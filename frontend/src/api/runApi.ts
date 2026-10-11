import type {
  ApprovalDecisionKind,
  AnswerBasis,
  CitationEntry,
  PermissionMode,
  ReasoningLevel,
  RunDetail,
  RunEvent,
  RunEventTyped,
  RunImageRef,
  RunStreamFrame,
  RunSummary,
  SourceType,
} from "../types";
import {
  CLOSING_EVENT_TYPES,
  RUN_SCHEMA_VERSION,
  USER_INPUT_MAX_CHARS,
} from "../types";

/**
 * 任务（run）API 客户端 —— V1.2.0 取代旧的 `POST /api/chat` 单发流式入口。
 *
 * 与后端契约（`app/api/routes/runs.py` / `approvals.py` / `agent/events.py`）：
 * - 创建：`POST /api/workspaces/{w}/runs` → **202** `RunDetail`（PRD 13.2）
 * - 快照：`GET  /api/workspaces/{w}/runs/{r}` → `RunDetail`（刷新后据此恢复状态与卡片）
 * - 补读：`GET  /api/workspaces/{w}/runs/{r}/event-log?after=` → JSON 事件快照（首屏用，不必等 SSE）
 * - 订阅：`GET  /api/workspaces/{w}/runs/{r}/events?after=`（SSE，原生 `EventSource`）
 * - 卡片：`POST .../runs/{r}/approvals/{id}` / `.../inputs/{id}`
 * - 生命周期：`POST .../runs/{r}/cancel` / `.../resume`
 *
 * 成功判定一律走 `response.ok`（覆盖 2xx）——创建是 **202 Accepted** 而不是 201，
 * 任何写死具体状态码的判定都会把正常路径误判成失败。
 *
 * 断线语义（PRD 12.4）：SSE 断开只结束**订阅**，不取消执行；`EventSource` 自带
 * 重连并自动回传 `Last-Event-ID`，因此补读不需要手写。
 */

const API_ROOT = "/api/workspaces";

/** 错误类别：界面据此给出不同文案，绝不静默失败（PRD 12.5 / 13.2）。 */
export type RunErrorCode =
  | "stale_client"
  | "busy"
  | "not_found"
  | "too_long"
  | "conflict"
  | "protocol"
  | "unavailable"
  | "network";

export class RunApiError extends Error {
  readonly status: number;
  readonly code: RunErrorCode;

  constructor(code: RunErrorCode, status: number, message: string) {
    super(message);
    this.name = "RunApiError";
    this.code = code;
    this.status = status;
  }
}

/** 错误码 → 用户可读文案（唯一来源，避免各组件自行编文案）。 */
export function describeRunError(error: unknown): string {
  if (error instanceof RunApiError) {
    switch (error.code) {
      case "stale_client":
        // 后端把旧入口 `/api/chat` 结构性移除后返回 410。
        return "客户端版本过旧，请刷新页面";
      case "busy":
        return "当前会话还有未完成的任务，请先处理当前任务或停止后继续";
      case "too_long":
        return `问题过长（上限 ${USER_INPUT_MAX_CHARS} 字），请拆分后再问`;
      case "not_found":
        return "任务不存在或已被移除";
      case "unavailable":
        return "本地服务尚未就绪，请稍后重试";
      case "protocol":
        return "事件协议错误，请刷新页面重试";
      case "conflict":
        return "该操作已失效（可能已被处理或会话已移动）";
      default:
        return error.message;
    }
  }
  return error instanceof Error && error.message !== ""
    ? error.message
    : "无法连接本地服务，请确认后端已启动";
}

function codeForStatus(status: number): RunErrorCode {
  if (status === 410) return "stale_client";
  if (status === 409) return "busy";
  if (status === 404) return "not_found";
  if (status === 422) return "too_long";
  if (status === 503) return "unavailable";
  if (status === 502) return "protocol";
  return "network";
}

async function readDetail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (typeof body === "object" && body !== null && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === "string" && detail !== "") return detail;
    }
    if (typeof body === "object" && body !== null && "message" in body) {
      const message = (body as { message: unknown }).message;
      if (typeof message === "string" && message !== "") return message;
    }
  } catch {
    /* 非 JSON：用状态码兜底 */
  }
  return "";
}

async function toError(response: Response): Promise<RunApiError> {
  const code = codeForStatus(response.status);
  const detail = await readDetail(response);
  const fallback = describeRunError(new RunApiError(code, response.status, ""));
  return new RunApiError(code, response.status, detail || fallback);
}

async function requestJson<T>(
  url: string,
  init?: RequestInit,
  signal?: AbortSignal,
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, { ...init, signal });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new RunApiError("network", 0, "无法连接本地服务，请确认后端已启动");
  }
  if (!response.ok) throw await toError(response);
  return (await response.json()) as T;
}

function runsPath(workspaceId: string): string {
  return `${API_ROOT}/${encodeURIComponent(workspaceId)}/runs`;
}

// =====================================================================
// 负载校验（PRD 13.3 末段：未知版本或负载校验失败 → 协议错误，不降级）
// =====================================================================

/** 每个事件类型必须存在的 payload 字段（与 `events.py` 的模型一一对应）。 */
const REQUIRED_PAYLOAD_FIELDS: Record<string, string[]> = {
  "run.status": ["status"],
  "route.decided": ["decision"],
  "plan.updated": [],
  "agent.started": ["agent_kind"],
  "agent.completed": ["agent_kind"],
  "model.reasoning": ["step_id", "stream_id", "title", "text"],
  "tool.started": ["tool_name"],
  "tool.completed": ["tool_name"],
  "approval.requested": ["approval"],
  "approval.resolved": ["approval_id", "status"],
  "input.requested": ["request"],
  "input.resolved": ["input_request_id", "answer"],
  "evidence.added": [],
  "verification.updated": ["result"],
  "answer.meta": ["answer_id", "answer_basis"],
  "answer.delta": ["answer_id", "text"],
  "run.completed": ["answer_id"],
  "run.failed": ["error"],
  "run.cancelled": ["stop_reason"],
  "memory.updated": [],
  "memory.failed": ["reason"],
};

/** 负载字段的轻量类型检查：数组字段必须是数组，否则视为契约漂移。 */
const ARRAY_PAYLOAD_FIELDS: Record<string, string[]> = {
  "plan.updated": ["tasks", "covered_requirements", "uncovered_requirements"],
  "evidence.added": ["items"],
  "verification.updated": ["pending"],
  "answer.meta": ["citations", "basis_components"],
  "route.decided": [],
  "run.completed": ["gaps"],
};

export interface EventParseFailure {
  ok: false;
  /** 协议错误原因（只进诊断，不回显原始报文，PRD 12.5）。 */
  reason: string;
}

export type EventParseOutcome = { ok: true; event: RunEventTyped } | EventParseFailure;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * 校验并归一化一个事件（信封 + payload）。
 *
 * 未知 `schema_version`、未知 `type`、缺字段或类型不符一律判为协议错误 ——
 * 与后端 `parse_event` 同口径，保证「前端显示协议错误」而不是偷偷丢帧。
 */
export function parseRunEvent(raw: unknown): EventParseOutcome {
  if (!isRecord(raw)) return { ok: false, reason: "事件不是对象" };

  const schemaVersion = raw.schema_version;
  if (typeof schemaVersion !== "number") {
    return { ok: false, reason: "缺少 schema_version" };
  }
  if (schemaVersion !== RUN_SCHEMA_VERSION) {
    return {
      ok: false,
      reason: `未知事件版本 schema_version=${schemaVersion}，本版本只支持 ${RUN_SCHEMA_VERSION}`,
    };
  }
  if (typeof raw.run_id !== "string" || raw.run_id === "") {
    return { ok: false, reason: "缺少 run_id" };
  }
  const seq = raw.event_seq;
  if (typeof seq !== "number" || !Number.isInteger(seq) || seq < 1) {
    return { ok: false, reason: "event_seq 必须是 >=1 的整数" };
  }
  const type = raw.type;
  if (typeof type !== "string" || !(type in REQUIRED_PAYLOAD_FIELDS)) {
    return { ok: false, reason: `未知事件类型: ${String(type)}` };
  }
  if (!isRecord(raw.payload)) {
    return { ok: false, reason: `事件 ${type} 缺少 payload` };
  }

  const payload = raw.payload;
  if (type === "model.reasoning" && ["step_id", "stream_id", "title", "text"].some(
    (field) => typeof payload[field] !== "string",
  )) return { ok: false, reason: "模型思考事件字段必须是文本" };
  for (const field of REQUIRED_PAYLOAD_FIELDS[type]) {
    if (!(field in payload)) {
      return { ok: false, reason: `事件 ${type} 的 payload 缺少 ${field}` };
    }
  }
  for (const field of ARRAY_PAYLOAD_FIELDS[type] ?? []) {
    if (field in payload && !Array.isArray(payload[field])) {
      return { ok: false, reason: `事件 ${type} 的 payload.${field} 必须是数组` };
    }
  }
  if (type === "answer.delta" && typeof payload.index !== "number") {
    return { ok: false, reason: "answer.delta 缺少 index" };
  }

  return { ok: true, event: raw as unknown as RunEventTyped };
}

/** SSE `data` 文本 → 事件或协议错误帧。 */
export function decodeRunFrame(data: string): RunStreamFrame | null {
  let decoded: unknown;
  try {
    decoded = JSON.parse(data);
  } catch {
    return null; // 坏帧跳过，不崩
  }
  if (isRecord(decoded) && decoded.type === "error" && typeof decoded.error_code === "string") {
    return {
      type: "error",
      error_code: decoded.error_code,
      message: typeof decoded.message === "string" ? decoded.message : "事件协议错误",
    };
  }
  const outcome = parseRunEvent(decoded);
  return outcome.ok ? outcome.event : null;
}

// =====================================================================
// 端点
// =====================================================================

export interface CreateRunInput {
  conversationId: string;
  message: string;
  clientRequestId?: string;
  permissionMode?: PermissionMode;
  reasoningLevel?: ReasoningLevel;
  /** 本次 run 附带的图片引用（`data:image/...` 内联或 http(s)），最多 4 张。 */
  images?: string[];
  /**
   * 来源 run（PRD 5.4 末句）：会话移到别的执行上下文后，新执行承接自哪一次 run。
   *
   * 后端的 `POST /runs` 接受该字段但**不在 `RunDetail` 里回显** —— 前端必须自己记住
   * 并展示，否则用户会以为历史凭空消失。
   */
  originRunId?: string | null;
  replaceFromMessageId?: string | null;
}

/** 一次 run 允许附带的图片张数（与后端 `MAX_RUN_IMAGES` 一致，超出 422）。 */
export const MAX_RUN_IMAGES = 4;

/** 会话内幂等 id：重复请求由后端按此返回同一 run（PRD 13.2）。 */
export function nextClientRequestId(): string {
  const random =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `cr-${random}`;
}

/** 创建并启动一次执行（含用户消息落库）；成功是 **202 Accepted**。 */
export async function createRun(
  workspaceId: string,
  input: CreateRunInput,
  signal?: AbortSignal,
): Promise<RunDetail> {
  return requestJson<RunDetail>(
    runsPath(workspaceId),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversation_id: input.conversationId,
        message: input.message,
        client_request_id: input.clientRequestId ?? nextClientRequestId(),
        permission_mode: input.permissionMode ?? null,
        reasoning_level: input.reasoningLevel ?? null,
        // 图片引用（data URI 或 http(s)），后端上限 4 张（超出 422）；结果见 RunDetail.image_refs。
        images: input.images ?? [],
        // 来源 run（PRD 5.4）：只在确有承接关系时带上。
        origin_run_id: input.originRunId ?? null,
        replace_from_message_id: input.replaceFromMessageId ?? null,
      }),
    },
    signal,
  );
}

/**
 * 某会话的 run 列表（按创建时间）。
 *
 * 用途：会话**静态移动**（无活动 run、`stopped_run_ids` 为空）时，仍要按 PRD 5.4 末句
 * 记录「新的执行承接自哪一次 run」——取列表里最后一次执行。
 */
export async function listConversationRuns(
  workspaceId: string,
  conversationId: string,
  limit?: number,
  signal?: AbortSignal,
): Promise<RunSummary[]> {
  const params = new URLSearchParams({ conversation_id: conversationId });
  if (limit !== undefined) params.set("limit", String(limit));
  const url = `${runsPath(workspaceId)}?${params.toString()}`;
  const payload = await requestJson<{ runs?: RunSummary[] }>(url, undefined, signal);
  return Array.isArray(payload.runs) ? payload.runs : [];
}

/** run 快照：刷新页面后恢复状态、子任务、待处理卡片与末事件位置（PRD 14.3）。 */
export async function getRun(
  workspaceId: string,
  runId: string,
  signal?: AbortSignal,
): Promise<RunDetail> {
  return requestJson<RunDetail>(
    `${runsPath(workspaceId)}/${encodeURIComponent(runId)}`,
    undefined,
    signal,
  );
}

/** 首屏事件快照（JSON，不必等 SSE）：返回事件列表与最大序号。 */
export async function fetchEventLog(
  workspaceId: string,
  runId: string,
  after = 0,
  signal?: AbortSignal,
): Promise<{ events: RunEventTyped[]; lastSeq: number; protocolError: string | null }> {
  const url = `${runsPath(workspaceId)}/${encodeURIComponent(runId)}/event-log?after=${after}`;
  let response: Response;
  try {
    response = await fetch(url, { signal });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new RunApiError("network", 0, "无法连接本地服务，请确认后端已启动");
  }
  if (!response.ok) {
    // 502 = 事件协议错误：明确上报，不静默返回空列表。
    if (response.status === 502) {
      const detail = await readDetail(response);
      return { events: [], lastSeq: after, protocolError: detail || "事件协议错误" };
    }
    throw await toError(response);
  }
  const body = (await response.json()) as { events?: unknown };
  const events: RunEventTyped[] = [];
  let lastSeq = after;
  let protocolError: string | null = null;
  for (const raw of Array.isArray(body.events) ? body.events : []) {
    const outcome = parseRunEvent(raw);
    if (!outcome.ok) {
      protocolError = outcome.reason;
      continue;
    }
    events.push(outcome.event);
    lastSeq = Math.max(lastSeq, outcome.event.event_seq);
  }
  return { events, lastSeq, protocolError };
}

export interface RunSubscriptionHandlers {
  /** 已通过校验的事件（同一 `event_seq` 只回调一次）。 */
  onEvent: (event: RunEventTyped) => void;
  /** 协议错误（未知版本 / 负载校验失败 / 服务端 error 帧）。 */
  onProtocolError: (message: string) => void;
  /** 订阅关闭（收到终态帧或服务端断开且不再重连）。 */
  onClose: () => void;
}

export interface RunSubscription {
  close: () => void;
  /** 已收到（并已去重）的最大事件序号。 */
  lastSeq: () => number;
}

export interface SubscribeRunOptions {
  /** 从该序号之后补读（缺省 0）；`EventSource` 重连时浏览器自动带 `Last-Event-ID`。 */
  after?: number;
}

/**
 * 订阅 run 事件流（原生 `EventSource`，GET SSE）。
 *
 * 去重：以 `event_seq` 为游标，重复序号（重连补读重叠段）直接丢弃。
 * 关闭：收到 `run.completed` / `run.failed` / `run.cancelled` 后主动关闭，
 * 避免 `EventSource` 对已终态的流反复重连。
 */
export function subscribeRunEvents(
  workspaceId: string,
  runId: string,
  handlers: RunSubscriptionHandlers,
  options: SubscribeRunOptions = {},
): RunSubscription {
  const url = `${runsPath(workspaceId)}/${encodeURIComponent(runId)}/events?after=${
    options.after ?? 0
  }`;
  const source = new EventSource(url);
  let cursor = options.after ?? 0;
  let closed = false;

  const close = (): void => {
    if (closed) return;
    closed = true;
    source.close();
    handlers.onClose();
  };

  source.onmessage = (message: MessageEvent<string>): void => {
    const frame = decodeRunFrame(message.data);
    if (frame === null) {
      handlers.onProtocolError("事件负载校验失败");
      return;
    }
    if ("type" in frame && frame.type === "error") {
      handlers.onProtocolError(frame.message);
      close();
      return;
    }
    const event = frame as RunEventTyped;
    if (event.event_seq <= cursor) return; // 重连重叠段：去重
    cursor = event.event_seq;
    handlers.onEvent(event);
    if (CLOSING_EVENT_TYPES.includes(event.type)) close();
  };

  source.onerror = (): void => {
    if (closed) return;
    // CONNECTING = 浏览器正在自动重连（带 Last-Event-ID），不算失败。
    if (source.readyState === EventSource.CONNECTING) return;
    close();
  };

  return { close, lastSeq: () => cursor };
}

// =====================================================================
// 卡片与生命周期
// =====================================================================

export interface ApprovalSubmitInput {
  decision: ApprovalDecisionKind;
  editedParameters?: Record<string, unknown> | null;
  note?: string;
}

/** 提交批准决定（允许本次 / 允许本轮同类操作 / 拒绝）。 */
export async function submitApproval(
  workspaceId: string,
  runId: string,
  approvalId: string,
  input: ApprovalSubmitInput,
  signal?: AbortSignal,
): Promise<unknown> {
  const url = `${runsPath(workspaceId)}/${encodeURIComponent(runId)}/approvals/${encodeURIComponent(
    approvalId,
  )}`;
  return requestJson<unknown>(
    url,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        decision: input.decision,
        edited_parameters: input.editedParameters ?? null,
        note: input.note ?? "",
      }),
    },
    signal,
  );
}

/** 提交澄清回答；`questionVersion` 不符由后端 409 拦下（PRD 13.2）。 */
export async function submitInput(
  workspaceId: string,
  runId: string,
  inputRequestId: string,
  answer: string,
  questionVersion?: number,
  signal?: AbortSignal,
): Promise<unknown> {
  const url = `${runsPath(workspaceId)}/${encodeURIComponent(runId)}/inputs/${encodeURIComponent(
    inputRequestId,
  )}`;
  return requestJson<unknown>(
    url,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answer, question_version: questionVersion ?? null }),
    },
    signal,
  );
}

/** 停止：立即请求取消主任务与活动子任务。 */
export async function cancelRun(
  workspaceId: string,
  runId: string,
  reason = "user_stopped",
  signal?: AbortSignal,
): Promise<RunDetail> {
  return requestJson<RunDetail>(
    `${runsPath(workspaceId)}/${encodeURIComponent(runId)}/cancel`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reason }),
    },
    signal,
  );
}

/** 继续：从记录位置重新驱动，已完成子任务不重复执行（PRD 12.3 / 12.4）。 */
export async function resumeRun(
  workspaceId: string,
  runId: string,
  signal?: AbortSignal,
): Promise<RunDetail> {
  return requestJson<RunDetail>(
    `${runsPath(workspaceId)}/${encodeURIComponent(runId)}/resume`,
    { method: "POST" },
    signal,
  );
}

/** 标签 → 来源类型（引用编号前缀的唯一来源，PRD 8.3）。 */
export function sourceTypeOfLabel(label: string): SourceType | null {
  const prefix = label.slice(0, 1).toUpperCase();
  if (prefix === "S") return "knowledge";
  if (prefix === "W") return "web";
  if (prefix === "M") return "memory";
  if (prefix === "H") return "history";
  return null;
}

// =====================================================================
// 图片引用文案（唯一来源）
//
// 贴图是**有条件生效**的：只有后端 `status=described` 才代表原图已缓存 + 描述已入库、
// `verify_image` 本轮真的能核对。其余状态都必须明确告诉用户「本轮不参与核对」，并给出
// 原因与下一步 —— 界面不得把 cached/unavailable 说成已生效（不伪造成功）。
// 文案直接由 `status` + `reason` 驱动，前端不自行猜测。
// =====================================================================

/** 需要用户采取的下一步动作。 */
export type RunImageAction = "configure_vision" | "replace_image" | null;

export interface RunImageOutcome {
  /** 状态标签（界面直接显示）。 */
  label: string;
  /** 本轮是否真的参与核对 —— 只有 `described` 为真。 */
  effective: boolean;
  /** 未生效时的说明；生效时为 null。 */
  notice: string | null;
  /** 建议的下一步动作。 */
  action: RunImageAction;
}

/** 单条图片引用的用户可读结论。 */
export function describeRunImage(ref: RunImageRef): RunImageOutcome {
  const reason = ref.reason === "" ? "" : `（${ref.reason}）`;
  switch (ref.status) {
    case "described":
      return {
        label: "已生效 · 本轮参与核对",
        effective: true,
        notice: null,
        action: null,
      };
    case "cached":
      return {
        label: "已保存 · 本轮不参与核对",
        effective: false,
        notice: `图片已随提问保存到本工作区，但尚未识图${reason}，本轮不会参与核对。配置视觉模型后，会在下次启动时自动补做。`,
        action: "configure_vision",
      };
    case "pending_vision":
      return {
        label: "识图未完成 · 本轮不参与核对",
        effective: false,
        notice: `图片已登记，但识图没有成功${reason}，本轮不会参与核对。配置或修复视觉模型后会补做。`,
        action: "configure_vision",
      };
    case "unavailable":
      return {
        label: "未取得图片 · 本轮不参与核对",
        effective: false,
        notice: `没有取得这张图片的内容${reason}，本轮不会参与核对。请更换为可用的图片或链接。`,
        action: "replace_image",
      };
    case "rejected":
      return {
        label: "未被接受",
        effective: false,
        notice: `这张图片没有被接受${reason}，本轮不会参与核对。请更换后重试。`,
        action: "replace_image",
      };
    default:
      return {
        label: `状态未知（${ref.status || "空"}）`,
        effective: false,
        notice: "后端返回了未知的图片处理状态，按「未生效」处理，本轮不参与核对。",
        action: null,
      };
  }
}

/** 一组图片引用的整体结论：**全部** `described` 才算本轮全部生效。 */
export function summarizeRunImages(refs: RunImageRef[]): {
  total: number;
  effective: number;
  allEffective: boolean;
  /** 整体结论一句话；全部生效时为 null。 */
  notice: string | null;
} {
  const outcomes = refs.map(describeRunImage);
  const effective = outcomes.filter((outcome) => outcome.effective).length;
  const allEffective = refs.length > 0 && effective === refs.length;
  return {
    total: refs.length,
    effective,
    allEffective,
    notice: allEffective
      ? null
      : `共 ${refs.length} 张图片，其中 ${effective} 张本轮参与核对`,
  };
}

/** 引用 → 正文可点击标记用的紧凑来源条目（复用既有 MarkdownContent/CitationPanel）。 */
export function citationLabelSet(citations: CitationEntry[]): Set<string> {
  return new Set(citations.map((citation) => citation.label));
}

/** 答案依据首值（basis_components 只有一项时即答案依据）。 */
export function basisFromComponents(
  basis: AnswerBasis,
  components: SourceType[],
): AnswerBasis {
  if (basis !== "mixed") return basis;
  return components.length === 1 ? (components[0] as AnswerBasis) : "mixed";
}
