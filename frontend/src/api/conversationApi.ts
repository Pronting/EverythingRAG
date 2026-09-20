import type { Conversation, ConversationMessage, ConversationSummary } from "../types";

/** 会话 API 客户端：列表 / 新建 / 读取 / 追加消息 / AI 标题。 */

const CONVERSATIONS_ENDPOINT = "/api/conversations";

async function readErrorDetail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (typeof body === "object" && body !== null && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === "string" && detail !== "") return detail;
    }
  } catch {
    /* 非 JSON */
  }
  return `本地服务返回异常（HTTP ${response.status}）`;
}

/** 会话摘要列表（按最近更新倒序，不含归档与空会话）。
 *
 * 传入 workspaceId 时只返回该工作区（侧边栏分区切换）；缺省不过滤（跨分区）。
 */
export async function listConversations(
  query = "",
  signal?: AbortSignal,
  workspaceId?: string | null,
): Promise<ConversationSummary[]> {
  const params = new URLSearchParams();
  if (query) params.set("q", query);
  if (workspaceId) params.set("workspace", workspaceId);
  const suffix = params.toString();
  const response = await fetch(suffix ? `${CONVERSATIONS_ENDPOINT}?${suffix}` : CONVERSATIONS_ENDPOINT, { signal });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  return (await response.json()) as ConversationSummary[];
}

/** 已归档会话列表（设置页「归档对话」，按归档时间倒序）。 */
export async function listArchivedConversations(signal?: AbortSignal): Promise<ConversationSummary[]> {
  const response = await fetch(`${CONVERSATIONS_ENDPOINT}/archived`, { signal });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  return (await response.json()) as ConversationSummary[];
}

/** 新建空会话（在指定工作区，缺省默认分区），返回摘要。 */
export async function createConversation(workspaceId?: string | null): Promise<ConversationSummary> {
  const query = workspaceId ? `?workspace=${encodeURIComponent(workspaceId)}` : "";
  const response = await fetch(`${CONVERSATIONS_ENDPOINT}${query}`, { method: "POST" });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  return (await response.json()) as ConversationSummary;
}

// =====================================================================
// 会话移动（PRD 5.4 / 13.2；A27 / A28）
//
// ⚠️ 两条硬要求，任何一条缺失都会让真实 UI 上的移动失败：
// 1. **`confirm`**：会话携带历史或存在活动任务时，后端返回 409 要求显式确认
//    （PRD 5.4 首句：移动会携带历史内容）。**必须在用户确认之后**才置 true ——
//    默认静默 true 等于绕过用户确认语义。
// 2. **`expected_context_epoch`**：乐观并发校验。后端只在运行期核对它，不一致 → 409。
//
// `t35` 把 `POST /{id}/move`（拖拽）与 `POST /{id}/workspace`（右键）收敛到**同一个**
// 状态机实现，两者请求体/响应/错误映射完全一致，所以这里两条路径共用一套字段。
//
// ⚠️ 诚实声明：E2E 用 `page.route` 造响应只能验证**请求形状与前端分支**，无法证明
// 前后端契约在真实服务上可用（这正是该缺陷此前被 Playwright 全绿掩盖的原因）。
// =====================================================================

/** 移动冲突（409）的两种成因，界面处理方式不同。 */
export type MoveConflictKind = "confirm_required" | "context_epoch_conflict" | "unknown";

export class MoveConflictError extends Error {
  readonly kind: MoveConflictKind;
  readonly detail: string;

  constructor(kind: MoveConflictKind, detail: string) {
    super(detail);
    this.name = "MoveConflictError";
    this.kind = kind;
    this.detail = detail;
  }
}

export class MoveApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "MoveApiError";
    this.status = status;
  }
}

/**
 * 409 detail → 成因分类。
 *
 * 后端两种 409 共用状态码，只能靠 detail 文案区分（契约如此）：
 * - 缺确认："移动会话会携带历史内容，请确认后再移动（confirm=true，PRD 5.4）"
 * - epoch 不符："会话上下文版本已变化：期望 X，实际 Y" / "…无法核对 expected_context_epoch…"
 * 判 epoch 优先：确认文案里不含 `context_epoch`，而 epoch 文案里一定含。
 */
export function classifyMoveConflict(detail: string): MoveConflictKind {
  if (/context_epoch|上下文版本/i.test(detail)) return "context_epoch_conflict";
  if (/confirm|确认/i.test(detail)) return "confirm_required";
  return "unknown";
}

/** 一次移动的结果（后端纯追加字段；同区排序时 `steps` 为空）。 */
export interface MoveOutcome extends ConversationSummary {
  from_workspace_id?: string | null;
  context_epoch?: number | null;
  previous_context_epoch?: number | null;
  stopped_run_ids?: string[];
  superseded_approvals?: number;
  superseded_inputs?: number;
  memory_snapshot_rebound?: boolean;
  permission_rebound?: boolean;
  steps?: string[];
}

export interface MoveOptions {
  /**
   * 用户已确认「移动会携带历史内容、并停止当前任务」。
   *
   * 会话**无历史且无活动任务**时后端不要求确认，可省略（保持既有交互）。
   */
  confirm?: boolean;
  /**
   * 预期 `context_epoch`（乐观并发）。缺省/`null` = 不做该校验。
   *
   * ⚠️ `GET /api/conversations/{id}` **不返回** `context_epoch`（`Conversation` 存档模型
   * 没有该字段），所以前端只能在拿到过真实值时才发：
   * `RunDetail.context_epoch`（后端在建 run 时从会话复制）与移动响应的 `context_epoch`。
   * 见 `knownConversationEpoch()`。
   */
  expectedContextEpoch?: number | null;
}

// ------------------------------------------------- 已观测到的 context_epoch
//
// 后端没有「读会话 context_epoch」的端点，但会在两处把它交给前端：建 run 的响应
// （`RunDetail.context_epoch`）与移动响应（`MoveOutcome.context_epoch`）。
// 这里缓存「最近一次观测到的值」，移动时作为乐观并发预期发出。
//
// 注意这是**观测值**不是权威读数：期间若有他人移动/上下文压缩，值会过期并触发 409。
// 那时 `forgetConversationEpoch()` 丢弃它，让用户重新确认后以 null 重试（不再无限重试）。

const observedEpochs = new Map<string, number>();

/** 记下后端交给我们的会话 `context_epoch`（仅接受 >=1 的整数）。 */
export function rememberConversationEpoch(conversationId: string, epoch: unknown): void {
  if (typeof epoch !== "number" || !Number.isInteger(epoch) || epoch < 1) return;
  observedEpochs.set(conversationId, epoch);
}

/** 丢弃已观测值（epoch 冲突后调用，避免用同一个过期值反复重试）。 */
export function forgetConversationEpoch(conversationId: string): void {
  observedEpochs.delete(conversationId);
}

/** 当前已知的 `context_epoch`；从未观测到则为 null（此时不发该校验字段）。 */
export function knownConversationEpoch(conversationId: string): number | null {
  return observedEpochs.get(conversationId) ?? null;
}

async function postMove(url: string, body: unknown): Promise<MoveOutcome> {
  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new MoveApiError(0, "无法连接本地服务，请确认后端已启动");
  }
  if (response.status === 409) {
    const detail = await readErrorDetail(response);
    throw new MoveConflictError(classifyMoveConflict(detail), detail);
  }
  if (!response.ok) throw new MoveApiError(response.status, await readErrorDetail(response));
  const payload = (await response.json()) as MoveOutcome;
  // 把后端回传的权威 epoch 记下来，供后续移动做乐观并发校验。
  if (typeof payload.context_epoch === "number") {
    rememberConversationEpoch(payload.id, payload.context_epoch);
  }
  return payload;
}

/**
 * 侧边栏拖拽落点：把会话移到目标工作区（缺省原地）的第 targetIndex 位。
 *
 * 跨区 == 归属变更，与 `moveConversationToWorkspace` 是同一条状态机（`t35` 收敛），
 * 因此同样需要 `confirm` / `expected_context_epoch`；同区排序不需要确认。
 */
export async function moveConversationInSidebar(
  id: string,
  target: { workspaceId?: string; targetIndex?: number },
  options: MoveOptions = {},
): Promise<MoveOutcome> {
  return postMove(`${CONVERSATIONS_ENDPOINT}/${encodeURIComponent(id)}/move`, {
    workspace_id: target.workspaceId ?? null,
    target_index: target.targetIndex ?? null,
    confirm: options.confirm ?? false,
    expected_context_epoch: options.expectedContextEpoch ?? null,
  });
}

/**
 * 把会话移动到指定工作区（分区归类）；右键「移动到工作区」走这里。
 *
 * 会话携带历史或存在活动任务时**必须**由用户确认后传 `confirm: true`，否则 409。
 */
export async function moveConversationToWorkspace(
  id: string,
  workspaceId: string,
  options: MoveOptions = {},
): Promise<MoveOutcome> {
  return postMove(`${CONVERSATIONS_ENDPOINT}/${encodeURIComponent(id)}/workspace`, {
    workspace_id: workspaceId,
    confirm: options.confirm ?? false,
    expected_context_epoch: options.expectedContextEpoch ?? null,
  });
}

/** 读取完整会话（含消息）。 */
export async function getConversation(id: string): Promise<Conversation> {
  const response = await fetch(`${CONVERSATIONS_ENDPOINT}/${encodeURIComponent(id)}`);
  if (!response.ok) throw new Error(await readErrorDetail(response));
  return (await response.json()) as Conversation;
}

/** 从第 index 条消息起截断会话（「编辑重发」：该条及其之后的轮次作废）。 */
export async function truncateConversation(id: string, index: number): Promise<number> {
  const response = await fetch(`${CONVERSATIONS_ENDPOINT}/${encodeURIComponent(id)}/truncate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ index }),
  });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  const body = (await response.json()) as { message_count?: number };
  return body.message_count ?? 0;
}

/** 追加一轮消息（user + assistant），返回摘要。 */
export async function appendConversationMessages(
  id: string,
  messages: Pick<
    ConversationMessage,
    "role" | "content" | "sources" | "answer_basis" | "policy_version"
  >[],
): Promise<ConversationSummary> {
  const response = await fetch(`${CONVERSATIONS_ENDPOINT}/${encodeURIComponent(id)}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
  });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  return (await response.json()) as ConversationSummary;
}

/** 用对话模型生成会话标题，返回新标题。 */
export async function generateConversationTitle(id: string): Promise<string> {
  const response = await fetch(`${CONVERSATIONS_ENDPOINT}/${encodeURIComponent(id)}/generate-title`, {
    method: "POST",
  });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  const data = (await response.json()) as { title: string };
  return data.title;
}

/** 删除会话。 */
export async function deleteConversation(id: string): Promise<void> {
  const response = await fetch(`${CONVERSATIONS_ENDPOINT}/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!response.ok) throw new Error(await readErrorDetail(response));
}

/** 归档会话：从侧边栏隐藏，可在设置页「归档对话」恢复。 */
export async function archiveConversation(id: string): Promise<ConversationSummary> {
  const response = await fetch(`${CONVERSATIONS_ENDPOINT}/${encodeURIComponent(id)}/archive`, {
    method: "POST",
  });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  return (await response.json()) as ConversationSummary;
}

/** 恢复已归档会话。 */
export async function restoreConversation(id: string): Promise<ConversationSummary> {
  const response = await fetch(`${CONVERSATIONS_ENDPOINT}/${encodeURIComponent(id)}/restore`, {
    method: "POST",
  });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  return (await response.json()) as ConversationSummary;
}

/** 分支会话：深拷贝源会话消息为独立新会话，返回分支摘要。 */
export async function forkConversation(id: string): Promise<ConversationSummary> {
  const response = await fetch(`${CONVERSATIONS_ENDPOINT}/${encodeURIComponent(id)}/fork`, {
    method: "POST",
  });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  return (await response.json()) as ConversationSummary;
}
