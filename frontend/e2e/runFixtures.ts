import { expect, type Page, type Route } from "@playwright/test";

/**
 * E2E 的 V1.2 任务（run）契约夹具。
 *
 * 全部离线：`page.route` 拦截 `/api`，**不访问真实后端**（`/api` 代理指向死端口）。
 * 关键约束（决定验收成败）：迁移后测试必须显式证明「至少一次命中 `/runs`」且
 * 「零次命中 `/api/chat`」—— 否则 stub 会拦住一个再也没人调用的旧端点，测试等于空跑。
 */

export const SCHEMA_VERSION = 2;

export const STATUS_BODY = JSON.stringify({
  app: { name: "everything-rag", version: "0.2.0" },
  config: {
    wizard_completed: false,
    embed_configured: true,
    chat_configured: true,
    vision_configured: false,
    vision_enabled: false,
    search_enabled: true,
    search_configured: true,
  },
  knowledge: {
    file_count: 0,
    chunk_count: 0,
    image_count: 0,
    image_tasks: { pending: 0, done: 0, failed: 0, total: 0 },
    last_sync_at: null,
    needs_rebuild: false,
  },
  privacy: { outbound_state: "local-only", providers: {} },
});

// ------------------------------------------------------------------ 事件构造

export interface V2Frame {
  type: string;
  payload: Record<string, unknown>;
  task_id?: string | null;
}

export function runStatus(status: string, phase = "", message = ""): V2Frame {
  return { type: "run.status", payload: { status, phase, message } };
}

export function routeDecided(overrides: Record<string, unknown> = {}): V2Frame {
  return {
    type: "route.decided",
    payload: {
      decision: {
        question: "q",
        resolved_question: "q",
        intent: "research",
        needs_knowledge: true,
        needs_web: false,
        needs_history: false,
        needs_general: false,
        freshness: "none",
        entities: [],
        scope_evidence_ids: [],
        unresolved_terms: [],
        reason_summary: "先查本地知识，不足再联网",
        ...overrides,
      },
    },
  };
}

export interface TaskSeed {
  task_id: string;
  question: string;
  agent_kind?: string;
  status?: string;
  depends_on?: string[];
  requirement_ids?: string[];
  /** 父任务 id（t25 起 run 详情带，PRD 14.2 父子任务显示）。 */
  parent_task_id?: string | null;
}

export function planUpdated(tasks: TaskSeed[], uncovered: string[] = []): V2Frame {
  return {
    type: "plan.updated",
    payload: {
      tasks: tasks.map((task) => ({
        task_id: task.task_id,
        question: task.question,
        agent_kind: task.agent_kind ?? "rag",
        status: task.status ?? "queued",
        depends_on: task.depends_on ?? [],
        requirement_ids: task.requirement_ids ?? [],
      })),
      covered_requirements: [],
      uncovered_requirements: uncovered,
    },
  };
}

export function agentStarted(
  agentKind: string,
  options: { taskId?: string | null; question?: string; threadId?: string } = {},
): V2Frame {
  return {
    type: "agent.started",
    payload: {
      agent_kind: agentKind,
      thread_id: options.threadId ?? "",
      task_id: options.taskId ?? null,
      question: options.question ?? "",
    },
    task_id: options.taskId ?? null,
  };
}

export function agentCompleted(
  agentKind: string,
  options: { taskId?: string | null; status?: string; summary?: string } = {},
): V2Frame {
  return {
    type: "agent.completed",
    payload: {
      agent_kind: agentKind,
      status: options.status ?? "completed",
      summary: options.summary ?? "",
    },
    task_id: options.taskId ?? null,
  };
}

export function toolStarted(
  toolName: string,
  options: { argsSummary?: string; operationId?: string; taskId?: string | null } = {},
): V2Frame {
  return {
    type: "tool.started",
    payload: {
      tool_name: toolName,
      args_summary: options.argsSummary ?? "",
      operation_id: options.operationId ?? `op-${toolName}`,
      task_id: options.taskId ?? null,
    },
    task_id: options.taskId ?? null,
  };
}

export function toolCompleted(
  toolName: string,
  options: {
    operationId?: string;
    status?: string;
    resultCount?: number;
    sourceCount?: number;
    errorCode?: string;
    taskId?: string | null;
  } = {},
): V2Frame {
  return {
    type: "tool.completed",
    payload: {
      tool_name: toolName,
      operation_id: options.operationId ?? `op-${toolName}`,
      status: options.status ?? "succeeded",
      result_count: options.resultCount ?? 1,
      source_count: options.sourceCount ?? 1,
      error_code: options.errorCode ?? "",
    },
    task_id: options.taskId ?? null,
  };
}

export function evidenceAdded(
  items: { evidence_id: string; source_type: string; title: string; source_ref: string; snippet?: string }[],
  taskId: string | null = null,
): V2Frame {
  return {
    type: "evidence.added",
    payload: {
      items: items.map((item) => ({
        evidence_id: item.evidence_id,
        source_type: item.source_type,
        title: item.title,
        source_ref: item.source_ref,
        snippet: item.snippet ?? "",
      })),
    },
    task_id: taskId,
  };
}

export interface ApprovalSeed {
  approval_id: string;
  tool_name?: string;
  purpose?: string;
  query_or_urls?: string[];
  date_range?: string;
  task_question?: string;
  redaction_notice?: string;
  task_id?: string | null;
}

export function approvalPayload(seed: ApprovalSeed): Record<string, unknown> {
  return {
    approval_id: seed.approval_id,
    run_id: "run-1",
    task_id: seed.task_id ?? null,
    interrupt_id: "int-1",
    operation_id: "op-web-search",
    args_hash: "hash-1",
    context_epoch: 1,
    rule_version: "v1",
    display: {
      purpose: seed.purpose ?? "需要联网检索最新版本",
      tool_name: seed.tool_name ?? "web_search",
      query_or_urls: seed.query_or_urls ?? ["风控指标定义 最新版"],
      date_range: seed.date_range ?? "2024-01-01 ~ 2026-09-19",
      task_question: seed.task_question ?? "风控指标定义是什么？",
      redaction_notice: seed.redaction_notice ?? "已脱敏：发送内容不含真实姓名（显示为「张三」等稳定称谓）",
      parameters: { query: (seed.query_or_urls ?? ["风控指标定义 最新版"])[0] },
    },
    status: "pending",
    decision: null,
    scope: null,
    superseded_reason: "",
    created_at: "2026-09-19T10:00:00Z",
    resolved_at: null,
  };
}

export function approvalRequested(seed: ApprovalSeed): V2Frame {
  return { type: "approval.requested", payload: { approval: approvalPayload(seed) } };
}

export function approvalResolved(approvalId: string, decisionKind: string): V2Frame {
  return {
    type: "approval.resolved",
    payload: { approval_id: approvalId, status: "approved", decision_kind: decisionKind },
  };
}

export interface InputSeed {
  input_request_id: string;
  question: string;
  options?: string[];
  reason?: string;
  task_id?: string | null;
}

export function inputPayload(seed: InputSeed): Record<string, unknown> {
  return {
    input_request_id: seed.input_request_id,
    run_id: "run-1",
    task_id: seed.task_id ?? null,
    question: seed.question,
    options: seed.options ?? [],
    reason: seed.reason ?? "",
    status: "pending",
    answer: null,
    question_version: 1,
    created_at: "2026-09-19T10:00:00Z",
    resolved_at: null,
  };
}

export function inputRequested(seed: InputSeed): V2Frame {
  return { type: "input.requested", payload: { request: inputPayload(seed) } };
}

export function inputResolved(inputRequestId: string, answer: string): V2Frame {
  return { type: "input.resolved", payload: { input_request_id: inputRequestId, answer } };
}

export function verificationUpdated(
  verdict: string,
  options: { issues?: unknown[]; conflicts?: string[]; uncovered?: string[]; pending?: string[] } = {},
): V2Frame {
  return {
    type: "verification.updated",
    payload: {
      result: {
        verdict,
        issues: options.issues ?? [],
        covered_requirements: ["R1"],
        uncovered_requirements: options.uncovered ?? [],
        open_questions: [],
        conflicts: options.conflicts ?? [],
        checked_at: "2026-09-19T10:00:00Z",
      },
      pending: options.pending ?? [],
    },
  };
}

export interface CitationSeed {
  label: string;
  source_type?: string;
  evidence_id?: string;
  workspace_id?: string;
  workspace_name?: string;
  source_ref?: string;
  snippet?: string;
  source_status?: string;
}

// ------------------------------------------------------------ 图片引用（t25）

export interface ImageRefSeed {
  status: string;
  kind?: string;
  reason?: string;
  image_id?: string;
  block_id?: string;
  content_hash?: string;
  src_ref?: string;
}

/** 一条 `RunDetail.image_refs` 条目（字段与后端 `ImageRef` 同形）。 */
export function imageRefPayload(seed: ImageRefSeed, index = 0): Record<string, unknown> {
  return {
    image_id: seed.image_id ?? `run-image-block-${index + 1}`,
    block_id: seed.block_id ?? `run-image-block-${index + 1}`,
    content_hash: seed.content_hash ?? `hash-image-${index + 1}`,
    kind: seed.kind ?? "data_uri",
    status: seed.status,
    reason: seed.reason ?? "",
    src_ref: seed.src_ref ?? `run-image/${index + 1}`,
    recorded_at: "2026-09-19T10:00:00Z",
  };
}

/**
 * 缺省图片引用：每条请求里的图片回一条 `cached`。
 *
 * `cached` 正是「视觉 provider 未配置」时后端的真实形态（原图已缓存 + 任务已登记、
 * 但没有描述块）—— 因此缺省行为就足以验证界面**不**把贴图说成已生效。
 */
export function defaultImageRefs(images: string[]): Record<string, unknown>[] {
  return images.map((image, index) =>
    imageRefPayload(
      {
        status: "cached",
        kind: image.startsWith("data:") ? "data_uri" : image.startsWith("http") ? "url" : "invalid",
        reason: "视觉模型未配置",
      },
      index,
    ),
  );
}

export function citationPayload(seed: CitationSeed): Record<string, unknown> {
  return {
    label: seed.label,
    source_type: seed.source_type ?? "knowledge",
    evidence_id: seed.evidence_id ?? `ev-${seed.label}`,
    workspace_id: seed.workspace_id ?? "default",
    workspace_name: seed.workspace_name ?? "默认",
    source_ref: seed.source_ref ?? "notes.md",
    snippet: seed.snippet ?? "",
    source_status: seed.source_status ?? "active",
  };
}

export function answerMeta(
  citations: CitationSeed[],
  options: { answerId?: string; basis?: string; components?: string[]; status?: string } = {},
): V2Frame {
  return {
    type: "answer.meta",
    payload: {
      answer_id: options.answerId ?? "ans-1",
      answer_basis: options.basis ?? "knowledge",
      basis_components: options.components ?? ["knowledge"],
      citations: citations.map(citationPayload),
      content_hash: "hash-ans-1",
      status: options.status ?? "running",
    },
  };
}

export function answerDelta(text: string, index = 0, answerId = "ans-1"): V2Frame {
  return { type: "answer.delta", payload: { answer_id: answerId, index, text } };
}

export function runCompleted(
  options: { answerId?: string; status?: string; gaps?: string[] } = {},
): V2Frame {
  return {
    type: "run.completed",
    payload: {
      answer_id: options.answerId ?? "ans-1",
      status: options.status ?? "completed",
      content_hash: "hash-ans-1",
      usage: {},
      gaps: options.gaps ?? [],
    },
  };
}

export function runFailed(message: string, options: { code?: string; resumable?: boolean } = {}): V2Frame {
  return {
    type: "run.failed",
    payload: {
      error: {
        error_code: options.code ?? "retrieval_failed",
        message,
        node: "research",
        operation_id: null,
        resumable: options.resumable ?? true,
        required_action: "可以点击继续从此处重试",
        created_at: "2026-09-19T10:00:00Z",
      },
    },
  };
}

export function runCancelled(reason = "user_stopped", resumable = true): V2Frame {
  return { type: "run.cancelled", payload: { stop_reason: reason, resumable } };
}

// ------------------------------------------------------------------ 编码

export function encodeV2Events(runId: string, frames: V2Frame[], after = 0): string {
  // `retry:` 让浏览器在服务端关流后按这个间隔重连 —— 没有它，等待中的 run 会以
  // 零间隔疯狂重连（测试桩立即关流造成），既刷日志也干扰断言。
  const head = after === 0 ? "retry: 30000\n\n" : "";
  return (
    head +
    frames
      .map((frame, index) => {
        const seq = index + 1;
        if (seq <= after) return "";
        const body = {
          schema_version: SCHEMA_VERSION,
          run_id: runId,
          event_seq: seq,
          execution_id: "exec-1",
          task_id: frame.task_id ?? null,
          type: frame.type,
          created_at: "2026-09-19T10:00:00Z",
          payload: frame.payload,
        };
        return `id: ${seq}\ndata: ${JSON.stringify(body)}\n\n`;
      })
      .join("")
  );
}

// ------------------------------------------------------------------ 桩

export interface StubRunOptions {
  /** 每个 run 的事件序列；传函数则按第几个 run（0 起）返回不同序列。 */
  frames?: V2Frame[] | ((runIndex: number) => V2Frame[]);
  /** 创建 run 时的初始状态。 */
  status?: string;
  /** 详情里返回的待批准/待澄清卡片。 */
  approvals?: ApprovalSeed[];
  inputs?: InputSeed[];
  /** 详情里返回的子任务。 */
  tasks?: TaskSeed[];
  /** 非空时，创建 run 的响应会等到该 Promise resolve（制造「发送中」窗口）。 */
  gate?: Promise<void>;
  /** 会话创建返回的 id。 */
  conversationId?: string;
  /** 新建会话的响应体（覆盖默认）。 */
  conversationBody?: unknown;
  /** 历史会话消息（切换会话时返回）。 */
  conversationMessages?: unknown[];
  /**
   * 不注册 runs 路由：让 run 请求真正落到 dev server（用于「停止生成」用例验证
   * vite.config.ts 的慢速 SSE 夹具 —— 它是最易漏的耦合点）。
   */
  skipRuns?: boolean;
  /**
   * run 详情里返回的图片引用处理结果（`RunDetail.image_refs`）。
   *
   * 传函数则按第几个 run（0 起）与请求体里的 `images` 生成，便于断言「请求带了 N 张
   * → 详情回 N 条状态」。缺省时按请求逐条回 `cached`（视觉未配置的常见形态）。
   */
  imageRefs?:
    | Record<string, unknown>[]
    | ((images: string[], runIndex: number) => Record<string, unknown>[]);
}

export interface RunStub {
  /** 所有请求的 pathname（断言「命中 /runs」「零次 /api/chat」用）。 */
  paths: string[];
  /** 每次创建的 run id。 */
  createdRuns: string[];
  /** 批准提交记录。 */
  approvalSubmits: { approvalId: string; decision: string; edited: unknown }[];
  /** 澄清提交记录。 */
  inputSubmits: { inputRequestId: string; answer: string }[];
  /** 取消 / 恢复调用次数。 */
  cancels: number;
  resumes: number;
  /** 每次创建 run 请求体里带的权限模式与思考深度（断言「设置落到请求」）。 */
  createRequests: { permissionMode: string | null; reasoningLevel: string | null }[];
  /** 每次创建 run 请求体里的图片引用（断言「贴图落到请求」）。 */
  createImages: string[][];
  /** 命中 `/runs` 的请求数。 */
  runRequestCount: () => number;
  /** 命中 `/api/chat` 的请求数（迁移后必须为 0）。 */
  chatRequestCount: () => number;
}

/**
 * 安装完整离线后端桩：status / settings / conversations / workspaces / runs。
 *
 * 注册顺序很关键：Playwright 的路由按**注册顺序的逆序**匹配，所以通用工作区列表
 * 路由必须先注册，具体的 runs 路由后注册才能生效。
 */
export async function stubRunBackend(page: Page, options: StubRunOptions = {}): Promise<RunStub> {
  const paths: string[] = [];
  const runs = new Map<
    string,
    {
      frames: V2Frame[];
      status: string;
      approvals: ApprovalSeed[];
      inputs: InputSeed[];
      tasks: TaskSeed[];
      permissionMode: string;
      reasoningLevel: string;
      imageRefs: Record<string, unknown>[];
    }
  >();
  let runCounter = 0;
  const conversationId = options.conversationId ?? "conv-1";

  page.on("request", (request) => {
    try {
      paths.push(new URL(request.url()).pathname);
    } catch {
      /* data: / about: 之类 */
    }
  });

  const stub: RunStub = {
    paths,
    createdRuns: [],
    approvalSubmits: [],
    inputSubmits: [],
    cancels: 0,
    resumes: 0,
    createRequests: [],
    createImages: [],
    runRequestCount: () => paths.filter((path) => path.includes("/runs")).length,
    chatRequestCount: () => paths.filter((path) => path === "/api/chat").length,
  };

  const json = (route: Route, body: unknown, status = 200) =>
    route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });

  await page.route("**/api/status", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: STATUS_BODY }),
  );
  await page.route("**/api/settings", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "{}" }),
  );

  await page.route("**/api/conversations**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    const method = route.request().method();
    if (path === "/api/conversations" && method === "GET") return json(route, []);
    if (path === "/api/conversations" && method === "POST") {
      return json(
        route,
        options.conversationBody ?? {
          id: conversationId,
          title: "新对话",
          message_count: 0,
          workspace_id: "default",
        },
        201,
      );
    }
    if (path.endsWith("/messages")) return json(route, { message_count: 0 });
    if (path.endsWith("/generate-title")) return json(route, { title: "AI 标题" });
    if (path.endsWith("/truncate")) return json(route, { message_count: 0 });
    if (method === "GET") {
      return json(route, {
        id: conversationId,
        title: "新对话",
        workspace_id: "default",
        messages: options.conversationMessages ?? [],
      });
    }
    return json(route, {});
  });

  // 工作区命名空间用**单个路由派发**（不做多路由优先级依赖）：
  // - `/runs` 前缀 → run 端点（skipRuns 时 fallback 到网络，让 dev server 夹具接管）
  // - 其余 → 工作区列表
  // 单路由 + 显式派发避免了「fallback 落到另一个已注册 stub」的假绿灯陷阱。
  await page.route("**/api/workspaces**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    if (path.includes("/runs")) {
      if (options.skipRuns === true) return route.fallback();
      await handleRunRoute(route, url, path);
      return;
    }
    return json(route, [
      {
        id: "default",
        name: "默认",
        is_default: true,
        created_at: "",
        updated_at: "",
        conversation_count: 0,
      },
    ]);
  });

  const detailOf = (runId: string) => {
    const entry = runs.get(runId);
    const lastSeq = entry === undefined ? 0 : entry.frames.length;
    return {
      run_id: runId,
      conversation_id: conversationId,
      workspace_id: "default",
      status: entry?.status ?? "queued",
      permission_mode: entry?.permissionMode ?? "ask",
      reasoning_level: entry?.reasoningLevel ?? "high",
      answer_id: null,
      error_code: null,
      stop_reason: null,
      resumable: entry?.status === "failed",
      context_epoch: 1,
      created_at: "2026-09-19T10:00:00Z",
      updated_at: "2026-09-19T10:00:05Z",
      events_url: `/api/workspaces/default/runs/${runId}/events`,
      tasks: (entry?.tasks ?? []).map((task) => ({
        task_id: task.task_id,
        agent_kind: task.agent_kind ?? "rag",
        question: task.question,
        status: task.status ?? "queued",
        depends_on: task.depends_on ?? [],
        requirement_ids: task.requirement_ids ?? [],
        thread_id: `run:${runId}:task:${task.task_id}`,
        result_ref: null,
        // t25 起 run 详情带父任务 id（PRD 14.2 父子任务显示）。
        parent_task_id: task.parent_task_id ?? null,
      })),
      approvals: (entry?.approvals ?? []).map(approvalPayload),
      inputs: (entry?.inputs ?? []).map(inputPayload),
      image_refs: entry?.imageRefs ?? [],
      last_event_seq: lastSeq,
    };
  };

  /** run 端点派发（由上面的工作区路由调用）。 */
  const handleRunRoute = async (
    route: Route,
    url: URL,
    path: string,
  ): Promise<void> => {
    const method = route.request().method();
    const match = /\/api\/workspaces\/([^/]+)\/runs(?:\/([^/]+))?(?:\/([^/]+))?(?:\/([^/]+))?/.exec(path);
    const runId = match?.[2];
    const action = match?.[3];
    const actionId = match?.[4];

    // 创建 run：POST /api/workspaces/{w}/runs → **202 Accepted**（PRD 13.2）
    if (runId === undefined || action === undefined) {
      if (method === "GET") return json(route, detailOf(runId ?? "run-1"));
      if (options.gate !== undefined) await options.gate;
      runCounter += 1;
      const newRunId = `run-${runCounter}`;
      const frames =
        typeof options.frames === "function"
          ? options.frames(runCounter - 1)
          : (options.frames ?? [runStatus("running")]);
      // 回显请求里带的权限/思考设置与图片引用：前端「落到请求」的断言靠这里核。
      const body = (route.request().postDataJSON() ?? {}) as {
        permission_mode?: string | null;
        reasoning_level?: string | null;
        images?: string[];
      };
      stub.createRequests.push({
        permissionMode: body.permission_mode ?? null,
        reasoningLevel: body.reasoning_level ?? null,
      });
      const images = Array.isArray(body.images) ? body.images : [];
      stub.createImages.push(images);
      const imageRefs =
        typeof options.imageRefs === "function"
          ? options.imageRefs(images, runCounter - 1)
          : (options.imageRefs ?? defaultImageRefs(images));
      runs.set(newRunId, {
        frames,
        status: options.status ?? "running",
        approvals: options.approvals ?? [],
        inputs: options.inputs ?? [],
        tasks: options.tasks ?? [],
        permissionMode: body.permission_mode ?? "ask",
        reasoningLevel: body.reasoning_level ?? "high",
        imageRefs,
      });
      stub.createdRuns.push(newRunId);
      return json(route, detailOf(newRunId), 202);
    }

    // 事件流：GET .../runs/{r}/events
    if (action === "events") {
      const after = Number(url.searchParams.get("after") ?? "0");
      const frames = runs.get(runId)?.frames ?? [];
      return route.fulfill({
        status: 200,
        contentType: "text/event-stream",
        body: encodeV2Events(runId, frames, after),
      });
    }

    // JSON 快照：GET .../runs/{r}/event-log
    if (action === "event-log") {
      const frames = runs.get(runId)?.frames ?? [];
      return json(route, {
        events: frames.map((frame, index) => ({
          schema_version: SCHEMA_VERSION,
          run_id: runId,
          event_seq: index + 1,
          execution_id: "exec-1",
          task_id: frame.task_id ?? null,
          type: frame.type,
          created_at: "2026-09-19T10:00:00Z",
          payload: frame.payload,
        })),
      });
    }

    if (action === "cancel") {
      stub.cancels += 1;
      const entry = runs.get(runId);
      if (entry !== undefined) {
        entry.status = "cancelled";
        entry.frames = [...entry.frames, runStatus("cancelled"), runCancelled()];
      }
      return json(route, detailOf(runId));
    }

    if (action === "resume") {
      stub.resumes += 1;
      const entry = runs.get(runId);
      if (entry !== undefined) entry.status = "running";
      return json(route, detailOf(runId));
    }

    if (action === "approvals" && actionId !== undefined && method === "POST") {
      const bodyJson = (route.request().postDataJSON() ?? {}) as {
        decision?: string;
        edited_parameters?: unknown;
      };
      stub.approvalSubmits.push({
        approvalId: actionId,
        decision: bodyJson.decision ?? "",
        edited: bodyJson.edited_parameters ?? null,
      });
      const entry = runs.get(runId);
      if (entry !== undefined) entry.approvals = [];
      return json(route, { approval_id: actionId, status: "approved" });
    }

    if (action === "inputs" && actionId !== undefined && method === "POST") {
      const bodyJson = (route.request().postDataJSON() ?? {}) as { answer?: string };
      stub.inputSubmits.push({ inputRequestId: actionId, answer: bodyJson.answer ?? "" });
      const entry = runs.get(runId);
      if (entry !== undefined) entry.inputs = [];
      return json(route, { input_request_id: actionId, status: "answered" });
    }

    // run 详情：GET .../runs/{r}
    return json(route, detailOf(runId));
  };

  return stub;
}

/**
 * 加强验收断言（决定成败）：
 * - 至少一次命中 `/runs`（证明前端真的走了新任务接口）；
 * - 零次命中 `/api/chat`（证明旧入口已彻底迁移，stub 没有空跑）。
 */
export function expectRunOnlyTraffic(stub: RunStub): void {
  expect(stub.runRequestCount(), "应至少一次请求 /runs").toBeGreaterThanOrEqual(1);
  expect(stub.chatRequestCount(), "不得再请求 /api/chat").toBe(0);
}
