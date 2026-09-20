import type {
  AgentKind,
  EvidenceSummary,
  PlanTaskSummary,
  ProcessEntry,
  ProcessEntryKind,
  ProcessState,
  ProcessTask,
  RetrievalMeta,
  RunEventTyped,
  RunStatus,
  TaskStatus,
  ToolCompletedPayload,
  ToolStartedPayload,
  VerificationResult,
  VerifierVerdict,
} from "../types";

/**
 * 过程区域归约器（PRD 14.2）。
 *
 * 纯函数：`applyRunEvent(state, event) -> state`。过程条目**全部由实际事件生成**，
 * 不预置文案 —— 也就是说，界面显示的「正在查询风控指标定义」「正在读取对应章节」
 * 只能来自 `tool.started` / `agent.started` 的真实负载。
 *
 * 去重：以 `event_seq` 为游标（调用方也可在订阅层先去重，两层都幂等）。
 */

/** 工具 → 行动说明（PRD 14.2：写成行动句，而不是工具名）。 */
export const TOOL_ACTION_LABELS: Record<string, string> = {
  lookup_catalog: "正在查询知识目录",
  get_workspace_overview: "正在获取工作区概览",
  search_knowledge: "正在检索知识库",
  // 相关性筛选（PRD §8.2）：评分期间不产出任何正文片段，进度只能靠工具事件先行上报
  // （集成说明：`main` 的 ChatService 用 `/api/chat` 的 `tool=rerank` 帧，这里改由运行层
  // 发布的 `tool.started` / `tool.completed`（tool_name="rerank"）驱动，事件类型与负载
  // 均沿用冻结契约里的既有字段）。
  rerank: "正在筛选资料",
  read_document: "正在读取对应章节",
  search_history: "正在检索历史会话",
  read_history: "正在读取历史原文",
  web_search: "正在联网搜索",
  read_web_page: "正在读取网页原文",
  verify_image: "正在核对图片",
  ask_user: "正在向你确认",
};

/**
 * 相关性筛选进度事件的专用收口文案。
 *
 * 后端发布 rerank 进度时只带 `tool_name` / `status` / `error_code`（`result_count`
 * 与 `source_count` 未发布、取模型默认 0），所以**不能**把 0 当成「没有资料通过筛选」。
 * 入选资料以本轮来源为准（见 `SourceFooter`）。
 */
const RERANK_DONE_TEXT = "筛选资料：已完成（入选资料见下方来源）";

/** Agent 种类 → 行动说明。 */
export const AGENT_ACTION_LABELS: Record<string, string> = {
  intent: "正在理解问题",
  planner: "正在拆解研究任务",
  general: "正在组织通用回答",
  rag: "正在本地知识中查证",
  web: "正在联网核实",
  synthesis: "正在汇总结论",
  verifier: "正在核对答案",
  approval_reviewer: "正在评估操作影响",
  memory_extractor: "正在整理工作区记忆",
};

/** 工具完成后按结果数量补一句明确说明（0 结果也必须说清楚）。 */
export function toolResultText(payload: ToolCompletedPayload): string {
  const action = TOOL_ACTION_LABELS[payload.tool_name] ?? `正在执行 ${payload.tool_name}`;
  if (payload.status !== "succeeded") {
    const reason = payload.error_code === "" ? "执行失败" : `执行失败（${payload.error_code}）`;
    return `${action.replace("正在", "")}：${reason}`;
  }
  // 筛选进度事件不发布计数（见 RERANK_DONE_TEXT）：按「已完成」收口，不编造条数。
  if (payload.tool_name === "rerank") return RERANK_DONE_TEXT;
  if (payload.result_count === 0) {
    return `${action.replace("正在", "")}：未找到可用资料`;
  }
  const sourceNote = payload.source_count > 0 ? `，来自 ${payload.source_count} 个来源` : "";
  return `${action.replace("正在", "")}：命中 ${payload.result_count} 条资料${sourceNote}`;
}

/**
 * 相关性筛选完成事件 → 本轮筛选状态（PRD §8.2 的三态）。
 *
 * 仅供 `tool.completed`（tool_name="rerank"）使用；其余工具返回 null。
 *
 * 为什么不区分 `filtered` / `empty`：筛选进度事件**不发布** `result_count`
 * （取默认 0），把它当「0 条入选」会把「已筛选出资料」误报成「没有资料通过筛选」。
 * 因此这里只断言**筛选流程已成功完成**（`filtered`）——此时正文来源正是该工具的
 * 入选结果；未入选则本轮没有知识来源、脚注不渲染。失败时不改状态：真实错误由
 * `tool.completed.error_code` 与 `run.failed` 如实展示，绝不把失败说成「未启用」。
 * `disabled` 由 `null` 表达（前端按来源自身字段判定）。
 *
 * 契约缺口（已上报 captain）：`result_count` / 每来源的 `rerank_score` 等字段目前
 * 只到工具结果层，未进入事件流，故本函数不涉及它们。
 */
export function retrievalFromRerankTool(
  payload: ToolCompletedPayload,
): RetrievalMeta | null {
  if (payload.tool_name !== "rerank") return null;
  if (payload.status !== "succeeded") return null;
  return { status: "filtered", profile_id: null };
}

/**
 * 过程区域里**正在进行**的最后一步文案（进度提示用）。
 *
 * 纯函数：只读 `entries`（全部由实际事件归约而来），无正在进行的步骤时返回 null。
 *
 * 优先取正在进行的**工具**步骤：阶段叙述（`run.status` 的 message）可能整轮都不再收到
 * 收口事件，若让它盖住工具步骤，用户会在筛选期间看到「正在检索知识库」而不是真正的
 * 当前动作（PRD §8.2：评分期间必须如实显示「正在筛选资料」）。
 */
export function runningEntryText(state: ProcessState): string | null {
  const running = state.entries.filter((entry) => entry.status === "running");
  const tool = running.filter((entry) => entry.kind === "tool").at(-1);
  const latest = running.at(-1);
  return (tool ?? latest)?.text ?? null;
}

/** 核对结论 → 行动说明（PRD 8.2 / 14.2）。 */
export function verificationText(result: VerificationResult): string {
  const verdict: VerifierVerdict = result.verdict;
  switch (verdict) {
    case "pass":
      return "核对通过";
    case "revise":
      return "核对发现需要修正的陈述";
    case "research":
      return "发现资料差异，继续查阅发布说明";
    case "clarify":
      return "需要向你确认后才能继续";
    case "insufficient":
      return "现有资料不足以支持结论";
    default:
      return "核对进行中";
  }
}

function nextSeq(state: ProcessState, seq: number): boolean {
  return seq > state.lastSeq;
}

function upsertEntry(state: ProcessState, entry: ProcessEntry): ProcessState {
  const index = state.entries.findIndex((item) => item.id === entry.id);
  if (index < 0) {
    return { ...state, entries: [...state.entries, entry] };
  }
  const entries = state.entries.slice();
  entries[index] = entry;
  return { ...state, entries };
}

function upsertTask(state: ProcessState, task: ProcessTask): ProcessState {
  const index = state.tasks.findIndex((item) => item.taskId === task.taskId);
  if (index < 0) return { ...state, tasks: [...state.tasks, task] };
  const tasks = state.tasks.slice();
  tasks[index] = task;
  return { ...state, tasks };
}

function taskFromSummary(summary: PlanTaskSummary): ProcessTask {
  return {
    taskId: summary.task_id,
    question: summary.question,
    agentKind: summary.agent_kind,
    status: summary.status,
    dependsOn: summary.depends_on ?? [],
    // `PlanTaskSummary` 未携带 parent_task_id（契约缺口，见交付说明）：
    // 依赖关系是唯一可得的父子线索。
    parentTaskId: null,
    childTaskIds: [],
    sources: [],
    resultSummary: "",
  };
}

/** 空状态。 */
export function emptyProcessState(): ProcessState {
  return { entries: [], tasks: [], threads: [], lastSeq: 0 };
}

/**
 * 记下本轮**实际使用**的权限模式与思考深度（PRD 14.1 末句）。
 *
 * 记的是随请求真正发给后端的那份设置，不是菜单里的「当前草稿」——否则用户改了菜单
 * 但请求没带上时，过程区域会显示一个并未生效的值。
 * 不推进 `lastSeq`（这不是事件），因此不会影响后续事件的去重。
 */
export function withRunSettings(state: ProcessState, text: string): ProcessState {
  const existing = state.entries.find((item) => item.id === "settings");
  const entry: ProcessEntry = {
    id: "settings",
    kind: "status",
    text,
    seq: existing?.seq ?? 0,
    at: existing?.at ?? "",
    status: "done",
    taskId: null,
    tag: "settings",
  };
  const index = state.entries.findIndex((item) => item.id === "settings");
  if (index < 0) return { ...state, entries: [...state.entries, entry] };
  const entries = state.entries.slice();
  entries[index] = entry;
  return { ...state, entries };
}

/**
 * 记下本轮**承接自哪一次执行**（PRD 5.4 末句）。
 *
 * 会话被移动到别的工作区后，原区的执行已停止、历史只读；新执行必须让用户看见承接
 * 关系，否则会以为历史凭空消失。后端接受 `origin_run_id` 但**不回显**，所以这个值
 * 只能由前端带过来。不推进 `lastSeq`（不是事件）。
 */
export function withOriginRun(state: ProcessState, originRunId: string): ProcessState {
  const entry: ProcessEntry = {
    id: "origin-run",
    kind: "status",
    text: `本次执行承接自原工作区的执行 ${originRunId}`,
    seq: 0,
    at: "",
    status: "done",
    taskId: null,
    tag: "origin_run",
  };
  const index = state.entries.findIndex((item) => item.id === "origin-run");
  if (index < 0) return { ...state, entries: [...state.entries, entry] };
  const entries = state.entries.slice();
  entries[index] = entry;
  return { ...state, entries };
}

/** 按序归约一批事件（首屏 `event-log` 快照用它）。 */
export function reduceRunEvents(
  events: RunEventTyped[],
  initial: ProcessState = emptyProcessState(),
): ProcessState {
  return events.reduce<ProcessState>((state, event) => applyRunEvent(state, event), initial);
}

/** 终态说明（PRD 14.2：`completed_with_gaps` 与 `failed` 各有自己的说明）。 */
export function terminalStatusText(status: RunStatus): string {
  switch (status) {
    case "completed":
      return "已完成并核对通过";
    case "completed_with_gaps":
      return "已完成，但部分资料不足（已在正文标明影响）";
    case "failed":
      return "执行失败（已完成的研究记录仍可查看，可从此处继续）";
    case "cancelled":
      return "已停止";
    case "interrupted":
      return "已中断，可继续";
    case "waiting_approval":
      return "等待你的批准";
    case "waiting_input":
      return "等待你的回答";
    case "waiting_children":
      return "等待并行子任务完成";
    case "cancelling":
      return "正在停止…";
    default:
      return "正在处理";
  }
}

/** 归约单个事件（纯函数，重复 `event_seq` 幂等）。 */
export function applyRunEvent(state: ProcessState, event: RunEventTyped): ProcessState {
  if (!nextSeq(state, event.event_seq)) return state;
  const base: ProcessState = { ...state, lastSeq: event.event_seq };
  const seq = event.event_seq;
  const at = event.created_at;
  const entry = (partial: Omit<ProcessEntry, "seq" | "at">): ProcessEntry => ({
    ...partial,
    seq,
    at,
  });

  switch (event.type) {
    case "run.status": {
      const payload = event.payload;
      // 有 message 的是**进度叙述**：独立条目，保留历史（每条叙述都是一步）。
      // 没有 message 的只维护「当前状态」这一条（按 id 覆盖，不刷屏）。
      if (payload.message !== "") {
        return upsertEntry(
          base,
          entry({
            id: `status-note:${seq}`,
            kind: "status",
            text: payload.message,
            status: payload.status === "failed" ? "failed" : "running",
            taskId: event.task_id,
            tag: payload.status,
            details: payload.phase === "" ? [] : [`阶段：${payload.phase}`],
          }),
        );
      }
      return upsertEntry(
        base,
        entry({
          id: "status",
          kind: "status",
          text: terminalStatusText(payload.status),
          status:
            payload.status === "failed"
              ? "failed"
              : payload.status === "cancelling" || payload.status === "running"
                ? "running"
                : "done",
          taskId: event.task_id,
          tag: payload.status,
          details: payload.phase === "" ? [] : [`阶段：${payload.phase}`],
        }),
      );
    }

    case "route.decided": {
      const decision = event.payload.decision;
      const chosen: string[] = [];
      if (decision.needs_knowledge) chosen.push("本地知识");
      if (decision.needs_web) chosen.push("联网核实");
      if (decision.needs_history) chosen.push("历史会话");
      if (decision.needs_general) chosen.push("通用解释");
      const text =
        decision.reason_summary !== ""
          ? decision.reason_summary
          : `处理路径：${chosen.length > 0 ? chosen.join(" + ") : "直接回答"}`;
      return upsertEntry(
        base,
        entry({
          id: "route",
          kind: "route",
          text,
          status: "done",
          taskId: null,
          tag: decision.intent,
          details: [`来源：${chosen.length > 0 ? chosen.join("、") : "无外部资料"}`],
        }),
      );
    }

    case "plan.updated": {
      const payload = event.payload;
      let next = base;
      for (const summary of payload.tasks ?? []) {
        next = upsertTask(next, taskFromSummary(summary));
      }
      const count = (payload.tasks ?? []).length;
      const details: string[] = [];
      if (count > 0) details.push(`共 ${count} 个研究任务`);
      if ((payload.uncovered_requirements ?? []).length > 0) {
        details.push(`尚无任务覆盖的要求：${(payload.uncovered_requirements ?? []).join("、")}`);
      }
      return upsertEntry(
        next,
        entry({
          id: "plan",
          kind: "plan",
          text: count > 0 ? `已拆解为 ${count} 个子问题` : "无需拆分，直接处理",
          status: "done",
          taskId: null,
          details,
        }),
      );
    }

    case "agent.started": {
      const payload = event.payload;
      const action = AGENT_ACTION_LABELS[payload.agent_kind] ?? "正在处理";
      let next = base;
      if (payload.thread_id !== "" && !next.threads.includes(payload.thread_id)) {
        next = { ...next, threads: [...next.threads, payload.thread_id] };
      }
      if (payload.task_id !== null) {
        const existing = next.tasks.find((item) => item.taskId === payload.task_id);
        next = upsertTask(next, {
          taskId: payload.task_id,
          question: payload.question !== "" ? payload.question : (existing?.question ?? ""),
          agentKind: payload.agent_kind,
          status: "running",
          dependsOn: existing?.dependsOn ?? [],
          parentTaskId: existing?.parentTaskId ?? null,
          childTaskIds: existing?.childTaskIds ?? [],
          sources: existing?.sources ?? [],
          resultSummary: existing?.resultSummary ?? "",
        });
      }
      const question = payload.question !== "" ? `：${payload.question}` : "";
      return upsertEntry(
        next,
        entry({
          // 稳定 id：同一 (任务, Agent) 只占一条，`agent.completed` 能原地收口。
          id: `agent:${payload.task_id ?? "main"}:${payload.agent_kind}`,
          kind: "agent",
          text: `${action}${question}`,
          status: "running",
          taskId: payload.task_id,
          tag: payload.agent_kind,
        }),
      );
    }

    case "agent.completed": {
      const payload = event.payload;
      const action = (AGENT_ACTION_LABELS[payload.agent_kind] ?? "正在处理").replace("正在", "已");
      const taskId = event.task_id;
      let next = base;
      if (taskId !== null) {
        const existing = next.tasks.find((item) => item.taskId === taskId);
        if (existing) {
          next = upsertTask(next, {
            ...existing,
            status: payload.status,
            resultSummary: payload.summary !== "" ? payload.summary : existing.resultSummary,
          });
        }
      }
      return upsertEntry(
        next,
        entry({
          id: `agent:${taskId ?? "main"}:${payload.agent_kind}`,
          kind: "agent",
          text: payload.summary !== "" ? payload.summary : action,
          status: payload.status === "failed" ? "failed" : "done",
          taskId,
          tag: payload.agent_kind,
        }),
      );
    }

    case "tool.started": {
      const payload: ToolStartedPayload = event.payload;
      const action = TOOL_ACTION_LABELS[payload.tool_name] ?? `正在执行 ${payload.tool_name}`;
      const details =
        payload.args_summary === "" ? [] : [`参数：${payload.args_summary}`];
      return upsertEntry(
        base,
        entry({
          id: `tool:${payload.operation_id || payload.tool_name}`,
          kind: "tool",
          text: action,
          status: "running",
          taskId: payload.task_id,
          tag: payload.tool_name,
          details,
        }),
      );
    }

    case "tool.completed": {
      const payload: ToolCompletedPayload = event.payload;
      return upsertEntry(
        base,
        entry({
          id: `tool:${payload.operation_id || payload.tool_name}`,
          kind: "tool",
          text: toolResultText(payload),
          status: payload.status === "succeeded" ? "done" : "failed",
          taskId: event.task_id,
          tag: payload.tool_name,
          details:
            payload.error_code === "" ? [] : [`错误码：${payload.error_code}`],
        }),
      );
    }

    case "evidence.added": {
      const items: EvidenceSummary[] = event.payload.items ?? [];
      let next = base;
      if (event.task_id !== null) {
        const existing = next.tasks.find((item) => item.taskId === event.task_id);
        if (existing) {
          const known = new Set(existing.sources.map((item) => item.evidence_id));
          const merged = [
            ...existing.sources,
            ...items.filter((item) => !known.has(item.evidence_id)),
          ];
          next = upsertTask(next, { ...existing, sources: merged });
        }
      }
      const titles = items
        .map((item) => item.title || item.source_ref)
        .filter((title) => title !== "");
      return upsertEntry(
        next,
        entry({
          id: `evidence:${event.event_seq}`,
          kind: "evidence",
          text: `找到 ${items.length} 条资料`,
          status: "done",
          taskId: event.task_id,
          details: titles,
        }),
      );
    }

    case "approval.requested": {
      const approval = event.payload.approval;
      return upsertEntry(
        base,
        entry({
          id: `approval:${approval.approval_id}`,
          kind: "approval",
          text: approval.display.purpose || `需要批准：${approval.display.tool_name}`,
          status: "waiting",
          taskId: approval.task_id,
          tag: approval.display.tool_name,
          details: (approval.display.query_or_urls ?? []).slice(0, 3),
        }),
      );
    }

    case "approval.resolved": {
      const payload = event.payload;
      const label =
        payload.decision_kind === "deny"
          ? "已拒绝该操作"
          : payload.decision_kind === "allow_scope"
            ? "已允许本轮同类操作"
            : "已允许本次操作";
      return upsertEntry(
        base,
        entry({
          id: `approval:${payload.approval_id}`,
          kind: "approval",
          text: label,
          status: payload.decision_kind === "deny" ? "failed" : "done",
          taskId: event.task_id,
        }),
      );
    }

    case "input.requested": {
      const request = event.payload.request;
      return upsertEntry(
        base,
        entry({
          id: `input:${request.input_request_id}`,
          kind: "input",
          text: request.question,
          status: "waiting",
          taskId: request.task_id,
          details: request.options ?? [],
        }),
      );
    }

    case "input.resolved": {
      const payload = event.payload;
      return upsertEntry(
        base,
        entry({
          id: `input:${payload.input_request_id}`,
          kind: "input",
          text: `已澄清：${payload.answer}`,
          status: "done",
          taskId: event.task_id,
        }),
      );
    }

    case "verification.updated": {
      const result = event.payload.result;
      const details: string[] = [];
      if ((result.conflicts ?? []).length > 0) {
        details.push(`发现版本差异：${(result.conflicts ?? []).join("；")}`);
      }
      for (const issue of result.issues ?? []) {
        details.push(`${issue.check}：${issue.required_action || issue.statement}`);
      }
      const pending = event.payload.pending ?? [];
      if (pending.length > 0) details.push(`待补充：${pending.join("；")}`);
      return upsertEntry(
        base,
        entry({
          id: "verification",
          kind: "verification",
          text: verificationText(result),
          status: result.verdict === "pass" ? "done" : "waiting",
          taskId: event.task_id,
          tag: result.verdict,
          details: details.slice(0, 8),
        }),
      );
    }

    case "memory.updated": {
      const payload = event.payload;
      return upsertEntry(
        base,
        entry({
          id: "memory",
          kind: "memory",
          text: `工作区记忆已更新（新增 ${payload.entries_added} 条，更新 ${payload.entries_updated} 条）`,
          status: "done",
          taskId: null,
        }),
      );
    }

    case "memory.failed": {
      return upsertEntry(
        base,
        entry({
          id: "memory",
          kind: "memory",
          text: `记忆维护失败：${event.payload.reason}（答案本身已保留）`,
          status: "failed",
          taskId: null,
        }),
      );
    }

    case "run.completed": {
      const payload = event.payload;
      const gaps = payload.gaps ?? [];
      const text =
        payload.status === "completed_with_gaps"
          ? terminalStatusText("completed_with_gaps")
          : terminalStatusText("completed");
      return upsertEntry(
        base,
        entry({
          id: "status",
          kind: "status",
          text,
          status: "done",
          taskId: null,
          tag: payload.status,
          details: gaps,
        }),
      );
    }

    case "run.failed": {
      const error = event.payload.error;
      return upsertEntry(
        base,
        entry({
          id: "status",
          kind: "error",
          text: `${error.message || "执行失败"}${error.resumable ? "（可从此处继续）" : ""}`,
          status: "failed",
          taskId: null,
          tag: error.error_code,
          details: error.required_action === "" ? [] : [`建议：${error.required_action}`],
        }),
      );
    }

    case "run.cancelled": {
      return upsertEntry(
        base,
        entry({
          id: "status",
          kind: "status",
          text: event.payload.resumable ? "已停止（可继续）" : "已停止",
          status: "done",
          taskId: null,
          tag: event.payload.stop_reason,
        }),
      );
    }

    case "answer.meta":
    case "answer.delta": {
      // 正文与来源走答案通道，不产生过程条目（PRD 14.2 正文在核对完成后展示）。
      return base;
    }

    default:
      return base;
  }
}

/** 父任务整体完成情况（PRD 14.2：父任务展示整体完成情况）。 */
export function taskProgress(tasks: ProcessTask[]): {
  total: number;
  done: number;
  running: number;
  waiting: number;
  failed: number;
  text: string;
} {
  const count = (predicate: (status: TaskStatus) => boolean): number =>
    tasks.filter((task) => predicate(task.status)).length;
  const done = count((status) => status === "completed" || status === "completed_with_gaps");
  const running = count((status) => status === "running" || status === "queued");
  const waiting = count(
    (status) =>
      status === "waiting_approval" || status === "waiting_input" || status === "waiting_children",
  );
  const failed = count((status) => status === "failed" || status === "cancelled");
  const total = tasks.length;
  const parts: string[] = [`${done}/${total} 已完成`];
  if (running > 0) parts.push(`${running} 进行中`);
  if (waiting > 0) parts.push(`${waiting} 等待处理`);
  if (failed > 0) parts.push(`${failed} 失败`);
  return { total, done, running, waiting, failed, text: parts.join("，") };
}

/** 已识别的 Agent 种类是否属于研究任务（过程区域可展开的研究任务）。 */
export function isResearchAgent(kind: AgentKind | string): boolean {
  return kind === "rag" || kind === "web" || kind === "general";
}
