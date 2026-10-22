import { useEffect, useState } from "react";
import type { ProcessEntry, ProcessEntryKind, ProcessState, ProcessTask } from "../types";
import { taskProgress, terminalStatusText } from "../run/processState";
import { formatActivityDuration } from "../run/elapsed";

/**
 * 过程时间线（PRD 14.2）。
 *
 * 每个步骤展示实际动作与结果，展开后查看来源正文。
 * 模型思考按调用分别追加，整个过程支持折叠与事件恢复。
 *
 * 三条不变量：
 * 1. 条目文本全部由 `run/processState.ts` 从**实际事件**归约而来，本组件不编文案；
 * 2. 行内文字单行截断（`title` 保留全文），保证环节多时也是一屏能扫完的清单；
 * 3. 研究任务（子问题 / 读过的来源 / 结果摘要）仍按 PRD 14.2 保留可展开能力。
 */

/** Agent 种类 → 用户可读职责名。 */
const AGENT_LABELS: Record<string, string> = {
  intent: "理解问题", planner: "安排任务", general: "通用回答", rag: "知识检索",
  web: "联网查询", synthesis: "整理回答", verifier: "核对回答",
  approval_reviewer: "权限确认", memory_extractor: "保存记忆",
};

/** 工具名 → 行首短标签（对齐「Grep · 参数」式单行摘要）。 */
const TOOL_LABELS: Record<string, string> = {
  lookup_catalog: "目录",
  get_workspace_overview: "工作区",
  search_knowledge: "检索",
  rerank: "筛选",
  read_document: "读取",
  search_history: "历史检索",
  read_history: "历史读取",
  web_search: "联网",
  read_web_page: "网页",
  verify_image: "图片",
  ask_user: "确认",
};

/** 条目种类 → 行首标签（工具 / Agent 命中时优先用更具体的标签）。 */
const KIND_LABELS: Record<ProcessEntryKind, string> = {
  route: "路径",
  plan: "计划",
  agent: "Agent",
  tool: "工具",
  evidence: "资料",
  approval: "批准",
  input: "澄清",
  verification: "核对",
  status: "状态",
  memory: "记忆",
  error: "错误",
};

const STATUS_GLYPH: Record<ProcessEntry["status"], string> = {
  running: "◐",
  done: "✓",
  failed: "✕",
  waiting: "!",
  cancelled: "−",
};

const STATUS_TEXT: Record<ProcessEntry["status"], string> = {
  running: "进行中",
  done: "已完成",
  failed: "失败",
  waiting: "等待处理",
  cancelled: "已终止",
};

/** 行首标签：工具看工具名，Agent 看职责，其余看条目种类。 */
function entryLabel(entry: ProcessEntry): string {
  if (entry.kind === "tool") {
    return TOOL_LABELS[entry.tag ?? ""] ?? entry.tag ?? KIND_LABELS.tool;
  }
  if (entry.kind === "agent") {
    return AGENT_LABELS[entry.tag ?? ""] ?? KIND_LABELS.agent;
  }
  return KIND_LABELS[entry.kind] ?? "步骤";
}

function taskStatusClass(task: ProcessTask): string {
  if (task.status === "failed") return "failed";
  if (task.status === "cancelled") return "cancelled";
  if (task.status === "completed" || task.status === "completed_with_gaps") return "done";
  if (
    task.status === "waiting_approval" ||
    task.status === "waiting_input" ||
    task.status === "waiting_children"
  ) {
    return "waiting";
  }
  return "running";
}

/** 单个环节：一行摘要 + 独立折叠的细节。 */
function ProcessRow({ entry }: { entry: ProcessEntry }) {
  const [open, setOpen] = useState(false);
  const details = entry.details ?? [];
  return (
    <li
      className={`process-row process-row-${entry.status}`}
      data-kind={entry.kind}
      data-testid={`process-entry-${entry.kind}`}
    >
      <button
        type="button"
        className="process-row-header"
        data-testid="process-row-toggle"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <span className="process-row-glyph" aria-hidden="true">
          {STATUS_GLYPH[entry.status]}
        </span>
        <span className="process-row-kind">{entryLabel(entry)}</span>
        <span className="process-row-sep" aria-hidden="true">
          ·
        </span>
        <span className="process-row-text" title={entry.text}>
          {entry.text}
        </span>
        <span className="process-row-status">{STATUS_TEXT[entry.status]}</span>
        <span className={`process-row-chevron${open ? " open" : ""}`} aria-hidden="true">
          ⌄
        </span>
      </button>
      {open && (
        <div className="process-row-detail" data-testid="process-row-detail">
          {details.length > 0 && (
            <ul className="process-row-detail-list">
              {details.map((detail, index) => (
                <li key={`${entry.id}-${index}`}>{detail}</li>
              ))}
            </ul>
          )}
          {entry.sources?.map((source, index) => (
            <details className="process-source" key={`${entry.id}:source:${index}`}>
              <summary>{source.title}</summary>
              <div className="process-source-body">{source.text || "本次结果没有附带正文。"}</div>
            </details>
          ))}
          {details.length === 0 && !entry.sources?.length && (
            <p className="process-row-detail-text">{entry.text}</p>
          )}
        </div>
      )}
      {entry.reasoning?.map((stream, index) => (
        <details className="process-reasoning" key={stream.streamId} data-testid="process-reasoning">
          <summary>模型思考{index > 0 ? `（第 ${index + 1} 次调用）` : ""}</summary>
          <div className="process-reasoning-body">{stream.text}</div>
        </details>
      ))}
    </li>
  );
}

/** 研究任务：子问题 / 依赖 / 读过的来源 / 结果摘要（PRD 14.2 保留可展开）。 */
function ResearchTaskItem({ task }: { task: ProcessTask }) {
  const [open, setOpen] = useState(false);
  const state = taskStatusClass(task);
  return (
    <li className={`process-task process-task-${state}`} data-testid={`process-task-${task.taskId}`}>
      <button
        type="button"
        className="process-task-header"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <span className="process-task-glyph" aria-hidden="true">
          {STATUS_GLYPH[state === "done" ? "done" : state === "failed" ? "failed" : state === "waiting" ? "waiting" : "running"]}
        </span>
        <span className="process-task-question">{task.question || task.taskId}</span>
        <span className="process-task-agent" data-testid={`process-task-agent-${task.taskId}`}>
          {AGENT_LABELS[task.agentKind] ?? task.agentKind}
        </span>
        <span className="process-task-state" data-testid={`process-task-state-${task.taskId}`}>
          {terminalStatusText(task.status)}
        </span>
      </button>
      {open && (
        <div className="process-task-detail" data-testid={`process-task-detail-${task.taskId}`}>
          <p className="process-task-detail-label">子问题</p>
          <p className="process-task-detail-value">{task.question || "（未提供子问题）"}</p>
          <p className="process-task-detail-label">读过的来源（{task.sources.length}）</p>
          {task.sources.length === 0 ? (
            <p className="process-task-detail-value">尚未读取来源</p>
          ) : (
            <ul className="process-task-sources">
              {task.sources.map((source) => (
                <li key={source.evidence_id}>
                  <span className="process-source-title">
                    {source.title && source.title !== source.source_ref ? source.title : source.snippet || "相关资料"}
                  </span>
                </li>
              ))}
            </ul>
          )}
          <p className="process-task-detail-label">结果摘要</p>
          <p className="process-task-detail-value">
            {task.resultSummary || "（该任务尚未返回摘要）"}
          </p>
        </div>
      )}
    </li>
  );
}

export interface ProcessAreaProps {
  process: ProcessState;
  /** 当前执行区间仍在运行时，每秒更新用时。 */
  live?: boolean;
}

/** 过程默认折叠为进度摘要，展开和收起由用户点击控制。 */
export function ProcessArea({ process, live = false }: ProcessAreaProps) {
  const entries = process.entries.filter((entry) => entry.kind !== "evidence" && entry.id !== "settings");
  const running = entries.some((entry) => entry.status === "running");
  const [open, setOpen] = useState(false);
  const [now, setNow] = useState<number>(() => Date.now());

  useEffect(() => {
    if (!live) return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [live]);

  if (entries.length === 0 && process.tasks.length === 0) return null;

  const progress = taskProgress(process.tasks);
  const duration = formatActivityDuration(process.activity ?? [], live ? now : null);
  const summary = [
    `${entries.length} 步`,
    process.tasks.length > 0 ? progress.text : null,
    duration === null ? null : `用时 ${duration}`,
  ]
    .filter((part): part is string => part !== null)
    .join(" · ");

  return (
    <section className={`process-area${open ? " process-area-open" : " process-area-collapsed"}`} data-testid="process-area">
      <button
        type="button"
        className="process-head"
        data-testid="process-head"
        aria-expanded={open}
        onClick={() => {
          setOpen((value) => !value);
        }}
      >
        <span className={`process-head-dot${running ? " active" : ""}`} aria-hidden="true" />
        <span className="process-head-label">
          {running ? "处理中" : "处理过程"}（{summary}）
        </span>
        <span
          className={`process-head-chevron${open ? " open" : ""}`}
          aria-hidden="true"
        >
          ⌄
        </span>
      </button>
      {open && (
        <>
          {process.tasks.length > 0 && (
            <details className="process-tasks" data-testid="process-tasks">
              <summary className="process-section-label">
                查询的问题（<span data-testid="process-task-progress">{progress.text}</span>）
              </summary>
              <ul className="process-task-list">
                {process.tasks.map((task) => (
                  <ResearchTaskItem key={task.taskId} task={task} />
                ))}
              </ul>
            </details>
          )}
          <ol className="process-timeline" data-testid="process-entries">
            {entries.map((entry) => (
              <ProcessRow key={entry.id} entry={entry} />
            ))}
          </ol>
        </>
      )}
    </section>
  );
}

export default ProcessArea;
