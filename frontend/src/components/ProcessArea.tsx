import { useState } from "react";
import type { ProcessEntry, ProcessState, ProcessTask } from "../types";
import { taskProgress } from "../run/processState";

/**
 * 过程区域（PRD 14.2）。
 *
 * 按时间顺序显示计划、Agent、工具、资料、批准、澄清、核对条目；研究任务可展开看
 * 子问题、读过的来源与结果摘要；并行任务分别显示状态，父任务展示整体完成情况。
 *
 * 条目文本全部由 `run/processState.ts` 从**实际事件**归约而来，本组件不编文案。
 */

const STATUS_GLYPH: Record<ProcessEntry["status"], string> = {
  running: "◐",
  done: "✓",
  failed: "✕",
  waiting: "!",
};

const STATUS_TEXT: Record<ProcessEntry["status"], string> = {
  running: "进行中",
  done: "已完成",
  failed: "失败",
  waiting: "等待处理",
};

function taskStatusClass(task: ProcessTask): string {
  if (task.status === "failed" || task.status === "cancelled") return "failed";
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
          {task.agentKind}
        </span>
        <span className="process-task-state" data-testid={`process-task-state-${task.taskId}`}>
          {task.status}
        </span>
      </button>
      {open && (
        <div className="process-task-detail" data-testid={`process-task-detail-${task.taskId}`}>
          <p className="process-task-detail-label">子问题</p>
          <p className="process-task-detail-value">{task.question || "（未提供子问题）"}</p>
          {task.dependsOn.length > 0 && (
            <>
              <p className="process-task-detail-label">依赖</p>
              <p className="process-task-detail-value">{task.dependsOn.join("、")}</p>
            </>
          )}
          <p className="process-task-detail-label">读过的来源（{task.sources.length}）</p>
          {task.sources.length === 0 ? (
            <p className="process-task-detail-value">尚未读取来源</p>
          ) : (
            <ul className="process-task-sources">
              {task.sources.map((source) => (
                <li key={source.evidence_id}>
                  <span className="process-source-type">{source.source_type}</span>
                  <span className="process-source-title">{source.title || source.source_ref}</span>
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
  /** 初始是否展开（等待批准/澄清时默认展开，让用户立刻看到）。 */
  defaultOpen?: boolean;
}

/** 过程区域：折叠容器 + 条目时间线 + 研究任务列表。 */
export function ProcessArea({ process, defaultOpen = false }: ProcessAreaProps) {
  const [open, setOpen] = useState(defaultOpen);
  const entries = process.entries;
  if (entries.length === 0 && process.tasks.length === 0) return null;

  const progress = taskProgress(process.tasks);
  const running = entries.some((entry) => entry.status === "running");

  return (
    <section className={`process-area${open ? " open" : ""}`} data-testid="process-area">
      <button
        type="button"
        className="process-toggle"
        aria-expanded={open}
        data-testid="process-toggle"
        onClick={() => setOpen((value) => !value)}
      >
        <span className={`process-toggle-dot${running ? " active" : ""}`} aria-hidden="true" />
        <span className="process-toggle-label">
          {running ? "处理中" : "处理过程"}（{entries.length} 步
          {process.tasks.length > 0 ? ` · ${progress.text}` : ""}）
        </span>
        <span className="process-chevron" aria-hidden="true">
          {open ? "−" : "+"}
        </span>
      </button>
      {open && (
        <div className="process-body">
          {process.tasks.length > 0 && (
            <div className="process-tasks" data-testid="process-tasks">
              <p className="process-section-label">
                研究任务（父任务整体完成情况：<span data-testid="process-task-progress">{progress.text}</span>）
              </p>
              <ul className="process-task-list">
                {process.tasks.map((task) => (
                  <ResearchTaskItem key={task.taskId} task={task} />
                ))}
              </ul>
            </div>
          )}
          <ol className="process-entries" data-testid="process-entries">
            {entries.map((entry) => (
              <li
                key={entry.id}
                className={`process-entry process-entry-${entry.status}`}
                data-kind={entry.kind}
                data-testid={`process-entry-${entry.kind}`}
              >
                <span className="process-entry-glyph" aria-hidden="true">
                  {STATUS_GLYPH[entry.status]}
                </span>
                <span className="process-entry-text">{entry.text}</span>
                <span className="process-entry-status">{STATUS_TEXT[entry.status]}</span>
                {(entry.details ?? []).length > 0 && (
                  <ul className="process-entry-details">
                    {(entry.details ?? []).map((detail, index) => (
                      <li key={`${entry.id}-${index}`}>{detail}</li>
                    ))}
                  </ul>
                )}
              </li>
            ))}
          </ol>
        </div>
      )}
    </section>
  );
}

export default ProcessArea;
