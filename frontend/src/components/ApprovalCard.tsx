import { useState } from "react";
import type { ApprovalDecisionKind, ApprovalRequest, InputRequest } from "../types";

/**
 * 批准与澄清卡片（PRD 14.3）。
 *
 * 卡片直接位于对话中，待处理时保持可见；展示操作目的、工具名、将发送的查询或
 * URL、日期范围、所属子任务与人名脱敏提示；按钮「允许本次」「允许本轮同类操作」
 * 「拒绝」+ 编辑查询入口。
 */

function editableKey(approval: ApprovalRequest): string {
  const parameters = approval.display.parameters ?? {};
  if ("query" in parameters) return "query";
  if ("url" in parameters) return "url";
  if ("urls" in parameters) return "urls";
  return "query";
}

function editableValue(approval: ApprovalRequest): string {
  const parameters = approval.display.parameters ?? {};
  const key = editableKey(approval);
  const value = parameters[key];
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.join("\n");
  return (approval.display.query_or_urls ?? []).join("\n");
}

export interface ApprovalCardProps {
  approval: ApprovalRequest;
  busy?: boolean;
  error?: string | null;
  onDecide: (
    approvalId: string,
    decision: ApprovalDecisionKind,
    editedParameters?: Record<string, unknown> | null,
  ) => void;
}

export function ApprovalCard({ approval, busy = false, error = null, onDecide }: ApprovalCardProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(() => editableValue(approval));
  const display = approval.display;

  const decide = (decision: ApprovalDecisionKind): void => {
    if (!editing) {
      onDecide(approval.approval_id, decision);
      return;
    }
    const key = editableKey(approval);
    const raw = draft.trim();
    const value: unknown =
      key === "urls" ? raw.split(/\r?\n/).map((line) => line.trim()).filter((line) => line !== "") : raw;
    onDecide(approval.approval_id, decision, { [key]: value });
  };

  return (
    <div className="chat-card approval-card" data-testid={`approval-card-${approval.approval_id}`}>
      <div className="chat-card-head">
        <span className="chat-card-badge">需要批准</span>
        <span className="chat-card-tool" data-testid="approval-tool">
          {display.tool_name}
        </span>
      </div>
      <p className="chat-card-purpose" data-testid="approval-purpose">
        {display.purpose}
      </p>
      <dl className="chat-card-fields">
        <div>
          <dt>将发送的查询或地址</dt>
          <dd data-testid="approval-query">
            {(display.query_or_urls ?? []).length === 0 ? "（无）" : (display.query_or_urls ?? []).join("；")}
          </dd>
        </div>
        {display.date_range !== "" && (
          <div>
            <dt>日期范围</dt>
            <dd data-testid="approval-date-range">{display.date_range}</dd>
          </div>
        )}
        {display.task_question !== "" && (
          <div>
            <dt>所属子任务</dt>
            <dd data-testid="approval-task">{display.task_question}</dd>
          </div>
        )}
      </dl>
      {display.redaction_notice !== "" && (
        <p className="chat-card-notice" data-testid="approval-redaction">
          {display.redaction_notice}
        </p>
      )}
      {editing && (
        <textarea
          className="chat-card-editor"
          data-testid="approval-editor"
          aria-label="编辑查询"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
        />
      )}
      {error !== null && (
        <p className="chat-card-error" role="alert" data-testid="approval-error">
          {error}
        </p>
      )}
      <div className="chat-card-actions">
        <button
          type="button"
          className="chat-card-primary"
          data-testid="approval-allow-once"
          disabled={busy}
          onClick={() => decide("allow_once")}
        >
          允许本次
        </button>
        <button
          type="button"
          className="chat-card-secondary"
          data-testid="approval-allow-scope"
          disabled={busy}
          onClick={() => decide("allow_scope")}
        >
          允许本轮同类操作
        </button>
        <button
          type="button"
          className="chat-card-danger"
          data-testid="approval-deny"
          disabled={busy}
          onClick={() => decide("deny")}
        >
          拒绝
        </button>
        <button
          type="button"
          className="chat-card-link"
          data-testid="approval-edit-toggle"
          aria-expanded={editing}
          onClick={() => setEditing((value) => !value)}
        >
          {editing ? "取消编辑" : "编辑查询"}
        </button>
      </div>
    </div>
  );
}

export interface ClarificationCardProps {
  request: InputRequest;
  busy?: boolean;
  error?: string | null;
  onSubmit: (inputRequestId: string, answer: string) => void;
}

/** 澄清卡片：选项可直接点，也可自由作答。 */
export function ClarificationCard({
  request,
  busy = false,
  error = null,
  onSubmit,
}: ClarificationCardProps) {
  const [answer, setAnswer] = useState("");
  const submit = (value: string): void => {
    const trimmed = value.trim();
    if (trimmed === "") return;
    onSubmit(request.input_request_id, trimmed);
  };

  return (
    <div className="chat-card input-card" data-testid={`input-card-${request.input_request_id}`}>
      <div className="chat-card-head">
        <span className="chat-card-badge">需要澄清</span>
      </div>
      <p className="chat-card-purpose" data-testid="input-question">
        {request.question}
      </p>
      {request.reason !== "" && <p className="chat-card-notice">{request.reason}</p>}
      {(request.options ?? []).length > 0 && (
        <div className="chat-card-options" data-testid="input-options">
          {request.options.map((option) => (
            <button
              key={option}
              type="button"
              className="chat-card-option"
              disabled={busy}
              onClick={() => submit(option)}
            >
              {option}
            </button>
          ))}
        </div>
      )}
      <div className="chat-card-answer">
        <input
          className="chat-card-input"
          data-testid="input-answer"
          aria-label="澄清回答"
          value={answer}
          placeholder="输入你的回答"
          onChange={(event) => setAnswer(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              submit(answer);
            }
          }}
        />
        <button
          type="button"
          className="chat-card-primary"
          data-testid="input-submit"
          disabled={busy || answer.trim() === ""}
          onClick={() => submit(answer)}
        >
          回复
        </button>
      </div>
      {error !== null && (
        <p className="chat-card-error" role="alert" data-testid="input-error">
          {error}
        </p>
      )}
    </div>
  );
}

export default ApprovalCard;
