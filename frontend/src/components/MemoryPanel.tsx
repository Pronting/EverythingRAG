import { useCallback, useEffect, useState } from "react";
import type { MemoryEntry, MemoryKind, MemoryView } from "../api/memoryApi";
import {
  addMemory,
  correctMemory,
  describeMemoryError,
  forgetMemory,
  getMemory,
  memoryKindLabel,
  memoryStatusLabel,
  setMemoryEnabled,
} from "../api/memoryApi";

/**
 * 工作区记忆页（PRD 11.2 / 14.4）。
 *
 * 显示简短条目、类别、来源与更新时间，支持修改、删除、开关和主动保存。
 *
 * ⚠️ 「记忆是否开启」只用**本页读到的 `enabled`** 判断 ——
 * `WorkspaceContextView.memory_enabled` 在后端恒为 `true`（未接线），不能作为开关依据。
 */

const KIND_ORDER: MemoryKind[] = [
  "response_preference",
  "topic",
  "progress",
  "confirmed_conclusion",
];

export interface MemoryPanelProps {
  workspaceId: string;
  onClose: () => void;
  onNotify?: (type: "success" | "error", message: string) => void;
}

export default function MemoryPanel({ workspaceId, onClose, onNotify }: MemoryPanelProps) {
  const [view, setView] = useState<MemoryView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [draft, setDraft] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState("");

  const load = useCallback(async (): Promise<void> => {
    try {
      const next = await getMemory(workspaceId);
      setView(next);
      setError(null);
    } catch (loadError) {
      setError(describeMemoryError(loadError));
    }
  }, [workspaceId]);

  useEffect(() => {
    void load();
  }, [load]);

  const run = async (action: () => Promise<void>): Promise<void> => {
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (actionError) {
      const message = describeMemoryError(actionError);
      setError(message);
      onNotify?.("error", message);
    } finally {
      setBusy(false);
    }
  };

  const handleToggle = (): void => {
    if (view === null) return;
    const next = !view.enabled;
    void run(async () => {
      await setMemoryEnabled(workspaceId, next, view.revision);
      await load();
      onNotify?.("success", next ? "已开启工作区记忆" : "已关闭工作区记忆");
    });
  };

  const handleAdd = (): void => {
    const text = draft.trim();
    if (text === "" || view === null) return;
    void run(async () => {
      await addMemory(workspaceId, text, { expectedRevision: view.revision });
      setDraft("");
      await load();
      onNotify?.("success", "已保存到工作区记忆");
    });
  };

  const handleSaveEdit = (entry: MemoryEntry): void => {
    const text = editText.trim();
    if (text === "" || text === entry.text || view === null) {
      setEditingId(null);
      return;
    }
    void run(async () => {
      await correctMemory(workspaceId, entry.id, {
        text,
        expectedRevision: view.revision,
      });
      setEditingId(null);
      await load();
      onNotify?.("success", "已修改该记忆条目");
    });
  };

  const handleForget = (entry: MemoryEntry): void => {
    if (view === null) return;
    void run(async () => {
      await forgetMemory(workspaceId, entry.id, view.revision);
      await load();
      onNotify?.("success", "已删除该记忆条目");
    });
  };

  return (
    <div className="memory-overlay" role="dialog" aria-label="工作区记忆" data-testid="memory-panel">
      <div className="memory-sheet">
        <div className="memory-head">
          <h2 className="memory-title">工作区记忆</h2>
          <button
            type="button"
            className="memory-close"
            aria-label="关闭工作区记忆"
            onClick={onClose}
          >
            ×
          </button>
        </div>

        {view !== null && (
          <div className="memory-switch-row">
            <label className="memory-switch">
              <input
                type="checkbox"
                data-testid="memory-enabled"
                checked={view.enabled}
                disabled={busy}
                onChange={handleToggle}
              />
              <span>开启工作区记忆</span>
            </label>
            <span className="memory-status" data-testid="memory-enabled-state">
              {view.enabled ? "已开启" : "已关闭"}
            </span>
            <span className="memory-meta">revision {view.revision}</span>
          </div>
        )}

        {error !== null && (
          <p className="memory-error" role="alert" data-testid="memory-error">
            {error}
          </p>
        )}

        <div className="memory-add">
          <input
            className="memory-add-input"
            data-testid="memory-new-text"
            aria-label="新记忆内容"
            placeholder="主动保存一条记忆（例如「回答用中文，语气自然」）"
            value={draft}
            disabled={busy}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                handleAdd();
              }
            }}
          />
          <button
            type="button"
            className="memory-add-button"
            data-testid="memory-add"
            disabled={busy || draft.trim() === ""}
            onClick={handleAdd}
          >
            保存
          </button>
        </div>

        {view === null ? (
          <p className="memory-empty">正在读取…</p>
        ) : view.entries.length === 0 ? (
          <p className="memory-empty" data-testid="memory-empty">
            这个工作区还没有记忆条目。
          </p>
        ) : (
          <ul className="memory-list" data-testid="memory-list">
            {view.entries.map((entry) => (
              <li className="memory-item" key={entry.id} data-testid={`memory-item-${entry.id}`}>
                <div className="memory-item-head">
                  <span className="memory-kind" data-testid={`memory-kind-${entry.id}`}>
                    {memoryKindLabel(entry.kind)}
                  </span>
                  <span className="memory-item-status">{memoryStatusLabel(entry.status)}</span>
                  <span className="memory-item-time" data-testid={`memory-updated-${entry.id}`}>
                    更新于 {entry.updated_at}
                  </span>
                </div>
                {editingId === entry.id ? (
                  <div className="memory-item-edit">
                    <input
                      className="memory-edit-input"
                      data-testid={`memory-edit-input-${entry.id}`}
                      aria-label="修改记忆内容"
                      value={editText}
                      onChange={(event) => setEditText(event.target.value)}
                    />
                    <button
                      type="button"
                      className="memory-save-button"
                      data-testid={`memory-save-${entry.id}`}
                      disabled={busy}
                      onClick={() => handleSaveEdit(entry)}
                    >
                      保存修改
                    </button>
                    <button
                      type="button"
                      className="memory-cancel-button"
                      onClick={() => setEditingId(null)}
                    >
                      取消
                    </button>
                  </div>
                ) : (
                  <p className="memory-item-text" data-testid={`memory-text-${entry.id}`}>
                    {entry.text}
                  </p>
                )}
                <div className="memory-item-foot">
                  <span className="memory-item-source" data-testid={`memory-source-${entry.id}`}>
                    来源：{entry.source_conversation_id || "用户主动保存"}
                  </span>
                  <button
                    type="button"
                    className="memory-edit-button"
                    data-testid={`memory-edit-${entry.id}`}
                    disabled={busy}
                    onClick={() => {
                      setEditingId(entry.id);
                      setEditText(entry.text);
                    }}
                  >
                    修改
                  </button>
                  <button
                    type="button"
                    className="memory-delete-button"
                    data-testid={`memory-delete-${entry.id}`}
                    disabled={busy}
                    onClick={() => handleForget(entry)}
                  >
                    删除
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}

        {view !== null && (
          <p className="memory-foot">
            共 {view.entries.length} 条 · 类别：{KIND_ORDER.map(memoryKindLabel).join(" / ")}
          </p>
        )}
      </div>
    </div>
  );
}
