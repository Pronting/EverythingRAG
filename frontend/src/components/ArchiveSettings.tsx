import { useCallback, useEffect, useState } from "react";
import {
  deleteConversation,
  listArchivedConversations,
  restoreConversation,
} from "../api/conversationApi";
import { listWorkspaces } from "../api/workspaceApi";
import type { ConversationSummary, ToastType, Workspace } from "../types";

interface ArchiveSettingsProps {
  /** 恢复成功后通知父级刷新侧边栏会话列表。 */
  onConversationsChanged: () => void;
  onNotify: (type: ToastType, message: string) => void;
}

/** 归档时间显示：本地化到分钟，日期无效时回退原串。 */
function formatArchivedAt(value: string | null | undefined): string {
  if (!value) return "归档时间未知";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/**
 * 设置页「归档对话」：列出所有已归档会话，可恢复到侧边栏，也可彻底删除。
 *
 * 进入该页（组件挂载）时拉取一次归档列表；恢复/删除后就地更新，无需整页刷新。
 */
export default function ArchiveSettings({
  onConversationsChanged,
  onNotify,
}: ArchiveSettingsProps) {
  const [items, setItems] = useState<ConversationSummary[]>([]);
  // 工作区名称映射：归档项里也标出它原本属于哪个分区
  const [workspaceNames, setWorkspaceNames] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  // 删除需二次确认：记住待删除会话（不做原生 confirm，保持与整体 UI 一致）
  const [pendingDelete, setPendingDelete] = useState<ConversationSummary | null>(null);

  const load = useCallback((): void => {
    setLoading(true);
    setError(null);
    listArchivedConversations()
      .then((list) => setItems(list))
      .catch((cause: unknown) =>
        setError(cause instanceof Error ? cause.message : String(cause)),
      )
      .finally(() => setLoading(false));
    listWorkspaces()
      .then((list: Workspace[]) =>
        setWorkspaceNames(Object.fromEntries(list.map((workspace) => [workspace.id, workspace.name]))),
      )
      .catch(() => {
        /* 工作区名称只是附加信息，取不到就不显示 */
      });
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const handleRestore = async (item: ConversationSummary): Promise<void> => {
    setBusyId(item.id);
    try {
      await restoreConversation(item.id);
      setItems((prev) => prev.filter((entry) => entry.id !== item.id));
      onConversationsChanged();
      onNotify("success", `已恢复「${item.title}」到对话列表`);
    } catch (cause) {
      onNotify("error", cause instanceof Error ? cause.message : "恢复失败，请稍后重试");
    } finally {
      setBusyId(null);
    }
  };

  const handleDelete = async (item: ConversationSummary): Promise<void> => {
    setBusyId(item.id);
    try {
      await deleteConversation(item.id);
      setItems((prev) => prev.filter((entry) => entry.id !== item.id));
      setPendingDelete(null);
      onNotify("success", `已删除「${item.title}」`);
    } catch (cause) {
      onNotify("error", cause instanceof Error ? cause.message : "删除失败，请稍后重试");
    } finally {
      setBusyId(null);
    }
  };

  return (
    <>
      <div className="archive-heading">
        <h3>归档对话</h3>
        <span className="archive-count">{items.length} 条</span>
      </div>
      <p className="archive-hint">
        归档的对话不再出现在侧边栏，但完整保留在这里；点「恢复」即可让它重新回到对话列表。
      </p>

      {loading && <p className="archive-empty">正在加载归档…</p>}
      {!loading && error !== null && (
        <p className="archive-empty archive-error" role="alert">
          {error}
        </p>
      )}
      {!loading && error === null && items.length === 0 && (
        <p className="archive-empty">还没有归档的对话。在侧边栏右键任意对话即可归档。</p>
      )}

      {items.length > 0 && (
        <ul className="archive-list">
          {items.map((item) => (
            <li key={item.id} className="archive-item">
              <div className="archive-item-body">
                <span className="archive-item-title" title={item.title}>
                  {item.title}
                </span>
                <span className="archive-item-meta">
                  {item.message_count} 条消息 · {formatArchivedAt(item.archived_at)}
                  {item.workspace_id && workspaceNames[item.workspace_id]
                    ? ` · 原分区「${workspaceNames[item.workspace_id]}」`
                    : ""}
                  {item.forked_from_title ? ` · 分支自「${item.forked_from_title}」` : ""}
                </span>
              </div>
              <div className="archive-item-actions">
                <button
                  type="button"
                  className="archive-restore-btn"
                  disabled={busyId === item.id}
                  onClick={() => void handleRestore(item)}
                  aria-label={`恢复对话 ${item.title}`}
                >
                  恢复
                </button>
                <button
                  type="button"
                  className="archive-delete-btn"
                  disabled={busyId === item.id}
                  onClick={() => setPendingDelete(item)}
                  aria-label={`删除对话 ${item.title}`}
                >
                  删除
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}

      {pendingDelete !== null && (
        <div className="archive-confirm" role="alertdialog" aria-label="确认删除对话">
          <p>
            确定要删除「{pendingDelete.title}」吗？该操作不可撤销，分支对话会保留。
          </p>
          <div className="archive-confirm-actions">
            <button
              type="button"
              className="model-test-button"
              onClick={() => setPendingDelete(null)}
            >
              取消
            </button>
            <button
              type="button"
              className="archive-delete-btn"
              disabled={busyId === pendingDelete.id}
              onClick={() => void handleDelete(pendingDelete)}
            >
              确认删除
            </button>
          </div>
        </div>
      )}
    </>
  );
}
