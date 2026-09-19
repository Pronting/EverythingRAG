import { useEffect, useRef, useState } from "react";
import type { ConversationSummary, Workspace } from "../types";
import type { ConversationMenuState } from "./Sidebar";

/** 拖拽落点：插到某个会话的前/后，或整个工作区末尾。 */
export interface DropTarget {
  workspaceId: string;
  /** 兄弟会话 id；落在它之前/之后。 */
  beforeId: string | null;
  afterId: string | null;
}

/** 侧边栏 UI 偏好（折叠状态 / 已展开条数）：本地记忆，刷新后保持。 */
const COLLAPSED_KEY = "everything-rag-workspace-collapsed";
const VISIBLE_KEY = "everything-rag-workspace-visible";
/** 每个工作区默认显示多少条会话，点「展开显示」每次追加多少条。 */
const DEFAULT_VISIBLE = 5;
const VISIBLE_STEP = 10;

function readCollapsed(): string[] {
  try {
    const raw = localStorage.getItem(COLLAPSED_KEY);
    if (raw === null) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === "string") : [];
  } catch {
    return [];
  }
}

function readVisibleCounts(): Record<string, number> {
  try {
    const raw = localStorage.getItem(VISIBLE_KEY);
    if (raw === null) return {};
    const parsed: unknown = JSON.parse(raw);
    return typeof parsed === "object" && parsed !== null ? (parsed as Record<string, number>) : {};
  } catch {
    return {};
  }
}

function persist(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* 忽略：存储不可用时仅失去记忆 */
  }
}

interface WorkspaceTreeProps {
  workspaces: Workspace[];
  /** 当前全部活跃会话（不带分区过滤，由本组件按 workspace_id 分组）。 */
  conversations: ConversationSummary[];
  activeConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onOpenConversationMenu: (state: ConversationMenuState) => void;
  onNewChat: (workspaceId: string) => void;
  onRenameWorkspace: (id: string, name: string) => Promise<void>;
  onDeleteWorkspace: (workspace: Workspace) => Promise<void>;
  onCreateWorkspace: () => Promise<void>;
  /** 拖拽落点：把会话移动到目标位置（跨工作区或同区排序）。 */
  onDropConversation: (conversationId: string, target: DropTarget) => void;
  /** 拖拽工作区分区排序（按新顺序落库）。 */
  onReorderWorkspaces: (orderedIds: string[]) => void;
  /** 置顶 / 取消置顶工作区（别针图标）。 */
  onToggleWorkspacePin: (workspace: Workspace) => void;
}

/** 会话所属工作区（缺省落到默认分区，兼容旧数据）。 */
function workspaceOf(conversation: ConversationSummary): string {
  return conversation.workspace_id ?? "default";
}

/**
 * 侧边栏工作区文件夹树（对齐 Codex）：每个工作区是一个可折叠文件夹，
 * 会话列在其下；支持拖拽排序与拖到别的工作区，也支持新建/重命名/删除工作区。
 */
export default function WorkspaceTree({
  workspaces,
  conversations,
  activeConversationId,
  onSelectConversation,
  onOpenConversationMenu,
  onNewChat,
  onRenameWorkspace,
  onDeleteWorkspace,
  onCreateWorkspace,
  onDropConversation,
  onReorderWorkspaces,
  onToggleWorkspacePin,
}: WorkspaceTreeProps) {
  // 折叠状态与「已展开条数」都从 localStorage 恢复：进入界面时保持上次的样子
  const [collapsed, setCollapsed] = useState<Set<string>>(() => new Set(readCollapsed()));
  const [visibleCounts, setVisibleCounts] = useState<Record<string, number>>(readVisibleCounts);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [draftName, setDraftName] = useState("");
  const [menuWorkspaceId, setMenuWorkspaceId] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<Workspace | null>(null);
  const [dragging, setDragging] = useState<string | null>(null);
  const [dropTarget, setDropTarget] = useState<DropTarget | null>(null);
  const [dropWorkspaceId, setDropWorkspaceId] = useState<string | null>(null);
  // 工作区自身拖拽排序：dragWorkspaceId 为被拖的分区，dropBeforeId 为落点
  const [dragWorkspaceId, setDragWorkspaceId] = useState<string | null>(null);
  const [dropBeforeId, setDropBeforeId] = useState<string | null>(null);
  const renameInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (renamingId !== null) renameInputRef.current?.select();
  }, [renamingId]);

  // 点别处 / Esc 关闭文件夹操作菜单
  useEffect(() => {
    if (menuWorkspaceId === null) return;
    const close = (): void => setMenuWorkspaceId(null);
    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key === "Escape") setMenuWorkspaceId(null);
    };
    document.addEventListener("pointerdown", close);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", close);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [menuWorkspaceId]);

  const toggleCollapsed = (id: string): void => {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      persist(COLLAPSED_KEY, [...next]);
      return next;
    });
  };

  /** 显示更多：每次追加 VISIBLE_STEP 条，并记忆该分区已展开的条数。 */
  const showMore = (id: string): void => {
    setVisibleCounts((prev) => {
      const next = { ...prev, [id]: (prev[id] ?? DEFAULT_VISIBLE) + VISIBLE_STEP };
      persist(VISIBLE_KEY, next);
      return next;
    });
  };

  const visibleLimit = (id: string): number => visibleCounts[id] ?? DEFAULT_VISIBLE;

  const submitRename = async (id: string): Promise<void> => {
    const name = draftName.trim();
    setRenamingId(null);
    if (name === "") return;
    await onRenameWorkspace(id, name);
  };

  const startDrag = (conversationId: string): void => {
    setDragging(conversationId);
    setDropTarget(null);
    setDropWorkspaceId(null);
  };

  const endDrag = (): void => {
    setDragging(null);
    setDropTarget(null);
    setDropWorkspaceId(null);
  };

  const commitDrop = (): void => {
    if (dragging !== null) {
      const target =
        dropTarget ??
        (dropWorkspaceId !== null
          ? { workspaceId: dropWorkspaceId, beforeId: null, afterId: null }
          : null);
      if (target !== null) onDropConversation(dragging, target);
    }
    endDrag();
  };

  const endWorkspaceDrag = (): void => {
    setDragWorkspaceId(null);
    setDropBeforeId(null);
  };

  /** 工作区落点：把被拖的分区插到目标分区之前（或末尾），落库新顺序。 */
  const commitWorkspaceDrop = (targetId: string | null): void => {
    const movingId = dragWorkspaceId;
    endWorkspaceDrag();
    if (movingId === null || movingId === targetId) return;
    const remaining = workspaces.filter((workspace) => workspace.id !== movingId).map((workspace) => workspace.id);
    const insertAt = targetId === null ? remaining.length : remaining.indexOf(targetId);
    if (insertAt < 0) return;
    const nextOrder = [...remaining];
    nextOrder.splice(insertAt, 0, movingId);
    if (nextOrder.join() === workspaces.map((workspace) => workspace.id).join()) return;
    onReorderWorkspaces(nextOrder);
  };

  const renderConversation = (
    conversation: ConversationSummary,
    workspaceId: string,
    siblings: ConversationSummary[],
    index: number,
  ) => {
    const previous = siblings[index - 1];
    const next = siblings[index + 1];
    const isDropBefore =
      dropTarget !== null &&
      dropTarget.workspaceId === workspaceId &&
      dropTarget.afterId === previous?.id &&
      dropTarget.beforeId === conversation.id;
    const isDragging = dragging === conversation.id;
    return (
      <div
        key={conversation.id}
        className={`ws-conversation${conversation.id === activeConversationId ? " active" : ""}${
          isDragging ? " dragging" : ""
        }${isDropBefore ? " drop-before" : ""}`}
        draggable
        onDragStart={(event) => {
          event.dataTransfer.effectAllowed = "move";
          event.dataTransfer.setData("text/plain", conversation.id);
          startDrag(conversation.id);
        }}
        onDragEnd={endDrag}
        onDragOver={(event) => {
          if (dragging === null || dragging === conversation.id) return;
          event.preventDefault();
          event.stopPropagation();
          const box = event.currentTarget.getBoundingClientRect();
          const after = event.clientY > box.top + box.height / 2;
          const beforeId = after ? next?.id ?? null : conversation.id;
          const afterId = after ? conversation.id : previous?.id ?? null;
          setDropWorkspaceId(null);
          setDropTarget({ workspaceId, beforeId, afterId });
        }}
        onDrop={(event) => {
          event.preventDefault();
          event.stopPropagation();
          commitDrop();
        }}
      >
        <button
          type="button"
          className="ws-conversation-main"
          onClick={() => onSelectConversation(conversation.id)}
          onContextMenu={(event) => {
            event.preventDefault();
            onOpenConversationMenu({ conversation, x: event.clientX, y: event.clientY });
          }}
          title={conversation.title}
        >
          {conversation.title}
        </button>
      </div>
    );
  };

  return (
    <div className="workspace-tree">
      {workspaces.map((workspace) => {
        const children = conversations
          .filter((conversation) => workspaceOf(conversation) === workspace.id)
          .sort((left, right) => (left.sort_index ?? Number.MAX_SAFE_INTEGER) - (right.sort_index ?? Number.MAX_SAFE_INTEGER));
        const isCollapsed = collapsed.has(workspace.id);
        const isDropZone = dropWorkspaceId === workspace.id;
        const limit = visibleLimit(workspace.id);
        // 当前打开的对话必须可见：即使它排在第 5 条之后，也要把它所在窗口一起展示出来
        const activeIndex = children.findIndex((item) => item.id === activeConversationId);
        const shown = children.slice(0, activeIndex >= limit ? activeIndex + 1 : limit);
        const remaining = children.length - shown.length;
        const isWorkspaceDropTarget = dropBeforeId === workspace.id && dragWorkspaceId !== null;
        return (
          <section
            key={workspace.id}
            className={`ws-folder${dragWorkspaceId === workspace.id ? " dragging" : ""}${
              isWorkspaceDropTarget ? " drop-before" : ""
            }`}
          >
            <div
              className={`ws-folder-header${isDropZone ? " drop-into" : ""}`}
              draggable={renamingId === null}
              onDragStart={(event) => {
                if (renamingId !== null) return;
                event.dataTransfer.effectAllowed = "move";
                event.dataTransfer.setData("text/plain", `workspace:${workspace.id}`);
                setDragWorkspaceId(workspace.id);
                setDropBeforeId(null);
              }}
              onDragEnd={endWorkspaceDrag}
              onDragOver={(event) => {
                // 会话拖进本分区 → 移动会话；分区互拖 → 调整分区顺序
                if (dragging !== null) {
                  event.preventDefault();
                  setDropTarget(null);
                  setDropWorkspaceId(workspace.id);
                  return;
                }
                if (dragWorkspaceId === null || dragWorkspaceId === workspace.id) return;
                event.preventDefault();
                const box = event.currentTarget.getBoundingClientRect();
                const after = event.clientY > box.top + box.height / 2;
                if (after) {
                  const index = workspaces.findIndex((item) => item.id === workspace.id);
                  setDropBeforeId(workspaces[index + 1]?.id ?? null);
                } else {
                  setDropBeforeId(workspace.id);
                }
              }}
              onDrop={(event) => {
                event.preventDefault();
                if (dragging !== null) {
                  commitDrop();
                  return;
                }
                commitWorkspaceDrop(dropBeforeId);
              }}
            >
              <button
                type="button"
                className="ws-folder-toggle"
                aria-expanded={!isCollapsed}
                aria-label={`${isCollapsed ? "展开" : "收起"}工作区「${workspace.name}」`}
                onClick={() => toggleCollapsed(workspace.id)}
              >
                <svg width="9" height="9" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                  <path d="M6 3l5 5-5 5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
              <svg className="ws-folder-icon" width="13" height="13" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <path
                  d="M2 5.5A1.5 1.5 0 0 1 3.5 4h3l1.2 1.6h4.8A1.5 1.5 0 0 1 14 7.1v4.4A1.5 1.5 0 0 1 12.5 13h-9A1.5 1.5 0 0 1 2 11.5z"
                  stroke="currentColor"
                  strokeWidth="1.3"
                  strokeLinejoin="round"
                />
              </svg>
              {renamingId === workspace.id ? (
                <form
                  className="ws-folder-rename"
                  onSubmit={(event) => {
                    event.preventDefault();
                    void submitRename(workspace.id);
                  }}
                >
                  <input
                    ref={renameInputRef}
                    className="ws-folder-rename-input"
                    value={draftName}
                    maxLength={20}
                    aria-label="工作区新名称"
                    onChange={(event) => setDraftName(event.target.value)}
                    onBlur={() => setRenamingId(null)}
                  />
                </form>
              ) : (
                <button
                  type="button"
                  className="ws-folder-name"
                  title={workspace.name}
                  onDoubleClick={() => {
                    setRenamingId(workspace.id);
                    setDraftName(workspace.name);
                  }}
                  onClick={() => onNewChat(workspace.id)}
                >
                  {workspace.name}
                </button>
              )}
              <span className="ws-folder-count">{children.length}</span>
              <button
                type="button"
                className={`ws-folder-pin${workspace.pinned ? " active" : ""}`}
                aria-pressed={workspace.pinned === true}
                aria-label={`${workspace.pinned ? "取消置顶" : "置顶"}工作区「${workspace.name}」`}
                title={workspace.pinned ? "取消置顶" : "置顶"}
                onClick={(event) => {
                  event.stopPropagation();
                  onToggleWorkspacePin(workspace);
                }}
              >
                <svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                  <path
                    d="M6 2h4l-.6 3.2 2.1 2.1H4.5l2.1-2.1zM8 7.3V14"
                    stroke="currentColor"
                    strokeWidth="1.3"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    fill={workspace.pinned === true ? "currentColor" : "none"}
                  />
                </svg>
              </button>
              <button
                type="button"
                className="ws-folder-menu-btn"
                aria-label={`工作区「${workspace.name}」操作`}
                aria-haspopup="menu"
                aria-expanded={menuWorkspaceId === workspace.id}
                onClick={(event) => {
                  event.stopPropagation();
                  setMenuWorkspaceId((current) => (current === workspace.id ? null : workspace.id));
                }}
              >
                <svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
                  <circle cx="3.4" cy="8" r="1.3" />
                  <circle cx="8" cy="8" r="1.3" />
                  <circle cx="12.6" cy="8" r="1.3" />
                </svg>
              </button>
              {menuWorkspaceId === workspace.id && (
                <div
                  className="ws-folder-menu"
                  role="menu"
                  aria-label={`工作区「${workspace.name}」操作`}
                  onPointerDown={(event) => event.stopPropagation()}
                >
                  <button
                    type="button"
                    role="menuitem"
                    className="ws-folder-menu-item"
                    onClick={() => {
                      setMenuWorkspaceId(null);
                      onNewChat(workspace.id);
                    }}
                  >
                    在此工作区新建对话
                  </button>
                  <button
                    type="button"
                    role="menuitem"
                    className="ws-folder-menu-item"
                    onClick={() => {
                      setMenuWorkspaceId(null);
                      setRenamingId(workspace.id);
                      setDraftName(workspace.name);
                    }}
                  >
                    重命名工作区
                  </button>
                  <button
                    type="button"
                    role="menuitem"
                    className="ws-folder-menu-item danger"
                    disabled={workspace.is_default}
                    title={workspace.is_default ? "默认工作区不可删除" : "其中对话会被归档"}
                    onClick={() => {
                      setMenuWorkspaceId(null);
                      setPendingDelete(workspace);
                    }}
                  >
                    删除工作区
                  </button>
                </div>
              )}
            </div>

            {!isCollapsed && (
              <div className="ws-folder-children">
                {shown.map((conversation, index) =>
                  renderConversation(conversation, workspace.id, children, index),
                )}
                {remaining > 0 && (
                  <button
                    type="button"
                    className="ws-folder-more"
                    data-workspace-id={workspace.id}
                    onClick={() => showMore(workspace.id)}
                    title={`还有 ${remaining} 条`}
                  >
                    展开显示
                  </button>
                )}
                {children.length === 0 && (
                  <button
                    type="button"
                    className="ws-folder-empty"
                    onClick={() => onNewChat(workspace.id)}
                  >
                    暂无对话，点此新建
                  </button>
                )}
              </div>
            )}
          </section>
        );
      })}

      <button
        type="button"
        className="ws-folder-new"
        onClick={() => void onCreateWorkspace()}
        aria-label="新建工作区"
      >
        ＋ 新建工作区
      </button>

      {pendingDelete !== null && (
        <div className="workspace-confirm" role="alertdialog" aria-label="确认删除工作区">
          <p>
            删除「{pendingDelete.name}」后，其中 {pendingDelete.conversation_count} 条对话会被归档，
            可在设置页「归档对话」找回。
          </p>
          <div className="workspace-confirm-actions">
            <button type="button" className="workspace-form-cancel" onClick={() => setPendingDelete(null)}>
              取消
            </button>
            <button
              type="button"
              className="workspace-delete-ok"
              onClick={() => {
                const target = pendingDelete;
                setPendingDelete(null);
                void onDeleteWorkspace(target);
              }}
            >
              确认删除
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
