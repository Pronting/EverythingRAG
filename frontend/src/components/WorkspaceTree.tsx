import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { ConversationSummary, Workspace } from "../types";
import type { ConversationMenuState } from "./Sidebar";
import {
  DEFAULT_VISIBLE,
  expandVisible,
  pagingFooter,
  remainingCount,
  resetVisible,
  shownCount,
  visibleLimitFor,
} from "./workspacePaging";

/**
 * 拖拽落点：把会话放进哪个工作区。
 *
 * 自 2026-09-22 起**只有跨工作区**这一种拖拽语义（工作区内拖拽排序已取消，见文件头
 * 「最近活动优先」说明），所以落点只需要一个目标工作区 id。
 */
export interface DropTarget {
  workspaceId: string;
}

/**
 * 侧边栏 UI 偏好。
 *
 * - 折叠状态（`COLLAPSED_KEY`）**仍持久化**：刷新后保持上次的样子；
 * - 「已展开条数」**刻意不持久化**（`LEGACY_VISIBLE_KEY` 仅用于清理旧值）：默认 5 条、
 *   展开到底可「收起」、折叠工作区与重启应用都回到默认 —— 具体规则见
 *   `workspacePaging.ts`（纯函数，有单测）。
 */
const COLLAPSED_KEY = "everything-rag-workspace-collapsed";
const LEGACY_VISIBLE_KEY = "everything-rag-workspace-visible";
/**
 * 双击判定窗口：只用于把「同一次双击的第二下」让给重命名，
 * **绝不延迟第一下**的折叠 —— 单次点击必须当帧生效（用户 2026-09-22 的明确要求：
 * 原先「先等 220ms 再折叠」让展开/收起像慢了一秒）。
 */
const DOUBLE_CLICK_WINDOW_MS = 320;

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

function persist(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* 忽略：存储不可用时仅失去记忆 */
  }
}

/** 清掉旧版本记录的「已展开条数」：新版重启即回默认，留着只会误导排查。 */
function dropLegacyVisibleCounts(): void {
  try {
    localStorage.removeItem(LEGACY_VISIBLE_KEY);
  } catch {
    /* 忽略：存储不可用不影响功能 */
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
  /**
   * 用户点了某个工作区（文件夹头）：把它记成「当前工作区」。
   *
   * 侧边栏不再有单独的「当前分区切换器」，所以「新对话落到哪个工作区」只能靠这里捕获——
   * 否则新对话永远落在默认工作区（用户 2026-09-22 的明确要求）。
   */
  onSelectWorkspace: (workspaceId: string) => void;
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
 * 可折叠容器：折叠 / 展开时用**像素高度**做过渡动画。
 *
 * 刻意不用 CSS `grid-template-rows: 0fr ↔ 1fr`：那条路依赖较新的浏览器特性，
 * 而且在「系统关了动画」(prefers-reduced-motion) 或老一点的 Chromium 上会直接瞬变，
 * 用户看到的就是「一下就折叠」。这里量一次真实高度、写进行内高度再过渡，
 * 任何浏览器、任何动画偏好下都是同一段平滑动画。
 */
function FolderChildren({
  collapsed,
  children,
}: {
  collapsed: boolean;
  children: React.ReactNode;
}) {
  const outerRef = useRef<HTMLDivElement>(null);
  const innerRef = useRef<HTMLDivElement>(null);
  const firstRender = useRef(true);

  useLayoutEffect(() => {
    const outer = outerRef.current;
    const inner = innerRef.current;
    if (outer === null || inner === null) return;
    // 首次渲染直接落到目标状态：刷新后保持收起，不该补播一次动画
    if (firstRender.current) {
      firstRender.current = false;
      outer.style.height = collapsed ? "0px" : "auto";
      return;
    }
    const from = outer.getBoundingClientRect().height;
    const to = collapsed ? 0 : inner.getBoundingClientRect().height;
    // 先钉住「当前高度」并强制一次布局，再写目标高度：否则两次修改会被合并，过渡不触发
    outer.style.height = `${from}px`;
    void outer.offsetHeight;
    outer.style.height = `${to}px`;
  }, [collapsed]);

  return (
    <div
      className="ws-folder-children"
      data-collapsed={collapsed}
      ref={outerRef}
      inert={collapsed}
      onTransitionEnd={(event) => {
        // 展开完成后回到 auto：之后「展开显示」把列表变长时高度还能自然跟上
        if (event.propertyName === "height" && !collapsed && outerRef.current !== null) {
          outerRef.current.style.height = "auto";
        }
      }}
    >
      <div className="ws-folder-children-inner" ref={innerRef}>
        {children}
      </div>
    </div>
  );
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
  onSelectWorkspace,
  onRenameWorkspace,
  onDeleteWorkspace,
  onCreateWorkspace,
  onDropConversation,
  onReorderWorkspaces,
  onToggleWorkspacePin,
}: WorkspaceTreeProps) {
  // 折叠状态从 localStorage 恢复；「已展开条数」只在本次会话的内存里（重启即回默认 5 条）
  const [collapsed, setCollapsed] = useState<Set<string>>(() => new Set(readCollapsed()));
  const [visibleCounts, setVisibleCounts] = useState<Record<string, number>>({});
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [draftName, setDraftName] = useState("");
  const [menuWorkspaceId, setMenuWorkspaceId] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<Workspace | null>(null);
  const [dragging, setDragging] = useState<string | null>(null);
  const [dropWorkspaceId, setDropWorkspaceId] = useState<string | null>(null);
  // 工作区自身拖拽排序：dragWorkspaceId 为被拖的分区，dropBeforeId 为落点
  const [dragWorkspaceId, setDragWorkspaceId] = useState<string | null>(null);
  const [dropBeforeId, setDropBeforeId] = useState<string | null>(null);
  const renameInputRef = useRef<HTMLInputElement>(null);
  const lastFolderClickRef = useRef<{ id: string; at: number }>({ id: "", at: 0 });
  /**
   * 被拖拽会话的所属工作区：用来判断这次拖拽是否真的跨区。
   * 同区拖拽在本版已无语义（排序恒为「最近活动优先」），不再触发落点高亮与移动调用。
   */
  const draggingWorkspaceId = ((): string | null => {
    if (dragging === null) return null;
    const row = conversations.find((item) => item.id === dragging);
    return row === undefined ? null : workspaceOf(row);
  })();

  useEffect(() => {
    dropLegacyVisibleCounts();
  }, []);

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
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
        // 折叠工作区 ⇒ 该工作区的对话列表也回默认条数：再次展开时就是默认的 5 条，
        // 而不是延续折叠前「越点越长」的窗口（用户 2026-09-20 的明确要求）。
        setVisibleCounts((counts) => resetVisible(counts, id));
      }
      persist(COLLAPSED_KEY, [...next]);
      return next;
    });
  };

  /**
   * 点击工作区 = **立刻**折叠/展开（不再新建对话），同时把它记为「当前工作区」。
   *
   * 双击仍是重命名，但重命名不能再靠「先等一会儿」实现：
   * - 第一下：当帧折叠（用户要的及时响应）；
   * - 第二下（`DOUBLE_CLICK_WINDOW_MS` 内）：直接返回，不折叠 —— 否则会连折两次，
   *   然后由 `onDoubleClick` 把第一下撤销回来并打开重命名。
   * 慢速的两次单击（超出窗口）按两次独立点击处理，与浏览器 `dblclick` 的判定一致。
   */
  const handleFolderClick = (id: string): void => {
    const now = Date.now();
    const last = lastFolderClickRef.current;
    if (last.id === id && now - last.at < DOUBLE_CLICK_WINDOW_MS) {
      lastFolderClickRef.current = { id, at: 0 };
      return;
    }
    lastFolderClickRef.current = { id, at: now };
    toggleCollapsed(id);
  };

  /** 双击工作区 = 重命名：撤销第一下造成的折叠，让展开状态保持原样。 */
  const handleFolderDoubleClick = (workspace: Workspace): void => {
    lastFolderClickRef.current = { id: workspace.id, at: 0 };
    toggleCollapsed(workspace.id);
    setRenamingId(workspace.id);
    setDraftName(workspace.name);
  };

  /** 展开显示：每次追加 VISIBLE_STEP 条（只改内存，不落盘）。 */
  const showMore = (id: string): void => {
    setVisibleCounts((prev) => expandVisible(prev, id));
  };

  /** 收起：把该工作区恢复到默认条数（列表到底后才出现这个按钮）。 */
  const showLess = (id: string): void => {
    setVisibleCounts((prev) => resetVisible(prev, id));
  };

  const visibleLimit = (id: string): number => visibleLimitFor(visibleCounts, id);

  const submitRename = async (id: string): Promise<void> => {
    const name = draftName.trim();
    setRenamingId(null);
    if (name === "") return;
    await onRenameWorkspace(id, name);
  };

  const startDrag = (conversationId: string): void => {
    setDragging(conversationId);
    setDropWorkspaceId(null);
  };

  const endDrag = (): void => {
    setDragging(null);
    setDropWorkspaceId(null);
  };

  const commitDrop = (): void => {
    if (dragging !== null && dropWorkspaceId !== null) {
      onDropConversation(dragging, { workspaceId: dropWorkspaceId });
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
  ) => {
    const isDragging = dragging === conversation.id;
    // 同区拖拽没有语义（排序永远是最近活动优先）⇒ 不给出落点高亮，也不拦截鼠标事件，
    // 免得让人以为「拖一下就能定住顺序」。只有跨区拖拽才接受落点。
    const acceptsDrop = dragging !== null && !isDragging && draggingWorkspaceId !== workspaceId;
    return (
      <div
        key={conversation.id}
        data-conversation-id={conversation.id}
        className={`ws-conversation${conversation.id === activeConversationId ? " active" : ""}${
          isDragging ? " dragging" : ""
        }`}
        draggable
        onDragStart={(event) => {
          event.dataTransfer.effectAllowed = "move";
          event.dataTransfer.setData("text/plain", conversation.id);
          startDrag(conversation.id);
        }}
        onDragEnd={endDrag}
        onDragOver={(event) => {
          if (!acceptsDrop) return;
          event.preventDefault();
          event.stopPropagation();
          setDropWorkspaceId(workspaceId);
        }}
        onDrop={(event) => {
          if (!acceptsDrop) return;
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
        // 顺序完全由后端给出：侧边栏列表一律「最近活动优先」（2026-09-22 需求 2），
        // 这里不再按任何手工位次重排（旧数据里的 sort_index 一并失效）。
        const children = conversations.filter(
          (conversation) => workspaceOf(conversation) === workspace.id,
        );
        const isCollapsed = collapsed.has(workspace.id);
        const isDropZone = dropWorkspaceId === workspace.id;
        const limit = visibleLimit(workspace.id);
        // 当前打开的对话必须可见：即使它排在第 5 条之后，也要把它所在窗口一起展示出来
        const activeIndex = children.findIndex((item) => item.id === activeConversationId);
        const shownTotal = shownCount(children.length, limit, activeIndex);
        const shown = children.slice(0, shownTotal);
        const remaining = remainingCount(children.length, shownTotal);
        // 「展开显示」还能加；已经到底且确实展开过 ⇒ 换成「收起」（回到默认 5 条）
        const footer = pagingFooter(children.length, shownTotal, limit);
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
              onClick={() => {
                // 整行可点（图标 / 计数 / 空白处都算）：点了就立刻折叠，并记为当前工作区。
                onSelectWorkspace(workspace.id);
                handleFolderClick(workspace.id);
              }}
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
                  if (draggingWorkspaceId !== null && draggingWorkspaceId === workspace.id) return;
                  event.preventDefault();
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
                  onClick={(event) => event.stopPropagation()}
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
                  aria-expanded={!isCollapsed}
                  onDoubleClick={(event) => {
                    event.stopPropagation();
                    handleFolderDoubleClick(workspace);
                  }}
                  onClick={(event) => {
                    // 名称按钮单独接管点击：否则会连带触发整行的 onClick 而折叠两次
                    event.stopPropagation();
                    onSelectWorkspace(workspace.id);
                    handleFolderClick(workspace.id);
                  }}
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
                  // 整行的 onClick 现在也负责折叠：菜单项里的点击必须止住冒泡，
                  // 否则点「重命名 / 删除 / 在此新建对话」会顺手把文件夹折叠掉。
                  onClick={(event) => event.stopPropagation()}
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

            {/*
              折叠/展开是**动画**而不是瞬间切换：列表始终挂载，高度由 FolderChildren
              逐帧过渡（见 App.css `.ws-folder-children`）。收起时用 `inert` 把里面的
              会话移出焦点与点击范围，否则 Tab 还能走到看不见的条目上。
            */}
            <FolderChildren collapsed={isCollapsed}>
              {shown.map((conversation) => renderConversation(conversation, workspace.id))}
              {footer === "expand" && (
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
              {footer === "collapse" && (
                <button
                  type="button"
                  className="ws-folder-more ws-folder-less"
                  data-workspace-id={workspace.id}
                  onClick={() => showLess(workspace.id)}
                  title={`收起为默认 ${DEFAULT_VISIBLE} 条`}
                >
                  收起
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
            </FolderChildren>
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
