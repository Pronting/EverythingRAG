import { useEffect, useState } from "react";
import type { MouseEvent as ReactMouseEvent } from "react";
import type { Workspace } from "../types";
import type { ConversationAction, ConversationMenuState } from "./Sidebar";

interface ConversationMenuProps {
  state: ConversationMenuState;
  /** 该项操作正在执行（禁用菜单，避免重复提交）。 */
  busy: boolean;
  /** 全部工作区（用于「移动到工作区」子菜单）。 */
  workspaces: Workspace[];
  onSelect: (action: ConversationAction) => void;
  /** 把当前会话移动到指定工作区。 */
  onMoveToWorkspace: (workspaceId: string) => void;
  onClose: () => void;
}

/** 菜单宽高估算值：用于把菜单夹回视口内，避免贴边被裁掉。 */
const MENU_WIDTH = 196;
const MENU_HEIGHT = 176;

/**
 * 会话右键菜单：Fork（分支）/ 移动到工作区 / 归档 / 删除。
 *
 * 定位在鼠标处并夹回视口；点击别处、滚动、按 Esc、失焦都会关闭。
 */
export default function ConversationMenu({
  state,
  busy,
  workspaces,
  onSelect,
  onMoveToWorkspace,
  onClose,
}: ConversationMenuProps) {
  const [moveOpen, setMoveOpen] = useState(false);

  useEffect(() => {
    const close = (): void => onClose();
    const onKeyDown = (event: KeyboardEvent): void => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("pointerdown", close);
    document.addEventListener("scroll", close, true);
    window.addEventListener("resize", close);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", close);
      document.removeEventListener("scroll", close, true);
      window.removeEventListener("resize", close);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [onClose]);

  const left = Math.max(8, Math.min(state.x, window.innerWidth - MENU_WIDTH - 8));
  const top = Math.max(8, Math.min(state.y, window.innerHeight - MENU_HEIGHT - 8));

  const pick = (action: ConversationAction) => (event: ReactMouseEvent): void => {
    event.stopPropagation();
    onSelect(action);
  };

  // 只有存在「别的」工作区时才显示子菜单（当前分区不需要移动）
  const otherWorkspaces = workspaces.filter(
    (workspace) => workspace.id !== state.conversation.workspace_id,
  );

  return (
    <div
      className="conversation-menu"
      role="menu"
      aria-label={`对话「${state.conversation.title}」操作`}
      style={{ left, top }}
      onPointerDown={(event) => event.stopPropagation()}
    >
      <button
        type="button"
        role="menuitem"
        className="conversation-menu-item"
        disabled={busy}
        onClick={pick("fork")}
      >
        <svg width="13" height="13" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <path
            d="M5 2v5.5a2 2 0 0 0 2 2h4M5 14v-3.5M11 7V5.5M11 9.5V11"
            stroke="currentColor"
            strokeWidth="1.4"
            strokeLinecap="round"
          />
          <circle cx="5" cy="2" r="1.4" stroke="currentColor" strokeWidth="1.2" />
          <circle cx="5" cy="14" r="1.4" stroke="currentColor" strokeWidth="1.2" />
          <circle cx="11" cy="2.5" r="1.4" stroke="currentColor" strokeWidth="1.2" />
          <circle cx="11" cy="13" r="1.4" stroke="currentColor" strokeWidth="1.2" />
        </svg>
        <span>Fork 分支对话</span>
      </button>

      {otherWorkspaces.length > 0 && (
        <div className={`conversation-menu-sub${moveOpen ? " open" : ""}`}>
          <button
            type="button"
            role="menuitem"
            aria-haspopup="menu"
            aria-expanded={moveOpen}
            className="conversation-menu-item"
            disabled={busy}
            onClick={(event) => {
              event.stopPropagation();
              setMoveOpen((value) => !value);
            }}
          >
            <svg width="13" height="13" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <path
                d="M2 5.5A1.5 1.5 0 0 1 3.5 4h3l1.2 1.6h4.8A1.5 1.5 0 0 1 14 7.1v4.4A1.5 1.5 0 0 1 12.5 13h-9A1.5 1.5 0 0 1 2 11.5z"
                stroke="currentColor"
                strokeWidth="1.3"
                strokeLinejoin="round"
              />
            </svg>
            <span>移动到工作区</span>
            <span className="conversation-menu-chevron" aria-hidden="true">
              <svg width="9" height="9" viewBox="0 0 16 16" fill="none">
                <path d="M6 3l5 5-5 5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </span>
          </button>
          {moveOpen && (
            <div className="conversation-menu-flyout" role="menu" aria-label="选择工作区">
              {otherWorkspaces.map((workspace) => (
                <button
                  key={workspace.id}
                  type="button"
                  role="menuitem"
                  className="conversation-menu-item"
                  disabled={busy}
                  data-workspace-id={workspace.id}
                  onClick={(event) => {
                    event.stopPropagation();
                    onMoveToWorkspace(workspace.id);
                  }}
                >
                  <span className="conversation-menu-name">{workspace.name}</span>
                  <span className="conversation-menu-count">{workspace.conversation_count}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      <button
        type="button"
        role="menuitem"
        className="conversation-menu-item"
        disabled={busy}
        onClick={pick("archive")}
      >
        <svg width="13" height="13" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <path
            d="M2.5 5.5h11v7a1 1 0 0 1-1 1h-9a1 1 0 0 1-1-1v-7ZM1.5 2.5h13v3h-13zM6.5 8.5h3"
            stroke="currentColor"
            strokeWidth="1.3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <span>归档对话</span>
      </button>
      <button
        type="button"
        role="menuitem"
        className="conversation-menu-item danger"
        disabled={busy}
        onClick={pick("delete")}
      >
        <svg width="13" height="13" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <path
            d="M3 4.5h10M6.5 4.5V3h3v1.5M4.5 4.5l.7 8a1 1 0 0 0 1 .9h3.6a1 1 0 0 0 1-.9l.7-8"
            stroke="currentColor"
            strokeWidth="1.3"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <span>删除对话</span>
      </button>
    </div>
  );
}
