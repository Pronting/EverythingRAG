import type { StatusResponse } from "../types";
import ImportPanel from "./ImportPanel";

interface SidebarProps {
  status: StatusResponse | null;
  statusError: string | null;
  onNewChat: () => void;
  onImported: () => void;
  onOpenSettings: () => void;
}

/** ChatGPT 风格左侧边栏：品牌 + 新对话 + 知识库管理（导入）+ 设置入口 + 隐私状态。 */
export default function Sidebar({
  status,
  statusError,
  onNewChat,
  onImported,
  onOpenSettings,
}: SidebarProps) {
  const knowledge = status?.knowledge;
  const outboundState = status?.privacy.outbound_state ?? "local-only";
  const hasOutbound = outboundState !== "local-only";

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="sidebar-logo" aria-hidden="true">
          E
        </div>
        <div>
          <h1 className="sidebar-title">Everything RAG</h1>
          <p className="sidebar-subtitle">个人知识第二大脑</p>
        </div>
      </div>

      <button type="button" className="new-chat-btn" onClick={onNewChat}>
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <path d="M8 3v10M3 8h10" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
        </svg>
        新对话
      </button>

      <div className="kb-card">
        <div className="kb-card-header">
          <span className="kb-card-title">知识库</span>
          <span className="kb-count-badge">
            {knowledge !== undefined ? `${knowledge.file_count} 文档` : "—"}
          </span>
        </div>
        <p className="kb-stats">
          共 <strong>{knowledge?.chunk_count ?? 0}</strong> 个语义块
          {(knowledge?.chunk_count ?? 0) > 0 ? " · 已导入" : " · 尚未导入"}
        </p>
        <ImportPanel onImported={onImported} />
      </div>

      <div className="sidebar-footer">
        {statusError !== null ? (
          <span className="status-error">后端未连接</span>
        ) : (
          <>
            <span
              className={`privacy-dot${hasOutbound ? " has-outbound" : ""}`}
              aria-hidden="true"
            />
            <span className="sidebar-privacy">隐私基线：{outboundState}</span>
          </>
        )}
        <button
          type="button"
          className="sidebar-settings-btn"
          onClick={onOpenSettings}
          aria-label="设置"
          title="设置"
        >
          <svg width="15" height="15" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path
              d="M8 5a3 3 0 1 0 0 6 3 3 0 0 0 0-6Zm5.5 3a5.5 5.5 0 0 0-.08-.9l1.3-1a.4.4 0 0 0 .1-.5l-1.2-2.1a.4.4 0 0 0-.5-.15l-1.55.62a5.5 5.5 0 0 0-1.56-.9L9.7 2.3a.4.4 0 0 0-.4-.3H7.7a.4.4 0 0 0-.4.3l-.3 1.67a5.5 5.5 0 0 0-1.55.9L3.9 4.85a.4.4 0 0 0-.5.15L2.2 7.1a.4.4 0 0 0 .1.5l1.3 1a5.5 5.5 0 0 0 0 1.8l-1.3 1a.4.4 0 0 0-.1.5l1.2 2.1a.4.4 0 0 0 .5.15l1.55-.62c.47.36 1 .66 1.55.9l.3 1.67c.03.18.2.3.4.3h2.4a.4.4 0 0 0 .4-.3l.3-1.67a5.5 5.5 0 0 0 1.56-.9l1.54.62a.4.4 0 0 0 .5-.15l1.2-2.1a.4.4 0 0 0-.1-.5l-1.3-1a5.5 5.5 0 0 0 .09-.9Z"
              stroke="currentColor"
              strokeWidth="1.2"
            />
          </svg>
        </button>
      </div>
    </aside>
  );
}
