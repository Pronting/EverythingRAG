import type { StatusResponse } from "../types";
import ImportPanel from "./ImportPanel";

interface SidebarProps {
  status: StatusResponse | null;
  statusError: string | null;
  onNewChat: () => void;
  onImported: () => void;
}

/** ChatGPT 风格左侧边栏：品牌 + 新对话 + 知识库管理（导入）+ 隐私状态。 */
export default function Sidebar({ status, statusError, onNewChat, onImported }: SidebarProps) {
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
            隐私基线：{outboundState}
          </>
        )}
      </div>
    </aside>
  );
}
