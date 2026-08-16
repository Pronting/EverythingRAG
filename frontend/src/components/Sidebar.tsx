import { useState } from "react";
import type { ConversationSummary, StatusResponse } from "../types";
import BlockBrowser from "./BlockBrowser";
import ImportPanel from "./ImportPanel";

const KB_COLLAPSED_KEY = "everything-rag-kb-collapsed";
const SIDEBAR_COLLAPSED_KEY = "everything-rag-sidebar-collapsed";

/** 读取知识库卡片是否收起（未设置时默认收起，避免长期占用侧边栏视野）。 */
function readCollapsed(): boolean {
  try {
    const value = localStorage.getItem(KB_COLLAPSED_KEY);
    return value === null ? true : value === "1";
  } catch {
    return true;
  }
}

/** 读取侧边栏是否折叠（未设置时默认展开）。 */
function readSidebarCollapsed(): boolean {
  try {
    return localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "1";
  } catch {
    return false;
  }
}

interface SidebarProps {
  status: StatusResponse | null;
  statusError: string | null;
  conversations: ConversationSummary[];
  activeConversationId: string | null;
  onNewChat: () => void;
  onSelectConversation: (id: string) => void;
  onImported: () => void;
  onOpenSettings: () => void;
}

/** ChatGPT 风格左侧边栏：品牌 + 新对话 + 会话搜索/历史 + 知识库管理 + 设置入口 + 隐私状态；
 *  支持折叠为图标栏。 */
export default function Sidebar({
  status,
  statusError,
  conversations,
  activeConversationId,
  onNewChat,
  onSelectConversation,
  onImported,
  onOpenSettings,
}: SidebarProps) {
  const knowledge = status?.knowledge;
  const outboundState = status?.privacy.outbound_state ?? "local-only";
  const hasOutbound = outboundState !== "local-only";

  // 知识库卡片可折叠：收起后只留「知识库 + 文档数」一行，展开才显示导入面板
  const [collapsed, setCollapsed] = useState<boolean>(readCollapsed);
  // 整个侧边栏可折叠为图标栏
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(readSidebarCollapsed);
  // 会话搜索关键字
  const [query, setQuery] = useState("");
  // 语义块浏览弹窗
  const [blocksOpen, setBlocksOpen] = useState(false);

  const toggleCollapsed = (): void => {
    setCollapsed((prev) => {
      const next = !prev;
      try {
        localStorage.setItem(KB_COLLAPSED_KEY, next ? "1" : "0");
      } catch {
        /* 忽略：存储不可用时仅失去记忆 */
      }
      return next;
    });
  };

  const toggleSidebar = (): void => {
    setSidebarCollapsed((prev) => {
      const next = !prev;
      try {
        localStorage.setItem(SIDEBAR_COLLAPSED_KEY, next ? "1" : "0");
      } catch {
        /* 忽略：存储不可用时仅失去记忆 */
      }
      return next;
    });
  };

  const normalizedQuery = query.trim().toLowerCase();
  const filteredConversations =
    normalizedQuery === ""
      ? conversations
      : conversations.filter((c) => c.title.toLowerCase().includes(normalizedQuery));

  return (
    <aside className={`sidebar${sidebarCollapsed ? " collapsed" : ""}`}>
      <div className="sidebar-brand">
        <div className="sidebar-logo" aria-hidden="true">
          E
        </div>
        {!sidebarCollapsed && <h1 className="sidebar-title">Everything RAG</h1>}
      </div>

      <button type="button" className="new-chat-btn" onClick={onNewChat} aria-label="新对话" title="新对话">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <path d="M8 3v10M3 8h10" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
        </svg>
        {!sidebarCollapsed && <span>新对话</span>}
      </button>

      {!sidebarCollapsed && (
        <div className="conversation-search">
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <circle cx="7" cy="7" r="4.5" stroke="currentColor" strokeWidth="1.4" />
            <path d="M10.5 10.5 14 14" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
          </svg>
          <input
            className="conversation-search-input"
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="搜索对话"
            aria-label="搜索对话"
          />
        </div>
      )}

      {!sidebarCollapsed && (
        <nav className="conversation-list" aria-label="会话历史">
          {filteredConversations.map((conversation) => (
            <button
              key={conversation.id}
              type="button"
              className={`conversation-item${conversation.id === activeConversationId ? " active" : ""}`}
              onClick={() => onSelectConversation(conversation.id)}
              title={conversation.title}
            >
              <span className="conversation-item-title">{conversation.title}</span>
            </button>
          ))}
          {filteredConversations.length === 0 && (
            <p className="conversation-empty">
              {conversations.length === 0 ? "暂无会话，点击「新对话」开始" : "未找到匹配的会话"}
            </p>
          )}
        </nav>
      )}

      {!sidebarCollapsed && (
        <div className={`kb-card${collapsed ? " collapsed" : ""}`}>
          <button
            type="button"
            className="kb-card-header"
            onClick={toggleCollapsed}
            aria-expanded={!collapsed}
            aria-label={collapsed ? "展开知识库" : "收起知识库"}
            title={collapsed ? "展开知识库管理" : "收起知识库管理"}
          >
            <span className="kb-card-title">知识库</span>
            <span className="kb-count-badge">
              {knowledge !== undefined ? `${knowledge.file_count} 文档` : "—"}
            </span>
            <span className="kb-card-chevron" aria-hidden="true">
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <path
                  d="M3 4.5 6 7.5l3-3"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </span>
          </button>
          <div className="kb-card-body">
            <button
              type="button"
              className="kb-stats kb-stats-btn"
              onClick={() => setBlocksOpen(true)}
              disabled={(knowledge?.chunk_count ?? 0) === 0}
              title="点击浏览语义块（按文本/图片分类，支持搜索）"
            >
              共 <strong>{knowledge?.chunk_count ?? 0}</strong> 个语义块
              {(knowledge?.chunk_count ?? 0) > 0 ? " · 已导入" : " · 尚未导入"}
            </button>
            {knowledge?.image_tasks !== undefined && knowledge.image_tasks.total > 0 && (
              <p className="kb-stats kb-image-progress">
                图片识别 <strong>{knowledge.image_tasks.done}</strong>/{knowledge.image_tasks.total}
                {knowledge.image_tasks.pending > 0 && (
                  <span className="kb-image-pending"> · 生成中 {knowledge.image_tasks.pending}</span>
                )}
                {knowledge.image_tasks.failed > 0 && (
                  <span className="kb-image-failed"> · 失败 {knowledge.image_tasks.failed}</span>
                )}
              </p>
            )}
            <ImportPanel onImported={onImported} />
          </div>
        </div>
      )}

      <div className="sidebar-footer">
        {!sidebarCollapsed &&
          (statusError !== null ? (
            <span className="status-error">后端未连接</span>
          ) : (
            <>
              <span
                className={`privacy-dot${hasOutbound ? " has-outbound" : ""}`}
                aria-hidden="true"
              />
              <span className="sidebar-privacy">隐私基线：{outboundState}</span>
            </>
          ))}
        <button
          type="button"
          className="sidebar-collapse-btn"
          onClick={toggleSidebar}
          aria-label={sidebarCollapsed ? "展开侧边栏" : "收起侧边栏"}
          title={sidebarCollapsed ? "展开侧边栏" : "收起侧边栏"}
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            {sidebarCollapsed ? (
              <path d="M6 3l5 5-5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            ) : (
              <path d="M10 3 5 8l5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            )}
          </svg>
        </button>
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

      <BlockBrowser open={blocksOpen} onClose={() => setBlocksOpen(false)} />
    </aside>
  );
}
