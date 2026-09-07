import { useEffect, useState } from "react";
import type { ConversationSummary } from "../types";
import { listConversations } from "../api/conversationApi";
import BrandMark from "./BrandMark";

const SIDEBAR_COLLAPSED_KEY = "everything-rag-sidebar-collapsed";

/** 读取侧边栏是否折叠（未设置时默认展开）。 */
function readSidebarCollapsed(): boolean {
  try {
    const stored = localStorage.getItem(SIDEBAR_COLLAPSED_KEY);
    return stored === null ? window.matchMedia("(max-width: 700px)").matches : stored === "1";
  } catch {
    return false;
  }
}

interface SidebarProps {
  statusError: string | null;
  conversations: ConversationSummary[];
  activeConversationId: string | null;
  onNewChat: () => void;
  onSelectConversation: (id: string) => void;
  onOpenSettings: () => void;
}

/** ChatGPT 风格左侧边栏：品牌 + 新对话 + 会话搜索/历史 + 知识库管理 + 设置入口 + 隐私状态；
 *  支持折叠为图标栏。 */
export default function Sidebar({
  statusError,
  conversations,
  activeConversationId,
  onNewChat,
  onSelectConversation,
  onOpenSettings,
}: SidebarProps) {
  // 整个侧边栏可折叠为图标栏
  const [sidebarCollapsed, setSidebarCollapsed] = useState<boolean>(readSidebarCollapsed);
  // 会话搜索关键字
  const [query, setQuery] = useState("");
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

  const normalizedQuery = query.trim();
  const [results, setResults] = useState<ConversationSummary[]>([]);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    if (!normalizedQuery) { setSearching(false); setSearchError(false); return; }
    setSearching(true);
    setResults([]);
    setSearchError(false);
    const timer = window.setTimeout(() => {
      listConversations(normalizedQuery, controller.signal).then((items) => {
        if (!controller.signal.aborted) setResults(items);
      }).catch(() => {
        if (!controller.signal.aborted) setSearchError(true);
      }).finally(() => {
        if (!controller.signal.aborted) setSearching(false);
      });
    }, 250);
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [normalizedQuery, conversations]);
  const filteredConversations = normalizedQuery ? results : conversations;

  return (
    <aside className={`sidebar${sidebarCollapsed ? " collapsed" : ""}`}>
      <div className="sidebar-brand">
        <div className="sidebar-logo" aria-hidden="true">
          <BrandMark />
        </div>
        {!sidebarCollapsed && <div><h1 className="sidebar-title">Everything RAG</h1><span className="brand-caption">你的个人知识空间</span></div>}
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
            placeholder="搜索标题和内容"
            maxLength={200}
            aria-label="搜索对话"
          />
        </div>
      )}

      {!sidebarCollapsed && (
        <nav className="conversation-list" aria-label="会话历史">
          <div className="section-eyebrow">{normalizedQuery ? "搜索结果" : "最近对话"} <span>{filteredConversations.length.toString().padStart(2, "0")}</span></div>
          {filteredConversations.map((conversation) => (
            <button
              key={conversation.id}
              type="button"
              className={`conversation-item${conversation.id === activeConversationId ? " active" : ""}`}
              onClick={() => onSelectConversation(conversation.id)}
              title={conversation.title}
            >
              <span className="conversation-item-title">{conversation.title}</span>
              {normalizedQuery && conversation.match_excerpt && <span className="conversation-match">{conversation.match_excerpt}</span>}
            </button>
          ))}
          {filteredConversations.length === 0 && (
            <p className="conversation-empty">
              {searching ? "搜索中…" : searchError ? "搜索失败，请稍后重试" : normalizedQuery ? "未找到匹配的对话" : "暂无会话，点击「新对话」开始"}
            </p>
          )}
        </nav>
      )}

      <div className="sidebar-footer">
        {!sidebarCollapsed && statusError !== null && <span className="status-error">后端未连接</span>}
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

    </aside>
  );
}
