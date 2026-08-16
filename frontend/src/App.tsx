import { useCallback, useEffect, useRef, useState } from "react";
import {
  appendConversationMessages,
  createConversation,
  generateConversationTitle,
  getConversation,
  listConversations,
} from "./api/conversationApi";
import Chat from "./components/Chat";
import { CitationPanel } from "./components/CitationPanel";
import SettingsModal from "./components/SettingsModal";
import Sidebar from "./components/Sidebar";
import { getSettings } from "./api/settingsApi";
import { persistTheme, readStoredTheme } from "./theme";
import type {
  ConversationMessage,
  ConversationSummary,
  Source,
  StatusResponse,
  ThemeValue,
  ToastType,
} from "./types";
import "./App.css";

/** 应用外壳（ChatGPT 风格）：左侧边栏（会话历史/知识库/设置）+ 主聊天区。 */
export default function App() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [avatars, setAvatars] = useState<{ user: string | null; agent: string | null }>({
    user: null,
    agent: null,
  });
  const [theme, setTheme] = useState<ThemeValue>(() => readStoredTheme());
  const persistedThemeRef = useRef<ThemeValue>(readStoredTheme());
  const transitionTimeoutRef = useRef<number | null>(null);
  const [toast, setToast] = useState<{ type: ToastType; message: string } | null>(null);
  const toastTimerRef = useRef<number | null>(null);

  // 会话管理
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [chatMessages, setChatMessages] = useState<ConversationMessage[]>([]);
  const [citationPanel, setCitationPanel] = useState<{ sources: Source[]; index: number } | null>(
    null,
  );

  const loadStatus = useCallback((): void => {
    fetch("/api/status")
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json() as Promise<StatusResponse>;
      })
      .then((data) => setStatus(data))
      .catch((error: unknown) =>
        setStatusError(error instanceof Error ? error.message : String(error)),
      );
  }, []);

  const loadConversations = useCallback((): void => {
    listConversations()
      .then(setConversations)
      .catch((error: unknown) => setStatusError(error instanceof Error ? error.message : String(error)));
  }, []);

  // 应用主题到 <html>：animate=true 时带约 1.5s 全局过渡，否则即时（用于初始加载）。
  const applyTheme = useCallback((next: ThemeValue, animate: boolean): void => {
    const root = document.documentElement;
    // 清理上一次尚未结束的过渡定时器，避免快速连续切换时过渡被提前打断
    if (transitionTimeoutRef.current !== null) {
      window.clearTimeout(transitionTimeoutRef.current);
      transitionTimeoutRef.current = null;
    }
    if (animate && root.dataset.theme !== next) {
      root.classList.add("theme-transition");
      void root.offsetWidth; // 强制回流，确保过渡属性先生效再切换主题值
      root.dataset.theme = next;
      transitionTimeoutRef.current = window.setTimeout(() => {
        root.classList.remove("theme-transition");
        transitionTimeoutRef.current = null;
      }, 1600);
    } else if (root.dataset.theme !== next) {
      root.dataset.theme = next;
      root.classList.remove("theme-transition");
    }
    setTheme(next);
  }, []);

  // 加载设置（头像 + 主题；设置页保存后也刷新），失败静默回退默认
  const loadSettings = useCallback((): void => {
    getSettings()
      .then((view) => {
        setAvatars(view.avatars ?? { user: null, agent: null });
        const next = view.theme ?? "light";
        persistedThemeRef.current = next;
        persistTheme(next);
        applyTheme(next, false);
      })
      .catch(() => {
        /* 忽略：加载失败保持默认 */
      });
  }, [applyTheme]);

  useEffect(() => {
    loadStatus();
    loadConversations();
    loadSettings();
  }, [loadStatus, loadConversations, loadSettings]);

  // 设置页切换主题：实时应用 + 过渡动画（是否持久化由「保存」决定）
  const handleThemeChange = useCallback(
    (next: ThemeValue) => applyTheme(next, true),
    [applyTheme],
  );

  // 轻提示（成功/失败）：顶部居中弹出，4 秒后自动消失
  const showToast = useCallback((type: ToastType, message: string): void => {
    setToast({ type, message });
    if (toastTimerRef.current !== null) window.clearTimeout(toastTimerRef.current);
    toastTimerRef.current = window.setTimeout(() => setToast(null), 4000);
  }, []);

  const handleNewChat = async (): Promise<void> => {
    try {
      const conversation = await createConversation();
      setConversations((prev) => [conversation, ...prev]);
      setActiveConversationId(conversation.id);
      setChatMessages([]);
      setCitationPanel(null);
    } catch (error) {
      setStatusError(error instanceof Error ? error.message : String(error));
    }
  };

  const handleSelectConversation = async (id: string): Promise<void> => {
    if (id === activeConversationId) return;
    try {
      const conversation = await getConversation(id);
      setActiveConversationId(conversation.id);
      setChatMessages(conversation.messages);
      setCitationPanel(null);
    } catch (error) {
      setStatusError(error instanceof Error ? error.message : String(error));
    }
  };

  /** 一轮问答完成后：持久化消息；首次交互用 AI 生成标题；刷新侧边栏列表。
   *
   * 默认会话（未点「新对话」）首次问答时惰性新建会话——否则 activeConversationId
   * 为 null，本轮会被直接丢弃、不进侧边栏。
   */
  const handleExchangeComplete = useCallback(
    async (
      userContent: string,
      assistantContent: string,
      thinking: string,
      sources: Source[] | null,
    ) => {
      const newMessages: ConversationMessage[] = [
        {
          role: "user",
          content: userContent,
          thinking: null,
          sources: null,
          created_at: new Date().toISOString(),
        },
        {
          role: "assistant",
          content: assistantContent,
          thinking,
          sources,
          created_at: new Date().toISOString(),
        },
      ];
      const isFirst = chatMessages.length === 0;

      let conversationId = activeConversationId;
      if (conversationId === null) {
        // 惰性新建会话：同步更新 activeId + 消息列表，避免 Chat 因 id 变化重置而丢消息
        try {
          const conversation = await createConversation();
          conversationId = conversation.id;
          setActiveConversationId(conversation.id);
          setConversations((prev) => [conversation, ...prev]);
          setChatMessages(newMessages);
        } catch (error) {
          setStatusError(error instanceof Error ? error.message : String(error));
          return;
        }
      } else {
        setChatMessages((prev) => [...prev, ...newMessages]);
      }

      try {
        await appendConversationMessages(conversationId, newMessages);
        if (isFirst) {
          await generateConversationTitle(conversationId);
        }
        loadConversations();
      } catch (error) {
        setStatusError(error instanceof Error ? error.message : String(error));
      }
    },
    [activeConversationId, chatMessages.length, loadConversations],
  );

  const handleCitationSelect = useCallback((sources: Source[], index: number): void => {
    setCitationPanel({ sources, index });
  }, []);

  return (
    <div className="app">
      <Sidebar
        status={status}
        statusError={statusError}
        conversations={conversations}
        activeConversationId={activeConversationId}
        onNewChat={() => void handleNewChat()}
        onSelectConversation={(id) => void handleSelectConversation(id)}
        onImported={loadStatus}
        onOpenSettings={() => setSettingsOpen(true)}
      />
      <main className="app-main">
        <Chat
          conversationId={activeConversationId}
          initialMessages={chatMessages}
          onExchangeComplete={handleExchangeComplete}
          onChatComplete={loadStatus}
          avatars={avatars}
          onCitationSelect={handleCitationSelect}
          searchConfigured={status?.config.search_configured ?? false}
        />
      </main>
      {citationPanel !== null && (
        <CitationPanel
          sources={citationPanel.sources}
          index={citationPanel.index}
          onClose={() => setCitationPanel(null)}
          onSelect={(index) =>
            setCitationPanel((prev) => (prev === null ? prev : { ...prev, index }))
          }
        />
      )}
      <SettingsModal
        open={settingsOpen}
        onClose={() => {
          setSettingsOpen(false);
          // 未保存则回退到已持久化的主题
          applyTheme(persistedThemeRef.current, true);
        }}
        onSaved={() => {
          // 当前主题即为已保存主题（同步到持久化基准 + 本地快照）
          persistedThemeRef.current = theme;
          persistTheme(theme);
          loadStatus();
          loadSettings();
        }}
        theme={theme}
        onThemeChange={handleThemeChange}
        onNotify={showToast}
      />
      {toast !== null && (
        <div className={`toast toast-${toast.type}`} role="status" aria-live="polite">
          {toast.type === "success" ? "✓ " : "⚠ "}
          {toast.message}
        </div>
      )}
    </div>
  );
}
