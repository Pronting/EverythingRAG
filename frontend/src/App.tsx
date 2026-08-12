import { useCallback, useEffect, useState } from "react";
import {
  appendConversationMessages,
  createConversation,
  generateConversationTitle,
  getConversation,
  listConversations,
} from "./api/conversationApi";
import Chat from "./components/Chat";
import SettingsModal from "./components/SettingsModal";
import Sidebar from "./components/Sidebar";
import type { ConversationMessage, ConversationSummary, Source, StatusResponse } from "./types";
import "./App.css";

/** 应用外壳（ChatGPT 风格）：左侧边栏（会话历史/知识库/设置）+ 主聊天区。 */
export default function App() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);

  // 会话管理
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [chatMessages, setChatMessages] = useState<ConversationMessage[]>([]);

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

  useEffect(() => {
    loadStatus();
    loadConversations();
  }, [loadStatus, loadConversations]);

  const handleNewChat = async (): Promise<void> => {
    try {
      const conversation = await createConversation();
      setConversations((prev) => [conversation, ...prev]);
      setActiveConversationId(conversation.id);
      setChatMessages([]);
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
    } catch (error) {
      setStatusError(error instanceof Error ? error.message : String(error));
    }
  };

  /** 一轮问答完成后：持久化消息；首次交互用 AI 生成标题；刷新侧边栏列表。 */
  const handleExchangeComplete = useCallback(
    async (userContent: string, assistantContent: string, sources: Source[] | null) => {
      if (activeConversationId === null) return;
      const isFirst = chatMessages.length === 0;
      try {
        await appendConversationMessages(activeConversationId, [
          { role: "user", content: userContent, sources: null },
          { role: "assistant", content: assistantContent, sources },
        ]);
        if (isFirst) {
          await generateConversationTitle(activeConversationId);
        }
        loadConversations();
      } catch (error) {
        setStatusError(error instanceof Error ? error.message : String(error));
      }
    },
    [activeConversationId, chatMessages.length, loadConversations],
  );

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
        />
      </main>
      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onSaved={loadStatus}
      />
    </div>
  );
}
