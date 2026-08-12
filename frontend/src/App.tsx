import { useCallback, useEffect, useState } from "react";
import Chat from "./components/Chat";
import SettingsModal from "./components/SettingsModal";
import Sidebar from "./components/Sidebar";
import type { StatusResponse } from "./types";
import "./App.css";

/** 应用外壳（ChatGPT 风格）：左侧边栏（知识库/导入/设置）+ 主聊天区。 */
export default function App() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState(0);
  const [settingsOpen, setSettingsOpen] = useState(false);

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

  useEffect(() => {
    loadStatus();
  }, [loadStatus]);

  const handleNewChat = useCallback((): void => {
    setSessionId((current) => current + 1);
  }, []);

  return (
    <div className="app">
      <Sidebar
        status={status}
        statusError={statusError}
        onNewChat={handleNewChat}
        onImported={loadStatus}
        onOpenSettings={() => setSettingsOpen(true)}
      />
      <main className="app-main">
        <Chat sessionId={sessionId} onChatComplete={loadStatus} />
      </main>
      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onSaved={loadStatus}
      />
    </div>
  );
}
