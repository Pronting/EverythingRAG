import { useEffect, useState } from "react";
import Chat from "./components/Chat";
import type { StatusResponse } from "./types";
import "./App.css";

/** 应用外壳：页头（标题 + 隐私基线）+ 问答界面。status 拉取失败不阻塞 Chat。 */
export default function App() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);

  useEffect(() => {
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

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1 className="app-title">Everything RAG — 个人知识第二大脑</h1>
          <p className="app-subtitle">纯本地运行 · 答案永远带来源</p>
        </div>
        <div className="status-pill">
          {statusError !== null && <span className="status-error">后端未连接</span>}
          {status !== null && (
            <span>
              隐私基线：<strong>{status.privacy.outbound_state}</strong>
            </span>
          )}
        </div>
      </header>
      <main className="app-main">
        <Chat />
      </main>
    </div>
  );
}
