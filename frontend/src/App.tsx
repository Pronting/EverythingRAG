import { useEffect, useState } from "react";

interface Status {
  app: { name: string; version: string };
  privacy: { outbound_state: string };
}

/** MVP 占位页：显示后端 /api/status，验证前后端链路。 */
export default function App() {
  const [status, setStatus] = useState<Status | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/status")
      .then((r) => r.json())
      .then((d: Status) => setStatus(d))
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <main style={{ fontFamily: "system-ui", padding: 24, maxWidth: 720 }}>
      <h1>Everything RAG — 个人知识第二大脑</h1>
      <p>纯本地运行 · 答案永远带来源</p>
      <h2>后端状态</h2>
      {error && <p style={{ color: "#c00" }}>连接后端失败：{error}</p>}
      {status && (
        <>
          <p>
            应用 <strong>{status.app.name}</strong> v{status.app.version}
            ｜隐私基线：<strong>{status.privacy.outbound_state}</strong>
          </p>
          <pre style={{ background: "#f5f5f5", padding: 12, borderRadius: 8 }}>
            {JSON.stringify(status, null, 2)}
          </pre>
        </>
      )}
    </main>
  );
}
