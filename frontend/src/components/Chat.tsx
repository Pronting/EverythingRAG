import { useEffect, useRef, useState } from "react";
import { streamChat } from "../api/sse";
import type { Source } from "../types";
import { MarkdownContent } from "./MarkdownContent";
import { SourceCard } from "./SourceCard";

/** 单条消息：user / assistant；assistant 携带来源块、流式状态与错误。 */
interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  sources: Source[] | null;
  done: boolean;
  error: string | null;
}

const THINKING_TEXT = "思考中…";

/** 问答界面：输入 -> POST /api/chat -> 手写 SSE 解析 -> 流式渲染 + 来源块。 */
export default function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const nextIdRef = useRef(1);
  const streamMessageIdRef = useRef<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // 卸载时中断进行中的流
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const updateMessage = (id: number, update: (m: ChatMessage) => ChatMessage): void => {
    setMessages((prev) => prev.map((m) => (m.id === id ? update(m) : m)));
  };

  const handleSend = async (): Promise<void> => {
    const message = input.trim();
    if (message === "" || isSending) return;

    const userId = nextIdRef.current++;
    const assistantId = nextIdRef.current++;
    const userMessage: ChatMessage = {
      id: userId,
      role: "user",
      content: message,
      sources: null,
      done: true,
      error: null,
    };
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      sources: null,
      done: false,
      error: null,
    };
    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setInput("");
    setIsSending(true);
    streamMessageIdRef.current = assistantId;

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await streamChat(
        message,
        {
          onMeta: (sources) => {
            if (streamMessageIdRef.current !== assistantId) return;
            updateMessage(assistantId, (m) => ({ ...m, sources }));
          },
          onToken: (text) => {
            if (streamMessageIdRef.current !== assistantId) return;
            updateMessage(assistantId, (m) => ({ ...m, content: m.content + text }));
          },
          onDone: () => {
            if (streamMessageIdRef.current !== assistantId) return;
            streamMessageIdRef.current = null;
            abortRef.current = null;
            updateMessage(assistantId, (m) => ({ ...m, done: true }));
            setIsSending(false);
          },
          onError: (errorMessage) => {
            if (streamMessageIdRef.current !== assistantId) return;
            streamMessageIdRef.current = null;
            abortRef.current = null;
            updateMessage(assistantId, (m) => ({ ...m, done: true, error: errorMessage }));
            setIsSending(false);
          },
        },
        controller.signal,
      );
    } finally {
      // 兜底：流结束但未收到 done/error（如坏帧流）时恢复 UI 并标记消息完成。
      streamMessageIdRef.current = null;
      abortRef.current = null;
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId && !m.done && !m.error ? { ...m, done: true } : m,
        ),
      );
      setIsSending(false);
    }
  };

  return (
    <div className="chat">
      <div className="chat-messages" role="log" aria-live="polite">
        {messages.length === 0 && (
          <p className="chat-empty">输入问题，基于你的本地知识库回答。</p>
        )}
        {messages.map((msg) => (
          <div key={msg.id} className={`message message-${msg.role}`}>
            <div className="message-role">{msg.role === "user" ? "你" : "助手"}</div>
            {msg.role === "user" ? (
              <div className="message-content user-content">{msg.content}</div>
            ) : msg.error !== null ? (
              <div className="message-error" role="alert">
                {msg.error}
              </div>
            ) : msg.done ? (
              <div className="message-content assistant-content">
                <MarkdownContent content={msg.content} />
              </div>
            ) : msg.content === "" ? (
              <div className="message-content assistant-content thinking">{THINKING_TEXT}</div>
            ) : (
              <div className="message-content assistant-content streaming-text">{msg.content}</div>
            )}
            {msg.role === "assistant" && msg.sources !== null && msg.sources.length > 0 && (
              <div className="sources">
                <h3 className="sources-title">来源</h3>
                {msg.sources.map((s) => (
                  <SourceCard key={s.block_id} source={s} />
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      <form
        className="chat-input-bar"
        onSubmit={(e) => {
          e.preventDefault();
          void handleSend();
        }}
      >
        <input
          className="chat-input"
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="问点什么，比如：这份文档讲了什么？"
          disabled={isSending}
          aria-label="消息输入框"
        />
        <button className="chat-send" type="submit" disabled={isSending || input.trim() === ""}>
          {isSending ? "生成中…" : "发送"}
        </button>
      </form>
    </div>
  );
}
