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

interface ChatProps {
  /** 会话 id：变化时清空消息（侧边栏「新对话」触发）。 */
  sessionId: number;
  /** 一次问答结束（含 done/error/中断）后回调：父级刷新 /api/status（隐私审计状态）。 */
  onChatComplete?: () => void;
}

const THINKING_TEXT = "思考中…";

const SUGGESTIONS = ["知识库里有什么内容？", "帮我总结一下导入的文档", "Everything RAG 是做什么的？"];

function UserAvatar() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M12 12a5 5 0 1 0 0-10 5 5 0 0 0 0 10Zm0 2c-4.7 0-8 3-8 6 0 1 .8 2 2 2h12c1.2 0 2-1 2-2 0-3-3.3-6-8-6Z" />
    </svg>
  );
}

function AssistantAvatar() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M12 2l1.9 5.6L19.5 9.5l-5.6 1.9L12 17l-1.9-5.6L4.5 9.5l5.6-1.9L12 2Z" />
      <path d="M19 14l.9 2.6 2.6.9-2.6.9L19 21l-.9-2.6-2.6-.9 2.6-.9L19 14Z" opacity=".7" />
    </svg>
  );
}

/** 问答界面（ChatGPT 风格）：空态建议 → 消息流（头像+内容）→ 底部圆角输入条。 */
export default function Chat({ sessionId, onChatComplete }: ChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const nextIdRef = useRef(1);
  const streamMessageIdRef = useRef<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // 新对话：清空消息、重置 id
  useEffect(() => {
    setMessages([]);
    nextIdRef.current = 1;
  }, [sessionId]);

  // 卸载时中断进行中的流
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const updateMessage = (id: number, update: (m: ChatMessage) => ChatMessage): void => {
    setMessages((prev) => prev.map((m) => (m.id === id ? update(m) : m)));
  };

  const handleSend = async (preset?: string): Promise<void> => {
    const message = (preset ?? input).trim();
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
      onChatComplete?.(); // 问答结束刷新 status（隐私审计状态跟随）
    }
  };

  return (
    <div className="chat">
      {messages.length === 0 ? (
        <div className="chat-empty">
          <h2>你想了解什么？</h2>
          <p>基于你的本地知识库，答案永远带来源</p>
          <div className="suggestion-chips">
            {SUGGESTIONS.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                className="suggestion-chip"
                onClick={() => void handleSend(suggestion)}
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="chat-messages" role="log" aria-live="polite">
          {messages.map((msg) => (
            <div key={msg.id} className={`message-row message-${msg.role}`}>
              <div
                className={`message-avatar ${msg.role}`}
                aria-hidden="true"
              >
                {msg.role === "user" ? <UserAvatar /> : <AssistantAvatar />}
              </div>
              <div className="message-body">
                {msg.role === "user" ? (
                  <div className="message-content user">{msg.content}</div>
                ) : msg.error !== null ? (
                  <div className="message-error" role="alert">
                    {msg.error}
                  </div>
                ) : msg.done ? (
                  <div className="message-content">
                    <MarkdownContent content={msg.content} />
                  </div>
                ) : msg.content === "" ? (
                  <div className="message-content thinking">{THINKING_TEXT}</div>
                ) : (
                  <div className="message-content streaming">{msg.content}</div>
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
            </div>
          ))}
        </div>
      )}

      <div className="chat-input-wrap">
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
          <button
            className="chat-send"
            type="submit"
            disabled={isSending || input.trim() === ""}
            aria-label="发送"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M4 11.5 20 4l-7.5 16-2-6.5L4 11.5Z" />
            </svg>
          </button>
        </form>
      </div>
    </div>
  );
}
