import { useCallback, useEffect, useRef, useState } from "react";
import { streamChat } from "../api/sse";
import type { ConversationMessage, Source } from "../types";
import { MarkdownContent } from "./MarkdownContent";
import { CitationTooltip } from "./CitationTooltip";
import { SourceFooter } from "./SourceFooter";

/** 单条消息：user / assistant；assistant 携带思维链、来源块、流式状态与错误。 */
interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  thinking: string;
  sources: Source[] | null;
  done: boolean;
  error: string | null;
}

interface ChatProps {
  /** 当前会话 id：变化时加载该会话的历史消息。 */
  conversationId: string | null;
  /** 当前会话的已持久化消息（切换会话时由父级传入）。 */
  initialMessages: ConversationMessage[];
  /** 一轮问答完成（非错误）后回调，供父级持久化 + AI 标题。 */
  onExchangeComplete: (
    userContent: string,
    assistantContent: string,
    thinking: string,
    sources: Source[] | null,
  ) => void;
  /** 问答结束（含错误/中断）后回调：父级刷新 /api/status（隐私审计状态）。 */
  onChatComplete?: () => void;
  /** 用户 / Agent 头像 URL（设置页配置），null 时用内置 SVG。 */
  avatars?: { user: string | null; agent: string | null };
  /** 点击正文引用序号 / 来源脚注后回调：打开右侧来源详情面板。 */
  onCitationSelect?: (sources: Source[], index: number) => void;
  /** 联网搜索是否已配置（启用 + provider 凭证齐备）；false 时按钮置灰。 */
  searchConfigured?: boolean;
}

const THINKING_TEXT = "思考中…";

const SUGGESTIONS = ["知识库里有什么内容？", "帮我总结一下导入的文档", "Everything RAG 是做什么的？"];

function UserAvatar({ src }: { src: string | null }) {
  if (src !== null) {
    return <img src={src} alt="" className="avatar-img" />;
  }
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M12 12a5 5 0 1 0 0-10 5 5 0 0 0 0 10Zm0 2c-4.7 0-8 3-8 6 0 1 .8 2 2 2h12c1.2 0 2-1 2-2 0-3-3.3-6-8-6Z" />
    </svg>
  );
}

function AssistantAvatar({ src }: { src: string | null }) {
  if (src !== null) {
    return <img src={src} alt="" className="avatar-img" />;
  }
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M12 2l1.9 5.6L19.5 9.5l-5.6 1.9L12 17l-1.9-5.6L4.5 9.5l5.6-1.9L12 2Z" />
      <path d="M19 14l.9 2.6 2.6.9-2.6.9L19 21l-.9-2.6-2.6-.9 2.6-.9L19 14Z" opacity=".7" />
    </svg>
  );
}

function toChatMessages(messages: ConversationMessage[]): ChatMessage[] {
  return messages.map((message, index) => ({
    id: index + 1,
    role: message.role,
    content: message.content,
    thinking: message.thinking ?? "",
    sources: message.sources,
    done: true,
    error: null,
  }));
}

/** 思维链是否有实际内容：排除纯空白/纯标点（如模型偶发的「，。」），避免渲染空「思考过程」块。 */
function isMeaningfulThinking(text: string): boolean {
  return /[^\s\p{P}]/u.test(text);
}

/** 思维链块：DeepSeek 风格，默认折叠，点击展开查看推理过程；active 时灯泡呼吸闪烁。 */
function ThinkingBlock({ text, active = false }: { text: string; active?: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <div className={`thinking-block${open ? " open" : ""}${active ? " active" : ""}`}>
      <button
        type="button"
        className="thinking-toggle"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
      >
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M12 3c-2 0-3.5 1.2-4 2.6C7 6 7.2 7 8 7.4 6.5 8 5.5 9.3 5.5 11c0 1.4.7 2.6 1.8 3.4-.9.8-1.5 2-1.5 3.3 0 2.4 1.9 4.3 6.2 4.3s6.2-1.9 6.2-4.3c0-1.3-.6-2.5-1.5-3.3 1.1-.8 1.8-2 1.8-3.4 0-1.7-1-3-2.5-3.6.8-.4 1-1.4.5-2.4C15.5 4.2 14 3 12 3Zm0 2c1 0 1.6.5 1.8 1.1.2.6-.1 1.1-.7 1.2l-.5.1v2.6c.4.4.8 1 .8 1.8 0 .9-.7 1.6-1.4 1.9-.7-.3-1.4-1-1.4-1.9 0-.8.4-1.4.8-1.8V7.4l-.5-.1c-.6-.1-.9-.6-.7-1.2C10.4 5.5 11 5 12 5Z"
            fill="currentColor"
          />
        </svg>
        思考过程
        <span className="thinking-chevron" aria-hidden="true">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
            <path d="M2.5 4.5 6 8l3.5-3.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </span>
      </button>
      {open && <div className="thinking-content">{text}</div>}
    </div>
  );
}

/** 问答界面（ChatGPT 风格）：空态建议 → 消息流（头像+内容）→ 底部圆角输入条。 */
export default function Chat({
  conversationId,
  initialMessages,
  onExchangeComplete,
  onChatComplete,
  avatars,
  onCitationSelect,
  searchConfigured = false,
}: ChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [webSearch, setWebSearch] = useState(false);
  const [searching, setSearching] = useState(false);
  // 贴图：data URI（src）+ 文件名，纯文本模型由后端识图代理转文字
  const [images, setImages] = useState<{ src: string; name: string }[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [hoveredCitation, setHoveredCitation] = useState<{ source: Source; rect: DOMRect } | null>(
    null,
  );
  const nextIdRef = useRef(1);
  const streamMessageIdRef = useRef<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const chatRootRef = useRef<HTMLDivElement>(null);
  // 用户是否停留在滚动区底部：贴近底部时才自动跟随滚动（流式吐字时用户上滑可停止自动滚动）
  const isAtBottomRef = useRef(true);
  const exchangeRef = useRef<{
    user: string;
    assistant: string;
    thinking: string;
    sources: Source[] | null;
    error: boolean;
  } | null>(null);

  // 切换会话：加载历史消息、重置 id、回到底部跟随
  useEffect(() => {
    setMessages(toChatMessages(initialMessages));
    nextIdRef.current = initialMessages.length + 1;
    isAtBottomRef.current = true;
    // eslint-disable-next-line react-hooks/exhaustive-deps -- 仅在切换会话时加载
  }, [conversationId]);

  // 跟踪滚动容器（.app-main = .chat 的父级）是否贴底：用户上滑后 isAtBottomRef 置 false。
  useEffect(() => {
    const scroller = chatRootRef.current?.parentElement;
    if (!scroller) return;
    const onScroll = (): void => {
      const distance = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight;
      isAtBottomRef.current = distance < 40;
    };
    scroller.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => scroller.removeEventListener("scroll", onScroll);
  }, []);

  // 消息变化（新消息/流式 token 追加）时，仅当用户在底部才滚动到底部，
  // 避免流式吐字过程中把用户向上滑动阅读的动作强行拽回底部。
  useEffect(() => {
    if (!isAtBottomRef.current) return;
    const scroller = chatRootRef.current?.parentElement;
    if (scroller) scroller.scrollTop = scroller.scrollHeight;
  }, [messages]);

  // 卸载时中断进行中的流
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const updateMessage = (id: number, update: (m: ChatMessage) => ChatMessage): void => {
    setMessages((prev) => prev.map((m) => (m.id === id ? update(m) : m)));
  };

  const handleCitationHover = useCallback((source: Source | null, rect: DOMRect | null): void => {
    setHoveredCitation(source !== null && rect !== null ? { source, rect } : null);
  }, []);

  // 选图：读为 data URI 加入待发队列（最多 4 张，与后端一致）
  const handleAddImages = useCallback((files: FileList | null): void => {
    if (!files) return;
    const remaining = Math.max(0, 4 - images.length);
    Array.from(files)
      .slice(0, remaining)
      .forEach((file) => {
        const reader = new FileReader();
        reader.onload = () => {
          const src = typeof reader.result === "string" ? reader.result : "";
          if (src !== "") setImages((prev) => [...prev, { src, name: file.name }]);
        };
        reader.readAsDataURL(file);
      });
  }, [images.length]);

  const handleSend = async (preset?: string): Promise<void> => {
    const message = (preset ?? input).trim();
    if (message === "" || isSending) return;
    // 快照本轮联网搜索开关（发送后按钮可继续切换，不影响已发出请求）
    const useWebSearch = webSearch;
    // 快照本轮贴图（发送后清空输入，但请求体仍带本轮的图）
    const sentImages = images.map((img) => img.src);

    // 多轮上下文：把历史已完成消息（最近 12 条）传给后端，承接前文（代称/缩写/话题）
    const history = messages
      .filter((m) => (m.role === "user" || m.role === "assistant") && m.done && m.error === null)
      .slice(-12)
      .map((m) => ({ role: m.role as "user" | "assistant", content: m.content }));

    const userId = nextIdRef.current++;
    const assistantId = nextIdRef.current++;
    const userMessage: ChatMessage = {
      id: userId,
      role: "user",
      content: message,
      thinking: "",
      sources: null,
      done: true,
      error: null,
    };
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      thinking: "",
      sources: null,
      done: false,
      error: null,
    };
    // 发送新消息：回到底部跟随（用户可能在读历史时发送）
    isAtBottomRef.current = true;
    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setInput("");
    setImages([]);
    setIsSending(true);
    streamMessageIdRef.current = assistantId;
    exchangeRef.current = { user: message, assistant: "", thinking: "", sources: null, error: false };

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await streamChat(
        message,
        history,
        {
          onMeta: (sources) => {
            if (streamMessageIdRef.current !== assistantId) return;
            if (exchangeRef.current) exchangeRef.current.sources = sources;
            updateMessage(assistantId, (m) => ({ ...m, sources }));
          },
          onToken: (text) => {
            if (streamMessageIdRef.current !== assistantId) return;
            if (exchangeRef.current) exchangeRef.current.assistant += text;
            updateMessage(assistantId, (m) => ({ ...m, content: m.content + text }));
          },
          onReasoning: (text) => {
            if (streamMessageIdRef.current !== assistantId) return;
            if (exchangeRef.current) exchangeRef.current.thinking += text;
            updateMessage(assistantId, (m) => ({ ...m, thinking: m.thinking + text }));
          },
          onTool: (_tool, status) => {
            if (streamMessageIdRef.current !== assistantId) return;
            setSearching(status === "running");
          },
          onDone: () => {
            if (streamMessageIdRef.current !== assistantId) return;
            streamMessageIdRef.current = null;
            abortRef.current = null;
            setSearching(false);
            updateMessage(assistantId, (m) => ({ ...m, done: true }));
            setIsSending(false);
          },
          onError: (errorMessage) => {
            if (streamMessageIdRef.current !== assistantId) return;
            streamMessageIdRef.current = null;
            abortRef.current = null;
            setSearching(false);
            if (exchangeRef.current) exchangeRef.current.error = true;
            updateMessage(assistantId, (m) => ({ ...m, done: true, error: errorMessage }));
            setIsSending(false);
          },
        },
        controller.signal,
        { webSearch: useWebSearch, images: sentImages },
      );
    } finally {
      setSearching(false);
      // 兜底：流结束但未收到 done/error（如坏帧流）时恢复 UI 并标记消息完成。
      streamMessageIdRef.current = null;
      abortRef.current = null;
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId && !m.done && !m.error ? { ...m, done: true } : m,
        ),
      );
      setIsSending(false);
      if (exchangeRef.current && !exchangeRef.current.error) {
        const exchange = exchangeRef.current;
        onExchangeComplete(exchange.user, exchange.assistant, exchange.thinking, exchange.sources);
      }
      onChatComplete?.(); // 问答结束刷新 status（隐私审计状态跟随）
    }
  };

  const inputBar = (
    <form
      className="chat-input-bar"
      onSubmit={(e) => {
        e.preventDefault();
        void handleSend();
      }}
    >
      {images.length > 0 && (
        <div className="chat-image-chips">
          {images.map((img, index) => (
            <div key={`${img.name}-${index}`} className="chat-image-chip">
              <img src={img.src} alt={img.name} />
              <button
                type="button"
                className="chat-image-remove"
                onClick={() => setImages((prev) => prev.filter((_, i) => i !== index))}
                aria-label={`移除图片 ${img.name}`}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp,image/gif"
        multiple
        hidden
        onChange={(e) => {
          handleAddImages(e.target.files);
          e.target.value = "";
        }}
      />
      <button
        type="button"
        className="chat-attach"
        disabled={isSending || images.length >= 4}
        onClick={() => fileInputRef.current?.click()}
        aria-label="贴图"
        title="贴图（纯文本模型会先识图转文字）"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <rect x="3" y="5" width="18" height="14" rx="2" stroke="currentColor" strokeWidth="1.6" />
          <circle cx="9" cy="10" r="1.6" stroke="currentColor" strokeWidth="1.4" />
          <path
            d="M5 17l4.5-4.5 3 3L15 13l4 4"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>
      <button
        type="button"
        className={`chat-websearch${webSearch ? " active" : ""}`}
        disabled={!searchConfigured || isSending}
        onClick={() => setWebSearch((value) => !value)}
        aria-pressed={webSearch}
        aria-label="联网搜索"
        title={
          searchConfigured
            ? "联网搜索（开启后本问题会检索网络）"
            : "联网搜索未配置，请在设置中启用并配置"
        }
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.6" />
          <path
            d="M3 12h18M12 3c2.5 2.6 4 5.6 4 9s-1.5 6.4-4 9c-2.5-2.6-4-5.6-4-9s1.5-6.4 4-9Z"
            stroke="currentColor"
            strokeWidth="1.6"
          />
        </svg>
      </button>
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
  );

  return (
    <div className="chat" ref={chatRootRef}>
      {messages.length === 0 ? (
        <div className="chat-empty">
          <h2 className="chat-empty-title">随时准备好，知无不言</h2>
          <div className="chat-empty-input">{inputBar}</div>
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
              <div className={`message-avatar ${msg.role}`} aria-hidden="true">
                {msg.role === "user" ? (
                  <UserAvatar src={avatars?.user ?? null} />
                ) : (
                  <AssistantAvatar src={avatars?.agent ?? null} />
                )}
              </div>
              <div className="message-body">
                {msg.role === "assistant" && isMeaningfulThinking(msg.thinking) && (
                  <ThinkingBlock text={msg.thinking} active={!msg.done} />
                )}
                {msg.role === "user" ? (
                  <div className="message-content user">{msg.content}</div>
                ) : msg.error !== null ? (
                  <div className="message-error" role="alert">
                    {msg.error}
                  </div>
                ) : msg.content === "" ? (
                  <div className="message-content thinking">
                    <span className="thinking-spinner" aria-hidden="true" />
                    <span>{THINKING_TEXT}</span>
                  </div>
                ) : (
                  <div className={`message-content${msg.done ? "" : " streaming"}`}>
                    <MarkdownContent
                      content={msg.content}
                      sources={msg.sources}
                      onCitationClick={onCitationSelect}
                      onCitationHover={handleCitationHover}
                    />
                  </div>
                )}
                {msg.role === "assistant" && msg.sources !== null && msg.sources.length > 0 && (
                  <SourceFooter
                    sources={msg.sources}
                    onSelect={(index) => {
                      if (msg.sources) onCitationSelect?.(msg.sources, index);
                    }}
                  />
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {messages.length !== 0 && (
        <div className="chat-input-wrap">
          {searching && (
            <div className="chat-searching" role="status" aria-live="polite">
              <span className="chat-searching-dot" aria-hidden="true" />
              正在联网搜索…
            </div>
          )}
          {inputBar}
        </div>
      )}

      {hoveredCitation !== null && (
        <CitationTooltip source={hoveredCitation.source} anchor={hoveredCitation.rect} />
      )}
    </div>
  );
}
