import { useCallback, useEffect, useRef, useState } from "react";
import { streamChat } from "../api/sse";
import type { AnswerBasis, ConversationMessage, Source } from "../types";
import { MarkdownContent } from "./MarkdownContent";
import { CitationTooltip } from "./CitationTooltip";
import { SourceFooter } from "./SourceFooter";
import BrandMark from "./BrandMark";

/** 单条消息：assistant 携带可审计的答案依据、策略版本、来源与流式状态。 */
interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  sources: Source[] | null;
  answerBasis: AnswerBasis | null;
  policyVersion: string | null;
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
    sources: Source[] | null,
    answerBasis: AnswerBasis | null,
    policyVersion: string | null,
  ) => void;
  /** 问答结束（含错误/中断）后回调：父级刷新 /api/status（隐私审计状态）。 */
  onChatComplete?: () => void;
  /** 用户 / Agent 头像 URL（设置页配置），null 时用内置 SVG。 */
  avatars?: { user: string | null; agent: string | null };
  /** 点击正文引用序号 / 来源脚注后回调：打开右侧来源详情面板。 */
  onCitationSelect?: (sources: Source[], index: number) => void;
  /** 联网搜索是否已配置（启用 + provider 凭证齐备）；false 时按钮置灰。 */
  searchConfigured?: boolean;
  /** 仅本地知识模式：后端不会联网，前端同步禁用联网按钮。 */
  knowledgeOnly?: boolean;
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
  return <BrandMark />;
}

function toChatMessages(messages: ConversationMessage[]): ChatMessage[] {
  return messages.map((message, index) => ({
    id: index + 1,
    role: message.role,
    content: message.content,
    sources: message.sources,
    answerBasis: message.answer_basis ?? null,
    policyVersion: message.policy_version ?? null,
    done: true,
    error: null,
  }));
}

const BASIS_LABELS: Record<AnswerBasis, string> = {
  knowledge: "基于本地知识",
  web: "包含联网信息",
  general: "一般回答",
  product: "产品说明",
  catalog: "知识概览",
  insufficient: "现有内容不足",
};

/** 问答界面（ChatGPT 风格）：空态建议 → 消息流（头像+内容）→ 底部圆角输入条。 */
export default function Chat({
  conversationId,
  initialMessages,
  onExchangeComplete,
  onChatComplete,
  avatars,
  onCitationSelect,
  searchConfigured = false,
  knowledgeOnly = false,
}: ChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [webSearch, setWebSearch] = useState(false);
  const [searching, setSearching] = useState(false);
  const [toolLabel, setToolLabel] = useState("正在联网搜索…");
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
    sources: Source[] | null;
    answerBasis: AnswerBasis | null;
    policyVersion: string | null;
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

  useEffect(() => {
    if (knowledgeOnly) setWebSearch(false);
  }, [knowledgeOnly]);

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
      sources: null,
      answerBasis: null,
      policyVersion: null,
      done: true,
      error: null,
    };
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      sources: null,
      answerBasis: null,
      policyVersion: null,
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
    exchangeRef.current = {
      user: message,
      assistant: "",
      sources: null,
      answerBasis: null,
      policyVersion: null,
      error: false,
    };

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      await streamChat(
        message,
        history,
        {
          onMeta: (sources, answerBasis, policyVersion) => {
            if (streamMessageIdRef.current !== assistantId) return;
            if (exchangeRef.current) {
              exchangeRef.current.sources = sources;
              exchangeRef.current.answerBasis = answerBasis;
              exchangeRef.current.policyVersion = policyVersion;
            }
            updateMessage(assistantId, (m) => ({
              ...m,
              sources,
              answerBasis,
              policyVersion,
            }));
          },
          onToken: (text) => {
            if (streamMessageIdRef.current !== assistantId) return;
            if (exchangeRef.current) exchangeRef.current.assistant += text;
            updateMessage(assistantId, (m) => ({ ...m, content: m.content + text }));
          },
          onTool: (tool, status) => {
            if (streamMessageIdRef.current !== assistantId) return;
            setSearching(status === "running");
            setToolLabel(tool === "image_verify" ? "正在核对图片…" : "正在联网搜索…");
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
        onExchangeComplete(
          exchange.user,
          exchange.assistant,
          exchange.sources,
          exchange.answerBasis,
          exchange.policyVersion,
        );
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
        disabled={!searchConfigured || knowledgeOnly || isSending}
        onClick={() => setWebSearch((value) => !value)}
        aria-pressed={webSearch}
        aria-label="联网搜索"
        title={
          knowledgeOnly
            ? "已开启仅使用本地知识"
            : searchConfigured
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
        aria-label="消息输入框"
      />
      <span className="composer-label">{webSearch ? "联网搜索已开启" : "从一个好问题开始"}</span>
      <button
        className="chat-send"
        type="submit"
        disabled={isSending || input.trim() === ""}
        aria-label="发送"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M12 19V5m-6 6 6-6 6 6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
    </form>
  );

  return (
    <div className="chat" ref={chatRootRef}>
      {messages.length === 0 ? (
        <div className="chat-empty">
          <div className="hero-emblem"><BrandMark /><span className="emblem-node" /></div>
          <p className="hero-eyebrow">连接知识 · 发现新知</p>
          <h2 className="chat-empty-title">让知识，<span>彼此连接。</span></h2>
          <p className="chat-empty-description">从散落的笔记到清晰的答案，和你的知识库聊聊。</p>
          <div className="chat-empty-input">{inputBar}</div>
          <div className="suggestion-chips">
            {SUGGESTIONS.map((suggestion, index) => (
              <button
                key={suggestion}
                type="button"
                className="suggestion-chip"
                onClick={() => void handleSend(suggestion)}
              >
                <span className="suggestion-number" aria-hidden="true">0{index + 1}</span>
                <span className="suggestion-title">{["探索知识库", "提炼文档重点", "认识 Everything RAG"][index]}</span>
                <span className="suggestion-description">{suggestion}</span>
                <span className="suggestion-arrow" aria-hidden="true">↗</span>
              </button>
            ))}
          </div>
          <p className="chat-empty-note">随时准备好，知无不言</p>
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
                {msg.role === "assistant" && msg.answerBasis !== null && (
                  <span className={`answer-basis answer-basis-${msg.answerBasis}`}>
                    {BASIS_LABELS[msg.answerBasis]}
                  </span>
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
              {toolLabel}
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
