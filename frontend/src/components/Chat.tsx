import { useCallback, useEffect, useRef, useState } from "react";
import type { ChangeEvent, KeyboardEvent } from "react";
import { streamChat } from "../api/sse";
import type { AnswerBasis, ConversationMessage, Source } from "../types";import { MarkdownContent } from "./MarkdownContent";
import { CitationTooltip } from "./CitationTooltip";
import { SourceFooter } from "./SourceFooter";
import BrandMark from "./BrandMark";

/** 单条消息：assistant 携带可审计的答案依据、策略版本、来源与流式状态。 */
interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  /** 推理模型思维链；空串表示本轮没有思维链（不渲染折叠块）。 */
  reasoning: string;
  sources: Source[] | null;
  answerBasis: AnswerBasis | null;
  policyVersion: string | null;
  done: boolean;
  /** 用户点「停止生成」中断的回复（保留已生成内容，仅提示已停止）。 */
  stopped: boolean;
  error: string | null;
}

interface ChatProps {
  /** 当前会话 id：变化时加载该会话的历史消息。 */
  conversationId: string | null;
  /** 当前会话的已持久化消息（切换会话时由父级传入）。 */
  initialMessages: ConversationMessage[];
  /**
   * 父级修订号：父级改写了 initialMessages 后自增，Chat 据此重新载入消息。
   *
   * 用于「编辑重发」——父级把历史截断到编辑点，需要 Chat 重新按新历史渲染。
   */
  revision?: number;
  /** 一轮问答完成（非错误）后回调，供父级持久化 + AI 标题。 */
  onExchangeComplete: (
    userContent: string,
    assistantContent: string,
    sources: Source[] | null,
    answerBasis: AnswerBasis | null,
    policyVersion: string | null,
    reasoning: string | null,
  ) => void;
  /**
   * 用户点铅笔「编辑重发」：父级把会话历史从该条起截断并返回新的消息列表。
   *
   * Chat 本地只负责把文本回填输入框、移除该条及其之后的消息；真正的历史截断
   * 由父级落库，因此重发不会追加出重复轮次。
   */
  onEditUserMessage?: (
    messageIndex: number,
    content: string,
  ) => Promise<ConversationMessage[] | null>;
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
/** 输入框单行高度上限（与 .chat-input 的 max-height 对齐；超出改为内部滚动）。 */
const INPUT_MAX_HEIGHT = 120;

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
    reasoning: message.reasoning ?? "",
    // 归一化为 null：后端历史 / 伪造响应可能整个缺省 sources 键（undefined）
    sources: message.sources ?? null,
    answerBasis: message.answer_basis ?? null,
    policyVersion: message.policy_version ?? null,
    done: true,
    stopped: false,
    error: null,
  }));
}

/**
 * 思维链折叠块：默认折叠，点击标题栏展开查看模型思考过程。
 *
 * 只在思维链含可读字符时渲染（纯标点/空白视为无内容），流式期间标题栏图标呼吸发光。
 */
function ThinkingBlock({
  reasoning,
  streaming,
}: {
  reasoning: string;
  streaming: boolean;
}) {
  const [open, setOpen] = useState(false);
  const readable = reasoning.replace(/[\s\p{P}\p{S}]/gu, "");
  if (readable === "") return null;
  return (
    <div className={`thinking-block${open ? " open" : ""}${streaming ? " active" : ""}`}>
      <button
        type="button"
        className="thinking-toggle"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M9 18h6M10 21h4M12 3a6 6 0 0 0-3.6 10.8c.5.4.8 1 .8 1.7V16h5.6v-.5c0-.7.3-1.3.8-1.7A6 6 0 0 0 12 3Z"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <span>{streaming ? "思考中…" : `思考过程（${reasoning.length} 字）`}</span>
        <span className="thinking-chevron" aria-hidden="true">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
            <path
              d="M6 9l6 6 6-6"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </span>
      </button>
      {open && (
        <div className="thinking-content" data-testid="thinking-content">
          {reasoning}
        </div>
      )}
    </div>
  );
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
  revision = 0,
  onExchangeComplete,
  onEditUserMessage,
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
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [hoveredCitation, setHoveredCitation] = useState<{ source: Source; rect: DOMRect } | null>(
    null,
  );
  const nextIdRef = useRef(1);
  const streamMessageIdRef = useRef<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  /** 用户主动「停止生成」：区分卸载/切会话中断，避免把半截回答写进会话历史。 */
  const stoppedRef = useRef(false);
  const chatRootRef = useRef<HTMLDivElement>(null);
  // 用户是否停留在滚动区底部：贴近底部时才自动跟随滚动（流式吐字时用户上滑可停止自动滚动）
  const isAtBottomRef = useRef(true);
  const exchangeRef = useRef<{
    user: string;
    assistant: string;
    reasoning: string;
    sources: Source[] | null;
    answerBasis: AnswerBasis | null;
    policyVersion: string | null;
    error: boolean;
  } | null>(null);

  /**
   * 输入框高度自适应：先归零再按内容量高，最多 INPUT_MAX_HEIGHT，超出内部滚动。
   *
   * 归零是必需的——否则清除多行长文本后 textarea 不会缩回单行。
   */
  const syncInputHeight = useCallback((): void => {
    const el = inputRef.current;
    if (el === null) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, INPUT_MAX_HEIGHT)}px`;
    el.style.overflowY = el.scrollHeight > INPUT_MAX_HEIGHT ? "auto" : "hidden";
  }, []);

  // 切换会话：加载历史消息、重置 id、回到底部跟随
  useEffect(() => {
    setMessages(toChatMessages(initialMessages));
    nextIdRef.current = initialMessages.length + 1;
    isAtBottomRef.current = true;
    // eslint-disable-next-line react-hooks/exhaustive-deps -- 仅在切换会话时加载
  }, [conversationId]);

  // 父级改写了历史（编辑重发截断）：按新历史重载消息，避免旧轮次残留
  useEffect(() => {
    if (revision === 0) return;
    setMessages(toChatMessages(initialMessages));
    nextIdRef.current = initialMessages.length + 1;
    // eslint-disable-next-line react-hooks/exhaustive-deps -- 仅在父级修订时重载
  }, [revision]);

  /**
   * 点铅笔「编辑重发」：把该条提问回填输入框，移除该条及其后的所有消息。
   *
   * 真正的历史截断交给父级落库（见 onEditUserMessage），因此重发直接追加即可，
   * 不会出现「旧卡片 + 新卡片」两份重复轮次。
   */
  const handleEditRequest = (messageId: number): void => {
    if (isSending) return;
    const index = messages.findIndex((message) => message.id === messageId);
    if (index < 0) return;
    const target = messages[index];
    if (target.role !== "user") return;
    const content = target.content;
    // 先本地移除（立即反馈），再让父级截断持久化历史
    setMessages((prev) => prev.slice(0, index));
    setInput(content);
    requestAnimationFrame(() => {
      syncInputHeight();
      inputRef.current?.focus();
    });
    void onEditUserMessage?.(index, content);
  };

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
      reasoning: "",
      sources: null,
      answerBasis: null,
      policyVersion: null,
      done: true,
      stopped: false,
      error: null,
    };
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      reasoning: "",
      sources: null,
      answerBasis: null,
      policyVersion: null,
      done: false,
      stopped: false,
      error: null,
    };
    // 发送新消息：回到底部跟随（用户可能在读历史时发送），输入框收回单行
    isAtBottomRef.current = true;
    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setInput("");
    setImages([]);
    setIsSending(true);
    stoppedRef.current = false;
    streamMessageIdRef.current = assistantId;
    exchangeRef.current = {
      user: message,
      assistant: "",
      reasoning: "",
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
          onReasoning: (text) => {
            if (streamMessageIdRef.current !== assistantId) return;
            if (exchangeRef.current) exchangeRef.current.reasoning += text;
            updateMessage(assistantId, (m) => ({ ...m, reasoning: m.reasoning + text }));
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
      // 兜底：流结束但未收到 done/error（如坏帧流 / 用户中断）时恢复 UI 并标记消息完成。
      const stopped = stoppedRef.current;
      stoppedRef.current = false;
      streamMessageIdRef.current = null;
      abortRef.current = null;
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId && !m.done && !m.error ? { ...m, done: true, stopped } : m,
        ),
      );
      setIsSending(false);
      // 用户主动停止的半截回答不写进会话历史（历史只保留完整轮次）；卸载中断同理丢弃。
      if (exchangeRef.current && !exchangeRef.current.error && !stopped) {
        const exchange = exchangeRef.current;
        onExchangeComplete(
          exchange.user,
          exchange.assistant,
          exchange.sources,
          exchange.answerBasis,
          exchange.policyVersion,
          exchange.reasoning === "" ? null : exchange.reasoning,
        );
      }
      onChatComplete?.(); // 问答结束刷新 status（隐私审计状态跟随）
    }
  };

  /** 停止生成：中断 SSE 流，保留已生成内容（半截回答不入历史）。 */
  const handleStop = (): void => {
    if (!isSending || abortRef.current === null) return;
    stoppedRef.current = true;
    abortRef.current.abort();
  };

  const handleInputChange = (event: ChangeEvent<HTMLTextAreaElement>): void => {
    setInput(event.target.value);
    syncInputHeight();
  };

  /** Enter 发送；Shift+Enter 换行（对齐主流对话输入习惯）。 */
  const handleInputKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>): void => {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) return;
    event.preventDefault();
    void handleSend();
  };

  const inputBar = (
    <form
      className="chat-input-bar"
      onSubmit={(e) => {
        e.preventDefault();
        void handleSend();
      }}
    >
      <textarea
        ref={inputRef}
        className="chat-input"
        value={input}
        rows={1}
        onChange={handleInputChange}
        onKeyDown={handleInputKeyDown}
        placeholder="随心输入"
        aria-label="消息输入框"
      />
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
      <span className="composer-label">{webSearch ? "联网搜索已开启" : ""}</span>
      {isSending ? (
        <button
          type="button"
          className="chat-stop"
          onClick={handleStop}
          aria-label="停止生成"
          title="停止生成"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
            <rect x="6" y="6" width="12" height="12" rx="2" />
          </svg>
        </button>
      ) : (
        <button
          className="chat-send"
          type="submit"
          disabled={input.trim() === ""}
          aria-label="发送"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M12 19V5m-6 6 6-6 6 6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      )}
    </form>
  );

  return (
    <div className="chat" ref={chatRootRef}>
      {messages.length === 0 ? (
        <div className="chat-empty">
          {/* 品牌 + 标题居中，输入条下沉到底部（参考 Codex 首页布局） */}
          <div className="chat-empty-center">
            <div className="hero-emblem"><BrandMark /><span className="emblem-node" /></div>
            <h2 className="chat-empty-title">问点什么？</h2>
          </div>
          <div className="chat-empty-composer">
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
                {msg.role === "assistant" && msg.answerBasis !== null && (
                  <span className={`answer-basis answer-basis-${msg.answerBasis}`}>
                    {BASIS_LABELS[msg.answerBasis]}
                  </span>
                )}
                {msg.role === "user" ? (
                  <div className="message-content user">
                    {msg.content}
                    {!isSending && onEditUserMessage !== undefined && (
                      <button
                        type="button"
                        className="message-edit-btn"
                        onClick={() => handleEditRequest(msg.id)}
                        aria-label="编辑这条提问"
                        title="编辑这条提问并重新发送"
                      >
                        <svg width="13" height="13" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                          <path
                            d="M11.2 2.3a1.4 1.4 0 0 1 2 2L5.6 11.9l-2.6.7.7-2.6z"
                            stroke="currentColor"
                            strokeWidth="1.3"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                        </svg>
                      </button>
                    )}
                  </div>
                ) : msg.error !== null ? (
                  <div className="message-error" role="alert">
                    {msg.error}
                  </div>
                ) : msg.content === "" && msg.reasoning === "" ? (
                  msg.stopped ? (
                    <div className="message-content stopped">已停止生成</div>
                  ) : (
                    <div className="message-content thinking">
                      <span className="thinking-spinner" aria-hidden="true" />
                      <span>{THINKING_TEXT}</span>
                    </div>
                  )
                ) : (
                  msg.content !== "" && (
                    <div className={`message-content${msg.done ? "" : " streaming"}`}>
                      <MarkdownContent
                        content={msg.content}
                        sources={msg.sources}
                        onCitationClick={onCitationSelect}
                        onCitationHover={handleCitationHover}
                      />
                    </div>
                  )
                )}
                {msg.role === "assistant" && msg.stopped && msg.content !== "" && (
                  <p className="message-stopped-note">已停止生成</p>
                )}
                {msg.role === "assistant" && msg.error === null && (
                  <ThinkingBlock reasoning={msg.reasoning} streaming={!msg.done && !msg.stopped} />
                )}
                {msg.role === "assistant" && (msg.sources?.length ?? 0) > 0 && (
                  <SourceFooter
                    sources={msg.sources ?? []}
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
