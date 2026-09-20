import { useCallback, useEffect, useRef, useState } from "react";
import type { ChangeEvent, KeyboardEvent } from "react";
import {
  MAX_RUN_IMAGES,
  cancelRun,
  createRun,
  describeRunError,
  describeRunImage,
  nextClientRequestId,
  subscribeRunEvents,
  submitApproval,
  submitInput,
  summarizeRunImages,
} from "../api/runApi";
import type { RunSubscription } from "../api/runApi";
import { rememberConversationEpoch } from "../api/conversationApi";
import type {
  AnswerBasis,
  ApprovalDecisionKind,
  ApprovalRequest,
  ConversationMessage,
  InputRequest,
  PermissionMode,
  ProcessState,
  ReasoningLevel,
  RetrievalMeta,
  RunDetail,
  RunEventTyped,
  RunImageRef,
  RunStatus,
  Source,
} from "../types";
import { applyAnswerEvent, answerText, citationsToSources, emptyAnswerState } from "../run/answerState";
import type { AnswerState } from "../run/answerState";
import {
  applyRunEvent,
  emptyProcessState,
  profileIdFromCitations,
  relevanceEmptyNote,
  retrievalFromRerankTool,
  runningEntryText,
  terminalStatusText,
  withOriginRun,
  withRunSettings,
} from "../run/processState";
import { MarkdownContent } from "./MarkdownContent";
import { CitationTooltip } from "./CitationTooltip";
import { SourceFooter } from "./SourceFooter";
import { ProcessArea } from "./ProcessArea";
import { ApprovalCard, ClarificationCard } from "./ApprovalCard";
import { PERMISSION_LABELS, REASONING_LABELS, PermissionMenu } from "./PermissionMenu";
import BrandMark from "./BrandMark";

/** 单条消息：assistant 携带可审计的答案依据、来源与任务过程中的卡片。 */
interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  /** V1.1 落库的思维链（历史会话仍可能带）。V1.2 由过程区域取代。 */
  reasoning: string;
  sources: Source[] | null;
  /**
   * 本轮相关性筛选状态（PRD §8.2）。
   *
   * run 架构没有 `meta.retrieval`：该值由本轮 `tool.completed`（tool_name="rerank"）
   * 推出（见 `run/processState.retrievalFromRerankTool`）；历史记录不持久化该字段，
   * 读回为 null，此时界面按来源自身字段判定（见 `SourceFooter`）。
   */
  retrieval: RetrievalMeta | null;
  answerBasis: AnswerBasis | null;
  policyVersion: string | null;
  done: boolean;
  /** 用户「停止生成」后保留的半截回答。 */
  stopped: boolean;
  error: string | null;
  /** 任务 id（刷新后据此恢复状态，PRD 14.3）。 */
  runId: string | null;
  runStatus: RunStatus | null;
  /** `completed_with_gaps` 的缺口说明（缺少依据 + 对结论的影响）。 */
  gaps: string[];
  process: ProcessState;
  approvals: ApprovalRequest[];
  inputs: InputRequest[];
  /** 事件协议错误（未知版本 / 负载校验失败）：必须显示，不静默。 */
  protocolError: string | null;
  /** 本轮附带的图片引用处理结果（后端如实上报；界面据 status/reason 渲染）。 */
  imageRefs: RunImageRef[];
  /** 本轮随提问附带的图片张数（后端漏返回结果时也不静默）。 */
  attachedImages: number;
}

interface ChatProps {
  /** 当前会话 id：变化时加载该会话的历史消息。 */
  conversationId: string | null;
  /** 当前会话的已持久化消息（切换会话时由父级传入）。 */
  initialMessages: ConversationMessage[];
  /** 当前工作区 id：所有 run 端点都要它（跨区 404，PRD 13.2）。 */
  workspaceId: string;
  /**
   * 惰性确保会话存在并返回其 id。
   *
   * V1.2 的 `POST /runs` 必须带 `conversation_id`，所以会话要在创建 run **之前**
   * 存在（V1.1 是问答完成后再惰性建会话）。
   */
  ensureConversation: () => Promise<string | null>;
  /** 父级修订号：父级改写了 initialMessages 后自增，Chat 据此重新载入消息。 */
  revision?: number;
  /** 一轮问答完成（非错误）后回调，供父级刷新侧边栏与生成标题。 */
  onExchangeComplete: (
    userContent: string,
    assistantContent: string,
    sources: Source[] | null,
    answerBasis: AnswerBasis | null,
    policyVersion: string | null,
    reasoning: string | null,
  ) => void;
  /** 用户点铅笔「编辑重发」：父级把历史从该条起截断并返回新历史。 */
  onEditUserMessage?: (
    messageIndex: number,
    content: string,
  ) => Promise<ConversationMessage[] | null>;
  /** 问答结束（含错误/中断）后回调：父级刷新 /api/status。 */
  onChatComplete?: () => void;
  /** 用户 / Agent 头像 URL（设置页配置）。 */
  avatars?: { user: string | null; agent: string | null };
  /** 点击正文引用序号 / 来源脚注后回调：打开右侧来源详情面板。 */
  onCitationSelect?: (sources: Source[], index: number) => void;
  /** 会话当前权限模式（PRD 14.1，默认请求批准）。 */
  permissionMode?: PermissionMode;
  /** 会话当前思考深度（PRD 14.1，默认深度思考）。 */
  reasoningLevel?: ReasoningLevel;
  /** 权限模式变更（落到本会话后续请求）。 */
  onPermissionModeChange?: (mode: PermissionMode) => void;
  /** 思考深度变更。 */
  onReasoningLevelChange?: (level: ReasoningLevel) => void;
  /** 提示回调（父级轻提示），用于「还有活动任务」等提示。 */
  onNotify?: (type: "success" | "error", message: string) => void;
  /**
   * 本会话「承接自哪一次 run」（PRD 5.4 末句）。
   *
   * 会话被移动到别的工作区后，原区的执行已停止、历史只读；带上它让新执行记录来源，
   * 并在过程区域展示承接关系（后端接受 `origin_run_id` 但**不回显**）。
   */
  originRunId?: string | null;
}

const THINKING_TEXT = "思考中…";
const INPUT_MAX_HEIGHT = 120;

const SUGGESTIONS = ["知识库里有什么内容？", "帮我总结一下导入的文档", "Everything RAG 是做什么的？"];

function UserAvatar({ src }: { src: string | null }) {
  if (src !== null) return <img src={src} alt="" className="avatar-img" />;
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M12 12a5 5 0 1 0 0-10 5 5 0 0 0 0 10Zm0 2c-4.7 0-8 3-8 6 0 1 .8 2 2 2h12c1.2 0 2-1 2-2 0-3-3.3-6-8-6Z" />
    </svg>
  );
}

function AssistantAvatar({ src }: { src: string | null }) {
  if (src !== null) return <img src={src} alt="" className="avatar-img" />;
  return <BrandMark />;
}

/** 运行中的消息骨架（含任务字段的默认值）。 */
function emptyAssistant(id: number): ChatMessage {
  return {
    id,
    role: "assistant",
    content: "",
    reasoning: "",
    sources: null,
    // 筛选状态只由本轮事件推出（历史记录读回为 null，按来源字段自行判定）。
    retrieval: null,
    answerBasis: null,
    policyVersion: null,
    done: false,
    stopped: false,
    error: null,
    runId: null,
    runStatus: null,
    gaps: [],
    process: emptyProcessState(),
    approvals: [],
    inputs: [],
    protocolError: null,
    imageRefs: [],
    attachedImages: 0,
  };
}

function toChatMessages(messages: ConversationMessage[]): ChatMessage[] {
  return messages.map((message, index) => ({
    id: index + 1,
    role: message.role,
    content: message.content,
    reasoning: message.reasoning ?? "",
    sources: message.sources ?? null,
    // 筛选状态不随会话持久化：历史记录按来源字段自行判定（见 SourceFooter）。
    retrieval: null,
    answerBasis: message.answer_basis ?? null,
    policyVersion: message.policy_version ?? null,
    done: true,
    stopped: false,
    error: null,
    runId: null,
    runStatus: null,
    gaps: [],
    process: emptyProcessState(),
    approvals: [],
    inputs: [],
    protocolError: null,
    imageRefs: [],
    attachedImages: 0,
  }));
}

const BASIS_LABELS: Record<AnswerBasis, string> = {
  knowledge: "基于本地知识",
  web: "包含联网信息",
  general: "一般回答",
  product: "产品说明",
  catalog: "知识概览",
  memory: "基于工作区记忆",
  history: "基于历史会话",
  mixed: "多来源综合",
  insufficient: "现有内容不足",
};

/** 问答界面：空态建议 → 消息流（含过程区域与卡片）→ 底部输入条。 */
export default function Chat({
  conversationId,
  initialMessages,
  workspaceId,
  ensureConversation,
  revision = 0,
  onExchangeComplete,
  onEditUserMessage,
  onChatComplete,
  avatars,
  onCitationSelect,
  permissionMode = "ask",
  reasoningLevel = "high",
  onPermissionModeChange,
  onReasoningLevelChange,
  onNotify,
  originRunId = null,
}: ChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [cardBusy, setCardBusy] = useState(false);
  const [cardError, setCardError] = useState<string | null>(null);
  const [images, setImages] = useState<{ src: string; name: string }[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [hoveredCitation, setHoveredCitation] = useState<{ source: Source; rect: DOMRect } | null>(
    null,
  );
  const nextIdRef = useRef(1);
  const chatRootRef = useRef<HTMLDivElement>(null);
  const isAtBottomRef = useRef(true);

  /** 进行中的 run（订阅、答案归约、过程归约与去重游标）。 */
  const runRef = useRef<{
    assistantId: number;
    runId: string;
    subscription: RunSubscription | null;
    answer: AnswerState;
    process: ProcessState;
    stopped: boolean;
    protocolError: string | null;
  } | null>(null);
  const exchangeRef = useRef<{
    user: string;
    assistant: string;
    sources: Source[] | null;
    answerBasis: AnswerBasis | null;
    policyVersion: string | null;
    error: boolean;
  } | null>(null);

  const syncInputHeight = useCallback((): void => {
    const el = inputRef.current;
    if (el === null) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, INPUT_MAX_HEIGHT)}px`;
    el.style.overflowY = el.scrollHeight > INPUT_MAX_HEIGHT ? "auto" : "hidden";
  }, []);

  // 切换会话：加载历史消息、重置 id、回到底部跟随。
  // 进行中的任务不重置 —— 惰性建会话会在发送过程中改变 conversationId。
  useEffect(() => {
    if (runRef.current !== null) return;
    setMessages(toChatMessages(initialMessages));
    nextIdRef.current = initialMessages.length + 1;
    isAtBottomRef.current = true;
    // eslint-disable-next-line react-hooks/exhaustive-deps -- 仅在切换会话时加载
  }, [conversationId]);

  useEffect(() => {
    if (revision === 0 || runRef.current !== null) return;
    setMessages(toChatMessages(initialMessages));
    nextIdRef.current = initialMessages.length + 1;
    // eslint-disable-next-line react-hooks/exhaustive-deps -- 仅在父级修订时重载
  }, [revision]);

  const handleEditRequest = (messageId: number): void => {
    if (isSending) return;
    const index = messages.findIndex((message) => message.id === messageId);
    if (index < 0) return;
    const target = messages[index];
    if (target.role !== "user") return;
    const content = target.content;
    setMessages((prev) => prev.slice(0, index));
    setInput(content);
    requestAnimationFrame(() => {
      syncInputHeight();
      inputRef.current?.focus();
    });
    void onEditUserMessage?.(index, content);
  };

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

  useEffect(() => {
    if (!isAtBottomRef.current) return;
    const scroller = chatRootRef.current?.parentElement;
    if (scroller) scroller.scrollTop = scroller.scrollHeight;
  }, [messages]);

  // 卸载时只结束订阅（不取消 run）：断线不等于取消（PRD 12.4）。
  useEffect(() => {
    return () => {
      runRef.current?.subscription?.close();
      runRef.current = null;
    };
  }, []);

  const updateMessage = (id: number, update: (m: ChatMessage) => ChatMessage): void => {
    setMessages((prev) => prev.map((m) => (m.id === id ? update(m) : m)));
  };

  const handleCitationHover = useCallback((source: Source | null, rect: DOMRect | null): void => {
    setHoveredCitation(source !== null && rect !== null ? { source, rect } : null);
  }, []);

  const handleAddImages = useCallback(
    (files: FileList | null): void => {
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
    },
    [images.length],
  );

  // 事件回调里读取最新消息列表（避免闭包拿到过期状态）。
  const messagesRef = useRef<ChatMessage[]>([]);
  messagesRef.current = messages;

  /**
   * 把 run 快照（子任务 + 待处理卡片 + 状态）合入消息。
   *
   * 这里**不**走 `applyRunEvent`：快照不是事件，硬塞一个事件会污染
   * `process.lastSeq` 游标，导致后续真实事件（seq 更小）被当成重放全部丢弃。
   * 快照只合并任务列表本身，并按 task_id 保留事件流已积累的来源与摘要。
   */
  const applyRunDetail = useCallback((assistantId: number, detail: RunDetail): void => {
    const current = runRef.current;
    if (current !== null && current.assistantId === assistantId) {
      const tasks = [...current.process.tasks];
      for (const task of detail.tasks) {
        const index = tasks.findIndex((item) => item.taskId === task.task_id);
        const existing = index < 0 ? null : tasks[index];
        const merged = {
          taskId: task.task_id,
          question: task.question || (existing?.question ?? ""),
          agentKind: task.agent_kind,
          status: task.status,
          dependsOn: task.depends_on ?? [],
          // 父子的权威来源是 run 详情（`plan.updated` 事件不带该字段）。
          parentTaskId: task.parent_task_id ?? existing?.parentTaskId ?? null,
          childTaskIds: existing?.childTaskIds ?? [],
          sources: existing?.sources ?? [],
          resultSummary: existing?.resultSummary ?? "",
        };
        if (index < 0) tasks.push(merged);
        else tasks[index] = merged;
      }
      // 依据父子关系补全 childTaskIds（过程区域展示「父任务的整体完成情况」）。
      const parents = new Map(tasks.map((task) => [task.taskId, task]));
      for (const task of tasks) {
        const parent = task.parentTaskId === null ? null : parents.get(task.parentTaskId);
        if (parent === undefined || parent === null) continue;
        if (!parent.childTaskIds.includes(task.taskId)) {
          parent.childTaskIds = [...parent.childTaskIds, task.taskId];
        }
      }
      current.process = { ...current.process, tasks };
    }
    updateMessage(assistantId, (m) => ({
      ...m,
      runId: detail.run_id,
      runStatus: detail.status,
      process: current !== null && current.assistantId === assistantId ? current.process : m.process,
      approvals: detail.approvals ?? [],
      inputs: detail.inputs ?? [],
      // 图片引用的逐条结果（后端如实上报）；界面据 status/reason 渲染，
      // 不把 cached/unavailable 说成已生效。
      imageRefs: detail.image_refs ?? [],
    }));
  }, []);

  /**
   * 单个事件 → 过程区域 + 答案通道 + 卡片。
   *
   * 卡片列表一律用**函数式更新**合并（按 id 去重）：事件可能在同一次渲染前连续到达，
   * 从 `messages` 快照读旧值会把先到的卡片覆盖掉（实测会让批准卡片凭空消失）。
   */
  const handleRunEvent = useCallback(
    (assistantId: number, event: RunEventTyped): void => {
      const current = runRef.current;
      if (current === null || current.assistantId !== assistantId) return;

      current.process = applyRunEvent(current.process, event);
      current.answer = applyAnswerEvent(current.answer, event);

      const answer = current.answer;
      const sources = answer.citations.length > 0 ? citationsToSources(answer.citations) : null;
      const text = answerText(answer);

      const patch: Partial<ChatMessage> = {
        process: current.process,
        content: text,
        sources: sources ?? (answer.answerId === null ? undefined : []),
        answerBasis: answer.basis,
      };

      if (event.type === "run.status") patch.runStatus = event.payload.status;
      if (event.type === "tool.completed") {
        // 相关性筛选（PRD §8.2）：run 架构没有 `meta.retrieval`，筛选状态由
        // `tool.completed`（tool_name="rerank"）的真实入选/候选计数推出；
        // 非 rerank 工具返回 null，不改状态。
        const retrieval = retrievalFromRerankTool(event.payload);
        if (retrieval !== null) patch.retrieval = retrieval;
      }
      if (event.type === "answer.meta") {
        // 筛选档案 id 只随引用到达（事件负载没有该字段，events.py 只读），而
        // `answer.meta` 晚于 rerank 进度事件 ⇒ 在这里补全；缺记录保持 null，不编造。
        const profileId = profileIdFromCitations(event.payload.citations ?? []);
        if (profileId !== null) {
          updateMessage(assistantId, (m) => ({
            ...m,
            ...patch,
            retrieval: m.retrieval === null ? null : { ...m.retrieval, profile_id: profileId },
          }));
          return;
        }
      }
      if (event.type === "run.completed") {
        patch.runStatus = event.payload.status;
        patch.gaps = event.payload.gaps ?? [];
      }
      if (event.type === "run.cancelled") {
        patch.runStatus = "cancelled";
        patch.stopped = true;
      }
      if (event.type === "run.failed") {
        patch.runStatus = "failed";
        patch.error = `${event.payload.error.message}${
          event.payload.error.required_action === ""
            ? ""
            : `（${event.payload.error.required_action}）`
        }`;
      }

      if (event.type === "approval.requested") {
        const approval = event.payload.approval;
        updateMessage(assistantId, (m) => ({
          ...m,
          ...patch,
          runStatus: "waiting_approval",
          approvals: [
            ...m.approvals.filter((item) => item.approval_id !== approval.approval_id),
            approval,
          ],
        }));
        return;
      }
      if (event.type === "approval.resolved") {
        const resolvedId = event.payload.approval_id;
        updateMessage(assistantId, (m) => ({
          ...m,
          ...patch,
          approvals: m.approvals.filter((item) => item.approval_id !== resolvedId),
        }));
        return;
      }
      if (event.type === "input.requested") {
        const request = event.payload.request;
        updateMessage(assistantId, (m) => ({
          ...m,
          ...patch,
          runStatus: "waiting_input",
          inputs: [
            ...m.inputs.filter((item) => item.input_request_id !== request.input_request_id),
            request,
          ],
        }));
        return;
      }
      if (event.type === "input.resolved") {
        const resolvedId = event.payload.input_request_id;
        updateMessage(assistantId, (m) => ({
          ...m,
          ...patch,
          inputs: m.inputs.filter((item) => item.input_request_id !== resolvedId),
        }));
        return;
      }

      updateMessage(assistantId, (m) => ({ ...m, ...patch }));
    },
    [],
  );

  const finishRun = useCallback(
    (assistantId: number): void => {
      const current = runRef.current;
      if (current === null || current.assistantId !== assistantId) return;
      const answer = current.answer;
      const stopped = current.stopped;
      const protocolError = current.protocolError;
      const sources = answer.citations.length > 0 ? citationsToSources(answer.citations) : null;

      updateMessage(assistantId, (m) => ({
        ...m,
        done: true,
        stopped: stopped || m.stopped,
        sources: sources ?? m.sources,
        content: answerText(answer) || m.content,
        answerBasis: answer.basis ?? m.answerBasis,
      }));

      const exchange = exchangeRef.current;
      runRef.current = null;
      setIsSending(false);
      setStopping(false);
      setCardBusy(false);

      // 用户主动停止的半截回答不入历史；协议错误与运行失败也不当作成功轮次。
      if (exchange !== null && !stopped && !exchange.error && protocolError === null) {
        onExchangeComplete(
          exchange.user,
          answerText(answer) || exchange.assistant,
          sources,
          answer.basis,
          exchange.policyVersion,
          null,
        );
      }
      onChatComplete?.();
    },
    [onChatComplete, onExchangeComplete],
  );

  const handleSend = async (preset?: string): Promise<void> => {
    const message = (preset ?? input).trim();
    if (message === "" || isSending) return;
    if (runRef.current !== null) {
      onNotify?.("error", "当前会话还有未完成的任务，请先处理当前任务或停止后继续");
      return;
    }

    const userId = nextIdRef.current++;
    const assistantId = nextIdRef.current++;
    const userMessage: ChatMessage = {
      id: userId,
      role: "user",
      content: message,
      reasoning: "",
      sources: null,
      retrieval: null,
      answerBasis: null,
      policyVersion: null,
      done: true,
      stopped: false,
      error: null,
      runId: null,
      runStatus: null,
      gaps: [],
      process: emptyProcessState(),
      approvals: [],
      inputs: [],
      protocolError: null,
      imageRefs: [],
      attachedImages: 0,
    };
    isAtBottomRef.current = true;
    setMessages((prev) => [...prev, userMessage, emptyAssistant(assistantId)]);
    setInput("");
    setIsSending(true);
    setCardError(null);
    exchangeRef.current = {
      user: message,
      assistant: "",
      sources: null,
      answerBasis: null,
      policyVersion: null,
      error: false,
    };
    runRef.current = {
      assistantId,
      runId: "",
      subscription: null,
      answer: emptyAnswerState(),
      process: emptyProcessState(),
      stopped: false,
      protocolError: null,
    };

    // 快照本轮贴图（发送后清空输入，但请求体仍带本轮的图）：后端把每条引用解析并
    // 落本区图片库，逐条结果由 RunDetail.image_refs 如实回报。
    const sentImages = images.slice(0, MAX_RUN_IMAGES).map((img) => img.src);
    const attachedImages = sentImages.length;
    setImages([]);

    // 1) 会话必须先存在（POST /runs 需要 conversation_id）。
    let targetConversation: string | null = null;
    try {
      targetConversation = await ensureConversation();
    } catch (error) {
      failSend(assistantId, describeRunError(error));
      return;
    }
    if (targetConversation === null) {
      failSend(assistantId, "无法创建会话，请稍后重试");
      return;
    }

    // 2) 创建并启动 run（202 Accepted）；图片引用与来源 run 随请求一起提交。
    let detail: RunDetail;
    try {
      detail = await createRun(workspaceId, {
        conversationId: targetConversation,
        message,
        clientRequestId: nextClientRequestId(),
        permissionMode,
        reasoningLevel,
        images: sentImages,
        // PRD 5.4 末句：会话移动后在目标区继续提问，新执行记录承接来源。
        originRunId,
      });
    } catch (error) {
      failSend(assistantId, describeRunError(error));
      return;
    }

    const current = runRef.current;
    if (current === null || current.assistantId !== assistantId) return;
    current.runId = detail.run_id;
    // 记下后端交给我们的会话 context_epoch：移动时作为乐观并发预期（后端不回读该字段）。
    rememberConversationEpoch(targetConversation, detail.context_epoch);
    // 记录本轮实际附带的图片张数：即使后端漏返回 image_refs，界面也不静默。
    if (attachedImages > 0) {
      updateMessage(assistantId, (m) => ({ ...m, attachedImages }));
    }
    // 承接关系（PRD 5.4）：后端不回显，前端自己展示，避免用户以为历史消失。
    if (originRunId !== null && originRunId !== "") {
      current.process = withOriginRun(current.process, originRunId);
    }
    // 过程区域记录**实际发给本轮的**设置（后端详情回读，而非菜单草稿）。
    current.process = withRunSettings(
      current.process,
      `本轮设置：${PERMISSION_LABELS[detail.permission_mode]} · ${REASONING_LABELS[detail.reasoning_level]}`,
    );
    applyRunDetail(assistantId, detail);

    // 3) 订阅事件流：从 0 开始（后端先补读历史再实时推送），保证不漏事件。
    current.subscription = subscribeRunEvents(
      workspaceId,
      detail.run_id,
      {
        onEvent: (event) => handleRunEvent(assistantId, event),
        onProtocolError: (protocolMessage) => {
          const active = runRef.current;
          if (active === null || active.assistantId !== assistantId) return;
          active.protocolError = protocolMessage;
          updateMessage(assistantId, (m) => ({
            ...m,
            protocolError: protocolMessage,
            done: true,
          }));
        },
        onClose: () => finishRun(assistantId),
      },
      { after: 0 },
    );
  };

  /** 发送链路的失败收尾：恢复 UI 并把错误显示在该条消息上。 */
  const failSend = (assistantId: number, message: string): void => {
    const exchange = exchangeRef.current;
    if (exchange !== null) exchange.error = true;
    runRef.current = null;
    exchangeRef.current = null;
    updateMessage(assistantId, (m) => ({ ...m, done: true, error: message }));
    setIsSending(false);
    setStopping(false);
    onChatComplete?.();
  };

  /** 停止：调用取消 API，显示「正在停止」→「已停止」。 */
  const handleStop = async (): Promise<void> => {
    const current = runRef.current;
    if (!isSending || current === null) return;
    current.stopped = true;
    setStopping(true);
    updateMessage(current.assistantId, (m) => ({ ...m, runStatus: "cancelling" }));
    if (current.runId === "") {
      // run 还没建起来：只结束本地订阅。
      current.subscription?.close();
      finishRun(current.assistantId);
      return;
    }
    try {
      const detail = await cancelRun(workspaceId, current.runId, "user_stopped");
      updateMessage(current.assistantId, (m) => ({ ...m, runStatus: detail.status, stopped: true }));
    } catch (error) {
      updateMessage(current.assistantId, (m) => ({ ...m, stopped: true }));
      onNotify?.("error", describeRunError(error));
    }
    finishRun(current.assistantId);
  };

  const handleApprove = async (
    approvalId: string,
    decision: ApprovalDecisionKind,
    editedParameters?: Record<string, unknown> | null,
  ): Promise<void> => {
    const current = runRef.current;
    const runId = current?.runId ?? findRunIdByApproval(approvalId);
    if (runId === null || runId === "") return;
    setCardBusy(true);
    setCardError(null);
    try {
      await submitApproval(workspaceId, runId, approvalId, {
        decision,
        editedParameters: editedParameters ?? null,
      });
      setMessages((prev) =>
        prev.map((m) => ({
          ...m,
          approvals: m.approvals.filter((item) => item.approval_id !== approvalId),
        })),
      );
    } catch (error) {
      setCardError(describeRunError(error));
    } finally {
      setCardBusy(false);
    }
  };

  const handleClarify = async (inputRequestId: string, answer: string): Promise<void> => {
    const current = runRef.current;
    const runId = current?.runId ?? findRunIdByInput(inputRequestId);
    if (runId === null || runId === "") return;
    setCardBusy(true);
    setCardError(null);
    try {
      await submitInput(workspaceId, runId, inputRequestId, answer);
      setMessages((prev) =>
        prev.map((m) => ({
          ...m,
          inputs: m.inputs.filter((item) => item.input_request_id !== inputRequestId),
        })),
      );
    } catch (error) {
      setCardError(describeRunError(error));
    } finally {
      setCardBusy(false);
    }
  };

  const findRunIdByApproval = (approvalId: string): string | null => {
    for (const message of messagesRef.current) {
      if (message.approvals.some((item) => item.approval_id === approvalId)) return message.runId;
    }
    return null;
  };
  const findRunIdByInput = (inputRequestId: string): string | null => {
    for (const message of messagesRef.current) {
      if (message.inputs.some((item) => item.input_request_id === inputRequestId)) return message.runId;
    }
    return null;
  };

  const handleInputChange = (event: ChangeEvent<HTMLTextAreaElement>): void => {
    setInput(event.target.value);
    syncInputHeight();
  };

  const handleInputKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>): void => {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) return;
    event.preventDefault();
    void handleSend();
  };

  /**
   * 进行中步骤的进度提示（PRD §8.2 / §14.2）。
   *
   * 位置与语义沿用主线 `/api/chat` 时代的 `.chat-searching`（相关性筛选评分期间不产出
   * 任何正文片段，必须有独立进度提示），但文案**只来自 run 事件归约**：`tool.started`
   * 的 rerank 帧给出「正在筛选资料」，不再有旧 `onTool` 回调。
   * 等待批准 / 澄清 / 停止时不显示，避免把「等你回复」说成「正在检索」。
   */
  const activeStepText = ((): string | null => {
    if (messages.length === 0) return null;
    const active = messages[messages.length - 1];
    if (active.role !== "assistant" || active.done) return null;
    const status = active.runStatus;
    if (status !== null && status !== "running" && status !== "queued") return null;
    return runningEntryText(active.process);
  })();

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
        title="贴图：图片会随提问保存到本工作区；配置了视觉模型时本轮即可参与核对，否则需配置后补做"
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
      <PermissionMenu
        permissionMode={permissionMode}
        reasoningLevel={reasoningLevel}
        disabled={isSending}
        onPermissionChange={(mode) => onPermissionModeChange?.(mode)}
        onReasoningChange={(level) => onReasoningLevelChange?.(level)}
      />
      {isSending ? (
        <button
          type="button"
          className="chat-stop"
          onClick={() => void handleStop()}
          aria-label="停止生成"
          title="停止生成"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
            <rect x="6" y="6" width="12" height="12" rx="2" />
          </svg>
        </button>
      ) : (
        <button className="chat-send" type="submit" disabled={input.trim() === ""} aria-label="发送">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path
              d="M12 19V5m-6 6 6-6 6 6"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      )}
    </form>
  );

  return (
    <div className="chat" ref={chatRootRef}>
      {messages.length === 0 ? (
        <div className="chat-empty">
          <div className="chat-empty-center">
            <div className="hero-emblem">
              <BrandMark />
              <span className="emblem-node" />
            </div>
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
                  <span className="suggestion-number" aria-hidden="true">
                    0{index + 1}
                  </span>
                  <span className="suggestion-title">
                    {["探索知识库", "提炼文档重点", "认识 Everything RAG"][index]}
                  </span>
                  <span className="suggestion-description">{suggestion}</span>
                  <span className="suggestion-arrow" aria-hidden="true">
                    ↗
                  </span>
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
                ) : msg.content === "" ? (
                  msg.stopped ? (
                    <div className="message-content stopped">已停止生成</div>
                  ) : (
                    <div className="message-content thinking">
                      <span className="thinking-spinner" aria-hidden="true" />
                      <span>{THINKING_TEXT}</span>
                    </div>
                  )
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

                {msg.role === "assistant" && msg.protocolError !== null && (
                  <div className="message-error" role="alert" data-testid="protocol-error">
                    事件协议错误：{msg.protocolError}
                  </div>
                )}

                {msg.role === "assistant" && msg.stopped && (
                  <p className="message-stopped-note" data-testid="stopped-note">
                    {stopping && msg.runStatus === "cancelling" ? "正在停止…" : "已停止生成"}
                  </p>
                )}

                {msg.role === "assistant" && msg.answerBasis === "insufficient" && (
                  <p className="message-gaps-note" data-testid="insufficient-note">
                    现有资料不足以支持结论，已在正文标明缺少依据的部分。
                  </p>
                )}

                {/* 筛选跑了但一条都没入选（PRD §8.2 的 empty 态）：必须说出来，
                    否则用户只看到「没有来源」而不知道这是相关性筛选的结果。 */}
                {msg.role === "assistant" && msg.retrieval?.status === "empty" && (
                  <p className="message-relevance-note" data-testid="relevance-empty-note">
                    {relevanceEmptyNote(msg.retrieval)}
                  </p>
                )}

                {msg.role === "assistant" && msg.runStatus === "completed_with_gaps" && (
                  <div className="message-gaps" data-testid="gaps-note">
                    <p className="message-gaps-title">已完成，但部分问题资料不足</p>
                    <p className="message-gaps-line">
                      缺少依据：{msg.gaps.length > 0 ? msg.gaps.join("；") : "未标注具体缺口"}
                    </p>
                    <p className="message-gaps-line">
                      对结论的影响：正文中相关结论已标注为资料不足，未做无依据的综合。
                    </p>
                  </div>
                )}

                {msg.role === "assistant" && msg.runStatus === "failed" && (
                  <div className="message-error" role="alert" data-testid="failed-note">
                    执行失败，已完成的研究记录仍可查看，可从此处继续。
                  </div>
                )}

                {msg.role === "assistant" && msg.imageRefs.length > 0 && (
                  <div
                    className={`message-images${summarizeRunImages(msg.imageRefs).allEffective ? " effective" : ""}`}
                    data-testid="image-refs"
                  >
                    <p className="message-images-title" data-testid="image-refs-summary">
                      {summarizeRunImages(msg.imageRefs).allEffective
                        ? "图片引用：已生效，本轮参与核对"
                        : `图片引用：${summarizeRunImages(msg.imageRefs).notice}`}
                    </p>
                    <ul className="message-images-list">
                      {msg.imageRefs.map((ref, index) => {
                        const outcome = describeRunImage(ref);
                        return (
                          <li
                            key={`${ref.content_hash || ref.src_ref}-${index}`}
                            className={`message-image-item${outcome.effective ? " effective" : ""}`}
                            data-testid={`image-ref-${index}`}
                            data-status={ref.status}
                            data-effective={outcome.effective ? "true" : "false"}
                          >
                            <span className="message-image-label" data-testid={`image-ref-label-${index}`}>
                              {outcome.label}
                            </span>
                            {outcome.notice !== null && (
                              <span
                                className="message-image-notice"
                                data-testid={`image-ref-notice-${index}`}
                              >
                                {outcome.notice}
                              </span>
                            )}
                          </li>
                        );
                      })}
                    </ul>
                  </div>
                )}

                {/* 贴了图但后端没回结果：同样不静默，按「未生效」说明。 */}
                {msg.role === "assistant" &&
                  msg.attachedImages > 0 &&
                  msg.imageRefs.length === 0 && (
                    <div className="message-images" data-testid="image-refs-missing">
                      <p className="message-images-title">
                        图片引用：后端未返回处理结果，本轮不视为已生效
                      </p>
                      <p className="message-image-notice">
                        已附带 {msg.attachedImages} 张图片，但没有拿到逐条处理结果，本轮不参与核对。
                        请刷新页面查看，或稍后重试。
                      </p>
                    </div>
                  )}

                {msg.role === "assistant" && <ProcessArea process={msg.process} />}

                {msg.role === "assistant" &&
                  msg.approvals.map((approval) => (
                    <ApprovalCard
                      key={approval.approval_id}
                      approval={approval}
                      busy={cardBusy}
                      error={cardError}
                      onDecide={(approvalId, decision, edited) =>
                        void handleApprove(approvalId, decision, edited)
                      }
                    />
                  ))}

                {msg.role === "assistant" &&
                  msg.inputs.map((request) => (
                    <ClarificationCard
                      key={request.input_request_id}
                      request={request}
                      busy={cardBusy}
                      error={cardError}
                      onSubmit={(inputRequestId, answer) =>
                        void handleClarify(inputRequestId, answer)
                      }
                    />
                  ))}

                {msg.role === "assistant" && (msg.sources?.length ?? 0) > 0 && (
                  <SourceFooter
                    sources={msg.sources ?? []}
                    retrieval={msg.retrieval}
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
          {activeStepText !== null && (
            <div className="chat-searching" role="status" aria-live="polite">
              <span className="chat-searching-dot" aria-hidden="true" />
              {activeStepText}…
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
