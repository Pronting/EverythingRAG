import { useCallback, useEffect, useRef, useState } from "react";
import type { ChangeEvent, KeyboardEvent } from "react";
import {
  MAX_RUN_IMAGES,
  cancelRun,
  createRun,
  fetchEventLog,
  getRun,
  listConversationRuns,
  describeRunError,
  describeRunImage,
  nextClientRequestId,
  resumeRun,
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
  ModelSelectionView,
  ProviderView,
} from "../types";
import { applyAnswerEvent, answerText, citationsToSources, emptyAnswerState, reduceAnswerEvents } from "../run/answerState";
import type { AnswerState } from "../run/answerState";
import {
  applyRunEvent,
  emptyProcessState,
  profileIdFromCitations,
  reduceRunEvents,
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
import ModelPicker from "./ModelPicker";
import BrandMark from "./BrandMark";
import { requiredActionText } from "../util/runError";

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
  /**
   * 失败是否可从记录位置继续（后端 `run.failed.error.resumable` 的如实投影）。
   *
   * 用户实测缺陷 F07/F19：失败卡片此前只显示一段带占位符的 `POST /api/...` 文本
   * 和「可从此处继续」，界面上**没有任何可点击的入口**；T18 已完成五项研究、仅在
   * 生成阶段超时，用户也无法继续。现在按该字段渲染真实的「继续执行」按钮。
   */
  resumable: boolean;
}

/**
 * 进行中的 run（订阅、答案归约、过程归约与去重游标）。
 *
 * 提取成模块级类型：继续执行（`handleResume`）与发送（`handleSend`）要构造同一种对象。
 */
interface ActiveRun {
  assistantId: number;
  conversationId: string | null;
  runId: string;
  subscription: RunSubscription | null;
  answer: AnswerState;
  process: ProcessState;
  stopped: boolean;
  protocolError: string | null;
  replayMessage: ChatMessage | null;
  renderTimer: number | null;
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
  /** 后端接受编辑后更新父级的历史消息。 */
  onEditCommitted?: (messageIndex: number) => void;
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
  /** 模型选择器数据：供应商列表（按供应商分组）与当前选择。 */
  modelProviders?: ProviderView[];
  modelSelection?: ModelSelectionView | null;
  /** 当前生效的对话模型名（无选择时的兜底展示）。 */
  currentModel?: string | null;
  /** 选择模型：父级写设置并对之后的提问生效。 */
  onModelSelect?: (providerId: string, modelId: string) => void;
  /** 打开设置页（模型选择器空态时的入口）。 */
  onOpenSettings?: () => void;
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
    resumable: true,
  };
}

function toChatMessages(messages: ConversationMessage[]): ChatMessage[] {
  const rows: ChatMessage[] = messages.map((message, index) => ({
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
    // 保留 run_id：历史消息的**过程区域**靠它回读 `event-log` 还原（见 restoreHistoricalProcesses）。
    runId: message.run_id ?? null,
    runStatus: null,
    gaps: [],
    process: emptyProcessState(),
    approvals: [],
    inputs: [],
    protocolError: null,
    imageRefs: [],
    attachedImages: 0,
    resumable: true,
  }));
  const answered = new Set(rows.filter((row) => row.role === "assistant").map((row) => row.runId));
  const restored: ChatMessage[] = [];
  for (const row of rows) {
    restored.push({ ...row, id: restored.length + 1 });
    if (row.role === "user" && row.runId !== null && !answered.has(row.runId)) {
      restored.push({ ...emptyAssistant(restored.length + 1), runId: row.runId, done: true });
    }
  }
  return restored;
}

/** 打开会话时最多回读多少轮历史过程（避免一次拉太长会话、打满事件请求）。 */
const MAX_RESTORED_PROCESSES = 20;

/** 历史过程区域回读失败的提示（只在完全没恢复出来时提醒一次，避免噪音）。 */
const PROCESS_RESTORE_FAILED_TEXT = "历史过程区域加载失败，可刷新重试";

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
  onEditCommitted,
  onChatComplete,
  avatars,
  onCitationSelect,
  permissionMode = "ask",
  reasoningLevel = "off",
  onPermissionModeChange,
  onReasoningLevelChange,
  onNotify,
  originRunId = null,
  modelProviders = [],
  modelSelection = null,
  currentModel = null,
  onModelSelect,
  onOpenSettings,
}: ChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [processLoadState, setProcessLoadState] = useState<Record<string, "loading" | "loaded" | "failed">>({});
  const processLoadGenerationRef = useRef(0);
  const processLoadPendingRef = useRef(new Map<string, Promise<void>>());
  const processLoadedRef = useRef(new Set<string>());
  const [input, setInput] = useState("");
  const [editing, setEditing] = useState<{
    messageId: string;
    storedIndex: number;
    visibleIndex: number;
    previousInput: string;
  } | null>(null);
  const [editSubmitting, setEditSubmitting] = useState(false);
  const editRequestRef = useRef<{ key: string; id: string } | null>(null);
  const [isSending, setIsSending] = useState(false);
  /** 「正在停止 → 已停止」的两段式提示；run 取消完成后复位。 */
  const [stopping, setStopping] = useState(false);
  const completionRef = useRef({ onExchangeComplete, onChatComplete });
  completionRef.current = { onExchangeComplete, onChatComplete };
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

  /**
   * 进行中的 run（订阅、答案归约、过程归约与去重游标）。
   *
   * 提取成模块级类型：继续执行（`handleResume`）与发送（`handleSend`）要构造同一种对象。
   */
  const runRef = useRef<ActiveRun | null>(null);
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

  useEffect(() => {
    if (revision === 0 || runRef.current !== null) return;
    const generation = ++processLoadGenerationRef.current;
    processLoadPendingRef.current.clear();
    processLoadedRef.current.clear();
    setProcessLoadState({});
    setMessages(toChatMessages(initialMessages));
    nextIdRef.current = toChatMessages(initialMessages).length + 1;
    // 父级重载消息（编辑重发 / 切换分支）同样要还原历史过程区域。
    void restoreHistoricalProcesses(() => generation === processLoadGenerationRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- 仅在父级修订时重载
  }, [revision]);

  const handleEditRequest = async (messageId: number): Promise<void> => {
    if (isSending || runRef.current !== null) return;
    const index = messages.findIndex((message) => message.id === messageId);
    if (index < 0) return;
    const target = messages[index];
    if (target.role !== "user") return;
    const content = target.content;
    const storedIndex = target.runId === null ? -1 : initialMessages.findIndex((message) =>
      message.role === "user" && message.run_id === target.runId);
    const resolvedIndex = storedIndex >= 0 ? storedIndex : index;
    const saved = initialMessages[resolvedIndex];
    if (!saved?.message_id || saved.role !== "user" || saved.content !== content) {
      onNotify?.("error", "这条提问的记录尚未加载，请稍后重试");
      return;
    }
    setEditing({
      messageId: saved.message_id, storedIndex: resolvedIndex, visibleIndex: index,
      previousInput: editing?.previousInput ?? input,
    });
    editRequestRef.current = null;
    setInput(content);
    requestAnimationFrame(() => {
      syncInputHeight();
      inputRef.current?.focus();
    });
  };

  const handleCancelEdit = (): void => {
    if (editSubmitting || editing === null) return;
    setInput(editing.previousInput);
    setEditing(null);
    editRequestRef.current = null;
    requestAnimationFrame(syncInputHeight);
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
      if (runRef.current?.renderTimer !== null && runRef.current?.renderTimer !== undefined) {
        window.clearTimeout(runRef.current.renderTimer);
      }
      const subscription = runRef.current?.subscription;
      runRef.current = null;
      subscription?.close();
    };
  }, []);

  const updateMessage = (id: number, update: (m: ChatMessage) => ChatMessage): void => {
    setMessages((prev) => prev.map((m) => (m.id === id ? update(m) : m)));
  };

  /** 读取指定执行的持久事件；旧消息点击按钮时与近期自动恢复共用。 */
  const loadHistoricalProcess = useCallback(
    async (target: { id: number; runId: string }, alive: () => boolean): Promise<void> => {
      if (!alive() || processLoadedRef.current.has(target.runId)) return;
      const pending = processLoadPendingRef.current.get(target.runId);
      if (pending !== undefined) return pending;
      const loading = (async (): Promise<void> => {
        setProcessLoadState((state) => ({ ...state, [target.runId]: "loading" }));
        try {
          const { events, protocolError } = await fetchEventLog(workspaceId, target.runId, 0);
          if (protocolError !== null) throw new Error(protocolError);
          const detail = await getRun(workspaceId, target.runId);
          if (!alive()) return;
          const reduced = reduceRunEvents(events);
          const statuses = new Map(detail.tasks.map((task) => [task.task_id, task.status]));
          const process = { ...reduced, tasks: reduced.tasks.map((task) => ({
            ...task, status: statuses.get(task.taskId) ?? task.status,
          })) };
          const answer = reduceAnswerEvents(events);
          const failure = events.filter((event) => event.type === "run.failed").at(-1);
          updateMessage(target.id, (message) =>
            message.runId === target.runId && message.process.entries.length === 0
              && runRef.current?.runId !== target.runId
              ? {
                ...message, process, runStatus: detail.status,
                content: answerText(answer) || message.content,
                sources: answer.citations.length ? citationsToSources(answer.citations) : message.sources,
                answerBasis: answer.basis ?? message.answerBasis,
                gaps: answer.gaps,
                error: detail.status === "failed" && failure ? failure.payload.error.message : null,
                resumable: detail.resumable,
                approvals: detail.approvals ?? [],
                inputs: detail.inputs ?? [],
              }
              : message,
          );
          processLoadedRef.current.add(target.runId);
          setProcessLoadState((state) => ({ ...state, [target.runId]: "loaded" }));
        } catch (error) {
          if (alive()) setProcessLoadState((state) => ({ ...state, [target.runId]: "failed" }));
          throw error;
        }
      })();
      processLoadPendingRef.current.set(target.runId, loading);
      try {
        await loading;
      } finally {
        if (processLoadPendingRef.current.get(target.runId) === loading) {
          processLoadPendingRef.current.delete(target.runId);
        }
      }
    },
    [workspaceId],
  );

  const restoreHistoricalProcesses = useCallback(
    async (alive: () => boolean): Promise<void> => {
      const targets = toChatMessages(initialMessages)
        .filter((message) => message.role === "assistant" && message.runId !== null)
        .map((message) => ({ id: message.id, runId: message.runId as string }))
        .slice(-MAX_RESTORED_PROCESSES)
        .reverse();
      let failures = 0;
      for (const target of targets) {
        if (!alive()) return;
        try {
          await loadHistoricalProcess(target, alive);
        } catch {
          failures += 1;
        }
      }
      if (alive() && targets.length > 0 && failures === targets.length) {
        onNotify?.("error", PROCESS_RESTORE_FAILED_TEXT);
      }
    },
    [initialMessages, loadHistoricalProcess, onNotify],
  );

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
      error: ["queued", "running", "waiting_approval", "waiting_input", "waiting_children"].includes(detail.status) ? null : m.error,
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
    (assistantId: number, event: RunEventTyped, replay = false): void => {
      const current = runRef.current;
      if (current === null || current.assistantId !== assistantId) return;
      if (event.event_seq <= current.process.lastSeq) return;

      const commit = (update: (message: ChatMessage) => ChatMessage): void => {
        if (replay) {
          if (current.replayMessage === null) throw new Error("事件回放缺少消息初始状态");
          current.replayMessage = update(current.replayMessage);
          return;
        }
        if (current.renderTimer !== null) {
          window.clearTimeout(current.renderTimer);
          current.renderTimer = null;
        }
        updateMessage(assistantId, update);
      };

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

      if (event.type === "run.status") {
        patch.runStatus = event.payload.status;
        if (["queued", "running", "waiting_approval", "waiting_input", "waiting_children"].includes(event.payload.status)) {
          patch.error = null;
          patch.stopped = false;
          patch.done = false;
          if (exchangeRef.current !== null) exchangeRef.current.error = false;
        }
      }
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
          commit((m) => ({
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
        patch.error = null;
        patch.stopped = false;
      }
      if (event.type === "run.cancelled") {
        patch.resumable = event.payload.resumable;
        patch.runStatus = "cancelled";
        patch.stopped = true;
      }
      if (event.type === "run.failed") {
        if (exchangeRef.current !== null) exchangeRef.current.error = true;
        patch.runStatus = "failed";
        patch.resumable = event.payload.error.resumable !== false;
        const action = requiredActionText(event.payload.error);
        patch.error = `${event.payload.error.message}${action ? `（${action}）` : ""}`;
      }

      if (event.type === "approval.requested") {
        const approval = event.payload.approval;
        commit((m) => ({
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
        commit((m) => ({
          ...m,
          ...patch,
          approvals: m.approvals.filter((item) => item.approval_id !== resolvedId),
        }));
        return;
      }
      if (event.type === "input.requested") {
        const request = event.payload.request;
        commit((m) => ({
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
        commit((m) => ({
          ...m,
          ...patch,
          inputs: m.inputs.filter((item) => item.input_request_id !== resolvedId),
        }));
        return;
      }

      if (!replay && (event.type === "model.reasoning" || event.type === "answer.delta")) {
        if (current.renderTimer === null) {
          current.renderTimer = window.setTimeout(() => {
            current.renderTimer = null;
            if (runRef.current !== current) return;
            updateMessage(assistantId, (message) => ({
              ...message,
              process: current.process,
              content: answerText(current.answer),
            }));
          }, 80);
        }
        return;
      }
      commit((m) => ({ ...m, ...patch }));
      if (!replay && ["run.completed", "run.failed", "run.cancelled"].includes(event.type)) {
        // 结束事件不包含子任务状态，读取同一运行的权威记录更新进度。
        void getRun(workspaceId, event.run_id).then((detail) => {
          updateMessage(assistantId, (m) => {
            if (m.runId !== detail.run_id || m.runStatus !== detail.status
              || m.process.lastSeq > event.event_seq) return m;
            const statuses = new Map(detail.tasks.map((task) => [task.task_id, task.status]));
            return {
              ...m,
              process: {
                ...m.process,
                tasks: m.process.tasks.map((task) => ({
                  ...task,
                  status: statuses.get(task.taskId) ?? task.status,
                })),
              },
            };
          });
        }).catch((error) => onNotify?.("error", `任务状态更新失败：${describeRunError(error)}`));
      }
    },
    [workspaceId, onNotify],
  );

  const replayRunHistory = (assistantId: number, events: RunEventTyped[], seed: ChatMessage): void => {
    const current = runRef.current;
    if (current === null || current.assistantId !== assistantId) return;
    current.replayMessage = seed;
    for (const event of events) handleRunEvent(assistantId, event, true);
    const replayed = current.replayMessage;
    current.replayMessage = null;
    if (replayed !== null) updateMessage(assistantId, () => replayed);
  };

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
        process: current.process,
        done: true,
        stopped: stopped || m.stopped,
        sources: sources ?? m.sources,
        content: answerText(answer) || m.content,
        answerBasis: answer.basis ?? m.answerBasis,
      }));

      const exchange = exchangeRef.current;
      if (current.renderTimer !== null) window.clearTimeout(current.renderTimer);
      runRef.current = null;
      setIsSending(false);
      setStopping(false);
      setCardBusy(false);

      // 用户主动停止的半截回答不入历史；协议错误与运行失败也不当作成功轮次。
      if (exchange !== null && answer.completed && !stopped && !exchange.error && protocolError === null) {
        completionRef.current.onExchangeComplete(
          exchange.user,
          answerText(answer) || exchange.assistant,
          sources,
          answer.basis,
          exchange.policyVersion,
          null,
        );
      }
      completionRef.current.onChatComplete?.();
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

    const edit = editing;
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
      resumable: true,
    };
    isAtBottomRef.current = true;
    if (edit === null) {
      setMessages((prev) => [...prev, userMessage, emptyAssistant(assistantId)]);
      setInput("");
    } else {
      setEditSubmitting(true);
    }
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
      conversationId,
      runId: "",
      subscription: null,
      answer: emptyAnswerState(),
      process: emptyProcessState(),
      stopped: false,
      protocolError: null,
      replayMessage: null,
      renderTimer: null,
    };

    // 快照本轮贴图（发送后清空输入，但请求体仍带本轮的图）：后端把每条引用解析并
    // 落本区图片库，逐条结果由 RunDetail.image_refs 如实回报。
    const sentImages = images.slice(0, MAX_RUN_IMAGES).map((img) => img.src);
    const attachedImages = sentImages.length;
    if (edit === null) setImages([]);

    const rejectSend = (error: string): void => {
      if (edit === null) {
        failSend(assistantId, error);
        return;
      }
      runRef.current = null;
      exchangeRef.current = null;
      setIsSending(false);
      setEditSubmitting(false);
      onNotify?.("error", error);
    };

    // 1) 会话必须先存在（POST /runs 需要 conversation_id）。
    let targetConversation: string | null = null;
    try {
      targetConversation = await ensureConversation();
    } catch (error) {
      rejectSend(describeRunError(error));
      return;
    }
    if (targetConversation === null) {
      rejectSend("无法创建会话，请稍后重试");
      return;
    }
    const sending = runRef.current;
    if (sending === null || sending.assistantId !== assistantId) return;
    sending.conversationId = targetConversation;

    // 2) 创建并启动 run（202 Accepted）；图片引用与来源 run 随请求一起提交。
    let detail: RunDetail;
    const requestKey = JSON.stringify([edit?.messageId, message, sentImages, permissionMode, reasoningLevel]);
    if (edit !== null && editRequestRef.current?.key !== requestKey) {
      editRequestRef.current = { key: requestKey, id: nextClientRequestId() };
    }
    try {
      detail = await createRun(workspaceId, {
        conversationId: targetConversation,
        message,
        clientRequestId: edit !== null ? editRequestRef.current!.id : nextClientRequestId(),
        permissionMode,
        reasoningLevel,
        images: sentImages,
        // PRD 5.4 末句：会话移动后在目标区继续提问，新执行记录承接来源。
        originRunId,
        replaceFromMessageId: edit?.messageId,
      });
    } catch (error) {
      if (runRef.current === sending) rejectSend(describeRunError(error));
      return;
    }

    const current = runRef.current;
    if (current !== sending) {
      await cancelRun(workspaceId, detail.run_id, "user_stopped");
      return;
    }
    current.runId = detail.run_id;
    if (edit !== null) {
      setMessages((previous) => [
        ...previous.slice(0, edit.visibleIndex),
        { ...userMessage, runId: detail.run_id }, emptyAssistant(assistantId),
      ]);
      onEditCommitted?.(edit.storedIndex);
      setEditing(null);
      setEditSubmitting(false);
      setInput("");
      setImages([]);
      editRequestRef.current = null;
    } else {
      updateMessage(userId, (previous) => ({ ...previous, runId: detail.run_id }));
    }
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
    applyRunDetail(assistantId, detail);
    // 过程区域记录**实际发给本轮的**设置（后端详情回读，而非菜单草稿）。
    current.process = withRunSettings(
      current.process,
      `本轮设置：${PERMISSION_LABELS[detail.permission_mode]} · ${REASONING_LABELS[detail.reasoning_level]}`,
    );

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

  useEffect(() => {
    const previous = runRef.current;
    if (previous !== null && previous.conversationId === conversationId) return;
    ++processLoadGenerationRef.current;
    processLoadPendingRef.current.clear();
    processLoadedRef.current.clear();
    setProcessLoadState({});
    if (previous?.renderTimer !== null && previous?.renderTimer !== undefined) {
      window.clearTimeout(previous.renderTimer);
    }
    runRef.current = null;
    previous?.subscription?.close();
    exchangeRef.current = null;
    setIsSending(false);
    setEditing(null);
    setEditSubmitting(false);
    editRequestRef.current = null;
    setInput("");
    setCardError(null);
    setMessages(toChatMessages(initialMessages));
    nextIdRef.current = toChatMessages(initialMessages).length + 1;
    isAtBottomRef.current = true;
    if (conversationId === null) return;
    let disposed = false;
    const generation = processLoadGenerationRef.current;
    void restoreHistoricalProcesses(() => !disposed && generation === processLoadGenerationRef.current);

    const restore = async (): Promise<void> => {
      const runs = await listConversationRuns(workspaceId, conversationId);
      const lastMessage = initialMessages.at(-1);
      const active = runs.find((run) =>
        ["queued", "running", "waiting_approval", "waiting_input", "waiting_children", "cancelling"].includes(run.status),
      ) ?? (lastMessage?.role === "user"
        ? runs.find((run) => run.run_id === lastMessage.run_id) : undefined);
      if (disposed || active === undefined || runRef.current !== null) return;
      const history = await fetchEventLog(workspaceId, active.run_id);
      if (history.protocolError !== null) throw new Error(history.protocolError);
      const detail = await getRun(workspaceId, active.run_id);
      if (disposed || runRef.current !== null) return;
      const existing = toChatMessages(initialMessages).find((message) =>
        message.role === "assistant" && message.runId === active.run_id);
      const assistantId = existing?.id ?? nextIdRef.current++;
      if (existing === undefined) setMessages((items) => [...items, emptyAssistant(assistantId)]);
      const current = {
        assistantId, conversationId, runId: active.run_id,
        subscription: null as RunSubscription | null,
        answer: emptyAnswerState(), process: emptyProcessState(),
        stopped: false, protocolError: null,
        replayMessage: null, renderTimer: null,
      };
      runRef.current = current;
      setIsSending(true);
      // 历史包含多次执行，完整读取后才订阅新的事件。
      replayRunHistory(assistantId, history.events, existing ?? {
        ...emptyAssistant(assistantId), runId: active.run_id,
      });
      applyRunDetail(assistantId, detail);
      if (["completed", "completed_with_gaps", "failed", "cancelled"].includes(detail.status)
        && detail.last_event_seq <= history.lastSeq) {
        finishRun(assistantId);
        return;
      }
      current.subscription = subscribeRunEvents(workspaceId, active.run_id, {
        onEvent: (event) => handleRunEvent(assistantId, event),
        onProtocolError: (error) => updateMessage(assistantId, (message) => ({ ...message, protocolError: error })),
        onClose: () => finishRun(assistantId),
      }, { after: history.lastSeq });
    };
    void restore().catch((error) => { if (!disposed) onNotify?.("error", describeRunError(error)); });
    return () => { disposed = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- 按会话恢复一次订阅
  }, [conversationId, workspaceId]);

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

  /** 从失败位置继续执行，保留历史过程并订阅恢复请求之后的事件。 */
  const handleResume = async (assistantId: number, runId: string): Promise<void> => {
    if (isSending || runId === "") return;
    setIsSending(true);
    setCardBusy(false);
    setCardError(null);
    updateMessage(assistantId, (m) => ({
      ...m,
      error: null,
      protocolError: null,
      stopped: false,
      done: false,
      runStatus: "running",
    }));
    const current: ActiveRun = {
      assistantId,
      conversationId: runRef.current?.conversationId ?? conversationId,
      runId,
      subscription: null,
      answer: emptyAnswerState(),
      process: emptyProcessState(),
      stopped: false,
      protocolError: null,
      replayMessage: null,
      renderTimer: null,
    };
    runRef.current = current;
    let after = 0;
    try {
      // 在恢复请求前记录位置，后续产生的批准和答案均由订阅接收。
      const history = await fetchEventLog(workspaceId, runId);
      if (runRef.current !== current) return;
      if (history.protocolError !== null) throw new Error(history.protocolError);
      const seed = messagesRef.current.find((message) => message.id === assistantId);
      if (seed === undefined) throw new Error("继续执行时缺少原消息");
      replayRunHistory(assistantId, history.events, seed);
      after = history.lastSeq;
      const detail = await resumeRun(workspaceId, runId);
      if (runRef.current !== current) return;
      applyRunDetail(assistantId, detail);
      if (exchangeRef.current !== null) exchangeRef.current.error = false;
      updateMessage(assistantId, (m) => ({
        ...m,
        error: null,
        done: false,
        stopped: false,
        // 接口受理后后台开始执行，响应快照可能仍处于上一轮终态。
        runStatus: ["failed", "interrupted", "cancelled"].includes(detail.status) ? "queued" : detail.status,
      }));
    } catch (error) {
      failSend(assistantId, describeRunError(error));
      return;
    }
    if (runRef.current !== current) return;
    current.subscription = subscribeRunEvents(
      workspaceId,
      runId,
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
      { after },
    );
  };

  /** 停止：立即结束本地事件处理，同时通知后端取消执行。 */
  const handleStop = (): void => {
    const current = runRef.current;
    if (!isSending || current === null) return;
    current.stopped = true;
    setStopping(true);
    current.subscription?.close();
    updateMessage(current.assistantId, (m) => ({
      ...m,
      runStatus: "cancelled",
      stopped: true,
      done: true,
    }));
    const runId = current.runId;
    const assistantId = current.assistantId;
    finishRun(assistantId);
    if (current.runId === "") {
      return;
    }
    void cancelRun(workspaceId, runId, "user_stopped").catch((error) => {
      onNotify?.("error", describeRunError(error));
    });
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
      {editing !== null && (
        <div className="chat-edit-status" role="status">
          <span>{editSubmitting ? "正在提交修改…" : "正在编辑提问，提交后将替换这条及后续消息"}</span>
          <button type="button" onClick={handleCancelEdit} disabled={editSubmitting}>取消编辑</button>
        </div>
      )}
      <textarea
        ref={inputRef}
        className="chat-input"
        value={input}
        rows={1}
        onChange={handleInputChange}
        onKeyDown={handleInputKeyDown}
        placeholder="随心输入"
        aria-label="消息输入框"
        disabled={editSubmitting}
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
        disabled={isSending}
        onPermissionChange={(mode) => onPermissionModeChange?.(mode)}
      />
      <ModelPicker
        providers={modelProviders ?? []}
        selection={modelSelection ?? null}
        fallbackModel={currentModel ?? null}
        reasoningLevel={reasoningLevel}
        disabled={isSending}
        onSelect={(providerId, modelId) => onModelSelect?.(providerId, modelId)}
        onReasoningChange={(level) => onReasoningLevelChange?.(level)}
        onOpenSettings={() => onOpenSettings?.()}
      />
      {isSending ? (
        <button
          type="button"
          className="chat-stop"
          onClick={handleStop}
          disabled={editSubmitting}
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
            <div key={msg.id} className={`message-row message-${msg.role}`} data-run-id={msg.runId ?? undefined}>
              <div className={`message-avatar ${msg.role}`} aria-hidden="true">
                {msg.role === "user" ? (
                  <UserAvatar src={avatars?.user ?? null} />
                ) : (
                  <AssistantAvatar src={avatars?.agent ?? null} />
                )}
              </div>
              <div className="message-body">
                {/* 过程时间线在正文之前；`live` 使当前执行区间的用时每秒更新。 */}
                {msg.role === "assistant" && (
                  <ProcessArea
                    process={msg.process}
                    live={!msg.done && msg.error === null && !msg.stopped}
                  />
                )}
                {msg.role === "assistant" && msg.runId !== null && msg.done
                  && msg.process.entries.length === 0 && msg.process.tasks.length === 0
                  && runRef.current?.runId !== msg.runId && (
                    processLoadState[msg.runId] === "loaded" ? (
                      <span className="process-head process-load-status">本次执行没有处理过程记录</span>
                    ) : (
                      <button
                        type="button"
                        className="process-head process-load-button"
                        data-testid="process-load-button"
                        disabled={processLoadState[msg.runId] === "loading"}
                        onClick={() => {
                          if (msg.runId === null) return;
                          const generation = processLoadGenerationRef.current;
                          void loadHistoricalProcess(
                            { id: msg.id, runId: msg.runId },
                            () => generation === processLoadGenerationRef.current,
                          ).catch((error) => onNotify?.("error", `处理过程读取失败：${describeRunError(error)}`));
                        }}
                      >
                        {processLoadState[msg.runId] === "loading" ? "正在读取处理过程…"
                          : processLoadState[msg.runId] === "failed" ? "读取失败，点击重试"
                            : "查看处理过程"}
                      </button>
                    )
                  )}

                {msg.role === "assistant" && msg.answerBasis !== null && msg.answerBasis !== "insufficient" && (
                  <span className={`answer-basis answer-basis-${msg.answerBasis}`}>
                    {BASIS_LABELS[msg.answerBasis]}
                  </span>
                )}
                {msg.role === "user" ? (
                  <div className="message-content user">
                    {msg.content}
                    {!isSending && onEditCommitted !== undefined && (
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
                      <span>{msg.runStatus === "waiting_approval" ? "等待你的批准" :
                        msg.runStatus === "waiting_input" ? "等待你的回答" :
                        msg.done ? "本次任务已结束" : runningEntryText(msg.process) ?? THINKING_TEXT}</span>
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

                {/* 筛选跑了但一条都没入选（PRD §8.2 的 empty 态）：必须说出来，
                    否则用户只看到「没有来源」而不知道这是相关性筛选的结果。 */}
                {msg.role === "assistant" && msg.retrieval?.status === "empty" && (
                  <p className="message-relevance-note" data-testid="relevance-empty-note">
                    {relevanceEmptyNote(msg.retrieval)}
                  </p>
                )}

                {msg.role === "assistant" && (msg.runStatus === "failed"
                  || (msg.resumable && ["cancelled", "interrupted"].includes(msg.runStatus ?? ""))) && (
                  <div className="message-error" role="alert" data-testid="failed-note">
                    <span>{msg.runStatus === "failed" ? "执行失败" : "执行已停止"}，已完成的研究记录仍可查看。</span>
                    {msg.resumable && msg.runId !== null && msg.runId !== "" ? (
                      <button
                        type="button"
                        className="message-resume-button"
                        data-testid="resume-run"
                        data-run-id={msg.runId}
                        disabled={isSending}
                        onClick={() => {
                          void handleResume(msg.id, msg.runId as string);
                        }}
                      >
                        继续执行
                      </button>
                    ) : (
                      <span>该失败不能从记录位置继续，请重新提问。</span>
                    )}
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
