import { useCallback, useEffect, useRef, useState } from "react";
import {
  MoveApiError,
  MoveConflictError,
  archiveConversation,
  createConversation,
  deleteConversation,
  forgetConversationEpoch,
  forkConversation,
  generateConversationTitle,
  getConversation,
  knownConversationEpoch,
  listConversations,
  moveConversationInSidebar,
  moveConversationToWorkspace,
  truncateConversation,
} from "./api/conversationApi";
import type { MoveOutcome } from "./api/conversationApi";
import { listConversationRuns } from "./api/runApi";
import {
  createWorkspace,
  deleteWorkspace,
  listWorkspaces,
  renameWorkspace,
  reorderWorkspaces,
  setWorkspacePinned,
} from "./api/workspaceApi";
import Chat from "./components/Chat";
import BlockBrowser from "./components/BlockBrowser";
import { CitationPanel } from "./components/CitationPanel";
import ConfirmDialog from "./components/ConfirmDialog";
import ConversationMenu from "./components/ConversationMenu";
import MemoryPanel from "./components/MemoryPanel";
import SettingsModal from "./components/SettingsModal";
import Sidebar from "./components/Sidebar";
import type { ConversationAction, ConversationMenuState } from "./components/Sidebar";
import type { DropTarget } from "./components/WorkspaceTree";
import { getSettings } from "./api/settingsApi";
import { persistTheme, readStoredTheme } from "./theme";
import {
  DEFAULT_PERMISSION_MODE,
  DEFAULT_REASONING_LEVEL,
} from "./components/PermissionMenu";
import type {
  ConversationMessage,
  ConversationSummary,
  PermissionMode,
  ReasoningLevel,
  Source,
  StatusResponse,
  ThemeValue,
  ToastType,
  Workspace,
} from "./types";
import { DEFAULT_WORKSPACE_ID } from "./types";
import "./App.css";
import "./design.css";
import "./settings.css";

/** 应用外壳（ChatGPT 风格）：左侧边栏（会话历史/知识库/设置）+ 主聊天区。 */
export default function App() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [avatars, setAvatars] = useState<{ user: string | null; agent: string | null }>({
    user: null,
    agent: null,
  });
  const [knowledgeOnly, setKnowledgeOnly] = useState(false);
  const [theme, setTheme] = useState<ThemeValue>(() => readStoredTheme());
  const persistedThemeRef = useRef<ThemeValue>(readStoredTheme());
  const transitionTimeoutRef = useRef<number | null>(null);
  const [toast, setToast] = useState<{ type: ToastType; message: string } | null>(null);
  const toastTimerRef = useRef<number | null>(null);

  // 会话管理
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [chatMessages, setChatMessages] = useState<ConversationMessage[]>([]);
  /** 父级修订号：编辑重发截断历史后自增，让 Chat 按新历史重载消息。 */
  const [chatRevision, setChatRevision] = useState(0);
  const [citationPanel, setCitationPanel] = useState<{ sources: Source[]; index: number } | null>(
    null,
  );
  /**
   * 来源面板「读取完整原文」打开的块 id。
   *
   * 复用既有「语义块浏览」入口读全文；完整原文不进入本轮生成上下文（PRD §8.3）。
   */
  const [fullTextBlockId, setFullTextBlockId] = useState<string | null>(null);
  // 对话分区（工作区）：列表 + 当前分区（侧边栏会话列表只显示当前分区）
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [activeWorkspaceId, setActiveWorkspaceId] = useState<string | null>(null);
  // 会话右键菜单（Fork / 归档 / 删除）
  const [conversationMenu, setConversationMenu] = useState<ConversationMenuState | null>(null);
  const [menuBusy, setMenuBusy] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<ConversationSummary | null>(null);
  /**
   * 正在异步装载某条会话的消息（fork / 切换会话）。
   *
   * 装载窗口内若用户已经发问，本轮问答的消息不该再断言追加到该会话——
   * 否则历史会被重复写一遍（fork 场景尤其明显：分支本来就已含父对话的消息）。
   */
  const loadingConversationRef = useRef(false);

  // V1.2 会话级设置（PRD 14.1）：权限模式与思考深度，默认「请求批准 + 深度思考」。
  const [permissionMode, setPermissionMode] = useState<PermissionMode>(DEFAULT_PERMISSION_MODE);
  const [reasoningLevel, setReasoningLevel] = useState<ReasoningLevel>(DEFAULT_REASONING_LEVEL);
  const [memoryOpen, setMemoryOpen] = useState(false);

  /**
   * 待用户确认的移动（PRD 5.4）。
   *
   * 只有会话**携带历史/有活动任务**时才需要确认：移动会携带历史内容、停止当前任务，
   * 原区未发布的结果不会进入新执行。用户确认后才带 `confirm: true` 发送。
   */
  const [pendingMove, setPendingMove] = useState<{
    conversation: ConversationSummary;
    targetWorkspaceId: string;
    targetIndex?: number;
    reason: "needs_confirm" | "epoch_changed";
  } | null>(null);
  const [moveBusy, setMoveBusy] = useState(false);
  /**
   * 移动后「本次执行承接自哪一次 run」（PRD 5.4 末句）。
   *
   * 后端 `POST /runs` 接受 `origin_run_id` 但**不回显**，所以来源必须由前端记住并在
   * 过程区域展示，否则用户会以为历史凭空消失。
   */
  const [originRunIds, setOriginRunIds] = useState<Record<string, string>>({});

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

  // 轻提示（成功/失败）：顶部居中弹出，4 秒后自动消失
  const showToast = useCallback((type: ToastType, message: string): void => {
    setToast({ type, message });
    if (toastTimerRef.current !== null) window.clearTimeout(toastTimerRef.current);
    toastTimerRef.current = window.setTimeout(() => setToast(null), 4000);
  }, []);

  /** 加载全部活跃会话：侧边栏按工作区分组展示，不再按分区过滤。 */
  const loadConversations = useCallback((): void => {
    listConversations()
      .then(setConversations)
      .catch((error: unknown) => setStatusError(error instanceof Error ? error.message : String(error)));
  }, []);

  /** 加载工作区列表；当前分区失效时回落到默认分区。 */
  const loadWorkspaces = useCallback(async (): Promise<Workspace[]> => {
    try {
      const items = await listWorkspaces();
      setWorkspaces(items);
      setActiveWorkspaceId((current) => {
        if (current !== null && items.some((item) => item.id === current)) return current;
        const fallback = items.find((item) => item.is_default) ?? items[0];
        return fallback?.id ?? null;
      });
      return items;
    } catch (error) {
      setStatusError(error instanceof Error ? error.message : String(error));
      return [];
    }
  }, []);

  /** 新建工作区：只刷新文件夹树（不切换会话视图）。 */
  const handleCreateWorkspace = useCallback(
    async (name: string): Promise<void> => {
      try {
        const workspace = await createWorkspace(name);
        await loadWorkspaces();
        // 新对话默认落在这个新分区（用户刚建它就是要用）
        setActiveWorkspaceId(workspace.id);
        showToast("success", `已创建分区「${workspace.name}」`);
      } catch (error) {
        showToast("error", error instanceof Error ? error.message : "创建工作区失败");
      }
    },
    [loadWorkspaces, showToast],
  );

  const handleRenameWorkspace = useCallback(
    async (id: string, name: string): Promise<void> => {
      try {
        await renameWorkspace(id, name);
        await loadWorkspaces();
      } catch (error) {
        showToast("error", error instanceof Error ? error.message : "重命名失败");
      }
    },
    [loadWorkspaces, showToast],
  );

  /** 删除分区：其中对话由后端归档（可在设置页「归档对话」找回）。 */
  const handleDeleteWorkspace = useCallback(
    async (workspace: Workspace): Promise<void> => {
      try {
        const archived = await deleteWorkspace(workspace.id);
        await loadWorkspaces();
        await loadConversations();
        if (activeConversationId !== null && activeWorkspaceId === workspace.id) {
          setActiveConversationId(null);
          setChatMessages([]);
          setChatRevision(0);
        }
        showToast("success", `已删除分区「${workspace.name}」${archived > 0 ? `，${archived} 条对话已归档` : ""}`);
      } catch (error) {
        showToast("error", error instanceof Error ? error.message : "删除工作区失败");
      }
    },
    [activeConversationId, activeWorkspaceId, loadConversations, loadWorkspaces, showToast],
  );

  /** 拖拽工作区排序：先本地立即生效（拖拽手感），再把新顺序落库。 */
  const handleReorderWorkspaces = useCallback(
    (orderedIds: string[]): void => {
      const byId = new Map(workspaces.map((workspace) => [workspace.id, workspace]));
      const optimistic = orderedIds
        .map((id) => byId.get(id))
        .filter((workspace): workspace is Workspace => workspace !== undefined);
      if (optimistic.length === workspaces.length) setWorkspaces(optimistic);
      void reorderWorkspaces(orderedIds)
        .then((items) => setWorkspaces(items))
        .catch((error: unknown) => {
          showToast("error", error instanceof Error ? error.message : "调整工作区顺序失败");
          void loadWorkspaces();
        });
    },
    [loadWorkspaces, showToast, workspaces],
  );

  /** 置顶 / 取消置顶工作区（别针）。 */
  const handleToggleWorkspacePin = useCallback(
    async (workspace: Workspace): Promise<void> => {
      const next = workspace.pinned !== true;
      try {
        await setWorkspacePinned(workspace.id, next);
        await loadWorkspaces();
      } catch (error) {
        showToast("error", error instanceof Error ? error.message : "置顶失败，请稍后重试");
      }
    },
    [loadWorkspaces, showToast],
  );

  /**
   * 侧边栏拖拽落点：把会话移到目标工作区的指定位置（同区即为手动排序）。
   *
   * 目标位次按「落点前后的兄弟会话」在当前排序里的下标推算：
   * 插到 before 之前 / after 之后；落在文件夹空白处则排到末尾。
   */
  /**
   * 目标区里最后一次执行（用于**静态移动**的承接来源）。
   *
   * 静态移动没有 `stopped_run_ids`，但 PRD 5.4 末句仍要求新执行记录「承接自哪一次 run」，
   * 所以从原区的 run 列表里取最后一次。取不到就返回 null（不编造）。
   */
  const latestRunIdFor = useCallback(
    async (workspaceId: string, conversationId: string): Promise<string | null> => {
      try {
        const runs = await listConversationRuns(workspaceId, conversationId);
        return runs.length > 0 ? runs[runs.length - 1].run_id : null;
      } catch {
        return null;
      }
    },
    [],
  );

  /** 移动成功后：刷新列表、记录承接来源、如实告知影响了什么。 */
  const afterMove = useCallback(
    async (
      conversationId: string,
      targetWorkspaceId: string,
      outcome: MoveOutcome,
    ): Promise<void> => {
      await loadConversations();
      await loadWorkspaces();
      if (conversationId === activeConversationId) setActiveWorkspaceId(targetWorkspaceId);

      // PRD 5.4 末句：记录本次执行承接自哪一次 run（被停止的那次优先）。
      const stopped = outcome.stopped_run_ids ?? [];
      let origin = stopped.length > 0 ? stopped[stopped.length - 1] : null;
      if (origin === null && outcome.from_workspace_id) {
        origin = await latestRunIdFor(outcome.from_workspace_id, conversationId);
      }
      if (origin !== null) {
        setOriginRunIds((prev) => ({ ...prev, [conversationId]: origin }));
      }

      const effects: string[] = [];
      if (stopped.length > 0) effects.push(`已停止 ${stopped.length} 个进行中的任务`);
      if ((outcome.superseded_approvals ?? 0) > 0) {
        effects.push(`${outcome.superseded_approvals} 个待批准已失效`);
      }
      if ((outcome.superseded_inputs ?? 0) > 0) {
        effects.push(`${outcome.superseded_inputs} 个待澄清已失效`);
      }
      const target = workspaces.find((item) => item.id === targetWorkspaceId);
      const suffix = effects.length > 0 ? `（${effects.join("；")}）` : "";
      showToast("success", `已移动到「${target?.name ?? "工作区"}」${suffix}`);
    },
    [
      activeConversationId,
      latestRunIdFor,
      loadConversations,
      loadWorkspaces,
      showToast,
      workspaces,
    ],
  );

  /**
   * 会话移动的**唯一执行入口**（右键与拖拽共用；PRD 5.4 / A27）。
   *
   * 两条入口在后端已收敛到同一状态机（`t35`），所以这里也只走一套逻辑：
   * - 会话携带历史或存在活动任务时**必须**先拿到用户确认，才带 `confirm: true`；
   *   绝不默认静默 confirm（那等于绕过 PRD 5.4 的用户确认语义）。
   * - `expected_context_epoch` 用「后端交给前端过的最近观测值」；epoch 冲突 → 丢弃该值、
   *   让用户重新确认，**不自动重试**（避免用同一个过期值反复撞 409）。
   */
  const runMove = useCallback(
    async (
      conversation: ConversationSummary,
      targetWorkspaceId: string,
      targetIndex: number | undefined,
      confirm: boolean,
    ): Promise<void> => {
      const options = {
        confirm,
        expectedContextEpoch: knownConversationEpoch(conversation.id),
      };
      setMoveBusy(true);
      try {
        const outcome =
          targetIndex === undefined
            ? await moveConversationToWorkspace(conversation.id, targetWorkspaceId, options)
            : await moveConversationInSidebar(
                conversation.id,
                { workspaceId: targetWorkspaceId, targetIndex },
                options,
              );
        setPendingMove(null);
        setConversationMenu(null);
        await afterMove(conversation.id, targetWorkspaceId, outcome);
      } catch (error) {
        if (error instanceof MoveConflictError) {
          if (error.kind === "context_epoch_conflict") {
            // 观测值已过期：丢弃它，要求用户重新确认（重试时会以 null 发出，不再撞同一个值）。
            forgetConversationEpoch(conversation.id);
            showToast("error", "会话状态已变化，请重新确认后再移动");
            setPendingMove({
              conversation,
              targetWorkspaceId,
              targetIndex,
              reason: "epoch_changed",
            });
            return;
          }
          if (error.kind === "confirm_required" && !confirm) {
            setPendingMove({ conversation, targetWorkspaceId, targetIndex, reason: "needs_confirm" });
            return;
          }
          // 已经确认过却仍被判需要确认：说明判断口径与后端不一致，如实报错而不是再弹一次（防死循环）。
          showToast("error", error.detail || "移动被拒绝，请刷新页面后重试");
          return;
        }
        if (error instanceof MoveApiError) {
          showToast("error", error.message);
          return;
        }
        showToast("error", error instanceof Error ? error.message : "移动失败，请稍后重试");
      } finally {
        setMoveBusy(false);
      }
    },
    [afterMove, showToast],
  );

  /**
   * 侧边栏拖拽落点：把会话移到目标工作区的指定位置（同区即为手动排序）。
   *
   * 目标位次按「落点前后的兄弟会话」在当前排序里的下标推算：
   * 插到 before 之前 / after 之后；落在文件夹空白处则排到末尾。
   *
   * 跨区拖拽 == 归属变更，与右键「移动到工作区」同一条状态机；**同区排序不需要确认**
   * （后端也只排序，不递增 epoch、不写权威行）。
   */
  const handleDropConversation = useCallback(
    async (conversationId: string, target: DropTarget): Promise<void> => {
      const conversation = conversations.find((item) => item.id === conversationId);
      if (conversation === undefined) return;
      const siblings = conversations
        .filter(
          (item) =>
            item.id !== conversationId && (item.workspace_id ?? "default") === target.workspaceId,
        )
        .sort(
          (left, right) =>
            (left.sort_index ?? Number.MAX_SAFE_INTEGER) - (right.sort_index ?? Number.MAX_SAFE_INTEGER),
        );
      let targetIndex = siblings.length;
      if (target.beforeId !== null) {
        const index = siblings.findIndex((item) => item.id === target.beforeId);
        if (index >= 0) targetIndex = index;
      } else if (target.afterId !== null) {
        const index = siblings.findIndex((item) => item.id === target.afterId);
        if (index >= 0) targetIndex = index + 1;
      }

      const currentWorkspaceId = conversation.workspace_id ?? DEFAULT_WORKSPACE_ID;
      if (currentWorkspaceId === target.workspaceId) {
        // 同区：仅排序，不涉及归属，无需确认。
        await runMove(conversation, target.workspaceId, targetIndex, false);
        return;
      }
      // 跨区：携带历史时先让用户确认真实影响（空会话交给后端的 409 判定，不无谓打扰）。
      if (conversation.message_count > 0) {
        setPendingMove({
          conversation,
          targetWorkspaceId: target.workspaceId,
          targetIndex,
          reason: "needs_confirm",
        });
        return;
      }
      await runMove(conversation, target.workspaceId, targetIndex, false);
    },
    [conversations, runMove],
  );

  /** 右键菜单「移动到工作区」：移动到目标分区（排在末尾，位置由拖拽微调）。 */
  const handleMoveConversation = useCallback(
    async (conversation: ConversationSummary, workspaceId: string): Promise<void> => {
      if ((conversation.workspace_id ?? DEFAULT_WORKSPACE_ID) === workspaceId) return;
      // 携带历史的会话必须先确认（PRD 5.4 首句）；空会话照旧直接移。
      if (conversation.message_count > 0) {
        setPendingMove({ conversation, targetWorkspaceId: workspaceId, reason: "needs_confirm" });
        return;
      }
      await runMove(conversation, workspaceId, undefined, false);
    },
    [runMove],
  );

  // 应用主题到 <html>：animate=true 时带短促全局过渡，否则即时（用于初始加载）。
  const applyTheme = useCallback((next: ThemeValue, animate: boolean): void => {
    const root = document.documentElement;
    // 清理上一次尚未结束的过渡定时器，避免快速连续切换时过渡被提前打断
    if (transitionTimeoutRef.current !== null) {
      window.clearTimeout(transitionTimeoutRef.current);
      transitionTimeoutRef.current = null;
    }
    if (animate && root.dataset.theme !== next) {
      root.classList.add("theme-transition");
      void root.offsetWidth; // 强制回流，确保过渡属性先生效再切换主题值
      root.dataset.theme = next;
      transitionTimeoutRef.current = window.setTimeout(() => {
        root.classList.remove("theme-transition");
        transitionTimeoutRef.current = null;
      }, 260);
    } else if (root.dataset.theme !== next) {
      root.dataset.theme = next;
      root.classList.remove("theme-transition");
    }
    setTheme(next);
  }, []);

  // 加载设置（头像 + 主题；设置页保存后也刷新），失败静默回退默认
  const loadSettings = useCallback((): void => {
    getSettings()
      .then((view) => {
        setAvatars(view.avatars ?? { user: null, agent: null });
        setKnowledgeOnly(view.answer_preferences?.knowledge_only ?? false);
        const next = view.theme ?? "light";
        persistedThemeRef.current = next;
        persistTheme(next);
        applyTheme(next, false);
      })
      .catch(() => {
        /* 忽略：加载失败保持默认 */
      });
  }, [applyTheme]);

  // 启动：状态 + 设置 + 分区（分区到位后再拉会话列表，避免拉成「全部分区」）
  // 启动：状态 + 设置 + 工作区 + 全部会话（侧边栏自行按工作区分组）
  const bootstrappedRef = useRef(false);
  useEffect(() => {
    if (bootstrappedRef.current) return;
    bootstrappedRef.current = true;
    loadStatus();
    loadSettings();
    void loadWorkspaces();
    loadConversations();
  }, [loadStatus, loadSettings, loadWorkspaces, loadConversations]);

  // 图片后台生成中时，周期轮询 /api/status 刷新图片识别进度（每 3s）
  const imagePending = status?.knowledge.image_tasks?.pending ?? 0;
  useEffect(() => {
    if (imagePending <= 0) return;
    const timer = window.setInterval(loadStatus, 3000);
    return () => window.clearInterval(timer);
  }, [imagePending, loadStatus]);

  // 设置页切换主题：实时应用 + 过渡动画（是否持久化由「保存」决定）
  const handleThemeChange = useCallback(
    (next: ThemeValue) => applyTheme(next, true),
    [applyTheme],
  );

  const handleNewChat = async (workspaceId?: string): Promise<void> => {
    try {
      // 新对话归入指定工作区（点文件夹则在那个文件夹里建；否则落在最近使用的分区）
      const targetWorkspace = workspaceId ?? activeWorkspaceId;
      const conversation = await createConversation(targetWorkspace);
      setConversations((prev) => [conversation, ...prev]);
      setActiveConversationId(conversation.id);
      setActiveWorkspaceId(targetWorkspace ?? null);
      setChatMessages([]);
    setChatRevision(0);
      setCitationPanel(null);
    } catch (error) {
      setStatusError(error instanceof Error ? error.message : String(error));
    }
  };

  const handleSelectConversation = async (id: string): Promise<void> => {
    if (id === activeConversationId) return;
    loadingConversationRef.current = true;
    try {
      const conversation = await getConversation(id);
      setActiveConversationId(conversation.id);
      setChatMessages(conversation.messages);
      setChatRevision(0);
      setCitationPanel(null);
      // 记住这条对话所属分区：后续「新对话」默认落在同一个文件夹里
      if (conversation.workspace_id) setActiveWorkspaceId(conversation.workspace_id);
    } catch (error) {
      setStatusError(error instanceof Error ? error.message : String(error));
    } finally {
      loadingConversationRef.current = false;
    }
  };

  /** 一轮问答完成后：刷新侧边栏与标题。
   *
   * V1.2 起消息落库由后端 run 的 `commit_answer` 在同一次事务里完成（用户消息在
   * `POST /runs` 时写入），前端**不再**调用 `/messages` 追加 —— 否则同一轮会被
   * 写两遍。这里只负责「首次交互生成标题 + 刷新列表」这些纯界面副作用。
   */
  const handleExchangeComplete = useCallback(
    async (
      userContent: string,
      assistantContent: string,
      sources: Source[] | null,
      answerBasis: ConversationMessage["answer_basis"],
      policyVersion: string | null,
      reasoning: string | null,
    ) => {
      const newMessages: ConversationMessage[] = [
        {
          role: "user",
          content: userContent,
          sources: null,
          answer_basis: null,
          policy_version: null,
          reasoning: null,
          created_at: new Date().toISOString(),
        },
        {
          role: "assistant",
          content: assistantContent,
          sources,
          answer_basis: answerBasis,
          policy_version: policyVersion,
          reasoning,
          created_at: new Date().toISOString(),
        },
      ];
      const isFirst = chatMessages.length === 0;

      // 会话由 Chat 在创建 run 之前惰性建好（见 handleEnsureConversation）。
      const conversationId = activeConversationId;
      if (conversationId === null) return;

      setChatMessages((prev) => [...prev, ...newMessages]);

      // 会话消息仍在异步装载中：跳过标题等副作用，避免与装载竞态。
      if (loadingConversationRef.current) return;

      try {
        if (isFirst) {
          await generateConversationTitle(conversationId);
        }
        loadConversations();
        void loadWorkspaces(); // 新会话会让所属工作区的计数变化
      } catch (error) {
        setStatusError(error instanceof Error ? error.message : String(error));
      }
    },
    [activeConversationId, chatMessages.length, loadConversations, loadWorkspaces],
  );

  /**
   * 惰性确保会话存在并返回 id。
   *
   * V1.2 的 `POST /api/workspaces/{w}/runs` 必须带 `conversation_id`，所以会话要在
   * 创建 run **之前**存在（V1.1 是问答完成后再惰性建会话）。
   */
  const handleEnsureConversation = useCallback(async (): Promise<string | null> => {
    if (activeConversationId !== null) return activeConversationId;
    try {
      const conversation = await createConversation(activeWorkspaceId);
      setActiveConversationId(conversation.id);
      setConversations((prev) => [conversation, ...prev]);
      void loadWorkspaces();
      return conversation.id;
    } catch (error) {
      showToast(
        "error",
        error instanceof Error ? error.message : "无法创建会话，请稍后重试",
      );
      return null;
    }
  }, [activeConversationId, activeWorkspaceId, loadWorkspaces, showToast]);

  const handleCitationSelect = useCallback((sources: Source[], index: number): void => {
    setCitationPanel({ sources, index });
  }, []);

  /**
   * 编辑重发：把会话历史从第 index 条起截断（该条及其后的轮次作废）并返回新历史。
   *
   * 与 ChatGPT 的编辑语义一致：改写中间某条提问后，后面的问答都失效；
   * 真截断落库后重发只追加，因此不会出现「旧卡片 + 新卡片」两份重复轮次。
   */
  const handleEditUserMessage = useCallback(
    async (index: number, _content: string): Promise<ConversationMessage[] | null> => {
      if (activeConversationId === null) return null;
      const next = chatMessages.slice(0, index);
      // 截断期间禁止把问答结果再追加落库（历史基线正在变化）
      loadingConversationRef.current = true;
      try {
        await truncateConversation(activeConversationId, index);
        setChatMessages(next);
        setChatRevision((value) => value + 1);
        return next;
      } catch (error) {
        showToast("error", error instanceof Error ? error.message : "无法编辑这条提问，请稍后重试");
        return null;
      } finally {
        loadingConversationRef.current = false;
      }
    },
    [activeConversationId, chatMessages, showToast],
  );

  const closeConversationMenu = useCallback((): void => setConversationMenu(null), []);

  /** 选中的会话被归档/删除后清空当前会话视图（避免继续显示已不存在的对话）。 */
  const detachIfActive = useCallback((id: string): void => {
      if (activeConversationId !== id) return;
      setActiveConversationId(null);
      setChatMessages([]);
      setChatRevision(0);
      setCitationPanel(null);
    },
    [activeConversationId],
  );

  /** 归档：从侧边栏移除（可在设置页「归档对话」恢复）。 */
  const handleArchiveConversation = useCallback(
    async (conversation: ConversationSummary): Promise<void> => {
      setMenuBusy(true);
      try {
        await archiveConversation(conversation.id);
        setConversations((prev) => prev.filter((item) => item.id !== conversation.id));
        detachIfActive(conversation.id);
        showToast("success", `已归档「${conversation.title}」`);
      } catch (error) {
        showToast("error", error instanceof Error ? error.message : "归档失败，请稍后重试");
      } finally {
        setMenuBusy(false);
        setConversationMenu(null);
      }
    },
    [detachIfActive, showToast],
  );

  /** Fork：深拷贝源会话为新分支，立即打开继续追问，并嵌到源会话下方。 */
  const handleForkConversation = useCallback(
    async (conversation: ConversationSummary): Promise<void> => {
      setMenuBusy(true);
      try {
        const branch = await forkConversation(conversation.id);
        setConversations((prev) => [branch, ...prev]);
        loadingConversationRef.current = true;
        const detail = await getConversation(branch.id);
        setActiveConversationId(detail.id);
        setChatMessages(detail.messages);
        setCitationPanel(null);
        setConversationMenu(null);
        showToast("success", "已创建分支对话，可继续追问");
      } catch (error) {
        showToast("error", error instanceof Error ? error.message : "分支失败，请稍后重试");
      } finally {
        loadingConversationRef.current = false;
        setMenuBusy(false);
      }
    },
    [showToast],
  );

  const handleDeleteConversation = useCallback(
    async (conversation: ConversationSummary): Promise<void> => {
      setMenuBusy(true);
      try {
        await deleteConversation(conversation.id);
        setConversations((prev) => prev.filter((item) => item.id !== conversation.id));
        detachIfActive(conversation.id);
        setPendingDelete(null);
        showToast("success", `已删除「${conversation.title}」`);
      } catch (error) {
        showToast("error", error instanceof Error ? error.message : "删除失败，请稍后重试");
      } finally {
        setMenuBusy(false);
      }
    },
    [detachIfActive, showToast],
  );

  const handleConversationAction = (action: ConversationAction): void => {
    const conversation = conversationMenu?.conversation;
    if (conversation === undefined) return;
    if (action === "fork") void handleForkConversation(conversation);
    else if (action === "archive") void handleArchiveConversation(conversation);
    else {
      setConversationMenu(null);
      setPendingDelete(conversation);
    }
  };

  return (
    <div className="app">
      <Sidebar
        statusError={statusError}
        conversations={conversations}
        activeConversationId={activeConversationId}
        onNewChat={(workspaceId) => void handleNewChat(workspaceId)}
        onSelectConversation={(id) => void handleSelectConversation(id)}
        onOpenSettings={() => setSettingsOpen(true)}
        onOpenConversationMenu={setConversationMenu}
        workspaces={workspaces}
        onCreateWorkspace={async () => {
          // 侧边栏「＋ 新建工作区」没有输入框位置：直接用占位名，随后可双击重命名
          await handleCreateWorkspace(`新工作区 ${workspaces.length + 1}`);
        }}
        onRenameWorkspace={handleRenameWorkspace}
        onDeleteWorkspace={handleDeleteWorkspace}
        onDropConversation={(conversationId, target) => void handleDropConversation(conversationId, target)}
        onReorderWorkspaces={handleReorderWorkspaces}
        onToggleWorkspacePin={(workspace) => void handleToggleWorkspacePin(workspace)}
      />
      <main className="app-main">
        <header className="workspace-header">
          <div className="workspace-breadcrumb"><span>个人空间</span><span aria-hidden="true">/</span><strong>知识对话</strong></div>
          <button
            type="button"
            className="workspace-header-action"
            data-testid="open-memory"
            onClick={() => setMemoryOpen(true)}
          >
            工作区记忆
          </button>
        </header>
        <Chat
          conversationId={activeConversationId}
          initialMessages={chatMessages}
          workspaceId={activeWorkspaceId ?? DEFAULT_WORKSPACE_ID}
          ensureConversation={handleEnsureConversation}
          revision={chatRevision}
          onExchangeComplete={handleExchangeComplete}
          onEditUserMessage={handleEditUserMessage}
          onChatComplete={loadStatus}
          avatars={avatars}
          onCitationSelect={handleCitationSelect}
          permissionMode={permissionMode}
          reasoningLevel={reasoningLevel}
          onPermissionModeChange={setPermissionMode}
          onReasoningLevelChange={setReasoningLevel}
          onNotify={showToast}
          originRunId={
            activeConversationId === null ? null : (originRunIds[activeConversationId] ?? null)
          }
        />
      </main>
      {/*
        移动确认（PRD 5.4）：会话携带历史或存在活动任务时必须由用户确认。
        文案如实说明后果 —— 停止当前任务、原区未发布结果不进入新执行、原审批失效。
      */}
      <ConfirmDialog
        open={pendingMove !== null}
        title="移动会话到其他工作区"
        description={
          pendingMove === null
            ? ""
            : `「${pendingMove.conversation.title}」携带 ${
                pendingMove.conversation.message_count
              } 条历史消息。移动会：① 停止该会话正在执行的任务与子任务；② 未发布的结果不会进入新工作区的执行；③ 待批准/待澄清的卡片失效；④ 历史在新的工作区里只读保留。${
                pendingMove.reason === "epoch_changed"
                  ? "（会话状态刚刚发生了变化，请重新确认本次移动。）"
                  : ""
              }`
        }
        confirmLabel="确认移动"
        busy={moveBusy}
        testId="move-confirm"
        onConfirm={() => {
          if (pendingMove === null) return;
          void runMove(
            pendingMove.conversation,
            pendingMove.targetWorkspaceId,
            pendingMove.targetIndex,
            true,
          );
        }}
        onCancel={() => setPendingMove(null)}
      />
      {memoryOpen && (
        <MemoryPanel
          workspaceId={activeWorkspaceId ?? DEFAULT_WORKSPACE_ID}
          onClose={() => setMemoryOpen(false)}
          onNotify={showToast}
        />
      )}
      {conversationMenu !== null && (
        <ConversationMenu
          state={conversationMenu}
          busy={menuBusy}
          workspaces={workspaces}
          onSelect={handleConversationAction}
          onMoveToWorkspace={(workspaceId) => {
            const conversation = conversationMenu?.conversation;
            if (conversation !== undefined) void handleMoveConversation(conversation, workspaceId);
          }}
          onClose={closeConversationMenu}
        />
      )}
      <ConfirmDialog
        open={pendingDelete !== null}
        title="删除对话"
        description={
          pendingDelete === null
            ? ""
            : `确定要删除「${pendingDelete.title}」吗？该操作不可撤销，它的分支对话会保留。`
        }
        confirmLabel="确认删除"
        danger
        busy={menuBusy}
        onConfirm={() => {
          if (pendingDelete !== null) void handleDeleteConversation(pendingDelete);
        }}
        onCancel={() => setPendingDelete(null)}
      />
      {citationPanel !== null && (
        <CitationPanel
          sources={citationPanel.sources}
          index={citationPanel.index}
          onClose={() => setCitationPanel(null)}
          onOpenFullText={(blockId) => setFullTextBlockId(blockId)}
          onSelect={(index) =>
            setCitationPanel((prev) => (prev === null ? prev : { ...prev, index }))
          }
        />
      )}
      <BlockBrowser
        open={fullTextBlockId !== null}
        focusBlockId={fullTextBlockId}
        onClose={() => setFullTextBlockId(null)}
      />
      <SettingsModal
        status={status}
        onKnowledgeChanged={loadStatus}
        onConversationsChanged={() => {
          // 归档页恢复/删除会改变会话归属，顺带刷新工作区计数
          loadConversations();
          void loadWorkspaces();
        }}
        open={settingsOpen}
        onClose={() => {
          setSettingsOpen(false);
          // 未保存则回退到已持久化的主题
          applyTheme(persistedThemeRef.current, true);
        }}
        onSaved={() => {
          // 当前主题即为已保存主题（同步到持久化基准 + 本地快照）
          persistedThemeRef.current = theme;
          persistTheme(theme);
          loadStatus();
          loadSettings();
        }}
        theme={theme}
        onThemeChange={handleThemeChange}
        onNotify={showToast}
      />
      {toast !== null && (
        <div className={`toast toast-${toast.type}`} role="status" aria-live="polite">
          {toast.type === "success" ? "✓ " : "⚠ "}
          {toast.message}
        </div>
      )}
    </div>
  );
}
