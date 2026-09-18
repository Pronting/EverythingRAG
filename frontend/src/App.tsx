import { useCallback, useEffect, useRef, useState } from "react";
import {
  appendConversationMessages,
  archiveConversation,
  createConversation,
  deleteConversation,
  forkConversation,
  generateConversationTitle,
  getConversation,
  listConversations,
  moveConversationInSidebar,
  moveConversationToWorkspace,
  truncateConversation,
} from "./api/conversationApi";
import { createWorkspace, deleteWorkspace, listWorkspaces, renameWorkspace } from "./api/workspaceApi";
import Chat from "./components/Chat";
import { CitationPanel } from "./components/CitationPanel";
import ConfirmDialog from "./components/ConfirmDialog";
import ConversationMenu from "./components/ConversationMenu";
import SettingsModal from "./components/SettingsModal";
import Sidebar from "./components/Sidebar";
import type { ConversationAction, ConversationMenuState } from "./components/Sidebar";
import type { DropTarget } from "./components/WorkspaceTree";
import { getSettings } from "./api/settingsApi";
import { persistTheme, readStoredTheme } from "./theme";
import type {
  ConversationMessage,
  ConversationSummary,
  Source,
  StatusResponse,
  ThemeValue,
  ToastType,
  Workspace,
} from "./types";
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

  /**
   * 侧边栏拖拽落点：把会话移到目标工作区的指定位置（同区即为手动排序）。
   *
   * 目标位次按「落点前后的兄弟会话」在当前排序里的下标推算：
   * 插到 before 之前 / after 之后；落在文件夹空白处则排到末尾。
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

      try {
        await moveConversationInSidebar(conversationId, {
          workspaceId: target.workspaceId,
          targetIndex,
        });
        await loadConversations();
        await loadWorkspaces();
        if (conversationId === activeConversationId) setActiveWorkspaceId(target.workspaceId);
      } catch (error) {
        showToast("error", error instanceof Error ? error.message : "移动失败，请稍后重试");
      }
    },
    [activeConversationId, conversations, loadConversations, loadWorkspaces, showToast],
  );

  /** 右键菜单「移动到工作区」：移动到目标分区（排在末尾，位置由拖拽微调）。 */
  const handleMoveConversation = useCallback(
    async (conversation: ConversationSummary, workspaceId: string): Promise<void> => {
      setMenuBusy(true);
      try {
        await moveConversationToWorkspace(conversation.id, workspaceId);
        setConversationMenu(null);
        await loadConversations();
        await loadWorkspaces();
        if (activeConversationId === conversation.id) setActiveWorkspaceId(workspaceId);
        const target = workspaces.find((item) => item.id === workspaceId);
        showToast("success", `已移动到「${target?.name ?? "工作区"}」`);
      } catch (error) {
        showToast("error", error instanceof Error ? error.message : "移动失败，请稍后重试");
      } finally {
        setMenuBusy(false);
      }
    },
    [activeConversationId, loadConversations, loadWorkspaces, showToast, workspaces],
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

  /** 一轮问答完成后：持久化消息；首次交互用 AI 生成标题；刷新侧边栏列表。
   *
   * 默认会话（未点「新对话」）首次问答时惰性新建会话——否则 activeConversationId
   * 为 null，本轮会被直接丢弃、不进侧边栏。
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

      let conversationId = activeConversationId;
      if (conversationId === null) {
        // 惰性新建会话：同步更新 activeId + 消息列表，避免 Chat 因 id 变化重置而丢消息
        try {
          const conversation = await createConversation(activeWorkspaceId);
          conversationId = conversation.id;
          setActiveConversationId(conversation.id);
          setConversations((prev) => [conversation, ...prev]);
          setChatMessages(newMessages);
        } catch (error) {
          setStatusError(error instanceof Error ? error.message : String(error));
          return;
        }
      } else {
        setChatMessages((prev) => [...prev, ...newMessages]);
      }

      // 会话消息仍在异步装载中：本轮结果没有可靠的历史基线，不追加落库（见 ref 注释）。
      if (loadingConversationRef.current) return;

      try {
        await appendConversationMessages(conversationId, newMessages);
        if (isFirst) {
          await generateConversationTitle(conversationId);
        }
        loadConversations();
        void loadWorkspaces(); // 新会话会让所属工作区的计数变化
      } catch (error) {
        setStatusError(error instanceof Error ? error.message : String(error));
      }
    },
    [activeConversationId, activeWorkspaceId, chatMessages.length, loadConversations, loadWorkspaces],
  );

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
      />
      <main className="app-main">
        <header className="workspace-header">
          <div className="workspace-breadcrumb"><span>个人空间</span><span aria-hidden="true">/</span><strong>知识对话</strong></div>
        </header>
        <Chat
          conversationId={activeConversationId}
          initialMessages={chatMessages}
          revision={chatRevision}
          onExchangeComplete={handleExchangeComplete}
          onEditUserMessage={handleEditUserMessage}
          onChatComplete={loadStatus}
          avatars={avatars}
          onCitationSelect={handleCitationSelect}
          searchConfigured={status?.config.search_configured ?? false}
          knowledgeOnly={knowledgeOnly}
        />
      </main>
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
          onSelect={(index) =>
            setCitationPanel((prev) => (prev === null ? prev : { ...prev, index }))
          }
        />
      )}
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
