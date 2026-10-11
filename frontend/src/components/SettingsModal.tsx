import { useEffect, useRef, useState } from "react";
import {
  deleteAvatar,
  getSettings,
  testRerankConnection,
  updateSettings,
  uploadAvatar,
} from "../api/settingsApi";
import type { AvatarKind } from "../api/settingsApi";
import type {
  ModelCatalogEntry,
  ModelSelectionView,
  RerankProfileView,
  RerankStatus,
  RerankTestResult,
  StatusResponse,
  SettingsUpdate,
  SettingsView,
  ThemeValue,
  ToastType,
} from "../types";
import BrandMark from "./BrandMark";
import ArchiveSettings from "./ArchiveSettings";
import KnowledgeSettings from "./KnowledgeSettings";
import ModelProviders, {
  draftFromView,
  toProviderUpdate,
  validateProviderDrafts,
} from "./ModelProviders";
import type { ProviderDraft } from "./ModelProviders";
import SettingsSelect from "./SettingsSelect";

const SETTINGS_PAGES = [
  { id: "knowledge", label: "知识库", icon: "M4 4h6v16H4zM14 4h6v16h-6z" },
  { id: "archive", label: "归档对话", icon: "M3 7h18v12H3zM3 3h18v4H3zM9 12h6" },
  { id: "memory", label: "记忆", icon: "M12 3a5 5 0 0 0-3 9v3h6v-3a5 5 0 0 0-3-9Zm-3 12v3h6v-3M10 21h4" },
  { id: "appearance", label: "外观与头像", icon: "M12 3a9 9 0 1 0 9 9c0-1.5-1-2-2.5-2H16a2 2 0 0 1-2-2V5.5C14 4 13.5 3 12 3ZM7 10h.01M8 15h.01M12 17h.01" },
  { id: "models", label: "模型服务", icon: "M5 5h14v14H5zM9 9h6v6H9zM9 2v3m6-3v3M9 19v3m6-3v3M2 9h3m-3 6h3m14-6h3m-3 6h3" },
  { id: "rerank", label: "相关性筛选", icon: "M10 4h10M10 12h7M10 20h4M4 4l1.6 1.6L8.4 2.6M4 12l1.6 1.6L8.4 10.6" },
  { id: "search", label: "联网搜索", icon: "M3 12h18M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Zm0 0c5 5 5 13 0 18-5-5-5-13 0-18Z" },
  { id: "answers", label: "回答偏好", icon: "M4 5h16v12H9l-5 4V5Zm4 4h8m-8 4h5" },
] as const;
export type SettingsPage = typeof SETTINGS_PAGES[number]["id"];

/** 设置页状态文案（PRD §8.1）；不展示 RRF / tokenizer 等实现参数。 */
const RERANK_STATUS_LABELS: Record<RerankStatus, string> = {
  disabled: "未启用",
  not_configured: "缺少配置",
  profile_mismatch: "档案不匹配",
  invalid: "配置无效",
  error: "配置异常",
  needs_validation: "可用（档案需复测）",
  ready: "已就绪",
};

/** 校准档案状态文案：只有「已验证」才是生产可用档案（PRD §6.5）。 */
const RERANK_PROFILE_STATUS_LABELS: Record<RerankProfileView["status"], string> = {
  candidate: "候选（未验收）",
  quality_passed: "质量通过（未验收）",
  validated: "已验证",
};

/** 验证日期展示：只给日期，缺失即「未验证」（不猜、不补）。 */
function formatValidationDate(value: string | null): string {
  if (value === null || value === "") return "未验证";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toISOString().slice(0, 10);
}

function ThemePreview({ mode }: { mode: ThemeValue }) {
  return <span className={`theme-preview theme-preview-${mode}`} aria-hidden="true">
    <span className="theme-preview-sidebar"><i /><i /><i /></span>
    <span className="theme-preview-main"><BrandMark /><i /><span className="theme-preview-composer" /></span>
  </span>;
}

interface SettingsModalProps {
  status: StatusResponse | null;
  onKnowledgeChanged: () => void;
  /** 归档恢复/删除后回调：父级刷新侧边栏会话列表。 */
  onConversationsChanged: () => void;
  open: boolean;
  /**
   * 打开时落在哪一页（默认「外观与头像」）。
   *
   * 供「从别处跳进设置」的入口用：输入框模型选择器的「管理提供方」必须直接落到
   * **模型服务**，而不是默认页（用户实测反馈：跳到了外观与头像 = 跳错位置）。
   */
  initialPage?: SettingsPage;
  onClose: () => void;
  /** 保存成功后回调（父级刷新 /api/status 等）。 */
  onSaved: () => void;
  /** 当前主题（父级状态，实时应用）。 */
  theme: ThemeValue;
  /** 切换主题：父级实时应用 + 过渡动画。 */
  onThemeChange: (next: ThemeValue) => void;
  /** 保存结果轻提示（成功/失败），父级以 toast 展示。 */
  onNotify: (type: ToastType, message: string) => void;
}

/** 单个头像选择器：预览 + 选择文件 + 恢复默认。 */
function AvatarField({
  kind,
  label,
  currentUrl,
  file,
  preview,
  remove,
  onSelect,
  onRemove,
}: {
  kind: AvatarKind;
  label: string;
  currentUrl: string | null;
  file: File | null;
  preview: string | null;
  remove: boolean;
  onSelect: (file: File, preview: string) => void;
  onRemove: () => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const shown = remove ? null : preview ?? currentUrl;

  return (
    <div className="avatar-field">
      <div className="avatar-preview" aria-hidden="true">
        {shown !== null ? (
          <img src={shown} alt="" className="avatar-img" />
        ) : (
          <AvatarFallback kind={kind} />
        )}
      </div>
      <div className="avatar-controls">
        <span className="avatar-label">{label}</span>
        {(remove || file !== null) && <span className="avatar-filename">
          {remove ? "将恢复默认" : file?.name}
        </span>}
        <div className="avatar-buttons">
          <button
            type="button"
            className="avatar-pick"
            onClick={() => inputRef.current?.click()}
          >
            选择图片
          </button>
          {(currentUrl !== null || file !== null || remove) && (
            <button type="button" className="avatar-remove" onClick={onRemove}>
              恢复默认
            </button>
          )}
        </div>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp,image/gif"
        className="avatar-input"
        aria-label={`选择${label}`}
        onChange={(e) => {
          const selected = e.target.files?.[0];
          if (!selected) return;
          onSelect(selected, URL.createObjectURL(selected));
          e.target.value = ""; // 允许再次选择同一文件
        }}
      />
    </div>
  );
}

/** 默认头像 SVG（与聊天区一致，未自定义时展示）。 */
function AvatarFallback({ kind }: { kind: AvatarKind }) {
  return kind === "user" ? (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 12a5 5 0 1 0 0-10 5 5 0 0 0 0 10Zm0 2c-4.7 0-8 3-8 6 0 1 .8 2 2 2h12c1.2 0 2-1 2-2 0-3-3.3-6-8-6Z" />
    </svg>
  ) : (
    <BrandMark />
  );
}

/** 设置弹窗：模型、白名单回答偏好、外观与头像。核心 RAG 策略由后端统一管理。 */
export default function SettingsModal({
  open,
  status,
  onKnowledgeChanged,
  onConversationsChanged,
  initialPage = "appearance",
  onClose,
  onSaved,
  theme,
  onThemeChange,
  onNotify,
}: SettingsModalProps) {
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [activePage, setActivePage] = useState<SettingsPage>("appearance");
  const [testingModel, setTestingModel] = useState<string | null>(null);
  const [connectionResult, setConnectionResult] = useState<{ success: boolean; message: string; latency_ms?: number } | null>(null);
  const connectionDialog = useRef<HTMLDialogElement>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const modalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (bodyRef.current) bodyRef.current.scrollTop = 0;
  }, [activePage]);

  useEffect(() => {
    if (!open) return;
    const previousFocus = document.activeElement as HTMLElement | null;
    modalRef.current?.querySelector<HTMLButtonElement>(".settings-close")?.focus();
    return () => previousFocus?.focus();
  }, [open]);

  // 对话模型：多供应商注册表（草稿在本地编辑，保存时整表提交）
  const [providers, setProviders] = useState<ProviderDraft[]>([]);
  const [modelCatalog, setModelCatalog] = useState<ModelCatalogEntry[]>([]);
  const [modelProtocols, setModelProtocols] = useState<string[]>(["openai-completions"]);
  const [modelSelection, setModelSelection] = useState<{
    provider_id: string;
    model_id: string;
  } | null>(null);

  // 嵌入模型表单（统一云端 OpenAI 兼容 /embeddings）
  const [embedBaseUrl, setEmbedBaseUrl] = useState("");
  const [embedModel, setEmbedModel] = useState("");
  const [embedKey, setEmbedKey] = useState("");
  const [embedKeyCleared, setEmbedKeyCleared] = useState(false);
  const [embedKeySet, setEmbedKeySet] = useState(false);
  const [embedKeyHint, setEmbedKeyHint] = useState<string | null>(null);

  // 识图模型表单（OpenAI 兼容多模态，可选；未配置则图片不索引）
  const [visionBaseUrl, setVisionBaseUrl] = useState("");
  const [visionModel, setVisionModel] = useState("");
  const [visionKey, setVisionKey] = useState("");
  const [visionKeyCleared, setVisionKeyCleared] = useState(false);
  const [visionKeySet, setVisionKeySet] = useState(false);
  const [visionKeyHint, setVisionKeyHint] = useState<string | null>(null);

  const testConnection = async (kind: "embed" | "vision") => {
    if (testingModel) return;
    const config = kind === "embed" ? [embedBaseUrl, embedModel, embedKey, embedKeyCleared] : [visionBaseUrl, visionModel, visionKey, visionKeyCleared];
    setTestingModel(kind);
    setConnectionResult(null);
    connectionDialog.current?.showModal();
    try {
      const response = await fetch("/api/settings/test-connection", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind, base_url: config[0], model: config[1], api_key: config[3] ? "" : config[2] || undefined }),
      });
      if (!response.ok) throw new Error(response.status === 404 || response.status === 405 ? "请更新并重启本地服务后重试" : "测试请求失败，请检查配置后重试");
      setConnectionResult(await response.json());
    } catch (cause) {
      setConnectionResult({ success: false, message: cause instanceof Error ? cause.message : "无法连接本地服务" });
    } finally { setTestingModel(null); }
  };

  // 资料相关性筛选表单（PRD §8.1）：启停、服务地址、模型、密钥、连接测试、档案与状态
  const [rerankEnabled, setRerankEnabled] = useState(false);
  const [rerankEndpoint, setRerankEndpoint] = useState("");
  const [rerankModel, setRerankModel] = useState("");
  const [rerankKey, setRerankKey] = useState("");
  const [rerankKeyCleared, setRerankKeyCleared] = useState(false);
  const [rerankKeySet, setRerankKeySet] = useState(false);
  const [rerankKeyHint, setRerankKeyHint] = useState<string | null>(null);
  const [rerankProfileId, setRerankProfileId] = useState("");
  const [rerankProfiles, setRerankProfiles] = useState<RerankProfileView[]>([]);
  const [rerankStatus, setRerankStatus] = useState<RerankStatus | null>(null);
  const [rerankAcceptanceEnv, setRerankAcceptanceEnv] = useState(false);
  const [rerankTesting, setRerankTesting] = useState(false);
  const [rerankTestResult, setRerankTestResult] = useState<RerankTestResult | null>(null);
  const rerankDialog = useRef<HTMLDialogElement>(null);

  /**
   * 相关性筛选连接测试：用未保存的表单值提交（服务端不持久化测试密钥）。
   *
   * 结果只说明服务可达且协议匹配，`quality_checked` 恒为 false —— 绝不当成质量验收通过。
   */
  const testRerank = async (): Promise<void> => {
    if (rerankTesting) return;
    setRerankTesting(true);
    setRerankTestResult(null);
    rerankDialog.current?.showModal();
    try {
      const endpoint = rerankEndpoint.trim();
      const model = rerankModel.trim();
      const result = await testRerankConnection({
        ...(endpoint !== "" ? { endpoint } : {}),
        ...(model !== "" ? { model } : {}),
        ...(rerankProfileId !== "" ? { profile_id: rerankProfileId } : {}),
        ...(rerankKey !== "" && !rerankKeyCleared ? { api_key: rerankKey } : {}),
      });
      setRerankTestResult(result);
    } catch (cause) {
      setRerankTestResult({
        success: false,
        quality_checked: false,
        message: cause instanceof Error ? cause.message : "无法连接本地服务",
      });
    } finally {
      setRerankTesting(false);
    }
  };

  // 联网搜索表单（Tavily）
  const [searchEnabled, setSearchEnabled] = useState(false);
  const [searchProvider, setSearchProvider] = useState<"tavily" | "searxng">("tavily");
  const [searchKey, setSearchKey] = useState("");
  const [searchKeyCleared, setSearchKeyCleared] = useState(false);
  const [searchKeySet, setSearchKeySet] = useState(false);
  const [searchKeyHint, setSearchKeyHint] = useState<string | null>(null);
  const [searchBaseUrl, setSearchBaseUrl] = useState("");
  const [searchMaxResults, setSearchMaxResults] = useState(5);
  const [memoryEnabled, setMemoryEnabled] = useState(true);

  // 只影响表达方式的结构化偏好；不能覆盖身份、grounding、防注入和引用规则。
  const [answerLanguage, setAnswerLanguage] = useState<"auto" | "zh-CN" | "en">("auto");
  const [answerVerbosity, setAnswerVerbosity] = useState<
    "concise" | "balanced" | "detailed"
  >("balanced");
  const [answerTone, setAnswerTone] = useState<"natural" | "professional">("natural");
  const [answerFormat, setAnswerFormat] = useState<"auto" | "prose" | "bullets">("auto");
  const [knowledgeOnly, setKnowledgeOnly] = useState(false);
  const [customInstructions, setCustomInstructions] = useState("");
  const [originalEmbed, setOriginalEmbed] = useState<SettingsView["embed"] | null>(null);

  // 头像：当前 URL（来自设置）+ 待保存的本地选择（文件 / objectURL 预览 / 恢复默认标记）
  const [avatarUser, setAvatarUser] = useState<string | null>(null);
  const [avatarAgent, setAvatarAgent] = useState<string | null>(null);
  const [avatarUserFile, setAvatarUserFile] = useState<File | null>(null);
  const [avatarUserPreview, setAvatarUserPreview] = useState<string | null>(null);
  const [avatarUserRemove, setAvatarUserRemove] = useState(false);
  const [avatarAgentFile, setAvatarAgentFile] = useState<File | null>(null);
  const [avatarAgentPreview, setAvatarAgentPreview] = useState<string | null>(null);
  const [avatarAgentRemove, setAvatarAgentRemove] = useState(false);

  // 打开时加载当前设置
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setLoading(true);
    // 打开落在哪一页由调用方决定（侧边栏齿轮 = 默认外观页；模型选择器「管理提供方」= 模型服务）
    setActivePage(initialPage);
    setError(null);
    setSaved(false);
    setEmbedKey("");
    setEmbedKeyCleared(false);
    setVisionKey("");
    setVisionKeyCleared(false);
    setSearchKey("");
    setSearchKeyCleared(false);
    setRerankKey("");
    setRerankKeyCleared(false);
    setRerankTestResult(null);
    setAvatarUser(null);
    setAvatarAgent(null);
    setAvatarUserFile(null);
    setAvatarUserPreview(null);
    setAvatarUserRemove(false);
    setAvatarAgentFile(null);
    setAvatarAgentPreview(null);
    setAvatarAgentRemove(false);
    getSettings()
      .then((view) => {
        if (cancelled) return;
        setProviders((view.providers ?? []).map(draftFromView));
        setModelCatalog(view.model_catalog ?? []);
        setModelProtocols(view.model_protocols ?? ["openai-completions"]);
        setModelSelection(
          // 旧后端 / 旧视图可能整段缺失：只有真的给了选择才回填，否则按「未选择」处理
          // （`undefined` 与 `null` 都必须走这条路，否则读 `.provider_id` 会抛错，
          // 把整个设置回填链打断 —— 后面几段会静默停在默认值）。
          view.model_selection === null || view.model_selection === undefined
            ? null
            : { provider_id: view.model_selection.provider_id, model_id: view.model_selection.model_id },
        );

        setEmbedBaseUrl(view.embed.base_url ?? "");
        setEmbedModel(view.embed.model ?? "");
        setEmbedKeySet(view.embed.api_key_set);
        setEmbedKeyHint(view.embed.api_key_hint);
        setVisionBaseUrl(view.vision?.base_url ?? "");
        setVisionModel(view.vision?.model ?? "");
        setVisionKeySet(view.vision?.api_key_set ?? false);
        setVisionKeyHint(view.vision?.api_key_hint ?? null);
        setSearchEnabled(view.search?.enabled ?? false);
        setSearchProvider(view.search?.provider_type ?? "tavily");
        setSearchKeySet(view.search?.api_key_set ?? false);
        setSearchKeyHint(view.search?.api_key_hint ?? null);
        setSearchBaseUrl(view.search?.base_url ?? "");
        setSearchMaxResults(view.search?.max_results ?? 5);
        setMemoryEnabled(view.memory_enabled ?? true);
        // 相关性筛选：旧后端 / 旧配置缺少该段时按未启用处理（不伪造已配置）。
        setRerankEnabled(view.rerank?.enabled ?? false);
        setRerankEndpoint(view.rerank?.endpoint ?? "");
        setRerankModel(view.rerank?.model ?? "");
        setRerankKeySet(view.rerank?.api_key_set ?? false);
        setRerankKeyHint(view.rerank?.api_key_hint ?? null);
        setRerankProfileId(view.rerank?.profile_id ?? "");
        setRerankProfiles(view.rerank?.profiles ?? []);
        setRerankStatus(view.rerank?.status ?? "disabled");
        setRerankAcceptanceEnv(view.rerank?.acceptance_env ?? false);
        setAnswerLanguage(view.answer_preferences.language);
        setAnswerVerbosity(view.answer_preferences.verbosity);
        setAnswerTone(view.answer_preferences.tone);
        setAnswerFormat(view.answer_preferences.response_format);
        setKnowledgeOnly(view.answer_preferences.knowledge_only);
        setCustomInstructions(view.answer_preferences.custom_instructions ?? "");
        setOriginalEmbed({ ...view.embed });
        setAvatarUser(view.avatars?.user ?? null);
        setAvatarAgent(view.avatars?.agent ?? null);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, initialPage]);

  // 释放 object URL（切换预览或卸载时），避免内存泄漏
  useEffect(() => {
    return () => {
      if (avatarUserPreview !== null) URL.revokeObjectURL(avatarUserPreview);
      if (avatarAgentPreview !== null) URL.revokeObjectURL(avatarAgentPreview);
    };
  }, [avatarUserPreview, avatarAgentPreview]);



  // 嵌入模型是否变更（提示需重新导入）
  const embedDirty =
    originalEmbed !== null &&
    (embedBaseUrl !== (originalEmbed.base_url ?? "") ||
      embedModel !== (originalEmbed.model ?? ""));

  /** 当前选中的校准档案（展示状态与验证日期；PRD §8.1）。 */
  const selectedProfile =
    rerankProfiles.find((profile) => profile.profile_id === rerankProfileId) ?? null;

  /**
   * 供应商草稿里的当前选择投影（「使用中」标记与勾选态都看它）。
   *
   * 从**草稿**推导而不是直接用后端视图：用户刚改了显示名/模型名时，标记必须跟着变。
   * 选择指向已被删掉的模型时返回 null（界面不再显示「使用中」）。
   */
  const draftSelection: ModelSelectionView | null = (() => {
    if (modelSelection === null) return null;
    const provider = providers.find((item) => item.id === modelSelection.provider_id);
    const model = provider?.models.find((item) => item.id === modelSelection.model_id);
    if (provider === undefined || model === undefined) return null;
    return {
      provider_id: provider.id,
      provider_name: provider.name,
      model_id: model.id,
      model_name: model.name || model.id,
      vision: model.vision,
    };
  })();

  const handleSave = async (): Promise<void> => {
    // 嵌入：仅在填写时才校验/发送；未配置不应阻塞头像、主题等其它项保存。
    const embedBase = embedBaseUrl.trim();
    const embedModelT = embedModel.trim();
    const embedPartiallyFilled =
      (embedBase !== "" || embedModelT !== "") && !(embedBase !== "" && embedModelT !== "");
    if (embedPartiallyFilled) {
      setActivePage("models");
      const message = "云端嵌入需同时提供 Base URL 与模型名";
      setError(message);
      onNotify("error", message);
      return;
    }
    // 识图：可选，仅在填写时校验；未配置不阻塞其它项保存。
    const visionBase = visionBaseUrl.trim();
    const visionModelT = visionModel.trim();
    const visionPartiallyFilled =
      (visionBase !== "" || visionModelT !== "") && !(visionBase !== "" && visionModelT !== "");
    if (visionPartiallyFilled) {
      setActivePage("models");
      const message = "识图模型需同时提供 Base URL 与模型名";
      setError(message);
      onNotify("error", message);
      return;
    }
    // 相关性筛选：启用前先在本地校验必填项（服务端仍会独立校验并返回 422）。
    const rerankEndpointT = rerankEndpoint.trim();
    const rerankModelT = rerankModel.trim();
    const rerankHasKey = rerankKey.trim() !== "" || (rerankKeySet && !rerankKeyCleared);
    if (rerankEnabled && (!rerankEndpointT || !rerankModelT || !rerankHasKey || !rerankProfileId)) {
      setActivePage("rerank");
      const message = "启用资料相关性筛选需要服务地址、模型、API Key 与校准档案齐备";
      setError(message);
      onNotify("error", message);
      return;
    }
    // 供应商草稿：本地先校验（服务端仍会独立校验并返回 422）。
    const providerError = validateProviderDrafts(providers);
    if (providerError !== null) {
      setActivePage("models");
      setError(providerError);
      onNotify("error", providerError);
      return;
    }

    setSaving(true);
    setError(null);
    setSaved(false);
    try {
      // 头像：先应用「恢复默认」/「上传新图」，再保存其余设置
      if (avatarUserRemove) {
        await deleteAvatar("user");
        setAvatarUser(null);
      } else if (avatarUserFile !== null) {
        setAvatarUser(await uploadAvatar("user", avatarUserFile));
      }
      if (avatarAgentRemove) {
        await deleteAvatar("agent");
        setAvatarAgent(null);
      } else if (avatarAgentFile !== null) {
        setAvatarAgent(await uploadAvatar("agent", avatarAgentFile));
      }

      const update: SettingsUpdate = {
        memory_enabled: memoryEnabled,
        // 对话模型改为「供应商整表替换 + 当前选择」：不再直接写 chat（后端由选择推导 chat，
        // 直接写 chat 会被理解为「绕过供应商选择」，因此这里不发送）。
        providers: providers.map(toProviderUpdate),
        // 提交**草稿里仍然存在**的选择：用户刚删掉了当前选中模型时，这里必须提交「清除」，
        // 否则服务端会因为选择指向不存在的模型而 422（界面与实际配置分叉）。
        model_selection:
          draftSelection === null
            ? { provider_id: null, model_id: null }
            : { provider_id: draftSelection.provider_id, model_id: draftSelection.model_id },
        search: {
          provider_type: searchProvider,
          enabled: searchEnabled,
          max_results: searchMaxResults,
          ...(searchKey !== "" || searchKeyCleared
            ? { api_key: searchKeyCleared ? "" : searchKey }
            : {}),
          ...(searchBaseUrl.trim() !== "" ? { base_url: searchBaseUrl.trim() } : {}),
        },
        answer_preferences: {
          custom_instructions: customInstructions,
          language: answerLanguage,
          verbosity: answerVerbosity,
          tone: answerTone,
          response_format: answerFormat,
          knowledge_only: knowledgeOnly,
        },
        theme,
      };
      // 相关性筛选：局部更新语义（省略 = 保留原值；api_key 空串 = 清除）。
      // 旧配置缺少该段时这里首次写入；启用校验由服务端独立执行（失败返回 422 且不落盘）。
      update.rerank = {
        enabled: rerankEnabled,
        ...(rerankEndpointT !== "" ? { endpoint: rerankEndpointT } : {}),
        ...(rerankModelT !== "" ? { model: rerankModelT } : {}),
        profile_id: rerankProfileId,
        ...(rerankKey !== "" || rerankKeyCleared
          ? { api_key: rerankKeyCleared ? "" : rerankKey }
          : {}),
      };
      // 嵌入仅在完整配置时发送（避免云端嵌入的必填校验阻塞其它项保存）
      if (embedBase !== "" && embedModelT !== "") {
        update.embed = {
          mode: "cloud",
          base_url: embedBase,
          model: embedModelT,
          ...(embedKey !== "" || embedKeyCleared ? { api_key: embedKeyCleared ? "" : embedKey } : {}),
        };
      }
      // 识图仅在完整配置时发送（可选功能，未配置则图片不索引）
      if (visionBase !== "" && visionModelT !== "") {
        update.vision = {
          base_url: visionBase,
          model: visionModelT,
          ...(visionKey !== "" || visionKeyCleared
            ? { api_key: visionKeyCleared ? "" : visionKey }
            : {}),
        };
      }
      const savedView = await updateSettings(update);

      setSaved(true);
      setEmbedKey("");
      setEmbedKeyCleared(false);
      setVisionKey("");
      setVisionKeyCleared(false);
      setSearchKey("");
      setSearchKeyCleared(false);
      // 回显：用保存后返回的脱敏视图同步各段「已设置/尾4位」状态，
      // 否则保存成功后页面不刷新（API key 不会立刻显示为「已设置」）
      setProviders((savedView.providers ?? []).map(draftFromView));
      setModelCatalog(savedView.model_catalog ?? modelCatalog);
      setModelProtocols(savedView.model_protocols ?? modelProtocols);
      setModelSelection(
        savedView.model_selection === null || savedView.model_selection === undefined
          ? null
          : {
              provider_id: savedView.model_selection.provider_id,
              model_id: savedView.model_selection.model_id,
            },
      );

      setEmbedKeySet(savedView.embed.api_key_set);
      setEmbedKeyHint(savedView.embed.api_key_hint);
      setVisionKeySet(savedView.vision?.api_key_set ?? false);
      setVisionKeyHint(savedView.vision?.api_key_hint ?? null);
      setSearchEnabled(savedView.search?.enabled ?? false);
      setSearchProvider(savedView.search?.provider_type ?? "tavily");
      setSearchBaseUrl(savedView.search?.base_url ?? "");
      setSearchMaxResults(savedView.search?.max_results ?? 5);
      setRerankEnabled(savedView.rerank?.enabled ?? false);
      setRerankEndpoint(savedView.rerank?.endpoint ?? "");
      setRerankModel(savedView.rerank?.model ?? "");
      setRerankKeySet(savedView.rerank?.api_key_set ?? false);
      setRerankKeyHint(savedView.rerank?.api_key_hint ?? null);
      setRerankProfileId(savedView.rerank?.profile_id ?? "");
      setRerankProfiles(savedView.rerank?.profiles ?? []);
      setRerankStatus(savedView.rerank?.status ?? "disabled");
      setRerankAcceptanceEnv(savedView.rerank?.acceptance_env ?? false);
      setRerankKey("");
      setRerankKeyCleared(false);
      setSearchKeySet(savedView.search?.api_key_set ?? false);
      setSearchKeyHint(savedView.search?.api_key_hint ?? null);
      setAnswerLanguage(savedView.answer_preferences.language);
      setAnswerVerbosity(savedView.answer_preferences.verbosity);
      setAnswerTone(savedView.answer_preferences.tone);
      setAnswerFormat(savedView.answer_preferences.response_format);
      setKnowledgeOnly(savedView.answer_preferences.knowledge_only);
      setCustomInstructions(savedView.answer_preferences.custom_instructions ?? "");
      setAvatarUserFile(null);
      setAvatarUserPreview(null);
      setAvatarUserRemove(false);
      setAvatarAgentFile(null);
      setAvatarAgentPreview(null);
      setAvatarAgentRemove(false);
      onNotify("success", "保存成功");
      onSaved();
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      setError(message);
      onNotify("error", message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="settings-overlay" hidden={!open} role="dialog" aria-modal="true" aria-label="设置">
      <div className="settings-modal" ref={modalRef} onKeyDown={(event) => {
        if (event.key === "Escape") { event.stopPropagation(); onClose(); }
        if (event.key !== "Tab") return;
        const elements = Array.from(modalRef.current?.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex="0"]') ?? []).filter((element) => element.getClientRects().length > 0);
        const first = elements[0];
        const last = elements[elements.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      }}>
        <dialog ref={connectionDialog} className="knowledge-confirm connection-result" aria-labelledby="connection-result-title" onKeyDown={(event) => event.stopPropagation()}>
          <h3 id="connection-result-title">{testingModel ? "正在测试连接…" : connectionResult?.success ? "连接成功" : "连接失败"}</h3>
          {(testingModel || !connectionResult?.success) && <p role="status">{testingModel ? "正在请求模型服务" : connectionResult?.message}</p>}
          {!testingModel && connectionResult?.success && <p>响应耗时 {connectionResult.latency_ms} 毫秒</p>}
          <div className="knowledge-confirm-actions"><button type="button" className="model-test-button" onClick={() => connectionDialog.current?.close()}>关闭</button></div>
        </dialog>
        {/* 相关性筛选连接测试结果：只说明服务可达（quality_checked=false），不代表质量验收。 */}
        <dialog ref={rerankDialog} className="knowledge-confirm connection-result" aria-labelledby="rerank-result-title" onKeyDown={(event) => event.stopPropagation()}>
          <h3 id="rerank-result-title">
            {rerankTesting
              ? "正在测试筛选服务…"
              : rerankTestResult?.success === true
                ? "筛选服务连接成功"
                : "筛选服务连接失败"}
          </h3>
          {rerankTesting && <p role="status">正在用固定公开文本请求筛选服务</p>}
          {!rerankTesting && rerankTestResult !== null && rerankTestResult.success !== true && (
            <p role="status">{rerankTestResult.message ?? "请检查服务地址、模型与 API Key"}</p>
          )}
          {!rerankTesting && rerankTestResult?.success === true && (
            <>
              <p>
                模型 {rerankTestResult.model ?? rerankModel} · 耗时 {rerankTestResult.latency_ms ?? "—"} 毫秒 ·
                返回 {rerankTestResult.scored_count ?? 0} 条排序分数
              </p>
              <p className="settings-warning">
                连通成功只代表服务可达且响应结构可用（quality_checked=false），<b>不代表质量验收通过</b>，
                也不代表相关性判断一定准确。
              </p>
            </>
          )}
          <div className="knowledge-confirm-actions"><button type="button" className="model-test-button" onClick={() => rerankDialog.current?.close()}>关闭</button></div>
        </dialog>
        <div className="settings-header">
          <div className="settings-heading"><h2>设置</h2></div>
          <button type="button" className="settings-close" onClick={onClose} aria-label="关闭设置">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <path d="M3 3l10 10M13 3L3 13" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        <div className="settings-layout">
          <nav className="settings-nav" aria-label="设置分类">
            {SETTINGS_PAGES.map((page) => <button key={page.id} type="button" className={`settings-nav-item${activePage === page.id ? " active" : ""}`} aria-current={activePage === page.id ? "page" : undefined} onClick={() => setActivePage(page.id)}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={page.icon} /></svg>
              <span>{page.label}</span>
            </button>)}
          </nav>
        <div className="settings-body" ref={bodyRef}>
          <section className="settings-section settings-knowledge" hidden={activePage !== "knowledge"}>
            <KnowledgeSettings status={status} onChanged={onKnowledgeChanged} onNotify={onNotify} />
          </section>
          {/* 归档对话：只在切到该页时挂载，避免每次打开设置都多打一次归档接口 */}
          {activePage === "archive" && (
            <section className="settings-section settings-archive">
              <ArchiveSettings
                onConversationsChanged={onConversationsChanged}
                onNotify={onNotify}
              />
            </section>
          )}
          {loading && <p className="settings-loading">加载中…</p>}

          {!loading && (
            <>
              <section className="settings-section settings-memory" hidden={activePage !== "memory"}>
                <h3>记忆</h3>
                <p className="settings-hint">
                  启用后，任务可以读取工作区记忆与当前会话保存的记忆快照。
                </p>
                <label className="settings-field settings-toggle">
                  <span>启用记忆</span>
                  <input
                    type="checkbox"
                    data-testid="settings-memory-enabled"
                    checked={memoryEnabled}
                    onChange={(event) => setMemoryEnabled(event.target.checked)}
                  />
                </label>
                {!memoryEnabled && (
                  <p className="settings-hint" data-testid="settings-memory-disabled-note">
                    关闭后，新任务会跳过工作区记忆与会话记忆快照读取。
                  </p>
                )}
              </section>

              <section className="settings-section settings-appearance" hidden={activePage !== "appearance"}>
                <h3>外观</h3>
                <div className="settings-segmented theme-options" role="group" aria-label="主题">
                  <button
                    type="button"
                    className={`settings-seg${theme === "light" ? " active" : ""}`}
                    onClick={() => onThemeChange("light")}
                    aria-pressed={theme === "light"}
                  >
                    <ThemePreview mode="light" /><span className="theme-option-label">浅色<span className="theme-selected" aria-hidden="true">{theme === "light" ? "✓" : ""}</span></span>
                  </button>
                  <button
                    type="button"
                    className={`settings-seg${theme === "dark" ? " active" : ""}`}
                    onClick={() => onThemeChange("dark")}
                    aria-pressed={theme === "dark"}
                  >
                    <ThemePreview mode="dark" /><span className="theme-option-label">深色<span className="theme-selected" aria-hidden="true">{theme === "dark" ? "✓" : ""}</span></span>
                  </button>
                </div>
              </section>

              <section className="settings-section settings-model" hidden={activePage !== "models"}>
                <ModelProviders
                  providers={providers}
                  catalog={modelCatalog}
                  protocols={modelProtocols}
                  selection={draftSelection}
                  onChange={setProviders}
                  onSelect={(providerId, modelId) =>
                    setModelSelection({ provider_id: providerId, model_id: modelId })
                  }
                  onNotify={onNotify}
                />
              </section>

              <section className="settings-section settings-model" hidden={activePage !== "models"}>
                <div className="settings-section-heading"><h3>嵌入模型</h3><button type="button" className="model-test-button" disabled={testingModel !== null} onClick={() => void testConnection("embed")} aria-label="测试嵌入模型连接">测试连接</button></div>
                <label className="settings-field">
                      <span>Base URL</span>
                      <input
                        type="text"
                        value={embedBaseUrl}
                        onChange={(e) => setEmbedBaseUrl(e.target.value)}
                        placeholder="https://api.siliconflow.cn/v1"
                      />
                    </label>
                    <label className="settings-field">
                      <span>模型名</span>
                      <input
                        type="text"
                        value={embedModel}
                        onChange={(e) => setEmbedModel(e.target.value)}
                        placeholder="Qwen/Qwen3-Embedding-8B"
                      />
                    </label>
                    <div className="settings-field">
                      <span>API Key</span>
                      <div className="settings-key-row">
                        <input
                          type="password"
                          value={embedKey}
                          onChange={(e) => {
                            setEmbedKey(e.target.value);
                            setEmbedKeyCleared(false);
                          }}
                          placeholder={embedKeySet && !embedKeyCleared ? "已设置" : "API Key（可选）"}
                          aria-label="嵌入模型 API Key"
                        />
                        {embedKeySet && !embedKeyCleared && !embedKey && embedKeyHint !== null && (
                          <span className="settings-key-hint">{embedKeyHint}</span>
                        )}
                    {embedKeySet && !embedKeyCleared && (
                        <button
                          type="button"
                          className="settings-clear-key"
                          onClick={() => {
                            setEmbedKey("");
                            setEmbedKeyCleared(true);
                          }}
                        >
                          清除 Key
                        </button>
                      )}
                      </div>
                    </div>
                {embedDirty && (
                  <p className="settings-warning" role="alert">
                    ⚠️ 切换嵌入模型后向量空间不兼容，需<b>重新导入全部文档</b>。
                  </p>
                )}
              </section>

              <section className="settings-section settings-model" hidden={activePage !== "models"}>
                <div className="settings-section-heading"><h3>识图模型</h3><button type="button" className="model-test-button" disabled={testingModel !== null} onClick={() => void testConnection("vision")} aria-label="测试识图模型连接">测试连接</button></div>
                <label className="settings-field">
                  <span>Base URL</span>
                  <input
                    type="text"
                    value={visionBaseUrl}
                    onChange={(e) => setVisionBaseUrl(e.target.value)}
                    placeholder="https://api.example.com/v1"
                  />
                </label>
                <label className="settings-field">
                  <span>模型名</span>
                  <input
                    type="text"
                    value={visionModel}
                    onChange={(e) => setVisionModel(e.target.value)}
                    placeholder="qwen2-vl:7b / gpt-4o-mini"
                  />
                </label>
                <div className="settings-field">
                  <span>API Key</span>
                  <div className="settings-key-row">
                    <input
                      type="password"
                      value={visionKey}
                      onChange={(e) => {
                        setVisionKey(e.target.value);
                        setVisionKeyCleared(false);
                      }}
                      placeholder={
                        visionKeySet && !visionKeyCleared
                          ? "已设置"
                          : "API Key（可选）"
                      }
                      aria-label="识图模型 API Key"
                    />
                    {visionKeySet && !visionKeyCleared && !visionKey && visionKeyHint !== null && (
                      <span className="settings-key-hint">{visionKeyHint}</span>
                    )}
                    {visionKeySet && !visionKeyCleared && (
                    <button
                      type="button"
                      className="settings-clear-key"
                      onClick={() => {
                        setVisionKey("");
                        setVisionKeyCleared(true);
                      }}
                    >
                      清除 Key
                    </button>
                  )}
                  </div>
                </div>
              </section>

              <section className="settings-section settings-rerank" hidden={activePage !== "rerank"}>
                {/* 资料相关性筛选（PRD §8.1）：只暴露用户需要决定的参数，不展示 RRF/tokenizer 等实现细节。 */}
                <div className="settings-section-heading">
                  <h3>资料相关性筛选</h3>
                  <button
                    type="button"
                    className="model-test-button"
                    disabled={rerankTesting}
                    onClick={() => void testRerank()}
                    aria-label="测试相关性筛选连接"
                  >
                    测试连接
                  </button>
                </div>
                <label className="settings-field settings-toggle">
                  <span>启用资料相关性筛选</span>
                  <input
                    type="checkbox"
                    checked={rerankEnabled}
                    onChange={(e) => setRerankEnabled(e.target.checked)}
                    aria-label="启用资料相关性筛选"
                  />
                </label>
                <p className="settings-hint">
                  启用后，每轮问答会把「本轮问题 + 候选资料片段」发送到下面的服务地址做相关性排序，
                  只用于决定哪些资料进入本轮回答；完整原文不会外发，未收录的私人正文也不会发送。
                </p>
                <label className="settings-field">
                  <span>服务地址</span>
                  <input
                    type="text"
                    value={rerankEndpoint}
                    onChange={(e) => setRerankEndpoint(e.target.value)}
                    placeholder="https://api.siliconflow.cn/v1/rerank"
                    aria-label="筛选服务地址"
                  />
                </label>
                <label className="settings-field">
                  <span>模型名</span>
                  <input
                    type="text"
                    value={rerankModel}
                    onChange={(e) => setRerankModel(e.target.value)}
                    placeholder="BAAI/bge-reranker-v2-m3"
                    aria-label="筛选模型名"
                  />
                </label>
                <div className="settings-field">
                  <span>API Key</span>
                  <div className="settings-key-row">
                    <input
                      type="password"
                      value={rerankKey}
                      onChange={(e) => {
                        setRerankKey(e.target.value);
                        setRerankKeyCleared(false);
                      }}
                      placeholder={rerankKeySet && !rerankKeyCleared ? "已设置" : "筛选服务 API Key"}
                      aria-label="筛选服务 API Key"
                    />
                    {rerankKeySet && !rerankKeyCleared && !rerankKey && rerankKeyHint !== null && (
                      <span className="settings-key-hint">{rerankKeyHint}</span>
                    )}
                    {rerankKeySet && !rerankKeyCleared && (
                      <button
                        type="button"
                        className="settings-clear-key"
                        onClick={() => {
                          setRerankKey("");
                          setRerankKeyCleared(true);
                        }}
                      >
                        清除 Key
                      </button>
                    )}
                  </div>
                </div>
                <label className="settings-field">
                  <span>校准档案</span>
                  <SettingsSelect
                    value={rerankProfileId}
                    onChange={(e) => setRerankProfileId(e.target.value)}
                    aria-label="校准档案"
                  >
                    {rerankProfiles.length === 0
                      ? [<option key="none" value="">暂无可用档案</option>]
                      : rerankProfiles.map((profile) => (
                          <option key={profile.profile_id} value={profile.profile_id}>
                            {`${profile.profile_id}（${RERANK_PROFILE_STATUS_LABELS[profile.status] ?? profile.status}）`}
                          </option>
                        ))}
                  </SettingsSelect>
                </label>
                <p className="settings-hint">
                  当前状态：
                  {RERANK_STATUS_LABELS[rerankStatus ?? "disabled"] ?? "未启用"}
                  {rerankAcceptanceEnv && <span className="settings-badge">验收环境</span>}
                </p>
                <p className="settings-hint">
                  档案状态：{selectedProfile === null
                    ? "未选择档案"
                    : RERANK_PROFILE_STATUS_LABELS[selectedProfile.status] ?? selectedProfile.status}
                  {" · "}验证日期：{formatValidationDate(selectedProfile?.validated_at ?? null)}
                </p>
                {selectedProfile?.needs_validation === true && (
                  <p className="settings-warning" role="alert">
                    资料或索引已变化，该档案需要重新验证后仍可继续使用。
                  </p>
                )}
                {rerankStatus === "profile_mismatch" && (
                  <p className="settings-warning" role="alert">
                    当前模型或服务地址与验证档案不匹配，保存启用会被拒绝。
                  </p>
                )}
              </section>

              <section className="settings-section settings-search" hidden={activePage !== "search"}>
                <h3>联网搜索</h3>
                <label className="settings-field settings-toggle">
                  <span>启用联网搜索</span>
                  <input
                    type="checkbox"
                    checked={searchEnabled}
                    onChange={(e) => setSearchEnabled(e.target.checked)}
                    aria-label="启用联网搜索"
                  />
                </label>
                <label className="settings-field">
                  <span>搜索服务</span>
                  <SettingsSelect
                    value={searchProvider}
                    onChange={(e) => setSearchProvider(e.target.value as "tavily" | "searxng")}
                    aria-label="搜索服务"
                  >
                    <option value="tavily">Tavily（推荐）</option>
                    <option value="searxng" disabled>
                      SearXNG（即将支持）
                    </option>
                  </SettingsSelect>
                </label>
                <div className="settings-field">
                  <span>API Key</span>
                  <div className="settings-key-row">
                    <input
                      type="password"
                      value={searchKey}
                      onChange={(e) => {
                        setSearchKey(e.target.value);
                        setSearchKeyCleared(false);
                      }}
                      placeholder={
                        searchKeySet && !searchKeyCleared ? "已设置" : "Tavily API Key"
                      }
                      aria-label="联网搜索 API Key"
                    />
                    {searchKeySet && !searchKeyCleared && !searchKey && searchKeyHint !== null && (
                      <span className="settings-key-hint">{searchKeyHint}</span>
                    )}
                    {searchKeySet && !searchKeyCleared && (
                    <button
                      type="button"
                      className="settings-clear-key"
                      onClick={() => {
                        setSearchKey("");
                        setSearchKeyCleared(true);
                      }}
                    >
                      清除 Key
                    </button>
                  )}
                  </div>
                </div>
                <label className="settings-field">
                  <span>结果数（1–10）</span>
                  <input
                    type="number"
                    min={1}
                    max={10}
                    value={searchMaxResults}
                    onChange={(e) => {
                      const value = Number(e.target.value);
                      setSearchMaxResults(
                        Number.isFinite(value) ? Math.min(10, Math.max(1, value)) : 5,
                      );
                    }}
                    aria-label="联网搜索结果数"
                  />
                </label>
              </section>

              <section className="settings-section settings-preferences" hidden={activePage !== "answers"}>
                <h3>回答偏好</h3>
                <label className="settings-field">
                  <span>回答语言</span>
                  <SettingsSelect
                    value={answerLanguage}
                    onChange={(e) =>
                      setAnswerLanguage(e.target.value as "auto" | "zh-CN" | "en")
                    }
                    aria-label="回答语言"
                  >
                    <option value="auto">跟随问题</option>
                    <option value="zh-CN">简体中文</option>
                    <option value="en">英文</option>
                  </SettingsSelect>
                </label>
                <label className="settings-field">
                  <span>回答篇幅</span>
                  <SettingsSelect
                    value={answerVerbosity}
                    onChange={(e) =>
                      setAnswerVerbosity(
                        e.target.value as "concise" | "balanced" | "detailed",
                      )
                    }
                    aria-label="回答篇幅"
                  >
                    <option value="concise">简洁</option>
                    <option value="balanced">适中</option>
                    <option value="detailed">详细</option>
                  </SettingsSelect>
                </label>
                <label className="settings-field">
                  <span>回答语气</span>
                  <SettingsSelect
                    value={answerTone}
                    onChange={(e) =>
                      setAnswerTone(e.target.value as "natural" | "professional")
                    }
                    aria-label="回答语气"
                  >
                    <option value="natural">自然</option>
                    <option value="professional">专业</option>
                  </SettingsSelect>
                </label>
                <label className="settings-field">
                  <span>输出格式</span>
                  <SettingsSelect
                    value={answerFormat}
                    onChange={(e) =>
                      setAnswerFormat(e.target.value as "auto" | "prose" | "bullets")
                    }
                    aria-label="输出格式"
                  >
                    <option value="auto">自动</option>
                    <option value="prose">段落</option>
                    <option value="bullets">要点列表</option>
                  </SettingsSelect>
                </label>
                <label className="settings-field settings-toggle">
                  <span>仅使用本地知识</span>
                  <input
                    type="checkbox"
                    checked={knowledgeOnly}
                    onChange={(e) => setKnowledgeOnly(e.target.checked)}
                    aria-label="仅使用本地知识"
                  />
                </label>
              </section>

              <section className="settings-section settings-custom" hidden={activePage !== "answers"}>
                <h3>自定义提示词</h3>
                <textarea className="settings-prompt" aria-label="自定义提示词" rows={6} maxLength={4000}
                  value={customInstructions} onChange={(event) => setCustomInstructions(event.target.value)}
                  placeholder="你希望助手如何回答？" />
                <div className="custom-instructions-meta"><span>{customInstructions.length} / 4000</span></div>
                <button type="button" className="custom-instructions-clear" disabled={!customInstructions} onClick={() => setCustomInstructions("")}>清空提示词</button>
              </section>

              <section className="settings-section settings-avatars" hidden={activePage !== "appearance"}>
                <h3>头像</h3>
                <AvatarField
                  kind="user"
                  label="用户头像"
                  currentUrl={avatarUser}
                  file={avatarUserFile}
                  preview={avatarUserPreview}
                  remove={avatarUserRemove}
                  onSelect={(file, preview) => {
                    setAvatarUserFile(file);
                    setAvatarUserPreview(preview);
                    setAvatarUserRemove(false);
                  }}
                  onRemove={() => {
                    setAvatarUserFile(null);
                    setAvatarUserPreview(null);
                    setAvatarUserRemove(true);
                  }}
                />
                <AvatarField
                  kind="agent"
                  label="Agent 头像"
                  currentUrl={avatarAgent}
                  file={avatarAgentFile}
                  preview={avatarAgentPreview}
                  remove={avatarAgentRemove}
                  onSelect={(file, preview) => {
                    setAvatarAgentFile(file);
                    setAvatarAgentPreview(preview);
                    setAvatarAgentRemove(false);
                  }}
                  onRemove={() => {
                    setAvatarAgentFile(null);
                    setAvatarAgentPreview(null);
                    setAvatarAgentRemove(true);
                  }}
                />
              </section>
            </>
          )}

        </div>
        </div>
        <div className="settings-feedback" aria-live="polite">
          {error !== null && (
            <p className="settings-error" role="alert">
              {error}
            </p>
          )}
          {saved && <p className="settings-saved">已保存</p>}
        </div>

        <div className="settings-footer">
          <button type="button" className="settings-cancel" onClick={onClose}>
            取消
          </button>
          <button
            type="button"
            className="settings-save"
            hidden={activePage === "knowledge"}
            onClick={() => void handleSave()}
            disabled={loading || saving}
          >
            {saving ? "保存中…" : "保存"}
          </button>
        </div>
      </div>
    </div>
  );
}
