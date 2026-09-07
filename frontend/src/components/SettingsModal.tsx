import { useEffect, useRef, useState } from "react";
import { deleteAvatar, getSettings, updateSettings, uploadAvatar } from "../api/settingsApi";
import type { AvatarKind } from "../api/settingsApi";
import type { StatusResponse, SettingsUpdate, SettingsView, ThemeValue, ToastType } from "../types";
import BrandMark from "./BrandMark";
import KnowledgeSettings from "./KnowledgeSettings";
import SettingsSelect from "./SettingsSelect";

const SETTINGS_PAGES = [
  { id: "knowledge", label: "知识库", icon: "M4 4h6v16H4zM14 4h6v16h-6z" },
  { id: "appearance", label: "外观与头像", icon: "M12 3a9 9 0 1 0 9 9c0-1.5-1-2-2.5-2H16a2 2 0 0 1-2-2V5.5C14 4 13.5 3 12 3ZM7 10h.01M8 15h.01M12 17h.01" },
  { id: "models", label: "模型服务", icon: "M5 5h14v14H5zM9 9h6v6H9zM9 2v3m6-3v3M9 19v3m6-3v3M2 9h3m-3 6h3m14-6h3m-3 6h3" },
  { id: "search", label: "联网搜索", icon: "M3 12h18M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Zm0 0c5 5 5 13 0 18-5-5-5-13 0-18Z" },
  { id: "answers", label: "回答偏好", icon: "M4 5h16v12H9l-5 4V5Zm4 4h8m-8 4h5" },
] as const;
type SettingsPage = typeof SETTINGS_PAGES[number]["id"];

function ThemePreview({ mode }: { mode: ThemeValue }) {
  return <span className={`theme-preview theme-preview-${mode}`} aria-hidden="true">
    <span className="theme-preview-sidebar"><i /><i /><i /></span>
    <span className="theme-preview-main"><BrandMark /><i /><span className="theme-preview-composer" /></span>
  </span>;
}

interface SettingsModalProps {
  status: StatusResponse | null;
  onKnowledgeChanged: () => void;
  open: boolean;
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

  // 对话模型表单
  const [chatBaseUrl, setChatBaseUrl] = useState("");
  const [chatModel, setChatModel] = useState("");
  const [chatKey, setChatKey] = useState("");
  const [chatKeyCleared, setChatKeyCleared] = useState(false);
  const [chatKeySet, setChatKeySet] = useState(false);
  const [chatKeyHint, setChatKeyHint] = useState<string | null>(null);
  const [chatSupportsImage, setChatSupportsImage] = useState(false);

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

  const testConnection = async (kind: "chat" | "embed" | "vision") => {
    if (testingModel) return;
    const config = kind === "chat" ? [chatBaseUrl, chatModel, chatKey, chatKeyCleared] : kind === "embed" ? [embedBaseUrl, embedModel, embedKey, embedKeyCleared] : [visionBaseUrl, visionModel, visionKey, visionKeyCleared];
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

  // 联网搜索表单（Tavily）
  const [searchEnabled, setSearchEnabled] = useState(false);
  const [searchProvider, setSearchProvider] = useState<"tavily" | "searxng">("tavily");
  const [searchKey, setSearchKey] = useState("");
  const [searchKeyCleared, setSearchKeyCleared] = useState(false);
  const [searchKeySet, setSearchKeySet] = useState(false);
  const [searchKeyHint, setSearchKeyHint] = useState<string | null>(null);
  const [searchBaseUrl, setSearchBaseUrl] = useState("");
  const [searchMaxResults, setSearchMaxResults] = useState(5);

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
    setActivePage("appearance");
    setError(null);
    setSaved(false);
    setChatKey("");
    setChatKeyCleared(false);
    setEmbedKey("");
    setEmbedKeyCleared(false);
    setVisionKey("");
    setVisionKeyCleared(false);
    setSearchKey("");
    setSearchKeyCleared(false);
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
        setChatBaseUrl(view.chat.base_url ?? "");
        setChatModel(view.chat.model ?? "");
        setChatKeySet(view.chat.api_key_set);
        setChatKeyHint(view.chat.api_key_hint);
        setChatSupportsImage(view.chat.supports_image);
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
  }, [open]);

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
        chat: {
          base_url: chatBaseUrl.trim() || undefined,
          model: chatModel.trim() || undefined,
          supports_image: chatSupportsImage,
          ...(chatKey !== "" || chatKeyCleared ? { api_key: chatKeyCleared ? "" : chatKey } : {}),
        },
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
      setChatKey("");
      setChatKeyCleared(false);
      setEmbedKey("");
      setEmbedKeyCleared(false);
      setVisionKey("");
      setVisionKeyCleared(false);
      setSearchKey("");
      setSearchKeyCleared(false);
      // 回显：用保存后返回的脱敏视图同步各段「已设置/尾4位」状态，
      // 否则保存成功后页面不刷新（API key 不会立刻显示为「已设置」）
      setChatKeySet(savedView.chat.api_key_set);
      setChatKeyHint(savedView.chat.api_key_hint);
      setChatSupportsImage(savedView.chat.supports_image);
      setEmbedKeySet(savedView.embed.api_key_set);
      setEmbedKeyHint(savedView.embed.api_key_hint);
      setVisionKeySet(savedView.vision?.api_key_set ?? false);
      setVisionKeyHint(savedView.vision?.api_key_hint ?? null);
      setSearchEnabled(savedView.search?.enabled ?? false);
      setSearchProvider(savedView.search?.provider_type ?? "tavily");
      setSearchBaseUrl(savedView.search?.base_url ?? "");
      setSearchMaxResults(savedView.search?.max_results ?? 5);
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
          {loading && <p className="settings-loading">加载中…</p>}

          {!loading && (
            <>
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
                <div className="settings-section-heading"><h3>对话模型</h3><button type="button" className="model-test-button" disabled={testingModel !== null} onClick={() => void testConnection("chat")} aria-label="测试对话模型连接">测试连接</button></div>
                <label className="settings-field">
                  <span>Base URL</span>
                  <input
                    type="text"
                    value={chatBaseUrl}
                    onChange={(e) => setChatBaseUrl(e.target.value)}
                    placeholder="https://api.example.com/v1"
                  />
                </label>
                <label className="settings-field">
                  <span>模型名</span>
                  <input
                    type="text"
                    value={chatModel}
                    onChange={(e) => setChatModel(e.target.value)}
                    placeholder="deepseek-chat / gpt-4o-mini"
                  />
                </label>
                <div className="settings-field">
                  <span>API Key</span>
                  <div className="settings-key-row">
                    <input
                      type="password"
                      value={chatKey}
                      onChange={(e) => {
                        setChatKey(e.target.value);
                        setChatKeyCleared(false);
                      }}
                      placeholder={chatKeySet && !chatKeyCleared ? "已设置" : "API Key（可选）"}
                      aria-label="对话模型 API Key"
                    />
                    {chatKeySet && !chatKeyCleared && !chatKey && chatKeyHint !== null && (
                      <span className="settings-key-hint">{chatKeyHint}</span>
                    )}
                    {chatKeySet && !chatKeyCleared && (
                    <button
                      type="button"
                      className="settings-clear-key"
                      onClick={() => {
                        setChatKey("");
                        setChatKeyCleared(true);
                      }}
                    >
                      清除 Key
                    </button>
                  )}
                  </div>
                </div>
                <label className="settings-field settings-toggle">
                  <span>图片输入</span>
                  <input
                    type="checkbox"
                    checked={chatSupportsImage}
                    onChange={(e) => setChatSupportsImage(e.target.checked)}
                    aria-label="多模态模型"
                  />
                </label>
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
