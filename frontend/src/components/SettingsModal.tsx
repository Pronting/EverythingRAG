import { useEffect, useRef, useState } from "react";
import { deleteAvatar, getSettings, updateSettings, uploadAvatar } from "../api/settingsApi";
import type { AvatarKind } from "../api/settingsApi";
import type { SettingsUpdate, SettingsView, ThemeValue, ToastType } from "../types";

interface SettingsModalProps {
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
        <span className="avatar-filename">
          {remove ? "将恢复默认" : file !== null ? file.name : currentUrl !== null ? "已自定义" : "默认头像"}
        </span>
        <div className="avatar-buttons">
          <button
            type="button"
            className="avatar-pick"
            onClick={() => inputRef.current?.click()}
          >
            选择图片…
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
    <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2l1.9 5.6L19.5 9.5l-5.6 1.9L12 17l-1.9-5.6L4.5 9.5l5.6-1.9L12 2Z" />
      <path d="M19 14l.9 2.6 2.6.9-2.6.9L19 21l-.9-2.6-2.6-.9 2.6-.9L19 14Z" opacity=".7" />
    </svg>
  );
}

/** 设置弹窗：外观 / 对话模型 / 嵌入模型（云端）/ 系统提示词 / 头像。ChatGPT 风格分节表单。 */
export default function SettingsModal({
  open,
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

  // 联网搜索表单（Tavily）
  const [searchEnabled, setSearchEnabled] = useState(false);
  const [searchProvider, setSearchProvider] = useState<"tavily" | "searxng">("tavily");
  const [searchKey, setSearchKey] = useState("");
  const [searchKeyCleared, setSearchKeyCleared] = useState(false);
  const [searchKeySet, setSearchKeySet] = useState(false);
  const [searchKeyHint, setSearchKeyHint] = useState<string | null>(null);
  const [searchBaseUrl, setSearchBaseUrl] = useState("");
  const [searchMaxResults, setSearchMaxResults] = useState(5);

  const [systemPrompt, setSystemPrompt] = useState("");
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
        setSystemPrompt(view.system_prompt);
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

  if (!open) return null;

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
        system_prompt: systemPrompt,
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
    <div className="settings-overlay" role="dialog" aria-modal="true" aria-label="设置">
      <div className="settings-modal">
        <div className="settings-header">
          <h2>设置</h2>
          <button type="button" className="settings-close" onClick={onClose} aria-label="关闭设置">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <path d="M3 3l10 10M13 3L3 13" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        <div className="settings-body">
          {loading && <p className="settings-loading">加载中…</p>}

          {!loading && (
            <>
              <section className="settings-section">
                <h3>外观</h3>
                <div className="settings-segmented" role="radiogroup" aria-label="主题">
                  <button
                    type="button"
                    className={`settings-seg${theme === "light" ? " active" : ""}`}
                    onClick={() => onThemeChange("light")}
                    aria-pressed={theme === "light"}
                  >
                    浅色
                  </button>
                  <button
                    type="button"
                    className={`settings-seg${theme === "dark" ? " active" : ""}`}
                    onClick={() => onThemeChange("dark")}
                    aria-pressed={theme === "dark"}
                  >
                    深色
                  </button>
                </div>
                <p className="settings-hint">切换主题时带约 1.5 秒的平滑过渡，默认浅色。</p>
              </section>

              <section className="settings-section">
                <h3>对话模型</h3>
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
                      placeholder={chatKeySet && !chatKeyCleared ? "已设置，留空保持不变" : "API Key（可选）"}
                      aria-label="对话模型 API Key"
                    />
                    {chatKeySet && !chatKeyCleared && chatKeyHint !== null && (
                      <span className="settings-key-hint">…{chatKeyHint.slice(1)}</span>
                    )}
                  </div>
                  {chatKeySet && (
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
                <label className="settings-field settings-toggle">
                  <span>多模态模型（支持图片输入）</span>
                  <input
                    type="checkbox"
                    checked={chatSupportsImage}
                    onChange={(e) => setChatSupportsImage(e.target.checked)}
                    aria-label="多模态模型"
                  />
                </label>
                <p className="settings-hint">
                  关闭（默认）时若贴图，会先用识图模型把图片转成文字再交给本模型（实验性「识图代理」）；开启则本模型直接接收图片。
                </p>
              </section>

              <section className="settings-section">
                <h3>嵌入模型</h3>
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
                          placeholder={embedKeySet && !embedKeyCleared ? "已设置，留空保持不变" : "API Key（可选）"}
                          aria-label="嵌入模型 API Key"
                        />
                        {embedKeySet && !embedKeyCleared && embedKeyHint !== null && (
                          <span className="settings-key-hint">…{embedKeyHint.slice(1)}</span>
                        )}
                      </div>
                      {embedKeySet && (
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
                {embedDirty && (
                  <p className="settings-warning" role="alert">
                    ⚠️ 切换嵌入模型后向量空间不兼容，需<b>重新导入全部文档</b>。
                  </p>
                )}
              </section>

              <section className="settings-section">
                <h3>识图模型（图片转文字，可选）</h3>
                <p className="settings-hint">
                  用于识别文档中的图床图片并转成文字描述入库。不配置则图片仅存链接、不参与检索。
                </p>
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
                          ? "已设置，留空保持不变"
                          : "API Key（云端必填，本地 Ollama 留空）"
                      }
                      aria-label="识图模型 API Key"
                    />
                    {visionKeySet && !visionKeyCleared && visionKeyHint !== null && (
                      <span className="settings-key-hint">…{visionKeyHint.slice(1)}</span>
                    )}
                  </div>
                  {visionKeySet && (
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
                <p className="settings-hint">
                  ⚠️ 云端识图会把图床图片发送给服务商；本地 Ollama 端点则图片不出机。配置后需重新导入文档才会生成图片描述。
                </p>
              </section>

              <section className="settings-section">
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
                  <select
                    value={searchProvider}
                    onChange={(e) => setSearchProvider(e.target.value as "tavily" | "searxng")}
                    aria-label="搜索服务"
                  >
                    <option value="tavily">Tavily（推荐）</option>
                    <option value="searxng" disabled>
                      SearXNG（即将支持）
                    </option>
                  </select>
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
                        searchKeySet && !searchKeyCleared ? "已设置，留空保持不变" : "Tavily API Key"
                      }
                      aria-label="联网搜索 API Key"
                    />
                    {searchKeySet && !searchKeyCleared && searchKeyHint !== null && (
                      <span className="settings-key-hint">…{searchKeyHint.slice(1)}</span>
                    )}
                  </div>
                  {searchKeySet && (
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
                <p className="settings-hint">
                  ⚠️ 开启后，你的问题原文会发送给搜索服务商（Tavily）。本功能默认关闭，发送时需在输入框手动点亮「联网搜索」按钮。
                </p>
              </section>

              <section className="settings-section">
                <h3>系统提示词</h3>
                <p className="settings-hint">问答时随消息发送给对话模型，约束回答行为。</p>
                <textarea
                  className="settings-prompt"
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                  rows={5}
                  aria-label="系统提示词"
                />
              </section>

              <section className="settings-section">
                <h3>头像</h3>
                <p className="settings-hint">自定义聊天区中「你」与 Agent 的头像，图片仅保存在本地。</p>
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
