import { useEffect, useState } from "react";
import { getSettings, updateSettings } from "../api/settingsApi";
import type { SettingsUpdate, SettingsView } from "../types";

interface SettingsModalProps {
  open: boolean;
  onClose: () => void;
  /** 保存成功后回调（父级刷新 /api/status 等）。 */
  onSaved: () => void;
}

/** 设置弹窗：对话模型 / 嵌入模型（本地或云端）/ 系统提示词。ChatGPT 风格分节表单。 */
export default function SettingsModal({ open, onClose, onSaved }: SettingsModalProps) {
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

  // 嵌入模型表单
  const [embedMode, setEmbedMode] = useState<"local" | "cloud">("local");
  const [embedBaseUrl, setEmbedBaseUrl] = useState("");
  const [embedModel, setEmbedModel] = useState("");
  const [embedKey, setEmbedKey] = useState("");
  const [embedKeyCleared, setEmbedKeyCleared] = useState(false);
  const [embedKeySet, setEmbedKeySet] = useState(false);
  const [embedKeyHint, setEmbedKeyHint] = useState<string | null>(null);

  const [systemPrompt, setSystemPrompt] = useState("");
  const [originalEmbed, setOriginalEmbed] = useState<SettingsView["embed"] | null>(null);

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
    getSettings()
      .then((view) => {
        if (cancelled) return;
        setChatBaseUrl(view.chat.base_url ?? "");
        setChatModel(view.chat.model ?? "");
        setChatKeySet(view.chat.api_key_set);
        setChatKeyHint(view.chat.api_key_hint);
        setEmbedMode(view.embed.mode);
        setEmbedBaseUrl(view.embed.base_url ?? "");
        setEmbedModel(view.embed.model ?? "");
        setEmbedKeySet(view.embed.api_key_set);
        setEmbedKeyHint(view.embed.api_key_hint);
        setSystemPrompt(view.system_prompt);
        setOriginalEmbed({ ...view.embed });
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

  if (!open) return null;

  // 嵌入模型是否变更（提示需重新导入）
  const embedDirty =
    originalEmbed !== null &&
    (embedMode !== originalEmbed.mode ||
      (embedMode === "cloud" &&
        (embedBaseUrl !== (originalEmbed.base_url ?? "") ||
          embedModel !== (originalEmbed.model ?? ""))));

  const cloudEmbedValid =
    embedMode !== "cloud" || (embedBaseUrl.trim() !== "" && embedModel.trim() !== "");

  const handleSave = async (): Promise<void> => {
    if (!cloudEmbedValid) {
      setError("云端嵌入需同时提供 Base URL 与模型名");
      return;
    }
    setSaving(true);
    setError(null);
    setSaved(false);
    const update: SettingsUpdate = {
      chat: {
        base_url: chatBaseUrl.trim() || undefined,
        model: chatModel.trim() || undefined,
        ...(chatKey !== "" || chatKeyCleared ? { api_key: chatKeyCleared ? "" : chatKey } : {}),
      },
      embed: {
        mode: embedMode,
        ...(embedMode === "cloud"
          ? { base_url: embedBaseUrl.trim() || undefined, model: embedModel.trim() || undefined }
          : {}),
        ...(embedKey !== "" || embedKeyCleared ? { api_key: embedKeyCleared ? "" : embedKey } : {}),
      },
      system_prompt: systemPrompt,
    };
    try {
      await updateSettings(update);
      setSaved(true);
      setChatKey("");
      setChatKeyCleared(false);
      setEmbedKey("");
      setEmbedKeyCleared(false);
      onSaved();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
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
              </section>

              <section className="settings-section">
                <h3>嵌入模型</h3>
                <div className="settings-segmented" role="radiogroup" aria-label="嵌入模式">
                  <button
                    type="button"
                    className={`settings-seg ${embedMode === "local" ? "active" : ""}`}
                    onClick={() => setEmbedMode("local")}
                  >
                    本地（bge-m3）
                  </button>
                  <button
                    type="button"
                    className={`settings-seg ${embedMode === "cloud" ? "active" : ""}`}
                    onClick={() => setEmbedMode("cloud")}
                  >
                    云端
                  </button>
                </div>
                {embedMode === "cloud" && (
                  <>
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
                        placeholder="BAAI/bge-m3"
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
                  </>
                )}
                {embedDirty && (
                  <p className="settings-warning" role="alert">
                    ⚠️ 切换嵌入模型后向量空间不兼容，需<b>重新导入全部文档</b>。
                  </p>
                )}
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
