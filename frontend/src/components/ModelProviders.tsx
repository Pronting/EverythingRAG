import { useMemo, useState } from "react";
import { fetchProviderModels, testModelConnection } from "../api/settingsApi";
import type {
  ModelCatalogEntry,
  ModelSelectionView,
  ProviderModelView,
  ProviderView,
  ToastType,
} from "../types";

/**
 * 设置页「模型」：多供应商配置（可配置多个，按供应商分组管理模型）。
 *
 * 数据流：本组件只编辑**草稿**（``ProviderDraft``），点「保存」时由设置弹窗整表提交
 * （``PUT /api/settings`` 的 ``providers`` 是整表替换语义）。密钥永不回显：已保存的密钥
 * 在界面上只显示「已设置 + 尾4位」，用户输入新值才替换，点「清除 Key」才清除。
 *
 * 唯一出网动作是「获取可用模型」（服务端经唯一 OutboundClient 审计）；失败只展示稳定文案。
 */

/** 供应商草稿 = 脱敏视图 + 待保存的密钥输入状态。 */
export interface ProviderDraft extends ProviderView {
  /** 用户输入的新密钥（"" = 不改动已保存的密钥）。 */
  key_input: string;
  /** 用户点了「清除 Key」（保存时提交空串）。 */
  key_cleared: boolean;
}

export interface ModelProvidersProps {
  providers: ProviderDraft[];
  catalog: ModelCatalogEntry[];
  protocols: string[];
  selection: ModelSelectionView | null;
  onChange: (providers: ProviderDraft[]) => void;
  onSelect: (providerId: string, modelId: string) => void;
  onNotify: (type: ToastType, message: string) => void;
}

/** 供应商状态文案与圆点样式（不展示任何实现参数）。 */
const STATUS_LABELS: Record<ProviderView["status"], string> = {
  ready: "已就绪",
  needs_key: "缺少密钥",
  unconfigured: "未配置完整",
};

/** 生成一个未被占用的自定义供应商 id（与后端 ``next_custom_provider_id`` 同规则）。 */
function nextCustomProviderId(providers: ProviderDraft[]): string {
  const used = new Set(providers.map((item) => item.id));
  let index = 1;
  while (used.has(`custom-${index}`)) index += 1;
  return `custom-${index}`;
}

/** 校验供应商草稿：返回错误消息，null 表示可以保存。 */
export function validateProviderDrafts(providers: ProviderDraft[]): string | null {
  const ids = new Set<string>();
  for (const provider of providers) {
    const id = provider.id.trim();
    if (id === "") return "供应商 id 不能为空";
    if (ids.has(id)) return `供应商 id 重复：${id}`;
    ids.add(id);
    if (provider.name.trim() === "") return "供应商名称不能为空";
    const base = (provider.base_url ?? "").trim();
    if (base !== "" && !/^https?:\/\/\S+$/i.test(base)) {
      return `供应商「${provider.name}」的 API 地址需为 http(s) 地址`;
    }
    const modelIds = new Set<string>();
    for (const model of provider.models) {
      const modelId = model.id.trim();
      if (modelId === "") return `供应商「${provider.name}」里有空模型名`;
      if (modelIds.has(modelId)) return `供应商「${provider.name}」里模型 id 重复：${modelId}`;
      modelIds.add(modelId);
    }
  }
  return null;
}

/** 草稿 -> PUT 负载的一条供应商（密钥省略 = 保留原值，空串 = 清除）。 */
export function toProviderUpdate(provider: ProviderDraft): {
  id: string;
  name: string;
  kind: "builtin" | "custom";
  protocol: string;
  base_url: string | null;
  api_key?: string;
  models: { id: string; name: string | null; vision: boolean }[];
} {
  const apiKey = provider.key_cleared ? "" : provider.key_input;
  return {
    id: provider.id.trim(),
    name: provider.name.trim(),
    kind: provider.kind,
    protocol: provider.protocol,
    base_url: (provider.base_url ?? "").trim() === "" ? null : (provider.base_url ?? "").trim(),
    ...(provider.key_cleared || provider.key_input !== "" ? { api_key: apiKey } : {}),
    models: provider.models
      .map((model) => ({
        id: model.id.trim(),
        name: model.name.trim() === model.id.trim() ? null : model.name.trim() || null,
        vision: model.vision,
      }))
      .filter((model) => model.id !== ""),
  };
}

/** 视图（脱敏）-> 草稿。防御旧后端缺字段（`models` 缺失按空目录处理，不抛错）。 */
export function draftFromView(provider: ProviderView): ProviderDraft {
  return {
    ...provider,
    models: (provider.models ?? []).map((model) => ({ ...model })),
    key_input: "",
    key_cleared: false,
  };
}

function CheckIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <path d="M3.5 8.5l3 3 6-7" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export default function ModelProviders({
  providers,
  catalog,
  protocols,
  selection,
  onChange,
  onSelect,
  onNotify,
}: ModelProvidersProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [catalogOpen, setCatalogOpen] = useState(false);
  const [modelSearch, setModelSearch] = useState("");
  const [fetchingId, setFetchingId] = useState<string | null>(null);
  const [testingId, setTestingId] = useState<string | null>(null);

  const addedIds = useMemo(() => new Set(providers.map((item) => item.id)), [providers]);
  const availableCatalog = catalog.filter((item) => !addedIds.has(item.id));

  const patch = (id: string, changes: Partial<ProviderDraft>): void => {
    onChange(providers.map((item) => (item.id === id ? { ...item, ...changes } : item)));
  };

  const patchModel = (providerId: string, index: number, changes: Partial<ProviderModelView>): void => {
    onChange(
      providers.map((item) =>
        item.id === providerId
          ? { ...item, models: item.models.map((model, i) => (i === index ? { ...model, ...changes } : model)) }
          : item,
      ),
    );
  };

  const addBuiltin = (entry: ModelCatalogEntry): void => {
    onChange([
      ...providers,
      draftFromView({
        id: entry.id,
        name: entry.name,
        kind: "builtin",
        protocol: protocols[0] ?? "openai-completions",
        base_url: entry.base_url,
        api_key_set: false,
        api_key_hint: null,
        models: entry.models,
        status: "unconfigured",
        selected: false,
      }),
    ]);
    setCatalogOpen(false);
    setExpandedId(entry.id);
  };

  const addCustom = (): void => {
    const id = nextCustomProviderId(providers);
    onChange([
      ...providers,
      draftFromView({
        id,
        name: "自定义提供方",
        kind: "custom",
        protocol: protocols[0] ?? "openai-completions",
        base_url: "",
        api_key_set: false,
        api_key_hint: null,
        models: [],
        status: "unconfigured",
        selected: false,
      }),
    ]);
    setExpandedId(id);
  };

  const removeProvider = (id: string): void => {
    onChange(providers.filter((item) => item.id !== id));
    if (expandedId === id) setExpandedId(null);
  };

  const addModelRow = (providerId: string): void => {
    onChange(
      providers.map((item) =>
        item.id === providerId
          ? { ...item, models: [...item.models, { id: "", name: "", vision: false }] }
          : item,
      ),
    );
  };

  const removeModelRow = (providerId: string, index: number): void => {
    onChange(
      providers.map((item) =>
        item.id === providerId ? { ...item, models: item.models.filter((_, i) => i !== index) } : item,
      ),
    );
  };

  const restoreDefaultModels = (provider: ProviderDraft): void => {
    const entry = catalog.find((item) => item.id === provider.id);
    if (entry === undefined) return;
    patch(provider.id, { models: entry.models.map((model) => ({ ...model })) });
    onNotify("success", `已恢复「${provider.name}」的预置模型（保存后生效）`);
  };

  const fetchModels = async (provider: ProviderDraft): Promise<void> => {
    if (fetchingId !== null) return;
    const base = (provider.base_url ?? "").trim();
    if (base === "") {
      onNotify("error", "请先填写 API 地址");
      return;
    }
    setFetchingId(provider.id);
    try {
      const result = await fetchProviderModels({
        provider_id: provider.id,
        base_url: base,
        ...(provider.key_input !== "" && !provider.key_cleared ? { api_key: provider.key_input } : {}),
      });
      if (!result.success) {
        onNotify("error", result.message ?? "获取可用模型失败");
        return;
      }
      const existing = new Map(provider.models.map((model) => [model.id, model]));
      const merged: ProviderModelView[] = [...provider.models];
      let added = 0;
      for (const item of result.models) {
        if (existing.has(item.id)) continue;
        merged.push({ id: item.id, name: item.id, vision: false });
        added += 1;
      }
      patch(provider.id, { models: merged });
      onNotify(
        "success",
        added > 0
          ? `获取到 ${result.count} 个模型，新增 ${added} 个（保存后生效）`
          : `获取到 ${result.count} 个模型，都已在本目录中`,
      );
    } catch (error) {
      onNotify("error", error instanceof Error ? error.message : "获取可用模型失败");
    } finally {
      setFetchingId(null);
    }
  };

  /**
   * 用该供应商**未保存**的地址/密钥做一次真实连通性测试（服务端不落盘测试密钥）。
   *
   * 结果只说明服务可达且协议匹配；它不构成模型能力结论（非官方端点仍需声明能力）。
   */
  const testConnection = async (provider: ProviderDraft): Promise<void> => {
    if (testingId !== null) return;
    const base = (provider.base_url ?? "").trim();
    const model = provider.models[0]?.id.trim() ?? "";
    if (base === "" || model === "") {
      onNotify("error", "请先填写 API 地址与至少一个模型");
      return;
    }
    setTestingId(provider.id);
    try {
      const result = await testModelConnection({
        kind: "chat",
        base_url: base,
        model,
        ...(provider.key_input !== "" && !provider.key_cleared ? { api_key: provider.key_input } : {}),
      });
      if (result.success) {
        onNotify("success", `「${provider.name}」连接成功（${result.latency_ms ?? "—"} 毫秒）`);
      } else {
        onNotify("error", `「${provider.name}」连接失败：${result.message}`);
      }
    } catch (error) {
      onNotify("error", error instanceof Error ? error.message : "连接测试失败");
    } finally {
      setTestingId(null);
    }
  };

  const keyword = modelSearch.trim().toLowerCase();
  return (
    <div className="model-providers" data-testid="model-providers">
      <div className="settings-section-heading">
        <h3>模型</h3>
      </div>
      <p className="settings-hint">填入各提供方的 API 密钥即可使用其模型；可以配置多个提供方。</p>

      <ul className="provider-list">
        {providers.map((provider) => {
          const expanded = expandedId === provider.id;
          const isSelected = selection !== null && selection.provider_id === provider.id;
          const models =
            keyword === ""
              ? provider.models
              : provider.models.filter(
                  (model) =>
                    model.id.toLowerCase().includes(keyword) || model.name.toLowerCase().includes(keyword),
                );
          return (
            <li
              key={provider.id}
              className={`provider-card${expanded ? " expanded" : ""}`}
              data-testid={`provider-${provider.id}`}
              data-expanded={expanded ? "true" : "false"}
            >
              <div className="provider-head">
                <div className="provider-title">
                  <span className="provider-name">{provider.name}</span>
                  {provider.kind === "custom" && <span className="provider-badge">自定义</span>}
                  <span
                    className={`provider-dot provider-dot-${provider.status}`}
                    data-testid={`provider-status-${provider.id}`}
                    title={STATUS_LABELS[provider.status]}
                    aria-label={STATUS_LABELS[provider.status]}
                  />
                  {isSelected && (
                    <span className="provider-current" data-testid={`provider-current-${provider.id}`}>
                      使用中
                    </span>
                  )}
                </div>
                <div className="provider-actions">
                  <button
                    type="button"
                    className="provider-edit"
                    aria-expanded={expanded}
                    onClick={() => {
                      setModelSearch("");
                      setExpandedId(expanded ? null : provider.id);
                    }}
                  >
                    编辑
                  </button>
                  {provider.kind === "custom" && (
                    <button
                      type="button"
                      className="provider-remove"
                      onClick={() => removeProvider(provider.id)}
                      aria-label={`删除提供方 ${provider.name}`}
                    >
                      删除
                    </button>
                  )}
                </div>
              </div>

              {expanded && (
                <div className="provider-body">
                  <label className="settings-field">
                    <span>API 密钥</span>
                    <input
                      type="password"
                      value={provider.key_input}
                      aria-label={`${provider.name} API 密钥`}
                      placeholder={
                        provider.api_key_set && !provider.key_cleared
                          ? "已配置——输入新值可替换"
                          : "输入 API 密钥"
                      }
                      onChange={(event) =>
                        patch(provider.id, { key_input: event.target.value, key_cleared: false })
                      }
                    />
                  </label>
                  <div className="provider-key-row">
                    {provider.api_key_set && !provider.key_cleared && (
                      <span className="provider-key-hint" data-testid={`provider-key-hint-${provider.id}`}>
                        已设置 {provider.api_key_hint ?? ""}
                      </span>
                    )}
                    {provider.api_key_set && !provider.key_cleared && (
                      <button
                        type="button"
                        className="settings-clear-key"
                        onClick={() => patch(provider.id, { key_input: "", key_cleared: true })}
                      >
                        清除 Key
                      </button>
                    )}
                    {provider.key_cleared && <span className="provider-key-hint">保存后清除</span>}
                  </div>

                  <section className="provider-advanced">
                    <h4>自定义设置</h4>
                    <label className="settings-field">
                      <span>显示名称</span>
                      <input
                        type="text"
                        value={provider.name}
                        aria-label={`${provider.id} 显示名称`}
                        onChange={(event) => patch(provider.id, { name: event.target.value })}
                      />
                    </label>
                    <label className="settings-field">
                      <span>API 地址</span>
                      <input
                        type="text"
                        value={provider.base_url ?? ""}
                        aria-label={`${provider.name} API 地址`}
                        placeholder="https://api.example.com/v1"
                        onChange={(event) => patch(provider.id, { base_url: event.target.value })}
                      />
                    </label>
                    <label className="settings-field">
                      <span>API 协议</span>
                      <select
                        value={provider.protocol}
                        aria-label={`${provider.name} API 协议`}
                        onChange={(event) => patch(provider.id, { protocol: event.target.value })}
                      >
                        {protocols.map((protocol) => (
                          <option key={protocol} value={protocol}>
                            {protocol}
                          </option>
                        ))}
                      </select>
                    </label>
                  </section>

                  <div className="provider-models">
                    <div className="provider-models-head">
                      <div>
                        <span className="provider-models-title">模型目录</span>
                        <span className="provider-models-sub" data-testid={`provider-models-count-${provider.id}`}>
                          {provider.models.length === 0 ? "尚未添加模型" : `已配置 ${provider.models.length} 个模型`}
                        </span>
                      </div>
                      <div className="provider-models-actions">
                        <button
                          type="button"
                          className="provider-link"
                          disabled={!catalog.some((item) => item.id === provider.id)}
                          title={
                            catalog.some((item) => item.id === provider.id)
                              ? "恢复该提供方的预置模型列表"
                              : "自定义提供方没有预置模型"
                          }
                          onClick={() => restoreDefaultModels(provider)}
                        >
                          恢复默认模型
                        </button>
                        <button
                          type="button"
                          className="provider-link"
                          disabled={testingId !== null}
                          data-testid={`provider-test-${provider.id}`}
                          onClick={() => void testConnection(provider)}
                        >
                          {testingId === provider.id ? "测试中…" : "测试连接"}
                        </button>
                        <button
                          type="button"
                          className="provider-link"
                          disabled={fetchingId !== null}
                          data-testid={`provider-fetch-models-${provider.id}`}
                          onClick={() => void fetchModels(provider)}
                        >
                          {fetchingId === provider.id ? "获取中…" : "获取可用模型"}
                        </button>
                      </div>
                    </div>

                    <input
                      type="search"
                      className="provider-model-search"
                      value={modelSearch}
                      placeholder="搜索模型"
                      aria-label={`${provider.name} 搜索模型`}
                      onChange={(event) => setModelSearch(event.target.value)}
                    />

                    <ul className="provider-model-rows">
                      {models.map((model) => {
                        const index = provider.models.indexOf(model);
                        const modelSelected =
                          selection !== null &&
                          selection.provider_id === provider.id &&
                          selection.model_id === model.id;
                        return (
                          <li className="provider-model-row" key={`${provider.id}-${index}`}>
                            <input
                              type="text"
                              className="provider-model-id"
                              value={model.id}
                              aria-label={`${provider.name} 模型 ${index + 1} 名称`}
                              placeholder="模型名（真实调用名）"
                              onChange={(event) => patchModel(provider.id, index, { id: event.target.value })}
                            />
                            <input
                              type="text"
                              className="provider-model-name"
                              value={model.name}
                              aria-label={`${provider.name} 模型 ${index + 1} 显示名`}
                              placeholder="显示名称（可选）"
                              onChange={(event) => patchModel(provider.id, index, { name: event.target.value })}
                            />
                            <label className="provider-model-vision" title="该模型可直接接收图片输入">
                              <input
                                type="checkbox"
                                checked={model.vision}
                                aria-label={`${provider.name} 模型 ${index + 1} 是否支持图片`}
                                onChange={(event) =>
                                  patchModel(provider.id, index, { vision: event.target.checked })
                                }
                              />
                              <span>视觉</span>
                            </label>
                            <button
                              type="button"
                              className={`provider-model-use${modelSelected ? " active" : ""}`}
                              aria-pressed={modelSelected}
                              aria-label={`使用 ${model.name || model.id}`}
                              title="设为当前对话模型"
                              onClick={() => onSelect(provider.id, model.id)}
                            >
                              {modelSelected ? <CheckIcon /> : "选用"}
                            </button>
                            <button
                              type="button"
                              className="provider-model-remove"
                              aria-label={`删除模型 ${model.name || model.id}`}
                              onClick={() => removeModelRow(provider.id, index)}
                            >
                              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                                <path
                                  d="M4 7h16M9 7V4h6v3m-8 0 1 13h8l1-13M10 11v6m4-6v6"
                                  stroke="currentColor"
                                  strokeWidth="1.5"
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                />
                              </svg>
                            </button>
                          </li>
                        );
                      })}
                      {models.length === 0 && (
                        <li className="provider-model-empty">
                          {provider.models.length === 0 ? "还没有模型，点「添加模型」或「获取可用模型」" : "没有匹配的模型"}
                        </li>
                      )}
                    </ul>
                    <button type="button" className="provider-add-model" onClick={() => addModelRow(provider.id)}>
                      + 添加模型
                    </button>
                    <p className="settings-hint provider-capability-hint">
                      非官方 DeepSeek 端点的模型需要在启动环境声明能力（EVERYTHING_RAG_CHAT_CAPABILITIES），
                      否则问答会明确提示能力未声明。
                    </p>
                  </div>
                </div>
              )}
            </li>
          );
        })}
        {providers.length === 0 && (
          <li className="provider-empty" data-testid="providers-empty">
            还没有提供方。点下面的「添加提供方」选一个预置端点，或「添加自定义提供方」自己填。
          </li>
        )}
      </ul>

      <div className="provider-add-row">
        <button
          type="button"
          className="provider-add"
          aria-expanded={catalogOpen}
          onClick={() => setCatalogOpen((current) => !current)}
        >
          + 添加提供方
        </button>
        <button type="button" className="provider-add" onClick={addCustom}>
          + 添加自定义提供方
        </button>
      </div>

      {catalogOpen && (
        <ul className="provider-catalog" aria-label="预置提供方">
          {availableCatalog.map((entry) => (
            <li key={entry.id}>
              <button type="button" className="provider-catalog-item" onClick={() => addBuiltin(entry)}>
                <span className="provider-catalog-name">{entry.name}</span>
                <span className="provider-catalog-base">{entry.base_url}</span>
                <span className="provider-catalog-summary">{entry.summary}</span>
              </button>
            </li>
          ))}
          {availableCatalog.length === 0 && <li className="provider-model-empty">预置提供方都已添加</li>}
        </ul>
      )}
    </div>
  );
}
