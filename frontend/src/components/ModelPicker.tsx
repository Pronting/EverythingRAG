import { useEffect, useMemo, useRef, useState } from "react";
import type { ModelSelectionView, ProviderView, ReasoningLevel } from "../types";
import { REASONING_LABELS, REASONING_ORDER } from "./PermissionMenu";

/**
 * 输入框右下角的「模型与推理等级」选择器（与「权限」菜单同一行的右侧、发送键左边）。
 *
 * 一个父菜单管两件事（对齐 Codex / DSH 的模型菜单）：
 * - 根面板：「模型」「推理等级」两行，各显示当前值；
 * - 进「模型」：原来的可搜索模型清单，**按供应商分组**；
 * - 进「推理等级」：关闭 / 轻度 / 深度 / 最高（PRD 18.1，默认关闭）。
 *
 * 选择即写入设置（``PUT /api/settings`` 的 ``model_selection``、会话的 ``reasoning_level``）
 * 并对**之后的提问**生效；正在跑的那一轮不受影响（后端在每次请求开始时读取当前设置），
 * 所以发送中整个选择器不可改。
 *
 * 没有任何可用模型时给出去设置页的入口，而不是一个空菜单。
 */

export interface ModelPickerProps {
  providers: ProviderView[];
  selection: ModelSelectionView | null;
  /** 当前生效的对话模型名（没有选择时的兜底展示；来自 settings.chat）。 */
  fallbackModel: string | null;
  /** 当前会话的推理等级（与后端读回的值同源，不在前端另存一份）。 */
  reasoningLevel: ReasoningLevel;
  disabled?: boolean;
  onSelect: (providerId: string, modelId: string) => void;
  onReasoningChange: (level: ReasoningLevel) => void;
  onOpenSettings: () => void;
}

interface PickerOption {
  providerId: string;
  providerName: string;
  modelId: string;
  modelName: string;
  vision: boolean;
}

/** 父菜单当前展示哪一屏：根面板 / 模型清单 / 推理等级。 */
type Pane = "root" | "model" | "effort";

/** 右向小箭头（根面板的「进下一屏」提示）。 */
function ChevronRight() {
  return (
    <svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <path d="m6 3 5 5-5 5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export default function ModelPicker({
  providers,
  selection,
  fallbackModel,
  reasoningLevel,
  disabled = false,
  onSelect,
  onReasoningChange,
  onOpenSettings,
}: ModelPickerProps) {
  const [open, setOpen] = useState(false);
  const [pane, setPane] = useState<Pane>("root");
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const root = useRef<HTMLDivElement>(null);

  /** 只把「有模型」的供应商作为可选项（没模型的供应商点不出任何东西）。 */
  const options = useMemo<PickerOption[]>(
    () =>
      providers.flatMap((provider) =>
        provider.models.map((model) => ({
          providerId: provider.id,
          providerName: provider.name,
          modelId: model.id,
          modelName: model.name || model.id,
          vision: model.vision,
        })),
      ),
    [providers],
  );

  const visible = useMemo(() => {
    const keyword = query.trim().toLowerCase();
    if (keyword === "") return options;
    return options.filter(
      (option) =>
        option.modelName.toLowerCase().includes(keyword) ||
        option.modelId.toLowerCase().includes(keyword) ||
        option.providerName.toLowerCase().includes(keyword),
    );
  }, [options, query]);

  /** 按供应商分组（保持后端返回的顺序），搜索命中供应商名时该组整组保留。 */
  const groups = useMemo(() => {
    const result: { providerId: string; providerName: string; items: PickerOption[] }[] = [];
    for (const option of visible) {
      const last = result[result.length - 1];
      if (last !== undefined && last.providerId === option.providerId) {
        last.items.push(option);
      } else {
        result.push({
          providerId: option.providerId,
          providerName: option.providerName,
          items: [option],
        });
      }
    }
    return result;
  }, [visible]);

  useEffect(() => {
    if (!open) return;
    const closeOutside = (event: PointerEvent): void => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", closeOutside);
    return () => document.removeEventListener("pointerdown", closeOutside);
  }, [open]);

  useEffect(() => {
    if (disabled) setOpen(false);
  }, [disabled]);

  useEffect(() => {
    setActive(0);
  }, [query, open]);

  const label = selection?.model_name ?? fallbackModel ?? "选择模型";
  const hasOptions = options.length > 0;

  /** 打开时总是回到根面板：上次停在「模型」清单里不该决定这次看到什么。 */
  const openMenu = (): void => {
    setPane("root");
    setQuery("");
    setOpen(true);
  };

  const choose = (option: PickerOption): void => {
    setOpen(false);
    setQuery("");
    if (
      selection !== null &&
      selection.provider_id === option.providerId &&
      selection.model_id === option.modelId
    ) {
      return;
    }
    onSelect(option.providerId, option.modelId);
  };

  const chooseReasoning = (level: ReasoningLevel): void => {
    setOpen(false);
    if (level !== reasoningLevel) onReasoningChange(level);
  };

  const onKeyDown = (event: React.KeyboardEvent<HTMLDivElement>): void => {
    if (event.key === "Escape") {
      // 在子面板里先退回根面板，再按一次才关掉整个菜单
      if (open && pane !== "root") setPane("root");
      else setOpen(false);
      return;
    }
    // 方向键只在模型清单里移动高亮；根面板与推理等级用 Tab/Enter 即可
    if (pane !== "model") return;
    if (!["ArrowDown", "ArrowUp", "Home", "End", "Enter"].includes(event.key)) return;
    event.preventDefault();
    if (!open) {
      setOpen(true);
      return;
    }
    if (event.key === "Enter") {
      const option = visible[active];
      if (option !== undefined) choose(option);
      return;
    }
    setActive((current) => {
      if (visible.length === 0) return 0;
      if (event.key === "Home") return 0;
      if (event.key === "End") return visible.length - 1;
      return (current + (event.key === "ArrowDown" ? 1 : -1) + visible.length) % visible.length;
    });
  };

  return (
    <div className="model-picker" ref={root} onKeyDown={onKeyDown}>
      <button
        type="button"
        className="model-picker-trigger"
        data-testid="model-picker"
        aria-label="选择模型"
        aria-haspopup="listbox"
        aria-expanded={open}
        title="选择对话模型与推理等级：对之后的提问生效"
        disabled={disabled}
        onClick={() => (open ? setOpen(false) : openMenu())}
      >
        <span className="model-picker-label">{label}</span>
        <span className="model-picker-effort">{REASONING_LABELS[reasoningLevel]}</span>
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <path d="m4 10 4-4 4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>

      {open && (
        <div className="model-picker-menu" role="menu" aria-label="模型与推理等级">
          {pane === "root" && (
            <>
              <button
                type="button"
                role="menuitem"
                className="model-picker-cell"
                data-testid="model-picker-model-cell"
                onClick={() => {
                  setQuery("");
                  setPane("model");
                }}
              >
                <span className="model-picker-cell-label">模型</span>
                <span className="model-picker-cell-value">{label}</span>
                <ChevronRight />
              </button>
              <button
                type="button"
                role="menuitem"
                className="model-picker-cell"
                data-testid="reasoning-menu"
                data-value={reasoningLevel}
                title="修改后从下一次模型调用起生效，已发出的请求继续使用原设置"
                onClick={() => setPane("effort")}
              >
                <span className="model-picker-cell-label">推理等级</span>
                <span className="model-picker-cell-value">{REASONING_LABELS[reasoningLevel]}</span>
                <ChevronRight />
              </button>
            </>
          )}

          {pane === "model" && (
            <>
              <button
                type="button"
                className="model-picker-back"
                data-testid="model-picker-back"
                onClick={() => setPane("root")}
              >
                <svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                  <path d="M10 3 5 8l5 5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                <span>模型</span>
              </button>
              <input
                type="search"
                className="model-picker-search"
                value={query}
                placeholder="搜索模型或服务商"
                aria-label="搜索模型或服务商"
                autoFocus
                onChange={(event) => setQuery(event.target.value)}
              />
              <div className="model-picker-list">
                {groups.map((group) => (
                  <div className="model-picker-group" key={group.providerId} data-testid={`model-group-${group.providerId}`}>
                    <div className="model-picker-group-title">{group.providerName}</div>
                    {group.items.map((option) => {
                      const selected =
                        selection !== null &&
                        selection.provider_id === option.providerId &&
                        selection.model_id === option.modelId;
                      const index = visible.indexOf(option);
                      return (
                        <button
                          type="button"
                          key={`${option.providerId}-${option.modelId}`}
                          role="option"
                          aria-selected={selected}
                          data-testid={`model-option-${option.modelId}`}
                          className={`model-picker-option${index === active ? " active" : ""}${selected ? " selected" : ""}`}
                          onPointerMove={() => setActive(index)}
                          onClick={() => choose(option)}
                        >
                          <span className="model-picker-option-name">{option.modelName}</span>
                          {option.vision && <span className="model-picker-tag">视觉</span>}
                          <span className="model-picker-check" aria-hidden="true">
                            {selected ? "✓" : ""}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                ))}
                {options.length === 0 && (
                  <div className="model-picker-empty" data-testid="model-picker-empty">
                    <p>还没有可用的模型。</p>
                    <button
                      type="button"
                      className="model-picker-settings"
                      onClick={() => {
                        setOpen(false);
                        onOpenSettings();
                      }}
                    >
                      去设置里添加提供方
                    </button>
                  </div>
                )}
                {options.length > 0 && visible.length === 0 && (
                  <div className="model-picker-empty" data-testid="model-picker-no-match">
                    <p>没有匹配「{query}」的模型。</p>
                  </div>
                )}
              </div>
              {hasOptions && (
                <div className="model-picker-foot">
                  <button
                    type="button"
                    className="model-picker-settings"
                    onClick={() => {
                      setOpen(false);
                      onOpenSettings();
                    }}
                  >
                    管理提供方
                  </button>
                </div>
              )}
            </>
          )}

          {pane === "effort" && (
            <>
              <button
                type="button"
                className="model-picker-back"
                data-testid="model-picker-back"
                onClick={() => setPane("root")}
              >
                <svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                  <path d="M10 3 5 8l5 5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                <span>推理等级</span>
              </button>
              <div className="model-picker-list" role="listbox" aria-label="推理等级">
                {REASONING_ORDER.map((level) => {
                  const selected = level === reasoningLevel;
                  return (
                    <button
                      type="button"
                      key={level}
                      role="option"
                      aria-selected={selected}
                      data-testid={`reasoning-option-${level}`}
                      className={`model-picker-option${selected ? " selected" : ""}`}
                      onClick={() => chooseReasoning(level)}
                    >
                      <span className="model-picker-option-name">{REASONING_LABELS[level]}</span>
                      <span className="model-picker-check" aria-hidden="true">
                        {selected ? "✓" : ""}
                      </span>
                    </button>
                  );
                })}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
