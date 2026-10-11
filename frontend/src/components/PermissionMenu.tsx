import { useEffect, useId, useRef, useState } from "react";
import type { PermissionMode, ReasoningLevel } from "../types";

/**
 * 权限菜单（PRD 14.1）：请求批准 / 自动批准 / 完全访问权限，**默认请求批准**。
 *
 * 菜单展示**当前会话的实际设置**（值由父级从后端读回，不在前端另存一份，
 * 否则界面会与真正生效的设置分叉）。
 *
 * 推理等级（PRD 18.1：关闭 / 轻度 / 深度 / 最高，默认关闭思考）**不再单独占一个菜单**：
 * 它和模型选择同属输入框右下角的「模型与推理等级」父菜单（见 `ModelPicker.tsx`），
 * 这里的 `REASONING_LABELS` / `REASONING_ORDER` 仍是那套档位的唯一来源。
 */

export const PERMISSION_LABELS: Record<PermissionMode, string> = {
  ask: "请求批准",
  auto: "自动批准",
  full: "完全访问权限",
};

export const PERMISSION_HINTS: Record<PermissionMode, string> = {
  ask: "每次出网/读文件前都先问你",
  auto: "本轮同类操作不再逐次询问",
  full: "不再询问（仅在你确认风险后使用）",
};

export const REASONING_LABELS: Record<ReasoningLevel, string> = {
  off: "关闭思考",
  low: "轻度思考",
  high: "深度思考",
  max: "最高思考",
};

/** PRD 18.1 默认值：请求批准 + 关闭思考（与后端 `DEFAULT_REASONING_LEVEL` 一致）。 */
export const DEFAULT_PERMISSION_MODE: PermissionMode = "ask";
export const DEFAULT_REASONING_LEVEL: ReasoningLevel = "off";

const PERMISSION_ORDER: PermissionMode[] = ["ask", "auto", "full"];
export const REASONING_ORDER: ReasoningLevel[] = ["off", "low", "high", "max"];

export interface PermissionMenuProps {
  permissionMode: PermissionMode;
  onPermissionChange: (mode: PermissionMode) => void;
  disabled?: boolean;
}

interface ComposerSelectProps<T extends string> {
  label: string;
  testId: string;
  value: T;
  values: T[];
  labels: Record<T, string>;
  disabled: boolean;
  title?: string;
  onChange: (value: T) => void;
}

function ComposerSelect<T extends string>({
  label,
  testId,
  value,
  values,
  labels,
  disabled,
  title,
  onChange,
}: ComposerSelectProps<T>) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(() => Math.max(0, values.indexOf(value)));
  const root = useRef<HTMLDivElement>(null);
  const listId = useId();

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

  const choose = (next: T): void => {
    onChange(next);
    setOpen(false);
  };

  return (
    <div className="composer-select" ref={root}>
      <button
        type="button"
        className="composer-select-trigger"
        role="combobox"
        data-testid={testId}
        data-value={value}
        aria-label={label}
        aria-expanded={open}
        aria-controls={open ? listId : undefined}
        aria-activedescendant={open ? `${listId}-${active}` : undefined}
        disabled={disabled}
        title={title}
        onClick={() => {
          setActive(Math.max(0, values.indexOf(value)));
          setOpen((current) => !current);
        }}
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            setOpen(false);
            return;
          }
          if (!["ArrowDown", "ArrowUp", "Home", "End", "Enter", " "].includes(event.key)) {
            return;
          }
          event.preventDefault();
          if (!open) {
            setOpen(true);
            setActive(Math.max(0, values.indexOf(value)));
            return;
          }
          if (event.key === "Enter" || event.key === " ") {
            choose(values[active]);
            return;
          }
          setActive((current) => {
            if (event.key === "Home") return 0;
            if (event.key === "End") return values.length - 1;
            return (current + (event.key === "ArrowDown" ? 1 : -1) + values.length) % values.length;
          });
        }}
      >
        <span>{labels[value]}</span>
        <svg width="13" height="13" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <path d="m4 6 4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      {open && (
        <div className="composer-select-menu" id={listId} role="listbox" aria-label={label}>
          {values.map((item, index) => (
            <button
              type="button"
              id={`${listId}-${index}`}
              key={item}
              role="option"
              aria-selected={item === value}
              className={`composer-select-option${index === active ? " active" : ""}`}
              onPointerMove={() => setActive(index)}
              onClick={() => choose(item)}
            >
              <span>{labels[item]}</span>
              <span className="composer-select-check" aria-hidden="true">
                {item === value ? "✓" : ""}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function PermissionMenu({
  permissionMode,
  onPermissionChange,
  disabled = false,
}: PermissionMenuProps) {
  // 刻意不带「权限」前缀（用户 2026-09-22）：三个取值本身就是完整语义，
  // 前缀只是噪音；无障碍名仍由 `ComposerSelect` 的 aria-label「权限模式」兜住。
  return (
    <div className="chat-menus">
      <div className="chat-menu">
        <ComposerSelect
          label="权限模式"
          testId="permission-menu"
          value={permissionMode}
          values={PERMISSION_ORDER}
          labels={PERMISSION_LABELS}
          disabled={disabled}
          title={PERMISSION_HINTS[permissionMode]}
          onChange={onPermissionChange}
        />
      </div>
    </div>
  );
}

/** 归一化后端返回值：未知取值回落到默认（绝不显示空菜单）。 */
export function normalizePermission(value: unknown): PermissionMode {
  return typeof value === "string" && (PERMISSION_ORDER as string[]).includes(value)
    ? (value as PermissionMode)
    : DEFAULT_PERMISSION_MODE;
}

export function normalizeReasoning(value: unknown): ReasoningLevel {
  return typeof value === "string" && (REASONING_ORDER as string[]).includes(value)
    ? (value as ReasoningLevel)
    : DEFAULT_REASONING_LEVEL;
}

export default PermissionMenu;
