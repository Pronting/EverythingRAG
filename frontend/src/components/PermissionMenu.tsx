import type { PermissionMode, ReasoningLevel } from "../types";

/**
 * 权限菜单与思考菜单（PRD 14.1）。
 *
 * 权限菜单：请求批准 / 自动批准 / 完全访问权限，**默认请求批准**。
 * 思考菜单：关闭 / 轻度 / 深度 / 最高，**默认深度思考**。
 *
 * 两个菜单都展示**当前会话的实际设置**（值由父级从后端读回，不在前端另存一份，
 * 否则界面会与真正生效的设置分叉）。
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

/** PRD 18.1 默认值：请求批准 + 深度思考。 */
export const DEFAULT_PERMISSION_MODE: PermissionMode = "ask";
export const DEFAULT_REASONING_LEVEL: ReasoningLevel = "high";

const PERMISSION_ORDER: PermissionMode[] = ["ask", "auto", "full"];
const REASONING_ORDER: ReasoningLevel[] = ["off", "low", "high", "max"];

export interface PermissionMenuProps {
  permissionMode: PermissionMode;
  reasoningLevel: ReasoningLevel;
  onPermissionChange: (mode: PermissionMode) => void;
  onReasoningChange: (level: ReasoningLevel) => void;
  disabled?: boolean;
}

export function PermissionMenu({
  permissionMode,
  reasoningLevel,
  onPermissionChange,
  onReasoningChange,
  disabled = false,
}: PermissionMenuProps) {
  return (
    <div className="chat-menus">
      <label className="chat-menu" title={PERMISSION_HINTS[permissionMode]}>
        <span className="chat-menu-label">权限</span>
        <select
          className="chat-menu-select"
          data-testid="permission-menu"
          aria-label="权限模式"
          value={permissionMode}
          disabled={disabled}
          onChange={(event) => onPermissionChange(event.target.value as PermissionMode)}
        >
          {PERMISSION_ORDER.map((mode) => (
            <option key={mode} value={mode}>
              {PERMISSION_LABELS[mode]}
            </option>
          ))}
        </select>
      </label>
      <label className="chat-menu">
        <span className="chat-menu-label">思考</span>
        <select
          className="chat-menu-select"
          data-testid="reasoning-menu"
          aria-label="思考深度"
          value={reasoningLevel}
          disabled={disabled}
          onChange={(event) => onReasoningChange(event.target.value as ReasoningLevel)}
        >
          {REASONING_ORDER.map((level) => (
            <option key={level} value={level}>
              {REASONING_LABELS[level]}
            </option>
          ))}
        </select>
      </label>
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
