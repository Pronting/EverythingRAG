import type { ThemeValue } from "./types";

/** 主题在 localStorage 中的键（仅用于刷新时的反闪烁快照；后端 config.json 为准）。 */
export const THEME_STORAGE_KEY = "everything-rag-theme";

/** 读取已存主题（非法值回退浅色）。 */
export function readStoredTheme(): ThemeValue {
  try {
    return localStorage.getItem(THEME_STORAGE_KEY) === "dark" ? "dark" : "light";
  } catch {
    return "light";
  }
}

/** 持久化主题到 localStorage（失败静默）。 */
export function persistTheme(theme: ThemeValue): void {
  try {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    /* 忽略：存储不可用时仅失去刷新反闪烁能力 */
  }
}
