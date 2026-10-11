/**
 * 侧边栏会话列表的分页规则（**纯函数**，便于单测；不碰 React / localStorage）。
 *
 * 规则（2026-09-20 用户指定，取代「已展开条数会长期记忆」的旧行为）：
 *
 * 1. 每个工作区默认只显示 {@link DEFAULT_VISIBLE} 条，点「展开显示」每次追加
 *    {@link VISIBLE_STEP} 条；
 * 2. **显示到底之后给出「收起」**，把该工作区恢复到默认条数 —— 否则用户每点一次
 *    「展开显示」列表就永久变长，一个工作区会越堆越高；
 * 3. **折叠工作区**（以及**重启应用**）都回到默认条数。因此展开条数**只存在于本次
 *    会话的内存里**，不再写 localStorage（重启即归零是需求，不是遗漏）。
 */

/** 默认显示条数（同时是「收起」的目标值）。 */
export const DEFAULT_VISIBLE = 5;

/** 每次「展开显示」追加的条数。 */
export const VISIBLE_STEP = 10;

/** 列表底部按钮的形态。 */
export type PagingFooter = "expand" | "collapse" | "none";

/** 读取某个工作区的显示上限（缺省 / 非法值一律回到默认）。 */
export function visibleLimitFor(
  counts: Readonly<Record<string, number>>,
  workspaceId: string,
): number {
  const value = counts[workspaceId];
  if (typeof value !== "number" || !Number.isFinite(value) || value <= DEFAULT_VISIBLE) {
    return DEFAULT_VISIBLE;
  }
  return Math.floor(value);
}

/**
 * 实际显示条数。
 *
 * `activeIndex` 是当前打开的会话在该工作区里的下标：它必须可见（PRD 14.1 的既有约定），
 * 因此排在显示窗口之外时把窗口扩到它为止 —— 这也是「显示到底」判定要用**显示条数**而
 * 不是「显示上限」的原因。
 */
export function shownCount(total: number, limit: number, activeIndex: number): number {
  const available = Math.max(0, Math.floor(total));
  const wanted = activeIndex >= 0 && activeIndex >= limit ? activeIndex + 1 : limit;
  return Math.max(0, Math.min(available, wanted));
}

/** 还剩多少条没显示。 */
export function remainingCount(total: number, shown: number): number {
  return Math.max(0, Math.floor(total) - Math.max(0, Math.floor(shown)));
}

/** 底部按钮形态：还有更多 → 展开；已到底且确实展开过 → 收起；否则不渲染。 */
export function pagingFooter(total: number, shown: number, limit: number): PagingFooter {
  if (remainingCount(total, shown) > 0) return "expand";
  return limit > DEFAULT_VISIBLE ? "collapse" : "none";
}

/** 「展开显示」后的新状态（不可变更新，不落盘）。 */
export function expandVisible(
  counts: Readonly<Record<string, number>>,
  workspaceId: string,
): Record<string, number> {
  return { ...counts, [workspaceId]: visibleLimitFor(counts, workspaceId) + VISIBLE_STEP };
}

/** 「收起」/ 折叠工作区后的新状态：该工作区回到默认条数（删键即默认值）。 */
export function resetVisible(
  counts: Readonly<Record<string, number>>,
  workspaceId: string,
): Record<string, number> {
  if (!(workspaceId in counts)) return counts as Record<string, number>;
  const next = { ...counts };
  delete next[workspaceId];
  return next;
}
