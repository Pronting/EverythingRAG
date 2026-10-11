/** 过程区域只累计实际执行区间；等待用户处理和失败后的暂停时间不计入用时。 */

export interface ActivityInterval {
  startedAt: string;
  endedAt: string | null;
}

export function activitySeconds(
  intervals: readonly ActivityInterval[], liveEndMs?: number | null,
): number | null {
  let totalMs = 0;
  let measured = false;
  for (const interval of intervals) {
    const start = Date.parse(interval.startedAt);
    const end = interval.endedAt === null ? liveEndMs : Date.parse(interval.endedAt);
    if (!Number.isFinite(start) || typeof end !== "number" || !Number.isFinite(end)) continue;
    totalMs += Math.max(0, end - start);
    measured = true;
  }
  return measured ? Math.round(totalMs / 1000) : null;
}

export function formatActivityDuration(
  intervals: readonly ActivityInterval[], liveEndMs?: number | null,
): string | null {
  const seconds = activitySeconds(intervals, liveEndMs);
  return seconds === null ? null : formatElapsed(seconds);
}

/** 解析一批事件时间戳，忽略解析不出来的值（不用 0 兜底，避免伪造出巨大的跨度）。 */
export function parseEventTimes(values: readonly string[]): number[] {
  const times: number[] = [];
  for (const value of values) {
    const parsed = Date.parse(value);
    if (Number.isFinite(parsed)) times.push(parsed);
  }
  return times;
}

/**
 * 计算耗时秒数；无法诚实算出时返回 ``null``（**不返回 0**，界面据此不显示，而不是显示
 * 一个假的「0 秒」）。
 *
 * - `times`：事件时间戳（毫秒）。
 * - `liveEndMs`：run 仍在进行时的「现在」（毫秒）。给了它就用它当右端点（每秒走表），
 *   否则用最后一个事件时间（= 已结束的 run 的真实耗时）。
 */
export function elapsedSeconds(
  times: readonly number[],
  liveEndMs?: number | null,
): number | null {
  if (times.length === 0) return null;
  const start = Math.min(...times);
  const last = Math.max(...times);
  const end = typeof liveEndMs === "number" && Number.isFinite(liveEndMs)
    ? Math.max(last, liveEndMs)
    : last;
  if (times.length < 2 && typeof liveEndMs !== "number") return null;
  return Math.max(0, Math.round((end - start) / 1000));
}

/** 秒数 → 中文可读文案（60 秒内按秒，60 分钟内按「N分钟[M秒]」，再往上按小时）。 */
export function formatElapsed(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  if (total < 60) return `${total}秒`;
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  if (minutes < 60) return rest === 0 ? `${minutes}分钟` : `${minutes}分钟${rest}秒`;
  const hours = Math.floor(minutes / 60);
  return `${hours}小时${minutes % 60}分钟`;
}

/** 一步到位：事件时间戳（+ 可选的活的结束时刻）→ 展示文案；算不出返回 ``null``。 */
export function formatDuration(
  values: readonly string[],
  liveEndMs?: number | null,
): string | null {
  const seconds = elapsedSeconds(parseEventTimes(values), liveEndMs);
  return seconds === null ? null : formatElapsed(seconds);
}
