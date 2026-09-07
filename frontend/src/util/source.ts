import type { Source } from "../types";

/** 来源 chip 使用的紧凑标签；缺少向量分数时绝不伪造 0%。 */
export function sourceMatchLabel(source: Source): string {
  if (source.match_type === "expanded") return "章节补充";
  if (source.similarity === null || source.similarity === undefined) return "词法命中";
  return `${Math.round(source.similarity * 100)}%`;
}

/** 详情卡使用的完整说明。 */
export function sourceMatchDescription(source: Source): string {
  const label = sourceMatchLabel(source);
  return source.similarity === null || source.similarity === undefined ? label : `相关度 ${label}`;
}
