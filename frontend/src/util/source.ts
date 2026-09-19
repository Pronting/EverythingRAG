import type { Source } from "../types";

/** 该来源是否经过相关性筛选（PRD §8.3：新结果展示「已筛选资料」）。 */
export function isRelevanceChecked(source: Source): boolean {
  return source.relevance_checked === true;
}

/**
 * 排序分数展示格式：固定三位小数（PRD §8.3 的「排序分数 0.812」）。
 *
 * 排序分数是服务返回的相关性排序值，不是正确率、也不是概率——前端只按数值原样展示，
 * 绝不换算成百分比或「可信度」。
 */
export function formatRerankScore(score: number | null | undefined): string | null {
  if (typeof score !== "number" || !Number.isFinite(score)) return null;
  return score.toFixed(3);
}

/** 来源 chip 使用的紧凑标签；缺少向量分数时绝不伪造 0%。 */
export function sourceMatchLabel(source: Source): string {
  if (isRelevanceChecked(source)) return "已筛选";
  if (source.match_type === "expanded") return "章节补充";
  if (source.match_type === "memory") return "记忆";
  if (source.similarity === null || source.similarity === undefined) return "词法命中";
  return `${Math.round(source.similarity * 100)}%`;
}

/** 详细面板使用的完整说明：重排来源用「排序分数」，不显示成正确率/概率。 */
export function sourceMatchDescription(source: Source): string {
  if (isRelevanceChecked(source)) {
    const score = formatRerankScore(source.rerank_score);
    return score === null ? "已筛选资料" : `已筛选资料 · 排序分数 ${score}`;
  }
  const label = sourceMatchLabel(source);
  return source.similarity === null || source.similarity === undefined
    ? label
    : `向量相似度 ${label}`;
}

/** 原始向量分数标签：旧记录的数值一旦展示，就明确叫「向量相似度」（PRD §8.3）。 */
export function vectorSimilarityLabel(source: Source): string | null {
  const similarity = source.similarity;
  if (typeof similarity !== "number" || !Number.isFinite(similarity)) return null;
  return `向量相似度 ${Math.round(similarity * 100)}%`;
}

/** 来源脚注标题：有任一来源经过筛选即标「已筛选资料」（历史记录同样按字段判定）。 */
export function sourceFooterLabel(sources: Source[]): string {
  return sources.some(isRelevanceChecked) ? "已筛选资料" : "来源";
}

/** 组合成员关联类型的中文说明（组合成员均可追溯，PRD §8.3）。 */
export function memberRelationLabel(relation: string): string {
  if (relation === "supplement") return "补充原文";
  if (relation === "image_text") return "图片文字";
  return "主要原文";
}
