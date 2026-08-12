import type { Source } from "../types";

interface SourceCardProps {
  source: Source;
}

/** 来源块卡片：文件 + 标题路径 + 相似度 + 文本片段。 */
export function SourceCard({ source }: SourceCardProps) {
  const similarityPercent = Math.round(source.similarity * 100);
  return (
    <article className="source-card">
      <div className="source-card-meta">
        <span className="source-file" title={source.source_file}>
          {source.source_file}
        </span>
        {source.heading_path !== null && source.heading_path !== "" && (
          <span className="source-heading">{source.heading_path}</span>
        )}
        <span className="source-similarity">相关度 {similarityPercent}%</span>
      </div>
      <p className="source-text">{source.text}</p>
    </article>
  );
}
