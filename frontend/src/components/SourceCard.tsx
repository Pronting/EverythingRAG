import { useState } from "react";
import type { Source } from "../types";

interface SourceCardProps {
  source: Source;
}

/** 来源块卡片：默认折叠只显示文档名；点击展开显示 chunk 详细段落。 */
export function SourceCard({ source }: SourceCardProps) {
  const [expanded, setExpanded] = useState(false);
  const similarityPercent = Math.round(source.similarity * 100);

  return (
    <article className={`source-card${expanded ? " expanded" : ""}`}>
      <button
        type="button"
        className="source-card-header"
        onClick={() => setExpanded((value) => !value)}
        aria-expanded={expanded}
        aria-label={`${expanded ? "折叠" : "展开"}来源 ${source.source_file}`}
      >
        <span className="source-file" title={source.source_file}>
          {source.source_file}
        </span>
        {source.heading_path !== null && source.heading_path !== "" && (
          <span className="source-heading">{source.heading_path}</span>
        )}
        <span className="source-similarity">相关度 {similarityPercent}%</span>
        <span className={`source-chevron${expanded ? " open" : ""}`} aria-hidden="true">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
            <path d="M2.5 4.5 6 8l3.5-3.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </span>
      </button>
      {expanded && <p className="source-text">{source.text}</p>}
    </article>
  );
}
