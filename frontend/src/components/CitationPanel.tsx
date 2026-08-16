import type { Source } from "../types";
import { basename } from "../util/path";

interface CitationPanelProps {
  sources: Source[];
  index: number;
  onClose: () => void;
  onSelect: (index: number) => void;
}

/** 右侧来源详情面板：展示选中 chunk 全文 + 元数据，支持上/下一个来源切换与关闭。 */
export function CitationPanel({ sources, index, onClose, onSelect }: CitationPanelProps) {
  const source = sources[index];
  const isWeb = source.source_type === "web";
  const percent = Math.round(source.similarity * 100);

  return (
    <aside className="citation-panel" aria-label="来源详情">
      <header className="citation-panel-header">
        <span className="citation-panel-count">
          来源 {index + 1} / {sources.length}
        </span>
        <div className="citation-panel-nav">
          <button
            type="button"
            disabled={index <= 0}
            onClick={() => onSelect(index - 1)}
            aria-label="上一个来源"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
              <path d="M3 9 7 5l4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
          <button
            type="button"
            disabled={index >= sources.length - 1}
            onClick={() => onSelect(index + 1)}
            aria-label="下一个来源"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
              <path d="M3 5l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
          <button type="button" onClick={onClose} aria-label="关闭来源详情">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
              <path d="M3.5 3.5 10.5 10.5M10.5 3.5 3.5 10.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </button>
        </div>
      </header>
      <div className="citation-panel-body">
        {isWeb ? (
          <>
            <div className="citation-panel-file">{source.source_file}</div>
            {source.url !== null && source.url !== undefined && source.url !== "" && (
              <a
                className="citation-panel-link"
                href={source.url}
                target="_blank"
                rel="noreferrer noopener"
              >
                {source.url}
              </a>
            )}
          </>
        ) : (
          <>
            <div className="citation-panel-file">{basename(source.source_file)}</div>
            <div className="citation-panel-path" title={source.source_file}>
              {source.source_file}
            </div>
            {source.heading_path !== null && source.heading_path !== "" && (
              <div className="citation-panel-heading">{source.heading_path}</div>
            )}
            <div className="citation-panel-meta">相关度 {percent}%</div>
          </>
        )}
        <p className="citation-panel-text">{source.text}</p>
      </div>
    </aside>
  );
}
