import type { Source } from "../types";
import { basename } from "../util/path";

interface CitationTooltipProps {
  source: Source;
  anchor: DOMRect;
}

const TOOLTIP_WIDTH = 320;
const GAP = 8;
const ESTIMATED_HEIGHT = 200;

/** 悬停引用序号的悬浮卡片：文件名 + 标题路径 + 相关度 + 截断摘要，锚定在序号附近。 */
export function CitationTooltip({ source, anchor }: CitationTooltipProps) {
  const viewportWidth = window.innerWidth;
  const viewportHeight = window.innerHeight;

  let left = anchor.left + anchor.width / 2 - TOOLTIP_WIDTH / 2;
  left = Math.max(8, Math.min(left, viewportWidth - TOOLTIP_WIDTH - 8));

  let top = anchor.bottom + GAP;
  if (top + ESTIMATED_HEIGHT > viewportHeight - 8) {
    top = anchor.top - GAP - ESTIMATED_HEIGHT;
  }
  top = Math.max(8, top);

  const isWeb = source.source_type === "web";
  const percent = Math.round(source.similarity * 100);

  return (
    <div className="citation-tooltip" role="tooltip" style={{ left, top, width: TOOLTIP_WIDTH }}>
      <div className="citation-tooltip-file" title={source.source_file}>
        {isWeb ? source.source_file : basename(source.source_file)}
      </div>
      {!isWeb && source.heading_path !== null && source.heading_path !== "" && (
        <div className="citation-tooltip-heading">{source.heading_path}</div>
      )}
      {isWeb && source.url !== null && source.url !== undefined && source.url !== "" && (
        <div className="citation-tooltip-url">{source.url}</div>
      )}
      {!isWeb && <div className="citation-tooltip-meta">相关度 {percent}%</div>}
      <p className="citation-tooltip-text">{source.text}</p>
    </div>
  );
}
