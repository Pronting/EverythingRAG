import type { Source } from "../types";
import { basename } from "../util/path";

interface SourceFooterProps {
  sources: Source[];
  onSelect: (index: number) => void;
}

/** 联网搜索结果 chip：地球图标 + 标题，点击打开右侧详情（正文里带可跳转链接）。 */
function WebChip({ source }: { source: Source }) {
  return (
    <>
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.6" />
        <path
          d="M3 12h18M12 3c2.5 2.6 4 5.6 4 9s-1.5 6.4-4 9c-2.5-2.6-4-5.6-4-9s1.5-6.4 4-9Z"
          stroke="currentColor"
          strokeWidth="1.6"
        />
      </svg>
      <span className="source-chip-file" title={source.url ?? source.source_file}>
        {source.source_file || source.url}
      </span>
    </>
  );
}

/** 极简来源脚注：一行文件名 chips（点击打开右侧详情），替代原来的 chunk 长文列表。 */
export function SourceFooter({ sources, onSelect }: SourceFooterProps) {
  return (
    <div className="source-footer">
      <span className="source-footer-label">来源</span>
      {sources.map((source, index) => (
        <button
          key={source.block_id}
          type="button"
          className={`source-chip${source.source_type === "web" ? " source-chip-web" : ""}`}
          title={source.source_type === "web" ? source.url ?? source.source_file : source.source_file}
          onClick={() => onSelect(index)}
        >
          {source.source_type === "web" ? (
            <WebChip source={source} />
          ) : (
            <>
              <span className="source-chip-file">{basename(source.source_file)}</span>
              <span className="source-chip-sim">{Math.round(source.similarity * 100)}%</span>
            </>
          )}
        </button>
      ))}
    </div>
  );
}
