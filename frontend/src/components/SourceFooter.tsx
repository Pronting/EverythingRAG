import type { RetrievalMeta, Source } from "../types";
import { basename } from "../util/path";
import { sourceFooterLabel, sourceMatchLabel } from "../util/source";

interface SourceFooterProps {
  sources: Source[];
  /** 本轮筛选状态（PRD §8.2）；历史记录为 null，此时按来源字段自行判定。 */
  retrieval?: RetrievalMeta | null;
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
export function SourceFooter({ sources, retrieval, onSelect }: SourceFooterProps) {
  // 本轮筛选成功时脚注标题用「已筛选资料」；历史记录靠来源自带的 relevance_checked 判定。
  const label = retrieval?.status === "filtered" ? "已筛选资料" : sourceFooterLabel(sources);
  return (
    <div className="source-footer">
      <span className="source-footer-label">{label}</span>
      {sources.map((source, index) => (
        <button
          key={`${source.source_id ?? "legacy"}-${source.block_id}`}
          type="button"
          className={`source-chip${source.source_type === "web" ? " source-chip-web" : ""}${source.relevance_checked ? " source-chip-filtered" : ""}`}
          title={source.source_type === "web" ? source.url ?? source.source_file : source.source_file}
          onClick={() => onSelect(index)}
        >
          {source.source_type === "web" ? (
            <WebChip source={source} />
          ) : (
            <>
              <span className="source-chip-id">{source.source_id}</span>
              <span className="source-chip-file">{basename(source.source_file)}</span>
              <span className="source-chip-sim">{sourceMatchLabel(source)}</span>
            </>
          )}
        </button>
      ))}
    </div>
  );
}
