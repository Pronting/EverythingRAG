import type { MemberRef, Source } from "../types";
import { basename } from "../util/path";
import {
  formatRerankScore,
  memberRelationLabel,
  sourceMatchDescription,
  vectorSimilarityLabel,
} from "../util/source";
import { MarkdownContent } from "./MarkdownContent";

interface CitationPanelProps {
  sources: Source[];
  index: number;
  onClose: () => void;
  onSelect: (index: number) => void;
  /**
   * 打开既有「语义块浏览」全文入口读取完整原文。
   *
   * 完整原文不自动进入本轮生成上下文（PRD §8.3）：本轮采用的只有 evidence_text，
   * 这里只是给用户一个按 block 追溯全文的既有入口。
   */
  onOpenFullText?: (blockId: string) => void;
}

/** 成员范围展示：半开区间 [start, end)，end 省略（null）表示整个块。 */
function memberRange(member: MemberRef): string {
  if (member.end === null || member.end === undefined) return "整块原文";
  return `第 ${member.start}–${member.end} 字`;
}

/** 组合成员可追溯：成员原文片段逐个列出，并能跳到既有全文入口（PRD §8.3）。 */
function MemberList({
  members,
  onOpenFullText,
}: {
  members: MemberRef[];
  onOpenFullText?: (blockId: string) => void;
}) {
  return (
    <ul className="citation-panel-members">
      {members.map((member, memberIndex) => (
        <li key={`${member.block_id}-${member.start}-${memberIndex}`} className="citation-panel-member">
          <span className="citation-panel-member-relation">
            {memberRelationLabel(member.relation)}
          </span>
          <span className="citation-panel-member-range">{memberRange(member)}</span>
          {onOpenFullText !== undefined && (
            <button
              type="button"
              className="citation-panel-member-open"
              onClick={() => onOpenFullText(member.block_id)}
            >
              读取完整原文
            </button>
          )}
        </li>
      ))}
    </ul>
  );
}

/** 右侧来源详情面板：展示选中 chunk 全文 + 元数据，支持上/下一个来源切换与关闭。 */
export function CitationPanel({
  sources,
  index,
  onClose,
  onSelect,
  onOpenFullText,
}: CitationPanelProps) {
  const source = sources[index];
  const isWeb = source.source_type === "web";
  const filtered = source.relevance_checked === true;
  const score = formatRerankScore(source.rerank_score);
  // 旧记录 / 未启用筛选的来源：原始数值只叫「向量相似度」（PRD §8.3）。
  const similarity = vectorSimilarityLabel(source);
  const members = filtered ? source.member_refs ?? [] : [];

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
        {source.image_url && /^(https?:\/\/|\/api\/knowledge\/images\/[a-f0-9]{16,64}$)/i.test(source.image_url) && (
          <a className="citation-panel-link" href={source.image_url} target="_blank" rel="noreferrer noopener">查看原图</a>
        )}
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
            {filtered && (
              <div className="citation-panel-relevance">
                <span className="citation-panel-badge">已筛选资料</span>
                {score !== null && <span className="citation-panel-score">排序分数 {score}</span>}
              </div>
            )}
            <div className="citation-panel-meta">
              {filtered ? similarity ?? "未记录向量相似度" : sourceMatchDescription(source)}
            </div>
            {filtered && members.length > 0 && (
              <div className="citation-panel-composition">
                <div className="citation-panel-composition-title">证据组成（组合成员可追溯）</div>
                <MemberList members={members} onOpenFullText={onOpenFullText} />
              </div>
            )}
            {filtered && members.length === 0 && onOpenFullText !== undefined && (
              <button
                type="button"
                className="citation-panel-member-open"
                onClick={() => onOpenFullText(source.block_id)}
              >
                读取完整原文
              </button>
            )}
          </>
        )}
        {source.chunk_type === "image_description" ? (
          <div className="citation-panel-evidence"><MarkdownContent content={source.text} /></div>
        ) : <p className="citation-panel-text">{source.text}</p>}
      </div>
    </aside>
  );
}
