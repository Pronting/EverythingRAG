import { Fragment, useEffect, useMemo, useRef, useState } from "react";
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

function SearchableSource({ text }: { text: string }) {
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const current = useRef<HTMLElement>(null);
  const matches = useMemo(() => {
    if (!query) return [];
    const pattern = new RegExp(query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "giu");
    return Array.from(text.matchAll(pattern), match => ({ start: match.index, text: match[0] }));
  }, [text, query]);
  useEffect(() => {
    current.current?.scrollIntoView({ block: "nearest" });
  }, [matches, active]);
  const move = (direction: number) => {
    if (matches.length) setActive(value => (value + direction + matches.length) % matches.length);
  };
  return <>
    {text.length > 2000 && <div className="citation-search" role="search" aria-label="查找来源文字">
      <input
        aria-label="查找来源文字"
        placeholder="查找来源中的文字"
        value={query}
        onChange={event => { setQuery(event.target.value); setActive(0); }}
        onKeyDown={event => {
          if (event.key === "Enter") { event.preventDefault(); move(event.shiftKey ? -1 : 1); }
          if (event.key === "Escape") { setQuery(""); setActive(0); }
        }}
      />
      <span role="status">{query ? `${matches.length ? active + 1 : 0} / ${matches.length}` : ""}</span>
      <button type="button" aria-label="上一个匹配" disabled={!matches.length} onClick={() => move(-1)}>↑</button>
      <button type="button" aria-label="下一个匹配" disabled={!matches.length} onClick={() => move(1)}>↓</button>
    </div>}
    <p className="citation-panel-text">
      {matches.map((match, index) => <Fragment key={match.start}>
        {text.slice(index ? matches[index - 1].start + matches[index - 1].text.length : 0, match.start)}
        <mark ref={index === active ? current : undefined} className={index === active ? "citation-match-active" : undefined}>{match.text}</mark>
      </Fragment>)}
      {text.slice(matches.length ? matches[matches.length - 1].start + matches[matches.length - 1].text.length : 0)}
    </p>
  </>;
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
  const displayTitle = source.document_title || (isWeb ? source.source_file : basename(source.source_file));

  // 诊断属性（PRD 12.5：实现参数只进诊断，不占日常可见文案）：本轮筛选的校准档案 id 与
  // 证据视图载荷哈希自 t5 起随引用到达界面，这里挂成 data-* 便于排障与 E2E 核对
  // 「字段真的到了来源层」，而不是把它们写进用户可见的正文/元数据行。
  return (
    <aside
      className="citation-panel"
      aria-label="来源详情"
      data-profile-id={source.profile_id ?? ""}
      data-evidence-hash={source.evidence_hash ?? ""}
    >
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
            <div className="citation-panel-file">{displayTitle}</div>
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
            {source.heading_path !== null && source.heading_path !== "" && (
              <div className="citation-panel-heading">{source.heading_path}</div>
            )}
          </>
        ) : (
          <>
            <div className="citation-panel-file">{displayTitle}</div>
            {source.source_file !== displayTitle && !/^[a-f0-9]{24,}(?:#|$)/i.test(source.source_file) && (
              <div className="citation-panel-path" title={source.source_file}>
                {source.source_file}
              </div>
            )}
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
              {/* 分数与来源类型分别依据记录中的字段展示。 */}
              {filtered ? (similarity ?? (source.match_type === "lexical" ? "词法命中" : "未提供向量分数")) : sourceMatchDescription(source)}
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
        ) : <SearchableSource key={`${index}-${source.block_id}`} text={source.text} />}
      </div>
    </aside>
  );
}
