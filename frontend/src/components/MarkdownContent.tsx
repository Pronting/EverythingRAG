import { memo, useMemo } from "react";
import type { KeyboardEvent as ReactKeyboardEvent, MouseEvent as ReactMouseEvent } from "react";
import MarkdownIt from "markdown-it";
import DOMPurify from "dompurify";
import type { Source } from "../types";

interface MarkdownContentProps {
  content: string;
  /** 该条消息的来源；新答案用稳定 [S1]/[W1]，旧会话兼容 [N]。 */
  sources?: Source[] | null;
  /** 点击正文引用序号：把整组来源与当前下标交给上层，打开右侧详情面板。 */
  onCitationClick?: (sources: Source[], index: number) => void;
  /** 悬停正文引用序号：source=null 表示移出引用。 */
  onCitationHover?: (source: Source | null, rect: DOMRect | null) => void;
}

/**
 * markdown-it + DOMPurify：任何 HTML 渲染必须经过净化（XSS 红线）。
 * html:false 让 markdown-it 先把原始 HTML 转义为文本，DOMPurify 再兜底净化。
 * 额外注入 citation inline rule：把白名单 source_id 渲染成可交互徽章。
 */
function renderMarkdown(content: string, sources: Source[] | null): string {
  const md = new MarkdownIt({ html: false, linkify: true, breaks: true });
  // 放在 link 规则之前；代码块/行内代码已由更早规则消费，不会误伤。
  md.inline.ruler.before("link", "citation", (state, silent) => {
    const start = state.pos;
    if (state.src.charCodeAt(start) !== 0x5b /* '[' */) return false;
    const match = /^\[((?:S|W)\d+|\d+)\]/.exec(state.src.slice(start));
    if (match === null) return false;
    // 仅当后面紧跟 '(' 时是标准 Markdown 链接（[3](url)），交给 link 规则。
    // 相邻引用（如 [4][6]）要逐个转成徽章，不能当作链接引用 [text][ref] 丢弃。
    const next = state.src.charCodeAt(start + match[0].length);
    if (next === 0x28 /* '(' */) return false;
    if (silent) return true;
    const token = state.push("citation", "", 0);
    token.meta = { sourceId: match[1] };
    state.pos = start + match[0].length;
    return true;
  });

  md.renderer.rules.citation = (tokens, idx) => {
    const sourceId: string = tokens[idx].meta.sourceId;
    const legacyIndex = /^\d+$/.test(sourceId) ? Number(sourceId) - 1 : -1;
    const index = legacyIndex >= 0
      ? legacyIndex
      : (sources ?? []).findIndex((source) => source.source_id === sourceId);
    if (sources?.[index]) {
      return `<span class="citation" data-cite-index="${index}" role="button" tabindex="0">${sourceId}</span>`;
    }
    // 无效序号（模型虚构/超出召回范围）：直接移除，避免残留无交互价值的纯文本编号
    return "";
  };

  return DOMPurify.sanitize(md.render(content));
}

/** 把文本渲染为净化后的 Markdown；有效引用转为可交互徽章。 */
export const MarkdownContent = memo(function MarkdownContent({
  content,
  sources,
  onCitationClick,
  onCitationHover,
}: MarkdownContentProps) {
  const html = useMemo(() => renderMarkdown(content, sources ?? null), [content, sources]);

  const resolveCitation = (
    target: EventTarget | null,
  ): { source: Source; el: HTMLElement; index: number } | null => {
    if (!sources || sources.length === 0 || !(target instanceof HTMLElement)) return null;
    const el = target.closest<HTMLElement>(".citation");
    if (el === null) return null;
    const index = Number(el.dataset.citeIndex);
    const source = sources[index];
    return source ? { source, el, index } : null;
  };

  const handleMouseOver = (event: ReactMouseEvent<HTMLDivElement>): void => {
    if (!onCitationHover) return;
    const hit = resolveCitation(event.target);
    onCitationHover(hit ? hit.source : null, hit ? hit.el.getBoundingClientRect() : null);
  };

  const handleMouseOut = (event: ReactMouseEvent<HTMLDivElement>): void => {
    if (!onCitationHover) return;
    const el =
      event.target instanceof HTMLElement ? event.target.closest<HTMLElement>(".citation") : null;
    const related = event.relatedTarget;
    const stillInside = el !== null && related instanceof Node && el.contains(related);
    if (el !== null && !stillInside) onCitationHover(null, null);
  };

  const handleClick = (event: ReactMouseEvent<HTMLDivElement>): void => {
    const hit = resolveCitation(event.target);
    if (hit && onCitationClick && sources) onCitationClick(sources, hit.index);
  };

  const handleKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>): void => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const hit = resolveCitation(event.target);
    if (hit && onCitationClick && sources) {
      event.preventDefault();
      onCitationClick(sources, hit.index);
    }
  };

  return (
    <div
      className="markdown-body"
      dangerouslySetInnerHTML={{ __html: html }}
      onMouseOver={handleMouseOver}
      onMouseOut={handleMouseOut}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
    />
  );
});
