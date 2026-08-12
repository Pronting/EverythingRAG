import { useMemo } from "react";
import MarkdownIt from "markdown-it";
import DOMPurify from "dompurify";

/**
 * markdown-it + DOMPurify：任何 HTML 渲染必须经过净化（XSS 红线）。
 * html:false 让 markdown-it 先把原始 HTML 转义为文本，DOMPurify 再兜底净化。
 */
const md = new MarkdownIt({ html: false, linkify: true, breaks: true });

function sanitizeHtml(html: string): string {
  return DOMPurify.sanitize(html);
}

interface MarkdownContentProps {
  content: string;
}

/** 把文本渲染为净化后的 Markdown；默认文本化，禁止未净化 dangerouslySetInnerHTML。 */
export function MarkdownContent({ content }: MarkdownContentProps) {
  const html = useMemo(() => sanitizeHtml(md.render(content)), [content]);
  return <div className="markdown-body" dangerouslySetInnerHTML={{ __html: html }} />;
}
