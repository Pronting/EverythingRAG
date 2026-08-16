import { useEffect, useMemo, useState } from "react";
import { listBlocks } from "../api/blocksApi";
import type { BlockInfo } from "../types";

/** 分类：全部 / 文本 / 图片。 */
type Category = "all" | "text" | "image";

/** 每页条数。 */
const PAGE_SIZE = 20;

/** 从绝对路径提取文件名 + 直接父目录名（用于展示「哪个文件 / 哪个子目录」）。 */
function pathParts(sourceFile: string): { name: string; dir: string | null } {
  const parts = sourceFile.split(/[\\/]/).filter(Boolean);
  const name = parts[parts.length - 1] ?? sourceFile;
  const dir = parts.length >= 2 ? parts[parts.length - 2] : null;
  return { name, dir };
}

/** 是否图片描述块。 */
function isImageBlock(block: BlockInfo): boolean {
  return block.chunk_type === "image_description" || block.source_type === "image_description";
}

/** 折叠为单行摘要（超出截断）。 */
function summary(text: string, max = 120): string {
  const oneLine = text.replace(/\s+/g, " ").trim();
  return oneLine.length > max ? `${oneLine.slice(0, max)}…` : oneLine;
}

/** 语义块浏览弹窗：分类（文本/图片）+ 搜索（标题/目录/关键词）+ 分页。 */
export default function BlockBrowser({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [blocks, setBlocks] = useState<BlockInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState<Category>("all");
  const [page, setPage] = useState(1);
  // 点击行展开的块详情
  const [selected, setSelected] = useState<BlockInfo | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    setBlocks([]);
    setSearch("");
    setCategory("all");
    setPage(1);
    listBlocks()
      .then((resp) => {
        if (!cancelled) setBlocks(resp.blocks);
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    let result = blocks;
    if (category === "text") result = result.filter((b) => !isImageBlock(b));
    else if (category === "image") result = result.filter((b) => isImageBlock(b));
    if (q !== "") {
      result = result.filter(
        (b) =>
          (b.heading_path ?? "").toLowerCase().includes(q) ||
          b.source_file.toLowerCase().includes(q) ||
          b.text.toLowerCase().includes(q),
      );
    }
    return result;
  }, [blocks, search, category]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages);
  const pageBlocks = filtered.slice((safePage - 1) * PAGE_SIZE, safePage * PAGE_SIZE);

  const copyText = async (text: string): Promise<void> => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      /* 复制失败静默 */
    }
  };

  if (!open) return null;

  return (
    <div className="block-browser-overlay" role="dialog" aria-modal="true" aria-label="语义块浏览">
      <div className="block-browser">
        <div className="block-browser-header">
          <h2>语义块</h2>
          <span className="block-browser-total">
            {filtered.length} / {blocks.length} 块
          </span>
          <button type="button" className="settings-close" onClick={onClose} aria-label="关闭语义块浏览">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <path d="M3 3l10 10M13 3L3 13" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        <div className="block-browser-toolbar">
          <div className="block-browser-categories" role="tablist" aria-label="分类">
            {(["all", "text", "image"] as Category[]).map((c) => (
              <button
                key={c}
                type="button"
                role="tab"
                aria-selected={category === c}
                className={`block-cat${category === c ? " active" : ""}`}
                onClick={() => {
                  setCategory(c);
                  setPage(1);
                }}
              >
                {c === "all" ? "全部" : c === "text" ? "文本" : "图片"}
              </button>
            ))}
          </div>
          <input
            className="block-browser-search"
            type="search"
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(1);
            }}
            placeholder="搜索标题 / 目录 / 关键词…"
            aria-label="搜索语义块"
          />
        </div>

        <div className="block-browser-body">
          {loading && <p className="block-browser-hint">加载中…</p>}
          {error !== null && <p className="block-browser-hint error">{error}</p>}
          {!loading && error === null && pageBlocks.length === 0 && (
            <p className="block-browser-hint">无匹配的语义块</p>
          )}
          {!loading &&
            error === null &&
            pageBlocks.map((block) => {
              const { name, dir } = pathParts(block.source_file);
              const image = isImageBlock(block);
              return (
                <div
                  key={block.block_id}
                  className="block-item"
                  role="button"
                  tabIndex={0}
                  onClick={() => setSelected(block)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      setSelected(block);
                    }
                  }}
                >
                  <div className="block-item-head">
                    <span className={`block-badge${image ? " image" : " text"}`}>
                      {image ? "图片" : "文本"}
                    </span>
                    <span className="block-item-summary">{summary(block.text)}</span>
                  </div>
                  <div className="block-item-meta">
                    <span className="block-item-file">{name}</span>
                    {dir !== null && <span className="block-item-dir">{dir}</span>}
                    {block.heading_path !== null && (
                      <span className="block-item-heading">{block.heading_path}</span>
                    )}
                  </div>
                </div>
              );
            })}
        </div>

        <div className="block-browser-footer">
          <button
            type="button"
            className="block-pager"
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={safePage <= 1}
          >
            上一页
          </button>
          <span className="block-pager-info">
            第 {safePage} / {totalPages} 页
          </span>
          <button
            type="button"
            className="block-pager"
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={safePage >= totalPages}
          >
            下一页
          </button>
        </div>
      </div>

      {selected !== null && (
        <div className="block-detail-overlay" onClick={() => setSelected(null)}>
          <div
            className="block-detail"
            role="dialog"
            aria-modal="true"
            aria-label="块详情"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="block-detail-header">
              <span className={`block-badge${isImageBlock(selected) ? " image" : " text"}`}>
                {isImageBlock(selected) ? "图片" : "文本"}
              </span>
              <h3>块详情</h3>
              <button
                type="button"
                className="settings-close"
                onClick={() => setSelected(null)}
                aria-label="关闭详情"
              >
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                  <path d="M3 3l10 10M13 3L3 13" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                </svg>
              </button>
            </div>
            <div className="block-detail-body">
              <div className="block-detail-text">{selected.text}</div>
              <dl className="block-detail-meta">
                <div className="block-detail-row">
                  <dt>来源文件</dt>
                  <dd>{selected.source_file}</dd>
                </div>
                <div className="block-detail-row">
                  <dt>标题路径</dt>
                  <dd>{selected.heading_path ?? "—"}</dd>
                </div>
                <div className="block-detail-row">
                  <dt>块 ID</dt>
                  <dd>{selected.block_id}</dd>
                </div>
                <div className="block-detail-row">
                  <dt>来源类型</dt>
                  <dd>{selected.source_type}</dd>
                </div>
              </dl>
            </div>
            <div className="block-detail-footer">
              <button
                type="button"
                className="block-detail-copy"
                onClick={() => void copyText(selected.text)}
              >
                {copied ? "已复制" : "复制全文"}
              </button>
              <button type="button" className="block-pager" onClick={() => setSelected(null)}>
                关闭
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
