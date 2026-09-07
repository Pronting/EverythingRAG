import { useRef, useState } from "react";
import type { StatusResponse, ToastType } from "../types";
import ImportPanel from "./ImportPanel";
import BlockBrowser from "./BlockBrowser";

export default function KnowledgeSettings({ status, onChanged, onNotify }: {
  status: StatusResponse | null;
  onChanged: () => void;
  onNotify: (type: ToastType, message: string) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [blocksOpen, setBlocksOpen] = useState(false);
  const [revision, setRevision] = useState(0);
  const dialogRef = useRef<HTMLDialogElement>(null);
  const knowledge = status?.knowledge;
  const images = knowledge?.image_tasks;

  const removeKnowledge = async () => {
    if (deleting) return;
    setDeleting(true);
    setError(null);
    try {
      const response = await fetch("/api/knowledge", { method: "DELETE" });
      if (!response.ok) {
        if (response.status === 404 || response.status === 405) {
          throw new Error("当前后端版本不支持删除知识库，请关闭旧服务并重新启动 Everything RAG，再刷新页面");
        }
        const body = await response.json().catch(() => null);
        throw new Error(typeof body?.detail === "string" ? body.detail : "删除失败，请重试");
      }
      setRevision((value) => value + 1);
      setBlocksOpen(false);
      onChanged();
      dialogRef.current?.close();
      onNotify("success", "知识库已删除，可重新导入文档");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "无法连接本地服务，请稍后重试");
    } finally { setDeleting(false); }
  };

  return <>
    <h3>文档与索引</h3>
    <div className="knowledge-summary">
      <div><strong>{knowledge?.file_count ?? "—"}</strong><span>篇文档</span></div>
      <div><strong>{knowledge?.chunk_count ?? "—"}</strong><span>个语义块</span></div>
      <button type="button" className="import-source-btn" disabled={!knowledge?.chunk_count || deleting} onClick={() => setBlocksOpen(true)}>浏览内容</button>
    </div>
    {images && images.total > 0 && <p className="settings-hint">图片识别 {images.done} / {images.total}{images.pending > 0 ? ` · 处理中 ${images.pending}` : ""}{images.failed > 0 ? ` · 失败 ${images.failed}` : ""}</p>}
    <ImportPanel key={revision} onImported={onChanged} onBusyChange={setBusy} disabled={deleting} />
    <div className="knowledge-danger-zone">
      <div><h3>删除知识库</h3></div>
      <button type="button" className="knowledge-delete-btn" disabled={busy || deleting || (images?.pending ?? 0) > 0} onClick={() => { setError(null); dialogRef.current?.showModal(); }}>删除知识库</button>
    </div>
    <dialog ref={dialogRef} className="knowledge-confirm" aria-labelledby="knowledge-confirm-title" aria-describedby="knowledge-confirm-description" onCancel={(event) => { if (deleting) event.preventDefault(); }} onKeyDown={(event) => event.stopPropagation()}>
      <h3 id="knowledge-confirm-title">确认删除知识库？</h3>
      <p id="knowledge-confirm-description">将清空当前知识库的全部文档索引、图片识别缓存和同步记录，此操作无法撤销。原始文件、上传副本和历史对话会保留；其他模型的旧索引不受影响。</p>
      {error && <p className="import-error" role="alert">{error}</p>}
      <div className="knowledge-confirm-actions">
        <button type="button" className="import-source-btn" autoFocus disabled={deleting} onClick={() => dialogRef.current?.close()}>取消</button>
        <button type="button" className="knowledge-delete-btn" disabled={deleting} onClick={() => void removeKnowledge()}>{deleting ? "删除中…" : "确认删除"}</button>
      </div>
    </dialog>
    <BlockBrowser open={blocksOpen} onClose={() => setBlocksOpen(false)} />
  </>;
}
