import { type ChangeEvent, useEffect, useRef, useState } from "react";
import { getImportStatus, uploadFolder } from "../api/importApi";
import type { ImportStatus } from "../types";

const POLL_INTERVAL_MS = 1000;

interface ImportPanelProps {
  /** 导入成功后回调（父级刷新 /api/status 知识计数）。 */
  onImported: () => void;
}

/** 文件相对路径（webkitRelativePath 优先，普通文件用 basename）。 */
function fileRelPath(file: File): string {
  return file.webkitRelativePath || file.name;
}

/**
 * 选择式导入面板：可多选多个子文件夹 / 单个文件，先积累「待导入清单」，
 * 确认后再统一上传导入（敏感目录/文件不选即可排除）。
 */
export default function ImportPanel({ onImported }: ImportPanelProps) {
  const [pending, setPending] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [task, setTask] = useState<ImportStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current !== null) window.clearInterval(timerRef.current);
    };
  }, []);

  const stopPolling = (): void => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  };

  /** 轮询任务状态，terminal（done/error）时停止并刷新计数。 */
  const poll = (taskId: string): void => {
    const tick = async (): Promise<void> => {
      let status: ImportStatus;
      try {
        status = await getImportStatus(taskId);
      } catch (pollError) {
        stopPolling();
        setBusy(false);
        setError(pollError instanceof Error ? pollError.message : String(pollError));
        return;
      }
      setTask(status);
      if (status.status === "running") return;
      stopPolling();
      setBusy(false);
      if (status.status === "done") {
        setPending([]); // 导入成功，清空待导入清单
        onImported();
      }
    };
    stopPolling();
    timerRef.current = window.setInterval(() => void tick(), POLL_INTERVAL_MS);
    void tick();
  };

  /** 把选中的文件（筛选 .md）追加到待导入清单。 */
  const addFiles = (files: File[]): void => {
    const markdown = files.filter((file) => file.name.toLowerCase().endsWith(".md"));
    if (markdown.length === 0) {
      setError("所选内容中没有 Markdown 文件");
      return;
    }
    setError(null);
    setPending((prev) => [...prev, ...markdown]);
  };

  const handleFolderSelected = (event: ChangeEvent<HTMLInputElement>): void => {
    const input = event.target;
    addFiles(Array.from(input.files ?? []));
    input.value = ""; // 允许再次选择同一文件夹
  };

  const handleFileSelected = (event: ChangeEvent<HTMLInputElement>): void => {
    const input = event.target;
    addFiles(Array.from(input.files ?? []));
    input.value = "";
  };

  const handleImport = async (): Promise<void> => {
    if (pending.length === 0 || busy) return;
    setBusy(true);
    setError(null);
    setTask(null);
    try {
      const { task_id } = await uploadFolder(pending);
      poll(task_id);
    } catch (importError) {
      setBusy(false);
      setError(importError instanceof Error ? importError.message : String(importError));
    }
  };

  const clearPending = (): void => {
    if (busy) return;
    setPending([]);
    setError(null);
    setTask(null);
  };

  const progress = task?.progress;
  const report = task?.report;

  return (
    <section className="import-panel">
      <div className="import-source-buttons">
        <button
          type="button"
          className="import-source-btn"
          disabled={busy}
          onClick={() => folderInputRef.current?.click()}
          title="可按住 Ctrl/Shift 多选多个子文件夹"
        >
          选择文件夹
        </button>
        <button
          type="button"
          className="import-source-btn"
          disabled={busy}
          onClick={() => fileInputRef.current?.click()}
          title="选择单个或多个 Markdown 文件"
        >
          选择文件
        </button>
        <input
          type="file"
          ref={folderInputRef}
          multiple
          hidden
          {...({ webkitdirectory: "" } as object)}
          onChange={handleFolderSelected}
          aria-label="选择知识库文件夹（可多选）"
        />
        <input
          type="file"
          ref={fileInputRef}
          multiple
          accept=".md"
          hidden
          onChange={handleFileSelected}
          aria-label="选择知识库文件（可多选）"
        />
      </div>

      <p className="import-hint">可多选重点子文件夹 / 单个文件；敏感目录不选即可排除。</p>

      {pending.length > 0 && (
        <div className="import-pending">
          <div className="import-pending-head">
            <span className="import-pending-count">待导入 {pending.length} 个文件</span>
            <button type="button" className="import-pending-clear" onClick={clearPending}>
              清空
            </button>
          </div>
          <ul className="import-pending-list">
            {pending.slice(0, 4).map((file, index) => (
              <li key={`${fileRelPath(file)}-${index}`} title={fileRelPath(file)}>
                {fileRelPath(file)}
              </li>
            ))}
            {pending.length > 4 && <li className="import-pending-more">…还有 {pending.length - 4} 个</li>}
          </ul>
        </div>
      )}

      <button
        type="button"
        className="import-btn"
        onClick={() => void handleImport()}
        disabled={busy || pending.length === 0}
      >
        {busy ? "导入中…" : pending.length > 0 ? `开始导入（${pending.length}）` : "开始导入"}
      </button>

      {error !== null && (
        <p className="import-error" role="alert">
          {error}
        </p>
      )}

      {task !== null && task.status === "running" && progress !== undefined && (
        <div className="import-progress" role="status">
          <div className="import-progress-track">
            <div
              className="import-progress-fill"
              style={{
                width: `${
                  progress.files_scanned > 0
                    ? Math.min(100, Math.round((progress.files_parsed / progress.files_scanned) * 100))
                    : 0
                }%`,
              }}
            />
          </div>
          <div className="import-progress-meta">
            <span>
              解析 {progress.files_parsed} / {progress.files_scanned}
            </span>
            <span>
              {progress.files_scanned > 0
                ? Math.min(100, Math.round((progress.files_parsed / progress.files_scanned) * 100))
                : 0}
              %
            </span>
          </div>
          <p className="import-progress-line">
            扫描 {progress.files_scanned} · 解析 {progress.files_parsed} · 跳过{" "}
            {progress.files_skipped} · 块 {progress.chunks}
          </p>
        </div>
      )}

      {task !== null && task.status === "done" && report != null && (
        <div className="import-report">
          <p className="import-report-title">导入完成</p>
          <p className="import-report-meta">
            文档 {report.files_parsed}（扫描 {report.files_scanned} · 跳过{" "}
            {report.files_skipped}）· 块 {report.chunks} · 写入 {report.blocks_upserted}
          </p>
          {report.errors.length > 0 && (
            <p className="import-report-errors">失败原因：{report.errors.join("、")}</p>
          )}
        </div>
      )}

      {task !== null && task.status === "error" && (
        <p className="import-error" role="alert">
          {task.error ?? "导入失败"}
        </p>
      )}
    </section>
  );
}
