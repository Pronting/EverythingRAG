import { type ChangeEvent, useEffect, useRef, useState } from "react";
import { getImportStatus, uploadFolder } from "../api/importApi";
import type { ImportStatus } from "../types";

const POLL_INTERVAL_MS = 1000;

interface ImportPanelProps {
  /** 导入成功后回调（父级刷新 /api/status 知识计数）。 */
  onImported: () => void;
}

/** 知识库导入面板：选择文件夹 -> 上传 .md 到本地后端 -> 异步导入 -> 轮询进度 -> 展示报告。 */
export default function ImportPanel({ onImported }: ImportPanelProps) {
  const [busy, setBusy] = useState(false);
  const [task, setTask] = useState<ImportStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
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

  /** 轮询任务状态，terminal 状态（done/error）时停止并刷新计数。 */
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
      if (status.status === "done") onImported();
    };
    stopPolling();
    timerRef.current = window.setInterval(() => void tick(), POLL_INTERVAL_MS);
    void tick(); // 立即查一次，避免小任务延迟一整轮
  };

  const handleFolderSelected = async (event: ChangeEvent<HTMLInputElement>): Promise<void> => {
    const input = event.target;
    const files = Array.from(input.files ?? []).filter((file) =>
      file.name.toLowerCase().endsWith(".md"),
    );
    input.value = ""; // 允许再次选择同一文件夹时触发 change
    if (busy) return;
    if (files.length === 0) {
      setError("所选文件夹内没有 Markdown 文件");
      return;
    }
    setBusy(true);
    setError(null);
    setTask(null);
    try {
      const { task_id } = await uploadFolder(files);
      poll(task_id);
    } catch (uploadError) {
      setBusy(false);
      setError(uploadError instanceof Error ? uploadError.message : String(uploadError));
    }
  };

  const progress = task?.progress;
  const report = task?.report;

  return (
    <section className="import-panel">
      <div className="import-controls">
        <button
          className="import-start"
          type="button"
          disabled={busy}
          onClick={() => fileInputRef.current?.click()}
        >
          {busy ? "导入中…" : "选择文件夹"}
        </button>
        <input
          type="file"
          ref={fileInputRef}
          multiple
          hidden
          {...({ webkitdirectory: "" } as object)}
          onChange={(event) => void handleFolderSelected(event)}
          aria-label="选择知识库文件夹"
        />
        <span className="import-hint">选择文件夹后，其中的 Markdown 文件会被复制到本地知识库并导入</span>
      </div>

      {error !== null && (
        <p className="import-error" role="alert">
          {error}
        </p>
      )}

      {task !== null && task.status === "running" && progress !== undefined && (
        <p className="import-progress" role="status">
          扫描 {progress.files_scanned} · 解析 {progress.files_parsed} · 跳过{" "}
          {progress.files_skipped} · 块 {progress.chunks}
        </p>
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
