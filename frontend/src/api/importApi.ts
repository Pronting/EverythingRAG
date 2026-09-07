import type { ImportStatus } from "../types";

/** 导入 API 客户端：POST /api/import（异步启动）+ GET /api/import/status/{id}（轮询）。 */

const IMPORT_ENDPOINT = "/api/import";

export interface StartImportResult {
  task_id: string;
}

/** 启动异步导入，返回 task_id；后端校验失败时抛出含 detail 的可读错误。 */
export async function startImport(dir: string, mode: "full" | "incremental" = "full"): Promise<StartImportResult> {
  let response: Response;
  try {
    response = await fetch(IMPORT_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dir, mode }),
    });
  } catch {
    throw new Error("无法连接本地服务，请确认后端已启动");
  }
  if (!response.ok) {
    throw new Error(await readErrorDetail(response));
  }
  return (await response.json()) as StartImportResult;
}

/** 查询导入任务状态（running / done / error）。 */
export async function getImportStatus(taskId: string): Promise<ImportStatus> {
  let response: Response;
  try {
    response = await fetch(`${IMPORT_ENDPOINT}/status/${encodeURIComponent(taskId)}`);
  } catch {
    throw new Error("无法连接本地服务，请确认后端已启动");
  }
  if (!response.ok) {
    throw new Error(await readErrorDetail(response));
  }
  return (await response.json()) as ImportStatus;
}

/** 上传式导入：把选中的文件夹文件（.md）multipart 上传到本地后端并异步导入。

mode=incremental（默认）：重新选择同一批文件夹上传 = 增量对账（只处理新增/更新/删除）；
mode=full：仅入库/更新，不删除。
 */
export async function uploadFolder(
  files: File[],
  mode: "full" | "incremental" = "incremental",
  onProgress?: (percent: number | null) => void,
): Promise<StartImportResult> {
  const formData = new FormData();
  for (const file of files) {
    // webkitRelativePath 携带相对路径（如 subdir/a.md），供后端按原结构暂存
    const relativePath = file.webkitRelativePath || file.name;
    formData.append("files", file, relativePath);
  }
  formData.append("mode", mode);
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${IMPORT_ENDPOINT}/upload`);
    xhr.upload.onprogress = (event) => onProgress?.(
      event.lengthComputable ? Math.min(100, Math.round(event.loaded / event.total * 100)) : null,
    );
    xhr.onerror = () => reject(new Error("无法连接本地服务，请确认后端已启动"));
    xhr.onabort = () => reject(new Error("上传已取消"));
    xhr.onload = () => {
      try {
        const body = JSON.parse(xhr.responseText);
        if (xhr.status < 200 || xhr.status >= 300) {
          reject(new Error(typeof body.detail === "string" ? body.detail : `上传失败（HTTP ${xhr.status}）`));
        } else if (typeof body.task_id !== "string") {
          reject(new Error("上传响应异常，请重试"));
        } else resolve(body as StartImportResult);
      } catch { reject(new Error(`上传响应异常（HTTP ${xhr.status}）`)); }
    };
    xhr.send(formData);
  });
}

/** 一键增量同步：对全部已登记的知识库目录做增量同步，返回 task_id（走同一轮询）。 */
export async function syncKnowledge(): Promise<StartImportResult> {
  let response: Response;
  try {
    response = await fetch("/api/sync", { method: "POST" });
  } catch {
    throw new Error("无法连接本地服务，请确认后端已启动");
  }
  if (!response.ok) {
    throw new Error(await readErrorDetail(response));
  }
  return (await response.json()) as StartImportResult;
}

/** 尝试从错误响应体提取 detail 字符串；失败回退通用消息。 */
async function readErrorDetail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (typeof body === "object" && body !== null && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === "string" && detail !== "") return detail;
    }
  } catch {
    /* 非 JSON 响应，忽略 */
  }
  return `本地服务返回异常（HTTP ${response.status}）`;
}
