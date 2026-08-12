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
