import type { Workspace } from "../types";

/** 工作区（对话分区）API 客户端：列表 / 新建 / 改名 / 删除。 */

const WORKSPACES_ENDPOINT = "/api/workspaces";

async function readErrorDetail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (typeof body === "object" && body !== null && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === "string" && detail !== "") return detail;
    }
  } catch {
    /* 非 JSON */
  }
  return `本地服务返回异常（HTTP ${response.status}）`;
}

/** 工作区列表（默认分区在最前），带各区活跃会话数。 */
export async function listWorkspaces(signal?: AbortSignal): Promise<Workspace[]> {
  const response = await fetch(WORKSPACES_ENDPOINT, { signal });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  return (await response.json()) as Workspace[];
}

/** 新建工作区。 */
export async function createWorkspace(name: string): Promise<Workspace> {
  const response = await fetch(WORKSPACES_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  return (await response.json()) as Workspace;
}

/** 重命名工作区。 */
export async function renameWorkspace(id: string, name: string): Promise<Workspace> {
  const response = await fetch(`${WORKSPACES_ENDPOINT}/${encodeURIComponent(id)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  return (await response.json()) as Workspace;
}

/** 删除工作区（其中的对话会被归档，可在设置页找回）；返回归档条数。 */
export async function deleteWorkspace(id: string): Promise<number> {
  const response = await fetch(`${WORKSPACES_ENDPOINT}/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  const body = (await response.json()) as { archived?: number };
  return body.archived ?? 0;
}

/** 置顶 / 取消置顶工作区（别针图标）。 */
export async function setWorkspacePinned(id: string, pinned: boolean): Promise<Workspace> {
  const response = await fetch(`${WORKSPACES_ENDPOINT}/${encodeURIComponent(id)}/pin`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pinned }),
  });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  return (await response.json()) as Workspace;
}

/** 按给定顺序重写工作区位次（拖拽排序），返回新的完整顺序。 */
export async function reorderWorkspaces(ids: string[]): Promise<Workspace[]> {
  const response = await fetch(`${WORKSPACES_ENDPOINT}/order`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids }),
  });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  return (await response.json()) as Workspace[];
}
