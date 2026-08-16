import type { SettingsUpdate, SettingsView } from "../types";

/** 设置 API 客户端：GET/PUT /api/settings（key 脱敏视图）。 */

const SETTINGS_ENDPOINT = "/api/settings";

/** 读取设置（脱敏视图：key 只给 api_key_set + hint）。 */
export async function getSettings(): Promise<SettingsView> {
  let response: Response;
  try {
    response = await fetch(SETTINGS_ENDPOINT);
  } catch {
    throw new Error("无法连接本地服务，请确认后端已启动");
  }
  if (!response.ok) {
    throw new Error(await readErrorDetail(response));
  }
  return (await response.json()) as SettingsView;
}

/** 保存设置（局部更新），返回更新后的脱敏视图。 */
export async function updateSettings(update: SettingsUpdate): Promise<SettingsView> {
  let response: Response;
  try {
    response = await fetch(SETTINGS_ENDPOINT, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(update),
    });
  } catch {
    throw new Error("无法连接本地服务，请确认后端已启动");
  }
  if (!response.ok) {
    throw new Error(await readErrorDetail(response));
  }
  return (await response.json()) as SettingsView;
}

/** 头像类型：user=用户，agent=Agent 助手。 */
export type AvatarKind = "user" | "agent";

/** 上传头像（multipart），返回新的头像 URL。 */
export async function uploadAvatar(kind: AvatarKind, file: File): Promise<string> {
  const form = new FormData();
  form.append("file", file);
  let response: Response;
  try {
    response = await fetch(`/api/avatars/${kind}`, { method: "POST", body: form });
  } catch {
    throw new Error("无法连接本地服务，请确认后端已启动");
  }
  if (!response.ok) {
    throw new Error(await readErrorDetail(response));
  }
  const data = (await response.json()) as { url: string };
  return data.url;
}

/** 删除头像（恢复默认）。 */
export async function deleteAvatar(kind: AvatarKind): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`/api/avatars/${kind}`, { method: "DELETE" });
  } catch {
    throw new Error("无法连接本地服务，请确认后端已启动");
  }
  if (!response.ok) {
    throw new Error(await readErrorDetail(response));
  }
}

/** 尝试从错误响应体提取 detail；失败回退通用消息。 */
async function readErrorDetail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (typeof body === "object" && body !== null && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      if (typeof detail === "string" && detail !== "") return detail;
    }
  } catch {
    /* 非 JSON 响应 */
  }
  return `本地服务返回异常（HTTP ${response.status}）`;
}
