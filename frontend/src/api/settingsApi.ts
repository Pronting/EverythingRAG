import type { ProviderModelsResult, RerankTestResult, SettingsUpdate, SettingsView } from "../types";

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

/** 相关性筛选（重排）连接测试的可提交字段：未保存的值随请求体提交，服务端不持久化。 */
export interface RerankTestRequest {
  endpoint?: string;
  model?: string;
  /** 省略 = 复用已保存密钥（换地址时不复用，由服务端判定）。 */
  api_key?: string;
  profile_id?: string;
}

/**
 * POST /api/settings/rerank/test：真实服务连通性测试（PRD §8.1）。
 *
 * 服务端用固定公开文本、不读知识库；`quality_checked` 恒为 false ——
 * 连通成功不代表质量验收通过。
 */
export async function testRerankConnection(
  request: RerankTestRequest,
): Promise<RerankTestResult> {
  let response: Response;
  try {
    response = await fetch("/api/settings/rerank/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
  } catch {
    throw new Error("无法连接本地服务，请确认后端已启动");
  }
  if (!response.ok) {
    throw new Error(await readErrorDetail(response));
  }
  return (await response.json()) as RerankTestResult;
}

/** 连接测试结果：只说明服务可达与协议匹配，不代表能力/质量结论。 */
export interface ModelConnectionResult {
  success: boolean;
  message: string;
  latency_ms?: number;
}

/**
 * POST /api/settings/test-connection：用未保存的表单值做一次真实连通性测试。
 *
 * `api_key` 省略时服务端只在地址与已保存配置一致时复用密钥（换地址不复用）。
 */
export async function testModelConnection(request: {
  kind: "chat" | "embed" | "vision";
  base_url: string;
  model: string;
  api_key?: string;
}): Promise<ModelConnectionResult> {
  let response: Response;
  try {
    response = await fetch("/api/settings/test-connection", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
  } catch {
    throw new Error("无法连接本地服务，请确认后端已启动");
  }
  if (!response.ok) {
    throw new Error(
      response.status === 404 || response.status === 405
        ? "请更新并重启本地服务后重试"
        : "测试请求失败，请检查配置后重试",
    );
  }
  return (await response.json()) as ModelConnectionResult;
}

/**
 * POST /api/settings/providers/models：按 OpenAI 兼容端点拉取可用模型（「获取可用模型」）。
 *
 * 真实出网（服务端经唯一 OutboundClient 审计）；`api_key` 省略时服务端只在地址与已保存
 * 供应商一致时复用密钥，换地址不复用。失败返回 `success:false` + 稳定文案（不回显密钥）。
 */
export async function fetchProviderModels(request: {
  provider_id?: string;
  base_url: string;
  api_key?: string;
}): Promise<ProviderModelsResult> {
  let response: Response;
  try {
    response = await fetch("/api/settings/providers/models", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
  } catch {
    throw new Error("无法连接本地服务，请确认后端已启动");
  }
  if (!response.ok) {
    throw new Error(await readErrorDetail(response));
  }
  return (await response.json()) as ProviderModelsResult;
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
