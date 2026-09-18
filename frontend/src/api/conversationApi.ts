import type { Conversation, ConversationMessage, ConversationSummary } from "../types";

/** 会话 API 客户端：列表 / 新建 / 读取 / 追加消息 / AI 标题。 */

const CONVERSATIONS_ENDPOINT = "/api/conversations";

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

/** 会话摘要列表（按最近更新倒序，不含归档与空会话）。
 *
 * 传入 workspaceId 时只返回该工作区（侧边栏分区切换）；缺省不过滤（跨分区）。
 */
export async function listConversations(
  query = "",
  signal?: AbortSignal,
  workspaceId?: string | null,
): Promise<ConversationSummary[]> {
  const params = new URLSearchParams();
  if (query) params.set("q", query);
  if (workspaceId) params.set("workspace", workspaceId);
  const suffix = params.toString();
  const response = await fetch(suffix ? `${CONVERSATIONS_ENDPOINT}?${suffix}` : CONVERSATIONS_ENDPOINT, { signal });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  return (await response.json()) as ConversationSummary[];
}

/** 已归档会话列表（设置页「归档对话」，按归档时间倒序）。 */
export async function listArchivedConversations(signal?: AbortSignal): Promise<ConversationSummary[]> {
  const response = await fetch(`${CONVERSATIONS_ENDPOINT}/archived`, { signal });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  return (await response.json()) as ConversationSummary[];
}

/** 新建空会话（在指定工作区，缺省默认分区），返回摘要。 */
export async function createConversation(workspaceId?: string | null): Promise<ConversationSummary> {
  const query = workspaceId ? `?workspace=${encodeURIComponent(workspaceId)}` : "";
  const response = await fetch(`${CONVERSATIONS_ENDPOINT}${query}`, { method: "POST" });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  return (await response.json()) as ConversationSummary;
}

/** 侧边栏拖拽落点：把会话移到目标工作区（缺省原地）的第 targetIndex 位。 */
export async function moveConversationInSidebar(
  id: string,
  target: { workspaceId?: string; targetIndex?: number },
): Promise<ConversationSummary> {
  const response = await fetch(`${CONVERSATIONS_ENDPOINT}/${encodeURIComponent(id)}/move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      workspace_id: target.workspaceId ?? null,
      target_index: target.targetIndex ?? null,
    }),
  });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  return (await response.json()) as ConversationSummary;
}

/** 把会话移动到指定工作区（分区归类）。 */
export async function moveConversationToWorkspace(
  id: string,
  workspaceId: string,
): Promise<ConversationSummary> {
  const response = await fetch(`${CONVERSATIONS_ENDPOINT}/${encodeURIComponent(id)}/workspace`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ workspace_id: workspaceId }),
  });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  return (await response.json()) as ConversationSummary;
}

/** 读取完整会话（含消息）。 */
export async function getConversation(id: string): Promise<Conversation> {
  const response = await fetch(`${CONVERSATIONS_ENDPOINT}/${encodeURIComponent(id)}`);
  if (!response.ok) throw new Error(await readErrorDetail(response));
  return (await response.json()) as Conversation;
}

/** 从第 index 条消息起截断会话（「编辑重发」：该条及其之后的轮次作废）。 */
export async function truncateConversation(id: string, index: number): Promise<number> {
  const response = await fetch(`${CONVERSATIONS_ENDPOINT}/${encodeURIComponent(id)}/truncate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ index }),
  });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  const body = (await response.json()) as { message_count?: number };
  return body.message_count ?? 0;
}

/** 追加一轮消息（user + assistant），返回摘要。 */
export async function appendConversationMessages(
  id: string,
  messages: Pick<
    ConversationMessage,
    "role" | "content" | "sources" | "answer_basis" | "policy_version"
  >[],
): Promise<ConversationSummary> {
  const response = await fetch(`${CONVERSATIONS_ENDPOINT}/${encodeURIComponent(id)}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
  });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  return (await response.json()) as ConversationSummary;
}

/** 用对话模型生成会话标题，返回新标题。 */
export async function generateConversationTitle(id: string): Promise<string> {
  const response = await fetch(`${CONVERSATIONS_ENDPOINT}/${encodeURIComponent(id)}/generate-title`, {
    method: "POST",
  });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  const data = (await response.json()) as { title: string };
  return data.title;
}

/** 删除会话。 */
export async function deleteConversation(id: string): Promise<void> {
  const response = await fetch(`${CONVERSATIONS_ENDPOINT}/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!response.ok) throw new Error(await readErrorDetail(response));
}

/** 归档会话：从侧边栏隐藏，可在设置页「归档对话」恢复。 */
export async function archiveConversation(id: string): Promise<ConversationSummary> {
  const response = await fetch(`${CONVERSATIONS_ENDPOINT}/${encodeURIComponent(id)}/archive`, {
    method: "POST",
  });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  return (await response.json()) as ConversationSummary;
}

/** 恢复已归档会话。 */
export async function restoreConversation(id: string): Promise<ConversationSummary> {
  const response = await fetch(`${CONVERSATIONS_ENDPOINT}/${encodeURIComponent(id)}/restore`, {
    method: "POST",
  });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  return (await response.json()) as ConversationSummary;
}

/** 分支会话：深拷贝源会话消息为独立新会话，返回分支摘要。 */
export async function forkConversation(id: string): Promise<ConversationSummary> {
  const response = await fetch(`${CONVERSATIONS_ENDPOINT}/${encodeURIComponent(id)}/fork`, {
    method: "POST",
  });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  return (await response.json()) as ConversationSummary;
}
