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

/** 会话摘要列表（按最近更新倒序）。 */
export async function listConversations(query = "", signal?: AbortSignal): Promise<ConversationSummary[]> {
  const response = await fetch(query ? `${CONVERSATIONS_ENDPOINT}?q=${encodeURIComponent(query)}` : CONVERSATIONS_ENDPOINT, { signal });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  return (await response.json()) as ConversationSummary[];
}

/** 新建空会话，返回摘要。 */
export async function createConversation(): Promise<ConversationSummary> {
  const response = await fetch(CONVERSATIONS_ENDPOINT, { method: "POST" });
  if (!response.ok) throw new Error(await readErrorDetail(response));
  return (await response.json()) as ConversationSummary;
}

/** 读取完整会话（含消息）。 */
export async function getConversation(id: string): Promise<Conversation> {
  const response = await fetch(`${CONVERSATIONS_ENDPOINT}/${encodeURIComponent(id)}`);
  if (!response.ok) throw new Error(await readErrorDetail(response));
  return (await response.json()) as Conversation;
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
