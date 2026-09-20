/**
 * 工作区记忆 API 客户端（PRD 11 / 13.2；后端 `app/api/routes/memory.py`）。
 *
 * ⚠️ 记忆开关的唯一权威是**本模块返回的 `enabled`**。
 * `WorkspaceContextView.memory_enabled` 在后端恒为默认 `true`（尚未接线），
 * 用它判断记忆是否开启会永远显示「已开启」。
 */

export type MemoryKind = "response_preference" | "topic" | "progress" | "confirmed_conclusion";
export type MemoryEntryStatus = "active" | "forgotten" | "superseded" | "orphaned";

export interface MemoryEntry {
  id: string;
  kind: MemoryKind;
  text: string;
  source_conversation_id: string;
  source_message_ids: string[];
  created_at: string;
  updated_at: string;
  user_pinned: boolean;
  status: MemoryEntryStatus;
  superseded_by: string | null;
  operation_id: string;
}

export interface MemoryView {
  workspace_id: string;
  storage_key: string;
  schema_version: number;
  /** 工作区记忆开关：**这是判断「开/关」的唯一字段**。 */
  enabled: boolean;
  revision: number;
  invalidation_revision: number;
  updated_at: string;
  entries: MemoryEntry[];
  invalidations: unknown[];
  processed_cursors: Record<string, string>;
  applied_operations: number;
  snapshot_preview: {
    revision: number;
    invalidation_revision: number;
    hash: string;
    chars: number;
    truncated: boolean;
    overflow_count: number;
    needs_tidy: boolean;
  };
  blocked_source_conversations: string[];
}

export interface MemoryMutationResult {
  applied: boolean;
  reason: string;
  revision?: number;
  invalidation_revision?: number;
  enabled?: boolean;
  entry?: MemoryEntry | null;
}

const MEMORY_LABELS: Record<MemoryKind, string> = {
  response_preference: "回答要求",
  topic: "主题",
  progress: "进展",
  confirmed_conclusion: "已确认结论",
};

const STATUS_LABELS: Record<MemoryEntryStatus, string> = {
  active: "生效中",
  forgotten: "已忘记",
  superseded: "已被取代",
  orphaned: "来源已失效",
};

export function memoryKindLabel(kind: MemoryKind): string {
  return MEMORY_LABELS[kind] ?? kind;
}

export function memoryStatusLabel(status: MemoryEntryStatus): string {
  return STATUS_LABELS[status] ?? status;
}

export class MemoryApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "MemoryApiError";
    this.status = status;
  }
}

export function describeMemoryError(error: unknown): string {
  if (error instanceof MemoryApiError) {
    if (error.status === 409) return "记忆已被其它操作修改，请刷新后重试";
    if (error.status === 404) return "该记忆条目不存在";
    if (error.status === 422) return error.message || "输入不合法";
    if (error.status === 503) return "本地服务尚未就绪，请稍后重试";
    return error.message;
  }
  return error instanceof Error && error.message !== ""
    ? error.message
    : "无法连接本地服务，请确认后端已启动";
}

function memoryPath(workspaceId: string): string {
  return `/api/workspaces/${encodeURIComponent(workspaceId)}/memory`;
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch {
    throw new MemoryApiError(0, "无法连接本地服务，请确认后端已启动");
  }
  if (!response.ok) {
    let detail = "";
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* 非 JSON */
    }
    throw new MemoryApiError(response.status, detail || `本地服务返回异常（HTTP ${response.status}）`);
  }
  return (await response.json()) as T;
}

/** 读取工作区记忆（含开关与条目）。 */
export async function getMemory(workspaceId: string): Promise<MemoryView> {
  return request<MemoryView>(memoryPath(workspaceId));
}

/** 主动保存一条记忆（PRD 13.2 第 6 行）。 */
export async function addMemory(
  workspaceId: string,
  text: string,
  options: { kind?: MemoryKind; conversationId?: string; expectedRevision?: number } = {},
): Promise<MemoryMutationResult> {
  return request<MemoryMutationResult>(memoryPath(workspaceId), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      kind: options.kind ?? null,
      conversation_id: options.conversationId ?? null,
      expected_revision: options.expectedRevision ?? null,
    }),
  });
}

/** 开启 / 关闭工作区记忆（PRD 11.4）。 */
export async function setMemoryEnabled(
  workspaceId: string,
  enabled: boolean,
  expectedRevision?: number,
): Promise<MemoryMutationResult> {
  return request<MemoryMutationResult>(memoryPath(workspaceId), {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled, expected_revision: expectedRevision ?? null }),
  });
}

/** 人工纠正条目（内容/类别），并使已有快照失效（PRD 13.2 第 3 行）。 */
export async function correctMemory(
  workspaceId: string,
  memoryId: string,
  patch: { text?: string; kind?: MemoryKind; replacement?: string; expectedRevision?: number },
): Promise<MemoryMutationResult> {
  return request<MemoryMutationResult>(`${memoryPath(workspaceId)}/${encodeURIComponent(memoryId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: patch.text ?? null,
      kind: patch.kind ?? null,
      replacement: patch.replacement ?? "",
      expected_revision: patch.expectedRevision ?? null,
    }),
  });
}

/** 删除（遗忘）条目。 */
export async function forgetMemory(
  workspaceId: string,
  memoryId: string,
  expectedRevision?: number,
): Promise<MemoryMutationResult> {
  const suffix = expectedRevision === undefined ? "" : `?expected_revision=${expectedRevision}`;
  return request<MemoryMutationResult>(
    `${memoryPath(workspaceId)}/${encodeURIComponent(memoryId)}${suffix}`,
    { method: "DELETE" },
  );
}
