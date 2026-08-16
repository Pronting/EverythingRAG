/** SSE 来源块：对齐后端 chat_service._meta_frame 的来源投影（见 BlockMetadata）。 */
export interface Source {
  block_id: string;
  text: string;
  source_file: string;
  similarity: number;
  heading_path: string | null;
  anchor: string | null;
  chunk_type: string;
  platform: string;
  /** 来源类型："web" = 联网搜索结果（其余为知识库片段）。 */
  source_type?: string;
  /** 联网搜索结果的原始链接（仅 source_type === "web" 时存在）。 */
  url?: string | null;
}

/** SSE 帧联合类型：对齐后端帧契约 meta / token / reasoning / tool / done / error。 */
export type SSEFrame =
  | { type: "meta"; sources: Source[] }
  | { type: "token"; text: string }
  | { type: "reasoning"; text: string }
  | { type: "tool"; tool: string; status: string }
  | { type: "done" }
  | { type: "error"; message: string };

/** /api/status 响应中前端关心的字段（app 元信息 + 配置 + 知识计数 + 隐私基线）。 */
export interface StatusResponse {
  app: { name: string; version: string };
  config: {
    wizard_completed: boolean;
    embed_configured: boolean;
    chat_configured: boolean;
    vision_configured: boolean;
    vision_enabled: boolean;
    search_enabled: boolean;
    search_configured: boolean;
  };
  knowledge: {
    file_count: number;
    chunk_count: number;
    image_count: number;
    last_sync_at: string | null;
    needs_rebuild: boolean;
  };
  privacy: { outbound_state: string };
}

/** 导入任务轮询状态：对齐后端 /api/import/status 响应。 */
export type ImportTaskStatusValue = "running" | "done" | "error";

/** 全量导入的进度快照（后端 /api/import/status 的全量形态）。 */
export interface ImportProgressFull {
  files_scanned: number;
  files_parsed: number;
  files_skipped: number;
  chunks: number;
}

/** 全量导入的完成报告（后端 IngestReport 投影）。 */
export interface ImportReportFull {
  files_scanned: number;
  files_parsed: number;
  files_skipped: number;
  chunks: number;
  blocks_upserted: number;
  errors: string[];
}

/** 增量同步的进度快照（后端 /api/import/status 的增量形态）。 */
export interface SyncProgress {
  files_scanned: number;
  files_processed: number;
  files_skipped: number;
}

/** 增量同步的完成报告（后端 SyncReport 投影）。 */
export interface SyncReport {
  files_scanned: number;
  files_added: number;
  files_updated: number;
  files_deleted: number;
  files_unchanged: number;
  files_skipped: number;
  blocks_upserted: number;
  blocks_deleted: number;
  errors: string[];
}

/** 任务进度/报告的两态联合：全量导入（parsed/chunks）或增量同步（processed）。 */
export type ImportProgress = ImportProgressFull | SyncProgress;
export type ImportReport = ImportReportFull | SyncReport;

export interface ImportStatus {
  task_id: string;
  status: ImportTaskStatusValue;
  progress: ImportProgress;
  report: ImportReport | null;
  error: string | null;
  created_at: string;
  updated_at: string;
}

/** /api/settings 脱敏视图（key 只给 hint）。 */
export interface SettingsView {
  chat: {
    provider_type: string;
    base_url: string | null;
    model: string | null;
    api_key_set: boolean;
    api_key_hint: string | null;
    supports_image: boolean;
  };
  embed: {
    mode: "cloud";
    base_url: string | null;
    model: string | null;
    api_key_set: boolean;
    api_key_hint: string | null;
  };
  vision: {
    provider_type: string;
    base_url: string | null;
    model: string | null;
    api_key_set: boolean;
    api_key_hint: string | null;
  };
  search: {
    provider_type: "tavily" | "searxng";
    enabled: boolean;
    base_url: string | null;
    max_results: number;
    api_key_set: boolean;
    api_key_hint: string | null;
  };
  system_prompt: string;
  avatars: {
    user: string | null;
    agent: string | null;
  };
  theme: ThemeValue;
}

/** 主题：浅色 / 深色（默认浅色）。 */
export type ThemeValue = "light" | "dark";

/** 轻提示类型：成功 / 失败。 */
export type ToastType = "success" | "error";

/** PUT /api/settings 局部更新负载（null = 保留原值；api_key 空串 = 清除）。 */
export interface SettingsUpdate {
  chat?: {
    provider_type?: string;
    base_url?: string;
    model?: string;
    api_key?: string | null;
    supports_image?: boolean;
  };
  embed?: {
    mode?: "cloud";
    base_url?: string;
    model?: string;
    api_key?: string | null;
  };
  vision?: {
    provider_type?: string;
    base_url?: string;
    model?: string;
    api_key?: string | null;
  };
  search?: {
    provider_type?: "tavily" | "searxng";
    enabled?: boolean;
    base_url?: string;
    max_results?: number;
    api_key?: string | null;
  };
  system_prompt?: string;
  theme?: ThemeValue;
}

/** 持久化的对话消息（含思维链快照与检索来源快照）。 */
export interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
  thinking: string | null;
  sources: Source[] | null;
  created_at: string;
}

/** 会话摘要（侧边栏列表项）。 */
export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

/** 完整会话（含消息列表）。 */
export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: ConversationMessage[];
}
