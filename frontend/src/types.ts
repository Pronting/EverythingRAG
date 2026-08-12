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
}

/** SSE 帧联合类型：对齐后端帧契约 meta / token / done / error。 */
export type SSEFrame =
  | { type: "meta"; sources: Source[] }
  | { type: "token"; text: string }
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

export interface ImportProgress {
  files_scanned: number;
  files_parsed: number;
  files_skipped: number;
  chunks: number;
}

export interface ImportReport {
  files_scanned: number;
  files_parsed: number;
  files_skipped: number;
  chunks: number;
  blocks_upserted: number;
  errors: string[];
}

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
  };
  embed: {
    mode: "local" | "cloud";
    base_url: string | null;
    model: string | null;
    api_key_set: boolean;
    api_key_hint: string | null;
  };
  system_prompt: string;
}

/** PUT /api/settings 局部更新负载（null = 保留原值；api_key 空串 = 清除）。 */
export interface SettingsUpdate {
  chat?: {
    provider_type?: string;
    base_url?: string;
    model?: string;
    api_key?: string | null;
  };
  embed?: {
    mode?: "local" | "cloud";
    base_url?: string;
    model?: string;
    api_key?: string | null;
  };
  system_prompt?: string;
}
