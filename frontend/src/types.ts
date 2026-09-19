/** SSE 来源块：对齐后端 chat_service._meta_frame 的来源投影（见 BlockMetadata）。 */
export interface Source {
  image_url?: string | null;
  /** 本轮稳定引用编号：本地来源 S1...，联网来源 W1...。 */
  source_id: string;
  block_id: string;
  text: string;
  source_file: string;
  /**
   * 仅向量/混合命中有余弦相关度；纯词法与章节补充为 null。
   *
   * 含义固定为**原始向量相似度**：启用相关性筛选后它仍是向量分数，
   * 重排分数只进 ``rerank_score``，不得改写本字段（PRD §8.3）。
   */
  similarity: number | null;
  match_type?: "dense" | "hybrid" | "lexical" | "expanded" | "web" | "memory" | "rerank";
  heading_path: string | null;
  anchor: string | null;
  chunk_type: string;
  platform: string;
  /** 来源类型："web" = 联网搜索结果（其余为知识库片段）。 */
  source_type?: string;
  /** 联网搜索结果的原始链接（仅 source_type === "web" 时存在）。 */
  url?: string | null;
  /** 本轮证据经过相关性筛选（新结果展示「已筛选资料」）；旧记录缺省为未筛选。 */
  relevance_checked?: boolean;
  /** 筛选服务的排序分数（PRD §8.3 的「排序分数 0.812」）；不是正确率或概率。 */
  rerank_score?: number | null;
  /** 分数类型："rerank" = 相关性筛选排序分数。 */
  score_type?: string;
  /** 产生该排序分数的校准档案 id（未启用筛选的来源为 null）。 */
  profile_id?: string | null;
  /** 证据视图载荷哈希（成员集合与评分正文的稳定指纹）。 */
  evidence_hash?: string | null;
  /** 组合成员引用：主证据之外的补充成员逐个可追溯（PRD §8.3）。 */
  member_refs?: MemberRef[] | null;
  /** 本轮检索范围标识（诊断用；日常页面不展示）。 */
  scope_id?: string | null;
  /** 知识库版本（诊断用；日常页面不展示）。 */
  index_revision?: string | null;
  /** 证据视图类型："block" / "support_bundle" / "image_text"。 */
  view_kind?: string | null;
}

/** 证据视图成员引用：原文半开区间 [start, end)，end=null 表示整段成员。 */
export interface MemberRef {
  block_id: string;
  content_hash: string;
  relation: "primary" | "supplement" | "image_text";
  start: number;
  end: number | null;
  block_char_offset: number | null;
}

/** meta.retrieval（PRD §8.2）：仅 CONTENT_QA 下发，报告本轮筛选状态与计数。 */
export interface RetrievalMeta {
  /** filtered = 已筛选；empty = 无通过筛选的资料；disabled = 未启用筛选。 */
  status: "filtered" | "empty" | "disabled";
  profile_id?: string | null;
  candidate_count?: number;
  selected_count?: number;
  /** 诊断：候选里查不到标题的数量（日常页面不展示实现参数）。 */
  missing_headings?: number;
}

export type AnswerBasis =
  | "knowledge"
  | "web"
  | "general"
  | "product"
  | "catalog"
  | "insufficient";

export type QueryMode = "PRODUCT_HELP" | "KNOWLEDGE_OVERVIEW" | "CONTENT_QA";

/**
 * SSE 帧联合类型。
 *
 * `reasoning` 为推理模型思维链增量（本地展示，前端默认折叠）；
 * 原始 provider 报文不会越过 SSE 边界。
 */
export type SSEFrame =
  | {
      type: "meta";
      sources: Source[];
      answer_basis: AnswerBasis;
      policy_version: string;
      query_mode: QueryMode;
      /** 可选：本轮相关性筛选状态（PRD §8.2，仅 CONTENT_QA 下发）。 */
      retrieval?: RetrievalMeta;
    }
  | { type: "reasoning"; text: string }
  | { type: "token"; text: string }
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
    image_tasks: {
      pending: number;
      done: number;
      failed: number;
      total: number;
    };
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
  rerank: RerankSettingsView;
  answer_preferences: {
    language: "auto" | "zh-CN" | "en";
    verbosity: "concise" | "balanced" | "detailed";
    tone: "natural" | "professional";
    response_format: "auto" | "prose" | "bullets";
    knowledge_only: boolean;
    custom_instructions: string;
  };
  policy: {
    version: string;
    managed: boolean;
  };
  avatars: {
    user: string | null;
    agent: string | null;
  };
  theme: ThemeValue;
}

/** 语义块（GET /api/blocks 返回项，供浏览面板分类/搜索/分页）。 */
export interface BlockInfo {
  block_id: string;
  text: string;
  chunk_type: string;
  source_type: string;
  source_file: string;
  heading_path: string | null;
  anchor: string | null;
  platform: string;
}

/** 主题：浅色 / 深色（默认浅色）。 */
export type ThemeValue = "light" | "dark";

/** 轻提示类型：成功 / 失败。 */
export type ToastType = "success" | "error";

/** 重排设置状态：未启用 / 缺配置 / 档案失配 / 语料变化需复测 / 可用。 */
export type RerankStatus =
  | "disabled"
  | "not_configured"
  | "profile_mismatch"
  | "invalid"
  | "error"
  | "needs_validation"
  | "ready";

/** 校准档案投影（PRD §8.1：档案状态与验证日期；不展示 tokenizer 等实现参数）。 */
export interface RerankProfileView {
  profile_id: string;
  status: "candidate" | "quality_passed" | "validated";
  model: string;
  generated_at: string | null;
  validated_at: string | null;
  needs_validation: boolean;
}

/** /api/settings 的重排段（GET 脱敏视图）。 */
export interface RerankSettingsView {
  enabled: boolean;
  provider_type: string;
  endpoint: string;
  model: string | null;
  api_key_set: boolean;
  api_key_hint: string | null;
  profile_id: string | null;
  connect_timeout_seconds: number;
  read_timeout_seconds: number;
  configured: boolean;
  status: RerankStatus;
  /** 隔离验收环境（启动参数开启，页面需显示「验收环境」）。 */
  acceptance_env: boolean;
  profile: RerankProfileView | null;
  profiles: RerankProfileView[];
}

/** POST /api/settings/rerank/test 响应：只表示连通性，不代表质量验收通过。 */
export interface RerankTestResult {
  success: boolean;
  quality_checked: false;
  model?: string;
  endpoint?: string;
  endpoint_host?: string;
  profile_id?: string | null;
  latency_ms?: number | null;
  candidate_count?: number;
  scored_count?: number;
  trace_id?: string | null;
  usage_reported?: boolean;
  code?: string;
  message?: string;
}

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
  rerank?: {
    enabled?: boolean;
    provider_type?: string;
    endpoint?: string;
    model?: string;
    api_key?: string | null;
    profile_id?: string;
    connect_timeout_seconds?: number;
    read_timeout_seconds?: number;
  };
  answer_preferences?: {
    language?: "auto" | "zh-CN" | "en";
    verbosity?: "concise" | "balanced" | "detailed";
    tone?: "natural" | "professional";
    response_format?: "auto" | "prose" | "bullets";
    knowledge_only?: boolean;
    custom_instructions?: string;
  };
  theme?: ThemeValue;
}

/** 持久化的对话消息（只含可展示正文、思维链、依据标签与来源快照）。 */
export interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
  sources: Source[] | null;
  answer_basis: AnswerBasis | null;
  policy_version: string | null;
  /** 推理模型思维链；旧会话 / 非推理模型为 null（前端不渲染折叠块）。 */
  reasoning?: string | null;
  created_at: string;
}

/** 会话摘要（侧边栏列表项 / 设置页归档列表项）。 */
export interface ConversationSummary {
  match_excerpt?: string;
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  /** 已归档：不在侧边栏出现，只能在设置页「归档对话」查看与恢复。 */
  archived?: boolean;
  archived_at?: string | null;
  /** fork 来源会话 id；非空表示这是某条对话的分支（侧边栏折叠在父对话下方）。 */
  parent_id?: string | null;
  /** fork 时父对话标题（归档页展示来源用）。 */
  forked_from_title?: string | null;
  /** 所属工作区（对话分区）id。 */
  workspace_id?: string;
  /** 工作区内手工排序位（拖拽排序后写入）；null/缺省 = 按更新时间。 */
  sort_index?: number | null;
}

/** 完整会话（含消息列表）。 */
export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  messages: ConversationMessage[];
  archived?: boolean;
  archived_at?: string | null;
  parent_id?: string | null;
  forked_from_title?: string | null;
  workspace_id?: string;
}

/** 工作区（对话分区）：侧边栏文件夹树的条目。 */
export interface Workspace {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
  /** 默认分区：不可删除。 */
  is_default: boolean;
  /** 置顶（别针）：排在其他分区之前。 */
  pinned?: boolean;
  /** 手工拖拽排序位；null/缺省 = 未排过（按创建时间）。 */
  order_index?: number | null;
  /** 该分区下的活跃会话数。 */
  conversation_count: number;
}

/** 默认工作区 id（后端常量，旧会话 / 未指定归属时落这里）。 */
export const DEFAULT_WORKSPACE_ID = "default";
