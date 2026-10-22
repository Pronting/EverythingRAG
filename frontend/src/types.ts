/** SSE 来源块：对齐后端 chat_service._meta_frame 的来源投影（见 BlockMetadata）。 */
export interface Source {
  image_url?: string | null;
  /** 来源文档或网页的可读标题；缺省时继续使用 source_file。 */
  document_title?: string | null;
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
  match_type?: "dense" | "hybrid" | "lexical" | "expanded" | "read" | "web" | "memory" | "rerank";
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

/**
 * 本轮相关性筛选状态与计数（PRD §8.2）。
 *
 * 集成说明（V1.2 的 run 架构）：旧架构由 `POST /api/chat` 的 `meta.retrieval` 下发；
 * run 架构**没有**该负载字段，前端改由**既有事件**推导（见 `run/processState.ts`
 * 的 `retrievalFromToolEvents`）：`tool.completed`（tool_name="rerank"）成功且
 * `result_count > 0` ⇒ `filtered`；成功且为 0 ⇒ `empty`；全程无 rerank 事件 ⇒
 * `disabled`。**未新增/未改动任何后端契约**（字段回落为 null 时界面按来源自身字段判定）。
 */
export interface RetrievalMeta {
  /** filtered = 已筛选；empty = 无通过筛选的资料；disabled = 未启用筛选。 */
  status: "filtered" | "empty" | "disabled";
  profile_id?: string | null;
  candidate_count?: number;
  selected_count?: number;
  /** 诊断：候选里查不到标题的数量（日常页面不展示实现参数）。 */
  missing_headings?: number;
}

/** 答案依据（PRD 8.3，9 值；与 app.agent.contracts.AnswerBasis 同步）。 */
export type AnswerBasis =
  | "knowledge"
  | "web"
  | "general"
  | "product"
  | "catalog"
  | "memory"
  | "history"
  | "mixed"
  | "insufficient";

export type QueryMode = "PRODUCT_HELP" | "KNOWLEDGE_OVERVIEW" | "CONTENT_QA";

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

/** 供应商下的一个模型：`id` 是真实调用名，`name` 只用于显示。 */
export interface ProviderModelView {
  id: string;
  name: string;
  /** 多模态（可直接接收图片输入）：选中它即开启「图片输入」。 */
  vision: boolean;
}

/** 供应商可用状态：已就绪 / 缺少密钥 / 配置不完整。 */
export type ProviderStatus = "ready" | "needs_key" | "unconfigured";

/** /api/settings 的供应商投影（密钥只给 api_key_set + 尾4位）。 */
export interface ProviderView {
  id: string;
  name: string;
  kind: "builtin" | "custom";
  protocol: string;
  base_url: string | null;
  api_key_set: boolean;
  api_key_hint: string | null;
  models: ProviderModelView[];
  status: ProviderStatus;
  /** 是否是当前选中的供应商。 */
  selected: boolean;
}

/** 预置目录项（设置页「添加提供方」列表）。 */
export interface ModelCatalogEntry {
  id: string;
  name: string;
  base_url: string;
  summary: string;
  requires_api_key: boolean;
  docs_url: string | null;
  models: ProviderModelView[];
}

/** 当前选中的供应商 + 模型（界面触发器与勾选态都看它）。 */
export interface ModelSelectionView {
  provider_id: string;
  provider_name: string;
  model_id: string;
  model_name: string;
  vision: boolean;
}

/** POST /api/settings/providers/models：按端点拉取可用模型（真实出网，服务端不落盘密钥）。 */
export interface ProviderModelsResult {
  success: boolean;
  message?: string;
  models: { id: string }[];
  count: number;
  latency_ms?: number;
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
  /** 对话模型供应商列表（可配置多个；`providers` 为整表替换语义）。 */
  providers: ProviderView[];
  /** 当前选中的供应商 + 模型；null = 未选择。 */
  model_selection: ModelSelectionView | null;
  /** 预置供应商目录（「添加提供方」）。 */
  model_catalog: ModelCatalogEntry[];
  /** 本版支持的 API 协议（设置页下拉选项）。 */
  model_protocols: string[];
  memory_enabled: boolean;
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

/** PUT /api/settings 中的一条供应商（整表替换：每条都要给全）。 */
export interface ProviderUpdate {
  id: string;
  name: string;
  kind: "builtin" | "custom";
  protocol?: string;
  base_url?: string | null;
  /** 省略 = 保留已保存密钥；空串 = 清除；非空 = 设置。 */
  api_key?: string;
  models: { id: string; name?: string | null; vision: boolean }[];
}

/** PUT /api/settings 局部更新负载（null = 保留原值；api_key 空串 = 清除）。 */
export interface SettingsUpdate {
  memory_enabled?: boolean;
  chat?: {
    provider_type?: string;
    base_url?: string;
    model?: string;
    api_key?: string | null;
    supports_image?: boolean;
  };
  /** 整表替换：省略的供应商即被删除。 */
  providers?: ProviderUpdate[];
  /** 当前选中的供应商 + 模型；两个字段都为空 = 清除选择。 */
  model_selection?: { provider_id: string | null; model_id: string | null };
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
  message_id?: string | null;
  run_id?: string | null;
  citations?: CitationEntry[] | null;
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
}

/** 完整会话（含消息列表）。 */
export interface Conversation {
  permission_mode?: PermissionMode;
  reasoning_level?: ReasoningLevel;
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
  default_permission_mode?: PermissionMode;
  default_reasoning_level?: ReasoningLevel;
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

// =====================================================================
// V1.2.0 run 契约（schema_version = 2）
//
// 单一事实来源是 `backend/app/agent/contracts.py` 与 `events.py`；本区只做
// 类型投影。任何字段增删都必须与后端同时改，否则 SSE 负载校验会失败
// （PRD 13.3 末段：未知版本或负载校验失败显示协议错误，不静默降级）。
// =====================================================================

/** 事件/负载 schema 版本（PRD 13.3）。 */
export const RUN_SCHEMA_VERSION = 2;

/** 用户输入长度上限（PRD 13.2 / DEFAULT_USER_INPUT_MAX_CHARS）。 */
export const USER_INPUT_MAX_CHARS = 32000;

/** run 与 task 共用同一套状态取值（PRD 12.3，11 值）。 */
export type RunStatus =
  | "queued"
  | "running"
  | "waiting_children"
  | "waiting_approval"
  | "waiting_input"
  | "interrupted"
  | "failed"
  | "cancelling"
  | "cancelled"
  | "completed"
  | "completed_with_gaps";

export type TaskStatus = RunStatus;

/** run 已进入终态（不再有新事件）。 */
export const TERMINAL_RUN_STATUSES: RunStatus[] = [
  "failed",
  "cancelled",
  "completed",
  "completed_with_gaps",
];

/** 子任务正在等待外部输入（卡片保持可见，PRD 14.3）。 */
export const WAITING_RUN_STATUSES: RunStatus[] = [
  "waiting_approval",
  "waiting_input",
  "waiting_children",
];

export type AgentKind =
  | "intent"
  | "planner"
  | "general"
  | "rag"
  | "web"
  | "synthesis"
  | "verifier"
  | "approval_reviewer"
  | "memory_extractor";

/** 权限模式（PRD 10.1，默认请求批准）。 */
export type PermissionMode = "ask" | "auto" | "full";

/** 思考深度（PRD 9.1，默认关闭思考）。 */
export type ReasoningLevel = "off" | "low" | "high" | "max";

export type ApprovalDecisionKind = "allow_once" | "allow_scope" | "deny";
export type ApprovalStatus = "pending" | "approved" | "denied" | "superseded";
export type InputRequestStatus = "pending" | "answered" | "superseded";

/** 引用/答案依据的来源类型（PRD 8.3，6 值；决定 [S]/[W]/[M]/[H] 前缀）。 */
export type SourceType = "knowledge" | "web" | "memory" | "history" | "catalog" | "product";

/** 证据来源类型（PRD 8.1，8 值；后两者不产生文档引用编号）。 */
export type EvidenceSourceType = SourceType | "general_knowledge" | "proposal";

export type VerifierVerdict = "pass" | "revise" | "research" | "clarify" | "insufficient";

/** 核对项目（PRD 8.2，八项）。 */
export type VerificationCheck =
  | "requirement_coverage"
  | "factual_support"
  | "numbers_and_units"
  | "version_and_scope"
  | "source_conflict"
  | "internal_fact"
  | "citation_validity"
  | "material_status";

export type VerificationSeverity = "low" | "medium" | "high";

/** 已保存引用的来源状态（知识被删/工作区归档时保留片段并标记，PRD 14.4）。 */
export type CitationSourceStatus = "active" | "deleted" | "archived";

export type IntentKind =
  | "general"
  | "research"
  | "product"
  | "catalog"
  | "memory"
  | "clarification";

export type Freshness = "none" | "current" | "date_range";

/** 20 类 SSE 事件 type（PRD 13.3，见架构文档 D1）。 */
export type EventType =
  | "run.status"
  | "route.decided"
  | "plan.updated"
  | "agent.started"
  | "agent.completed"
  | "model.reasoning"
  | "tool.started"
  | "tool.completed"
  | "approval.requested"
  | "approval.resolved"
  | "input.requested"
  | "input.resolved"
  | "evidence.added"
  | "verification.updated"
  | "answer.meta"
  | "answer.delta"
  | "run.completed"
  | "run.failed"
  | "run.cancelled"
  | "memory.updated"
  | "memory.failed";

/** 订阅据此关闭的事件（终态帧）。 */
export const CLOSING_EVENT_TYPES: EventType[] = [
  "run.completed",
  "run.failed",
  "run.cancelled",
];

// ---------------------------------------------------------------- 契约对象

/**
 * 引用条目（PRD 8.3）：稳定编号 + 来源身份 + 保存片段 + 来源状态。
 *
 * 后 7 个字段是后端**纯追加**（集成 t5/t7，captain 已批准）：run 架构没有旧
 * `/api/chat` 的 `meta.sources`，main 展示过的「已筛选资料」信息只能经
 * `answer.meta.citations` 到达界面。**全部可选**：旧记录/旧后端不带这些键，
 * 缺省时界面**不显示**该信息（不填 0、不编造分数，PRD §8.3）。
 */
export interface CitationEntry {
  label: string;
  source_type: SourceType;
  evidence_id: string;
  workspace_id: string;
  workspace_name: string;
  source_ref: string;
  snippet: string;
  source_status: CitationSourceStatus;
  /** 本条来源是否经过相关性筛选（PRD §8.3「已筛选资料」）；缺省视为未筛选。 */
  relevance_checked?: boolean;
  /** 筛选服务的排序分数（例如 0.812）；未启用筛选或服务未给出时为 null（**不填 0**）。 */
  rerank_score?: number | null;
  /** 分数类型："rerank" = 相关性筛选排序分数；其余情况为空串。 */
  score_type?: string;
  /** 产生该排序分数的校准档案 id；未筛选时为空串。 */
  profile_id?: string;
  /** 证据视图载荷哈希（成员集合与评分正文的稳定指纹）；未筛选时为空串。 */
  evidence_hash?: string;
  /** 组合成员引用（与 `MemberRef` 同形，后端 `MemberRef.to_dict()` 原样投影）。 */
  member_refs?: MemberRef[];
  /**
   * **原始向量分数**（检索阶段的相似度；PRD §304：不得存入重排分数）。
   *
   * 缺少分数时保留来源类型；仅明确标记为 lexical 的记录显示「词法命中」。
   * **绝不**编造 0%，也绝不把 `rerank_score` 写进这个字段。
   */
  similarity?: number | null;
  match_type?: Source["match_type"];
  /**
   * 证据所属**章节标题**（人读的位置，如「3.3 记忆管理 > 3.3.5 记忆关联与概念标签」）。
   *
   * 与 `source_ref`（文档 ID / 版本 / block ID，仅供定位）分开表达：来源面板显示正文 +
   * 章节标题，内部标识只用于定位。旧记录缺该键时为空串（界面不显示标题行）。
   */
  heading_path?: string;
  /** 证据所属文档或网页的完整标题；旧记录缺省为空串。 */
  document_title?: string;
}

export interface WebReadRange {
  offset: number;
  length: number;
  total: number;
  complete: boolean;
}

/** 证据对象（PRD 8.1，12 字段）。 */
export interface Evidence {
  evidence_id: string;
  workspace_id: string;
  run_id: string;
  task_id: string | null;
  source_type: EvidenceSourceType;
  source_ref: string;
  content: string;
  content_hash: string;
  document_revision: string;
  retrieved_at: string;
  supported_requirements: string[];
  limitations: string[];
}

export interface RouteDecision {
  question: string;
  resolved_question: string;
  intent: IntentKind;
  needs_knowledge: boolean;
  needs_web: boolean;
  needs_history: boolean;
  needs_general: boolean;
  freshness: Freshness;
  entities: string[];
  scope_evidence_ids: string[];
  unresolved_terms: string[];
  reason_summary: string;
}

export interface VerificationIssue {
  check: VerificationCheck;
  requirement_ids: string[];
  statement: string;
  evidence_ids: string[];
  required_action: string;
  severity: VerificationSeverity;
}

export interface VerificationResult {
  verdict: VerifierVerdict;
  issues: VerificationIssue[];
  covered_requirements: string[];
  uncovered_requirements: string[];
  open_questions: string[];
  conflicts: string[];
  checked_at: string;
}

export interface AgentError {
  error_code: string;
  message: string;
  node: string;
  operation_id: string | null;
  resumable: boolean;
  required_action: string;
  created_at: string;
}

export interface AnswerArtifact {
  answer_id: string;
  run_id: string;
  workspace_id: string;
  content: string;
  citations: CitationEntry[];
  verification: VerificationResult | null;
  content_hash: string;
  answer_basis: AnswerBasis;
  basis_components: SourceType[];
  status: RunStatus;
  created_at: string;
}

/** 批准卡片的作用范围（「允许本轮同类操作」的匹配键，四项全等）。 */
export interface ApprovalScope {
  run_id: string;
  tool_category: string;
  question_scope: string;
  rule_version: string;
}

/** 批准卡片展示字段（PRD 14.3：目的/工具/查询或 URL/日期范围/所属子任务/脱敏提示）。 */
export interface ApprovalDisplay {
  purpose: string;
  tool_name: string;
  query_or_urls: string[];
  date_range: string;
  task_question: string;
  redaction_notice: string;
  parameters: Record<string, unknown>;
}

/** 批准请求（PRD 10.2，11 字段 + 展示与范围）。 */
export interface ApprovalRequest {
  approval_id: string;
  run_id: string;
  task_id: string | null;
  interrupt_id: string;
  operation_id: string;
  args_hash: string;
  context_epoch: number;
  rule_version: string;
  display: ApprovalDisplay;
  status: ApprovalStatus;
  decision: unknown;
  scope: ApprovalScope | null;
  superseded_reason: string;
  created_at: string;
  resolved_at: string | null;
}

/** 澄清请求（PRD 13.2）。 */
export interface InputRequest {
  input_request_id: string;
  run_id: string;
  task_id: string | null;
  question: string;
  options: string[];
  reason: string;
  status: InputRequestStatus;
  answer: string | null;
  question_version: number;
  created_at: string;
  resolved_at: string | null;
}

// ---------------------------------------------------------------- run 摘要

export interface TaskSummary {
  task_id: string;
  agent_kind: string;
  question: string;
  status: TaskStatus;
  depends_on: string[];
  requirement_ids: string[];
  thread_id: string | null;
  result_ref: string | null;
  /**
   * 父任务 id（PRD 14.2 父子任务）。
   *
   * 只有 run 详情（`RunDetail.tasks[]`）会带；`plan.updated` 事件的负载里没有该字段
   * （后端固定契约未改），所以事件流阶段先用 `depends_on` 近似，快照到达后再补正。
   */
  parent_task_id: string | null;
}

/** 图片引用形态（后端 `app/api/run_images.py::_kind_of`）。 */
export type RunImageKind = "data_uri" | "url" | "invalid";

/**
 * 一条图片引用的处理结果（PRD 13.2「图片引用」）。
 *
 * `status` 是**如实上报**的处理程度，界面必须照它显示，不得把非 `described` 说成已生效：
 * - `described`：原图已缓存 + 描述已入库 → `verify_image` 本轮可核对；
 * - `cached`：原图已缓存、任务已登记，但**未识图**（视觉未配置），待续跑补做；
 * - `pending_vision`：登记成功但流水线报告处理失败（例如识图模型报错）；
 * - `unavailable`：引用形态可识别，但当前环境取不到字节（未配置下载通道）；
 * - `rejected`：引用本身不可用（非法 data URI / 非 http(s) / 超出张数上限）。
 */
export type RunImageStatus =
  | "described"
  | "cached"
  | "pending_vision"
  | "unavailable"
  | "rejected";

export interface RunImageRef {
  image_id: string;
  block_id: string;
  content_hash: string;
  kind: RunImageKind | string;
  status: RunImageStatus | string;
  reason: string;
  src_ref: string;
  recorded_at: string;
}

export interface RunSummary {
  run_id: string;
  conversation_id: string;
  workspace_id: string;
  status: RunStatus;
  permission_mode: PermissionMode;
  reasoning_level: ReasoningLevel;
  answer_id: string | null;
  error_code: string | null;
  stop_reason: string | null;
  resumable: boolean;
  context_epoch: number;
  created_at: string;
  updated_at: string;
}

/** `GET/POST .../runs/{id}` 详情：状态 + 子任务 + 待处理卡片 + 末事件位置。 */
export interface RunDetail extends RunSummary {
  events_url: string;
  tasks: TaskSummary[];
  approvals: ApprovalRequest[];
  inputs: InputRequest[];
  /** 本次 run 附带的图片引用逐条处理结果（PRD 13.2；后端如实上报）。 */
  image_refs: RunImageRef[];
  /** 末事件位置：刷新后从这里继续补读（PRD 14.3）。 */
  last_event_seq: number;
  // Last-Event-ID 的服务端等价物见 `event-log?after=`。
}

// ---------------------------------------------------------------- 事件负载

export interface ToolStartedPayload {
  tool_name: string;
  args_summary: string;
  operation_id: string;
  task_id: string | null;
  summary?: string;
}

export interface ProcessSource {
  title: string;
  text: string;
}

export interface ModelReasoningPayload {
  step_id: string;
  stream_id: string;
  title: string;
  text: string;
}

export interface ToolCompletedPayload {
  tool_name: string;
  operation_id: string;
  status: string;
  result_count: number;
  source_count: number;
  error_code: string;
  summary?: string;
  details?: string[];
  sources?: ProcessSource[];
}

export interface RunStatusPayload {
  model_call?: { call_id: string; ended_at: string | null; outcome: string; usage: Record<string, number> | null } | null;
  status: RunStatus;
  phase: string;
  message: string;
}

export interface PlanTaskSummary {
  task_id: string;
  question: string;
  agent_kind: AgentKind;
  status: TaskStatus;
  depends_on: string[];
  requirement_ids: string[];
}

export interface PlanUpdatedPayload {
  tasks: PlanTaskSummary[];
  covered_requirements: string[];
  uncovered_requirements: string[];
}

export interface AgentStartedPayload {
  agent_kind: AgentKind;
  thread_id: string;
  task_id: string | null;
  question: string;
}

export interface AgentCompletedPayload {
  agent_kind: AgentKind;
  status: TaskStatus;
  summary: string;
}

export interface EvidenceSummary {
  evidence_id: string;
  source_type: EvidenceSourceType;
  title: string;
  source_ref: string;
  snippet: string;
}

export interface EvidenceAddedPayload {
  items: EvidenceSummary[];
}

export interface ApprovalRequestedPayload {
  approval: ApprovalRequest;
}

export interface ApprovalResolvedPayload {
  approval_id: string;
  status: string;
  decision_kind: ApprovalDecisionKind | null;
}

export interface InputRequestedPayload {
  request: InputRequest;
}

export interface InputResolvedPayload {
  input_request_id: string;
  answer: string;
}

export interface VerificationUpdatedPayload {
  result: VerificationResult;
  pending: string[];
}

export interface AnswerMetaPayload {
  answer_id: string;
  answer_basis: AnswerBasis;
  basis_components: SourceType[];
  citations: CitationEntry[];
  content_hash: string;
  status: RunStatus;
}

export interface AnswerDeltaPayload {
  answer_id: string;
  index: number;
  text: string;
  draft?: boolean;
  replace?: boolean;
}

export interface UsageInfo {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  reasoning_tokens: number;
  prompt_cache_hit_tokens: number;
  prompt_cache_miss_tokens: number;
}

export interface RunCompletedPayload {
  answer_id: string;
  status: RunStatus;
  content_hash: string;
  usage: UsageInfo;
  gaps: string[];
}

export interface RunFailedPayload {
  error: AgentError;
}

export interface RunCancelledPayload {
  stop_reason: string;
  resumable: boolean;
}

export interface MemoryUpdatedPayload {
  revision: number;
  invalidation_revision: number;
  entries_added: number;
  entries_updated: number;
}

export interface MemoryFailedPayload {
  reason: string;
  retryable: boolean;
}

/** 统一事件对象（PRD 13.3）：SSE `data` 即本模型的 JSON。 */
export interface RunEvent<T extends EventType = EventType> {
  schema_version: number;
  run_id: string;
  event_seq: number;
  execution_id: string;
  task_id: string | null;
  type: T;
  created_at: string;
  payload: unknown;
  object_id?: string | null;
  version?: number | null;
}

/** 协议错误帧（不在 20 类存储事件里）：界面据此显示协议错误，不静默丢帧。 */
export interface RunProtocolErrorFrame {
  type: "error";
  error_code: string;
  message: string;
}

/** SSE `data` 的两种形态。 */
export type RunStreamFrame = RunEvent | RunProtocolErrorFrame;

/** 带具体负载的强类型事件，供事件归约器使用。 */
export type RunEventTyped =
  | RunEvent<"model.reasoning"> & { payload: ModelReasoningPayload }
  | RunEvent<"run.status"> & { payload: RunStatusPayload }
  | RunEvent<"route.decided"> & { payload: { decision: RouteDecision } }
  | RunEvent<"plan.updated"> & { payload: PlanUpdatedPayload }
  | RunEvent<"agent.started"> & { payload: AgentStartedPayload }
  | RunEvent<"agent.completed"> & { payload: AgentCompletedPayload }
  | RunEvent<"tool.started"> & { payload: ToolStartedPayload }
  | RunEvent<"tool.completed"> & { payload: ToolCompletedPayload }
  | RunEvent<"approval.requested"> & { payload: ApprovalRequestedPayload }
  | RunEvent<"approval.resolved"> & { payload: ApprovalResolvedPayload }
  | RunEvent<"input.requested"> & { payload: InputRequestedPayload }
  | RunEvent<"input.resolved"> & { payload: InputResolvedPayload }
  | RunEvent<"evidence.added"> & { payload: EvidenceAddedPayload }
  | RunEvent<"verification.updated"> & { payload: VerificationUpdatedPayload }
  | RunEvent<"answer.meta"> & { payload: AnswerMetaPayload }
  | RunEvent<"answer.delta"> & { payload: AnswerDeltaPayload }
  | RunEvent<"run.completed"> & { payload: RunCompletedPayload }
  | RunEvent<"run.failed"> & { payload: RunFailedPayload }
  | RunEvent<"run.cancelled"> & { payload: RunCancelledPayload }
  | RunEvent<"memory.updated"> & { payload: MemoryUpdatedPayload }
  | RunEvent<"memory.failed"> & { payload: MemoryFailedPayload };

// ------------------------------------------------------- 过程区域条目模型

/** 过程区域条目种类（PRD 14.2：计划/Agent/工具/资料/批准/澄清/核对/状态）。 */
export type ProcessEntryKind =
  | "route"
  | "plan"
  | "agent"
  | "tool"
  | "evidence"
  | "approval"
  | "input"
  | "verification"
  | "status"
  | "memory"
  | "error";

/** 过程区域条目：全部由**实际事件**生成，不是预先编好的文案。 */
export interface ProcessEntry {
  /** 稳定 key：同一条目被后续事件更新时复用（如工具 running → succeeded）。 */
  id: string;
  kind: ProcessEntryKind;
  text: string;
  /** 事件序号（排序与去重依据）。 */
  seq: number;
  at: string;
  status: "running" | "done" | "failed" | "waiting" | "cancelled";
  taskId: string | null;
  /** 工具名 / Agent 名 / 核对项目键（测试与样式挂钩）。 */
  tag?: string;
  /** 可展开的细项（读过的来源、结果摘要、核对发现）。 */
  details?: string[];
  sources?: ProcessSource[];
  reasoning?: { streamId: string; text: string }[];
}

/** 过程区域内的研究任务视图（父任务展示整体完成情况，PRD 14.2）。 */
export interface ProcessTask {
  taskId: string;
  question: string;
  agentKind: string;
  status: TaskStatus;
  dependsOn: string[];
  parentTaskId: string | null;
  /** 子问题（本任务派生出的子任务 id）。 */
  childTaskIds: string[];
  /** 读过的来源（source_ref + 标题）。 */
  sources: EvidenceSummary[];
  /** 结果摘要（agent.completed.summary）。 */
  resultSummary: string;
}

/** 过程区域的完整归约结果。 */
export interface ProcessState {
  entries: ProcessEntry[];
  tasks: ProcessTask[];
  /** 每次实际执行的起止时间；暂停等待用户或服务恢复的时间不计入用时。 */
  activity: { startedAt: string; endedAt: string | null }[];
  /** 主图与子图的 thread_id（`run:<id>:main` / `run:<id>:task:<task_id>`）。 */
  threads: string[];
  /** 已处理的最大事件序号（去重与断线补读的游标）。 */
  lastSeq: number;
}

/** 批准/澄清卡片在对话中的位置（PRD 14.3：直接位于对话中）。 */
export interface PendingCards {
  approvals: ApprovalRequest[];
  inputs: InputRequest[];
}
