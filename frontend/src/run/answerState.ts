import type {
  AnswerBasis,
  CitationEntry,
  RunEventTyped,
  RunStatus,
  Source,
  SourceType,
} from "../types";

/**
 * 答案通道归约器（PRD 14.2 / 8.3）。
 *
 * 事件顺序：`answer.meta`（正文之前先发来源）→ `answer.delta*` → `run.completed`。
 *
 * 两条硬规则：
 * 1. **按片段位置更新**：`answer.delta.index` 决定落位，重复 `index` 覆盖而不是追加，
 *    因此重连补读、重复事件都不会让正文出现重复段落。
 * 2. **引用编号在最终候选答案形成时分配**（后端 `assign_citation_ids`），前端只按
 *    `label` 消费；同一来源复用同一 label，绝不重新编号。
 */

export interface AnswerState {
  answerId: string | null;
  basis: AnswerBasis | null;
  basisComponents: SourceType[];
  citations: CitationEntry[];
  /** 片段按 `index` 落位；稀疏（缺号）时按序拼接，不补空。 */
  chunks: string[];
  status: RunStatus | null;
  gaps: string[];
  contentHash: string;
  completed: boolean;
}

export function emptyAnswerState(): AnswerState {
  return {
    answerId: null,
    basis: null,
    basisComponents: [],
    citations: [],
    chunks: [],
    status: null,
    gaps: [],
    contentHash: "",
    completed: false,
  };
}

/** 正文（按片段位置拼接）。 */
export function answerText(state: AnswerState): string {
  return state.chunks.join("");
}

/** 已落位的片段数（去重后）。 */
export function answerChunkCount(state: AnswerState): number {
  return state.chunks.filter((chunk) => chunk !== "").length;
}

export function applyAnswerEvent(state: AnswerState, event: RunEventTyped): AnswerState {
  switch (event.type) {
    case "answer.meta": {
      const payload = event.payload;
      // 新一轮答案（answer_id 变化）：整块重置，避免上一轮片段串进来。
      const carry = state.answerId === payload.answer_id ? state : emptyAnswerState();
      return {
        ...carry,
        answerId: payload.answer_id,
        basis: payload.answer_basis,
        basisComponents: payload.basis_components ?? [],
        citations: payload.citations ?? [],
        contentHash: payload.content_hash,
        status: payload.status,
      };
    }

    case "answer.delta": {
      const payload = event.payload;
      // 片段属于另一个答案：说明 meta 缺失，按新答案从该位置起收敛。
      const base =
        state.answerId !== null && state.answerId !== payload.answer_id
          ? { ...emptyAnswerState(), answerId: payload.answer_id }
          : state;
      const chunks = base.chunks.slice();
      const index = Math.max(0, payload.index);
      while (chunks.length <= index) chunks.push("");
      // 同一位置重复到达 → 覆盖（幂等），不是追加。
      chunks[index] = payload.text;
      return { ...base, chunks };
    }

    case "run.completed": {
      const payload = event.payload;
      if (state.answerId !== null && payload.answer_id !== "" && state.answerId !== payload.answer_id) {
        // 完成帧指向的答案与已收到片段不一致：保留已收内容，仅记录完成状态。
        return { ...state, completed: true, status: payload.status };
      }
      return {
        ...state,
        answerId: state.answerId ?? payload.answer_id,
        status: payload.status,
        contentHash: payload.content_hash || state.contentHash,
        gaps: payload.gaps ?? [],
        completed: true,
      };
    }

    default:
      return state;
  }
}

export function reduceAnswerEvents(
  events: RunEventTyped[],
  initial: AnswerState = emptyAnswerState(),
): AnswerState {
  return events.reduce<AnswerState>((state, event) => applyAnswerEvent(state, event), initial);
}

/**
 * 图片证据的 `source_ref` 形态：后端把原图挂在 `/api/knowledge/images/<hash>`，
 * 也接受直链图片。命中即按图片证据渲染（PRD 8.1 的图片方案 B）。
 */
const IMAGE_REF = /^(?:\/api\/knowledge\/images\/[a-f0-9]{16,64}|https?:\/\/\S+\.(?:png|jpe?g|webp|gif))$/i;

/**
 * 引用条目 → 正文/脚注用的来源条目。
 *
 * `similarity` 恒为 `null`：`CitationEntry` 仍不携带向量分数，界面必须显示
 * 「词法命中」而不是伪造 0%（PRD 8.3：来源位置可核）。
 *
 * 「已筛选资料」字段（集成 t5）**按原值搬运**：后端把本轮真实筛选结果
 * （`relevance_checked` / `rerank_score` / `score_type` / `profile_id` /
 * `evidence_hash` / `member_refs`）写在引用上，这里只拷贝 —— 记录缺失时保持缺省
 * （`false` / `null` / `""` / `[]`），**绝不补造分数或成员**。
 */
export function citationToSource(citation: CitationEntry): Source {
  const isWeb = citation.source_type === "web";
  const isImage = IMAGE_REF.test(citation.source_ref);
  // 只有后端明确标了「经筛选」才搬分数与成员：旧记录/半截 payload 不得被当成已筛选。
  const filtered = citation.relevance_checked === true;
  return {
    source_id: citation.label,
    block_id: citation.evidence_id,
    text: citation.snippet,
    source_file: citation.source_ref,
    similarity: null,
    heading_path: null,
    anchor: null,
    // 图片证据：面板要走 Markdown 渲染（表格/描述），并给出「查看原图」入口。
    chunk_type: isImage ? "image_description" : "evidence",
    platform: citation.source_type,
    source_type: isWeb ? "web" : citation.source_type,
    url: isWeb ? citation.source_ref : null,
    image_url: isImage ? citation.source_ref : null,
    relevance_checked: filtered,
    rerank_score: filtered ? (citation.rerank_score ?? null) : null,
    score_type: filtered ? (citation.score_type ?? "") : "",
    profile_id: filtered ? (citation.profile_id ?? "") : "",
    evidence_hash: filtered ? (citation.evidence_hash ?? "") : "",
    member_refs: filtered ? (citation.member_refs ?? []) : [],
  };
}

/** 全部引用 → 来源列表（顺序即引用编号顺序，S1/W1/M1/H1）。 */
export function citationsToSources(citations: CitationEntry[]): Source[] {
  return citations.map(citationToSource);
}

/**
 * 已失效来源的提示（PRD 14.4）：知识已删除或原工作区归档时保留已保存片段并标记。
 */
export function citationStatusLabel(citation: CitationEntry): string | null {
  if (citation.source_status === "deleted") return "来源已删除（保留已保存片段）";
  if (citation.source_status === "archived") return "原工作区已归档（保留已保存片段）";
  return null;
}

/** 跨工作区引用提示：移动会话携带的历史引用显示原工作区名称（PRD 14.4）。 */
export function citationWorkspaceLabel(
  citation: CitationEntry,
  currentWorkspaceId: string,
): string | null {
  if (citation.workspace_id === "" || citation.workspace_id === currentWorkspaceId) return null;
  return citation.workspace_name || citation.workspace_id;
}
