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

/** /api/status 响应中前端关心的字段（app 元信息 + 隐私基线）。 */
export interface StatusResponse {
  app: { name: string; version: string };
  privacy: { outbound_state: string };
}
