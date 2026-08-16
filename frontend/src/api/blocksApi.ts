import type { BlockInfo } from "../types";

/** 语义块浏览 API：GET /api/blocks（全量返回，前端本地分类/搜索/分页）。 */

export interface BlocksResponse {
  total: number;
  blocks: BlockInfo[];
}

/** 拉取知识库全部语义块（含来源文件 / 标题路径 / 类型元数据）。 */
export async function listBlocks(): Promise<BlocksResponse> {
  let response: Response;
  try {
    response = await fetch("/api/blocks");
  } catch {
    throw new Error("无法连接本地服务，请确认后端已启动");
  }
  if (!response.ok) {
    throw new Error(`本地服务返回异常（HTTP ${response.status}）`);
  }
  return (await response.json()) as BlocksResponse;
}
