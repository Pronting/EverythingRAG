import type { Source, SSEFrame } from "../types";

/** streamChat 回调集：SSE 帧到达时按型分发。 */
export interface StreamChatHandlers {
  onMeta: (sources: Source[]) => void;
  onToken: (text: string) => void;
  onReasoning: (text: string) => void;
  onTool: (tool: string, status: string) => void;
  onDone: () => void;
  onError: (message: string) => void;
}

/** streamChat 可选参数：联网搜索开关、贴图（data URI）。 */
export interface StreamChatOptions {
  webSearch?: boolean;
  images?: string[];
}

/** 多轮对话历史中的一条。 */
export interface HistoryMessage {
  role: "user" | "assistant";
  content: string;
}

const ENDPOINT = "/api/chat";

/**
 * 手写 SSE 客户端：fetch + ReadableStream 增量解码，按 `\n\n` 切帧、
 * 取 `data: ` 后的 JSON 并按 type 分发。不使用 EventSource / 第三方 SSE 库。
 * - 帧内 JSON.parse 失败 / 未知帧型 / 非 data: 行 -> 跳过，不崩溃。
 * - 传入 AbortSignal 可在请求进行中中断（如组件卸载）。
 */
export async function streamChat(
  message: string,
  history: HistoryMessage[],
  handlers: StreamChatHandlers,
  signal?: AbortSignal,
  options?: StreamChatOptions,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({
        message,
        history,
        web_search: options?.webSearch ?? false,
        images: options?.images ?? [],
      }),
      signal,
    });
  } catch (error) {
    if (isAbortError(error)) return;
    handlers.onError(toErrorMessage(error, "无法连接本地服务，请确认后端已启动"));
    return;
  }

  if (!response.ok) {
    handlers.onError(`本地服务返回异常（HTTP ${response.status}），请稍后重试`);
    return;
  }
  if (!response.body) {
    handlers.onError("当前浏览器不支持流式响应，请更换浏览器");
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let sep = buffer.indexOf("\n\n");
      while (sep !== -1) {
        const frameText = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        dispatchFrame(frameText, handlers);
        sep = buffer.indexOf("\n\n");
      }
    }
    // 尾部余量：流结束后残余的最后一帧
    buffer += decoder.decode();
    if (buffer.trim() !== "") {
      dispatchFrame(buffer, handlers);
    }
  } catch (error) {
    if (isAbortError(error)) return;
    handlers.onError(toErrorMessage(error, "接收响应中断，请稍后重试"));
  } finally {
    reader.releaseLock();
  }
}

/** 解析单帧 `data: {json}`，失败静默跳过。 */
function dispatchFrame(frameText: string, handlers: StreamChatHandlers): void {
  const line = frameText.trimStart();
  if (!line.startsWith("data:")) return;
  const payload = line.slice("data:".length).trim();
  if (payload === "") return;

  let frame: SSEFrame;
  try {
    frame = JSON.parse(payload) as SSEFrame;
  } catch {
    return; // 坏帧：跳过不崩
  }

  switch (frame.type) {
    case "meta":
      if (Array.isArray(frame.sources)) handlers.onMeta(frame.sources);
      break;
    case "token":
      if (typeof frame.text === "string") handlers.onToken(frame.text);
      break;
    case "reasoning":
      if (typeof frame.text === "string") handlers.onReasoning(frame.text);
      break;
    case "tool":
      if (typeof frame.tool === "string" && typeof frame.status === "string") {
        handlers.onTool(frame.tool, frame.status);
      }
      break;
    case "done":
      handlers.onDone();
      break;
    case "error":
      if (typeof frame.message === "string") handlers.onError(frame.message);
      break;
    default:
      break; // 未知帧型，忽略
  }
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function toErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message !== "" ? error.message : fallback;
}
