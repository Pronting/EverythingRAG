import { expect, test, type Page } from "@playwright/test";

/** canned /api/chat SSE 帧：meta（2 个来源块）+ 5 个 token（聚合「你好，世界」）+ done。 */
const CHAT_SSE_FRAMES: unknown[] = [
  {
    type: "meta",
    sources: [
      {
        block_id: "chunk-1",
        text: "这是第一份笔记的片段内容。",
        source_file: "notes.md",
        similarity: 0.92,
        heading_path: "项目笔记 > 概述",
        anchor: "sec-overview",
        chunk_type: "text",
        platform: "local",
      },
      {
        block_id: "chunk-2",
        text: "关于架构决策的第二段参考内容。",
        source_file: "architecture.md",
        similarity: 0.85,
        heading_path: "架构 > 决策",
        anchor: "sec-decision",
        chunk_type: "text",
        platform: "local",
      },
    ],
  },
  { type: "token", text: "你" },
  { type: "token", text: "好" },
  { type: "token", text: "，" },
  { type: "token", text: "世" },
  { type: "token", text: "界" },
  { type: "done" },
];

const ERROR_SSE_FRAMES: unknown[] = [
  { type: "error", message: "知识库尚未建立索引，请先同步文档" },
];

/** 把帧序列编码为线上 SSE 格式：每帧一行 `data: {json}\n\n`。 */
function encodeSse(frames: unknown[]): string {
  return frames.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join("");
}

const STATUS_BODY = JSON.stringify({
  app: { name: "everything-rag", version: "0.1.0" },
  config: {
    wizard_completed: false,
    embed_configured: false,
    chat_configured: false,
    vision_configured: false,
    vision_enabled: false,
  },
  knowledge: {
    file_count: 0,
    chunk_count: 0,
    image_count: 0,
    last_sync_at: null,
    needs_rebuild: false,
  },
  privacy: { outbound_state: "zero_outbound", providers: {} },
});

/** 拦截本地 /api 请求：status 返回 canned JSON，chat 返回 canned SSE。 */
async function stubBackend(page: Page, chatSse: string): Promise<void> {
  await page.route("**/api/status", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: STATUS_BODY }),
  );
  await page.route("**/api/chat", (route) =>
    route.fulfill({ status: 200, contentType: "text/event-stream", body: chatSse }),
  );
}

test("发消息后流式渲染完整回答与来源块", async ({ page }) => {
  await stubBackend(page, encodeSse(CHAT_SSE_FRAMES));

  await page.goto("/");

  // 1. 侧边栏品牌渲染 + 输入框可见
  await expect(page.getByRole("heading", { name: "Everything RAG" })).toBeVisible();
  const input = page.getByRole("textbox", { name: "消息输入框" });
  await expect(input).toBeVisible();

  // 2. 输入消息并发送，断言聚合后的完整文本（多 token 拼接）
  await input.fill("这份文档讲了什么？");
  await page.getByRole("button", { name: "发送" }).click();
  await expect(page.getByText("你好，世界")).toBeVisible();

  // 3. 来源块展示：默认折叠只显示文档名，展开后显示 chunk 段落
  await expect(page.getByText("notes.md")).toBeVisible();
  await expect(page.getByText("架构 > 决策")).toBeVisible();
  await expect(page.getByText("相关度 92%")).toBeVisible();
  // 折叠态：段落文本不可见
  await expect(page.getByText("这是第一份笔记的片段内容。")).not.toBeVisible();
  // 展开第一个来源卡 -> 段落可见
  await page.getByRole("button", { name: /展开来源 notes\.md/ }).click();
  await expect(page.getByText("这是第一份笔记的片段内容。")).toBeVisible();

  // 4. 发送结束后输入框恢复可用
  await expect(input).toBeEnabled();
});

test("error 帧渲染错误消息且不白屏", async ({ page }) => {
  await stubBackend(page, encodeSse(ERROR_SSE_FRAMES));

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Everything RAG" })).toBeVisible();

  const input = page.getByRole("textbox", { name: "消息输入框" });
  await input.fill("测试错误路径");
  await page.getByRole("button", { name: "发送" }).click();

  await expect(page.getByText("知识库尚未建立索引，请先同步文档")).toBeVisible();
  await expect(input).toBeEnabled();
});

test("连续两轮对话的用户问题与回答都完整显示（自动滚动到底部）", async ({ page }) => {
  // 每次 /api/chat 都返回同一组帧（meta + 你好，世界 + done）
  await stubBackend(page, encodeSse(CHAT_SSE_FRAMES));
  await page.goto("/");

  const input = page.getByRole("textbox", { name: "消息输入框" });

  // 第一轮
  await input.fill("第一轮问题");
  await page.getByRole("button", { name: "发送" }).click();
  await expect(page.getByText("第一轮问题")).toBeVisible();
  await expect(page.getByText("你好，世界").first()).toBeVisible();

  // 第二轮：用户问题与回答都应可见（此前因消息区不自动滚动而被折叠区截断）
  await input.fill("第二轮问题");
  await page.getByRole("button", { name: "发送" }).click();
  await expect(page.getByText("第二轮问题")).toBeVisible();
  await expect(page.getByText("你好，世界").nth(1)).toBeVisible();

  // 输入框恢复可用
  await expect(input).toBeEnabled();
});
