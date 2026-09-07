import { expect, test, type Page } from "@playwright/test";

/** canned /api/chat SSE 帧：meta（2 个来源块）+ 5 个 token（聚合「你好，世界」）+ done。 */
const CHAT_SSE_FRAMES: unknown[] = [
  {
    type: "meta",
    answer_basis: "knowledge",
    policy_version: "rag-policy-v3",
    query_mode: "CONTENT_QA",
    sources: [
      {
        source_id: "S1",
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
        source_id: "S2",
        block_id: "chunk-2",
        text: "关于架构决策的第二段参考内容。",
        source_file: "architecture.md",
        similarity: null,
        match_type: "lexical",
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

test("图片引用展示表格证据并可打开缓存原图", async ({ page }) => {
  await stubBackend(page, encodeSse([
    { type: "meta", answer_basis: "knowledge", policy_version: "rag-policy-v3", query_mode: "CONTENT_QA",
      sources: [{ source_id: "S1", block_id: "image", source_file: "黑五.md", heading_path: "成绩",
        chunk_type: "image_description", platform: "local", similarity: .8, anchor: null,
        image_url: "/api/knowledge/images/aaaaaaaaaaaaaaaa",
        text: "|日期|GMV|\n|---|---|\n|2024-12-02|4826万|" }] },
    {type: "token", text: "数据见[S1]。"}, {type: "done"},
  ]));
  await page.goto("/");
  await page.getByRole("textbox", {name: "消息输入框"}).fill("查看图表数据");
  await page.getByRole("button", {name: "发送", exact: true}).click();
  await page.locator(".markdown-body .citation").click();
  const panel = page.getByRole("complementary", {name: "来源详情"});
  await expect(panel.getByRole("link", {name: "查看原图"})).toHaveAttribute("href", "/api/knowledge/images/aaaaaaaaaaaaaaaa");
  await expect(panel.getByRole("cell", {name: "4826万"})).toBeVisible();
});

test("发消息后流式渲染完整回答与来源脚注", async ({ page }) => {
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

  // 3. 来源脚注：一行文件名 chips（不再展示整段 chunk 长文）
  await expect(page.getByText("基于本地知识")).toBeVisible();
  await expect(page.getByText("notes.md")).toBeVisible();
  await expect(page.getByText("architecture.md")).toBeVisible();
  await expect(page.getByText("92%")).toBeVisible();
  await expect(page.getByText("词法命中")).toBeVisible();
  await expect(page.getByText("0%")).not.toBeVisible();
  // 默认不渲染 chunk 正文
  await expect(page.getByText("这是第一份笔记的片段内容。")).not.toBeVisible();
  // 点击来源 chip -> 右侧详情面板展示该 chunk 全文
  await page.getByRole("button", { name: /notes\.md/ }).click();
  await expect(page.locator(".citation-panel")).toBeVisible();
  await expect(
    page.locator(".citation-panel").getByText("这是第一份笔记的片段内容。"),
  ).toBeVisible();

  // 4. 发送结束后输入框恢复可用
  await expect(input).toBeEnabled();
});

test("正文引用序号可悬停预览、点击打开来源详情", async ({ page }) => {
  const CITATION_FRAMES: unknown[] = [
    {
      type: "meta",
      answer_basis: "knowledge",
      policy_version: "rag-policy-v3",
      query_mode: "CONTENT_QA",
      sources: [
        {
          source_id: "S1",
          block_id: "c1",
          text: "引用片段正文内容。",
          source_file: "ref.md",
          similarity: 0.9,
          heading_path: "章节一",
          anchor: null,
          chunk_type: "text",
          platform: "local",
        },
      ],
    },
    { type: "token", text: "答案是" },
    { type: "token", text: " " },
    { type: "token", text: "[S1]" },
    { type: "token", text: "。" },
    { type: "done" },
  ];
  await stubBackend(page, encodeSse(CITATION_FRAMES));
  await page.goto("/");

  const input = page.getByRole("textbox", { name: "消息输入框" });
  await input.fill("测试引用");
  await page.getByRole("button", { name: "发送" }).click();

  // 正文 [S1] 渲染为可交互上标徽章
  const citation = page.locator(".markdown-body .citation");
  await expect(citation).toHaveText("S1");

  // 悬停 -> 悬浮卡片展示文件名 + 摘要
  await citation.hover();
  await expect(page.locator(".citation-tooltip").getByText("ref.md")).toBeVisible();
  await expect(
    page.locator(".citation-tooltip").getByText("引用片段正文内容。"),
  ).toBeVisible();

  // 点击 -> 右侧详情面板展示 chunk 全文
  await citation.click();
  await expect(page.locator(".citation-panel")).toBeVisible();
  await expect(page.locator(".citation-panel").getByText("引用片段正文内容。")).toBeVisible();
  await expect(page.locator(".citation-panel-count")).toContainText("1 / 1");
});

test("相邻引用逐个转徽章，无效序号被移除", async ({ page }) => {
  const ADJACENT_FRAMES: unknown[] = [
    {
      type: "meta",
      answer_basis: "knowledge",
      policy_version: "rag-policy-v3",
      query_mode: "CONTENT_QA",
      sources: [
        {
          source_id: "S1",
          block_id: "a1",
          text: "来源 A 的片段。",
          source_file: "a.md",
          similarity: 0.9,
          heading_path: null,
          anchor: null,
          chunk_type: "text",
          platform: "local",
        },
        {
          source_id: "S2",
          block_id: "a2",
          text: "来源 B 的片段。",
          source_file: "b.md",
          similarity: 0.8,
          heading_path: null,
          anchor: null,
          chunk_type: "text",
          platform: "local",
        },
      ],
    },
    { type: "token", text: "见 " },
    { type: "token", text: "[S1][S2]" },
    { type: "token", text: "，无效引用 " },
    { type: "token", text: "[S9]" },
    { type: "token", text: "。" },
    { type: "done" },
  ];
  await stubBackend(page, encodeSse(ADJACENT_FRAMES));
  await page.goto("/");

  const input = page.getByRole("textbox", { name: "消息输入框" });
  await input.fill("测试相邻引用");
  await page.getByRole("button", { name: "发送" }).click();

  // 相邻的 [S1][S2] 各转成一个徽章，编号与来源一致
  const citations = page.locator(".markdown-body .citation");
  await expect(citations).toHaveCount(2);
  await expect(citations.nth(0)).toHaveText("S1");
  await expect(citations.nth(1)).toHaveText("S2");

  // 不在 meta 白名单的 [S9] 被整体移除
  await expect(page.locator(".markdown-body")).not.toContainText("[S9]");
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

test("即使收到旧 reasoning 帧也不展示或持久化原始思维链", async ({ page }) => {
  const REASONING_FRAMES: unknown[] = [
    { type: "reasoning", text: "让我想想" },
    { type: "reasoning", text: "这个问题" },
    { type: "token", text: "答案是" },
    { type: "token", text: "42" },
    { type: "done" },
  ];
  await stubBackend(page, encodeSse(REASONING_FRAMES));
  await page.goto("/");

  const input = page.getByRole("textbox", { name: "消息输入框" });
  await input.fill("测试思维链");
  await page.getByRole("button", { name: "发送" }).click();

  // 原始 reasoning 被忽略，正文正常显示。
  await expect(page.getByRole("button", { name: /思考过程/ })).not.toBeVisible();
  await expect(page.getByText("让我想想这个问题")).not.toBeVisible();
  await expect(page.getByText("答案是42")).toBeVisible();
});

test("纯标点思维链不渲染「思考过程」块", async ({ page }) => {
  const TRIVIAL_REASONING_FRAMES: unknown[] = [
    { type: "reasoning", text: "，" },
    { type: "reasoning", text: "。" },
    { type: "token", text: "答案是" },
    { type: "token", text: "42" },
    { type: "done" },
  ];
  await stubBackend(page, encodeSse(TRIVIAL_REASONING_FRAMES));
  await page.goto("/");

  const input = page.getByRole("textbox", { name: "消息输入框" });
  await input.fill("测试纯标点思维链");
  await page.getByRole("button", { name: "发送" }).click();

  // 正文正常显示，纯标点的思维链不渲染「思考过程」块
  await expect(page.getByText("答案是42")).toBeVisible();
  await expect(page.getByRole("button", { name: /思考过程/ })).not.toBeVisible();
});


test("生成期间可聚焦输入框并保留下一条草稿，不重复发送", async ({ page }) => {
  await stubBackend(page, encodeSse(CHAT_SSE_FRAMES));
  let release!: () => void;
  const gate = new Promise<void>(resolve => { release = resolve; });
  let requests = 0;
  await page.route("**/api/chat", async route => {
    requests += 1;
    await gate;
    await route.fulfill({ status: 200, contentType: "text/event-stream", body: encodeSse(CHAT_SSE_FRAMES) });
  });
  await page.goto("/");
  const input = page.getByRole("textbox", { name: "消息输入框" });
  await input.fill("第一条问题");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await input.click();
  await expect(input).toBeFocused();
  await input.fill("下一条草稿");
  await input.press("Enter");
  await expect(page.getByRole("button", { name: "发送", exact: true })).toBeDisabled();
  release();
  await expect(page.getByRole("button", { name: "发送", exact: true })).toBeEnabled();
  await expect(input).toHaveValue("下一条草稿");
  expect(requests).toBe(1);
});
