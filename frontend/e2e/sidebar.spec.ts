import { expect, test, type Page } from "@playwright/test";

const STATUS_BODY = JSON.stringify({
  app: { name: "everything-rag", version: "0.1.0" },
  config: {
    wizard_completed: false,
    embed_configured: true,
    chat_configured: true,
    vision_configured: false,
    vision_enabled: false,
  },
  knowledge: { file_count: 0, chunk_count: 0, image_count: 0, last_sync_at: null, needs_rebuild: false },
  privacy: { outbound_state: "local-only", providers: {} },
});

const CONVERSATIONS = [
  { id: "c1", title: "红人系统架构", message_count: 3 },
  { id: "c2", title: "简历写法笔记", message_count: 1 },
];

/** 拦截本地 /api 请求：status / conversations / settings 返回 canned JSON。 */
async function stubBackend(page: Page): Promise<void> {
  await page.route("**/api/status", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: STATUS_BODY }),
  );
  await page.route("**/api/conversations", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(CONVERSATIONS),
    }),
  );
  await page.route("**/api/settings", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "{}" }),
  );
}

test("空态：显示标语且输入框可见", async ({ page }) => {
  await stubBackend(page);
  await page.goto("/");

  await expect(page.getByText("随时准备好，知无不言")).toBeVisible();
  await expect(page.getByRole("textbox", { name: "消息输入框" })).toBeVisible();
});

test("侧边栏可折叠/展开", async ({ page }) => {
  await stubBackend(page);
  await page.goto("/");

  // 初始展开：品牌标题可见
  await expect(page.getByRole("heading", { name: "Everything RAG" })).toBeVisible();
  await page.getByRole("button", { name: "收起侧边栏" }).click();
  // 折叠后：标题隐藏（仅剩图标栏）
  await expect(page.getByRole("heading", { name: "Everything RAG" })).not.toBeVisible();
  await page.getByRole("button", { name: "展开侧边栏" }).click();
  await expect(page.getByRole("heading", { name: "Everything RAG" })).toBeVisible();
});

test("侧边栏可搜索过滤会话", async ({ page }) => {
  await stubBackend(page);
  await page.goto("/");

  await expect(page.getByText("红人系统架构")).toBeVisible();
  await expect(page.getByText("简历写法笔记")).toBeVisible();

  const search = page.getByRole("textbox", { name: "搜索对话" });
  await search.fill("简历");
  await expect(page.getByText("红人系统架构")).not.toBeVisible();
  await expect(page.getByText("简历写法笔记")).toBeVisible();

  await search.fill("不存在的关键词");
  await expect(page.getByText("未找到匹配的会话")).toBeVisible();
});
