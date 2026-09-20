import { expect, test } from "@playwright/test";
import { answerDelta, expectRunOnlyTraffic, runCompleted, runStatus, stubRunBackend } from "./runFixtures";

/** 会话流桩：维护内存 store，模拟创建 / 列表 / 追加 / AI 标题。 */
async function stubConversations(page: import("@playwright/test").Page) {
  const convs: { id: string; title: string; message_count: number }[] = [
    { id: "c1", title: "历史对话", message_count: 2 },
  ];
  let nextId = 2;

  const stub = await stubRunBackend(page, {
    conversationId: "c2",
    frames: [runStatus("running"), answerDelta("你好世界"), runCompleted()],
  });

  await page.route("**/api/status", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        app: { name: "everything-rag", version: "0.1.0" },
        config: { wizard_completed: false, embed_configured: true, chat_configured: true, vision_configured: false, vision_enabled: false },
        knowledge: { file_count: 0, chunk_count: 0, image_count: 0, last_sync_at: null, needs_rebuild: false },
        privacy: { outbound_state: "local-only", providers: {} },
      }),
    }),
  );

  // 工作区列表 / status / settings 由 stubRunBackend 提供；这里不要再注册
  // `**/api/workspaces**`（后注册者优先，会把同前缀的 /runs 路由一起吃掉）。

  await page.route("**/api/conversations**", (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    // 只处理列表 / 新建；其余（messages、generate-title）交给后面更具体的路由
    if (path !== "/api/conversations") return route.fallback();
    if (route.request().method() === "GET") {
      route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(convs) });
    } else {
      const created = { id: `c${nextId++}`, title: "新对话", message_count: 0, workspace_id: "default" };
      convs.unshift(created);
      route.fulfill({ status: 201, contentType: "application/json", body: JSON.stringify(created) });
    }
  });

  await page.route("**/api/conversations/*/generate-title", (route) => {
    if (convs[0]) convs[0].title = "AI 标题";
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ title: "AI 标题" }) });
  });

  return stub;
}

test("会话历史：新对话、交换持久化、AI 标题更新侧边栏", async ({ page }) => {
  const stub = await stubConversations(page);
  await page.goto("/");

  // 加载时侧边栏（默认工作区文件夹下）显示历史会话
  await expect(page.getByText("历史对话", { exact: true })).toBeVisible();

  // 新对话 -> 会话列表出现「新对话」项
  await page.getByRole("button", { name: "新对话" }).click();
  await expect(page.getByText("新对话").first()).toBeVisible();

  // 发一条消息 -> 交换完成后 AI 生成标题并更新侧边栏
  const input = page.getByRole("textbox", { name: "消息输入框" });
  await input.fill("总结一下我的知识库");
  await page.getByRole("button", { name: "发送" }).click();
  await expect(page.getByText("你好世界")).toBeVisible();

  // 侧边栏标题由「新对话」变为 AI 生成标题
  await expect(page.getByText("AI 标题")).toBeVisible();
  expectRunOnlyTraffic(stub);
});


test("对话搜索显示正文命中片段并可打开历史", async ({ page }) => {
  const stub = await stubConversations(page);
  await page.route("**/api/conversations?*", route => route.fulfill({ json: [{ id: "c1", title: "历史对话", message_count: 2, workspace_id: "default", match_excerpt: "这里讨论了并行上传的实现" }] }));
  await page.route("**/api/conversations/c1", route => route.fulfill({ json: { id: "c1", title: "历史对话", workspace_id: "default", messages: [{ role: "user", content: "并行上传如何实现？", created_at: "2026-09-05" }] } }));
  await page.goto("/");
  await page.getByRole("textbox", { name: "搜索对话" }).fill("并行上传");
  await expect(page.getByText("这里讨论了并行上传的实现")).toBeVisible();
  await page.locator(".conversation-list").getByText("历史对话").click();
  await expect(page.getByText("并行上传如何实现？")).toBeVisible();
  await page.getByRole("textbox", { name: "搜索对话" }).fill("");
  await expect(page.getByText("这里讨论了并行上传的实现")).not.toBeVisible();
  await expect(page.getByText("隐私基线", { exact: false })).not.toBeVisible();
  // 本用例只读历史，不发问：不应产生任何 run 请求，也绝不触碰旧入口
  expect(stub.runRequestCount()).toBe(0);
  expect(stub.chatRequestCount()).toBe(0);
});
