import { expect, test } from "@playwright/test";

const SSE_FRAMES = [
  { type: "meta", sources: [] },
  { type: "token", text: "你好" },
  { type: "token", text: "世界" },
  { type: "done" },
];

function encodeSse(): string {
  return SSE_FRAMES.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join("");
}

/** 会话流桩：维护内存 store，模拟创建 / 列表 / 追加 / AI 标题。 */
async function stubConversations(page: import("@playwright/test").Page): Promise<void> {
  const convs: { id: string; title: string; message_count: number }[] = [
    { id: "c1", title: "历史对话", message_count: 2 },
  ];
  let nextId = 2;

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

  await page.route("**/api/conversations", (route) => {
    if (route.request().method() === "GET") {
      route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(convs) });
    } else {
      const created = { id: `c${nextId++}`, title: "新对话", message_count: 0 };
      convs.unshift(created);
      route.fulfill({ status: 201, contentType: "application/json", body: JSON.stringify(created) });
    }
  });

  await page.route("**/api/conversations/*/messages", (route) => {
    const target = convs[0];
    if (target) target.message_count += 2;
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(convs[0] ?? {}) });
  });

  await page.route("**/api/conversations/*/generate-title", (route) => {
    if (convs[0]) convs[0].title = "AI 标题";
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ title: "AI 标题" }) });
  });

  await page.route("**/api/chat", (route) =>
    route.fulfill({ status: 200, contentType: "text/event-stream", body: encodeSse() }),
  );
}

test("会话历史：新对话、交换持久化、AI 标题更新侧边栏", async ({ page }) => {
  await stubConversations(page);
  await page.goto("/");

  // 加载时侧边栏显示历史会话
  await expect(page.getByText("历史对话")).toBeVisible();

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
});
