import { expect, test, type Page } from "@playwright/test";

const STATUS_BODY = JSON.stringify({
  app: { name: "everything-rag", version: "0.1.0" },
  config: {
    wizard_completed: false,
    embed_configured: true,
    chat_configured: true,
    vision_configured: false,
    vision_enabled: false,
    search_enabled: false,
    search_configured: false,
  },
  knowledge: { file_count: 0, chunk_count: 0, image_count: 0, last_sync_at: null, needs_rebuild: false },
  privacy: { outbound_state: "local-only", providers: {} },
});

const ANSWER_FRAMES: unknown[] = [
  { type: "meta", answer_basis: "knowledge", policy_version: "rag-policy-v3", query_mode: "CONTENT_QA", sources: [] },
  { type: "token", text: "分支里的回答" },
  { type: "done" },
];

/** 把帧序列编码为线上 SSE 格式。 */
function encodeSse(frames: unknown[]): string {
  return frames.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join("");
}

/**
 * 分支对话的竞态回归：fork 后消息仍在异步装载时用户就发问，
 * 这一轮结果不能追加落库（否则分支历史会被重复写一遍）。
 */
test("分支装载未完成时提问不重复写入历史", async ({ page }: { page: Page }) => {
  const calls: string[] = [];
  let releaseDetail!: () => void;
  const detailGate = new Promise<void>((resolve) => {
    releaseDetail = resolve;
  });

  await page.route("**/api/status", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: STATUS_BODY }),
  );
  await page.route("**/api/settings", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "{}" }),
  );
  await page.route("**/api/workspaces**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([
        { id: "default", name: "默认", is_default: true, created_at: "", updated_at: "", conversation_count: 1 },
      ]),
    }),
  );
  await page.route("**/api/chat", (route) =>
    route.fulfill({ status: 200, contentType: "text/event-stream", body: encodeSse(ANSWER_FRAMES) }),
  );
  await page.route("**/api/conversations**", async (route) => {
    const url = new URL(route.request().url());
    const segments = url.pathname.split("/").filter((part) => part !== "");
    const [, , first, second] = segments;
    calls.push(`${route.request().method()} ${url.pathname}`);
    const json = (body: unknown, status = 200) =>
      route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
    if (first === "archived") return json([]);
    if (second === "fork") {
      return json({ id: "b1", title: "源对话 · 分支", message_count: 1, created_at: "", updated_at: "", parent_id: "p1" }, 201);
    }
    if (second === "messages") return json({ ok: true });
    if (first === "b1") {
      await detailGate; // 卡住「读取分支消息」，制造装载窗口
      return json({
        id: "b1",
        title: "源对话 · 分支",
        created_at: "",
        updated_at: "",
        parent_id: "p1",
        messages: [{ role: "user", content: "继承来的历史", created_at: "" }],
      });
    }
    if (first === "p1") {
      return json({ id: "p1", title: "源对话", created_at: "", updated_at: "", messages: [] });
    }
    return json([{ id: "p1", title: "源对话", message_count: 1, created_at: "", updated_at: "" }]);
  });

  await page.goto("/");
  await page.getByText("源对话", { exact: true }).click({ button: "right" });
  await page.getByRole("menuitem", { name: /Fork 分支对话/ }).click();

  // 装载窗口内提问并等整轮结束
  const input = page.getByRole("textbox", { name: "消息输入框" });
  await input.fill("分支里的追问");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await expect(page.getByText("分支里的回答")).toBeVisible();

  releaseDetail();
  await expect(page.getByText("继承来的历史")).toBeVisible();
  expect(calls.filter((call) => call.includes("/messages"))).toHaveLength(0);
});
