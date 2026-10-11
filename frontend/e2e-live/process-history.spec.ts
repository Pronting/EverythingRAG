import { expect, test } from "@playwright/test";

const RUN_ID = "run-49146dfb288f4ca0abe1e28731fded80";
const CONVERSATION_ID = "4ef3eae9683a4a9aa9a11cefb17175be";

test("真实执行的大量思考事件在刷新后完整展示", async ({ page, request }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));

  const response = await request.get(`/api/workspaces/default/runs/${RUN_ID}/event-log?after=0`);
  expect(response.ok()).toBe(true);
  const log = await response.json() as {
    events: Array<{ type: string; payload: { stream_id?: string; text?: string } }>;
  };
  const expected = new Map<string, string>();
  for (const event of log.events) {
    if (event.type !== "model.reasoning") continue;
    const streamId = event.payload.stream_id;
    const text = event.payload.text;
    if (streamId === undefined || text === undefined) throw new Error("真实思考事件缺少必要字段");
    expected.set(streamId, (expected.get(streamId) ?? "") + text);
  }
  expect(expected.size).toBe(6);

  await page.goto("/");
  const conversation = page.locator(`[data-conversation-id="${CONVERSATION_ID}"]`);
  await expect(conversation).toBeVisible();
  await conversation.getByRole("button").first().click();

  const assistant = page.locator(`.message-assistant[data-run-id="${RUN_ID}"]`);
  await expect(assistant.getByTestId("process-head")).toBeVisible({ timeout: 30_000 });
  const head = assistant.getByTestId("process-head");
  if (await head.getAttribute("aria-expanded") === "false") await head.click();
  const actual = await assistant.getByTestId("process-reasoning").locator(".process-reasoning-body").allTextContents();
  expect(actual.sort()).toEqual([...expected.values()].sort());
  expect(errors).toEqual([]);

  await page.reload();
  await expect(assistant.getByTestId("process-head")).toBeVisible({ timeout: 30_000 });
  if (await head.getAttribute("aria-expanded") === "false") await head.click();
  const refreshed = await assistant.getByTestId("process-reasoning").locator(".process-reasoning-body").allTextContents();
  expect(refreshed.sort()).toEqual([...expected.values()].sort());
  expect(errors).toEqual([]);
});

test("真实多轮会话可主动读取较早过程且并发点击只请求一次", async ({ page, context, request }) => {
  const conversationId = "489c8e7f76334b3dac8a835058d30644";
  const response = await request.get(`/api/conversations/${conversationId}`);
  expect(response.ok()).toBe(true);
  const conversation = await response.json() as {
    messages: Array<{ role: string; run_id: string | null }>;
  };
  const assistants = conversation.messages.filter((message) => message.role === "assistant");
  expect(assistants.length).toBe(12);
  const firstRunId = assistants[0].run_id;
  if (firstRunId === null) throw new Error("真实会话首轮缺少 run_id");

  await page.addInitScript((id) => sessionStorage.setItem("everything-rag-conversation", id), conversationId);
  const network = await context.newCDPSession(page);
  await network.send("Network.enable");
  await network.send("Network.emulateNetworkConditions", {
    offline: false, latency: 180, downloadThroughput: -1, uploadThroughput: -1,
  });
  let eventLogRequests = 0;
  page.on("request", (item) => {
    if (item.url().includes(`/runs/${firstRunId}/event-log?after=0`)) eventLogRequests += 1;
  });

  await page.goto("/");
  const firstAssistant = page.locator(`.message-assistant[data-run-id="${firstRunId}"]`);
  const loadButton = firstAssistant.getByTestId("process-load-button");
  await expect(loadButton).toBeVisible({ timeout: 20_000 });
  await loadButton.evaluate((element) => {
    (element as HTMLButtonElement).click();
    (element as HTMLButtonElement).click();
  });
  await expect(firstAssistant.getByTestId("process-head")).toBeVisible({ timeout: 20_000 });
  await expect(page.getByTestId("process-head")).toHaveCount(12, { timeout: 40_000 });
  expect(eventLogRequests).toBe(1);

  await page.reload();
  await expect(firstAssistant.getByTestId("process-head")).toBeVisible({ timeout: 40_000 });
});
