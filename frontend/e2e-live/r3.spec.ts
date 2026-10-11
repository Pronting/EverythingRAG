import { expect, test } from "@playwright/test";

type EventItem = {
  type: string;
  payload: Record<string, unknown>;
};

async function eventLog(request: any, runId: string): Promise<EventItem[]> {
  try {
    const response = await request.get(`/api/workspaces/default/runs/${runId}/event-log`);
    if (!response.ok()) return [];
    const body = await response.json();
    return body.events ?? [];
  } catch {
    return [];
  }
}

async function runStatus(request: any, runId: string): Promise<string> {
  try {
    const response = await request.get(`/api/workspaces/default/runs/${runId}`);
    if (!response.ok()) return "unavailable";
    return (await response.json()).status;
  } catch {
    return "unavailable";
  }
}

test("G8 alternate wording survives backend restart", async ({ page, request }) => {
  test.setTimeout(360_000);
  await page.goto("/");
  await page.getByLabel("新对话", { exact: true }).click();
  await page.getByRole("combobox", { name: "权限模式" }).click();
  await page.getByRole("option", { name: "完全访问权限", exact: true }).click();
  await expect(page.getByTestId("model-picker")).toContainText("deepseek/deepseek-v4.1-flash");
  await page.getByTestId("model-picker").click();
  await page.getByTestId("reasoning-menu").click();
  await page.getByTestId("reasoning-option-low").click();
  await expect(page.getByTestId("model-picker")).toContainText("轻度思考");

  const question =
    "请联网查 AWS 官方文档，核验 Bedrock AgentCore Memory 的 CreateMemory 请求中 eventExpiryDuration 允许填写的最小值、最大值和单位，并给出页面链接。";
  await page.getByRole("textbox", { name: "消息输入框" }).fill(question);
  const created = page.waitForResponse(
    (response) => response.request().method() === "POST" && /\/runs$/.test(response.url()),
  );
  await page.getByRole("button", { name: "发送", exact: true }).click();
  const run = await (await created).json();
  console.log(`R3_RUN_ID=${run.run_id}`);

  await expect
    .poll(async () => {
      const events = await eventLog(request, run.run_id);
      return events.some((event) => event.type === "tool.completed");
    }, { timeout: 120_000, intervals: [500, 1000, 2000] })
    .toBe(true);
  console.log("R3_FIRST_TOOL_COMPLETED");

  await page.waitForTimeout(30_000);
  await expect
    .poll(async () => runStatus(request, run.run_id), {
      timeout: 240_000,
      intervals: [1000, 2000, 5000],
    })
    .toMatch(/^completed/);
  await page.reload();
  await expect(page.locator(".message-assistant .markdown-body").last()).toContainText(/最小(?:值)?\s*3/);
  await expect(page.locator(".message-assistant .markdown-body").last()).toContainText(/最大(?:值)?\s*365/);

  const events = await eventLog(request, run.run_id);
  const started = events.filter((event) => event.type === "tool.started");
  const toolNames = started.map((event) => String(event.payload.tool_name ?? ""));
  expect(toolNames.filter((name) => name === "web_search")).toHaveLength(1);
  expect(toolNames.filter((name) => name === "read_web_page")).toHaveLength(1);
  expect(toolNames).not.toContain("search_knowledge");
  expect(toolNames).not.toContain("read_document");
  const operationIds = started
    .map((event) => String(event.payload.operation_id ?? ""))
    .filter(Boolean);
  expect(new Set(operationIds).size).toBe(operationIds.length);

  const conversation = await (
    await request.get(`/api/conversations/${run.conversation_id}`)
  ).json();
  const assistant = conversation.messages.filter(
    (message: { role: string }) => message.role === "assistant",
  ).at(-1);
  expect(assistant.content).toMatch(/最小(?:值)?\s*3/);
  expect(assistant.content).toMatch(/最大(?:值)?\s*365/);
  const webCitations = (assistant.citations ?? []).filter(
    (citation: { source_type: string }) => citation.source_type === "web",
  );
  expect(webCitations.length).toBeGreaterThan(0);
  expect(new Set(webCitations.map((item: { source_ref: string }) => item.source_ref)).size).toBe(
    webCitations.length,
  );
  expect(webCitations.every((item: { source_ref: string }) => item.source_ref.includes("docs.aws.amazon.com"))).toBe(true);
  await expect(page.getByTestId("process-reasoning")).toHaveCount(0);
});

test("duplicate approval requests resolve once", async ({ page, request }) => {
  test.setTimeout(240_000);
  await page.goto("/");
  await page.getByLabel("新对话", { exact: true }).click();
  await page.getByRole("combobox", { name: "权限模式" }).click();
  await page.getByRole("option", { name: "请求批准", exact: true }).click();
  await page.getByTestId("model-picker").click();
  await page.getByTestId("reasoning-menu").click();
  await page.getByTestId("reasoning-option-low").click();
  await page.getByRole("textbox", { name: "消息输入框" }).fill(
    "请联网查 Python 官方文档，说明 asyncio.TaskGroup 的用途并给出来源。",
  );
  const created = page.waitForResponse(
    (response) => response.request().method() === "POST" && /\/runs$/.test(response.url()),
  );
  await page.getByRole("button", { name: "发送", exact: true }).click();
  const run = await (await created).json();
  await expect(page.getByTestId(/approval-card-/).first()).toBeVisible({ timeout: 60_000 });
  await page.reload();
  await expect(page.getByTestId(/approval-card-/).first()).toBeVisible({ timeout: 15_000 });
  const detail = await (
    await request.get(`/api/workspaces/default/runs/${run.run_id}`)
  ).json();
  const approvalId = detail.approvals[0].approval_id;
  const endpoint = `/api/workspaces/default/runs/${run.run_id}/approvals/${approvalId}`;
  const [first, second] = await Promise.all([
    request.post(endpoint, { data: { decision: "allow_once" } }),
    request.post(endpoint, { data: { decision: "allow_once" } }),
  ]);
  expect(first.status()).toBe(200);
  expect(second.status()).toBe(200);
  const conflict = await request.post(endpoint, { data: { decision: "deny" } });
  expect(conflict.status()).toBe(409);
  const cancelled = await request.post(
    `/api/workspaces/default/runs/${run.run_id}/cancel`,
    { data: { reason: "approval_regression_complete" } },
  );
  expect(cancelled.status()).toBe(200);
  const events = await eventLog(request, run.run_id);
  expect(events.filter((event) => event.type === "approval.resolved")).toHaveLength(1);
});

test("off mode records no reasoning event", async ({ page, request }) => {
  test.setTimeout(180_000);
  await page.goto("/");
  await page.getByLabel("新对话", { exact: true }).click();
  await expect(page.getByTestId("model-picker")).toContainText("关闭思考");
  await page.getByRole("textbox", { name: "消息输入框" }).fill(
    "请联网查 Python 官方文档，说明 asyncio.TaskGroup 的用途并给出来源。",
  );
  const created = page.waitForResponse(
    (response) => response.request().method() === "POST" && /\/runs$/.test(response.url()),
  );
  await page.getByRole("button", { name: "发送", exact: true }).click();
  const run = await (await created).json();
  await expect
    .poll(async () => runStatus(request, run.run_id), { timeout: 150_000 })
    .toMatch(/^completed/);
  const events = await eventLog(request, run.run_id);
  expect(events.filter((event) => event.type === "model.reasoning")).toHaveLength(0);
  expect(
    events.filter(
      (event) =>
        event.type === "tool.started" &&
        ["web_search", "read_web_page", "search_knowledge", "read_document"].includes(
          String(event.payload.tool_name ?? ""),
        ),
    ),
  ).toHaveLength(0);
  await expect(page.getByTestId("process-reasoning")).toHaveCount(0);
});

for (const level of ["high", "max"] as const) {
  test(`${level} keeps web enabled`, async ({ page, request }) => {
    test.setTimeout(120_000);
    await page.goto("/");
    await page.getByLabel("新对话", { exact: true }).click();
    await page.getByRole("combobox", { name: "权限模式" }).click();
    await page.getByRole("option", { name: "完全访问权限", exact: true }).click();
    await page.getByTestId("model-picker").click();
    await page.getByTestId("reasoning-menu").click();
    await page.getByTestId(`reasoning-option-${level}`).click();
    await page.getByRole("textbox", { name: "消息输入框" }).fill(
      "请联网查 Python 官方文档，确认 asyncio.TaskGroup 从哪个 Python 版本开始提供。",
    );
    const created = page.waitForResponse(
      (response) => response.request().method() === "POST" && /\/runs$/.test(response.url()),
    );
    await page.getByRole("button", { name: "发送", exact: true }).click();
    const run = await (await created).json();
    await expect
      .poll(async () => {
        const events = await eventLog(request, run.run_id);
        return events.some(
          (event) =>
            event.type === "tool.started" && event.payload.tool_name === "web_search",
        );
      }, { timeout: 90_000, intervals: [500, 1000, 2000] })
      .toBe(true);
    const events = await eventLog(request, run.run_id);
    expect(
      events.some(
        (event) =>
          event.type === "tool.started" && event.payload.tool_name === "search_knowledge",
      ),
    ).toBe(false);
    const detail = await (
      await request.get(`/api/workspaces/default/runs/${run.run_id}`)
    ).json();
    expect(detail.reasoning_level).toBe(level);
    const cancelled = await request.post(
      `/api/workspaces/default/runs/${run.run_id}/cancel`,
      { data: { reason: `${level}_web_policy_checked` } },
    );
    expect(cancelled.status()).toBe(200);
  });
}
