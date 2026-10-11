import { expect, test } from "@playwright/test";

test("深度思考完成记忆系统问答", async ({ page, request }) => {
  test.setTimeout(120_000);
  await page.goto("/");
  await page.getByRole("combobox", { name: "权限模式" }).click();
  await page.getByRole("option", { name: "完全访问权限", exact: true }).click();
  await page.getByRole("combobox", { name: "思考深度" }).click();
  await page.getByRole("option", { name: "深度思考", exact: true }).click();
  await page.getByRole("textbox", { name: "消息输入框" }).fill("货拉拉的记忆系统有淘汰策略吗。类似openclaw的那种方案");
  const created = page.waitForResponse((response) => response.request().method() === "POST" && /\/runs$/.test(response.url()));
  const started = Date.now();
  await page.getByRole("button", { name: "发送", exact: true }).click();
  const run = await (await created).json();
  await expect(page.getByRole("button", { name: "停止生成" })).toBeHidden({ timeout: 90_000 });
  expect(Date.now() - started).toBeLessThan(90_000);
  await expect(page.locator(".message-assistant .markdown-body")).not.toBeEmpty();
  const detail = await (await request.get(`/api/workspaces/default/runs/${run.run_id}`)).json();
  expect(detail.status).toMatch(/^completed/);
  expect(detail.reasoning_level).toBe("high");
  const log = await (await request.get(`/api/workspaces/default/runs/${run.run_id}/event-log`)).json();
  const events = log.events as { type: string; payload: { phase?: string } }[];
  expect(events.filter((event) => event.type === "verification.updated")).toHaveLength(0);
  expect(events.filter((event) => event.payload.phase === "synthesis")).toHaveLength(1);
  const synthesis = events.findIndex((event) => event.payload.phase === "synthesis");
  expect(events.slice(synthesis + 1).filter((event) => event.type === "tool.started")).toHaveLength(0);
});
