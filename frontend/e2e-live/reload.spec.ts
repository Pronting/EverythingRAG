import { expect, test } from "@playwright/test";

test("生成期间刷新并继续停止操作", async ({ page, request }) => {
  await page.goto("/");
  await page.getByRole("textbox", { name: "消息输入框" }).fill("请详细解释十种常见光学现象，每种说明原理和生活中的例子。");
  const created = page.waitForResponse((response) => response.request().method() === "POST" && /\/runs$/.test(response.url()));
  await page.getByRole("button", { name: "发送", exact: true }).click();
  const run = await (await created).json();
  await expect(page.getByRole("button", { name: "停止生成" })).toBeVisible();
  await page.reload();
  await expect(page.getByRole("button", { name: "停止生成" })).toBeVisible();
  await expect(page.locator(".message-user")).toHaveCount(1);
  await expect(page.locator(".message-assistant")).toHaveCount(1);
  await page.getByRole("button", { name: "停止生成" }).click();
  await expect.poll(async () => (await (await request.get(`/api/workspaces/default/runs/${run.run_id}`)).json()).status).toBe("cancelled");
  await page.reload();
  await expect(page.getByTestId("stopped-note")).toBeVisible();
  await expect(page.getByRole("button", { name: "停止生成" })).toBeHidden();
  await expect(page.locator(".message-assistant")).toHaveCount(1);
});
