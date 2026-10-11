import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";

const base = resolve(process.cwd(), "../.tmp_probe/agent-eval40-20260930");

function record(directory: string, id: string): { run_id: string; conversation_id: string } {
  return JSON.parse(readFileSync(resolve(base, directory, `${id}.json`), "utf8"));
}

test("真实章节引用显示读取来源并在刷新后保持", async ({ page }) => {
  const item = record("repair-header-10005", "Q11");
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.addInitScript(id => sessionStorage.setItem("everything-rag-conversation", id), item.conversation_id);
  await page.goto("http://127.0.0.1:10005/", { waitUntil: "domcontentloaded" });
  const message = page.locator(`.message-assistant[data-run-id="${item.run_id}"]`);
  for (let attempt = 0; attempt < 2; attempt++) {
    await expect(message.locator(".source-chip").first()).toBeVisible({ timeout: 30000 });
    await message.locator(".source-chip").first().click();
    const panel = page.getByRole("complementary", { name: "来源详情" });
    await expect(panel).toBeVisible();
    await expect(panel.locator(".citation-panel-meta")).toContainText("章节读取");
    await expect(panel).not.toContainText("词法命中");
    await expect(panel).toContainText("Rules");
    await panel.getByRole("button", { name: "关闭来源详情" }).click();
    if (attempt === 0) await page.reload({ waitUntil: "domcontentloaded" });
  }
  expect(errors).toEqual([]);
});

test("真实失败记录显示中文恢复建议且保留继续按钮", async ({ page, request }) => {
  const item = record("results-10002", "Q24");
  const response = await request.get(`http://127.0.0.1:10002/api/workspaces/default/runs/${item.run_id}`);
  expect(response.ok()).toBe(true);
  expect((await response.json()).status).toBe("failed");
  await page.addInitScript(id => sessionStorage.setItem("everything-rag-conversation", id), item.conversation_id);
  await page.goto("http://127.0.0.1:10002/", { waitUntil: "domcontentloaded" });
  const message = page.locator(`.message-assistant[data-run-id="${item.run_id}"]`);
  for (let attempt = 0; attempt < 2; attempt++) {
    await expect(message.getByRole("button", { name: "继续执行", exact: true })).toBeVisible({ timeout: 30000 });
    await expect(message).toContainText("查询嵌入超时");
    await expect(message).toContainText("点击“继续执行”重试当前任务");
    expect(await message.textContent()).not.toContain("POST /api/");
    expect(await message.textContent()).not.toContain("{workspace_id}");
    if (attempt === 0) await page.reload({ waitUntil: "domcontentloaded" });
  }
});
