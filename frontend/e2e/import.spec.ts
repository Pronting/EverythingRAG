import { expect, test, type Page } from "@playwright/test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const TASK_ID = "task-123";

/** 打开设置中的知识库管理。 */
async function expandKnowledgeBase(page: Page): Promise<void> {
  await page.getByRole("button", { name: "设置", exact: true }).click();
  await page.getByRole("button", { name: "知识库", exact: true }).click();
}

/** 状态桩：导入前 knowledge 为 0；imported=true 后返回真实计数（验证导入后刷新）。 */
function statusBody(imported: boolean): string {
  return JSON.stringify({
    app: { name: "everything-rag", version: "0.1.0" },
    config: {
      wizard_completed: false,
      embed_configured: true,
      chat_configured: true,
      vision_configured: false,
      vision_enabled: false,
    },
    knowledge: {
      file_count: imported ? 2 : 0,
      chunk_count: imported ? 3 : 0,
      image_count: 0,
      last_sync_at: imported ? "2026-08-12T00:00:00Z" : null,
      needs_rebuild: false,
    },
    privacy: { outbound_state: "local-only", providers: {} },
  });
}

/** 导入任务状态桩：前两次轮询 running（带进度），之后 done（带报告）。 */
function taskBody(running: boolean): string {
  return JSON.stringify({
    task_id: TASK_ID,
    status: running ? "running" : "done",
    progress: {
      files_scanned: 2,
      files_parsed: running ? 1 : 2,
      files_skipped: 0,
      chunks: running ? 1 : 3,
    },
    report: running
      ? null
      : {
          files_scanned: 2,
          files_parsed: 2,
          files_skipped: 0,
          chunks: 3,
          blocks_upserted: 3,
          errors: [],
        },
    error: null,
    created_at: "2026-08-12T00:00:00Z",
    updated_at: "2026-08-12T00:00:00Z",
  });
}

/** 拦截 /api/status 与 /api/import/upload、/api/import/status。 */
async function stubBackend(page: Page): Promise<void> {
  let imported = false;

  await page.route("**/api/status", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: statusBody(imported) }),
  );

  await page.route("**/api/import/upload", (route) =>
    route.fulfill({ status: 202, contentType: "application/json", body: JSON.stringify({ task_id: TASK_ID }) }),
  );

  let pollCount = 0;
  await page.route(`**/api/import/status/${TASK_ID}`, (route) => {
    pollCount += 1;
    const running = pollCount <= 2; // 前两次 running，第三次 done
    if (!running) imported = true; // 完成后标记，供 /api/status 返回新计数
    route.fulfill({ status: 200, contentType: "application/json", body: taskBody(running) });
  });
}

/** 建一个含 2 个 .md（含子目录）的临时文件夹。 */
function makeCorpus(): string {
  const dir = mkdtempSync(join(tmpdir(), "erag-e2e-"));
  mkdirSync(join(dir, "sub"));
  writeFileSync(join(dir, "a.md"), "# A\n\n内容 A\n");
  writeFileSync(join(dir, "sub", "b.md"), "# B\n\n内容 B\n");
  return dir;
}

test("多选文件夹/文件积累待导入清单，开始导入后显示进度与报告", async ({ page }) => {
  const corpus = makeCorpus();
  try {
    await stubBackend(page);
    await page.goto("/");
    await expandKnowledgeBase(page);

    // 初始状态：知识库为空
    await expect(page.locator(".knowledge-summary").getByText("个语义块")).toBeVisible();

    // 选一个文件夹（含子目录 2 个 .md）-> 进入待导入清单
    await page.locator('input[webkitdirectory]').setInputFiles(corpus);
    await expect(page.getByText("待导入 2 个文件")).toBeVisible();

    // 再单独追加一个文件
    await page.getByLabel("选择知识库文件（可多选）").setInputFiles([
      { name: "extra.md", mimeType: "text/markdown", buffer: Buffer.from("# Extra\n\n内容\n") },
    ]);
    await expect(page.getByText("待导入 3 个文件")).toBeVisible();

    // 开始导入 -> 导入中 -> 进度
    await page.getByRole("button", { name: "开始导入" }).click();
    await expect(page.getByRole("button", { name: "导入中…" })).toBeVisible();
    await expect(page.getByText(/扫描 2 · 解析 1 · 跳过 0 · 块 1/)).toBeVisible();

    // 完成报告 + 待导入清单清空
    await expect(page.getByText("导入完成")).toBeVisible();
    await expect(page.getByText(/块 3 · 写入 3/)).toBeVisible();
    await expect(page.getByText(/待导入/)).not.toBeVisible();

    // 导入后 /api/status 刷新：设置页知识库计数更新
    await expect(page.locator(".knowledge-summary strong").nth(1)).toHaveText("3");
    await expect(page.locator(".knowledge-summary strong").first()).toHaveText("2");
  } finally {
    rmSync(corpus, { recursive: true, force: true });
  }
});

test("所选内容中没有 Markdown 时提示", async ({ page }) => {
  const dir = mkdtempSync(join(tmpdir(), "erag-e2e-"));
  try {
    writeFileSync(join(dir, "note.txt"), "hello");
    await stubBackend(page);
    await page.goto("/");
    await expandKnowledgeBase(page);

    await page.locator('input[webkitdirectory]').setInputFiles(dir);

    await expect(page.getByText("所选内容中没有 Markdown 文件")).toBeVisible();
    // 无可导入内容 -> 开始导入按钮禁用
    await expect(page.getByRole("button", { name: "开始导入" })).toBeDisabled();
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("上传失败：后端 400 展示可读错误", async ({ page }) => {
  await page.route("**/api/status", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: statusBody(false) }),
  );
  await page.route("**/api/import/upload", (route) =>
    route.fulfill({
      status: 400,
      contentType: "application/json",
      body: JSON.stringify({ detail: "所选内容中没有 Markdown 文件" }),
    }),
  );

  const corpus = makeCorpus();
  try {
    await page.goto("/");
    await expandKnowledgeBase(page);
    await page.locator('input[webkitdirectory]').setInputFiles(corpus);
    await page.getByRole("button", { name: "开始导入" }).click();

    await expect(page.getByText("所选内容中没有 Markdown 文件")).toBeVisible();
    await expect(page.getByRole("button", { name: "开始导入" })).toBeEnabled();
  } finally {
    rmSync(corpus, { recursive: true, force: true });
  }
});


test("上传阶段立即显示进度，关闭设置后仍持续轮询", async ({ page }) => {
  await stubBackend(page);
  let release!: () => void;
  const gate = new Promise<void>((resolve) => { release = resolve; });
  await page.route("**/api/import/upload", async (route) => {
    await gate;
    await route.fulfill({ status: 202, json: { task_id: TASK_ID } });
  });
  await page.goto("/");
  await expandKnowledgeBase(page);
  await page.getByLabel("选择知识库文件（可多选）").setInputFiles({ name: "a.md", mimeType: "text/markdown", buffer: Buffer.from("# A") });
  await page.getByRole("button", { name: "开始导入" }).click();
  await expect(page.getByRole("progressbar", { name: "文件上传进度" })).toBeVisible();
  await expect(page.getByRole("button", { name: "删除知识库", exact: true })).toBeDisabled();
  await page.getByRole("button", { name: "关闭设置" }).click();
  release();
  await expect.poll(async () => page.locator('.knowledge-summary strong').first().textContent()).toBe("2");
  await expandKnowledgeBase(page);
  await expect(page.getByRole("progressbar", { name: "导入完成" })).toHaveAttribute("aria-valuenow", "100");
});

test("删除需二次确认，取消不发请求，失败可重试；适配手机和深色", async ({ page }) => {
  await stubBackend(page);
  let calls = 0;
  await page.route("**/api/knowledge", (route) => {
    calls += 1;
    return route.fulfill(calls === 1 ? { status: 409, json: { detail: "导入仍在进行，请稍后重试" } } : { json: { deleted_chunks: 3 } });
  });
  await page.goto("/");
  await expandKnowledgeBase(page);
  for (const width of [1280, 390]) {
    await page.setViewportSize({ width, height: 844 });
    await page.evaluate((theme) => { document.documentElement.dataset.theme = theme; }, width === 390 ? "dark" : "light");
    await page.getByRole("button", { name: "删除知识库", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "确认删除知识库？" });
    await expect(dialog).toBeVisible();
    const box = await dialog.boundingBox();
    expect(box!.x).toBeGreaterThanOrEqual(0);
    expect(box!.x + box!.width).toBeLessThanOrEqual(width);
    await page.screenshot({ path: `test-results/knowledge-confirm-${width}.png` });
    await dialog.getByRole("button", { name: "取消" }).click();
    await page.screenshot({ path: `test-results/knowledge-settings-${width}.png` });
    expect(calls).toBe(0);
  }
  await page.getByRole("button", { name: "删除知识库", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "确认删除知识库？" });
  await dialog.getByRole("button", { name: "确认删除", exact: true }).click();
  await expect(dialog.getByRole("alert")).toHaveText("导入仍在进行，请稍后重试");
  await dialog.getByRole("button", { name: "确认删除", exact: true }).click();
  await expect(dialog).not.toBeVisible();
  await expect(page.getByText("知识库已删除，可重新导入文档")).toBeVisible();
  expect(calls).toBe(2);
});


test("旧后端缺少删除接口时显示明确的更新提示", async ({ page }) => {
  await stubBackend(page);
  await page.route("**/api/knowledge", route => route.fulfill({ status: 405, json: { detail: "Method Not Allowed" } }));
  await page.goto("/");
  await expandKnowledgeBase(page);
  await page.getByRole("button", { name: "删除知识库", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "确认删除知识库？" });
  await dialog.getByRole("button", { name: "确认删除", exact: true }).click();
  await expect(dialog.getByRole("alert")).toContainText("当前后端版本不支持删除知识库");
  await expect(dialog.getByRole("button", { name: "确认删除", exact: true })).toBeEnabled();
});
