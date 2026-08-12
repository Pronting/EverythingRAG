import { expect, test, type Page } from "@playwright/test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const TASK_ID = "task-123";

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

test("选择文件夹上传导入：显示进度与报告，导入后知识库计数刷新", async ({ page }) => {
  const corpus = makeCorpus();
  try {
    await stubBackend(page);
    await page.goto("/");

    // 初始状态：知识库为空
    await expect(page.getByText("共 0 个语义块 · 尚未导入")).toBeVisible();

    // 通过文件夹选择触发上传导入
    await page.locator('input[type="file"]').setInputFiles(corpus);

    // 按钮进入导入中
    await expect(page.getByRole("button", { name: "导入中…" })).toBeVisible();

    // 实时进度（running 快照）
    await expect(page.getByText(/扫描 2 · 解析 1 · 跳过 0 · 块 1/)).toBeVisible();

    // 完成报告
    await expect(page.getByText("导入完成")).toBeVisible();
    await expect(page.getByText(/块 3 · 写入 3/)).toBeVisible();

    // 导入后 /api/status 刷新：侧边栏知识库计数更新
    await expect(page.getByText("共 3 个语义块 · 已导入")).toBeVisible();
    await expect(page.getByText("2 文档")).toBeVisible();
  } finally {
    rmSync(corpus, { recursive: true, force: true });
  }
});

test("所选文件夹内没有 Markdown 时提示", async ({ page }) => {
  const dir = mkdtempSync(join(tmpdir(), "erag-e2e-"));
  try {
    writeFileSync(join(dir, "note.txt"), "hello");
    await stubBackend(page);
    await page.goto("/");

    await page.locator('input[type="file"]').setInputFiles(dir);

    await expect(page.getByText("所选文件夹内没有 Markdown 文件")).toBeVisible();
    await expect(page.getByRole("button", { name: "选择文件夹" })).toBeEnabled();
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
    await page.locator('input[type="file"]').setInputFiles(corpus);

    await expect(page.getByText("所选内容中没有 Markdown 文件")).toBeVisible();
    await expect(page.getByRole("button", { name: "选择文件夹" })).toBeEnabled();
  } finally {
    rmSync(corpus, { recursive: true, force: true });
  }
});
