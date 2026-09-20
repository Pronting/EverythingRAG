import { expect, test, type Page } from "@playwright/test";
import { answerDelta, runCompleted, runStatus, stubRunBackend } from "./runFixtures";

/**
 * 工作区记忆页（PRD 11.2 / 14.4）。
 *
 * 显示简短条目、类别、来源与更新时间；支持修改、删除、开关与主动保存。
 *
 * ⚠️ 记忆开关的权威是**记忆 API 的 `enabled`**：本文件专门覆盖「工作区视图说
 * memory_enabled=true，但记忆 API 说已关闭」时界面必须显示「已关闭」。
 */

const SIMPLE_FRAMES = [runStatus("running"), answerDelta("回答"), runCompleted()];

interface MemoryEntrySeed {
  id: string;
  kind: string;
  text: string;
  source_conversation_id?: string;
  updated_at?: string;
  status?: string;
}

/** 装一个内存版记忆 API（GET/POST/PATCH/DELETE 全走这里），并记录收到的请求体。 */
async function stubMemory(
  page: Page,
  options: { entries?: MemoryEntrySeed[]; enabled?: boolean } = {},
): Promise<{
  calls: { method: string; path: string; body: unknown }[];
  enabled: () => boolean;
  entries: () => MemoryEntrySeed[];
}> {
  const entries: MemoryEntrySeed[] = options.entries ?? [
    {
      id: "m1",
      kind: "response_preference",
      text: "回答用中文，语气自然",
      source_conversation_id: "c1",
      updated_at: "2026-09-18T09:00:00Z",
      status: "active",
    },
    {
      id: "m2",
      kind: "topic",
      text: "关注风控指标口径",
      source_conversation_id: "",
      updated_at: "2026-09-17T08:00:00Z",
      status: "active",
    },
  ];
  let enabled = options.enabled ?? true;
  const calls: { method: string; path: string; body: unknown }[] = [];

  const view = (): unknown => ({
    workspace_id: "default",
    storage_key: "ws-default",
    schema_version: 1,
    enabled,
    revision: 3,
    invalidation_revision: 1,
    updated_at: "2026-09-18T09:00:00Z",
    entries,
    invalidations: [],
    processed_cursors: {},
    applied_operations: 0,
    snapshot_preview: {
      revision: 3,
      invalidation_revision: 1,
      hash: "h",
      chars: 40,
      truncated: false,
      overflow_count: 0,
      needs_tidy: false,
    },
    blocked_source_conversations: [],
  });

  await page.route("**/api/workspaces/*/memory**", (route) => {
    const url = new URL(route.request().url());
    const method = route.request().method();
    const raw = route.request().postData();
    const body: unknown = raw === null || raw === "" ? null : (JSON.parse(raw) as unknown);
    calls.push({ method, path: url.pathname, body });
    const json = (payload: unknown, status = 200) =>
      route.fulfill({ status, contentType: "application/json", body: JSON.stringify(payload) });

    const suffix = url.pathname.split("/memory")[1] ?? "";
    const memoryId = suffix.replace(/^\//, "");

    if (method === "GET" && memoryId === "") return json(view());
    if (method === "PATCH" && memoryId === "") {
      enabled = Boolean((body as { enabled?: boolean }).enabled);
      return json({ applied: true, reason: "switch", enabled, revision: 4, invalidation_revision: 2 });
    }
    if (method === "POST") {
      const text = String((body as { text?: string }).text ?? "");
      entries.push({
        id: `m${entries.length + 1}`,
        kind: "topic",
        text,
        source_conversation_id: "",
        updated_at: "2026-09-19T10:00:00Z",
        status: "active",
      });
      return json({ applied: true, reason: "added", revision: 5, entry: entries[entries.length - 1] }, 201);
    }
    if (method === "PATCH" && memoryId !== "") {
      const target = entries.find((entry) => entry.id === memoryId);
      const text = (body as { text?: string | null }).text;
      if (target !== undefined && typeof text === "string" && text !== "") target.text = text;
      return json({ applied: true, reason: "corrected", revision: 6, entry: target ?? null });
    }
    if (method === "DELETE" && memoryId !== "") {
      const index = entries.findIndex((entry) => entry.id === memoryId);
      if (index >= 0) entries.splice(index, 1);
      return json({ applied: true, reason: "forgotten", revision: 7 });
    }
    return json({}, 404);
  });

  return { calls, enabled: () => enabled, entries: () => entries };
}

test("记忆页显示条目的类别、来源与更新时间", async ({ page }) => {
  await stubRunBackend(page, { frames: SIMPLE_FRAMES });
  await stubMemory(page, {
    entries: [
      {
        id: "m1",
        kind: "response_preference",
        text: "回答用中文，语气自然",
        source_conversation_id: "c1",
        updated_at: "2026-09-18T09:00:00Z",
      },
      {
        id: "m2",
        kind: "confirmed_conclusion",
        text: "风控口径以内部规范为准",
        source_conversation_id: "",
        updated_at: "2026-09-17T08:00:00Z",
      },
    ],
  });

  await page.goto("/");
  await page.getByTestId("open-memory").click();

  await expect(page.getByTestId("memory-panel")).toBeVisible();
  await expect(page.getByTestId("memory-text-m1")).toHaveText("回答用中文，语气自然");
  await expect(page.getByTestId("memory-kind-m1")).toHaveText("回答要求");
  await expect(page.getByTestId("memory-source-m1")).toContainText("c1");
  await expect(page.getByTestId("memory-updated-m1")).toContainText("2026-09-18T09:00:00Z");
  await expect(page.getByTestId("memory-kind-m2")).toHaveText("已确认结论");
  await expect(page.getByTestId("memory-source-m2")).toContainText("用户主动保存");
});

test("记忆开关读到 API 的 enabled 并可用关闭", async ({ page }) => {
  const stub = await stubMemory(page, { enabled: true });
  await page.goto("/");
  await page.getByTestId("open-memory").click();

  // 开关状态只认记忆 API 的 enabled
  await expect(page.getByTestId("memory-enabled")).toBeChecked();
  await expect(page.getByTestId("memory-enabled-state")).toHaveText("已开启");

  // 勾选框状态来自后端 `enabled`，切换要等 PATCH → 重新 GET 才变；用 click 而非
  // uncheck（Playwright 的 uncheck 会立即校验 DOM 状态，异步提交必然误判）。
  await page.getByTestId("memory-enabled").click();
  await expect(page.getByTestId("memory-enabled-state")).toHaveText("已关闭");
  expect(stub.enabled()).toBe(false);
  const patch = stub.calls.find((call) => call.method === "PATCH");
  expect(patch?.body).toMatchObject({ enabled: false, expected_revision: 3 });
});

test("记忆开关陷阱：后端工作区视图 enabled 恒为 true 时不能据此显示", async ({ page }) => {
  // 模拟「WorkspaceContextView.memory_enabled 恒为 true（未接线）」而记忆 API 已关闭。
  await stubRunBackend(page, { frames: SIMPLE_FRAMES });
  await stubMemory(page, { enabled: false });

  await page.goto("/");
  // 工作区相关接口仍会返回 memory_enabled: true 之类的字段
  await page.route("**/api/workspaces", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([
        {
          id: "default",
          name: "默认",
          is_default: true,
          created_at: "",
          updated_at: "",
          conversation_count: 0,
          memory_enabled: true,
        },
      ]),
    }),
  );
  await page.getByTestId("open-memory").click();
  await expect(page.getByTestId("memory-enabled-state")).toHaveText("已关闭");
  await expect(page.getByTestId("memory-enabled")).not.toBeChecked();
});

test("主动保存新增记忆条目", async ({ page }) => {
  const stub = await stubMemory(page, { entries: [] });
  await page.goto("/");
  await page.getByTestId("open-memory").click();

  await expect(page.getByTestId("memory-empty")).toBeVisible();
  await page.getByTestId("memory-new-text").fill("以后回答先给结论");
  await page.getByTestId("memory-add").click();

  await expect(page.getByTestId("memory-list")).toContainText("以后回答先给结论");
  const post = stub.calls.find((call) => call.method === "POST");
  expect(post?.body).toMatchObject({ text: "以后回答先给结论", expected_revision: 3 });
});

test("修改记忆条目的内容", async ({ page }) => {
  const stub = await stubMemory(page);
  await page.goto("/");
  await page.getByTestId("open-memory").click();

  await page.getByTestId("memory-edit-m1").click();
  const editor = page.getByTestId("memory-edit-input-m1");
  await expect(editor).toHaveValue("回答用中文，语气自然");
  await editor.fill("回答用中文，简洁一点");
  await page.getByTestId("memory-save-m1").click();

  await expect(page.getByTestId("memory-text-m1")).toHaveText("回答用中文，简洁一点");
  const patch = stub.calls.find((call) => call.method === "PATCH" && call.path.endsWith("/m1"));
  expect(patch?.body).toMatchObject({ text: "回答用中文，简洁一点", expected_revision: 3 });
});

test("删除记忆条目", async ({ page }) => {
  const stub = await stubMemory(page);
  await page.goto("/");
  await page.getByTestId("open-memory").click();

  await expect(page.getByTestId("memory-item-m2")).toBeVisible();
  await page.getByTestId("memory-delete-m2").click();

  await expect(page.getByTestId("memory-item-m2")).toHaveCount(0);
  const del = stub.calls.find((call) => call.method === "DELETE");
  expect(del?.path).toContain("/m2");
  expect(stub.entries().some((entry) => entry.id === "m2")).toBe(false);
});

test("记忆 API 冲突时显示可读错误，不静默失败", async ({ page }) => {
  await stubMemory(page);
  await page.route("**/api/workspaces/*/memory/*", (route) =>
    route.fulfill({
      status: 409,
      contentType: "application/json",
      body: JSON.stringify({ detail: "revision 冲突" }),
    }),
  );
  await page.goto("/");
  await page.getByTestId("open-memory").click();

  await page.getByTestId("memory-delete-m1").click();
  await expect(page.getByTestId("memory-error")).toContainText("记忆已被其它操作修改");
  // 冲突后条目仍在（没有假装删除成功）
  await expect(page.getByTestId("memory-item-m1")).toBeVisible();
});
