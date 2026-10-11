import { expect, test, type Page, type Route } from "@playwright/test";

/**
 * 会话移动的前端对齐（PRD 5.4 / 13.2；A27）。
 *
 * ⚠️ **诚实声明（不要删）**：本文件用 `page.route` 造响应，验证的是**请求形状与前端
 * 分支**（是否发了 `confirm`/`expected_context_epoch`、409 是否弹了确认、拖拽是否走了
 * 同一条契约）。它**不能**证明前后端契约在真实服务上可用 —— 真实联通属集成范畴。
 * 这正是该缺陷此前被「Playwright 75 绿」掩盖的原因。
 */

const STATUS_BODY = JSON.stringify({
  app: { name: "everything-rag", version: "0.2.0" },
  config: {
    wizard_completed: false,
    embed_configured: true,
    chat_configured: true,
    vision_configured: false,
    vision_enabled: false,
    search_enabled: false,
    search_configured: false,
  },
  knowledge: {
    file_count: 0,
    chunk_count: 0,
    image_count: 0,
    image_tasks: { pending: 0, done: 0, failed: 0, total: 0 },
    last_sync_at: null,
    needs_rebuild: false,
  },
  privacy: { outbound_state: "local-only", providers: {} },
});

/** 后端 409 文案（逐字取自 `conversation_move.py`）。 */
const CONFIRM_REQUIRED_DETAIL = "移动会话会携带历史内容，请确认后再移动（confirm=true，PRD 5.4）";
const epochConflictDetail = (expected: number, actual: number): string =>
  `会话上下文版本已变化：期望 ${expected}，实际 ${actual}`;

interface MoveCall {
  path: string;
  body: { workspace_id?: string | null; target_index?: number | null; confirm?: boolean; expected_context_epoch?: number | null };
}

interface MoveStubOptions {
  /** 携带历史的消息数（>0 时前端会先弹确认）。 */
  messageCount?: number;
  /** 后端当前 context_epoch。 */
  currentEpoch?: number;
  /** 强制「未确认即 409」，模拟后端认为需要确认（例如存在活动 run）。 */
  requireConfirm?: boolean;
}

interface MoveStub {
  calls: MoveCall[];
  /** 每次建 run 的请求体（断言 `origin_run_id` 真的发出去了）。 */
  runBodies: { origin_run_id?: string | null }[];
  /** 生效中的移动端点 pathname 集合（断言拖拽没打旁路端点）。 */
  paths: () => string[];
  setCurrentEpoch: (value: number) => void;
}

/** 两个工作区 + 一条带历史的会话 + 会真实校验 confirm/epoch 的移动端点。 */
async function stubMoveBackend(page: Page, options: MoveStubOptions = {}): Promise<MoveStub> {
  const calls: MoveCall[] = [];
  const runBodies: { origin_run_id?: string | null }[] = [];
  const messageCount = options.messageCount ?? 6;
  let currentEpoch = options.currentEpoch ?? 3;
  const conversations = [
    {
      id: "c1",
      title: "带历史的会话",
      message_count: messageCount,
      created_at: "2026-09-18T09:00:00Z",
      updated_at: "2026-09-18T09:00:00Z",
      archived: false,
      archived_at: null,
      parent_id: null,
      forked_from_title: null,
      workspace_id: "default",
      sort_index: null,
    },
    {
      id: "c2",
      title: "另一条默认对话",
      message_count: messageCount,
      created_at: "2026-09-18T08:00:00Z",
      updated_at: "2026-09-18T08:00:00Z",
      archived: false,
      archived_at: null,
      parent_id: null,
      forked_from_title: null,
      workspace_id: "default",
      sort_index: null,
    },
  ];

  const json = (route: Route, body: unknown, status = 200) =>
    route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });

  await page.route("**/api/status", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: STATUS_BODY }),
  );
  await page.route("**/api/settings", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "{}" }),
  );
  await page.route("**/api/workspaces", (route) =>
    json(route, [
      { id: "default", name: "默认", is_default: true, created_at: "", updated_at: "", conversation_count: 1 },
      { id: "research", name: "研究", is_default: false, created_at: "", updated_at: "", conversation_count: 0 },
    ]),
  );

  await page.route("**/api/conversations**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const segments = url.pathname.split("/").filter((part) => part !== "");
    // /api/conversations/{id}/{action}
    const [, , id, action] = segments;

    if (action === "move" || action === "workspace") {
      const body = (request.postDataJSON() ?? {}) as MoveCall["body"];
      calls.push({ path: url.pathname, body });
      const target = body.workspace_id ?? null;
      const needsConfirm = options.requireConfirm === true || messageCount > 0;
      if (needsConfirm && body.confirm !== true) {
        return json(route, { detail: CONFIRM_REQUIRED_DETAIL }, 409);
      }
      const expected = body.expected_context_epoch;
      if (typeof expected === "number" && expected !== currentEpoch) {
        return json(route, { detail: epochConflictDetail(expected, currentEpoch) }, 409);
      }
      currentEpoch += 1;
      const row = conversations.find((item) => item.id === (decodeURIComponent(id) || "c1")) ?? conversations[0];
      if (target !== null) row.workspace_id = target;
      return json(route, {
        ...row,
        from_workspace_id: "default",
        context_epoch: currentEpoch,
        previous_context_epoch: currentEpoch - 1,
        stopped_run_ids: ["run-7"],
        superseded_approvals: 1,
        superseded_inputs: 0,
        memory_snapshot_rebound: true,
        permission_rebound: false,
        steps: ["confirmed", "runs_stopped", "app_db_moved"],
      });
    }

    if (request.method() === "POST" && id === undefined) {
      const created = { ...conversations[0], id: "c1", message_count: 0 };
      return json(route, created, 201);
    }
    if (request.method() === "GET" && id === undefined) return json(route, conversations);
    if (id !== undefined) {
      return json(route, { ...conversations[0], messages: [] });
    }
    return json(route, []);
  });

  // run 端点：给「观测 context_epoch」与「origin_run_id」用。
  await page.route("**/api/workspaces/*/runs**", async (route) => {
    const url = new URL(route.request().url());
    const method = route.request().method();
    const match = /\/api\/workspaces\/([^/]+)\/runs(?:\/([^/]+))?(?:\/([^/]+))?/.exec(url.pathname);
    const runId = match?.[2];
    if (runId === undefined) {
      if (method === "GET") {
        return json(route, {
          runs: [
            {
              run_id: "run-5",
              conversation_id: "c1",
              workspace_id: "default",
              status: "completed",
              permission_mode: "ask",
              reasoning_level: "high",
              answer_id: null,
              error_code: null,
              stop_reason: null,
              resumable: false,
              context_epoch: currentEpoch,
              created_at: "",
              updated_at: "",
            },
          ],
        });
      }
      // 创建 run：回一个带 context_epoch 的详情，并把请求体交给测试断言。
      const body = (route.request().postDataJSON() ?? {}) as { origin_run_id?: string | null };
      runBodies.push(body);
      return json(
        route,
        {
          run_id: "run-9",
          conversation_id: "c1",
          workspace_id: "default",
          status: "running",
          permission_mode: "ask",
          reasoning_level: "high",
          answer_id: null,
          error_code: null,
          stop_reason: null,
          resumable: false,
          context_epoch: currentEpoch,
          created_at: "",
          updated_at: "",
          events_url: "/api/workspaces/default/runs/run-9/events",
          tasks: [],
          approvals: [],
          inputs: [],
          image_refs: [],
          last_event_seq: 0,
        },
        202,
      );
    }
    if (match?.[3] === "events") {
      const body = [
        { type: "run.status", payload: { status: "running", phase: "", message: "" } },
        {
          type: "answer.meta",
          payload: { answer_id: "a1", answer_basis: "general", basis_components: ["general"], citations: [], content_hash: "", status: "running" },
        },
        { type: "answer.delta", payload: { answer_id: "a1", index: 0, text: "回答" } },
        {
          type: "run.completed",
          payload: { answer_id: "a1", status: "completed", content_hash: "", usage: {}, gaps: [] },
        },
      ].map((frame, index) => {
        const seq = index + 1;
        return `id: ${seq}\ndata: ${JSON.stringify({
          schema_version: 2,
          run_id: "run-9",
          event_seq: seq,
          execution_id: "e",
          task_id: null,
          created_at: "",
          payload: frame.payload,
          type: frame.type,
        })}\n\n`;
      });
      return route.fulfill({ status: 200, contentType: "text/event-stream", body: `retry: 30000\n\n${body.join("")}` });
    }
    if (match?.[3] === "cancel") return json(route, { run_id: runId, status: "cancelled" });
    return json(route, { run_id: runId ?? "run-9", status: "completed" });
  });

  return {
    calls,
    runBodies,
    paths: () => calls.map((call) => call.path),
    setCurrentEpoch: (value: number) => {
      currentEpoch = value;
    },
  };
}

/** 记录所有请求 pathname：守住「≥1 命中 /runs、0 命中 /api/chat」这条加强口径。 */
function trackPaths(page: Page): string[] {
  const paths: string[] = [];
  page.on("request", (request) => {
    try {
      paths.push(new URL(request.url()).pathname);
    } catch {
      /* data: / about: */
    }
  });
  return paths;
}

/** 与 `runFixtures.expectRunOnlyTraffic` 同一口径（本文件自带桩，故就地实现）。 */
function expectRunOnlyTraffic(paths: string[]): void {
  expect(paths.filter((path) => path.includes("/runs")).length).toBeGreaterThanOrEqual(1);
  expect(paths.filter((path) => path === "/api/chat").length).toBe(0);
}

/**
 * 侧边栏会话条目。
 *
 * 作用域限定在文件夹树里（`.ws-folder`），确保条目一定属于某个工作区 —— 拖拽落点的
 * `workspaceId` 由所在文件夹决定，取扁平列表可能落到没有归属的容器上。
 */
function sidebarItem(page: Page, title: string) {
  return page.locator(".ws-folder").getByText(title).first();
}

/** 侧边栏拖拽：HTML5 drag 需要手动补 mousemove（与 workspaces.spec 同一手法）。 */
async function dragTo(page: Page, from: ReturnType<Page["getByText"]>, target: ReturnType<Page["locator"]>): Promise<void> {
  const source = await from.boundingBox();
  const destination = await target.boundingBox();
  if (source === null || destination === null) throw new Error("拖拽目标不可见");
  await page.mouse.move(source.x + source.width / 2, source.y + source.height / 2);
  await page.mouse.down();
  await page.mouse.move(destination.x + destination.width / 2, destination.y + destination.height / 2, { steps: 8 });
  await page.waitForTimeout(80);
  await page.mouse.move(destination.x + destination.width / 2, destination.y + destination.height / 2, { steps: 3 });
  await page.mouse.up();
  await page.waitForTimeout(80);
}

test("右键移动：未确认前不发请求，确认后请求体带 confirm 与 expected_context_epoch", async ({ page }) => {
  const stub = await stubMoveBackend(page);
  await page.goto("/");

  await sidebarItem(page, "带历史的会话").click({ button: "right" });
  await page.getByRole("menuitem", { name: "移动到工作区" }).click();
  await page.getByRole("menuitem", { name: /^研究/ }).click();

  // ① 未确认前**一个请求都没发**（不得默认静默 confirm:true）
  const dialog = page.getByRole("alertdialog", { name: "移动会话到其他工作区" });
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText("停止该会话正在执行的任务");
  await expect(dialog).toContainText("待批准/待澄清的卡片失效");
  expect(stub.calls).toHaveLength(0);

  // ② 确认后才发，且字段齐全
  await dialog.getByRole("button", { name: "确认移动" }).click();
  await expect.poll(() => stub.calls.length, { message: "确认后应发一次移动请求" }).toBe(1);
  const call = stub.calls[0];
  expect(call.path).toBe("/api/conversations/c1/workspace");
  expect(call.body.confirm).toBe(true);
  // 从未观测到 epoch ⇒ 显式发 null（不是省略、不是编造）
  expect("expected_context_epoch" in call.body).toBe(true);
  expect(call.body.expected_context_epoch).toBeNull();
  expect(call.body.workspace_id).toBe("research");

  // ③ 成功后如实告知影响了什么
  await expect(page.locator(".toast")).toContainText("已移动到「研究」");
  await expect(page.locator(".toast")).toContainText("已停止 1 个进行中的任务");
  await expect(page.locator(".toast")).toContainText("1 个待批准已失效");
});

test("观测到 context_epoch 后，移动请求带上真实观测值", async ({ page }) => {
  // messageCount=0：避免「先确认再发」的交互干扰本次要断言的乐观并发字段。
  const stub = await stubMoveBackend(page, { currentEpoch: 7, messageCount: 0 });
  await page.goto("/");

  // 先发一轮：建 run 的响应带 context_epoch=7 ⇒ 前端记下它
  await page.getByRole("textbox", { name: "消息输入框" }).fill("先问一句");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await expect(page.locator(".markdown-body").first()).toContainText("回答");

  await sidebarItem(page, "带历史的会话").click({ button: "right" });
  await page.getByRole("menuitem", { name: "移动到工作区" }).click();
  await page.getByRole("menuitem", { name: /^研究/ }).click();

  await expect
    .poll(() => stub.calls.length, { message: `calls=${JSON.stringify(stub.calls)}` })
    .toBe(1);
  expect(stub.calls[0].body.expected_context_epoch).toBe(7);
  expect(stub.calls[0].body.workspace_id).toBe("research");
  await expect(page.locator(".toast")).toContainText("已移动到「研究」");
});

test("缺确认的 409：弹出真实确认而不是静默失败", async ({ page }) => {
  // 会话「看起来」没有历史（message_count=0），但后端因存在活动 run 仍要求确认。
  const stub = await stubMoveBackend(page, { messageCount: 0, requireConfirm: true });
  await page.goto("/");

  await sidebarItem(page, "带历史的会话").click({ button: "right" });
  await page.getByRole("menuitem", { name: "移动到工作区" }).click();
  await page.getByRole("menuitem", { name: /^研究/ }).click();

  // 第一次以 confirm=false 发出 → 后端 409 → 界面弹确认（而不是静默失败/toast 之后无声）
  await expect.poll(() => stub.calls.length, { message: "应先发一次未确认请求" }).toBe(1);
  expect(stub.calls[0].body.confirm).toBe(false);

  const dialog = page.getByRole("alertdialog", { name: "移动会话到其他工作区" });
  await expect(dialog).toBeVisible();
  await dialog.getByRole("button", { name: "确认移动" }).click();

  await expect.poll(() => stub.calls.length, { message: "确认后应重发" }).toBe(2);
  expect(stub.calls[1].body.confirm).toBe(true);
  await expect(page.locator(".toast")).toContainText("已移动到「研究」");
});

test("context_epoch 过期：提示会话状态已变化并要求重新确认，不自动重试", async ({ page }) => {
  // messageCount=0：让第一次请求直接进入 epoch 校验（不经确认门），断言更干净。
  const stub = await stubMoveBackend(page, { currentEpoch: 7, messageCount: 0 });
  await page.goto("/");

  // 观测到 epoch=7
  await page.getByRole("textbox", { name: "消息输入框" }).fill("先问一句");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await expect(page.locator(".markdown-body").first()).toContainText("回答");

  // 期间会话状态被他人改变
  stub.setCurrentEpoch(9);

  await sidebarItem(page, "带历史的会话").click({ button: "right" });
  await page.getByRole("menuitem", { name: "移动到工作区" }).click();
  await page.getByRole("menuitem", { name: /^研究/ }).click();

  // ① 请求带了过期观测值 → 409 epoch
  await expect
    .poll(() => stub.calls.length, { message: `calls=${JSON.stringify(stub.calls)}` })
    .toBe(1);
  expect(stub.calls[0].body.expected_context_epoch).toBe(7);

  // ② 明确提示「会话状态已变化」，并重新弹确认
  await expect(page.locator(".toast")).toContainText("会话状态已变化");
  const dialog = page.getByRole("alertdialog", { name: "移动会话到其他工作区" });
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText("会话状态刚刚发生了变化");

  // ③ **不得自动重试**：等一小会儿仍是 1 次请求
  await page.waitForTimeout(600);
  expect(stub.calls).toHaveLength(1);

  // ④ 用户重新确认 → 丢弃过期观测值、以 null 发出 → 成功
  await dialog.getByRole("button", { name: "确认移动" }).click();
  await expect.poll(() => stub.calls.length).toBe(2);
  expect(stub.calls[1].body.expected_context_epoch).toBeNull();
  expect(stub.calls[1].body.confirm).toBe(true);
  await expect(page.locator(".toast")).toContainText("已移动到「研究」");
});

test("拖拽跨区也受同一确认门约束（不再是不经状态机的旁路）", async ({ page }) => {
  // messageCount=0 ⇒ 前端不会预先弹确认，直接把 confirm=false 发出去；
  // requireConfirm=true ⇒ 后端（状态机）返回 409。若拖拽仍走旧的「只改存档」旁路，
  // 这里不会有任何 409，也就不会出现确认弹窗 —— 因此本用例是结构性证明。
  const stub = await stubMoveBackend(page, { messageCount: 0, requireConfirm: true });
  await page.goto("/");

  await dragTo(page, sidebarItem(page, "带历史的会话"), page.locator(".ws-folder-name").nth(1));

  await expect.poll(() => stub.calls.length, { message: "拖拽应发出一次移动请求" }).toBe(1);
  expect(stub.calls[0].path).toBe("/api/conversations/c1/move");
  expect(stub.calls[0].body.confirm).toBe(false);
  expect(stub.calls[0].body.workspace_id).toBe("research");
  // 字段集合与右键完全一致（两处同一契约）；2026-09-22 起不再有 `target_index`
  // （侧边栏恒为最近活动优先，工作区内手工排序已取消）
  expect(Object.keys(stub.calls[0].body).sort()).toEqual([
    "confirm",
    "expected_context_epoch",
    "workspace_id",
  ]);

  // 状态机要求确认 → 界面必须弹确认，而不是静默失败
  await page.getByRole("alertdialog", { name: "移动会话到其他工作区" }).getByRole("button", { name: "确认移动" }).click();
  await expect.poll(() => stub.calls.length).toBe(2);
  expect(stub.calls[1].body.confirm).toBe(true);
  await expect(page.locator(".toast")).toContainText("已移动到「研究」");
});

// 同区拖拽「不排序、不弹确认」由
// `workspaces.spec.ts::拖拽会话只能跨工作区移动，同区内拖拽不再排序` 覆盖
// （那里显式断言同区拖拽不产生任何请求、也不出现确认弹窗）；此处不重复实现脆弱的
// 物理拖拽，避免同一行为两处维护。

test("移动后在目标区继续提问：新 run 携带 origin_run_id 并在过程区域展示承接关系", async ({ page }) => {
  // messageCount=0 ⇒ 移动不需要确认；服务端返回 stopped_run_ids=["run-7"] 作为承接来源。
  const stub = await stubMoveBackend(page, { messageCount: 0, currentEpoch: 4 });
  const paths = trackPaths(page);
  await page.goto("/");

  await sidebarItem(page, "带历史的会话").click({ button: "right" });
  await page.getByRole("menuitem", { name: "移动到工作区" }).click();
  await page.getByRole("menuitem", { name: /^研究/ }).click();
  await expect(page.locator(".toast")).toContainText("已移动到「研究」");

  // 进入被移动的会话（真实用户路径：移动后在目标区打开它继续问）
  await sidebarItem(page, "带历史的会话").click();

  // 在目标区继续提问：新 run 必须记录承接来源（PRD 5.4 末句）
  await page.getByRole("textbox", { name: "消息输入框" }).fill("移动后接着问");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await expect(page.locator(".markdown-body").first()).toContainText("回答");

  await expect
    .poll(() => stub.runBodies.length, { message: `runBodies=${JSON.stringify(stub.runBodies)}` })
    .toBe(1);
  expect(stub.runBodies[0].origin_run_id).toBe("run-7");

  // 过程时间线如实展示承接关系，避免用户以为历史凭空消失（环节行默认可见）
  await expect(page.getByTestId("process-entries").first()).toContainText(
    "本次执行承接自原工作区的执行 run-7",
  );
  expectRunOnlyTraffic(paths);
});

test("未发生移动时不携带 origin_run_id（不编造承接关系）", async ({ page }) => {
  const stub = await stubMoveBackend(page, { currentEpoch: 4 });
  const paths = trackPaths(page);
  await page.goto("/");

  await page.getByRole("textbox", { name: "消息输入框" }).fill("普通提问");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await expect(page.locator(".markdown-body").first()).toContainText("回答");

  await expect.poll(() => stub.runBodies.length).toBe(1);
  expect(stub.runBodies[0].origin_run_id).toBeNull();
  // 没有承接关系：过程时间线里不得出现「承接自原工作区」这一环节
  // （环节行默认可见，因此断言内容而不是断言容器不存在）
  await expect(page.getByTestId("process-entries")).not.toContainText("承接自原工作区");
  expectRunOnlyTraffic(paths);
});

