import { expect, test } from "@playwright/test";
import {
  agentCompleted,
  agentStarted,
  answerDelta,
  answerMeta,
  approvalRequested,
  approvalResolved,
  encodeV2Events,
  evidenceAdded,
  expectRunOnlyTraffic,
  inputRequested,
  inputResolved,
  planUpdated,
  runCompleted,
  runFailed,
  runStatus,
  routeDecided,
  stubRunBackend,
  toolCompleted,
  toolStarted,
  verificationUpdated,
} from "./runFixtures";

/**
 * 过程区域、批准/澄清卡片、停止与断线刷新（PRD 14.2 / 14.3 / 16.2 A23）。
 *
 * 全部离线（page.route，代理指向死端口）；每条用例都用 `expectRunOnlyTraffic`
 * 证明「至少一次 /runs」且「零次 /api/chat」。
 */

test("并行任务分别显示状态，父任务展示整体完成情况", async ({ page }) => {
  const stub = await stubRunBackend(page, {
    frames: [
      runStatus("running"),
      planUpdated([
        { task_id: "t1", question: "风控指标定义是什么？", agent_kind: "rag" },
        { task_id: "t2", question: "最新版本有什么变化？", agent_kind: "web", depends_on: ["t1"] },
        { task_id: "t3", question: "内部实现细节", agent_kind: "rag", status: "waiting_approval" },
      ]),
      agentStarted("rag", { taskId: "t1", question: "风控指标定义是什么？" }),
      agentStarted("web", { taskId: "t2", question: "最新版本有什么变化？" }),
      agentCompleted("rag", { taskId: "t1", summary: "定义来自内部规范文档" }),
      agentCompleted("web", { taskId: "t2", summary: "发布说明确认了口径调整" }),
      // t3 卡在等待批准：只发批准卡片，任务状态由 run 快照给出（waiting_approval）
      approvalRequested({
        approval_id: "ap-t3",
        purpose: "读取内部实现文档",
        tool_name: "read_document",
        task_id: "t3",
      }),
      answerDelta("综合结论。"),
      runCompleted(),
    ],
    tasks: [
      { task_id: "t1", question: "风控指标定义是什么？", agent_kind: "rag", status: "completed" },
      { task_id: "t2", question: "最新版本有什么变化？", agent_kind: "web", depends_on: ["t1"], status: "completed" },
      { task_id: "t3", question: "内部实现细节", agent_kind: "rag", status: "waiting_approval" },
    ],
  });

  await page.goto("/");
  await page.getByRole("textbox", { name: "消息输入框" }).fill("并行任务测试");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await expect(page.getByText("综合结论。")).toBeVisible();

  await page.getByTestId("process-toggle").click();
  // 父任务整体完成情况
  await expect(page.getByTestId("process-task-progress")).toContainText("2/3 已完成");
  await expect(page.getByTestId("process-task-progress")).toContainText("1 等待处理");
  // 并行任务分别显示状态
  await expect(page.getByTestId("process-task-state-t1")).toHaveText("completed");
  await expect(page.getByTestId("process-task-state-t2")).toHaveText("completed");
  await expect(page.getByTestId("process-task-state-t3")).toHaveText("waiting_approval");
  expectRunOnlyTraffic(stub);
});

test("研究任务可展开看子问题、读过的来源与结果摘要", async ({ page }) => {
  const stub = await stubRunBackend(page, {
    frames: [
      runStatus("running"),
      planUpdated([{ task_id: "t1", question: "风控指标定义是什么？", agent_kind: "rag" }]),
      agentStarted("rag", { taskId: "t1", question: "风控指标定义是什么？" }),
      evidenceAdded(
        [{ evidence_id: "ev-1", source_type: "knowledge", title: "风控规范.md", source_ref: "风控规范.md", snippet: "定义…" }],
        "t1",
      ),
      agentCompleted("rag", { taskId: "t1", summary: "定义来自内部规范文档第 3 节" }),
      answerDelta("答案。"),
      runCompleted(),
    ],
    tasks: [{ task_id: "t1", question: "风控指标定义是什么？", agent_kind: "rag", status: "completed" }],
  });

  await page.goto("/");
  await page.getByRole("textbox", { name: "消息输入框" }).fill("展开测试");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await expect(page.getByText("答案。")).toBeVisible();

  await page.getByTestId("process-toggle").click();
  const detail = page.getByTestId("process-task-detail-t1");
  await expect(detail).not.toBeVisible();

  await page.getByTestId("process-task-t1").getByRole("button").first().click();
  await expect(detail).toBeVisible();
  await expect(detail).toContainText("子问题");
  await expect(detail).toContainText("风控指标定义是什么？");
  await expect(detail).toContainText("读过的来源（1）");
  await expect(detail).toContainText("风控规范.md");
  await expect(detail).toContainText("结果摘要");
  await expect(detail).toContainText("定义来自内部规范文档第 3 节");
  expectRunOnlyTraffic(stub);
});

test("批准卡片展示目的/工具/查询/日期范围/子任务/脱敏提示，并支持编辑查询", async ({ page }) => {
  const stub = await stubRunBackend(page, {
    frames: [
      runStatus("waiting_approval"),
      approvalRequested({
        approval_id: "ap-1",
        tool_name: "web_search",
        purpose: "需要联网检索最新版本",
        query_or_urls: ["风控指标定义 最新版"],
        date_range: "2024-01-01 ~ 2026-09-19",
        task_question: "风控指标定义是什么？",
        redaction_notice: "已脱敏：发送内容不含真实姓名",
      }),
    ],
  });

  await page.goto("/");
  await page.getByRole("textbox", { name: "消息输入框" }).fill("批准卡片测试");
  await page.getByRole("button", { name: "发送", exact: true }).click();

  const card = page.getByTestId("approval-card-ap-1");
  await expect(card).toBeVisible();
  await expect(card.getByTestId("approval-tool")).toHaveText("web_search");
  await expect(card.getByTestId("approval-purpose")).toHaveText("需要联网检索最新版本");
  await expect(card.getByTestId("approval-query")).toContainText("风控指标定义 最新版");
  await expect(card.getByTestId("approval-date-range")).toHaveText("2024-01-01 ~ 2026-09-19");
  await expect(card.getByTestId("approval-task")).toHaveText("风控指标定义是什么？");
  await expect(card.getByTestId("approval-redaction")).toContainText("已脱敏");

  // 三个按钮 + 编辑查询入口
  await expect(card.getByTestId("approval-allow-once")).toBeVisible();
  await expect(card.getByTestId("approval-allow-scope")).toBeVisible();
  await expect(card.getByTestId("approval-deny")).toBeVisible();
  await expect(card.getByTestId("approval-edit-toggle")).toBeVisible();

  // 编辑查询：改文本后「允许本次」把 edited_parameters 一起提交
  await card.getByTestId("approval-edit-toggle").click();
  const editor = card.getByTestId("approval-editor");
  await expect(editor).toBeVisible();
  await editor.fill("改写后的查询词");
  await card.getByTestId("approval-allow-once").click();

  await expect
    .poll(() => stub.approvalSubmits.length, { message: "应提交一次批准" })
    .toBe(1);
  expect(stub.approvalSubmits[0].decision).toBe("allow_once");
  expect(stub.approvalSubmits[0].edited).toEqual({ query: "改写后的查询词" });
  // 提交后卡片消失
  await expect(page.getByTestId("approval-card-ap-1")).not.toBeVisible();
  expectRunOnlyTraffic(stub);
});

test("「允许本轮同类操作」与「拒绝」各自提交正确决定", async ({ page }) => {
  const stub = await stubRunBackend(page, {
    frames: [
      runStatus("waiting_approval"),
      approvalRequested({ approval_id: "ap-scope" }),
      runStatus("waiting_approval"),
      approvalRequested({ approval_id: "ap-deny" }),
    ],
  });

  await page.goto("/");
  await page.getByRole("textbox", { name: "消息输入框" }).fill("两种决定");
  await page.getByRole("button", { name: "发送", exact: true }).click();

  await page.getByTestId("approval-card-ap-scope").getByTestId("approval-allow-scope").click();
  await expect.poll(() => stub.approvalSubmits.length).toBe(1);
  expect(stub.approvalSubmits[0]).toEqual({ approvalId: "ap-scope", decision: "allow_scope", edited: null });

  await page.getByTestId("approval-card-ap-deny").getByTestId("approval-deny").click();
  await expect.poll(() => stub.approvalSubmits.length).toBe(2);
  expect(stub.approvalSubmits[1].decision).toBe("deny");
  expectRunOnlyTraffic(stub);
});

test("澄清卡片可直接点选项或自由作答，提交后卡片消失", async ({ page }) => {
  const stub = await stubRunBackend(page, {
    frames: [
      runStatus("waiting_input"),
      inputRequested({
        input_request_id: "in-1",
        question: "你说的「风控指标」指哪一套？",
        options: ["内部规范", "监管口径"],
        reason: "两套口径的结论不同",
      }),
    ],
  });

  await page.goto("/");
  await page.getByRole("textbox", { name: "消息输入框" }).fill("澄清测试");
  await page.getByRole("button", { name: "发送", exact: true }).click();

  const card = page.getByTestId("input-card-in-1");
  await expect(card.getByTestId("input-question")).toHaveText("你说的「风控指标」指哪一套？");
  await expect(card.getByTestId("input-options")).toContainText("内部规范");
  await card.getByRole("button", { name: "内部规范" }).click();

  await expect.poll(() => stub.inputSubmits.length).toBe(1);
  expect(stub.inputSubmits[0]).toEqual({ inputRequestId: "in-1", answer: "内部规范" });
  await expect(page.getByTestId("input-card-in-1")).not.toBeVisible();
  expectRunOnlyTraffic(stub);
});

test("过程区域显示核对发现与继续研究说明", async ({ page }) => {
  const stub = await stubRunBackend(page, {
    frames: [
      runStatus("running"),
      verificationUpdated("research", {
        conflicts: ["口径 v1.2 与 v1.1 不一致"],
        pending: ["核对发布说明"],
      }),
      answerDelta("结论待确认。"),
      runCompleted({ status: "completed_with_gaps", gaps: ["版本差异未解决"] }),
    ],
  });

  await page.goto("/");
  await page.getByRole("textbox", { name: "消息输入框" }).fill("核对测试");
  await page.getByRole("button", { name: "发送", exact: true }).click();

  await page.getByTestId("process-toggle").click();
  const entries = page.getByTestId("process-entries");
  await expect(entries).toContainText("发现资料差异，继续查阅发布说明");
  await expect(entries).toContainText("发现版本差异：口径 v1.2 与 v1.1 不一致");
  await expect(entries).toContainText("待补充：核对发布说明");
  expectRunOnlyTraffic(stub);
});

test("completed_with_gaps 展示已核实内容、缺少依据与对结论的影响", async ({ page }) => {
  const stub = await stubRunBackend(page, {
    frames: [
      runStatus("running"),
      answerMeta([{ label: "S1", source_ref: "内部规范.md", snippet: "已核实片段" }]),
      answerDelta("已核实的内容[S1]，其余无法确认。"),
      runCompleted({ status: "completed_with_gaps", gaps: ["缺少发布说明依据"] }),
    ],
  });

  await page.goto("/");
  await page.getByRole("textbox", { name: "消息输入框" }).fill("资料不足测试");
  await page.getByRole("button", { name: "发送", exact: true }).click();

  // 正文照常展示（已核实内容可查看）
  await expect(page.getByText(/已核实的内容/)).toBeVisible();
  const gaps = page.getByTestId("gaps-note");
  await expect(gaps).toBeVisible();
  await expect(gaps).toContainText("已完成，但部分问题资料不足");
  await expect(gaps).toContainText("缺少依据：缺少发布说明依据");
  await expect(gaps).toContainText("对结论的影响");
  expectRunOnlyTraffic(stub);
});

test("failed 保留已完成研究记录并给出继续入口", async ({ page }) => {
  const stub = await stubRunBackend(page, {
    frames: [
      runStatus("running"),
      planUpdated([{ task_id: "t1", question: "子问题", agent_kind: "rag" }]),
      agentStarted("rag", { taskId: "t1", question: "子问题" }),
      agentCompleted("rag", { taskId: "t1", summary: "本地已确认的部分" }),
      runFailed("检索服务暂时不可用", { resumable: true }),
    ],
    tasks: [{ task_id: "t1", question: "子问题", agent_kind: "rag", status: "completed" }],
  });

  await page.goto("/");
  await page.getByRole("textbox", { name: "消息输入框" }).fill("失败测试");
  await page.getByRole("button", { name: "发送", exact: true }).click();

  await expect(page.getByText(/检索服务暂时不可用/)).toBeVisible();
  await expect(page.getByText(/可以点击继续从此处重试/)).toBeVisible();
  await expect(page.getByTestId("failed-note")).toBeVisible();
  // 已完成研究记录仍可查看
  await page.getByTestId("process-toggle").click();
  await expect(page.getByTestId("process-entries")).toContainText("本地已确认的部分");
  expectRunOnlyTraffic(stub);
});

test("未知事件版本或负载校验失败时显示协议错误，不静默", async ({ page }) => {
  // 手写一个 schema_version=99 的帧：前端必须报协议错误而不是丢掉。
  const badFrame = `id: 1\ndata: ${JSON.stringify({
    schema_version: 99,
    run_id: "run-1",
    event_seq: 1,
    execution_id: "exec-1",
    task_id: null,
    type: "run.status",
    created_at: "2026-09-19T10:00:00Z",
    payload: { status: "running", phase: "", message: "" },
  })}\n\n`;
  const stub = await stubRunBackend(page, {
    frames: [runStatus("running"), answerDelta("正常片段"), runCompleted()],
  });
  await page.route("**/api/workspaces/*/runs/*/events**", (route) =>
    route.fulfill({ status: 200, contentType: "text/event-stream", body: badFrame }),
  );

  await page.goto("/");
  await page.getByRole("textbox", { name: "消息输入框" }).fill("协议错误测试");
  await page.getByRole("button", { name: "发送", exact: true }).click();

  const banner = page.getByTestId("protocol-error");
  await expect(banner).toBeVisible();
  await expect(banner).toContainText("事件协议错误");
  expectRunOnlyTraffic(stub);
});

test("A23 断线补读：重连重复投递同一序号区间时正文不重复追加", async ({ page }) => {
  const frames = [
    runStatus("running"),
    answerMeta([{ label: "S1", source_ref: "notes.md", snippet: "片段" }]),
    answerDelta("前半"),
    answerDelta("后半", 1),
    runCompleted(),
  ];
  const stub = await stubRunBackend(page, { frames });

  // 模拟「断线后重连、服务端从 0 重发」：同一批 event_seq 投递两遍。
  // 前端必须按事件编号去重，正文只能出现一次。
  await page.route("**/api/workspaces/*/runs/*/events**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "text/event-stream",
      body: encodeV2Events("run-1", frames) + encodeV2Events("run-1", frames),
    }),
  );

  await page.goto("/");
  await page.getByRole("textbox", { name: "消息输入框" }).fill("断线测试");
  await page.getByRole("button", { name: "发送", exact: true }).click();

  const body = page.locator(".markdown-body").first();
  await expect(body).toContainText("前半后半");
  // 重复投递不得导致正文翻倍
  await expect(body).not.toContainText("前半后半前半后半");
  await expect(page.getByText("前半后半", { exact: true })).toHaveCount(1);
  expect(stub.chatRequestCount()).toBe(0);
});

test("同一会话仍有活动任务时，再问会提示先处理当前任务", async ({ page }) => {
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const stub = await stubRunBackend(page, {
    frames: [runStatus("running"), answerDelta("慢回答"), runCompleted()],
    gate,
  });

  await page.goto("/");
  const input = page.getByRole("textbox", { name: "消息输入框" });
  await input.fill("第一个问题");
  await page.getByRole("button", { name: "发送", exact: true }).click();

  // 发送中再提交：输入框被禁用（发送按钮让位给停止），不产生第二个 run
  await input.fill("第二个问题");
  await expect(page.getByRole("button", { name: "发送", exact: true })).toHaveCount(0);

  release();
  await expect(page.getByRole("button", { name: "发送", exact: true })).toBeVisible();
  expect(stub.createdRuns.length).toBe(1);
  expectRunOnlyTraffic(stub);
});

test("批准与澄清卡片直接位于对话中，且重新进入会话后仍在快照里", async ({ page }) => {
  const stub = await stubRunBackend(page, {
    frames: [runStatus("waiting_approval"), approvalRequested({ approval_id: "ap-restore" })],
    approvals: [{ approval_id: "ap-restore", tool_name: "web_search" }],
    inputs: [{ input_request_id: "in-restore", question: "选哪一套口径？", options: ["内部规范"] }],
  });

  await page.goto("/");
  await page.getByRole("textbox", { name: "消息输入框" }).fill("卡片恢复测试");
  await page.getByRole("button", { name: "发送", exact: true }).click();

  // 卡片直接位于消息流里（不在弹窗/侧栏）
  const card = page.getByTestId("approval-card-ap-restore");
  await expect(card).toBeVisible();
  await expect(page.locator(".message-row.message-assistant")).toContainText("web_search");

  // 后端 run 快照里的待处理澄清也会渲染成卡片（重进会话即可恢复）
  await expect(page.getByTestId("input-card-in-restore")).toBeVisible();
  await expect(page.getByTestId("input-card-in-restore")).toContainText("选哪一套口径？");
  expectRunOnlyTraffic(stub);
});
