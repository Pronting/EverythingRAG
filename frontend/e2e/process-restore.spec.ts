import { expect, test } from "@playwright/test";
import {
  answerDelta,
  expandProcessArea,
  planUpdated,
  runCompleted,
  runStatus,
  stubRunBackend,
  toolCompleted,
  toolStarted,
} from "./runFixtures";

/**
 * 离开会话再回来：**历史过程区域必须还原**（用户实测缺陷回归）。
 *
 * 过程条目只由 run 事件归约而来，而事件不随会话消息持久化 —— 前端必须按消息上的
 * `run_id` 回读 `event-log` 快照才能还原。并且正文已经在场时，过程区域应当
 * **默认折叠成一行摘要**（步数 / 任务进度 / 用时），点开才看完整环节流。
 */

const HISTORY_RUN_ID = "run-hist";
const ANSWER_TEXT = "上一轮的答案正文。";

function historyEvents() {
  const frames = [
    runStatus("running", "route", "正在理解问题"),
    planUpdated([
      {
        task_id: "t1",
        question: "货拉拉的记忆系统有淘汰策略吗",
        agent_kind: "rag",
        status: "completed",
      },
    ]),
    toolStarted("search_knowledge", { operationId: "op-1", argsSummary: "query=货拉拉" }),
    toolCompleted("search_knowledge", { operationId: "op-1", resultCount: 2, sourceCount: 1 }),
    answerDelta(ANSWER_TEXT),
    runCompleted(),
  ];
  return frames.map((frame, index) => ({
    schema_version: 2,
    run_id: HISTORY_RUN_ID,
    event_seq: index + 1,
    execution_id: "exec-1",
    task_id: frame.task_id ?? null,
    type: frame.type,
    created_at: `2026-09-20T09:00:${String(index).padStart(2, "0")}Z`,
    payload: frame.payload,
  }));
}

test("重开会话后历史过程区域按事件快照还原：默认折叠、可展开", async ({ page }) => {
  const stub = await stubRunBackend(page, {
    conversationId: "c1",
    conversationMessages: [
      {
        role: "user",
        content: "货拉拉的记忆系统有淘汰策略吗",
        created_at: "2026-09-20T08:59:59Z",
      },
      {
        role: "assistant",
        content: ANSWER_TEXT,
        run_id: HISTORY_RUN_ID,
        answer_basis: "knowledge",
        sources: [],
        policy_version: null,
        created_at: "2026-09-20T09:00:06Z",
      },
    ],
  });

  // 侧边栏列出这条历史会话（Playwright 后注册的路由优先，覆盖 stub 的空列表）。
  // 用 URL 谓词而不是 glob：会话列表请求**不带**查询串（App 侧再按工作区分组）。
  await page.route(
    (url) => url.pathname === "/api/conversations",
    (route) =>
      route.fulfill({
        json: [{ id: "c1", title: "历史对话", message_count: 2, workspace_id: "default" }],
      }),
  );
  // 这一轮的**事件快照**：只有它能让过程区域还原
  await page.route(
    (url) => url.pathname.endsWith(`/runs/${HISTORY_RUN_ID}/event-log`),
    (route) => route.fulfill({ json: { events: historyEvents() } }),
  );

  await page.goto("/");
  await page.locator(".conversation-list").getByText("历史对话").click();
  await expect(page.getByText(ANSWER_TEXT)).toBeVisible();

  const head = page.getByTestId("process-head").first();
  await expect(head).toBeVisible();
  // 正文在场 ⇒ 过程区域默认折叠成一行摘要
  await expect(head).toHaveAttribute("aria-expanded", "false");
  await expect(head).toContainText("步");
  await expect(head).toContainText("已完成");
  // 用时来自事件的实际时间戳（首末相差 5 秒），不是编出来的
  await expect(head).toContainText("用时 5秒");
  await expect(page.getByTestId("process-entries")).toHaveCount(0);

  // 展开后能看到完整环节流（真实事件负载归约出的说明）
  await expandProcessArea(page);
  const entries = page.getByTestId("process-entries");
  await expect(entries).toContainText("检索知识库：命中 2 条资料，来自 1 个来源");
  await expect(page.getByTestId("process-task-state-t1")).toHaveText("已完成");

  // 历史读取只走事件快照：不得新建 run，也不得触碰旧入口
  expect(stub.createdRuns.length).toBe(0);
  expect(stub.chatRequestCount()).toBe(0);
});
