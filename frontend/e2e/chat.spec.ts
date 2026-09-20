import { expect, test } from "@playwright/test";
import {
  answerDelta,
  answerMeta,
  encodeV2Events,
  evidenceAdded,
  agentCompleted,
  agentStarted,
  expectRunOnlyTraffic,
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
 * 问答主链路（V1.2：`POST /api/workspaces/{w}/runs` + `GET .../events` SSE）。
 *
 * 「/api/chat」在本文件里**一次都不出现**：旧入口已被结构性移除（后端只剩 410），
 * 所有用例都通过 `expectRunOnlyTraffic` 显式证明「至少一次命中 /runs」且
 * 「零次命中 /api/chat」—— 否则 stub 会拦住一个再没人调用的旧端点，测试等于空跑。
 */

/** 常见答案序列：理解问题 → 计划 → 研究 → 资料 → 核对 → 正文 → 完成。 */
function fullRunFrames(): ReturnType<typeof runStatus>[] {
  return [
    runStatus("running", "route", "正在理解问题"),
    routeDecided(),
    planUpdated([{ task_id: "t1", question: "这份文档讲了什么？" }]),
    agentStarted("rag", { taskId: "t1", question: "这份文档讲了什么？", threadId: "run:1:task:t1" }),
    toolStarted("search_knowledge", { operationId: "op-1", taskId: "t1" }),
    toolCompleted("search_knowledge", { operationId: "op-1", taskId: "t1", resultCount: 2, sourceCount: 2 }),
    evidenceAdded(
      [
        { evidence_id: "ev-1", source_type: "knowledge", title: "notes.md", source_ref: "notes.md", snippet: "这是第一份笔记的片段内容。" },
        { evidence_id: "ev-2", source_type: "knowledge", title: "architecture.md", source_ref: "architecture.md", snippet: "关于架构决策的第二段参考内容。" },
      ],
      "t1",
    ),
    agentCompleted("rag", { taskId: "t1", summary: "本地文档确认了标题与架构决策" }),
    verificationUpdated("pass"),
    answerMeta(
      [
        { label: "S1", source_ref: "notes.md", snippet: "这是第一份笔记的片段内容。" },
        { label: "S2", source_ref: "architecture.md", snippet: "关于架构决策的第二段参考内容。" },
      ],
      { basis: "knowledge", components: ["knowledge"] },
    ),
    answerDelta("你"),
    answerDelta("好", 1),
    answerDelta("，", 2),
    answerDelta("世", 3),
    answerDelta("界", 4),
    runCompleted(),
  ];
}

test("图片引用展示表格证据并可打开缓存原图", async ({ page }) => {
  const stub = await stubRunBackend(page, {
    frames: [
      runStatus("running"),
      answerMeta(
        [
          {
            label: "S1",
            source_ref: "/api/knowledge/images/aaaaaaaaaaaaaaaa",
            snippet: "|日期|GMV|\n|---|---|\n|2024-12-02|4826万|",
          },
        ],
        { basis: "knowledge", components: ["knowledge"] },
      ),
      answerDelta("数据见[S1]。"),
      runCompleted(),
    ],
  });

  await page.goto("/");
  await page.getByRole("textbox", { name: "消息输入框" }).fill("查看图表数据");
  await page.getByRole("button", { name: "发送", exact: true }).click();

  await expect(page.getByText("数据见")).toBeVisible();
  await page.locator(".markdown-body .citation").click();
  const panel = page.getByRole("complementary", { name: "来源详情" });
  await expect(panel.getByRole("link", { name: "查看原图" })).toHaveAttribute(
    "href",
    "/api/knowledge/images/aaaaaaaaaaaaaaaa",
  );
  await expect(panel.getByRole("cell", { name: "4826万" })).toBeVisible();
  expectRunOnlyTraffic(stub);
});

test("发消息后流式渲染完整回答与来源脚注", async ({ page }) => {
  const stub = await stubRunBackend(page, { frames: fullRunFrames() });

  await page.goto("/");

  // 1. 侧边栏品牌渲染 + 输入框可见
  await expect(page.getByRole("heading", { name: "Everything RAG" })).toBeVisible();
  const input = page.getByRole("textbox", { name: "消息输入框" });
  await expect(input).toBeVisible();

  // 2. 输入消息并发送，断言按片段位置拼接的完整文本
  await input.fill("这份文档讲了什么？");
  await page.getByRole("button", { name: "发送" }).click();
  await expect(page.getByText("你好，世界")).toBeVisible();

  // 3. 答案依据 + 来源脚注（CitationEntry 不带向量分数 → 显示「词法命中」，不伪造 0%）
  await expect(page.getByText("基于本地知识")).toBeVisible();
  await expect(page.getByText("notes.md")).toBeVisible();
  await expect(page.getByText("architecture.md")).toBeVisible();
  await expect(page.getByText("词法命中").first()).toBeVisible();
  await expect(page.getByText("0%")).not.toBeVisible();

  // 4. 过程区域由真实事件生成（不是预置文案）
  await page.getByTestId("process-toggle").click();
  const entries = page.getByTestId("process-entries");
  await expect(entries).toContainText("正在理解问题");
  await expect(entries).toContainText("先查本地知识，不足再联网");
  await expect(entries).toContainText("已拆解为 1 个子问题");
  await expect(entries).toContainText("本地文档确认了标题与架构决策");
  await expect(entries).toContainText("检索知识库：命中 2 条资料");
  await expect(entries).toContainText("找到 2 条资料");
  await expect(entries).toContainText("核对通过");
  await expect(entries).toContainText("已完成并核对通过");

  // 5. 点击来源 chip -> 右侧详情面板展示该片段全文
  await page.getByRole("button", { name: /notes\.md/ }).click();
  await expect(page.locator(".citation-panel")).toBeVisible();
  await expect(page.locator(".citation-panel").getByText("这是第一份笔记的片段内容。")).toBeVisible();

  // 6. 发送结束后输入框恢复可用
  await expect(input).toBeEnabled();
  expectRunOnlyTraffic(stub);
});

test("正文引用序号可悬停预览、点击打开来源详情", async ({ page }) => {
  const stub = await stubRunBackend(page, {
    frames: [
      runStatus("running"),
      answerMeta([{ label: "S1", source_ref: "ref.md", snippet: "引用片段正文内容。" }]),
      answerDelta("答案是"),
      answerDelta(" ", 1),
      answerDelta("[S1]", 2),
      answerDelta("。", 3),
      runCompleted(),
    ],
  });
  await page.goto("/");

  const input = page.getByRole("textbox", { name: "消息输入框" });
  await input.fill("测试引用");
  await page.getByRole("button", { name: "发送" }).click();

  const citation = page.locator(".markdown-body .citation");
  await expect(citation).toHaveText("S1");

  await citation.hover();
  await expect(page.locator(".citation-tooltip").getByText("ref.md")).toBeVisible();
  await expect(page.locator(".citation-tooltip").getByText("引用片段正文内容。")).toBeVisible();

  await citation.click();
  await expect(page.locator(".citation-panel")).toBeVisible();
  await expect(page.locator(".citation-panel").getByText("引用片段正文内容。")).toBeVisible();
  await expect(page.locator(".citation-panel-count")).toContainText("1 / 1");
  expectRunOnlyTraffic(stub);
});

test("相邻引用逐个转徽章，无效序号被移除", async ({ page }) => {
  const stub = await stubRunBackend(page, {
    frames: [
      runStatus("running"),
      answerMeta([
        { label: "S1", source_ref: "a.md", snippet: "来源 A 的片段。" },
        { label: "S2", source_ref: "b.md", snippet: "来源 B 的片段。" },
      ]),
      answerDelta("见 "),
      answerDelta("[S1][S2]", 1),
      answerDelta("，无效引用 ", 2),
      answerDelta("[S9]", 3),
      answerDelta("。", 4),
      runCompleted(),
    ],
  });
  await page.goto("/");

  const input = page.getByRole("textbox", { name: "消息输入框" });
  await input.fill("测试相邻引用");
  await page.getByRole("button", { name: "发送" }).click();

  const citations = page.locator(".markdown-body .citation");
  await expect(citations).toHaveCount(2);
  await expect(citations.nth(0)).toHaveText("S1");
  await expect(citations.nth(1)).toHaveText("S2");
  await expect(page.locator(".markdown-body")).not.toContainText("[S9]");
  expectRunOnlyTraffic(stub);
});

test("run.failed 渲染错误消息且不白屏", async ({ page }) => {
  const stub = await stubRunBackend(page, {
    frames: [runStatus("running"), runFailed("知识库尚未建立索引，请先同步文档")],
  });

  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Everything RAG" })).toBeVisible();

  const input = page.getByRole("textbox", { name: "消息输入框" });
  await input.fill("测试错误路径");
  await page.getByRole("button", { name: "发送" }).click();

  await expect(page.getByText(/知识库尚未建立索引/)).toBeVisible();
  await expect(input).toBeEnabled();
  expectRunOnlyTraffic(stub);
});

test("连续两轮对话的用户问题与回答都完整显示（自动滚动到底部）", async ({ page }) => {
  const stub = await stubRunBackend(page, { frames: fullRunFrames() });
  await page.goto("/");

  const input = page.getByRole("textbox", { name: "消息输入框" });

  await input.fill("第一轮问题");
  await page.getByRole("button", { name: "发送" }).click();
  await expect(page.getByText("第一轮问题")).toBeVisible();
  await expect(page.getByText("你好，世界").first()).toBeVisible();

  await input.fill("第二轮问题");
  await page.getByRole("button", { name: "发送" }).click();
  await expect(page.getByText("第二轮问题")).toBeVisible();
  await expect(page.getByText("你好，世界").nth(1)).toBeVisible();

  await expect(input).toBeEnabled();
  // 两轮 = 两个 run
  expect(stub.createdRuns.length).toBe(2);
  expectRunOnlyTraffic(stub);
});

test("处理过程默认折叠，点击可展开逐步说明", async ({ page }) => {
  // V1.2 起模型的思维链不再越过 SSE 边界（20 类事件里没有 reasoning），
  // 逐步过程由 PRD 14.2 的过程区域承载，条目全部来自真实事件。
  const stub = await stubRunBackend(page, {
    frames: [
      runStatus("running", "route", "正在理解问题"),
      planUpdated([{ task_id: "t1", question: "风控指标定义是什么？", agent_kind: "rag" }]),
      agentStarted("rag", { taskId: "t1", question: "风控指标定义是什么？", threadId: "run:1:task:t1" }),
      toolStarted("read_document", { operationId: "op-read", argsSummary: "风控指标定义", taskId: "t1" }),
      toolCompleted("read_document", { operationId: "op-read", taskId: "t1", resultCount: 1 }),
      answerDelta("答案是42"),
      runCompleted(),
    ],
  });

  await page.goto("/");
  const input = page.getByRole("textbox", { name: "消息输入框" });
  await input.fill("测试处理过程");
  await page.getByRole("button", { name: "发送" }).click();

  await expect(page.getByText("答案是42")).toBeVisible();

  // 默认折叠：条目不可见
  const toggle = page.getByTestId("process-toggle");
  await expect(toggle).toHaveAttribute("aria-expanded", "false");
  await expect(page.getByTestId("process-entries")).not.toBeVisible();

  // 展开：逐步说明可见（行动句来自真实事件负载）
  await toggle.click();
  await expect(toggle).toHaveAttribute("aria-expanded", "true");
  const entries = page.getByTestId("process-entries");
  await expect(entries).toContainText("正在理解问题");
  await expect(entries).toContainText("正在本地知识中查证：风控指标定义是什么？");
  // 工具条目在完成后收口为「动作：结果」，不残留 running 文案
  await expect(entries).toContainText("读取对应章节：命中 1 条资料");
  await expect(entries).toContainText("已完成并核对通过");

  // 研究任务可展开看子问题与结果摘要
  await expect(page.getByTestId("process-tasks")).toContainText("风控指标定义是什么？");
  await page.getByTestId("process-task-t1").getByRole("button").first().click();
  await expect(page.getByTestId("process-task-detail-t1")).toContainText("子问题");
  expectRunOnlyTraffic(stub);
});

test("生成中可点击「停止生成」中断真流式输出并保留已生成内容", async ({ page }) => {
  // 只有本用例不 stub run 端点：由 dev server 的慢速 SSE 夹具提供（见 vite.config.ts），
  // 前端拿到的才是真正的「吐一半就停住」的流。其余接口照常 stub。
  // 这一条是「/api/chat 中间件」迁移的守门用例：夹具没挂到新 runs 路径上就会失败。
  const stub = await stubRunBackend(page, { skipRuns: true });

  await page.goto("/");

  const input = page.getByRole("textbox", { name: "消息输入框" });
  await input.fill("测试中断");
  await page.getByRole("button", { name: "发送", exact: true }).click();

  // 流式已吐出的正文保留在消息流里；生成中发送键让位给「停止生成」
  await expect(page.getByText("已经生成的前半")).toBeVisible();
  const stop = page.getByRole("button", { name: "停止生成" });
  await expect(stop).toBeVisible();
  await stop.click();

  // 中断后：标注已停止、正文保留、恢复可发送状态
  await expect(page.getByTestId("stopped-note")).toBeVisible();
  await expect(page.getByText("已经生成的前半")).toBeVisible();
  await expect(page.getByRole("button", { name: "发送", exact: true })).toBeVisible();
  await expect(stop).not.toBeVisible();
  await expect(page.getByText("测试中断")).toBeVisible();
  // 已吐出的正文不会被清空（停止只结束本轮，不回滚已收到内容）
  await expect(page.locator(".markdown-body")).toContainText("已经生成的前半");
  // 取消走的是新任务接口（本用例不 stub run 端点，由 dev server 夹具应答）
  expect(stub.paths.some((path) => path.endsWith("/cancel"))).toBe(true);
  expectRunOnlyTraffic(stub);
});

test("Shift+Enter 换行、Enter 发送，输入框随内容自动增高", async ({ page }) => {
  const stub = await stubRunBackend(page, { frames: fullRunFrames() });
  await page.goto("/");

  const input = page.getByRole("textbox", { name: "消息输入框" });
  const heightOf = (): Promise<number> =>
    input.evaluate((el) => (el as HTMLTextAreaElement).getBoundingClientRect().height);

  const singleLineHeight = await heightOf();
  await input.fill("第一行");
  await input.press("Shift+Enter");
  await input.type("第二行");
  await expect(input).toHaveValue("第一行\n第二行");

  await input.fill(Array.from({ length: 6 }, (_, index) => `第 ${index} 行内容`).join("\n"));
  await expect
    .poll(heightOf, { message: "输入框应随内容增高" })
    .toBeGreaterThan(singleLineHeight);

  await input.fill("用回车发送");
  await input.press("Enter");
  await expect(page.getByText("你好，世界")).toBeVisible();
  await expect(input).toHaveValue("");
  await expect.poll(heightOf).toBeLessThanOrEqual(singleLineHeight + 1);
  expectRunOnlyTraffic(stub);
});

test("生成期间可聚焦输入框并保留下一条草稿，不重复发送", async ({ page }) => {
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const stub = await stubRunBackend(page, { frames: fullRunFrames(), gate });
  await page.goto("/");

  const input = page.getByRole("textbox", { name: "消息输入框" });
  await input.fill("第一条问题");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await input.click();
  await expect(input).toBeFocused();
  await input.fill("下一条草稿");
  await input.press("Enter");

  // 生成中：不重复发送，草稿保留
  await expect(input).toHaveValue("下一条草稿");
  release();
  await expect(page.getByRole("button", { name: "发送", exact: true })).toBeEnabled();
  await expect(input).toHaveValue("下一条草稿");
  expect(stub.createdRuns.length).toBe(1);
  expectRunOnlyTraffic(stub);
});

test("悬浮铅笔可召回提问重新编辑，重发后不追加重复卡片", async ({ page }) => {
  const truncates: { index: number }[] = [];
  const stub = await stubRunBackend(page, {
    frames: (runIndex) =>
      runIndex === 0
        ? [runStatus("running"), answerDelta("第一轮回答"), runCompleted()]
        : [runStatus("running"), answerDelta("改写后的回答"), runCompleted()],
  });
  await page.route("**/api/conversations/conv-1/truncate", (route) => {
    truncates.push(route.request().postDataJSON() as { index: number });
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ message_count: 0 }),
    });
  });
  await page.route("**/api/conversations/conv-1", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ id: "conv-1", title: "新对话", workspace_id: "default", messages: [] }),
    }),
  );
  await page.goto("/");

  const input = page.getByRole("textbox", { name: "消息输入框" });
  await input.fill("第一个问题");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await expect(page.getByText("第一轮回答")).toBeVisible();

  await input.fill("第二个问题");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await expect(page.getByText("第一轮回答")).toBeVisible();
  await expect(page.locator(".message-row.message-user")).toHaveCount(2);

  const firstQuestion = page.locator(".message-row.message-user").first();
  const pencil = firstQuestion.getByRole("button", { name: "编辑这条提问" });
  await expect(pencil).toBeAttached();
  await firstQuestion.locator(".message-content.user").hover();
  await expect(pencil).toHaveCSS("opacity", "1");
  await pencil.click();

  await expect(input).toHaveValue("第一个问题");
  await expect(page.locator(".message-row.message-user")).toHaveCount(0);
  await expect(page.getByText("第一轮回答")).toHaveCount(0);
  expect(truncates).toEqual([{ index: 0 }]);

  await input.fill("第一个问题（改写）");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await expect(page.getByText("改写后的回答")).toBeVisible();
  await expect(page.locator(".message-row.message-user")).toHaveCount(1);
  await expect(page.getByText("第一个问题（改写）")).toBeVisible();
  await expect(page.getByText("第二个问题")).toHaveCount(0);
  expect(stub.createdRuns.length).toBe(3);
  expectRunOnlyTraffic(stub);
});
