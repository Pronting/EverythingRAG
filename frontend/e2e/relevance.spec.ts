import { expect, test, type Page } from "@playwright/test";
import {
  answerDelta,
  answerMeta,
  expectRunOnlyTraffic,
  runCompleted,
  runStatus,
  stubRunBackend,
  toolCompleted,
  toolStarted,
  type RunStub,
} from "./runFixtures";

/**
 * RAG-REL-001 前端接入（PRD §8.1/§8.2/§8.3）：设置页「资料相关性筛选」、问答进度、
 * 来源的「已筛选资料」脚注与旧记录兼容。
 *
 * 集成说明（merge/v120-into-main）：旧的 `POST /api/chat`（meta.retrieval + `tool` 帧）
 * 已按 PRD 13.2 结构性下线，本文件**全部**改到 V1.2 的任务（run）架构上：
 * - 进度：`tool.started` / `tool.completed`（`tool_name="rerank"`）——运行层发布的既有事件；
 * - 来源：`answer.meta` 的 `CitationEntry`（run 架构的引用卡片契约）；
 * - 守门口径：问答用例调用 `expectRunOnlyTraffic`（≥1 次命中 `/runs`、0 次命中 `/api/chat`）；
 *   设置页用例不发起任务，改用本地 `expectNoLegacyChatTraffic`（只守 0 次 `/api/chat`）——
 *   两者都不让 stub 拦住一个再也没人调用的旧端点。
 *
 * 契约现状（集成 t5 后）：后端把**真实**的每来源筛选字段（`relevance_checked` /
 * `rerank_score` / `score_type` / `profile_id` / `evidence_hash` / `member_refs`）
 * 随 `answer.meta.citations` 下发，并在 rerank 的 `tool.completed` 上给真实
 * `result_count`（入选数）/ `source_count`（候选数）——本文件据此断言「真实值真的被
 * 渲染出来」（排序分数 / 组合成员 range / chip 与徽标），同时保留「字段缺失时**不显示**、
 * 不伪造」与「绝不出现正确率 / 概率 / 0%」的用例。
 * 引用契约仍不带**向量**相似度 ⇒ 已筛选来源的向量分数行如实显示「未记录向量相似度」，
 * 而不是编一个百分比。
 */

const STATUS_BODY = {
  app: { name: "everything-rag", version: "0.1.0" },
  config: {
    wizard_completed: false,
    embed_configured: true,
    chat_configured: true,
    vision_configured: false,
    vision_enabled: false,
  },
  knowledge: { file_count: 0, chunk_count: 0, image_count: 0, last_sync_at: null, needs_rebuild: false },
  privacy: { outbound_state: "local-only", providers: {} },
};

/** 关键断言用的测试密钥：只允许出现在 password 输入框里，绝不回显到页面文本。 */
const UNSAVED_KEY = "sk-rerank-unsaved-1234";
const SAVED_KEY_HINT = "...brkz";

const RERANK_VIEW = {
  enabled: false,
  provider_type: "siliconflow_rerank",
  endpoint: "https://api.siliconflow.cn/v1/rerank",
  model: "BAAI/bge-reranker-v2-m3",
  api_key_set: true,
  api_key_hint: SAVED_KEY_HINT,
  profile_id: "profile-live",
  connect_timeout_seconds: 5,
  read_timeout_seconds: 60,
  configured: true,
  status: "ready",
  acceptance_env: false,
  profile: {
    profile_id: "profile-live",
    status: "validated",
    model: "BAAI/bge-reranker-v2-m3",
    generated_at: "2026-07-20T09:00:00+00:00",
    validated_at: "2026-08-01T10:00:00+00:00",
    needs_validation: false,
  },
  profiles: [
    {
      profile_id: "profile-live",
      status: "validated",
      model: "BAAI/bge-reranker-v2-m3",
      generated_at: "2026-07-20T09:00:00+00:00",
      validated_at: "2026-08-01T10:00:00+00:00",
      needs_validation: false,
    },
    {
      profile_id: "profile-candidate",
      status: "candidate",
      model: "BAAI/bge-reranker-v2-m3",
      generated_at: "2026-08-10T09:00:00+00:00",
      validated_at: null,
      needs_validation: true,
    },
  ],
};

const SETTINGS_VIEW = {
  chat: {
    provider_type: "openai_compatible",
    base_url: "https://api.example.com/v1",
    model: "deepseek-chat",
    api_key_set: true,
    api_key_hint: "...abcd",
    supports_image: false,
  },
  embed: {
    mode: "cloud",
    base_url: "https://embed.example.com/v1",
    model: "bge-m3",
    api_key_set: false,
    api_key_hint: null,
  },
  vision: {
    provider_type: "openai_compatible",
    base_url: null,
    model: null,
    api_key_set: false,
    api_key_hint: null,
  },
  search: {
    provider_type: "tavily",
    enabled: false,
    base_url: null,
    max_results: 5,
    api_key_set: false,
    api_key_hint: null,
  },
  rerank: RERANK_VIEW,
  answer_preferences: {
    language: "auto",
    verbosity: "balanced",
    tone: "natural",
    response_format: "auto",
    knowledge_only: false,
    custom_instructions: "",
  },
  policy: { version: "rag-policy-v3", managed: true },
  avatars: { user: null, agent: null },
  theme: "light",
};

/** 语义块库：来源面板「读取完整原文」的既有入口读它。 */
const BLOCKS_BODY = {
  total: 1,
  blocks: [
    {
      block_id: "chunk-b",
      text: "补充成员的完整原文全文。",
      chunk_type: "text",
      source_type: "local_file",
      source_file: "notes.md",
      heading_path: "项目笔记 > 概述",
      anchor: "sec-overview",
      platform: "local",
    },
  ],
};

/** 语义块库桩：来源面板「读取完整原文」走既有「语义块浏览」入口读它。 */
async function stubBlocks(page: Page): Promise<void> {
  await page.route("**/api/blocks", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(BLOCKS_BODY) }),
  );
}

/**
 * 基础后端桩：App 启动需要 status/settings（外加来源面板用的 blocks）。
 *
 * 复用 `stubRunBackend`（它已按 V1.2 契约桩好 status/conversations/workspaces/runs），
 * 再**后注册**设置页用例需要的完整 settings 视图：Playwright 按注册顺序逆序匹配，
 * 后注册的路由生效（既有设置用例依赖同一条性质）。
 */
async function stubShell(page: Page, settingsView: unknown = SETTINGS_VIEW): Promise<RunStub> {
  const stub = await stubRunBackend(page);
  await page.route("**/api/status", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(STATUS_BODY) }),
  );
  await page.route("**/api/settings", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(settingsView) }),
  );
  await stubBlocks(page);
  return stub;
}

async function send(page: Page, message: string): Promise<void> {
  await page.getByRole("textbox", { name: "消息输入框" }).fill(message);
  await page.getByRole("button", { name: "发送", exact: true }).click();
}

/**
 * 设置页用例不发起任何任务（不点「发送」），因此不适用「≥1 次命中 `/runs`」，
 * 但仍必须守住「零次命中 `/api/chat`」——旧入口已结构性下线。
 */
function expectNoLegacyChatTraffic(stub: RunStub): void {
  expect(stub.chatRequestCount(), "不得再请求 /api/chat").toBe(0);
}

test("问答显示「正在筛选资料」进度，停止后进度关闭", async ({ page }) => {
  // 只有本用例不 stub run 端点：由 dev server 的慢速 SSE 夹具提供（见 vite.config.ts），
  // 夹具里带 rerank 的 tool.started 帧。这一条同时是「夹具已从 /api/chat 迁到
  // /api/workspaces/{w}/runs + .../events」的守门用例：挂错路径就会失败。
  const stub = await stubRunBackend(page, { skipRuns: true });
  await page.goto("/");
  await send(page, "筛选进度测试");

  // 评分期间必须有独立进度提示（此时正文还没开始吐字）
  const progress = page.locator(".chat-searching");
  await expect(progress).toBeVisible();
  await expect(progress).toContainText("正在筛选资料");
  // 用户可见的进度文案不暴露 RRF / tokenizer 等实现参数
  await expect(progress).not.toContainText("RRF");
  await expect(progress).not.toContainText("tokenizer");

  // 夹具随后吐出正文片段：进度提示不得挡住正文
  await expect(page.getByText("已经生成的前半")).toBeVisible();

  // 过程区域同步显示筛选这一步（文案来自真实事件，不是预置）
  await page.getByTestId("process-toggle").click();
  await expect(page.getByTestId("process-entries")).toContainText("正在筛选资料");

  // 停止生成后进度关闭，已吐出的正文保留
  await page.getByRole("button", { name: "停止生成" }).click();
  await expect(page.getByTestId("stopped-note")).toBeVisible();
  await expect(page.locator(".chat-searching")).not.toBeVisible();
  await expect(page.getByText("已经生成的前半")).toBeVisible();
  expect(stub.paths.some((path) => path.endsWith("/cancel"))).toBe(true);
  expectRunOnlyTraffic(stub);
});

test("已筛选来源渲染真实排序分数、组合成员与证据哈希", async ({ page }) => {
  // 后端 t5 起：每来源筛选字段随 `answer.meta.citations` 到达（纯追加、带默认值）。
  // 本用例给**真实值**，断言界面真的把它们渲染出来（t2 时期这些字段到不了来源层）。
  await stubBlocks(page);
  const stub = await stubRunBackend(page, {
    frames: [
      runStatus("running", "research", "正在检索知识库"),
      toolStarted("rerank", { operationId: "op-rerank" }),
      // result_count = 入选数、source_count = 候选数（运行层发布的真实计数）
      toolCompleted("rerank", { operationId: "op-rerank", status: "succeeded", resultCount: 1, sourceCount: 30 }),
      answerMeta(
        [
          {
            label: "S1",
            source_ref: "notes.md",
            snippet: "本轮实际采用的证据片段。",
            relevance_checked: true,
            rerank_score: 0.812,
            score_type: "rerank",
            profile_id: "profile-live",
            evidence_hash: "deadbeef",
            member_refs: [
              {
                block_id: "chunk-a",
                content_hash: "hash-a",
                relation: "primary",
                start: 120,
                end: 480,
                block_char_offset: 120,
              },
              {
                block_id: "chunk-b",
                content_hash: "hash-b",
                relation: "supplement",
                start: 0,
                end: null,
                block_char_offset: 0,
              },
            ],
          },
        ],
        { basis: "knowledge" },
      ),
      answerDelta("答案见 [S1]。"),
      runCompleted(),
    ],
  });
  await page.goto("/");
  await send(page, "已筛选来源测试");
  await expect(page.getByText("答案见")).toBeVisible();

  // 脚注状态由 rerank 完成事件推出；chip 由**来源自身**的 relevance_checked 渲染
  await expect(page.locator(".source-footer-label")).toHaveText("已筛选资料");
  await expect(page.locator(".source-chip-sim")).toHaveText("已筛选");
  await expect(page.locator(".source-chip-filtered")).toHaveCount(1);

  // 过程区域如实收口筛选这一步（真实入选/候选计数）
  await page.getByTestId("process-toggle").click();
  await expect(page.getByTestId("process-entries")).toContainText("筛选资料：入选 1 条资料（候选 30 条）");

  // 详细面板：排序分数、组合成员 range、原文；档案 id 与证据哈希进诊断属性（不占可见文案）
  await page.getByRole("button", { name: /notes\.md/ }).click();
  const panel = page.getByRole("complementary", { name: "来源详情" });
  await expect(panel.locator(".citation-panel-badge")).toHaveText("已筛选资料");
  await expect(panel.locator(".citation-panel-score")).toHaveText("排序分数 0.812");
  await expect(panel.locator(".citation-panel-score")).not.toContainText("%");
  await expect(panel).not.toContainText("正确率");
  await expect(panel).not.toContainText("概率");
  await expect(panel).toHaveAttribute("data-profile-id", "profile-live");
  await expect(panel).toHaveAttribute("data-evidence-hash", "deadbeef");
  // 引用契约仍不带**向量**相似度：不得因此编一个百分比，如实显示「未记录向量相似度」
  await expect(panel.locator(".citation-panel-meta")).toHaveText("未记录向量相似度");
  await expect(panel.locator(".citation-panel-text")).toHaveText("本轮实际采用的证据片段。");

  // 组合成员可追溯：主证据 + 补充原文各带范围
  const members = panel.locator(".citation-panel-member");
  await expect(members).toHaveCount(2);
  await expect(members.nth(0)).toContainText("主要原文");
  await expect(members.nth(0)).toContainText("第 120–480 字");
  await expect(members.nth(1)).toContainText("补充原文");
  await expect(members.nth(1)).toContainText("整块原文");

  // 读取完整原文：复用既有「语义块浏览」入口，完整原文不进入本轮来源片段
  await members.nth(1).getByRole("button", { name: "读取完整原文" }).click();
  await expect(page.getByRole("dialog", { name: "语义块浏览" })).toBeVisible();
  const detail = page.getByRole("dialog", { name: "块详情" });
  await expect(detail.getByText("补充成员的完整原文全文。")).toBeVisible();
  await expect(detail.getByText("chunk-b")).toBeVisible();
  expectRunOnlyTraffic(stub);
});

test("有候选但一条都没入选：显示 empty 态说明，来源为空", async ({ page }) => {
  // PRD §8.2 的 empty 态：`result_count=0` 且 `source_count>0`（有候选、无一条入选）。
  // 必须让用户看见「这是筛选的结果」，而不是只剩一片空白。
  const stub = await stubRunBackend(page, {
    frames: [
      runStatus("running", "research", "正在检索知识库"),
      toolStarted("rerank", { operationId: "op-rerank" }),
      toolCompleted("rerank", { operationId: "op-rerank", status: "succeeded", resultCount: 0, sourceCount: 3 }),
      answerMeta([], { basis: "insufficient" }),
      answerDelta("现有资料不足以支持结论。"),
      runCompleted({ status: "completed_with_gaps", gaps: ["缺少可核对的资料"] }),
    ],
  });
  await page.goto("/");
  await send(page, "无材料入选测试");

  const note = page.getByTestId("relevance-empty-note");
  await expect(note).toBeVisible();
  await expect(note).toHaveText("本轮没有资料通过相关性筛选（候选 3 条全部未入选）。");

  // 过程区域同步说清这一步的真实结果
  await page.getByTestId("process-toggle").click();
  await expect(page.getByTestId("process-entries")).toContainText("筛选资料：候选 3 条全部未入选");

  // 没有来源 ⇒ 不出现脚注/徽标；empty ≠「已筛选出资料」，也不得伪造分数
  await expect(page.locator(".source-footer")).toHaveCount(0);
  await expect(page.locator(".citation-panel-badge")).toHaveCount(0);
  await expect(page.getByText("已筛选资料")).not.toBeVisible();
  await expect(page.locator(".chat-searching")).not.toBeVisible();
  await expect(note).not.toContainText("%");
  expectRunOnlyTraffic(stub);
});

test("筛选完成但引用缺字段：只显示事件级结论，不伪造每来源分数", async ({ page }) => {
  // 旧记录 / 半截 payload：rerank 事件在（筛选真的跑过），但引用上没有追加字段。
  // 脚注（事件级）可以说「已筛选资料」；**每来源**的分数/成员/哈希必须留空 ——
  // 契约默认值不得被渲染成 0 或假分数（PRD §8.3）。
  const stub = await stubRunBackend(page, {
    frames: [
      runStatus("running", "research", "正在检索知识库"),
      toolStarted("rerank", { operationId: "op-rerank" }),
      toolCompleted("rerank", { operationId: "op-rerank", status: "succeeded", resultCount: 1, sourceCount: 5 }),
      answerMeta([{ label: "S1", source_ref: "legacy.md", snippet: "旧记录引用的片段。" }], {
        basis: "knowledge",
      }),
      answerDelta("答案见 [S1]。"),
      runCompleted(),
    ],
  });
  await page.goto("/");
  await send(page, "字段缺失测试");
  await expect(page.getByText("答案见")).toBeVisible();

  // 事件级：筛选跑过且入选 1 条 ⇒ 脚注说「已筛选资料」
  await expect(page.locator(".source-footer-label")).toHaveText("已筛选资料");
  // 每来源级：引用没带 relevance_checked ⇒ chip 仍是匹配方式标签，不是「已筛选」
  await expect(page.locator(".source-chip-sim")).toHaveText("词法命中");
  await expect(page.locator(".source-chip-filtered")).toHaveCount(0);

  await page.getByRole("button", { name: /legacy\.md/ }).click();
  const panel = page.getByRole("complementary", { name: "来源详情" });
  await expect(panel.locator(".citation-panel-badge")).toHaveCount(0);
  await expect(panel.locator(".citation-panel-score")).toHaveCount(0);
  await expect(panel).not.toContainText("排序分数");
  await expect(panel).not.toContainText("0%");
  await expect(panel).not.toContainText("正确率");
  await expect(panel).not.toContainText("概率");
  // 诊断属性为空 ⇒ 确实没有值可显示，而不是显示了一个 0
  await expect(panel).toHaveAttribute("data-profile-id", "");
  await expect(panel).toHaveAttribute("data-evidence-hash", "");
  expectRunOnlyTraffic(stub);
});

test("未启用筛选的来源保留匹配方式标签与「向量相似度」命名口径", async ({ page }) => {
  // 未启用相关性筛选：运行层不发布任何 rerank 事件 ⇒ 脚注回到「来源」，
  // 引用不带向量分数 ⇒ 显示「词法命中」而不是伪造 0%/百分比。
  const stub = await stubRunBackend(page, {
    frames: [
      runStatus("running", "research", "正在检索知识库"),
      toolStarted("search_knowledge", { operationId: "op-search" }),
      toolCompleted("search_knowledge", { operationId: "op-search", resultCount: 2, sourceCount: 2 }),
      answerMeta(
        [
          { label: "S1", source_ref: "legacy.md", snippet: "旧记录的命中片段。" },
          { label: "S2", source_ref: "legacy.md", snippet: "纯词法命中的旧片段。" },
        ],
        { basis: "knowledge" },
      ),
      answerDelta("旧记录回答。"),
      runCompleted(),
    ],
  });
  await page.goto("/");
  await send(page, "旧记录测试");
  await expect(page.getByText("旧记录回答。")).toBeVisible();

  // 未筛选：脚注仍是「来源」；本轮没有 rerank 事件 ⇒ 不出现筛选进度提示
  await expect(page.locator(".source-footer-label")).toHaveText("来源");
  await expect(page.locator(".chat-searching")).not.toBeVisible();
  const chips = page.locator(".source-chip-sim");
  await expect(chips).toHaveCount(2);
  await expect(chips.nth(0)).toHaveText("词法命中");
  await expect(chips.nth(1)).toHaveText("词法命中");
  await expect(chips.nth(0)).not.toContainText("%");

  // 旧记录面板：只有向量相似度的命名口径（此处引用不带分数）与原文，不出现「排序分数」
  await page.getByRole("button", { name: /legacy\.md/ }).first().click();
  const panel = page.getByRole("complementary", { name: "来源详情" });
  await expect(panel.locator(".citation-panel-meta")).toContainText("词法命中");
  await expect(panel).not.toContainText("排序分数");
  await expect(panel).not.toContainText("0%");

  // 下一个来源同样不伪造分数
  await panel.getByRole("button", { name: "下一个来源" }).click();
  await expect(panel.locator(".citation-panel-meta")).toContainText("词法命中");
  await expect(panel).not.toContainText("0%");
  expectRunOnlyTraffic(stub);
});

test("设置页相关性筛选：状态、档案验证日期、密钥不回显、连接测试不代表质量通过", async ({ page }) => {
  const stub = await stubShell(page);
  let tested: unknown = null;
  await page.route("**/api/settings/rerank/test", (route) => {
    tested = route.request().postDataJSON();
    return route.fulfill({
      json: {
        success: true,
        quality_checked: false,
        model: "BAAI/bge-reranker-v2-m3",
        endpoint: "https://api.siliconflow.cn/v1/rerank",
        profile_id: "profile-live",
        latency_ms: 312,
        candidate_count: 2,
        scored_count: 2,
        trace_id: "tr-1",
        usage_reported: true,
      },
    });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "设置", exact: true }).click();
  await page.getByRole("button", { name: "相关性筛选", exact: true }).click();

  const section = page.locator(".settings-rerank");
  await expect(section.getByRole("heading", { name: "资料相关性筛选" })).toBeVisible();
  // 日常页面不展示 RRF / tokenizer 等实现参数
  await expect(section).not.toContainText("RRF");
  await expect(section).not.toContainText("tokenizer");
  // 说明外呼内容为本轮问题与候选片段
  await expect(section).toContainText("本轮问题 + 候选资料片段");
  await expect(section).toContainText("完整原文不会外发");
  // 状态 / 档案状态 / 验证日期 / 只显示末四位
  await expect(section).toContainText("当前状态：已就绪");
  await expect(section).toContainText("档案状态：已验证");
  await expect(section).toContainText("验证日期：2026-08-01");
  await expect(section).toContainText(SAVED_KEY_HINT);
  await expect(page.getByLabel("筛选服务 API Key")).toHaveValue("");

  // 未保存的字段可以提交测试；密钥不持久化、不回显到页面文本
  await page.getByLabel("筛选服务 API Key").fill(UNSAVED_KEY);
  await page.getByRole("button", { name: "测试相关性筛选连接" }).click();
  const dialog = page.getByRole("dialog", { name: "筛选服务连接成功", exact: true });
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText("耗时 312 毫秒");
  await expect(dialog).toContainText("返回 2 条排序分数");
  await expect(dialog).toContainText("不代表质量验收通过");
  await expect(dialog).toContainText("quality_checked=false");
  expect(tested).toMatchObject({
    endpoint: "https://api.siliconflow.cn/v1/rerank",
    model: "BAAI/bge-reranker-v2-m3",
    profile_id: "profile-live",
    api_key: UNSAVED_KEY,
  });
  await dialog.getByRole("button", { name: "关闭", exact: true }).click();
  expect(await page.locator("body").innerText()).not.toContain(UNSAVED_KEY);
  expectNoLegacyChatTraffic(stub);
});

test("设置页相关性筛选：启用保存写入 rerank 字段，422 显示后端原因", async ({ page }) => {
  const stub = await stubShell(page);
  let view: Record<string, unknown> = structuredClone(SETTINGS_VIEW) as unknown as Record<string, unknown>;
  let putBody: Record<string, unknown> | null = null;
  let reject = false;
  await page.route("**/api/settings", (route) => {
    if (route.request().method() === "GET") {
      return route.fulfill({ json: view });
    }
    putBody = route.request().postDataJSON() as Record<string, unknown>;
    if (reject) {
      return route.fulfill({ status: 422, json: { detail: "当前模型或设置与验证档案不匹配" } });
    }
    view = { ...view, rerank: { ...RERANK_VIEW, enabled: true } };
    return route.fulfill({ json: view });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "设置", exact: true }).click();
  await page.getByRole("button", { name: "相关性筛选", exact: true }).click();

  await page.getByLabel("启用资料相关性筛选").check();
  await page.getByLabel("筛选服务 API Key").fill(UNSAVED_KEY);
  await page.getByRole("combobox", { name: "校准档案" }).click();
  await page.getByRole("option", { name: /profile-live/ }).click();
  await page.getByRole("button", { name: "保存", exact: true }).click();
  await expect(page.getByText("已保存", { exact: true })).toBeVisible();
  expect(putBody).toMatchObject({
    rerank: {
      enabled: true,
      endpoint: "https://api.siliconflow.cn/v1/rerank",
      model: "BAAI/bge-reranker-v2-m3",
      api_key: UNSAVED_KEY,
      profile_id: "profile-live",
    },
  });

  // 服务端拒绝（422）时如实展示原因，不假装保存成功
  reject = true;
  await page.getByRole("button", { name: "保存", exact: true }).click();
  await expect(page.locator(".settings-error")).toHaveText("当前模型或设置与验证档案不匹配");
  await expect(page.getByRole("dialog", { name: "设置", exact: true })).toBeVisible();
  expectNoLegacyChatTraffic(stub);
});

test("设置页相关性筛选：手机与桌面布局不溢出", async ({ page }) => {
  const stub = await stubShell(page);
  await page.goto("/");
  await page.getByRole("button", { name: "设置", exact: true }).click();
  for (const width of [1280, 390]) {
    await page.setViewportSize({ width, height: 844 });
    await page.getByRole("button", { name: "相关性筛选", exact: true }).click();
    const section = page.locator(".settings-rerank");
    await expect(section.getByRole("heading", { name: "资料相关性筛选" })).toBeVisible();
    for (const target of [
      section,
      section.getByRole("button", { name: "测试相关性筛选连接" }),
      section.getByLabel("筛选服务 API Key"),
    ]) {
      const box = await target.boundingBox();
      expect(box!.x).toBeGreaterThanOrEqual(0);
      expect(box!.x + box!.width).toBeLessThanOrEqual(width);
    }
  }
  expectNoLegacyChatTraffic(stub);
});

test("设置页相关性筛选：启用但缺档案时前端拦截，不发起保存", async ({ page }) => {
  let puts = 0;
  const viewWithoutProfiles = structuredClone(SETTINGS_VIEW);
  viewWithoutProfiles.rerank = { ...RERANK_VIEW, enabled: false, profile_id: null, profiles: [] };
  const stub = await stubShell(page, viewWithoutProfiles);
  await page.route("**/api/settings", (route) => {
    if (route.request().method() === "PUT") {
      puts += 1;
      return route.fulfill({ json: viewWithoutProfiles });
    }
    return route.fulfill({ json: viewWithoutProfiles });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "设置", exact: true }).click();
  await page.getByRole("button", { name: "相关性筛选", exact: true }).click();
  await expect(page.locator(".settings-rerank")).toContainText("暂无可用档案");

  await page.getByLabel("启用资料相关性筛选").check();
  await page.getByRole("button", { name: "保存", exact: true }).click();
  await expect(page.locator(".settings-error")).toHaveText(
    "启用资料相关性筛选需要服务地址、模型、API Key 与校准档案齐备",
  );
  expect(puts).toBe(0);
  expectNoLegacyChatTraffic(stub);
});
