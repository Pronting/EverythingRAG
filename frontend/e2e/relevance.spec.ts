import { expect, test, type Page } from "@playwright/test";

/**
 * RAG-REL-001 前端接入（PRD §8.1/§8.2/§8.3）：
 * 设置页「资料相关性筛选」、问答进度、来源的「已筛选资料 / 排序分数」与旧记录兼容。
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

function encodeSse(frames: unknown[]): string {
  return frames.map((frame) => `data: ${JSON.stringify(frame)}\n\n`).join("");
}

/** 基础后端 stub：App 启动需要 status/settings/workspaces/conversations。 */
async function stubShell(page: Page, settingsView: unknown = SETTINGS_VIEW): Promise<void> {
  await page.route("**/api/status", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(STATUS_BODY) }),
  );
  await page.route("**/api/settings", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(settingsView) }),
  );
  await page.route("**/api/workspaces**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([
        { id: "default", name: "默认", is_default: true, created_at: "", updated_at: "", conversation_count: 0 },
      ]),
    }),
  );
  await page.route("**/api/conversations**", (route) => {
    const url = new URL(route.request().url());
    if (route.request().method() === "POST") {
      return route.fulfill({
        status: 201,
        contentType: "application/json",
        body: JSON.stringify({ id: "conv-1", title: "新对话", message_count: 0, workspace_id: "default" }),
      });
    }
    if (url.pathname === "/api/conversations") {
      return route.fulfill({ status: 200, contentType: "application/json", body: "[]" });
    }
    return route.fulfill({ status: 200, contentType: "application/json", body: "{}" });
  });
  await page.route("**/api/blocks", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(BLOCKS_BODY) }),
  );
}

async function send(page: Page, message: string): Promise<void> {
  await page.getByRole("textbox", { name: "消息输入框" }).fill(message);
  await page.getByRole("button", { name: "发送", exact: true }).click();
}

const FILTERED_FRAMES = [
  { type: "tool", tool: "rerank", status: "running" },
  {
    type: "meta",
    answer_basis: "knowledge",
    policy_version: "rag-policy-v3",
    query_mode: "CONTENT_QA",
    retrieval: { status: "filtered", profile_id: "profile-live", candidate_count: 30, selected_count: 1 },
    sources: [
      {
        source_id: "S1",
        block_id: "chunk-a",
        text: "本轮实际采用的证据片段。",
        source_file: "notes.md",
        similarity: 0.88,
        match_type: "rerank",
        heading_path: "项目笔记 > 概述",
        anchor: "sec-overview",
        chunk_type: "text",
        platform: "local",
        relevance_checked: true,
        rerank_score: 0.812,
        score_type: "rerank",
        profile_id: "profile-live",
        evidence_hash: "deadbeef",
        member_refs: [
          { block_id: "chunk-a", content_hash: "hash-a", relation: "primary", start: 120, end: 480, block_char_offset: 120 },
          { block_id: "chunk-b", content_hash: "hash-b", relation: "supplement", start: 0, end: null, block_char_offset: 0 },
        ],
      },
    ],
  },
  { type: "tool", tool: "rerank", status: "done" },
  { type: "token", text: "答案见 [S1]。" },
  { type: "done" },
];

/** 旧记录 / 未启用重排：只有向量相似度与匹配方式标签。 */
const LEGACY_FRAMES = [
  {
    type: "meta",
    answer_basis: "knowledge",
    policy_version: "rag-policy-v3",
    query_mode: "CONTENT_QA",
    retrieval: { status: "disabled", profile_id: "", candidate_count: 0, selected_count: 2 },
    sources: [
      {
        source_id: "S1",
        block_id: "legacy-1",
        text: "旧记录的向量命中片段。",
        source_file: "legacy.md",
        similarity: 0.92,
        heading_path: "旧章节",
        anchor: null,
        chunk_type: "text",
        platform: "local",
      },
      {
        source_id: "S2",
        block_id: "legacy-2",
        text: "纯词法命中的旧片段。",
        source_file: "legacy.md",
        similarity: null,
        match_type: "lexical",
        heading_path: "旧章节",
        anchor: null,
        chunk_type: "text",
        platform: "local",
      },
    ],
  },
  { type: "token", text: "旧记录回答。" },
  { type: "done" },
];

test("问答显示「正在筛选资料」进度，结束后正常出答案", async ({ page }) => {
  // /api/chat 不 stub：由 dev server 的慢速 SSE 夹具提供（含 rerank running 帧）。
  await stubShell(page);
  await page.goto("/");
  await send(page, "筛选进度测试");

  // 评分期间不得先出答案：只有进度提示
  await expect(page.getByText("正在筛选资料", { exact: false })).toBeVisible();
  await expect(page.getByText("已经生成的前半")).toBeVisible();

  // 用户可见的进度文案不暴露 RRF / tokenizer 等实现参数
  const progress = page.locator(".chat-searching");
  await expect(progress).not.toContainText("RRF");
  await expect(progress).not.toContainText("tokenizer");

  // 停止生成后进度关闭
  await page.getByRole("button", { name: "停止生成" }).click();
  await expect(page.getByText("已停止生成")).toBeVisible();
  await expect(page.locator(".chat-searching")).not.toBeVisible();
});

test("已筛选来源展示排序分数、组合成员与完整原文入口", async ({ page }) => {
  await stubShell(page);
  await page.route("**/api/chat", (route) =>
    route.fulfill({ status: 200, contentType: "text/event-stream", body: encodeSse(FILTERED_FRAMES) }),
  );
  await page.goto("/");
  await send(page, "已筛选来源测试");
  await expect(page.getByText("答案见")).toBeVisible();

  // 来源脚注标「已筛选资料」，chip 不再是向量百分比
  await expect(page.locator(".source-footer-label")).toHaveText("已筛选资料");
  await expect(page.locator(".source-chip-sim")).toHaveText("已筛选");

  // 详细面板：排序分数 0.812，且绝不显示成正确率 / 概率百分比
  await page.getByRole("button", { name: /notes\.md/ }).click();
  const panel = page.getByRole("complementary", { name: "来源详情" });
  await expect(panel.locator(".citation-panel-badge")).toHaveText("已筛选资料");
  await expect(panel.locator(".citation-panel-score")).toHaveText("排序分数 0.812");
  await expect(panel.locator(".citation-panel-score")).not.toContainText("%");
  await expect(panel).not.toContainText("正确率");
  await expect(panel).not.toContainText("概率");
  // 原始向量分数保持独立字段，明确叫「向量相似度」
  await expect(panel.locator(".citation-panel-meta")).toContainText("向量相似度 88%");
  // 面板展示的就是本轮实际采用的 evidence_text
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
  const browser = page.getByRole("dialog", { name: "语义块浏览" });
  await expect(browser).toBeVisible();
  const detail = page.getByRole("dialog", { name: "块详情" });
  await expect(detail.getByText("补充成员的完整原文全文。")).toBeVisible();
  await expect(detail.getByText("chunk-b")).toBeVisible();
});

test("未启用筛选的来源保留匹配方式标签与「向量相似度」命名", async ({ page }) => {
  await stubShell(page);
  await page.route("**/api/chat", (route) =>
    route.fulfill({ status: 200, contentType: "text/event-stream", body: encodeSse(LEGACY_FRAMES) }),
  );
  await page.goto("/");
  await send(page, "旧记录测试");
  await expect(page.getByText("旧记录回答。")).toBeVisible();

  // 未筛选：脚注仍是「来源」，匹配方式标签保留
  await expect(page.locator(".source-footer-label")).toHaveText("来源");
  await expect(page.locator(".source-chip-sim").nth(0)).toHaveText("92%");
  await expect(page.locator(".source-chip-sim").nth(1)).toHaveText("词法命中");
  await expect(page.getByText("已筛选资料")).not.toBeVisible();

  // 旧记录原始数值明确叫「向量相似度」，且不出现「排序分数」
  await page.getByRole("button", { name: /legacy\.md/ }).first().click();
  const panel = page.getByRole("complementary", { name: "来源详情" });
  await expect(panel.locator(".citation-panel-meta")).toHaveText("向量相似度 92%");
  await expect(panel).not.toContainText("排序分数");

  // 纯词法来源没有向量分数，绝不伪造 0%
  await panel.getByRole("button", { name: "下一个来源" }).click();
  await expect(panel.locator(".citation-panel-meta")).toHaveText("词法命中");
  await expect(panel).not.toContainText("0%");
});

test("设置页相关性筛选：状态、档案验证日期、密钥不回显、连接测试不代表质量通过", async ({ page }) => {
  await stubShell(page);
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
});

test("设置页相关性筛选：启用保存写入 rerank 字段，422 显示后端原因", async ({ page }) => {
  await stubShell(page);
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
});

test("设置页相关性筛选：手机与桌面布局不溢出", async ({ page }) => {
  await stubShell(page);
  await page.goto("/");
  await page.getByRole("button", { name: "设置", exact: true }).click();
  for (const width of [1280, 390]) {
    await page.setViewportSize({ width, height: 844 });
    await page.getByRole("button", { name: "相关性筛选", exact: true }).click();
    const section = page.locator(".settings-rerank");
    await expect(section.getByRole("heading", { name: "资料相关性筛选" })).toBeVisible();
    for (const target of [section, section.getByRole("button", { name: "测试相关性筛选连接" }), section.getByLabel("筛选服务 API Key")]) {
      const box = await target.boundingBox();
      expect(box!.x).toBeGreaterThanOrEqual(0);
      expect(box!.x + box!.width).toBeLessThanOrEqual(width);
    }
  }
});

test("设置页相关性筛选：启用但缺档案时前端拦截，不发起保存", async ({ page }) => {
  let puts = 0;
  const viewWithoutProfiles = structuredClone(SETTINGS_VIEW);
  viewWithoutProfiles.rerank = { ...RERANK_VIEW, enabled: false, profile_id: null, profiles: [] };
  await stubShell(page);
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
});
