import { expect, test, type Page } from "@playwright/test";

const STATUS_BODY = JSON.stringify({
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
});

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

test("自定义菜单支持键盘与手机布局，自定义提示词可回填并清空保存", async ({ page }) => {
  await stubSettings(page);
  let view = structuredClone(SETTINGS_VIEW);
  await page.route("**/api/settings", (route) => {
    if (route.request().method() === "PUT") {
      const update = route.request().postDataJSON();
      view = { ...view, ...update, answer_preferences: { ...view.answer_preferences, ...update.answer_preferences } };
    }
    return route.fulfill({ json: view });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "设置", exact: true }).click();
  await page.getByRole("button", { name: "回答偏好", exact: true }).click();
  const language = page.getByRole("combobox", { name: "回答语言" });
  for (const width of [1280, 390]) {
    await page.setViewportSize({ width, height: 844 });
    await language.click();
    await expect(page.getByRole("listbox", { name: "回答语言" })).toBeVisible();
    const box = await page.getByRole("listbox", { name: "回答语言" }).boundingBox();
    expect(box!.x).toBeGreaterThanOrEqual(0);
    expect(box!.x + box!.width).toBeLessThanOrEqual(width);
    await page.keyboard.press("End");
    await page.keyboard.press("Enter");
    await expect(language).toHaveText("英文");
    await expect(language).toHaveAttribute("aria-expanded", "false");
    await language.click();
    await page.keyboard.press("Escape");
    await expect(page.getByRole("dialog", { name: "设置" })).toBeVisible();
    await expect(language).toHaveAttribute("aria-expanded", "false");
  }
  const custom = page.getByRole("textbox", { name: "自定义提示词", exact: true });
  await custom.fill("从设计师的视角回答，先给结论。");
  await page.getByRole("button", { name: "保存", exact: true }).click();
  await expect(page.getByText("已保存", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "关闭设置" }).click();
  await page.getByRole("button", { name: "设置", exact: true }).click();
  await page.getByRole("button", { name: "回答偏好", exact: true }).click();
  await expect(custom).toHaveValue("从设计师的视角回答，先给结论。");
  await page.getByRole("button", { name: "清空提示词" }).click();
  await page.getByRole("button", { name: "保存", exact: true }).click();
  await expect(page.getByText("已保存", { exact: true })).toBeVisible();
  expect(view.answer_preferences.custom_instructions).toBe("");
});

/** 拦截 status 与 settings（GET 返回视图；PUT 记录请求体并返回原视图）。 */
async function stubSettings(page: Page): Promise<{ putBody: () => unknown }> {
  let captured: unknown = null;
  await page.route("**/api/status", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: STATUS_BODY }),
  );
  await page.route("**/api/settings", (route) => {
    if (route.request().method() === "GET") {
      route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(SETTINGS_VIEW) });
    } else {
      captured = route.request().postDataJSON();
      route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(SETTINGS_VIEW) });
    }
  });
  return { putBody: () => captured };
}

test("设置弹窗：核心策略只读 + 保存结构化回答偏好", async ({ page }) => {
  const { putBody } = await stubSettings(page);
  await page.goto("/");

  // 打开设置
  await page.getByRole("button", { name: "设置" }).click();
  const dialog = page.getByRole("dialog", { name: "设置" });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole("heading", { name: "头像", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "模型服务", exact: true }).click();
  await expect(dialog.getByRole("heading", { name: "对话模型" })).toBeVisible();
  await expect(dialog.getByRole("heading", { name: "嵌入模型" })).toBeVisible();
  await expect(page.getByLabel("对话模型 API Key")).toHaveValue("");
  await page.getByRole("button", { name: "回答偏好", exact: true }).click();
  await expect(dialog.getByRole("heading", { name: "回答偏好" })).toBeVisible();
  await expect(dialog.getByText(/rag-policy-v3/)).not.toBeVisible();
  await expect(dialog.getByRole("textbox", { name: "系统提示词" })).not.toBeVisible();

  // 修改结构化偏好并保存
  for (const [label, option] of [["回答语言", "简体中文"], ["回答篇幅", "简洁"], ["回答语气", "专业"], ["输出格式", "要点列表"]]) {
    await page.getByRole("combobox", { name: label }).click();
    await page.getByRole("option", { name: option, exact: true }).click();
  }
  await page.getByLabel("自定义提示词", { exact: true }).fill("请以产品设计师的视角回答，先给结论。 ");
  await page.getByLabel("仅使用本地知识").check();
  await page.getByRole("button", { name: "保存" }).click();

  await expect(page.getByText("已保存")).toBeVisible();
  expect(putBody()).toMatchObject({
    answer_preferences: {
      language: "zh-CN",
      verbosity: "concise",
      tone: "professional",
      response_format: "bullets",
      knowledge_only: true,
      custom_instructions: "请以产品设计师的视角回答，先给结论。 ",
    },
  });
  expect(putBody()).not.toHaveProperty("system_prompt");
});

test("设置弹窗：修改嵌入模型显示重建提示", async ({ page }) => {
  await stubSettings(page);
  await page.goto("/");
  await page.getByRole("button", { name: "设置" }).click();

  // 改动嵌入模型名 -> 提示需重新导入
  await page.getByRole("button", { name: "模型服务", exact: true }).click();
  await page.getByPlaceholder("Qwen/Qwen3-Embedding-8B").fill("another-embed-model");
  await expect(page.getByText(/重新导入全部文档/)).toBeVisible();
});

test("设置分类保留未保存内容，键盘焦点限制在弹窗内，取消还原主题", async ({ page }) => {
  await stubSettings(page);
  await page.goto("/");
  await page.getByRole("button", { name: "设置", exact: true }).click();
  const close = page.getByRole("button", { name: "关闭设置" });
  await close.focus();
  await page.keyboard.press("Shift+Tab");
  await expect(page.getByRole("button", { name: "保存", exact: true })).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(close).toBeFocused();
  await page.getByRole("button", { name: "模型服务", exact: true }).click();
  await page.getByPlaceholder("deepseek-chat / gpt-4o-mini").fill("draft-model");
  await page.getByRole("navigation", { name: "设置分类" }).getByRole("button", { name: "联网搜索", exact: true }).click();
  await page.getByLabel("启用联网搜索").check();
  await page.getByRole("button", { name: "模型服务", exact: true }).click();
  await expect(page.getByPlaceholder("deepseek-chat / gpt-4o-mini")).toHaveValue("draft-model");
  await page.getByRole("button", { name: "外观与头像", exact: true }).click();
  await page.getByRole("button", { name: "深色", exact: true }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog", { name: "设置" })).not.toBeVisible();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await expect(page.getByRole("button", { name: "设置", exact: true })).toBeFocused();
});

test("设置弹窗：切换深色主题实时应用并随保存持久化", async ({ page }) => {
  const { putBody } = await stubSettings(page);
  await page.goto("/");
  await page.getByRole("button", { name: "设置" }).click();

  // 实时应用：点击「深色」后 <html> 根节点 data-theme 变为 dark
  await page.getByRole("button", { name: "深色" }).click();
  await expect
    .poll(() => page.evaluate(() => document.documentElement.dataset.theme))
    .toBe("dark");

  // 保存 -> PUT 负载携带 theme=dark
  await page.getByRole("button", { name: "保存" }).click();
  await expect(page.getByText("已保存")).toBeVisible();
  expect(putBody()).toMatchObject({ theme: "dark" });
});

test("设置弹窗：选择用户头像并保存上传", async ({ page }) => {
  await stubSettings(page);
  let uploaded = false;
  await page.route("**/api/avatars/user", (route) => {
    if (route.request().method() === "POST") {
      uploaded = true;
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ kind: "user", url: "/api/avatars/user-test.png", filename: "user-test.png" }),
      });
    } else {
      route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ kind: "user", url: null }) });
    }
  });
  await page.goto("/");
  await page.getByRole("button", { name: "设置" }).click();

  // 通过隐藏 file input 选择头像（打开系统资源管理器选择文件）
  await page
    .getByLabel("选择用户头像")
    .setInputFiles({ name: "me.png", mimeType: "image/png", buffer: Buffer.from([0x89, 0x50, 0x4e, 0x47]) });

  // 预览显示文件名 + 出现「恢复默认」
  await expect(page.getByText("me.png")).toBeVisible();
  await expect(page.getByRole("button", { name: "恢复默认" })).toBeVisible();

  await page.screenshot({ path: "test-results/compact-avatar-actions.png" });

  // 保存 -> 触发头像上传
  await page.getByRole("button", { name: "保存" }).click();
  await expect(page.getByText("已保存")).toBeVisible();
  expect(uploaded).toBe(true);
});


test("精简设置的密钥操作可清除保存，手机与桌面不溢出", async ({ page }) => {
  const { putBody } = await stubSettings(page);
  await page.goto("/");
  await page.getByRole("button", { name: "设置", exact: true }).click();
  for (const width of [1280, 390]) {
    await page.setViewportSize({ width, height: 844 });
    await page.evaluate((theme) => { document.documentElement.dataset.theme = theme; }, width === 1280 ? "light" : "dark");
    await page.getByRole("button", { name: "外观与头像", exact: true }).click();
    await page.screenshot({ path: `test-results/compact-appearance-${width}.png` });
    await page.getByRole("button", { name: "模型服务", exact: true }).click();
    const key = page.getByLabel("对话模型 API Key");
    await expect(key).toBeVisible();
    const clear = page.getByRole("button", { name: "清除 Key", exact: true });
    const box = await clear.boundingBox();
    expect(box!.x + box!.width).toBeLessThanOrEqual(width);
    await page.screenshot({ path: `test-results/compact-models-${width}.png` });
  }
  await page.getByRole("button", { name: "清除 Key", exact: true }).click();
  await expect(page.getByLabel("对话模型 API Key")).toHaveValue("");
  await page.getByRole("button", { name: "保存", exact: true }).click();
  await expect(page.getByText("已保存", { exact: true })).toBeVisible();
  expect(putBody()).toMatchObject({ chat: { api_key: "" } });
});


test("模型连接测试使用未保存配置，结果弹窗支持成功和失败", async ({ page }) => {
  await stubSettings(page);
  const calls: unknown[] = [];
  await page.route("**/api/settings/test-connection", route => {
    calls.push(route.request().postDataJSON());
    return route.fulfill({ json: calls.length === 1 ? { success: true, message: "连接成功", latency_ms: 120 } : { success: false, message: "认证失败，请检查 API Key" } });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "设置", exact: true }).click();
  await page.getByRole("button", { name: "模型服务", exact: true }).click();
  await page.getByRole("textbox", { name: "模型名", exact: true }).first().fill("draft-model");
  await page.getByRole("button", { name: "测试对话模型连接" }).click();
  const result = page.getByRole("dialog", { name: "连接成功", exact: true });
  await expect(result).toBeVisible();
  await expect(result.getByText("响应耗时 120 毫秒")).toBeVisible();
  expect(calls[0]).toMatchObject({ kind: "chat", model: "draft-model" });
  await page.screenshot({ path: "test-results/model-connection-success.png" });
  await result.getByRole("button", { name: "关闭", exact: true }).click();
  await page.getByRole("button", { name: "测试嵌入模型连接" }).click();
  await expect(page.getByRole("dialog", { name: "连接失败", exact: true })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog", { name: "设置", exact: true })).toBeVisible();
});
