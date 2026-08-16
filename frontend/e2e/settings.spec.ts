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
  },
  embed: {
    mode: "cloud",
    base_url: "https://embed.example.com/v1",
    model: "bge-m3",
    api_key_set: false,
    api_key_hint: null,
  },
  system_prompt:
    "你是 Everything RAG 个人知识助手。请严格基于给定的参考上下文作答，不要编造上下文之外的内容；若上下文无法回答该问题，请如实说明。引用来源时只能使用上下文中标注的序号，不得虚构来源。",
  avatars: { user: null, agent: null },
  theme: "light",
};

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

test("设置弹窗：配置对话模型 + 保存系统提示词", async ({ page }) => {
  const { putBody } = await stubSettings(page);
  await page.goto("/");

  // 打开设置
  await page.getByRole("button", { name: "设置" }).click();
  const dialog = page.getByRole("dialog", { name: "设置" });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole("heading", { name: "对话模型" })).toBeVisible();
  await expect(dialog.getByRole("heading", { name: "嵌入模型" })).toBeVisible();
  await expect(dialog.getByRole("heading", { name: "系统提示词" })).toBeVisible();
  await expect(dialog.getByRole("heading", { name: "头像" })).toBeVisible();

  // 对话模型已回填
  await expect(page.getByLabel("对话模型 API Key")).toHaveValue("");

  // 修改系统提示词并保存
  const prompt = page.getByRole("textbox", { name: "系统提示词" });
  await prompt.fill("你是测试助手，只用中文回答。");
  await page.getByRole("button", { name: "保存" }).click();

  await expect(page.getByText("已保存")).toBeVisible();
  expect(putBody()).toMatchObject({ system_prompt: "你是测试助手，只用中文回答。" });
});

test("设置弹窗：修改嵌入模型显示重建提示", async ({ page }) => {
  await stubSettings(page);
  await page.goto("/");
  await page.getByRole("button", { name: "设置" }).click();

  // 改动嵌入模型名 -> 提示需重新导入
  await page.getByPlaceholder("Qwen/Qwen3-Embedding-8B").fill("another-embed-model");
  await expect(page.getByText(/重新导入全部文档/)).toBeVisible();
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

  // 保存 -> 触发头像上传
  await page.getByRole("button", { name: "保存" }).click();
  await expect(page.getByText("已保存")).toBeVisible();
  expect(uploaded).toBe(true);
});
