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
  embed: { mode: "local", base_url: null, model: null, api_key_set: false, api_key_hint: null },
  system_prompt:
    "你是 Everything RAG 个人知识助手。请严格基于给定的参考上下文作答，不要编造上下文之外的内容；若上下文无法回答该问题，请如实说明。引用来源时只能使用上下文中标注的序号，不得虚构来源。",
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

  // 对话模型已回填
  await expect(page.getByLabel("对话模型 API Key")).toHaveValue("");

  // 修改系统提示词并保存
  const prompt = page.getByRole("textbox", { name: "系统提示词" });
  await prompt.fill("你是测试助手，只用中文回答。");
  await page.getByRole("button", { name: "保存" }).click();

  await expect(page.getByText("已保存")).toBeVisible();
  expect(putBody()).toMatchObject({ system_prompt: "你是测试助手，只用中文回答。" });
});

test("设置弹窗：切换云端嵌入显示字段与重建提示", async ({ page }) => {
  await stubSettings(page);
  await page.goto("/");
  await page.getByRole("button", { name: "设置" }).click();

  await page.getByRole("button", { name: "云端" }).click();
  await expect(page.getByLabel("嵌入模型 API Key")).toBeVisible();
  await expect(page.getByLabel("嵌入模型 API Key")).toBeEnabled();
  // 切换嵌入模型 -> 重建提示
  await expect(page.getByText(/重新导入全部文档/)).toBeVisible();
});
