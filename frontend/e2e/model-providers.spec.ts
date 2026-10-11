import { expect, test, type Page, type Route } from "@playwright/test";

/**
 * 多供应商模型配置 + 输入框模型选择器（用户诉求：设置页可配多个供应商；
 * 输入框右下角发送键旁新增模型选择按钮；模型按供应商分组；UI 适配站点主题）。
 *
 * 全部离线（`page.route` 拦截 `/api`），验证的是**前端真实契约**：
 * - 设置页「模型」里的增删改与整表替换负载（省略 api_key = 保留，空串 = 清除）；
 * - 「获取可用模型」把结果并入模型目录（不自动保存）；
 * - 输入框模型选择器按供应商分组、可搜索、选择后写 `model_selection` 并更新触发器；
 * - 深浅色主题下组件的实际背景色跟着主题变量走（不是写死白色）。
 */

const STATUS_BODY = JSON.stringify({
  app: { name: "everything-rag", version: "0.2.0" },
  config: { wizard_completed: false, embed_configured: true, chat_configured: true },
  knowledge: { file_count: 0, chunk_count: 0, image_count: 0, last_sync_at: null, needs_rebuild: false },
  privacy: { outbound_state: "local-only", providers: {} },
});

const DEEPSEEK = {
  id: "deepseek",
  name: "DeepSeek",
  kind: "builtin",
  protocol: "openai-completions",
  base_url: "https://api.deepseek.com/v1",
  api_key_set: true,
  api_key_hint: "...abcd",
  models: [
    { id: "deepseek-flash", name: "DeepSeek Flash", vision: true },
    { id: "deepseek-v4-pro", name: "DeepSeek V4 Pro", vision: false },
  ],
  status: "ready",
  selected: true,
};

const COMMANDE = {
  id: "custom-1",
  name: "commande code",
  kind: "custom",
  protocol: "openai-completions",
  base_url: "https://api.commandcode.ai/provider/v1",
  api_key_set: false,
  api_key_hint: null,
  models: [{ id: "deepseek/deepseek-v4.1-flash", name: "DeepSeek V4.1 Flash", vision: false }],
  status: "ready",
  selected: false,
};

const CATALOG = [
  {
    id: "deepseek",
    name: "DeepSeek",
    base_url: "https://api.deepseek.com/v1",
    summary: "官方端点；预置模型已内置能力声明",
    requires_api_key: true,
    docs_url: null,
    models: [{ id: "deepseek-flash", name: "DeepSeek Flash", vision: true }],
  },
  {
    id: "siliconflow",
    name: "硅基流动",
    base_url: "https://api.siliconflow.cn/v1",
    summary: "OpenAI 兼容聚合端点",
    requires_api_key: true,
    docs_url: null,
    models: [],
  },
];

interface Stub {
  putBodies: unknown[];
  modelsRequests: unknown[];
  /** 当前服务端视图（PUT 会按后端的合并语义更新它）。 */
  view: Record<string, unknown>;
}

/** 安装离线桩：status / settings（GET+PUT）/ conversations / workspaces / 获取可用模型。 */
async function stubBackend(page: Page): Promise<Stub> {
  const stub: Stub = {
    putBodies: [],
    modelsRequests: [],
    view: {
      chat: {
        provider_type: "openai_compatible",
        base_url: "https://api.deepseek.com/v1",
        model: "deepseek-flash",
        api_key_set: true,
        api_key_hint: "...abcd",
        supports_image: true,
      },
      providers: [structuredClone(DEEPSEEK), structuredClone(COMMANDE)],
      model_selection: {
        provider_id: "deepseek",
        provider_name: "DeepSeek",
        model_id: "deepseek-flash",
        model_name: "DeepSeek Flash",
        vision: true,
      },
      model_catalog: CATALOG,
      model_protocols: ["openai-completions"],
      embed: { mode: "cloud", base_url: null, model: null, api_key_set: false, api_key_hint: null },
      vision: { provider_type: "openai_compatible", base_url: null, model: null, api_key_set: false, api_key_hint: null },
      search: { provider_type: "tavily", enabled: false, base_url: null, max_results: 5, api_key_set: false, api_key_hint: null },
      rerank: { enabled: false, status: "disabled", profiles: [], profile: null, profile_id: null, provider_type: "siliconflow_rerank", endpoint: "https://api.siliconflow.cn/v1/rerank", model: null, api_key_set: false, api_key_hint: null, connect_timeout_seconds: 5, read_timeout_seconds: 60, configured: false, acceptance_env: false },
      memory_enabled: true,
      answer_preferences: { language: "auto", verbosity: "balanced", tone: "natural", response_format: "auto", knowledge_only: false, custom_instructions: "" },
      policy: { version: "rag-policy-v3", managed: true },
      avatars: { user: null, agent: null },
      theme: "light",
    },
  };

  await page.route("**/api/status", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: STATUS_BODY }),
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
  await page.route("**/api/conversations**", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify([]) }),
  );
  await page.route("**/api/settings/providers/models", (route) => {
    stub.modelsRequests.push(route.request().postDataJSON());
    return route.fulfill({
      json: {
        success: true,
        message: "获取到 2 个模型",
        models: [{ id: "deepseek-chat" }, { id: "deepseek-reasoner" }],
        count: 2,
        latency_ms: 18,
      },
    });
  });
  await page.route("**/api/settings", (route: Route) => {
    if (route.request().method() !== "PUT") {
      return route.fulfill({ json: stub.view });
    }
    const update = (route.request().postDataJSON() ?? {}) as Record<string, unknown>;
    stub.putBodies.push(update);
    // 后端语义：providers 整表替换（省略即删除），model_selection 决定生效 chat。
    if (Array.isArray(update.providers)) {
      stub.view.providers = update.providers;
      const selection = stub.view.model_selection as { provider_id: string; model_id: string } | null;
      const stillThere =
        selection !== null &&
        (update.providers as { id: string; models: { id: string }[] }[]).some(
          (provider) =>
            provider.id === selection.provider_id &&
            provider.models.some((model) => model.id === selection.model_id),
        );
      if (!stillThere) {
        stub.view.model_selection = null;
        (stub.view.chat as Record<string, unknown>).base_url = null;
        (stub.view.chat as Record<string, unknown>).model = null;
      }
    }
    if (update.model_selection !== undefined) {
      const next = update.model_selection as { provider_id: string | null; model_id: string | null };
      // 与后端 `selection_view` 同语义：从供应商注册表解析供应商名与**显示**模型名
      const providers = stub.view.providers as {
        id: string;
        name: string;
        models: { id: string; name: string; vision: boolean }[];
      }[];
      const provider = providers.find((item) => item.id === next.provider_id);
      const model = provider?.models.find((item) => item.id === next.model_id);
      if (next.provider_id === null || next.model_id === null || provider === undefined || model === undefined) {
        stub.view.model_selection = null;
      } else {
        stub.view.model_selection = {
          provider_id: provider.id,
          provider_name: provider.name,
          model_id: model.id,
          model_name: model.name || model.id,
          vision: model.vision,
        };
      }
      const chosen = stub.view.model_selection as { model_id: string } | null;
      (stub.view.chat as Record<string, unknown>).model = chosen?.model_id ?? null;
    }
    return route.fulfill({ json: stub.view });
  });
  return stub;
}

async function openSettingsModels(page: Page): Promise<void> {
  await page.getByRole("button", { name: "设置", exact: true }).click();
  await page.getByRole("button", { name: "模型服务", exact: true }).click();
}

test("设置页模型列表：预置供应商可编辑、自定义供应商可删除、可按目录添加提供方", async ({ page }) => {
  const stub = await stubBackend(page);
  await page.goto("/");
  await openSettingsModels(page);

  // 初始：DeepSeek（使用中，无删除）+ commande code（自定义，有删除）
  await expect(page.getByTestId("provider-deepseek").getByText("使用中")).toBeVisible();
  await expect(page.getByTestId("provider-deepseek").getByRole("button", { name: "删除" })).toHaveCount(0);
  await expect(page.getByTestId("provider-custom-1").getByRole("button", { name: "删除" })).toBeVisible();

  // 添加预置提供方：目录里 deepseek 已添加，只剩硅基流动可选
  await page.getByRole("button", { name: "+ 添加提供方" }).click();
  await expect(page.getByRole("list", { name: "预置提供方" }).getByText("硅基流动")).toBeVisible();
  await page.getByRole("list", { name: "预置提供方" }).getByRole("button", { name: /硅基流动/ }).click();
  await expect(page.getByTestId("provider-siliconflow")).toBeVisible();

  // 添加自定义提供方 + 填一条模型
  await page.getByRole("button", { name: "+ 添加自定义提供方" }).click();
  const custom2 = page.getByTestId("provider-custom-2");
  await expect(custom2).toHaveAttribute("data-expanded", "true");
  await custom2.getByLabel("custom-2 显示名称").fill("本地网关");
  await custom2.getByLabel("本地网关 API 地址").fill("http://127.0.0.1:8000/v1");
  await custom2.getByLabel("本地网关 API 密钥").fill("sk-local-0001");
  await page.getByRole("button", { name: "+ 添加模型" }).last().click();
  await custom2.getByLabel("本地网关 模型 1 名称").fill("qwen3-8b");
  await custom2.getByLabel("本地网关 模型 1 显示名").fill("Qwen3 8B");
  await custom2.getByLabel("本地网关 模型 1 是否支持图片").check();
  // 选用这条新模型
  await custom2.getByRole("button", { name: "使用 Qwen3 8B" }).click();

  await page.getByRole("button", { name: "保存", exact: true }).click();
  await expect(page.getByText("已保存", { exact: true })).toBeVisible();

  const body = stub.putBodies.at(-1) as {
    providers: { id: string; name: string; base_url: string | null; api_key?: string; models: { id: string; name: string | null; vision: boolean }[] }[];
    model_selection: { provider_id: string; model_id: string };
  };
  expect(body.providers.map((item) => item.id)).toEqual(["deepseek", "custom-1", "siliconflow", "custom-2"]);
  const added = body.providers.find((item) => item.id === "custom-2")!;
  expect(added).toMatchObject({ name: "本地网关", base_url: "http://127.0.0.1:8000/v1", api_key: "sk-local-0001" });
  expect(added.models).toEqual([{ id: "qwen3-8b", name: "Qwen3 8B", vision: true }]);
  // 未改动的供应商不带 api_key（保留服务端已保存的密钥）
  expect(body.providers.find((item) => item.id === "deepseek")!.api_key).toBeUndefined();
  expect(body.model_selection).toEqual({ provider_id: "custom-2", model_id: "qwen3-8b" });
});

test("删除当前使用的供应商后，界面明确回到「未选择模型」", async ({ page }) => {
  const stub = await stubBackend(page);
  await page.goto("/");
  await openSettingsModels(page);

  await page.getByTestId("provider-deepseek").getByRole("button", { name: "编辑" }).click();
  await page.getByRole("button", { name: "删除模型 DeepSeek Flash" }).click();
  await page.getByRole("button", { name: "删除模型 DeepSeek V4 Pro" }).click();
  await page.getByRole("button", { name: "保存", exact: true }).click();
  await expect(page.getByText("已保存", { exact: true })).toBeVisible();

  // 服务端把选择清空（模型被删）→ 界面不再显示「使用中」，输入框触发器回到「选择模型」
  await expect(page.getByTestId("provider-deepseek").getByText("使用中")).toHaveCount(0);
  await page.getByRole("button", { name: "关闭设置" }).click();
  await expect(page.getByTestId("model-picker")).toContainText("选择模型");
  expect((stub.view.chat as { model: string | null }).model).toBeNull();
});

test("获取可用模型把结果并入模型目录（保存前不落库）", async ({ page }) => {
  const stub = await stubBackend(page);
  await page.goto("/");
  await openSettingsModels(page);

  await page.getByTestId("provider-custom-1").getByRole("button", { name: "编辑" }).click();
  await page.getByTestId("provider-fetch-models-custom-1").click();
  await expect(page.getByText(/获取到 2 个模型，新增 2 个/)).toBeVisible();
  await expect(page.getByTestId("provider-models-count-custom-1")).toHaveText("已配置 3 个模型");
  // 请求带该供应商的地址；密钥未填写时省略（服务端只在地址一致时复用已保存的密钥）
  expect(stub.modelsRequests[0]).toEqual({
    provider_id: "custom-1",
    base_url: "https://api.commandcode.ai/provider/v1",
  });
  // 未点保存 -> 没有任何 PUT
  expect(stub.putBodies).toHaveLength(0);
});

test("输入框模型选择器：按供应商分组、可搜索、选择后写设置并更新触发器", async ({ page }) => {
  const stub = await stubBackend(page);
  await page.goto("/");

  const picker = page.getByTestId("model-picker");
  await expect(picker).toContainText("DeepSeek Flash");
  await picker.click();
  // 父菜单根面板：先选「模型」才进模型清单（推理等级与它同一个父菜单）
  await expect(page.getByTestId("reasoning-menu")).toBeVisible();
  await page.getByTestId("model-picker-model-cell").click();

  // 按供应商分组：组标题就是供应商名，选项在各自组内
  const deepseekGroup = page.getByTestId("model-group-deepseek");
  const commandeGroup = page.getByTestId("model-group-custom-1");
  await expect(deepseekGroup).toContainText("DeepSeek");
  await expect(deepseekGroup.getByTestId("model-option-deepseek-flash")).toBeVisible();
  await expect(commandeGroup).toContainText("commande code");
  await expect(commandeGroup.getByTestId("model-option-deepseek/deepseek-v4.1-flash")).toBeVisible();
  // 多模态模型带「视觉」标记（能力如实展示）
  await expect(deepseekGroup.getByTestId("model-option-deepseek-flash")).toContainText("视觉");

  // 搜索：按模型名过滤，只剩匹配项
  await page.getByLabel("搜索模型或服务商").fill("V4.1");
  await expect(deepseekGroup).toHaveCount(0);
  await expect(commandeGroup.getByTestId("model-option-deepseek/deepseek-v4.1-flash")).toBeVisible();
  // 按供应商名搜索：整组保留
  await page.getByLabel("搜索模型或服务商").fill("commande");
  await expect(commandeGroup.getByTestId("model-option-deepseek/deepseek-v4.1-flash")).toBeVisible();

  await commandeGroup.getByTestId("model-option-deepseek/deepseek-v4.1-flash").click();
  await expect(page.getByText(/已切换到 DeepSeek V4.1 Flash/)).toBeVisible();
  await expect(picker).toContainText("DeepSeek V4.1 Flash");
  expect(stub.putBodies.at(-1)).toMatchObject({
    model_selection: { provider_id: "custom-1", model_id: "deepseek/deepseek-v4.1-flash" },
  });
});

test("模型选择器的「管理提供方」直接落到设置-模型服务页（不能停在默认页）", async ({ page }) => {
  await stubBackend(page);
  await page.goto("/");

  await page.getByTestId("model-picker").click();
  await page.getByTestId("model-picker-model-cell").click();
  await page.getByRole("button", { name: "管理提供方" }).click();

  const dialog = page.getByRole("dialog", { name: "设置" });
  await expect(dialog).toBeVisible();
  // 落点必须是「模型服务」：供应商卡片可见 + 左侧导航该项为当前页
  await expect(page.getByTestId("provider-deepseek")).toBeVisible();
  await expect(page.getByRole("heading", { name: "模型", exact: true })).toBeVisible();
  await expect(
    page.getByRole("navigation", { name: "设置分类" }).getByRole("button", { name: "模型服务", exact: true }),
  ).toHaveAttribute("aria-current", "page");
  // 默认页（外观与头像）的内容不应同时露出来
  await expect(page.getByRole("heading", { name: "头像", exact: true })).not.toBeVisible();
});

test("模型选择器与供应商卡片适配深浅色主题（跟随主题变量，不是写死白色）", async ({ page }) => {
  await stubBackend(page);
  await page.goto("/");

  const readColors = async () => {
    await page.getByTestId("model-picker").click();
    const menu = await page
      .locator(".model-picker-menu")
      .evaluate((node) => getComputedStyle(node).backgroundColor);
    await page.keyboard.press("Escape");
    await openSettingsModels(page);
    const card = await page
      .getByTestId("provider-deepseek")
      .evaluate((node) => getComputedStyle(node).backgroundColor);
    await page.getByRole("button", { name: "关闭设置" }).click();
    return { menu, card };
  };

  const light = await readColors();
  await page.evaluate(() => {
    document.documentElement.dataset.theme = "dark";
  });
  const dark = await readColors();

  expect(light).toEqual({ menu: "rgb(255, 255, 255)", card: "rgb(255, 255, 255)" });
  expect(dark).toEqual({ menu: "rgb(25, 36, 30)", card: "rgb(25, 36, 30)" });
});
