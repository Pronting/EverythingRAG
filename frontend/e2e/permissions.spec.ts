import { expect, test, type Page } from "@playwright/test";
import {
  answerDelta,
  expandProcessArea,
  expectRunOnlyTraffic,
  runCompleted,
  runStatus,
  stubRunBackend,
} from "./runFixtures";

/**
 * 权限菜单与思考菜单（PRD 14.1）。
 *
 * 口径：菜单展示**当前会话实际设置**；默认「请求批准 + 关闭思考」；用户改动后必须
 * 真正落到创建任务的请求体（`permission_mode` / `reasoning_level`），并由过程区域
 * 记录本轮**实际使用值**。
 */

const SIMPLE_FRAMES = [runStatus("running"), answerDelta("回答"), runCompleted()];

/** 推理等级在模型选择器的父菜单下：先打开选择器，再进「推理等级」子面板。 */
async function openReasoningMenu(page: Page): Promise<void> {
  await page.getByTestId("model-picker").click();
  await page.getByTestId("reasoning-menu").click();
}

test("权限菜单与「模型与推理等级」选择器位于输入框底部工具行", async ({ page }) => {
  await stubRunBackend(page, { frames: SIMPLE_FRAMES });
  await page.goto("/");

  const input = await page.getByRole("textbox", { name: "消息输入框" }).boundingBox();
  const attach = await page.getByRole("button", { name: "贴图" }).boundingBox();
  const permission = await page.getByTestId("permission-menu").boundingBox();
  const picker = await page.getByTestId("model-picker").boundingBox();

  expect(input).not.toBeNull();
  expect(attach).not.toBeNull();
  expect(permission).not.toBeNull();
  expect(picker).not.toBeNull();
  expect(permission!.y).toBeGreaterThan(input!.y + input!.height - 2);
  expect(
    Math.abs(permission!.y + permission!.height / 2 - (attach!.y + attach!.height / 2)),
  ).toBeLessThan(4);
  expect(
    Math.abs(picker!.y + picker!.height / 2 - (attach!.y + attach!.height / 2)),
  ).toBeLessThan(4);

  // 推理等级不再单独占一个工具位：它和模型选择同在模型选择器的父菜单下
  await page.getByTestId("model-picker").click();
  await expect(page.getByTestId("model-picker-model-cell")).toBeVisible();
  await expect(page.getByTestId("reasoning-menu")).toBeVisible();
});

test("权限菜单与「模型与推理等级」默认是「请求批准」与「关闭思考」", async ({ page }) => {
  const stub = await stubRunBackend(page, { frames: SIMPLE_FRAMES });
  await page.goto("/");

  await expect(page.getByTestId("permission-menu")).toHaveAttribute("data-value", "ask");
  // 推理等级在父菜单里，触发器上直接显示当前档位
  await expect(page.getByTestId("model-picker")).toContainText("关闭思考");

  // 选项文案与 PRD 14.1 一致
  const permission = page.getByTestId("permission-menu");
  await permission.click();
  await expect(
    page.getByRole("listbox", { name: "权限模式" }).getByRole("option").locator("span:first-child"),
  ).toHaveText(["请求批准", "自动批准", "完全访问权限"]);
  await page.keyboard.press("Escape");

  await page.getByTestId("model-picker").click();
  await expect(page.getByTestId("reasoning-menu")).toHaveAttribute("data-value", "off");
  await page.getByTestId("reasoning-menu").click();
  await expect(
    page.getByRole("listbox", { name: "推理等级" }).getByRole("option").locator("span:first-child"),
  ).toHaveText(["关闭思考", "轻度思考", "深度思考", "最高思考"]);
  await page.keyboard.press("Escape");

  // 默认设置也要落到请求体
  await page.getByRole("textbox", { name: "消息输入框" }).fill("默认设置");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await expect.poll(() => stub.createRequests.length).toBe(1);
  expect(stub.createRequests[0]).toEqual({ permissionMode: "ask", reasoningLevel: "off" });
  expectRunOnlyTraffic(stub);
});

test("切换到完全访问权限与最高思考后落到创建任务的请求体", async ({ page }) => {
  const stub = await stubRunBackend(page, { frames: SIMPLE_FRAMES });
  await page.goto("/");

  await page.getByTestId("permission-menu").click();
  await page.getByRole("option", { name: "完全访问权限" }).click();
  await openReasoningMenu(page);
  await page.getByRole("option", { name: "最高思考" }).click();
  await expect(page.getByTestId("permission-menu")).toHaveAttribute("data-value", "full");
  await expect(page.getByTestId("model-picker")).toContainText("最高思考");

  await page.getByRole("textbox", { name: "消息输入框" }).fill("改设置");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await expect(page.locator(".markdown-body").first()).toContainText("回答");

  await expect.poll(() => stub.createRequests.length).toBe(1);
  expect(stub.createRequests[0]).toEqual({ permissionMode: "full", reasoningLevel: "max" });

  // 过程时间线记录本轮**实际使用值**（后端详情回读，不是菜单草稿）；
  // 正文在场 ⇒ 过程区域默认折叠，断言前先展开。
  await expandProcessArea(page);
  await expect(page.getByTestId("process-entries").first()).toContainText(
    "本轮设置：完全访问权限 · 最高思考",
  );
  expectRunOnlyTraffic(stub);
});

test("设置在同一会话内保持，并在后续轮次继续落到请求体", async ({ page }) => {
  const stub = await stubRunBackend(page, { frames: SIMPLE_FRAMES });
  await page.goto("/");

  await page.getByTestId("permission-menu").click();
  await page.getByRole("option", { name: "自动批准" }).click();
  await openReasoningMenu(page);
  await page.getByRole("option", { name: "轻度思考" }).click();

  const input = page.getByRole("textbox", { name: "消息输入框" });
  await input.fill("第一次");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await expect(page.locator(".markdown-body").first()).toContainText("回答");

  // 菜单值不被重置
  await expect(page.getByTestId("permission-menu")).toHaveAttribute("data-value", "auto");
  await expect(page.getByTestId("model-picker")).toContainText("轻度思考");

  await input.fill("第二次");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await expect.poll(() => stub.createRequests.length).toBe(2);
  expect(stub.createRequests[1]).toEqual({ permissionMode: "auto", reasoningLevel: "low" });

  // 两轮各自记录实际使用值（每轮一个过程区域；默认折叠 ⇒ 逐轮展开后再断言）
  await expect(page.getByTestId("process-area")).toHaveCount(2);
  await expandProcessArea(page, 0);
  await expect(page.getByTestId("process-entries").first()).toContainText(
    "本轮设置：自动批准 · 轻度思考",
  );
  expectRunOnlyTraffic(stub);
});

test("发送中菜单不可改（避免改了菜单却与在飞请求不一致）", async ({ page }) => {
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const stub = await stubRunBackend(page, { frames: SIMPLE_FRAMES, gate });
  await page.goto("/");

  await page.getByRole("textbox", { name: "消息输入框" }).fill("发送中");
  await page.getByRole("button", { name: "发送", exact: true }).click();

  await expect(page.getByTestId("permission-menu")).toBeDisabled();
  await expect(page.getByTestId("model-picker")).toBeDisabled();

  release();
  await expect(page.getByTestId("permission-menu")).toBeEnabled();
  await expect(page.getByTestId("model-picker")).toBeEnabled();
  expectRunOnlyTraffic(stub);
});
