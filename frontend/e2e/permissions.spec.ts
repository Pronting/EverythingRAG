import { expect, test } from "@playwright/test";
import { answerDelta, expectRunOnlyTraffic, runCompleted, runStatus, stubRunBackend } from "./runFixtures";

/**
 * 权限菜单与思考菜单（PRD 14.1）。
 *
 * 口径：菜单展示**当前会话实际设置**；默认「请求批准 + 深度思考」；用户改动后必须
 * 真正落到创建任务的请求体（`permission_mode` / `reasoning_level`），并由过程区域
 * 记录本轮**实际使用值**。
 */

const SIMPLE_FRAMES = [runStatus("running"), answerDelta("回答"), runCompleted()];

test("权限菜单与思考菜单默认是「请求批准」与「深度思考」", async ({ page }) => {
  const stub = await stubRunBackend(page, { frames: SIMPLE_FRAMES });
  await page.goto("/");

  await expect(page.getByTestId("permission-menu")).toHaveValue("ask");
  await expect(page.getByTestId("reasoning-menu")).toHaveValue("high");

  // 选项文案与 PRD 14.1 一致
  const permission = page.getByTestId("permission-menu");
  await expect(permission.locator("option")).toHaveText([
    "请求批准",
    "自动批准",
    "完全访问权限",
  ]);
  const reasoning = page.getByTestId("reasoning-menu");
  await expect(reasoning.locator("option")).toHaveText([
    "关闭思考",
    "轻度思考",
    "深度思考",
    "最高思考",
  ]);

  // 默认设置也要落到请求体
  await page.getByRole("textbox", { name: "消息输入框" }).fill("默认设置");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await expect.poll(() => stub.createRequests.length).toBe(1);
  expect(stub.createRequests[0]).toEqual({ permissionMode: "ask", reasoningLevel: "high" });
  expectRunOnlyTraffic(stub);
});

test("切换到完全访问权限与最高思考后落到创建任务的请求体", async ({ page }) => {
  const stub = await stubRunBackend(page, { frames: SIMPLE_FRAMES });
  await page.goto("/");

  await page.getByTestId("permission-menu").selectOption("full");
  await page.getByTestId("reasoning-menu").selectOption("max");
  await expect(page.getByTestId("permission-menu")).toHaveValue("full");
  await expect(page.getByTestId("reasoning-menu")).toHaveValue("max");

  await page.getByRole("textbox", { name: "消息输入框" }).fill("改设置");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await expect(page.locator(".markdown-body").first()).toContainText("回答");

  await expect.poll(() => stub.createRequests.length).toBe(1);
  expect(stub.createRequests[0]).toEqual({ permissionMode: "full", reasoningLevel: "max" });

  // 过程区域记录本轮**实际使用值**（后端详情回读，不是菜单草稿）
  await page.getByTestId("process-toggle").first().click();
  await expect(page.getByTestId("process-entries").first()).toContainText(
    "本轮设置：完全访问权限 · 最高思考",
  );
  expectRunOnlyTraffic(stub);
});

test("设置在同一会话内保持，并在后续轮次继续落到请求体", async ({ page }) => {
  const stub = await stubRunBackend(page, { frames: SIMPLE_FRAMES });
  await page.goto("/");

  await page.getByTestId("permission-menu").selectOption("auto");
  await page.getByTestId("reasoning-menu").selectOption("low");

  const input = page.getByRole("textbox", { name: "消息输入框" });
  await input.fill("第一次");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await expect(page.locator(".markdown-body").first()).toContainText("回答");

  // 菜单值不被重置
  await expect(page.getByTestId("permission-menu")).toHaveValue("auto");
  await expect(page.getByTestId("reasoning-menu")).toHaveValue("low");

  await input.fill("第二次");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await expect.poll(() => stub.createRequests.length).toBe(2);
  expect(stub.createRequests[1]).toEqual({ permissionMode: "auto", reasoningLevel: "low" });

  // 两轮各自记录实际使用值（条目默认折叠，先展开第一轮）
  await expect(page.getByTestId("process-toggle")).toHaveCount(2);
  await page.getByTestId("process-toggle").first().click();
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
  await expect(page.getByTestId("reasoning-menu")).toBeDisabled();

  release();
  await expect(page.getByTestId("permission-menu")).toBeEnabled();
  expectRunOnlyTraffic(stub);
});
