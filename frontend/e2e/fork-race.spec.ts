import { expect, test, type Page } from "@playwright/test";
import { answerDelta, expectRunOnlyTraffic, runCompleted, runStatus, stubRunBackend } from "./runFixtures";

/**
 * 分支对话的竞态回归：fork 后消息仍在异步装载时用户就发问，
 * 这一轮结果不能追加落库（否则分支历史会被重复写一遍）。
 *
 * V1.2：消息落库由后端 run 完成，前端不再调用 `/messages` —— 本用例继续守住
 * 「装载窗口内不产生任何 `/messages` 写入」这条不变量。
 */
test("分支装载未完成时提问不重复写入历史", async ({ page }: { page: Page }) => {
  const calls: string[] = [];
  let releaseDetail!: () => void;
  const detailGate = new Promise<void>((resolve) => {
    releaseDetail = resolve;
  });

  const stub = await stubRunBackend(page, {
    frames: [runStatus("running"), answerDelta("分支里的回答"), runCompleted()],
  });

  // 工作区列表 / settings / status 由 stubRunBackend 提供；这里不要再注册
  // `**/api/workspaces**`（后注册者优先，会把同前缀的 /runs 路由一起吃掉）。
  await page.route("**/api/conversations**", async (route) => {
    const url = new URL(route.request().url());
    const segments = url.pathname.split("/").filter((part) => part !== "");
    const [, , first, second] = segments;
    calls.push(`${route.request().method()} ${url.pathname}`);
    const json = (body: unknown, status = 200) =>
      route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
    if (first === "archived") return json([]);
    if (second === "fork") {
      return json({ id: "b1", title: "源对话 · 分支", message_count: 1, created_at: "", updated_at: "", parent_id: "p1" }, 201);
    }
    if (second === "messages") return json({ ok: true });
    if (first === "b1") {
      await detailGate; // 卡住「读取分支消息」，制造装载窗口
      return json({
        id: "b1",
        title: "源对话 · 分支",
        created_at: "",
        updated_at: "",
        parent_id: "p1",
        messages: [{ role: "user", content: "继承来的历史", created_at: "" }],
      });
    }
    if (first === "p1") {
      return json({ id: "p1", title: "源对话", created_at: "", updated_at: "", messages: [] });
    }
    return json([{ id: "p1", title: "源对话", message_count: 1, created_at: "", updated_at: "" }]);
  });

  await page.goto("/");
  await page.getByText("源对话", { exact: true }).click({ button: "right" });
  await page.getByRole("menuitem", { name: /Fork 分支对话/ }).click();

  // 装载窗口内提问并等整轮结束
  const input = page.getByRole("textbox", { name: "消息输入框" });
  await input.fill("分支里的追问");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await expect(page.getByText("分支里的回答")).toBeVisible();

  releaseDetail();
  await expect(page.getByText("继承来的历史")).toBeVisible();
  expect(calls.filter((call) => call.includes("/messages"))).toHaveLength(0);
  expectRunOnlyTraffic(stub);
});
