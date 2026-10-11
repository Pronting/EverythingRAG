import { expect, test } from "@playwright/test";
import {
  answerDelta,
  expandProcessArea,
  expectRunOnlyTraffic,
  imageRefPayload,
  planUpdated,
  runCompleted,
  runStatus,
  stubRunBackend,
} from "./runFixtures";

/**
 * 贴图收口（PRD 13.2「图片引用」+ t25 契约）。
 *
 * 贴图是**有条件生效**的：只有后端 `status=described` 才代表原图已缓存 + 描述已入库、
 * `verify_image` 本轮真能核对；`cached` / `pending_vision` / `unavailable` / `rejected`
 * 都必须明确告诉用户「本轮不参与核对」。本文件同时守住两件事：
 * 1. 图片引用**真的随请求提交**（不是前端丢弃后嘴上说支持）；
 * 2. 降级时界面**如实说明**（不伪造成功）。
 */

/** 一张 1x1 PNG 的 data URI（前端贴图就是这种形态）。 */
const PNG_A =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg==";
const PNG_B =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";

const ANSWER_FRAMES = [runStatus("running"), answerDelta("看到了图片。"), runCompleted()];

/** 通过文件选择器贴图（走真实的 FileReader → data URI 路径）。 */
async function attachImages(
  page: import("@playwright/test").Page,
  files: { name: string; body: Buffer }[],
): Promise<void> {
  // Playwright 的 setInputFiles 要 `{name, mimeType, buffer}`；这里的 `body` 只是
  // 本文件的入参名，必须映射成 `buffer`，否则会以「expected Buffer, got undefined」失败。
  await page.setInputFiles(
    'input[type="file"]',
    files.map((file) => ({ name: file.name, mimeType: "image/png", buffer: file.body })),
  );
}

test("带图片创建 run：请求体 images 含 data URI，且 202 视为成功", async ({ page }) => {
  const stub = await stubRunBackend(page, {
    frames: ANSWER_FRAMES,
    // 视觉已配置 → 每条 described（本轮真的参与核对）
    imageRefs: (images) =>
      images.map((_, index) => imageRefPayload({ status: "described" }, index)),
  });

  await page.goto("/");
  await attachImages(page, [
    { name: "chart-a.png", body: Buffer.from(PNG_A) },
    { name: "chart-b.png", body: Buffer.from(PNG_B) },
  ]);
  await expect(page.locator(".chat-image-chip")).toHaveCount(2);

  await page.getByRole("textbox", { name: "消息输入框" }).fill("看一下这两张图");
  await page.getByRole("button", { name: "发送", exact: true }).click();

  // ① 请求体确实带上了 images（两条 data URI）
  await expect.poll(() => stub.createImages.length, { message: "应发出一次建 run 请求" }).toBe(1);
  expect(stub.createImages[0]).toHaveLength(2);
  expect(stub.createImages[0][0].startsWith("data:image/png;base64,")).toBe(true);
  expect(stub.createImages[0][1].startsWith("data:image/png;base64,")).toBe(true);

  // 202 被当作成功：回答正常渲染（若写死 201 判定，这里会走失败分支）
  await expect(page.getByText("看到了图片。")).toBeVisible();
  await expect(page.getByRole("alert")).toHaveCount(0);

  // ② 生效时如实宣称生效（不夸大也不缩水）
  const refs = page.getByTestId("image-refs");
  await expect(refs).toBeVisible();
  await expect(page.getByTestId("image-refs-summary")).toContainText("已生效");
  await expect(page.getByTestId("image-ref-0")).toHaveAttribute("data-effective", "true");
  await expect(page.getByTestId("image-ref-label-0")).toContainText("已生效");
  await expect(page.getByTestId("image-ref-label-0")).toContainText("本轮参与核对");
  // 生效时不该出现任何「不参与核对」的说明
  await expect(refs).not.toContainText("本轮不参与核对");

  expectRunOnlyTraffic(stub);
});

test("视觉未配置（cached）：请求照常带图，界面明确说不参与核对", async ({ page }) => {
  // 缺省桩行为就是 cached（原图已缓存 + 任务已登记，但没有描述块）
  const stub = await stubRunBackend(page, { frames: ANSWER_FRAMES });

  await page.goto("/");
  await attachImages(page, [{ name: "chart-a.png", body: Buffer.from(PNG_A) }]);
  await page.getByRole("textbox", { name: "消息输入框" }).fill("这张图说明了什么？");
  await page.getByRole("button", { name: "发送", exact: true }).click();

  await expect.poll(() => stub.createImages.length).toBe(1);
  expect(stub.createImages[0]).toHaveLength(1);

  // 界面必须说清楚「已保存但不参与核对」，不得宣称生效
  await expect(page.getByTestId("image-refs-summary")).toContainText("本轮参与核对");
  await expect(page.getByTestId("image-refs-summary")).toContainText("共 1 张图片，其中 0 张本轮参与核对");

  const item = page.getByTestId("image-ref-0");
  await expect(item).toHaveAttribute("data-status", "cached");
  await expect(item).toHaveAttribute("data-effective", "false");
  await expect(page.getByTestId("image-ref-label-0")).toContainText("已保存");
  await expect(page.getByTestId("image-ref-label-0")).toContainText("本轮不参与核对");
  await expect(page.getByTestId("image-ref-notice-0")).toContainText("尚未识图");
  await expect(page.getByTestId("image-ref-notice-0")).toContainText("配置视觉模型");
  // 不能出现「已生效」这种伪造成功的措辞
  await expect(page.getByTestId("image-refs")).not.toContainText("已生效");

  expectRunOnlyTraffic(stub);
});

test("多种降级状态各自如实显示（unavailable / rejected）", async ({ page }) => {
  const stub = await stubRunBackend(page, {
    frames: ANSWER_FRAMES,
    imageRefs: [
      imageRefPayload({ status: "unavailable", reason: "未配置图片下载通道", kind: "url" }, 0),
      imageRefPayload({ status: "rejected", reason: "非法 data URI", kind: "invalid" }, 1),
    ],
  });

  await page.goto("/");
  await attachImages(page, [
    { name: "remote.png", body: Buffer.from(PNG_A) },
    { name: "broken.png", body: Buffer.from(PNG_B) },
  ]);
  await page.getByRole("textbox", { name: "消息输入框" }).fill("这两张能用吗");
  await page.getByRole("button", { name: "发送", exact: true }).click();

  await expect(page.getByTestId("image-ref-0")).toHaveAttribute("data-status", "unavailable");
  await expect(page.getByTestId("image-ref-0")).toHaveAttribute("data-effective", "false");
  await expect(page.getByTestId("image-ref-label-0")).toContainText("未取得图片");
  await expect(page.getByTestId("image-ref-notice-0")).toContainText("未配置图片下载通道");

  await expect(page.getByTestId("image-ref-1")).toHaveAttribute("data-status", "rejected");
  await expect(page.getByTestId("image-ref-label-1")).toContainText("未被接受");
  await expect(page.getByTestId("image-ref-notice-1")).toContainText("非法 data URI");

  await expect(page.getByTestId("image-refs-summary")).toContainText("其中 0 张本轮参与核对");
  expectRunOnlyTraffic(stub);
});

test("后端未返回图片结果时不静默：按「未生效」明说", async ({ page }) => {
  // imageRefs: [] 而请求确实带了图 —— 界面必须自己指出这个缺口
  const stub = await stubRunBackend(page, { frames: ANSWER_FRAMES, imageRefs: [] });

  await page.goto("/");
  await attachImages(page, [{ name: "chart-a.png", body: Buffer.from(PNG_A) }]);
  await page.getByRole("textbox", { name: "消息输入框" }).fill("图呢？");
  await page.getByRole("button", { name: "发送", exact: true }).click();

  await expect.poll(() => stub.createImages.length).toBe(1);
  expect(stub.createImages[0]).toHaveLength(1);

  const missing = page.getByTestId("image-refs-missing");
  await expect(missing).toBeVisible();
  await expect(missing).toContainText("后端未返回处理结果");
  await expect(missing).toContainText("本轮不视为已生效");
  await expect(missing).toContainText("已附带 1 张图片");
  // 不得出现 image-refs 这种「有结果」的区块
  await expect(page.getByTestId("image-refs")).toHaveCount(0);
  expectRunOnlyTraffic(stub);
});

test("无图片时不渲染图片引用区（不产生噪音）", async ({ page }) => {
  const stub = await stubRunBackend(page, { frames: ANSWER_FRAMES });

  await page.goto("/");
  await page.getByRole("textbox", { name: "消息输入框" }).fill("纯文字提问");
  await page.getByRole("button", { name: "发送", exact: true }).click();

  await expect(page.getByText("看到了图片。")).toBeVisible();
  expect(stub.createImages[0]).toEqual([]);
  await expect(page.getByTestId("image-refs")).toHaveCount(0);
  await expect(page.getByTestId("image-refs-missing")).toHaveCount(0);
  expectRunOnlyTraffic(stub);
});

test("父子任务按 run 详情的 parent_task_id 归组显示", async ({ page }) => {
  const stub = await stubRunBackend(page, {
    frames: [
      runStatus("running"),
      planUpdated([
        { task_id: "t1", question: "风控指标定义是什么？", agent_kind: "rag" },
        { task_id: "t2", question: "最新版本有什么变化？", agent_kind: "web" },
      ]),
      answerDelta("综合结论。"),
      runCompleted(),
    ],
    tasks: [
      { task_id: "t1", question: "风控指标定义是什么？", agent_kind: "rag", status: "completed" },
      {
        task_id: "t2",
        question: "最新版本有什么变化？",
        agent_kind: "web",
        status: "completed",
        parent_task_id: "t1",
      },
    ],
  });

  await page.goto("/");
  await page.getByRole("textbox", { name: "消息输入框" }).fill("父子任务测试");
  await page.getByRole("button", { name: "发送", exact: true }).click();
  await expect(page.getByText("综合结论。")).toBeVisible();
  await expandProcessArea(page);

  // 研究任务行默认可见，点开看细节
  await page.getByTestId("process-task-t1").getByRole("button").first().click();
  const detail = page.getByTestId("process-task-detail-t1");
  await expect(detail).toContainText("子问题");
  await expect(detail).toContainText("风控指标定义是什么？");
  expectRunOnlyTraffic(stub);
});
