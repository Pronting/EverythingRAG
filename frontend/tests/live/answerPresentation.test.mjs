import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { chromium, expect } from "@playwright/test";
import { createServer } from "vite";

test("真实回答完成后保留Markdown，过程默认折叠且保持用户选择", async () => {
  const events = JSON.parse(await readFile(
    new URL("../../../.tmp_probe/answer-stream/general.json", import.meta.url), "utf8",
  ));
  const draftIndex = events.findLastIndex((event) => event.type === "answer.delta");
  assert.ok(draftIndex > 0);
  const server = await createServer({
    server: { host: "127.0.0.1", port: 0 },
    optimizeDeps: {
      noDiscovery: true,
      include: ["react", "react-dom/client", "react/jsx-dev-runtime", "markdown-it", "dompurify"],
    },
  });
  let browser;
  try {
    await server.listen();
    browser = await chromium.launch({ headless: true });
    const page = await browser.newPage();
    const errors = [];
    page.on("pageerror", (error) => {
      errors.push(error.message);
      console.error(error.message);
    });
    const { port } = server.httpServer.address();
    const url = `http://127.0.0.1:${port}/tests/live/answerPresentation.html`;
    await page.goto(url);
    await page.waitForFunction(() => typeof window.renderEvents === "function");
    const render = async (records, live) => {
      await page.evaluate(({ records, live }) => window.renderEvents(records, live), { records, live });
    };
    const head = page.getByTestId("process-head");
    const running = events.slice(0, draftIndex + 1);
    await render(running, true);
    await expect(head).toHaveAttribute("aria-expanded", "false");
    await expect(page.getByTestId("process-entries")).toHaveCount(0);
    await expect(page.locator(".markdown-body h1, .markdown-body h2, .markdown-body h3").first()).toBeVisible();
    await expect(page.locator(".markdown-body li").first()).toBeVisible();
    await expect(page.locator(".markdown-body pre code").first()).toBeVisible();
    const streamedHtml = await page.locator(".markdown-body").innerHTML();
    await head.click();
    await expect(head).toHaveAttribute("aria-expanded", "true");
    await render(events, false);
    await expect(head).toContainText("处理过程");
    await expect(head).toHaveAttribute("aria-expanded", "true");
    assert.equal(await page.locator(".markdown-body").innerHTML(), streamedHtml);
    await head.click();
    await render(running, true);
    await expect(head).toContainText("处理中");
    await expect(head).toHaveAttribute("aria-expanded", "false");
    await render(events, false);
    await expect(head).toContainText("处理过程");
    await expect(head).toHaveAttribute("aria-expanded", "false");
    await page.reload();
    await page.waitForFunction(() => typeof window.renderEvents === "function");
    await render(events, false);
    await expect(head).toHaveAttribute("aria-expanded", "false");
    assert.equal(await page.locator(".markdown-body").innerHTML(), streamedHtml);
    assert.deepEqual(errors, []);
  } finally {
    if (browser) await browser.close();
    await server.close();
  }
});
