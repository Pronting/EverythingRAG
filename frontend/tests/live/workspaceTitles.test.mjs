import test from "node:test";
import assert from "node:assert/strict";
import { chromium, request } from "@playwright/test";

const baseURL = process.env.EVERYTHING_RAG_LIVE_URL ?? "http://127.0.0.1:10017";

async function waitFor(check, message, timeout = 180000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    const value = await check();
    if (value) return value;
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(message);
}

test("真实服务：恢复执行生成标题，切换文件夹保持当前对话的工作区", { timeout: 240000 }, async () => {
  const api = await request.newContext({ baseURL });
  const workspaceResponse = await api.post("/api/workspaces", { data: { name: "标题验证" } });
  assert.equal(workspaceResponse.status(), 201);
  const workspace = await workspaceResponse.json();
  const conversationResponse = await api.post(`/api/conversations?workspace=${workspace.id}`);
  assert.equal(conversationResponse.status(), 201);
  const conversation = await conversationResponse.json();
  const runResponse = await api.post(`/api/workspaces/${workspace.id}/runs`, {
    data: {
      conversation_id: conversation.id,
      message: "只使用通用知识，用一句话解释Python的for循环。",
      reasoning_level: "off", permission_mode: "ask",
    },
  });
  assert.equal(runResponse.status(), 202);
  const run = await runResponse.json();
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const errors = [];
  const failures = [];
  const requests = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("response", (response) => {
    if (response.url().includes("/api/") && response.status() >= 400) failures.push({ url: response.url(), status: response.status() });
  });
  page.on("request", (request) => requests.push(request.url()));
  try {
    await page.addInitScript((id) => sessionStorage.setItem("everything-rag-conversation", id), conversation.id);
    await page.goto(baseURL);
    await page.locator(`.message-user[data-run-id="${run.run_id}"]`).waitFor({ timeout: 30000 });
    const saved = await waitFor(async () => {
      const response = await api.get(`/api/conversations/${conversation.id}`);
      assert.equal(response.status(), 200);
      const detail = await response.json();
      return detail.title !== "新对话" && detail.messages.length === 2 ? detail : null;
    }, "恢复执行完成后没有生成对话标题");
    assert.ok(saved.title.length > 0 && saved.title.length <= 60);
    assert.match(saved.title, /Python|循环|for/i);
    await page.locator(`.ws-conversation-main[title="${saved.title}"]`).waitFor({ timeout: 10000 });
    await page.locator(`.message-assistant[data-run-id="${run.run_id}"] .process-area`).waitFor({ timeout: 10000 });
    const before = requests.length;
    await page.locator(".ws-folder-header").filter({ hasText: "默认" }).click();
    await page.waitForTimeout(1000);
    assert.equal(await page.getByText("历史过程区域加载失败，可刷新重试", { exact: true }).count(), 0);
    assert.equal(await page.getByText("读取失败，点击重试", { exact: true }).count(), 0);
    assert.equal(requests.slice(before).filter((url) => url.includes(`/api/workspaces/default/runs/${run.run_id}`)).length, 0);
    const followupResponse = page.waitForResponse((response) => response.request().method() === "POST"
      && new URL(response.url()).pathname.endsWith("/runs"), { timeout: 30000 });
    await page.locator(".chat-input:visible").fill("只使用通用知识，用一句话解释range函数。");
    await page.locator(".chat-input:visible").press("Enter");
    const started = await followupResponse;
    assert.equal(started.status(), 202);
    assert.ok(new URL(started.url()).pathname.startsWith(`/api/workspaces/${workspace.id}/`));
    await waitFor(async () => {
      const response = await api.get(`/api/conversations/${conversation.id}`);
      const detail = await response.json();
      return detail.messages.length === 4;
    }, "切换文件夹后的追问没有完成");
    const manualTitle = "Python循环学习";
    assert.equal((await api.put(`/api/conversations/${conversation.id}/title`, { data: { title: manualTitle } })).status(), 200);
    const preserved = await api.post(`/api/conversations/${conversation.id}/generate-title?if_untitled=true`);
    assert.equal(preserved.status(), 200);
    assert.equal((await preserved.json()).title, manualTitle);
    const detail = await (await api.get(`/api/conversations/${conversation.id}`)).json();
    assert.equal(detail.title, manualTitle);
    assert.deepEqual(errors, []);
    assert.deepEqual(failures, []);
    console.log(JSON.stringify({ title: saved.title, messages: detail.messages.length, run: run.run_id, workspace: workspace.id }));
  } finally {
    await browser.close();
    await api.delete(`/api/conversations/${conversation.id}`);
    await api.delete(`/api/workspaces/${workspace.id}`);
    await api.dispose();
  }
});
