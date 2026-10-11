import { expect, test, type Locator, type Page } from "@playwright/test";

const STATUS_BODY = JSON.stringify({
  app: { name: "everything-rag", version: "0.1.0" },
  config: {
    wizard_completed: false,
    embed_configured: true,
    chat_configured: true,
    vision_configured: false,
    vision_enabled: false,
    search_enabled: false,
    search_configured: false,
  },
  knowledge: { file_count: 0, chunk_count: 0, image_count: 0, last_sync_at: null, needs_rebuild: false },
  privacy: { outbound_state: "local-only", providers: {} },
});

interface Row {
  id: string;
  title: string;
  workspace_id: string;
  archived?: boolean;
  archived_at?: string | null;
}

interface WorkspaceRow {
  id: string;
  name: string;
  is_default: boolean;
  pinned?: boolean;
  order_index?: number | null;
}

/**
 * 可变的工作区后端替身：新建/重命名/删除/移动/拖拽排序/置顶都会真实改变数据，
 * 断言的是「分区与顺序」这件事本身，而不是「点了按钮」。
 */
class WorkspaceStub {
  workspaces: WorkspaceRow[] = [
    { id: "default", name: "默认", is_default: true, order_index: 0 },
    { id: "research", name: "研究", is_default: false, order_index: 1 },
  ];
  conversations: Row[] = [
    { id: "c1", title: "默认区对话", workspace_id: "default" },
    { id: "c2", title: "研究区对话", workspace_id: "research" },
    { id: "c3", title: "另一条默认对话", workspace_id: "default" },
  ];
  /** 记录落库的拖拽目标，便于断言「拖到哪去了」。 */
  moves: { id: string; workspaceId: string | null }[] = [];
  /** 每次 `POST /api/conversations` 带的 `?workspace=`（新对话归属的最终判据）。 */
  createdIn: string[] = [];

  counts(): Record<string, number> {
    const result: Record<string, number> = {};
    for (const workspace of this.workspaces) result[workspace.id] = 0;
    for (const conversation of this.conversations) {
      if (conversation.archived) continue;
      if (conversation.workspace_id in result) result[conversation.workspace_id] += 1;
    }
    return result;
  }

  payload(): unknown[] {
    const counts = this.counts();
    return this.sorted().map((workspace) => ({ ...workspace, conversation_count: counts[workspace.id] ?? 0 }));
  }

  /** 后端语义：置顶在前 → 手工位次 → 创建顺序（与后端 _workspace_sort_key 一致）。 */
  sorted(): WorkspaceRow[] {
    return [...this.workspaces].sort((left, right) => {
      const pinnedDelta = (left.pinned === true ? 0 : 1) - (right.pinned === true ? 0 : 1);
      if (pinnedDelta !== 0) return pinnedDelta;
      return (left.order_index ?? Number.MAX_SAFE_INTEGER) - (right.order_index ?? Number.MAX_SAFE_INTEGER);
    });
  }

  /**
   * 某工作区里的会话顺序（后端语义：**最近活动优先**）。
   *
   * 2026-09-22（需求 2）起服务端不再有手工位次：stub 里也没有 `sort_index`，
   * 顺序回落到数组顺序（等价于「后来的排在后面」这个稳定的可见顺序）。
   */
  ordered(workspaceId: string): Row[] {
    return this.conversations.filter((row) => row.workspace_id === workspaceId && !row.archived);
  }

  /** 拖拽排序落库：按给定 id 顺序重写位次（不触碰置顶状态）。 */
  reorder(ids: string[]): void {
    ids.forEach((id, position) => {
      const found = this.workspaces.find((workspace) => workspace.id === id);
      if (found !== undefined) found.order_index = position;
    });
  }
}

async function stubBackend(page: Page, stub = new WorkspaceStub()): Promise<WorkspaceStub> {
  await page.route("**/api/status", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: STATUS_BODY }),
  );
  await page.route("**/api/settings", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "{}" }),
  );
  await page.route("**/api/workspaces**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const segments = url.pathname.split("/").filter(Boolean);
    const id = decodeURIComponent(segments[2] ?? "");
    const action = segments[3] ?? "";
    const json = (body: unknown, status = 200) =>
      route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });

    // PUT /api/workspaces/order：拖拽排序落库
    if (request.method() === "PUT" && id === "order") {
      const body = request.postDataJSON() as { ids: string[] };
      stub.reorder(body.ids);
      return json(stub.payload());
    }
    if (request.method() === "POST" && id === "") {
      const body = request.postDataJSON() as { name: string };
      const created: WorkspaceRow = { id: `ws-${stub.workspaces.length + 1}`, name: body.name, is_default: false };
      stub.workspaces.push(created);
      return json({ ...created, conversation_count: 0 }, 201);
    }
    // PUT /api/workspaces/{id}/pin：置顶
    if (request.method() === "PUT" && action === "pin") {
      const body = request.postDataJSON() as { pinned: boolean };
      const found = stub.workspaces.find((workspace) => workspace.id === id);
      if (found === undefined) return json({ detail: "工作区不存在" }, 404);
      found.pinned = body.pinned;
      return json({ ...found, conversation_count: stub.counts()[id] ?? 0 });
    }
    if (request.method() === "PUT") {
      const body = request.postDataJSON() as { name: string };
      const found = stub.workspaces.find((workspace) => workspace.id === id);
      if (found === undefined) return json({ detail: "工作区不存在" }, 404);
      found.name = body.name;
      return json({ ...found, conversation_count: stub.counts()[id] ?? 0 });
    }
    if (request.method() === "DELETE") {
      const index = stub.workspaces.findIndex((workspace) => workspace.id === id);
      if (index < 0) return json({ detail: "工作区不存在" }, 404);
      if (stub.workspaces[index].is_default) return json({ detail: "默认工作区不可删除" }, 400);
      stub.workspaces.splice(index, 1);
      let archived = 0;
      for (const conversation of stub.conversations) {
        if (conversation.workspace_id === id && !conversation.archived) {
          conversation.archived = true;
          conversation.archived_at = "2026-09-17T10:00:00Z";
          archived += 1;
        }
      }
      return json({ archived });
    }
    // GET：返回当前顺序（置顶在前 → 手工位次）
    return json(stub.payload());
  });
  // 任务端点：必须比上面的 `**/api/workspaces**` **后注册**（Playwright 按注册的逆序匹配），
  // 否则会被工作区列表路由吃掉。这里只要求「建会话发生在起 run 之前」，所以给一个最小 202。
  await page.route("**/api/workspaces/*/runs", (route) =>
    route.fulfill({
      status: 202,
      contentType: "application/json",
      body: JSON.stringify({ run_id: "run-1", events_url: "" }),
    }),
  );
  await page.route("**/api/conversations**", (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const segments = url.pathname.split("/").filter((part) => part !== "");
    const [, , first, second] = segments;
    const json = (body: unknown, status = 200) =>
      route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });

    if (first === "archived") return json(stub.conversations.filter((row) => row.archived));
    if (request.method() === "POST" && second === "move") {
      const conversationId = decodeURIComponent(first);
      const body = request.postDataJSON() as { workspace_id: string | null };
      const found = stub.conversations.find((row) => row.id === conversationId);
      if (found === undefined) return json({ detail: "会话不存在" }, 404);
      stub.moves.push({ id: conversationId, workspaceId: body.workspace_id });
      if (body.workspace_id !== null) found.workspace_id = body.workspace_id;
      return json(found);
    }
    if (request.method() === "POST" && second === "workspace") {
      const conversationId = decodeURIComponent(first);
      const body = request.postDataJSON() as { workspace_id: string };
      const found = stub.conversations.find((row) => row.id === conversationId);
      if (found === undefined) return json({ detail: "会话不存在" }, 404);
      found.workspace_id = body.workspace_id;
      return json(found);
    }
    if (request.method() === "POST" && second === "truncate") return json({ message_count: 0 });
    if (request.method() === "POST" && first === undefined) {
      const workspaceId = url.searchParams.get("workspace") ?? "default";
      stub.createdIn.push(workspaceId);
      const created: Row = {
        id: `new-${stub.conversations.length + 1}`,
        title: "新对话",
        workspace_id: workspaceId,
      };
      stub.conversations.push(created);
      return json(created, 201);
    }
    if (first === undefined) {
      const query = url.searchParams.get("q")?.trim() ?? "";
      const visible = stub.conversations.filter(
        (row) => !row.archived && (query === "" || row.title.includes(query)),
      );
      return json(visible.map((row) => ({ ...row, message_count: 1, created_at: "", updated_at: "" })));
    }
    const found = stub.conversations.find((row) => row.id === decodeURIComponent(first));
    if (found === undefined) return json({ detail: "会话不存在" }, 404);
    return json({
      ...found,
      created_at: "",
      updated_at: "",
      messages: [{ role: "user", content: `内容-${found.title}`, created_at: "" }],
    });
  });
  return stub;
}

/** 打开某个工作区文件夹的「⋯」菜单。 */
async function openFolderMenu(page: Page, name: string): Promise<void> {
  await page.getByRole("button", { name: `工作区「${name}」操作` }).click();
  await expect(page.getByRole("menu", { name: `工作区「${name}」操作` })).toBeVisible();
}

/** 用鼠标事件模拟 HTML5 拖拽（Playwright 的 dragTo 对 HTML5 DnD 支持有限）。 */
async function dragTo(page: Page, source: Locator, target: Locator): Promise<void> {
  const from = await source.boundingBox();
  const to = await target.boundingBox();
  if (from === null || to === null) throw new Error("拖拽源/目标不可见");
  await page.mouse.move(from.x + from.width / 2, from.y + from.height / 2);
  await page.mouse.down();
  await page.mouse.move(to.x + to.width / 2, to.y + to.height / 2, { steps: 8 });
  // 多停一下：并列跑多个 worker 时事件可能比 React 状态更新更快
  await page.waitForTimeout(80);
  await page.mouse.move(to.x + to.width / 2, to.y + to.height / 2, { steps: 3 });
  await page.mouse.up();
  await page.waitForTimeout(80);
}

test("侧边栏按工作区文件夹分组显示，含名称与计数", async ({ page }) => {
  await stubBackend(page);
  await page.goto("/");

  await expect(page.getByRole("button", { name: "工作区「默认」操作" })).toBeVisible();
  await expect(page.getByRole("button", { name: "工作区「研究」操作" })).toBeVisible();
  await expect(page.locator(".ws-folder").nth(0).getByText("默认区对话")).toBeVisible();
  await expect(page.locator(".ws-folder").nth(0).getByText("另一条默认对话")).toBeVisible();
  await expect(page.locator(".ws-folder").nth(1).getByText("研究区对话")).toBeVisible();
  // 计数：默认 2 条、研究 1 条
  await expect(page.locator(".ws-folder-count").nth(0)).toHaveText("2");
  await expect(page.locator(".ws-folder-count").nth(1)).toHaveText("1");
});

test("折叠状态被记住：刷新后仍是收起的样子", async ({ page }) => {
  await stubBackend(page);
  await page.goto("/");

  // 收起「研究」后刷新：折叠状态应从 localStorage 恢复
  await page.getByRole("button", { name: "收起工作区「研究」" }).click();
  // 列表是**动画收起**的（内容仍在 DOM 里，收完后隐藏），所以断言「看不见」而非「不存在」
  await expect(page.getByText("研究区对话")).not.toBeVisible();

  await page.reload();
  await expect(page.getByRole("button", { name: "工作区「研究」操作" })).toBeVisible();
  await expect(page.getByRole("button", { name: "展开工作区「研究」" })).toBeVisible();
  await expect(page.getByText("研究区对话")).not.toBeVisible();

  // 再展开也应记住
  await page.getByRole("button", { name: "展开工作区「研究」" }).click();
  await expect(page.getByText("研究区对话")).toBeVisible();
  await page.reload();
  await expect(page.getByText("研究区对话")).toBeVisible();
});

test("会话列表默认 5 条：可展开、到底后可收起、刷新与折叠都回默认", async ({ page }) => {
  const stub = await stubBackend(page);
  // 默认工作区补到 19 条：5 → 15 → 19
  for (let index = 1; index <= 17; index += 1) {
    stub.conversations.push({ id: `extra-${index}`, title: `额外对话 ${index}`, workspace_id: "default" });
  }
  await page.goto("/");

  const folder = page.locator(".ws-folder").first();
  const more = folder.getByRole("button", { name: "展开显示" });
  // 精确匹配：文件夹头的 `aria-label="收起工作区「默认」"` 也含「收起」，不精确会误命中
  const less = folder.getByRole("button", { name: "收起", exact: true });
  await expect(folder.locator(".ws-conversation")).toHaveCount(5);
  await expect(more).toHaveAttribute("title", "还有 14 条");
  await expect(less).toHaveCount(0);

  await more.click();
  await expect(folder.locator(".ws-conversation")).toHaveCount(15);
  await expect(more).toHaveAttribute("title", "还有 4 条");

  await more.click();
  await expect(folder.locator(".ws-conversation")).toHaveCount(19);
  // 列表到底：不再有「展开显示」，改为给出「收起」
  await expect(more).toHaveCount(0);
  await expect(less).toBeVisible();

  // 收起 → 回到默认 5 条（不再是「点了就一直长」）
  await less.click();
  await expect(folder.locator(".ws-conversation")).toHaveCount(5);
  await expect(more).toHaveAttribute("title", "还有 14 条");

  // 展开条数**不持久化**：刷新（等价重启）后回到默认 5 条
  await more.click();
  await expect(folder.locator(".ws-conversation")).toHaveCount(15);
  await page.reload();
  await expect(page.locator(".ws-folder").first().locator(".ws-conversation")).toHaveCount(5);

  // 折叠工作区再展开，同样回到默认 5 条（而不是折叠前的 15 条）
  const lessAfterReload = page.locator(".ws-folder").first().getByRole("button", { name: "展开显示" });
  await lessAfterReload.click();
  await expect(page.locator(".ws-folder").first().locator(".ws-conversation")).toHaveCount(15);
  await page.getByRole("button", { name: "收起工作区「默认」" }).click();
  await page.getByRole("button", { name: "展开工作区「默认」" }).click();
  await expect(page.locator(".ws-folder").first().locator(".ws-conversation")).toHaveCount(5);
});

test("文件夹可折叠，文件夹菜单可在此新建对话（落库推迟到首次提问）", async ({ page }) => {
  const stub = await stubBackend(page);
  await page.goto("/");

  await page.getByRole("button", { name: "收起工作区「研究」" }).click();
  await expect(page.getByText("研究区对话")).not.toBeVisible();
  await page.getByRole("button", { name: "展开工作区「研究」" }).click();
  await expect(page.getByText("研究区对话")).toBeVisible();

  // 「在此工作区新建对话」= 清空当前会话视图并把目标工作区记为当前工作区。
  // 会话本体推迟到第一次提问才落库（`handleEnsureConversation`），所以这里**不该**有任何
  // POST /api/conversations：这正是「重复点新对话只创建一个」的保证。
  const before = stub.conversations.length;
  await openFolderMenu(page, "研究");
  await page.getByRole("menuitem", { name: "在此工作区新建对话" }).click();
  expect(stub.conversations.length).toBe(before);
  await expect(page.getByRole("textbox", { name: "消息输入框" })).toHaveValue("");
  await expect(page.locator(".ws-folder").nth(1).getByText("新对话")).toHaveCount(0);
  // 菜单项里的点击不得冒泡到整行：文件夹必须保持展开（否则点菜单会顺手折叠）
  await expect(page.getByText("研究区对话")).toBeVisible();
});

test("重复点新对话不会堆积空会话（去重）", async ({ page }) => {
  const stub = await stubBackend(page);
  await page.goto("/");

  const before = stub.conversations.length;
  const newChat = page.getByRole("button", { name: "新对话" });
  await newChat.click();
  await newChat.click();
  await newChat.click();

  // 空会话一个都不落库：侧边栏也不会冒出「新对话」条目
  expect(stub.conversations.length).toBe(before);
  await expect(page.locator(".ws-conversation", { hasText: "新对话" })).toHaveCount(0);
});

test("点过某个工作区后，新对话记在那个工作区名下", async ({ page }) => {
  const stub = await stubBackend(page);
  await page.goto("/");

  // 点「研究」文件夹头（折叠/展开的同时把它记为当前工作区）
  await page.locator(".ws-folder").nth(1).locator(".ws-folder-name").click();
  await page.getByRole("button", { name: "新对话" }).click();
  expect(stub.createdIn).toHaveLength(0); // 仍不落库（去重）

  // 首次提问时才建会话：必须建在「研究」里（而不是默认工作区）
  await page.getByRole("textbox", { name: "消息输入框" }).fill("你好");
  await page.getByRole("button", { name: "发送" }).click();
  await expect.poll(() => stub.createdIn.at(-1) ?? null).toBe("research");
});

test("点击工作区名是折叠/展开，不会新建对话", async ({ page }) => {
  const stub = await stubBackend(page);
  await page.goto("/");

  const folderName = page.locator(".ws-folder").nth(1).locator(".ws-folder-name");
  const before = stub.conversations.length;

  await folderName.click();
  await expect(page.getByText("研究区对话")).not.toBeVisible();
  expect(stub.conversations.length).toBe(before);

  await folderName.click();
  await expect(page.getByText("研究区对话")).toBeVisible();
  expect(stub.conversations.length).toBe(before);
});

test("新建工作区、双击重命名、删除时其中对话被归档", async ({ page }) => {
  const stub = await stubBackend(page);
  await page.goto("/");

  // 新建
  await page.getByRole("button", { name: "新建工作区" }).click();
  await expect(page.getByRole("button", { name: /工作区「新工作区 3」操作/ })).toBeVisible();

  // 双击重命名「研究」→「长期项目」
  await page.locator(".ws-folder").nth(1).locator(".ws-folder-name").dblclick();
  const renameInput = page.getByRole("textbox", { name: "工作区新名称" });
  await renameInput.fill("长期项目");
  await renameInput.press("Enter");
  await expect(page.getByRole("button", { name: "工作区「长期项目」操作" })).toBeVisible();

  // 默认工作区不可删除
  await openFolderMenu(page, "默认");
  await expect(page.getByRole("menuitem", { name: "删除工作区" })).toBeDisabled();
  await page.keyboard.press("Escape");

  // 删除「长期项目」：其中对话归档
  await openFolderMenu(page, "长期项目");
  await page.getByRole("menuitem", { name: "删除工作区" }).click();
  await expect(page.getByRole("alertdialog", { name: "确认删除工作区" })).toBeVisible();
  await page.getByRole("button", { name: "确认删除" }).click();

  await expect(page.getByRole("button", { name: "工作区「长期项目」操作" })).toHaveCount(0);
  expect(stub.conversations.find((row) => row.id === "c2")?.archived).toBe(true);
  await expect(page.locator(".toast")).toContainText("已删除分区「长期项目」");
});

test("右键「移动到工作区」可把对话挪到别的文件夹", async ({ page }) => {
  const stub = await stubBackend(page);
  await page.goto("/");

  await page.getByText("默认区对话").click({ button: "right" });
  await page.getByRole("menuitem", { name: "移动到工作区" }).click();
  await page.getByRole("menuitem", { name: /^研究/ }).click();

  // 会话携带历史 ⇒ PRD 5.4 要求先确认；确认后才真正移动。
  const dialog = page.getByRole("alertdialog", { name: "移动会话到其他工作区" });
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText("停止该会话正在执行的任务");
  await expect(dialog).toContainText("历史在新的工作区里只读保留");
  // 未确认前不得发生移动
  expect(stub.moves).toHaveLength(0);
  await dialog.getByRole("button", { name: "确认移动" }).click();

  await expect(page.locator(".ws-folder").nth(1).getByText("默认区对话")).toBeVisible();
  await expect(page.locator(".toast")).toContainText("已移动到「研究」");
  expect(stub.conversations.find((row) => row.id === "c1")?.workspace_id).toBe("research");
  expect(stub.counts()).toMatchObject({ default: 1, research: 2 });
});

test("拖拽会话只能跨工作区移动，同区内拖拽不再排序", async ({ page }) => {
  const stub = await stubBackend(page);
  await page.goto("/");

  // 1) 同区拖拽：侧边栏恒为「最近活动优先」，区内位置不由拖拽决定 ⇒ 不产生任何请求
  await dragTo(page, page.getByText("另一条默认对话"), page.getByText("默认区对话"));
  await expect(page.getByRole("alertdialog", { name: "移动会话到其他工作区" })).toHaveCount(0);
  expect(stub.moves).toHaveLength(0);
  expect(stub.ordered("default").map((row) => row.id)).toEqual(["c1", "c3"]);

  // 2) 跨区拖动：归属变更 ⇒ 先确认（PRD 5.4）
  await dragTo(page, page.getByText("研究区对话"), page.locator(".ws-folder-name").first());
  const dialog = page.getByRole("alertdialog", { name: "移动会话到其他工作区" });
  await expect(dialog).toBeVisible();
  expect(stub.moves).toHaveLength(0); // 未确认前不得发生移动
  await dialog.getByRole("button", { name: "确认移动" }).click();

  await expect.poll(() => stub.moves.at(-1) ?? null).toMatchObject({ id: "c2", workspaceId: "default" });
  await expect.poll(() => stub.conversations.find((row) => row.id === "c2")?.workspace_id).toBe("default");
  await expect(page.locator(".ws-folder").nth(0).getByText("研究区对话")).toBeVisible();
});

test("工作区可置顶（别针）并排到最前", async ({ page }) => {
  const stub = await stubBackend(page);
  await page.goto("/");

  const folderNames = () => page.locator(".ws-folder-name").allTextContents();
  await expect(page.locator(".ws-folder-name")).toHaveCount(2);
  expect(await folderNames()).toEqual(["默认", "研究"]);

  // 悬浮「研究」出现别针 → 点击置顶
  await page.locator(".ws-folder").nth(1).hover();
  await page.getByRole("button", { name: "置顶工作区「研究」" }).click();

  await expect.poll(folderNames).toEqual(["研究", "默认"]);
  expect(stub.workspaces.find((workspace) => workspace.id === "research")?.pinned).toBe(true);

  // 再点一次取消置顶 → 回到原位次
  const unpin = page.getByRole("button", { name: "取消置顶工作区「研究」" });
  await expect(unpin).toHaveAttribute("aria-pressed", "true");
  await unpin.click();
  await expect.poll(folderNames).toEqual(["默认", "研究"]);
  expect(stub.workspaces.find((workspace) => workspace.id === "research")?.pinned).toBe(false);
});

test("拖拽工作区可调整顺序并落库", async ({ page }) => {
  const stub = await stubBackend(page);
  await page.goto("/");

  const folderNames = () => page.locator(".ws-folder-name").allTextContents();
  await expect(page.locator(".ws-folder-name")).toHaveCount(2);
  expect(await folderNames()).toEqual(["默认", "研究"]);

  // 拖「研究」到「默认」上方，等排序请求真正落库后再断言最终顺序
  const reorderRequest = page.waitForRequest(
    (request) => request.url().includes("/api/workspaces/order") && request.method() === "PUT",
  );
  await dragTo(page, page.locator(".ws-folder-header").nth(1), page.locator(".ws-folder-header").first());
  await reorderRequest;

  await expect.poll(folderNames).toEqual(["研究", "默认"]);
  // 落库的位次：研究(0) 在 默认(1) 之前
  await expect
    .poll(() => stub.sorted().map((workspace) => workspace.id))
    .toEqual(["research", "default"]);
  // 工作区拖拽不应被当成「把会话拖进该分区」
  expect(stub.moves).toHaveLength(0);
  expect(stub.conversations.find((row) => row.id === "c2")?.workspace_id).toBe("research");
});

test("搜索跨全部工作区，结果直接列出", async ({ page }) => {
  await stubBackend(page);
  await page.goto("/");

  await page.getByRole("textbox", { name: "搜索对话" }).fill("研究区");
  await expect(page.getByText("研究区对话")).toBeVisible();
  // 搜索结果里不再显示文件夹树
  await expect(page.locator(".ws-folder")).toHaveCount(0);

  await page.getByRole("textbox", { name: "搜索对话" }).fill("不存在的关键词");
  await expect(page.getByText("未找到匹配的对话")).toBeVisible();
});
