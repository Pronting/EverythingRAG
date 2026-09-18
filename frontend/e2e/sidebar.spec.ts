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

/** 会话摘要（不带类型注解：Playwright 自行编译 e2e，避免跨目录 import）。 */
interface SidebarConversation {
  id: string;
  title: string;
  message_count: number;
  created_at: string;
  updated_at: string;
  workspace_id?: string;
  parent_id?: string | null;
  forked_from_title?: string | null;
  archived?: boolean;
  archived_at?: string | null;
}

/** 两个顶层会话 + 一条挂在 c1 下的分支（对齐后端 parent_id）+ 一条归档项。 */
function conversationFixtures(): SidebarConversation[] {
  return [
    { id: "c1", title: "红人系统架构", message_count: 3, created_at: "", updated_at: "2026-09-03T10:00:00Z", workspace_id: "default" },
    { id: "c2", title: "简历写法笔记", message_count: 1, created_at: "", updated_at: "2026-09-02T10:00:00Z", workspace_id: "default" },
    {
      id: "c1-fork",
      title: "红人系统架构 · 分支",
      message_count: 3,
      created_at: "",
      updated_at: "2026-09-01T10:00:00Z",
      workspace_id: "default",
      parent_id: "c1",
      forked_from_title: "红人系统架构",
    },
  ];
}

/**
 * 可变的会话后端替身：归档/恢复/fork 会真实改变列表，让流程断言有意义
 * （而不是只断言「点了按钮」）。
 */
class ConversationStub {
  readonly conversations: SidebarConversation[] = conversationFixtures();
  readonly archived: SidebarConversation[] = [];
  archivedRequests = 0;

  /** 新建一条分支（源会话必须存在）。 */
  fork(id: string): SidebarConversation {
    const parent = this.conversations.find((item) => item.id === id);
    if (parent === undefined) throw new Error(`未知会话 ${id}`);
    const branch: SidebarConversation = {
      id: `${id}-fork-${this.conversations.length}`,
      title: `${parent.title} · 分支`,
      message_count: parent.message_count,
      created_at: "",
      updated_at: new Date().toISOString(),
      workspace_id: parent.workspace_id,
      parent_id: parent.id,
      forked_from_title: parent.title,
    };
    this.conversations.push(branch);
    return branch;
  }

  archive(id: string): void {
    const index = this.conversations.findIndex((item) => item.id === id);
    if (index < 0) return;
    const [removed] = this.conversations.splice(index, 1);
    this.archived.unshift({ ...removed, archived: true, archived_at: "2026-09-04T09:30:00Z" });
  }

  restore(id: string): void {
    const index = this.archived.findIndex((item) => item.id === id);
    if (index < 0) return;
    const [removed] = this.archived.splice(index, 1);
    this.conversations.unshift({ ...removed, archived: false, archived_at: null });
  }
}

/** 拦截本地 /api 请求：会话列表由可变的 stub 后端提供，其余返回 canned JSON。 */
async function stubBackend(page: Page, stub = new ConversationStub()): Promise<ConversationStub> {
  await page.route("**/api/status", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: STATUS_BODY }),
  );
  await page.route("**/api/settings", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: "{}" }),
  );
  // 工作区文件夹树需要一个默认工作区，否则侧边栏拉不到分组
  await page.route("**/api/workspaces**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([
        {
          id: "default",
          name: "默认",
          is_default: true,
          created_at: "",
          updated_at: "",
          conversation_count: stub.conversations.length,
        },
      ]),
    }),
  );
  // 会话接口用单一 dispatcher 路由：多个重叠 glob（如 **/api/conversations/* 与
  // **/api/conversations/*/fork）在 Playwright 中会互相抢占，导致 POST /fork 落到
  // 列表分支上（返回数组、前端拿不到 id 而崩页）。按 pathname 显式分派才可靠。
  await page.route("**/api/conversations**", (route) => {
    const url = new URL(route.request().url());
    const segments = url.pathname.split("/").filter((part) => part !== "");
    // ["api", "conversations", ...rest]
    const [, , first, second] = segments;
    const json = (body: unknown, status = 200) =>
      route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });

    // GET /api/conversations/archived（必须先于 /{id} 判断，与后端路由顺序一致）
    if (first === "archived") {
      stub.archivedRequests += 1;
      return json(stub.archived);
    }
    const id = first === undefined ? "" : decodeURIComponent(first);
    if (second === "archive") {
      stub.archive(id);
      return json({ ok: true });
    }
    if (second === "restore") {
      stub.restore(id);
      return json({ ok: true });
    }
    if (second === "fork") {
      return json(stub.fork(id), 201);
    }
    if (second === "messages") {
      return json({ ok: true });
    }
    if (id === "") {
      // 列表 / 搜索（?q=）
      const query = url.searchParams.get("q")?.trim() ?? "";
      return json(
        query === "" ? stub.conversations : stub.conversations.filter((item) => item.title.includes(query)),
      );
    }
    // GET /api/conversations/{id}：完整会话（含消息）
    const found = [...stub.conversations, ...stub.archived].find((item) => item.id === id);
    if (found === undefined) return json({ detail: "会话不存在" }, 404);
    return json({
      ...found,
      messages: [
        { role: "user", content: `关于${found.title}的问题`, created_at: "" },
        { role: "assistant", content: `关于${found.title}的回答`, created_at: "" },
      ],
    });
  });
  return stub;
}

/** 打开设置弹窗并切到「归档对话」页。 */
async function openArchivePage(page: Page): Promise<void> {
  await page.getByRole("button", { name: "设置" }).click();
  await page.getByRole("button", { name: "归档对话" }).click();
}

test("空态：只保留大字标题、输入框与建议词", async ({ page }) => {
  await stubBackend(page);
  await page.goto("/");

  await expect(page.getByRole("heading", { name: "问点什么？" })).toBeVisible();
  await expect(page.getByRole("textbox", { name: "消息输入框" })).toBeVisible();
  // 已按「删除多余信息」清理：眉标 / 副标题 / 底部小字 / 顶栏英文标语 / 分组标题都不再渲染
  await expect(page.getByText("连接知识 · 发现新知")).toHaveCount(0);
  await expect(page.getByText("从散落的笔记到清晰的答案，和你的知识库聊聊。")).toHaveCount(0);
  await expect(page.getByText("随时准备好，知无不言")).toHaveCount(0);
  await expect(page.getByText("EVERYTHING, CONNECTED.")).toHaveCount(0);
  await expect(page.getByText("最近对话")).toHaveCount(0);
  // 输入框不再有「从一个好问题开始」的小字
  await expect(page.getByText("从一个好问题开始")).toHaveCount(0);
});

test("侧边栏可折叠/展开", async ({ page }) => {
  await stubBackend(page);
  await page.goto("/");

  // 初始展开：品牌标题可见
  await expect(page.getByRole("heading", { name: "Everything RAG" })).toBeVisible();
  await page.getByRole("button", { name: "收起侧边栏" }).click();
  // 折叠后：标题隐藏（仅剩图标栏）
  await expect(page.getByRole("heading", { name: "Everything RAG" })).not.toBeVisible();
  await page.getByRole("button", { name: "展开侧边栏" }).click();
  await expect(page.getByRole("heading", { name: "Everything RAG" })).toBeVisible();
});

test("侧边栏可搜索过滤会话", async ({ page }) => {
  await stubBackend(page);
  await page.goto("/");

  const list = page.locator(".conversation-list");
  await expect(list.getByText("红人系统架构", { exact: true })).toBeVisible();
  await expect(list.getByText("简历写法笔记")).toBeVisible();

  const search = page.getByRole("textbox", { name: "搜索对话" });
  await search.fill("简历");
  await expect(list.getByText("红人系统架构", { exact: true })).toHaveCount(0);
  await expect(list.getByText("简历写法笔记")).toBeVisible();

  await search.fill("不存在的关键词");
  await expect(page.getByText("未找到匹配的对话")).toBeVisible();
});

test("Fork 出的分支与源对话同列在一个工作区文件夹下", async ({ page }) => {
  const stub = await stubBackend(page);
  await page.goto("/");

  const folder = page.locator(".ws-folder").first();
  await expect(folder.getByText("红人系统架构", { exact: true })).toBeVisible();
  await expect(folder.getByText("红人系统架构 · 分支")).toBeVisible();
  expect(stub.conversations.filter((item) => item.parent_id === "c1")).toHaveLength(1);
});

test("右键菜单可 Fork 出继承全部内容的子对话并嵌在父对话下", async ({ page }) => {
  const stub = await stubBackend(page);
  await page.goto("/");

  await page.getByText("红人系统架构", { exact: true }).click({ button: "right" });
  const menu = page.getByRole("menu", { name: /红人系统架构/ });
  await expect(menu).toBeVisible();

  // 点击别处关闭菜单（不误触其它操作）
  await page.locator(".workspace-header").click();
  await expect(menu).not.toBeVisible();

  await page.getByText("红人系统架构", { exact: true }).click({ button: "right" });
  await page.getByRole("menuitem", { name: /Fork 分支对话/ }).click();

  // Fork 后立即打开新会话：新分支出现在同一工作区文件夹里（与源对话相邻）
  const list = page.locator(".conversation-list");
  await expect(list.getByText("红人系统架构 · 分支").first()).toBeVisible();
  await expect(page.getByText("关于红人系统架构 · 分支的问题")).toBeVisible();
  await expect(page.getByText("关于红人系统架构 · 分支的回答")).toBeVisible();
  expect(stub.conversations.filter((item) => item.parent_id === "c1")).toHaveLength(2);
});

test("右键归档后从侧边栏消失，可在设置页归档列表恢复", async ({ page }) => {
  const stub = await stubBackend(page);
  await page.goto("/");

  await page.getByText("简历写法笔记").click({ button: "right" });
  await page.getByRole("menuitem", { name: /归档对话/ }).click();

  // 归档后：侧边栏不再出现，搜索也搜不到
  // （注意用 .conversation-list 限定范围：提示条里也含同样的标题文字）
  await expect(page.locator(".conversation-list").getByText("简历写法笔记")).toHaveCount(0);
  await expect(page.locator(".toast")).toContainText("已归档「简历写法笔记」");
  await page.getByRole("textbox", { name: "搜索对话" }).fill("简历");
  await expect(page.getByText("未找到匹配的对话")).toBeVisible();
  await page.getByRole("textbox", { name: "搜索对话" }).fill("");

  // 设置页「归档对话」列出该会话，可一键恢复
  await openArchivePage(page);
  const archivePage = page.locator(".settings-archive");
  await expect(archivePage.getByText("简历写法笔记")).toBeVisible();
  await expect(archivePage.getByText(/1 条消息 · 2026\/09\/04/)).toBeVisible();
  await archivePage.getByRole("button", { name: "恢复对话 简历写法笔记" }).click();

  await expect(archivePage.getByText("还没有归档的对话。在侧边栏右键任意对话即可归档。")).toBeVisible();
  await page.getByRole("button", { name: "关闭设置" }).click();
  await expect(page.locator(".conversation-list").getByText("简历写法笔记")).toBeVisible();
  expect(stub.archivedRequests).toBeGreaterThan(0);
});

test("归档页删除对话需二次确认", async ({ page }) => {
  await stubBackend(page);
  await page.goto("/");

  await page.getByText("简历写法笔记").click({ button: "right" });
  await page.getByRole("menuitem", { name: /归档对话/ }).click();
  await openArchivePage(page);

  const archivePage = page.locator(".settings-archive");
  await archivePage.getByRole("button", { name: "删除对话 简历写法笔记" }).click();
  await expect(archivePage.getByText(/确定要删除「简历写法笔记」吗？/)).toBeVisible();

  // 取消不发请求
  await archivePage.getByRole("button", { name: "取消" }).click();
  await expect(archivePage.getByRole("button", { name: "删除对话 简历写法笔记" })).toBeVisible();

  // 确认后从归档列表移除
  await archivePage.getByRole("button", { name: "删除对话 简历写法笔记" }).click();
  await archivePage.getByRole("button", { name: "确认删除" }).click();
  await expect(archivePage.getByText("还没有归档的对话。在侧边栏右键任意对话即可归档。")).toBeVisible();
});

test("右键菜单删除对话需二次确认，取消不删", async ({ page }) => {
  await stubBackend(page);
  await page.goto("/");

  await page.getByText("红人系统架构", { exact: true }).click({ button: "right" });
  await page.getByRole("menuitem", { name: /删除对话/ }).click();

  const dialog = page.getByRole("alertdialog", { name: "删除对话" });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByText(/确定要删除「红人系统架构」吗？/)).toBeVisible();

  // 取消：会话仍在列表里
  await dialog.getByRole("button", { name: "取消" }).click();
  await expect(dialog).not.toBeVisible();
  await expect(page.getByText("红人系统架构", { exact: true })).toBeVisible();
});
