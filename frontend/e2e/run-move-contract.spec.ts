/**
 * t37 独立复核：会话移动 **API 客户端契约**（不经过 UI 交互）。
 *
 * 与实现方 `e2e/run-move.spec.ts` 的角度不同：那个文件从**界面**驱动移动并断言请求体；
 * 本文件直接在页面里 `import` 客户端模块调用两个导出函数，断言：
 * 1. 两个入口（`/move` 拖拽、`/workspace` 右键）的**请求体形状**：字段始终存在，
 *    `confirm` 缺省为 `false`（绝不静默 true）、`expected_context_epoch` 缺省为 `null`；
 *    2026-09-22（需求 2）起拖拽入口**不再发 `target_index`**（侧边栏恒为最近活动优先）；
 * 2. 显式传入 `confirm` / `expectedContextEpoch` 时如实发出；
 * 3. **两种 409 的分类**（缺确认 vs epoch 过期）——这是界面分支的依据，直接测它比只测
 *    「界面上弹了窗」更接近契约本身；
 * 4. 观测值缓存语义（`rememberConversationEpoch`/`knownConversationEpoch`/`forget...`）。
 *
 * ⚠️ 局限（与实现方一致，不得删除）：本文件用 `page.route` 造响应 ⇒ 只能证明
 * **请求形状与前端分支**，不能证明前后端契约在真实服务上可用（真实联通性仍未验证）。
 */
import { expect, test } from "@playwright/test";
import type { Page, Route } from "@playwright/test";

const MOVE_CALLS: { url: string; body: Record<string, unknown> }[] = [];

/** 拦截移动端点：记录请求体，按 `reply` 决定响应。 */
async function stubMove(page: Page, reply: (route: Route) => Promise<void>): Promise<void> {
  MOVE_CALLS.length = 0;
  await page.route("**/api/conversations/**", async (route) => {
    const request = route.request();
    if (request.method() !== "POST") return route.fallback();
    MOVE_CALLS.push({
      url: new URL(request.url()).pathname,
      body: (request.postDataJSON() ?? {}) as Record<string, unknown>,
    });
    await reply(route);
  });
}

async function importClient(page: Page): Promise<void> {
  await page.goto("/");
}

test("两入口请求体形状：confirm 缺省 false、expected_context_epoch 缺省 null", async ({
  page,
}) => {
  await importClient(page);
  await stubMove(page, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ id: "c1", workspace_id: "ws-2", context_epoch: 2 }),
    }),
  );

  const result = await page.evaluate(async () => {
    const mod = await import("/src/api/conversationApi.ts");
    const summary = { id: "c1", title: "会话", workspace_id: "ws-1", message_count: 3 };
    const viaWorkspace = await mod.moveConversationToWorkspace("c1", "ws-2");
    const viaDrag = await mod.moveConversationInSidebar("c1", {
      workspaceId: "ws-2",
    });
    return { summary, viaWorkspace, viaDrag };
  });

  expect(MOVE_CALLS).toHaveLength(2);
  const [workspaceCall, dragCall] = MOVE_CALLS;
  expect(workspaceCall.url).toBe("/api/conversations/c1/workspace");
  expect(dragCall.url).toBe("/api/conversations/c1/move");

  for (const call of MOVE_CALLS) {
    expect(call.body.workspace_id).toBe("ws-2");
    expect("confirm" in call.body, "confirm 必须始终出现在请求体里").toBe(true);
    expect(call.body.confirm, "缺省不得静默 confirm:true").toBe(false);
    expect(
      "expected_context_epoch" in call.body,
      "expected_context_epoch 必须始终出现在请求体里（未观测到则显式 null，不省略）",
    ).toBe(true);
    expect(call.body.expected_context_epoch).toBeNull();
  }
  expect(dragCall.body.target_index).toBeUndefined();
  // 响应里的权威 epoch 会被记下（供后续乐观并发使用）
  expect(result.viaWorkspace.context_epoch).toBe(2);
});

test("显式 confirm / expectedContextEpoch 如实发出", async ({ page }) => {
  await importClient(page);
  await stubMove(page, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ id: "c1", workspace_id: "ws-2", context_epoch: 3 }),
    }),
  );

  await page.evaluate(async () => {
    const mod = await import("/src/api/conversationApi.ts");
    await mod.moveConversationToWorkspace("c1", "ws-2", {
      confirm: true,
      expectedContextEpoch: 8,
    });
  });

  expect(MOVE_CALLS).toHaveLength(1);
  expect(MOVE_CALLS[0].body.confirm).toBe(true);
  expect(MOVE_CALLS[0].body.expected_context_epoch).toBe(8);
});

test("两种 409 分类 + 观测值缓存语义（界面分支的依据）", async ({ page }) => {
  await importClient(page);

  const outcome = await page.evaluate(async () => {
    const mod = await import("/src/api/conversationApi.ts");

    const classify = {
      confirm: mod.classifyMoveConflict(
        "移动会话会携带历史内容，请确认后再移动（confirm=true，PRD 5.4）",
      ),
      epoch: mod.classifyMoveConflict("会话上下文版本已变化：期望 2，实际 5（PRD 13.2）"),
      noEpochYet: mod.classifyMoveConflict(
        "该会话尚无 app.db 归属记录，无法核对 expected_context_epoch（PRD 13.2）",
      ),
    };

    // 观测值缓存：只接受 >=1 的整数；遗忘后回到 null
    mod.rememberConversationEpoch("c1", 7);
    const afterRemember = mod.knownConversationEpoch("c1");
    mod.rememberConversationEpoch("c1", 0);
    const ignoresZero = mod.knownConversationEpoch("c1");
    mod.forgetConversationEpoch("c1");
    const afterForget = mod.knownConversationEpoch("c1");

    return { classify, afterRemember, ignoresZero, afterForget };
  });

  expect(outcome.classify.confirm).toBe("confirm_required");
  expect(outcome.classify.epoch).toBe("context_epoch_conflict");
  // 「尚无归属记录」也含 context_epoch 字样 ⇒ 归为 epoch 类（不是编造值重试）
  expect(outcome.classify.noEpochYet).toBe("context_epoch_conflict");
  expect(outcome.afterRemember).toBe(7);
  expect(outcome.ignoresZero).toBe(7);
  expect(outcome.afterForget).toBeNull();
});

test("409 会抛出带分类的 MoveConflictError（不是普通 Error）", async ({ page }) => {
  await importClient(page);
  await stubMove(page, (route) =>
    route.fulfill({
      status: 409,
      contentType: "application/json",
      body: JSON.stringify({ detail: "会话上下文版本已变化：期望 2，实际 5（PRD 13.2）" }),
    }),
  );

  const captured = await page.evaluate(async () => {
    const mod = await import("/src/api/conversationApi.ts");
    try {
      await mod.moveConversationInSidebar("c1", { workspaceId: "ws-2" }, { confirm: true });
      return { threw: false } as Record<string, unknown>;
    } catch (error) {
      const typed = error as { name?: string; kind?: string };
      return { threw: true, name: typed.name, kind: typed.kind };
    }
  });

  expect(captured.threw).toBe(true);
  expect(captured.name).toBe("MoveConflictError");
  expect(captured.kind).toBe("context_epoch_conflict");
});
