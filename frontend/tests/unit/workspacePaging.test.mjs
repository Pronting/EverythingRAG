import test from "node:test";
import assert from "node:assert/strict";

import {
  DEFAULT_VISIBLE,
  VISIBLE_STEP,
  expandVisible,
  pagingFooter,
  remainingCount,
  resetVisible,
  shownCount,
  visibleLimitFor,
} from "../../src/components/workspacePaging.ts";

test("默认 5 条、每次展开追加 10 条", () => {
  assert.equal(DEFAULT_VISIBLE, 5);
  assert.equal(VISIBLE_STEP, 10);
  assert.equal(visibleLimitFor({}, "ws-1"), 5);
  assert.equal(visibleLimitFor({ "ws-1": 7 }, "ws-1"), 7);
});

test("非法 / 过期值一律回到默认（负数、0、NaN、字符串）", () => {
  assert.equal(visibleLimitFor({ "ws-1": 0 }, "ws-1"), 5);
  assert.equal(visibleLimitFor({ "ws-1": -3 }, "ws-1"), 5);
  assert.equal(visibleLimitFor({ "ws-1": Number.NaN }, "ws-1"), 5);
  assert.equal(visibleLimitFor({ "ws-1": 5 }, "ws-1"), 5);
  assert.equal(visibleLimitFor({ "ws-1": "12" }, "ws-1"), 5);
});

test("展开两次 = 25 条；收起后回到默认 5 条", () => {
  let counts = {};
  counts = expandVisible(counts, "ws-1");
  assert.equal(visibleLimitFor(counts, "ws-1"), 15);
  counts = expandVisible(counts, "ws-1");
  assert.equal(visibleLimitFor(counts, "ws-1"), 25);

  counts = resetVisible(counts, "ws-1");
  assert.equal(visibleLimitFor(counts, "ws-1"), 5);
  // 收起是幂等的，且不影响别的分区
  assert.deepEqual(resetVisible(counts, "ws-1"), counts);
  const other = expandVisible(counts, "ws-2");
  assert.equal(visibleLimitFor(other, "ws-1"), 5);
  assert.equal(visibleLimitFor(other, "ws-2"), 15);
});

test("底部按钮：还有更多 → 展开；到底且展开过 → 收起；没展开过 → 不渲染", () => {
  // 20 条、默认窗口 5 → 还能展开
  assert.equal(pagingFooter(20, shownCount(20, 5, -1), 5), "expand");
  // 20 条、已展开到 25（到底）→ 收起
  assert.equal(pagingFooter(20, shownCount(20, 25, -1), 25), "collapse");
  // 7 条、展开过一次（上限 15）但列表只有 7 条 → 到底 → 收起
  assert.equal(pagingFooter(7, shownCount(7, 15, -1), 15), "collapse");
  // 4 条、从未展开 → 不渲染
  assert.equal(pagingFooter(4, shownCount(4, 5, -1), 5), "none");
  // 恰好 5 条、从未展开 → 不渲染
  assert.equal(pagingFooter(5, shownCount(5, 5, -1), 5), "none");
});

test("当前打开的会话必须可见：窗口按它扩到为止，且算作「已到底」", () => {
  // 打开的会话排在第 12 位（0 基），窗口至少到 13
  assert.equal(shownCount(20, 5, 12), 13);
  assert.equal(remainingCount(20, 13), 7);
  assert.equal(pagingFooter(20, shownCount(20, 5, 12), 5), "expand");
  // 打开的会话是最后一条 → 全部显示，但没展开过 → 仍不渲染「收起」
  assert.equal(shownCount(9, 5, 8), 9);
  assert.equal(pagingFooter(9, shownCount(9, 5, 8), 5), "none");
  // 窗口比列表长时不会越界
  assert.equal(shownCount(3, 25, -1), 3);
  assert.equal(remainingCount(3, 25), 0);
});
