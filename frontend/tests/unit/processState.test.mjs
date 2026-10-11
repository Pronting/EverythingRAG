import test from "node:test";
import assert from "node:assert/strict";

import { reduceRunEvents } from "../../src/run/processState.ts";

test("真实取消后恢复：已取消的模型步骤保持终止状态", () => {
  const events = [
    { event_seq: 7, type: "run.status", created_at: "2026-09-27T03:40:56Z", payload: {
      status: "running", phase: "model:00c9a1d92f5b4faa8bbf17206a71ed04",
      message: "正在选择原文章节",
      model_call: { outcome: "pending", ended_at: null },
    } },
    { event_seq: 8, type: "run.status", created_at: "2026-09-27T03:40:58Z", payload: {
      status: "cancelling", phase: "model:00c9a1d92f5b4faa8bbf17206a71ed04",
      message: "选择原文章节已取消，供应商未返回用量",
      model_call: { outcome: "CancelledError", ended_at: "2026-09-27T03:40:58.762023+08:00" },
    } },
    { event_seq: 9, type: "run.cancelled", created_at: "2026-09-27T03:40:58Z", payload: {
      stop_reason: "user_cancelled", resumable: true,
    } },
    { event_seq: 13, type: "run.status", created_at: "2026-09-27T03:41:04Z", payload: {
      status: "running", phase: "model:resumed", message: "正在选择原文章节",
      model_call: { outcome: "pending", ended_at: null },
    } },
  ];
  const previous = reduceRunEvents(events.slice(0, 3));
  const resumed = reduceRunEvents(events);
  const cancelledId = "phase:model:00c9a1d92f5b4faa8bbf17206a71ed04";
  assert.equal(previous.entries.find((item) => item.id === cancelledId)?.status, "cancelled");
  assert.equal(resumed.entries.find((item) => item.id === cancelledId)?.status, "cancelled");
  assert.equal(resumed.entries.find((item) => item.id === "phase:model:resumed")?.status, "running");
});
