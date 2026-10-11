import test from "node:test";
import assert from "node:assert/strict";

import {
  activitySeconds,
  formatActivityDuration,
  elapsedSeconds,
  formatDuration,
  formatElapsed,
  parseEventTimes,
} from "../../src/run/elapsed.ts";
import { reduceRunEvents } from "../../src/run/processState.ts";

const T = (second) => `2026-09-20T09:00:${String(second).padStart(2, "0")}Z`;

test("解析时间戳：解析不出来的值被忽略，不用 0 兜底", () => {
  assert.deepEqual(parseEventTimes([T(0), "", "not-a-time", T(5)]), [
    Date.parse(T(0)),
    Date.parse(T(5)),
  ]);
  assert.deepEqual(parseEventTimes([]), []);
});

test("已结束的 run：首末事件时间差（成对时间戳）", () => {
  assert.equal(elapsedSeconds(parseEventTimes([T(0), T(5)])), 5);
  assert.equal(formatDuration([T(0), T(5)]), "5秒");
  assert.equal(formatDuration([T(0), T(5), T(3)]), "5秒"); // 乱序也无所谓
});

test("时间戳不足时**不显示**，而不是显示「0 秒」", () => {
  assert.equal(elapsedSeconds([]), null);
  assert.equal(elapsedSeconds(parseEventTimes([T(3)])), null);
  assert.equal(formatDuration([""]), null);
  assert.equal(formatDuration([T(3)]), null); // 只有一条且 run 已结束 ⇒ 无从算跨度
});

test("运行中给「现在」当右端点 ⇒ 每秒都在走（静默期不再冻住）", () => {
  const times = parseEventTimes([T(0), T(2)]);
  const now = Date.parse(T(30));
  assert.equal(elapsedSeconds(times, now), 30);
  assert.equal(formatDuration([T(0), T(2)], now), "30秒");
  assert.equal(formatDuration([T(0), T(2)], Date.parse(T(31))), "31秒");
  // 只有一条时间戳 + 活的右端点 ⇒ 也能算（run 刚开始，只有一个事件）
  assert.equal(formatDuration([T(0)], Date.parse(T(7))), "7秒");
});

test("活的右端点不会把数字算小（时钟回拨 / 事件时间在未来）", () => {
  const times = parseEventTimes([T(0), T(40)]);
  assert.equal(elapsedSeconds(times, Date.parse(T(10))), 40);
});

test("文案分档：秒 / 分钟 / 小时", () => {
  assert.equal(formatElapsed(0), "0秒");
  assert.equal(formatElapsed(59), "59秒");
  assert.equal(formatElapsed(60), "1分钟");
  assert.equal(formatElapsed(125), "2分钟5秒");
  assert.equal(formatElapsed(1336), "22分钟16秒"); // 用户截图里的那个数
  assert.equal(formatElapsed(3600), "1小时0分钟");
  assert.equal(formatElapsed(7900), "2小时11分钟");
});

test("真实检索超时后跨天恢复：用时只累计两次执行", () => {
  // run-c1e4da11e064450784fdf47206555114 的事件序号与时间。
  const question = "帮我查查货拉拉记忆系统怎样处理重复信息，提供资料来源。";
  const route = {
    decision: {
      question, resolved_question: question, intent: "research",
      needs_knowledge: true, needs_web: false, needs_history: false, needs_general: false,
    },
  };
  const events = [
    { event_seq: 1, type: "route.decided", created_at: "2026-09-24T05:22:57Z", payload: route },
    { event_seq: 2, type: "run.status", created_at: "2026-09-24T05:22:57Z", payload: {
      status: "running", phase: "plan", message: "联网策略：思考等级=low、允许联网=True；联网额度：查询 0/2、网页 0/3、用时 0/20s",
    } },
    { event_seq: 3, type: "plan.updated", created_at: "2026-09-24T05:22:57Z", payload: {
      tasks: [{ task_id: "t-q1-knowledge", question, agent_kind: "rag", status: "queued", depends_on: [] }],
    } },
    { event_seq: 5, type: "tool.started", created_at: "2026-09-24T05:22:57Z", payload: {
      tool_name: "search_knowledge", operation_id: "search_knowledge:t-q1-knowledge:1:ecec47b76c044862902d2b5644c82f4d",
      task_id: "t-q1-knowledge", summary: "搜索资料：帮我查查货拉拉记忆系统怎样处理重复信息，提供资料来源",
    } },
    { event_seq: 6, type: "tool.completed", created_at: "2026-09-24T05:23:28Z", payload: {
      tool_name: "search_knowledge", operation_id: "search_knowledge:t-q1-knowledge:1:ecec47b76c044862902d2b5644c82f4d",
      status: "failed", result_count: 0, source_count: 0, error_code: "service_error",
    } },
    { event_seq: 7, type: "run.failed", created_at: "2026-09-24T05:23:28Z", payload: {
      error: { error_code: "service_error", message: "查询嵌入超时，可继续执行以重新请求", resumable: true, required_action: "" },
    } },
    { event_seq: 8, type: "route.decided", created_at: "2026-09-27T03:12:14Z", payload: route },
    { event_seq: 312, type: "run.completed", created_at: "2026-09-27T03:13:15Z", payload: {
      answer_id: "ans-ffab6d836d3ad6fd709000db", status: "completed", gaps: [],
    } },
  ];
  const process = reduceRunEvents(events);
  assert.deepEqual(process.activity, [
    { startedAt: "2026-09-24T05:22:57Z", endedAt: "2026-09-24T05:23:28Z" },
    { startedAt: "2026-09-27T03:12:14Z", endedAt: "2026-09-27T03:13:15Z" },
  ]);
  assert.equal(activitySeconds(process.activity), 92);
  assert.equal(formatActivityDuration(process.activity), "1分钟32秒");
  assert.equal(formatActivityDuration(reduceRunEvents(events.slice(0, -1)).activity, Date.parse("2026-09-27T03:12:44Z")), "1分钟1秒");
  const failed = reduceRunEvents(events.slice(0, -2));
  assert.equal(failed.entries.find((entry) => entry.id === "phase:plan")?.status, "done");
  assert.equal(failed.entries.find((entry) => entry.id.startsWith("tool:search_knowledge"))?.status, "failed");
});
