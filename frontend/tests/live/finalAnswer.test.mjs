import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import { answerText, applyAnswerEvent, emptyAnswerState } from "../../src/run/answerState.ts";

for (const name of ["general", "sourced", "discussion-0", "discussion-1"]) {
  test(`真实事件：${name} 的正文持续追加，完成时保持内容`, {
    skip: process.env.EVERYTHING_RAG_LIVE_EVAL !== "1",
  }, async () => {
    const file = new URL(`../../../.tmp_probe/answer-stream/${name}.json`, import.meta.url);
    const events = JSON.parse(await readFile(file, "utf8"));
    let state = emptyAnswerState();
    let previous = "";
    const answerIds = new Set();
    for (const event of events) {
      state = applyAnswerEvent(state, event);
      const text = answerText(state);
      assert.ok(text.startsWith(previous), `事件 ${event.event_seq} 改写了已经显示的内容`);
      if (event.type === "answer.delta") {
        assert.equal(event.payload.draft, false);
        assert.equal(event.payload.replace, false);
        answerIds.add(event.payload.answer_id);
      }
      if (previous && event.payload.model_call) {
        assert.ok(event.payload.model_call.ended_at, "正文显示后启动了额外模型调用");
      }
      previous = text;
    }
    assert.equal(answerIds.size, 1);
    assert.ok(previous.length > 0);
    assert.equal(state.completed, true);
    assert.equal(state.answerId, [...answerIds][0]);
    const duplicate = events.find(event => event.type === "answer.delta");
    assert.deepEqual(applyAnswerEvent(state, duplicate), state);
  });
}
