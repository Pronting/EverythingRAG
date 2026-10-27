import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import {
  answerText, applyAnswerEvent, emptyAnswerState, reduceAnswerEvents,
} from "../../src/run/answerState.ts";

async function readEvents(name) {
  const path = new URL(`../../../.tmp_probe/answer-stream/${name}.json`, import.meta.url);
  return JSON.parse(await readFile(path, "utf8"));
}

for (const name of ["general", "sourced", "resumed"]) {
  test(`真实事件 ${name}：逐步显示、重连去重和最终正文一致`, async () => {
    const events = await readEvents(name);
    let state = emptyAnswerState();
    let updates = 0;
    for (const event of events) {
      const previous = answerText(state);
      state = applyAnswerEvent(state, event);
      if (event.type === "answer.delta") {
        assert.equal(state.completed, false);
        if (answerText(state) !== previous) updates += 1;
      }
      assert.deepEqual(applyAnswerEvent(state, event), state);
    }
    assert.ok(updates >= 1);
    assert.equal(state.completed, true);
    const finalId = events.findLast((event) => event.type === "run.completed").payload.answer_id;
    const content = events.filter((event) => (
      event.type === "answer.delta" && event.payload.answer_id === finalId
    )).map((event) => event.payload.text).join("");
    assert.equal(answerText(state), content);
    assert.equal(state.answerId, finalId);
    const draftIndex = events.findIndex((event) => event.type === "answer.delta");
    const prefix = reduceAnswerEvents(events.slice(0, draftIndex + 1));
    const reconnected = reduceAnswerEvents(events.slice(draftIndex), prefix);
    assert.deepEqual(reconnected, state);
    assert.deepEqual(reduceAnswerEvents(events, state), state);
    if (name === "sourced") {
      assert.ok(state.citations.length > 0);
      assert.ok(answerText(state).includes("[S1]"));
    }
  });
}

test("真实取消事件保留已收到的正文且不标记完成", async () => {
  const events = await readEvents("cancelled");
  const state = reduceAnswerEvents(events);
  assert.ok(answerText(state).length > 0);
  assert.equal(state.completed, false);
  assert.equal(events.at(-1).type, "run.cancelled");
});
