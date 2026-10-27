import test, { after, before } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import { createServer } from "vite";

let server;
let reduceRunEvents;
before(async () => {
  server = await createServer({
    configFile: false, server: { middlewareMode: true }, appType: "custom",
    optimizeDeps: { noDiscovery: true, include: [] },
  });
  ({ reduceRunEvents } = await server.ssrLoadModule("/src/run/processState.ts"));
});
after(async () => { await server.close(); });

for (const caseId of ["Q10", "Q11"]) {
  test(`真实记录 ${caseId}：完成状态简洁且保留资料不足状态`, async () => {
    const path = new URL(
      `../../../.tmp_probe/agent-eval40-20260930/results-10001/${caseId}.json`, import.meta.url,
    );
    const record = JSON.parse(await readFile(path, "utf8"));
    const completed = record.events.find((event) => event.type === "run.completed");
    assert.ok(completed.payload.gaps.length > 0);
    const process = reduceRunEvents(record.events);
    const status = process.entries.find((entry) => entry.id === "status");
    assert.equal(status.text, "已完成，资料不足");
    assert.equal(status.tag, "completed_with_gaps");
    assert.equal(status.status, "done");
    assert.equal(status.details, undefined);
  });
}
