import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

// 本地 Web 形态：dev 模式把 /api 代理到后端。
// 开发时可用环境变量 EVERYTHING_RAG_BACKEND_URL 指定（默认 127.0.0.1:9999）。
const backendUrl = process.env.EVERYTHING_RAG_BACKEND_URL || "http://127.0.0.1:9999";

// E2E：把 V1.2 的任务（run）端点换成可控的慢速 SSE 夹具，用真流式验证「停止生成」；
// 同时把 /api 代理指向不可达地址，任何没被 page.route stub 的请求都会立刻失败，
// 避免测试意外打到用户真实运行中的后端（读到真实会话数据、甚至写坏它）。
// 仅当显式设置 EVERYTHING_RAG_SSE_FIXTURE=slow 时生效。
//
// V1.2 迁移：旧夹具挂在 `POST /api/chat`（该入口已结构性移除，只剩 410），现在挂在
// `POST /api/workspaces/{w}/runs`（创建任务）+ `GET .../runs/{r}/events`（SSE）。
const sseFixture = process.env.EVERYTHING_RAG_SSE_FIXTURE ?? "";
const SSE_FIXTURE_DELAY_MS = Number(process.env.EVERYTHING_RAG_SSE_FIXTURE_DELAY_MS ?? "6000");
// 1 端口回环：连接必然被拒，不会连到真实后端
const e2eBackendUrl = "http://127.0.0.1:1";

const FIXTURE_RUN_ID = "fixture-run-1";

/** 夹具 run 详情（与后端 `RunDetail` 同形，含 `image_refs`）。 */
function fixtureRunDetail(imageRefs: unknown[] = []): string {
  return JSON.stringify({
    run_id: FIXTURE_RUN_ID,
    conversation_id: "fixture-conv-1",
    workspace_id: "default",
    status: "running",
    permission_mode: "ask",
    reasoning_level: "high",
    answer_id: null,
    error_code: null,
    stop_reason: null,
    resumable: false,
    context_epoch: 1,
    created_at: "2026-09-19T10:00:00Z",
    updated_at: "2026-09-19T10:00:00Z",
    events_url: `/api/workspaces/default/runs/${FIXTURE_RUN_ID}/events`,
    tasks: [],
    approvals: [],
    inputs: [],
    image_refs: imageRefs,
    last_event_seq: 0,
  });
}

/** 已吐出的前半段：answer.meta + 一段正文，之后长时间不吐字。 */
const FIXTURE_EVENTS: { type: string; payload: unknown }[] = [
  {
    type: "run.status",
    payload: { status: "running", phase: "research", message: "正在检索知识库" },
  },
  {
    type: "answer.meta",
    payload: {
      answer_id: "fixture-answer-1",
      answer_basis: "knowledge",
      basis_components: ["knowledge"],
      citations: [],
      content_hash: "",
      status: "running",
    },
  },
  {
    type: "answer.delta",
    payload: { answer_id: "fixture-answer-1", index: 0, text: "已经生成的前半" },
  },
];

function sseFixturePlugin(): Plugin {
  return {
    name: "everything-rag-run-sse-fixture",
    configureServer(server) {
      server.middlewares.use("/api/workspaces", (request, response, next) => {
        if (sseFixture !== "slow") {
          response.statusCode = 404;
          response.end("run sse fixture disabled");
          return;
        }
        // Connect 风格中间件会**剥掉挂载前缀**：request.url 只剩 `/default/runs/...`，
        // 所以这里必须把 `/api/workspaces` 拼回来再解析，否则所有路由判断都会失配。
        const raw = request.url ?? "/";
        const queryIndex = raw.indexOf("?");
        const pathOnly = queryIndex < 0 ? raw : raw.slice(0, queryIndex);
        const search = queryIndex < 0 ? "" : raw.slice(queryIndex);
        const url = new URL(`/api/workspaces${pathOnly}${search}`, "http://127.0.0.1");
        const path = url.pathname;
        const method = request.method ?? "GET";
        const notFound = (): void => {
          response.statusCode = 404;
          response.end("not found in fixture");
        };

        // 创建 run：POST /api/workspaces/{w}/runs → **202 Accepted**（PRD 13.2）
        if (method === "POST" && /^\/api\/workspaces\/[^/]+\/runs\/?$/.test(path)) {
          response.writeHead(202, { "Content-Type": "application/json" });
          response.end(fixtureRunDetail());
          return;
        }
        // 最后事件位置快照：GET /api/workspaces/{w}/runs/{r}
        if (method === "GET" && /^\/api\/workspaces\/[^/]+\/runs\/[^/]+\/?$/.test(path)) {
          response.writeHead(200, { "Content-Type": "application/json" });
          response.end(fixtureRunDetail());
          return;
        }
        // 取消：POST /api/workspaces/{w}/runs/{r}/cancel
        if (method === "POST" && /\/cancel$/.test(path)) {
          response.writeHead(200, { "Content-Type": "application/json" });
          response.end(
            JSON.stringify({ ...(JSON.parse(fixtureRunDetail()) as object), status: "cancelled" }),
          );
          return;
        }
        // SSE 订阅：GET /api/workspaces/{w}/runs/{r}/events
        if (method === "GET" && /\/events$/.test(path)) {
          response.writeHead(200, {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            Connection: "keep-alive",
          });
          const after = Number(url.searchParams.get("after") ?? "0");
          FIXTURE_EVENTS.forEach((frame, index) => {
            const seq = index + 1;
            if (seq <= after) return;
            const body = {
              schema_version: 2,
              run_id: FIXTURE_RUN_ID,
              event_seq: seq,
              execution_id: "fixture-exec",
              task_id: null,
              type: frame.type,
              created_at: "2026-09-19T10:00:00Z",
              payload: frame.payload,
            };
            response.write(`id: ${seq}\ndata: ${JSON.stringify(body)}\n\n`);
          });
          // 之后不再吐字：客户端只能靠 cancel 结束这一轮（模拟长回答生成中）。
          const timer = setTimeout(() => {
            const seq = FIXTURE_EVENTS.length + 1;
            const body = {
              schema_version: 2,
              run_id: FIXTURE_RUN_ID,
              event_seq: seq,
              execution_id: "fixture-exec",
              task_id: null,
              type: "run.completed",
              created_at: "2026-09-19T10:00:10Z",
              payload: { answer_id: "fixture-answer-1", status: "completed", content_hash: "", usage: {}, gaps: [] },
            };
            response.write(`id: ${seq}\ndata: ${JSON.stringify(body)}\n\n`);
            response.end();
          }, SSE_FIXTURE_DELAY_MS);
          // 客户端断开（订阅关闭）：Node 会在响应对象上抛出 EPIPE / ECONNRESET
          response.on("error", () => clearTimeout(timer));
          response.on("close", () => clearTimeout(timer));
          return;
        }
        void next;
        notFound();
      });
    },
  };
}

/** 仅在 E2E 关注流式中断的用例里启用；正常开发生效为零。 */
const plugins = [react(), ...(sseFixture === "slow" ? [sseFixturePlugin()] : [])];

export default defineConfig({
  plugins,
  build: { outDir: "dist" },
  server: {
    // 仅绑定本机回环地址（隐私红线：不开放远程访问；Playwright E2E 与 /api 代理均走 127.0.0.1）
    host: "127.0.0.1",
    // E2E 验证实时流：关闭压缩，避免中间层缓冲 SSE 分片
    ...(sseFixture === "slow" ? { headers: { "Cache-Control": "no-store" } } : {}),
    proxy: {
      "/api": {
        target: sseFixture === "slow" ? e2eBackendUrl : backendUrl,
        changeOrigin: true,
      },
    },
  },
});
