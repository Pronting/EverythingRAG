import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

// 本地 Web 形态：dev 模式把 /api 代理到后端。
// 开发时可用环境变量 EVERYTHING_RAG_BACKEND_URL 指定（默认 127.0.0.1:9999）。
const backendUrl = process.env.EVERYTHING_RAG_BACKEND_URL || "http://127.0.0.1:9999";

// E2E：把 /api/chat 换成可控的慢速 SSE 夹具，用真流式验证「停止生成」；
// 同时把 /api 代理指向不可达地址，任何没被 page.route stub 的请求都会立刻失败，
// 避免测试意外打到用户真实运行中的后端（读到真实会话数据、甚至写坏它）。
// 仅当显式设置 EVERYTHING_RAG_SSE_FIXTURE=slow 时生效。
const sseFixture = process.env.EVERYTHING_RAG_SSE_FIXTURE ?? "";
const SSE_FIXTURE_DELAY_MS = Number(process.env.EVERYTHING_RAG_SSE_FIXTURE_DELAY_MS ?? "6000");
// 1 端口回环：连接必然被拒，不会连到真实后端
const e2eBackendUrl = "http://127.0.0.1:1";

function sseFixturePlugin(): Plugin {
  return {
    name: "everything-rag-sse-fixture",
    configureServer(server) {
      server.middlewares.use("/api/chat", (request, response) => {
        if (sseFixture !== "slow") {
          response.statusCode = 404;
          response.end("sse fixture disabled");
          return;
        }
        const frames = [
          {
            type: "meta",
            answer_basis: "knowledge",
            policy_version: "rag-policy-v3",
            query_mode: "CONTENT_QA",
            sources: [],
          },
          { type: "reasoning", text: "夹具思维链：先确认检索命中" },
          { type: "token", text: "已经生成的前半" },
        ];
        response.writeHead(200, {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        });
        for (const frame of frames) {
          response.write(`data: ${JSON.stringify(frame)}\n\n`);
        }
        // 之后不再吐字：客户端只能靠 abort 结束这一轮（模拟长回答生成中）
        const timer = setTimeout(() => {
          response.end('data: {"type":"done"}\n\n');
        }, SSE_FIXTURE_DELAY_MS);
        // 客户端断开（fetch abort）：Node 会在响应对象上抛出 EPIPE / ECONNRESET
        response.on("error", () => clearTimeout(timer));
        response.on("close", () => clearTimeout(timer));
        void request;
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
