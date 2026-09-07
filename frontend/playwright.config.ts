import { defineConfig } from "@playwright/test";

const e2ePort = Number(process.env.EVERYTHING_RAG_E2E_PORT ?? "41739");
const e2eBaseUrl = `http://127.0.0.1:${e2ePort}`;

/**
 * Playwright E2E 配置：webServer 在独立端口启动 Vite dev，
 * 用 page.route 拦截 /api/*，不依赖真实后端或其它已运行项目。
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  fullyParallel: true,
  use: {
    baseURL: e2eBaseUrl,
  },
  webServer: {
    command: `npm run dev -- --host 127.0.0.1 --port ${e2ePort} --strictPort`,
    url: e2eBaseUrl,
    // 不能复用任意已监听端口：否则会把另一项目的页面当成本项目并产生整组假失败。
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
