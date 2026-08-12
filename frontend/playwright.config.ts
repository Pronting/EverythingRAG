import { defineConfig } from "@playwright/test";

/**
 * Playwright E2E 配置：webServer 起 Vite dev（localhost:5173），
 * 用 page.route 拦截 /api/*，不依赖真实后端。
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  fullyParallel: true,
  use: {
    baseURL: "http://127.0.0.1:5173",
  },
  webServer: {
    command: "npm run dev",
    url: "http://127.0.0.1:5173",
    reuseExistingServer: true,
    timeout: 120_000,
  },
});
