import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 本地 Web 形态：dev 模式把 /api 代理到后端。
// 开发时可用环境变量 EVERYTHING_RAG_BACKEND_URL 指定（默认 127.0.0.1:9999）。
const backendUrl = process.env.EVERYTHING_RAG_BACKEND_URL || "http://127.0.0.1:9999";

export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist" },
  server: {
    // 仅绑定本机回环地址（隐私红线：不开放远程访问；Playwright E2E 与 /api 代理均走 127.0.0.1）
    host: "127.0.0.1",
    proxy: {
      "/api": { target: backendUrl, changeOrigin: true },
    },
  },
});
