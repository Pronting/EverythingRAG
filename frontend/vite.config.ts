import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 本地 Web 形态：dev 模式把 /api 代理到后端。
// 后端端口动态时用环境变量 EVERYTHING_RAG_BACKEND_URL 指定（默认 127.0.0.1:8000）。
const backendUrl = process.env.EVERYTHING_RAG_BACKEND_URL || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist" },
  server: {
    proxy: {
      "/api": { target: backendUrl, changeOrigin: true },
    },
  },
});
