import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e-live",
  timeout: 180_000,
  workers: 1,
  outputDir: "../.tmp_probe/live-results",
  reporter: [["list"], ["json", { outputFile: "../.tmp_probe/live-results.json" }]],
  use: {
    baseURL: "http://127.0.0.1:9999",
    viewport: { width: 1440, height: 900 },
    actionTimeout: 10_000,
    screenshot: "off",
    video: "off",
    trace: "off",
  },
});
