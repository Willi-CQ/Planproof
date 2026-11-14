import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 30_000,
  workers: 2,
  use: {
    baseURL: "http://127.0.0.1:5174",
    viewport: { width: 1440, height: 1000 },
    launchOptions: process.env.PLANPROOF_BROWSER
      ? { executablePath: process.env.PLANPROOF_BROWSER }
      : {},
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  webServer: {
    command: "npm run preview -- --port 5174",
    url: "http://127.0.0.1:5174",
    reuseExistingServer: false,
  },
});
