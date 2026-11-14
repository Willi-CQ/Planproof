import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// This branch uses its observation-enabled backend on 8003; main remains on 8002.
const proxy = {
  "/api": {
    target: process.env.PLANPROOF_API_TARGET || "http://127.0.0.1:8003",
    changeOrigin: false,
    timeout: 0,
    proxyTimeout: 0,
  },
};
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { host: "127.0.0.1", port: 5175, strictPort: true, proxy },
  preview: { host: "127.0.0.1", port: 5175, strictPort: true, proxy },
});
