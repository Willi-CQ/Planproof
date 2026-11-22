import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { cp } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { createRequire } from "node:module";

const pdfRoot = dirname(createRequire(import.meta.url).resolve("pdfjs-dist/package.json"));

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
  plugins: [react(), tailwindcss(), {
    name: "pdf-reader-assets",
    async writeBundle(options) {
      if (!options.dir) return;
      await Promise.all(["cmaps", "standard_fonts", "wasm", "iccs"].map(folder =>
        cp(resolve(pdfRoot, folder), resolve(options.dir!, "pdfjs", folder), {recursive: true})));
    },
  }],
  server: { host: "127.0.0.1", port: 5175, strictPort: true, proxy },
  preview: { host: "127.0.0.1", port: 5175, strictPort: true, proxy },
});
