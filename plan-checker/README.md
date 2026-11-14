# STOA 2D Plan Assistant

React/Vite frontend based on the supplied Stoa construction-plan interface. GitHub Pages serves only the built static files; it cannot run Gemini or store provider credentials.

## Frontend configuration

Set the repository Actions variable `VITE_STOA_API_BASE_URL` to the HTTPS origin of the chat API, then rerun the Pages workflow. The URL is public. Do not use a `VITE_GEMINI_API_KEY`, `VITE_NVIDIA_API_KEY`, or any other provider secret.

The frontend sends `POST /api/chat` with `{ query, plan, history, uploadedImage }`. The server should call Gemini with its server-side credential and return `{ reply }`. For uploaded drawings, the client sends the selected image as base64 with each chat turn. Configure CORS for `https://willi-cq.github.io` on the API server.

When `VITE_STOA_API_BASE_URL` is unset, the interface still builds and can be explored, but chat requests show that the API URL is not configured.

## Local frontend

```sh
bun install --frozen-lockfile
cp .env.example .env.local
bun run dev
```

Set `VITE_STOA_API_BASE_URL` in `.env.local` to the server origin. Keep Gemini and NVIDIA keys in the server's secret store, never in this frontend project.
