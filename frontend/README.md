# Product demo frontend

This is the React layout supplied by Johan on `origin/front` at `ea92de4`,
adapted for Planproof's existing question agent. `iamd-johan-dev` at `6081daf`
was also reviewed; its static interface is the older version. The React split
workspace, typography and visual language are retained. Scripted example
conversations and an alternate `/api/chat` server are not used as real results.

The agent baseline is `2bf56b05bd65cbcebe8eec7188742f0aeb0224c0`, also saved locally
as `backup/agent-before-frontend-20261114`. This integration changes no files in
`backend/`, `feedback/`, `ml/`, `geometry/` or `deterministic/`. It does not change
agent prompts, routing, permissions, provider recovery, budgets or model weights.

## Run

1. Start the existing backend using [the MVP handoff](../docs/MVP_FRONTEND_HANDOFF.md).
   Its default port is 8002. Keep provider keys in the backend's private `.env`.
2. Use Node 22.12+ (Node 24 tested), then run from `frontend/`:

   ```sh
   npm ci
   npm run build
   npm start
   ```

3. Open **http://127.0.0.1:5173**. `npm run dev` is the development alternative.
   Both modes proxy `/api` to the unchanged backend on port 8002 without an added
   short inference timeout. `PLANPROOF_API_TARGET` can override that server-side
   target. No provider secrets belong in Vite variables or browser storage.

This is a local demo server, not an internet-facing deployment. GitHub Pages
cannot host the Python agent or this proxy. The old `/planwise/` interface on
port 8002 remains intact for compatibility; the new product UI is on 5173.

## Actual request flow

- Upload original PDF/PNG/JPEG bytes as multipart `file` and 1-based `page` to
  `/api/documents`; render the canonical returned image. No browser resizing,
  filtering or re-encoding. The existing 10 MiB / 150 MP limits still apply.
- PDF page changes explicitly upload the selected page and clear previous answers.
- The user grants Gemini and optional Nemotron permissions. Ask sends only
  `{question, mode: "auto", allow_gemini, allow_nemotron}` to the existing `/ask`.
  Previous messages are display history, not injected model context.
- Agent-selected algorithms, Nemotron and reflection remain optional. There is
  no algorithms-first frontend workflow and no browser-generated model answer.
- Canonical citations are drawn in original source pixels; selecting a citation
  highlights and centers its region. Invalid/wrong-page boxes are omitted.
- Answer, abstention, outage, configuration and permission outcomes remain
  distinct. Zero is a valid answer; null is never converted into zero. Coverage,
  uncertainty and completed tool counters come from the actual response.
- One request at a time; changing uploads/pages and duplicate submission are
  disabled while working. No automatic retry of paid requests. A network error
  does not prove the server stopped; re-upload is explicit on document expiry.

## Verify

```sh
npm run lint
npm run build
npx playwright install chromium
npm test
```

Browser tests use clearly simulated API responses to verify contracts, source
coordinates, permissions, duplicate prevention, failures and mobile layout.
They do not establish model accuracy. On Windows an installed browser may be
selected with `PLANPROOF_BROWSER` pointing to `msedge.exe`.

No drawings, training datasets, credentials or private evaluation results are
included. Generated `dist`, dependency installs and test reports are ignored.
