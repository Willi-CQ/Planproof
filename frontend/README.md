# Premium product demo — design branch

This version lives on `design/premium-agent`, isolated from the main MVP at
`bf64999`. It builds on Johan's React frontend integration with a minimal,
Apple-inspired workspace using Meng To's design-first skill. See [DESIGN.md](DESIGN.md).
Users upload a drawing and talk to Planproof; no model selection is required.
There are no scripted answers or alternate `/api/chat` server.

The agent baseline is `2bf56b05bd65cbcebe8eec7188742f0aeb0224c0`, also saved locally
as `backup/agent-before-frontend-20261114`. This integration changes no files in
`backend/`, `feedback/`, `ml/`, `geometry/` or `deterministic/`. It does not change
agent prompts, routing, permission enforcement, provider recovery, budgets or model weights.

## Run

1. Start the existing backend using [the MVP handoff](../docs/MVP_FRONTEND_HANDOFF.md).
   Its default port is 8002. Keep provider keys in the backend's private `.env`.
2. Use Node 22.12+ (Node 24 tested), then run from `frontend/`:

   ```sh
   npm ci
   npm run build
   npm start
   ```

3. Open **http://127.0.0.1:5175**. `npm run dev` is the development alternative.
   Both modes proxy `/api` to the unchanged backend on port 8002 without an added
   short inference timeout. `PLANPROOF_API_TARGET` can override that server-side
   target. No provider secrets belong in Vite variables or browser storage.

This is a local demo server, not an internet-facing deployment. GitHub Pages
cannot host the Python agent or this proxy. The old `/planwise/` interface on
port 8002 remains intact. The main MVP remains on 5173; this design preview is
separate on 5175. Do not merge this branch into main until the design is accepted.

## Actual request flow

- Upload original PDF/PNG/JPEG bytes as multipart `file` and 1-based `page` to
  `/api/documents`; render the canonical returned image. No browser resizing,
  filtering or re-encoding. The existing 10 MiB / 150 MP limits still apply.
- The bottom toolbar is the only page navigator: previous/next or enter a page
  number. Successful PDF page changes clear previous answers. Failed page loads
  retain the current source and conversation; sending is disabled during loading.
- Only explicit send calls Planproof's AI services, with the unchanged payload
  `{question, mode: "auto", allow_gemini: true, allow_nemotron: true}`. Upload,
  restore, and suggested questions never call inference. Previous messages are
  display history, not model context. The composer has no footer or model controls.
- Agent-selected algorithms, Nemotron and reflection remain optional. There is
  no algorithms-first frontend workflow and no browser-generated model answer.
- Canonical citations are drawn in original source pixels; selecting a citation
  highlights and centers its region. Invalid/wrong-page boxes are omitted.
- Answer, abstention, outage, configuration and permission outcomes remain
  distinct. Zero is a valid answer; null is never converted into zero. Coverage,
  uncertainty remain accessible under “About this answer”. Internal tool counters
  and model diagnostics are not presented in the product conversation.
- One request at a time; changing uploads/pages and duplicate submission are
  disabled while working. No automatic retry of paid requests. A network error
  does not prove the server stopped; re-upload is explicit on document expiry.

## Local drawing library

The chat spans the desktop workspace from its top edge; New drawing sits in its
header. Your drawings opens a small, keyboard-accessible picker with remove controls.

Original files, the last viewed page, its conversation, draft, zoom and selected
evidence are saved automatically in browser IndexedDB (`planproof-library`, v1).
Content hashes deduplicate originals. The library contains no provider credentials.
[IndexedDB](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API) supports
structured browser records and original file blobs without a separate server database.

Refreshing or selecting a saved drawing reuploads its unchanged original to the
existing document endpoint, renewing expired backend document IDs. No question is
resent. Interrupted questions are marked; their draft is retained for explicit retry.
If the backend is unavailable, originals and conversations remain saved and can be
reopened from Your drawings later. If rendering dimensions change, retain history
but suppress mismatched source boxes. The current page's history clears on a
successful page flip, just as before; the library is not a per-page chat archive.

This is local to this browser profile and origin (use the same host and port).
Clearing site data removes it; it is not a backup or cross-device sync. Storage
failures are visible and do not prevent asking questions. Removing a saved drawing
removes its local original and conversation, not backend uploads or source files.
Home closes the active workspace without deleting saved drawings.

## Verify

```sh
npm run lint
npm run build
npx playwright install chromium
npm test
```

Browser tests use clearly simulated API responses to verify contracts, source
coordinates, explicit send, duplicate prevention, failures and mobile layout.
They do not establish model accuracy. On Windows an installed browser may be
selected with `PLANPROOF_BROWSER` pointing to `msedge.exe`.

Validation: 21 browser checks cover the UI, including page navigation failure
recovery and narrow-screen controls. TypeScript and production build passed. One real
end-to-end test used a generated, explicitly synthetic sheet: the unchanged agent
returned the printed 24 m² room area and a source citation. This proves the UI/API
flow, not real construction-plan accuracy. Private test artifacts are not included.

No drawings, training datasets, credentials or private evaluation results are
included. Generated `dist`, dependency installs and test reports are ignored.
