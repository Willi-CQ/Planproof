# MVP startup and frontend handoff

The primary path is **question + drawing → Gemini-selected tools → grounded
answer or explicit abstention**. ML/geometry, original-pixel crops, Nemotron
inspection and reflection are optional tools. This is a working local 2D MVP;
the latest evaluation and remaining reliability limits are in
[AGENT_COHORT50.md](AGENT_COHORT50.md).

## Start from a fresh clone

Use Python 3.11 or 3.12 and Git LFS. Run these commands from the repository root:

```text
git lfs pull --include="models/baselines/v1/corrected-3500-global/**"
python tools/mvp.py setup
```

`setup` creates separate host, ML and geometry environments under the ignored
`.artifacts/venvs/` directory. Dependency installation needs internet access and
downloads PyTorch; it can take several minutes. It creates `.env` from the example
only if `.env` does not already exist. Fill these values locally:

```dotenv
GEMINI_API_KEY=<your Gemini key>
GEMINI_MODEL=gemini-4-carbon
NVIDIA_NEMOTRON_KEY=<your hosted event endpoint key>
```

The model ID must be available to your project. Keep keys on the server; never
place them in browser code, frontend environment bundles, commits or screenshots.

```text
python tools/mvp.py run
```

Open `http://127.0.0.1:8002/planwise/`. API docs: `/docs`; machine-readable contract:
`/openapi.json`. The server binds localhost. For existing environments, the explicit
launcher remains available:

```text
python backend/run_local.py --ml-python <ml-python> --geometry-python <geometry-python> --env-file <private-Nemotron-env> --gemini-env-file <private-Gemini-env> --port 8002
```

The same `.env` can hold both keys. All workers run from this checkout's source;
API credentials are removed from algorithm subprocess environments. The approved
baseline is `models/baselines/v1/corrected-3500-global`. No training datasets or
private evaluation files are required to start the MVP. Upload your own plan.
`/api/demo` is explicitly synthetic and is for UI plumbing, not an accuracy demo.

## Files for frontend work

- `plan-checker/index.html`, `app.js`, `styles.css`: currently served interface.
- `backend/app.py`: upload, page images, source frames and HTTP contracts.
- `backend/question_agent.py`: coordinator, optional tools and final-answer schema.
- `backend/agent_pipeline.py`, `visual_specialist.py`: real algorithms and Nemotron.
- [QUESTION_AGENT.md](QUESTION_AGENT.md): routing, budgets and evidence contracts.

The newer teammate design on `iamd-johan-dev` is separate UI work. Connect it to
the general `/ask` endpoint below. The legacy label check, disputed-door demo and
fixed algorithms-first `/reason` path are separate flows; they are not the general
question agent. Keep algorithm implementations on the server while changing UI.

For a separate frontend dev server, proxy `/api` to `http://127.0.0.1:8002` so the
client keeps using relative URLs. The current app uses same-origin requests and
does not enable unrestricted cross-origin access.

## Request sequence

1. `GET /api/v1/agent` reports Gemini configuration presence and model ID.
   `connection_verified: false` means it is **not** a live connectivity test.
2. `POST /api/documents` with multipart fields `file` and `page` (1-based).
   PDF, PNG and JPEG are accepted; maximum upload 10 MiB and image 16 million pixels.
   PNG/JPEG have only page 1. Only the selected PDF page is supplied to the agent.
3. Use the returned `document_id`, `image_url`, `width` and `height`. Render
   `image_url`, which is the canonical source used for coordinates.
4. `POST /api/v1/documents/{document_id}/ask` with the user's question and explicit
   provider permissions. Do not call the algorithms separately first.

```javascript
const form = new FormData();
form.append('file', fileInput.files[0]);
form.append('page', String(selectedPage));
const uploadResponse = await fetch('/api/documents', { method: 'POST', body: form });
if (!uploadResponse.ok) throw new Error(`Upload failed: ${uploadResponse.status}`);
const document = await uploadResponse.json();

const answerResponse = await fetch(`/api/v1/documents/${document.document_id}/ask`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    question,
    mode: 'auto',
    allow_gemini: geminiConsent,
    allow_nemotron: nemotronConsent
  })
});
if (!answerResponse.ok) throw new Error(`Request failed: ${answerResponse.status}`);
const result = await answerResponse.json();
```

Optional `unit` is a hint such as `count` or `m2`, not a reference answer.
Modes: `auto` (all authorized tools), `visual_only` (crops/Nemotron without
algorithms), `direct` (Gemini overview only). Permissions default false.

## Render results correctly

The response contains `status`, `answer`, `source`, `image_url`,
`canonical_citations`, `tool_trace`, `algorithm_used`, `nemotron_calls`,
`reflection_calls`, `seconds`, `stop_reason` and model/usage information when
inference ran. Configuration/permission failures can return `answer: null` and
omit inference-only fields. An HTTP 200 alone does not mean the question was solved.

| State | UI behavior |
| --- | --- |
| `status: ok`, `answer.status: answered` | Show answer/unit, concise explanation, coverage, uncertainties and source citations. |
| `answer.status: abstain` | Show insufficient evidence or execution failure; never render null as zero. |
| `status: unavailable` | Show failure and allow an explicit user retry. Keep any source context visible. |
| `status: not_configured` | Explain that server configuration is missing. Do not ask for keys inside a public chat field. |
| `status: not_authorized` | Request provider consent before another call. |
| HTTP 404 | The in-memory document expired; upload the source again. |
| HTTP 413 / 415 / 422 | Show upload size, format or validation feedback. |

The server retains eight uploaded documents in memory; uploads disappear on
restart and can be evicted by later uploads. Each `/ask` is a fresh independent
question, not a persistent chat session. Keep the original file available for
re-upload. Clear stale answers/overlays when the selected file or PDF page changes.

`canonical_citations[].canonical_bbox` is `[x0,y0,x1,y1]` in ORIGINAL rendered page
pixels, with `page_index` zero-based. Prefer an SVG overlay with
`viewBox="0 0 <source.width> <source.height>"` directly over the returned image.
Keep the image's aspect ratio and account for any CSS letterboxing. Do not draw
`answer.citations` directly over the page: those boxes can be local crop coordinates.
Backend transformations already handle them in `canonical_citations`.

The endpoint is synchronous and currently does not stream intermediate events.
Show an honest processing indicator, disable duplicate submission and avoid a
short frontend/proxy timeout. The 600-second admission budget is not a hard wall
limit: an in-flight operation can finish later. Browser cancellation does not
guarantee server-side cancellation. Do not silently retry expensive requests.
Only display tool stages as completed from the actual returned trace, not from a
timer animation. No claim of production latency or reliable dense-plan area QA.

## Verification

The existing tested environments passed 134 host/feedback/cohort tests and ten
browser checks. The bootstrap has two additional command-contract checks; its
fresh dependency downloads were not repeated in this workspace. Run host checks:

```text
.artifacts/venvs/host/Scripts/python -m pytest backend feedback/tests tools/test_agent_cohort.py tools/test_mvp.py -q
```

On macOS/Linux use `.artifacts/venvs/host/bin/python` instead. Real browser smoke
check (requires Edge on Windows, or set `PLANPROOF_BROWSER` to a compatible binary):

```text
node tools/test_question_agent_ui.mjs <local-plan.png>
```

That check uploads a real file but simulates the answer for UI testing; it does
not spend model calls or establish answer accuracy. The frozen 50-question study
used real calls and remains a separate, private artifact.
