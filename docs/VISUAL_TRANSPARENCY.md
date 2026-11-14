# Source-linked run inspection

## Compact default (2026-11-15)

Live work defaults to one short status (for example, **Calling Nemotron**) and
numbered source-view buttons. Colored dashed boxes and matching labels identify
active inspections; they are distinct from green supported and amber uncertain
object markers. Prompts, timings, raw findings and tool roles remain under **Details**.
The final answer shows two briefing paragraphs with inline citations. Further
explanation, scope and uncertainty notes share one **More information** disclosure;
there is no separate verification section. Detection previews are always visible
at the end, including all located count candidates. Completion/abstention status
remains explicit. Completed activity leaves the chat; the navigation inspector
still provides the full saved event history and the **Download task report** PDF
includes a compact activity summary. The portable replay retains its activity
controls. No extra inference is added by presentation.

The PDF pairs worker starts and returns using stage/inspection identity, retains
failed and uncertain outcomes, and groups provider connection attempts. Long
prompts/findings are explicitly shortened; full records remain in the saved task.
Source-bound citations and links, source identity, and failure/coverage information
remain included. Two-column evidence previews, compressed images and natural
pagination replace full-page crops and forced section breaks. Export does not run
models or change the source drawing.

The design branch exposes the existing agent's actual execution. It adds no
mandatory tool path, model-generated narration, accuracy score or inference call.
The old synchronous `/api/v1/documents/{id}/ask` contract remains compatible.

## Run API

- `POST /api/v1/documents/{document_id}/runs`: `{request_id: UUID, request: AgentRequest}`.
  Returns 202 after durable registration. Reusing an ID with the same source and
  request returns the existing run; a mismatched request returns 409. Six slots
  admit two executing jobs and four waiting jobs; excess requests return 429.
- `GET /api/v1/runs/{request_id}?after=0`: current status, source frame, ordered
  events after the requested sequence, last sequence and terminal agent result.
- `GET /api/v1/runs/{request_id}/evidence`: allowlisted geometry/comparisons already
  produced during this run, or `not_run`. This endpoint never invokes algorithms.
- `POST /api/v1/runs/{request_id}/report`: `{document_id, question?, filename?}`.
  Downloads a PDF from the terminal saved run and the matching registered source page.
  New runs retain the exact question; `question` is a legacy fallback only. The client
  cannot supply an answer or trace. Running tasks and source mismatches return 409;
  missing runs/pages return 404. No model or algorithm runs during export.

States are queued, running, completed, failed and interrupted. Completed means
execution returned, not that the answer is correct or even answered; preserve the
existing answer/abstention/unavailable fields. The UI polls every second while
visible and every three seconds when hidden. Sequence merging deduplicates events.
It never automatically resubmits a question after a connection failure.

## Execution and persistence

`DrawingAgent` accepts an optional fail-isolated observer. Existing tool arguments,
outputs, provider histories, budgets and routing remain unchanged. The live-briefing
update adds only final-presentation instructions and optional fields described below.
Hooks record Gemini turns, actual tool starts/results, cached calls, explicit
self-checks and transport attempts. Geometry projection reads the existing job;
the compact model-facing catalog is unchanged. Delivered evidence IDs are mapped
back to the full artifact IDs so available output is not confused with supplied
evidence. A self-check is not independent verification.

The single-process local backend uses a bounded thread executor. Atomic per-run
JSON snapshots under `PLANPROOF_AGENT_TRACE_ROOT/runs` retain ordered events and
results, including visual bundles. Registering a new run must succeed on disk
before execution starts. Later telemetry persistence failures preserve the agent
result in memory and expose a warning. On restart, queued/running records become
interrupted; they are never reexecuted. Existing algorithm workers and transport
budgets still govern execution. Do not run multiple server processes against the
same run directory; distributed ownership is outside this local demo's scope.

This version retains run files locally until explicitly removed. They can contain
drawing observations and must remain ignored/private. There is no new public
hosting or authentication layer. Main continues on 8002; start this branch's
backend on 8003 and its frontend on 5175. The frontend proxy defaults to 8003;
`PLANPROOF_API_TARGET` can override it.

## Evidence and coordinate contract

Public events contain sequence, elapsed time, stage/state, measured duration,
allowlisted findings/checks, image IDs and canonical crop/citation boxes. All nested
page indices are remapped to the selected PDF page at the host boundary. Provider
payloads, raw errors, filesystem paths and model thought/signature data are excluded
from the observation projection.

Bundles contain canonical boxes/polygons (including holes), source family, class,
recorded rule/method and role/status, plus the existing spatial comparison records.
Independent deterministic evidence, ML predictions and ML-assisted room/wall/opening
geometry remain distinct. No comparison is presented as ground truth or an accuracy
percentage. Missing candidates are not zero objects; inspected crops do not prove
complete coverage. Virtual or unknown opening classes keep their recorded labels.

The frontend checks raster hash, page index and dimensions before showing inspector
overlays. Crops are views of the unchanged canonical source image. Canvas overlays
have a bounded backing resolution (4096-pixel longest edge), retain polygon holes,
and share the source image's pan/zoom transform. Accessible lists provide selection
without requiring canvas hit-testing. Final citations use the existing SVG layer.

The Layers view includes a movable ML/rules comparison divider over the same
source image. Selecting a candidate shows an unchanged source crop, a plain-language
description of its recorded rule, and measured cues where available (for example
swing-arc support, glazing strokes or boundary support). These are measurements,
not calibrated correctness probabilities. ML-assisted boundaries have a separate
purple dashed layer and are excluded from the independent ML/rules comparison.

Browser IndexedDB retains the run ID before dispatch, received events, final result
and fetched bundles. Refresh reopens the original and observes the same active run.
Completed replay and layer filters are local/read-only. Legacy saved answers retain
their citations and explicitly report missing execution details. Browser storage is
not cross-device backup; server restart cannot recover a provider request in flight.

## Verification

- Backend: idempotency, request mismatch, bounded queue, restart interruption,
  delta sequences, canonical mapping, observer failure isolation and read-only GETs.
- Existing agent fixtures: identical answers, citations and tool/provider call counts
  with observation enabled; no changes to recovery or frozen benchmark behavior.
- Browser: live reconnect and refresh, replay, source mismatch, class/provenance
  toggles, object comparisons, Follow pause, mobile/keyboard access and local replay.
- UI fixtures are explicitly simulated and are not accuracy evidence. Real smoke
  tests use a generated synthetic sheet; private artifacts are never committed.

Local validation: 223 backend/feedback/tool checks and 25 browser checks passed,
plus lint/type checks and production build. One live synthetic question returned
the printed OFFICE A area of 24 square metres with a citation and five recorded
events in 9.766 seconds. A separate visual review used 396 already-computed shapes
and 222 comparisons from a 10200-by-6600 drawing; that review made no inference
calls and is not a new agent answer or an accuracy benchmark.

## Live briefing update

The default running view combines an in-conversation work log with source outlines
on the drawing. The former floating plan activity card has been removed. Matching
start/result events form one step. Pending steps have no fabricated
outcome; interrupted steps stay interrupted. Source regions appear without opening
the inspector. Only recorded/requested coordinates are shown; there is no inference
of model attention. Retained crops remain previous source regions rather than model attention.
Follow is an icon toggle in the bottom drawing toolbar beside Task details, with
a persistent accessible name, pressed state and on/off icon. It can be enabled
before a run; manual navigation pauses it. The toolbar wraps on narrow screens.

Raw layers now require Diagnostics -> Show raw algorithm output. They are never
automatically loaded or treated as the user's final findings. A returned detector
candidate can be mistaken; this change does not alter detector quality.

`submit_answer` accepts two optional presentation fields:

- `briefing`: up to 8 statements with `section` (answer/basis/limitation), `text`,
  and `citations` (1-based indices into the model's submitted final citation list).
- `citation_details`: `{citation, kind, label}` for each classified source; kind
  is text, detection or measurement. These are model-proposed source roles.

The host retains the validated canonical answer/citations. Invalid optional
presentation items are discarded, never promoted to evidence or used to reject a
valid core answer. Source indices are remapped after host-owned measurement citations
are prepended. Unknown references and uncited answer/basis statements are omitted.
The original explanation remains a complete fallback. No second model rewrites the
answer, and neither prose structure nor citation validation certifies correctness.

The browser preserves source numbering when filtering invalid/off-page citations.
New statements have inline source buttons; legacy explanations keep their original
paragraph and list its sources without guessing which sentence each one supports.
Only sources classified as detection have compact cards below the text and default
plan boxes. Other sources highlight on selection. Approximate area values carry an
approximation marker. Material limitations remain visible; detailed notes expand.

Validation on the combined measurement/design baseline: 300 Python checks and 28
browser checks pass. One actual synthetic OFFICE A run returned the printed24m2 in
58s, with15events, a crop and calculate_area, three cited briefing statements and
four source-role labels. No detector or Nemotron call was needed. This verifies
integration and citation remapping, not improved accuracy or latency on real plans.
Private desktop/mobile screenshots replay this saved run with no new inference.

## Compact task details and downloadable report

The bottom drawing toolbar has a small **Task details** button. It opens the latest
task's existing Steps / Diagnostics inspector; it is disabled before the first task.
Live on-plan activity and the chat work log remain automatic while the task runs.
Completed work no longer adds a second expandable recap inside the conversation.

**Download task report · PDF** replaces the previous answer-inspection link. Only the
latest terminal task offers a download. Selecting an older citation or typing a new
draft does not change which question is exported. A running latest task hides the
previous report; failed/interrupted tasks can export their own incomplete record.
Downloads have a pending state, prevent duplicate requests and surface retryable errors
without discarding the answer or resubmitting the question.

The PDF includes the exact question, answer/estimate status, briefing with clickable
source references, scope and uncertainties, a source overview, numbered citation
crops with canonical pixel bounds, recorded execution events, and the source identity.
Missing citations remain explicitly missing. It exports selected public observations,
not raw provider payloads, private reasoning, credentials, or local artifact paths.
Source previews are scaled inside the PDF; originals and inference inputs are unchanged.

Rendering uses the existing PyMuPDF dependency and its
[Story document layout](https://pymupdf.readthedocs.io/en/latest/story-class.html),
with in-memory image resources, escaped text, internal links and pagination. PDF work
shares the existing PyMuPDF lock. The backend returns the file with `no-store` and does
not save another copy. Browser downloads are user-owned files. A report requires the
saved run and a registered matching source page; reopen the original saved drawing if
the page has expired. A PDF records evidence, not a new correctness certification.

Validation: 322 Python checks, TypeScript, Ruff and production build passed. All 31
browser scenarios passed; the existing reconnect scenario exceeded its 30-second
deadline during the full concurrent run and passed in 8.6 seconds when rerun alone.
Report checks cover clickable citations, real PDF/image rendering, escaping, pagination,
source mismatch, missing/running runs, abstention, legacy question fallback, zero values,
private-field exclusion, failed downloads and no extra inference. Synthetic report
fixtures and private visual screenshots are not accuracy benchmarks.

Live export smoke check: an existing synthetic OFFICE A result produced a 104 KB,
three-page PDF with one internal citation link and two source images. Desktop, 390px
and 320px browser views were checked with no page errors or toolbar overflow. The
run was replayed, not inferred again. Preview activation waited for the user's active
task to complete and preserved its 30 recorded events.

## Proposed next evidence experience

See [Evidence experience and model-use plan](EVIDENCE_EXPERIENCE_PLAN.md) for the
proposed correlation, delivery/usage, coverage and claim-link contracts. These
follow-on contracts remain planned. The toolbar Follow toggle and the tool-card
experience described below are implemented.

## Visible tool execution

The conversation now displays distinct cards for Nemotron, the combined ML/rules
request, source crops, measurements and agent coordination. Observed families and
active-step/returned-step counts summarize the run. Batch dispatch belongs to the
agent; its worker inspections carry the Nemotron labels, avoiding double-counting
the parent as another active Nemotron worker.

Each card retains its original start sequence/time when the result arrives out
of order. Recorded start/return offsets and measured durations explain when work
happened; absent/null timings stay absent. No invented percentage or completion
estimate. Active inspections and unresolved/failed work remain in the default
list even when more recent steps arrive; older completed work can be expanded.

Nemotron cards show stable detail numbers, the supplied question, larger clickable
source previews, recorded sent resolution when available, reported findings and
uncertainty. More findings and long questions expand in place. Drawing-region
labels match the detail numbers. All active regions are rendered (the old limit
of three is removed); these outlines represent requested views, not detections.
Follow responds when the active region set changes, including out-of-order worker
completion. It remains optional and pauses on manual drawing navigation.

ML/rules cards explain learned candidates and pixel/geometry cues, including ML
dependencies. The UI displays the unique recorded delivered-ID count when known;
missing counts do not become zero. Separate ML/rule timings are not recorded, so
the UI treats them as one combined request. Cached and pending results remain
explicit. Pending history describes that particular request, not the final answer.
"Explore raw ML + rule candidates" opens the existing inspector by explicit user
action; no inference and no automatic overlays. Layer selection still controls
what appears on the drawing.

After completion, "Tools and source checks" preserves the same cards in a closed
recap with a visible attention count. Returned work uses a neutral return marker;
uncertainty, missing results and partial work retain caution labels. These displays
do not establish claim-level provenance or model comprehension.

Validation: all42browser scenarios passed, followed by5focused checks after
refining Follow and pending-history text; TypeScript/build pass. The prior60/63
event crash reproductions replay through completion/history with zero errors.
Mocked desktop/390px/320px screenshots were inspected; keyboard access and narrow
layout pass. Zero provider calls. Concurrent counting changes are separately owned.

## Running-inspection white-screen regression

Parallel inspection start events can contain `seconds: null` because their duration
is not known yet. Both the live work log and Task details previously checked only
for `undefined`, then called `toFixed()` on null. React unmounted the page while
the server kept processing. Refresh could appear to fix the live log once worker
results replaced the start events; Task details still contained the raw starts.

Both views now share a duration formatter that accepts only finite, nonnegative
numbers. Unknown timings have no displayed duration; measured zero stays `0.0s`.
The public event projection omits unknown/invalid durations from new events, while
the frontend remains compatible with existing records containing null. Neither
tool state nor findings are changed, and no question is resubmitted.

Validation: TypeScript/build and Ruff pass; 28 run-tracking/routing Python tests
and all 18 transparency browser tests pass, including six duration cases through
live work, Task details and completion. Two local saved-event replays reproduced
the crash before the fix, then rendered all 60 and 63 events through completion
and restored history without browser errors. These replays used isolated mocked
API routes and made zero model calls; they do not measure answer accuracy.
