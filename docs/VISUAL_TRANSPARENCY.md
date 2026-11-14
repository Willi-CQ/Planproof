# Source-linked run inspection

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

The default running view now combines an in-conversation work log with a plan activity
card. Matching start/result events form one step. Pending steps have no fabricated
outcome; interrupted steps stay interrupted. Source regions appear without opening
the inspector. Only recorded/requested coordinates are shown; there is no inference
of model attention. When the latest step has no image, the card calls the retained
crop the last source region. Follow is optional, and manual navigation pauses it.

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
