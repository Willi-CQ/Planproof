# Architecture pitch graphic

Use `architecture.png` in a 16:9 deck (3840 × 2160). `architecture.svg` is the
editable vector original; `architecture.pdf` is a one-page vector export.

Suggested narration:

> Planproof turns a drawing question into an answer you can inspect. Gemini
> scopes the task and coordinates the work. Our local detectors and geometry
> provide candidates, while NVIDIA Nemotron inspects selected drawing details
> in parallel. Measurements use explicit source dimensions and local arithmetic.
> The host checks citation coverage and calculation consistency, then returns
> the answer with drawing highlights, uncertainty and a replayable PDF report.

## Source and scope

Verified against product source `6d1a240` on 15 November 2026. This is the
interactive `auto` architecture, not the tool-free comparison baseline.
Arrows between Gemini and tools represent requests and returned evidence;
they do not mean every tool runs for every question. Training is offline;
evaluation does not supply reference answers to inference.

| Graphic component | Repository evidence |
| --- | --- |
| UI, upload, source frame, runs and PDF | `frontend/`, `backend/app.py`, `backend/run_tracking.py`, `backend/task_report.py` |
| Gemini orchestration and answer submission | `backend/question_agent.py`, `docs/QUESTION_AGENT.md` |
| Required inspection, native tiles and four shared workers | `backend/specialist_routing.py`, `docs/NEMOTRON_ORCHESTRATION.md` |
| Localized counting and source-coordinate reconciliation | `backend/count_evidence.py`, `docs/LOCALIZED_COUNTING.md` |
| Optional ML, rules, hybrid rooms and cache | `backend/algorithm_jobs.py`, `backend/agent_pipeline.py`, `ml/`, `deterministic/` |
| Measurement registration and validation | `backend/measurements.py`, `docs/PLAN_MEASUREMENTS.md` |
| Training and frozen exports | `ml/TRAINING_RESULTS.md`, `models/baselines/v1/README.md` |
| Comparison and review tooling | `tools/benchmark_agent.py`, `feedback/`, `docs/AGENT_COHORT50.md`, `docs/LOCAL_VISUAL_REPLAY.md` |

Independent classical evidence and ML-assisted geometry retain separate
provenance inside the local tools box. ML/rule outputs are candidates, not
ground truth. Source-bound citation checks do not prove semantic correctness,
and a count can remain incomplete. Polygon takeoffs are explicitly approximate.
The illustration uses invented plan linework, no private drawing or credentials.
No benchmark, cost or latency improvement is claimed on the slide.

Historical worktrees are development checkouts, not deployed services. The
legacy `geometry/` reference tree and fixed door experiments are not drawn as
required stages of the current question pipeline. Package details are grouped
to keep the full flow readable in a pitch.

## Rebuild

With the frontend Playwright dependency and Chromium installed:

```powershell
node tools/export_architecture.mjs
```

The exporter blocks network access and checks that text remains inside the
slide and card padding, and that the PDF contains exactly one page. Set
`PLANPROOF_CHROMIUM_PATH` to an existing Chromium executable if using a separately
installed browser. Visually inspect the result after editing labels or positioning.
