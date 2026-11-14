# Nemotron feedback and Stoa benchmark

The two feedback routes, local Python environment and fixed 100-question Stoa
pack are prepared. Paste the **full inference key** after `VERDA_INFERENCE_KEY=`
in `../.env`. This one ignored file serves feedback, self-feedback and benchmarking.
An existing environment variable of the same name takes precedence.

From the parent Planproof workspace in PowerShell:

```powershell
.\_planproof_nemotron\nemotron.cmd check
.\_planproof_nemotron\nemotron.cmd benchmark
```

`check` authenticates and checks `/v1/models`; the benchmark makes the actual
image-conditioned calls. Reports are saved under this worktree's
`.artifacts/nemotron/runs/<UTC timestamp>/`. No inference call was made during
preparation. Live model access/accuracy remain unverified until the key is supplied.
The `.cmd` launcher works when PowerShell script execution is disabled.

## Visual results and complete retention

Open `.artifacts/nemotron/comparison/index.html`: accuracy charts, paired outcomes,
question map, drawing/answer inspection, evidence boxes, time/token usage and
saved-run comparison. It shows not-run states until real results exist.
Every benchmark updates this report and creates a verified local backup. Exact
requests without credentials, raw responses, all source/reference data and
critique stages are retained. See [comparison and preservation](COMPARISON.md)
for the future Gemini Carbon import contract, storage layout and restore checks.

## Feedback integration

| Purpose | Python method | Later-agent tool |
| --- | --- | --- |
| Visual feedback in the Jev role | `FeedbackService.review` | `nemotron_visual_feedback` |
| Critique the later agent's own draft | `FeedbackService.critique_draft` | `nemotron_self_feedback` |

Jev was a planned crop-tool boundary, with no runtime to replace. This package
implements the replacement inference boundary and consumes the evidence engine's
actual `get_region` output. A Gemini agent is not required. No network work occurs
on import or during deterministic extraction; the host explicitly calls a route.

```python
from pathlib import Path
from planproof_feedback import create_service
from planproof_feedback.config import load_config
from planproof_feedback.regions import load_region

config = load_config(Path("feedback/config.toml"))
service = create_service(config)
images, evidence = load_region(Path(".artifacts/crops/door/region.json"), config["images"])
feedback = service.review("How many doors?", images, candidate=3, evidence=evidence)
self_feedback = service.critique_draft(
    "How many doors?", images, draft={"answer": 3, "unit": "count"}, evidence=evidence,
)
```

Both routes return advisory `supported`, `revise` or `insufficient_evidence`
verdicts, issues, optional suggested answers and citations. The adapter verifies
the clean crop hash, retains observation IDs and maps resized-image citations
back to canonical page coordinates. It leaves source evidence and drafts intact.
Citation checks establish supplied-image identity and bounds, **not semantic
truth**. Same-model self-review is correlated feedback, not independent verification.

`planproof_feedback.tools.TOOL_DEFINITIONS` exports two OpenAI-style function
schemas. The later agent's host registers known clean crop IDs and dispatches:

```python
from planproof_feedback.tools import FeedbackTools, TOOL_DEFINITIONS

tools = FeedbackTools(service, config["images"], {
    "door-detail": Path(".artifacts/crops/door/region.json"),
})
result = tools.call("nemotron_self_feedback", {
    "region_id": "door-detail", "question": "How many doors?", "draft": "Three doors.",
})
```

The model cannot supply arbitrary filesystem paths. No model-proposed external
tools execute. Provider selection for the two roles is independent in
`config.toml`; Nemotron is the implemented backend. The future agent's primary
model remains the host application's decision.

Run inside this worktree for standalone use:

```powershell
.\nemotron.cmd feedback --region .artifacts/crops/door/region.json --question "How many doors?" --candidate "Three" --out .artifacts/feedback.json
.\nemotron.cmd self-feedback --image drawing.png --question "How many doors?" --candidate "@draft.json" --out .artifacts/critique.json
.\nemotron.cmd answer --image drawing.pdf --question "How many doors?" --unit count --type count --self-review --out .artifacts/answer.json
```

The bounded `answer_with_self_feedback` workflow makes at most four calls:
draft, critique, one revision if requested, final critique. An agent that already
has its own draft can call only `critique_draft`. Invalid JSON, missing images,
out-of-frame citations and truncated answers are explicit failures.

## Random zero-shot Stoa benchmark

The ready pack is `.artifacts/nemotron/stoa-random-v1/`. It contains 100 unique
questions selected uniformly with seed `planproof-nemotron-stoa-v1`, independently
of answer values or model outputs. Eligible population: 100,938 of 101,635
reference rows with exact-path or exact-basename PNG/JPEG/PDF sources. The 697
unresolved 2D joins are excluded. There are no inferred IFC model-to-floor joins
and the final partition is never opened.

This random draw contains 85 count and 15 area questions: 61 CubiCasa5K, 38 Swiss
Dwellings and one IFC-rendered plan. Rare question types were not oversampled.
All 61 original PNGs and 39 original PDFs render successfully, as 100 single-page
inputs. Original bytes, ZIP CRCs and SHA256s are preserved, together with the
archive README and all three source license/credit notices. Payloads total
24,612,508 bytes; byte-range requests fetched 53,703,413 bytes, not the full 9.77 GB
archive. These materials remain ignored local artifacts.

The direct baseline receives only the question, requested unit/type and clean
source drawings. It gets no examples, ML/geometry observations, tools or reference
answers. Predictions are persisted before the evaluator opens
`evaluator/answers.json`. This is a zero-shot diagnostic, not the organizer's
official evaluator or random answer guessing.

Counts require exact integers. Area/length/volume use the archive README's 2%
relative tolerance, with compatible unit conversion; zero references require
exact zero. Text uses normalized exact match as a local surrogate because the
README describes free-form text without supplying its scorer. Abstentions,
malformed responses, invalid citations and case failures stay in the accuracy
denominator. Reports separate accuracy, answer coverage, source/type/mode,
failures, elapsed time and token usage. Semantic citation support needs human
review. Dollar cost is unknown; missing usage fields are explicit, and failed
requests/retries may incur unreported usage.

Compare workflows on the same fixed questions, from this worktree:

```powershell
.\nemotron.cmd benchmark --mode both --out .artifacts/nemotron/runs/comparison-v1
.\nemotron.cmd benchmark --mode both --out .artifacts/nemotron/runs/comparison-v1 --resume
```

Direct uses 100 calls; self-feedback uses up to 400 stage calls with a fresh draft
(normally 200–400 for valid drafts). Compare `metrics.by_mode`;
pooled metrics combine modes. This compares workflows, not the isolated effect
of critique on identical cached drafts. Temperature zero does not guarantee
deterministic remote responses. Transient transport failures retry at most twice;
authentication/model rejection stops the benchmark early.

Each completed case is saved atomically. Resume checks the pack/source hashes,
prompt, provider/model, rendering settings and runtime code hashes. Changes need
a new run. Already recorded case errors stay recorded. Outputs: `run.json`,
`predictions.json`, `report.json` and `scores.csv`.

Use this sample for reporting, never prompt/threshold/model selection or training.
It is not unseen final evaluation; upstream overlap and pretraining exposure are
unknown. Preserve credits when displaying drawings. Original datasets and frozen
ML models remain unchanged. A whole-page view limited to 2048 pixels can miss
detail; crop feedback is a separate task, with recorded coordinate transforms.
PDFs render at up to 144 dpi within the image limit. Pages exceeding `max_pages`
are rejected rather than silently omitted.

## Recreate and validate

Already done locally. On another Windows checkout, from this worktree:

```powershell
uv --cache-dir .artifacts/uv-cache venv .venv --python 3.12
uv --cache-dir .artifacts/uv-cache pip install --python .venv/Scripts/python.exe -e './feedback[test]'
Copy-Item .env.example .env
.\nemotron.cmd benchmark-prepare
.\.venv\Scripts\python.exe -m pytest -c feedback/pyproject.toml feedback/tests -q
.\.venv\Scripts\ruff.exe check feedback
.\.venv\Scripts\ruff.exe format --check feedback
```

Adjust private registry paths in `config.toml` if needed. A different seed/size
requires a different `pack_dir`. Preparation needs no inference key. Tests use
generated fixtures and simulated providers, exercising real PDF rendering,
transport compatibility/failures, role routing, bounded revision, coordinate
transforms, tool boundaries, answer isolation, scoring, integrity and resume.
They do not measure Nemotron accuracy. See [sources and access gates](SOURCES.md).
