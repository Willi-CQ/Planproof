# Mixed 50-question agent evaluation

## Fixed protocol

This evaluation expands the eight-question Gemini development pilot to 50
different drawings from the existing 100-case Stoa reference pack. Selection reads
question/source metadata only, never answers or previous benchmark scores. It
excludes all eight prior pilot drawings by source hash, not just question ID.
Every selected case uses a distinct source drawing; all selected PDFs have one
page. The IFC source is an existing 2D PDF, not a 3D/BIM reconstruction.

| Category | Questions |
| --- | ---: |
| Room counts / labels | 14 |
| Fixture counts | 10 |
| Door and window counts | 8 |
| Areas: room, named-room totals, apartment and floor totals | 12 |
| Displayed floor layouts | 4 |
| Apartment counts | 2 |

Sources: 24 CubiCasa PNGs, 25 Swiss dwelling PDFs and one IFC-derived PDF.
Languages: 26 English, 10 Finnish, 6 German, 7 French and 1 Italian. Overall,
38 questions ask counts and 12 ask areas. Source/category quotas are fixed in
`tools/select_agent_cohort.py`; selection balances languages and exact question
templates within each stratum and breaks ties with seeded SHA256 ranking.
The seed is `planproof-agent50-v1-20261114`.

This deliberately mixed sample is not proportional to the challenge population.
It comes from a reference pack previously used for Nemotron testing, although
these drawings were excluded from the eight-case Gemini pilot. No independent
held-out detector/generalization claim follows from it. Rare strata have too few
cases for standalone accuracy claims. Source training overlap has not been audited.

## Inference and evaluator boundaries

Run the unchanged question-aware agent in `auto` and `direct` modes for each
case: 100 completed-attempt records. Both conditions receive identical question
text and canonical page pixels. Half the pairs start with automatic routing,
half with direct Gemini. Up to three cases run concurrently; the existing host
lock serializes ML/geometry preparation. Timings include any queueing and provider
variation; summed request time differs from experiment wall-clock time.

The agent decides whether to use algorithms, crops, Nemotron or reflection.
Its current crop limits and execution budgets remain unchanged from the eight-case
pilot. The proposed independent-count-before-draft policy has **not** been added
to this experiment. Runtime source hashes/snapshots and the baseline checksum
manifest are recorded before inference. The runner is frozen at source `62ecbb7`.

Reference answers are opened only after all scheduled attempts finish and the
exact case/mode Cartesian product is verified. Existing scoring remains unchanged:
exact nonnegative integer counts; compatible area units with 2% relative tolerance
(exact zero if the reference is zero). Abstentions, execution failures and wrong
answers stay in the full denominator. Citation identity/bounds are validated,
but semantic correctness of citations and claimed coverage are not scored.

Each completed attempt is persisted immediately. `--resume` validates the frozen
binding and skips completed successes **and failures**. An already-written trace
can be recovered if its record write was interrupted. An interrupted in-flight
provider request may have unreported cost and no durable response; resume does not
promise exactly-once execution for such requests. Unexpected orchestration errors
are recorded by exception type without provider secrets, and block final scoring.

## Run and inspect locally

Configure private key and worker paths as in [QUESTION_AGENT.md](QUESTION_AGENT.md).
Do not commit datasets, source drawings, labels, predictions or environment files.

```powershell
python tools/select_agent_cohort.py --manifest <pack>/manifest.json --exclude <prior-pilot>/binding.json --out .artifacts/agent50-selection-v1.json
python tools/run_agent_cohort.py --pack <pack> --selection .artifacts/agent50-selection-v1.json --out .artifacts/agent50-v1 --workers 3
# Only after an interrupted run, with identical source and configuration:
python tools/run_agent_cohort.py --pack <pack> --selection .artifacts/agent50-selection-v1.json --out .artifacts/agent50-v1 --workers 3 --resume
python tools/build_agent_report.py .artifacts/agent50-v1
```

The local run directory contains `binding.json`, implementation snapshots,
inference-only `predictions.json`, per-case traces/records/source renders, and
the final `report.json`. This session also retains `cohort-inputs.json`, a
50-question input inventory linking the original local source files without
answers. The final report includes paired wins/regressions and breakdowns by
category, source, language and question type. The viewer supports category and
outcome filters, source citations, actual Nemotron crops and recorded self-checks.

Four offline regression tests cover answer-independent reproducible selection,
prior-drawing exclusion, sparse-stratum rejection, room/fixture distinctions,
the evaluator completeness boundary, failed-attempt denominators and unchanged
area tolerance. Run `python -m pytest tools/test_agent_cohort.py -q`.

## Outcome

Completed all 100 attempts in 3,983 seconds (66.4 minutes wall time). All scores
below include failed attempts. Counts require exact matches; areas use the local
2% evaluator tolerance. These are development results, not official competition
scores or population accuracy estimates.

| Measure | Adaptive agent | Direct Gemini |
| --- | ---: | ---: |
| Correct / all questions | **33/50 (66%)** | **27/50 (54%)** |
| Returned answers | 35/50 | 30/50 |
| Execution failures | 15 | 20 |
| Incorrect returned answers | 2 | 3 |
| Median elapsed seconds | 64.164 | 63.344 |
| Maximum elapsed seconds | 566.110 | 182.719 |
| Summed request seconds | 7,125.218 | 4,475.518 |
| Reported Gemini tokens | 1,458,269 | 163,524 |
| Reported Nemotron tokens | 33,310 | 0 |

The agent gained six correct answers, with no case correct in direct mode becoming
incorrect in adaptive mode. **All six gains occurred where direct Gemini had an
execution failure.** On the 29 cases where both returned answers, both scored
27/29. This is an observed completion/robustness advantage, not evidence that the
specialist corrected a wrong direct answer. Conditional accuracy was 33/35 for
the agent and 27/30 for direct mode; these selected successful subsets do not
predict accuracy if all failures were repaired.

| Category | Agent | Direct |
| --- | ---: | ---: |
| Rooms / space labels | 13/14 | 12/14 |
| Fixtures | 8/10 | 8/10 |
| Doors / windows | 3/8 | 0/8 |
| Areas | 3/12 | 1/12 |
| Displayed floor layouts | 4/4 | 4/4 |
| Apartments | 2/2 | 2/2 |

By source: CubiCasa 18/24 versus 16/24; Swiss drawings 14/25 versus 11/25;
the single IFC-derived 2D plan 1/1 versus 0/1. Single-case results cannot establish
coverage of an entire source family.

### Failure diagnosis and evidence limits

All 15 adaptive failures stopped in the Gemini generation/response-handling path:
seven `ServerError`, two `RemoteProtocolError`, two `ReadTimeout`, one `ConnectError`,
one `ReadError`, and two `ValueError`. Server HTTP codes, exact finish reasons and
exception bodies were not retained; do not label every failure a confirmed timeout
or infer whether a truncated/unreturned answer would have been correct. There
were zero algorithm or specialist tool-validation failures. A brief connection
error burst affected both conditions; a later credential-free HTTPS probe found
the API host reachable, without diagnosing that earlier interruption.

Both actual count mismatches were shared by direct and adaptive modes. One
concerned fixtures in a partially visible neighboring unit: the answers explicitly
included that unit. This suggests a scope/annotation-boundary question requiring
review, not a demonstrated hallucinated object. The other was a window undercount;
Nemotron endorsed the proposed count. Neither reference was relabeled. One
additional direct area answer missed the 2% tolerance while the adaptive attempt
failed. Exact partner questions, labels and images remain in local artifacts.

The adaptive condition used algorithm catalogs on 19 questions (21 requests),
145 coordinate crops, two registered room crops, 22 Nemotron calls and 15 optional
self-checks. No fixed sequence was imposed. All 48 specialist crop references
covered between 0.264% and 57.148% of page area. Algorithm calls totaled 866.516s;
specialist inspections/reviews totaled 24.577s, with a maximum 3.797s per call.
The remaining adaptive elapsed time was primarily Gemini/host work, including
failed requests. Summed time includes concurrent work and queueing.

The agent used 1.59 times the summed elapsed time and 8.92 times the reported Gemini
tokens. Failed requests can have unreported usage; these are not price estimates.
The largest practical gaps are long/failing Gemini turns, dense-plan area/opening
questions, and scope-aware object inventories. Independent crop counting before
draft disclosure remains a candidate experiment, not a proven repair or moat.

Audit: exactly 50 unique cases times two modes; all 79 frozen implementation
hashes/snapshots match; identical paired rasters and model IDs; no prior-pilot
source overlap; all 226 final citation boxes in bounds. Semantic citation accuracy
and model coverage claims are unscored. 134 backend/feedback/runner tests and ten
browser checks pass, including the complete report's category/failure filters,
actual crop display and mobile layout. Raw results and audit stay under
`.artifacts/agent50-v1/`; the original eight-question experiment is unchanged.
