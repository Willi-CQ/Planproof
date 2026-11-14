# Adaptive agent pilot — 14 November 2026

The eight-question development pilot scored **7/8 (87.5%) in both adaptive and
direct Gemini modes**. Both missed the same case; there were no paired wins or
regressions, no abstentions and no execution/tool failures. Routing is working,
but this experiment does not establish an accuracy benefit from tools.

## Frozen comparison

Sixteen real inference attempts used `gemini-4-carbon`: adaptive first, then a
fresh direct control for each question. The selected eight cases are all CubiCasa
count questions from the existing Stoa reference pack. They were previously
inspected during development, are not representative of the complete challenge,
and are not an independent held-out evaluation of the trained detector.

Runtime source was frozen at `c8c42e9`; six implementation hashes and copied
source snapshots still match after completion. Both conditions used identical
canonical pixels and questions. The runner opened reference answers only after
all attempts finished. No prompt, model, weights, evaluator or policy was changed
in response to these results. Conditions were not randomized; provider latency
and stochastic generation limit comparisons from this single run.

| Measure | Agent chooses tools | Direct Gemini |
| --- | ---: | ---: |
| Exact-count correct | 7/8 | 7/8 |
| Execution / tool failures | 0 / 0 | 0 / 0 |
| Total elapsed time | 448.563 s | 378.937 s |
| Median per-question time | 38.844 s | 26.742 s |
| Longest question | 127.141 s | 166.219 s |
| Reported Gemini tokens | 170,333 | 36,386 |
| Reported Nemotron tokens | 7,077 | 0 |
| Algorithm catalog requests | 1 | 0 |
| Nemotron calls | 5 | 0 |
| Recorded self-checks | 4 | 0 |

Adaptive routing took 18.4% more total time and reported about 4.7 times as many
Gemini tokens, plus Nemotron usage. Token counts are provider-reported totals,
not prices; retries/failed requests can have unreported usage. Zero execution
failures describes completed outcomes, not proof that no transport retry occurred.

## Decisions actually observed

| Pilot case / category | Adaptive route | Exact-count result |
| --- | --- | --- |
| 1 — displayed floor layouts | Direct answer | Correct |
| 2 — bathroom fixtures | Two crops → Nemotron inspection → self-check | Correct |
| 3 — bedroom labels | Direct answer | Correct |
| 4 — kitchen labels | Direct answer | Correct |
| 5 — bathroom fixtures | Three crops → Nemotron → closer crop → Nemotron → self-check | Correct |
| 6 — displayed floor layouts | One label crop → answer | Correct |
| 7 — windows | ML/geometry catalog → three crops → Nemotron draft review → self-check | Incorrect |
| 8 — bathroom fixtures | Two crops → Nemotron inspection → self-check | Correct |

All routes were model-selected; the host did not prescribe this order or force
reflection. The window case requested window/door evidence and received the
catalog successfully. Gemini chose coordinate crops rather than `crop_room` in
this pilot. Registered room cropping is covered by a separate live integration
probe, not by an observed call in this experiment.

The specialist received 11 crop references across five calls, ranging from 1.7%
to 55.9% of original page area per crop. No whole-page overview was supplied to
Nemotron. Twelve `crop_region` calls were made overall; some crops were inspected
only by Gemini or reused. The host enforces a 75% maximum area per specialist
crop. Smaller crops alone do not guarantee complete coverage or correct answers.

## Shared error and implications

The window question was undercounted by one in both conditions. Nemotron marked
Gemini's proposed count as supported, and the recorded self-check reported no
remaining uncertainty. Algorithms, model agreement and reflection therefore
failed to catch this error. The reference score remains unchanged; the report
does not relabel the case to match model consensus.

The review request exposed Gemini's count and detailed interpretation to Nemotron.
Anchoring is a plausible contributor, but this run cannot establish causation or
identify the precise missed object. An independent crop inventory before revealing
a proposed count is a sensible **next experiment**, preserving Gemini's choice
of whether to request it. It has not been implemented or scored here. Coverage
statements in self-checks are model claims, not a host-verified coverage certificate.

All 41 final citations have valid source bounds. This is coordinate validation,
not measured semantic citation accuracy. Exact-count scores do not prove that
every citation correctly identifies its claimed object. No general 87.5% accuracy
claim, detector improvement, or superiority over direct Gemini is supported.

## Local inspection and reproduction

Private artifacts are under `.artifacts/agent-pilot8-v1/`:

- `index.html`: question selector, original drawing, both answers, citation overlays,
  actual specialist crop images, feedback, optional self-checks and complete traces.
- `report.json`, `predictions.json`: scored outcomes and inference-only records.
- `binding.json`, `implementation/`: fixed cohort/source settings and snapshots.
- `audit.json`: paired-image, source-hash, crop-limit and citation-bounds checks.
- Per-case directories: source render and separate automatic/direct traces.

With private environment paths configured as in [QUESTION_AGENT.md](QUESTION_AGENT.md):

```powershell
python tools/benchmark_agent.py --pack <frozen-pack> --selection <pilot-binding.json> --count 8 --modes auto,direct --out .artifacts/<new-run>
python tools/build_agent_report.py .artifacts/<new-run>
node tools/test_question_agent_ui.mjs <local-plan.png> .artifacts/<new-run>/index.html
```

The output directory must be new. Do not overwrite this run when changing routing.
Current policy passed 130 backend/feedback tests; geometry was unchanged from its
previous 69 passing tests. Eight browser checks passed, including real local
upload with a clearly simulated UI answer and display of the actual completed
pilot/crops. Those browser checks are not additional accuracy samples. Raw partner
drawings, labels, outputs, credentials and private brain files remain local/ignored.
