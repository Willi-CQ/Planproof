# Direct versus algorithm-guided Nemotron

The local comparison uses exactly the eight question IDs from the saved
`guided-pilot-v2/binding.json`. These are eight selected CubiCasa count questions
from Stoa's original archive, within our frozen 100-question reference pack.
They are not an official eight-question Stoa preset or an unseen test set.

The viewer puts the original drawing beside both answers and exposes the actual
algorithm catalog, routing decision, selected clean crops, citation overlays,
timings and complete saved provider records. Failed or invalid outputs remain
visible and count against the score. Earlier benchmark outputs are preserved.

## Conditions

- Direct: clean source image and question, with no algorithms or tools.
- Guided: frozen ML, deterministic geometry and room proposals, Nemotron routing,
  up to three registered room crops, then an evidence-conditioned Nemotron answer.
- Same model: `nemotron-3.5-super-vl` at the supplied hosted Verda endpoint.
- Matched answer settings: temperature 0, max_tokens 4096, thinking disabled.
  Routing uses the same model and temperature, with max_tokens 1024.
- Matching overview image hashes are asserted. The direct and guided answering
  stages use the same answer prompt; guided gets additional evidence/context.
- Labels never enter model inputs. Scoring follows completion of all attempts.
  Counts must be exact; citation identity and bounds must pass validation.

The experiment saves a source snapshot of the selected runtime, including ML and
deterministic code. Frozen weight/calibration hashes are recorded. It does not
change the other chat's working checkout, train anything, or tune on Stoa labels.

## Local commands

From the parent Planproof workspace, run `tools/run_paired_pilot.py` with the
feedback Python environment and these required arguments:

| Argument | Local value |
| --- | --- |
| `--runtime` | `_planproof_agent` (checkout containing the guided runtime) |
| `--pack` | `_planproof_nemotron/.artifacts/nemotron/stoa-random-v1` |
| `--binding` | `_planproof_agent/.artifacts/guided-pilot-v2/binding.json` |
| `--out` | A new directory beneath `_planproof_nemotron/.artifacts/nemotron/` |
| `--view` | `_planproof_nemotron/.artifacts/nemotron/comparison` |
| `--env-file` | `_planproof_nemotron/.env` |
| `--ml-python` | `_planproof_ml/.venv/Scripts/python.exe` |
| `--geometry-python` | `_planproof_evidence/.venv/Scripts/python.exe` |

The credential stays in the existing ignored `.env`; requests omit Authorization
from their journals. Never put the key in command arguments or source files.

Use `--resume` only to continue the same saved experiment. It validates the
runtime and settings, retains all actual provider attempts and only permits
retrying a presentation failure that saved no provider request. Local error
records are archived. It can reuse completed algorithm evidence after checking
the original image/model receipt. Browser file locks cannot cancel model calls.

For this run, direct calls were first recorded in `paired-pilot-v1`; its guided
worker launch failed before inference. `paired-pilot-v2 --direct-from ...` retains
the matching new direct records and runs the guided condition. A direct call
blocked by a presentation error before making any request is completed on resume.
This preserves real model failures and avoids retrying for a more favorable score.
Actual chronological order is recorded; it is not a randomized trial.

Regenerate the offline page without API calls:

```text
python tools/build_paired_report.py RUN_DIRECTORY COMPARISON_DIRECTORY
```

`test_paired_report.mjs HTML_PATH OWNED_BROWSER_PROFILE` checks the real page with
an isolated headless Chrome profile. Unit regressions in
`tools/tests/test_paired_pilot.py` cover denominator semantics, failure journals,
Windows file locks, and restrictions on retrying provider attempts.
