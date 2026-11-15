# Production cohort audit

This compares a frozen historical adaptive run with the current application's
upload, resumable run, and polling API. The current observer, interactive limits,
mandatory specialist routing, and answer presentation are exercised. It is not a
direct invocation of the historical benchmark's more permissive agent settings.

## Measured result, 2026-11-15

The authorized pass completed all 50 cases once, with pauses after regressions
29 and 46. In-flight requests were retained; no completed case was rerun. Source
was frozen at the pre-repair implementation. All canonical raster hashes match.

| Measure | Historical adaptive | Current production, pre-repair |
| --- | ---: | ---: |
| Strict correct | 33/50 (66%) | 30/50 (60%) |
| Numeric count agreement / same area tolerance | 33/50 | 32/50 |
| Returned answers | 35 | 36 |
| Median seconds | 64.164 | 64.850 |
| Mean seconds | 142.504 | 66.726 |
| Longest seconds | 566.110 | 121.562 |
| Reported Gemini tokens | 1,458,269 | 3,090,015 |
| Nemotron calls | 22 | 369 |
| Cases receiving algorithm evidence | 19 | 0 |

Strict regressions: 6, 9 (count-unit format), 29 (area/provider interruption), 46
(door inspection). Gain: 30. Wrong numeric outputs: 23, 28, 37, 44. Fourteen
abstentions remain in the denominator. Areas score 2/12 and openings 2/8; these
are material weaknesses. Cases 23 and 44 used traced polygon estimates:
whole-unit rounding worsened their output, but case 23's underlying boundary estimate was already
outside tolerance. Fixing formatting does not fix geometric interpretation.

Thirteen current cases requested algorithms, but none received their finished
catalogue during the run. The audit started with a separate cold job cache;
background CPU jobs often finished too late, and some ready results were not
requested again. This is a measured delivery limitation, not evidence that the
detector's weights regressed. All 15 baseline model-file checksums match. The
current presentation adapter preserves the core answer and citations of all 50
saved historical outputs; observation callbacks pass scripted parity checks.

The implementation now includes the repairs below, with 420 automated Python
checks passing. The offline review passed all 50 case navigation/filter checks,
390px layout, and zero external requests/browser errors. Saved runtime outputs
are unchanged. **There is no fresh post-repair live accuracy measurement.**
Do not describe this cohort as an independent test set or the system as fully
validated. Warm algorithm delivery and end-to-end area/door correctness remain
unproven by this pass.

## Integrity and interpretation

- Bind the exact case IDs and order, original questions, source-file hashes,
  reference-manifest hash, historical-report hash, model ID, and current source
  hashes. Snapshot the implementation before inference. Verify the detector files
  against their existing checksum manifest independently.
- Send original drawings; verify canonical raster equality after upload. Keep the
  historical unit hint to avoid changing the benchmark question's contract.
  Ordinary browser chat omits this optional hint, which is a comparison limit.
- Never load evaluator answers into the inference process. Open them only after
  all requested cases have terminal records. Missing, duplicated, or nonterminal
  attempts must fail evaluation, not disappear from the denominator.
- Retain the historical scorer: exact integer counts and 2% area tolerance.
  Report count-value agreement as a separate diagnostic when a semantically
  correct number has an incompatible unit; never silently replace the score.
- Failures, abstentions, incorrect answers, unit mismatches, and regressions are
  different outcomes. A completed HTTP request is not necessarily an answer.
- The same development cohort has been inspected previously. This is a
  historical comparison, not a simultaneous randomized control or independent
  generalization estimate. Citation bounds do not prove semantic correctness.

## Reproduction

Start an isolated instance of the current production backend, with the existing
Gemini/Nemotron configuration and separate trace/job directories. No provider
keys belong in command arguments, reports, or this repository. Prepare first:

```powershell
python tools/audit_production_cohort.py --baseline <historical-run> --out <private-output>
```

`--execute` performs the authorized paid pass, at most two cases concurrently.
`--url` defaults to `http://127.0.0.1:8005`. A `STOP` file in the output directory
pauses new admissions while retaining in-flight results. Persistent request IDs
prevent automatic resubmission of ambiguous or completed attempts. Preserve pause
receipts; investigate a regression before resuming. A modified implementation must
use a new run directory, never be substituted midway through a comparison.

Build the private offline review after evaluation:

```powershell
python tools/build_production_audit_report.py --audit <private-output> --baseline <historical-run> --translations <verified-english-translations.json> --out <review.html>
node tools/check_production_audit.mjs <review.html> <browser-receipt.json>
```

The review embeds the original pages, original and translated questions, reference
answer, before/after outputs, timings, source boxes, and recorded public activity.
It checks trace/report agreement and raster identity. It makes no network requests.
Images, questions, answers, provider traces, and generated reports stay ignored and
local; only generic tools, tests, and aggregate documentation are tracked.

## Repairs found during the audit

Repairs are developed separately while the comparison source remains frozen:

1. Door counting used any mention of a leaf to select leaf counting, including an
   exclusion clause. It now uses the requested count noun. Openings and individual
   leaves remain distinct, with the entity retained in `count_basis` and prose.
2. Machine count units could drift to entity names. Completed localized counts use
   `count`; an explicit API count hint normalizes known count-noun units while
   preserving values, citations, explanations, physical units, and unknown units.
3. Rejected specialist output lacked a reason. Safe validation codes identify the
   schema, counting convention, inventory, or coordinate problem without exporting
   provider text or weakening validation.
4. Working turns and retries could be admitted with fewer than ten seconds and
   fail before reserved finalization. Interactive admission now requires ten
   seconds; otherwise it hands existing evidence to bounded finalization. Total
   time and request caps are unchanged. Authentication and arbitrary HTTP 400
   errors still are not blindly retried.
5. Raster polygon estimates were rounded to whole output units, adding substantial
   relative error for small rooms or rejecting positive sub-unit areas. Three
   significant figures now preserve useful numeric information across units;
   the approximate flag, scale/boundary uncertainty, and cited inputs remain.
   This does not correct an inaccurately selected polygon.
6. Nemotron sometimes used `verdict: observed` for an ordinary visual inspection.
   That observation/review enum confusion caused otherwise usable detections to
   be discarded. The host now demotes that one alias to `uncertain`, records the
   normalization, and applies all remaining validation. It does not normalize a
   review of a draft, infer missing objects, or accept malformed/truncated JSON.
   Offline replay of twelve saved count responses from the door regression
   changed accepted inspection payloads from three to nine. This is parser/evidence
   retention improvement, not a new end-to-end answer or accuracy measurement.

The fourth repair is supported by observed sub-six-second HTTP 400 failures,
the installed SDK's deadline-header construction, and the firsthand
[Gemini deadline rejection report](https://github.com/livekit/agents/issues/3723).
The original error bodies were deliberately not saved, so the precise server
message for those historical failures cannot be confirmed. The
[official SDK tests](https://github.com/googleapis/python-genai/blob/main/google/genai/tests/client/test_http_options.py)
document the timeout-to-server-header behavior. Sources checked 2026-11-15.

Offline regression tests demonstrate repaired host behavior, not improved live
model accuracy. Never relabel pre-repair predictions as a new measured pass.
