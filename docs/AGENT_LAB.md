# Agent improvement and regression protocol

The product UI is frozen. Agent experiments run in separate containers using the
production HTTP route, the same real detector checkpoint, and unchanged source
drawings. Never run an experimental benchmark against the live demo container.

## Datasets and controls

- Keep the existing 50-case cohort unchanged as a regression check. Run a fresh
  deployed control: an old recorded score alone does not control provider changes.
- `prepare_agent_lab.py` selects fresh questions without reading their answers,
  excludes prior drawing names and hashes, and separates development from holdout.
  It reads only the reference registry, never the sealed final challenge set.
- Evaluate new behavior on development data. Freeze a candidate revision and
  runtime image before running the holdout. If holdout results influence another
  change, that set is now development evidence and must be described as such.
- `prepare_agent_policy_suite.py` creates explicitly synthetic checks for units,
  rotations, missing scale, prompt injection, and scope. Report these separately
  from accuracy on real drawings. An API failure is not a successful abstention.

The stored selection is disjoint by source file and hash, not proven independent
by building family. Some questions require information outside a supplied sheet.
Flag these for review while preserving the original strict denominator and labels.
An agent holdout is not automatically held out from the frozen detector: Stoa's
CubiCasa examples can share publisher source IDs with the external CubiCasa
training corpus. Audit the checkpoint's frozen manifest and report those overlaps
separately. A separate download or changed raster encoding does not remove shared
source identity; train-pool membership also does not prove sampling in every run.

## Running a matched trial

Set `PYTHONPATH=backend;feedback/src;deterministic/src;tools` on Windows (colon
separators on Linux). Supply a private pack and a receipt containing the actual
server image digest and deployed revision. Each trial gets a new output directory.

```sh
python tools/run_agent_lab.py --pack PRIVATE_PACK --split development \
  --out PRIVATE_TRIAL --url LOOPBACK_API --revision DEPLOYED_COMMIT \
  --runtime-receipt PRIVATE_RECEIPT --workers 2 --execute --score
```

For a synthetic policy run omit `--score`, then use
`score_agent_policy_suite.py --pack PRIVATE_PACK --run PRIVATE_TRIAL`.
For held-out runs, omit `--score` until all records are terminal and the candidate
freeze has been recorded. Run again with the same binding and `--score` to evaluate;
completed requests are retained and are not billed again.

The runner binds source hashes, runtime, model, unit-hint policy, selected IDs and
its own code hash. It records admission IDs before execution and saves terminal
responses. An ambiguous admission must be investigated rather than silently
submitted again. A `STOP` file stops new admissions while retaining in-flight work.
Keep at most two workers per condition and four total on the shared GPU.

New bindings also hash the imported production runner. Legacy reference packs may
lack category/language annotations; those are recorded as unspecified/unknown
instead of losing a completed response. Existing frozen trials keep their original
bindings and runner revision; do not edit old records or rerun inference to upgrade
reporting metadata.

Exact integer counts and the existing 2% area tolerance are unchanged. All failures
and abstentions count in the denominator. Answers remain in evaluator-only files;
the agent gets the question and publisher unit hint identically in both conditions.

## Review and release

`compare_agent_trials.py` checks matched pixels, questions and references, reports
case-level gains and losses, and includes descriptive uncertainty. A small gain
on development data is not proof of generalization. Report latency alongside
provider failures and shared-load limitations. Reported seconds begin after upload
and canonical-frame preparation; they include question execution and HTTP polling,
not file upload/render time. Token totals can include cached
input and are not a monetary cost estimate.

`build_agent_lab_report.py` creates a private offline HTML review with source
thumbnails, questions, reference and model answers, citations and tool traces.
Originals and raw records remain private; only reusable tools, tests and this
protocol belong in Git. Do not upload drawings, benchmark answers, credentials,
model weights or the personal `.brain` with an ordinary source-code push.

`audit_agent_evidence.py --run PRIVATE_TRIAL --out PRIVATE_AUDIT.json` checks
retained source identity, citation bounds and delegation/count contracts. It also
records provider interruptions and tool usage. This is a structural check, not a
claim that a citation proves the answer or that the whole drawing was inspected.

Promote only after reviewing both the unchanged regression cohort and held-out
results. Retain the previous source/image for rollback, verify the frontend tree
is identical, and keep the live demo running during preparation.

## Backend changes under evaluation

Native PDF text is a bounded locator, not trusted evidence. Vector paths are
candidate boundaries, not automatically rooms. Registered polygon measurement
requires independent cited scale anchors and complete inspected source coverage;
curved/compound paths and overlapping regions are rejected. Nemotron verification
and calibrated-measurement guards still apply.

Requested asynchronous ML/deterministic catalogues are delivered between turns
when ready. Their public tool boxes use original-page normalized coordinates in
0..1000, matching crop tools; internal room/source coordinates remain native
pixels. Algorithm suggestions are fallible navigation aids, never totals.

Door inventory prompts distinguish physical symbol visibility from an entrance
role. A role-specific total still needs adjacent context and labels, and uncertain
objects or incomplete inspections still prevent an established total.

Region inspections can request a clockwise quarter-turn for upside-down or
sideways text. Only Nemotron's detail view is rotated, using lossless PNG output;
the original drawing, selected page bounds and all returned citation coordinates
stay in their existing source frame. The host validates locations before applying
the inverse transform. This capability is an experiment until live comparisons
show whether it helps; rotation does not restore missing source detail.
