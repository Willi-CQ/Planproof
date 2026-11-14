# Visual comparison and data preservation

Open `.artifacts/nemotron/comparison/index.html` in this worktree. It works
offline, without a server, external scripts or uploads. Preparation already
created the report. It deliberately shows **not run**, not zero accuracy, until
actual inference results exist.

The report provides:

- Run selectors for Nemotron, Gemini Carbon and later saved runs/modes.
- Accuracy by question type, paired wins/losses and the accuracy difference on
  shared completed questions. Partial runs keep pending questions separate.
- A per-question outcome map, source/type/outcome filters and text search.
- Original drawings, exact sent-image previews, reference answers, both model
  answers, explanations and preserved critique stages.
- Evidence boxes overlaid only when their image ID/hash matches the preview.
  Different frames are identified rather than silently rescaled.
- Timing and token usage, including missing usage. No invented dollar costs.
- A downloadable comparison snapshot; original JSON/CSV files remain available.

The reference answer is evaluator-only; the exported model input file omits it.
Citation boxes identify proposed evidence, not independently verified truth.
Scores characterize this reference sample. They must not guide prompt/model tuning.

## Compare Gemini Carbon later

`comparison/questions-for-model.json` exports the frozen 100 question IDs, text,
units/types, source input SHA256s and exact sent-image files/metadata. It contains
**no reference answers**. Use these inputs in the later Gemini runner. Carbon is
the owner's comparison label; no undocumented Carbon API/schema is assumed.

Export the Gemini runner's results using this provider-neutral contract:

```json
{
  "schema": "planproof-predictions-v1",
  "dataset_sha256": "copy from questions-for-model.json",
  "model": "the actual Gemini model/version used",
  "mode": "direct",
  "image_policy": {"max_edge": 2048, "max_pages": 12, "pdf_dpi": 144},
  "records": [
    {
      "id": "the exported question ID",
      "input_sha256": "the exported original source hash",
      "status": "ok",
      "answer": {
        "status": "answered",
        "answer": 3,
        "unit": "count",
        "explanation": "Generated format example, not a Stoa result.",
        "citations": []
      },
      "latency_seconds": 1.5,
      "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    }
  ]
}
```

The numeric values above are generated schema examples. Record the actual model,
timing and provider usage; leave unavailable usage absent rather than inventing
it. A source-grounded answer should include citations and the image metadata
used by the run. For image overlays, each record may include `images` matching
the exported image metadata and citation `image_id`/`bbox` fields in sent-image
pixels. Foreign runs without verifiable image frames retain their citations as
raw data, without drawing potentially misleading boxes. The importer scores
answer values; foreign citation validity is not inferred from numeric correctness.

Use `answer.status="abstain"` with `answer=null` for abstention. Use a non-`ok`
record status for a failed request. A partial export may contain fewer than 100
records; omitted questions stay pending. Include all tool/crop/critique history
and raw provider output in additional fields if available; original import bytes
are preserved. Do not include credentials or Authorization headers.

Register, independently score, visualize and back up the run:

```powershell
.\nemotron.cmd benchmark-import --input path/to/gemini-carbon.json --label "Gemini Carbon"
```

The importer rejects unknown/duplicate question IDs, different dataset IDs and
source hash mismatches. It recomputes scores with the same frozen local reference
answers, ignoring supplied correctness flags. The original import is copied
unchanged into a new run directory, with a checksum receipt. A changed image or
prompt policy is visible as a workflow difference. This does not establish
identical reasoning conditions between models.

The report's **Add saved report** control can load a generated `report.json` or
comparison snapshot for local viewing. For durable registration use the command
above. Browser imports are in-memory until a snapshot is exported; the original
file remains on disk. No local-storage-only persistence is used.

## Everything retained

All live inference requests are journaled under `.artifacts/nemotron/calls/`:

- Exact request JSON, prompts, source-image payloads and inference settings.
- Every returned provider response, including reasoning fields if returned,
  before final-answer/schema checks; reasoning is never substituted for an answer.
- Retry HTTP errors, timing/request metadata and the validated feedback result.
- Authentication headers are omitted and the known key is redacted from response
  bodies. The real `.env` is never archived or committed.

Every benchmark persists predictions after each completed case. A completed,
failed or interrupted CLI benchmark then rebuilds the visual report and creates
an immutable local ZIP under `.artifacts/nemotron/backups/`. If report/backup
creation itself fails, original inputs, journals and incremental predictions
remain on disk and the error is explicit. No automatic pruning occurs.

Backups include the original drawings and source notices, all selected questions
and reference answers, frozen manifest and readiness receipt, exact preview
assets, every run/import/audit file, source/configuration snapshot and the full
existing reference-question registry plus archive inventory. Final-partition data
is never opened. Backups exclude previous backups, caches and the real `.env`.
Every payload has a SHA256 in `BACKUP-MANIFEST.json`; the archive passes ZIP CRC
and all payload hashes before a verification receipt is written.

```powershell
.\nemotron.cmd report
.\nemotron.cmd backup
.\nemotron.cmd verify-backup .artifacts/nemotron/backups/<archive>.zip
```

For restoration, verify the ZIP first. `evidence/` contains the complete local
benchmark folder, `registries/` the original reference/inventory files and
`source/` the implementation snapshot. Preserve relative artifact paths when
restoring `evidence/` so the report and original-drawing links remain connected.
Reinstall dependencies from the saved source; supply the key separately.

Local ZIPs are a verified second copy, not a claim that a cloud backup has synced.
The current workspace is inside OneDrive, but sync state is unverified. No data
or code was uploaded/published by this work.
