# Offline evaluation review

`tools/build_case_review.py` turns a completed paired run into one self-contained
HTML file containing every question, reference answer, adaptive/direct result,
original drawing, citation overlay, specialist crops and recorded trace.

It distinguishes correct answers, wrong answers, execution failures and
abstentions. Filters cover outcomes, question categories, English search and
agent-only successes. Source images are embedded at their original resolution;
no external network access is needed. The current presentation targets the
completed 50-case cohort, not arbitrary benchmark sizes.

```text
python tools/build_case_review.py --run <completed-run-directory> --answers <source-pack>/evaluator/answers.json --translations <private-translations.json> --out .artifacts/review.html
node tools/test_case_review.mjs .artifacts/review.html
```

Use the backend Python environment (Pillow is required). The browser check uses
Microsoft Edge on Windows; set `PLANPROOF_BROWSER` to another Chromium executable
if needed. It starts an isolated headless profile and checks against the embedded
records without calling a model.

The translation JSON maps case IDs to `original` and `english` strings. Every
non-English question needs a translation matching the exact original question;
preserve quoted drawing labels and identifiers. Translations affect display only.
Reference-pack hashes, paired completeness, answer/trace consistency and decoded
RGB raster hashes are verified before generation. Original evaluation files and
scores are unchanged.

The generated file contains private drawings and reference labels. Keep it and
translations under ignored `.artifacts/`; commit only the generic source and
documentation. No dataset, answer file or private report belongs in GitHub.

The latest local review passed ten browser checks covering all 50 displayed
questions and answers, all 24 translations, outcome/category/search filters,
canonical citations, failure explanations, actual crop pixels, zoom, mobile
layout and absence of external requests or runtime errors.

## Additional original-resolution drawings

`tools/extend_case_review.py` adds a separate human-review tab to the same artifact.
The frozen 50-case payload and scores stay unchanged. Additional runs are never
silently mixed into that benchmark. Back up the original HTML before rebuilding:

```text
python tools/extend_case_review.py --base <frozen-review.html> --inventory <inference-only-inventory.json> --results <probe-results-directory> --out <same-review.html>
node tools/test_case_review.mjs <same-review.html>
```

The inventory declares a source root and cases with IDs, numbers, relative image
paths, original byte hashes/dimensions and fixed questions. Results use the
`run_resilience_probe.py` record format. The builder checks question/ID matching
and verifies the actual upload's source hash equals the original file hash.
Missing results remain explicitly pending. Rebuilding atomically replaces only
the additional review section; original images are embedded without resizing.

Each drawing shows its fixed question, original-resolution image, zoom controls,
an independent human-answer field, and an explicit reveal button. Counts remain
unscored until a human supplies an answer. Local browser storage holds the answer,
notes, timestamp and whether the model answer had already been revealed; export
downloads a durable JSON copy. Human answers never enter an inference request.
The review can display model citations, crop feedback, tool routing, recovery and
raw traces. A match to a human count is not an official benchmark label.

## Large source images

PNG/JPEG uploads and algorithm source validation support up to 150 million pixels
with the existing 10MiB compressed upload limit. Original spatial resolution and
RGB pixel hashes remain unchanged for opaque PNG drawings. Transparency is still
composited on white; JPEG EXIF orientation retains the prior behavior. PDF rendering
keeps its existing separate budget. The ML inference reader accepts the same
150MP input size; training dataset reader limits are unchanged.

Gemini's overview still fits within 2048 pixels per side. Every requested crop
starts from the preserved full-resolution source, then applies the model's own
image encoding limit with explicit coordinate transforms. The ML detector already
uses a bounded working raster and maps predictions back to original coordinates.
Increasing upload capacity does not mean every model processes every source pixel
simultaneously. Run these very large drawings sequentially on memory-constrained
hosts; algorithm subprocesses retain their existing timeouts and failure reporting.

`backend/run_local.py` selects feedback source from the launched checkout so an
interpreter reused from another worktree cannot silently import stale limits.
Test with that checkout's packages (or set PYTHONPATH to `feedback/src` when using
a sibling editable environment). `backend/test_large_drawings.py` checks uploads
above the old limit, unchanged raster hashes, native detail coordinates and finite
decode rejection. The review extension tests reject changed or mismatched inputs.
