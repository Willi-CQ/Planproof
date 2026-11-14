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
