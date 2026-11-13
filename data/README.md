# Data registry and local layout

Tracked files in this folder contain source metadata and acquisition guidance only. The local working copy has ignored dataset files under `downloads/`, `extracted/`, and `evaluation/`; none are tracked. The root `README.md` remains empty until the team has a project description.

## Why raw datasets are not committed

- GitHub blocks ordinary Git files above 100 MiB; Git LFS has plan-dependent per-file limits and metered storage/bandwidth. The Stoa full archive is about 9.77 GB, so it cannot be handled as a normal repository file.
- The Stoa archive README gives per-file licenses and credits, including NC, SA, and ND conditions. External collections also have dataset-specific terms; a public download link is not blanket permission to mirror the data into this repository.
- Keep the Git repository reproducible and lightweight. Fetch approved files from their publishers into local ignored folders and preserve source IDs, licenses, and checksums in a local manifest.

## Tracked files

- `manifest.csv`: dataset source, role, current Stoa-preview overlap, access state, terms, and collection notes.
- `training-plan.md`: external training-source roles, access gates, split policy, and local-only data boundary.
- `README.md`: instructions and storage policy.

## Local-only layout

```text
data/
├── downloads/   # original external archives, ignored by Git
├── extracted/   # source files, ignored by Git
├── processed/   # derived training data, ignored by Git
├── evaluation/  # Stoa records and final evaluation data, ignored by Git
├── manifest.csv # tracked metadata only
└── training-plan.md # tracked acquisition/training policy only
```

Do not mirror Stoa data or external archives to GitHub. Stoa is evaluation/reference only; it must never enter model training, tuning, augmentation, or model selection. Before downloading an external source, inspect its current license and provenance. Keep external train/validation/test splits grouped by source drawing/building so alternate formats or floors cannot leak across splits.

## Start with these sources

Stoa is reserved for final client evaluation and reference only; do not use its drawings, questions, answers, labels, or derivatives to train or tune any model. See [training-plan.md](training-plan.md) for the external training roles, acquisition gates, and split rules. Raw source data, local evaluation records, processed samples, checkpoints, and weights must stay in ignored local folders. Use IFC samples for parser smoke tests, not as image detector training data.
