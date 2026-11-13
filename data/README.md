# Data registry and local layout

This folder contains source metadata and acquisition guidance. It does not contain copied dataset binaries. The root `README.md` remains empty until the team has a project description.

## Why raw datasets are not committed

- GitHub blocks ordinary Git files above 100 MiB; Git LFS has plan-dependent per-file limits and metered storage/bandwidth. The Stoa full archive is about 9.77 GB, so it cannot be handled as a normal repository file.
- The Stoa archive README gives per-file licenses and credits, including NC, SA, and ND conditions. External collections also have dataset-specific terms; a public download link is not blanket permission to mirror the data into this repository.
- Keep the Git repository reproducible and lightweight. Fetch approved files from their publishers into local ignored folders and preserve source IDs, licenses, and checksums in a local manifest.

## Tracked files

- `manifest.csv`: dataset source, role, current Stoa-preview overlap, access state, terms, and collection notes.
- `README.md`: instructions and storage policy.

## Local-only layout

```text
data/
├── downloads/   # original archives, ignored by Git
├── extracted/   # source files, ignored by Git
├── processed/   # derived training/evaluation data, ignored by Git
└── manifest.csv # tracked metadata only
```

Do not mirror Stoa data or external archives to GitHub. Before downloading a source, inspect its current license and provenance. For Stoa, retain its per-file `LICENSE.md` records and preserve credits; do not alter ND material. Keep train/validation/test splits grouped by source drawing so alternate formats or pages from one drawing cannot leak across splits.

## Start with these sources

For direct floor-plan parsing, evaluate Stoa's own labeled examples first, then FloorPlanCAD and CVC-FP. HouseExpo may help room-category/topology recognition, but it is a residential layout dataset and its data rights must be checked separately from its MIT-licensed code. Structured3D is synthetic and access requires accepting its terms. Use IFC samples for parser smoke tests, not as a substitute for the Stoa evaluation set.
