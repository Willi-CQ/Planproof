# Original-source validation — 2026-11-14

Data and model artifacts live on the GPU volume only. These observations describe
the independently acquired publisher releases, not Stoa's client dataset.

## CubiCasa5K

All 5,000 drawings were converted and deeply audited. The ZIP's embedded lists
contain 4,200/400/400 original train/validation/test drawings. Exact image and
geometry grouping, plus a visually confirmed near-duplicate pair (high_quality
1506 and 1742), produce the frozen clean project split: 4,193/404/403 drawings,
4,978 groups. This is not the unchanged publisher benchmark.

Use `processed/cubicasa-clean-manifest.jsonl`, fingerprint
`dd84fe59c6a6744a9b7bced6bddc5b89d99c139849e9b1492ec2238ebd40bb27`.
Its ground-truth polygons use F1_scaled raster pixels, independently of the SVG
viewport. Large original rasters are capped at 4,096 pixels with the identical
geometry scale recorded. Unlabeled decorative NaN paths are skipped; malformed
semantic geometry still raises errors. No drawing was silently dropped.

The old fast-baseline probe used the pre-curation split. It is a performance
diagnostic and must not be used as a transfer parent.

## ResPlan

The 17,000-entry pickle is shuffled. The canonical split refers to `plan['id']`,
not the list index. Read the [publisher's split description](https://github.com/m-agour/ResPlan)
before changing this reader. The 683 augmentation IDs are excluded by default.
The canonical original corpus contains 16,317 plans. Exact duplicate grouping
produces 13,048/1,635/1,634 project train/validation/test records, 16,297 groups.

Use `resplan-canonical-processed/manifest.jsonl`, fingerprint
`5f2c255d251db92a7d4a09169ef9d6b1bc766fd87de310c61839edad9c43a5c9`.
The earlier `resplan-processed` manifest used archive indices and is quarantined.
Migration reused 15,567 geometry exports, rendered 750 missing originals, and
re-rendered eight reused examples to verify pixel and target hashes exactly.
Stable publisher ID and archive index are recorded separately.

## FloorPlanCAD

The original three archives contain 15,663 SVGs: 3,760 train-00, 6,401 train-01,
and 5,502 test-00. Preserve those directories to avoid filename collisions.
The original SVG uses `semantic-id` and `instance-id` attributes. Its raw IDs
are **3–8 doors, 9–11 windows, and 30 stairs**, as confirmed by the authors'
[raw-to-benchmark remapping](https://github.com/VITA-Group/CADTransformer/blob/main/config/anno_config.py).
Benchmark category IDs are different. The checked map is
`configs/floorplancad-v2-map.json`; raw 28 is a urinal, not stairs.

Inputs are monochrome linework rendered from SVG geometry, without annotation
colors. Supervision covers object boxes only; wall/room masks are not fabricated.
Eight actual examples were visually checked for box alignment.

Group all crops with the same filename prefix. The provided test archives share
many prefixes with training, so they remain outside training and checkpoint
selection. Derive project validation from held-out training families instead.
Also exclude 241 training SVGs without semantic annotations; do not label these
as fully annotated negative pages. Explicit source-relative exclusions and
reasons are retained in GPU provenance. Expected usable training-source count:
9,920, before project family splitting. These are project metrics, not a
reproduction of the published benchmark.

## Curriculum safeguards

Freeze each manifest. Compare prior-stage training hashes against each new
held-out cohort. Keep CubiCasa validation IDs, targets and splits unchanged when
adding sources. Use separately audited source tensors and hard links on the same
GPU filesystem. Source weights initialize a fresh optimizer and schedule through
`--init-weights`; `--resume` retains its strict exact-data/config contract.

Replay recipes start with 70% CubiCasa / 30% ResPlan, then 65% CubiCasa /
15% ResPlan / 20% FloorPlanCAD. These are experiments, not guarantees of better
accuracy. Preserve the previous best checkpoint and compare matching validation
cohorts before adopting a candidate. Never tune on Stoa or final test questions.
