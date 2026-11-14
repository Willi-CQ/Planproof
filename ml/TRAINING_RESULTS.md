# Supervised training — 14 November 2026

The selected detector completed CubiCasa baseline training, ResPlan transfer and
replay refinement with corrected explicit-object supervision. Its PyTorch and
ONNX inference bundle is exported, verified and backed up locally outside Git.
Selection was locked before the full held-out test results. That evaluation is
complete with unchanged weights and thresholds. A separate 3000-update CAD object
specialist is also exported and verified.

## Selected detector

Checkpoint: `runs/corrected-refine-selected/final-review.pt`, refinement step 3500
(the run completed 4000 updates; the final update was not automatically selected).
Operating point: global view, canvas 1024, mask thresholds
`[0.35, 0.65, 0.5, 0.5, 0.5]`, center thresholds `[0.1, 0.1, 0.1, 0.1]`,
and minimum wall/room component area 2048 working-image pixels.

### Full post-selection test

| Test source | Drawings | Wall IoU | Object mAP @ .5 | Room-instance F1 @ IoU .5 |
| --- | ---: | ---: | ---: | ---: |
| CubiCasa5K | 403 | .6450 | .5421 | .6393 |
| ResPlan | 1,648 | .7817 | .6319 | .8389 |
| Source macro | 2,051 | .7134 | .5870 | .7391 |

Door/window AP50 is .7434/.7270 on CubiCasa and .9477/.9406 on ResPlan.
Stair AP50 is .1559 on 377 CubiCasa instances and .0074 on 79 ResPlan instances.
Stairs remain unreliable. These component scores are not Stoa QA accuracy or
publisher benchmark reproductions. The test report did not select or tune models.

### Selection validation

Thresholds were selected on 32 validation drawings per source and then kept fixed
for the larger 128-per-source confirmation below.

| Validation source | Drawings | Wall IoU | Raw room-mask IoU | Room-instance F1 @ IoU .5 | Object mAP @ .5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CubiCasa5K | 128 | .6477 | .8773 | .6474 | .5767 |
| ResPlan | 128 | .7828 | .8918 | .8505 | .6819 |
| Source macro | 256 | .7152 | — | .7489 | .6293 |

Door/window AP50 is .7688/.7673 on CubiCasa and .9492/.9480 on ResPlan.
Stairs remain weak: .1941 AP50 on 96 CubiCasa instances, .1485 on seven ResPlan
instances. Room-category macro F1 is .5433/.7579, conditional on 716/799 matched
rooms with known category labels; it is not an all-room classification score.
The provisional targets (wall IoU .80, room F1 .80, object mAP .70) have **not** been met.

These are project validation metrics, not publisher benchmark reproductions,
final test scores, or Stoa question-answering accuracy. All current comparisons
use `project-v2-original-pixel-coordinates-known-object-coverage`; historical v1
reports must not be compared directly with these results.

The chosen candidate passed matching 256-drawing comparisons against the
preserved structural fallback and the earlier 2500-update candidate. Against the
2500 candidate with the same component filter, composite score improved by .0163,
with no per-source wall/object/room component dropping by more than .02.
The calibration cohort contains only one ResPlan stair: its AP is unstable, so
the larger cohort above is the validation headline.

## Data actually used

| Original source | Prepared originals | Project train / val / test | Role in accepted lineage |
| --- | ---: | --- | --- |
| CubiCasa5K | 5,000 | 4,193 / 404 / 403 | Baseline and replay throughout |
| ResPlan | 16,317 | 13,014 / 1,655 / 1,648 | Transfer and replay |
| FloorPlanCAD | 9,920 usable source-train SVGs | 8,697 / 1,223 / 0 | Separate object specialist; not in the selected main lineage |
| HouseExpo | 35,126 JSON originals downloaded | No full prepared training split | Eight label previews only; not trained |

Other catalog entries are not completed training runs. Stoa files and questions
were not used for training, threshold tuning, or checkpoint selection.
Original upstream datasets can share drawings with Stoa; a separate download
does not establish unseen final application questions or drawings.

See [DATA_VALIDATION.md](DATA_VALIDATION.md) for publisher checksums, raw CAD label
IDs, the ResPlan stable-ID correction and the conservative family exclusions.
FloorPlanCAD's 5,502 official test SVGs were excluded from both training and
selection because their filename families overlap the publisher train partition.
Another 241 unannotated train SVGs were explicitly excluded.

Frozen mixed-CubiCasa/ResPlan manifest fingerprint:
`46b36620848e0d23a9d042513bd990a7ffa33cde582f5eaaec3828f2a17a8a95`.
It contains 17,207 train, 2,059 validation and 2,051 test drawings.

## What supervision changed

The initial 512-pixel baseline learned masks but localized opening boxes poorly.
Higher-resolution views and scale-sensitive GIoU supervision made opening
localization usable; replay protected the original domain during transfer.
Cached task-specific validation calibration improved instance splitting without
additional training. Class-balanced heads and bounded low-learning-rate structural
refinement improved the validated global-view model.

An audit found 85 of 609 annotated stair centers inside ignored pixels across
200 stair-containing CubiCasa training drawings. Explicit complete-object positive
peaks now remain supervised there; unknown negatives stay ignored. Evaluation
also retains predictions inside known object coverage. This is a confirmed partial
issue, not proof that it explains every stair failure. Target-transform revision 2
prevents silently resuming an old optimizer with changed target semantics.

CAD transfer and class-balanced mixed-source training did not establish a better
universal detector. Their checkpoints remain separate. Full-view CAD-focused
training improved object AP on 128 matching CAD validation drawings from .1372
to .4032 at step1500 and **.4675 at step3000**. Final AP50 is .7108 door (285 GT),
.4479 window (85 GT) and .2436 stairs (22 GT). The final continuation improved
every class and passed the predeclared mean/per-class regression rule.
On the same CAD cohort the selected residential model scored .0250 object mAP,
supporting separate models for these domains. This is an object specialist, with
no claim of reliable CAD wall/room extraction or a publisher benchmark reproduction.
A CAD viewport audit ruled out out-of-frame annotations as the main explanation.
No claim is made that additional datasets automatically improve accuracy.

The existing minimum-component-area setting also mattered: on the same 2500
checkpoint and 256 validation drawings, room-instance F1 increased from .5465
to .7301 after filtering tiny fragments. The source-controlled
`scripts/calibrate_component_area.py` caches one inference pass per drawing,
tests a bounded area grid and confirms the selected profile through the ordinary
evaluator. The working image has a maximum long side of 4096; the area threshold
is not a physical square-metre cutoff and can remove legitimate tiny features.

```bash
python ml/scripts/calibrate_component_area.py \
  --checkpoint /data/planproof/runs/corrected-refine-selected/final-review.pt \
  --manifest /data/planproof/processed/mixed-resplan-manifest.jsonl \
  --calibration /data/planproof/reviews/corrected-final-calibration32.json \
  --areas 4 32 128 512 2048 --limit-per-source 32 \
  --out /data/planproof/reviews/area-calibration.json
```

## Integration bundle

The selected bundle is stored on the GPU at
`/srv/planproof-data/releases/corrected-3500-global`, with a local ignored backup
under `.artifacts/releases/corrected-3500-global`. `SELECTED.json` in the parent
release directory points to it. The earlier `structural-1693-global` fallback is
preserved independently.

The bundle contains `detector.pt`, `detector.onnx`, `wall.onnx`, `metadata.json`,
the rebound `calibration.json`, validation/selection evidence, real-input parity,
CLI integration evidence, dataset source notices and `SHA256SUMS`.
`test.json` contains the complete post-selection 2051-drawing report;
`selection-lock.json` records the frozen weights and thresholds before results.
The slim PyTorch checkpoint has identical model parameters and
omits optimizer state. Keep this bundle outside Git.

```bash
planproof-ml infer --checkpoint /path/to/release/detector.pt \
  --calibration /path/to/release/calibration.json --global-view \
  --image PAGE.png --document-id drawing-123 --page-index 0 \
  --out .artifacts/inference/drawing-123 --device cpu
```

Use `--device cuda` when running on the GPU. Input PDF pages need rendering first.
The output includes original-page pixel coordinates, instance IDs, polygons,
boxes, confidence and room-category predictions, plus a visual overlay. It does
not determine metric scale, perform OCR, or answer questions. Gemini and the
deterministic reader should treat these predictions as independently verifiable
evidence, with special caution around stairs and room counts.

All five ONNX heads and the wall-only export passed PyTorch parity on an actual
validation drawing at 1024 pixels (`rtol=1e-3`, `atol=1e-4`). All fallback files
and all selected-release files were checksum-verified after local backup; the
actual selected slim checkpoint loaded
and ran a finite five-head CPU forward locally. Tiled inference needs separate
calibration and validation; do not reuse this global calibration in tiled mode.

### CAD object bundle

`/srv/planproof-data/releases/cad-objects-3000-global` and the corresponding local
ignored `.artifacts/releases/cad-objects-3000-global` contain the CAD specialist.
The parent `CAD_OBJECTS.json` pointer declares its role and validation scores.

Use the same inference command with this bundle's checkpoint and calibration.
Its checkpoint declares `intended_classes = [door, window, stairs]`; the CLI
automatically limits JSON and overlays to these classes. Dense probability-map
export is rejected for this object-only release. Python callers of `detect`
should retain the declared classes before exposing results.

Its ONNX outputs are **centers and boxes only**. The center tensor has four
channels: channel0 is reserved, while channels1–3 represent door/window/stairs.
Boxes use the existing 12-channel encoding. Inspect `metadata.json` rather than
assuming every model exports five heads. Reproduce with `export --objects-only`.

The full-head CAD export failed strict real-input parity in unsupported mask/type
outputs. It was retained as a rejected diagnostic artifact. The object-only
export publishes the validated task: both outputs passed the original
rtol=1e-3/atol=1e-4 gate on an actual CAD drawing. No tolerance was relaxed.
The actual CLI also passed object-only output, original-coordinate overlay and
input-preservation checks. Its fixed threshold profile is validated, not a
statistical calibration of correctness probabilities.

## Continuing training

Use `--init-weights /path/to/detector.pt` with a new run/config for another compatible
dataset. It loads the shared architecture/schema and resets optimizer, schedule
and counters while recording parent provenance. Exact `--resume` requires the
original training checkpoint and matching config, manifest and world size.
Use the corrected frozen manifests, not historical diagnostic imports.

The selected artifact is intended for the hackathon evidence pipeline. Reliable
stairs, precise room counts, room categories, metric measurements and Stoa QA
accuracy remain integration/evaluation work, not demonstrated capabilities.

74 CPU tests and Ruff lint/format checks pass, including cached component-area
parity, calibration binding, dynamic object-only ONNX parity and non-vacuous CLI
scope filtering. The release's real-input and CLI checks supplement
fixture tests; no remote GitHub CI or multi-GPU execution is claimed.
