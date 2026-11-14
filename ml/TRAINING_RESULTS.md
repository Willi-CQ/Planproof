# Supervised training — 14 November 2026

The overnight runs produced a working detector and a validated inference bundle.
The accepted fallback is the CubiCasa → ResPlan → structural-refinement lineage.
A bounded refinement using corrected explicit-object supervision is still being
compared; this report will record the final selection before handoff.

## Measured fallback

Checkpoint: `runs/structural-refine/final-review.pt`, structural step 1693.
Operating point: global view, canvas 1024, mask thresholds
`[0.35, 0.65, 0.5, 0.5, 0.5]`, center thresholds `[0.1, 0.1, 0.1, 0.1]`.
Thresholds were selected on 32 validation drawings per source and then kept fixed
for the larger 128-per-source confirmation below.

| Validation source | Drawings | Wall IoU | Raw room-mask IoU | Room-instance F1 @ IoU .5 | Object mAP @ .5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CubiCasa5K | 128 | .6433 | .8679 | .4214 | .4991 |
| ResPlan | 128 | .7694 | .8780 | .6711 | .5897 |
| Source macro | 256 | .7064 | — | .5463 | .5444 |

Door/window AP50 is .7148/.6999 on CubiCasa and .8792/.8901 on ResPlan.
Stairs are weak: .0826 AP50 on 96 CubiCasa instances, zero on seven ResPlan
instances. Room-mask quality does not establish reliable room counts or categories.
The provisional .80/.80/.70 targets have **not** been met.

These are project validation metrics, not publisher benchmark reproductions,
final test scores, or Stoa question-answering accuracy. All current comparisons
use `project-v2-original-pixel-coordinates-known-object-coverage`; historical v1
reports must not be compared directly with these results.

## Data actually used

| Original source | Prepared originals | Project train / val / test | Role in accepted lineage |
| --- | ---: | --- | --- |
| CubiCasa5K | 5,000 | 4,193 / 404 / 403 | Baseline and replay throughout |
| ResPlan | 16,317 | 13,014 / 1,655 / 1,648 | Transfer and replay |
| FloorPlanCAD | 9,920 usable source-train SVGs | 8,697 / 1,223 / 0 | Separate transfer experiments; not adopted |
| HouseExpo | 35,126 JSON originals downloaded | No full prepared training split | Eight label previews only; not trained |

Other catalog entries are not completed training runs. Stoa files and questions
were not used for training, threshold tuning, or checkpoint selection.

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

CAD transfer, class-balanced mixed-source training and a box-size refinement did
not establish a better universal detector. Their checkpoints remain separate.
A CAD viewport audit ruled out out-of-frame annotations as the main explanation.
No claim is made that additional datasets automatically improve accuracy.

## Integration bundle

The validated fallback is stored on the GPU at
`/srv/planproof-data/releases/structural-1693-global`, with a local ignored backup
under `.artifacts/releases/structural-1693-global`.

The bundle contains `detector.pt`, `detector.onnx`, `wall.onnx`, `metadata.json`,
the rebound `calibration.json`, validation evidence, real-input parity evidence
and `SHA256SUMS`. The slim PyTorch checkpoint has identical model parameters and
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
were checksum-verified after local backup, and the actual slim checkpoint loaded
and ran a finite five-head CPU forward locally. Tiled inference needs separate
calibration and validation; do not reuse this global calibration in tiled mode.

## Continuing training

Use `--init-weights /path/to/detector.pt` with a new run/config for another compatible
dataset. It loads the shared architecture/schema and resets optimizer, schedule
and counters while recording parent provenance. Exact `--resume` requires the
original training checkpoint and matching config, manifest and world size.
Use the corrected frozen manifests, not historical diagnostic imports.

The selected artifact is intended for the hackathon evidence pipeline. Reliable
stairs, precise room counts, room categories, metric measurements and Stoa QA
accuracy remain integration/evaluation work, not demonstrated capabilities.
