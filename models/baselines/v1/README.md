# Planproof trained baseline v1

This is a frozen baseline for **product inference and future training**. The
models are stored using [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage),
with SHA256 manifests so a recovered copy can be checked independently.

## Restore and verify

```bash
git lfs install
git clone --branch ml/trained-baseline-v1 https://github.com/Willi-CQ/Planproof.git
cd Planproof
git lfs pull
python ml/scripts/verify_baseline.py
```

This repository is private, so use your authorized GitHub account. A raw LFS
pointer is not the model; the verifier reports this explicitly. `manifest.json`
lists every published payload's exact size and SHA256. Bundle-level checksums
and validation reports are preserved too. Publish improvements under a new
version rather than replacing this baseline.

## Models and intended use

| Bundle | Use | Verified result |
| --- | --- | --- |
| `corrected-3500-global/` | Main wall, room and opening detector | Full 2051-drawing project test: wall IoU .7134, object mAP50 .5870, room F1 .7391 |
| `cad-objects-3000-global/` | CAD doors, windows and stairs only | 128-drawing project validation: object mAP50 .4675; door .7108, window .4479, stairs .2436 |
| `structural-1693-global/` | Preserved earlier fallback | Historical validation evidence included; not the selected production default |

`SELECTED.json` identifies the main bundle; `CAD_OBJECTS.json` identifies the
specialist. Each includes `detector.pt`, ONNX output, a checksum-bound profile,
model metadata and measured evidence. The main model also has `wall.onnx` and
the complete held-out `test.json`.

The main ONNX publishes five heads. CAD ONNX publishes only centers and boxes;
the final CLI automatically filters its JSON/overlays to door/window/stairs.
Both profiles use **global1024**, so keep `--global-view`.

## Product inference

Install the pinned ML environment following [ml/README.md](../../../ml/README.md).
Inference requires only a bundle and a rendered page, not the training datasets.

```bash
planproof-ml infer \
  --checkpoint models/baselines/v1/corrected-3500-global/detector.pt \
  --calibration models/baselines/v1/corrected-3500-global/calibration.json \
  --global-view --image PAGE.png --document-id drawing-123 --page-index 0 \
  --out .artifacts/inference/main --device cpu
```

Use `--device cuda` on a compatible GPU. Substitute the CAD bundle for object
evidence from CAD drawings. Outputs contain original-image pixel coordinates,
instance IDs, confidence, model provenance and an overlay. Render PDF/vector
pages first and preserve the page-to-pixel transform in the application.

Stairs remain unreliable; exact counts, room categories, metric measurements
and end-to-end Stoa answer accuracy need independent verification. These are
component metrics, not publisher benchmark reproductions. Probabilities are
not correctness guarantees. See [the measured report](../../../ml/TRAINING_RESULTS.md).

## Continue training

The inference checkpoints support fresh training on compatible original datasets:

```bash
planproof-ml train --config NEW_RECIPE.yaml --manifest ORIGINAL_DATA_MANIFEST.jsonl \
  --init-weights models/baselines/v1/corrected-3500-global/detector.pt \
  --run .artifacts/runs/next-experiment --max-steps 100 --max-minutes 10
```

This retains learned weights and resets optimizer, schedule and step counters.
Use the CAD weights or CubiCasa-only seed for the corresponding experiment.
The original full training checkpoints are also preserved:

| File under `training-checkpoints/` | Purpose |
| --- | --- |
| `cubicasa-only-1000.pt` | CubiCasa-only seed before cross-dataset transfer |
| `main-selected-3500.pt` | Exact selected main training state |
| `cad-selected-3000.pt` | Exact selected CAD training state |
| `structural-fallback-1693.pt` | Earlier fallback training state |

These retain optimizer/scaler/RNG/configuration state. `CHECKPOINTS.json` records
their steps, source commits and dataset fingerprints. Exact `--resume` requires
the matching historical code, config, world size and reconstructed dataset;
legacy target processing must not silently resume under revision2. For a new
dataset or schedule, use `--init-weights`. Original completed schedules should
not be extended by editing a resumed configuration.

`TRAINABILITY_VERIFICATION.json` verifies a real fresh optimizer update from the
main, CAD and CubiCasa-only weights using generated fixtures. It is a runtime
check, not another accuracy benchmark; accepted weights remain unchanged.

## Data boundary

No training images, labels, dataset split manifests, Stoa questions, `.brain`,
agent instructions, credentials or machine access details are published here.
Dataset reconstruction metadata is kept only in the local complete backup.
Original data must be downloaded independently on the training machine. Stoa
did not train or select these weights; upstream datasets can overlap the
application corpus, so this does not establish unseen Stoa QA accuracy.

Publisher source notices and available licence texts are retained with the
bundles. No claim of unrestricted commercial rights is made. The original
implementation snapshot is source commit
`e761b8410f8805c371bb7cb6e4a19a8afaad2ae5` in this branch's ancestry.
