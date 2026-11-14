# Local verification — 2026-11-13

Status: ready to connect a GPU host for CUDA and real-source validation.

- Python 3.12.13; PyTorch 2.5.1+cpu; Windows x86_64.
- 40 tests passed. Coverage includes dataset adapters, coverage masking, grouped
  split leakage, geometry transforms, archive safety, download resume/checksums,
  validation-only experiment comparison, losses, exact CPU checkpoint resume,
  original-coordinate inference, and dynamic-shape ONNX parity.
- Ruff lint and format checks passed. Bash preparation script syntax passed.
- Dependency consistency check passed.
- The CLI generated-fixture check trained one update, resumed to update two,
  evaluated a held-out fixture, and exported full and wall-only ONNX models with
  ONNX Runtime parity. No publisher data or pretrained weights were downloaded.
- The required-GPU probe correctly refused this CPU-only environment.
- Root README remains empty; private notes, instructions, datasets and model
  artifacts are excluded from source control and the Docker build context.

Reproduce with the commands in [README.md](README.md). The GitHub CI workflow is
prepared but has not run remotely. CUDA, DDP on real GPUs, the Linux container
build, actual publisher archive imports, and real-data accuracy are not locally
verified. These checks happen on the GPU host using [GPU_READY.md](GPU_READY.md).

Contract fixtures do not establish that every variant of a publisher's export
format is supported. Inspect real ground-truth overlays before long training.

## Hackathon update — 2026-11-14

All 42 CPU tests passed after adding bounded sessions, STOP-file checkpoint/resume,
run status and health warnings. Ruff lint/format and Bash syntax checks passed.
The initial launch profile now uses 512px, 2000 updates maximum and explicit
session time limits. See [HACKATHON.md](HACKATHON.md). GPU performance and actual
training quality remain unmeasured until a host is connected.

## First A100 validation — 2026-11-14

The pinned Linux container built on an A100-SXM4-80GB with driver580.178.04.
The CUDA forward/backward smoke passed with finite gradients; the ImageNet
encoder initialized and cached successfully. Original CubiCasa ZIP verified
against the publisher's MD5 and extracted on persistent GPU storage.

Real overlays exposed a CubiCasa coordinate bug: polygons already use
F1_scaled raster pixels, while SVG viewport dimensions can differ. The source
adapter now preserves those pixels and requires the matching scaled raster.
Eight corrected ground-truth overlays were visually checked. The original ZIP
also contains train/val/test lists; nonexistent supplemental GitHub URLs were
removed. Regression coverage checks pixel alignment and embedded split files.

Preparation accepts 1-16 bounded CPU workers; CubiCasa workers parse and convert
drawings in separate processes, preserving deterministic record order. Serial
and parallel fixtures produce identical manifests. The full corpus import and
real-data model accuracy are still pending at this checkpoint.


## Supervised original-source training ? 2026-11-14

59 CPU tests pass on the overnight source, including strict transfer/resume,
deterministic prefetch parity, optimized postprocessing/metric equivalence,
original CAD raw IDs, shuffled ResPlan publisher IDs, explicit import exclusions,
paired GIoU/reference parity, gradient coverage, and pre-validation checkpoints.
Ruff lint/format pass. The pinned A100 container has completed clean-data CUDA
updates, validation and checkpoint saves. This verifies execution, not the
requested accuracy targets. See DATA_VALIDATION.md for dataset evidence.

The initial 512-pixel model learns walls and rooms but has weak opening-box
localization. A bounded overlap-loss refinement is evaluated independently;
previous baseline weights are preserved. Final numerical claims belong in the
measured run report after matched-cohort validation, not this readiness record.
