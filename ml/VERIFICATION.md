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
