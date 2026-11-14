# Connecting the training GPU

For the actual trained release, start with [TRAINING_RESULTS.md](TRAINING_RESULTS.md).
The validated model uses a 1024-pixel global view and mode-bound calibration;
include `--global-view` in inference and evaluation commands for that release.
[HACKATHON.md](HACKATHON.md) describes the earlier bounded-run workflow. The
baseline commands below are preparation examples, not the completed run history.

The training implementation is prepared on branch **ml-training-preparation**.
The deterministic code remains on its separate branch. The root README stays empty.
No training archives, annotations, model files or private brain notes belong in GitHub.

## Readiness boundary

Prepared in code: source registry/acquisition, dataset preparation and coverage masks,
grouped splits/audits, detector, baseline/curriculum recipes, training and resume,
validation/calibration, test reports, original-pixel inference/overlays, raw
probability artifacts, ONNX export and an isolated wall-logit export.

CPU checks cover contracts; the pinned Linux image, CUDA forward/backward,
real-source imports, bounded training, validation, trained-weight ONNX parity and
local artifact backup have also been verified on the A100 host. See the measured
report for source counts and quality limits. Passing execution checks does not
meet the model's provisional accuracy targets.

## Host

Use Linux x86_64 with an NVIDIA GPU and persistent storage. A **24 GB GPU** is
the initial planning baseline; larger GPUs allow larger microbatches. Use at
least 16 GB host RAM, more for original high-resolution sources. CubiCasa alone
downloads 5.47 GB; prepared labels and checkpoints add storage. Inspect free space
before additional sources; a full-corpus storage total is not established.

The supplied image pins **PyTorch 2.5.1 / CUDA 12.4**. Confirm GPU architecture
and driver compatibility. Newer GPUs requiring newer CUDA/PyTorch need a separately
validated image rather than bypassing the CUDA probe.

An authorized remote A100 is configured. Machine access details stay in private
local operational notes, outside this source handoff.

## Obtain code

Transfer/checkout branch ml-training-preparation after publishing its code,
or transfer a source archive of that local branch. Do not copy the private
workspace, brain notes or laptop datasets. No automatic push or deployment is
part of preparation.

## Container setup and baseline data

Run from the repository root:

~~~bash
docker build -f ml/Dockerfile -t planproof-training .
docker run --rm --gpus all --shm-size=8g planproof-training doctor --require-gpu --smoke
~~~

The CUDA probe performs a model forward/backward pass with random weights and
does not download data or encoder weights.

Prepare the first real corpus on a mounted persistent volume:

~~~bash
mkdir -p /srv/planproof-data
docker run --rm --gpus all --shm-size=8g \
  -v /srv/planproof-data:/data/planproof \
  --entrypoint bash planproof-training \
  ml/scripts/gpu_prepare.sh /data/planproof
~~~

This downloads original CubiCasa5K on the GPU host, extracts it, converts labels,
preserves publisher split information, audits it, renders ground-truth inspection
overlays, and checks the baseline config. Review those overlays before a long run.
Review cross-split perceptual candidates; confirmed duplicate groups must be
joined before training. Never move individual tiles between splits.

Start with 100 updates to measure actual throughput and memory:

~~~bash
docker run --rm --gpus all --shm-size=8g \
  -v /srv/planproof-data:/data/planproof planproof-training \
  train --config ml/configs/baseline.yaml \
  --manifest /data/planproof/processed/manifest.jsonl \
  --run /data/planproof/runs/baseline --max-steps 100
~~~

The full schedule is retained. Continue the same run:

~~~bash
docker run --rm --gpus all --shm-size=8g \
  -v /srv/planproof-data:/data/planproof planproof-training \
  train --config ml/configs/baseline.yaml \
  --manifest /data/planproof/processed/manifest.jsonl \
  --run /data/planproof/runs/baseline \
  --resume /data/planproof/runs/baseline/last.pt
~~~

If memory requires a smaller microbatch, change micro_batch before the real run,
retaining effective_batch: 16. Resume refuses changed data/config/schema/world size.
Interrupted jobs resume the last completed checkpoint, not an incomplete update.

## Existing GPU Python environment

With compatible CUDA PyTorch and Python 3.11 or 3.12:

~~~bash
python -m pip install wheel==0.45.1 -r ml/requirements.lock
python -m pip install --no-build-isolation --no-deps -e ml
planproof-ml doctor --require-gpu --smoke
bash ml/scripts/gpu_prepare.sh /absolute/persistent/planproof-data
~~~

Multiple compatible GPUs use:

~~~bash
torchrun --standalone --nproc_per_node=2 --module planproof_ml train \
  --config ml/configs/baseline.yaml \
  --manifest /data/planproof/processed/manifest.jsonl \
  --run /data/planproof/runs/baseline-ddp
~~~

Effective batch must divide by micro_batch times WORLD_SIZE. The DDP path is
implemented; its first multi-GPU verification happens on that host.

## Calibrate, test, export

~~~bash
planproof-ml calibrate --checkpoint /data/planproof/runs/baseline/best.pt \
  --manifest /data/planproof/processed/manifest.jsonl \
  --out /data/planproof/runs/baseline/calibration.json

planproof-ml evaluate --checkpoint /data/planproof/runs/baseline/best.pt \
  --manifest /data/planproof/processed/manifest.jsonl --split test \
  --calibration /data/planproof/runs/baseline/calibration.json \
  --out /data/planproof/reports/baseline

planproof-ml export --checkpoint /data/planproof/runs/baseline/best.pt \
  --calibration /data/planproof/runs/baseline/calibration.json \
  --out /data/planproof/exports/baseline
~~~

Calibration selects validation thresholds; model scores are not calibrated
probabilities of correctness. Test results and code readiness remain separate.
These commands never use the Stoa challenge corpus.
