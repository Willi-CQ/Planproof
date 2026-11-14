# Hackathon: first weights early, protect retries

The user has 33 hours from Saturday 01:00 to Sunday 10:00 submission. These are
planning targets, not measured training times or a promise of sufficient accuracy.
This supersedes the 40k-update research schedule as the initial launch path.

The connected training host has one A100 80GB. Run curriculum stages sequentially;
use CPU workers for source preparation while training. No DDP setup is required.
This prioritizes time to working weights with the prepared stack; no claim of
absolute speed versus RTX PRO 6000, which we have not benchmarked for our detector.
[Verda pricing](https://verda.com/pricing?currency=eur), checked 2026-11-14:
one A100 80GB EUR2.142/h; estimated two cards EUR4.284/h, eight hours EUR34.272.
Confirm the actual 2x console quote; storage/taxes are additional.

Start only with original CubiCasa. Do not delay first weights to import every
source. Preserve groups and held-out splits, audit and inspect real annotations.
The original publisher ZIP contains its train/val/test lists. For a host with
sufficient RAM, `prepare --workers 8` converts CubiCasa in bounded parallel
processes; the default is one worker. Its annotations use F1_scaled raster pixels,
not SVG viewport fitting. Review actual overlays before training.

## Short runs

configs/hackathon.yaml: ImageNet encoder, 512px, effective batch8, microbatch4,
2000 updates maximum, validation/checkpoints every100, fixed eight examples/source.
Tiny symbols may lose accuracy at 512px. This uses a full real corpus, not --limit.
hackathon-crops.yaml changes only crop probability .5 to .75 for a controlled
alternative. More crops are not assumed to improve results.

After preparation, measure the actual path:

~~~bash
CUDA_VISIBLE_DEVICES=0 planproof-ml train \
  --config ml/configs/hackathon.yaml \
  --manifest /data/planproof/processed/manifest.jsonl \
  --run /data/planproof/runs/fast-baseline --max-steps 100 --max-minutes 10
~~~

Review validation overlays, loss, throughput and memory. If batch4 does not fit,
change micro_batch before a fresh run. Resume rejects changed recipes. Warm the
encoder initialization once before launching simultaneous model initializations.
Continue for at most 60 additional minutes:

~~~bash
CUDA_VISIBLE_DEVICES=0 planproof-ml train \
  --config ml/configs/hackathon.yaml \
  --manifest /data/planproof/processed/manifest.jsonl \
  --run /data/planproof/runs/fast-baseline \
  --resume /data/planproof/runs/fast-baseline/last.pt --max-minutes 60
~~~

In another terminal, after the first run is healthy:

~~~bash
CUDA_VISIBLE_DEVICES=1 planproof-ml train \
  --config ml/configs/hackathon-crops.yaml \
  --manifest /data/planproof/processed/manifest.jsonl \
  --run /data/planproof/runs/fast-crops --max-minutes 60
~~~

Time caps finish the current update, validate and save: they may overrun by an
update, validation and checkpoint I/O. Training exit does not stop cloud billing.
Save artifacts and stop/release compute separately.

## Active supervision

~~~bash
planproof-ml status --run /data/planproof/runs/fast-baseline --hourly-rate 2.142
nvidia-smi
~~~

Check every 5-10 minutes while actively connected. status.json/training.jsonl
record loss, throughput, peak memory and latest validation. failure.json records
caught errors. status flags stale runs, recent failures and three validation
checks without >.002 improvement. ETA/cost omit idle billing, prior sessions,
storage and taxes. Inspect room merges, missing openings/stairs, empty predictions,
category mistakes and alignment in overlays; scores alone do not prove usefulness.
Stop flat or declining runs and keep best.pt for integration.

A graceful pause uses:

~~~bash
touch /data/planproof/runs/fast-baseline/STOP
~~~

It validates/checkpoints after the next update. Remove STOP before resume.
Nonfinite losses/gradients raise errors; retain earlier completed checkpoints.
Hard process kills can lose updates since the latest checkpoint.

The GPU host is connected and supervised over SSH. Live supervision requires an
active connected session. Time caps protect runs between checks, not account-wide
spending. Never send private SSH keys in chat. Checkpoints are saved before
validation, so a validation failure preserves the completed optimizer updates;
only successfully evaluated improvements replace best.pt.

## Targets relative to the user's clock

- Saturday 01:00-02:00: connect, source preparation/checks, throughput probe.
- Saturday 04:00-06:00: target first evaluated/exportable candidate in the app;
  dependent on import success and measured quality.
- Saturday daytime: targeted corrections, validation comparison, export winner.
- Saturday 18:00: target freeze of working weights. Keep the remaining time for
  integration, testing, demo and recovery.
- Sunday 10:00: final submission; retain a previously working model during retries.

Default-threshold export can unblock first integration. Do bounded calibration
later rather than blocking on an exhaustive sweep or full-corpus benchmark.
Test/Stoa questions never select thresholds or models.

## Continuing into other datasets

Use `train --init-weights /data/planproof/runs/previous/best.pt` for a new run
on a new immutable manifest and a transfer recipe. This loads compatible model
parameters, resets the optimizer/schedule/best score, and records the source
checkpoint SHA-256 and source dataset/config fingerprints. `--resume` retains
the strict same-manifest/config/world-size contract; the two options are exclusive.

`transfer-resplan.yaml` keeps 70% CubiCasa replay. The subsequent
`transfer-floorplancad.yaml` keeps 65% CubiCasa and 15% ResPlan replay. These are
initial bounded experiments, not measured claims of improvement. Run a short
probe and inspect matching validation cohorts before extending either stage.
Keep the previous best weights and reject material original-source regressions.

Freeze a baseline manifest next to its prepared files before adding sources.
Separate release partitions during FloorPlanCAD extraction and group related
filename prefixes conservatively to keep cropped drawing families together.
Audit new data against prior-stage training groups before selecting checkpoints.
Do not mix Stoa into any stage.

See [real-source validation](DATA_VALIDATION.md) before changing adapters or
assembling another curriculum. Diagnostic imports and the old pre-curation
manifests must not initialize production stages. Session --max-steps is an
absolute optimizer-step ceiling, including updates completed before --resume.

`refine-cubicasa.yaml` and `detail-cubicasa.yaml` are bounded fallback recipes,
using the clean baseline through --init-weights. The latter raises input size to
1024 pixels. They and the transfer recipes use paired GIoU box loss while keeping
the existing width/height/offset inference encoding. The original baseline keeps
its Smooth L1 loss and exact-resume config unchanged. Original-pixel opening boxes
had weak overlap even at ground-truth centers in the early baseline; image-sized
Smooth L1 errors underweighted thin-object proportions. GIoU directly supervises
overlap across object scales. This motivates an ablation, not a promised gain.
See the [GIoU paper](https://giou.stanford.edu/).

The measured 1024-pixel detail probe runs faster per update with effective batch4
and microbatch2 than the initial 512-pixel batch8 run. Transfer recipes therefore
use the detail resolution and box loss weight0.5. This balances localization
against structural masks; the earlier weight2 GIoU run is preserved as an
ablation. Raw room-mask IoU is reported separately from instance F1 so a good
foreground mask cannot hide failed room splitting.

Calibration uses one cached prediction per validation drawing, then independently
searches wall thresholds, room-mask/room-center thresholds, and opening-center
thresholds. A final ordinary evaluation confirms their combined result. It never
uses test data and the CLI refuses a different training-manifest fingerprint.
Candidate settings, source cohorts and inference-pass counts are retained.
Thresholds describe an operating point, not correctness probabilities.

`balanced-symbol-refine.yaml` is a bounded rare-class experiment. It averages
center losses per covered class and GIoU per represented object class, so a
stair target is not divided by the count of unrelated doors/windows. Legacy
recipes retain their original reductions. It samples 50% CubiCasa, 10% ResPlan,
40% CAD at low learning rates and validates 32 examples/source every250 updates,
including CAD stairs absent from the original eight-example preview. Start with
250 updates, inspect per-class changes and regressions, then decide whether to
continue. Keep the validated ResPlan checkpoint as the fallback.

`structural-refine.yaml` independently refines the validated model with 80% real
CubiCasa and 20% ResPlan, avoiding inheritance of CAD regressions.
`scale-refine.yaml` tests a small log-width/height loss alongside GIoU, with
60/10/30% replay. Stair diagnostics showed predicted boxes contained inside much
larger targets, where overlap gradients are weak. The relative-size term is
scale invariant and does not change inference encoding. Its default weight is
zero, preserving exact-resume behavior for all earlier recipes. Both recipes
start with a bounded probe and retain the previous validated checkpoint.
