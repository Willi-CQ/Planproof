# Floor-plan detection training

Start with [GPU_READY.md](GPU_READY.md) when attaching the GPU. This package is a
standalone implementation on branch **ml-training-preparation**. It does not wire
the deterministic packages or call Gemini.

## Model and output

The first detector uses a U-Net/ResNet-34 shared encoder/decoder, initialized
from ImageNet for real training. Five mask channels describe wall body, room
interior, door, window and stairs. Auxiliary heads predict room boundaries,
four instance-center heatmaps, per-class box width/height/offsets, and ten room
category logits. Box supervision allows CAD instances to train localization
without inventing filled masks.

Room categories are kitchen, living, dining, bedroom, bathroom, hallway, storage,
garage, balcony and other. Unknown source categories remain unsupervised;
unknown predictions remain unknown. Elevators/escalators are not stairs.

Inference combines a global view and overlapping detailed tiles. It returns
original-page pixel coordinates with origin at the top left and xyxy boxes:
document id, page index, image size, schema/model version, instance id, class,
confidence, optional room category, and polygons with holes. Raw probability
maps can be saved separately. Probabilities and model predictions stay separate
from later deterministic evidence. The input image is never overwritten.

~~~bash
planproof-ml infer --checkpoint /data/runs/baseline/best.pt \
  --image /data/input/page.png --document-id drawing-123 --page-index 0 \
  --calibration /data/runs/baseline/calibration.json \
  --save-probabilities --out /data/results/page
~~~

PDF pages must first be rendered by the existing PDF reader. This package does
not infer metric units, answer questions, perform OCR, or reconstruct a BIM.

## Dataset roles and acquisition

The original-source catalog is available without network calls:

~~~bash
planproof-ml sources
~~~

| Source | Role | Reader/acquisition behavior |
| --- | --- | --- |
| CubiCasa5K | Primary real drawings | Original model.svg and paired PNG; publisher train/val/test lists downloaded separately |
| FloorPlanCAD | CAD object instances | Original semanticId/instanceId SVG; explicit release-specific map; black-stroke render when paired PNG is absent |
| CVC-FP | External test only in v1 | Named SVG structural annotations and paired raster; publisher endpoint must be reachable |
| ResPlan | Synthetic geometry/room types | Original geometry pickle plus publisher split; explicit trust flag required |
| ModRes | Synthetic cleaned geometry | Annotated GeoJSON, geometry dictionaries, or recognized indexed-face JSON; incompatible TopologicPy exports must be converted to the interchange contract |
| RPLAN | Synthetic room/layout input | Original RGBA boundary/type/instance/interior channels; independent monochrome input render |
| SESYD | Symbol augmentation | XML boxes or named SVG annotations; template group must be recoverable, otherwise use explicit interchange grouping |
| HouseExpo | Weak room-category supervision | Original JSON vertices and category boxes; no invented dense room masks |
| Structured3D | Derived room-plan supervision | Original scene annotation_3d.json incidence matrices; no invented wall/opening labels |
| ROBIN / ZInD | Future auxiliary tasks | Cataloged; not part of the v1 structural training loader |

Readers are tested against generated schema fixtures. **Real archive imports are
not claimed verified locally.** GPU preparation must pass deep audit and visual
inspection. Unknown/mismatched schemas stop preparation rather than silently
emitting empty labels. Access-dependent sources remain access-dependent.

Examples on the GPU machine:

~~~bash
planproof-ml fetch --source floorplancad --data-root /data --extract
planproof-ml prepare --source floorplancad --raw /data/extracted/floorplancad \
  --class-map ml/configs/floorplancad-35.json --out /data/processed

planproof-ml fetch --source modres --data-root /data --extract
planproof-ml prepare --source modres --raw /data/extracted/modres --out /data/processed

planproof-ml fetch --source resplan --data-root /data --extract
planproof-ml prepare --source resplan --raw /data/extracted/resplan \
  --trust-pickle --out /data/processed

planproof-ml fetch --source houseexpo --data-root /data --extract
planproof-ml prepare --source houseexpo --raw /data/extracted/houseexpo --out /data/processed
~~~

For a source with a publisher-provided access URL, supply it explicitly with
fetch --url, then extract and prepare. No access agreement, credential, or download
link is fabricated. Download receipts retain URLs, byte counts, hashes and source
access notes. Known publisher checksums are checked; incomplete downloads do not
become usable files.

### Annotation interchange for release variants

A source exporter can write one **.annotation.json** file per image:

~~~json
{
  "id": "drawing-1",
  "image": "drawing-1.png",
  "group_id": "template:01",
  "complete": ["wall", "room"],
  "instance_complete": ["door", "window"],
  "features": [
    {"class": "wall", "outer": [[1,1],[20,1],[20,4],[1,4]], "holes": []},
    {"class": "room", "room_type": "kitchen", "outer": [[4,4],[18,4],[18,18],[4,18]]},
    {"class": "door", "box_only": true, "bbox": [8,1,12,5]}
  ]
}
~~~

Completeness is a declaration from annotation knowledge, not a list of present
objects. It enables negative supervision only where a class is fully annotated.
Otherwise only known positives are supervised. Source images must remain within
the raw source directory. SESYD requires a template group, not an image-level group.
The normalized adapter is supported for SESYD and release-specific exporters.

### Prepared representation

Preparation stores paired PNG/NPZ samples and a JSONL manifest under the private
data root. Targets include masks, per-instance ids, coverage masks, room types,
object boxes, weak category boxes and explicitly ignored pixels. The manifest
records source/family/original group, publisher/project split, input/target hashes,
annotation provenance and geometry fingerprints. No training artifact belongs in Git.

Run all source preparations before training; imports update the combined manifest.
Exact decoded-image and normalized-geometry duplicates are grouped across sources.
Publisher test/validation membership takes precedence. Sources with only a train
and test split get a deterministic group-level validation subset.
Reassignments are recorded: this clean project split is not presented as an
unchanged publisher benchmark.

~~~bash
planproof-ml inspect --manifest /data/processed/manifest.jsonl \
  --per-source 12 --out /data/inspection
planproof-ml audit --manifest /data/processed/manifest.jsonl \
  --deep --near --out /data/audit.json
~~~

Review near-duplicate candidates rather than treating visual similarity as proof.
Apply confirmed duplicate aliases with a JSON mapping from record ids to shared
group names:

~~~bash
planproof-ml regroup --manifest /data/processed/manifest.jsonl \
  --mapping /data/reviewed-groups.json
~~~

Whole existing groups move together. A new fingerprint requires a new training run.
Augmentations and crops happen only after the split. Stoa's challenge package is
rejected, while independently downloaded original CubiCasa5K remains eligible.
Diagnostic --limit imports must be replaced with full imports before production training.

## Recipes and reproducibility

Both control and curriculum runs have a 40,000-update cap and effective batch 16.
The baseline uses only CubiCasa5K. Curriculum stages are:

| Stage | Updates | Sampling |
| --- | ---: | --- |
| Synthetic preparation | 8,000 | ModRes 40%, RPLAN 40%, SESYD 20% |
| Mixed | 24,000 | CubiCasa 60%, FloorPlanCAD 25%, ModRes 6%, RPLAN 6%, SESYD 3% |
| Real refinement | 8,000 | CubiCasa 75%, FloorPlanCAD 25%; learning rate scale 0.1 |

ResPlan may replace ModRes via an explicit config, not by duplicating its source plans.
Missing required sources stop preflight; they are never silently substituted.
Optional HouseExpo and Structured3D ablation configs change one source family at a time.

AdamW starts with encoder/head learning rates 1e-4/3e-4, weight decay .01,
per-phase warmup/cosine scheduling, mixed precision on CUDA, accumulation and
gradient clipping. Half of training views are full plans and half detailed crops.
Transforms preserve targets and ignore padding/clipped center supervision.
Input-only noise includes quarter rotations, reflection, contrast/noise,
grayscale, blur, JPEG and conservative ink thickening.

The combined loss uses masked BCE+Dice, room-category CE, weighted boundary BCE,
center focal loss, supervised box regression and weak region-category CE.
Unavailable labels contribute no negative loss.

Logs retain configuration/schema/data fingerprints, source sampling recipes,
software/commit, optimizer/scaler/RNG state, losses and validation reports.
Resume refuses incompatible data/config/world size. CPU tests check exact resume;
cross-device GPU bitwise equality is not promised.

## Evaluation and model selection

Every 1,000 updates, use a fixed global-view validation preview to select checkpoints.
After training, calibrate thresholds on the full validation set using tiled inference,
then run final validation/test reports in that same inference mode. The test set
never selects thresholds or checkpoints.

Selection score is 0.4 wall IoU + 0.3 object mAP50 + 0.3 room-instance F1.
Reports include per-source raw/postprocessed wall IoU, wall boundary F1,
room-instance F1 at mask IoU .5, object AP50/AP50:95, and conditional room-category
macro F1 with support counts. Absent ground-truth classes are null, not perfect scores.
AP uses documented 101-point interpolation on project boxes; it is not claimed
to reproduce a publisher's CAD-primitive PQ or COCO evaluator.

Provisional targets are wall IoU >= .80, room F1 >= .80 and object mAP50 >= .70.
These are experiment acceptance targets, not measured quality or GPU readiness.
All three object classes must have held-out support before numerical acceptance can pass.
See [RESEARCH.md](RESEARCH.md) for scores and protocol limitations.

Compare full validation reports for the control and each curriculum/optional-source run:

~~~bash
planproof-ml compare --baseline /data/reports/baseline/val-metrics.json \
  --candidate /data/reports/curriculum/val-metrics.json \
  --out /data/reports/selection.json
~~~

The command requires identical common-source validation cohorts and inference protocol.
It selects a candidate only above .005 aggregate improvement with no per-source
component regression above .02. It refuses test reports. Evaluate held-out test data
after choosing the recipe using validation.

## Tests and exports

~~~bash
pytest ml/tests -q
planproof-ml doctor --smoke
planproof-ml smoke --device cpu
~~~

The final command trains, resumes, evaluates and checks ONNX parity using generated
fixtures in a temporary directory. It downloads no datasets or pretrained weights
and makes no real-data accuracy claim. Supply an ignored --out directory to retain artifacts.

Export includes the full detector, wall-only logits and metadata with schema,
normalization, thresholds, hashes, training step and dataset fingerprint.
Keep weights and exports in the private artifact store, not GitHub.
