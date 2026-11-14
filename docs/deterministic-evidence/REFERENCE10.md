# Ten real labeled development plans

**Local only. Do not publish drawings, labels, predictions or review artifacts.**
The owner superseded the synthetic-dataset proposal with ten already-labeled
source validation plans. This is an iterative development set, not a new final
test. Small generated arrays remain useful only for unit-test contracts.

## What is already prepared

Under ignored `.artifacts/reference10-v1/` in this local worktree:

- Four CubiCasa5K validation drawings, including two stair-positive cases and
  lower/higher annotation counts within a fixed hash-ranked pool.
- Four ResPlan validation drawings, with equivalent diversity. These are the
  existing prepared renders of labeled source layouts, not new invented plans.
- Two FloorPlanCAD validation drawings: a stair-positive and an object-dense case.
- Unmodified prepared PNGs, original prepared NPZ targets, readable ground-truth
  JSON, class masks, valid-region masks, source records and label overlays.
- Frozen main-model predictions on all ten and CAD-specialist predictions on its
  two CAD cases. No weights, thresholds or calibration were changed.
- Offline `review.html`, `failure-report.json`, `failure-table.csv`, error overlays
  and a contact sheet. Deterministic inference is explicitly NOT RUN.

There are77 annotated room instances,108 doors,110 windows and8 stairs in the
supported annotation scopes. Walls have semantic masks, not physical-wall counts.
CAD walls/rooms are unannotated. Two ResPlan cases lack complete stair-instance
coverage; N/A is not an exact count of zero.

All ten come from existing project validation splits; publisher test records,
training/final-test groups and exact image/geometry hash collisions with those
splits were excluded. Selection was based on annotation metadata/counts, not model
performance. The original images, labels and weights remain unchanged.

## How to inspect failures

Open `.artifacts/reference10-v1/review.html` locally. Choose a plan and compare
the original with existing labels, main ML, CAD ML or their error overlays.
Red error boxes are missed labels; amber boxes are unmatched predictions at
IoU0.5. A correct total count can still have both misses and false positives.

The report records per-plan counts, matched objects, misses, false positives,
room-mask matching, matched-room pixel-area error and wall-mask IoU. Unknown or
unrun sources remain explicit. Room area error is conditional on matched rooms;
read it alongside room misses. These diagnostic metrics are not the model's
published AP benchmark or end-to-end question accuracy.

Manual labeling is not needed to begin the five-class comparison. Inspect label
quality and record disputed annotations separately before changing any source
labels. This pack does not provide complete OCR transcriptions, dimension-line
associations or validated physical scale. Those tasks need additional focused
annotations later; pixel masks alone do not establish square metres.

## Reproduce and rerun

The tools use NumPy, OpenCV, Pillow and SciPy. Contract tests use pytest. Existing
local verification used NumPy1.26.4, OpenCV-headless4.11.0.86, Pillow11.1.0,
SciPy1.15.2 and pytest8.3.5. The inference worker additionally uses the already
pinned ML runtime; the CPU report tool does not need PyTorch or GPU access.

Export from the existing prepared source corpus, not a new download:

```bash
python tools/prepare_reference10.py --root PREPARED_DATA_ROOT --out .artifacts/reference10-v1
```

The exporter refuses to overwrite an existing pack. It records source manifest
and target/image hashes and retains available attribution notices. Preserve the
selection so algorithms can be compared on identical drawings over time.

After running a method, write one `planproof-detection-v1` prediction document per
case to `.artifacts/reference10-v1/baselines/METHOD/CASE_ID.json`. Use the pack's
`source_id` as document ID, page0, original image dimensions and pixel-edge xyxy
coordinates. Keep object IDs unique. Overlay PNGs use the same case basename;
the report generates a separate error overlay from predictions and source pixels.

```bash
python tools/review_reference10.py --pack .artifacts/reference10-v1 --method main --method cad --method deterministic
python -m pytest tools/tests/test_reference10.py -q --basetemp .artifacts/pytest-reference10
```

Do not create empty prediction files for algorithms that have not run. Missing
files produce NOT RUN; a genuinely empty detector result is a different outcome.
Ground-truth NPZ files must never be used as classifier inputs. They are read only
by the evaluator after predictions exist. Keep later ML-assisted results separate
from independent deterministic predictions.

No GitHub push was made for this local work. Preserve the frozen ML baseline and
the project's sealed final evaluation while iterating on these ten drawings.
