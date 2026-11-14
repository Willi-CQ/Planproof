# Local 2D evidence engine

CPU-only PNG/PDF geometry and independent-versus-ML cross-checks. The default
pixel backend now runs the teammate's pinned GitHub extraction routines through
a validated adapter. See the
[integration and interactive Evidence Lab](../docs/deterministic-evidence/INTEGRATION.md).
The older reference backend and its [measured results](../docs/deterministic-evidence/IMPLEMENTATION.md)
remain documented. Classical recognition is not a reliable replacement for ML.

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e './deterministic[test]'
.venv/Scripts/planproof-evidence analyze drawing.png --out .artifacts/example
.venv/Scripts/planproof-evidence analyze drawing.pdf --out .artifacts/pdf
.venv/Scripts/planproof-evidence crop .artifacts/example/evidence.json --page 0 --bbox 100 200 600 700 --out .artifacts/crop
.venv/Scripts/python -m pytest deterministic/tests tools/tests -q
```

On Linux use `.venv/bin/` instead. Python 3.11+; validated locally with Python
3.12. No Torch, OCR service, GPU, cloud account, dataset or model download is
required for analysis. The repository root README stays empty.

```python
from planproof_evidence import analyze, get_region

evidence = analyze('drawing.png', '.artifacts/job')
crop = get_region('.artifacts/job/evidence.json', 0, [100, 200, 600, 700],
                  '.artifacts/crop', padding_px=20)
```

Outputs include `evidence.json`, exact canonical rasters, numbered class overlays,
wall-vector/gap overlays, ink/wall/skeleton masks with transforms, raw segments,
paired boundaries, medial thickness estimates, noded graphs, room-region
diagnostics, opening attachments, unknown pattern records and pixel quantities.
Holes and disconnected components survive. Gap hypotheses remain separate from
observed walls. The crop API returns clean source pixels and the actual crop
transform; it verifies the stored source raster checksum before reading it.

Counts are explicit memberships of candidate symbols/regions. Physical wall,
door-assembly, stair-flight and exact full-plan counts remain unresolved. Numbers
called `support` measure the named rule feature; they are not calibrated
correctness probabilities. Room categories require contained text. No implicit
metres, square metres, typical-door scale or printed-scale fallback is used.

## Import the frozen detector

Pass `--ml-predictions binding.json`. A binding is an explicit trusted link
between an existing prediction and the exact decoded RGB raster:

```json
{
  "sources": [{
    "name": "main",
    "page_index": 0,
    "image_size": [1200, 800],
    "document_id": "the-original-ml-document-id",
    "raster_sha256": "64-lowercase-hex-characters",
    "prediction_path": "predictions/page-0.json",
    "prediction_sha256": "64-lowercase-hex-characters",
    "model_sha256": "64-lowercase-hex-characters",
    "profile_sha256": "64-lowercase-hex-characters"
  }]
}
```

The strings above are placeholders, not real hashes. Raster hashing is SHA256 of
`str(rgb_uint8.shape).encode() + rgb_uint8.tobytes()`. Paths are resolved against
the binding file; absolute paths are supported for trusted local imports. The
caller must supply a trustworthy model/profile binding: the CPU importer does
not load weights to independently attest those two hashes. Prediction bytes,
frame, page, raster hash, sizes, classes, finite scores and rings are checked.
Multiple model sources stay separate. Object-only CAD classes stay unavailable
for wall/room quantities. Source ML rings with self-touching topology are repaired
explicitly; originals, reasons and normalized geometry remain separately visible.
Malformed/out-of-frame rings are rejected.

The engine compares independent observations to imported ML and records spatial
agreement, class/geometry conflict and unmatched proposals. It also produces a
separate hybrid pixel check and ML-mask-derived wall thickness/graph. A hybrid
descendant cannot independently confirm its ML parent. No union count or claimed
accuracy improvement is manufactured. Dense probability maps are explicitly
unsupported; passing `ml_maps` raises an error rather than guessing a transform.

## Track 1 adapter boundary

`analyze(..., pixel_adapter=adapter)` replaces the reference pixel front end.
The callable receives only `(original_rgb_uint8, Config)` and returns raw
primitives. Track 2 calls `pixels.vectorize_pixels` on those primitives; Person 1
does not implement line fitting, wall pairs, topology or classifiers. Its source
must be independent of floorplan ML and evaluation labels. In particular:

- required: `working_size=[W,H]`, `working_to_canonical=3x3`, binary uint8
  `ink`, `wall_mask` arrays and `raw_segments` in that working frame;
- optional: measured `stroke_px`, independent candidate records from
  `pixels.candidate`, availability, warnings, OCR `text_spans` and
  unclassified-pattern diagnostics. Full-page diagonal scaling is the supported
  adapter frame; normalize any tile/rotation transforms before returning;
- candidate polygons must be valid and in-frame; support must name the measured
  feature; `class` must be wall/room/door/window/stairs;
- OCR availability must be explicit. OCR records must retain `ocr` provenance;
  native PDF text is separately recorded as `native_pdf`.

The default path caches extraction using the raster hash, config and extraction
source hashes. Model/profile/prediction changes do not invalidate independent
pixels; hybrid comparisons are recomputed. Custom adapters bypass cache reads.
There is no evaluation-label argument or import in this package.

## Measurements and native PDF limits

`measurements.parse_dimension` retains missing-unit and locale ambiguity.
`associate_dimension` returns nearby line alternatives requiring endpoint/extension
verification. `calibrate` requires at least two distinct verified explicit-unit
anchors from the same scope and rejects inconsistent scales. `quantities` squares
the length factor for area. These are tested utilities; the default pipeline does
not automatically verify dimension endpoints, so page calibration remains
unresolved. Printed areas are not substituted for calculated polygon areas.

PDFium renders each page within a pixel budget. A reversible matrix includes
CropBox/rotation. Straight native paths and form transforms, text spans and native
line topology are retained separately. Bezier controls are reported as skipped;
raster extraction still sees curves. Native paths may include furniture and
dimensions and are not automatically physical walls. The public renderer API
is described in the [PDFium wrapper documentation](https://pypdfium2.readthedocs.io/en/stable/python_api.html).

## Repeat the ten-plan check

For ML-assisted room boundaries, wall roles, opening rules and clean per-room
context exports, use the optional [room-boundary postprocessor](../docs/deterministic-evidence/ROOM_BOUNDARIES.md).
Its primary room geometry preserves ML by default; deterministic refinements
remain inspectable alternatives. The guide includes the real-plan feedback loop.

The already labeled development pack and frozen ML outputs are local artifacts,
never tracked datasets. Keep the original frozen pack unchanged:

```powershell
.venv/Scripts/python tools/run_reference10_geometry.py --pack .artifacts/reference10-v1 --out .artifacts/reference10-track2-v6
.venv/Scripts/python tools/review_reference10.py --pack .artifacts/reference10-track2-v6 --method main --method cad --method deterministic
```

Open the derived pack's `review.html`. Inspect missed labels and false positives,
vector gaps and evidence JSON alongside the unchanged source. CAD wall/room
coverage and incomplete stair labels stay N/A. Labels are loaded only by the
separate evaluator. This small stair-enriched, repeatedly inspected development
cohort cannot establish final Stoa/Gemini answer accuracy.
