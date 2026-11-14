# Shared evidence contracts

Status: the local CPU baseline implements `analyze` and `get_region`, PNG/PDF
frames, evidence validation, source-bound ML imports and candidate comparisons.
See [implementation status and limits](IMPLEMENTATION.md). This remains the
target contract for Track 1 integration; dense maps and automatic dimension
endpoint verification are not implemented.

## Coordinate and identity invariants

- The canonical frame is the exact decoded PNG image or the recorded PDF page
  render: pixel edges, origin top-left, x right, y down. Preserve original PNG
  dimensions; internal downsampling/tiling always records its inverse transform.
- Boxes are `[x0, y0, x1, y1]`, with `0 <= x0 < x1 <= width` and analogous y.
  Crops use exclusive right/bottom edges. Preserve floating-point coordinates
  until raster drawing; no hidden inclusive/exclusive conversions.
- PDF pages have separate sizes, CropBox/rotation metadata and 3x3 homogeneous
  source-to-canonical transforms and inverses. A detail render has its own
  crop-to-canonical transform. Never reuse the last page's dimensions for all pages.
- `document_id` is derived from source-file SHA256; `page_index` is zero-based.
  `source_sha256` identifies original file bytes. `raster_sha256` identifies decoded
  RGB pixel bytes, with dimensions/color convention recorded. They are different
  concepts, even when both are hashes. Preserve this binding through ML execution.
- IDs are deterministic within a document/page/config version and namespaced by
  method. Preserve source IDs after fitting or merging through `dependency_ids`.
  Sort outputs stably; record seed/configuration for stochastic numerical helpers.
- Polygon objects have an outer ring and zero or more holes. Do not drop holes,
  disconnected components or inner boundaries to obtain a prettier contour.
- Lengths/areas are pixels/pixels squared until a supported calibration exists.
  Geometry, frames, quantities and JSON numbers must be finite; JSON uses no NaN.

## Minimum shared objects

Implement typed records plus JSON serializers/validators for these concepts:

| Object | Required content |
| --- | --- |
| `PageFrame` | Document/page IDs, source/raster hashes, image size, source/crop transforms, page metadata |
| `Observation` | ID, kind, source family, independent/hybrid mode, geometry/frame, method/config version, dependency IDs, warnings |
| `PixelEvidence` | Frame, artifact references for observed/processed masks, raw segments, contour rings, text spans, symbol candidates, availability and coverage |
| `VectorEvidence` | Frame reference, fitted primitives, wall/junction graph, region polygons, opening associations, measurements and topology diagnostics |
| `Comparison` | Source IDs, relation, supporting geometry, disagreement region, warnings and crop reference |
| `Aggregate` | Scope, class and counting unit, member IDs, candidate duplicate groups, value or unresolved status, coverage limits |
| `EvidenceDocument` | Schema version, pages, all observations/comparisons/aggregates, calibration records and artifact manifest |

An artifact reference contains a relative path under the job directory, SHA256,
dimensions/frame and artifact kind. Dense masks/maps stay in local files, not
large nested JSON arrays. Do not accept paths outside the job root.

`source_family` is one of `native_pdf`, `classical_pixels`, `ocr`, `ml`, `hybrid`.
`mode` is `independent` or `hybrid`; it describes dependency on the floorplan ML
proposals, not statistical independence from every recognition model. OCR is
recognition evidence and must never be described as exact native source text.
Numeric model/OCR scores retain named semantics. Do not create an overall
correctness confidence by averaging unrelated scores or assigning a constant.

## Two public operations

```python
analyze(input_path, out_dir, *, ml_predictions=None, ml_maps=None, config=None)
get_region(evidence_path, page_index, bbox, out_dir, *, padding_px=0)
```

`analyze` returns an `EvidenceDocument` and writes `evidence.json`, numbered
pixel/vector overlays and a local artifact manifest. It may report partial
extraction if an optional source is unavailable. Invalid input, identity/frame
mismatch or malformed geometry produces an explicit error; do not silently
rescale unrelated ML results to fit the page.

`get_region` returns a clean image crop, intersecting observation IDs and a
crop-to-canonical transform. The requested box is in canonical pixels; clip the
padded box to the page and return the actual clipped box. Crop content is never
replaced by an annotated image; provide an optional separate overlay.

Executable local CLI equivalents after installing `deterministic/`:

```bash
planproof-evidence analyze page.png --out .artifacts/evidence/page
planproof-evidence analyze drawing.pdf --out .artifacts/evidence/drawing
planproof-evidence analyze page.png --ml-predictions .artifacts/ml-binding.json --out .artifacts/evidence/page
planproof-evidence crop .artifacts/evidence/page/evidence.json --page 0 --bbox 100 200 600 700 --out .artifacts/crops/region
```

The ML binding document identifies model/profile SHA256, source raster identity,
prediction JSON and any map references. Different model outputs are a list of
separately identified sources, not one concatenated instance list. Track 1 owns
validation and preserves original prediction IDs. Track 2 consumes normalized
observations without importing PyTorch.

## Relationship to the existing ML JSON

Input schema is `planproof-detection-v1`; it contains `document_id`, `page_index`,
`image_size`, `coordinate_system`, `model_version`, `inference_mode`, thresholds
and `instances`. Instances have `id`, `class`, `bbox`, `confidence` and optional
polygons with holes and room categories. The adapter must preserve these fields
and add provenance through the binding; do not change the frozen ML schema.

The binding must establish the same raster, page and transform. Matching width
and height alone does not prove two images are the same. If launching ML, set
the document ID/page explicitly and record the input raster hash. Imported
predictions need equivalent trusted binding; otherwise report them as unbound
and exclude them from spatial agreement/count claims.

Main dense probability maps can have working-image rather than original-image
resolution. Read their recorded sizes and transform before sampling. Do not
index them using original pixels directly. CAD dense maps are unsupported.

## Compact illustrative observation

This is generated example data, not a result on a Stoa drawing. The full document
also contains the frame, source observations and artifact references it names.

```json
{
  "schema_version": "planproof-evidence-v1",
  "observation": {
    "id": "p0-hybrid-door-0001",
    "kind": "door_candidate",
    "source_family": "hybrid",
    "mode": "hybrid",
    "page_index": 0,
    "frame_id": "p0-canonical",
    "bbox": [100, 200, 160, 270],
    "method": "arc-gap-refinement-v1",
    "dependency_ids": ["ml-main:p0-door-0002", "pixel:arc-0010", "pixel:line-0021"],
    "warnings": ["classification remains a candidate"],
    "crop_id": "p0-region-0001"
  }
}
```

An independent counterpart may use the same visible arc/line but must have no ML
parents or ML-driven search restriction. Searching only where ML found objects
cannot establish independent full-page recall.

## Topology and measurement rules

- Keep raw observations immutable. Store fitted/snapped geometry separately,
  retaining parents and numerical residuals or pixel-support summaries.
- Record inferred bridges separately from observed edges. A room dependent on a
  repaired gap is a hypothesis, not an observed fully closed region.
- Record outer rings, holes, dangles, cuts and invalid rings. Preserve a failed
  room candidate and reason; do not force closure or discard contradictory ink.
- A dimension observation links raw text, parsed alternatives, unit evidence,
  dimension/extension endpoints and the region it measures. Bare `2400`, `2.40`
  or `4` cannot be assigned metres merely from plausible magnitude or formatting.
- A calibration record names its page/view region, unit, source references,
  transform/factor and consistency checks. Distinct detail views may have distinct
  scales. Native printed scale alone may be stale after resizing.
- Explicit dimension text can be reported without calibrating the whole drawing.
  Computed physical geometry needs a compatible supported calibration. Missing or
  contradictory calibration keeps derived physical values unresolved.
- Printed area and calculated area are separate observations. Keep net-room,
  centerline-region and gross-area definitions separate. Retain original unit
  alongside normalized values; area conversion uses the squared length factor.

## Matching, counts and disagreement

Use spatial indexing for candidate matches. Compare only compatible classes,
frames, scopes and quantity definitions; preserve matching metrics and thresholds
in the versioned configuration. Avoid interpreting IoU alone as correctness.

Relations are `agreement`, `class_conflict`, `geometry_conflict`,
`measurement_conflict`, `unmatched_ml` or `unmatched_geometry`. A disabled or
unavailable detector is availability metadata, not a negative observation.
One source missing an object is not automatically a contradiction.

Aggregate membership must be explicit. Deduplicate one physical candidate seen
by multiple methods; retain unresolved duplicate groups as alternatives rather
than taking the union or adding source counts. Preserve each source's count too.
Do not report exact count if coverage or duplicate resolution does not support it.

Define what is counted: door assembly versus leaves, stair flight versus treads,
semantic room versus enclosed region. Wall connected components or graph edges
are not counts of physical walls. These distinctions cannot be repaired by
rounding or a model-confidence threshold.

## Caching and failure visibility

Cache extraction by source hash, page, renderer/frame policy, algorithm version,
configuration and OCR engine/language versions. Cache hybrid comparisons by the
additional model/profile/prediction/map identities. Changing ML must not invalidate
independent extraction unnecessarily. Repeated runs must not retain stale crops.

Differentiate successful empty output, unavailable source, failed extraction and
unsupported interpretation. Include partial page/view coverage and diagnostics
in JSON and overlays. No network model downloads or external inference calls
should occur implicitly during CPU extraction.
