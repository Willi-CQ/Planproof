# Track 1: pixel evidence, OCR and local refinement

Owner: **teammate**. Start with [the architecture](README.md) and
[shared contracts](CONTRACTS.md). Track 2 owns the harder vector/topology work;
you own a complete extraction path, tests and visual artifacts, not just inputs.

## Deliverable

A source-preserving `PixelEvidence` producer for PNG/PDF that works without ML,
plus a separate ML-assisted local refinement pass. Export raw primitives, text,
candidate regions, masks, transforms, crops and availability diagnostics.

Own `deterministic/` package/CLI setup, `frames`, `raster`, `pixels`, `ocr`,
`ml_adapter` and `crops` modules plus corresponding tests. Track 2 owns the
`contracts.py` initial fixture/types implementation after joint agreement; review
the frame serialization together before either track depends on it.

## Implementation order

1. Build a minimal installable CPU package and CLI. Load a PNG without modifying
   the source; normalize color/alpha handling explicitly and retain original
   dimensions. Render each PDF page using pypdfium2 with bounded overview pixels.
   Expose per-page/source/detail-crop transforms and record EXIF/orientation policy.
2. Return one generated fixture through `PixelEvidence` immediately. Track 2 uses
   this and generated primitives while real extraction develops. Do not wait for
   OCR or ML to deliver the first valid record.
3. Extract grayscale/ink, edges, components, contour rings and raw segments.
   Compare Otsu/adaptive thresholding for filled and paired-outline styles.
   Use LSD with Canny/Hough fallback as a starting algorithm, not a universal
   optimum. Keep thresholded and repaired masks distinct from observed pixels.
4. Preserve disconnected components and holes. Estimate local stroke size so
   kernels/tolerances scale with resolution. Tile large pages with overlap and
   restore all results to canonical pixels; mark coverage and deduplicate overlap.
   Keep slanted/curved evidence rather than forcing axis alignment.
5. Extract OCR spans with text, box/orientation, language/engine and recognition
   score semantics. Track 2 adds native PDF text/vector extraction and merges
   duplicate text sources by provenance. Preserve raw dimension strings; Track 2
   parses/associates them. Missing OCR/language packs is an explicit partial result.
6. Propose visible opening/stair regions using arcs, leaves, parallel lines and
   tread profiles. Store evidence and alternative interpretations. Pixel cues
   alone do not prove semantic class; do not identify elevators as stairs.
7. Adapt frozen ML JSON/maps with explicit image/model binding. For ML-assisted
   refinement, inspect padded ROIs in the original pixels, preserving the original
   proposal and every change. A missing pixel-supported component is unresolved;
   do not return the ML polygon as if pixels confirmed it.
8. Write clean crops, optional separate overlays and cacheable artifact references.
   The overlay numbering must match observation IDs used by Track 2's final JSON.

## ConstructQ ideas to adapt selectively

These paths are mirrored references under this branch's
`geometry/planproof_geometry/src/planproof_geometry/ai/`:

| Module | Useful idea | Required adaptation |
| --- | --- | --- |
| `plan_extract_raster.py` | LSD/Hough, OCR and image extraction | Keep PNG pixels; remove invented DPI/physical scale |
| `v3/pixel_bim/refine/classical.py` | Local thresholding and morphology | Preserve holes/components; no label-as-truth fallback |
| `v3/pixel_bim/refine/wall_dispatcher.py` | Filled versus paired-outline processing | Adapt local style and support; no whole BIM dispatcher import |
| `v3/pixel_bim/refine/ml_gravity.py` | Soft search/support region around proposals | Hybrid provenance; no old model loader or forced polygon confirmation |
| `v3/pixel_bim/stairs/stair_pixel_extractor.py` | Visible tread/profile/PCA cues | No DIN-expected tread count or fabricated dimensions |

Some existing helpers operate on whole-page masks or use fixed pixel/metre
thresholds. Reimplement compact pure functions where needed; do not copy a large
pipeline merely to keep its function names. Retain attribution for adapted code.

## Track boundary

You hand over observed/processed mask references, raw segments, contours/rings,
text spans, candidate boxes, frames and source dependencies. Track 2 owns fitted
vectors, merging, graph topology, physical calibration, final count semantics
and source comparison. Do not silently repair topology or assign metric units
inside the pixel extractor.

For local refinement, return its parents and modified region. Track 2 must be
able to trace a hybrid vector all the way back to the original ML observation.
Both paths must coexist; the hybrid output never overwrites independent results.

## Required checks

- Source PNG/PDF hashes unchanged; alpha, rotated pages, CropBox, differently sized
  pages, boundary-touching boxes and exclusive crop edges handled correctly.
- Canonical/crop transforms round-trip within 0.5 pixel; overlay alignment within
  one canonical pixel on controlled fixtures, including downsampled/tiled cases.
- Filled/outlined walls, disconnected pieces, courtyards/holes, faint strokes,
  text touching lines and slanted walls remain visible in suitable primitives.
- Tiling does not duplicate candidates or omit seams. Memory use is bounded and
  measured; do not create a full-resolution mask per object unnecessarily.
- OCR missing/failure differs from genuinely empty text; multilingual raw text
  survives JSON serialization. Rotated dimensions retain orientation evidence.
- Independent extraction is identical with ML absent or present. Only the separate
  hybrid pass changes when ML inputs change; malformed/unbound maps are rejected.
- Clean crops contain original source content, with an accurate page transform.
- Record cold/warm CPU time, memory and cache invalidation behavior on the chosen
  development examples. No ConstructQ timing claims become Planproof results.

Complete your own unit/integration tests and one visual inspection before handing
off. Do not leave coordinate validation for the other person to discover at merge.
