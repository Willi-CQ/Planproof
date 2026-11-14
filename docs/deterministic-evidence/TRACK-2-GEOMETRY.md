# Track 2: vector geometry, measurements and ML cross-checking

Owner: **project owner/Codex pair**. This is the harder workstream. Start with
[the architecture](README.md) and [shared contracts](CONTRACTS.md).

Your result must make quantities and disagreements defensible. A clean-looking
polygon is not enough: connectivity, units, object membership and evidence lineage
must also be correct. Existing `geometry/` supplies references, not trusted answers.

## Deliverable and owned modules

Own `contracts.py` after joint agreement, the first generated handoff fixture,
and `vectors`, `topology`, `native_pdf`, `measurements`, `compare`, `aggregate`
and `evidence` modules with their tests under the new standalone package.
Do not take over Track 1's renderer/OCR/crop implementation or import the former
ConstructQ application to make tests work.

Return `VectorEvidence`, source-specific quantities, comparison records and the
final `EvidenceDocument`/JSON. Emit vector overlay primitives that Track 1's
renderer can draw with stable observation numbering.

## 1. Start without waiting for pixel extraction

Implement a generated fixture with known wall boundaries, one hole, an opening,
text/dimension boxes and an intentionally incorrect ML proposal. Publish its
`PixelEvidence` representation immediately after the shared contract kickoff.
It is a diagnostic fixture, not a dataset or an accuracy benchmark.

Implement native PDF extraction using pdfplumber behind the same frame contract.
Preserve characters/text, lines, curves, fills and stroke attributes with source
IDs. Convert PDF primitives using the recorded page transform; do not invent a
second raster frame. On scanned PDFs this source can be empty without failure.
PNG support must not depend on any native PDF vectors.

## 2. Fit geometry while preserving visual support

Use contour rings for outer/inner boundaries. For filled wall-mask centerlines,
start with skeletonization, distance-transform thickness, Hough segments and
collinear merging. Keep source-mask identity explicit: classical-mask vectors
and ML-mask vectors belong to different evidence paths.

Record pixel fit/support and parent primitive IDs. Distance-transform thickness
is meaningful only for an appropriate filled body, not arbitrary thin ink.
For paired outlines, fit line pairs and retain their actual boundaries; reject
pairings with insufficient overlap/support or inconsistent perpendicular distance.
Evaluate centerlines and boundaries separately.

Normalize/nodify segments and build junction/path connectivity; use skeleton
path evidence to resolve missing/ambiguous junctions rather than treating every
Hough crossing as a physical connection. Spatial indexes bound candidate searches.
Merge duplicates before computing lengths; do not turn every graph edge into a
physical wall object.

Optional pixel snapping uses distance to observed ink plus edge-direction
agreement, bounded in pixel space until calibrated. Preserve slanted geometry.
Never apply Manhattan/right-angle snapping globally. A cleaner vector that crosses
unsupported white space or erases an opening is a failed refinement.

## 3. Construct rooms and opening relationships

Node supported boundary linework and use Shapely `polygonize_full` so valid faces,
cuts, dangles and invalid rings all remain available. Combine boundary/body
evidence to distinguish the wall surface from the room interior.

Preserve holes and multiple components. Report room-region candidates separately
from semantic room names/types. One enclosed face need not equal one room, and
an open-plan named room need not have its own closed face.

Attach opening candidates to plausible nearby wall boundaries/axes using overlap,
orientation and local support. Keep unattached candidates. Do not force a nearby
window into the wrong wall just to produce complete associations.

Gap repair is bounded by image support and exposed as an inferred edge with its
own ID. Rooms/areas depending on it inherit that dependency. Do not unconditionally
close every door-sized gap or fill every enclosed hole into wall material.

## 4. Associate dimensions and compute quantities

Consume Track 1 text spans and native PDF text, preserving raw text and origin.
Parse explicit units, decimal commas/points and area notation. Keep ambiguous
numbers as alternatives; avoid classifying `2.40` as a room number solely by format
or `2400` as millimetres solely by magnitude.

Match dimension values with main lines, ticks/arrows and extension endpoints,
then associate them to the relevant view/region. Prefer direct declared dimension
evidence over inferring metric size from an object's typical dimensions.

Represent calibration as scoped evidence with supporting anchors and consistency
diagnostics. Cross-check compatible anchors; expose conflicting scales and separate
detail views. No default 1:100, assumed image DPI or 0.85m door calibration.
Uncalibrated geometry can still provide pixel quantities and spatial relationships.

Compute lengths, areas with holes, containment and adjacency locally. Keep printed
area and computed area distinct; distinguish net room interior from wall-axis or
gross-region area. Do not derive volume from an assumed height in this 2D phase.

## 5. Compare ML and geometry without manufacturing certainty

Consume normalized ML observations supplied by Track 1; preserve model/profile
identity, supported classes and original scores. Main and CAD are correlated ML
sources, not separate independent votes. CAD has no supported room/wall evidence.

Match candidates by compatible semantics, frame, overlap, boundary distance,
orientation and local support as applicable. Version thresholds and tune them on
development/reference drawings only. Distinguish true conflicts from unmatched
candidates and unavailable detection sources.

Create identity groups for likely duplicate physical objects. Preserve each
source's membership and count. Unresolved matches remain alternatives; never sum
ML and classical counts or select exact counts just because scores are high.
Wall graph edges, room faces, door leaves and stair treads have explicit counting
units and are not interchangeable with walls, rooms, door assemblies and flights.

Final JSON records scope/coverage, dependencies, quantities and crop regions.
Request source crops from Track 1 for disagreements, including enough surrounding
wall/text context to evaluate the issue. This prepares Jev; it does not call Jev.

## Reference algorithms to inspect

All paths below are under
`geometry/planproof_geometry/src/planproof_geometry/ai/` in this branch:

| Module | Useful part | Do not inherit |
| --- | --- | --- |
| `plan_wall_vectorize.py` | Skeleton/distance/Hough/collinear merge | Constant 0.85 confidence or implied production verification |
| `wall_mask_graph.py` | Medial-axis paths, junctions and thickness | Metre-only API, automatic Manhattan snap or unchecked closing |
| `v3/pixel_bim/extrude/pixel_snap.py` | Ink-distance field and local edge fit | Metric tolerance without calibration or ML-clipped ink called independent |
| `v3/pixel_bim/openings/gap_geometry.py` | Local PCA axes and opening projections | Label-bounded width called an independently measured gap |
| `plan_scale_infer.py` | Dimension-line association and robust scale candidates | Printed ratio suppressing checks, arbitrary-line matches or hidden ML hints |

The mask-vectorization helpers were found implemented in the inspected ConstructQ
source but not found called in its AI tree. Its default pipeline is annotation-
driven BIM; do not transplant that orchestrator. Avoid `plan_matching.py`: its
scan/database/3D purpose is not the required 2D ML-evidence comparator.

## Required tests and acceptance

- Known generated counts/lengths/areas exactly recover the expected semantic
  objects within declared raster tolerance; fixtures include corners, T/X junctions,
  diagonal walls, parallel close walls, duplicate segments and nested holes.
- Translation/resolution changes do not alter line pairing because of incorrect
  absolute-coordinate distance formulas. Curves survive as curves or explicitly
  tolerance-bounded approximations, not forced right-angle walls.
- Broken/open boundaries return diagnostics; inferred repairs are traceable.
  Pixel-supported openings are not closed away by morphology/vector snapping.
- Missing, contradictory or region-incompatible scale never yields asserted
  metric geometry. Explicit text values remain reportable as text evidence.
- ML-only, geometry-only, duplicate, class-conflict and hybrid cases preserve
  their proper dependency/availability semantics. Partially missing detection
  coverage cannot create an exact zero or full-plan count.
- Unit tests verify count member IDs, double-door/stair counting units, holes in
  area calculations and separation of printed versus calculated quantities.
- Integration runs without PyTorch installed by consuming prediction JSON, and
  without ML at all. Reject mismatched page/image bindings.
- Development report compares ML-only/classical-only/combined counts and objects,
  matched-room geometry/area error, unresolved coverage, cold/warm CPU cost and
  memory. Final QA accuracy remains a separate later integration measurement.

Use your six-hour planning allocation for a coherent baseline first. Implement
the declared pipeline with diagnostics and tests before exploring replacement
optimizers; do not start a new learned vectorization model or a whole ILP system.
