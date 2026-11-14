# Research basis and legacy assessment

Reviewed 14 November 2026. These are algorithm references, not evidence of
Planproof accuracy or permission to redistribute datasets/models.

## Primary references and decisions

| Source | What it establishes | What we use |
| --- | --- | --- |
| [Raster-to-Vector, ICCV 2017](https://art-programmer.github.io/floorplan-transformation.html), [author code](https://github.com/art-programmer/FloorplanTransformation) | Learned junctions plus integer programming for consistent floorplan primitives | Explicit junction/topology constraints and traceable vector primitives; no new learned detector or full legacy runtime |
| [Floor-SP, ICCV 2019](https://jcchen.me/floor-sp/), [author code](https://github.com/woodfrog/floor-sp) | Neural data terms plus room-wise shortest-path/consistency optimization on RGB-D-derived evidence | Boundary-support and shared-room consistency ideas; not a PNG/PDF drop-in or independent classical detector |
| [pdfplumber](https://github.com/jsvine/pdfplumber) | Character, line, curve, style and visual-debugging interfaces, primarily for machine-generated PDFs | Native PDF evidence alongside raster evidence, retaining styles and source coordinates |
| [pypdfium2 API](https://pypdfium2.readthedocs.io/en/stable/python_api.html) | Rendering and page transforms; PDFium calls are not thread-safe | Explicit renderer boundary; serialize PDFium calls or isolate processes, never unsafe concurrent calls |
| [OpenCV morphological line extraction](https://docs.opencv.org/4.x/dd/dd7/tutorial_morph_lines_detection.html) | Erosion/dilation with directional structuring elements | Inspectable raster primitives; adapt direction/scale and do not mistake the tutorial for floorplan accuracy evidence |
| [Shapely polygonize_full](https://shapely.readthedocs.io/en/stable/reference/shapely.polygonize_full.html) | Polygons plus cut edges, dangles and invalid rings | Expose topology failure instead of hiding incomplete room boundaries |
| [TU Wien dimension-extraction thesis, 2024](https://www.cg.tuwien.ac.at/research/publications/2024/riegelnegg-2024-aeo/) | Engineering-drawing pipeline associating dimension geometry, OCR and ratio consistency | Localized text/geometry association and scale checks; different drawing domain, not guaranteed construction-plan transfer |
| [eDOCr2 author code](https://github.com/javvi51/edocr2) | Region-specific recognition for primarily mechanical/production drawings | OCR-region organization as reference; no unmeasured recognizer replacement or training in this phase |

The learned systems above show that geometry and ML can cooperate. They do not
show that geometric postprocessing is statistically independent of its ML input.
Reported paper benchmark numbers do not become our component or QA accuracy.

## ConstructQ findings that matter here

The owner supplied ConstructQ as a local reference; its relevant source was
inspected read-only by separate pixel/vector reviewers. No original environment
is required to implement this handoff; mirrored algorithm references are already
under this branch's `geometry/` folder. No private application notes, customer
data, credentials or original model artifacts are part of this handoff.

Confirmed in inspected source:

- The default orchestration is `perfect_bim` and requires annotation sidecars.
  The older V3 pipeline is a fallback, not the default reconstruction path.
- Production door diagnostics call mode-aware wall refinement. Paired-outline
  walls use classical processing; filled-wall refinement can use ML gravity.
- The classical helper clips to trusted user polygons, selects a largest component
  and uses an external contour. That can discard valid pieces/holes for our task.
- ML-gravity can fall back to the user polygon when model support is missing.
  With an ML proposal in place of a human label, this would falsely confirm it.
- Standalone mask vectorizers implement skeleton/medial-axis paths, distance
  thickness and Hough/graph outputs, but no AI-tree production callsites were
  found for those helpers. Their presence is not a verified production benchmark.
- Some paths assume a typical 0.85m door for scale, fixed metric tolerances,
  right-angle snapping, constant segment scores or expected stair dimensions.
  Those assumptions are not acceptable factual evidence for Planproof QA.

Positive concepts to retain are drawing-style-sensitive processing, local raster
support, ink-distance snapping, explicit opening axes, graph structure and
diagnostics. Reimplement/adapt them as small tested functions with source lineage.

## Branch readiness and reuse policy

The historical snapshot is not an installed or verified evidence engine. Static
inspection found a malformed test identifier in the old CAD test suite, missing
optional modules/dependencies and application-specific imports. Do not repair the
entire snapshot as a prerequisite; test only deliberately adapted code in the new
standalone package. Existing geometry quantities and old benchmark notes must not
be presented as newly measured results on Stoa.

Preserve applicable copyright/licence notices for code actually adapted. Read
the relevant source notices rather than assuming every copied file has the same
licence. Do not copy data or private application configuration to make a reference
run. Keep raw training/evaluation sources, prediction artifacts and crops local.

## Practical comparison protocol

Use generated fixtures for exact contracts and the existing development/reference
drawing partition for selecting geometric rules. Group by source drawing/family;
do not select near-duplicate pages into opposing partitions. Do not use sealed
final-evaluation answers or completed ML test results to tune this engine.

Run ML-only, classical-only and combined evidence on the same canonical images,
scope/classes and matching rules. Report supports and failure counts alongside
precision/recall, count error, region overlap and area error. Keep unmatched,
unresolved and unsupported cases visible; do not report errors only on easy
matched examples. Record algorithm/config/model hashes and cold/warm local cost.

The end-to-end Gemini/Jev QA comparison belongs to later integration. This phase
delivers evidence and crop interfaces; it does not establish answer accuracy by
borrowing detector, OCR or research-paper benchmark scores.
