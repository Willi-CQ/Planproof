# Geometry adaptation record

The integrated pixel front end also contains the team's GitHub contribution,
pinned at `24e37023c9441c657050b8595f1cf738230e6b17` (original implementation
`d4b8c003c250bfdccb9e47b8b4d73c367597b21d`). Source/function hashes and the
adapter changes are recorded in [TRACK1_SOURCE.json](TRACK1_SOURCE.json) and the
[integration guide](../docs/deterministic-evidence/INTEGRATION.md). Original pure
pixel routines are preserved apart from import/format normalization; no blanket
license is inferred from the duplicate upstream application's package metadata.

This implementation adapts algorithmic ideas from the owner's supplied local
ConstructQ-derived geometry snapshot. It does not import its BIM application or
claim that these helpers were production-validated for this task.

| New implementation | Supplied reference under `geometry/` | Adaptation |
| --- | --- | --- |
| `geometry.fit_line`, `merge_collinear`, `skeleton_lines` | `planproof_geometry/src/planproof_geometry/ai/plan_wall_vectorize.py` | Pixel units, orthogonal fits, supported gaps, residuals and measured thickness; no constant correctness score |
| `geometry.pair_boundaries` | `planproof_cad/src/planproof_cad/cad/wall_reconstruction.py` | Mutual nearest boundaries, shared projected overlap, translation-invariant relative normal offsets; no DXF or metric defaults |
| `geometry.node_graph`, `net_regions` | `ai/wall_mask_graph.py`; CAD `room_detection.py` | Exact noding and polygonize diagnostics; holes and dangles; no automatic Manhattan snapping or forced closure |
| `geometry.snap_polygon` | `ai/v3/pixel_bim/extrude/pixel_snap.py` | Reused ink-distance-field idea with bounded pixel attraction, area/topology guards and preserved source geometry |
| `pixels.extract_pixels`, `mask_polygons` | `ai/v3/pixel_bim/refine/classical.py`; `extrude/paired_outline.py` | Threshold/core/outline evidence, preserved islands/holes, explicit ambiguity; no human-label or ML polygon treated as truth |
| `pixels.detect_arcs`, `repeated_parallel_symbols` | `ai/plan_heuristic_openings.py` | Scale-free angular sampling, leaf support, parallel-line and wall context; no guessed metres |
| `pixels.repeated_parallel_symbols` | `ai/v3/pixel_bim/stairs/stair_pixel_extractor.py` | Local axes/repeated projections with rail evidence; no prescribed DIN tread count |
| `geometry.project_opening` | `ai/v3/pixel_bim/openings/gap_geometry.py` | Local axis projection, explicitly a symbol extent rather than independent clear width |
| `measurements` | `ai/plan_measure_parse.py`, `plan_scale_infer.py`, `plan_transform.py` | Explicit units, ambiguous formatting retained, distinct verified anchors and weighted median; no implicit metre or 1:100 fallback |
| `inputs` | `pdf/pdfium_adapter.py` | Small standalone crop/rotation-aware renderer and native path/text boundary |

`ai/` and `pdf/` in the abbreviated paths mean
`geometry/planproof_geometry/src/planproof_geometry/ai/` and its sibling `pdf/`.
The new comparison/provenance/API layer is purpose-built for 2D evidence.

The supplied CAD package states: Copyright (c) 2026 Willi Horn. All rights
reserved. See its [LICENSE](../geometry/planproof_cad/LICENSE) and
[NOTICE](../geometry/planproof_cad/NOTICE). No new blanket open-source license is
asserted for the supplied reference collection or this project.

Runtime libraries remain separate unmodified pip dependencies: NumPy (BSD), SciPy
(BSD), OpenCV (Apache-2.0 for the pinned version), Pillow (HPND), scikit-image
(BSD), Shapely (BSD; bundled GEOS LGPL), and pypdfium2 (Apache-2.0/BSD-3-Clause;
PDFium and its bundled third-party notices apply). Their distributions include
their own license files. No reference dataset, source drawing or model weight is
distributed by this package.
