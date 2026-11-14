# Planproof geometry

## Start here: current PNG/PDF evidence work

**Read [the implementation handoff](../docs/deterministic-evidence/README.md) before writing code.**
It defines the current architecture, two-person ownership, interfaces, research,
and acceptance checks. The owner/Codex pair takes **Track 2: vector geometry,
measurements and ML cross-checking**. The teammate takes Track 1: pixel evidence.

The ML baseline already exists on the separate
[`ml/trained-baseline-v1` branch](https://github.com/Willi-CQ/Planproof/tree/ml/trained-baseline-v1).
This branch contains reference code, not a completed deterministic evidence engine.
The new engine is PNG-first with PDF support and produces JSON, 2D overlays and
crops. It does not create BIM or 3D models or implement Gemini/Jev inference.

**Everything below is a historical reference inventory.** Existing modules were
extracted from a larger application and contain assumptions unsuitable for this
task. Do not install or connect the entire legacy pipeline. Selectively adapt
algorithms into the new standalone package, with tests and preserved attribution.
The older statements about training a model later, production 3D entry points and
which CAD package to keep do not define the current implementation scope.

## Historical inventory

## What to use later

| Input the new model, or the file, already gives you | Code to run | Result |
| --- | --- | --- |
| DXF with layers | `planproof_cad` | Walls, doors, windows, rooms, `scene.json`, quantities, IFC4, GLB. No learned model. |
| IFC | `planproof_geometry.ai.ifc_index` | Stored length, area, and volume from quantity sets, plus element ids. The mesh is separate (`ifcopenshell.geom` is not used here). |
| PDF vectors and text | `planproof_geometry.pdf` (`PdfDatei`) and `ai/plan_extract_pdf.py` | Lines and text boxes in pdfium coordinates. Raster size is capped by `ai/pdf_render_budget.py`. |
| Dimension strings | `ai/plan_measure_parse.py` | Parsed measure text. It does not read the drawing by itself. |
| Raster lines | `ai/plan_wall_vectorize.py` and `ai/cv_compat.py` | Pixel segments from a probabilistic Hough transform. |
| Wall mask or wall polygons, in metres | `ai/v3/pixel_bim/extrude` | Shapely polygons extruded to a trimesh, then GLB. |
| Door and window boxes plus the wall mask | `ai/v3/pixel_bim/openings` and `ai/v3/perfect_bim` | Opening voids, frames, glass, bridges. |
| Stair marks on the raster | `ai/v3/pixel_bim/stairs` | Stair solids. |
| Annotations JSON (class, box or polygon, page) | `ai/perfect_bim_pipeline.py` | `model.glb`, `semantics.json`, `quantity_takeoff.csv`, viewer. This is the production 3D entry. It still expects that sidecar. It does not detect the plan. |
| GLB | `ai/ifc_export.py` | IFC written from the GLB. |

`planproof_cad` is the DXF path to keep. The older `ai/dwg_bim` copy inside Planproof was not copied. DWG still needs an external ODA File Converter, which is not in this folder.

## Where the new model plugs in

`ai/v3/pixel_bim/refine/wall_dispatcher.py` sends a filled wall body through `refine_walls_ml_gravity`. That function calls `planproof_geometry.ai.plan_wall_segmentation.infer_wall_mask`. The loader file is included so the call site stays valid. Its weights are not included. Point `infer_wall_mask` at the new model, or skip it: `ai/v3/pixel_bim/refine/classical.py` refines a labeled polygon without a network.

Door and window meshes do not contain a detector. They consume boxes and polygons. Emit those from the new model in the annotation sidecar shape that `perfect_bim_pipeline.py` already reads (`class`, `bbox` or polygon).

Furniture naming in `ai/furniture/classify.py` calls an external classifier. The mesh code can place a box without that call. Leave it off until a model is chosen.

## Deterministic coordinate frames

Keep the frame on each primitive. Do not mix them in one list without a conversion.

- pdfium, via `PdfDatei`: PDF page space, usually origin at the bottom left.
- `plan_wall_vectorize` and OpenCV: pixels, origin at the top left, y downward.
- DXF in `planproof_cad`: unitless numbers. The named unit is `$INSUNITS`. `LINE` is in world coordinates. Many other entities are in object coordinates until converted.
- IFC quantities: project units, scaled to metres by `ifcopenshell.util.unit.calculate_unit_scale`.
- Extrusion: polygons must already be in metres. `plan_to_bim_v3_pipeline._calibrate_px_to_m` estimates scale from a typical door width of 0.85 m. Replace that once a measured scale exists.

## Left behind on purpose

- YOLO weights, wall-segmentation weights, and `ai/training`.
- The cloud worker.
- PaddleOCR (`ocr_paddle.py`). `plan_extract_raster.py` still calls it from inside the text function.
- `planproof_cad/goldset/data` (DXF derived from Swiss Dwellings). The goldset runner and its notes were copied. The DXF files were not.
- PyMuPDF. PDF reading goes through pdfium.
- GAEB invoicing, CPM scheduling, and the labour-safety rule engine.

## Added so the deterministic set is complete

These implementations were brought in on 2026-11-13 after a file-by-file search. They are still not called. No weights were copied.

- Vector plan: line merge, wall reconstruction, door-arc and window-line detection, opening splits, scale and dimension parsing, point-to-metre transform, measurement anchors, raster Canny and Hough in `plan_extract_raster.py`.
- Scene and meshes: `plan_build_scene.py`, `plan_bim_meshes.py`, `plan_bim_gltf_export.py`, `plan_gltf_walls.py`, `plan_quantity_takeoff.py`, `domain/scene_model.py`, `domain/wall_graph.py`.
- DXF in core: `ai/dwg_bim`. `planproof_cad` remains the later package of the same pipeline.
- Quantities and rules: `standards/reb/aufmass.py` plus DA11 and DA12 readers, DIN snap tables, catalog U-value, IDS checks, AABB clash and federation.
- Scan geometry: seeded RANSAC, floor-plan projection, rooms, openings, shells, registration, and scan-versus-plan comparison. `scan_to_bim/ml/semantic_segmentation.py` is included only because the plane step imports its label table. It loads no weights from this folder. `load_segmenter()` stays empty until a model file is supplied later.

`standards/reb/__init__.py` is the one file written here rather than copied. The original init also exported GAEB X31, which imports the invoicing builder. That builder is outside this grant. The quantity functions are unchanged.

Left out on purpose: YOLO, PaddleOCR, LLM calls, GAEB invoicing, CPM scheduling, and the labour-safety rule engine. `plan_extract_raster.py` still imports `ocr_paddle` inside its text function. That module is not in this folder.

## Not wired

Do not add this folder to a package path, a job runner, or an API route as part of this handover. Imports still say `planproof_geometry` and `planproof_cad`. They resolve only after those packages are installed in an environment that is not set up here.

`planproof_cad` declares numpy, shapely, ezdxf, pydantic, and ifcopenshell. The 3D raster path also needs opencv, pypdfium2, and trimesh at the call sites that extrude masks. Those dependencies are not installed in Planproof.
