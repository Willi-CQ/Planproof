# Floor-plan model data plan

**Updated:** 2026-11-13  
**Purpose:** assemble external supervision for the floor-plan perception and room-structure models. Dataset files, processed samples, splits, checkpoints, and model weights belong in local ignored storage and must never be committed or pushed.

**Current training readiness:** ResPlan is collected and can support a room-graph/category component, but it contains vector geometry rather than source images. No cleared raster image/annotation corpus for the visual detector is local yet. CVC-FP is the first target; its publisher endpoint timed out. FloorPlanCAD, CubiCasa5K, and the other sources remain behind rights, duplicate, or agreement checks. Do not start GPU training for the image detector until such a corpus is cleared and locally validated.

## Data boundary

- **Stoa is the sealed client/evaluation corpus.** Do not use Stoa drawings, questions, answers, labels, derivatives, or embeddings for training, augmentation, prompt tuning, threshold selection, or model selection. Its archive contains source-specific conditions and the held-out answers are the final benchmark.
- Every external source stays in its own directory under `data/downloads/<ID>/`, then `data/extracted/<ID>/` and `data/processed/<ID>/`. Keep original license, attribution, source URL, version/date, archive checksum, and any accepted terms alongside it.
- Before external data enters training, compare drawings/layouts against Stoa's evaluation groups and against other training sources. Quarantine suspected duplicates; keep source groups intact across splits. Do not combine benchmark numbers across domains without reporting each source separately.
- The project GitHub may contain this registry and acquisition instructions only. Raw and extracted data, caches, generated samples, question/answer files, checkpoints, and weights remain ignored locally.

## Recommended corpus by training role

| Dataset | Training role | Fit to this model | Current status and gate |
| --- | --- | --- | --- |
| [FloorPlanCAD](https://floorplancad.github.io/) | Primary vector/CAD pretraining | 15,663 real residential and commercial CAD plans with fine-grained symbol labels; best match to technical drawing geometry. | **Acquire after source-rights review.** Annotation/site are CC BY-NC 4.0, but authors say they do not own drawing copyright and users accept responsibility. Keep source files local; retain attribution. |
| [CVC-FP](https://dag.cvc.uab.es/dataset/cvc-fp-database-for-structural-floor-plan-analysis/) | Primary raster segmentation/detection | 122 real scanned plans with ground truth for rooms, walls, doors, windows, parking doors, separations, and relations. | **Pending collection.** Publisher states CC BY-NC 4.0 and links the complete archive, but the endpoint timed out from this environment. Small, high-value supervised set; deduplicate against Stoa before training. |
| [CubiCasa5K](https://github.com/CubiCasa/CubiCasa5k) | Primary raster segmentation baseline | 5,000 images with dense semantic polygons for rooms and plan elements. | **Quarantine pending deduplication.** A CubiCasa5K folder is already inside the Stoa archive; do not train from upstream copies until overlap with final Stoa evaluation drawings is measured and all overlapping groups are excluded. Dataset has separate NC-SA terms. |
| [Structured3D](https://structured3d-dataset.org/) | Synthetic geometry pretraining/augmentation | 3,500 synthetic homes with dense 2D/3D structure labels; useful for geometry and room boundaries, with a synthetic-to-real gap. | **Access agreement required.** Obtain and review its Terms of Use before requesting/downloading. Do not infer data rights from MIT code. |
| [HouseExpo](https://github.com/TeaganLi/HouseExpo) | Room category/topology auxiliary training | 35,126 layouts and 252,550 rooms; helpful for room categories and connectivity, weak for technical-drawing visual detection. | **Do not collect/use yet.** HouseExpo is based on SUNCG. Its public README points to a Drive download, but the repository's MIT license covers software and does not state a separate data grant; SUNCG's original access required a signed agreement. |
| [SESYD](http://mathieu.delalandre.free.fr/projects/sesyd/) | Synthetic symbol-detector pretraining | Synthetic floor-plan/document symbols and vector labels can augment symbol spotting. | **Terms and exact release contents need review.** Public project page/torrent is not a data license. |
| [RPLAN / DeepLayout](http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html) | Room segmentation/category auxiliary training | About 80,000 rasterized residential plans with room-layout semantics; scale is useful, style/region transfer is uncertain. | **Access and reuse terms unverified.** Obtain source terms before acquisition. |
| [ResPlan](https://github.com/m-agour/ResPlan) | Room-graph and semantic-category model training | 17,000 vector layouts with walls, doors, windows, 17-category labels, metric coordinates, and canonical splits. Useful to a graph model after geometry extraction, not a raster detector. | **Collected locally.** Dataset release identifies CC BY 4.0; 100,106,537-byte archive. Keep its supplied split and remove/flag its documented near-duplicate clusters before claiming clean evaluation. |
| [ModRes](https://arxiv.org/abs/2610.17178) | Room-graph training / data-quality comparison | Cleaned/enriched ResPlan derivative with 2D geometry, images, 3D representations, and complementary graphs. | **Release and terms need confirmation.** Check for source overlap with ResPlan before adding; do not double-count the same plans. |
| [ROBIN](https://github.com/gesstalt/ROBIN) | Small plan retrieval/classification auxiliary set | Building-plan category and retrieval labels; not dense detector supervision. | **Inspect repository data license and overlaps before use.** |
| [Zillow Indoor Dataset (ZInD)](https://github.com/zillow/zind) | Plan-to-room-image grounding auxiliary training | Floor plans paired with panoramas; may support room-label grounding but differs from construction drawings. | **Review dataset agreement and use terms; registration/access gate.** |
| [FloorNet](https://openaccess.thecvf.com/content_ECCV_2018/html/Chen_Liu_FloorNet_A_Unified_ECCV_2018_paper.html) | Optional 3D/scan-to-plan training | 155 home scans with plan annotations; only useful if a scan-to-plan branch is implemented. | **Author release/terms need confirmation.** Not part of the initial 2D drawing detector corpus. |
| [FloorPlanQA](https://github.com/OldDeLorean/FloorplanQA) | Reasoning evaluation only | Structured-layout questions test spatial reasoning after geometry is parsed; it does not train image perception. | **Keep outside detector training.** Check linked layout dataset license if used for evaluation. |

## Auxiliary data for separate pipeline components

These datasets can help the deterministic BIM/IFC or physical-space components. They are not floor-plan image detector training data and must not be mixed into its training manifest.

| Dataset | Separate use | Status |
| --- | --- | --- |
| [buildingSMART Sample-Test-Files](https://github.com/buildingSMART/Sample-Test-Files) | IFC schema/parser smoke tests | Review repository and file-level terms; metadata/example files only. |
| [buildingSMART Certification datasets](https://github.com/buildingSMART/Certification-datasets) | IFC interoperability and parser tests | Review file-level terms; not model training. |
| [Open IFC Model Repository, University of Auckland](https://openifcmodel.cs.auckland.ac.nz/) | IFC storey/space/quantity extraction | Check per-model terms; not raster training. |
| [IFC product geometry (Zenodo 10730758)](https://zenodo.org/records/10730758) | IFC object-geometry experiments | Check Zenodo record and source-model terms; derived component meshes, not floor plans. |
| [BIMvoice IFC samples](https://github.com/BIMvoice/ifcsamples) | IFC version coverage and ingestion tests | Preserve source/license manifests; curator does not own models, so terms remain source-specific. |
| Auckland, RWTH, KIT, buildingSMART, and OSArch collections linked from the [IFC geometry record](https://zenodo.org/records/10730758) | Expand IFC parser/model coverage | Check each source repository and model license separately; not raster training. |
| [ScanNet](https://github.com/ScanNet/ScanNet), [ScanNet++](https://github.com/scannetpp/scannetpp), [S3DIS](https://buildingparser.stanford.edu/dataset.html), [Matterport3D](https://niessner.github.io/Matterport/), [HM3D](https://github.com/facebookresearch/habitat-matterport3d-dataset) | Optional floor-surface/indoor 3D segmentation | Registration, terms, or non-commercial conditions apply. Modality mismatch for 2D drawing detection; acquire only if a 3D surface branch is in scope. |

## Not training sources

Stoa's `cubicasa5k`, `swiss_dwellings`, `opening_design`, `ifc_bench`, `ifc_models`, `europeana`, `finna`, `habs`, and `wikimedia` files are all part of the client corpus, whether or not every source has supplied QA labels. They remain evaluation/reference material only. Keep the locally generated Stoa question partitions under evaluation-only names and paths; never name or load them as training data.

## Split and reporting rules

1. Split at the original drawing/building/listing/model level, never at tile/page/question level. Keep alternate formats and all floors from one building together.
2. Preserve publisher-provided train/validation/test splits where their group boundaries and duplicate handling are documented. For new splits, fix and record the seed and group key.
3. Deduplicate by archive hash, normalized vector geometry, and perceptual image hash before any fit. Exclude all confirmed Stoa evaluation overlap from external training and record the excluded group IDs locally.
4. Keep synthetic, residential-layout, archival, scanned technical drawing, CAD, and BIM results as separate slices. Report label mapping, class coverage, and per-source metrics.
5. For every dataset, keep `source.json` locally with source/version/observed date, download URL, license/terms snapshot, attribution, SHA-256, archive size, extraction date, exclusions, and split mapping. Never store credentials or private access tokens there.

## Acquisition status

- ResPlan is collected locally at `data/downloads/ResPlan/ResPlan.zip` (100,106,537 bytes; SHA-256 `F718DE8865E51BBE93B49B584798E3C536ED6B4B8A5D32F01B56812F389AEB46`), with its README, CC BY 4.0 data license, and canonical `split.json`; the archive lists `ResPlan.pkl` and was extracted locally. Do not load the pickle until its contents are handled in an isolated data-processing workflow.
- CVC-FP remains pending. The publisher links `CVC-FP.zip` and states CC BY-NC 4.0, but both HTTP and HTTPS endpoints timed out from this environment; no archive bytes were collected.
- HouseExpo's image archive is publicly linked, but is not collected because its underlying SUNCG rights are not established by the MIT software license. Non-commercial intent alone does not supply a missing data grant.
- Other sources remain pending acquisition or rights/access review. A published link or catalog entry does not mean bytes have been collected.
- Stoa's existing local question files are evaluation references only and are being renamed/re-homed to make that boundary explicit.
