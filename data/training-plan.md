# Floor-plan training datasets for the GPU machine

Updated: 2026-11-13. This is the planned GPU download list, not an inventory of datasets on the laptop. Machine-readable manifest: [gpu-training-datasets.csv](gpu-training-datasets.csv).

Download original publisher releases on the GPU machine later. The Stoa challenge archive and its questions are the final application corpus and are not the training-data source. Original CubiCasa5K is included even though Stoa contains examples from the same source. A shared source name does not exclude an upstream release from this training plan.

| Dataset | Priority | Intended training use | Original release |
| --- | --- | --- | --- |
| CubiCasa5K | Core | Rooms, walls, openings, and object segmentation | https://github.com/CubiCasa/CubiCasa5k |
| FloorPlanCAD | Core | CAD symbol and element detection; vector/raster supervision | https://floorplancad.github.io/ |
| CVC-FP | External test only (v1) | Independent structural detection evaluation | https://dag.cvc.uab.es/dataset/cvc-fp-database-for-structural-floor-plan-analysis/ |
| RPLAN | Core | Residential room categories, boundaries, and layout semantics | http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html |
| SESYD | Core | Synthetic floor-plan symbol detection and augmentation | http://mathieu.delalandre.free.fr/projects/sesyd/ |
| ResPlan | Supplementary | Room categories and graphs; render paired images/labels for synthetic training | https://github.com/m-agour/ResPlan |
| HouseExpo | Supplementary | Room categories and layout/topology supervision | https://github.com/TeaganLi/HouseExpo |
| Structured3D | Supplementary | Synthetic room geometry and structure; derive suitable plan views | https://structured3d-dataset.org/ |
| ModRes | Supplementary | Cleaned ResPlan-derived geometry and room graphs | https://arxiv.org/abs/2610.17178 |
| ROBIN | Optional | Floor-plan classification/retrieval auxiliary task | https://github.com/gesstalt/ROBIN |
| Zillow Indoor Dataset (ZInD) | Optional | Room geometry and plan/panorama grounding | https://github.com/zillow/zind |

Vector and scene datasets need preprocessing into compatible input/label pairs. They can provide synthetic plan images and room-structure supervision; their source formats differ from annotated real drawing images. ResPlan and ModRes share provenance and must not be counted as independent plans.

## Download and training notes

- GPU downloads happen later. No further training-data download onto the laptop is needed.
- The CSV records publisher pages and known download routes. It is a manifest, not an executable downloader. Agreement-dependent links are resolved on the GPU machine; some releases require access requests or license review.
- Preserve publisher licenses, attribution, versions, checksums, and splits. Listing a source does not grant access or permission to use its data.
- Keep original drawing/building groups together across training, validation, and test. Generate augmentations after splitting and group ResPlan/ModRes relatives together.
- Original releases can share drawings with Stoa. When benchmarking the finished system later, identify this overlap and report seen versus unseen drawings; downloading originals separately does not establish an independent test set. This does not exclude CubiCasa5K from the training-source list.
- Training archives, labels, generated samples, and split data stay outside Git or in ignored data/downloads/, data/extracted/, and data/processed/. No training data goes to GitHub.

FloorNet and indoor scan datasets remain potential 3D extensions; IFC samples support the deterministic BIM parser; FloorPlanQA supports reasoning evaluation. They remain in the broader registry and are not part of this 2D floor-plan training download list.

ResPlan was downloaded previously into ignored local storage. Stoa question references are under ignored data/evaluation/. Local download state is separate from this planned GPU corpus.

## Dataset size and format audit (2026-11-13)

The GPU CSV records sample counts, sample units, verified byte sizes where available, publisher size displays, source links, technical fit, and conversion requirements. Unknown byte sizes are explicit; no complete corpus disk total is established. CubiCasa5K ZIP is 5,469,495,706 bytes (5.47 GB) from Zenodo metadata; its publisher's approximately 105 GB figure describes the optional preprocessed LMDB. ResPlan ZIP is 100,106,537 bytes from the earlier local collection. HouseExpo's JSON archive is displayed as 24.4 MB on GitHub; its PNG archive is separate. Original ROBIN.zip is displayed as 15.1 MB. ZInD's publisher states about 40 GB for the full data.

Technical compatibility is assessed for the intended label task, not as a claim that releases share a format. CubiCasa5K/CVC-FP provide structural image supervision; FloorPlanCAD needs primitive-label conversion; RPLAN needs semantic-channel decoding and independent inputs/targets; ResPlan/HouseExpo/ModRes support rendered or room-graph supervision; Structured3D needs scene-to-plan preprocessing. Missing class labels must not be treated as background negatives. ROBIN and ZInD remain optional auxiliary tasks. Source counts are not a unique-data sum: ModRes derives from ResPlan and Structured3D counts house scenes, not equivalent annotated drawing images.

## Executable training pipeline

Implementation and current source-adapter boundaries: [ML guide](../ml/README.md). GPU setup, first-source import and launch commands: [GPU readiness](../ml/GPU_READY.md). The baseline uses CubiCasa5K; curriculum and optional-source ablations are explicit configs. CVC-FP is held out from training in v1. No real corpus or weights are included in source control.
