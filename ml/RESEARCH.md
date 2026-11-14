# Detection research and chosen baseline

Research checked 2026-11-13. Scores below are reported by authors, not reproduced
Planproof results. Class sets, annotation versions and evaluation protocols differ.

| Source | Reported result | Relevance |
| --- | --- | --- |
| [CubiCasa5K paper, Table 5](https://arxiv.org/html/1904.01920v1) and [official implementation](https://github.com/CubiCasa/CubiCasa5k) | Original raw test room mIoU 57.5%; icons 55.7%. Polygonized outputs score 49.3% and 41.6%. | Reproducible multitask baseline; preserve dense predictions and measure postprocessing separately. |
| [Multi-Unit Floor Plan Recognition, Tables 4–5](https://arxiv.org/html/2408.01526v1) | Adapted MDA/MACU: CubiCasa five-class mean F1 .86 and IoU .76; CVC-FP three-class CAB2 mean F1 .90 and IoU .82. | Dense segmentation and boundary learning are relevant; processed protocols do not imply these scores on untouched full-class releases. |
| [SymPoint-V2](https://arxiv.org/html/2407.01928v1), [code](https://github.com/nicehuster/SymPointV2) | Component ablation reports FloorPlanCAD PQ 90.1 with layer enhancement and position-guided training. | Strong CAD primitive labeling; requires vector/layer cues, not a raster detector benchmark. |
| [Emit versus detect preprint](https://arxiv.org/html/2608.25608v1), [code](https://github.com/Cyprinus12138/fpvec-lab) | Corrected 296-plan CubiCasa subset: dense graph readout wall F1 .818 versus sequence .790 at normalized tolerance .05. | Supports dense localization as an initial choice; distinct subset and newer preprint, not a replicated result or direct comparison with original CubiCasa metrics. |
| [RoomFormer](https://github.com/ywyue/RoomFormer) | Reconstruction benchmark uses projections from 3D scans on Structured3D/SceneCAD. | Useful geometry research, but scores cannot serve as drawing-image detector accuracy. |

The engineering baseline uses [SMP's supported U-Net/ResNet-34](https://github.com/qubvel-org/segmentation_models.pytorch).
Its choice is for a controlled, maintainable first experiment and compatibility
with mask-based consumers, not a claim that U-Net beats these papers.

Data-format evidence for adapters: [CubiCasa reader](https://github.com/CubiCasa/CubiCasa5k/blob/master/floortrans/loaders/house.py),
[SVG utilities](https://github.com/CubiCasa/CubiCasa5k/blob/master/floortrans/loaders/svg_utils.py),
[original CAD attributes](https://github.com/nicehuster/SymPoint/blob/main/parse_svg.py),
[ResPlan geometry](https://github.com/m-agour/ResPlan/blob/main/resplan_utils.py),
[ModRes release](https://github.com/emmashelley/ModRes), and
[Structured3D](https://github.com/bertjiazheng/Structured3D).

Adapters do not import or execute these repositories. Code here is independently
implemented around the declared formats; actual archive validation happens on
the GPU host. Dataset and code terms remain attached to their original sources.
