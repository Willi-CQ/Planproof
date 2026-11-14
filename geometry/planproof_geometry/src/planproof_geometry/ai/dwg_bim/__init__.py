"""DWG/DXF -> BIM pipeline.

Erzeugt ein vollwertiges BIM-Modell direkt aus Vektor-CAD (DWG/DXF) statt aus
gerastertem PDF. Weil die Geometrie bereits als exakte Linien/Polylinien auf
benannten Layern und Tuer/Fenster-Bloecke vorliegt, entfaellt die komplette
Raster-/ML-/OCR-Kette (px_to_m-Schaetzung, YOLO-Detection, Wall-Segmentation).

Architektur (maximale Wiederverwendung der bestehenden Downstream-Bausteine):

    DWG --(ODA/LibreDWG)--> DXF --(ezdxf)--> Entities
      -> layer_classifier   (Layer/Block-Name -> Bauteilklasse, DACH+AIA)
      -> wall_reconstruction (Parallellinien-Paare -> Achse + Dicke)
      -> WallGraph           (domain.wall_graph; Junctions + parametrische Oeffnungen)
      -> opening_extraction  (Tuer/Fenster-Bloecke -> ParametricOpening)
      -> room_detection      (shapely-Polygonize -> Raeume + TEXT-Label)
      -> build_scene_from_wall_graph (-> SceneModel)            [REUSE]
      -> export_scene_to_gltf        (-> model.glb)             [REUSE]
      -> IfcExporter.export          (-> plan.ifc, parametrisch)[REUSE]

Der BIM-Output ist echtes IFC4 (IfcWallStandardCase + IfcOpeningElement via
IfcRelVoidsElement/IfcRelFillsElement + IfcDoor/IfcWindow/IfcSpace, Material-
Layer-Sets aus dem DIN-Katalog, Multi-Storey) plus eine glTF-Ansicht.
"""
from __future__ import annotations

from .pipeline import DwgBimResult, run_dwg_to_bim_pipeline

__all__ = [
    "DwgBimResult",
    "run_dwg_to_bim_pipeline",
]
