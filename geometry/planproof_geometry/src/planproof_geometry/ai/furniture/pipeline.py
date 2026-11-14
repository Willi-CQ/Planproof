"""Orchestrierung des Moebel-Schritts der Plan-to-BIM-Pipeline.

Bindet die Bausteine zusammen:
  Annotationen -> Raeume (rooms) + Crops (crop) + Raum-Zuordnung
              -> EU-LLM-Klassifikation (classify)
              -> deterministische Platzierung (placement)
              -> Mesh + Szene-Append (mesh)

Reihenfolge der Verantwortung (research-belegt):
  LOKALISIEREN = User/YOLO-Label (Bbox)  |  ERKENNEN = LLM (benennt)  |
  PLATZIEREN = deterministisch (Geometrie)

Soft-fail durchgehend: ohne Classifier/cv2/Bild laeuft der Schritt weiter und
setzt generische Platzhalter — er bricht die BIM-Erzeugung nie.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..label_schema import to_plan_element_class
from .classify import (
    ClassifyInput,
    FurnitureClassification,
    FurnitureClassifier,
    classify_furniture,
)
from .crop import extract_crop
from .placement import PlacedFurniture, place_furniture
from .rooms import RoomRegion, _annotation_bbox_m, assign_room, extract_rooms

logger = logging.getLogger(__name__)

BBoxM = Tuple[float, float, float, float]


@dataclass
class FurnitureResult:
    """Ergebnis des Moebel-Schritts."""

    placements: List[PlacedFurniture] = field(default_factory=list)
    rooms: List[RoomRegion] = field(default_factory=list)
    classifications: Dict[str, FurnitureClassification] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_items(self) -> int:
        return sum(len(p.items) for p in self.placements)

    def to_semantics(self) -> List[Dict[str, Any]]:
        """Kompakte Semantik-Liste fuer semantics_v3.json / Diagnose."""
        out: List[Dict[str, Any]] = []
        for pf in self.placements:
            for item in pf.items:
                out.append({
                    "bbox_id": pf.bbox_id,
                    "catalog_id": item.catalog_id,
                    "order": item.order,
                    "room_id": pf.room_id,
                    "room_type": pf.room_type,
                    "zone_type": pf.zone_type,
                    "facing_deg": round(item.facing_deg, 1),
                    "height_m": round(item.height_m, 3),
                    "confidence": round(item.confidence, 3),
                    "source": pf.source,
                })
        return out


def _ann_bbox_px(ann: Dict[str, Any], scale_xy: Tuple[float, float]) -> Optional[BBoxM]:
    """Bbox eines Annotations in Render-Pixeln (plan-px * scale_xy)."""
    sx, sy = scale_xy
    if ann.get("type") == "polygon":
        pts = ann.get("points") or []
        if not pts:
            return None
        xs = [float(p[0]) * sx for p in pts]
        ys = [float(p[1]) * sy for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))
    bbox = ann.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = bbox
    xs = sorted([float(x0) * sx, float(x1) * sx])
    ys = sorted([float(y0) * sy, float(y1) * sy])
    return (xs[0], ys[0], xs[1], ys[1])


def _is_furniture(ann: Dict[str, Any]) -> bool:
    return to_plan_element_class(ann.get("class") or "") == "furniture"


def _is_wall(ann: Dict[str, Any]) -> bool:
    return to_plan_element_class(ann.get("class") or "") == "wall"


def _center(bbox: BBoxM) -> Tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def _long_edge(bbox: BBoxM) -> float:
    x0, y0, x1, y1 = bbox
    return max(abs(x1 - x0), abs(y1 - y0))


def run_furniture_step(
    annotations: Sequence[Dict[str, Any]],
    image=None,
    *,
    scale_xy: Tuple[float, float] = (1.0, 1.0),
    px_to_m: float = 1.0,
    classifier: Optional[FurnitureClassifier] = None,
    cache: Optional[Dict[str, FurnitureClassification]] = None,
    request_id: str = "furniture",
) -> FurnitureResult:
    """Den kompletten Moebel-Schritt ausfuehren.

    Args:
        annotations: alle Annotationen (wir filtern Moebel/Raum/Wand selbst).
        image: gerendertes Plan-Bild (numpy), fuer Crops. None -> nur Raum-Prior.
        scale_xy: plan-px -> render-px.
        px_to_m: render-px -> Meter (aus Tuer-Median kalibriert).
        classifier: EU-Vision-LLM-Transport. None -> generische Platzhalter.
        cache: optionaler Klassifikations-Cache (deterministische Re-Runs).
    """
    result = FurnitureResult()

    # Raum-Kontext: Label-Text, sonst LLM-UNABHAENGIG per OCR aus dem Plan gelesen.
    rooms = extract_rooms(annotations, scale_xy=scale_xy, px_to_m=px_to_m, image=image)
    result.rooms = rooms
    result.metrics["furniture_rooms_detected"] = sum(1 for r in rooms if r.room_type)

    walls_m: List[BBoxM] = []
    for ann in annotations:
        if _is_wall(ann):
            bm = _annotation_bbox_m(ann, scale_xy, px_to_m)
            if bm is not None:
                walls_m.append(bm)

    furniture_anns = [a for a in annotations if _is_furniture(a)]
    result.metrics["furniture_labels"] = len(furniture_anns)
    if not furniture_anns:
        return result

    # --- Crops + Raum-Zuordnung vorbereiten ---
    inputs: List[ClassifyInput] = []
    meta: Dict[str, Tuple[BBoxM, Optional[RoomRegion]]] = {}
    for idx, ann in enumerate(furniture_anns):
        bbox_m = _annotation_bbox_m(ann, scale_xy, px_to_m)
        if bbox_m is None:
            continue
        bbox_id = str(ann.get("id") or f"furniture_{idx}")
        # Grosszuegiger Snap: Raum-Label ist der (kleine) Namens-Text, Moebel
        # liegen ueber den Raum verteilt -> Containment zuerst, sonst naechstes
        # Raum-Label bis ~6 m (Raum-Kontext als Hinweis, nicht harte Logik).
        room = assign_room(_center(bbox_m), rooms, max_snap_m=6.0)

        crop = None
        bbox_px = _ann_bbox_px(ann, scale_xy)
        if image is not None and bbox_px is not None:
            crop = extract_crop(image, bbox_px)

        inputs.append(ClassifyInput(
            bbox_id=bbox_id,
            crop=crop,
            room=room,
            bbox_long_edge_m=_long_edge(bbox_m),
        ))
        meta[bbox_id] = (bbox_m, room)

    # --- Klassifizieren (EU-LLM oder Fallback) ---
    usage_sink: Dict[str, Any] = {}
    classifications = classify_furniture(
        inputs, classifier=classifier, request_id=request_id, cache=cache,
        usage_sink=usage_sink,
    )
    result.classifications = classifications

    # --- Platzieren ---
    for bbox_id, (bbox_m, room) in meta.items():
        cls = classifications.get(bbox_id)
        if cls is None:
            continue
        pf = place_furniture(
            bbox_m, cls, walls=walls_m,
            room_id=(room.room_id if room else None),
        )
        result.placements.append(pf)

    result.metrics["furniture_placed"] = len(result.placements)
    result.metrics["furniture_items"] = result.n_items
    result.metrics["furniture_llm_classified"] = sum(
        1 for c in classifications.values() if c.source == "llm"
    )
    result.metrics["furniture_runs"] = sum(
        1 for p in result.placements if len(p.items) > 1
    )
    # Echte LLM-Token-Usage + Kosten (Transparenz; vom Endpoint via Transport geliefert).
    if usage_sink:
        result.metrics["furniture_llm_input_tokens"] = usage_sink.get("prompt_tokens", 0)
        result.metrics["furniture_llm_output_tokens"] = usage_sink.get("completion_tokens", 0)
        result.metrics["furniture_llm_calls"] = usage_sink.get("calls", 0)
        if usage_sink.get("model"):
            result.metrics["furniture_llm_model"] = usage_sink["model"]
        if usage_sink.get("cost_usd") is not None:
            result.metrics["furniture_llm_cost_usd"] = usage_sink["cost_usd"]
    return result


def append_furniture_to_glb(
    glb_path,
    annotations: Sequence[Dict[str, Any]],
    image=None,
    *,
    scale_xy: Tuple[float, float] = (1.0, 1.0),
    px_to_m: float = 1.0,
    classifier: Optional[FurnitureClassifier] = None,
    cache: Optional[Dict[str, FurnitureClassification]] = None,
    request_id: str = "furniture",
) -> Tuple[FurnitureResult, int]:
    """Moebel platzieren und als Knoten in eine bestehende GLB schreiben.

    Laedt die Szene aus ``glb_path``, fuehrt den Moebel-Schritt aus, haengt die
    Knoten an und re-exportiert in dieselbe Datei. Bestehende Geometrie
    (Waende/Tueren/Fenster) bleibt unberuehrt. Gibt (Result, n_knoten) zurueck.

    Soft-fail: bei jedem Fehler bleibt die GLB unveraendert, n_knoten=0.
    """
    from pathlib import Path

    from .mesh import append_furniture_to_scene

    result = run_furniture_step(
        annotations, image, scale_xy=scale_xy, px_to_m=px_to_m,
        classifier=classifier, cache=cache, request_id=request_id,
    )
    if not result.placements:
        return result, 0

    try:
        import trimesh
        path = Path(glb_path)
        scene = trimesh.load(str(path), force="scene")
        added = append_furniture_to_scene(scene, result.placements)
        if added:
            scene.export(str(path))
        result.metrics["furniture_nodes_added"] = added
        return result, added
    except Exception as exc:  # noqa: BLE001 — darf die BIM-Erzeugung nie brechen
        logger.warning("append_furniture_to_glb failed: %s", exc)
        result.metrics["furniture_nodes_added"] = 0
        return result, 0
