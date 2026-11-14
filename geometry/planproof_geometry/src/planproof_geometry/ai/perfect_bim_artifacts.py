"""Ergebnis-Artefakte der perfect_bim-Pipeline.

Seit ADR-014 ist perfect_bim die Produktions-Pipeline — sie lieferte aber nur
GLB + IFC. Die Desktop-Ergebnis-Features (Bauteil-Konsole, Konfidenz-Heatmap,
Compare-Sync, Mengenliste/GAEB/DIN-276, Material-Tab) sind gegen den
V1/V3-Bundle-Vertrag gebaut (``semantics.json``, ``quantity_takeoff.csv``,
``materializations.json``) und liefen seit dem Umstieg leer. Dieses Modul
stellt den Vertrag aus der Mensch-First-Wahrheit wieder her:

* :func:`build_semantics` — Segmente 1:1 aus der ``annotations.json``
  (Pixel-Bbox, Klasse, Seite, Konfidenz) plus ``stats.segments_by_class``.
  Die Klassen bleiben das rohe Label-Vokabular (``wall_exterior``,
  ``door_single`` ...), weil die GLB-Knoten der Tuer-Pipeline mit genau
  diesen Prefixen benannt sind (Viewer ``classMatches``).
* :func:`build_quantity_rows` — Mengen aus Label-Geometrie mal Tuer-
  kalibriertem Massstab. Es ist DIESELBE Kalibrierung, mit der die
  GLB-Geometrie gebaut wird (``merge._calibrate_px_to_m``) — Mengenliste
  und Modell sind damit konsistent.
* :func:`build_materializations` — Template-Zuweisung + DIN-Validierung
  ueber die Catalog-Rules-Engine inkl. ``manual_template_overrides.json``
  (SIMS-Switcher-Roundtrip).

Alle Builder sind pure Funktionen (testbar ohne Pipeline-Lauf);
:func:`write_result_artifacts` ist der soft-fail Orchestrator fuer den
Pipeline-Step — ein Artefakt-Fehler darf einen gruenen Lauf nie kippen.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from planproof_geometry.ai.manual_bridge import (
    _annotations_for_page,
    load_annotations_file,
)

logger = logging.getLogger(__name__)

# Standard-Tuerhoehe (DIN 18101-nahe Annahme) fuer die Flaechen-Schaetzung in
# der Mengenliste. Im Plan steht nur die Breite — die Hoehe ist eine
# ausgewiesene Annahme (Eigenschaft-Spalte), kein Messwert.
_DOOR_HEIGHT_M = 2.01

# Plausibles Wanddicken-Fenster (m). Ausserhalb -> Dicke gilt als nicht
# messbar (Label umfasst mehr als die Wand) und bleibt leer.
_WALL_THICKNESS_RANGE_M = (0.04, 0.60)

_WALL_CLASSES = {"wall", "wall_exterior", "wall_interior"}
_DOOR_CLASSES = {"door", "door_single", "door_double", "door_sliding"}
_WINDOW_CLASSES = {"window", "window_large"}
_COLUMN_CLASSES = {"column", "beam"}
_STAIR_CLASSES = {"stairs", "stairs_spiral"}

_CSV_HEADER = (
    "Kategorie;ID;Typ;Laenge_m;Hoehe_m;Dicke_m;"
    "Flaeche_m2;Volumen_m3;Material;Eigenschaft"
)


def _page_numbers(annotations_file: Dict[str, Any]) -> List[int]:
    """Alle Seiten mit Annotations: ``pages``-Keys oder Seite 1 (Top-Level)."""
    pages = annotations_file.get("pages")
    if isinstance(pages, dict) and pages:
        numbers = []
        for key in pages:
            try:
                numbers.append(int(key))
            except (TypeError, ValueError):
                continue
        if numbers:
            return sorted(set(numbers))
    return [1]


def _ann_bbox_px(ann: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """Bbox (x_min, y_min, x_max, y_max) in Annotation-Pixeln; Polygon -> Huelle."""
    if ann.get("type") == "polygon":
        points = ann.get("points") or []
        if len(points) < 3:
            return None
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        return (min(xs), min(ys), max(xs), max(ys))
    bbox = ann.get("bbox")
    if not bbox or len(bbox) < 4:
        return None
    x0, y0, x1, y1 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def calibrate_px_to_m(annotations_file: Dict[str, Any]) -> Optional[float]:
    """Meter pro Annotation-Pixel — identische Kalibrierung wie das Merge-Modul.

    Delegiert an ``merge._calibrate_px_to_m`` (Median der ``door_single``-
    Breiten gegen 0,85 m), damit Mengenliste und GLB-Geometrie nie
    auseinanderlaufen. ``scale_xy=(1,1)`` weil wir direkt im
    Annotation-Pixelraum rechnen. Kalibriert auf Seite 1; Fallback: alle
    Seiten zusammen (gleicher Plan-Massstab ueber Geschosse).
    """
    from planproof_geometry.ai.v3.perfect_bim._script_loader import load_merge_module

    merge = load_merge_module()
    page_one, _ = _annotations_for_page(annotations_file, 1)
    px_to_m = merge._calibrate_px_to_m(page_one, (1.0, 1.0))
    if px_to_m is not None:
        return float(px_to_m)
    combined: List[Dict[str, Any]] = []
    for page in _page_numbers(annotations_file):
        anns, _ = _annotations_for_page(annotations_file, page)
        combined.extend(anns)
    px_to_m = merge._calibrate_px_to_m(combined, (1.0, 1.0))
    return float(px_to_m) if px_to_m is not None else None


# ── Semantics ─────────────────────────────────────────────────────────────

def build_semantics(
    annotations_file: Dict[str, Any],
    *,
    model_glb_path: Path,
) -> Dict[str, Any]:
    """Volle ``semantics.json`` aus der Annotations-Wahrheit.

    Segment-Schema folgt dem Frontend-Accessor (``semanticsTypes.ts``):
    ``id``/``class``/``confidence``/``source``/``page`` + ``bbox_px`` als
    ``[x, y, w, h]``. ``image_size``/``page_sizes`` erlauben dem Client die
    Normalisierung auf [0..1] fuer Plan-Overlays (Heatmap, Compare-Sync).
    """
    segments: List[Dict[str, Any]] = []
    page_sizes: Dict[str, Dict[str, int]] = {}
    by_class: Dict[str, int] = {}

    for page in _page_numbers(annotations_file):
        anns, (img_w, img_h) = _annotations_for_page(annotations_file, page)
        page_sizes[str(page)] = {"width": img_w, "height": img_h}
        for idx, ann in enumerate(anns):
            cls = str(ann.get("class") or "unknown")
            bbox = _ann_bbox_px(ann)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            seg_id = str(ann.get("id") or f"p{page}_m{idx}")
            segments.append({
                "id": seg_id,
                "class": cls,
                "confidence": float(ann.get("confidence", 1.0)),
                "source": str(ann.get("source") or "manual"),
                "page": page,
                "bbox_px": [x0, y0, x1 - x0, y1 - y0],
            })
            by_class[cls] = by_class.get(cls, 0) + 1

    first_page = page_sizes.get(str(_page_numbers(annotations_file)[0]), {"width": 0, "height": 0})
    try:
        glb_size_kb = round(model_glb_path.stat().st_size / 1024, 1)
    except OSError:
        glb_size_kb = None

    return {
        "schema_version": "v3",
        "pipeline_mode": "perfect_bim",
        "model_glb": model_glb_path.name,
        "model_glb_size_kb": glb_size_kb,
        "image_size": first_page,
        "page_sizes": page_sizes,
        "segments": segments,
        "stats": {
            "model_format": "glb",
            "segments_by_class": by_class,
            "total_segments": len(segments),
        },
    }


# ── Mengenliste ───────────────────────────────────────────────────────────

def build_quantity_rows(
    annotations_file: Dict[str, Any],
    *,
    wall_height_m: float,
    px_to_m: float,
) -> List[Dict[str, Any]]:
    """Mengenzeilen (CSV-Vertrag ``quantityCsv.ts``) aus den Labels.

    Bewusst nur belastbare Groessen: Wandlaengen/-flaechen aus Bbox-Langseite
    mal Wandhoehe, Tuerbreiten aus der Bbox-KURZseite (die Bbox umfasst den
    Aufschwenkbogen — die kurze Seite ist die Oeffnungsbreite, identisch zur
    Kalibrierungs-Konvention), Fensterbreiten aus der Langseite. Raeume werden
    NICHT erfunden (ein Raumlabel-Bbox ist keine Raumflaeche).
    """
    rows: List[Dict[str, Any]] = []
    counters = {"wand": 0, "tuer": 0, "fenster": 0, "stuetze": 0}

    def _next_id(category: str, prefix: str) -> str:
        counters[category] += 1
        return f"{prefix}_{counters[category]:03d}"

    for page in _page_numbers(annotations_file):
        anns, _ = _annotations_for_page(annotations_file, page)
        for ann in anns:
            cls = str(ann.get("class") or "")
            bbox = _ann_bbox_px(ann)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            long_m = max(x1 - x0, y1 - y0) * px_to_m
            short_m = min(x1 - x0, y1 - y0) * px_to_m
            ann_ref = str(ann.get("id") or "")

            if cls in _WALL_CLASSES:
                wall_type = (
                    "exterior" if cls == "wall_exterior"
                    else "interior" if cls == "wall_interior"
                    else ""
                )
                lo, hi = _WALL_THICKNESS_RANGE_M
                thickness = short_m if lo <= short_m <= hi else None
                area = long_m * wall_height_m
                rows.append({
                    "category": "Wand",
                    "id": _next_id("wand", "WALL"),
                    "type": wall_type,
                    "length_m": long_m,
                    "height_m": wall_height_m,
                    "thickness_m": thickness,
                    "area_m2": area,
                    "volume_m3": area * thickness if thickness else None,
                    "material": "",
                    "property": f"aus Label {ann_ref} (Seite {page})",
                })
            elif cls in _DOOR_CLASSES:
                width = short_m
                rows.append({
                    "category": "Tuer",
                    "id": _next_id("tuer", "DOOR"),
                    "type": cls.removeprefix("door_") or "single",
                    "length_m": width,
                    "height_m": _DOOR_HEIGHT_M,
                    "thickness_m": None,
                    "area_m2": width * _DOOR_HEIGHT_M,
                    "volume_m3": None,
                    "material": "",
                    "property": f"Hoehe {_DOOR_HEIGHT_M} m angenommen; aus Label {ann_ref}",
                })
            elif cls in _WINDOW_CLASSES:
                rows.append({
                    "category": "Fenster",
                    "id": _next_id("fenster", "WIN"),
                    "type": "large" if cls == "window_large" else "standard",
                    "length_m": long_m,
                    "height_m": None,
                    "thickness_m": None,
                    "area_m2": None,
                    "volume_m3": None,
                    "material": "",
                    "property": f"Breite aus Label {ann_ref} (Seite {page})",
                })
            elif cls in _COLUMN_CLASSES:
                rows.append({
                    "category": "Stuetze",
                    "id": _next_id("stuetze", "COL"),
                    "type": cls,
                    "length_m": long_m,
                    "height_m": wall_height_m,
                    "thickness_m": short_m,
                    "area_m2": long_m * short_m,
                    "volume_m3": long_m * short_m * wall_height_m,
                    "material": "",
                    "property": f"Querschnitt aus Label {ann_ref} (Seite {page})",
                })
    return rows


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_quantity_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """CSV im Vertrag von ``quantityCsv.ts``: BOM, ``;``, Punkt-Dezimal."""
    lines = [_CSV_HEADER]
    for r in rows:
        lines.append(";".join([
            _fmt(r["category"]), _fmt(r["id"]), _fmt(r["type"]),
            _fmt(r["length_m"]), _fmt(r["height_m"]), _fmt(r["thickness_m"]),
            _fmt(r["area_m2"]), _fmt(r["volume_m3"]),
            _fmt(r["material"]), _fmt(r["property"]),
        ]))
    lines.append("")
    lines.append(f"ZUSAMMENFASSUNG;{len(rows)} Positionen")
    path.write_text("\ufeff" + "\n".join(lines), encoding="utf-8")


# ── Materializations ──────────────────────────────────────────────────────

def _locate_template_overrides(input_path: Optional[Path]) -> Optional[Path]:
    """Spiegelt die Sibling-Konvention aus ``template_assignment.py``."""
    if input_path is None:
        return None
    conv = input_path.parent / "plan_bim_data" / input_path.stem / "manual_template_overrides.json"
    if conv.exists():
        return conv
    sibling = input_path.parent / f"{input_path.stem}.manual_template_overrides.json"
    if sibling.exists():
        return sibling
    return None


def build_materializations(
    annotations_file: Dict[str, Any],
    *,
    wall_height_m: float,
    px_to_m: Optional[float],
    input_path: Optional[Path] = None,
    low_confidence_threshold: float = 0.5,
) -> Optional[Dict[str, Any]]:
    """Template-Zuweisungen + DIN-Validierung fuer die gelabelten Bauteile.

    Element-IDs sind die Annotation-IDs — exakt die Keys, unter denen der
    SIMS-Switcher im Desktop seine Overrides ablegt (Roundtrip ueber
    ``manual_template_overrides.json``). Gibt ``None`` zurueck, wenn der
    Template-Katalog nicht ladbar ist (z. B. Seed-Verzeichnis fehlt im
    Container) — dann wird KEINE Datei geschrieben statt einer leeren Luege.
    """
    from planproof_geometry.ai.template_assignment import TemplateAssignmentStep
    from planproof_geometry.catalog.assignment_rules import (
        DetectedElement,
        apply_forced_overrides,
        assign_element,
        validate_templates_exist,
    )
    from planproof_geometry.catalog.din_validator import validate_assignments

    loader = TemplateAssignmentStep()
    if not loader._load_catalog():
        return None

    detected: List[DetectedElement] = []
    for page in _page_numbers(annotations_file):
        anns, _ = _annotations_for_page(annotations_file, page)
        for idx, ann in enumerate(anns):
            cls = str(ann.get("class") or "")
            bbox = _ann_bbox_px(ann)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            long_m = max(x1 - x0, y1 - y0) * px_to_m if px_to_m else None
            short_m = min(x1 - x0, y1 - y0) * px_to_m if px_to_m else None
            elem_id = str(ann.get("id") or f"p{page}_m{idx}")

            if cls in _WALL_CLASSES:
                lo, hi = _WALL_THICKNESS_RANGE_M
                thickness = short_m if short_m is not None and lo <= short_m <= hi else None
                detected.append(DetectedElement(
                    id=elem_id, kind="WALL",
                    length_m=long_m, thickness_m=thickness,
                    height_m=wall_height_m, width_m=None, area_m2=None,
                    is_external=(
                        True if cls == "wall_exterior"
                        else False if cls == "wall_interior"
                        else None
                    ),
                    is_load_bearing=None, is_ground_slab=None,
                    label=None, room_type_hint=None,
                ))
            elif cls in _DOOR_CLASSES:
                detected.append(DetectedElement(
                    id=elem_id, kind="DOOR",
                    length_m=None, thickness_m=None,
                    height_m=_DOOR_HEIGHT_M, width_m=short_m, area_m2=None,
                    is_external=None, is_load_bearing=None, is_ground_slab=None,
                    label=None, room_type_hint=None,
                ))
            elif cls in _WINDOW_CLASSES:
                detected.append(DetectedElement(
                    id=elem_id, kind="WINDOW",
                    length_m=None, thickness_m=None,
                    height_m=None, width_m=long_m, area_m2=None,
                    is_external=None, is_load_bearing=None, is_ground_slab=None,
                    label=None, room_type_hint=None,
                ))
            elif cls in _STAIR_CLASSES:
                detected.append(DetectedElement(
                    id=elem_id, kind="STAIR",
                    length_m=long_m, thickness_m=None,
                    height_m=wall_height_m, width_m=short_m, area_m2=None,
                    is_external=None, is_load_bearing=None, is_ground_slab=None,
                    label=None, room_type_hint=None,
                ))

    if not detected:
        return None

    assignments = validate_templates_exist(
        [assign_element(d) for d in detected], loader._template_keys,
    )

    overrides_path = _locate_template_overrides(input_path)
    if overrides_path is not None:
        try:
            raw = json.loads(overrides_path.read_text(encoding="utf-8"))
            forced = {
                eid: v["template_key"]
                for eid, v in (raw.get("overrides") or {}).items()
                if isinstance(v, dict) and v.get("template_key")
            }
            assignments = apply_forced_overrides(assignments, forced)
        except Exception as exc:
            logger.warning("manual_template_overrides.json lesen fehlgeschlagen: %s", exc)

    violations = validate_assignments(assignments, loader._templates_by_key)
    flagged = [
        a.element_id for a in assignments
        if a.template_key is None or a.confidence < low_confidence_threshold
    ]

    return {
        "source": "perfect_bim_annotations",
        "template_assignments": [
            {
                "element_id": a.element_id,
                "element_kind": a.element_kind,
                "template_key": a.template_key,
                "template_version": a.template_version,
                "confidence": a.confidence,
                "rationale": a.rationale,
            }
            for a in assignments
        ],
        "din_violations": [
            {
                "element_id": v.element_id,
                "element_kind": v.element_kind,
                "din_reference": v.din_reference,
                "severity": v.severity,
                "message": v.message,
                "actual_value": v.actual_value,
                "required_value": v.required_value,
            }
            for v in violations
        ],
        "flagged_assignments": flagged,
    }


# ── Orchestrator (Pipeline-Step) ──────────────────────────────────────────

def write_result_artifacts(
    solution_dir: Path,
    *,
    ann_path: Path,
    model_glb_path: Path,
    input_path: Optional[Path],
    wall_height_m: float,
    log: Callable[[str], None],
) -> Dict[str, Any]:
    """Schreibt semantics.json / quantity_takeoff.csv / materializations.json /
    meta.json in den Solution-Ordner (landet damit im inneren Solution-Zip UND
    unter ``outputs/<stem>_solution/`` im Bundle — beide Client-Lookups).

    Jeder Teil ist soft-fail: ein Artefakt-Fehler kippt den Lauf nicht,
    sondern wird geloggt. Rueckgabe: Metrik-Snippets fuer ``state.metrics``.
    """
    metrics: Dict[str, Any] = {}
    annotations_file = load_annotations_file(ann_path)
    if annotations_file is None:
        log("perfect_bim: artifacts skipped — annotations.json nicht lesbar")
        return metrics

    semantics: Optional[Dict[str, Any]] = None
    try:
        semantics = build_semantics(annotations_file, model_glb_path=model_glb_path)
        (solution_dir / "semantics.json").write_text(
            json.dumps(semantics, indent=2), encoding="utf-8",
        )
        stats = semantics["stats"]
        metrics["segments_by_class"] = stats["segments_by_class"]
        metrics["total_segments"] = stats["total_segments"]
        metrics["semantics_path"] = str(solution_dir / "semantics.json")
        log(f"perfect_bim: semantics.json — {stats['total_segments']} Segmente")
    except Exception as exc:
        log(f"perfect_bim: semantics.json failed (soft) — {exc}")

    px_to_m: Optional[float] = None
    try:
        px_to_m = calibrate_px_to_m(annotations_file)
        if px_to_m is not None:
            metrics["px_to_m"] = round(px_to_m, 6)
    except Exception as exc:
        log(f"perfect_bim: scale calibration failed (soft) — {exc}")

    if px_to_m is not None:
        try:
            rows = build_quantity_rows(
                annotations_file, wall_height_m=wall_height_m, px_to_m=px_to_m,
            )
            if rows:
                write_quantity_csv(rows, solution_dir / "quantity_takeoff.csv")
                metrics["quantity_rows"] = len(rows)
                log(f"perfect_bim: quantity_takeoff.csv — {len(rows)} Positionen")
        except Exception as exc:
            log(f"perfect_bim: quantity_takeoff.csv failed (soft) — {exc}")
    else:
        log("perfect_bim: Mengenliste uebersprungen — Massstab nicht kalibrierbar (keine door_single-Labels)")

    try:
        mat = build_materializations(
            annotations_file,
            wall_height_m=wall_height_m,
            px_to_m=px_to_m,
            input_path=input_path,
        )
        if mat is not None:
            (solution_dir / "materializations.json").write_text(
                json.dumps(mat, indent=2), encoding="utf-8",
            )
            metrics["template_assignments"] = {
                "total": len(mat["template_assignments"]),
                "assigned": sum(1 for a in mat["template_assignments"] if a["template_key"]),
                "flagged_low_confidence": len(mat["flagged_assignments"]),
                "din_violations": len(mat["din_violations"]),
            }
            log(
                f"perfect_bim: materializations.json — "
                f"{metrics['template_assignments']['assigned']}/"
                f"{metrics['template_assignments']['total']} zugewiesen, "
                f"{metrics['template_assignments']['din_violations']} DIN-Befunde"
            )
        else:
            log("perfect_bim: materializations uebersprungen — Katalog/Elemente nicht verfuegbar")
    except Exception as exc:
        log(f"perfect_bim: materializations.json failed (soft) — {exc}")

    try:
        meta = {
            "pipeline": "plan_to_bim",
            "pipeline_mode": "perfect_bim",
            "model_glb_size_kb": semantics.get("model_glb_size_kb") if semantics else None,
            **{k: v for k, v in metrics.items() if k != "semantics_path"},
        }
        (solution_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception as exc:
        log(f"perfect_bim: meta.json failed (soft) — {exc}")

    return metrics
