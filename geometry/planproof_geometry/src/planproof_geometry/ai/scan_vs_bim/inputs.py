"""Input-Lader: manifest.json + plan_components.json -> PlanElement-Liste.

Das Job-Input-Zip (von der Dispatch-Route assembliert) enthaelt:

* ``manifest.json``       — {schema_version, project_id, plan_version_id,
                             up_axis, scan_filename, options{...}}
* ``plan_components.json`` — {version_id, up_axis, components: [dict, ...]}
                             mit exakt der dict-Form des Compare-Adapters
                             (id/node_path/name/class_code/length_m/width_m/
                             height_m/area_m2/center_x/center_y/center_z/
                             room_key/polygon)
* die Punktwolken-Datei    — .ply/.las/.laz/.e57/.xyz/.pts

Klassen-Normalisierung ist bewusst dieselbe kleine Stichwort-Tabelle wie in
``scan_plan_compare`` (Backend) — hier dupliziert, weil planproof_geometry keine
Backend-Imports kennt. Bei Aenderungen BEIDE Stellen pflegen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .types import PlanElement

_KNOWN_CLASSES = {"wall", "door", "window", "room", "floor", "ceiling", "stairs", "beam", "column", "roof"}
_CLASS_KEYWORDS = (
    ("wall", "wall"),
    ("wand", "wall"),
    ("door", "door"),
    ("tuer", "door"),
    ("tur", "door"),
    ("window", "window"),
    ("fenster", "window"),
    ("room", "room"),
    ("raum", "room"),
    ("floor", "floor"),
    ("boden", "floor"),
    ("ceiling", "ceiling"),
    ("decke", "ceiling"),
    ("stair", "stairs"),
    ("treppe", "stairs"),
    ("beam", "beam"),
    ("column", "column"),
    ("stuetze", "column"),
)

MANIFEST_NAME = "manifest.json"
PLAN_COMPONENTS_NAME = "plan_components.json"
SCAN_EXTENSIONS = frozenset({".ply", ".las", ".laz", ".e57", ".xyz", ".pts"})


def norm_class(class_code: Optional[str], name: Optional[str]) -> Optional[str]:
    if class_code and class_code.lower() in _KNOWN_CLASSES:
        return class_code.lower()
    nm = (name or "").lower()
    for kw, cls in _CLASS_KEYWORDS:
        if kw in nm:
            return cls
    return None


def find_inputs(input_paths: List[Path]) -> Tuple[Path, Path, Path]:
    """(manifest, plan_components, scan_file) aus den entpackten Job-Inputs."""
    manifest = None
    components = None
    scan = None
    for p in input_paths:
        low = p.name.lower()
        if low == MANIFEST_NAME:
            manifest = p
        elif low == PLAN_COMPONENTS_NAME:
            components = p
        elif p.suffix.lower() in SCAN_EXTENSIONS and scan is None:
            scan = p
    if manifest is None:
        raise ValueError("scan_vs_bim: manifest.json fehlt im Job-Input")
    if components is None:
        raise ValueError("scan_vs_bim: plan_components.json fehlt im Job-Input")
    if scan is None:
        raise ValueError("scan_vs_bim: keine Punktwolken-Datei im Job-Input")
    return manifest, components, scan


def load_manifest(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("manifest.json: Objekt erwartet")
    for key in ("project_id", "plan_version_id"):
        if not data.get(key):
            raise ValueError(f"manifest.json: Feld '{key}' fehlt")
    data.setdefault("schema_version", 1)
    data.setdefault("up_axis", "Y")
    data.setdefault("options", {})
    return data


def _canonical_aabb(
    c: Dict[str, Any], up_axis: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """``aabb_min``/``aabb_max`` (Welt) -> kanonischer Frame (x, h, z).

    Die achsentreue Welt-AABB stammt aus dem GLB-Walk der Dispatch-Route
    (die DB-Spalten length/width sind nach Magnitude sortiert — dort ist die
    Horizontal-Orientierung verloren). Bei up_axis Y ist der Welt-Frame
    bereits kanonisch; bei Z wird (x, y, z) -> (x, z, y) getauscht.
    """
    amin, amax = c.get("aabb_min"), c.get("aabb_max")
    if (
        not isinstance(amin, (list, tuple))
        or not isinstance(amax, (list, tuple))
        or len(amin) != 3
        or len(amax) != 3
    ):
        return None, None
    lo = np.array([float(v) for v in amin], dtype=np.float64)
    hi = np.array([float(v) for v in amax], dtype=np.float64)
    if up_axis.upper() == "Z":
        lo = lo[[0, 2, 1]]
        hi = hi[[0, 2, 1]]
    return np.minimum(lo, hi), np.maximum(lo, hi)


def load_plan_elements(path: Path, up_axis: str = "Y") -> List[PlanElement]:
    data = json.loads(path.read_text(encoding="utf-8"))
    comps = data.get("components") if isinstance(data, dict) else data
    if not isinstance(comps, list):
        raise ValueError("plan_components.json: Liste 'components' erwartet")
    out: List[PlanElement] = []
    for c in comps:
        if not isinstance(c, dict):
            continue
        aabb_min, aabb_max = _canonical_aabb(c, up_axis)
        poly = c.get("polygon")
        out.append(
            PlanElement(
                id=str(c.get("id") or c.get("node_path") or ""),
                node_path=str(c.get("node_path") or ""),
                name=c.get("name"),
                cls=norm_class(c.get("class_code"), c.get("name")),
                aabb_min=aabb_min,
                aabb_max=aabb_max,
                room_key=c.get("room_key"),
                polygon=poly if isinstance(poly, list) else None,
                area_m2=c.get("area_m2"),
            )
        )
    return out
