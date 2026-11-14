"""DXF laden + DWG->DXF-Bruecke + Einheiten -> Meter + Entity-Extraktion.

Liest eine ``.dxf`` direkt via ezdxf. Eine ``.dwg`` wird zuerst ueber einen austauschbaren
Konverter (``dwg_adapter``: ODA File Converter, LibreDWG dwg2dxf oder keiner) nach DXF umgesetzt.
DXF-Eingaben funktionieren immer ohne Zusatzsoftware.

Gelesen werden LINE, LWPOLYLINE, POLYLINE, TEXT, MTEXT und INSERT. Bloecke, die keine Tuer oder
kein Fenster sind, werden aufgeloest (verschachtelt, mit Massstab und Drehung, Tiefenlimit), damit
Waende in Bloecken nicht verloren gehen; Tuer- und Fensterbloecke bleiben Oeffnungen mit ihrer
orientierten Bounding-Box. HATCH wird nicht gelesen.

Alle extrahierten Koordinaten werden auf **Meter** skaliert (aus ``$INSUNITS``, mit
Heuristik-Fallback bei ``unitless``) und auf einen lokalen Ursprung (Minimum der Geometrie)
verschoben. Die Einheit wird immer auf Plausibilitaet geprueft (Gebaeude-Diagonale 3 bis 500 m);
bei Unplausibilitaet gibt es eine Warnung mit Vorschlag. ``config['dxf_unit']`` hat Vorrang.
"""

from __future__ import annotations

import logging
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import layer_classifier as lc
from .dwg_adapter import DwgConversionError, convert_dwg

logger = logging.getLogger(__name__)

Point2 = Tuple[float, float]

# Gebaeude-Diagonale, die als plausibel gilt (Meter)
PLAUSIBLE_DIAG_M = (3.0, 500.0)
DEFAULT_MAX_BLOCK_DEPTH = 3


class DxfLoadError(Exception):
    """DXF konnte nicht gelesen/geparst werden."""


# ── Einheiten ────────────────────────────────────────────────────────────
# $INSUNITS-Code -> Meter-Faktor. Quelle: DXF-Reference / ezdxf.units.
_INSUNITS_TO_M = {
    0: None,  # unitless -> Heuristik
    1: 0.0254,  # Inches
    2: 0.3048,  # Feet
    4: 0.001,  # Millimeters
    5: 0.01,  # Centimeters
    6: 1.0,  # Meters
    8: 0.0254e-6,  # Microinches (selten)
    9: 0.0254e-3,  # Mils
    10: 0.9144,  # Yards
    13: 1e-9,  # Nanometers
    14: 1e-6,  # Micrometers
    15: 0.1,  # Decimeters
    16: 10.0,  # Decameters
    17: 100.0,  # Hectometers
    18: 1000.0,  # Kilometers
}

_UNIT_LABEL_TO_M = {
    "mm": 0.001,
    "millimeter": 0.001,
    "millimeters": 0.001,
    "cm": 0.01,
    "centimeter": 0.01,
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "in": 0.0254,
    "inch": 0.0254,
    "inches": 0.0254,
    "ft": 0.3048,
    "feet": 0.3048,
    "foot": 0.3048,
}


@dataclass
class RawSegment:
    """Eine 2D-Linie (LINE oder ein Polyline-Abschnitt), in Meter."""

    a: Point2
    b: Point2
    layer: str
    source: int = -1  # laufende Nummer der Quell-Entity (Kanten einer Polylinie teilen sie)
    closed: bool = False  # Kante einer geschlossenen Polylinie (Kontur)

    def length(self) -> float:
        return math.hypot(self.b[0] - self.a[0], self.b[1] - self.a[1])


@dataclass
class RawInsert:
    """Eine Block-Referenz (INSERT), Position in Meter."""

    name: str
    layer: str
    pos: Point2
    rotation_deg: float
    xscale: float
    yscale: float
    # Ausdehnung des Blocks in seinem eigenen (gedrehten) Koordinatensystem, nicht die AABB
    width_m: Optional[float] = None
    height_m: Optional[float] = None
    attribs: dict = field(default_factory=dict)
    # Vier Ecken der orientierten Bounding-Box in Meter (Weltkoordinaten) und deren Mitte
    corners: Optional[List[Point2]] = None
    center: Optional[Point2] = None
    expanded: bool = False  # Inhalt wurde in Linien/Texte aufgeloest
    depth: int = 0  # 0 = direkt im Modellbereich


@dataclass
class RawText:
    """Eine TEXT/MTEXT-Annotation, Position in Meter."""

    text: str
    pos: Point2
    layer: str


@dataclass
class ExtractedDxf:
    """Aus der DXF extrahierte Roh-Geometrie (alles in Meter, origin-verschoben)."""

    segments: List[RawSegment]
    inserts: List[RawInsert]
    texts: List[RawText]
    unit_scale: float  # Faktor Zeichnungseinheit -> Meter
    unit_label: str  # "mm"/"cm"/"m"/... oder "unitless(heuristic)"
    origin: Point2  # subtrahierter Ursprung (in Meter, vor Shift)
    bbox: Tuple[Point2, Point2]  # (min, max) in Meter nach Shift
    insunits_code: Optional[int]
    warnings: List[str] = field(default_factory=list)
    unit_check: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)


# ── DWG -> DXF Bruecke ───────────────────────────────────────────────────


def _convert_dwg_to_dxf(dwg_path: Path, work_dir: Path, converter: Any = None) -> Path:
    """Kompatibilitaets-Einstieg: DWG->DXF ueber den ausgewaehlten Konverter (``dwg_adapter``).

    Ohne ``converter`` entscheidet die Umgebungsvariable ``CQPLAN_DWG_CONVERTER``; ohne beides
    gibt es die klare Fehlermeldung von ``NoConverter``. Fehler der Konverter werden nicht
    verschluckt.
    """
    return convert_dwg(Path(dwg_path), Path(work_dir), converter)


# ── Einheiten-Aufloesung ─────────────────────────────────────────────────


def _resolve_unit_scale(
    doc,
    config: dict,
    raw_bbox_diag: float,
) -> Tuple[float, str, Optional[int]]:
    """Liefert (meter_faktor, label, insunits_code).

    Prioritaet: config['dxf_unit'] > $INSUNITS > Heuristik aus Bbox-Diagonale.
    """
    forced = (config or {}).get("dxf_unit")
    if forced:
        f = _UNIT_LABEL_TO_M.get(str(forced).lower())
        if f:
            return f, str(forced).lower(), None

    code: Optional[int] = None
    try:
        code = int(doc.header.get("$INSUNITS", 0))
    except Exception:  # noqa: BLE001
        code = None

    if code is not None and code in _INSUNITS_TO_M and _INSUNITS_TO_M[code] is not None:
        factor = _INSUNITS_TO_M[code]
        label = {0.001: "mm", 0.01: "cm", 1.0: "m", 0.0254: "in", 0.3048: "ft"}.get(
            factor, f"x{factor}"
        )
        return factor, label, code

    # Heuristik fuer unitless: eine reale Gebaeude-Diagonale liegt grob bei
    # 5..200 m. Skaliere so, dass die Diagonale in dieses Fenster faellt.
    if raw_bbox_diag <= 0:
        return 1.0, "unitless(assumed-m)", code
    for factor, label in ((0.001, "mm"), (0.01, "cm"), (1.0, "m")):
        diag_m = raw_bbox_diag * factor
        if PLAUSIBLE_DIAG_M[0] <= diag_m <= PLAUSIBLE_DIAG_M[1]:
            return factor, f"unitless(heuristic:{label})", code
    # Default: groesste Plausibilitaet - wenn Rohwerte > 1000, mm; sonst m.
    if raw_bbox_diag > 1000:
        return 0.001, "unitless(heuristic:mm)", code
    return 1.0, "unitless(heuristic:m)", code


def _check_unit_plausibility(
    raw_diag: float, unit_scale: float, unit_label: str, code: Optional[int], forced: bool
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Prueft die Diagonale in Metern IMMER, nicht nur bei unitless. Gibt (Befund, Warnung)
    zurueck."""
    lo, hi = PLAUSIBLE_DIAG_M
    diag_m = raw_diag * unit_scale
    plausible = lo <= diag_m <= hi
    suggestion: Optional[str] = None
    if not plausible:
        for label, factor in (("mm", 0.001), ("cm", 0.01), ("m", 1.0)):
            if lo <= raw_diag * factor <= hi and abs(factor - unit_scale) > 1e-12:
                suggestion = label
                break
    check: Dict[str, Any] = {
        "plausible": plausible,
        "diag_m": round(diag_m, 3),
        "unit": unit_label,
        "insunits": code,
        "forced": forced,
        "range_m": [lo, hi],
        "suggestion": suggestion,
    }
    if plausible:
        return check, None
    herkunft = (
        "per config dxf_unit gesetzt"
        if forced
        else (f"laut $INSUNITS={code}" if code not in (None, 0) else "per Heuristik")
    )
    text = (
        f"Einheit unplausibel: Die Zeichnung ist in der Einheit {unit_label} ({herkunft}) "
        f"{diag_m:.4g} m groß (Diagonale); erwartet sind {lo:g} bis {hi:g} m."
    )
    if suggestion:
        text += (
            f" Mit {suggestion} wären es {raw_diag * _UNIT_LABEL_TO_M[suggestion]:.4g} m. "
            f"Wenn das stimmt, config dxf_unit={suggestion!r} setzen; "
            "die Einheit wurde nicht geändert."
        )
    else:
        text += (
            " Keine der Einheiten mm, cm, m ergibt eine plausible Größe; config dxf_unit prüfen."
        )
    return check, text


# ── Geometrie-Extraktion ─────────────────────────────────────────────────


def _polyline_segments(points: List[Point2], closed: bool) -> List[Tuple[Point2, Point2]]:
    segs: List[Tuple[Point2, Point2]] = []
    for k in range(len(points) - 1):
        segs.append((points[k], points[k + 1]))
    if closed and len(points) >= 3:
        segs.append((points[-1], points[0]))
    return segs


def _text_anchor(e) -> Optional[Point2]:
    """Bezugspunkt eines TEXT/MTEXT (bei ausgerichtetem TEXT der Ausrichtungspunkt)."""
    try:
        if e.dxftype() == "TEXT":
            _align, p1, p2 = e.get_placement()
            p = p2 if (p2 is not None and e.dxf.hasattr("align_point")) else p1
            return (float(p.x), float(p.y))
        ins = e.dxf.insert
        return (float(ins.x), float(ins.y))
    except Exception:  # noqa: BLE001
        try:
            ins = e.dxf.insert
            return (float(ins.x), float(ins.y))
        except Exception:  # noqa: BLE001
            return None


class _Collector:
    """Sammelt Linien, Texte und Inserts in Zeichnungseinheiten (Block-Inhalte aufgeloest)."""

    def __init__(self, doc, max_depth: int) -> None:
        self.doc = doc
        self.max_depth = max_depth
        self.lines: List[Tuple[Point2, Point2, str, int, bool]] = []
        self.texts: List[Tuple[str, Point2, str]] = []
        self.inserts: List[Tuple[Any, str, int, bool]] = []  # (entity, eff_layer, depth, expanded)
        self.warnings: List[str] = []
        self.hatches = 0
        self.expanded_blocks = 0
        self._source = 0
        self._bbox_cache: Dict[str, Optional[Tuple[float, float, float, float]]] = {}
        self._depth_warned: set = set()

    def block_bbox(self, name: str) -> Optional[Tuple[float, float, float, float]]:
        """Lokale Bounding-Box des Blocks relativ zum Basispunkt: (minx, miny, maxx, maxy)."""
        if name in self._bbox_cache:
            return self._bbox_cache[name]
        res = None
        try:
            import ezdxf.bbox  # noqa: PLC0415

            layout = self.doc.blocks.get(name)
            if layout is not None:
                box = ezdxf.bbox.extents(layout, fast=False)
                if box.has_data:
                    bp = layout.block.dxf.base_point
                    res = (
                        float(box.extmin.x - bp.x),
                        float(box.extmin.y - bp.y),
                        float(box.extmax.x - bp.x),
                        float(box.extmax.y - bp.y),
                    )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Block-Bbox fehlgeschlagen (%s): %s", name, exc)
        self._bbox_cache[name] = res
        return res

    def walk(self, entities, parent_layer: Optional[str], depth: int) -> None:
        for e in entities:
            etype = e.dxftype()
            layer = str(getattr(e.dxf, "layer", "0"))
            if layer == "0" and parent_layer is not None:
                layer = parent_layer  # Layer 0 im Block erbt den Layer des INSERT
            try:
                if etype == "LINE":
                    s, t = e.dxf.start, e.dxf.end
                    self._source += 1
                    self.lines.append(
                        (
                            (float(s.x), float(s.y)),
                            (float(t.x), float(t.y)),
                            layer,
                            self._source,
                            False,
                        )
                    )
                elif etype == "LWPOLYLINE":
                    pts = [(float(x), float(y)) for x, y, *_ in e.get_points("xyseb")]
                    closed = bool(getattr(e, "closed", False) or e.is_closed)
                    self._source += 1
                    for a, b in _polyline_segments(pts, closed):
                        self.lines.append((a, b, layer, self._source, closed))
                elif etype == "POLYLINE":
                    if getattr(e, "is_poly_face_mesh", False) or getattr(
                        e, "is_polygon_mesh", False
                    ):
                        continue
                    pts = [(float(v.dxf.location.x), float(v.dxf.location.y)) for v in e.vertices]
                    closed = bool(getattr(e, "is_closed", False))
                    self._source += 1
                    for a, b in _polyline_segments(pts, closed):
                        self.lines.append((a, b, layer, self._source, closed))
                elif etype in ("TEXT", "MTEXT"):
                    txt = e.plain_text() if hasattr(e, "plain_text") else str(e.dxf.text)
                    pos = _text_anchor(e)
                    if pos is not None:
                        self.texts.append((str(txt), pos, layer))
                elif etype == "HATCH":
                    self.hatches += 1
                elif etype == "INSERT":
                    self._insert(e, layer, depth)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Entity %s uebersprungen: %s", etype, exc)

    def _insert(self, e, layer: str, depth: int) -> None:
        name = str(e.dxf.name)
        cls = lc.classify_block(name, layer).cls
        is_opening = cls in lc.OPENING_CLASSES
        # Nur Bloecke aufloesen, die Waende, Raeume oder Unbekanntes enthalten koennen
        wants_expand = not is_opening and cls not in (
            lc.FURNITURE,
            lc.STAIR,
            lc.COLUMN,
            lc.RAILING,
            lc.ANNOTATION,
        )
        expanded = False
        if wants_expand:
            if depth + 1 > self.max_depth:
                if name not in self._depth_warned:
                    self._depth_warned.add(name)
                    self.warnings.append(
                        f"Blocktiefe {self.max_depth} erreicht: Block {name!r} wird nicht weiter "
                        "aufgelöst, sein Inhalt fehlt (config max_block_depth erhöhen)."
                    )
            else:
                expanded = True
                self.expanded_blocks += 1
                self.walk(e.virtual_entities(), layer, depth + 1)
        self.inserts.append((e, layer, depth, expanded))


def _insert_geometry(
    e, bbox_local: Optional[Tuple[float, float, float, float]], unit_scale: float
) -> Tuple[Optional[float], Optional[float], Optional[List[Point2]], Optional[Point2]]:
    """Orientierte Bounding-Box eines INSERT in Meter (unverschoben): (Breite, Hoehe, Ecken,
    Mitte)."""
    if bbox_local is None:
        return None, None, None, None
    ins = e.dxf.insert
    xs = float(getattr(e.dxf, "xscale", 1.0) or 1.0)
    ys = float(getattr(e.dxf, "yscale", 1.0) or 1.0)
    rot = math.radians(float(getattr(e.dxf, "rotation", 0.0) or 0.0))
    c, s = math.cos(rot), math.sin(rot)
    x0, y0, x1, y1 = bbox_local
    corners: List[Point2] = []
    for lx, ly in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        px, py = lx * xs, ly * ys
        wx = float(ins.x) + px * c - py * s
        wy = float(ins.y) + px * s + py * c
        corners.append((wx * unit_scale, wy * unit_scale))
    w = abs(x1 - x0) * abs(xs) * unit_scale
    h = abs(y1 - y0) * abs(ys) * unit_scale
    cx = sum(p[0] for p in corners) / 4.0
    cy = sum(p[1] for p in corners) / 4.0
    return (w if w > 1e-6 else None), (h if h > 1e-6 else None), corners, (cx, cy)


def load_and_extract(
    input_path: str | Path, config: Optional[dict] = None, work_dir: Optional[str | Path] = None
) -> ExtractedDxf:
    """Laedt DWG/DXF, skaliert auf Meter, extrahiert Linien/Bloecke/Text.

    Parameters
    ----------
    input_path : Pfad zu .dwg oder .dxf
    config     : {'dxf_unit': 'mm'|'cm'|'m'|'in'|'ft'} erzwingt Einheit;
                 sonst $INSUNITS bzw. Heuristik. 'converter' waehlt den DWG-Konverter,
                 'max_block_depth' das Tiefenlimit der Blockaufloesung (Standard 3).
    work_dir   : Arbeitsverzeichnis fuer DWG->DXF-Konvertierung.
    """
    import ezdxf  # noqa: PLC0415

    config = config or {}
    p = Path(input_path)
    if not p.exists():
        raise DxfLoadError(f"Eingabedatei nicht gefunden: {p}")

    suffix = p.suffix.lower()
    if suffix == ".dwg":
        wd = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="dwgbim_"))
        dxf_path = _convert_dwg_to_dxf(p, wd, config.get("converter"))
    elif suffix == ".dxf":
        dxf_path = p
    else:
        raise DxfLoadError(f"Nicht unterstuetzte Endung '{suffix}' (erwarte .dwg/.dxf)")

    try:
        doc = ezdxf.readfile(str(dxf_path))
    except Exception as exc:  # noqa: BLE001
        # ezdxf.recover fuer leicht beschaedigte Dateien
        try:
            from ezdxf import recover  # noqa: PLC0415

            doc, _auditor = recover.readfile(str(dxf_path))
        except Exception as exc2:  # noqa: BLE001
            raise DxfLoadError(f"DXF-Parse-Fehler: {exc2}") from exc

    # --- Pass 1: Rohgeometrie in Zeichnungseinheiten sammeln (fuer Bbox) ---
    max_depth = int(config.get("max_block_depth", DEFAULT_MAX_BLOCK_DEPTH))
    col = _Collector(doc, max_depth)
    col.walk(doc.modelspace(), None, 0)
    raw_lines, raw_texts = col.lines, col.texts

    # Rohe Bbox aus Linien und Einfuegepunkten
    xs: List[float] = []
    ys: List[float] = []
    for a, b, *_ in raw_lines:
        xs += [a[0], b[0]]
        ys += [a[1], b[1]]
    for e, _lay, _d, _x in col.inserts:
        try:
            ins = e.dxf.insert
            xs.append(float(ins.x))
            ys.append(float(ins.y))
        except Exception:  # noqa: BLE001
            pass
    if not xs:
        raise DxfLoadError("DXF enthaelt keine verwertbare Geometrie (Linien/Bloecke).")
    raw_diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))

    unit_scale, unit_label, code = _resolve_unit_scale(doc, config, raw_diag)
    unit_check, unit_warning = _check_unit_plausibility(
        raw_diag, unit_scale, unit_label, code, forced=bool(config.get("dxf_unit"))
    )

    # Ursprung = Bbox-Min in Metern, damit Koordinaten klein/positiv werden
    ox = min(xs) * unit_scale
    oy = min(ys) * unit_scale

    def to_m(pt: Point2) -> Point2:
        return (pt[0] * unit_scale - ox, pt[1] * unit_scale - oy)

    # --- Pass 2: in Meter konvertieren ---
    segments = [
        RawSegment(a=to_m(a), b=to_m(b), layer=lyr, source=src, closed=closed)
        for a, b, lyr, src, closed in raw_lines
    ]

    inserts: List[RawInsert] = []
    for e, eff_layer, depth, expanded in col.inserts:
        try:
            ins = e.dxf.insert
            name = str(e.dxf.name)
            rot = float(getattr(e.dxf, "rotation", 0.0) or 0.0)
            xs_ = float(getattr(e.dxf, "xscale", 1.0) or 1.0)
            ys_ = float(getattr(e.dxf, "yscale", 1.0) or 1.0)
            w, h, corners, center = _insert_geometry(e, col.block_bbox(name), unit_scale)
            attribs: dict = {}
            try:
                for att in e.attribs:
                    attribs[str(att.dxf.tag)] = str(att.dxf.text)
            except Exception:  # noqa: BLE001
                pass
            inserts.append(
                RawInsert(
                    name=name,
                    layer=eff_layer,
                    pos=to_m((float(ins.x), float(ins.y))),
                    rotation_deg=rot,
                    xscale=xs_,
                    yscale=ys_,
                    width_m=w,
                    height_m=h,
                    attribs=attribs,
                    corners=[(q[0] - ox, q[1] - oy) for q in corners] if corners else None,
                    center=(center[0] - ox, center[1] - oy) if center else None,
                    expanded=expanded,
                    depth=depth,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("INSERT uebersprungen: %s", exc)

    texts = [RawText(text=t, pos=to_m(pos), layer=lyr) for t, pos, lyr in raw_texts]

    # Bbox nach Shift
    sxs = [c for seg in segments for c in (seg.a[0], seg.b[0])]
    sys_ = [c for seg in segments for c in (seg.a[1], seg.b[1])]
    bbox = ((min(sxs), min(sys_)), (max(sxs), max(sys_))) if sxs else ((0.0, 0.0), (0.0, 0.0))

    warnings: List[str] = []
    if code in (None, 0) and not config.get("dxf_unit"):
        warnings.append(
            f"Zeichnung ohne explizite Einheit ($INSUNITS={code}); "
            f"Heuristik gewaehlt: {unit_label}. Bei falschem Massstab "
            f"config['dxf_unit'] setzen."
        )
    if unit_warning:
        warnings.append(unit_warning)
    warnings.extend(col.warnings)

    logger.info(
        "DXF geladen: %d Linien, %d Inserts, %d Texte, Einheit=%s (scale=%g), bbox=%.1fx%.1f m",
        len(segments),
        len(inserts),
        len(texts),
        unit_label,
        unit_scale,
        bbox[1][0] - bbox[0][0],
        bbox[1][1] - bbox[0][1],
    )

    return ExtractedDxf(
        segments=segments,
        inserts=inserts,
        texts=texts,
        unit_scale=unit_scale,
        unit_label=unit_label,
        origin=(ox, oy),
        bbox=bbox,
        insunits_code=code,
        warnings=warnings,
        unit_check=unit_check,
        stats={
            "hatch_entities_ignored": col.hatches,
            "blocks_expanded": col.expanded_blocks,
            "max_block_depth": max_depth,
        },
    )


__all__ = [
    "load_and_extract",
    "ExtractedDxf",
    "RawSegment",
    "RawInsert",
    "RawText",
    "DxfLoadError",
    "DwgConversionError",
    "PLAUSIBLE_DIAG_M",
]
