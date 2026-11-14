"""DXF laden + DWG->DXF-Bruecke + Einheiten -> Meter + Entity-Extraktion.

Liest eine ``.dxf`` direkt via ezdxf. Eine ``.dwg`` wird zuerst nach DXF
konvertiert ueber eine pluggbare Bruecke (Reihenfolge):

  1. ezdxf ``odafc``-Addon (nutzt ODA File Converter, falls konfiguriert)
  2. ``ODAFileConverter`` CLI (env ``ODA_FILE_CONVERTER`` oder auf PATH)
  3. ``dwg2dxf`` (LibreDWG, env ``DWG2DXF`` oder auf PATH)

Findet sich keine Bruecke, wirft ``DwgConversionError`` mit klarer Anleitung —
DXF-Eingaben funktionieren immer ohne Zusatzsoftware.

Alle extrahierten Koordinaten werden auf **Meter** skaliert (aus ``$INSUNITS``,
mit Heuristik-Fallback bei ``unitless``) und auf einen lokalen Ursprung
(Bounding-Box-Min der Geometrie) verschoben, damit Downstream in kleinen,
positiven Meter-Koordinaten rechnet.
"""
from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

Point2 = Tuple[float, float]


class DxfLoadError(Exception):
    """DXF konnte nicht gelesen/geparst werden."""


class DwgConversionError(Exception):
    """DWG->DXF-Konvertierung nicht moeglich (keine Bruecke verfuegbar)."""


# ── Einheiten ────────────────────────────────────────────────────────────
# $INSUNITS-Code -> Meter-Faktor. Quelle: DXF-Reference / ezdxf.units.
_INSUNITS_TO_M = {
    0: None,        # unitless -> Heuristik
    1: 0.0254,      # Inches
    2: 0.3048,      # Feet
    4: 0.001,       # Millimeters
    5: 0.01,        # Centimeters
    6: 1.0,         # Meters
    8: 0.0254e-6,   # Microinches (selten)
    9: 0.0254e-3,   # Mils
    10: 0.9144,     # Yards
    13: 1e-9,       # Nanometers
    14: 1e-6,       # Micrometers
    15: 0.1,        # Decimeters
    16: 10.0,       # Decameters
    17: 100.0,      # Hectometers
    18: 1000.0,     # Kilometers
}

_UNIT_LABEL_TO_M = {
    "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
    "cm": 0.01, "centimeter": 0.01,
    "m": 1.0, "meter": 1.0, "meters": 1.0,
    "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
    "ft": 0.3048, "feet": 0.3048, "foot": 0.3048,
}


@dataclass
class RawSegment:
    """Eine 2D-Linie (LINE oder ein Polyline-Abschnitt), in Meter."""
    a: Point2
    b: Point2
    layer: str

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
    width_m: Optional[float] = None   # aus Block-Bbox/ATTRIB, falls ermittelbar
    height_m: Optional[float] = None
    attribs: dict = field(default_factory=dict)


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
    unit_scale: float                 # Faktor Zeichnungseinheit -> Meter
    unit_label: str                   # "mm"/"cm"/"m"/... oder "unitless(heuristic)"
    origin: Point2                    # subtrahierter Ursprung (in Meter, vor Shift)
    bbox: Tuple[Point2, Point2]       # (min, max) in Meter nach Shift
    insunits_code: Optional[int]
    warnings: List[str] = field(default_factory=list)


# ── DWG -> DXF Bruecke ───────────────────────────────────────────────────

def _convert_dwg_to_dxf(dwg_path: Path, work_dir: Path) -> Path:
    """Konvertiert DWG->DXF ueber die erste verfuegbare Bruecke."""
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dxf = work_dir / (dwg_path.stem + ".dxf")

    # 1) ezdxf odafc-Addon (nutzt ODA File Converter, falls registriert)
    try:
        from ezdxf.addons import odafc  # type: ignore

        if getattr(odafc, "is_installed", lambda: False)():
            logger.info("DWG->DXF via ezdxf odafc-Addon")
            doc = odafc.readfile(str(dwg_path))
            doc.saveas(str(out_dxf))
            if out_dxf.exists():
                return out_dxf
    except Exception as exc:  # noqa: BLE001
        logger.debug("odafc-Addon nicht nutzbar: %s", exc)

    # 2) ODAFileConverter CLI
    oda = os.environ.get("ODA_FILE_CONVERTER") or shutil.which("ODAFileConverter")
    if oda:
        try:
            # ODAFileConverter <in_dir> <out_dir> <out_ver> <out_type> <recurse> <audit>
            in_dir = str(dwg_path.parent)
            cmd = [oda, in_dir, str(work_dir), "ACAD2018", "DXF", "0", "1", dwg_path.name]
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
            if out_dxf.exists():
                logger.info("DWG->DXF via ODAFileConverter CLI")
                return out_dxf
        except Exception as exc:  # noqa: BLE001
            logger.debug("ODAFileConverter CLI fehlgeschlagen: %s", exc)

    # 3) LibreDWG dwg2dxf
    dwg2dxf = os.environ.get("DWG2DXF") or shutil.which("dwg2dxf")
    if dwg2dxf:
        try:
            cmd = [dwg2dxf, "-o", str(out_dxf), str(dwg_path)]
            subprocess.run(cmd, check=True, capture_output=True, timeout=300)
            if out_dxf.exists():
                logger.info("DWG->DXF via LibreDWG dwg2dxf")
                return out_dxf
        except Exception as exc:  # noqa: BLE001
            logger.debug("dwg2dxf fehlgeschlagen: %s", exc)

    raise DwgConversionError(
        "DWG kann nicht gelesen werden: keine DWG->DXF-Bruecke gefunden. "
        "Optionen: (a) ODA File Converter installieren und Pfad in ENV "
        "ODA_FILE_CONVERTER setzen (bzw. ezdxf odafc registrieren), oder "
        "(b) LibreDWG 'dwg2dxf' auf PATH/ENV DWG2DXF bereitstellen, oder "
        "(c) die Zeichnung manuell als .dxf exportieren. "
        "DXF-Eingaben funktionieren immer ohne Zusatzsoftware."
    )


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
        if 3.0 <= diag_m <= 500.0:
            return factor, f"unitless(heuristic:{label})", code
    # Default: groesste Plausibilitaet — wenn Rohwerte > 1000, mm; sonst m.
    if raw_bbox_diag > 1000:
        return 0.001, "unitless(heuristic:mm)", code
    return 1.0, "unitless(heuristic:m)", code


# ── Geometrie-Extraktion ─────────────────────────────────────────────────

def _polyline_segments(points: List[Point2], closed: bool) -> List[Tuple[Point2, Point2]]:
    segs: List[Tuple[Point2, Point2]] = []
    for k in range(len(points) - 1):
        segs.append((points[k], points[k + 1]))
    if closed and len(points) >= 3:
        segs.append((points[-1], points[0]))
    return segs


def _insert_extent_m(insert, unit_scale: float) -> Tuple[Optional[float], Optional[float]]:
    """Block-Referenz-Bbox in Meter (width, height) ueber virtuelle Entities."""
    try:
        import ezdxf.bbox  # noqa: PLC0415

        box = ezdxf.bbox.extents([insert], fast=True)
        if box.has_data:
            size = box.size
            w = abs(float(size.x)) * unit_scale
            h = abs(float(size.y)) * unit_scale
            return (w if w > 1e-6 else None, h if h > 1e-6 else None)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Insert-Bbox fehlgeschlagen (%s): %s", getattr(insert.dxf, "name", "?"), exc)
    return None, None


def load_and_extract(input_path: str | Path, config: Optional[dict] = None,
                     work_dir: Optional[str | Path] = None) -> ExtractedDxf:
    """Laedt DWG/DXF, skaliert auf Meter, extrahiert Linien/Bloecke/Text.

    Parameters
    ----------
    input_path : Pfad zu .dwg oder .dxf
    config     : {'dxf_unit': 'mm'|'cm'|'m'|'in'|'ft'} erzwingt Einheit;
                 sonst $INSUNITS bzw. Heuristik.
    work_dir   : Arbeitsverzeichnis fuer DWG->DXF-Konvertierung.
    """
    import ezdxf  # noqa: PLC0415

    config = config or {}
    p = Path(input_path)
    if not p.exists():
        raise DxfLoadError(f"Eingabedatei nicht gefunden: {p}")

    wd = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="dwgbim_"))

    suffix = p.suffix.lower()
    if suffix == ".dwg":
        dxf_path = _convert_dwg_to_dxf(p, wd)
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

    msp = doc.modelspace()

    # --- Pass 1: Rohgeometrie in Zeichnungseinheiten sammeln (fuer Bbox) ---
    raw_lines: List[Tuple[Point2, Point2, str]] = []
    raw_inserts: List = []
    raw_texts: List[Tuple[str, Point2, str]] = []

    for e in msp:
        etype = e.dxftype()
        layer = str(getattr(e.dxf, "layer", "0"))
        try:
            if etype == "LINE":
                s = e.dxf.start
                t = e.dxf.end
                raw_lines.append(((float(s.x), float(s.y)), (float(t.x), float(t.y)), layer))
            elif etype == "LWPOLYLINE":
                pts = [(float(x), float(y)) for x, y, *_ in e.get_points("xyseb")]
                closed = bool(getattr(e, "closed", False) or e.is_closed)
                for a, b in _polyline_segments(pts, closed):
                    raw_lines.append((a, b, layer))
            elif etype == "POLYLINE":
                pts = [(float(v.dxf.location.x), float(v.dxf.location.y)) for v in e.vertices]
                closed = bool(getattr(e, "is_closed", False))
                for a, b in _polyline_segments(pts, closed):
                    raw_lines.append((a, b, layer))
            elif etype == "INSERT":
                raw_inserts.append(e)
            elif etype in ("TEXT", "MTEXT"):
                txt = e.plain_text() if hasattr(e, "plain_text") else str(e.dxf.text)
                ins = e.dxf.insert
                raw_texts.append((str(txt), (float(ins.x), float(ins.y)), layer))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Entity %s uebersprungen: %s", etype, exc)

    # Rohe Bbox aus Linien (oder Inserts, falls keine Linien)
    xs: List[float] = []
    ys: List[float] = []
    for a, b, _ in raw_lines:
        xs += [a[0], b[0]]
        ys += [a[1], b[1]]
    for e in raw_inserts:
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

    # Ursprung = Bbox-Min in Metern, damit Koordinaten klein/positiv werden
    ox = min(xs) * unit_scale
    oy = min(ys) * unit_scale

    def to_m(pt: Point2) -> Point2:
        return (pt[0] * unit_scale - ox, pt[1] * unit_scale - oy)

    # --- Pass 2: in Meter konvertieren ---
    segments = [RawSegment(a=to_m(a), b=to_m(b), layer=lyr) for a, b, lyr in raw_lines]

    inserts: List[RawInsert] = []
    for e in raw_inserts:
        try:
            ins = e.dxf.insert
            name = str(e.dxf.name)
            rot = float(getattr(e.dxf, "rotation", 0.0) or 0.0)
            xs_ = float(getattr(e.dxf, "xscale", 1.0) or 1.0)
            ys_ = float(getattr(e.dxf, "yscale", 1.0) or 1.0)
            w, h = _insert_extent_m(e, unit_scale)
            attribs: dict = {}
            try:
                for att in e.attribs:
                    attribs[str(att.dxf.tag)] = str(att.dxf.text)
            except Exception:  # noqa: BLE001
                pass
            inserts.append(RawInsert(
                name=name,
                layer=str(e.dxf.layer),
                pos=to_m((float(ins.x), float(ins.y))),
                rotation_deg=rot,
                xscale=xs_,
                yscale=ys_,
                width_m=w,
                height_m=h,
                attribs=attribs,
            ))
        except Exception as exc:  # noqa: BLE001
            logger.debug("INSERT uebersprungen: %s", exc)

    texts = [RawText(text=t, pos=to_m(pos), layer=lyr) for t, pos, lyr in raw_texts]

    # Bbox nach Shift
    sxs = [c for seg in segments for c in (seg.a[0], seg.b[0])]
    sys_ = [c for seg in segments for c in (seg.a[1], seg.b[1])]
    bbox = ((min(sxs), min(sys_)), (max(sxs), max(sys_))) if sxs else ((0.0, 0.0), (0.0, 0.0))

    warnings: List[str] = []
    if code in (None, 0):
        warnings.append(
            f"Zeichnung ohne explizite Einheit ($INSUNITS={code}); "
            f"Heuristik gewaehlt: {unit_label}. Bei falschem Massstab "
            f"config['dxf_unit'] setzen."
        )

    logger.info(
        "DXF geladen: %d Linien, %d Inserts, %d Texte, Einheit=%s (scale=%g), "
        "bbox=%.1fx%.1f m",
        len(segments), len(inserts), len(texts), unit_label, unit_scale,
        bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1],
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
    )


__all__ = [
    "load_and_extract",
    "ExtractedDxf",
    "RawSegment",
    "RawInsert",
    "RawText",
    "DxfLoadError",
    "DwgConversionError",
]
