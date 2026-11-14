"""Layer-/Block-Name -> Bauteilklasse.

Die CAD-Semantik steckt in Layer- und Blocknamen. Dieser Klassifikator deckt
die vier in DACH ueblichen Namens-Dialekte ab:

  1. AIA / US National CAD Standard:  A-WALL, A-DOOR, A-GLAZ, A-COLS ...
  2. Oeffentliche DACH-Layerlisten (BLB NRW u.a.): A_Wand, A_Tuer_i, A_Fenster
  3. Tool-Deutsch (Allplan/ArchiCAD-Templates): Waende, Tueren, Fenster
  4. Frei/gemischt: Substring-Heuristik als Fallback

Wichtig fuer DACH: oeffentliche Standards **verbieten Umlaute**, daher tauchen
real ``Tuer``/``Stuetze``/``Moebel`` UND ``Tür``/``Stütze``/``Möbel`` auf. Wir
falten Umlaute (ue/oe/ae/ss) und matchen gegen beide Formen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Bauteilklassen (kompatibel mit build_scene/_KIND_MAP und ifc_export)
WALL = "wall"
WALL_EXTERIOR = "wall_exterior"
WALL_INTERIOR = "wall_interior"
DOOR = "door"
WINDOW = "window"
COLUMN = "column"
BEAM = "beam"
STAIR = "stair"
RAILING = "railing"
SLAB = "slab"
ROOM = "room"
FURNITURE = "furniture"
ANNOTATION = "annotation"  # Bemassung/Text/Schraffur -> ignorieren
UNKNOWN = "unknown"

# Klassen, die als Wand-Geometrie weiterverarbeitet werden
WALL_CLASSES = frozenset({WALL, WALL_EXTERIOR, WALL_INTERIOR})
# Klassen, die als Oeffnungs-Block verarbeitet werden
OPENING_CLASSES = frozenset({DOOR, WINDOW})


_UMLAUT_MAP = str.maketrans(
    {
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "ß": "ss",
        "Ä": "ae",
        "Ö": "oe",
        "Ü": "ue",
    }
)


def fold(name: str) -> str:
    """Lowercase + Umlaut-Faltung fuer robustes Keyword-Matching."""
    return (name or "").translate(_UMLAUT_MAP).lower()


# Keyword -> Klasse, **Reihenfolge = spezifisch vor allgemein**.
# Alle Keywords bereits gefaltet (ascii). Ein Treffer = "Keyword ist Substring
# des gefalteten Namens".
_KEYWORDS: Tuple[Tuple[str, str], ...] = (
    # Annotation zuerst aussortieren (sonst faengt "wand"/"raum" Bemassung)
    ("bemassung", ANNOTATION),
    ("vermassung", ANNOTATION),
    ("dimension", ANNOTATION),
    ("a-dims", ANNOTATION),
    ("a-anno", ANNOTATION),
    ("a-glaz-sill", WINDOW),  # AIA: Fenstersohlbank zaehlt zu Fenster
    # Gelaender/Bruestung vor Treppe (Geländer sitzt oft am Treppen-Layer)
    ("gelaender", RAILING),
    ("brustung", RAILING),
    ("bruestung", RAILING),
    ("railing", RAILING),
    ("handlauf", RAILING),
    # Aussen-/Innenwand spezifisch vor generischer Wand
    ("wand_aussen", WALL_EXTERIOR),
    ("wand-aussen", WALL_EXTERIOR),
    ("aussenwand", WALL_EXTERIOR),
    ("wand_a", WALL_EXTERIOR),
    ("aw_", WALL_EXTERIOR),
    ("a-wall-exterior", WALL_EXTERIOR),
    ("wand_innen", WALL_INTERIOR),
    ("wand-innen", WALL_INTERIOR),
    ("innenwand", WALL_INTERIOR),
    ("trennwand", WALL_INTERIOR),
    ("wand_i", WALL_INTERIOR),
    ("iw_", WALL_INTERIOR),
    ("a-wall-interior", WALL_INTERIOR),
    # Generische Wand
    ("a-wall", WALL),
    ("a_wand", WALL),
    ("ar_wand", WALL),
    ("wand", WALL),
    ("waende", WALL),
    ("wall", WALL),
    ("mauerwerk", WALL),
    ("mauer", WALL),
    # Tueren
    ("a-door", DOOR),
    ("a_tuer", DOOR),
    ("tueren", DOOR),
    ("tuer", DOOR),
    ("door", DOOR),
    # Fenster (AIA: GLAZ = Verglasung)
    ("a-glaz", WINDOW),
    ("a_fenster", WINDOW),
    ("fenster", WINDOW),
    ("window", WINDOW),
    ("verglasung", WINDOW),
    ("glaz", WINDOW),
    # Stuetzen / Traeger
    ("a-cols", COLUMN),
    ("stuetze", COLUMN),
    ("stutze", COLUMN),
    ("pfeiler", COLUMN),
    ("column", COLUMN),
    ("a-beam", BEAM),
    ("unterzug", BEAM),
    ("traeger", BEAM),
    ("beam", BEAM),
    # Treppen
    ("a-flor-strs", STAIR),
    ("treppe", STAIR),
    ("stiege", STAIR),
    ("stair", STAIR),
    # Decken / Boden / Platten
    ("a-slab", SLAB),
    ("bodenplatte", SLAB),
    ("decke", SLAB),
    ("a-flor-slab", SLAB),
    # Raeume
    ("raum-polygon", ROOM),
    ("raumstempel", ROOM),
    ("a-area", ROOM),
    ("a_raum", ROOM),
    ("raum", ROOM),
    ("room", ROOM),
    # Moebel / Einrichtung
    ("a-furn", FURNITURE),
    ("moebel", FURNITURE),
    ("einrichtung", FURNITURE),
    ("sanitaer", FURNITURE),
    ("furniture", FURNITURE),
)


@dataclass
class Classification:
    cls: str
    confidence: float
    signal: str  # welcher Name/Keyword matchte
    source: str  # "layer" | "block" | "default"
    # Tragend (True), nichttragend (False) oder aus dem Namen nicht ableitbar (None); nur bei
    # Wandklassen.
    bearing: Optional[bool] = None


# Namensbestandteile (gefaltet, durch Nicht-Alphanumerisches getrennt), die einen Layer zu
# Beschriftung,
# Schraffur oder Hilfsgeometrie machen. Sie gewinnen gegen jedes Bauteil-Stichwort: 300_WAND_TXT ist
# Text, A-WALL-PATT ist Schraffur, 0_RAUMPOLYGON_ABZUG ist eine Abzugsflaeche und kein Raum.
_ANNOTATION_TOKENS = frozenset(
    {
        "txt", "text", "tx", "kennz", "schraffur", "schraff", "patt", "pattern", "hatch",
        "annahme", "abzug", "oberhalb", "bemassung", "vermassung", "dim", "dims", "dimension",
        "anno", "annotation",
    }
)  # fmt: skip

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_AUF_TOKEN = re.compile(r"(?:^|[^a-z0-9])auf(?:[^a-z0-9]|$)")
_NICHT_TRAGEND = re.compile(r"nicht[^a-z0-9]*tragend|untragend")


def _tokens(folded: str) -> List[str]:
    return [t for t in _NON_ALNUM.split(folded) if t]


def _head(folded: str) -> str:
    """Der Teil vor einem Wort 'auf': 'fenster_auf_aussenwaenden' nennt das Fenster, die Wand ist
    nur der Wirt."""
    m = _AUF_TOKEN.search(folded)
    if m and m.start() > 0:
        return folded[: m.start()]
    return folded


def _keyword(folded: str) -> Optional[Tuple[str, str]]:
    for kw, cls in _KEYWORDS:
        if kw in folded:
            return cls, kw
    return None


def _match(name: str) -> Optional[Tuple[str, str]]:
    folded = _head(fold(name))
    toks = _tokens(folded)
    for t in toks:
        if t in _ANNOTATION_TOKENS:
            return ANNOTATION, t
    if "id" in toks:
        # ID_Raum ist der Raumstempel; ID_Tuer, ID_Fenst sind Kennzeichnungen
        m = _keyword(folded)
        return m if (m and m[0] == ROOM) else (ANNOTATION, "id")
    return _keyword(folded)


def _bearing_of(name: str) -> Optional[bool]:
    folded = fold(name)
    if _NICHT_TRAGEND.search(folded):
        return False
    if "tragend" in folded:
        return True
    return None


def classify_layer(layer_name: str) -> Classification:
    """Klassifiziert anhand des Layer-Namens."""
    m = _match(layer_name)
    if m:
        cls, kw = m
        if cls == ANNOTATION:
            return Classification(cls=cls, confidence=0.9, signal=kw, source="layer")
        # Exakte AIA-/Praefix-Treffer sind verlaesslicher als lose Substrings
        conf = 0.95 if (kw.startswith(("a-", "a_", "aw_", "iw_")) or "_" in kw) else 0.8
        bearing = _bearing_of(_head(fold(layer_name))) if cls in WALL_CLASSES else None
        return Classification(cls=cls, confidence=conf, signal=kw, source="layer", bearing=bearing)
    return Classification(cls=UNKNOWN, confidence=0.0, signal=layer_name, source="layer")


_TEXT_TOKENS = frozenset({"txt", "text", "tx"})


def is_generic_text_layer(layer_name: str) -> bool:
    """Allgemeiner Textlayer (A_Text, 0_TEXT, A-ANNO-TEXT) ohne Bauteil- oder Bemassungsbezug."""
    folded = _head(fold(layer_name))
    toks = _tokens(folded)
    if not any(t in _TEXT_TOKENS for t in toks):
        return False
    if any(t in _ANNOTATION_TOKENS - _TEXT_TOKENS - {"anno", "annotation"} for t in toks):
        return False  # Bemassung, Schraffur, Abzug ...
    kw = _keyword(folded)
    return kw is None or kw[0] == ANNOTATION


def is_room_text_layer(layer_name: str) -> bool:
    """Layer, auf dem Raumbezeichnungen stehen (A_Raum, A-AREA, Tx_Raum, ID_Raum, STIFT_7_RAUM)."""
    folded = fold(layer_name)
    toks = _tokens(folded)
    if "abzug" in toks:
        return False
    return "raum" in folded or "room" in folded or "area" in toks


def classify_block(block_name: str, layer_name: str = "") -> Classification:
    """Klassifiziert eine Block-Referenz.

    Block-Name hat Prioritaet; der Layer dient als Korrektiv/Fallback. Das
    faengt den ArchiCAD-Fall ab, in dem Tueren/Fenster auf dem Wand-Layer
    sitzen (dann traegt der Block-Name die Wahrheit) ebenso wie anonyme
    Bloecke (``*U123``), bei denen erst der Layer Semantik liefert.
    """
    bm = _match(block_name)
    if bm and bm[0] in (DOOR, WINDOW, STAIR, COLUMN, FURNITURE, RAILING):
        return Classification(cls=bm[0], confidence=0.9, signal=bm[1], source="block")

    lm = _match(layer_name)
    if lm and lm[0] in (DOOR, WINDOW, STAIR, COLUMN, FURNITURE, RAILING):
        return Classification(cls=lm[0], confidence=0.75, signal=lm[1], source="layer")

    # Block-Name lieferte irgendeine (auch Wand-)Klasse?
    if bm:
        return Classification(cls=bm[0], confidence=0.6, signal=bm[1], source="block")
    if lm:
        return Classification(cls=lm[0], confidence=0.5, signal=lm[1], source="layer")
    return Classification(cls=UNKNOWN, confidence=0.0, signal=block_name, source="block")


def match_template_key(layer_name: str, known_keys) -> Optional[str]:
    """DWG-Bonus: Layer-Name kodiert oft den exakten Aufbau
    (z.B. ``A_Wand_KS240_tragend`` -> Katalog ``iw_ks240_tragend``).

    Versucht, einen Katalog-Template-Key direkt aus dem Layer-Namen zu treffen,
    indem die Token-Ueberlappung maximiert wird. Liefert den besten Key oder
    None, wenn die Ueberlappung zu schwach ist.
    """
    folded = fold(layer_name)
    layer_tokens = {t for t in _tokenize(folded) if t}
    if not layer_tokens:
        return None
    best_key: Optional[str] = None
    best_score = 0.0
    for key in known_keys:
        ktoks = {t for t in _tokenize(fold(key)) if t}
        if not ktoks:
            continue
        overlap = layer_tokens & ktoks
        if not overlap:
            continue
        # Stark gewichten: Anteil der Template-Tokens, die im Layer vorkommen
        score = len(overlap) / len(ktoks)
        # Mind. ein "inhaltliches" Token (nicht nur aw/iw) muss matchen
        meaningful = {t for t in overlap if t not in ("aw", "iw", "a", "i", "w")}
        if not meaningful:
            continue
        if score > best_score:
            best_score = score
            best_key = key
    return best_key if best_score >= 0.5 else None


def _tokenize(s: str):
    out = []
    cur = ""
    for ch in s:
        if ch.isalnum():
            cur += ch
        else:
            if cur:
                out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


__all__ = [
    "Classification",
    "classify_layer",
    "classify_block",
    "match_template_key",
    "fold",
    "WALL",
    "WALL_EXTERIOR",
    "WALL_INTERIOR",
    "DOOR",
    "WINDOW",
    "COLUMN",
    "BEAM",
    "STAIR",
    "RAILING",
    "SLAB",
    "ROOM",
    "FURNITURE",
    "ANNOTATION",
    "UNKNOWN",
    "WALL_CLASSES",
    "OPENING_CLASSES",
]
