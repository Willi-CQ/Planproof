"""Tabellengetriebene Testmatrix fuer den Layer-Klassifikator.

Die Soll-Klassen stammen aus der Aufgabenstellung (Pipeline-Qualitaet, Defekt 12): echte
Layerlisten aus AIA, DACH-Praxis, Sachsen-SIB und ArchiCAD-Mustern. Layernamen duerfen ae/oe/ue
enthalten (es sind Layernamen); Texte und Meldungen dieses Moduls verwenden echte Umlaute.
"""

from __future__ import annotations

import pytest

from planproof_cad.cad import layer_classifier as lc

# (Layername, Soll-Klasse). WALL steht fuer "irgendeine Wandklasse" ausser dort, wo aussen/innen
# gefordert ist.
WAND = "WAND_BELIEBIG"

GEOMETRIE_SOLL = [
    # Wand
    ("A_Wand", WAND),
    ("300_WAND", WAND),
    ("300_WAND_TRAGEND", WAND),
    ("300_WAND_NICHTTRAGEND", WAND),
    ("A-WALL", WAND),
    ("A-WALL-FULL", WAND),
    ("A_Wand_a", lc.WALL_EXTERIOR),
    ("A_Wand_i", lc.WALL_INTERIOR),
    # Tuer
    ("A_Tuer", lc.DOOR),
    ("300_TUEROEFFNUNGEN", lc.DOOR),
    ("A-DOOR", lc.DOOR),
    # Fenster
    ("A_Fenster", lc.WINDOW),
    ("300_FENSTEROEFFNUNGEN", lc.WINDOW),
    ("A-GLAZ", lc.WINDOW),
    ("A-GLAZ-SILL", lc.WINDOW),
    ("FENSTER_AUF_AUSSENWAENDE", lc.WINDOW),
    ("FENSTER_AUF_AUSSENWÄNDE", lc.WINDOW),
    # Raum
    ("0_RAUMPOLYGON", lc.ROOM),
    ("ID_Raum", lc.ROOM),
]

KEINE_GEOMETRIE = [
    "300_WAND_TXT",
    "300_TUER_TXT",
    "300_TUER_KENNZ_OEFF",
    "300_FENSTER_TXT",
    "300_FENSTER_KENNZ_OEFF",
    "Tx_Raum",
    "Tx_Tuer",
    "Tx_Fenster",
    "Tx_Bemaßung",
    "ID_Tuer",
    "ID_Fenst",
    "A_Schraffur",
    "A-WALL-PATT",
    "A_Annahme",
    "Z_Abzug",
    "0_RAUMPOLYGON_ABZUG",
    "A_Fenster_oberhalb",
    "A_Treppen_oberhalb",
]

PEN_LAYER = ["STIFT_23", "STIFT_18", "AA_23_PEN", "AA_05_PEN"]

# (Layername, bearing)
TRAGEND = [
    ("300_WAND_TRAGEND", True),
    ("300_WAND_NICHTTRAGEND", False),
    ("A_Wand_nicht_tragend", False),
    ("A_Wand_KS240_tragend", True),
    ("A-WALL-NICHT-TRAGEND", False),
    ("A_Wand", None),
    ("300_WAND", None),
    ("A-WALL", None),
    ("A_Wand_a", None),
]


@pytest.mark.parametrize("layer,soll", GEOMETRIE_SOLL)
def test_geometrie_layer(layer, soll):
    c = lc.classify_layer(layer)
    if soll == WAND:
        assert c.cls in lc.WALL_CLASSES, f"{layer}: {c}"
    else:
        assert c.cls == soll, f"{layer}: {c}"


@pytest.mark.parametrize("layer", KEINE_GEOMETRIE)
def test_layer_ohne_geometrie(layer):
    c = lc.classify_layer(layer)
    assert c.cls in (lc.ANNOTATION, lc.UNKNOWN), f"{layer}: {c}"
    assert c.cls not in lc.WALL_CLASSES and c.cls not in lc.OPENING_CLASSES and c.cls != lc.ROOM


@pytest.mark.parametrize("layer", PEN_LAYER)
def test_pen_layer_sind_unknown(layer):
    assert lc.classify_layer(layer).cls == lc.UNKNOWN


@pytest.mark.parametrize("layer,bearing", TRAGEND)
def test_bearing_feld(layer, bearing):
    c = lc.classify_layer(layer)
    assert c.bearing is bearing, f"{layer}: {c}"


@pytest.mark.parametrize(
    "layer", [x for x, _ in GEOMETRIE_SOLL if not x.startswith(("A_Wand", "300_WAND", "A-WALL"))]
)
def test_bearing_nur_bei_waenden(layer):
    assert lc.classify_layer(layer).bearing is None


def test_fenster_auf_wand_ist_kein_wandlayer():
    """Der Teil nach 'AUF' nennt nur den Wirt; zaehlt nicht."""
    for name in ("FENSTER_AUF_AUSSENWAENDE", "FENSTER_AUF_AUSSENWÄNDE", "Fenster_auf_Innenwaenden"):
        assert lc.classify_layer(name).cls == lc.WINDOW
    assert lc.classify_layer("TUER_AUF_INNENWAENDEN").cls == lc.DOOR


def test_block_klassifikation_bleibt():
    assert lc.classify_block("AC_Tuer_0900", "STIFT_23").cls == lc.DOOR
    assert lc.classify_block("AC_Fenster_1200", "STIFT_18").cls == lc.WINDOW
    assert lc.classify_block("DR_0900", "A-DOOR").cls == lc.DOOR
    assert lc.classify_block("WN_1200", "A-GLAZ").cls == lc.WINDOW


def test_raumtext_layer():
    for name in ("A_Raum", "A-AREA", "0_RAUMPOLYGON", "ID_Raum", "Tx_Raum", "STIFT_7_RAUM"):
        assert lc.is_room_text_layer(name), name
    for name in ("A-ANNO-DIMS", "Tx_Tuer", "STIFT_23", "A_Text"):
        assert not lc.is_room_text_layer(name), name
