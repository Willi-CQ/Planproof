"""PDF lesen und rastern ueber pdfium (pypdfium2) - EIN Adapter fuer API und Worker.

## Warum es diese Datei gibt (14.09.2026)

PyMuPDF steht unter AGPL-3.0 oder einer kommerziellen Artifex-Lizenz. Es lief in
API und Worker ohne Lizenz - ein Befund jeder Due Diligence (known-issues
14.09.2026, Notfallplan Verkauf). pdfium (die PDF-Engine von Chromium) steht unter
BSD-3-Clause/Apache-2.0, pypdfium2 ebenso. Der Wechsel war seit April geplant.

## Die Semantik ist die von PyMuPDF - gemessen, nicht angenommen

Alle Aufrufer waren auf PyMuPDFs Antworten kalibriert (die Begehung auf die
Masse echter Protokolle, Plan-zu-BIM auf die Pixel der Rasterung, der Suchindex
auf seinen Klartext). Deshalb antwortet dieser Adapter so, wie PyMuPDF
geantwortet hat. Sonde am 14.09. auf Seiten mit /Rotate 90, versetzter CropBox
und verschachtelten Form-XObjects:

* Text, Bilder und Linien liegen im UNGEDREHTEN Seitenraum, Ursprung oben links
  an der CropBox, y nach unten. So antwortet PyMuPDF auch auf gedrehten Seiten.
* :meth:`PdfDatei.seitengroesse` ist dagegen die SICHTBARE (gedrehte) Groesse -
  wie ``page.rect`` bei PyMuPDF.
* Rendergroessen sind pixelgleich (A4 bei Faktor 2: 1190 x 1684, gedreht
  1684 x 1190, beschnitten 960 x 1540 - in beiden Bibliotheken). Dafuer rundet
  :meth:`PdfDatei._rendern` wie MuPDF in float32, nicht wie pypdfium2: ein
  Scan mit 595,3846 pt ergab bei 130 dpi sonst 1076 statt 1075 Pixel. Gemessen
  am 14.09. an 195 Seitenbildern und 579 Ausschnitten: alle gleich gross.
* Zeichen in Form-XObjects liefert pdfium bereits im Seitenraum, Bilder und
  Pfade dagegen im Raum ihres Formulars - die Kette der Formmatrizen wird hier
  angewendet, dann stimmen sie mit PyMuPDF ueberein.
* Text ausserhalb der sichtbaren Seite (CropBox) liefert MuPDF nicht - eine
  zugeschnittene Klausurseite zeigte sonst ihre weggeschnittene Titelseite mit.

pdfium kennt Zeichen, keine Spans. :meth:`PdfDatei.textspans` baut sie nach den
Schwellen von MuPDFs stext-Device (Abstaende in Vielfachen der Schriftgroesse):
unter 0,15 kein Zwischenraum, bis 0,8 ein Leerzeichen, darueber eine neue Zeile;
ein Wechsel von Farbe oder Groesse beginnt einen neuen Span. Die Reihenfolge ist
die des Inhaltsstroms wie bei PyMuPDF - pdfium ordnet Zeichen auf gedrehten
Seiten sonst selbst um (gemessen: "Datum Foto / rot text / Beschreibung").

**Die Hoehe eines Spans rechnet dieser Adapter wie MuPDF: Grundlinie plus
Ascender, Grundlinie plus Descender der Schrift** - nicht aus pdfiums
"Loose Box". Gemessen an einem echten Begehungsprotokoll lag deren Oberkante
2,3 bis 3,6 pt tiefer; die Begehung schneidet ihre Zeilenbaender an dieser Kante,
und auf Seite 4 des Protokolls vom 10.12.2025 rutschte dadurch ein Satzstueck in
die Nachbarzeile - ein anderer Zeilenschluessel, an dem die Entscheidung eines
Menschen haengt. pdfiums ``FPDFFont_GetAscent/GetDescent`` liefert genau MuPDFs
Werte (DIN-Regular, DIN-Bold: alle Kanten auf 0,01 pt gleich) - ausser bei den
14 Standardschriften ohne Schriftbeschreibung: dort setzt MuPDF eigene
Ersatzschriften ein (:data:`_MUPDF_STANDARDSCHRIFTEN`, gemessen). Dass eine
Beschreibung fehlt, sieht der Adapter daran, dass pdfium genau den Wert SEINER
Ersatzschrift meldet - und welche das ist, haengt von den Schriften der Maschine
ab (Windows: Arial, Produktions-Image: DejaVu). Dieser Wert wird deshalb im
laufenden Prozess gemessen (:func:`_ascender_ohne_beschreibung`), nicht eingetragen.

**Die Breite eines Zeichens kommt aus pdfium** (PDF-Breitentabelle), nicht aus
dem Schriftprogramm wie bei MuPDF. Widersprechen sich beide, weicht das
Leerzeichen ab: gemessen an EINER Stelle in drei Protokollen, "->Abdeckung"
statt MuPDFs "-> Abdeckung" - gesetzt ist das Wort 0,5 pt hinter dem Pfeil,
pdfium liegt also naeher am Blatt. Beschreibung und Zeilenschluessel bleiben gleich.

**Auch der Klartext kommt aus dieser Zeilenbildung, nicht aus pdfiums eigener
Textausgabe.** Gemessen an 57 echten PDFs: pdfiums ``get_text_range`` klebt eng
stehende Tabellenzellen zusammen - in einem Aufmass wurden aus "3" und "19"
75-mal "319", aus Datum und Kopfzeile "2026druckdatum" -, trennt Hochgestelltes
ab ("m2" -> "m 2") und laesst Steuerzeichen durch. Fuer einen Suchindex ueber
Mengen und Preise ist das eine falsche Zahl, keine Formsache.

**Was bleibt, gemessen am 14.09. an 57 PDFs mit 308 724 Woertern** (Wort-F1
gegen PyMuPDF: Median 1,0, p10 0,993; Seitenzahl und Scan-Entscheidung ueberall
gleich):

* Zeichen ausserhalb der BMP (Formelsatz, U+1D43C mathematisches I) liefert
  pdfium als zwei UTF-16-Haelften; sie werden hier wieder zusammengesetzt.
  Einzeln waren es Surrogate - kein Wort, und als UTF-8 nicht speicherbar.
* Steht ein echtes Leerzeichen am Ende eines Textobjekts und beginnt das
  naechste Objekt noch in dessen Breite (Tabulator, enger Blocksatz), verschluckt
  pdfium es. 56 Wortpaare kleben so zusammen (0,02 %), etwa "1____" statt
  "1 ____". Der Rahmen des Textobjekts verraet es nicht: von 2514 Objekten, die
  ueber ihr letztes Zeichen hinausreichen, trugen 49 wirklich ein Leerzeichen -
  also wird hier nichts geraten.
* Zeichen ohne Unicode-Tabelle: pdfium nimmt den Zeichencode, MuPDF den
  Glyphennamen. In Cambria-Math-Formeln wird so aus U+02C6 (Zirkumflex) ein
  "a" mit Gravis. Schriften ganz ohne Abbildung (ein Formblatt) liefern in
  beiden Faellen Fuellbytes.

## Nebenlaeufigkeit

pdfium ist NICHT threadsicher, auch nicht mit verschiedenen Dokumenten. PyMuPDF
war es ebenso wenig, hielt aber den GIL ueber jeden Aufruf - ctypes gibt ihn frei.
In der API rastern OCR, Begehung, Vision und Indexierung in verschiedenen Threads.
Deshalb laeuft JEDER pdfium-Aufruf dieses Moduls unter einer prozessweiten
Sperre, und kein pdfium-Objekt verlaesst das Modul: Seiten werden je Aufruf
geoeffnet und geschlossen, das Dokument explizit - nie durch den Garbage
Collector in einem fremden Thread.
"""

from __future__ import annotations

import ctypes
import io
import math
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, NamedTuple, Self

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

#: Die eine Sperre fuer alle pdfium-Aufrufe im Prozess (siehe Modulkopf).
_SPERRE = threading.RLock()

Matrix = tuple[float, float, float, float, float, float]
Rahmen = tuple[float, float, float, float]
_EINHEIT: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

#: Schwellen aus MuPDFs stext-Device, in Vielfachen der Schriftgroesse.
_LEERZEICHEN_AB = 0.15
_NEUE_ZEILE_AB = 0.8
_GRUNDLINIE_MAX = 0.8
#: Groessenunterschied, ab dem ein neuer Span beginnt (relativ).
_GROESSE_TOLERANZ = 0.05
#: Tiefe, bis zu der Form-XObjects verfolgt werden - Schutz gegen entartete Dateien.
_FORM_TIEFE_MAX = 16
#: Stuetzstellen je kubischer Bezier-Kurve - wie ``plan_extract_pdf`` bisher.
BEZIER_SCHRITTE = 8

#: pdfium setzt Trennstriche am Zeilenende teils als Steuerzeichen.
_TRENNSTRICH_ZEICHEN = (0x02, 0xFFFE)

#: MuPDFs Ascender/Descender (je em) fuer die 14 Standardschriften ohne
#: Schriftbeschreibung - MuPDF setzt dann eigene Ersatzschriften ein. Gemessen am
#: 14.09.2026 mit PyMuPDF 1.27 (``insert_text`` je Schrift, ``span["ascender"]``).
_MUPDF_STANDARDSCHRIFTEN: dict[str, tuple[float, float]] = {
    "Helvetica": (1.075, -0.299),
    "Helvetica-Bold": (1.070, -0.307),
    "Helvetica-Oblique": (1.070, -0.284),
    "Helvetica-BoldOblique": (1.073, -0.309),
    "Times-Roman": (1.053, -0.281),
    "Times-Bold": (1.044, -0.341),
    "Times-Italic": (0.951, -0.270),
    "Times-BoldItalic": (0.972, -0.324),
    "Courier": (0.932, -0.317),
    "Courier-Bold": (1.007, -0.393),
    "Courier-Oblique": (0.920, -0.317),
    "Courier-BoldOblique": (0.997, -0.393),
    "Symbol": (1.010, -0.293),
    "ZapfDingbats": (0.819, -0.144),
}
#: Was pdfium fuer eine Standardschrift OHNE Schriftbeschreibung meldet (Ascender je
#: em, je Name, gemessen von :func:`_ascender_ohne_beschreibung`). Meldet eine Schrift
#: im Dokument genau das, gibt es keine Beschreibung, und MuPDF nimmt seine
#: Ersatzschrift; sonst stammt der Wert aus der Beschreibung, und die nimmt MuPDF auch.
#:
#: Bis zum 15.09.2026 stand hier eine feste Tabelle, gemessen unter Windows
#: (Helvetica 0,905 - das ist Arial). pdfium nimmt fuer eine nicht eingebettete
#: Standardschrift aber die Schrift, die das SYSTEM hat: im Produktions-Image und auf
#: dem CI-Runner DejaVu (Helvetica 0,945, Helvetica-Bold 0,962, Times 0,878, Courier
#: 0,803). Die Tabelle traf dort nie, und solche Spans bekamen pdfiums Hoehe statt
#: MuPDFs - Oberkante 1,3 pt tiefer, Unterkante 0,75 pt hoeher (Core-CI rot seit
#: 14.09., im Image ``prod-20260915-8bb395b6`` nachgemessen).
_OHNE_BESCHREIBUNG: dict[str, float | None] = {}


@dataclass(frozen=True)
class TextSpan:
    """Ein Textstueck mit einheitlicher Farbe und Groesse - wie ein PyMuPDF-Span."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    groesse: float
    #: Fuellfarbe als 0xRRGGBB - dieselbe Kodierung wie ``span["color"]``.
    farbe: int

    @property
    def bbox(self) -> Rahmen:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass(frozen=True)
class Linien:
    """Die Vektorgeometrie einer Seite als Liniensegmente."""

    #: Sichtbare Pfad-Objekte - das Gegenstueck zu ``len(page.get_drawings())``.
    pfade: int
    #: ``(ax, ay, bx, by)`` je Segment; Kurven sind in Stuetzstellen zerlegt.
    segmente: list[Rahmen]


@dataclass(frozen=True)
class Gerendert:
    """Ein gerastertes PNG samt seiner Pixelgroesse."""

    png: bytes
    breite: int
    hoehe: int


class _Zeichen(NamedTuple):
    rang: int
    index: int
    zeichen: str
    ox: float
    oy: float
    links: float
    unten: float
    rechts: float
    oben: float
    groesse: float
    farbe: int
    rx: float
    ry: float
    #: Wo der naechste Buchstabe beginnen wuerde (Ursprung + Vorschub).
    px: float
    py: float


# ── Geometrie ────────────────────────────────────────────────────────


def _mal(m: Matrix, n: Matrix) -> Matrix:
    """Erst ``m``, dann ``n`` anwenden - Zeilenvektor-Konvention wie im PDF."""
    a, b, c, d, e, f = m
    aa, bb, cc, dd, ee, ff = n
    return (
        a * aa + b * cc,
        a * bb + b * dd,
        c * aa + d * cc,
        c * bb + d * dd,
        e * aa + f * cc + ee,
        e * bb + f * dd + ff,
    )


def _punkt(m: Matrix, x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = m
    return (a * x + c * y + e, b * x + d * y + f)


def _rechteck(m: Matrix, links: float, unten: float, rechts: float, oben: float) -> Rahmen:
    """Achsparalleler Rahmen um das transformierte Rechteck (links, unten, rechts, oben)."""
    ecken = [_punkt(m, x, y) for x in (links, rechts) for y in (unten, oben)]
    xs = [p[0] for p in ecken]
    ys = [p[1] for p in ecken]
    return (min(xs), min(ys), max(xs), max(ys))


def _objekt_matrix(objekt: Any) -> Matrix:
    m = pdfium_c.FS_MATRIX()
    if not pdfium_c.FPDFPageObj_GetMatrix(objekt, m):
        return _EINHEIT
    return (m.a, m.b, m.c, m.d, m.e, m.f)


def _objekt_rahmen(objekt: Any) -> Rahmen | None:
    links, unten, rechts, oben = (ctypes.c_float() for _ in range(4))
    if not pdfium_c.FPDFPageObj_GetBounds(objekt, links, unten, rechts, oben):
        return None
    return (links.value, unten.value, rechts.value, oben.value)


def _zeiger(handle: Any) -> int:
    return ctypes.cast(handle, ctypes.c_void_p).value or 0


def _objekte(seite_roh: Any) -> Iterator[tuple[Any, int, Matrix]]:
    """Alle Seitenobjekte in Inhaltsreihenfolge, Formulare eingeschlossen.

    Liefert ``(objekt, typ, kette)``: ``kette`` bildet den Raum des Containers
    auf den Seitenraum ab (Einheit fuer Objekte direkt auf der Seite).
    """

    def gehen(anzahl: int, holen: Any, kette: Matrix, tiefe: int) -> Iterator[tuple[Any, int, Matrix]]:
        for i in range(anzahl):
            objekt = holen(i)
            if not objekt:
                continue
            typ = pdfium_c.FPDFPageObj_GetType(objekt)
            yield objekt, typ, kette
            if typ == pdfium_c.FPDF_PAGEOBJ_FORM and tiefe < _FORM_TIEFE_MAX:
                innen = _mal(_objekt_matrix(objekt), kette)
                yield from gehen(
                    pdfium_c.FPDFFormObj_CountObjects(objekt),
                    lambda j, o=objekt: pdfium_c.FPDFFormObj_GetObject(o, j),
                    innen,
                    tiefe + 1,
                )

    yield from gehen(
        pdfium_c.FPDFPage_CountObjects(seite_roh),
        lambda i: pdfium_c.FPDFPage_GetObject(seite_roh, i),
        _EINHEIT,
        0,
    )


# ── Das Dokument ─────────────────────────────────────────────────────


PdfQuelle = bytes | bytearray | memoryview | str | Path


class PdfDatei:
    """Ein geoeffnetes PDF. Jede Methode haelt die Sperre nur fuer ihren eigenen Aufruf.

    Benutzung::

        with PdfDatei(daten) as pdf:
            for i in range(pdf.seitenzahl):
                spans = pdf.textspans(i)
    """

    def __init__(self, quelle: PdfQuelle) -> None:
        # Pfade werden in Python gelesen: so haengt das Oeffnen nicht davon ab,
        # wie pdfium Nicht-ASCII-Pfade unter Windows behandelt.
        daten = bytes(quelle) if isinstance(quelle, (bytes, bytearray, memoryview)) else Path(quelle).read_bytes()
        with _SPERRE:
            self._pdf = pdfium.PdfDocument(daten)
            self.seitenzahl = len(self._pdf)
        self._offen = True

    # Kontextmanager
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.schliessen()

    def __len__(self) -> int:
        return self.seitenzahl

    def schliessen(self) -> None:
        if not self._offen:
            return
        with _SPERRE:
            self._pdf.close()
        self._offen = False

    def _seite(self, index: int) -> Any:
        if not 0 <= index < self.seitenzahl:
            raise IndexError(f"Seite {index} ausserhalb des PDFs ({self.seitenzahl} Seiten)")
        return self._pdf[index]

    # ── Masse ──

    def seitengroesse(self, index: int) -> tuple[float, float]:
        """Sichtbare Groesse in Punkt, Drehung beruecksichtigt - wie ``page.rect``."""
        with _SPERRE:
            seite = self._seite(index)
            try:
                breite, hoehe = seite.get_size()
            finally:
                seite.close()
        return float(breite), float(hoehe)

    # ── Text ──

    def _zeichen_der_seite(self, index: int) -> tuple[list[_Zeichen], Rahmen]:
        with _SPERRE:
            seite = self._seite(index)
            try:
                cropbox = seite.get_cropbox()
                reihenfolge = {
                    _zeiger(o): rang
                    for rang, (o, typ, _k) in enumerate(_objekte(seite.raw))
                    if typ == pdfium_c.FPDF_PAGEOBJ_TEXT
                }
                textseite = seite.get_textpage()
                try:
                    zeichen = _zeichen_lesen(textseite.raw, reihenfolge, cropbox)
                finally:
                    textseite.close()
            finally:
                seite.close()
        return zeichen, cropbox

    def textspans(self, index: int) -> list[TextSpan]:
        """Die Spans der Seite in Inhaltsreihenfolge, Koordinaten wie PyMuPDF."""
        return [span for span, _zeile in _spans_bilden(*self._zeichen_der_seite(index))]

    def seitentext(self, index: int) -> str:
        """Der Klartext der Seite wie ``page.get_text()``: eine Zeile je Textzeile, ``\\n`` am Ende."""
        zeilen: list[list[str]] = []
        letzte = -1
        for span, zeile in _spans_bilden(*self._zeichen_der_seite(index)):
            if zeile != letzte:
                zeilen.append([])
                letzte = zeile
            zeilen[-1].append(span.text)
        return "".join("".join(teile) + "\n" for teile in zeilen)

    # ── Bilder und Linien ──

    def bildrahmen(self, index: int) -> list[Rahmen]:
        """Rahmen aller Bildplatzierungen - wie ``page.get_image_info()``."""
        with _SPERRE:
            seite = self._seite(index)
            try:
                links, _unten, _rechts, oben = seite.get_cropbox()
                rahmen: list[Rahmen] = []
                for objekt, typ, kette in _objekte(seite.raw):
                    if typ != pdfium_c.FPDF_PAGEOBJ_IMAGE:
                        continue
                    lokal = _objekt_rahmen(objekt)
                    if lokal is None:
                        continue
                    x0, y0, x1, y1 = _rechteck(kette, *lokal)
                    rahmen.append((x0 - links, oben - y1, x1 - links, oben - y0))
            finally:
                seite.close()
        return rahmen

    def linien(self, index: int, *, bezier_schritte: int = BEZIER_SCHRITTE) -> Linien:
        """Die sichtbaren Vektorpfade als Liniensegmente - wie ``page.get_drawings()``."""
        with _SPERRE:
            seite = self._seite(index)
            try:
                links, _unten, _rechts, oben = seite.get_cropbox()
                pfade = 0
                segmente: list[Rahmen] = []
                for objekt, typ, kette in _objekte(seite.raw):
                    if typ != pdfium_c.FPDF_PAGEOBJ_PATH or not _pfad_sichtbar(objekt):
                        continue
                    pfade += 1
                    for ax, ay, bx, by in _pfad_segmente(objekt, _mal(_objekt_matrix(objekt), kette), bezier_schritte):
                        segmente.append((ax - links, oben - ay, bx - links, oben - by))
            finally:
                seite.close()
        return Linien(pfade=pfade, segmente=segmente)

    # ── Rastern ──

    def rendern_grau(self, index: int, zoom: float) -> np.ndarray:
        """Graustufenbild ``(hoehe, breite)`` uint8 - wie ``get_pixmap(colorspace=csGRAY)``."""
        feld = self._rendern(index, zoom, grau=True)
        if feld.ndim == 3:
            feld = feld[:, :, 0]
        return np.ascontiguousarray(feld)

    def rendern_rgb(self, index: int, zoom: float) -> np.ndarray:
        """RGB-Bild ``(hoehe, breite, 3)`` uint8."""
        return np.ascontiguousarray(self._rendern(index, zoom, grau=False))

    def png(
        self,
        index: int,
        *,
        zoom: float | None = None,
        dpi: float | None = None,
        grau: bool = False,
        ausschnitt: Rahmen | None = None,
    ) -> Gerendert:
        """Seite oder Ausschnitt als PNG. ``ausschnitt`` in denselben Koordinaten wie Text und Bilder.

        Anders als PyMuPDFs ``clip`` gilt der Ausschnitt auch auf gedrehten Seiten
        im Raum der Bildrahmen - PyMuPDF legte ihn dort in den gedrehten Raum und
        schnitt damit eine andere Stelle aus als die, deren Rahmen es gemeldet hatte.
        """
        if zoom is None:
            zoom = (dpi or 72.0) / 72.0
        feld = self._rendern(index, zoom, grau=grau, ausschnitt=ausschnitt)
        if grau and feld.ndim == 3:
            feld = feld[:, :, 0]
        return png_aus_feld(feld)

    #: MuPDFs Toleranz beim Runden eines Rechtecks auf Pixel (``fz_round_rect``).
    _RUNDUNG = np.float32(0.001)

    @classmethod
    def _pixel(cls, wert: float, zoom: float, *, aufrunden: bool) -> int:
        """Punkt mal Zoom auf ganze Pixel - in MuPDFs float32-Arithmetik.

        MuPDF rechnet Rechtecke in ``float``: ``ceilf(x1 * zoom - 0.001f)``. Liegt
        das Produkt auf einer Kante dieser Toleranz, entscheidet die float-Rundung
        ueber das Pixel (gemessen: 97,3 pt bei Faktor 0,37 sind in double 36,001,
        in float etwas mehr - MuPDF rundet auf 37).
        """
        produkt = np.float32(wert) * np.float32(zoom)
        if aufrunden:
            return math.ceil(produkt - cls._RUNDUNG)
        return math.floor(produkt + cls._RUNDUNG)

    @staticmethod
    def _sichtbar(seite: Any, ausschnitt: Rahmen) -> Rahmen:
        """Ausschnitt (ungedrehter Raum, oben links) -> Rechteck auf der SICHTBAREN Seite (oben links)."""
        links, unten, rechts, oben = seite.get_cropbox()
        drehung = seite.get_rotation() % 360
        breite_u, hoehe_u = rechts - links, oben - unten
        x0, y0, x1, y1 = ausschnitt
        x0, x1 = max(0.0, min(x0, x1)), min(breite_u, max(x0, x1))
        y0, y1 = max(0.0, min(y0, y1)), min(hoehe_u, max(y0, y1))
        if x1 < x0 or y1 < y0:
            raise ValueError(f"Ausschnitt {ausschnitt} liegt ausserhalb der sichtbaren Seite")
        # Ungedreht (x, y von oben) -> sichtbar, Drehung im Uhrzeigersinn wie /Rotate.
        if drehung == 90:
            punkte = [(hoehe_u - y, x) for x in (x0, x1) for y in (y0, y1)]
        elif drehung == 180:
            punkte = [(breite_u - x, hoehe_u - y) for x in (x0, x1) for y in (y0, y1)]
        elif drehung == 270:
            punkte = [(y, breite_u - x) for x in (x0, x1) for y in (y0, y1)]
        else:
            punkte = [(x, y) for x in (x0, x1) for y in (y0, y1)]
        return (
            min(p[0] for p in punkte),
            min(p[1] for p in punkte),
            max(p[0] for p in punkte),
            max(p[1] for p in punkte),
        )

    def _rendern(self, index: int, zoom: float, *, grau: bool, ausschnitt: Rahmen | None = None) -> np.ndarray:
        """Seite oder Ausschnitt rastern - im Pixelraster von PyMuPDF.

        Bewusst NICHT ueber ``PdfPage.render``: pypdfium2 rundet die Bildgroesse
        mit ``ceil(breite * zoom)``, MuPDF mit ``ceil(breite * zoom - 0,001)``.
        Liegt das Produkt knapp ueber einer ganzen Zahl - ein Scan mit
        595,3846 pt bei 130 dpi ergibt 1075,00002 -, wurde das Bild eine Spalte
        breiter als bisher (gemessen 14.09.). Ausschnitte rundet MuPDF als Ganzes
        nach aussen, pypdfium2s ``crop`` jeden Rand fuer sich (416 x 312 statt
        418 x 313). Hier rastert pdfium die ganze Seite in MuPDFs Raster, und der
        Ausschnitt ist ein Fenster in genau dieses Raster.
        """
        with _SPERRE:
            seite = self._seite(index)
            try:
                breite, hoehe = seite.get_size()
                voll_b = self._pixel(breite, zoom, aufrunden=True)
                voll_h = self._pixel(hoehe, zoom, aufrunden=True)
                if ausschnitt is None:
                    x0, y0, x1, y1 = 0, 0, voll_b, voll_h
                else:
                    sx0, sy0, sx1, sy1 = self._sichtbar(seite, ausschnitt)
                    x0 = max(0, self._pixel(sx0, zoom, aufrunden=False))
                    y0 = max(0, self._pixel(sy0, zoom, aufrunden=False))
                    x1 = min(voll_b, self._pixel(sx1, zoom, aufrunden=True))
                    y1 = min(voll_h, self._pixel(sy1, zoom, aufrunden=True))
                b, h = x1 - x0, y1 - y0
                if b < 1 or h < 1:
                    # MuPDF gab hier ein Bild ohne Pixel zurueck, pypdfium2 einen Fehler.
                    raise ValueError(f"Nichts zu rastern: {b} x {h} Pixel (Seite {index}, Zoom {zoom:.3f})")
                format_ = pdfium_c.FPDFBitmap_Gray if grau else pdfium_c.FPDFBitmap_BGR
                bitmap = pdfium.PdfBitmap.new_native(b, h, format_, rev_byteorder=not grau)
                try:
                    bitmap.fill_rect((255, 255, 255, 255), 0, 0, b, h)
                    flags = pdfium_c.FPDF_ANNOT | (pdfium_c.FPDF_GRAYSCALE if grau else pdfium_c.FPDF_REVERSE_BYTE_ORDER)
                    pdfium_c.FPDF_RenderPageBitmap(bitmap, seite, -x0, -y0, voll_b, voll_h, 0, flags)
                    feld = np.array(bitmap.to_numpy(), copy=True)
                finally:
                    bitmap.close()
            finally:
                seite.close()
        if feld.ndim == 3 and feld.shape[2] == 4:
            feld = feld[:, :, :3]
        return feld

    # ── Anlagen ──

    def anlagen(self) -> list[tuple[str, bytes]]:
        """Eingebettete Dateien (EmbeddedFiles) als ``(name, inhalt)``."""
        with _SPERRE:
            ergebnis: list[tuple[str, bytes]] = []
            for i in range(self._pdf.count_attachments()):
                anlage = self._pdf.get_attachment(i)
                try:
                    ergebnis.append((str(anlage.get_name() or f"anlage-{i}"), bytes(anlage.get_data())))
                except Exception:  # noqa: BLE001, S112 - eine kaputte Anlage kostet nicht die anderen
                    continue
        return ergebnis


# ── Hilfen ───────────────────────────────────────────────────────────


def png_aus_feld(feld: np.ndarray) -> Gerendert:
    """Ein uint8-Feld (grau oder RGB) als PNG."""
    from PIL import Image

    bild = Image.fromarray(feld, mode="L" if feld.ndim == 2 else "RGB")
    puffer = io.BytesIO()
    bild.save(puffer, format="PNG")
    return Gerendert(png=puffer.getvalue(), breite=int(bild.width), hoehe=int(bild.height))


def seitenzahl(quelle: PdfQuelle) -> int:
    with PdfDatei(quelle) as pdf:
        return pdf.seitenzahl


def _pfad_sichtbar(objekt: Any) -> bool:
    fuellung, strich = ctypes.c_int(), ctypes.c_int()
    if not pdfium_c.FPDFPath_GetDrawMode(objekt, fuellung, strich):
        return True
    return bool(fuellung.value) or bool(strich.value)


def _pfad_segmente(objekt: Any, m: Matrix, bezier_schritte: int) -> Iterator[Rahmen]:
    """Die Segmente EINES Pfads im PDF-Seitenraum (y nach oben)."""
    start: tuple[float, float] | None = None
    aktuell: tuple[float, float] | None = None
    kontrolle: list[tuple[float, float]] = []
    x, y = ctypes.c_float(), ctypes.c_float()
    for k in range(pdfium_c.FPDFPath_CountSegments(objekt)):
        segment = pdfium_c.FPDFPath_GetPathSegment(objekt, k)
        if not segment or not pdfium_c.FPDFPathSegment_GetPoint(segment, x, y):
            continue
        p = _punkt(m, x.value, y.value)
        art = pdfium_c.FPDFPathSegment_GetType(segment)
        if art == pdfium_c.FPDF_SEGMENT_MOVETO or aktuell is None:
            start = aktuell = p
            kontrolle = []
        elif art == pdfium_c.FPDF_SEGMENT_LINETO:
            yield (aktuell[0], aktuell[1], p[0], p[1])
            aktuell = p
        elif art == pdfium_c.FPDF_SEGMENT_BEZIERTO:
            kontrolle.append(p)
            if len(kontrolle) == 3:
                yield from _bezier(aktuell, kontrolle[0], kontrolle[1], kontrolle[2], bezier_schritte)
                aktuell = kontrolle[2]
                kontrolle = []
        if pdfium_c.FPDFPathSegment_GetClose(segment) and start is not None and aktuell is not None:
            if aktuell != start:
                yield (aktuell[0], aktuell[1], start[0], start[1])
            aktuell = start


def _bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    schritte: int,
) -> Iterator[Rahmen]:
    def stelle(t: float) -> tuple[float, float]:
        u = 1 - t
        return (
            u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0],
            u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1],
        )

    punkte = [stelle(i / schritte) for i in range(schritte + 1)]
    for a, b in pairwise(punkte):
        yield (a[0], a[1], b[0], b[1])


def _standardfamilie(name: str) -> str | None:
    """Welche der 14 Standardschriften ist das - oder keine?"""
    grund = name.split("+", 1)[-1].replace(",", "-")
    if grund in _MUPDF_STANDARDSCHRIFTEN:
        return grund
    return None


def _ascender_ohne_beschreibung(standard: str) -> float | None:
    """Ascender je em, den pdfium auf DIESER Maschine fuer ``standard`` ohne Beschreibung meldet.

    ``FPDFText_LoadStandardFont`` legt die Schrift ohne Schriftbeschreibung an; pdfium
    greift also zur selben Ersatzschrift wie bei einer solchen Schrift im Dokument -
    im selben Prozess, mit denselben Systemschriften. Einmal je Name gemessen.
    """
    with _SPERRE:
        if standard not in _OHNE_BESCHREIBUNG:
            wert = None
            dokument = pdfium_c.FPDF_CreateNewDocument()
            try:
                schrift = pdfium_c.FPDFText_LoadStandardFont(dokument, standard.encode("ascii"))
                if schrift:
                    try:
                        auf = ctypes.c_float()
                        # pdfium rechnet ganzzahlige Tausendstel: bei 1000 pt ist der Wert exakt.
                        if pdfium_c.FPDFFont_GetAscent(schrift, ctypes.c_float(1000.0), auf):
                            wert = auf.value / 1000.0
                    finally:
                        pdfium_c.FPDFFont_Close(schrift)
            finally:
                pdfium_c.FPDF_CloseDocument(dokument)
            _OHNE_BESCHREIBUNG[standard] = wert
        return _OHNE_BESCHREIBUNG[standard]


def _schrifthoehe(objekt: Any, groesse: float) -> tuple[float, float] | None:
    """(Ascender, Descender) in Punkt bei ``groesse`` - so, wie MuPDF sie nimmt.

    ``None`` heisst: unbekannt, der Aufrufer nimmt pdfiums Zeichenrahmen.
    """
    schrift = pdfium_c.FPDFTextObj_GetFont(objekt)
    if not schrift:
        return None
    auf, ab = ctypes.c_float(), ctypes.c_float()
    hat_werte = bool(
        pdfium_c.FPDFFont_GetAscent(schrift, ctypes.c_float(groesse), auf)
        and pdfium_c.FPDFFont_GetDescent(schrift, ctypes.c_float(groesse), ab)
        and auf.value > ab.value
    )
    if not pdfium_c.FPDFFont_GetIsEmbedded(schrift):
        puffer = ctypes.create_string_buffer(256)
        pdfium_c.FPDFFont_GetBaseFontName(schrift, puffer, 256)
        standard = _standardfamilie(puffer.value.decode("latin-1", "replace"))
        if standard is not None:
            ersatz = _ascender_ohne_beschreibung(standard)
            ohne_beschreibung = not hat_werte or (
                ersatz is not None and groesse > 0 and abs(auf.value / groesse - ersatz) < 0.002
            )
            if ohne_beschreibung:
                hoch, tief = _MUPDF_STANDARDSCHRIFTEN[standard]
                return hoch * groesse, tief * groesse
    if not hat_werte:
        return None
    return auf.value, ab.value


def _zeichen_lesen(
    textseite: Any,
    reihenfolge: dict[int, int],
    cropbox: Rahmen | None = None,
) -> list[_Zeichen]:
    """Alle echten (nicht von pdfium erzeugten) Zeichen samt Geometrie und Stil.

    **Stil je Textobjekt, Lage je Zeichen.** Die erste Fassung fragte pdfium je
    Zeichen zehnmal - beim 209-seitigen EU-Arbeitsprogramm 47 s gegen 3 s bei
    PyMuPDF, auf der einzigen Indexier-Bahn der API. Schriftgroesse, Richtung,
    Farbe und Schrifthoehe sind aber je Textobjekt (ein Tj/TJ-Lauf) konstant: sie
    werden einmal je Objekt gelesen.

    **Erzeugte Leerzeichen** haengt pdfium teils an das Textobjekt des Nachbarn -
    sie werden deshalb ueber ``IsGenerated`` erkannt, aber nur bei Leerzeichen
    gefragt (Zeilenenden fallen als Steuerzeichen ohnehin weg).

    **Nur was auf der Seite steht.** Zeichen ganz ausserhalb der CropBox fallen weg.
    """
    ergebnis: list[_Zeichen] = []
    matrix = pdfium_c.FS_MATRIX()
    ox, oy = ctypes.c_double(), ctypes.c_double()
    box = pdfium_c.FS_RECTF()
    rot, gruen, blau, alpha = (ctypes.c_uint() for _ in range(4))
    ohne_rang = len(reihenfolge)
    stile: dict[int, tuple[int, float, float, float, int, tuple[float, float] | None]] = {}
    cl, cu, cr, co = cropbox if cropbox is not None else (-math.inf, -math.inf, math.inf, math.inf)
    textobjekt = pdfium_c.FPDFText_GetTextObject
    unicode_von = pdfium_c.FPDFText_GetUnicode
    ursprung = pdfium_c.FPDFText_GetCharOrigin
    rahmen_von = pdfium_c.FPDFText_GetLooseCharBox
    erzeugt = pdfium_c.FPDFText_IsGenerated
    anzahl = pdfium_c.FPDFText_CountChars(textseite)
    zweite_haelfte = -1
    for i in range(anzahl):
        if i == zweite_haelfte:
            continue
        code = unicode_von(textseite, i)
        if 0xD800 <= code <= 0xDFFF:
            # Zeichen ausserhalb der BMP (Formelsatz: U+1D43C, kursives I) liefert pdfium als
            # zwei UTF-16-Haelften mit demselben Rahmen. Einzeln waeren es zwei
            # Surrogate - kein Wort, und nicht als UTF-8 speicherbar (gemessen 14.09.
            # an Word-Formeln in Cambria Math).
            folge = unicode_von(textseite, i + 1) if i + 1 < anzahl else 0
            if code <= 0xDBFF and 0xDC00 <= folge <= 0xDFFF:
                code = 0x10000 + ((code - 0xD800) << 10) + (folge - 0xDC00)
                zweite_haelfte = i + 1
            else:
                code = 0xFFFD
        if code in (0x20, *_TRENNSTRICH_ZEICHEN) and erzeugt(textseite, i) == 1:
            continue  # von pdfium erzeugt - Leerraum baut die Span-Bildung selbst
        if code in _TRENNSTRICH_ZEICHEN or (code == 0 and pdfium_c.FPDFText_IsHyphen(textseite, i) == 1):
            code = 0x2D
        elif code == 0x09:
            code = 0x20
        # Steuerzeichen und Nicht-Zeichen tragen keinen Text: pdfium laesst sie
        # bei Schriften ohne Unicode-Tabelle durch (gemessen: 854 in einem Formblatt).
        if code < 0x20 or code in (0x7F, 0xFFFF):
            continue
        objekt = textobjekt(textseite, i)
        zeiger = _zeiger(objekt)
        if not zeiger or not rahmen_von(textseite, i, box):
            continue
        stil = stile.get(zeiger)
        if stil is None:
            pdfium_c.FPDFText_GetMatrix(textseite, i, matrix)
            schrift = float(pdfium_c.FPDFText_GetFontSize(textseite, i))
            skala = math.sqrt(abs(matrix.a * matrix.d - matrix.b * matrix.c))
            groesse = schrift * skala if skala > 1e-9 else schrift
            laenge = math.hypot(matrix.a, matrix.b)
            farbe = 0
            if pdfium_c.FPDFText_GetFillColor(textseite, i, rot, gruen, blau, alpha):
                farbe = (rot.value << 16) | (gruen.value << 8) | blau.value
            stil = (
                reihenfolge.get(zeiger, ohne_rang),
                groesse,
                matrix.a / laenge if laenge > 1e-9 else 1.0,
                matrix.b / laenge if laenge > 1e-9 else 0.0,
                farbe,
                _schrifthoehe(objekt, groesse),
            )
            stile[zeiger] = stil
        rang, groesse, rx, ry, farbe, hoehe = stil
        ursprung(textseite, i, ox, oy)
        x, y = ox.value, oy.value
        # Laenge in Schreibrichtung aus pdfiums Zeichenrahmen, Hoehe quer dazu aus
        # der Schrift (Ascender/Descender) - wie MuPDF.
        bl, bb, br, bt = box.left, box.bottom, box.right, box.top
        if ry == 0.0 and rx == 1.0:
            # Der Normalfall, waagerechter Text - ohne Eckenrechnung.
            links, rechts = (bl, br) if bl <= br else (br, bl)
            if hoehe is not None:
                unten, oben = y + hoehe[1], y + hoehe[0]
            else:
                unten, oben = (bb, bt) if bb <= bt else (bt, bb)
            px, py = rechts, y
        else:
            ecken_lr = ((bl, bb), (br, bb), (bl, bt), (br, bt))
            entlang = [(ex - x) * rx + (ey - y) * ry for ex, ey in ecken_lr]
            a0, a1 = min(entlang), max(entlang)
            if hoehe is not None:
                h0, h1 = hoehe[1], hoehe[0]
            else:
                quer = [-(ex - x) * ry + (ey - y) * rx for ex, ey in ecken_lr]
                h0, h1 = min(quer), max(quer)
            ecken = [(x + rx * a - ry * h, y + ry * a + rx * h) for a in (a0, a1) for h in (h0, h1)]
            links = min(e[0] for e in ecken)
            rechts = max(e[0] for e in ecken)
            unten = min(e[1] for e in ecken)
            oben = max(e[1] for e in ecken)
            px, py = x + rx * a1, y + ry * a1
        if rechts < cl or links > cr or oben < cu or unten > co:
            continue
        ergebnis.append(_Zeichen(rang, i, chr(code), x, y, links, unten, rechts, oben, groesse, farbe, rx, ry, px, py))
    ergebnis.sort(key=lambda z: (z.rang, z.index))
    return ergebnis


def _spans_bilden(zeichen: list[_Zeichen], cropbox: Rahmen) -> list[tuple[TextSpan, int]]:
    """Zeichen -> ``(Span, Zeilennummer)`` nach den Schwellen von MuPDFs stext-Device."""
    links_cb, _unten_cb, _rechts_cb, oben_cb = cropbox
    spans: list[tuple[TextSpan, int]] = []

    text: list[str] = []
    rahmen = [0.0, 0.0, 0.0, 0.0]  # links, unten, rechts, oben (PDF-Raum)
    stil: tuple[int, float] | None = None
    richtung = (1.0, 0.0)
    stift = (0.0, 0.0)
    zeile = 0

    def abschliessen() -> None:
        if text and stil is not None:
            spans.append(
                (
                    TextSpan(
                        x0=rahmen[0] - links_cb,
                        y0=oben_cb - rahmen[3],
                        x1=rahmen[2] - links_cb,
                        y1=oben_cb - rahmen[1],
                        text="".join(text),
                        groesse=round(stil[1], 2),
                        farbe=stil[0],
                    ),
                    zeile,
                )
            )

    def beginnen(z: _Zeichen, zeichen_text: str, box: Rahmen) -> None:
        nonlocal stil, richtung
        text.clear()
        text.append(zeichen_text)
        rahmen[:] = list(box)
        stil = (z.farbe, z.groesse)
        richtung = (z.rx, z.ry)

    for z in zeichen:
        box = (z.links, z.unten, z.rechts, z.oben)
        if stil is None:
            beginnen(z, z.zeichen, box)
            stift = (z.px, z.py)
            continue
        dx, dy = z.ox - stift[0], z.oy - stift[1]
        entlang = dx * richtung[0] + dy * richtung[1]
        quer = -dx * richtung[1] + dy * richtung[0]
        groesse = max(z.groesse, 1e-3)
        gleiche_richtung = z.rx * richtung[0] + z.ry * richtung[1] > 0.99
        neue_zeile = (
            not gleiche_richtung
            or abs(quer) > _GRUNDLINIE_MAX * groesse
            or entlang < -_NEUE_ZEILE_AB * groesse
            or entlang > _NEUE_ZEILE_AB * groesse
        )
        if neue_zeile:
            abschliessen()
            zeile += 1
            beginnen(z, z.zeichen, box)
        else:
            leer = entlang > _LEERZEICHEN_AB * groesse and z.zeichen != " " and not text[-1].endswith(" ")
            farbe, alt_groesse = stil
            stilwechsel = z.farbe != farbe or abs(z.groesse - alt_groesse) > _GROESSE_TOLERANZ * max(
                z.groesse, alt_groesse, 1e-3
            )
            if stilwechsel:
                abschliessen()
                if leer:
                    # MuPDF setzt das eingefuegte Leerzeichen als eigenen Span im
                    # Stil des Folgezeichens (gemessen: ' ' gruen vor 'Basis').
                    luecke = (
                        min(stift[0], z.ox),
                        min(stift[1], z.unten),
                        max(stift[0], z.ox),
                        max(stift[1], z.oben),
                    )
                    beginnen(z, " ", luecke)
                    abschliessen()
                beginnen(z, z.zeichen, box)
            else:
                if leer:
                    text.append(" ")
                text.append(z.zeichen)
                rahmen[0] = min(rahmen[0], box[0])
                rahmen[1] = min(rahmen[1], box[1])
                rahmen[2] = max(rahmen[2], box[2])
                rahmen[3] = max(rahmen[3], box[3])
        stift = (z.px, z.py)
    abschliessen()
    return spans


__all__ = [
    "BEZIER_SCHRITTE",
    "Gerendert",
    "Linien",
    "PdfDatei",
    "PdfQuelle",
    "Rahmen",
    "TextSpan",
    "png_aus_feld",
    "seitenzahl",
]
