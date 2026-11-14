"""Aufmass-Austauschdateien ``.d12`` lesen — Plan 27, S2 (deterministisch, kein Modell).

Der IBD-Bestand traegt 234 ``.d12``-Dateien, alle im Rechnungsordner neben den
Aufmassblaettern. Zwei Dialekte, beide am 11.08.2026 gegen echte Dateien aus
dem Bestand verifiziert (958 Ansatzzeilen), nicht gegen eine Vermutung:

**1. GAEB-VB 23.004, Datenart 12 (Festformat).** Kopfsatz ``00`` mit der
Ausgabe-Kennung ``23.004``, ein ``120``-Satz mit der GAEB-90-OZ-Maske
(z. B. ``22N112233PPPP0000``), dann je Ansatz ein ``122``-Satz, Endesatz
``99``. Spalten der ``122``-Saetze (0-basiert, konstant ueber alle
gemessenen Zeilen):

* 0-2    ``122`` (Satzart)
* 3-13   Ordnungszahl (11 Stellen It. Maske, z. B. ``00107020210``)
* 19-30  Erlaeuterung — im Bestand die Raumnummer (``03.013``, ``EG.033``)
* 44-46  Formelnummer ``091`` (Formel 91 = freier Rechenansatz; im gesamten
         Bestand die einzige vorkommende Formel)
* ab 47  der Rechenansatz als Ausdruck, abgeschlossen mit ``=``

Nach REB tragen die Austauschdateien **kein Ergebnis** — die Menge entsteht
durch Auswertung des Ansatzes (Freies REB Buch 2019, S. 14: 'In der
DA11-Datei werden kein Leistungsverzeichnis und kein Ergebnis abgestellt,
sondern lediglich die Ansaetze"). Ausgewertet wird hier mit Decimal und
REB-Rundung auf 3 Stellen, wie im Rechenkern :mod:`.aufmass`. Die **Einheit
steht nicht in der Datei** (sie steht im LV) — sie wird als Luecke
ausgewiesen, nicht erfunden.

**2. RIB-Klartext** (Kopfzeile ``REB23.003 AUFMASS``): ``S21``-Zeilen mit
Blatt, Position, Ansatztext, ``=``, Menge und Einheit; ``S99 SUMMENSATZ``.
Hier steht die Menge im Klartext in der Datei und wird gelesen, nicht
gerechnet — der Ansatztext ist RIB-frei formatiert und traegt Etiketten
(``A=13.14*H=2.19``), deren Rechenweg nicht spezifiziert ist. Gelesen wird,
was dasteht; eine Nachrechnung findet nicht statt.

Unbekannte Satzarten und nicht auswertbare Ansaetze werden als Fehler je
Zeile ausgewiesen — die Datei bleibt lesbar, nichts wird geraten.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

#: REB-Rundung (wie :mod:`.aufmass`): 3 Nachkommastellen, kaufmaennisch.
_RUNDUNG = Decimal("0.001")

_KLARTEXT_KOPF = re.compile(r"^REB\s?23\.00[34]\s+AUFMASS", re.IGNORECASE)
_S21 = re.compile(
    r"^S21\s+(?P<blatt>\S+)\s+(?P<oz>\S+)\s+(?P<ansatz>.*?)=\s*(?P<menge>-?[\d.,]+)\s*(?P<einheit>\S*)\s*$"
)

#: Erlaubte Zeichen eines freien Rechenansatzes (Formel 91): Zahlen mit
#: deutschem Komma, Grundrechenarten, Klammern. Alles andere (z. B.
#: Blattadress-Aufrufe wie ``0006C0``) ist im Bestand nicht belegt und wird
#: als nicht auswertbar gemeldet statt falsch gerechnet.
_AUSDRUCK_ERLAUBT = re.compile(r"^[\d.,+\-*/() ]+$")
_AUSDRUCK_TOKEN = re.compile(r"\d+(?:,\d+)?|[+\-*/()]")


@dataclass(slots=True)
class Ansatz:
    """Eine Aufmasszeile: Position, Ort, Rechenansatz, Menge."""

    oz: str
    ort: str | None
    ansatz: str
    menge: Decimal | None
    einheit: str | None
    zeile: int
    fehler: str | None = None
    #: REB-Kennzeichen der Zeile (nur DA11 traegt es: ``H``/``Z``/``P``/``S``).
    #: ``H`` = Hilfswert; er wird gespeichert, aber NICHT mitsummiert.
    kennzeichen: str | None = None


@dataclass(slots=True)
class PositionsSumme:
    oz: str
    menge: Decimal
    einheit: str | None
    anzahl_ansaetze: int


@dataclass(slots=True)
class D12Ergebnis:
    dialekt: str  # "gaeb_23004_da12" | "rib_klartext"
    ansaetze: list[Ansatz]
    positionen: list[PositionsSumme]
    warnungen: list[str] = field(default_factory=list)

    @property
    def fehlerzeilen(self) -> int:
        return sum(1 for a in self.ansaetze if a.fehler)


class D12Unlesbar(ValueError):
    """Die Datei ist keine lesbare Aufmass-Austauschdatei."""


def _runde(wert: Decimal) -> Decimal:
    return wert.quantize(_RUNDUNG, rounding=ROUND_HALF_UP)


def ansatz_auswerten(ausdruck: str) -> Decimal:
    """Wertet einen freien Rechenansatz (Formel 91) aus — Decimal, kein eval.

    Zugelassen sind Zahlen (deutsches Komma), ``+ - * /`` und Klammern.
    Rekursiver Abstieg, Punkt-vor-Strich, wirft ``ValueError`` bei allem,
    was nicht dazugehoert.
    """
    ausdruck = ausdruck.strip()
    if not ausdruck or not _AUSDRUCK_ERLAUBT.match(ausdruck):
        raise ValueError(f"Ansatz nicht auswertbar: {ausdruck!r}")
    tokens = _AUSDRUCK_TOKEN.findall(ausdruck)
    rest = "".join(tokens)
    if rest.replace(" ", "") != ausdruck.replace(" ", ""):
        raise ValueError(f"Ansatz enthaelt unbekannte Zeichen: {ausdruck!r}")
    pos = 0

    def _summe() -> Decimal:
        nonlocal pos
        wert = _produkt()
        while pos < len(tokens) and tokens[pos] in "+-":
            op = tokens[pos]
            pos += 1
            rechts = _produkt()
            wert = wert + rechts if op == "+" else wert - rechts
        return wert

    def _produkt() -> Decimal:
        nonlocal pos
        wert = _faktor()
        while pos < len(tokens) and tokens[pos] in "*/":
            op = tokens[pos]
            pos += 1
            rechts = _faktor()
            wert = wert * rechts if op == "*" else wert / rechts
        return wert

    def _faktor() -> Decimal:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError(f"Ansatz endet unerwartet: {ausdruck!r}")
        tok = tokens[pos]
        if tok == "(":
            pos += 1
            wert = _summe()
            if pos >= len(tokens) or tokens[pos] != ")":
                raise ValueError(f"Klammer nicht geschlossen: {ausdruck!r}")
            pos += 1
            return wert
        if tok == "-":
            pos += 1
            return -_faktor()
        if tok == "+":
            pos += 1
            return _faktor()
        pos += 1
        try:
            return Decimal(tok.replace(",", "."))
        except InvalidOperation as exc:
            raise ValueError(f"Keine Zahl: {tok!r}") from exc

    try:
        ergebnis = _summe()
    except (InvalidOperation, ZeroDivisionError) as exc:
        raise ValueError(f"Ansatz nicht auswertbar: {ausdruck!r} ({exc})") from exc
    if pos != len(tokens):
        raise ValueError(f"Ansatz nicht vollstaendig gelesen: {ausdruck!r}")
    return ergebnis


def _dekodieren(daten: bytes) -> str:
    for encoding in ("cp1252", "latin-1"):
        try:
            return daten.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise D12Unlesbar("Datei ist nicht als Text dekodierbar.")


def _positionssummen(ansaetze: list[Ansatz]) -> list[PositionsSumme]:
    reihenfolge: list[str] = []
    summen: dict[str, Decimal] = {}
    zaehler: dict[str, int] = {}
    einheiten: dict[str, set[str]] = {}
    for a in ansaetze:
        if a.menge is None:
            continue
        # REB 23.003: Hilfswerte (Kennzeichen "H") werden "nicht zu den
        # Positions-, Zwischen- und Gesamtsummen addiert, sondern nur
        # gespeichert" (Freies REB Buch 2019, Abschnitt 2.2). Sie zaehlen
        # hier also nicht mit — sonst stuende ein Zwischenwert doppelt drin.
        if a.kennzeichen == "H":
            continue
        if a.oz not in summen:
            summen[a.oz] = Decimal(0)
            zaehler[a.oz] = 0
            einheiten[a.oz] = set()
            reihenfolge.append(a.oz)
        summen[a.oz] += a.menge
        zaehler[a.oz] += 1
        if a.einheit:
            einheiten[a.oz].add(a.einheit)
    return [
        PositionsSumme(
            oz=oz,
            menge=_runde(summen[oz]),
            # Nur eine EINDEUTIGE Einheit wird uebernommen — gemischte oder
            # fehlende Einheiten bleiben None (Luecke, kein Ratewert).
            einheit=next(iter(einheiten[oz])) if len(einheiten[oz]) == 1 else None,
            anzahl_ansaetze=zaehler[oz],
        )
        for oz in reihenfolge
    ]


def _lese_gaeb_da12(zeilen: list[str]) -> D12Ergebnis:
    ansaetze: list[Ansatz] = []
    warnungen: list[str] = []
    bekannte_praefixe = ("00", "99", "120", "121", "122", "123")
    for nr, zeile in enumerate(zeilen, start=1):
        if not zeile.strip():
            continue
        if not zeile.startswith("122"):
            if not zeile.startswith(bekannte_praefixe):
                warnungen.append(
                    f"Zeile {nr}: unbekannte Satzart {zeile[:3]!r} uebersprungen"
                )
            continue
        oz = zeile[3:14].strip()
        ort = zeile[19:31].strip() or None
        formel = zeile[44:47]
        gleich = zeile.find("=", 47)
        ausdruck = zeile[47:gleich].strip() if gleich >= 0 else ""
        if formel != "091":
            ansaetze.append(
                Ansatz(
                    oz=oz,
                    ort=ort,
                    ansatz=ausdruck,
                    menge=None,
                    einheit=None,
                    zeile=nr,
                    fehler=f"Formel {formel.strip() or '?'} wird nicht ausgewertet (nur Formel 91 ist im Bestand belegt)",
                )
            )
            continue
        if gleich < 0:
            ansaetze.append(
                Ansatz(
                    oz=oz,
                    ort=ort,
                    ansatz=zeile[47:80].strip(),
                    menge=None,
                    einheit=None,
                    zeile=nr,
                    fehler="Ansatz ohne abschliessendes '='",
                )
            )
            continue
        try:
            menge = _runde(ansatz_auswerten(ausdruck))
            fehler = None
        except ValueError as exc:
            menge, fehler = None, str(exc)
        ansaetze.append(
            Ansatz(
                oz=oz,
                ort=ort,
                ansatz=ausdruck,
                menge=menge,
                einheit=None,
                zeile=nr,
                fehler=fehler,
            )
        )
    if not ansaetze:
        raise D12Unlesbar("Keine 122-Ansatzzeilen gefunden.")
    warnungen.append(
        "Einheiten stehen nach GAEB-VB 23.004 nicht in der DA12-Datei, sondern im LV — Luecke, kein Ratewert."
    )
    return D12Ergebnis(
        dialekt="gaeb_23004_da12",
        ansaetze=ansaetze,
        positionen=_positionssummen(ansaetze),
        warnungen=warnungen,
    )


def _lese_rib_klartext(zeilen: list[str]) -> D12Ergebnis:
    ansaetze: list[Ansatz] = []
    warnungen: list[str] = []
    for nr, zeile in enumerate(zeilen, start=1):
        if not zeile.startswith("S21"):
            continue
        m = _S21.match(zeile)
        if m is None:
            ansaetze.append(
                Ansatz(
                    oz="?",
                    ort=None,
                    ansatz=zeile[4:].strip(),
                    menge=None,
                    einheit=None,
                    zeile=nr,
                    fehler="S21-Zeile nicht lesbar",
                )
            )
            continue
        try:
            menge = (
                Decimal(m.group("menge").replace(".", "").replace(",", "."))
                if "," in m.group("menge")
                else Decimal(m.group("menge"))
            )
        except InvalidOperation:
            menge = None
        ansaetze.append(
            Ansatz(
                oz=m.group("oz"),
                ort=m.group("blatt"),
                ansatz=m.group("ansatz").strip(),
                # Die Menge steht hier IN der Datei und wird gelesen. Der
                # Ansatztext ist RIB-frei etikettiert (z. B. 'A=13.14*H=2.19')
                # und wird bewusst NICHT nachgerechnet.
                menge=menge,
                einheit=m.group("einheit") or None,
                zeile=nr,
                fehler=None if menge is not None else "Menge nicht lesbar",
            )
        )
    if not ansaetze:
        raise D12Unlesbar("Keine S21-Aufmasszeilen gefunden.")
    return D12Ergebnis(
        dialekt="rib_klartext",
        ansaetze=ansaetze,
        positionen=_positionssummen(ansaetze),
        warnungen=warnungen,
    )


def aufmass_lesen(daten: bytes) -> D12Ergebnis:
    """Liest eine REB-Aufmassdatei — der Dialekt wird am Satzaufbau erkannt.

    Neben den beiden ``.d12``-Dialekten wird auch die **DA11** gelesen
    (:mod:`.da11`) — dieselbe Regelung, andere Datenart, 80 statt ~175
    Stellen. Der Import steht bewusst hier in der Funktion: :mod:`.da11`
    nutzt die Datenklassen und den Rechenkern dieses Moduls, ein Import auf
    Modulebene waere ein Zirkel.
    """
    from .da11 import da11_lesen, ist_da11

    text = _dekodieren(daten)
    zeilen = text.splitlines()
    if not zeilen:
        raise D12Unlesbar("Datei ist leer.")
    if _KLARTEXT_KOPF.match(zeilen[0]):
        return _lese_rib_klartext(zeilen)
    if any(z.startswith("122") for z in zeilen[:200]):
        return _lese_gaeb_da12(zeilen)
    if ist_da11(zeilen):
        return da11_lesen(daten)
    raise D12Unlesbar(
        f"Weder GAEB-DA12-Saetze noch DA11-Saetze noch RIB-Klartextkopf gefunden "
        f"(Kopf: {zeilen[0][:40]!r})."
    )
