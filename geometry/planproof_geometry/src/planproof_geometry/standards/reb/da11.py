"""REB-VB 23.003 Datenart 11 (``.d11``) schreiben und lesen - der Rueckweg der Menge.

Die DA11 ist der Weg, auf dem eine in Planproof ermittelte Menge in die
Abrechnungswelt des Auftraggebers kommt: eine ASCII-Datei mit **80 Stellen
Satzlaenge**, die *kein Ergebnis* traegt, sondern nur die Rechenansaetze. Der
Empfaenger rechnet selbst nach - genau das macht sie pruefbar.

**Format-Grundlage (21.08.2026, Primaerquelle).** "Das Freie REB Buch",
Version 2019 (MWM, Herausgeber des BVBS-zertifizierten MWM-Libero),
Abschnitt 2.2 "Die Austauschdatei DA11", Seite 15. Die dortige Feldtabelle
ist hier 1:1 umgesetzt und zusaetzlich gegen die **109 Musterzeilen** der
beiden Musterdateien in Abschnitt 2.8 gemessen (Ausgabe 1979 und 2009):
jede Musterzeile ist nach ``rstrip`` exakt 75 Zeichen lang, Spalte 12 ist in
allen 109 Zeilen leer, Spalte 13 traegt ausschliesslich die dokumentierten
Kennzeichen (leer, ``*``, ``H``, ``Z``), Spalten 70-73 sind immer vierstellig
numerisch. Die Spaltenlage ist damit nicht angenommen, sondern gemessen.

Satzaufbau eines Ansatzsatzes (1-basiert, wie im Buch)::

     1- 2  Datenart "11"
     3-11  Ordnungszahl (9 Stellen)
    12     V
    13     Kennzeichen (leer | * | H | Z | P | S)
    14-22  Erlaeuterung (9 Stellen)
    23     ggf. Vorzeichen fuer den Faktor
    24-29  Faktor (keine Eingabe = 1)
    30-31  Formelnummer
    32-69  Werte (38 Stellen)
    70-73  Blattnummer
    74     Zeile (A-Z)
    75     Index (0-9)
    76-80  zur besonderen Verwendung

**Warum ausschliesslich Formel 91.** Der Formelkatalog der REB 23.003 hat 25
bzw. 26 Formeln; ihre Wertefelder codieren Zahlen mit einer *impliziten*
Nachkommastelligkeit (im Muster steht ``3250`` fuer 3,250 m). Diese Stelligkeit
steht in der frei verfuegbaren Quelle **nirgends** - die Formeltafeln der
Seiten 7-12 sind Bilder ohne Textebene. Eine geratene Kommastelle waere ein
Faktor 1000 auf einer Abrechnungsmenge. Formel 91 ("Freie Formel") traegt
stattdessen den Rechenansatz woertlich mit deutschem Komma, ist im Buch
ausdruecklich beschrieben (max. 6 bzw. 20 Zeilen zu je 38 Zeichen, mit ``=``
abgeschlossen) und ist die einzige Formel, die im gesamten IBD-Bestand
vorkommt. Sie ist exakt, pruefbar - und :func:`da11_lesen` rechnet jeden
geschriebenen Ansatz mit demselben Kern nach, mit dem er entstanden ist.

**Was hier bewusst leer bleibt.** Das Faktorfeld (24-29) und sein Vorzeichen
(23). In den Mustern steht dort ``  2000`` (zwei gleiche Achsen) und ``  1000``
(GAEB-Fachdokumentation 3.3, Beispiel zu Kapitel 7.5) - beides passt zu einer
impliziten Kommastelligkeit von 3, aber geschrieben steht das nirgends. Die
Vermutung ist also gut belegt und bleibt trotzdem eine. Faktor und Anzahl wandern
darum in den Rechenansatz selbst (``...*2``), wo sie eindeutig sind und vom
eigenen Leser nachgerechnet werden koennen. Ein leeres Faktorfeld heisst nach
Buch "= 1", also genau das, was dann gilt.

**Rundung, ehrlich benannt.** Der Rechenkern von Planproof rundet jeden Ansatz
auf 3 Nachkommastellen und summiert danach - das ist die Methode der **REB
23.003 Ausgabe 1979**. Die **Ausgabe 2009** rundet intern gar nicht mehr
(Buch, Abschnitt 2.7) und kann darum in der dritten Nachkommastelle
abweichen; das Buch zeigt das an einem Beispiel (10,421 gegen 10,422). Die
DA11 traegt kein Ergebnis, der Unterschied entsteht also erst beim Empfaenger.
Wer exakte Uebereinstimmung braucht, schreibt mit ``ausgabe="1979"``; die
Voreinstellung ist 2009, weil nur sie freie Ordnungszahl-Masken erlaubt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal

from .d12 import Ansatz, D12Ergebnis, D12Unlesbar, _positionssummen, _runde, ansatz_auswerten

#: Satzlaenge nach Buch, Abschnitt 2.2 ("ASCII-Datei mit 80 Stellen Satzlaenge").
SATZLAENGE = 80

#: Spaltenlage (0-basiert, halboffen) - aus der Feldtabelle Seite 15.
SP_DATENART = (0, 2)
SP_OZ = (2, 11)
SP_V = (11, 12)
SP_KENNZEICHEN = (12, 13)
SP_ERLAEUTERUNG = (13, 22)
SP_VORZEICHEN = (22, 23)
SP_FAKTOR = (23, 29)
SP_FORMEL = (29, 31)
SP_WERTE = (31, 69)
SP_BLATT = (69, 73)
SP_ZEILE = (73, 74)
SP_INDEX = (74, 75)

#: Kommentarzeilen (Kennzeichen ``*``) tragen ihren Text ueber die Felder
#: Erlaeuterung bis Werte hinweg: "steht im Spaltenbereich 14-69 der Kommentar".
SP_KOMMENTAR = (13, 69)

#: Breite des Wertefelds - "maximal 6 bzw. 20 (Ausgabe 2009) Zeilen (mit je
#: 38 Zeichen)".
WERTE_BREITE = SP_WERTE[1] - SP_WERTE[0]

#: Zeilenzahl eines Formel-91-Ansatzes je Ausgabe (Buch, Abschnitt 2.1).
MAX_ZEILEN = {"1979": 6, "2009": 20}

#: Die einzige Formel, die geschrieben und ausgewertet wird (siehe Modul-Doku).
FORMEL_FREI = "91"

#: Kennzeichen nach Buch, Abschnitt 2.2. ``E`` gibt es nur 1979, ``S`` nur 2009.
KENNZEICHEN_JE_AUSGABE = {
    "1979": frozenset({" ", "H", "E", "Z", "P"}),
    "2009": frozenset({" ", "H", "Z", "P", "S"}),
}

#: Hilfswerte werden "nicht zu den Positions-, Zwischen- und Gesamtsummen
#: addiert, sondern nur gespeichert" (Buch, Abschnitt 2.2).
KZ_HILFSWERT = "H"

#: Kopfsatz: Datenart 00, ab Spalte 11 die Ausgabe-Kennung, dann das Jahr.
KOPF_DATENART = "00"
AUSGABE_KENNUNG = "23.003"
SP_KOPF_KENNUNG = (10, 16)
SP_KOPF_JAHR = (16, 20)
SP_KOPF_PROJEKT = (20, 68)
SP_KOPF_MASKE = (68, 77)

#: Die DA11 ist eine ASCII-/ANSI-Datei; cp1252 haelt die Ein-Byte-Regel ein,
#: ohne die eine feste Spaltenlage nicht existieren kann.
KODIERUNG = "cp1252"
ZEILENENDE = "\r\n"

#: Blattadresse: "Blattnummer mit 4 Stellen", "Zeilen-Nummer mit 1 Stelle
#: (Buchstaben A-Z)", "Index mit 1 Stelle (0-9)".
_ZEILEN_BUCHSTABEN = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_MAX_BLATT = 9999

#: Ein Rechenansatz zerfaellt in unteilbare Atome: Zahlen (deutsches Komma),
#: Grundrechenarten, Klammern, der Abschluss. An Atomgrenzen wird umgebrochen,
#: nie mitten in einer Zahl.
_ATOM = re.compile(r"\d+(?:,\d+)?|[+\-*/()=]")

_OZ_TRENNER = re.compile(r"[.\-/\s]+")


class DA11Unschreibbar(ValueError):
    """Die Ansaetze lassen sich nicht als DA11 abstellen."""


@dataclass(slots=True)
class Mengenansatz:
    """Ein Ansatz, wie er in die DA11 geschrieben wird.

    ``ausdruck`` ist der freie Rechenansatz OHNE abschliessendes ``=`` (das
    setzt der Writer). Zugelassen ist genau das, was :func:`ansatz_auswerten`
    auswerten kann - was der eigene Kern nicht nachrechnen kann, wird nicht
    geschrieben, sondern ausgesteuert.
    """

    oz: str
    ausdruck: str
    erlaeuterung: str | None = None
    kommentar: str | None = None
    kennzeichen: str = " "


@dataclass(slots=True)
class Ausgesteuert:
    """Ein Ansatz, der bewusst NICHT in der Datei steht - mit Grund."""

    oz: str
    grund: str


@dataclass(slots=True)
class SatzBau:
    """Die erzeugten Ansatzsaetze - flach und je Ansatz gruppiert.

    Die Gruppierung braucht die X31, die dieselben Saetze nach Ordnungszahl
    sortiert in ihre Positionen haengt. Sie entstehen in EINEM Lauf, damit die
    Blattadressen ueber die ganze Datei eindeutig bleiben - eine Adresse ist der
    Anker, ueber den ein Rechenansatz referenziert wird.
    """

    saetze: list[str]
    je_ansatz: list[tuple[Mengenansatz, list[str]]]
    oz_maske: str
    ausgesteuert: list[Ausgesteuert] = field(default_factory=list)
    warnungen: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DA11Ergebnis:
    daten: bytes
    saetze: list[str]
    ansaetze: int
    oz_maske: str
    ausgabe: str
    ausgesteuert: list[Ausgesteuert] = field(default_factory=list)
    warnungen: list[str] = field(default_factory=list)


# --- Hilfen ---------------------------------------------------------------


def _feld(zeile: str, spalten: tuple[int, int]) -> str:
    return zeile[spalten[0] : spalten[1]]


def _setze(puffer: list[str], spalten: tuple[int, int], wert: str) -> None:
    """Schreibt ``wert`` linksbuendig in das Feld - laenger wird abgeschnitten.

    Das Abschneiden ist nur fuer Textfelder gedacht (Erlaeuterung, Kommentar,
    Projektname); Zahlenfelder werden vorher geprueft und sonst ausgesteuert.
    """
    breite = spalten[1] - spalten[0]
    text = wert[:breite].ljust(breite)
    puffer[spalten[0] : spalten[1]] = list(text)


def _leerzeile() -> list[str]:
    return [" "] * SATZLAENGE


def zahl(wert: Decimal) -> str:
    """Decimal als REB-Zahl: deutsches Komma, keine Exponentialschreibweise."""
    text = format(wert, "f")
    return text.replace(".", ",")


class _Adressen:
    """Vergibt Blattadressen - Buchstabe laeuft schnell, Index langsam.

    Das ist die Reihenfolge, in der die Musterdateien lesen (0001B0, 0001C0,
    0001D0). 260 Adressen je Blatt, danach das naechste Blatt.
    """

    def __init__(self, blatt: int = 1) -> None:
        self._blatt = blatt
        self._buchstabe = 0
        self._index = 0

    def naechste(self) -> str:
        if self._blatt > _MAX_BLATT:
            raise DA11Unschreibbar(
                f"Mehr als {_MAX_BLATT * 260} Saetze - die Blattnummer der REB 23.003 hat 4 Stellen."
            )
        adresse = f"{self._blatt:04d}{_ZEILEN_BUCHSTABEN[self._buchstabe]}{self._index}"
        self._buchstabe += 1
        if self._buchstabe >= len(_ZEILEN_BUCHSTABEN):
            self._buchstabe = 0
            self._index += 1
            if self._index > 9:
                self._index = 0
                self._blatt += 1
        return adresse


def _oz_segmente(oz: str) -> list[str]:
    """Zerlegt eine Ordnungszahl in ihre Segmente (Punkte, Striche, Schraegstriche)."""
    return [teil for teil in _OZ_TRENNER.split(oz.strip()) if teil]


def oz_maske_bauen(ozs: list[str]) -> tuple[str, list[str]]:
    """Leitet die OZ-Maske aus den tatsaechlichen Ordnungszahlen ab.

    Die Maske beschreibt den Aufbau der Ordnungszahl: je Hierarchiestufe deren
    Ziffer, fuer die Positionsnummer ``P`` (Buch, Abschnitt 2.5: ``1122PPPPI``
    = 2 Stellen Los, 2 Stellen Titel, 4 Stellen Position, 1 Stelle Index).

    Haben nicht alle Ordnungszahlen dasselbe Stellenmuster, laesst sich keine
    gemeinsame Hierarchie-Maske angeben. Dann wird eine flache Maske
    (``PPPPPPPP``) geschrieben und der Verlust benannt - die Ziffern selbst
    bleiben vollstaendig, nur die Gliederung ist dann nicht deklariert.
    """
    warnungen: list[str] = []
    muster: set[tuple[int, ...]] = set()
    for oz in ozs:
        segmente = _oz_segmente(oz)
        if segmente:
            muster.add(tuple(len(s) for s in segmente))
    if not muster:
        return "P" * 9, warnungen
    if len(muster) == 1:
        breiten = next(iter(muster))
        maske = ""
        for stufe, breite in enumerate(breiten[:-1], start=1):
            maske += str(stufe)[-1] * breite
        maske += "P" * breiten[-1]
        return maske, warnungen
    breite = max(sum(m) for m in muster)
    warnungen.append(
        f"Die {len(muster)} verschiedenen Ordnungszahl-Stellenmuster lassen keine gemeinsame "
        f"Hierarchie-Maske zu - geschrieben wird die flache Maske {'P' * breite!r}. Die Ziffern "
        "der Ordnungszahlen bleiben vollstaendig, ihre Gliederung ist in der Datei aber nicht "
        "deklariert."
    )
    return "P" * breite, warnungen


def oz_stellen(oz: str) -> str:
    """Ordnungszahl als Ziffernfolge ohne Trenner (so steht sie in der DA11)."""
    return "".join(_oz_segmente(oz))


def oz_aus_maske(ziffern: str, maske: str) -> str:
    """Setzt die Trenner anhand der Maske wieder ein - Gegenstueck zu :func:`oz_stellen`."""
    ziffern = ziffern.strip()
    if not maske or not ziffern:
        return ziffern
    # Gleiche aufeinanderfolgende Maskenzeichen bilden ein Segment.
    segmente: list[int] = []
    letztes = ""
    for zeichen in maske:
        if zeichen == letztes:
            segmente[-1] += 1
        else:
            segmente.append(1)
            letztes = zeichen
    teile: list[str] = []
    pos = 0
    for breite in segmente:
        if pos >= len(ziffern):
            break
        teile.append(ziffern[pos : pos + breite])
        pos += breite
    if pos < len(ziffern):
        teile.append(ziffern[pos:])
    return ".".join(t for t in teile if t)


def _kodiere(text: str, warnungen: list[str], feldname: str) -> str:
    """Haelt die Ein-Byte-Regel ein: was cp1252 nicht kann, wird benannt."""
    try:
        text.encode(KODIERUNG)
    except UnicodeEncodeError:
        ersetzt = text.encode(KODIERUNG, errors="replace").decode(KODIERUNG)
        warnungen.append(
            f"{feldname}: Zeichen ausserhalb des DA11-Zeichensatzes (cp1252) durch '?' ersetzt - "
            f"{text!r} wurde zu {ersetzt!r}."
        )
        return ersetzt
    return text


def _teile_ausdruck(payload: str) -> list[str]:
    """Bricht den Rechenansatz an Atomgrenzen auf Zeilen zu 38 Zeichen um.

    Umgebrochen wird nur zwischen Atomen - eine Zahl wird nie zerrissen.
    """
    atome = _ATOM.findall(payload)
    if "".join(atome) != payload:
        raise DA11Unschreibbar(f"Rechenansatz enthaelt unbekannte Zeichen: {payload!r}")
    zeilen: list[str] = []
    aktuell = ""
    for atom in atome:
        if len(atom) > WERTE_BREITE:
            raise DA11Unschreibbar(
                f"Ein einzelner Wert ist laenger als das Wertefeld ({WERTE_BREITE} Zeichen): {atom!r}"
            )
        if len(aktuell) + len(atom) > WERTE_BREITE:
            zeilen.append(aktuell)
            aktuell = atom
        else:
            aktuell += atom
    if aktuell:
        zeilen.append(aktuell)
    return zeilen


def _kommentarzeilen(text: str) -> list[str]:
    """Bricht einen langen Kommentar auf die Kommentarbreite (Spalten 14-69) um."""
    breite = SP_KOMMENTAR[1] - SP_KOMMENTAR[0]
    worte = text.split()
    zeilen: list[str] = []
    aktuell = ""
    for wort in worte:
        kandidat = f"{aktuell} {wort}".strip()
        if len(kandidat) > breite:
            if aktuell:
                zeilen.append(aktuell)
            # Ein einzelnes Wort, das breiter als das Feld ist (Pfad, URL), wird
            # umgebrochen statt abgeschnitten - abschneiden waere stiller Verlust.
            while len(wort) > breite:
                zeilen.append(wort[:breite])
                wort = wort[breite:]
            aktuell = wort
        else:
            aktuell = kandidat
    if aktuell:
        zeilen.append(aktuell)
    return zeilen


# --- Schreiben ------------------------------------------------------------


def _saetze_fuer_ansatz(
    ansatz: Mengenansatz,
    *,
    oz_feld: str,
    adressen: _Adressen,
    max_zeilen: int,
    warnungen: list[str],
) -> list[str]:
    saetze: list[str] = []

    if ansatz.kommentar:
        for text in _kommentarzeilen(_kodiere(ansatz.kommentar, warnungen, "Kommentar")):
            puffer = _leerzeile()
            _setze(puffer, SP_DATENART, "11")
            _setze(puffer, SP_OZ, oz_feld)
            _setze(puffer, SP_KENNZEICHEN, "*")
            _setze(puffer, SP_KOMMENTAR, text)
            adresse = adressen.naechste()
            _setze(puffer, SP_BLATT, adresse[:4])
            _setze(puffer, SP_ZEILE, adresse[4])
            _setze(puffer, SP_INDEX, adresse[5])
            saetze.append("".join(puffer))

    teile = _teile_ausdruck(ansatz.ausdruck + "=")
    if len(teile) > max_zeilen:
        raise DA11Unschreibbar(
            f"Der Rechenansatz braucht {len(teile)} Zeilen, die Formel 91 erlaubt in der "
            f"Ausgabe dieser Datei hoechstens {max_zeilen}."
        )

    erlaeuterung = _kodiere(ansatz.erlaeuterung or "", warnungen, "Erlaeuterung")
    for nr, teil in enumerate(teile):
        letzte = nr == len(teile) - 1
        puffer = _leerzeile()
        _setze(puffer, SP_DATENART, "11")
        _setze(puffer, SP_OZ, oz_feld)
        _setze(puffer, SP_FORMEL, FORMEL_FREI)
        _setze(puffer, SP_WERTE, teil)
        if letzte:
            # Buch, Abschnitt 2.2: "bei mehrzeiligen Rechenansaetzen das
            # Kennzeichen in der letzten Zeile einzutragen".
            _setze(puffer, SP_KENNZEICHEN, ansatz.kennzeichen or " ")
            if erlaeuterung:
                _setze(puffer, SP_ERLAEUTERUNG, erlaeuterung)
        adresse = adressen.naechste()
        _setze(puffer, SP_BLATT, adresse[:4])
        _setze(puffer, SP_ZEILE, adresse[4])
        _setze(puffer, SP_INDEX, adresse[5])
        saetze.append("".join(puffer))
    return saetze


def da11_saetze(
    ansaetze: list[Mengenansatz],
    *,
    ausgabe: str = "2009",
    oz_maske: str | None = None,
    oz_im_satz: bool = True,
) -> SatzBau:
    """Erzeugt die Ansatzsaetze (ohne Kopfsatz) - gemeinsamer Kern von DA11 und X31.

    ``oz_im_satz=False`` laesst die Ordnungszahl im Satz leer: die X31 traegt sie
    ueber das uebergeordnete ``Item`` und blendet die Stellen 1-11 aus
    (GAEB-Fachdokumentation 7.2.2). Dann entfaellt auch die 9-Stellen-Grenze der
    DA11 - genau das ist der Grund, warum die X31 bis zu 14 Stellen kann.
    """
    if ausgabe not in MAX_ZEILEN:
        raise DA11Unschreibbar(f"Unbekannte REB-Ausgabe {ausgabe!r} - bekannt sind {sorted(MAX_ZEILEN)}.")
    erlaubte_kz = KENNZEICHEN_JE_AUSGABE[ausgabe]
    max_zeilen = MAX_ZEILEN[ausgabe]

    warnungen: list[str] = []
    oz_breite = SP_OZ[1] - SP_OZ[0]
    if not oz_im_satz:
        oz_maske = ""
    elif oz_maske is None:
        # Die Maske beschreibt, was tatsaechlich in der Datei landet: zu lange
        # Ordnungszahlen werden weiter unten einzeln ausgesteuert und duerfen
        # die Maske der uebrigen nicht sprengen.
        passende = [a.oz for a in ansaetze if len(oz_stellen(a.oz)) <= oz_breite]
        if not passende:
            raise DA11Unschreibbar(
                f"Keine Ordnungszahl passt in die {oz_breite} Stellen der DA11 - die X31 traegt "
                "bis zu 14 Stellen."
            )
        oz_maske, maske_warnungen = oz_maske_bauen(passende)
        warnungen.extend(maske_warnungen)
    if len(oz_maske) > oz_breite:
        raise DA11Unschreibbar(
            f"Die OZ-Maske {oz_maske!r} braucht {len(oz_maske)} Stellen, die DA11 hat {oz_breite}."
        )

    saetze: list[str] = []
    je_ansatz: list[tuple[Mengenansatz, list[str]]] = []
    ausgesteuert: list[Ausgesteuert] = []
    adressen = _Adressen()
    for ansatz in ansaetze:
        ziffern = oz_stellen(ansatz.oz)
        if not ziffern:
            ausgesteuert.append(Ausgesteuert(oz=ansatz.oz, grund="Ordnungszahl fehlt."))
            continue
        if oz_im_satz and not ziffern.isdigit():
            ausgesteuert.append(
                Ausgesteuert(
                    oz=ansatz.oz,
                    grund=f"Ordnungszahl enthaelt nicht nur Ziffern ({ziffern!r}); die DA11 fuehrt "
                    "eine reine Ziffernfolge nach OZ-Maske. Die X31 traegt sie unveraendert.",
                )
            )
            continue
        if oz_im_satz and len(ziffern) > oz_breite:
            ausgesteuert.append(
                Ausgesteuert(
                    oz=ansatz.oz,
                    grund=f"Ordnungszahl hat {len(ziffern)} Stellen, die DA11 traegt hoechstens "
                    f"{oz_breite}. Die X31 traegt bis zu 14 Stellen.",
                )
            )
            continue
        if (ansatz.kennzeichen or " ") not in erlaubte_kz:
            ausgesteuert.append(
                Ausgesteuert(
                    oz=ansatz.oz,
                    grund=f"Kennzeichen {ansatz.kennzeichen!r} gibt es in der Ausgabe {ausgabe} nicht "
                    f"(erlaubt: {sorted(erlaubte_kz)}).",
                )
            )
            continue
        # Nichts wird abgestellt, was der eigene Kern nicht nachrechnen kann.
        try:
            ansatz_auswerten(ansatz.ausdruck)
        except ValueError as exc:
            ausgesteuert.append(Ausgesteuert(oz=ansatz.oz, grund=f"Rechenansatz nicht auswertbar: {exc}"))
            continue
        try:
            eigene = _saetze_fuer_ansatz(
                ansatz,
                oz_feld=ziffern.ljust(oz_breite) if oz_im_satz else " " * oz_breite,
                adressen=adressen,
                max_zeilen=max_zeilen,
                warnungen=warnungen,
            )
        except DA11Unschreibbar as exc:
            ausgesteuert.append(Ausgesteuert(oz=ansatz.oz, grund=str(exc)))
            continue
        saetze.extend(eigene)
        je_ansatz.append((ansatz, eigene))
    return SatzBau(
        saetze=saetze,
        je_ansatz=je_ansatz,
        oz_maske=oz_maske,
        ausgesteuert=ausgesteuert,
        # Dieselbe Warnung 500-mal ist keine Information mehr; die Aussteuerungen
        # bleiben davon unberuehrt - jede einzelne wird weiter benannt.
        warnungen=list(dict.fromkeys(warnungen)),
    )


def kopfsatz(*, projekt: str, ausgabe: str, oz_maske: str, warnungen: list[str]) -> str:
    """Startsatz (Datenart 00) mit Projektinformation, REB-Ausgabe und OZ-Maske.

    Die Feldtabelle des Buches beschreibt nur die Ansatzsaetze; die Spaltenlage
    des Kopfsatzes ist an der Musterdatei der Ausgabe 2009 **gemessen** (Kennung
    ab Spalte 11, Jahr ab 17, Maske ab 69) und deckt sich mit den ``00``-Saetzen
    der 234 DA12-Dateien des IBD-Bestands, die ihre Kennung ``23.004`` ebenfalls
    ab Spalte 11 tragen.
    """
    puffer = _leerzeile()
    _setze(puffer, SP_DATENART, KOPF_DATENART)
    _setze(puffer, SP_KOPF_KENNUNG, AUSGABE_KENNUNG)
    _setze(puffer, SP_KOPF_JAHR, ausgabe)
    _setze(puffer, SP_KOPF_PROJEKT, _kodiere(projekt, warnungen, "Projektbezeichnung"))
    _setze(puffer, SP_KOPF_MASKE, oz_maske)
    return "".join(puffer)


def da11_schreiben(
    ansaetze: list[Mengenansatz],
    *,
    projekt: str = "",
    ausgabe: str = "2009",
    oz_maske: str | None = None,
) -> DA11Ergebnis:
    """Schreibt eine DA11-Datei aus Rechenansaetzen.

    Args:
        ansaetze: die Ansaetze; ihr ``ausdruck`` muss vom eigenen Kern
            auswertbar sein, sonst wird der Ansatz ausgesteuert.
        projekt: Projektbezeichnung fuer den Kopfsatz (48 Stellen).
        ausgabe: ``"2009"`` (Voreinstellung, freie OZ-Maske) oder ``"1979"``.
        oz_maske: optionale Vorgabe; sonst aus den Ordnungszahlen abgeleitet.

    Returns:
        :class:`DA11Ergebnis` mit den Bytes, der Satzliste, den bewusst nicht
        geschriebenen Ansaetzen (mit Grund) und den Warnungen.
    """
    bau = da11_saetze(ansaetze, ausgabe=ausgabe, oz_maske=oz_maske)
    kopf = kopfsatz(projekt=projekt, ausgabe=ausgabe, oz_maske=bau.oz_maske, warnungen=bau.warnungen)
    alle = [kopf, *bau.saetze]
    text = ZEILENENDE.join(zeile.rstrip() for zeile in alle) + ZEILENENDE
    return DA11Ergebnis(
        daten=text.encode(KODIERUNG),
        saetze=bau.saetze,
        ansaetze=len(bau.je_ansatz),
        oz_maske=bau.oz_maske,
        ausgabe=ausgabe,
        ausgesteuert=bau.ausgesteuert,
        warnungen=bau.warnungen,
    )


# --- Lesen ----------------------------------------------------------------


def _dekodieren(daten: bytes) -> str:
    for kodierung in (KODIERUNG, "latin-1"):
        try:
            return daten.decode(kodierung)
        except UnicodeDecodeError:
            continue
    raise D12Unlesbar("Datei ist nicht als Text dekodierbar.")


def ist_da11(zeilen: list[str]) -> bool:
    """Erkennt eine DA11 an ihren Ansatzsaetzen (Datenart 11, Formelfeld 30-31)."""
    for zeile in zeilen[:200]:
        if len(zeile) >= SP_FORMEL[1] and zeile.startswith("11") and _feld(zeile, SP_FORMEL).strip():
            return True
    return False


def _adresse(zeile: str) -> str:
    return (_feld(zeile, SP_BLATT) + _feld(zeile, SP_ZEILE) + _feld(zeile, SP_INDEX)).strip()


def _aufgeloest(ausdruck: str, werte: dict[str, Decimal]) -> str:
    """Ersetzt Blattadress-Aufrufe durch ihren gespeicherten Wert.

    Buch, Abschnitt 2.2: Adressen gekennzeichneter Zeilen (1979) bzw. saemtliche
    Zeilen (2009) koennen spaeter wieder aufgerufen werden. Eine Adresse ist
    vierstellig numerisch + Buchstabe + Ziffer und damit von einer Zahl mit
    deutschem Komma eindeutig unterscheidbar.
    """
    return re.sub(
        r"\d{4}[A-Z]\d",
        lambda m: zahl(werte[m.group(0)]) if m.group(0) in werte else m.group(0),
        ausdruck,
    )


def da11_lesen(daten: bytes) -> D12Ergebnis:
    """Liest eine DA11-Datei - Ansaetze, Mengen, Positionssummen.

    Ausgewertet wird ausschliesslich Formel 91 (siehe Modul-Doku); jede andere
    Formelnummer wird als Zeile mit Fehlergrund gefuehrt statt geraten.
    Blattadress-Aufrufe werden aufgeloest, solange der aufgerufene Wert schon
    gelesen ist; Vorwaertsverweise (in der Ausgabe 2009 erlaubt) werden in einem
    zweiten Durchlauf nachgezogen.
    """
    text = _dekodieren(daten)
    zeilen = text.splitlines()
    if not zeilen:
        raise D12Unlesbar("Datei ist leer.")

    warnungen: list[str] = []
    maske = ""
    for zeile in zeilen:
        if zeile.startswith(KOPF_DATENART) and _feld(zeile, SP_KOPF_KENNUNG).strip().startswith("23."):
            maske = _feld(zeile, SP_KOPF_MASKE).strip()
            jahr = _feld(zeile, SP_KOPF_JAHR).strip()
            if jahr:
                warnungen.append(f"REB 23.003 Ausgabe {jahr} laut Kopfsatz.")
            break

    ansaetze: list[Ansatz] = []
    kommentare = 0
    # (Rohansatz, Kennzeichen, OZ, Erlaeuterung, Adresse, Zeilennummer)
    roh: list[tuple[str, str, str, str, str, int]] = []
    puffer = ""
    offene_oz = ""
    for nr, zeile in enumerate(zeilen, start=1):
        if not zeile.strip():
            continue
        if not zeile.startswith("11"):
            if not zeile.startswith(KOPF_DATENART) and not zeile.startswith("99"):
                warnungen.append(f"Zeile {nr}: unbekannte Datenart {zeile[:2]!r} uebersprungen")
            continue
        zeile = zeile.ljust(SATZLAENGE)
        kennzeichen = _feld(zeile, SP_KENNZEICHEN)
        if kennzeichen == "*":
            kommentare += 1
            continue
        formel = _feld(zeile, SP_FORMEL).strip()
        werte = _feld(zeile, SP_WERTE).rstrip()
        ziffern = _feld(zeile, SP_OZ).strip()
        oz = oz_aus_maske(ziffern, maske) if maske else ziffern
        if formel != FORMEL_FREI:
            ansaetze.append(
                Ansatz(
                    oz=oz,
                    ort=_feld(zeile, SP_ERLAEUTERUNG).strip() or None,
                    ansatz=werte.strip(),
                    menge=None,
                    einheit=None,
                    zeile=nr,
                    fehler=f"Formel {formel or '?'} wird nicht ausgewertet (nur Formel 91 traegt den "
                    "Rechenansatz woertlich - siehe Modul-Doku)",
                    kennzeichen=kennzeichen.strip() or None,
                )
            )
            continue
        if puffer and oz != offene_oz:
            # Ein unabgeschlossener Ansatz endet an der Ordnungszahl-Grenze.
            warnungen.append(
                f"Zeile {nr}: Rechenansatz der Ordnungszahl {offene_oz!r} ohne abschliessendes "
                "'=' - uebersprungen."
            )
            puffer = ""
        offene_oz = oz
        puffer += werte.strip()
        if "=" not in puffer:
            # Mehrzeiliger Ansatz - erst der Gleichheitssatz schliesst ihn ab.
            continue
        ausdruck = puffer.split("=", 1)[0]
        puffer = ""
        roh.append(
            (
                ausdruck,
                kennzeichen.strip(),
                oz,
                _feld(zeile, SP_ERLAEUTERUNG).strip(),
                _adresse(zeile),
                nr,
            )
        )
    if puffer:
        warnungen.append("Letzter Rechenansatz ohne abschliessendes '=' - uebersprungen.")

    gespeichert: dict[str, Decimal] = {}
    offen: list[int] = []
    ergebnisse: list[Ansatz | None] = [None] * len(roh)
    for durchlauf in (1, 2):
        for i, (ausdruck, kennzeichen, oz, erlaeuterung, adresse, nr) in enumerate(roh):
            if durchlauf == 2 and i not in offen:
                continue
            try:
                menge = _runde(ansatz_auswerten(_aufgeloest(ausdruck, gespeichert)))
                fehler = None
            except ValueError as exc:
                menge, fehler = None, str(exc)
            if menge is None and durchlauf == 1:
                offen.append(i)
            if menge is not None and adresse:
                gespeichert[adresse] = menge
            ergebnisse[i] = Ansatz(
                oz=oz,
                ort=erlaeuterung or None,
                ansatz=ausdruck,
                menge=menge,
                einheit=None,
                zeile=nr,
                fehler=fehler,
                kennzeichen=kennzeichen or None,
            )
    ansaetze.extend(a for a in ergebnisse if a is not None)
    ansaetze.sort(key=lambda a: a.zeile)

    if not ansaetze:
        raise D12Unlesbar("Keine DA11-Ansatzsaetze gefunden.")
    if kommentare:
        warnungen.append(f"{kommentare} Kommentarzeilen (Kennzeichen '*') tragen keine Menge.")
    hilfswerte = sum(1 for a in ansaetze if a.kennzeichen == KZ_HILFSWERT)
    if hilfswerte:
        warnungen.append(
            f"{hilfswerte} Hilfswerte (Kennzeichen 'H') sind nach REB 23.003 nicht Teil der "
            "Positionssumme - sie werden nur gespeichert."
        )
    warnungen.append(
        "Einheiten stehen nicht in der DA11, sondern im Leistungsverzeichnis - Luecke, kein Ratewert."
    )
    return D12Ergebnis(
        dialekt="reb_23003_da11",
        ansaetze=ansaetze,
        positionen=_positionssummen(ansaetze),
        warnungen=warnungen,
    )
