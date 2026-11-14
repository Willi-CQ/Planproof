"""REB-23.003 Formelaufmaß / Mengenermittlung (K4-Rest — Abrechnungs-Phase).

Nativer Rechenkern fuer das formelbasierte Aufmass nach REB-VB 23.003: eine
Mengenermittlung besteht aus ``AufmassLine``-Ansaetzen, die eine geometrische
Formel aus dem Formelkatalog mit benannten Parametern auswerten, mit Vorzeichen/
Faktor (Zugabe ``+`` / Abzug ``-``) und Wiederholungs-Anzahl. Jeder Ansatz wird
REB-konform auf ``round_ndigits`` (Default 3) gerundet, danach je Position summiert.

Optional gegen die LV-Soll-Menge geprueft (Mengenkontrolle): Ist vs Soll, Differenz
absolut/prozentual, Flag ``over``/``under``/``match`` innerhalb Toleranz.

Reiner, deterministischer Decimal-Rechenkern (kein DB/IO). Ein Ansatz mit fehlenden/
ungueltigen Parametern liefert ``quantity=None`` + ``error`` statt zu werfen — die
Mengenermittlung bleibt so immer anzeigbar (Builder-tauglich).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

# REB rechnet mit ausreichender Stellenzahl; PI als Decimal-Konstante.
_PI = Decimal("3.14159265358979323846")
_ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class Formula:
    """Eine geometrische Formel des Katalogs."""

    key: str
    label: str
    dimension: str  # "laenge" | "flaeche" | "volumen" | "stueck"
    params: tuple[str, ...]
    fn: Callable[[dict[str, Decimal]], Decimal]
    #: Derselbe Rechenweg als REB-Formel-91-Ausdruck, mit ``{param}``-Platzhaltern
    #: und deutschem Komma. Die Reihenfolge der Operationen ist dieselbe wie in
    #: ``fn`` — ein Test rechnet beide Wege gegeneinander, damit sie nicht
    #: auseinanderlaufen koennen.
    ansatz: str = ""


def _req(p: dict[str, Decimal], *names: str) -> list[Decimal]:
    vals = []
    for n in names:
        v = p.get(n)
        if v is None:
            raise KeyError(n)
        vals.append(v)
    return vals


# ── Formelkatalog (REB-23.003-Geometrie) ──────────────────────────────────
# Adressiert ueber sprechende Keys; ``dimension`` gibt die resultierende Groesse.
_CATALOG: dict[str, Formula] = {}


def _register(key: str, label: str, dim: str, params: tuple[str, ...], fn, ansatz: str = "") -> None:
    _CATALOG[key] = Formula(key=key, label=label, dimension=dim, params=params, fn=fn, ansatz=ansatz)


def _f_wert(p):
    (v,) = _req(p, "v")
    return v


def _f_laenge(p):
    (v,) = _req(p, "l")
    return v


def _f_rechteck(p):
    a, b = _req(p, "l", "b")
    return a * b


def _f_dreieck(p):
    g, h = _req(p, "g", "h")
    return g * h / Decimal(2)


def _f_trapez(p):
    a, c, h = _req(p, "a", "c", "h")
    return (a + c) / Decimal(2) * h


def _f_kreis(p):
    (d,) = _req(p, "d")
    return _PI * d * d / Decimal(4)


def _f_kreisring(p):
    da, di = _req(p, "d_a", "d_i")
    if di > da:
        raise ValueError("Innendurchmesser d_i > Aussendurchmesser d_a.")
    return _PI * (da * da - di * di) / Decimal(4)


def _f_quader(p):
    a, b, h = _req(p, "l", "b", "h")
    return a * b * h


def _f_zylinder(p):
    d, h = _req(p, "d", "h")
    return _PI * d * d / Decimal(4) * h


def _f_prisma(p):
    a, l_ = _req(p, "a_flaeche", "l")
    return a * l_


#: PI als REB-Zahl (deutsches Komma) — derselbe Wert wie ``_PI``, damit der
#: geschriebene Rechenansatz exakt das rechnet, was der Kern gerechnet hat.
_PI_ANSATZ = str(_PI).replace(".", ",")

_register("wert", "Direktwert", "stueck", ("v",), _f_wert, "{v}")
_register("laenge", "Laenge", "laenge", ("l",), _f_laenge, "{l}")
_register("rechteck", "Rechteck-Flaeche", "flaeche", ("l", "b"), _f_rechteck, "{l}*{b}")
_register("dreieck", "Dreieck-Flaeche", "flaeche", ("g", "h"), _f_dreieck, "{g}*{h}/2")
_register("trapez", "Trapez-Flaeche", "flaeche", ("a", "c", "h"), _f_trapez, "({a}+{c})/2*{h}")
_register("kreis", "Kreis-Flaeche", "flaeche", ("d",), _f_kreis, _PI_ANSATZ + "*{d}*{d}/4")
_register(
    "kreisring",
    "Kreisring-Flaeche",
    "flaeche",
    ("d_a", "d_i"),
    _f_kreisring,
    _PI_ANSATZ + "*({d_a}*{d_a}-{d_i}*{d_i})/4",
)
_register("quader", "Quader-Volumen", "volumen", ("l", "b", "h"), _f_quader, "{l}*{b}*{h}")
_register("zylinder", "Zylinder-Volumen", "volumen", ("d", "h"), _f_zylinder, _PI_ANSATZ + "*{d}*{d}/4*{h}")
_register("prisma", "Prisma-Volumen", "volumen", ("a_flaeche", "l"), _f_prisma, "{a_flaeche}*{l}")


def catalog() -> dict[str, dict]:
    """Liefert den Formelkatalog als serialisierbares Dict (fuer UI/Tools)."""
    return {
        f.key: {"label": f.label, "dimension": f.dimension, "params": list(f.params)}
        for f in _CATALOG.values()
    }


def formula_keys() -> frozenset[str]:
    return frozenset(_CATALOG.keys())


def reb_zahl(wert: Decimal) -> str:
    """Decimal als REB-Zahl: deutsches Komma, negative Werte geklammert.

    Die Klammer ist nicht Kosmetik: ``2*-1`` ist zwar eindeutig, aber nicht jede
    fremde REB-Anwendung liest ein Vorzeichen direkt hinter dem Operator.
    ``2*(-1)`` liest jede.
    """
    text = format(wert, "f").replace(".", ",")
    return f"({text})" if wert < _ZERO else text


def formel_ausdruck(line: AufmassLine) -> str:
    """Formuliert einen Aufmass-Ansatz als REB-Formel-91-Rechenansatz.

    Das ist der Uebersetzer fuer den Rueckweg: was der Rechenkern als Formel mit
    benannten Parametern rechnet, wird hier zu genau demselben Rechenweg in
    Schreibweise der Freien Formel - dieselbe Reihenfolge der Operationen,
    dasselbe PI, Faktor und Anzahl als Multiplikatoren angehaengt (das
    DA11-Faktorfeld bleibt bewusst leer, siehe :mod:`.da11`).

    Raises:
        ValueError: unbekannte Formel, fehlender Parameter oder eine Formel ohne
            hinterlegten Ansatz - nichts davon wird geraten.
    """
    formula = _CATALOG.get(line.formula)
    if formula is None:
        raise ValueError(f"Unbekannte Formel '{line.formula}'.")
    if not formula.ansatz:
        raise ValueError(f"Fuer die Formel '{line.formula}' ist kein Rechenansatz hinterlegt.")
    werte: dict[str, str] = {}
    for name in formula.params:
        wert = line.params.get(name)
        if wert is None:
            raise ValueError(f"Parameter '{name}' fehlt fuer Formel '{formula.key}'.")
        werte[name] = reb_zahl(wert)
    ausdruck = formula.ansatz.format(**werte)
    if line.factor != Decimal(1):
        ausdruck += "*" + reb_zahl(line.factor)
    if line.count != Decimal(1):
        ausdruck += "*" + reb_zahl(line.count)
    return ausdruck


# ── Ansatz-Modell ─────────────────────────────────────────────────────────


@dataclass(slots=True)
class AufmassLine:
    """Ein Aufmass-Ansatz (eine Zeile der Mengenermittlung)."""

    formula: str
    params: dict[str, Decimal] = field(default_factory=dict)
    factor: Decimal = Decimal(1)  # +1 Zugabe, -1 Abzug
    count: Decimal = Decimal(1)  # Anzahl gleicher Elemente
    position_no: str | None = None  # Zuordnung zur LV-Position
    comment: str | None = None


@dataclass(slots=True)
class LineResult:
    formula: str
    dimension: str | None
    position_no: str | None
    base_value: Decimal | None  # Formelwert vor Faktor/Anzahl
    quantity: Decimal | None  # base * factor * count, gerundet
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "formula": self.formula,
            "dimension": self.dimension,
            "position_no": self.position_no,
            "base_value": float(self.base_value) if self.base_value is not None else None,
            "quantity": float(self.quantity) if self.quantity is not None else None,
            "error": self.error,
        }


@dataclass(slots=True)
class PositionTotal:
    position_no: str | None
    quantity: Decimal
    # Mengenkontrolle vs Soll (optional befuellt).
    soll: Decimal | None = None
    diff: Decimal | None = None
    diff_pct: Decimal | None = None
    status: str | None = None  # "over" | "under" | "match"

    def to_dict(self) -> dict:
        return {
            "position_no": self.position_no,
            "quantity": float(self.quantity),
            "soll": float(self.soll) if self.soll is not None else None,
            "diff": float(self.diff) if self.diff is not None else None,
            "diff_pct": float(self.diff_pct) if self.diff_pct is not None else None,
            "status": self.status,
        }


@dataclass(slots=True)
class AufmassResult:
    lines: list[LineResult]
    positions: list[PositionTotal]
    total: Decimal
    n_lines: int
    n_errors: int

    def to_dict(self) -> dict:
        return {
            "total": float(self.total),
            "n_lines": self.n_lines,
            "n_errors": self.n_errors,
            "lines": [line.to_dict() for line in self.lines],
            "positions": [p.to_dict() for p in self.positions],
        }


def _round(v: Decimal, ndigits: int) -> Decimal:
    q = Decimal(1).scaleb(-ndigits)  # 10^-ndigits
    return v.quantize(q, rounding=ROUND_HALF_UP)


def _compute_line(line: AufmassLine, ndigits: int) -> LineResult:
    formula = _CATALOG.get(line.formula)
    if formula is None:
        return LineResult(
            formula=line.formula,
            dimension=None,
            position_no=line.position_no,
            base_value=None,
            quantity=None,
            error=f"Unbekannte Formel '{line.formula}'.",
        )
    try:
        # Geometrische Laengen/Werte muessen nicht-negativ sein (ausser reiner Wert,
        # der auch negativ als Abzug erlaubt ist — dafuer aber lieber factor=-1).
        for name, val in line.params.items():
            if not isinstance(val, Decimal):
                raise ValueError(f"Parameter '{name}' ist kein Decimal.")
        base = formula.fn(line.params)
        qty = _round(base * line.factor * line.count, ndigits)
        return LineResult(
            formula=formula.key,
            dimension=formula.dimension,
            position_no=line.position_no,
            base_value=_round(base, ndigits),
            quantity=qty,
        )
    except KeyError as exc:
        return LineResult(
            formula=formula.key,
            dimension=formula.dimension,
            position_no=line.position_no,
            base_value=None,
            quantity=None,
            error=f"Parameter {exc} fehlt fuer Formel '{formula.key}'.",
        )
    except (ValueError, InvalidOperation, ZeroDivisionError) as exc:
        return LineResult(
            formula=formula.key,
            dimension=formula.dimension,
            position_no=line.position_no,
            base_value=None,
            quantity=None,
            error=str(exc),
        )


def compute_aufmass(
    lines: list[AufmassLine],
    *,
    soll_by_position: dict[str, Decimal] | None = None,
    round_ndigits: int = 3,
    tolerance_pct: Decimal = Decimal("0.1"),
) -> AufmassResult:
    """Berechnet eine Mengenermittlung (REB-Formelaufmaß).

    Args:
        lines: Ansaetze der Mengenermittlung.
        soll_by_position: optionale LV-Soll-Mengen je ``position_no`` fuer die
            Mengenkontrolle (Ist vs Soll).
        round_ndigits: Rundungsstellen je Ansatz + Summe (REB-Konvention, Default 3).
        tolerance_pct: Toleranz in %, innerhalb derer Ist==Soll als ``match`` gilt.

    Returns:
        AufmassResult mit Zeilen-Ergebnissen, Positions-Summen (inkl. Mengenkontrolle)
        und Gesamtsumme.
    """
    line_results = [_compute_line(line, round_ndigits) for line in lines]

    # Positions-Summen (nach position_no gruppiert, Reihenfolge des Erst-Auftritts).
    order: list[str | None] = []
    sums: dict[str | None, Decimal] = {}
    for lr in line_results:
        if lr.quantity is None:
            continue
        key = lr.position_no
        if key not in sums:
            sums[key] = _ZERO
            order.append(key)
        sums[key] += lr.quantity

    positions: list[PositionTotal] = []
    for key in order:
        qty = _round(sums[key], round_ndigits)
        pt = PositionTotal(position_no=key, quantity=qty)
        if soll_by_position and key is not None and key in soll_by_position:
            soll = soll_by_position[key]
            pt.soll = soll
            pt.diff = _round(qty - soll, round_ndigits)
            if soll != _ZERO:
                pt.diff_pct = _round((qty - soll) / soll * Decimal(100), 2)
                if abs(pt.diff_pct) <= tolerance_pct:
                    pt.status = "match"
                elif pt.diff_pct > _ZERO:
                    pt.status = "over"
                else:
                    pt.status = "under"
            else:
                pt.status = "over" if qty > _ZERO else "match"
        positions.append(pt)

    total = _round(sum((p.quantity for p in positions), _ZERO), round_ndigits)
    n_errors = sum(1 for lr in line_results if lr.error is not None)
    return AufmassResult(
        lines=line_results,
        positions=positions,
        total=total,
        n_lines=len(line_results),
        n_errors=n_errors,
    )
