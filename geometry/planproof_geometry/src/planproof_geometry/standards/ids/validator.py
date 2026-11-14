"""IDS-Validator: prueft Element-Dicts gegen ein ``IdsDocument`` (K5-Rest openBIM).

Fuer jede Spezifikation werden die zutreffenden Elemente (applicability) ermittelt und
gegen die Requirements geprueft. Ergebnis: pro Spec pass/fail-Zaehlung + Fehlerliste,
plus Gesamt-Verdikt. Reiner Rechenkern (kein XML, kein IO).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .spec import IdsDocument, Specification

# Audit H5: hartes Wall-Clock-Gesamtbudget. Der Per-Match-Timeout in ``spec._safe_
# fullmatch`` bremst EINEN Match; ueber elements(<=50k) x requirements koennte die
# Summe dennoch eskalieren. Dieses Budget deckelt die Gesamtlaufzeit fail-safe.
_DEFAULT_TIME_BUDGET_S = 10.0


def _element_ref(element: dict, index: int) -> str:
    return str(
        element.get("global_id")
        or element.get("name")
        or (element.get("attributes") or {}).get("Name")
        or f"#{index} {element.get('ifc_class', '?')}"
    )


@dataclass(slots=True)
class FailureItem:
    element_ref: str
    ifc_class: str | None
    facet: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "element_ref": self.element_ref,
            "ifc_class": self.ifc_class,
            "facet": self.facet,
            "reason": self.reason,
        }


@dataclass(slots=True)
class SpecResult:
    name: str
    description: str | None
    n_applicable: int
    n_passed: int
    n_failed: int
    status: str  # "pass" | "fail" | "not_applicable"
    failures: list[FailureItem] = field(default_factory=list)
    failures_truncated: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "n_applicable": self.n_applicable,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "status": self.status,
            "failures": [f.to_dict() for f in self.failures],
            "failures_truncated": self.failures_truncated,
        }


@dataclass(slots=True)
class IdsReport:
    title: str
    n_specs: int
    n_elements: int
    total_applicable: int
    total_passed: int
    total_failed: int
    is_pass: bool
    specs: list[SpecResult] = field(default_factory=list)
    budget_exceeded: bool = False

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "n_specs": self.n_specs,
            "n_elements": self.n_elements,
            "total_applicable": self.total_applicable,
            "total_passed": self.total_passed,
            "total_failed": self.total_failed,
            "is_pass": self.is_pass,
            "budget_exceeded": self.budget_exceeded,
            "specs": [s.to_dict() for s in self.specs],
        }


def _validate_spec(
    spec: Specification,
    elements: list[dict],
    max_failures: int,
    deadline: float | None = None,
) -> tuple[SpecResult, bool]:
    n_applicable = 0
    n_passed = 0
    n_failed = 0
    failures: list[FailureItem] = []
    truncated = False
    budget_hit = False

    for i, el in enumerate(elements):
        # Audit H5: Gesamtbudget zwischen den Elementen pruefen -> selbst bei sehr
        # vielen Elementen/Requirements bleibt die Laufzeit gedeckelt.
        if deadline is not None and time.monotonic() > deadline:
            budget_hit = True
            break
        if not spec.applies_to(el):
            continue
        n_applicable += 1
        element_ok = True
        for req in spec.requirements:
            ok, reason = req.check_requirement(el)
            if not ok:
                element_ok = False
                if len(failures) < max_failures:
                    failures.append(
                        FailureItem(
                            element_ref=_element_ref(el, i),
                            ifc_class=el.get("ifc_class"),
                            facet=req.describe(),
                            reason=reason or "Verstoss",
                        )
                    )
                else:
                    truncated = True
        if element_ok:
            n_passed += 1
        else:
            n_failed += 1

    if n_applicable == 0:
        status = "not_applicable"
    elif n_failed == 0:
        status = "pass"
    else:
        status = "fail"

    return (
        SpecResult(
            name=spec.name,
            description=spec.description,
            n_applicable=n_applicable,
            n_passed=n_passed,
            n_failed=n_failed,
            status=status,
            failures=failures,
            failures_truncated=truncated,
        ),
        budget_hit,
    )


def validate_ids(
    elements: list[dict],
    document: IdsDocument,
    *,
    max_failures_per_spec: int = 200,
    time_budget_s: float = _DEFAULT_TIME_BUDGET_S,
) -> IdsReport:
    """Validiert Element-Dicts gegen ein IDS-Dokument.

    Args:
        elements: Liste von Element-Dicts (Schema siehe ``spec``-Modul-Docstring).
        document: geparstes ``IdsDocument``.
        max_failures_per_spec: Obergrenze der protokollierten Einzelfehler je Spec.
        time_budget_s: hartes Wall-Clock-Gesamtbudget (Audit H5 ReDoS-Schutz).

    Returns:
        IdsReport — pro Spec pass/fail + Fehler, plus Gesamt-Verdikt (``is_pass``).
        ``budget_exceeded=True`` signalisiert einen vorzeitigen Abbruch.
    """
    deadline = time.monotonic() + time_budget_s if time_budget_s and time_budget_s > 0 else None
    spec_results: list[SpecResult] = []
    budget_exceeded = False
    for s in document.specifications:
        result, hit = _validate_spec(s, elements, max_failures_per_spec, deadline)
        spec_results.append(result)
        if hit:
            budget_exceeded = True
            break
    total_applicable = sum(s.n_applicable for s in spec_results)
    total_passed = sum(s.n_passed for s in spec_results)
    total_failed = sum(s.n_failed for s in spec_results)
    is_pass = total_failed == 0 and not budget_exceeded
    return IdsReport(
        title=document.title,
        n_specs=len(spec_results),
        n_elements=len(elements),
        total_applicable=total_applicable,
        total_passed=total_passed,
        total_failed=total_failed,
        is_pass=is_pass,
        specs=spec_results,
        budget_exceeded=budget_exceeded,
    )
