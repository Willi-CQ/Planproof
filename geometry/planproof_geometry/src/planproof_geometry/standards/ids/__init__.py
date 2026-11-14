"""buildingSMART IDS (Information Delivery Specification) — K5-Rest openBIM.

Nativer IDS-Parser + Validator: prueft ein IFC-Modell (als Element-Dicts) gegen
pruefbare Informationsanforderungen (Pset-Pflichtfelder, Klassifikationen, Materialien,
Attribute) — die openBIM-Pruefkante analog Solibri/IDS-Audit.

Public-API:
- parse_ids(xml) -> IdsDocument
- validate_ids(elements, document) -> IdsReport
- validate(elements, xml) -> IdsReport  (Einschritt-Convenience)
"""

from __future__ import annotations

from .parser import IdsParseError, parse_ids
from .spec import (
    Facet,
    IdsDocument,
    Specification,
    ValueConstraint,
)
from .validator import (
    FailureItem,
    IdsReport,
    SpecResult,
    validate_ids,
)


def validate(elements: list[dict], xml: bytes | str, *, max_failures_per_spec: int = 200) -> IdsReport:
    """Convenience: parst IDS-XML und validiert die Elemente in einem Schritt."""
    return validate_ids(elements, parse_ids(xml), max_failures_per_spec=max_failures_per_spec)


__all__ = [
    "Facet",
    "FailureItem",
    "IdsDocument",
    "IdsParseError",
    "IdsReport",
    "SpecResult",
    "Specification",
    "ValueConstraint",
    "parse_ids",
    "validate",
    "validate_ids",
]
