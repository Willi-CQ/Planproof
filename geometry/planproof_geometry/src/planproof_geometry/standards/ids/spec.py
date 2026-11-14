"""IDS-Datenmodell: Facets, Wert-Constraints, Spezifikationen (K5-Rest — openBIM).

buildingSMART IDS (Information Delivery Specification) beschreibt pruefbare
Informationsanforderungen an ein IFC-Modell. Eine ``Specification`` hat eine
``applicability`` (auf welche Elemente sie zutrifft) und ``requirements`` (was fuer
diese Elemente gelten muss) — beide als Liste von ``Facet``.

Facet-Typen: entity | attribute | property | classification | material.
Wert-Constraints: simpleValue (exakt), enumeration (xs:enumeration), pattern
(xs:pattern-Regex). Kardinalitaet der Requirements: required | optional | prohibited.

Reiner Datentyp- + Match-Layer (kein XML, kein IO) — der Parser (``parser.py``) baut
diese Objekte, der Validator (``validator.py``) wertet sie gegen Element-Dicts aus.

Element-Dict-Schema (Producer z.B. ``ifc_index.parse_ifc_for_ids``):
    {
      "ifc_class": "IFCWALL",              # UPPER, Pflicht
      "predefined_type": "SOLIDWALL",      # optional
      "attributes": {"Name": "W-01", ...}, # IFC-Attribute
      "properties": {"Pset_WallCommon": {"FireRating": "F90", "IsExternal": true}},
      "materials": ["Beton", "Daemmung"],
      "classifications": [{"system": "Uniclass", "value": "EF_25_10"}],
    }
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ── ReDoS-Schutz fuer untrusted <xs:pattern>-Regex (Audit H5) ───────────────
# Ein IDS-Dokument ist voll attacker-kontrolliert; ``xs:pattern`` landete frueher
# direkt in ``re.fullmatch`` (kein Timeout). Ein Pattern wie ``(a+)+$`` gegen einen
# langen Nicht-Match-String pinnt einen CPU-Kern minutenlang (catastrophic
# backtracking) -> ueber die vielen Elemente ein API-weiter Threadpool-DoS.
# Primaerschutz: die ``regex``-Engine mit ``timeout=`` (unterbricht Backtracking).
# Fallback ohne die Dependency: klassische ReDoS-Formen fail-closed ablehnen +
# Pattern-/Input-Laenge cappen.
_MAX_PATTERN_LEN = 1000
_MAX_MATCH_INPUT = 8192
_PATTERN_MATCH_TIMEOUT_S = 0.05

try:  # pragma: no cover - Verfuegbarkeit abhaengig vom Deploy-Image
    import regex as _regex_engine

    _HAS_TIMEOUT_ENGINE = True
except Exception:  # noqa: BLE001 - jede Import-Stoerung -> Fallback
    _regex_engine = None
    _HAS_TIMEOUT_ENGINE = False

# Heuristik fuer klassische "evil regex": ein quantifizierter Gruppen-Ausdruck,
# dessen Inhalt selbst einen unbegrenzten Quantor traegt (z.B. (a+)+, (a*)*, (.+)+).
_EVIL_REGEX_SHAPE = re.compile(r"\([^)]*[+*][^)]*\)\s*[+*{]")
_pattern_cache: dict[str, Any] = {}


def _safe_fullmatch(pattern: str, s: str) -> bool:
    """re.fullmatch-Ersatz mit ReDoS-Schutz fuer untrusted Patterns."""
    if len(pattern) > _MAX_PATTERN_LEN:
        return False
    if len(s) > _MAX_MATCH_INPUT:
        s = s[:_MAX_MATCH_INPUT]
    if _HAS_TIMEOUT_ENGINE:
        try:
            return _regex_engine.fullmatch(pattern, s, timeout=_PATTERN_MATCH_TIMEOUT_S) is not None
        except (_regex_engine.error, TimeoutError, ValueError):
            return False
    # Kein Timeout-Engine: klassische ReDoS-Form fail-closed ablehnen.
    if _EVIL_REGEX_SHAPE.search(pattern):
        return False
    compiled = _pattern_cache.get(pattern)
    if compiled is None:
        try:
            compiled = re.compile(pattern)
        except re.error:
            return False
        if len(_pattern_cache) < 512:
            _pattern_cache[pattern] = compiled
    try:
        return compiled.fullmatch(s) is not None
    except re.error:
        return False


def _norm(v: Any) -> str:
    """Normalisiert einen Wert fuer den Vergleich (String, getrimmt)."""
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()


@dataclass(frozen=True, slots=True)
class ValueConstraint:
    """Einschraenkung eines Wertes: simpleValue, Enumeration oder Regex-Pattern.

    Ist alles ``None``, ist der Constraint reine Praesenz-Pruefung (Wert egal).
    """

    simple: str | None = None
    enumeration: tuple[str, ...] | None = None
    pattern: str | None = None

    def is_presence_only(self) -> bool:
        return self.simple is None and self.enumeration is None and self.pattern is None

    def matches(self, actual: Any) -> bool:
        if actual is None:
            return False
        s = _norm(actual)
        if self.simple is not None:
            return s.casefold() == _norm(self.simple).casefold()
        if self.enumeration is not None:
            return any(s.casefold() == _norm(e).casefold() for e in self.enumeration)
        if self.pattern is not None:
            return _safe_fullmatch(self.pattern, s)
        return True  # Praesenz genuegt


# Facet-Kinds.
ENTITY = "entity"
ATTRIBUTE = "attribute"
PROPERTY = "property"
CLASSIFICATION = "classification"
MATERIAL = "material"

# Kardinalitaet.
REQUIRED = "required"
OPTIONAL = "optional"
PROHIBITED = "prohibited"


@dataclass(frozen=True, slots=True)
class Facet:
    """Eine IDS-Facet (in applicability oder requirements)."""

    kind: str
    # entity
    entity_name: ValueConstraint | None = None
    predefined_type: ValueConstraint | None = None
    # attribute
    attr_name: ValueConstraint | None = None
    # property
    pset: ValueConstraint | None = None
    base_name: ValueConstraint | None = None
    # classification
    system: ValueConstraint | None = None
    # gemeinsamer Wert (attribute/property/classification/material)
    value: ValueConstraint | None = None
    # nur fuer requirements relevant
    cardinality: str = REQUIRED

    # ── Praesenz + Match ──────────────────────────────────────────────────

    def _attr_lookup(self, element: dict, name_c: ValueConstraint) -> Any:
        attrs = element.get("attributes") or {}
        # simpleValue -> direkter Key; sonst erste passende.
        if name_c.simple is not None:
            for k, v in attrs.items():
                if _norm(k).casefold() == _norm(name_c.simple).casefold():
                    return v
            return None
        for k, v in attrs.items():
            if name_c.matches(k):
                return v
        return None

    def _prop_lookup(self, element: dict) -> tuple[bool, bool, Any]:
        """Liefert (praesent, wert_stimmt, wert) fuer die property-Facet.

        ``praesent`` = Pset+Property existiert (Wert egal); ``wert_stimmt`` =
        zusaetzlich passt der Wert-Constraint. Beide werden getrennt gebraucht
        (required/optional/prohibited).
        """
        props = element.get("properties") or {}
        present = False
        result: Any = None
        for pset_name, entries in props.items():
            if self.pset is not None and not self.pset.matches(pset_name):
                continue
            if not isinstance(entries, dict):
                continue
            for prop_name, val in entries.items():
                if self.base_name is not None and not self.base_name.matches(prop_name):
                    continue
                present = True
                result = val
                if self.value is None or self.value.matches(val):
                    return True, True, val
        return present, False, result

    def present(self, element: dict) -> bool:
        """Container-Praesenz ohne Wert-Pruefung (fuer optional/prohibited-Logik)."""
        if self.kind == ENTITY:
            return True  # jedes Element hat eine Klasse
        if self.kind == ATTRIBUTE:
            if self.attr_name is None:
                return bool(element.get("attributes"))
            return self._attr_lookup(element, self.attr_name) is not None
        if self.kind == PROPERTY:
            present, _, _ = self._prop_lookup(element)
            return present
        if self.kind == CLASSIFICATION:
            for c in element.get("classifications") or []:
                if self.system is None or self.system.matches(c.get("system")):
                    return True
            return False
        if self.kind == MATERIAL:
            return bool(element.get("materials"))
        return False

    def satisfied(self, element: dict) -> bool:
        """Wert-bewusste Erfuellung (Praesenz UND Wert stimmen)."""
        if self.kind == ENTITY:
            cls = element.get("ifc_class")
            if self.entity_name is not None and not self.entity_name.matches(cls):
                return False
            if self.predefined_type is not None and not self.predefined_type.matches(
                element.get("predefined_type")
            ):
                return False
            return True
        if self.kind == ATTRIBUTE:
            if self.attr_name is None:
                return False
            actual = self._attr_lookup(element, self.attr_name)
            if actual is None:
                return False
            return self.value is None or self.value.matches(actual)
        if self.kind == PROPERTY:
            _, matched, _ = self._prop_lookup(element)
            return matched
        if self.kind == CLASSIFICATION:
            for c in element.get("classifications") or []:
                if self.system is not None and not self.system.matches(c.get("system")):
                    continue
                if self.value is None or self.value.matches(c.get("value")):
                    return True
            return False
        if self.kind == MATERIAL:
            mats = element.get("materials") or []
            if self.value is None:
                return bool(mats)
            return any(self.value.matches(m) for m in mats)
        return False

    def check_requirement(self, element: dict) -> tuple[bool, str | None]:
        """Prueft die Facet als Requirement unter Beruecksichtigung der Kardinalitaet.

        Returns (ok, reason) — reason nur bei Verstoss.
        """
        ok_value = self.satisfied(element)
        if self.cardinality == PROHIBITED:
            if ok_value:
                return False, f"{self.describe()} ist unzulaessig (prohibited), aber vorhanden."
            return True, None
        if self.cardinality == OPTIONAL:
            if self.present(element) and not ok_value:
                return False, f"{self.describe()} vorhanden, aber Wert stimmt nicht."
            return True, None
        # required
        if not ok_value:
            if not self.present(element):
                return False, f"{self.describe()} fehlt."
            return False, f"{self.describe()} vorhanden, aber Wert stimmt nicht."
        return True, None

    def describe(self) -> str:
        def c(vc: ValueConstraint | None) -> str:
            if vc is None:
                return "*"
            if vc.simple is not None:
                return vc.simple
            if vc.enumeration is not None:
                return "|".join(vc.enumeration)
            if vc.pattern is not None:
                return f"/{vc.pattern}/"
            return "*"

        if self.kind == ENTITY:
            s = f"entity {c(self.entity_name)}"
            if self.predefined_type is not None:
                s += f".{c(self.predefined_type)}"
            return s
        if self.kind == ATTRIBUTE:
            return f"attribute {c(self.attr_name)}={c(self.value)}"
        if self.kind == PROPERTY:
            return f"property {c(self.pset)}.{c(self.base_name)}={c(self.value)}"
        if self.kind == CLASSIFICATION:
            return f"classification {c(self.system)}:{c(self.value)}"
        if self.kind == MATERIAL:
            return f"material {c(self.value)}"
        return self.kind


@dataclass(slots=True)
class Specification:
    """Eine IDS-Spezifikation: applicability + requirements."""

    name: str
    applicability: list[Facet] = field(default_factory=list)
    requirements: list[Facet] = field(default_factory=list)
    ifc_version: str | None = None
    description: str | None = None

    def applies_to(self, element: dict) -> bool:
        """Element faellt unter diese Spec, wenn ALLE applicability-Facets erfuellt sind."""
        return all(f.satisfied(element) for f in self.applicability)


@dataclass(slots=True)
class IdsDocument:
    """Ein geparstes IDS-Dokument (Titel + Spezifikationen)."""

    title: str
    specifications: list[Specification] = field(default_factory=list)
