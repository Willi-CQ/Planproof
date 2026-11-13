"""IDS-XML-Parser (buildingSMART IDS 1.0) — K5-Rest openBIM.

Liest ein ``.ids``-XML (untrusted -> defusedxml, XXE-sicher wie der GAEB-Parser) in
``IdsDocument`` / ``Specification`` / ``Facet``. Namespace-tolerant ueber Local-Name-
Matching (IDS-Default-NS ``http://standards.buildingsmart.org/IDS`` + ``xs:``-Restriction).

Unterstuetzte Facets: entity, attribute, property, classification, material.
Wert-Constraints: ``<simpleValue>`` sowie ``<xs:restriction>`` mit ``<xs:enumeration>``
und ``<xs:pattern>``. Kardinalitaet: ``cardinality``-Attribut bzw. minOccurs/maxOccurs.
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

from defusedxml.ElementTree import fromstring as _safe_fromstring

from .spec import (
    ATTRIBUTE,
    CLASSIFICATION,
    ENTITY,
    MATERIAL,
    OPTIONAL,
    PROHIBITED,
    PROPERTY,
    REQUIRED,
    Facet,
    IdsDocument,
    Specification,
    ValueConstraint,
)


class IdsParseError(ValueError):
    """Strukturelles Problem im IDS-XML."""


def _local(tag: str) -> str:
    return tag.split("}")[-1]


def _find(el: ET.Element | None, name: str) -> ET.Element | None:
    if el is None:
        return None
    for child in el:
        if _local(child.tag) == name:
            return child
    return None


def _findall(el: ET.Element | None, name: str) -> list[ET.Element]:
    if el is None:
        return []
    return [child for child in el if _local(child.tag) == name]


def _text(el: ET.Element | None) -> str | None:
    if el is None or el.text is None:
        return None
    t = el.text.strip()
    return t or None


def _parse_value(container: ET.Element | None) -> ValueConstraint | None:
    """Parst einen Wert-Container (z.B. <name>, <value>) in einen ValueConstraint."""
    if container is None:
        return None
    simple = _find(container, "simpleValue")
    if simple is not None:
        return ValueConstraint(simple=_text(simple))
    restriction = _find(container, "restriction")
    if restriction is not None:
        enums = [e.get("value") for e in _findall(restriction, "enumeration") if e.get("value") is not None]
        pattern_el = _find(restriction, "pattern")
        pattern = pattern_el.get("value") if pattern_el is not None else None
        if enums:
            return ValueConstraint(enumeration=tuple(enums))
        if pattern:
            return ValueConstraint(pattern=pattern)
        return ValueConstraint()  # Restriction ohne unterstuetztes Kind -> Praesenz
    # Container ohne simpleValue/restriction: Praesenz genuegt.
    return ValueConstraint()


def _cardinality(facet_el: ET.Element) -> str:
    card = facet_el.get("cardinality")
    if card in (REQUIRED, OPTIONAL, PROHIBITED):
        return card
    min_occurs = facet_el.get("minOccurs")
    max_occurs = facet_el.get("maxOccurs")
    if max_occurs == "0":
        return PROHIBITED
    if min_occurs == "0":
        return OPTIONAL
    return REQUIRED


def _parse_facet(facet_el: ET.Element) -> Facet | None:
    kind = _local(facet_el.tag)
    card = _cardinality(facet_el)
    if kind == ENTITY:
        return Facet(
            kind=ENTITY,
            entity_name=_parse_value(_find(facet_el, "name")),
            predefined_type=_parse_value(_find(facet_el, "predefinedType"))
            if _find(facet_el, "predefinedType") is not None
            else None,
            cardinality=card,
        )
    if kind == ATTRIBUTE:
        return Facet(
            kind=ATTRIBUTE,
            attr_name=_parse_value(_find(facet_el, "name")),
            value=_parse_value(_find(facet_el, "value")) if _find(facet_el, "value") is not None else None,
            cardinality=card,
        )
    if kind == PROPERTY:
        base = _find(facet_el, "baseName")
        if base is None:
            base = _find(facet_el, "name")  # aeltere IDS-Varianten
        return Facet(
            kind=PROPERTY,
            pset=_parse_value(_find(facet_el, "propertySet")),
            base_name=_parse_value(base),
            value=_parse_value(_find(facet_el, "value")) if _find(facet_el, "value") is not None else None,
            cardinality=card,
        )
    if kind == CLASSIFICATION:
        return Facet(
            kind=CLASSIFICATION,
            system=_parse_value(_find(facet_el, "system")) if _find(facet_el, "system") is not None else None,
            value=_parse_value(_find(facet_el, "value")) if _find(facet_el, "value") is not None else None,
            cardinality=card,
        )
    if kind == MATERIAL:
        return Facet(
            kind=MATERIAL,
            value=_parse_value(_find(facet_el, "value")) if _find(facet_el, "value") is not None else None,
            cardinality=card,
        )
    return None


def _parse_facets(container: ET.Element | None) -> list[Facet]:
    if container is None:
        return []
    out: list[Facet] = []
    for child in container:
        facet = _parse_facet(child)
        if facet is not None:
            out.append(facet)
    return out


def parse_ids(xml: bytes | str) -> IdsDocument:
    """Parst ein IDS-XML in ein ``IdsDocument``.

    Raises:
        IdsParseError: bei ungueltigem/unsicherem XML oder fehlendem <ids>-Root.
    """
    data = xml.encode("utf-8") if isinstance(xml, str) else xml
    try:
        root = _safe_fromstring(data)
    except ET.ParseError as exc:
        raise IdsParseError(f"Ungueltiges XML: {exc}") from exc
    except Exception as exc:  # defusedxml EntitiesForbidden o.ae.
        raise IdsParseError(f"Ungueltiges/unsicheres XML: {exc}") from exc

    if _local(root.tag) != "ids":
        raise IdsParseError(f"Erwartet <ids>-Root, gefunden <{_local(root.tag)}>.")

    info = _find(root, "info")
    title = _text(_find(info, "title")) or "IDS"

    specs_container = _find(root, "specifications")
    specifications: list[Specification] = []
    for spec_el in _findall(specs_container, "specification"):
        spec = Specification(
            name=spec_el.get("name") or "(unbenannt)",
            ifc_version=spec_el.get("ifcVersion"),
            description=_text(_find(spec_el, "description")),
            applicability=_parse_facets(_find(spec_el, "applicability")),
            requirements=_parse_facets(_find(spec_el, "requirements")),
        )
        specifications.append(spec)

    return IdsDocument(title=title, specifications=specifications)
