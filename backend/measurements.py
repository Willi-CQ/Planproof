"""Cited area arithmetic and calibrated raster takeoff. No inferred units or scale.

The host validates arithmetic, geometry and source coordinates, not the model's
reading of text or choice of boundary. Raster takeoffs are always estimates.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from shapely.geometry import LineString, Polygon, box

from planproof_feedback.images import validate_citations
from planproof_feedback.regions import canonical_citations


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


Finite = Annotated[float, Field(allow_inf_nan=False)]
Point = Annotated[list[Finite], Field(min_length=2, max_length=2)]
Box = Annotated[list[Finite], Field(min_length=4, max_length=4)]
Ring = Annotated[list[Point], Field(min_length=3, max_length=64)]
AreaUnit = Literal["ft2", "m2", "in2", "cm2", "mm2"]


class Source(Input):
    image_id: str = Field(min_length=1, max_length=80)
    bbox: Box


class Reading(Input):
    text: str = Field(min_length=1, max_length=100)
    citation: Source
    # Used ONLY when the drawing explicitly establishes units for bare numbers.
    unit_context: str | None = Field(default=None, max_length=20)
    unit_citation: Source | None = None
    number_format: Literal["decimal_point", "decimal_comma"] | None = None


class Rectangle(Input):
    label: str = Field(min_length=1, max_length=100)
    length: Reading
    width: Reading
    operation: Literal["add", "subtract"] = "add"


class AreaRequest(Input):
    target: str = Field(min_length=1, max_length=200)
    method: Literal["printed", "rectangles"]
    output_unit: AreaUnit
    boundary_basis: str = Field(min_length=1, max_length=600)
    region_citation: Source
    approximate: bool
    printed_area: Reading | None = None
    rectangles: list[Rectangle] = Field(default_factory=list, max_length=16)


class Anchor(Input):
    dimension: Reading
    endpoints: Annotated[list[Point], Field(min_length=2, max_length=2)]


class PolygonRequest(Input):
    target: str = Field(min_length=1, max_length=200)
    output_unit: AreaUnit
    boundary_basis: str = Field(min_length=1, max_length=600)
    region_citation: Source
    view_bbox: Box
    polygon: Ring
    holes: list[Ring] = Field(default_factory=list, max_length=16)
    anchors: list[Anchor] = Field(min_length=2, max_length=6)


FACTORS = {
    "mm": Fraction(1, 1000),
    "cm": Fraction(1, 100),
    "m": Fraction(1),
    "in": Fraction(127, 5000),
    "ft": Fraction(381, 1250),
}
ALIASES = {
    "sf": "ft2",
    "sqft": "ft2",
    "squarefeet": "ft2",
    "squarefoot": "ft2",
    "sqm": "m2",
    "squaremeters": "m2",
    "squaremetres": "m2",
    "sqin": "in2",
    "sqcm": "cm2",
    "sqmm": "mm2",
    "feet": "ft",
    "foot": "ft",
    "inches": "in",
    "inch": "in",
    "meters": "m",
    "metres": "m",
    "meter": "m",
    "metre": "m",
}


def normalize_unit(unit):
    text = re.sub(r"[\s.\^]", "", unit.lower().replace("²", "2"))
    return ALIASES.get(text, text)


def _number(text, number_format=None):
    if re.fullmatch(r"\d+\s+\d+/\d+|\d+/\d+", text):
        parts = text.split()
        numerator, denominator = map(int, parts[-1].split("/"))
        if not denominator or numerator >= denominator:
            raise ValueError("Use a proper fraction with a nonzero denominator")
        return (
            Fraction(int(parts[0])) + Fraction(numerator, denominator)
            if len(parts) == 2
            else Fraction(numerator, denominator)
        )
    if number_format == "decimal_comma":
        if not re.fullmatch(r"(?:\d+|\d{1,3}(?:\.\d{3})+)(?:,\d+)?", text):
            raise ValueError("Invalid decimal-comma number")
        return Fraction(text.replace(".", "").replace(",", "."))
    if "," in text and number_format != "decimal_point":
        raise ValueError(
            "Comma separators require explicit number_format from the drawing context"
        )
    if not re.fullmatch(r"(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?", text):
        raise ValueError("Unreadable or ambiguous dimension number")
    return Fraction(text.replace(",", ""))


def parse_quantity(text, unit_context=None, number_format=None):
    """Return exact SI value, dimensional power and written unit (no eval)."""
    text = (
        text.strip()
        .lower()
        .translate(
            str.maketrans(
                {"′": "'", "’": "'", "″": '"', "“": '"', "”": '"', "–": "-", "−": "-"}
            )
        )
    )
    for glyph, replacement in {
        "½": " 1/2",
        "¼": " 1/4",
        "¾": " 3/4",
        "⅛": " 1/8",
        "⅜": " 3/8",
        "⅝": " 5/8",
        "⅞": " 7/8",
    }.items():
        text = text.replace(glyph, replacement)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(?<=\d)-(?=\d+/\d+)", " ", text)
    number = r"(?:\d+\s+\d+/\d+|\d+/\d+|\d+(?:[.,]\d+)?)"
    imperial = re.fullmatch(
        rf"(\d+(?:[.,]\d+)?)\s*(?:'|ft|feet|foot)\s*(?:-?\s*({number})\s*(?:\"|in|inches|inch))?",
        text,
    )
    if imperial:
        feet, inches = imperial.groups()
        inch_value = (
            _number(inches, number_format) if inches is not None else Fraction(0)
        )
        feet_value = _number(feet, number_format)
        if inch_value >= 12 or inches is not None and feet_value.denominator != 1:
            raise ValueError("Invalid feet/inches notation")
        value, power, unit = (
            feet_value * FACTORS["ft"] + inch_value * FACTORS["in"],
            1,
            "ft",
        )
    else:
        match = re.fullmatch(r"([\d,. /]+)\s*([a-z²2^ .]+|\")?", text)
        if not match:
            raise ValueError(
                "Expected an explicit length or area, not a formula or scale ratio"
            )
        raw, suffix = match.groups()
        unit = (
            normalize_unit(suffix.strip())
            if suffix
            else normalize_unit(unit_context or "")
        )
        if unit == '"':
            unit = "in"
        power = 2 if unit.endswith("2") else 1
        base = unit[:-1] if power == 2 else unit
        if base not in FACTORS:
            raise ValueError("An explicit supported unit is required")
        value = _number(raw.strip(), number_format) * FACTORS[base] ** power
    if value <= 0:
        raise ValueError("Measurement must be positive")
    if unit_context and normalize_unit(unit_context) != unit:
        raise ValueError("Unit context conflicts with the written dimension")
    return value, power, unit


def _reading(reading, power, images, citations):
    if (reading.unit_context is None) != (reading.unit_citation is None):
        raise ValueError("Unit context requires its own source citation")
    evidence = [reading.citation.model_dump()]
    if reading.unit_citation:
        evidence.append(reading.unit_citation.model_dump())
    validate_citations(evidence, images)
    value, actual_power, unit = parse_quantity(
        reading.text, reading.unit_context, reading.number_format
    )
    if actual_power != power:
        raise ValueError("Length and area units cannot be interchanged")
    citations.extend(evidence)
    return value, {
        "text": reading.text,
        "unit": unit,
        "number_format": reading.number_format,
        "si_value": float(value),
        "citations": evidence,
    }


def _base(request, images):
    citations = [request.region_citation.model_dump()]
    validate_citations(citations, images)
    return citations


def _result(
    request, area, citations, images, *, method, formula, approximate, **details
):
    value = float(area / FACTORS[request.output_unit[:-1]] ** 2)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Calculated area must be finite and positive")
    # Three significant figures keep raster estimates concise without rounding a
    # small room to zero or introducing large, unit-dependent numeric error.
    # This is reporting resolution, not a claim about boundary/scale accuracy.
    decimals = (
        2 - math.floor(math.log10(value))
        if approximate and method == "scaled_polygon"
        else 2
    )
    answer = round(value, decimals)
    if answer <= 0:
        raise ValueError("Area is too small for the requested reporting unit")
    unique = []
    for citation in citations:
        if citation not in unique:
            unique.append(citation)
    return {
        "status": "ok",
        "method": method,
        "target": request.target,
        "value": answer,
        "unrounded_value": value,
        "unit": request.output_unit,
        "area_m2": float(area),
        "approximate": approximate,
        "boundary_basis": request.boundary_basis,
        "formula": formula,
        "citations": unique,
        "canonical_citations": canonical_citations(unique, images),
        "verification": "Arithmetic, geometry and citation bounds checked. Text readings, dimension endpoints and room boundary are model-proposed, not independently verified.",
        **details,
    }


def calculate_area(args, images):
    request = AreaRequest.model_validate(args)
    citations = _base(request, images)
    readings, components = [], []
    if request.method == "printed":
        if request.printed_area is None or request.rectangles:
            raise ValueError("Printed area requires one label and no rectangles")
        area, reading = _reading(request.printed_area, 2, images, citations)
        readings.append(reading)
        formula = request.printed_area.text + " converted to " + request.output_unit
    else:
        if request.printed_area is not None or not request.rectangles:
            raise ValueError("Rectangle calculation requires dimensioned components")
        area, expressions = Fraction(0), []
        for rectangle in request.rectangles:
            length, l_reading = _reading(rectangle.length, 1, images, citations)
            width, w_reading = _reading(rectangle.width, 1, images, citations)
            readings.extend([l_reading, w_reading])
            sign = 1 if rectangle.operation == "add" else -1
            area += sign * length * width
            expressions.append(
                f"{'+' if sign == 1 else '-'} ({rectangle.length.text} x {rectangle.width.text})"
            )
            components.append(
                {
                    "label": rectangle.label,
                    "operation": rectangle.operation,
                    "area_m2": float(length * width),
                }
            )
        formula = " ".join(expressions).removeprefix("+ ")
    return _result(
        request,
        area,
        citations,
        images,
        method=request.method,
        formula=formula,
        approximate=request.approximate,
        readings=readings,
        components=components,
    )


def measure_polygon(args, images, page_size):
    request = PolygonRequest.model_validate(args)
    citations = _base(request, images)
    width, height = page_size

    def native(point):
        x, y = point
        if not (0 <= x <= 1000 and 0 <= y <= 1000):
            raise ValueError("Geometry must use normalized original-page coordinates")
        return [x * width / 1000, y * height / 1000]

    x0, y0, x1, y1 = request.view_bbox
    if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
        raise ValueError("Select a valid single drawing view")
    view = box(*native([x0, y0]), *native([x1, y1]))
    polygon = Polygon(
        [native(p) for p in request.polygon],
        [[native(p) for p in ring] for ring in request.holes],
    )
    if not polygon.is_valid or polygon.area <= 0 or not view.covers(polygon):
        raise ValueError(
            "Boundary must be simple, inside the view, with contained nonoverlapping holes"
        )
    lines, scales, readings, anchors = [], [], [], []
    for anchor in request.anchors:
        distance, reading = _reading(anchor.dimension, 1, images, citations)
        line = LineString([native(p) for p in anchor.endpoints])
        if line.length < 20 or not view.covers(line):
            raise ValueError(
                "Dimension anchors must span at least 20 native pixels within the same view"
            )
        if any(line.hausdorff_distance(previous) <= 3 for previous in lines):
            raise ValueError("Two distinct dimension anchors are required")
        lines.append(line)
        scales.append(float(distance) / line.length)
        readings.append(reading)
        anchors.append(
            {
                "endpoints": list(map(list, line.coords)),
                "length_px": line.length,
                "length_m": float(distance),
                "m_per_px": scales[-1],
            }
        )
    spread = max(scales) / min(scales) - 1
    if spread > 0.03:
        raise ValueError(
            "Dimension scales disagree by more than 3%; recheck endpoints, units and view"
        )
    factor = sum(a["length_px"] * a["length_m"] for a in anchors) / sum(
        a["length_px"] ** 2 for a in anchors
    )
    area = polygon.area * factor**2
    return _result(
        request,
        area,
        citations,
        images,
        method="scaled_polygon",
        formula=f"({polygon.area:.2f} px2 outer-minus-holes) x ({factor:.9g} m/px)^2, converted to {request.output_unit}",
        approximate=True,
        readings=readings,
        anchors=anchors,
        calibration={
            "m_per_px": factor,
            "relative_spread": spread,
            "tolerance": 0.03,
            "scope": request.view_bbox,
        },
        canonical_polygon=list(map(list, polygon.exterior.coords)),
        canonical_holes=[list(map(list, ring.coords)) for ring in polygon.interiors],
        area_px2=polygon.area,
    )
