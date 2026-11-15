"""Deterministic measurement contracts; synthetic geometry is not model accuracy."""

from copy import deepcopy
from fractions import Fraction

import pytest

from measurements import parse_quantity
from test_question_agent import agent, final, response
from token_recovery import compact_observations


SOURCE = {"image_id": "overview", "bbox": [1, 1, 20, 20]}


def reading(text):
    return {
        "text": text,
        "citation": deepcopy(SOURCE),
        **({"number_format": "decimal_point"} if "," in text else {}),
    }


def rectangle_request():
    return {
        "target": "Exercise room",
        "method": "rectangles",
        "output_unit": "ft2",
        "boundary_basis": "Internal face to internal face",
        "approximate": False,
        "region_citation": deepcopy(SOURCE),
        "rectangles": [
            {
                "label": "Main room",
                "length": reading("20'-6\""),
                "width": reading("12'-3\""),
            }
        ],
    }


def polygon_request():
    # Page is 200 x 100. Both independent anchors imply 0.1 m/native pixel.
    # 80x60px room minus a 20x20px void = 4400px2 = 44m2.
    return {
        "target": "Exercise room",
        "output_unit": "m2",
        "boundary_basis": "Visible inner faces; central void excluded",
        "region_citation": deepcopy(SOURCE),
        "view_bbox": [0, 0, 1000, 1000],
        "polygon": [[100, 200], [500, 200], [500, 800], [100, 800]],
        "holes": [[[200, 400], [300, 400], [300, 600], [200, 600]]],
        "anchors": [
            {"dimension": reading("10 m"), "endpoints": [[100, 100], [600, 100]]},
            {"dimension": reading("8 m"), "endpoints": [[800, 100], [800, 900]]},
        ],
    }


@pytest.mark.parametrize(
    "text,expected,power",
    [
        ("12'-3\"", Fraction(37338, 10000), 1),
        ("12′–3½″", Fraction(37465, 10000), 1),
        ("12 ft 3 1/2 in", Fraction(37465, 10000), 1),
        ('7/8"', Fraction(889, 40000), 1),
        ("1/2 in", Fraction(127, 10000), 1),
        ("2.5 m", Fraction(5, 2), 1),
        ("250 cm", Fraction(5, 2), 1),
        ("2500 mm", Fraction(5, 2), 1),
        ("1,250 SF", Fraction(145161, 1250), 2),
        ("24 m²", Fraction(24), 2),
        ("144 in2", Fraction(145161, 1562500), 2),
        ("10000 cm2", Fraction(1), 2),
        ("1000000 mm2", Fraction(1), 2),
        ("16 sq. ft.", 16 * Fraction(381, 1250) ** 2, 2),
    ],
)
def test_explicit_units_and_imperial_fractions(text, expected, power):
    value, actual_power, _ = parse_quantity(text, number_format="decimal_point")
    assert float(value) == pytest.approx(float(expected))
    assert actual_power == power


@pytest.mark.parametrize(
    "text",
    [
        "12",
        "nan m",
        "inf ft2",
        "0 ft",
        "-5 m",
        "12'-15\"",
        "2.5'-3\"",
        "2/0 in",
        "3/2 in",
        "1,25 m",
        "1:100",
        "1\" = 40'",
        "20 x 30 ft",
        "100 SF approx",
        "__import__('os')",
        "5 yd",
    ],
)
def test_ambiguous_invalid_or_executable_text_rejected(text):
    with pytest.raises(ValueError):
        parse_quantity(text)


@pytest.mark.parametrize(
    "text,format,expected",
    [
        ("1,250 m", "decimal_comma", 1.25),
        ("1,250 m", "decimal_point", 1250),
        ("1.250,5 m2", "decimal_comma", 1250.5),
        ("1,250.5 m2", "decimal_point", 1250.5),
        ("2,5 ft", "decimal_comma", 0.762),
        ("12'-3-1/2\"", None, 3.7465),
    ],
)
def test_explicit_number_conventions(text, format, expected):
    assert float(parse_quantity(text, number_format=format)[0]) == pytest.approx(
        expected
    )


def test_ambiguous_thousands_or_decimal_separator_requires_context():
    with pytest.raises(ValueError, match="number_format"):
        parse_quantity("1,250 m")


def test_rectangle_arithmetic_and_unit_conversion():
    a = agent()
    request = rectangle_request()
    result, images = a.execute("calculate_area", request)
    assert result["value"] == 251.12  # ties-to-even, from exact fraction
    assert result["unrounded_value"] == 251.125
    assert result["measurement_id"] == "M1" and not images
    request["output_unit"] = "m2"
    converted, _ = a.execute("calculate_area", request)
    assert converted["value"] == 23.33
    assert converted["measurement_id"] == "M2"
    a.algorithm_factory.assert_not_called()
    a.specialist_factory.assert_not_called()


def test_composite_rectangles_subtract_cutouts():
    a = agent()
    request = rectangle_request()
    request["rectangles"] = [
        {"label": "Main", "length": reading("20 ft"), "width": reading("15 ft")},
        {
            "label": "Void",
            "length": reading("5 ft"),
            "width": reading("4 ft"),
            "operation": "subtract",
        },
    ]
    assert a.execute("calculate_area", request)[0]["value"] == 280
    request["rectangles"][0]["operation"] = "subtract"
    with pytest.raises(ValueError):
        a.execute("calculate_area", request)


def test_printed_area_does_not_require_calibration():
    request = rectangle_request()
    request.update(method="printed", rectangles=[], printed_area=reading("1,250 SF"))
    assert agent().execute("calculate_area", request)[0]["value"] == 1250
    request["printed_area"]["text"] = "1250 ft"
    with pytest.raises(ValueError, match="Length and area"):
        agent().execute("calculate_area", request)


def test_explicit_unit_note_is_required_for_bare_dimension():
    request = rectangle_request()
    length = request["rectangles"][0]["length"]
    length.update(text="2500", unit_context="mm")
    with pytest.raises(ValueError, match="own source citation"):
        agent().execute("calculate_area", request)
    length["unit_citation"] = deepcopy(SOURCE)
    assert agent().execute("calculate_area", request)[0]["value"] == 100.48
    length["text"] = "2500 ft"
    with pytest.raises(ValueError, match="conflicts"):
        agent().execute("calculate_area", request)


def test_crop_citations_map_back_to_native_page():
    a = agent()
    a.execute("crop_region", {"bbox": [500, 0, 1000, 1000]})
    request = rectangle_request()
    request["rectangles"][0]["length"]["citation"]["image_id"] = "crop-1"
    result = a.execute("calculate_area", request)[0]
    assert result["canonical_citations"][1]["canonical_bbox"] == [101, 1, 120, 20]


@pytest.mark.parametrize(
    "citation",
    [
        {"image_id": "unknown", "bbox": [1, 1, 20, 20]},
        {"image_id": "overview", "bbox": [1, 1, 201, 20]},
        {"image_id": "overview", "bbox": [1, 1, float("nan"), 20]},
        {"image_id": "overview", "bbox": [True, 1, 20, 20]},
    ],
)
def test_invalid_source_citation_never_registers_measurement(citation):
    a = agent()
    request = rectangle_request()
    request["rectangles"][0]["length"]["citation"] = citation
    with pytest.raises(ValueError):
        a.execute("calculate_area", request)
    assert not a.measurements


def test_polygon_holes_and_original_page_aspect_ratio():
    result = agent().execute("measure_polygon", polygon_request())[0]
    assert result["value"] == 44 and result["approximate"]
    assert result["area_px2"] == 4400
    assert result["calibration"]["m_per_px"] == pytest.approx(0.1)
    assert result["canonical_polygon"][0] == [20, 20]
    assert result["canonical_holes"][0][0] == [40, 40]


@pytest.mark.parametrize(
    "length,width,expected",
    [("2 m", "1.6 m", 1.76), ("1 m", "0.8 m", 0.44), ("0.1 m", "0.08 m", 0.0044)],
)
def test_small_polygon_estimates_preserve_value_and_approximate_provenance(
    length, width, expected
):
    value = agent()
    request = polygon_request()
    request["anchors"][0]["dimension"]["text"] = length
    request["anchors"][1]["dimension"]["text"] = width
    measured = value.execute("measure_polygon", request)[0]
    assert measured["value"] == pytest.approx(expected)
    assert measured["approximate"]
    answer = value.finish(
        {**final(), "answer": measured["value"], "unit": "m2", "measurement_id": "M1"}
    )
    assert answer["answer"] == measured["value"]
    assert "Approximate" in answer["explanation"]
    assert answer["citations"]


@pytest.mark.parametrize("unit", ["m2", "ft2", "cm2", "mm2", "in2"])
def test_polygon_rounding_error_is_bounded_across_output_units(unit):
    request = polygon_request()
    request["output_unit"] = unit
    result = agent().execute("measure_polygon", request)[0]
    assert abs(result["value"] / result["unrounded_value"] - 1) <= 0.005
    assert result["area_m2"] == pytest.approx(44)


def test_nonrectangular_room_uses_polygon_not_bounding_box():
    request = polygon_request()
    request.update(
        polygon=[
            [100, 200],
            [500, 200],
            [500, 400],
            [300, 400],
            [300, 800],
            [100, 800],
        ],
        holes=[],
    )
    result = agent().execute("measure_polygon", request)[0]
    assert result["value"] == 32


@pytest.mark.parametrize(
    "mutation",
    [
        "one_anchor",
        "duplicate",
        "reverse_duplicate",
        "conflicting_scale",
        "bad_polygon",
        "hole_outside",
        "overlapping_holes",
        "negative_point",
        "nan",
        "tiny_anchor",
        "outside_view",
        "wrong_quantity",
        "bool",
    ],
)
def test_polygon_fails_closed(mutation):
    request = polygon_request()
    if mutation == "one_anchor":
        request["anchors"].pop()
    elif mutation in ("duplicate", "reverse_duplicate"):
        request["anchors"][1] = deepcopy(request["anchors"][0])
        if mutation == "reverse_duplicate":
            request["anchors"][1]["endpoints"].reverse()
    elif mutation == "conflicting_scale":
        request["anchors"][1]["dimension"]["text"] = "80 m"
    elif mutation == "bad_polygon":
        request["polygon"] = [[100, 200], [500, 800], [500, 200], [100, 800]]
    elif mutation == "hole_outside":
        request["holes"] = [[[600, 400], [700, 400], [700, 600], [600, 600]]]
    elif mutation == "overlapping_holes":
        request["holes"].append(deepcopy(request["holes"][0]))
    elif mutation == "negative_point":
        request["polygon"][0][0] = -1
    elif mutation == "nan":
        request["polygon"][0][0] = float("nan")
    elif mutation == "tiny_anchor":
        request["anchors"][0]["endpoints"][1] = [101, 100]
    elif mutation == "outside_view":
        request["view_bbox"] = [0, 0, 600, 1000]
    elif mutation == "wrong_quantity":
        request["anchors"][0]["dimension"]["text"] = "10 m2"
    elif mutation == "bool":
        request["polygon"][0][0] = True
    a = agent()
    with pytest.raises(ValueError):
        a.execute("measure_polygon", request)
    assert not a.measurements


def test_final_area_is_bound_to_calculation_with_all_input_citations():
    a = agent()
    request = rectangle_request()
    request["rectangles"][0]["length"]["citation"]["bbox"] = [30, 20, 60, 30]
    a.execute("calculate_area", request)
    answer = {
        **final(),
        "answer": 251.12,
        "unit": "square feet",
        "measurement_id": "M1",
    }
    result = a.finish(answer)
    assert result["unit"] == "ft2" and len(result["citations"]) == 2
    for patch in (
        {"measurement_id": None},
        {"measurement_id": "M99"},
        {"answer": 999},
        {"unit": "m2"},
        {"answer": "251.12"},
    ):
        with pytest.raises(ValueError):
            a.finish({**answer, **patch})


def test_raster_result_cannot_suppress_approximate_label():
    a = agent()
    a.execute("measure_polygon", polygon_request())
    result = a.finish({**final(), "answer": 44, "unit": "m2", "measurement_id": "M1"})
    assert "Approximate" in result["explanation"]
    assert any("Approximate" in item for item in result["uncertainties"])


def test_direct_control_cannot_use_measurements_but_can_read_labels():
    a = agent(mode="direct")
    with pytest.raises(ValueError):
        a.execute("calculate_area", rectangle_request())
    assert a.finish({**final(), "answer": 24, "unit": "m2"})["answer"] == 24


def test_full_scripted_tool_flow_and_token_recovery_preserve_provenance():
    a = agent(
        response("calculate_area", rectangle_request()),
        response(
            "submit_answer",
            {**final(), "answer": 251.12, "unit": "ft2", "measurement_id": "M1"},
        ),
        mode="visual_only",
        allow=False,
    )
    result = a.run()
    assert result["status"] == "ok" and result["measurement"]["value"] == 251.12
    record = compact_observations(result["tool_trace"])["observations"][0]
    assert record["result"]["measurement_id"] == "M1"
    assert record["result"]["citations"] and record["result"]["formula"]


def test_tool_validation_reason_is_available_for_input_repair():
    request = polygon_request()
    request["anchors"][1]["dimension"]["text"] = "80 m"
    a = agent(response("measure_polygon", request), response("submit_answer", final()))
    result = a.run()
    assert "scales disagree" in result["tool_trace"][0]["result"]["instruction"]


def test_function_schemas_build_and_have_bounded_geometry():
    from question_agent import declarations

    tools = {
        item.name: item.parameters_json_schema for item in declarations("auto", True)
    }
    assert tools["measure_polygon"]["properties"]["anchors"]["minItems"] == 2
    assert tools["calculate_area"]["additionalProperties"] is False
