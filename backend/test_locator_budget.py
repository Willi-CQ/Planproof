"""Native locator allowance cannot enlarge visual/provider limits."""
from test_question_agent import agent


def row(name, **extra):
    return {"name": name, "result": {}, **extra}


def test_native_locators_leave_room_for_real_measurements():
    value = agent()
    value.tool_log = [row("read_text")] + [row("get_vector_regions") for _ in range(5)]
    value.tool_log += [row("inspect_regions") for _ in range(6)]
    assert value.tool_budget_used == 6
    value.tool_log += [row("measure_vector_area") for _ in range(6)]
    assert value.tool_budget_used == 12


def test_only_six_locators_are_free_and_raster_cap_is_preserved():
    value = agent()
    value.tool_log = [row("read_text") for _ in range(7)]
    assert value.tool_budget_used == 1
    value.tool_log = [row("crop_region", result={"error_code": "ValueError"}) for _ in range(18)]
    assert value.tool_budget_used == 12


def test_failed_metadata_and_auto_delivery_cannot_increase_allowance():
    value = agent()
    value.tool_log = [row("read_text", result={"error_code": "RuntimeError"}) for _ in range(7)]
    assert value.tool_budget_used == 7
    value.tool_log = [row("read_text", result={"error_code": "ValidationError"}) for _ in range(9)]
    value.tool_log += [row("get_vector_regions", automatic_delivery=True) for _ in range(9)]
    assert value.tool_budget_used == 12


def test_raw_event_cap_remains_finite_with_native_lookups():
    value = agent()
    value.tool_log = [row("get_vector_regions") for _ in range(6)]
    value.tool_log += [row("crop_region", result={"error_code": "ValueError"}) for _ in range(17)]
    assert value.tool_budget_used == 0
    value.tool_log.append(row("crop_region", result={"error_code": "ValueError"}))
    assert value.tool_budget_used == 12
