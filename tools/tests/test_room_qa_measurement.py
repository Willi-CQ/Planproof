from pathlib import Path

from PIL import Image
import pytest

from tools.free_agent_room_qa.answers import compile_question
from tools.free_agent_room_qa.room_geometry import bar_candidates, crop_outline, inspect_room

DRAWING = Path(__file__).resolve().parents[2] / "frontend/src/assets/plan-sample.png"
ANCHOR = {"id":"hall", "bbox":[510,494,613,513]}


@pytest.mark.parametrize("q", [
    "mark me the hallway, how big is it?",
    "mark me the hallway in the drawing, tell me how large it is",
    "Show me the kitchen on this sheet and tell me how big it is.",
    "Markiere das Büro auf dieser Zeichnung und sag mir, wie groß es ist.",
    "Show me the hallway and tell me how big it is.",
    "What's the area of the corridor?",
    "Markiere den Flur und sag mir, wie groß er ist.",
    "Wie viele Quadratmeter hat der Flur?",
    "What is the area of room R101?",
])
def test_natural_area_questions(q):
    assert compile_question(q)["op"] == "measure_room"


def test_location_and_boundaries():
    assert compile_question("Zeig mir den Flur.")["op"] == "locate_room"
    assert compile_question('Find the label "hallway".')["op"] == "locate_text"
    assert compile_question("How much does the hallway cost?")["op"] == "unsupported"
    assert compile_question("Mark the hallway and measure the bedroom.")["op"] == "unsupported"
    assert compile_question("Mark the hallway in the basement, tell me how large it is.")["op"] == "unsupported"


def test_pixels_drive_outline_and_scale_candidates():
    with Image.open(DRAWING) as image:
        polygon = crop_outline(image, ANCHOR, [ANCHOR])
        assert polygon is not None
        assert polygon.bounds == pytest.approx((160,420,960,580), abs=4)
        bars = bar_candidates(image)
        assert any(abs(a-160)<4 and abs(b-360)<4 for a,b,*_ in bars)


def test_missing_scale_keeps_the_marked_outline(monkeypatch):
    monkeypatch.setattr("tools.free_agent_room_qa.room_geometry.read_scale", lambda *args, **kwargs: None)
    with Image.open(DRAWING) as image:
        result = inspect_room(image, {"lines":[{"text":"HALLWAY","bbox":ANCHOR["bbox"]}]},
                              compile_question("How big is the hallway?"))
    assert result["status"] == "partial" and result["reason_code"] == "room_scale_unverified"
    assert result["evidence"][0]["polygon"] and result["unit"] is None
