import base64
from copy import deepcopy
from dataclasses import replace
from io import BytesIO
import json

from PIL import Image
import pytest

from specialist_views import original_box, rotated_views
from specialist_routing import InspectionRegion
from test_visual_specialist import observation, setup


def encoded_image(image):
    pixels = Image.new("RGB", (200, 100), "white")
    pixels.paste("red", (20, 20, 80, 70))
    buffer = BytesIO()
    pixels.save(buffer, format="PNG")
    return replace(image, data_url="data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())


@pytest.mark.parametrize("rotation,display_box,display_size", [
    (0, [100, 200, 400, 700], [200, 100]),
    (90, [300, 100, 800, 400], [100, 200]),
    (180, [600, 300, 900, 800], [200, 100]),
    (270, [200, 600, 700, 900], [100, 200]),
])
def test_rotated_pixels_and_inverse_citations_match_original_landmark(rotation, display_box, display_size):
    raw = observation()
    raw["citations"][0]["bbox"] = display_box
    specialist, backend, image = setup(raw)
    image = encoded_image(image)
    original = deepcopy(image)
    result = specialist.answer("Read this label", [image], rotation_degrees=rotation)
    assert result["status"] == "ok"
    assert result["result"]["citations"][0]["bbox"] == [20, 20, 80, 70]
    assert result["canonical_citations"][0]["canonical_bbox"] == [340, 440, 460, 540]
    content = backend.complete.call_args.args[0][1]["content"]
    payload = json.loads(content[0]["text"])
    assert payload["images"][0]["size_pixels"] == display_size
    sent = Image.open(BytesIO(base64.b64decode(content[1]["image_url"]["url"].split(",")[1])))
    # Real displayed pixels, not just a matching pair of inverse formulas.
    mask = sent.point(lambda _: 0).convert("L")
    mask.putdata([255 if rgb == (255, 0, 0) else 0 for rgb in sent.getdata()])
    assert list(mask.getbbox()) == [int(v / 1000 * display_size[j % 2]) for j, v in enumerate(display_box)]
    assert image == original


def test_rotated_count_objects_return_in_original_image_coordinates():
    raw = {**observation(), "answer": None, "scope_complete": True, "citations": [],
           "objects": [{"image_id": "crop-1", "bbox": [300, 100, 800, 400],
                        "unit": "door_opening", "status": "supported", "reason": "Visible door leaf"}]}
    specialist, _, image = setup(raw)
    result = specialist.answer("COUNT_UNIT=door_opening", [encoded_image(image)], rotation_degrees=90)
    assert result["result"]["objects"][0]["bbox"] == [20, 20, 80, 70]
    assert result["result"]["scope_complete"] is True
    assert result["canonical_citations"][0]["canonical_bbox"] == [340, 440, 460, 540]


@pytest.mark.parametrize("invalid", [True, False, 0.0, 90.0, 45, -90, 360, "180"])
def test_invalid_rotation_rejected_before_provider_access(invalid):
    specialist, backend, image = setup(observation())
    with pytest.raises(ValueError):
        specialist.answer("Inspect", [image], rotation_degrees=invalid)
    backend.complete.assert_not_called()
    with pytest.raises(ValueError):
        InspectionRegion(question="Inspect", bbox=[0, 0, 200, 200], rotation_degrees=invalid)


def test_identity_view_does_not_decode_or_reencode_image():
    _, _, image = setup(observation())
    assert rotated_views([image], 0)[0] is image
    assert original_box([100, 200, 400, 700], 0) == [100, 200, 400, 700]


def test_agent_dispatches_rotation_without_changing_source_crop_or_reusing_wrong_view():
    from test_specialist_routing import orchestrator, target

    agent = orchestrator()
    rotated = {"regions": [{**target(), "rotation_degrees": 180}]}
    upright = {"regions": [target()]}
    first, _ = agent.execute("inspect_regions", rotated, timeout_seconds=2)
    original, _ = agent.execute("inspect_regions", upright, timeout_seconds=2)
    again, _ = agent.execute("inspect_regions", rotated, timeout_seconds=2)
    calls = agent.specialist_factory.return_value.answer.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["rotation_degrees"] == 180
    assert "rotation_degrees" not in calls[1].kwargs
    assert first["inspections"][0]["image_ids"] == original["inspections"][0]["image_ids"]
    assert first["inspections"][0]["canonical_bbox"] == original["inspections"][0]["canonical_bbox"]
    assert first["inspections"][0]["inspection_view_rotation_clockwise_degrees"] == 180
    assert again["inspections"][0]["inspection_view_rotation_clockwise_degrees"] == 180
