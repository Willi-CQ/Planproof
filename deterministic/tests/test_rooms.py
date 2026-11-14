"""Room rules on controlled drawings; these tests are not real-data accuracy claims."""

import cv2
import numpy as np
import pytest
from shapely.geometry import box

from planproof_evidence.contracts import polygon_json
from planproof_evidence.geometry import mask_polygons
from planproof_evidence.rooms import (
    RoomRules,
    _gap_candidates,
    _profile_gaps,
    classify_gap,
    derive_rooms,
)


def example(opening=True, two_seeds=True):
    wall = np.zeros((240, 320), np.uint8)
    cv2.rectangle(wall, (30, 30), (290, 210), 1, 10)
    cv2.line(wall, (160, 30), (160, 210), 1, 10)
    if opening:
        wall[95:136, 150:171] = 0
    observations = [
        {
            "id": "ml-wall",
            "class": "wall",
            "page_index": 0,
            "source_family": "ml",
            "model_binding": {"name": "main"},
            "polygons": [polygon_json(p) for p in mask_polygons(wall)],
        }
    ]
    for i, bounds in enumerate(
        ([40, 40, 150, 200], [170, 40, 280, 200]) if two_seeds else ([40, 40, 280, 200],)
    ):
        observations.append(
            {
                "id": "seed-" + str(i),
                "class": "room",
                "page_index": 0,
                "source_family": "ml",
                "model_binding": {"name": "main"},
                "polygons": [polygon_json(box(*bounds))],
            }
        )
    frame = {"page_index": 0, "image_size": [320, 240], "id": "page0"}
    return wall, observations, frame


def test_wall_end_gap_is_not_an_arbitrary_segment_break():
    wall, _, _ = example()
    gaps, _, truncated = _gap_candidates(wall, 1, RoomRules())
    assert not truncated
    assert len(gaps) == 1
    assert 35 <= gaps[0]["length_px"] <= 45
    full, _, _ = example(False)
    assert _gap_candidates(full, 1, RoomRules())[0] == []


def symbol(kind, rotation=0):
    ink = np.zeros((220, 220), np.uint8)
    cv2.rectangle(ink, (10, 95), (79, 105), 1, -1)
    cv2.rectangle(ink, (141, 95), (210, 105), 1, -1)
    if kind == "door":
        cv2.line(ink, (80, 100), (80, 160), 1, 2)
        cv2.ellipse(ink, (80, 100), (60, 60), 0, 0, 90, 1, 2)
    elif kind == "window":
        cv2.line(ink, (80, 97), (140, 97), 1, 1)
        cv2.line(ink, (80, 103), (140, 103), 1, 1)
    elif kind == "ambiguous":
        cv2.line(ink, (90, 90), (130, 110), 1, 3)
    a, b = np.array([80.0, 100.0]), np.array([140.0, 100.0])
    if rotation:
        transform = cv2.getRotationMatrix2D((110, 110), rotation, 1)
        ink = cv2.warpAffine(ink, transform, (220, 220), flags=cv2.INTER_NEAREST)
        a, b = np.column_stack([np.array([a, b]), np.ones(2)]) @ transform.T
    return ink, {"segment": [a.tolist(), b.tolist()], "length_px": 60.0, "thickness_px": 10.0}


@pytest.mark.parametrize("kind", ["door", "window", "walkthrough", "ambiguous"])
@pytest.mark.parametrize("rotation", [0, 35, 90])
def test_rotated_gap_symbols(kind, rotation):
    ink, gap = symbol(kind, rotation)
    result = classify_gap(gap, ink, 1)
    assert result["type"] == ("unknown" if kind == "ambiguous" else kind)


def test_virtual_separator_partitions_but_does_not_become_wall():
    wall, observations, frame = example()
    before = wall.copy()
    result = derive_rooms(frame, observations, wall, {}, 1, rules=RoomRules(apply_refinements=True))
    assert np.array_equal(wall, before)
    bounded = [r for r in result["rooms"] if r["status"] == "bounded_seed_region"]
    assert len(bounded) == 2
    assert len(result["separators"]) == 1
    assert result["separators"][0]["physical_wall"] is False
    assert result["openings"][0]["type"] == "walkthrough"
    assert result["adjacency"][0]["relation"] == "possible_passage"
    assert result["adjacency"][0]["verified_access"] is False
    assert {w["role"] for w in result["walls"]} >= {"inner", "outer"}


def test_open_plan_seeds_do_not_invent_partition():
    wall, observations, frame = example()
    wall[36:205, 150:171] = 0
    observations[0]["polygons"] = [polygon_json(p) for p in mask_polygons(wall)]
    result = derive_rooms(frame, observations, wall, {}, 1)
    shared = [r for r in result["enclosures"] if len(r["ml_seed_ids"]) == 2]
    assert len(shared) == 1
    assert set(shared[0]["ml_seed_ids"]) == {"seed-0", "seed-1"}
    assert len(result["rooms"]) == 2
    assert all(r["status"] == "shared_space_ml_boundary" for r in result["rooms"])
    assert not result["separators"]


def test_cropped_building_keeps_outer_role_unknown():
    wall, observations, frame = example(False)
    wall[0:35, 25:36] = 1
    observations[0]["polygons"] = [polygon_json(p) for p in mask_polygons(wall)]
    result = derive_rooms(frame, observations, wall, {}, 1)
    assert result["diagnostics"]["wall_touches_image_edge"]
    assert all(w["role"] == "unknown" for w in result["walls"])


def test_no_supported_ml_wall_model_is_unavailable():
    wall, observations, frame = example()
    result = derive_rooms(frame, observations, wall, {}, 1, model_source="cad")
    assert result["status"] == "unavailable_no_ml_walls"
    assert not result["rooms"]


def test_connected_window_frames_still_produce_pixel_gap():
    ink, gap = symbol("window")
    wall = ink.copy()
    gaps = _profile_gaps(wall, ink, 1, RoomRules())
    assert any(abs(g["length_px"] - gap["length_px"]) < 8 for g in gaps)
    solid = np.zeros_like(ink)
    cv2.rectangle(solid, (10, 95), (210, 105), 1, -1)
    assert not _profile_gaps(solid, solid, 1, RoomRules())


def test_multipart_seed_is_retained_as_one_hypothesis():
    wall, observations, frame = example(False, False)
    observations[1]["polygons"] = [
        polygon_json(box(45, 45, 100, 90)),
        polygon_json(box(180, 140, 230, 190)),
    ]
    result = derive_rooms(frame, observations, wall, {}, 1)
    assert len(result["rooms"]) == 1
    assert len(result["rooms"][0]["polygons"]) == 2
    assert result["rooms"][0]["status"] == "unresolved_ml_boundary"


def test_refinement_is_separate_from_primary_ml_geometry_by_default():
    wall, observations, frame = example()
    result = derive_rooms(frame, observations, wall, {}, 1)
    assert all(r["primary_geometry"] == "retained_ml_seed" for r in result["rooms"])
    assert all(r["boundary_alternative"] for r in result["rooms"])
    assert result["rooms"][0]["polygons"] == observations[1]["polygons"]
    assert not any(d["applied_to_primary"] for d in result["diagnostics"]["refinement_decisions"])


def test_bundle_crop_and_artifact_tamper(tmp_path):
    import json
    from PIL import Image
    from planproof_evidence import analyze, build_room_boundaries, get_room_context
    from planproof_evidence.contracts import rgb_hash, sha256, write_json

    wall, observations, _ = example()
    image = np.repeat(((1 - wall) * 255)[:, :, None], 3, axis=2)
    path = tmp_path / "source.png"
    Image.fromarray(image).save(path)
    prediction = {
        "schema_version": "planproof-detection-v1",
        "model_version": "fixture",
        "document_id": "fixture",
        "page_index": 0,
        "image_size": [320, 240],
        "coordinate_system": "pixel_edges_xy_top_left",
        "instances": [
            {
                "id": o["id"],
                "class": o["class"],
                "polygons": o["polygons"],
                "confidence": 0.8,
                "bbox": list(box(0, 0, 320, 240).bounds)
                if o["class"] == "wall"
                else [40, 40, 280, 200],
            }
            for o in observations
        ],
    }
    write_json(tmp_path / "prediction.json", prediction)
    write_json(
        tmp_path / "binding.json",
        {
            "sources": [
                {
                    "name": "main",
                    "prediction_path": "prediction.json",
                    "prediction_sha256": sha256(tmp_path / "prediction.json"),
                    "model_sha256": "a" * 64,
                    "profile_sha256": "b" * 64,
                    "raster_sha256": rgb_hash(image),
                    "image_size": [320, 240],
                    "page_index": 0,
                    "document_id": "fixture",
                }
            ]
        },
    )
    job = tmp_path / "job"
    analyze(path, job, ml_predictions=tmp_path / "binding.json")
    original = sha256(job / "evidence.json")
    bundle = build_room_boundaries(job / "evidence.json", tmp_path / "rooms")
    assert sha256(job / "evidence.json") == original
    boundary_path = tmp_path / "rooms/room-boundaries.json"
    room_id = bundle["pages"][0]["rooms"][0]["id"]
    crop = get_room_context(boundary_path, room_id, tmp_path / "crop")
    x1, y1, x2, y2 = crop["bbox"]
    assert np.array_equal(np.asarray(Image.open(tmp_path / "crop/clean.png")), image[y1:y2, x1:x2])
    context = json.loads((tmp_path / "crop/agent-context.json").read_text())
    assert context["mode"] == "ml_assisted"
    assert any(
        o["source_family"] == "ml" for o in context["source_evidence_context"]["observations"]
    )
    with pytest.raises(ValueError, match="registered region"):
        get_room_context(boundary_path, "missing", tmp_path / "bad")
    with (job / "page-000/ink.png").open("ab") as file:
        file.write(b"modified")
    with pytest.raises(ValueError, match="artifact changed"):
        build_room_boundaries(job / "evidence.json", tmp_path / "tampered")
