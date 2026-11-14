"""Pure, bounded evidence selection for a coordinator's final recovery request.

These helpers do not call models, create crops, merge detections, or infer counts.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping


_WARNINGS = [
    "Whole-page coverage remains unknown; omitted or missing evidence never means absence.",
    "Specialist answers are local advisory observations, not verified facts. Never sum overlapping crop counts.",
    "Algorithm candidates may be ML-assisted; they are not independent confirmation or physical counts.",
]
_ALGORITHM_FIELDS = (
    "mode",
    "rooms",
    "observations",
    "omitted_observations",
    "openings",
    "omitted_openings",
    "warnings",
    "capabilities",
    "already_supplied_classes",
)
_SPECIALIST_FIELDS = (
    "status",
    "result",
    "advisory",
    "canonical_citations",
    "image_ids",
    "scope",
    "error",
)
_CROP_FIELDS = (
    "status",
    "image",
    "cached",
    "image_id",
    "canonical_bbox",
    "source_render_size",
    "sent_size",
    "coordinate_system",
    "sent_to_source_render",
    "crop_to_canonical",
    "page_index",
    "metadata",
    "scope",
    "error",
)


def _size(value):
    return len(json.dumps(value, ensure_ascii=True, allow_nan=False))


def _record(entry, index):
    name = entry.get("name")
    result = entry.get("result")
    if not isinstance(result, Mapping):
        return None
    if name == "get_algorithms":
        fields = _ALGORITHM_FIELDS + ("status", "error")
    elif name in ("inspect_nemotron", "review_answer"):
        fields = _SPECIALIST_FIELDS
    elif name in ("crop_region", "crop_room"):
        fields = _CROP_FIELDS
    else:
        # Reflection, proposed final answers and unknown tools are not evidence.
        return None
    record = {
        "tool_index": index,
        "tool": name,
        "result": {key: result[key] for key in fields if key in result},
    }
    args = entry.get("args")
    if isinstance(args, Mapping):
        # The question identifies the scope of an advisory answer. Never copy a
        # review's candidate/draft into the evidence used to finalize that draft.
        context = {key: args[key] for key in ("question", "image_ids") if key in args}
        if context:
            record["request_scope"] = context
    try:
        return json.loads(json.dumps(record, allow_nan=False))
    except (ValueError, TypeError, OverflowError):
        return None


def compact_observations(tool_log, max_chars=12000):
    """Return bounded JSON evidence, with omissions and scope explicitly marked.

    ``max_chars`` bounds the complete default ``json.dumps`` representation,
    including warnings. Too-small budgets raise ValueError. Specialist records
    are atomic: an answer is never retained while dropping its citations. Only
    algorithm candidate arrays may be shortened, with exact additional omission
    counts. Inputs are never mutated.
    """
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    entries = list(tool_log)
    output = {
        "coverage": "unknown",
        "warnings": list(_WARNINGS),
        "observations": [],
        "omitted_tool_records": len(entries),
        "truncated_tool_records": 0,
        "omission_note": "Omitted records include unsupported tools, drafts, reflection, invalid data and records exceeding the budget.",
    }
    if _size(output) > max_chars:
        raise ValueError("max_chars is too small for mandatory evidence limitations")
    records = []
    for index, entry in enumerate(entries):
        if isinstance(entry, Mapping):
            record = _record(entry, index)
            if record is not None:
                records.append(record)
    # Prioritize the actual visual observations over potentially large geometry
    # catalogs. Original indices preserve provenance and temporal ordering.
    records.sort(
        key=lambda item: (
            0
            if item["tool"] in ("inspect_nemotron", "review_answer")
            else 2
            if item["tool"] == "get_algorithms"
            else 1,
            item["tool_index"],
        )
    )
    for record in records:
        candidate = copy.deepcopy(output)
        candidate["observations"].append(record)
        candidate["omitted_tool_records"] -= 1
        if _size(candidate) <= max_chars:
            output = candidate
            continue
        if record["tool"] != "get_algorithms":
            continue
        shortened = copy.deepcopy(record)
        arrays = {
            key: shortened["result"][key]
            for key in ("rooms", "observations", "openings")
            if isinstance(shortened["result"].get(key), list)
        }
        if not any(arrays.values()):
            continue
        shortened["additional_omissions"] = {key: 0 for key in arrays}
        shortened["truncation_warning"] = (
            "Candidate lists are incomplete. Additional omissions are separate from upstream omissions; do not infer totals or absence."
        )
        candidate["observations"][-1] = shortened
        candidate["truncated_tool_records"] += 1
        while _size(candidate) > max_chars and any(arrays.values()):
            # Trim the largest array's tail, keeping each retained geometry item
            # complete (including its ID, source family, bbox and uncertainty).
            key = max(
                (key for key in arrays if arrays[key]),
                key=lambda key: _size(arrays[key]),
            )
            arrays[key].pop()
            shortened["additional_omissions"][key] += 1
        if _size(candidate) <= max_chars:
            output = candidate
    return output


def _geometry(image, page_frame):
    metadata = getattr(image, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    box = metadata.get("canonical_bbox")
    size = page_frame[:2]
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        return None
    values = [*box, *size]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        return None
    x0, y0, x1, y1 = box
    width, height = size
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        return None
    area = (x1 - x0) * (y1 - y0)
    if area > 0.75 * width * height:
        return None
    if metadata.get("page_index") != page_frame[2]:
        return None
    return tuple(box), area, page_frame


def _page_frame(images):
    for image in images:
        if getattr(image, "image_id", None) != "overview":
            continue
        metadata = getattr(image, "metadata", {})
        if not isinstance(metadata, Mapping):
            continue
        box = metadata.get("canonical_bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in box
        ):
            continue
        if box[0] == 0 and box[1] == 0 and box[2] > 0 and box[3] > 0:
            return (box[2], box[3], metadata.get("page_index"))
    return None


def select_detail_images(images, tool_log, max_count=2):
    """Choose existing, previously uninspected, nonoverlapping local crops.

    Overlap is limited to 20% of the smaller crop, not IoU. All selected images
    must share a source coordinate frame. This is scheduling, never evidence of
    complete coverage or disjoint physical objects. A valid overview is required
    to establish the original page frame: crop source_render_size describes the
    crop raster, not the full page.
    """
    if isinstance(max_count, bool) or not isinstance(max_count, int) or max_count < 0:
        raise ValueError("max_count must be a nonnegative integer")
    if max_count == 0:
        return []
    inspected = set()
    for entry in tool_log:
        if not isinstance(entry, Mapping) or entry.get("name") not in (
            "inspect_nemotron",
            "review_answer",
        ):
            continue
        result = entry.get("result")
        if not isinstance(result, Mapping) or result.get("status") != "ok":
            continue
        args = entry.get("args")
        ids = result.get(
            "image_ids", args.get("image_ids", []) if isinstance(args, Mapping) else []
        )
        if isinstance(ids, (list, tuple)):
            inspected.update(value for value in ids if isinstance(value, str))
    candidates = []
    seen = set()
    iterable = list(images.values() if isinstance(images, Mapping) else images)
    page_frame = _page_frame(iterable)
    if page_frame is None:
        return []
    for image in iterable:
        image_id = getattr(image, "image_id", None)
        if (
            not isinstance(image_id, str)
            or not image_id
            or image_id == "overview"
            or image_id in seen
            or image_id in inspected
        ):
            continue
        geometry = _geometry(image, page_frame)
        if geometry is not None:
            candidates.append((image_id, geometry))
            seen.add(image_id)
    selected = []
    for image_id, geometry in candidates:
        box, area, frame = geometry
        compatible = True
        for _, (other, other_area, other_frame) in selected:
            intersection = max(0, min(box[2], other[2]) - max(box[0], other[0])) * max(
                0, min(box[3], other[3]) - max(box[1], other[1])
            )
            if frame != other_frame or intersection > 0.20 * min(area, other_area):
                compatible = False
                break
        if compatible:
            selected.append((image_id, geometry))
            if len(selected) >= max_count:
                break
    return [image_id for image_id, _ in selected]
