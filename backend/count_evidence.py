"""Localized counting evidence. Geometry validates provenance, never object truth."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from planproof_feedback.regions import canonical_citations


class CountBasis(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    object_ids: list[str] | None = Field(default=None, max_length=1000)
    scope: Literal["complete", "partial"]
    unit: Literal["door_opening", "door_leaf"]


def door_count_unit(question):
    if re.search(r"\b(doors?|doorways?|türen|tueren)\b", question, re.I) and re.search(
        r"how many|count|number|amount|quantity|wie viele|anzahl", question, re.I
    ):
        # Count the requested noun, not words in eligibility/exclusion clauses.
        # "Doorways ... without a door leaf do not count" still asks for openings.
        target = re.search(
            r"\b(?:how\s+many|count|number\s+of|amount\s+of|quantity\s+of)\s+"
            r"(?:(?:the|all|individual|visible|separate|each|every)\s+)*"
            r"(?P<noun>(?:door\s+)?(?:leaves|leaf)|doorways?|doors?)\b",
            question,
            re.I,
        )
        return (
            "door_leaf"
            if target and re.search(r"\b(?:leaf|leaves)\b", target["noun"], re.I)
            else "door_opening"
        )
    return None


def overlap(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    aa, bb = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (aa + bb - intersection) if intersection else 0


def inventory(records, images, unit):
    """Stable IDs, same-source duplicate suppression, and visible ambiguity.

    Repeated-room arithmetic is deliberately not an input. Overlapping detections
    are only merged at high IoU; weaker overlaps stay visible and block an exact
    count. This conservative geometry is not a semantic verifier.
    """
    objects = []
    for image_id, record in records.items():
        for index, obj in enumerate(record.get("objects", [])):
            if obj["unit"] != unit or obj["image_id"] != image_id:
                continue
            citation = {"image_id": image_id, "bbox": obj["bbox"]}
            canonical = canonical_citations([citation], list(images.values()))[0]
            item = {
                **canonical,
                **obj,
                "id": f"{image_id}-D{index + 1}",
                "citation": citation,
                "sources": [image_id],
            }
            duplicate = next(
                (
                    o
                    for o in objects
                    if overlap(o["canonical_bbox"], item["canonical_bbox"])
                    >= (0.95 if image_id in o["sources"] else 0.5)
                ),
                None,
            )
            if duplicate:
                duplicate["sources"] = list(
                    dict.fromkeys([*duplicate["sources"], image_id])
                )
                if duplicate["status"] != item["status"]:
                    duplicate["status"] = "uncertain"
                    duplicate["reason"] = (
                        "Overlapping inspections disagree about this candidate."
                    )
            else:
                objects.append(item)
    for i, a in enumerate(objects):
        for b in objects[i + 1 :]:
            if overlap(a["canonical_bbox"], b["canonical_bbox"]) > 0.15:
                for obj in (a, b):
                    obj["status"] = "uncertain"
                    obj["reason"] = (
                        "Nearby overlapping candidates need a duplicate check."
                    )
    return {
        "unit": unit,
        "supported": sum(o["status"] == "supported" for o in objects),
        "uncertain": sum(o["status"] == "uncertain" for o in objects),
        "objects": objects,
        "verification": "Model-localized candidates; spatial validation and conservative deduplication are not independent verification.",
    }


def count_presentation(answer, evidence, *, complete=False):
    """Never promote an unsupported numeric draft into the count briefing."""
    supported, uncertain = evidence["supported"], evidence["uncertain"]
    noun = "door openings" if evidence["unit"] == "door_opening" else "door leaves"
    if complete:
        text = f"{answer['answer']:g} {noun} were counted from individually located symbols."
    elif not evidence["objects"]:
        text = "No door locations were established. The inspection is incomplete; this is not a count of zero."
    else:
        text = (
            f"Inspection is incomplete. Located {supported} supported {noun} candidates "
            f"and {uncertain} uncertain candidates in the inspected regions. "
            "These are provisional observations, not a verified whole-drawing total."
        )
    answer["explanation"] = text
    answer["briefing"] = [
        {
            "section": "answer" if complete else "limitation",
            "text": text,
            "citations": [],
        }
    ]
    answer["coverage"] = (
        "Agent reports the requested scope inspected; object identities and scope remain model judgments."
        if complete
        else "Partial inspection only. Uninspected, deferred or ambiguous regions cannot be counted as empty."
    )
    if not complete and answer.get("clarification_question"):
        answer["briefing"].append(
            {
                "section": "limitation",
                "text": answer["clarification_question"],
                "citations": [],
            }
        )
    # Do not reintroduce ungrounded totals through a generated limitation paragraph.
    # Specific visible ambiguities remain attached to each candidate's reason.
    answer["uncertainties"] = [
        "Boxes show model-supported (green) and uncertain (amber) candidates; neither is human-verified.",
        "Region selection and door roles remain model judgments; omitted regions are not independently ruled out.",
    ]
    if not complete:
        answer["uncertainties"].append(
            "Unresolved regions or candidates prevent an established whole-drawing total."
        )
        answer["count_basis"] = {
            "scope": "partial",
            "unit": evidence["unit"],
            "object_ids": [
                o["id"] for o in evidence["objects"] if o["status"] == "supported"
            ],
        }
    return answer
