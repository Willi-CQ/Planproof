from math import isfinite
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator


Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
Box = Annotated[list[float], Field(min_length=4, max_length=4)]
Provider = Literal["deterministic", "ml"]
Status = Literal["SUPPORTED", "NEEDS_REVIEW", "INSUFFICIENT_EVIDENCE"]
Reason = Literal[
    "analyzer_unavailable", "non_independent", "ambiguous_matches",
    "unmatched_detections", "coverage_unknown",
]
# Initial operating point for the demo. Tune against annotated real pages.
MATCH_IOU = 0.5


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class PageFrame(Contract):
    document_id: Name
    page_index: Annotated[int, Field(ge=0)]
    source_sha256: Digest
    raster_sha256: Digest
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]
    coordinate_system: Literal["pixel_edges_xy_top_left"] = "pixel_edges_xy_top_left"


class DoorDetection(Contract):
    id: Name
    bbox: Box

    @field_validator("bbox")
    @classmethod
    def validate_box(cls, box: list[float]) -> list[float]:
        left, top, right, bottom = box
        if not all(isfinite(value) for value in box) or not (0 <= left < right and 0 <= top < bottom):
            raise ValueError("Expected a finite, positive-area box in XYXY pixel order.")
        return box


class AnalyzerResult(Contract):
    schema_version: Literal["planproof-door-result-v1"]
    source: PageFrame
    version: Name
    status: Literal["ok", "partial", "unavailable"]
    mode: Literal["independent", "hybrid"]
    complete: bool = False
    detections: Annotated[list[DoorDetection], Field(max_length=80)] = Field(default_factory=list)

    @field_validator("detections")
    @classmethod
    def unique_ids(cls, detections: list[DoorDetection]) -> list[DoorDetection]:
        if len({detection.id for detection in detections}) != len(detections):
            raise ValueError("Detection IDs must be unique within an analyzer.")
        return detections


class DoorCountRequest(Contract):
    document_id: Name
    deterministic: AnalyzerResult | None = None
    ml: AnalyzerResult | None = None
    allow_gemini_review: bool = False


class Evidence(Contract):
    id: str
    provider: Provider
    bbox_xyxy: Box


class AnalyzerSummary(Contract):
    status: Literal["ok", "partial", "unavailable"]
    version: str | None
    detected_count: int | None
    complete: bool


class Comparison(Contract):
    status: Status
    answer: int | None = None
    reasons: list[Reason] = Field(default_factory=list)
    deterministic: AnalyzerSummary
    ml: AnalyzerSummary
    matched_pairs: list[tuple[str, str]] = Field(default_factory=list)
    matched_evidence: list[Evidence] = Field(default_factory=list)
    disputed_evidence: list[Evidence] = Field(default_factory=list)


def validate_source(result: AnalyzerResult | None, actual: PageFrame) -> None:
    if result is None:
        return
    if result.source != actual:
        raise ValueError("Result must identify the same source file and decoded page raster.")
    if any(box.bbox[2] > actual.width or box.bbox[3] > actual.height for box in result.detections):
        raise ValueError("A detection is outside the canonical page.")
    if result.status == "unavailable" and result.detections:
        raise ValueError("An unavailable analyzer cannot report detections.")


def intersection_over_union(a: list[float], b: list[float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def summarize(result: AnalyzerResult | None) -> AnalyzerSummary:
    if result is None or result.status == "unavailable":
        return AnalyzerSummary(status="unavailable", version=result.version if result else None,
                               detected_count=None, complete=False)
    return AnalyzerSummary(status=result.status, version=result.version,
                           detected_count=len(result.detections),
                           complete=result.status == "ok" and result.complete)


def possible_duplicates(detections: list[DoorDetection]) -> set[str]:
    duplicates = set()
    for index, item in enumerate(detections):
        for other in detections[index + 1:]:
            if intersection_over_union(item.bbox, other.bbox) >= MATCH_IOU:
                duplicates.update((item.id, other.id))
    return duplicates


def compare(
    deterministic: AnalyzerResult | None, ml: AnalyzerResult | None, source: PageFrame
) -> Comparison:
    for result in (deterministic, ml):
        validate_source(result, source)
    decision = Comparison(status="INSUFFICIENT_EVIDENCE",
                          deterministic=summarize(deterministic), ml=summarize(ml))
    if (deterministic is None or ml is None or
            "unavailable" in (decision.deterministic.status, decision.ml.status)):
        decision.reasons = ["analyzer_unavailable"]
        return decision
    if deterministic.mode != "independent" or ml.mode != "independent":
        decision.reasons = ["non_independent"]
        return decision

    a = sorted(deterministic.detections, key=lambda item: item.id)
    b = sorted(ml.detections, key=lambda item: item.id)
    ambiguous_a = possible_duplicates(a)
    ambiguous_b = possible_duplicates(b)
    a_matches = {item.id: [] for item in a}
    b_matches = {item.id: [] for item in b}
    for left in a:
        for right in b:
            if intersection_over_union(left.bbox, right.bbox) >= MATCH_IOU:
                a_matches[left.id].append(right.id)
                b_matches[right.id].append(left.id)

    decision.matched_pairs = [
        (f"deterministic:{left}", f"ml:{rights[0]}")
        for left, rights in a_matches.items()
        if left not in ambiguous_a and len(rights) == 1 and rights[0] not in ambiguous_b
        and len(b_matches[rights[0]]) == 1
    ]
    matched = {item_id for pair in decision.matched_pairs for item_id in pair}
    for provider, detections in (("deterministic", a), ("ml", b)):
        for item in detections:
            evidence = Evidence(id=f"{provider}:{item.id}", provider=provider, bbox_xyxy=item.bbox)
            if evidence.id in matched:
                decision.matched_evidence.append(evidence)
            else:
                decision.disputed_evidence.append(evidence)
    if decision.disputed_evidence:
        decision.status = "NEEDS_REVIEW"
        if ambiguous_a or ambiguous_b or any(len(matches) > 1 for matches in (*a_matches.values(), *b_matches.values())):
            decision.reasons.append("ambiguous_matches")
        if any(not matches for matches in (*a_matches.values(), *b_matches.values())):
            decision.reasons.append("unmatched_detections")
        return decision

    if not (decision.deterministic.complete and decision.ml.complete):
        decision.reasons = ["coverage_unknown"]
        return decision

    decision.status = "SUPPORTED"
    decision.answer = len(decision.matched_pairs)
    return decision
