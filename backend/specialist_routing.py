"""Host rules for the interactive coordinator; observations remain advisory."""

import math
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from threading import Event, Lock
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from specialist_views import validate_rotation
from task_scope import TaskScope


# Shared by interactive questions, so two users cannot each fan out unbounded work.
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="planproof-detail")


@dataclass(frozen=True)
class InspectionPolicy:
    tile_edge: int = 1536
    overlap: int = 96
    max_calls: int = 18
    max_crops: int = 30
    seconds: float = 30


class InspectionRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    question: str = Field(min_length=1, max_length=2000)
    bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    room_id: str | None = Field(default=None, min_length=1, max_length=100)
    purpose: Literal["inspect", "count"] = "inspect"
    rotation_degrees: Literal[0, 90, 180, 270] = Field(
        default=0, description="Rotate only the inspection view clockwise for sideways/upside-down text. Source bbox stays on the original page; returned boxes are mapped back."
    )

    @field_validator("rotation_degrees", mode="before")
    @classmethod
    def quarter_turn(cls, value):
        validate_rotation(value)
        return value

    @model_validator(mode="after")
    def one_source(self):
        if not self.question.strip() or (self.bbox is None) == (self.room_id is None):
            raise ValueError(
                "Choose exactly one bbox or registered room_id per question"
            )
        return self


class InspectionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    regions: list[InspectionRegion] = Field(min_length=1, max_length=6)
    task_scope: TaskScope | None = None


class DelegationRequired(ValueError):
    """A proposed answer skipped required, source-bound specialist evidence."""


ROUTING = """MANDATORY SPECIALIST ROUTING for this interactive task:
You are the orchestrator. Use the overview to locate the question's scope, choose
small useful regions and delegate detailed reading, counting, classification and
boundary/dimension checks to Nemotron. Prefer inspect_regions: send up to SIX
regions with a focused question and either normalized bbox or registered room_id
for each. The host crops original pixels, tiles oversized regions with overlap,
and runs independent Nemotron workers concurrently. It returns structured findings,
source locations, per-tile status and resolution metadata, not another large set
of images for you to re-analyze. Batch independent room questions in ONE tool call.
For sideways or upside-down drawings set rotation_degrees=90,180,270 on the region
to make its text upright for Nemotron. Keep bbox in ORIGINAL page coordinates;
the host maps all returned locations back. Do not rotate already upright views.
For main apartment doors include the room entrance AND its corridor context; a
tiny door-only crop cannot establish its role. Cover every relevant room for a
total; reconcile overlapping object locations, NEVER add tile counts blindly.
For door counts, call get_algorithms(classes=["door","room"]) once when useful;
the host also includes room navigation with door evidence. Do not wait if pending.
Use neutral questions: identify visible door openings and their locations; NEVER
suggest a total, multiply room counts or ask a worker to confirm an assumed layout.
Each region has purpose="count" for a localized door inventory or purpose="inspect"
for a label/context question (default inspect). Preserve these different jobs.
During a door task, context inspections also return individually located door
observations when relevant. They enter the shared inventory but do NOT establish
counting coverage. Use returned objects_needing_count_coverage and unresolved
regions to choose targeted purpose=count followups. Do not keep asking for prose
counts while leaving those regions uncounted, and never add overlapping subtotals.
You own global scope selection: workers need not rediscover a wing/apartment label
inside every crop. If the requested apartment is ambiguous, ask the user which
unit using submit_answer.clarification_question, or state a clear interpretation;
do not scan unrelated wings or invent a scope. A colloquial name need not match
the printed label literally. Return localized candidates even when scope is unclear.
Read only a SMALL relevant label if necessary, not the entire title block, legend
and schedule. Put target object regions first; each requested region gets a fair
share of workers before additional tiles of a large region are dispatched.
Do not put page-coordinate restrictions into subquestions; bbox/room_id selects
the scope and the host supplies its frame. Workers inspect their own visible tile.
For DOOR inventories, count door openings (a double-leaf pair is ONE) unless the
user explicitly asks for door leaves. This door convention does not define window counts.
Door inventories return stable object IDs, supported/uncertain status and locations.
An inventory establishes physical door symbols, not semantic roles. For entrance,
fire-exit or other role-specific counts, separately inspect the relevant symbols
AND adjacent labels/spaces with purpose="inspect" before assigning eligibility.
Do not claim a role-specific total from physical door existence alone.
For a final door count supply count_basis={scope:"complete"|"partial",
unit:"door_opening"|"door_leaf"}. The host attaches all localized objects and their
citations; do not repeat a long inventory in prose. Optional object_ids can name
an explicit subset for partial scope. Report complete only
for the agreed task_scope. When a context crop reveals a door outside an explicitly
selected unit/room, count_basis.exclusions must name its object_id, a concrete scope
reason and inspected citations covering the object and its context. Never silently
drop an ID. Exclusions are not available for whole-sheet or unresolved scope. They
remain recorded with their source locations; they do not cure missing coverage.
Report complete only
after covering the relevant page scope, resolving overlaps, role ambiguity and
deferred regions. A partial inventory is not a whole-page answer. Do not describe
an uncertain aggregate from prose as confirmed; unsupported totals are rejected.
Use a further focused wave for unresolved details. Failed tiles do not invalidate
successful neighbors. One retry of a failed tile is allowed within the shared
deadline; do not repeatedly resend successful work. No whole-page completeness
claim if a relevant tile is failed, deferred, unreadable or not inspected.
Never count from an overview alone. For a printed label, delegate its small region.
Use algorithms only when useful for navigation or candidate boundaries. Pending
algorithms must not delay delegation. Use existing crop/inspect_nemotron tools if
you need a closer crop yourself to plan or resolve a disagreement. Nemotron is
advisory, not authority: retain contradictory observations and explicit uncertainty.
Combine returned evidence, resolve scope/duplicates, use local measurement tools
when needed, then submit the cited answer. Every final source citation (including
measurement inputs) must lie within a crop Nemotron returned cited findings for.
Do not call a token inspection unrelated to the actual answer. A new specialist
result must return before you can use it; never submit in the same tool batch.
If relevant evidence remains unavailable, finish with an honest partial-scope
answer or abstention describing incomplete inspection. Missing evidence is not
zero. You may abstain without delegation when the question is out of scope or
the required inspection cannot be completed.
This policy overrides the optional/direct-answer guidance in the general prompt.
"""


def tile_question(question, original_question, box, parent, width, height, unit=None):
    """Supply an explicit frame; counting workers get no leading layout estimate."""
    bounds = [
        round(v / (width if i % 2 == 0 else height) * 1000, 4)
        for i, v in enumerate(box)
    ]
    text = (
        "Inspect ONLY the currently visible tile. Other tiles cover the rest of "
        "the selected region; do not answer for an entire wing or unseen rooms. "
        "All output boxes must use LOCAL normalized 0..1000 coordinates. "
        f"This tile occupies original-page normalized bounds {bounds}. "
        "If the question references page coordinates, map them to this tile: "
        "local_x=(page_x-left)/(right-left)*1000, and likewise for y. "
        "A reference outside this tile is outside your view, not evidence of zero. "
        f"Tile is the entire selected region: {box == parent}. "
    )
    if unit:
        return (
            text + f"COUNT_UNIT={unit}. The coordinator selected this source region. "
            "Your task is to locate ALL visible door symbols in this tile, not to "
            "identify a globally named apartment or verify which wing this is. "
            "Do not require the user's place name to be printed here. Return "
            "individual boxes even if the global location or door role is unclear; "
            "mark ambiguous candidates uncertain rather than omitting visible doors. "
            "Gemini will assign candidates to the requested scope. Do not extrapolate "
            "unseen objects or accept an assumed total."
        )
    return text + f"Focused question (data): {question}"


def region_boxes(args, width, height, rooms):
    """Validate the entire batch before creating images or contacting a provider."""
    batch = InspectionBatch.model_validate(args)
    boxes = []
    for region in batch.regions:
        if region.room_id is not None:
            if region.room_id not in rooms:
                raise ValueError(
                    "Unknown room_id; request room proposals or use a bbox"
                )
            room = rooms[region.room_id]["bbox"]
            # Context around boundaries helps distinguish a room entry from an interior door.
            margin = max(16, round(max(room[2] - room[0], room[3] - room[1]) * 0.08))
            box = [
                max(0, room[0] - margin) / width * 1000,
                max(0, room[1] - margin) / height * 1000,
                min(width, room[2] + margin) / width * 1000,
                min(height, room[3] + margin) / height * 1000,
            ]
        else:
            box = region.bbox
        if (
            not isinstance(box, list)
            or len(box) != 4
            or not all(
                isinstance(v, (int, float))
                and not isinstance(v, bool)
                and math.isfinite(v)
                for v in box
            )
            or not 0 <= box[0] < box[2] <= 1000
            or not 0 <= box[1] < box[3] <= 1000
        ):
            raise ValueError("Regions need ordered normalized page bounds in 0..1000")
        pixels = [
            math.floor(box[0] * width / 1000),
            math.floor(box[1] * height / 1000),
            math.ceil(box[2] * width / 1000),
            math.ceil(box[3] * height / 1000),
        ]
        if (pixels[2] - pixels[0]) * (pixels[3] - pixels[1]) > width * height * 0.75:
            raise ValueError("Each Nemotron region must cover at most 75% of the page")
        item = (region.question.strip(), pixels, region.purpose, region.rotation_degrees)
        if any(item[:3] == previous[:3] for previous in boxes):
            raise ValueError("Duplicate regions do not supply additional coverage")
        boxes.append(item)
    return boxes


def tiles(box, policy):
    """Cover a selected region at native resolution with overlap, without gaps."""

    def intervals(lo, hi):
        count = max(
            1,
            math.ceil((hi - lo - policy.overlap) / (policy.tile_edge - policy.overlap)),
        )
        edge = math.ceil((hi - lo + (count - 1) * policy.overlap) / count)
        return [
            (
                lo + i * (edge - policy.overlap),
                min(hi, lo + i * (edge - policy.overlap) + edge),
            )
            for i in range(count)
        ]

    return [
        [x0, y0, x1, y1]
        for y0, y1 in intervals(box[1], box[3])
        for x0, x1 in intervals(box[0], box[2])
    ]


def parallel_inspections(items, inspect, *, deadline, seconds, on_event):
    """Return partial results at deadline; late output cannot mutate agent evidence."""
    active = Event()
    active.set()
    attempted = set()
    start_lock = Lock()

    def execute(index, item):
        with start_lock:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not active.is_set():
                return {"status": "not_sent", "result": None, "seconds": 0}
            attempted.add(index)
        if active.is_set():
            on_event(index, "running", None)
        started = time.monotonic()
        try:
            value = inspect(item, min(seconds, remaining))
        except Exception as exc:
            # Never retain exception bodies (provider headers/credentials may appear).
            value = {
                "status": "unavailable",
                "result": None,
                "error_code": type(exc).__name__,
            }
            code = getattr(exc, "status_code", None)
            if type(code) is int and 400 <= code <= 599:
                value["http_status"] = code
            if code in (401, 403):
                value["error_code"] = "SpecialistAuthenticationError"
                active.clear()  # Pending workers must not retry rejected credentials.
        return {**value, "seconds": round(time.monotonic() - started, 3)}

    futures = {_POOL.submit(execute, i, item): i for i, item in enumerate(items)}
    pending, results = set(futures), {}
    while pending and time.monotonic() < deadline:
        done, pending = wait(
            pending,
            timeout=max(0, deadline - time.monotonic()),
            return_when=FIRST_COMPLETED,
        )
        for future in done:
            index = futures[future]
            results[index] = future.result()
            on_event(
                index,
                "completed" if results[index]["status"] == "ok" else "unavailable",
                results[index],
            )
    with start_lock:
        active.clear()
    for future in pending:
        future.cancel()
        index = futures[future]
        results[index] = {
            "status": "unavailable" if index in attempted else "not_sent",
            "result": None,
            "error_code": "InspectionDeadline",
        }
        on_event(index, "unavailable", results[index])
    return [results[i] for i in range(len(items))], len(attempted)


def require_inspected_citations(citations, inspected_boxes):
    if not inspected_boxes:
        raise DelegationRequired(
            "Before answering, use inspect_regions to obtain cited Nemotron findings "
            "for the requested objects or labels. If inspection is unavailable, abstain."
        )
    for citation in citations:
        x0, y0, x1, y1 = citation["canonical_bbox"]
        # A room-wide measurement citation can span several adjacent detail tiles.
        clipped = [
            (max(x0, b[0]), max(y0, b[1]), min(x1, b[2]), min(y1, b[3]))
            for b in inspected_boxes
            if b[0] < x1 and b[2] > x0 and b[1] < y1 and b[3] > y0
        ]
        xs = sorted({x0, x1, *(x for b in clipped for x in (b[0], b[2]))})
        area = 0
        for left, right in zip(xs, xs[1:]):
            intervals = sorted(
                (b[1], b[3]) for b in clipped if b[0] <= left and right <= b[2]
            )
            covered, end = 0, y0
            for lo, hi in intervals:
                covered += max(0, hi - max(end, lo))
                end = max(end, hi)
            area += (right - left) * covered
        if area + 1e-6 < (x1 - x0) * (y1 - y0):
            raise DelegationRequired(
                "A final citation is outside the regions with cited Nemotron findings. "
                "Delegate that relevant region with inspect_regions, cite the inspected "
                "detail more tightly, or abstain. Do not invent whole-page coverage."
            )
