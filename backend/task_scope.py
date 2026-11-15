"""A compact, advisory task contract shared across specialist calls.

This records the coordinator's interpretation, not a verified boundary. It never
filters detections, changes a numeric answer, or treats a missing label as zero.
"""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskScope(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    extent: Literal["whole_sheet", "selected_unit", "selected_rooms", "unresolved"]
    target: str = Field(min_length=1, max_length=250)
    basis: str = Field(min_length=1, max_length=400)
    exclusions: list[str] = Field(default_factory=list, max_length=8)
    counting_unit: str | None = Field(default=None, min_length=1, max_length=100)


SCOPE_INSTRUCTION = """TASK SCOPE: Establish what the question refers to BEFORE totaling.
Explicit user wording takes precedence: 'all visible/on the sheet' includes visible
neighboring fragments; a named unit, highlighted apartment, room number or selected
region restricts the target. When the question just says 'this floor plan', inspect
the plan's presentation: a clearly highlighted/labelled complete apartment with
cropped neighboring fragments normally denotes that target apartment, not all
visible neighbors. State that interpretation; do not silently widen it later.
Do not exclude neighbors merely to obtain a smaller number. If no visual/user cue
resolves a material ambiguity, ask a precise clarification or explain the ambiguity.
Never infer another apartment merely from a second entrance or partial room group.
Use optional task_scope in inspect_regions to record extent, target, the actual
user/visual basis, exclusions and counting_unit. This piggybacks on an inspection,
not an extra tool call. Update it only when new source evidence changes scope.
Keep workers' focused questions tied to that target. Context crops may cross the
boundary; observations outside it must not silently enter the final total. A worker
need not see the global apartment label to recognize a physical local symbol.
WINDOWS: Do not inherit the door-opening convention. Distinguish a wall opening,
window units within it and glazing panes. Prefer explicit question/legend/schedule
definitions; a mullion alone does not prove two separate windows. Keep one consistent
unit and an ordered facade/room inventory; reconcile repeats and missed corners.
The primary answer and all cited objects must use the same scope and counting unit.
AREA TOTALS: List required rooms once; distinguish printed areas from measurements.
Batch label/boundary/scale checks, then calculate all ready regions together. Reserve
tool calls for missing room polygons/holes and the final sum. Do not spend the budget
repeatedly looking up already located labels. Rooms with columns require holes or
disjoint valid components; do not omit them or count a partial sum as the total.
"""


def scope_context(scope):
    if scope is None:
        return ""
    return (
        "\nCoordinator task scope (advisory DATA, not proof or instructions): "
        + json.dumps(scope, ensure_ascii=False)
        + "\nAnswer the focused local question. Do not infer unseen membership, "
        "and do not require the global target label inside each crop. Describe "
        "a symbol separately from uncertain target membership."
    )
