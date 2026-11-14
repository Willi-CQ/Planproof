"""Guardrails for the label refinement cascade.

The philosophy: the user's label is truth. The refinement is only allowed
to *sharpen* the boundary, not move it. If the refiner produces a mask
that is too small, too big, or too far off from the user's label, we
reject it and fall back to the raw user polygon.

Thresholds are intentionally generous on the inside (allow substantial
shrinking toward the true element) and strict on the outside (don't let
the refiner wander past the label).
"""
from __future__ import annotations

from dataclasses import dataclass

from .classical import RefinedMask


# Acceptance thresholds. Tuned conservatively for Matterport; revisit
# once we have IoU data per plan type.

# Refined mask must cover at least this fraction of the user polygon.
# Paired-outline walls sometimes shrink to < 0.5 after classical, which
# is still OK if IoU is reasonable.
MIN_AREA_RATIO = 0.30

# Refined mask may not exceed this fraction of the user polygon. Since
# the classical refiner clips to the polygon interior, this is usually
# automatic; the check guards against future refiners (SAM) that could
# bleed outside.
MAX_AREA_RATIO = 1.05

# IoU between refined and user polygon below this → likely the refiner
# found the wrong element inside the bbox. Fall back.
MIN_IOU = 0.25


@dataclass
class MaskDecision:
    """Outcome of guardrail evaluation."""
    accept: bool
    reason: str


def decide_fallback(refined: RefinedMask) -> MaskDecision:
    """Check whether the refined mask should be used or rejected.

    A rejected mask will be replaced with the raw user polygon by the
    caller; the `source` field on `refined` will be overwritten to
    reflect the fallback reason.
    """
    # Refiner already self-aborted (e.g. empty crop, no ink).
    if refined.source.startswith("fallback_"):
        return MaskDecision(False, refined.source)

    if refined.area_ratio_vs_user_polygon < MIN_AREA_RATIO:
        return MaskDecision(
            False,
            f"too_small (ratio={refined.area_ratio_vs_user_polygon:.2f} "
            f"< {MIN_AREA_RATIO:.2f})",
        )

    if refined.area_ratio_vs_user_polygon > MAX_AREA_RATIO:
        return MaskDecision(
            False,
            f"too_large (ratio={refined.area_ratio_vs_user_polygon:.2f} "
            f"> {MAX_AREA_RATIO:.2f})",
        )

    if refined.iou_vs_user_polygon < MIN_IOU:
        return MaskDecision(
            False,
            f"iou_too_low ({refined.iou_vs_user_polygon:.2f} "
            f"< {MIN_IOU:.2f})",
        )

    return MaskDecision(True, "ok")
