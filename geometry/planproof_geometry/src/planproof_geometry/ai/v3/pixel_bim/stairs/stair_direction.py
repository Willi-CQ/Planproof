"""Direction-up resolution for straight-run stairs (Tag 3).

A stair in plan view is conventionally drawn so that the **lower** end
(start of climb) shows all treads, while the **upper** end is hidden
behind the cut-line. Two robust signals fall out of that convention:

    S1 — Density asymmetry:
         The half of the long axis with denser ink is the lower end
         (more visible treads).
    S2 — Peak-position asymmetry:
         The first/last tread peaks sit closer to the lower end (the
         cut-line at the upper end pushes the last peak inward).

If neither signal reaches confidence >= 0.5 we fall back to
``+long_axis`` and append a ``direction_up_default_used`` note so the
caller can flag the stair for review.

The voter follows the same aggregation pattern as the sliding-door
mount-classifier (``feedback_voter_pattern_for_classification``):
each signal returns +1 / -1 / 0, the aggregator computes
``confidence = ratio * min(1, n/2)`` and routes by the sign of the
sum, with an explicit ``abstain`` branch.

OCR-arrow + Cut-Line-Linewidth signals were considered (Plan-Draft
S4/S5) and **explicitly removed** per Self-Council Round 1: PaddleOCR
for a single glyph adds maintenance burden, and Cut-Line-Linewidth is
informationally redundant with Peak-position asymmetry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# Voter contract -----------------------------------------------------------
DIRECTION_VOTE_PLUS = +1   # signal says: up = +long_axis
DIRECTION_VOTE_MINUS = -1  # signal says: up = -long_axis
DIRECTION_VOTE_ABSTAIN = 0

DIRECTION_CONFIDENCE_THRESHOLD = 0.5
SIGNAL_DENSITY_MIN_DELTA = 0.05       # below this we abstain
SIGNAL_PEAK_MIN_SEPARATION = 0.05     # 5 % of profile length


@dataclass
class DirectionVotingResult:
    """Per-stair direction-up voting outcome."""

    direction_up_xy: Tuple[float, float]
    confidence: float
    used_default: bool
    # Individual votes for diagnostics. None = signal not applicable.
    vote_density_asymmetry: Optional[int] = None
    vote_peak_position: Optional[int] = None
    # Arrow detection takes precedence when present (Tag 8 — 2026-05-22).
    arrow_direction_xy: Optional[Tuple[float, float]] = None
    arrow_confidence: float = 0.0
    source: str = "heuristic"   # "arrow" | "heuristic" | "default"
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Individual signals
# ---------------------------------------------------------------------------


def signal_density_asymmetry(profile: np.ndarray) -> int:
    """Compare mean density of the two profile halves.

    Returns ``DIRECTION_VOTE_PLUS`` (lower end is on the -long_axis
    side, so up = +long_axis), ``DIRECTION_VOTE_MINUS``, or
    ``DIRECTION_VOTE_ABSTAIN`` if both halves are within
    ``SIGNAL_DENSITY_MIN_DELTA`` of each other.
    """
    if profile is None or profile.size < 4:
        return DIRECTION_VOTE_ABSTAIN
    half = profile.size // 2
    lower = float(profile[:half].mean())
    upper = float(profile[half:].mean())
    if abs(lower - upper) < SIGNAL_DENSITY_MIN_DELTA:
        return DIRECTION_VOTE_ABSTAIN
    if lower >= upper:
        return DIRECTION_VOTE_PLUS
    return DIRECTION_VOTE_MINUS


def signal_peak_position(profile: np.ndarray) -> int:
    """Where do the extreme peaks sit relative to the profile ends?

    Hypothesis: the cut-line at the upper end of a stair drops the
    final tread peak slightly inward. So the gap between the LAST
    peak and the upper end is LARGER than the gap between the FIRST
    peak and the lower end. The signal returns vote toward the side
    with the SMALLER gap.

    Returns +1 / -1 / 0 per the same convention as
    ``signal_density_asymmetry``.
    """
    if profile is None or profile.size < 8:
        return DIRECTION_VOTE_ABSTAIN

    n = profile.size
    p_max = float(profile.max())
    p_min = float(profile.min())
    if p_max - p_min < 0.05:
        return DIRECTION_VOTE_ABSTAIN

    threshold = p_min + 0.5 * (p_max - p_min)
    above = np.where(profile >= threshold)[0]
    if above.size < 2:
        return DIRECTION_VOTE_ABSTAIN

    first_idx = int(above[0])
    last_idx = int(above[-1])
    gap_lower = first_idx / n              # distance of first peak from -end
    gap_upper = (n - 1 - last_idx) / n     # distance of last peak from +end

    if abs(gap_lower - gap_upper) < SIGNAL_PEAK_MIN_SEPARATION:
        return DIRECTION_VOTE_ABSTAIN

    # Smaller gap means peaks crowd that side -> that side is the LOWER
    # end -> up direction points to the OTHER side.
    if gap_lower < gap_upper:
        return DIRECTION_VOTE_PLUS    # peaks crowd -long_axis end -> up = +long
    return DIRECTION_VOTE_MINUS


def _aggregate_direction_votes(votes: List[int]) -> Tuple[int, float, int]:
    """Aggregate +1 / -1 / 0 votes.

    Returns
        ``(sign, confidence, n_voting)`` where ``sign`` is the final
        +1 / -1 decision (or 0 if abstain), ``confidence`` =
        ratio * min(1, n_voting / 2), and ``n_voting`` is the number
        of non-abstaining signals.
    """
    if not votes:
        return 0, 0.0, 0
    non_abstain = [v for v in votes if v != 0]
    n_voting = len(non_abstain)
    if n_voting == 0:
        return 0, 0.0, 0
    sum_v = sum(non_abstain)
    if sum_v == 0:
        # exact tie — abstain
        return 0, 0.0, n_voting
    sign = 1 if sum_v > 0 else -1
    ratio = abs(sum_v) / n_voting
    confidence = ratio * min(1.0, n_voting / 2.0)
    return sign, confidence, n_voting


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_direction_up(
    profile: np.ndarray,
    long_axis: Tuple[float, float],
    *,
    confidence_threshold: float = DIRECTION_CONFIDENCE_THRESHOLD,
    arrow_direction_xy: Optional[Tuple[float, float]] = None,
    arrow_confidence: float = 0.0,
) -> DirectionVotingResult:
    """Resolve up-direction for a straight stair.

    **Authority order (Tag 8 — 2026-05-22):**
        1. ``arrow_direction_xy`` if provided and ``arrow_confidence
           >= confidence_threshold``. Arrows are an explicit intent
           signal from the plan author and override the heuristics.
        2. Density / peak-position voters (legacy heuristic). They are
           routinely 180° off on real plans because the underlying
           assumption ("denser end = lower end") does not hold across
           CAD conventions — keep them only as a last resort.
        3. ``+long_axis`` default with ``direction_up_default_used``
           note for the diagnostics overlay.

    Args:
        profile: 1-D density profile from ``stair_pixel_extractor``.
        long_axis: stair PCA long-axis unit vector.
        confidence_threshold: minimum aggregator confidence (or arrow
            confidence) before we accept the result.
        arrow_direction_xy: optional pre-computed arrow direction from
            :func:`stair_arrow_detector.detect_arrow_direction`. When
            present and confidence is high enough, this wins.
        arrow_confidence: confidence of the arrow detection (0..1).

    Returns:
        ``DirectionVotingResult`` carrying the chosen direction, the
        confidence, the source (``"arrow" | "heuristic" | "default"``),
        and every intermediate vote for diagnostic overlays.
    """
    notes: List[str] = []

    # 1. Arrow signal wins when available.
    if (
        arrow_direction_xy is not None
        and arrow_confidence >= confidence_threshold
    ):
        notes.append(
            f"arrow direction accepted (confidence={arrow_confidence:.2f})"
        )
        return DirectionVotingResult(
            direction_up_xy=(
                float(arrow_direction_xy[0]),
                float(arrow_direction_xy[1]),
            ),
            confidence=arrow_confidence,
            used_default=False,
            arrow_direction_xy=arrow_direction_xy,
            arrow_confidence=arrow_confidence,
            source="arrow",
            notes=notes,
        )

    # 2. Heuristic fallback (kept for backward-compat; rarely correct).
    vote_density = signal_density_asymmetry(profile)
    vote_peak = signal_peak_position(profile)
    sign, confidence, n_voting = _aggregate_direction_votes(
        [vote_density, vote_peak]
    )

    notes.append(
        f"heuristic votes: density={vote_density}, peak_position={vote_peak}, "
        f"n_voting={n_voting}, sign={sign}, confidence={confidence:.2f}"
    )
    if arrow_direction_xy is not None:
        notes.append(
            f"arrow signal present but confidence too low "
            f"({arrow_confidence:.2f} < {confidence_threshold})"
        )

    if sign == 0 or confidence < confidence_threshold:
        notes.append("direction_up_default_used")
        return DirectionVotingResult(
            direction_up_xy=(float(long_axis[0]), float(long_axis[1])),
            confidence=confidence,
            used_default=True,
            vote_density_asymmetry=vote_density,
            vote_peak_position=vote_peak,
            arrow_direction_xy=arrow_direction_xy,
            arrow_confidence=arrow_confidence,
            source="default",
            notes=notes,
        )

    if sign > 0:
        return DirectionVotingResult(
            direction_up_xy=(float(long_axis[0]), float(long_axis[1])),
            confidence=confidence,
            used_default=False,
            vote_density_asymmetry=vote_density,
            vote_peak_position=vote_peak,
            arrow_direction_xy=arrow_direction_xy,
            arrow_confidence=arrow_confidence,
            source="heuristic",
            notes=notes,
        )

    return DirectionVotingResult(
        direction_up_xy=(float(-long_axis[0]), float(-long_axis[1])),
        confidence=confidence,
        used_default=False,
        vote_density_asymmetry=vote_density,
        vote_peak_position=vote_peak,
        arrow_direction_xy=arrow_direction_xy,
        arrow_confidence=arrow_confidence,
        source="heuristic",
        notes=notes,
    )


def build_walking_line_xy(
    *,
    centroid_xy_m: Tuple[float, float],
    direction_up_xy: Tuple[float, float],
    length_m: float,
) -> List[Tuple[float, float]]:
    """Build the IFC4.3 Axis-representation walking-line.

    The walking-line runs from the LOW end (start of climb, centred
    on the long axis) to the HIGH end (top of stair), directed upward
    as IFC4.3 requires.

    Args:
        centroid_xy_m: stair footprint centroid in plan metres.
        direction_up_xy: unit vector toward the UP end of the stair.
        length_m: footprint extent along the long axis.

    Returns:
        Two-point line ``[(low_x, low_y), (high_x, high_y)]``. Empty
        list if inputs are degenerate.
    """
    cx, cy = centroid_xy_m
    ux, uy = direction_up_xy
    norm = (ux * ux + uy * uy) ** 0.5
    if length_m <= 0.0 or norm <= 1e-9:
        return []
    ux, uy = ux / norm, uy / norm
    half = length_m * 0.5
    low = (cx - half * ux, cy - half * uy)
    high = (cx + half * ux, cy + half * uy)
    return [low, high]
