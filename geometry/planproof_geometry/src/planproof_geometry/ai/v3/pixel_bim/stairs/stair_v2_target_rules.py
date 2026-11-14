"""Stair target classification — does the flight reach the next storey?

A flight does NOT always end at the next storey. It can also lead to a
terrace, a roof, a half-landing, or a few entrance steps. Misclassifying
would either flatten a real floor-to-floor stair (loses height) or push
a terrace stair into the ceiling (intersects the slab above). The rules
below decide the ENDPOINT per flight; START is always z=0 of the local
storey.

  R1  PAIRED:        junction-step detected with another flight
                     -> next floor (both legs share floor-to-floor)
  R2  LONG SOLO:     n_steps * DIN_RISE >= FLOOR_SNAP_RATIO * floor_h
                     -> next floor (snap top exactly to floor_h)
  R3  SHORT SOLO:    fewer steps than R2 requires
                     -> intermediate level at n_steps * DIN_RISE
  R4  ROOM-LABEL:    [TODO] adjacent room labeled
                     Terrasse / Balkon / Loggia / Dachterrasse
                     -> outdoor intermediate, override R2 even when long
  R5  EXTERIOR TOP:  [TODO] end-tread crosses an exterior wall outward
                     -> outdoor terrace, top = n*DIN_RISE
  R6  ROOF ACCESS:   [TODO] top storey AND end-tread inside a
                     stair_well-roof region -> roof access (top = n*DIN_RISE)
  R7  MULTI-FLOOR:   [TODO] stair on storey k continues on storey k+1
                     with aligned footprint -> chain heights across storeys

R1-R3 are active. R4-R7 are stubbed for follow-up sprints; the
classifier accepts optional signal flags so the caller can wire them up
without changing this module's interface.

ALGORITHM LOCKED 2026-05-28 — frozen baseline lives at
``c:/tmp/rebuild_bim_v2_LOCKED.py``.
"""
from __future__ import annotations

from typing import Tuple

DIN_RISE_M = 0.18           # DIN 18065 comfort-step rise
FLOOR_SNAP_RATIO = 0.85     # n*DIN_RISE >= ratio*floor_h => solo->next_floor

TargetLabel = str
TargetTop = float


def classify_stair_target(
    n_steps: int,
    *,
    is_paired: bool,
    floor_height_m: float,
    din_rise_m: float = DIN_RISE_M,
    snap_ratio: float = FLOOR_SNAP_RATIO,
    # R4-R7 hooks — accepted but not yet load-bearing.
    near_room_label: str = "",
    end_outside_exterior_wall: bool = False,
    on_top_storey: bool = False,
    end_in_roof_well: bool = False,
    chained_storey_above: bool = False,
) -> Tuple[TargetLabel, TargetTop]:
    """Decide the climb endpoint for a single flight.

    Returns:
        ``(target_label, top_m)`` where ``target_label`` is one of
        ``"next_floor_pair"``, ``"next_floor_solo"``, ``"intermediate"``
        (active R1-R3) and ``top_m`` is the absolute end-tread elevation
        relative to the local storey floor.
    """
    n = max(int(n_steps), 0)
    if is_paired:
        return "next_floor_pair", float(floor_height_m)            # R1
    if n * din_rise_m >= snap_ratio * floor_height_m:
        return "next_floor_solo", float(floor_height_m)            # R2
    # R4..R7 hooks (currently inactive — left in place so the caller can
    # wire signal flags without changing this module's interface).
    _ = near_room_label
    _ = end_outside_exterior_wall
    _ = on_top_storey
    _ = end_in_roof_well
    _ = chained_storey_above
    return "intermediate", float(n * din_rise_m)                   # R3
