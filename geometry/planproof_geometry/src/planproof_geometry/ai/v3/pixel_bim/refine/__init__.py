"""Phase 1 — Label refinement (Gravity Wells).

Pipeline: for each user label (polygon or bbox), produce a pixel-perfect
binary mask that tightly snaps to the underlying plan element.

Refinement cascade:
  1. classical_refiner   — crop + binarize + connected-components + contour.
                           Zero ML, deterministic, handles 80%+ of cases.
  2. chan_vese_refiner   — Chan-Vese level-set when classical is ambiguous.
                           Still zero ML.
  3. sam_refiner         — SAM-2 SAMRefiner pattern. Gated by a config flag
                           since it needs torch + the model on disk.

guardrails decide per label which path to run and when to fall back to the
raw user polygon (no change).
"""

from .classical import refine_label_classical, RefinedMask  # noqa: F401
from .guardrails import MaskDecision, decide_fallback  # noqa: F401
