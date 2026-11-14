"""DIN 18065:2020-08 reference values and soft-flag emitter for stairs.

This module is the SINGLE SOURCE OF TRUTH for stair-related DIN values
in the Plan-to-BIM pipeline. Values are corrected against the official
2020-08 edition — in particular the Schrittmaszregel band is
**590 - 650 mm** (not the previously assumed 630 +/- 20 mm).

Background:
    * DIN 18065:2020-08 "Gebaeudetreppen — Begriffe, Messregeln,
      Hauptmasze".
    * The Schrittmaszregel is the centrepiece of stair planning:
      2 * rise (s) + going (a) = 590 to 650 mm.
    * Dimension bands distinguish residential buildings with up to two
      dwelling units (WE <= 2) from "buildings in general":
        - WE<=2:   rise 140-200 mm, going 230-370 mm.
        - general: rise 140-190 mm, going 260-370 mm.
    * Step-to-step tolerance: actual rise/going must not deviate more
      than +/- 5 mm between consecutive treads. We model this as a
      soft-flag because a derived stair (rise = total_height / n_steps)
      is perfectly regular by construction; the flag triggers only
      when a per-tread extractor delivers non-uniform measurements.
    * Width: nutzbare Treppenlauf-Breite >= 800 mm (Wohngebaeude, WE<=2),
      >= 1000 mm fuer notwendige Treppe in allgemeinen Bauten.

The values here are SOFT constraints. The algorithm always emits a
result; deviations are appended as ``din_deviations`` records (severity
``warn``) on the StairRecord. Hard filtering is delegated to QualityPage
BIM-Audit (Sprint #6), where the user decides whether a deviation is
acceptable.

All limits are exposed as module-level constants so they can be
mirrored into ``standards/din/loader.py::DinNormSet`` in a follow-up
sprint without code-duplication risk.
"""
from __future__ import annotations

from typing import Any, Dict, List

# --- Schrittmaszregel band (DIN 18065:2020-08) -----------------------------
SCHRITTMASS_MIN_M: float = 0.590
SCHRITTMASS_MAX_M: float = 0.650
SCHRITTMASS_OPTIMUM_M: float = 0.630  # comfort target (17/29 yields 0.63)

# --- Dimension bands — residential (Wohngebaeude WE <= 2) ------------------
RISE_MIN_RESIDENTIAL_M: float = 0.140
RISE_MAX_RESIDENTIAL_M: float = 0.200
GOING_MIN_RESIDENTIAL_M: float = 0.230
GOING_MAX_RESIDENTIAL_M: float = 0.370

# --- Dimension bands — general buildings -----------------------------------
RISE_MIN_GENERAL_M: float = 0.140
RISE_MAX_GENERAL_M: float = 0.190
GOING_MIN_GENERAL_M: float = 0.260
GOING_MAX_GENERAL_M: float = 0.370

# --- Width ----------------------------------------------------------------
WIDTH_MIN_RESIDENTIAL_M: float = 0.800   # nutzbare Laufbreite
WIDTH_MIN_GENERAL_M: float = 1.000       # notwendige Treppe

# --- Step-to-step tolerance (regularity) ----------------------------------
STEP_TO_STEP_TOLERANCE_M: float = 0.005  # +/- 5 mm

# --- Comfort defaults (used when reverse-engineering a stair without
# --- explicit rise/going) -------------------------------------------------
DEFAULT_TARGET_GOING_M: float = 0.290    # 17/29 optimum -> 290 mm going
DEFAULT_TARGET_RISE_M: float = 0.170     # 17/29 optimum -> 170 mm rise


def derive_step_layout(
    length_m: float,
    total_height_m: float,
    *,
    going_target_m: float = DEFAULT_TARGET_GOING_M,
) -> Dict[str, Any]:
    """Compute (n_steps, rise_m, going_m, schrittmass_m) from length+height.

    Args:
        length_m: Footprint extent along the long axis (m).
        total_height_m: Floor-to-floor height the stair spans (m).
        going_target_m: Desired going for step-count estimation
            (default 29 cm, DIN comfort target).

    Returns:
        Dict with ``n_steps`` (>= 1), ``rise_m``, ``going_m``,
        ``schrittmass_m``. If inputs are degenerate (any <= 0),
        returns ``n_steps=1`` with zero-or-input values; the caller
        decides how to handle the degenerate case.
    """
    if length_m <= 0.0 or total_height_m <= 0.0 or going_target_m <= 0.0:
        return {
            "n_steps": 1,
            "rise_m": max(0.0, total_height_m),
            "going_m": max(0.0, length_m),
            "schrittmass_m": 2.0 * max(0.0, total_height_m) + max(0.0, length_m),
        }
    n_steps = max(1, int(round(length_m / going_target_m)))
    rise_m = total_height_m / n_steps
    going_m = length_m / n_steps
    return {
        "n_steps": n_steps,
        "rise_m": rise_m,
        "going_m": going_m,
        "schrittmass_m": 2.0 * rise_m + going_m,
    }


def validate_din(
    *,
    rise_m: float,
    going_m: float,
    width_m: float = 0.0,
    building_class: str = "residential",
) -> List[Dict[str, Any]]:
    """Return DIN 18065 soft-flag entries for a derived step layout.

    Each entry shape::

        {
            "code": "<short_id>",
            "kind": "rise_oor|going_oor|schrittmass_oor|width_below_min",
            "norm": "DIN 18065:2020-08",
            "metric": "<rise|going|schrittmass|width>",
            "value_m": <actual>,
            "expected_m": <range or value>,
            "severity": "warn",
            "message": "<human-readable de>",
        }

    Empty list = fully DIN-conform.

    Args:
        rise_m: Step rise (m).
        going_m: Step going / tread depth (m).
        width_m: Usable stair-flight width (m). Pass 0.0 to skip the
            width check.
        building_class: ``"residential"`` (Wohngebaeude WE<=2) or
            ``"general"`` (Gebaeude allgemein, notwendige Treppe).
    """
    devs: List[Dict[str, Any]] = []

    if building_class == "general":
        rise_min, rise_max = RISE_MIN_GENERAL_M, RISE_MAX_GENERAL_M
        going_min, going_max = GOING_MIN_GENERAL_M, GOING_MAX_GENERAL_M
        width_min = WIDTH_MIN_GENERAL_M
    else:
        rise_min, rise_max = RISE_MIN_RESIDENTIAL_M, RISE_MAX_RESIDENTIAL_M
        going_min, going_max = GOING_MIN_RESIDENTIAL_M, GOING_MAX_RESIDENTIAL_M
        width_min = WIDTH_MIN_RESIDENTIAL_M

    if not (rise_min <= rise_m <= rise_max):
        devs.append({
            "code": "rise_out_of_range",
            "kind": "rise_oor",
            "norm": "DIN 18065:2020-08",
            "metric": "rise",
            "value_m": round(float(rise_m), 4),
            "expected_m": [rise_min, rise_max],
            "severity": "warn",
            "message": (
                f"Steigung {rise_m * 100:.1f} cm außerhalb DIN-Bereich "
                f"{rise_min * 100:.0f}-{rise_max * 100:.0f} cm "
                f"({building_class})."
            ),
        })

    if not (going_min <= going_m <= going_max):
        devs.append({
            "code": "going_out_of_range",
            "kind": "going_oor",
            "norm": "DIN 18065:2020-08",
            "metric": "going",
            "value_m": round(float(going_m), 4),
            "expected_m": [going_min, going_max],
            "severity": "warn",
            "message": (
                f"Auftritt {going_m * 100:.1f} cm außerhalb DIN-Bereich "
                f"{going_min * 100:.0f}-{going_max * 100:.0f} cm "
                f"({building_class})."
            ),
        })

    schrittmass = 2.0 * rise_m + going_m
    if not (SCHRITTMASS_MIN_M <= schrittmass <= SCHRITTMASS_MAX_M):
        devs.append({
            "code": "schrittmass_out_of_range",
            "kind": "schrittmass_oor",
            "norm": "DIN 18065:2020-08",
            "metric": "schrittmass",
            "value_m": round(float(schrittmass), 4),
            "expected_m": [SCHRITTMASS_MIN_M, SCHRITTMASS_MAX_M],
            "severity": "warn",
            "message": (
                f"Schrittmasz 2*Steigung+Auftritt = {schrittmass * 100:.1f} "
                f"cm außerhalb DIN-Band {SCHRITTMASS_MIN_M * 100:.0f}-"
                f"{SCHRITTMASS_MAX_M * 100:.0f} cm."
            ),
        })

    if width_m > 0.0 and width_m < width_min:
        devs.append({
            "code": "width_below_minimum",
            "kind": "width_below_min",
            "norm": "DIN 18065:2020-08",
            "metric": "width",
            "value_m": round(float(width_m), 4),
            "expected_m": width_min,
            "severity": "warn",
            "message": (
                f"Laufbreite {width_m * 100:.1f} cm unter DIN-Minimum "
                f"{width_min * 100:.0f} cm ({building_class})."
            ),
        })

    return devs


def resolve_total_height_m(
    *,
    annotation_meta: Dict[str, Any] | None,
    config_value_m: float | None,
    pipeline_default_m: float,
) -> Dict[str, Any]:
    """Resolve floor-to-floor height for one stair label.

    Priority order (1 = highest):
        1. ``annotation_meta["total_height_m"]`` if positive — per-stair
           user override.
        2. ``config_value_m`` if positive — per-job pipeline config
           (``stair_floor_to_floor_m``).
        3. ``pipeline_default_m`` — final fallback
           (typically WALL_HEIGHT_M + ceiling_buildup_m, e.g. 2.95 m).

    Returns:
        Dict with ``height_m`` (float) and ``height_source`` (str:
        ``"annotation_meta"``, ``"config"``, or ``"pipeline_default"``).
    """
    if annotation_meta:
        raw = annotation_meta.get("total_height_m")
        if raw is not None:
            try:
                v = float(raw)
                if v > 0.0:
                    return {"height_m": v, "height_source": "annotation_meta"}
            except (TypeError, ValueError):
                pass

    if config_value_m is not None and config_value_m > 0.0:
        return {"height_m": float(config_value_m), "height_source": "config"}

    return {
        "height_m": float(pipeline_default_m),
        "height_source": "pipeline_default",
    }
