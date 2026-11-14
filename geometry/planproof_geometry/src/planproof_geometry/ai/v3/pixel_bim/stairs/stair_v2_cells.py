"""V2 cell-based step extractor — promoted from the locked prototype.

Heatmap-attributed outline + cells between continuous ink lines:
the user bbox is dilated 0.5 m as a label gravity well, dark ink inside
plus a confidence floor (cf >= 0.10) gives the stair body, then RAW dark
ink dilated by D = 2 cm carves the body into ONE cell per real tread.
Each cell is regrown by D + stroke so adjacent treads meet exactly on
the shared division line (gap-free, no synthetic risers).

A perpendicular sliver filter (perp-extent >= 50% of the run median)
drops cells that leaked across wall gaps below the stair body.

ALGORITHM LOCKED 2026-05-28 — frozen baseline lives at
``c:/tmp/rebuild_bim_v2_LOCKED.py``. Do not modify the body of
:func:`extract_steps_v2` without re-validating on Matterport Sample
(oyy4=4, pd2n=7, pm8f=6, pzya=3 cells + junction step pd2n+pm8f).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import cv2
import numpy as np


def _cmpx(m: float, px_to_m: float) -> int:
    """Metre -> pixel, clamped to at least 1 px."""
    return max(1, int(round(m / max(px_to_m, 1e-9))))


def extract_steps_v2(
    img: np.ndarray,
    poly_px: np.ndarray,
    px_to_m: float,
    cf: np.ndarray,
    arrow_dir: Optional[Sequence[float]] = None,
) -> List[np.ndarray]:
    """Extract per-step cell polygons from a single stair bbox.

    Args:
        img: grayscale image (HxW uint8), 0 = ink, 255 = paper.
        poly_px: user bbox as int32 ``(N, 2)`` pixel polygon.
        px_to_m: metres per image pixel.
        cf: confidence field, ``float32`` same shape as ``img``, in
            ``[0, 1]``. Comes from
            :func:`stair_confidence_field.build_confidence_field`.
        arrow_dir: unit climb direction in pixel coords (``(dx, dy)``),
            or ``None`` to skip the sliver filter.

    Returns:
        List of int32 ``(M, 2)`` cell contours in pixel coords. Empty
        list when the bbox holds no detectable treads.
    """
    if img is None or img.size == 0 or len(poly_px) < 3:
        return []
    H, W = img.shape[:2]
    labelmask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(labelmask, [poly_px.astype(np.int32)], 255)
    roi = cv2.dilate(
        labelmask,
        cv2.getStructuringElement(
            cv2.MORPH_RECT, (_cmpx(0.5, px_to_m) * 2 + 1,) * 2
        ),
    )
    thr = 140
    # Low confidence floor so a step shared ~50/50 with an adjacent label
    # (its far half sits in the neighbour bbox at cf~0.1-0.2) is still
    # attributed here -> the split step is recovered, while the
    # neighbour's CORE (cf~0 for us) stays excluded.
    if cf is None or cf.shape != img.shape:
        cf_mask = np.ones_like(img, dtype=bool)
    else:
        cf_mask = cf >= 0.10
    ink = ((img < thr) & (roi > 0) & cf_mask).astype(np.uint8) * 255
    ink_closed = cv2.morphologyEx(
        ink,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (_cmpx(0.04, px_to_m),) * 2),
    )
    _nC, lab, _stats, _ = cv2.connectedComponentsWithStats(ink_closed, 8)
    lbls = set(int(v) for v in lab[labelmask > 0]) - {0}
    stair_ink = np.zeros((H, W), np.uint8)
    for label in lbls:
        stair_ink[lab == label] = 255
    cnts, _ = cv2.findContours(stair_ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    outline = max(cnts, key=cv2.contourArea)
    filled = np.zeros((H, W), np.uint8)
    cv2.drawContours(filled, [outline], -1, 255, -1)

    # NOTE: an ink-FREE turning landing (a flat podest with no risers) is
    # deliberately NOT fabricated here. A forward flood across white to
    # annex it proved crop/context-dependent and produced phantom steps
    # on other flights (verified on oyy4: 4 real -> 7 spurious). Every
    # RISER-defined step (incl. one shared ~50/50 with a neighbour
    # recovered by the low confidence floor) is handled here; a
    # riser-less podest needs a dedicated landing detector
    # (opposing-flight aware) and is out of safe scope.
    _ = arrow_dir  # reserved for a future landing detector

    # Dilate the division-line ink by D to bridge tiny stroke breaks so
    # each riser fully spans the outline and cleanly separates the step
    # cells. Then grow EACH cell back by D (+stroke) so adjacent treads
    # meet exactly ON the division line — connected, gap-free. The
    # division line itself IS the shared boundary (used directly).
    D = _cmpx(0.02, px_to_m)
    ink_div = ((img < thr) & (filled > 0)).astype(np.uint8) * 255
    ink_cut = cv2.dilate(
        ink_div, cv2.getStructuringElement(cv2.MORPH_RECT, (D, D))
    )
    cells_mask = ((filled > 0) & (ink_cut == 0)).astype(np.uint8) * 255
    nc, clab, cstats, _ = cv2.connectedComponentsWithStats(cells_mask, 8)
    min_area = _cmpx(0.06, px_to_m) ** 2
    gpx = D + max(4, _cmpx(0.006, px_to_m))  # reclaim dilation + stroke
    kgrow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gpx, gpx))
    out: List[np.ndarray] = []
    for i in range(1, nc):
        if cstats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        m = (clab == i).astype(np.uint8) * 255
        m = cv2.dilate(m, kgrow)  # grow back to the division line
        m = ((m > 0) & (filled > 0)).astype(np.uint8) * 255  # stay in outline
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            continue
        c = max(cs, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, _cmpx(0.03, px_to_m), True)
        if len(c) >= 3:
            out.append(c.reshape(-1, 2))

    # SLIVER FILTER: a real tread spans the FULL stair width perpendicular
    # to the climb direction. A spurious cell that leaked across a wall
    # gap (e.g. a doorway strip below the stair body) is much narrower
    # along that axis. Drop cells whose perpendicular extent is < 50% of
    # the median across the run -> robust to wall-bleeding without
    # depending on knowing the wall polygon here.
    if arrow_dir is not None and len(out) >= 3:
        d = np.asarray(arrow_dir, np.float64)
        d = d / (np.linalg.norm(d) or 1.0)
        perp = np.array([-d[1], d[0]])
        ext = [float(np.ptp(c.astype(np.float64) @ perp)) for c in out]
        med = float(np.median(ext))
        out = [c for c, e in zip(out, ext) if e >= 0.5 * med]
    return out
