"""V2 arrow detector — per-CC triangle-base asymmetry, multi-arrow safe.

Each grey-band CC after a 7x7 OPEN keeps the WIDE parts (triangle base +
thick line body). The triangle base is WIDER perpendicular to the major
axis than the line end, so we measure the perpendicular extent at each
end of the CC and point AWAY from the narrower (line) end. Averaging
across all detected arrows + snapping to the dominant axis handles
single-flight (1 arrow) and L/U/winder (2+ arrows) cases identically.
When the averaged direction is too small (opposing arrows cancel), we
fall back to the bbox long axis using the largest head's displacement.

ALGORITHM LOCKED 2026-05-28 — frozen baseline lives at
``c:/tmp/rebuild_bim_v2_LOCKED.py``. Do not modify without re-validating
on Matterport Sample (oyy4/pd2n/pm8f/pzya directions).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .stair_v2_cells import _cmpx


def _arrow_dir_from_cc(cc_pix: np.ndarray) -> Optional[np.ndarray]:
    """Per-arrow direction (tail->tip) from one head_seed CC.

    PCA gives the major axis; the perpendicular extent at each end of
    the major-axis projection identifies the triangle-base (wider) vs
    line-end (narrower). The direction points from line-end to base.
    """
    if cc_pix is None or len(cc_pix) < 10:
        return None
    pm = cc_pix.mean(0)
    cov = np.cov((cc_pix - pm).T)
    if cov.ndim == 0 or cov.shape != (2, 2):
        return None
    _, vecs = np.linalg.eigh(cov)
    major = vecs[:, -1]
    proj = (cc_pix - pm) @ major
    L = float(np.ptp(proj))
    if L < 4.0:
        return None
    margin = max(5.0, L * 0.18)
    perp = np.array([-major[1], major[0]])
    lo_w = float(np.ptp(cc_pix[proj < proj.min() + margin] @ perp))
    hi_w = float(np.ptp(cc_pix[proj > proj.max() - margin] @ perp))
    return major if hi_w >= lo_w else -major


def detect_arrow_direction_v2(
    img: np.ndarray,
    poly_px: np.ndarray,
    px_to_m: float,
) -> Optional[np.ndarray]:
    """Detect the climb direction inside ``poly_px``.

    Args:
        img: grayscale plan crop (HxW uint8).
        poly_px: stair bbox as int32 ``(N, 2)`` pixel polygon.
        px_to_m: metres per image pixel.

    Returns:
        ``(2,) float64`` unit direction in pixel coords (tail->tip), or
        ``None`` when no valid arrow CC survives the area filter.
    """
    if img is None or img.size == 0 or len(poly_px) < 3:
        return None
    H, W = img.shape[:2]
    bbm = np.zeros((H, W), np.uint8)
    cv2.fillPoly(bbm, [poly_px.astype(np.int32)], 255)
    bbm = cv2.erode(
        bbm, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    )
    grey = ((img >= 70) & (img <= 215) & (bbm > 0)).astype(np.uint8) * 255
    head_seed = cv2.morphologyEx(
        grey,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    nb, lab, stats, _ = cv2.connectedComponentsWithStats(head_seed, 8)
    min_area = _cmpx(0.04, px_to_m) ** 2

    dirs = []
    sizes = []
    for i in range(1, nb):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        ys, xs = np.where(lab == i)
        cc = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        d = _arrow_dir_from_cc(cc)
        if d is not None:
            dirs.append(d)
            sizes.append(int(stats[i, cv2.CC_STAT_AREA]))
    if not dirs:
        return None

    avg = np.mean(dirs, axis=0)
    nrm = float(np.linalg.norm(avg))
    if nrm < 0.15:
        # Arrows cancel out (e.g. user bbox spans two opposing flights).
        # Fall back: align with the bbox long axis using the LARGEST CC.
        poly = np.asarray(poly_px, dtype=np.float64)
        bb_min = poly.min(0)
        bb_max = poly.max(0)
        long_is_x = (bb_max - bb_min)[0] >= (bb_max - bb_min)[1]
        long_unit = (
            np.array([1.0, 0.0]) if long_is_x else np.array([0.0, 1.0])
        )
        biggest = int(np.argmax(sizes))
        sign = 1.0 if (dirs[biggest] @ long_unit) >= 0 else -1.0
        return sign * long_unit
    avg = avg / nrm
    if abs(avg[0]) >= abs(avg[1]):
        return np.array([1.0 if avg[0] >= 0 else -1.0, 0.0])
    return np.array([0.0, 1.0 if avg[1] >= 0 else -1.0])
