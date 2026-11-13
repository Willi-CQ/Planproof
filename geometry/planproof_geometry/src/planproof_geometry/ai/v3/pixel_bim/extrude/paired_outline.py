"""Paired-outline mode detector for wall masks.

Some floor plans (Matterport, architectural drawings with outline-only
walls) draw walls as two parallel lines bracketing the wall interior;
others (filled floor plans) draw walls as solid regions.

If we morphologically close a filled-wall mask with a kernel matched
to typical wall thickness, we risk merging two adjacent thin partition
walls into one. Conversely, if we DON'T close a paired-outline mask,
each wall ends up as two parallel thin strips in the 3D export.

This detector decides which regime we're in by looking at how much of
the wall mask sits in patterns of parallel edges 8-30 cm apart:

    ratio = <wall_pixels_in_paired_pattern> / <total_wall_pixels>

If ratio > 0.40 -> paired_mode = True -> safe to morph-close.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from planproof_geometry.ai.cv_compat import hough_lines_p


@dataclass
class PairedOutlineReport:
    paired_mode: bool
    paired_ratio: float
    line_pair_count: int
    total_lines: int
    close_kernel_px: int   # kernel size to use for morph-close (0 if not paired)


def detect_paired_outline_mode(
    wall_mask: np.ndarray,
    *,
    px_per_m: float,
    min_gap_cm: float = 8.0,
    max_gap_cm: float = 30.0,
    ratio_threshold: float = 0.40,
    hough_min_len_m: float = 0.30,
) -> PairedOutlineReport:
    """Analyse the wall mask and decide whether paired-outline closing is safe.

    Parameters
    ----------
    wall_mask : np.ndarray
        uint8 mask, 0 = non-wall, > 0 = wall-pixel.
    px_per_m : float
        How many pixels correspond to one metre (invariant across resolutions).
    min_gap_cm, max_gap_cm : float
        Parallel-line gaps considered "wall thickness" candidates.
    ratio_threshold : float
        Fraction of lines that must participate in a pair to declare paired mode.
    hough_min_len_m : float
        Minimum Hough-line length in metres (0.3 m skips door-symbol ticks).
    """
    if wall_mask.size == 0 or wall_mask.max() == 0:
        return PairedOutlineReport(False, 0.0, 0, 0, 0)

    # Hough-line transform on the wall-mask skeleton to find candidate
    # wall edges. We use HoughLinesP with relaxed parameters because the
    # refinement has already removed most non-wall ink.
    mask_bin = (wall_mask > 0).astype(np.uint8) * 255
    # Edges within the wall mask: Canny is overkill; use morph_gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(mask_bin, cv2.MORPH_GRADIENT, kernel)

    min_len_px = max(10, int(hough_min_len_m * px_per_m))
    # OpenCV 4 liefert (N, 1, 4), OpenCV 5 liefert (N, 4) - der Helfer
    # normiert auf (N, 4), damit das Entpacken unten in beiden Welten gilt
    # (Befund 08.09.2026, siehe planproof_geometry.ai.cv_compat).
    lines = hough_lines_p(
        edges,
        rho=1, theta=np.pi / 180, threshold=50,
        minLineLength=min_len_px,
        maxLineGap=int(0.1 * px_per_m),  # 10 cm max gap along one line
    )
    if len(lines) < 2:
        return PairedOutlineReport(False, 0.0, 0, 0, 0)

    # Normalise lines as (angle_deg_0_180, unit normal, point on the line).
    #
    # Bis zum 08.09.2026 stand hier ``rho = x1*cos(theta) + y1*sin(theta)`` mit
    # ``theta`` = Winkel der LINIE. Das ist nicht die Hesse-Normalform: fuer eine
    # waagerechte Linie liefert es ``x1``, fuer eine senkrechte ``y1`` - die
    # Koordinate ENTLANG der Linie statt ihres Abstands. Die Normale steht
    # senkrecht auf der Linie, also ``(-sin, cos)``.
    infos = []
    for x1, y1, x2, y2 in lines:
        dx, dy = (x2 - x1), (y2 - y1)
        angle = np.degrees(np.arctan2(dy, dx)) % 180.0
        theta = np.radians(angle)
        normal = (float(-np.sin(theta)), float(np.cos(theta)))
        length = float(np.hypot(dx, dy))
        infos.append((angle, normal, (float(x1), float(y1)), length))

    n = len(infos)
    paired_count = 0
    min_gap_px = min_gap_cm / 100.0 * px_per_m
    max_gap_px = max_gap_cm / 100.0 * px_per_m

    # Naive O(n^2) pairing — n is typically < 500 even on large plans.
    for i in range(n):
        a_i, (nx_i, ny_i), (px_i, py_i), _ = infos[i]
        found_partner = False
        for j in range(n):
            if i == j:
                continue
            a_j, _, (px_j, py_j), _ = infos[j]
            # angles must be close (paired outlines are parallel)
            da = min(abs(a_i - a_j), 180 - abs(a_i - a_j))
            if da > 5.0:
                continue
            # Gemessen wird gegen die Normale VON i, nicht gegen zwei getrennte
            # rho-Werte: die Winkel laufen modulo 180, ein Paar kann also als
            # 0,3 Grad und 179,6 Grad erkannt werden. Dann zeigen die beiden
            # Normalen entgegengesetzt, und eine Differenz zweier rho-Werte
            # waere ihre SUMME statt des Abstands.
            gap = abs((px_j - px_i) * nx_i + (py_j - py_i) * ny_i)
            if min_gap_px <= gap <= max_gap_px:
                found_partner = True
                break
        if found_partner:
            paired_count += 1

    ratio = paired_count / n if n else 0.0
    paired = ratio >= ratio_threshold

    close_kernel_px = 0
    if paired:
        # Kernel size = max expected wall gap + a small pad. The actual
        # kernel shape is RECT (chosen by the caller in wall_dispatcher);
        # this report only reports the size, not the shape.
        close_kernel_px = int(max_gap_px) + 2
    return PairedOutlineReport(
        paired_mode=paired,
        paired_ratio=ratio,
        line_pair_count=paired_count,
        total_lines=n,
        close_kernel_px=close_kernel_px,
    )
