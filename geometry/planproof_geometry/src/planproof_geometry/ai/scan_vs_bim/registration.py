"""Registrierung: Roh-Punktwolke -> Plan-Welt-Frame (ohne ML, ohne Marker).

Pipeline (alle Schritte rein geometrisch, numpy/scipy):

1. **Einheiten-Sanity** — Scan ist metrisch (Laser); kein freier Scale-Fit,
   nur Snap auf {m, mm, cm, inch} mit Warnung (vgl.
   feedback_v3_class_c_is_scaling: Drift = Skalierung, nie stillschweigend).
2. **Up-Achse + Leveling** — dominante horizontale Ebene (Boden/Decke) via
   RANSAC (Reuse ``scan_to_bim.ransac``); Wolke wird auf h-up gedreht und
   exakt eingeebnet.
3. **Boden-Offset** — staerkster niedriger Peak im h-Histogramm -> Scan-Boden
   auf Plan-Bodenhoehe.
4. **Yaw** — Wandrichtungs-Histogramm (Struktur-Tensor auf dem 2D-Occupancy-
   Grid der Wandband-Punkte) mod 90 Grad; Plan-Waende sind achsparallel,
   also Kandidaten theta + k*90.
5. **Translation** — FFT-Kreuzkorrelation Wolken-Occupancy gegen
   Plan-Wand-Maske, bester Peak ueber alle 4 Yaw-Kandidaten.
6. **Feinschliff** — 2D-Punkt-zu-Wand-ICP via Distanzfeld
   (``scipy.ndimage.distance_transform_edt``) und Gauss-Newton ueber
   (dx, dz, dtheta).

Ergebnis ist eine homogene 4x4 (Roh -> Plan-Welt) + Guete-Metriken. Bei
Mehrgeschossern registriert v1 global (Warnung im Ergebnis).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..scan_to_bim.ransac import fit_plane_ransac
from .types import PlanElement, Registration

# Snap-Kandidaten fuer Einheiten (Meter-Faktor). Kein freier Fit!
_UNIT_CANDIDATES = (1.0, 0.001, 0.01, 0.0254)

# Registrierung rechnet auf einer Voxel-Teilstichprobe — Genauigkeit haengt an
# Zellgroessen, nicht an Punktdichte.
_REG_VOXEL_M = 0.04
_REG_MAX_POINTS = 400_000
_GRID_CELL_M = 0.05
_CORR_CELL_M = 0.10
_ICP_ITERATIONS = 12
_WALL_BAND_REL = (0.3, 0.85)  # Anteil der Plan-Wandhoehe fuer das Wandband


def _voxel_downsample(points: np.ndarray, voxel: float, max_points: int) -> np.ndarray:
    """Ein Punkt je Voxel (erster Treffer) — deterministisch, reines numpy."""
    if points.shape[0] == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    # Eindeutige Voxel via strukturiertem View (schneller als tuple-Set).
    _, idx = np.unique(keys, axis=0, return_index=True)
    sampled = points[np.sort(idx)]
    if sampled.shape[0] > max_points:
        step = int(math.ceil(sampled.shape[0] / max_points))
        sampled = sampled[::step]
    return sampled


def _estimate_unit_scale(
    points: np.ndarray, plan_diag_m: float
) -> Tuple[float, List[str]]:
    """Snap auf plausible Einheit. Kriterium: vertikale Ausdehnung in
    [1.8, 60] m UND Horizontal-Diagonale im Verhaeltnis [0.05, 20] zum Plan."""
    warnings: List[str] = []
    extent = points.max(axis=0) - points.min(axis=0)
    raw_diag = float(np.linalg.norm(extent))
    best = 1.0
    best_score = -np.inf
    for s in _UNIT_CANDIDATES:
        vert = float(np.sort(extent)[0]) * s  # kleinste Ausdehnung ~ Hoehe
        diag = raw_diag * s
        score = 0.0
        if 1.8 <= vert <= 60.0:
            score += 2.0
        if plan_diag_m > 0 and 0.05 <= diag / plan_diag_m <= 20.0:
            score += 1.0
        # Praeferenz fuer Meter bei Gleichstand (Scan-Formate sind i. d. R. metrisch).
        if s == 1.0:
            score += 0.5
        if score > best_score:
            best_score = score
            best = s
    if best != 1.0:
        warnings.append(
            f"Einheiten-Snap: Wolke wurde mit Faktor {best} auf Meter skaliert "
            f"(Roh-Diagonale {raw_diag:.1f})."
        )
    return best, warnings


def _axis_concentration(coords: np.ndarray, bin_m: float = 0.05) -> float:
    """Anteil des staerksten Histogramm-Bins — Boeden erzeugen scharfe Peaks."""
    lo, hi = float(coords.min()), float(coords.max())
    if hi - lo < bin_m:
        return 1.0
    bins = max(4, int((hi - lo) / bin_m))
    hist, _ = np.histogram(coords, bins=bins)
    return float(hist.max()) / float(coords.shape[0])


def _level_up_axis(points: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Rotation, die die Boden-Normale exakt auf h-up (Achse 1) legt.

    Up-Erkennung NICHT ueber die dominante RANSAC-Ebene (eine grosse Wand
    kann mehr Inlier haben als der Boden), sondern ueber die Histogramm-
    Konzentration: entlang der up-Achse liegen sehr viele Punkte in exakt
    einem Bin (Bodenplatte). Danach exaktes Leveling per Ebenen-Fit NUR auf
    den Boden-Peak-Punkten.
    """
    warnings: List[str] = []

    # Kandidaten: die 3 Datei-Achsen + Normale der dominanten RANSAC-Ebene
    # (faengt leicht verkippte Scans, die nicht achsen-ausgerichtet sind).
    candidates: List[np.ndarray] = [np.eye(3)[i] for i in range(3)]
    plane = fit_plane_ransac(
        points,
        distance_threshold=0.03,
        max_iterations=400,
        min_inliers=max(200, points.shape[0] // 30),
    )
    if plane is not None:
        candidates.append(plane.normal / np.linalg.norm(plane.normal))

    best_axis = candidates[1]
    best_score = -1.0
    for cand in candidates:
        cand = cand / np.linalg.norm(cand)
        score = _axis_concentration(points @ cand)
        if score > best_score:
            best_score = score
            best_axis = cand
    if best_score < 0.05:
        warnings.append(
            "Leveling: keine ausgepraegte horizontale Flaeche gefunden — "
            "up-Achse unsicher, Registrierung mit Vorsicht interpretieren."
        )

    # Exaktes Leveling: Ebene nur auf den Boden-Peak-Punkten fitten.
    coords = points @ best_axis
    lo, hi = float(coords.min()), float(coords.max())
    bins = max(4, int((hi - lo) / 0.05))
    hist, edges = np.histogram(coords, bins=bins)
    # Boden = staerkster Peak in der UNTEREN Haelfte (Decke ist der obere).
    half = len(hist) // 2 or 1
    peak = int(np.argmax(hist[:half])) if hist[:half].max() > 0 else int(np.argmax(hist))
    peak_center = (edges[peak] + edges[peak + 1]) / 2.0
    near_floor = np.abs(coords - peak_center) < 0.10
    normal = best_axis
    if near_floor.sum() >= 100:
        floor_pts = points[near_floor]
        centroid = floor_pts.mean(axis=0)
        _, _, vh = np.linalg.svd(floor_pts - centroid, full_matrices=False)
        fitted = vh[2]
        if float(np.dot(fitted, best_axis)) < 0:
            fitted = -fitted
        # Nur uebernehmen, wenn plausibel nah an der Kandidaten-Achse.
        if float(np.dot(fitted, best_axis)) > 0.9:
            normal = fitted

    normal = normal / np.linalg.norm(normal)
    # Vorzeichen: mehr Punkte OBERHALB des Bodens als unterhalb.
    above = float(np.mean(points @ normal > peak_center))
    if above < 0.5:
        normal = -normal

    # Rotation, die `normal` auf (0, 1, 0) abbildet (Rodrigues).
    target = np.array([0.0, 1.0, 0.0])
    v = np.cross(normal, target)
    s = float(np.linalg.norm(v))
    c = float(np.dot(normal, target))
    if s < 1e-9:
        rot = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rot = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    return rot, warnings


def _floor_height(points_h: np.ndarray) -> float:
    """Staerkster niedriger Peak im Hoehen-Histogramm = Scan-Bodenhoehe."""
    lo, hi = float(points_h.min()), float(points_h.max())
    if hi - lo < 0.5:
        return lo
    bins = max(20, int((hi - lo) / 0.05))
    hist, edges = np.histogram(points_h, bins=bins)
    # Nur die untere Haelfte betrachten (Decke ist der andere grosse Peak).
    half = len(hist) // 2 or 1
    peak = int(np.argmax(hist[:half]))
    return float((edges[peak] + edges[peak + 1]) / 2.0)


def _plan_wall_grid(
    walls: Sequence[PlanElement], cell: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterisiert Wand-/Stuetzen-Footprints in ein bool-Grid.

    Returns (grid[z, x], origin_xz). Grid traegt einen Rand von 2 m, damit
    Korrelation/ICP auch bei leicht groesserem Scan nicht abschneidet.
    """
    mins = np.array([w.aabb_min[[0, 2]] for w in walls])
    maxs = np.array([w.aabb_max[[0, 2]] for w in walls])
    lo = mins.min(axis=0) - 2.0
    hi = maxs.max(axis=0) + 2.0
    nx = max(4, int(math.ceil((hi[0] - lo[0]) / cell)))
    nz = max(4, int(math.ceil((hi[1] - lo[1]) / cell)))
    grid = np.zeros((nz, nx), dtype=bool)
    for w in walls:
        x0, z0 = w.aabb_min[[0, 2]]
        x1, z1 = w.aabb_max[[0, 2]]
        ix0 = max(0, int((x0 - lo[0]) / cell))
        ix1 = min(nx - 1, int((x1 - lo[0]) / cell))
        iz0 = max(0, int((z0 - lo[1]) / cell))
        iz1 = min(nz - 1, int((z1 - lo[1]) / cell))
        grid[iz0 : iz1 + 1, ix0 : ix1 + 1] = True
    return grid, lo


def _points_grid(
    xz: np.ndarray, lo: np.ndarray, shape: Tuple[int, int], cell: float
) -> np.ndarray:
    """2D-Occupancy der Punkte im selben Raster wie das Plan-Grid."""
    grid = np.zeros(shape, dtype=bool)
    ix = np.floor((xz[:, 0] - lo[0]) / cell).astype(np.int64)
    iz = np.floor((xz[:, 1] - lo[1]) / cell).astype(np.int64)
    ok = (ix >= 0) & (ix < shape[1]) & (iz >= 0) & (iz < shape[0])
    grid[iz[ok], ix[ok]] = True
    return grid


def _dominant_wall_angle(xz: np.ndarray, cell: float) -> float:
    """Dominante Wandrichtung (mod 90 Grad) via Struktur-Tensor des
    Occupancy-Grids der Wandband-Punkte."""
    lo = xz.min(axis=0) - cell
    hi = xz.max(axis=0) + cell
    nx = max(8, int(math.ceil((hi[0] - lo[0]) / cell)))
    nz = max(8, int(math.ceil((hi[1] - lo[1]) / cell)))
    grid = np.zeros((nz, nx), dtype=np.float32)
    ix = np.clip(((xz[:, 0] - lo[0]) / cell).astype(np.int64), 0, nx - 1)
    iz = np.clip(((xz[:, 1] - lo[1]) / cell).astype(np.int64), 0, nz - 1)
    np.add.at(grid, (iz, ix), 1.0)
    grid = np.minimum(grid, 4.0)  # Dichte kappen, sonst dominieren dichte Flecken

    gz, gx = np.gradient(grid)
    mag = np.hypot(gx, gz)
    mask = mag > 1e-6
    if not mask.any():
        return 0.0
    # Gradient steht senkrecht auf der Wand; Kantenrichtung = +90 Grad. Fuer
    # mod-90-Histogramm ist das irrelevant (gleiche Klasse).
    ang = np.arctan2(gz[mask], gx[mask])
    ang90 = np.mod(ang, math.pi / 2.0)
    hist, edges = np.histogram(ang90, bins=90, range=(0.0, math.pi / 2.0), weights=mag[mask])
    # Zirkulaer glaetten (Nachbar-Bins), dann Peak + Parabel-Subbin-Verfeinerung.
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    padded = np.concatenate([hist[-2:], hist, hist[:2]])
    smooth = np.convolve(padded, kernel, mode="valid") / kernel.sum()
    peak = int(np.argmax(smooth))
    left, mid, right = (
        smooth[(peak - 1) % len(smooth)],
        smooth[peak],
        smooth[(peak + 1) % len(smooth)],
    )
    denom = left - 2 * mid + right
    frac = 0.0 if abs(denom) < 1e-12 else 0.5 * (left - right) / denom
    bin_w = (math.pi / 2.0) / 90
    return float((peak + frac) * bin_w)


def _fft_correlate(plan_grid: np.ndarray, cloud_grid: np.ndarray) -> Tuple[int, int, float]:
    """Beste Verschiebung (dz, dx) via FFT-Kreuzkorrelation. Score = Overlap."""
    pf = np.fft.rfft2(plan_grid.astype(np.float32))
    cf = np.fft.rfft2(cloud_grid.astype(np.float32))
    corr = np.fft.irfft2(pf * np.conj(cf), s=plan_grid.shape)
    peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
    score = float(corr[peak])
    dz, dx = int(peak[0]), int(peak[1])
    # FFT liefert zyklische Shifts — in signierte Verschiebung umrechnen.
    if dz > plan_grid.shape[0] // 2:
        dz -= plan_grid.shape[0]
    if dx > plan_grid.shape[1] // 2:
        dx -= plan_grid.shape[1]
    return dz, dx, score


def _rot2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _icp_refine(
    xz: np.ndarray,
    plan_grid: np.ndarray,
    lo: np.ndarray,
    cell: float,
    iterations: int = _ICP_ITERATIONS,
) -> Tuple[float, np.ndarray, float, float]:
    """Gauss-Newton ueber (theta, tx, tz) auf dem Distanzfeld der Wand-Maske.

    Returns (dtheta, dt, rms_m, inlier_fraction).
    """
    from scipy import ndimage

    dist = ndimage.distance_transform_edt(~plan_grid) * cell
    gz, gx = np.gradient(dist, cell)

    def _sample(field: np.ndarray, pts: np.ndarray) -> np.ndarray:
        fx = (pts[:, 0] - lo[0]) / cell
        fz = (pts[:, 1] - lo[1]) / cell
        return ndimage.map_coordinates(field, [fz, fx], order=1, mode="nearest")

    theta = 0.0
    t = np.zeros(2)
    center = xz.mean(axis=0)
    rms = float("inf")
    inlier_fraction = 0.0
    for _ in range(iterations):
        cur = (xz - center) @ _rot2d(theta).T + center + t
        d = _sample(dist, cur)
        # Robust: nur Punkte nahe einer Wand ziehen lassen (Trimmed ICP).
        w = d < 0.30
        inlier_fraction = float(np.mean(w))
        if w.sum() < 50:
            break
        dw = d[w]
        rms = float(np.sqrt(np.mean(dw**2)))
        jx = _sample(gx, cur[w])
        jz = _sample(gz, cur[w])
        rel = cur[w] - center
        # d(residual)/d(theta) = grad . d(pos)/d(theta); d(pos)/d(theta) = (-rel_z, rel_x)
        jt = jx * (-rel[:, 1]) + jz * (rel[:, 0])
        jac = np.column_stack([jt, jx, jz])
        jtj = jac.T @ jac + np.eye(3) * 1e-6
        jtr = jac.T @ dw
        try:
            delta = np.linalg.solve(jtj, jtr)
        except np.linalg.LinAlgError:
            break
        theta -= float(delta[0])
        t -= delta[1:]
        if np.abs(delta).max() < 1e-5:
            break
    return theta, t, rms, inlier_fraction


def register(
    points: np.ndarray,
    plan_elements: Sequence[PlanElement],
    config: Optional[Dict] = None,
) -> Registration:
    """Registriert die Roh-Punktwolke auf den Plan-Welt-Frame (kanonisch x,h,z).

    ``points`` sind ROHE Datei-Koordinaten. Ergebnis-Matrix bildet Roh ->
    kanonischen Plan-Frame ab; der Aufrufer rechnet fuer Preview-Artefakte
    ggf. in den GLB-Welt-Frame zurueck (up_axis).
    """
    cfg = config or {}
    warnings: List[str] = []

    walls = [
        e
        for e in plan_elements
        if e.cls in ("wall", "column") and e.aabb_min is not None and e.aabb_max is not None
    ]
    if not walls:
        raise ValueError("scan_vs_bim: Plan enthaelt keine Waende/Stuetzen — Registrierung unmoeglich")
    if points.shape[0] < 1000:
        raise ValueError(f"scan_vs_bim: zu wenige Punkte fuer Registrierung ({points.shape[0]})")

    wall_min_h = float(min(w.aabb_min[1] for w in walls))
    wall_max_h = float(max(w.aabb_max[1] for w in walls))
    wall_height = wall_max_h - wall_min_h
    if wall_height > 4.5:
        warnings.append(
            "Plan umfasst vermutlich mehrere Geschosse — v1 registriert global; "
            "geschossweise Registrierung ist Follow-up."
        )
    plan_lo = np.min([w.aabb_min for w in walls], axis=0)
    plan_hi = np.max([w.aabb_max for w in walls], axis=0)
    plan_diag = float(np.linalg.norm((plan_hi - plan_lo)[[0, 2]]))

    # 1) Einheiten
    unit_scale, w1 = _estimate_unit_scale(points, plan_diag)
    warnings.extend(w1)
    unit_scale = float(cfg.get("unit_scale") or unit_scale)
    work = points * unit_scale

    # Downsample fuer alle folgenden Schritte
    work = _voxel_downsample(work, _REG_VOXEL_M, _REG_MAX_POINTS)

    # 2) Leveling / up-Achse
    level_rot, w2 = _level_up_axis(work)
    warnings.extend(w2)
    work = work @ level_rot.T

    # 3) Boden-Offset
    scan_floor = _floor_height(work[:, 1])
    h_shift = wall_min_h - scan_floor

    # 4) Wandband + Yaw
    band_lo = wall_min_h + _WALL_BAND_REL[0] * wall_height
    band_hi = wall_min_h + _WALL_BAND_REL[1] * wall_height
    h = work[:, 1] + h_shift
    band = work[(h >= band_lo) & (h <= band_hi)]
    if band.shape[0] < 500:
        warnings.append("Wenige Wandband-Punkte — Registrierung unsicher (Wolke unvollstaendig?).")
        band = work
    xz = band[:, [0, 2]]
    theta_cloud = _dominant_wall_angle(xz, _GRID_CELL_M)

    # 5) Translation je 90-Grad-Kandidat via FFT-Korrelation
    plan_grid, grid_lo = _plan_wall_grid(walls, _CORR_CELL_M)
    xz_center = xz.mean(axis=0)
    best = None
    for k in range(4):
        cand = -theta_cloud + k * (math.pi / 2.0)
        rot_xz = (xz - xz_center) @ _rot2d(cand).T
        # Grob in Plan-Naehe schieben (Zentroid), Rest macht die Korrelation.
        rot_xz = rot_xz + (grid_lo + np.array(plan_grid.shape[::-1]) * _CORR_CELL_M / 2.0)
        cloud_grid = _points_grid(rot_xz, grid_lo, plan_grid.shape, _CORR_CELL_M)
        dz, dx, score = _fft_correlate(plan_grid, cloud_grid)
        if best is None or score > best[0]:
            best = (score, cand, dx, dz, rot_xz)
    assert best is not None
    _, yaw, dx, dz, rot_xz = best
    shift = np.array([dx * _CORR_CELL_M, dz * _CORR_CELL_M])
    coarse_xz = rot_xz + shift

    # 6) ICP-Feinschliff auf feinerem Grid
    fine_grid, fine_lo = _plan_wall_grid(walls, _GRID_CELL_M)
    dtheta, dt, rms, inlier_fraction = _icp_refine(coarse_xz, fine_grid, fine_lo, _GRID_CELL_M)

    if inlier_fraction < 0.25:
        warnings.append(
            f"Registrierung schwach (Inlier {inlier_fraction:.0%}) — Overlay-Befunde "
            "mit Vorsicht interpretieren."
        )

    # 7) Gesamt-Matrix komponieren: Roh -> kanonischer Plan-Frame.
    #    Exakt die oben gerechnete Kette:
    #    p1   = level_rot @ (unit_scale * p_raw);  p1_h += h_shift
    #    q1   = R(yaw) @ (p1_xz - xz_center) + plan_grid_center + shift   (Korrelation)
    #    q2   = R(dtheta) @ (q1 - icp_center) + icp_center + dt           (ICP)
    icp_center = coarse_xz.mean(axis=0)
    plan_grid_center = grid_lo + np.array(plan_grid.shape[::-1]) * _CORR_CELL_M / 2.0

    def _yaw4(theta: float) -> np.ndarray:
        rot = _rot2d(theta)
        m = np.eye(4)
        m[0, 0], m[0, 2] = rot[0, 0], rot[0, 1]
        m[2, 0], m[2, 2] = rot[1, 0], rot[1, 1]
        return m

    def _trans4(x: float, h: float, z: float) -> np.ndarray:
        m = np.eye(4)
        m[0, 3], m[1, 3], m[2, 3] = x, h, z
        return m

    base = np.eye(4)
    base[:3, :3] = level_rot * unit_scale
    matrix = (
        _trans4(icp_center[0] + dt[0], 0.0, icp_center[1] + dt[1])
        @ _yaw4(dtheta)
        @ _trans4(-icp_center[0], 0.0, -icp_center[1])
        @ _trans4(plan_grid_center[0] + shift[0], 0.0, plan_grid_center[1] + shift[1])
        @ _yaw4(yaw)
        @ _trans4(-xz_center[0], 0.0, -xz_center[1])
        @ _trans4(0.0, h_shift, 0.0)
        @ base
    )

    return Registration(
        matrix=matrix,
        unit_scale=unit_scale,
        yaw_rad=yaw + dtheta,
        rms_m=rms if math.isfinite(rms) else 999.0,
        inlier_fraction=inlier_fraction,
        method="floor+manhattan+fftcorr+icp",
        warnings=warnings,
    )
