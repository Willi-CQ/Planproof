"""Vier-Klassen-Engine: Punktwolke (registriert) gegen Soll-Bauteile.

Kern-Idee (Scan-vs-BIM, Bosché-Familie): Die erwartete Bauteil-Oberflaeche
wird in Zellen gerastert und jede Zelle nach Evidenz klassifiziert:

* ``hit``      — Punkte im Toleranzband an der Flaeche (Bauteil da)
* ``through``  — Evidenz GEGEN Existenz: Punkte IM Volumen (physikalisch
                 unmoeglich bei gebautem Bauteil) oder Sichtstrahlen, die das
                 Volumen durchqueren (nur mit Sensor-Posen, E57)
* ``blocked``  — Punkte VOR der Flaeche, aber keine an ihr (verdeckt)
* ``empty``    — keinerlei Evidenz (nicht erfasst)

Daraus das Bauteil-Verdikt: gebaut / fehlt / abweichend / nicht einsehbar.
Ohne die 4. Klasse wuerde jede Verdeckung zum Phantom-Mangel.

Oeffnungen (Tuer/Fenster) werden aus der Wandflaeche maskiert und separat
verifiziert: Durchgangs-Evidenz = Oeffnung existiert; eine an der Wandebene
geschlossene Flaeche kann Tuerblatt ODER faelschlich durchgebaute Wand sein
-> ehrlich ``not_visible`` mit Foto-Check-Empfehlung, KEIN Phantom-Mangel.

Alles advisory (Handy-Scan-Genauigkeit >> DIN-Toleranzen); DIN 18202-Werte
(Ebenheit/Lot) sind Soft-Flags.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .types import (
    VERDICT_BUILT,
    VERDICT_DEVIATING,
    VERDICT_MISSING,
    VERDICT_NOT_VISIBLE,
    VERDICT_UNVERIFIABLE,
    VERIFIABLE_CLASSES,
    ElementVerdict,
    PlanElement,
    RoomSummary,
)

# Zell-Klassen (Prioritaet absteigend bei Mehrfach-Evidenz)
_CELL_OPENING = 4
_CELL_HIT = 3
_CELL_THROUGH = 2
_CELL_BLOCKED = 1
_CELL_EMPTY = 0


@dataclass
class EngineConfig:
    """Tuning-Knoepfe — Defaults fuer Handy-/Aufmass-Scans (advisory)."""

    cell_size_m: float = 0.10
    surface_tol_m: float = 0.05
    deviation_flag_mm: float = 30.0  # konsistent zum Compare-Noise-Floor
    min_hit_ratio: float = 0.35
    through_evidence_ratio: float = 0.20
    # Oeffnungen: schon wenige verlaessliche Durchgangs-Zellen belegen einen
    # Durchgang (Sichtkegel durch eine Tuer ist schmal) — separate Schwelle.
    opening_through_ratio: float = 0.05
    front_probe_m: float = 1.0
    min_cells: int = 6
    min_points_per_cell: int = 2
    engine_voxel_m: float = 0.02
    engine_max_points: int = 3_000_000

    @classmethod
    def from_dict(cls, cfg: Optional[Dict]) -> "EngineConfig":
        out = cls()
        for key, value in (cfg or {}).items():
            if hasattr(out, key) and isinstance(value, (int, float)):
                setattr(out, key, float(value))
        out.min_cells = int(out.min_cells)
        out.min_points_per_cell = int(out.min_points_per_cell)
        out.engine_max_points = int(out.engine_max_points)
        return out


@dataclass
class _Evidence:
    """Zell-Raster + Messwerte eines Flaechen-Checks.

    ``offset_sides`` traegt das Vorzeichen von ``d`` (welche der beiden
    Flaechen ein Hit-Punkt trifft): Eine lateral verschobene Wand liest auf
    der einen Seite +delta, auf der anderen -delta — nur seitengetrennt wird
    daraus Verschiebung statt (falschem) Null-Median.
    """

    grid: np.ndarray  # (nv, nu) int8 Zell-Klassen
    offsets_mm: np.ndarray = field(default_factory=lambda: np.empty(0))
    offset_sides: np.ndarray = field(default_factory=lambda: np.empty(0))
    offset_heights: np.ndarray = field(default_factory=lambda: np.empty(0))

    def ratios(self) -> Dict[str, float]:
        effective = self.grid != _CELL_OPENING
        n = int(effective.sum())
        if n == 0:
            return {"hit": 0.0, "through": 0.0, "blocked": 0.0, "empty": 1.0, "opening": 1.0, "cells": 0}
        g = self.grid[effective]
        total = float(self.grid.size)
        return {
            "hit": float(np.mean(g == _CELL_HIT)),
            "through": float(np.mean(g == _CELL_THROUGH)),
            "blocked": float(np.mean(g == _CELL_BLOCKED)),
            "empty": float(np.mean(g == _CELL_EMPTY)),
            "opening": float(np.sum(self.grid == _CELL_OPENING)) / total,
            "cells": n,
        }


def _severity_for_deviation(mm: float) -> str:
    if mm < 50.0:
        return "low"
    if mm < 100.0:
        return "medium"
    return "high"


def _bin(vals: np.ndarray, lo: float, cell: float, n: int) -> np.ndarray:
    return np.clip(((vals - lo) / cell).astype(np.int64), 0, n - 1)


def _mark_cells(
    grid: np.ndarray,
    iu: np.ndarray,
    iv: np.ndarray,
    cell_class: int,
    min_count: int = 1,
) -> None:
    """Zell-Klasse setzen, wenn genug Punkte und Prioritaet hoeher ist."""
    if iu.size == 0:
        return
    counts = np.zeros(grid.shape, dtype=np.int32)
    np.add.at(counts, (iv, iu), 1)
    mask = (counts >= min_count) & (grid < cell_class)
    grid[mask] = cell_class


def _wall_frame(elem: PlanElement) -> Optional[Tuple[int, int, float, float]]:
    """(thin_axis, long_axis, thickness, length) einer Wand/Stuetze.

    Achsen sind kanonische Indizes (0 = x, 2 = z); h ist immer Achse 1.
    """
    size = elem.size
    if size is None:
        return None
    dx, dz = float(size[0]), float(size[2])
    if dx <= dz:
        return 0, 2, dx, dz
    return 2, 0, dz, dx


def _openings_for_wall(
    wall: PlanElement, openings: Sequence[PlanElement], margin: float = 0.12
) -> List[PlanElement]:
    """Tuer-/Fenster-Elemente, deren AABB die Wand-AABB schneidet."""
    if wall.aabb_min is None:
        return []
    out = []
    for o in openings:
        if o.aabb_min is None or o.aabb_max is None:
            continue
        if np.all(o.aabb_max >= wall.aabb_min - margin) and np.all(
            o.aabb_min <= wall.aabb_max + margin
        ):
            out.append(o)
    return out


def _check_planar_surface(
    points: np.ndarray,
    origins: Optional[np.ndarray],
    scan_ids: Optional[np.ndarray],
    *,
    normal_axis: int,
    u_axis: int,
    v_axis: int,
    center_n: float,
    half_thickness: float,
    u_range: Tuple[float, float],
    v_range: Tuple[float, float],
    openings_uv: Sequence[Tuple[float, float, float, float]],
    cfg: EngineConfig,
) -> _Evidence:
    """Klassifiziert die Zellen einer (doppelseitigen) ebenen Soll-Flaeche.

    Funktioniert fuer Waende (normal_axis horizontal) und Boeden/Decken
    (normal_axis = 1). ``u``/``v`` spannen das Zellraster auf.
    """
    cell = cfg.cell_size_m
    tol = cfg.surface_tol_m
    u0, u1 = u_range
    v0, v1 = v_range
    nu = max(1, int(math.ceil((u1 - u0) / cell)))
    nv = max(1, int(math.ceil((v1 - v0) / cell)))
    grid = np.zeros((nv, nu), dtype=np.int8)

    # Oeffnungs-Zellen maskieren (nur Waende; Boeden haben keine)
    for ou0, ou1, ov0, ov1 in openings_uv:
        cu0 = max(0, int((ou0 - u0) / cell))
        cu1 = min(nu - 1, int(math.ceil((ou1 - u0) / cell)) - 1)
        cv0 = max(0, int((ov0 - v0) / cell))
        cv1 = min(nv - 1, int(math.ceil((ov1 - v0) / cell)) - 1)
        if cu1 >= cu0 and cv1 >= cv0:
            grid[cv0 : cv1 + 1, cu0 : cu1 + 1] = _CELL_OPENING

    d = points[:, normal_axis] - center_n
    u = points[:, u_axis]
    v = points[:, v_axis]
    in_footprint = (u >= u0) & (u <= u1) & (v >= v0) & (v <= v1)

    s = np.abs(d) - half_thickness  # 0 an der Oberflaeche, <0 im Volumen

    # 1) blocked: Punkte deutlich VOR einer der beiden Flaechen
    front = in_footprint & (s > tol) & (s <= cfg.front_probe_m)
    _mark_cells(grid, _bin(u[front], u0, cell, nu), _bin(v[front], v0, cell, nv), _CELL_BLOCKED)

    # 2) through (posefrei): Punkte IM Volumen — bei gebautem Bauteil unmoeglich
    interior = in_footprint & (s < -tol)
    _mark_cells(
        grid,
        _bin(u[interior], u0, cell, nu),
        _bin(v[interior], v0, cell, nv),
        _CELL_THROUGH,
        min_count=cfg.min_points_per_cell,
    )

    # 2b) through (mit Posen): Sichtstrahl kreuzt das Volumen, Ziel klar dahinter
    if origins is not None and scan_ids is not None and origins.shape[0] > 0:
        o = origins[np.clip(scan_ids, 0, origins.shape[0] - 1)]
        d_o = o[:, normal_axis] - center_n
        crosses = (np.sign(d_o) != np.sign(d)) & (np.abs(d) > half_thickness + tol)
        if crosses.any():
            denom = d_o - d
            t = np.where(np.abs(denom) > 1e-12, d_o / denom, 0.0)
            cu = o[:, u_axis] + t * (u - o[:, u_axis])
            cv = o[:, v_axis] + t * (v - o[:, v_axis])
            ray_ok = crosses & (t > 0.0) & (t < 1.0) & (cu >= u0) & (cu <= u1) & (cv >= v0) & (cv <= v1)
            _mark_cells(
                grid,
                _bin(cu[ray_ok], u0, cell, nu),
                _bin(cv[ray_ok], v0, cell, nv),
                _CELL_THROUGH,
                min_count=cfg.min_points_per_cell,
            )

    # 3) hit: Punkte im Toleranzband um eine der beiden Flaechen
    hit = in_footprint & (np.abs(s) <= tol)
    _mark_cells(
        grid,
        _bin(u[hit], u0, cell, nu),
        _bin(v[hit], v0, cell, nv),
        _CELL_HIT,
        min_count=cfg.min_points_per_cell,
    )

    # Messwerte: signierter Offset der Hit-Punkte relativ zur naeheren Flaeche
    # (positiv = ausserhalb der Soll-Flaeche), plus Seiten-Vorzeichen von d
    # fuer die Verschiebungs-/Dicken-Trennung im Verdikt.
    offsets = s[hit]
    sides = np.sign(d[hit])
    heights = v[hit]
    return _Evidence(
        grid=grid,
        offsets_mm=offsets * 1000.0,
        offset_sides=sides,
        offset_heights=heights,
    )


def _verdict_from_evidence(
    elem: PlanElement,
    ev: _Evidence,
    cfg: EngineConfig,
    *,
    measure_plumb: bool,
) -> ElementVerdict:
    r = ev.ratios()
    verdict = ElementVerdict(
        component_id=elem.id,
        node_path=elem.node_path,
        cls=elem.cls or "unknown",
        room_key=elem.room_key,
        verdict=VERDICT_NOT_VISIBLE,
        hit_ratio=r["hit"],
        through_ratio=r["through"],
        blocked_ratio=r["blocked"],
        empty_ratio=r["empty"],
        opening_ratio=r["opening"],
    )
    if r["cells"] < cfg.min_cells:
        verdict.verdict = VERDICT_NOT_VISIBLE
        verdict.detail = "Soll-Flaeche kleiner als Mindest-Zellzahl — keine belastbare Aussage."
        return verdict

    if r["through"] >= cfg.through_evidence_ratio and r["hit"] < 0.5:
        verdict.verdict = VERDICT_MISSING
        verdict.rule_id = f"svb:{verdict.cls}:missing"
        verdict.severity = "high" if verdict.cls == "wall" else "medium"
        verdict.detail = (
            f"Durchgangs-/Volumen-Evidenz auf {r['through']:.0%} der Flaeche — "
            "Bauteil am Soll-Ort nicht vorhanden."
        )
        return verdict

    if r["hit"] >= cfg.min_hit_ratio:
        verdict.verdict = VERDICT_BUILT
        if ev.offsets_mm.size >= 10:
            pos = ev.offsets_mm[ev.offset_sides > 0]
            neg = ev.offsets_mm[ev.offset_sides < 0]
            if pos.size >= 10 and neg.size >= 10:
                # Beide Flaechen gesehen: lateral Verschiebung und Dicken-
                # Abweichung trennen (eine verschobene Wand liest +d/-d —
                # ein naiver Gesamt-Median waere faelschlich ~0).
                med_pos, med_neg = float(np.median(pos)), float(np.median(neg))
                shift = (med_pos - med_neg) / 2.0
                thickness_delta = (med_pos + med_neg) / 2.0
                med = shift if abs(shift) >= abs(thickness_delta) else thickness_delta
                kind = "Lage" if abs(shift) >= abs(thickness_delta) else "Dicke"
            else:
                med = float(np.median(ev.offsets_mm))
                kind = "Flaeche"
            verdict.deviation_mm = med
            resid = ev.offsets_mm - float(np.median(ev.offsets_mm))
            verdict.flatness_mm = float(np.sqrt(np.mean(resid**2)))
            if measure_plumb and ev.offset_heights.size >= 10:
                side = 1.0 if (ev.offset_sides > 0).sum() >= (ev.offset_sides < 0).sum() else -1.0
                sel = ev.offset_sides == side
                h = ev.offset_heights[sel]
                if h.size >= 10 and float(h.max() - h.min()) > 0.5:
                    slope = float(np.polyfit(h, ev.offsets_mm[sel], 1)[0])  # mm pro m
                    verdict.plumb_mm_per_m = abs(slope)
            if abs(med) >= cfg.deviation_flag_mm:
                verdict.verdict = VERDICT_DEVIATING
                verdict.rule_id = f"svb:{verdict.cls}:deviation"
                verdict.severity = _severity_for_deviation(abs(med))
                verdict.detail = (
                    f"{kind}-Abweichung {med:+.0f} mm zum Soll (Median ueber "
                    f"{ev.offsets_mm.size} Punkte)."
                )
        return verdict

    # Weder gebaut-Evidenz noch Fehlt-Evidenz -> nicht einsehbar
    verdict.verdict = VERDICT_NOT_VISIBLE
    verdict.rule_id = f"svb:{verdict.cls}:not_visible"
    if r["blocked"] > r["empty"]:
        verdict.detail = "Flaeche ueberwiegend verdeckt (Material/Geraete/Sichtlinie)."
    else:
        verdict.detail = "Flaeche im Scan nicht erfasst."
    return verdict


def _check_wall(
    wall: PlanElement,
    openings: Sequence[PlanElement],
    points: np.ndarray,
    origins: Optional[np.ndarray],
    scan_ids: Optional[np.ndarray],
    cfg: EngineConfig,
) -> ElementVerdict:
    frame = _wall_frame(wall)
    assert frame is not None and wall.aabb_min is not None and wall.aabb_max is not None
    thin, long_ax, thickness, _length = frame

    openings_uv = []
    for o in _openings_for_wall(wall, openings):
        openings_uv.append(
            (
                float(o.aabb_min[long_ax]),
                float(o.aabb_max[long_ax]),
                float(o.aabb_min[1]),
                float(o.aabb_max[1]),
            )
        )

    ev = _check_planar_surface(
        points,
        origins,
        scan_ids,
        normal_axis=thin,
        u_axis=long_ax,
        v_axis=1,
        center_n=float(wall.center[thin]),
        half_thickness=max(0.01, thickness / 2.0),
        u_range=(float(wall.aabb_min[long_ax]), float(wall.aabb_max[long_ax])),
        v_range=(float(wall.aabb_min[1]), float(wall.aabb_max[1])),
        openings_uv=openings_uv,
        cfg=cfg,
    )
    return _verdict_from_evidence(wall, ev, cfg, measure_plumb=True)


def _check_slab(
    slab: PlanElement,
    points: np.ndarray,
    origins: Optional[np.ndarray],
    scan_ids: Optional[np.ndarray],
    cfg: EngineConfig,
) -> ElementVerdict:
    assert slab.aabb_min is not None and slab.aabb_max is not None
    thickness = max(0.02, float(slab.size[1]))
    ev = _check_planar_surface(
        points,
        origins,
        scan_ids,
        normal_axis=1,
        u_axis=0,
        v_axis=2,
        center_n=float(slab.center[1]),
        half_thickness=thickness / 2.0,
        u_range=(float(slab.aabb_min[0]), float(slab.aabb_max[0])),
        v_range=(float(slab.aabb_min[2]), float(slab.aabb_max[2])),
        openings_uv=(),
        cfg=cfg,
    )
    return _verdict_from_evidence(slab, ev, cfg, measure_plumb=False)


def _check_opening(
    opening: PlanElement,
    walls: Sequence[PlanElement],
    points: np.ndarray,
    origins: Optional[np.ndarray],
    scan_ids: Optional[np.ndarray],
    cfg: EngineConfig,
) -> ElementVerdict:
    """Oeffnungs-Verdikt: Durchgangs-Evidenz = Oeffnung existiert.

    Eine an der Wandebene geschlossene Flaeche ist NICHT als "fehlt" wertbar
    (Tuerblatt/Verglasung vs. durchgebaute Wand sind geometrisch identisch) —
    dann ehrlich ``not_visible`` + Foto-Check-Empfehlung.
    """
    host = None
    for w in walls:
        if opening in _openings_for_wall(w, [opening]):
            host = w
            break
    if host is None or opening.aabb_min is None:
        return ElementVerdict(
            component_id=opening.id,
            node_path=opening.node_path,
            cls=opening.cls or "unknown",
            room_key=opening.room_key,
            verdict=VERDICT_NOT_VISIBLE,
            detail="Keine Traeger-Wand in Soll-Geometrie gefunden.",
        )
    frame = _wall_frame(host)
    assert frame is not None
    thin, long_ax, thickness, _ = frame
    ev = _check_planar_surface(
        points,
        origins,
        scan_ids,
        normal_axis=thin,
        u_axis=long_ax,
        v_axis=1,
        center_n=float(host.center[thin]),
        half_thickness=max(0.01, thickness / 2.0),
        u_range=(float(opening.aabb_min[long_ax]), float(opening.aabb_max[long_ax])),
        v_range=(float(opening.aabb_min[1]), float(opening.aabb_max[1])),
        openings_uv=(),
        cfg=cfg,
    )
    r = ev.ratios()
    verdict = ElementVerdict(
        component_id=opening.id,
        node_path=opening.node_path,
        cls=opening.cls or "unknown",
        room_key=opening.room_key,
        verdict=VERDICT_NOT_VISIBLE,
        hit_ratio=r["hit"],
        through_ratio=r["through"],
        blocked_ratio=r["blocked"],
        empty_ratio=r["empty"],
    )
    if r["cells"] < cfg.min_cells:
        verdict.detail = "Oeffnung kleiner als Mindest-Zellzahl."
        return verdict
    if r["through"] >= cfg.opening_through_ratio:
        verdict.verdict = VERDICT_BUILT
        verdict.detail = f"Durchgang nachgewiesen ({r['through']:.0%} der Oeffnungsflaeche)."
        return verdict
    if r["hit"] >= cfg.min_hit_ratio:
        verdict.verdict = VERDICT_NOT_VISIBLE
        verdict.rule_id = f"svb:{verdict.cls}:closed_surface"
        verdict.severity = "low"
        verdict.detail = (
            "Flaeche an der Oeffnung geschlossen — Tuerblatt/Verglasung oder nicht "
            "ausgefuehrte Oeffnung. Foto-Verifikation empfohlen."
        )
        return verdict
    verdict.rule_id = f"svb:{verdict.cls}:not_visible"
    verdict.detail = "Oeffnungsbereich im Scan nicht erfasst."
    return verdict


def _room_summaries(verdicts: Sequence[ElementVerdict]) -> List[RoomSummary]:
    by_room: Dict[str, List[ElementVerdict]] = {}
    for v in verdicts:
        if v.room_key:
            by_room.setdefault(v.room_key, []).append(v)
    out: List[RoomSummary] = []
    for room_key in sorted(by_room):
        vs = by_room[room_key]
        counts: Dict[str, int] = {}
        for v in vs:
            counts[v.verdict] = counts.get(v.verdict, 0) + 1
        verifiable = [v for v in vs if v.verdict != VERDICT_UNVERIFIABLE]
        confirmed = sum(1 for v in verifiable if v.verdict in (VERDICT_BUILT, VERDICT_DEVIATING))
        pct = (confirmed / len(verifiable) * 100.0) if verifiable else 0.0
        out.append(RoomSummary(room_key=room_key, label=None, counts=counts, verified_pct=pct))
    return out


def run_engine(
    points: np.ndarray,
    plan_elements: Sequence[PlanElement],
    *,
    origins: Optional[np.ndarray] = None,
    scan_ids: Optional[np.ndarray] = None,
    config: Optional[Dict] = None,
) -> Tuple[List[ElementVerdict], List[RoomSummary], Dict]:
    """Alle Soll-Bauteile gegen die (bereits registrierte) Wolke pruefen.

    ``points`` im kanonischen Plan-Frame (x, h, z). Returns
    (verdicts, room_summaries, summary_dict).
    """
    cfg = EngineConfig.from_dict(config)

    walls = [e for e in plan_elements if e.cls in ("wall", "column") and e.aabb_min is not None]
    openings = [e for e in plan_elements if e.cls in ("door", "window") and e.aabb_min is not None]
    slabs = [e for e in plan_elements if e.cls in ("floor", "ceiling") and e.aabb_min is not None]

    verdicts: List[ElementVerdict] = []
    for w in walls:
        verdicts.append(_check_wall(w, openings, points, origins, scan_ids, cfg))
    for o in openings:
        verdicts.append(_check_opening(o, walls, points, origins, scan_ids, cfg))
    for s in slabs:
        verdicts.append(_check_slab(s, points, origins, scan_ids, cfg))

    # Nicht geometrisch verifizierbare Klassen ehrlich ausweisen
    for e in plan_elements:
        if e.cls in ("room", None):
            continue
        if e.cls not in VERIFIABLE_CLASSES and e.aabb_min is not None:
            verdicts.append(
                ElementVerdict(
                    component_id=e.id,
                    node_path=e.node_path,
                    cls=e.cls,
                    room_key=e.room_key,
                    verdict=VERDICT_UNVERIFIABLE,
                    detail="Klasse per Punktwolken-Maske nicht belastbar pruefbar.",
                )
            )

    rooms = _room_summaries(verdicts)

    counts: Dict[str, int] = {}
    for v in verdicts:
        counts[v.verdict] = counts.get(v.verdict, 0) + 1
    verifiable = [v for v in verdicts if v.verdict != VERDICT_UNVERIFIABLE]
    seen = [v for v in verifiable if v.verdict != VERDICT_NOT_VISIBLE]
    confirmed = sum(1 for v in seen if v.verdict in (VERDICT_BUILT, VERDICT_DEVIATING))
    summary = {
        "elements_total": len(verdicts),
        "counts": counts,
        "coverage_pct": round(100.0 * len(seen) / len(verifiable), 1) if verifiable else 0.0,
        "confirmed_pct_of_seen": round(100.0 * confirmed / len(seen), 1) if seen else 0.0,
        "walls": len(walls),
        "openings": len(openings),
        "slabs": len(slabs),
    }
    return verdicts, rooms, summary
