"""Scan-vs-BIM Orchestrator: Job-Inputs -> Registrierung -> Verdikte -> Artefakte.

Rein lokal (Dateien rein, Dateien raus) — Cloud-/DB-Anbindung macht die
Dispatch-Route (API) bzw. der Worker-Runner. Artefakte:

* ``scan_vs_bim_report.json``   — Registrierung + Vier-Klassen-Verdikte +
                                  Raum-Aggregat + Summary (Schema v1)
* ``registered_preview.ply``    — registrierte, ausgeduennte Wolke im
                                  Plan-Welt-Frame (Viewer-Overlay)
* ``meta.json``                 — Metrik-Snapshot fuer Run-History/Status
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..scan_to_bim.point_cloud_io import load_point_cloud_multi
from .engine import EngineConfig, run_engine
from .inputs import find_inputs, load_manifest, load_plan_elements
from .registration import register
from .types import Registration, ScanCloud, ScanVsBimReport

REPORT_SCHEMA_VERSION = 1
REPORT_FILENAME = "scan_vs_bim_report.json"
PREVIEW_FILENAME = "registered_preview.ply"
_PREVIEW_VOXEL_M = 0.05
_PREVIEW_MAX_POINTS = 400_000


@dataclass
class ScanVsBimRunResult:
    ok: bool
    error: Optional[str] = None
    report_path: Optional[str] = None
    preview_path: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def _voxel_indices(points: np.ndarray, voxel: float, max_points: int) -> np.ndarray:
    """Indizes einer deterministischen Voxel-Teilstichprobe (1 Punkt/Voxel)."""
    if points.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    idx = np.sort(idx)
    if idx.shape[0] > max_points:
        step = int(np.ceil(idx.shape[0] / max_points))
        idx = idx[::step]
    return idx


def _write_preview_ply(
    path: Path, points: np.ndarray, colors: Optional[np.ndarray]
) -> None:
    """Binary-LE PLY (xyz float32, optional rgb uchar) — Viewer-Overlay."""
    n = points.shape[0]
    has_rgb = colors is not None and colors.shape[0] == n
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += ["property float x", "property float y", "property float z"]
    if has_rgb:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header += ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        pts32 = points.astype(np.float32)
        if has_rgb:
            rgb8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
            row = struct.Struct("<3f3B")
            f.write(b"".join(row.pack(*p, *c) for p, c in zip(pts32, rgb8)))
        else:
            f.write(pts32.tobytes())


def _canonical_to_world(points: np.ndarray, up_axis: str) -> np.ndarray:
    """Kanonisch (x, h, z) -> GLB-Welt-Frame der Plan-Version."""
    if up_axis.upper() == "Z":
        return points[:, [0, 2, 1]]
    return points


def run_scan_vs_bim(
    input_paths: List[Path],
    output_dir: Path,
    work_dir: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
    logger_fn: Optional[Callable[[str], None]] = None,
) -> ScanVsBimRunResult:
    t0 = time.time()
    log = logger_fn or (lambda _msg: None)
    cfg = dict(config or {})
    warnings: List[str] = []

    try:
        manifest_path, components_path, scan_path = find_inputs(list(input_paths))
        manifest = load_manifest(manifest_path)
        up_axis = str(manifest.get("up_axis") or "Y")
        options = {**(manifest.get("options") or {}), **cfg}

        log(f"scan_vs_bim: lade Soll-Geometrie ({components_path.name})")
        elements = load_plan_elements(components_path, up_axis)
        usable = [e for e in elements if e.aabb_min is not None]
        log(f"scan_vs_bim: {len(usable)}/{len(elements)} Soll-Bauteile mit AABB")
        if not usable:
            raise ValueError("plan_components.json enthaelt keine Bauteile mit aabb_min/aabb_max")

        log(f"scan_vs_bim: lade Punktwolke ({scan_path.name})")
        points, colors, origins, scan_ids = load_point_cloud_multi(str(scan_path))
        cloud = ScanCloud(points=points, colors=colors, origins=origins, scan_ids=scan_ids)
        log(f"scan_vs_bim: {cloud.count} Punkte" + (f", {origins.shape[0]} Scan-Posen" if origins is not None else ", keine Posen"))

        log("scan_vs_bim: Registrierung (Boden + Manhattan + Korrelation + ICP)")
        reg: Registration = register(cloud.points, usable, options)
        warnings.extend(reg.warnings)
        log(
            f"scan_vs_bim: registriert — rms {reg.rms_m * 1000:.0f} mm, "
            f"Inlier {reg.inlier_fraction:.0%}, yaw {np.degrees(reg.yaw_rad):.1f} Grad"
        )

        pts_c = reg.apply(cloud.points)
        origins_c = reg.apply(cloud.origins) if cloud.origins is not None else None

        engine_cfg = EngineConfig.from_dict(options)
        idx = _voxel_indices(pts_c, engine_cfg.engine_voxel_m, engine_cfg.engine_max_points)
        pts_e = pts_c[idx]
        ids_e = cloud.scan_ids[idx] if cloud.scan_ids is not None else None
        log(f"scan_vs_bim: Engine auf {pts_e.shape[0]} Punkten ({len(usable)} Bauteile)")

        verdicts, rooms, summary = run_engine(
            pts_e, usable, origins=origins_c, scan_ids=ids_e, config=options
        )

        report = ScanVsBimReport(
            schema_version=REPORT_SCHEMA_VERSION,
            manifest={
                "project_id": manifest.get("project_id"),
                "plan_version_id": manifest.get("plan_version_id"),
                "scan_filename": scan_path.name,
                "up_axis": up_axis,
                "options": {k: v for k, v in options.items() if isinstance(v, (int, float, str, bool))},
            },
            registration=reg,
            elements=verdicts,
            rooms=rooms,
            summary=summary,
            warnings=warnings,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / REPORT_FILENAME
        report_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Viewer-Overlay: registrierte Wolke zurueck in den GLB-Welt-Frame
        pidx = _voxel_indices(pts_c, _PREVIEW_VOXEL_M, _PREVIEW_MAX_POINTS)
        preview_path = output_dir / PREVIEW_FILENAME
        _write_preview_ply(
            preview_path,
            _canonical_to_world(pts_c[pidx], up_axis),
            cloud.colors[pidx] if cloud.colors is not None else None,
        )

        metrics = {
            "points_total": cloud.count,
            "points_engine": int(pts_e.shape[0]),
            "scan_poses": int(cloud.origins.shape[0]) if cloud.origins is not None else 0,
            "registration_rms_mm": round(reg.rms_m * 1000.0, 1),
            "registration_inlier_fraction": round(reg.inlier_fraction, 4),
            "unit_scale": reg.unit_scale,
            **{f"verdict_{k}": v for k, v in summary.get("counts", {}).items()},
            "coverage_pct": summary.get("coverage_pct"),
            "confirmed_pct_of_seen": summary.get("confirmed_pct_of_seen"),
            "duration_s": round(time.time() - t0, 3),
        }
        (output_dir / "meta.json").write_text(
            json.dumps(
                {"pipeline": "scan_vs_bim", "schema_version": REPORT_SCHEMA_VERSION, **metrics},
                indent=2,
            ),
            encoding="utf-8",
        )

        return ScanVsBimRunResult(
            ok=True,
            report_path=str(report_path),
            preview_path=str(preview_path),
            metrics=metrics,
            warnings=warnings,
        )
    except Exception as e:  # noqa: BLE001 — Fehler traegt der Runner als Job-Fehler
        return ScanVsBimRunResult(ok=False, error=str(e), warnings=warnings)
