"""Point cloud I/O with format-specific loaders and numpy fallbacks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Supported file extensions
SUPPORTED_EXTENSIONS = {".ply", ".las", ".laz", ".e57", ".xyz", ".pts", ".zip"}

#: Obergrenze fuer den ASCII-PLY-Fallback (Issue #47). ASCII-PLY braucht rund
#: 25 Byte je Punkt auf der Platte und ein Python-Listenobjekt je Punkt im
#: Speicher — 50 Mio. Punkte sind bereits ueber 1 GB an Zwischenobjekten und
#: liegen weit ueber dem, was ``max_input_points`` (2 Mio.) danach behaelt. Wer
#: mehr braucht, liefert binaeres PLY oder LAS/LAZ; beide Pfade sind durch
#: ``np.frombuffer`` implizit an die reale Dateigroesse gebunden.
MAX_ASCII_VERTICES = 50_000_000


def load_point_cloud(
    path: str,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load a point cloud file and return (points, colors).

    Returns:
        points: (N, 3) float64 array of XYZ coordinates.
        colors: (N, 3) float32 array of RGB in [0..1], or None.
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".ply":
        return _load_ply(p)
    if ext in (".las", ".laz"):
        return _load_las(p)
    if ext == ".e57":
        return _load_e57(p)
    if ext in (".xyz", ".pts"):
        return _load_xyz(p)
    raise ValueError(f"Unsupported point cloud format: {ext}")


def detect_format(path: str) -> str:
    """Return the normalised extension of a point cloud file."""
    return Path(path).suffix.lower()


def load_point_cloud_multi(
    path: str,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load a point cloud incl. per-scan sensor poses where the format has them.

    E57 ist das einzige unterstuetzte Format mit Posen: ALLE Scans der Datei
    werden gelesen, jeweils mit der Scanner-Pose (Rotation+Translation) in
    Datei-Weltkoordinaten transformiert; die Translation ist die
    Sensor-Position des Scans. Fuer alle anderen Formate delegiert die
    Funktion an :func:`load_point_cloud` (origins/scan_ids = ``None``).

    Returns:
        points: (N, 3) float64.
        colors: (N, 3) float32 [0..1] oder None.
        origins: (M, 3) float64 Sensor-Positionen oder None.
        scan_ids: (N,) int32 Index in origins oder None.
    """
    p = Path(path)
    if p.suffix.lower() == ".e57":
        try:
            return _load_e57_multi(p)
        except ImportError:
            raise
        except Exception:
            # Einzelscan-Fallback (defekte Posen/Exoten): Verhalten wie bisher.
            points, colors = _load_e57(p)
            return points, colors, None, None
    points, colors = load_point_cloud(path)
    return points, colors, None, None


# ---------------------------------------------------------------------------
# PLY
# ---------------------------------------------------------------------------

def _load_ply(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load PLY via open3d, fall back to numpy parser."""
    try:
        return _load_ply_open3d(path)
    except Exception:
        return _load_ply_numpy(path)


def _load_ply_open3d(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float64)
    if points.shape[0] == 0:
        raise ValueError("open3d read 0 points")
    colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
    return points, colors


def _load_ply_numpy(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Minimal ASCII/binary-little-endian PLY parser (numpy only)."""
    with open(path, "rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        vertex_count = 0
        fmt = "ascii"
        prop_names: list[str] = []

        for line in header_lines:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("property"):
                parts = line.split()
                prop_names.append(parts[-1])

        if vertex_count == 0:
            raise ValueError("PLY file has 0 vertices")

        x_idx = prop_names.index("x") if "x" in prop_names else 0
        y_idx = prop_names.index("y") if "y" in prop_names else 1
        z_idx = prop_names.index("z") if "z" in prop_names else 2
        has_rgb = all(c in prop_names for c in ("red", "green", "blue"))

        if fmt == "ascii":
            # Issue #47: ``vertex_count`` ist eine Zahl AUS DER DATEI und steuerte
            # die Schleife ungeprueft. Zwei Fehler zugleich:
            #
            # 1. Kein EOF-Abbruch — am Dateiende liefert ``readline()`` ``b''``,
            #    ``.split()`` ist ``[]``, und ``rows.append([])`` laeuft trotzdem
            #    bis ``vertex_count`` durch. Ein ~80-Byte-Header mit
            #    ``element vertex 2000000000`` allokierte damit Gigabytes.
            # 2. Kein Deckel. ``max_points`` (steps/preprocess.py) greift erst
            #    NACH dem Laden — also zu spaet, um das Laden zu ueberleben.
            #
            # Der Schaden lief zudem in einem Thread mit 300-s-Timeout; Threads
            # sind nicht killbar, der verwaiste Thread allokierte danach weiter.
            rows = []
            for _ in range(vertex_count):
                zeile = f.readline()
                if not zeile:
                    break  # EOF: der Header hat mehr versprochen als die Datei haelt
                vals = zeile.decode("ascii", errors="replace").split()
                if not vals:
                    continue
                rows.append([float(v) for v in vals])
                if len(rows) > MAX_ASCII_VERTICES:
                    raise ValueError(
                        f"PLY hat mehr als {MAX_ASCII_VERTICES} ASCII-Punkte — "
                        "bitte als binaeres PLY, LAS/LAZ oder E57 hochladen."
                    )
            if not rows:
                raise ValueError(f"PLY-Header nennt {vertex_count} Punkte, die Datei enthaelt keine.")
            data = np.array(rows, dtype=np.float64)
        else:
            # binary_little_endian -- assume all float32 properties
            row_bytes = len(prop_names) * 4
            raw = f.read(vertex_count * row_bytes)
            data = np.frombuffer(raw, dtype=np.float32).reshape(vertex_count, len(prop_names)).astype(np.float64)

        points = np.column_stack([data[:, x_idx], data[:, y_idx], data[:, z_idx]])

        colors = None
        if has_rgb:
            r_idx = prop_names.index("red")
            g_idx = prop_names.index("green")
            b_idx = prop_names.index("blue")
            rgb = np.column_stack([data[:, r_idx], data[:, g_idx], data[:, b_idx]])
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            colors = rgb.astype(np.float32)

        return points, colors


# ---------------------------------------------------------------------------
# LAS / LAZ
# ---------------------------------------------------------------------------

def _load_las(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load LAS/LAZ via laspy."""
    try:
        import laspy
    except ImportError:
        raise ImportError(
            "laspy is required for LAS/LAZ files. "
            "Install with: pip install 'planproof-core[scan]'"
        )

    las = laspy.read(str(path))
    points = np.column_stack([
        las.x.astype(np.float64),
        las.y.astype(np.float64),
        las.z.astype(np.float64),
    ])

    colors = None
    if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
        rgb = np.column_stack([
            las.red.astype(np.float32),
            las.green.astype(np.float32),
            las.blue.astype(np.float32),
        ])
        max_val = rgb.max()
        if max_val > 1.0:
            rgb = rgb / max_val
        colors = rgb

    return points, colors


# ---------------------------------------------------------------------------
# E57
# ---------------------------------------------------------------------------

def _load_e57(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load E57 via pye57 (optional dependency)."""
    try:
        import pye57
    except ImportError:
        raise ImportError(
            "pye57 is required for E57 files. "
            "Install with: pip install 'planproof-core[scan-e57]'"
        )

    e57 = pye57.E57(str(path))
    data = e57.read_scan_raw(0)

    points = np.column_stack([
        data["cartesianX"].astype(np.float64),
        data["cartesianY"].astype(np.float64),
        data["cartesianZ"].astype(np.float64),
    ])

    colors = None
    if "colorRed" in data and "colorGreen" in data and "colorBlue" in data:
        rgb = np.column_stack([
            data["colorRed"].astype(np.float32),
            data["colorGreen"].astype(np.float32),
            data["colorBlue"].astype(np.float32),
        ])
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        colors = rgb

    return points, colors


def _e57_scan_pose(header) -> Tuple[np.ndarray, np.ndarray]:
    """(rotation 3x3, translation 3,) einer E57-Scan-Pose, defensiv gelesen.

    pye57-Versionen exponieren die Pose unterschiedlich (``rotation_matrix``
    vs. ``rotation``-Quaternion); fehlende Pose = Identitaet.
    """
    rot = np.eye(3)
    trans = np.zeros(3)
    try:
        rm = getattr(header, "rotation_matrix", None)
        if rm is not None:
            rot = np.asarray(rm, dtype=np.float64).reshape(3, 3)
        else:
            q = getattr(header, "rotation", None)
            if q is not None:
                q = np.asarray(q, dtype=np.float64).reshape(-1)
                if q.shape[0] == 4:
                    w, x, y, z = q
                    rot = np.array(
                        [
                            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                        ]
                    )
        tv = getattr(header, "translation", None)
        if tv is not None:
            trans = np.asarray(tv, dtype=np.float64).reshape(3)
    except Exception:
        rot, trans = np.eye(3), np.zeros(3)
    return rot, trans


def _load_e57_multi(
    path: Path,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """ALLE Scans einer E57-Datei inkl. Posen (Scan-vs-BIM: Okklusions-Check).

    Punkte werden mit der jeweiligen Scan-Pose in Datei-Weltkoordinaten
    gebracht; die Pose-Translation ist die Sensor-Position des Scans.
    """
    try:
        import pye57
    except ImportError:
        raise ImportError(
            "pye57 is required for E57 files. "
            "Install with: pip install 'planproof-core[scan-e57]'"
        )

    e57 = pye57.E57(str(path))
    scan_count = int(getattr(e57, "scan_count", 1) or 1)

    all_points: list = []
    all_colors: list = []
    all_ids: list = []
    origins: list = []
    any_colors = False

    for i in range(scan_count):
        data = e57.read_scan_raw(i)
        pts = np.column_stack(
            [
                data["cartesianX"].astype(np.float64),
                data["cartesianY"].astype(np.float64),
                data["cartesianZ"].astype(np.float64),
            ]
        )
        rot, trans = _e57_scan_pose(e57.get_header(i))
        pts = pts @ rot.T + trans
        origins.append(trans)
        all_points.append(pts)
        all_ids.append(np.full(pts.shape[0], i, dtype=np.int32))

        if "colorRed" in data and "colorGreen" in data and "colorBlue" in data:
            rgb = np.column_stack(
                [
                    data["colorRed"].astype(np.float32),
                    data["colorGreen"].astype(np.float32),
                    data["colorBlue"].astype(np.float32),
                ]
            )
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            all_colors.append(rgb)
            any_colors = True
        else:
            all_colors.append(None)

    points = np.vstack(all_points)
    scan_ids = np.concatenate(all_ids)
    colors = None
    if any_colors:
        filled = [
            c if c is not None else np.zeros((p.shape[0], 3), dtype=np.float32)
            for c, p in zip(all_colors, all_points)
        ]
        colors = np.vstack(filled)
    return points, colors, np.asarray(origins, dtype=np.float64), scan_ids


# ---------------------------------------------------------------------------
# XYZ / PTS (whitespace-delimited text)
# ---------------------------------------------------------------------------

def _load_xyz(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load space/tab-delimited text point cloud (XYZ or XYZRGB)."""
    # Skip comment lines and detect delimiter
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            # PTS format: first line may be point count
            parts = line.split()
            if len(parts) == 1:
                continue  # skip point-count header
            try:
                vals = [float(v) for v in parts]
                rows.append(vals)
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No valid point data in {path}")

    data = np.array(rows, dtype=np.float64)
    points = data[:, :3]

    colors = None
    if data.shape[1] >= 6:
        rgb = data[:, 3:6].astype(np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        colors = rgb

    return points, colors
