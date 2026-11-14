"""Hartes Kriterium #1 — Window-Labels zu Bloecken clustern.

Eingabe: alle ``window`` / ``window_large`` Annotationen aus der
``annotations.json``.
Ausgabe: Liste von ``WindowBlock``. Jeder Block ist ein Cluster
benachbarter Labels und repraesentiert eine zusammenhaengende
Wand-Lucke.

Adjazenz-Regel:
  Zwei Labels gehoeren in denselben Block, wenn ihre Bounding-Boxes
  sich beruehren oder ueberlappen — bzw. wenn der euklidische
  Bbox-zu-Bbox-Abstand kleiner gleich ``adjacency_tol_m`` ist
  (Default = 10 cm, deckt Fenster mit Mullion-Trennstuck ab).

Pro Block liefert das Modul:
  - ``count``      : Anzahl fusionierter Labels = Anzahl Fenster im Gap
  - ``classes``    : Klassen der enthaltenen Labels (window / window_large)
  - ``bbox_m``     : Union-Bbox aller Labels in Metern
  - ``centroid_m`` : Mittelpunkt der Union-Bbox
  - ``label_indices`` : Indizes aus der Original-Annotations-Liste

Walls-immutable: Wand-Geometrie wird in diesem Schritt nicht beruehrt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple


WINDOW_LABEL_CLASSES = frozenset({"window", "window_large"})

# Default-Adjazenz: 10 cm. Damit gehoeren benachbarte Fenster mit
# duennem Mullion zwischen sich noch in denselben Block, isolierte
# Fenster mit ≥ 30 cm Wand-Stuck dazwischen aber nicht mehr.
ADJACENCY_TOL_M = 0.10


@dataclass
class WindowBlock:
    """Ein zusammenhaengender Cluster von Window-Labels."""
    label_indices: List[int]
    count: int
    classes: List[str]
    bbox_m: Tuple[float, float, float, float]   # (x_min, y_min, x_max, y_max)
    member_bboxes_m: List[Tuple[float, float, float, float]] = field(default_factory=list)
    centroid_m: Tuple[float, float] = field(init=False)

    def __post_init__(self) -> None:
        x0, y0, x1, y1 = self.bbox_m
        self.centroid_m = (0.5 * (x0 + x1), 0.5 * (y0 + y1))

    def width_m(self) -> float:
        return self.bbox_m[2] - self.bbox_m[0]

    def height_m(self) -> float:
        return self.bbox_m[3] - self.bbox_m[1]

    def long_side_m(self) -> float:
        return max(self.width_m(), self.height_m())


def _annotation_bbox_m(
    ann: Dict[str, Any],
    scale_xy: Tuple[float, float],
    px_to_m: float,
) -> Tuple[float, float, float, float] | None:
    sx, sy = scale_xy
    if ann.get("type") == "polygon":
        pts = ann.get("points") or []
        if not pts:
            return None
        xs = [float(p[0]) * sx * px_to_m for p in pts]
        ys = [float(p[1]) * sy * px_to_m for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))
    bbox = ann.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    x0, y0, x1, y1 = bbox
    xs = sorted([float(x0) * sx * px_to_m, float(x1) * sx * px_to_m])
    ys = sorted([float(y0) * sy * px_to_m, float(y1) * sy * px_to_m])
    return (xs[0], ys[0], xs[1], ys[1])


def _bbox_distance_m(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """Euklidischer Abstand zweier achsenparalleler Rechtecke. 0 wenn sie
    sich beruehren oder ueberlappen."""
    dx = max(0.0, max(a[0] - b[2], b[0] - a[2]))
    dy = max(0.0, max(a[1] - b[3], b[1] - a[3]))
    return (dx * dx + dy * dy) ** 0.5


def _union_bbox(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    return (min(a[0], b[0]), min(a[1], b[1]),
            max(a[2], b[2]), max(a[3], b[3]))


def find_window_blocks(
    annotations: Sequence[Dict[str, Any]],
    *,
    scale_xy: Tuple[float, float] = (1.0, 1.0),
    px_to_m: float = 1.0,
    adjacency_tol_m: float = ADJACENCY_TOL_M,
) -> List[WindowBlock]:
    """Cluster benachbarte Window-Labels zu Bloecken.

    Zwei Labels gehen in denselben Block wenn ihr Bbox-Abstand in Metern
    ``≤ adjacency_tol_m`` ist (transitiv via Connected-Components).

    Reihenfolge der Bloecke ist deterministisch: aufsteigend nach dem
    kleinsten enthaltenen Original-Index.
    """
    items: List[Tuple[int, str, Tuple[float, float, float, float]]] = []
    for idx, ann in enumerate(annotations):
        cls = ann.get("class") or ""
        if cls not in WINDOW_LABEL_CLASSES:
            continue
        bbox_m = _annotation_bbox_m(ann, scale_xy, px_to_m)
        if bbox_m is None:
            continue
        items.append((idx, cls, bbox_m))

    n = len(items)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if _bbox_distance_m(items[i][2], items[j][2]) <= adjacency_tol_m:
                union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    blocks: List[WindowBlock] = []
    for member_indices in clusters.values():
        member_indices.sort(key=lambda k: items[k][0])
        union_bbox = items[member_indices[0]][2]
        for k in member_indices[1:]:
            union_bbox = _union_bbox(union_bbox, items[k][2])
        blocks.append(WindowBlock(
            label_indices=[items[k][0] for k in member_indices],
            count=len(member_indices),
            classes=[items[k][1] for k in member_indices],
            bbox_m=union_bbox,
            member_bboxes_m=[items[k][2] for k in member_indices],
        ))

    blocks.sort(key=lambda b: b.label_indices[0])
    return blocks
