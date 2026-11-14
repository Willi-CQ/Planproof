"""Hartes Kriterium #1 — Phase 3: Fenster IN den Block platzieren.

Pro ``ClippedWindowBlock`` mit ``count = N`` werden genau N Fenster
in den Block gesetzt. Die Block-Achse wird dafuer in N gleich grosse
Sub-Segmente unterteilt (Equipartition). Jedes Sub-Segment ist ein
``PlacedWindow`` mit:

  - 4-Punkt-Polygon (CCW) im Welt-Frame
  - tangentialer Sub-Span (s_a, s_b) auf der Block-Achse
  - dem Wand-Dicke-Feld vom Block geerbt
  - der Klasse aus den User-Labels (geordnet nach
    Tangent-Projektion ihrer Bbox-Centroids → robust gegen Annotator-
    Drift, klassen-konsistent zur Realitaet auf der Wand)

Klassenzuordnung:
  Member-Bboxes werden nach ihrer Projektion auf die Block-Tangente
  sortiert. Das ergibt eine Reihenfolge "links → rechts" entlang der
  Wand-Achse. Diese Reihenfolge bestimmt, welche Klasse das i-te
  Equipartition-Segment bekommt.

Hartes Kriterium #1 wird durch ``verify_placement_criterion_1``
geprueft. Der Verifier fuehrt drei Pruefungen pro Block durch:

  1. **count parity**       — exakt ``count`` PlacedWindows pro Block
  2. **inside polygon**     — jedes Fenster-Polygon liegt vollstaendig
                              im Block-Polygon (Toleranz 1 mm)
  3. **coverage**           — die Vereinigung der Fenster-Sub-Spans
                              deckt das Block-Achsen-Intervall
                              luckenlos und ueberlappungsfrei ab

Walls-immutable: Geometrie nur lesen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from .window_block_clipper import ClippedWindowBlock


# Numerische Toleranz fuer "innerhalb des Block-Polygons" und
# "luckenlose Abdeckung". 1 mm ist deutlich strenger als jede
# Annotator-Genauigkeit.
CRITERION_TOLERANCE_M = 1e-3


@dataclass
class PlacedWindow:
    """Ein einzelnes Fenster, equipartitioniert in einem Block."""
    block_idx: int                       # Index in clipped_blocks
    sub_index: int                       # 0..count-1 innerhalb des Blocks
    label_class: str                     # window | window_large
    polygon_m: List[Tuple[float, float]] # 4 Eckpunkte CCW im Welt-Frame
    block_count: int = 1                 # Gesamtanzahl Sub-Fenster im Block
    axis_s_lo_m: float = 0.0
    axis_s_hi_m: float = 0.0
    wall_thickness_m: float = 0.0
    # Block-Frame fuer Mesh-Builder (Frames + perp-Extension fuer Wand-Lueckenschluss)
    axis_origin_m: Tuple[float, float] = (0.0, 0.0)
    axis_tangent: Tuple[float, float] = (1.0, 0.0)
    axis_perp: Tuple[float, float] = (0.0, 1.0)
    axis_perp_lo_m: float = 0.0
    axis_perp_hi_m: float = 0.0
    centroid_m: Tuple[float, float] = field(init=False)

    @property
    def is_first_in_block(self) -> bool:
        return self.sub_index == 0

    @property
    def is_last_in_block(self) -> bool:
        return self.sub_index == self.block_count - 1

    def __post_init__(self) -> None:
        self.centroid_m = (
            sum(p[0] for p in self.polygon_m) / len(self.polygon_m),
            sum(p[1] for p in self.polygon_m) / len(self.polygon_m),
        )

    @property
    def width_m(self) -> float:
        return self.axis_s_hi_m - self.axis_s_lo_m


@dataclass
class CriterionResult:
    """Ergebnis der harten-Kriterium-Pruefung."""
    passed: bool
    n_blocks: int
    n_windows: int
    issues: List[str] = field(default_factory=list)
    per_block_stats: List[Dict[str, float]] = field(default_factory=list)


def _project(point: Tuple[float, float],
             origin: Tuple[float, float],
             axis: Tuple[float, float]) -> float:
    return (point[0] - origin[0]) * axis[0] + (point[1] - origin[1]) * axis[1]


def _bbox_centroid(bbox: Tuple[float, float, float, float]
                   ) -> Tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def _members_sorted_by_tangent(
    cb: ClippedWindowBlock,
) -> List[Tuple[float, str]]:
    """Member-Bboxes nach ihrer Tangent-Projektion sortieren."""
    origin = cb.axis_origin_m
    tangent = cb.axis_tangent
    pairs: List[Tuple[float, str]] = []
    for bbox, cls in zip(cb.block.member_bboxes_m, cb.block.classes):
        s = _project(_bbox_centroid(bbox), origin, tangent)
        pairs.append((s, cls))
    pairs.sort(key=lambda p: p[0])
    return pairs


def place_windows_in_blocks(
    clipped_blocks: Sequence[ClippedWindowBlock],
) -> List[PlacedWindow]:
    """Equipartition-Platzierung: pro Block exakt ``count`` Fenster.

    Algorithmus:
      1. Teile ``[s_lo, s_hi]`` in N gleich grosse Sub-Segmente.
      2. Jedes Sub-Segment wird zu einem ``PlacedWindow`` mit
         4-Eckpunkt-Polygon im Block-Frame, im Welt-Frame ausgegeben.
      3. Klassen-Zuordnung: Member-Bboxes nach Tangent-Projektion
         sortieren, i-tes Segment bekommt die i-te Klasse.
    """
    placed: List[PlacedWindow] = []
    for block_idx, cb in enumerate(clipped_blocks):
        count = cb.count
        if count <= 0:
            continue

        sorted_classes = [cls for _, cls in _members_sorted_by_tangent(cb)]
        # Falls Anzahl der Member-Klassen != count (defensiv), padde mit
        # 'window' — sollte in der Praxis nie passieren da count aus den
        # Membern kommt.
        while len(sorted_classes) < count:
            sorted_classes.append("window")
        sorted_classes = sorted_classes[:count]

        s_lo = cb.axis_s_lo_m
        s_hi = cb.axis_s_hi_m
        seg_len = (s_hi - s_lo) / count

        ox, oy = cb.axis_origin_m
        tx, ty = cb.axis_tangent
        px, py = cb.axis_perp
        perp_lo = cb.axis_perp_lo_m
        perp_hi = cb.axis_perp_hi_m

        def _world(s: float, t: float) -> Tuple[float, float]:
            return (ox + s * tx + t * px, oy + s * ty + t * py)

        for i in range(count):
            s_a = s_lo + i * seg_len
            s_b = s_lo + (i + 1) * seg_len
            polygon = [
                _world(s_a, perp_lo),
                _world(s_b, perp_lo),
                _world(s_b, perp_hi),
                _world(s_a, perp_hi),
            ]
            placed.append(PlacedWindow(
                block_idx=block_idx,
                sub_index=i,
                block_count=count,
                label_class=sorted_classes[i],
                polygon_m=polygon,
                axis_s_lo_m=float(s_a),
                axis_s_hi_m=float(s_b),
                wall_thickness_m=cb.wall_thickness_m,
                axis_origin_m=cb.axis_origin_m,
                axis_tangent=cb.axis_tangent,
                axis_perp=cb.axis_perp,
                axis_perp_lo_m=float(perp_lo),
                axis_perp_hi_m=float(perp_hi),
            ))
    return placed


def _point_in_axis_aligned_rect_block_frame(
    s: float, p: float, cb: ClippedWindowBlock,
    *, tolerance: float,
) -> bool:
    return (cb.axis_s_lo_m - tolerance <= s <= cb.axis_s_hi_m + tolerance
            and cb.axis_perp_lo_m - tolerance <= p <= cb.axis_perp_hi_m + tolerance)


def verify_placement_criterion_1(
    clipped_blocks: Sequence[ClippedWindowBlock],
    placed: Sequence[PlacedWindow],
    *,
    tolerance_m: float = CRITERION_TOLERANCE_M,
) -> CriterionResult:
    """Hartes Kriterium #1 verifizieren.

    Drei Pruefungen pro Block:
      1. count parity     — exakt ``count`` PlacedWindows
      2. inside polygon   — jedes Fenster-Polygon liegt im Block-Polygon
      3. coverage         — Sub-Spans decken Achsen-Intervall lueckenlos
                            und ueberlappungsfrei ab
    """
    issues: List[str] = []
    per_block: List[Dict[str, float]] = []
    by_block: Dict[int, List[PlacedWindow]] = {}
    for w in placed:
        by_block.setdefault(w.block_idx, []).append(w)

    for block_idx, cb in enumerate(clipped_blocks):
        windows = by_block.get(block_idx, [])
        n = len(windows)
        stats: Dict[str, float] = {
            "block_idx": float(block_idx),
            "expected_count": float(cb.count),
            "actual_count": float(n),
        }

        # 1. Count parity
        if n != cb.count:
            issues.append(
                f"block {block_idx}: count parity FAILED — "
                f"expected {cb.count}, got {n}"
            )

        # 2. Inside polygon
        inside_failures = 0
        for w in windows:
            for vx, vy in w.polygon_m:
                s = _project((vx, vy), cb.axis_origin_m, cb.axis_tangent)
                p = _project((vx, vy), cb.axis_origin_m, cb.axis_perp)
                if not _point_in_axis_aligned_rect_block_frame(
                    s, p, cb, tolerance=tolerance_m,
                ):
                    inside_failures += 1
                    break
        stats["inside_failures"] = float(inside_failures)
        if inside_failures > 0:
            issues.append(
                f"block {block_idx}: {inside_failures} window(s) "
                f"have a vertex outside the block polygon"
            )

        # 3. Coverage — Sub-Spans sollten luckenlos den Achsen-Bereich
        #    abdecken und sich nicht ueberlappen.
        if n > 0:
            sorted_w = sorted(windows, key=lambda w: w.axis_s_lo_m)
            gaps_total = 0.0
            overlaps_total = 0.0
            cursor = cb.axis_s_lo_m
            for w in sorted_w:
                if w.axis_s_lo_m > cursor + tolerance_m:
                    gaps_total += w.axis_s_lo_m - cursor
                elif w.axis_s_lo_m < cursor - tolerance_m:
                    overlaps_total += cursor - w.axis_s_lo_m
                cursor = max(cursor, w.axis_s_hi_m)
            tail_gap = cb.axis_s_hi_m - cursor
            if tail_gap > tolerance_m:
                gaps_total += tail_gap
            stats["coverage_gap_m"] = float(gaps_total)
            stats["coverage_overlap_m"] = float(overlaps_total)
            if gaps_total > tolerance_m:
                issues.append(
                    f"block {block_idx}: coverage gap of "
                    f"{gaps_total*1000:.1f} mm along axis"
                )
            if overlaps_total > tolerance_m:
                issues.append(
                    f"block {block_idx}: coverage overlap of "
                    f"{overlaps_total*1000:.1f} mm along axis"
                )

        per_block.append(stats)

    return CriterionResult(
        passed=(len(issues) == 0),
        n_blocks=len(clipped_blocks),
        n_windows=len(placed),
        issues=issues,
        per_block_stats=per_block,
    )
