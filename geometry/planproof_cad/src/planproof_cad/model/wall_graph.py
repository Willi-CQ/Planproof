"""Wall graph - junction + parametric-opening representation.

The SceneModel's polygon/CSG path cannot cleanly answer two questions that
every real BIM workflow needs:

  * "Which wall does this opening belong to?"
  * "Where along that wall does it sit?"

Both answers are implicit in a planar graph where nodes are wall junctions
(endpoint, T, L, X) and edges are wall runs carrying {thickness, height}.
Openings attach to edges as parametric tuples `(offset, width, sill, head)`.
This is the representation every commercial BIM system and every 2022+
academic floorplan paper converges on (Raster-to-Graph 2024, PolyRoom ECCV
2024, Cloud2BIM 2025). It also serialises 1:1 to IFC:

    WallEdge           -> IfcWallStandardCase (Axis + SweptSolid)
    ParametricOpening  -> IfcOpeningElement via IfcRelVoidsElement
                           filled by IfcDoor/IfcWindow via IfcRelFillsElement

Geometry is stored in metre y-up (matches SceneModel and the glTF exporter).
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Point2 = Tuple[float, float]


def _clip_segment_to_aabb(
    a: Point2,
    b: Point2,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> Tuple[Optional[float], Optional[float]]:
    """Liang-Barsky clip of segment A->B against axis-aligned rect.
    Returns (t_in, t_out) along A->B in [0,1], or (None, None) if no
    intersection. Used by ``edge_through_bbox`` for manual opening attach.
    """
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, a[0] - x1), (dx, x2 - a[0]), (-dy, a[1] - y1), (dy, y2 - a[1])):
        if abs(p) < 1e-12:
            if q < 0:
                return None, None
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return None, None
            if r > t0:
                t0 = r
        else:
            if r < t0:
                return None, None
            if r < t1:
                t1 = r
    if t1 <= t0:
        return None, None
    return t0, t1


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------

NODE_KINDS = frozenset({"endpoint", "L", "T", "X", "cross"})


@dataclass
class WallNode:
    """A junction in the wall graph.

    `kind` is purely advisory - it is recomputed from edge degree at any
    time via ``WallGraph.refresh_node_kinds``. Callers may leave it on the
    default "endpoint" and let the graph derive the real value.
    """

    id: str
    xy: Point2
    kind: str = "endpoint"
    attrs: Dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Parametric openings
# --------------------------------------------------------------------------

OPENING_KINDS = frozenset({"door", "window"})


@dataclass
class ParametricOpening:
    """A door or window attached to a WallEdge.

    Offset is measured from the edge's node_a along the edge direction.
    Width is along the edge (the physical opening width). Sill/head are
    vertical (metres above the floor).

    The mesh exporter uses exactly these four numbers - no nearest-wall
    lookup, no bbox rebuild, no rotation.
    """

    id: str
    kind: str  # "door" | "window"
    offset_m: float  # along-edge position of opening centre (0..edge_length)
    width_m: float  # along-edge extent
    sill_m: float = 0.0  # bottom of opening above floor
    head_m: float = 2.10  # top of opening above floor
    subtype: Optional[str] = None  # "door_single" / "door_double" / "door_sliding" / "window_large"
    confidence: float = 1.0
    source: str = "manual"  # "manual" | "yolo" | "heuristic"
    cuts_wall: bool = True  # False for surface-mount (e.g. sliding doors)
    attrs: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Sliding doors run in front of the wall on a rail - they do NOT
        # cut through it. Auto-flip cuts_wall so the mesh exporter knows
        # to render a panel in front instead of subtracting an opening.
        if self.subtype == "door_sliding":
            self.cuts_wall = False

    def start_m(self) -> float:
        return self.offset_m - self.width_m * 0.5

    def end_m(self) -> float:
        return self.offset_m + self.width_m * 0.5


# --------------------------------------------------------------------------
# Edges
# --------------------------------------------------------------------------


@dataclass
class WallEdge:
    """A wall run between two WallNodes.

    The edge stores thickness + height so the mesh exporter can build an
    extruded rectangle directly. Openings are owned by the edge - deleting
    the edge deletes its openings.
    """

    id: str
    node_a: str
    node_b: str
    thickness_m: float = 0.12
    height_m: float = 2.80
    wall_type: str = "interior"  # "interior" | "exterior"
    source: str = "manual"
    confidence: float = 1.0
    openings: List[ParametricOpening] = field(default_factory=list)
    attrs: Dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------


@dataclass
class WallGraph:
    """Planar graph of wall junctions + parametric openings.

    Internally a plain dict-of-dicts - cheap to build, cheap to mutate,
    cheap to serialise. NetworkX is used only when actually needed
    (outer face, planarity check, shortest path).
    """

    nodes: Dict[str, WallNode] = field(default_factory=dict)
    edges: Dict[str, WallEdge] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)

    # -- node ops ---------------------------------------------------------

    def add_node(
        self, xy: Point2, node_id: Optional[str] = None, kind: str = "endpoint"
    ) -> WallNode:
        nid = node_id or f"n_{len(self.nodes)}_{uuid.uuid4().hex[:6]}"
        node = WallNode(id=nid, xy=(float(xy[0]), float(xy[1])), kind=kind)
        self.nodes[nid] = node
        return node

    def find_or_create_node(self, xy: Point2, snap_radius_m: float) -> WallNode:
        """Return existing node within `snap_radius_m`, else create one."""
        if snap_radius_m > 0 and self.nodes:
            r2 = snap_radius_m * snap_radius_m
            best: Optional[Tuple[float, WallNode]] = None
            for n in self.nodes.values():
                dx = n.xy[0] - xy[0]
                dy = n.xy[1] - xy[1]
                d2 = dx * dx + dy * dy
                if d2 <= r2 and (best is None or d2 < best[0]):
                    best = (d2, n)
            if best is not None:
                return best[1]
        return self.add_node(xy)

    def node_degree(self, node_id: str) -> int:
        n = 0
        for e in self.edges.values():
            if e.node_a == node_id or e.node_b == node_id:
                n += 1
        return n

    def refresh_node_kinds(self) -> None:
        """Recompute node.kind from degree. Call after structural edits."""
        degree = {nid: 0 for nid in self.nodes}
        for e in self.edges.values():
            degree[e.node_a] = degree.get(e.node_a, 0) + 1
            degree[e.node_b] = degree.get(e.node_b, 0) + 1
        for nid, node in self.nodes.items():
            d = degree.get(nid, 0)
            if d <= 1:
                node.kind = "endpoint"
            elif d == 2:
                node.kind = "L"  # straight-through is also stored as L,
                # real L vs straight is not used downstream
            elif d == 3:
                node.kind = "T"
            else:
                node.kind = "X" if d == 4 else "cross"

    def prune_orphan_nodes(self) -> int:
        used: set[str] = set()
        for e in self.edges.values():
            used.add(e.node_a)
            used.add(e.node_b)
        orphans = [nid for nid in self.nodes if nid not in used]
        for nid in orphans:
            del self.nodes[nid]
        return len(orphans)

    # -- edge ops ---------------------------------------------------------

    def add_edge(
        self,
        node_a: str,
        node_b: str,
        *,
        thickness_m: float = 0.12,
        height_m: float = 2.80,
        wall_type: str = "interior",
        source: str = "manual",
        confidence: float = 1.0,
        edge_id: Optional[str] = None,
        attrs: Optional[Dict[str, object]] = None,
    ) -> WallEdge:
        if node_a == node_b:
            raise ValueError(f"self-loop edge rejected: {node_a}")
        if node_a not in self.nodes or node_b not in self.nodes:
            raise KeyError(
                f"missing node(s): a={node_a!r} b={node_b!r} (have {len(self.nodes)} nodes)"
            )
        eid = edge_id or f"e_{len(self.edges)}_{uuid.uuid4().hex[:6]}"
        edge = WallEdge(
            id=eid,
            node_a=node_a,
            node_b=node_b,
            thickness_m=float(thickness_m),
            height_m=float(height_m),
            wall_type=wall_type,
            source=source,
            confidence=float(confidence),
            attrs=dict(attrs or {}),
        )
        self.edges[eid] = edge
        return edge

    def add_edge_from_segment(
        self,
        a_xy: Point2,
        b_xy: Point2,
        *,
        thickness_m: float = 0.12,
        snap_radius_m: float = 0.0,
        **edge_kwargs,
    ) -> WallEdge:
        """Create or reuse nodes at ``a_xy``/``b_xy`` and add one edge."""
        na = self.find_or_create_node(a_xy, snap_radius_m)
        nb = self.find_or_create_node(b_xy, snap_radius_m)
        if na.id == nb.id:
            raise ValueError(
                f"segment endpoints collapsed to same node after snap (radius={snap_radius_m:.3f}m)"
            )
        return self.add_edge(na.id, nb.id, thickness_m=thickness_m, **edge_kwargs)

    def remove_edge(self, edge_id: str) -> None:
        self.edges.pop(edge_id, None)

    def edge_endpoints(self, edge_id: str) -> Tuple[Point2, Point2]:
        e = self.edges[edge_id]
        return self.nodes[e.node_a].xy, self.nodes[e.node_b].xy

    def edge_length(self, edge_id: str) -> float:
        a, b = self.edge_endpoints(edge_id)
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def edge_direction(self, edge_id: str) -> Tuple[float, float]:
        a, b = self.edge_endpoints(edge_id)
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L < 1e-9:
            return (1.0, 0.0)
        return (dx / L, dy / L)

    def edge_midpoint(self, edge_id: str) -> Point2:
        a, b = self.edge_endpoints(edge_id)
        return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)

    # -- projection / nearest-edge queries --------------------------------

    @staticmethod
    def _project_point_to_segment(
        px: float,
        py: float,
        ax: float,
        ay: float,
        bx: float,
        by: float,
    ) -> Tuple[float, float]:
        """Return (t, perp_dist) where t is clamped to [0, L] along AB.

        t is an absolute length along the segment, NOT a 0..1 parameter -
        callers want the along-edge offset in metres directly.
        """
        dx = bx - ax
        dy = by - ay
        L2 = dx * dx + dy * dy
        if L2 < 1e-12:
            return 0.0, math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / L2  # 0..1 param
        t_cl = max(0.0, min(1.0, t))
        qx = ax + t_cl * dx
        qy = ay + t_cl * dy
        L = math.sqrt(L2)
        return t_cl * L, math.hypot(px - qx, py - qy)

    def project_to_edge(
        self,
        edge_id: str,
        xy: Point2,
    ) -> Tuple[float, float]:
        """Return (offset_m_along_edge, perp_dist_m)."""
        a, b = self.edge_endpoints(edge_id)
        return self._project_point_to_segment(
            xy[0],
            xy[1],
            a[0],
            a[1],
            b[0],
            b[1],
        )

    def nearest_edge(
        self,
        xy: Point2,
        *,
        max_perp_m: float = 1.0,
    ) -> Optional[Tuple[WallEdge, float, float]]:
        """Return (edge, offset_m, perp_dist_m) or None.

        Rejects edges with perp distance > max_perp_m.
        """
        best: Optional[Tuple[WallEdge, float, float]] = None
        for e in self.edges.values():
            off, d = self.project_to_edge(e.id, xy)
            if d > max_perp_m:
                continue
            if best is None or d < best[2]:
                best = (e, off, d)
        return best

    def edge_through_bbox(
        self,
        bbox_m: Tuple[float, float, float, float],
        *,
        min_overlap_m: float = 0.05,
        inflate_by_thickness: bool = True,
        extra_slack_m: float = 0.20,
    ) -> Optional[Tuple[WallEdge, float, float]]:
        """Find the WallEdge whose centerline passes through the bbox
        (inflated by wall thickness/2 + slack to account for the way
        users draw opening bboxes BESIDE rather than ON the centerline).

        Returns (edge, centre_offset_m, intersection_length_m) or None.
        Multiple-edge tie-break: longest intersection wins (the wall
        most of whose body lies inside the bbox is "the" wall).

        Used for Manual-Lock opening attach: replaces nearest-edge by
        perp distance with "wall whose body sits inside the bbox" - fixes
        2026-04-21 review groups A (opening in falscher Wand) and G
        (offset entlang Wand falsch).
        """
        x1 = min(bbox_m[0], bbox_m[2])
        y1 = min(bbox_m[1], bbox_m[3])
        x2 = max(bbox_m[0], bbox_m[2])
        y2 = max(bbox_m[1], bbox_m[3])
        best: Optional[Tuple[WallEdge, float, float]] = None
        best_len = 0.0
        for e in self.edges.values():
            a, b = self.edge_endpoints(e.id)
            slack = (
                e.thickness_m * 0.5 + extra_slack_m
                if inflate_by_thickness and e.thickness_m > 0
                else extra_slack_m
            )
            t_in, t_out = _clip_segment_to_aabb(
                a,
                b,
                x1 - slack,
                y1 - slack,
                x2 + slack,
                y2 + slack,
            )
            if t_in is None:
                continue
            L = self.edge_length(e.id)
            interval_len = (t_out - t_in) * L
            if interval_len < min_overlap_m:
                continue
            if interval_len > best_len:
                best_len = interval_len
                offset_m = max(0.0, min(L, (t_in + t_out) * 0.5 * L))
                best = (e, offset_m, interval_len)
        return best

    # -- opening ops ------------------------------------------------------

    def attach_opening(
        self,
        edge_id: str,
        opening: ParametricOpening,
    ) -> None:
        edge = self.edges[edge_id]
        edge.openings.append(opening)
        edge.openings.sort(key=lambda o: o.offset_m)

    # -- NetworkX bridge --------------------------------------------------

    def to_networkx(self):
        """Return a NetworkX MultiGraph copy (nodes keyed by node-id).

        Used for planar-embedding / outer-face extraction. Edges carry
        ``edge_id`` in their attributes so callers can round-trip.
        """
        import networkx as nx

        g = nx.MultiGraph()
        for n in self.nodes.values():
            g.add_node(n.id, x=n.xy[0], y=n.xy[1], kind=n.kind)
        for e in self.edges.values():
            g.add_edge(
                e.node_a,
                e.node_b,
                edge_id=e.id,
                length=self.edge_length(e.id),
                thickness=e.thickness_m,
            )
        return g

    def outer_face_polygon(self) -> Optional[List[Point2]]:
        """Return the polygon around the outside of the graph, if planar.

        Uses a planar embedding when the graph is planar; otherwise None.
        The returned polygon is a list of node xy coordinates in walking
        order, closed implicitly (first != last).

        NOTE: this is a best-effort helper for the building outline.
        Geometrically-accurate exterior walls still come from the edges
        themselves - this just provides the outline as a polygon for
        UI / floor-slab purposes.
        """
        import networkx as nx

        g = nx.Graph()
        for n in self.nodes.values():
            g.add_node(n.id, pos=n.xy)
        for e in self.edges.values():
            g.add_edge(e.node_a, e.node_b)
        if g.number_of_edges() == 0:
            return None
        is_planar, embedding = nx.check_planarity(g, counterexample=False)
        if not is_planar or embedding is None:
            return None
        # Pick the outer face - NetworkX doesn't mark it, so we take the
        # face containing the leftmost node's leftmost edge.
        left_node = min(self.nodes.values(), key=lambda n: (n.xy[0], n.xy[1])).id
        neighbours = list(embedding.neighbors_cw_order(left_node))
        if not neighbours:
            return None
        face = embedding.traverse_face(left_node, neighbours[0])
        return [self.nodes[nid].xy for nid in face]

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return {
            "nodes": [
                {"id": n.id, "xy": list(n.xy), "kind": n.kind, "attrs": dict(n.attrs)}
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "id": e.id,
                    "node_a": e.node_a,
                    "node_b": e.node_b,
                    "thickness_m": e.thickness_m,
                    "height_m": e.height_m,
                    "wall_type": e.wall_type,
                    "source": e.source,
                    "confidence": e.confidence,
                    "openings": [
                        {
                            "id": o.id,
                            "kind": o.kind,
                            "offset_m": o.offset_m,
                            "width_m": o.width_m,
                            "sill_m": o.sill_m,
                            "head_m": o.head_m,
                            "subtype": o.subtype,
                            "confidence": o.confidence,
                            "source": o.source,
                            "cuts_wall": o.cuts_wall,
                            "attrs": dict(o.attrs),
                        }
                        for o in e.openings
                    ],
                    "attrs": dict(e.attrs),
                }
                for e in self.edges.values()
            ],
            "meta": dict(self.meta),
        }

    # -- summary ----------------------------------------------------------

    def stats(self) -> Dict[str, object]:
        n_doors = 0
        n_windows = 0
        total_length = 0.0
        for e in self.edges.values():
            total_length += self.edge_length(e.id)
            for o in e.openings:
                if o.kind == "door":
                    n_doors += 1
                elif o.kind == "window":
                    n_windows += 1
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "doors": n_doors,
            "windows": n_windows,
            "total_wall_length_m": round(total_length, 3),
        }


__all__ = [
    "WallNode",
    "WallEdge",
    "ParametricOpening",
    "WallGraph",
    "NODE_KINDS",
    "OPENING_KINDS",
]
