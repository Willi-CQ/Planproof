"""Tests for the junction-graph wall representation.

Pure data-structure checks - no pipeline integration. Day-1 deliverable
of the Plan2BIM Graph-Refactor sprint; ensures add/remove/project/outer-face
behave before any pipeline step depends on them.
"""

from __future__ import annotations

import pytest

from planproof_cad.model.wall_graph import (
    ParametricOpening,
    WallGraph,
)


def _square(g: WallGraph, *, snap=0.0):
    """Build a 4m x 3m rectangle with 4 edges. Returns node ids in order."""
    e1 = g.add_edge_from_segment((0, 0), (4, 0), snap_radius_m=snap)
    e2 = g.add_edge_from_segment((4, 0), (4, 3), snap_radius_m=snap)
    e3 = g.add_edge_from_segment((4, 3), (0, 3), snap_radius_m=snap)
    e4 = g.add_edge_from_segment((0, 3), (0, 0), snap_radius_m=snap)
    return e1, e2, e3, e4


def test_add_edge_snaps_to_existing_node():
    g = WallGraph()
    _square(g, snap=0.05)
    # 4 distinct corners, not 8 floating endpoints
    assert len(g.nodes) == 4
    assert len(g.edges) == 4


def test_no_snap_creates_distinct_nodes():
    g = WallGraph()
    _square(g, snap=0.0)
    # Without snapping every segment allocates its own endpoints - 4 edges
    # × 2 endpoints = 8 nodes even though geometrically they coincide. This
    # is the "raw" state before HealGraphStep runs.
    assert len(g.nodes) == 8
    assert len(g.edges) == 4


def test_self_loop_rejected():
    g = WallGraph()
    n = g.add_node((0, 0))
    with pytest.raises(ValueError):
        g.add_edge(n.id, n.id)


def test_refresh_node_kinds_marks_junctions():
    g = WallGraph()
    _square(g, snap=0.05)
    g.refresh_node_kinds()
    # All corners degree 2 → "L"
    assert all(n.kind == "L" for n in g.nodes.values())

    # Add a T junction: extra edge hitting one of the corners
    n_mid = g.add_node((2, 0), kind="endpoint")
    n_up = g.add_node((2, 1.5), kind="endpoint")
    # Split the bottom edge into two halves meeting at n_mid
    # (simulating a T by subdividing and adding a stub)
    g.add_edge(n_mid.id, n_up.id)
    g.refresh_node_kinds()
    # n_mid now has degree 1 → endpoint; n_up also endpoint
    assert g.nodes[n_up.id].kind == "endpoint"


def test_project_to_edge_returns_along_offset():
    g = WallGraph()
    a = g.add_node((0, 0))
    b = g.add_node((4, 0))
    e = g.add_edge(a.id, b.id)
    # Point at (1, 0.5) projects to (1, 0) → offset=1m, perp=0.5m
    off, perp = g.project_to_edge(e.id, (1.0, 0.5))
    assert off == pytest.approx(1.0, abs=1e-6)
    assert perp == pytest.approx(0.5, abs=1e-6)


def test_project_to_edge_clamps_before_a():
    g = WallGraph()
    a = g.add_node((0, 0))
    b = g.add_node((4, 0))
    e = g.add_edge(a.id, b.id)
    # Point before a → clamped to a, perp = distance to a
    off, perp = g.project_to_edge(e.id, (-1.0, 0.0))
    assert off == pytest.approx(0.0, abs=1e-6)
    assert perp == pytest.approx(1.0, abs=1e-6)


def test_nearest_edge_picks_closest():
    g = WallGraph()
    _square(g, snap=0.05)
    # Point (2, 0.1) is 0.1m above the bottom edge
    found = g.nearest_edge((2.0, 0.1), max_perp_m=1.0)
    assert found is not None
    edge, off, perp = found
    assert perp == pytest.approx(0.1, abs=1e-6)
    # Projected onto the bottom edge at x=2 → offset 2m
    assert off == pytest.approx(2.0, abs=1e-6)


def test_nearest_edge_respects_max_perp():
    g = WallGraph()
    _square(g, snap=0.05)
    # Point far outside any wall
    found = g.nearest_edge((10.0, 10.0), max_perp_m=0.5)
    assert found is None


def test_edge_length_and_direction():
    g = WallGraph()
    a = g.add_node((0, 0))
    b = g.add_node((3, 4))
    e = g.add_edge(a.id, b.id)
    assert g.edge_length(e.id) == pytest.approx(5.0, abs=1e-6)
    ux, uy = g.edge_direction(e.id)
    assert ux == pytest.approx(0.6, abs=1e-6)
    assert uy == pytest.approx(0.8, abs=1e-6)


def test_attach_opening_sorts_by_offset():
    g = WallGraph()
    a = g.add_node((0, 0))
    b = g.add_node((10, 0))
    e = g.add_edge(a.id, b.id)
    g.attach_opening(e.id, ParametricOpening(id="o2", kind="door", offset_m=6.0, width_m=0.85))
    g.attach_opening(e.id, ParametricOpening(id="o1", kind="window", offset_m=2.0, width_m=1.2))
    g.attach_opening(e.id, ParametricOpening(id="o3", kind="door", offset_m=8.5, width_m=0.9))
    assert [o.id for o in e.openings] == ["o1", "o2", "o3"]


def test_opening_start_end():
    o = ParametricOpening(id="d", kind="door", offset_m=3.0, width_m=0.85)
    assert o.start_m() == pytest.approx(2.575, abs=1e-6)
    assert o.end_m() == pytest.approx(3.425, abs=1e-6)


def test_prune_orphan_nodes():
    g = WallGraph()
    g.add_node((5, 5))
    a = g.add_node((0, 0))
    b = g.add_node((1, 0))
    g.add_edge(a.id, b.id)
    removed = g.prune_orphan_nodes()
    assert removed == 1
    assert len(g.nodes) == 2


def test_outer_face_rectangle():
    g = WallGraph()
    _square(g, snap=0.05)
    poly = g.outer_face_polygon()
    assert poly is not None
    assert len(poly) == 4
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    assert min(xs) == pytest.approx(0.0, abs=1e-6)
    assert max(xs) == pytest.approx(4.0, abs=1e-6)
    assert min(ys) == pytest.approx(0.0, abs=1e-6)
    assert max(ys) == pytest.approx(3.0, abs=1e-6)


def test_to_networkx_roundtrip():
    g = WallGraph()
    _square(g, snap=0.05)
    nx_g = g.to_networkx()
    assert nx_g.number_of_nodes() == 4
    assert nx_g.number_of_edges() == 4
    # Edge attrs carry edge_id so we can map back
    edge_ids = [d["edge_id"] for _, _, d in nx_g.edges(data=True)]
    assert set(edge_ids) == set(g.edges.keys())


def test_to_dict_is_json_friendly():
    import json

    g = WallGraph()
    _square(g, snap=0.05)
    first_edge = next(iter(g.edges.values()))
    g.attach_opening(
        first_edge.id,
        ParametricOpening(id="d1", kind="door", offset_m=2.0, width_m=0.85),
    )
    d = g.to_dict()
    # Must be serialisable
    json.dumps(d)
    assert d["edges"][0]["openings"][0]["kind"] == "door"


def test_stats():
    g = WallGraph()
    _square(g, snap=0.05)
    e = next(iter(g.edges.values()))
    g.attach_opening(e.id, ParametricOpening(id="d", kind="door", offset_m=2.0, width_m=0.85))
    g.attach_opening(e.id, ParametricOpening(id="w", kind="window", offset_m=3.5, width_m=1.0))
    s = g.stats()
    assert s["nodes"] == 4
    assert s["edges"] == 4
    assert s["doors"] == 1
    assert s["windows"] == 1
    # Perimeter = 2*(4+3) = 14 m
    assert s["total_wall_length_m"] == pytest.approx(14.0, abs=1e-3)


def test_nearest_edge_used_for_opening_attachment_flow():
    """Simulate the Day-4 flow: user bbox centroid -> edge -> offset."""
    g = WallGraph()
    _square(g, snap=0.05)

    # User labels a door whose bbox centroid is at (2.5, 0.08) - 8cm above
    # the bottom edge (representing typical user drift).
    found = g.nearest_edge((2.5, 0.08), max_perp_m=0.3)
    assert found is not None
    edge, off, perp = found
    assert perp == pytest.approx(0.08, abs=1e-6)
    assert off == pytest.approx(2.5, abs=1e-6)

    # Width from bbox (say 0.85m along x)
    g.attach_opening(
        edge.id,
        ParametricOpening(
            id="door1",
            kind="door",
            offset_m=off,
            width_m=0.85,
        ),
    )
    # Door start 2.075m, end 2.925m along edge - both inside [0, 4]
    o = edge.openings[0]
    assert o.start_m() > 0
    assert o.end_m() < g.edge_length(edge.id)
