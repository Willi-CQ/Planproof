"""Semantic classification of observations.

Converts raw pixel/vector/topology observations into semantic objects:
walls, rooms, doors, windows, and stairs.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .contracts import (
    Observation,
    ObservationKind,
    PageFrame,
    SourceFamily,
    Mode,
    VectorEvidence,
    PixelEvidence,
)


def classify_walls(vector_ev: VectorEvidence, frame: PageFrame) -> List[Observation]:
    """Classify wall edges as wall observations.
    
    Args:
        vector_ev: Vector evidence with wall edges
        frame: Page frame
    
    Returns:
        List of wall observations
    """
    walls = []
    
    for edge in vector_ev.wall_edges:
        # Compute bbox from geometry
        a = edge.geometry["a"]
        b = edge.geometry["b"]
        bbox = (
            min(a[0], b[0]),
            min(a[1], b[1]),
            max(a[0], b[0]),
            max(a[1], b[1]),
        )
        
        walls.append(Observation(
            id=f"wall-{edge.id}",
            kind=ObservationKind.WALL,
            source_family=SourceFamily.CLASSICAL_PIXELS,
            mode=Mode.INDEPENDENT,
            page_index=frame.page_index,
            frame_id=f"p{frame.page_index}-canonical",
            bbox=bbox,
            geometry={
                "a": edge.geometry["a"],
                "b": edge.geometry["b"],
                "length": edge.length_px,
                "thickness": edge.thickness_px,
                "start_node": edge.start_node,
                "end_node": edge.end_node,
            },
            method="wall_vectorization",
            method_version="v1",
        ))
    
    return walls


# Map RoomFace.semantic_class -> ObservationKind. Faces left
# "unclassified" (e.g. produced without extract_topology) keep legacy
# room behaviour so direct polygonize_faces consumers are unaffected.
_FACE_KIND_MAP = {
    "room": ObservationKind.ROOM,
    "sub_feature": ObservationKind.SUB_FEATURE,
    "non_room_loop": ObservationKind.NON_ROOM_LOOP,
}


def classify_rooms(vector_ev: VectorEvidence, frame: PageFrame) -> List[Observation]:
    """Classify room faces as observations using T2-15 semantic classes.
    
    Only faces classified "room" become ObservationKind.ROOM; smaller
    enclosed loops are preserved as SUB_FEATURE / NON_ROOM_LOOP evidence
    instead of inflating the room count.
    
    Args:
        vector_ev: Vector evidence with room faces
        frame: Page frame
    
    Returns:
        List of room/feature observations
    """
    rooms = []
    
    for face in vector_ev.room_faces:
        if not face.is_valid:
            continue
        
        kind = _FACE_KIND_MAP.get(face.semantic_class)
        if kind is None:
            kind = ObservationKind.ROOM  # unclassified -> legacy room
        
        # Compute bbox from polygon
        xs = [p[0] for p in face.polygon]
        ys = [p[1] for p in face.polygon]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        
        rooms.append(Observation(
            id=f"{kind.value}-{face.id}",
            kind=kind,
            source_family=SourceFamily.CLASSICAL_PIXELS,
            mode=Mode.INDEPENDENT,
            page_index=frame.page_index,
            frame_id=f"p{frame.page_index}-canonical",
            bbox=bbox,
            geometry={
                "polygon": face.polygon,
                "holes": face.holes,
                "area_px": face.area_px,
                "centroid": face.centroid,
                "depends_on_inferred": face.depends_on_inferred,
                "semantic_class": face.semantic_class,
            },
            method="room_polygonization",
            method_version="v1",
        ))
    
    return rooms


def classify_openings(
    vector_ev: VectorEvidence,
    pixel_ev: PixelEvidence,
    frame: PageFrame,
) -> List[Observation]:
    """Classify openings (doors/windows) from associations and candidates.
    
    Args:
        vector_ev: Vector evidence with opening associations
        pixel_ev: Pixel evidence with opening candidates
        frame: Page frame
    
    Returns:
        List of opening observations
    """
    openings = []

    # T2-03: openings outside the estimated plan region (legend bands,
    # title blocks, marginalia) are not door/window evidence. The raw
    # OPENING_CANDIDATE observations are preserved in pixel_ev regardless.
    plan_bbox = vector_ev.plan_region_bbox
    plan_ex = None
    if plan_bbox is not None:
        px0, py0, px1, py1 = plan_bbox
        mx = 0.10 * (px1 - px0)
        my = 0.10 * (py1 - py0)
        plan_ex = (px0 - mx, py0 - my, px1 + mx, py1 + my)

    def _in_plan(bbox) -> bool:
        if plan_ex is None or bbox is None:
            return True
        cx = 0.5 * (bbox[0] + bbox[2])
        cy = 0.5 * (bbox[1] + bbox[3])
        return plan_ex[0] <= cx <= plan_ex[2] and plan_ex[1] <= cy <= plan_ex[3]

    # Get opening candidates from pixel evidence
    candidates = {}
    for obs in pixel_ev.observations:
        if obs.kind == ObservationKind.OPENING_CANDIDATE:
            candidates[obs.id] = obs

    # T2-15 disambiguation: enclosed ink holes are ambiguous — a hole may
    # be a doorway/window reveal in a wall OR a room interior / fixture
    # cavity. Room-interior holes coincide with classified room faces
    # (faces are bounded by wall edges, so true wall openings fall
    # outside room polygons). Skip hole-type candidates whose centroid
    # lies inside a "room"-classified face; arc/gap candidates are kept.
    room_polys = [
        face.polygon for face in vector_ev.room_faces
        if face.is_valid and face.semantic_class == "room" and face.polygon
    ]

    def _inside_room(bbox) -> bool:
        if not room_polys or bbox is None:
            return False
        cx = 0.5 * (bbox[0] + bbox[2])
        cy = 0.5 * (bbox[1] + bbox[3])
        from .contracts import point_in_polygon
        return any(point_in_polygon((cx, cy), poly) for poly in room_polys)

    # Process associations
    for assoc in vector_ev.opening_associations:
        # Find the original candidate
        candidate = candidates.get(assoc.opening_id)
        if candidate is None:
            continue

        # Skip candidates outside the plan region (legend/title-block ink)
        if not _in_plan(candidate.bbox):
            continue

        # Skip enclosed holes that are actually room interiors
        geom = candidate.geometry or {}
        if geom.get("is_hole") and _inside_room(candidate.bbox):
            continue
        
        # Determine if door or window based on aspect ratio and orientation
        if candidate.bbox:
            x0, y0, x1, y1 = candidate.bbox
            width = x1 - x0
            height = y1 - y0
            aspect_ratio = width / height if height > 0 else 0
            
            # Heuristic: taller = door, wider = window
            if aspect_ratio < 0.8:
                kind = ObservationKind.DOOR
            else:
                kind = ObservationKind.WINDOW
        else:
            kind = ObservationKind.DOOR  # Default
        
        openings.append(Observation(
            id=f"opening-{assoc.opening_id}",
            kind=kind,
            source_family=SourceFamily.CLASSICAL_PIXELS,
            mode=Mode.INDEPENDENT,
            page_index=frame.page_index,
            frame_id=f"p{frame.page_index}-canonical",
            bbox=candidate.bbox,
            geometry={
                "position": assoc.position,
                "orientation": assoc.orientation,
                "wall_edge_id": assoc.wall_edge_id,
                "is_attached": assoc.is_attached,
                "overlap_px": assoc.overlap_px,
            },
            method="opening_classification",
            method_version="v1",
        ))
    
    return openings


def classify_stairs(pixel_ev: PixelEvidence, frame: PageFrame) -> List[Observation]:
    """Classify stair candidates as stair observations.
    
    Args:
        pixel_ev: Pixel evidence with stair candidates
        frame: Page frame
    
    Returns:
        List of stair observations
    """
    stairs = []
    
    for obs in pixel_ev.observations:
        if obs.kind == ObservationKind.STAIR_CANDIDATE:
            stairs.append(Observation(
                id=f"stair-{obs.id}",
                kind=ObservationKind.STAIRS,
                source_family=SourceFamily.CLASSICAL_PIXELS,
                mode=Mode.INDEPENDENT,
                page_index=frame.page_index,
                frame_id=f"p{frame.page_index}-canonical",
                bbox=obs.bbox,
                geometry=obs.geometry,
                method="stair_classification",
                method_version="v1",
            ))
    
    return stairs


def classify_all_semantic(
    vector_ev: VectorEvidence,
    pixel_ev: PixelEvidence,
    frame: PageFrame,
) -> List[Observation]:
    """Classify all observations into semantic objects.
    
    Args:
        vector_ev: Vector evidence
        pixel_ev: Pixel evidence
        frame: Page frame
    
    Returns:
        List of semantic observations (walls, rooms, doors, windows, stairs)
    """
    semantic_obs = []
    
    # Classify walls
    semantic_obs.extend(classify_walls(vector_ev, frame))
    
    # Classify rooms
    semantic_obs.extend(classify_rooms(vector_ev, frame))
    
    # Classify openings
    semantic_obs.extend(classify_openings(vector_ev, pixel_ev, frame))
    
    # Classify stairs
    semantic_obs.extend(classify_stairs(pixel_ev, frame))
    
    return semantic_obs
