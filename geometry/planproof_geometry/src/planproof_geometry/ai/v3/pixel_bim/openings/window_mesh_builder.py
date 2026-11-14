"""Hartes Kriterium #1 — BIM-Transfer: 2D-PlacedWindow → 3D-Meshes.

Pro ``PlacedWindow`` werden mehrere Mesh-Komponenten erzeugt:

  - **Parapet** (Brüstung): Wand-Material von 0 bis ``sill_m`` (DIN 0.90 m).
  - **Lintel** (Sturz): Wand-Material von ``head_m`` bis ``wall_height_m``.
  - **Frame** (Rahmen): 4 dünne Riegel (jambs links/rechts + Riegel
    oben/unten) im Glasband — macht Einzelfenster sichtbar; bei
    benachbarten Sub-Fenstern entstehen die "Mullions" automatisch
    durch das Aneinanderstossen der Jambs.
  - **Glass** (Verglasung): Inset-Polygon zwischen Frame-Bauteilen,
    Vertex-Alpha < 200 für Transmissions-Material im glTF-Postprocess.

Lückenschluss zur Wand (Task 2 von Willi):
  Parapet/Lintel/Frame werden in **perp**-Richtung um
  ``WALL_OVERLAP_M`` (5 cm) **erweitert**. Die Wand ist immutable
  (siehe Memory ``feedback_walls_immutable_openings_fit.md``); die
  Erweiterung ueberlappt visuell mit der Wand → keine Luecke mehr.
  Der Glas-Inset bleibt im echten Block-Perp-Range.

Konvention (gleich wie door-``mesh_builder.py``):
  - X = plan X
  - Z = -plan Y (PDF-Y-flip; PDF-Y geht runter, Szene-Y geht hoch)
  - Y = vertikal (oben)
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from .mesh_builder import OpeningMeshes
from .window_placement import PlacedWindow


# DIN-Standard-Fensterhoehen (DIN 18055 / DIN 18545)
DEFAULT_SILL_M = 0.90
DEFAULT_HEAD_M = 2.10

# Frame: ~6 cm seitlich, ~6 cm oben/unten — typische Wohnfenster-Werte.
DEFAULT_FRAME_WIDTH_M = 0.06
DEFAULT_FRAME_THICKNESS_M = 0.06

# Lueckenschluss zur Wand erfolgt jetzt im Clipper via short-side snapping
# (siehe window_block_clipper._snap_short_side_to_wall_edge). Der Mesh-
# Builder braucht keine perp-Extension mehr — Default 0.
WALL_OVERLAP_M = 0.0

# Glas-Material via Vertex-Alpha; der GLTF-Postprocess promotet das zu
# ``KHR_materials_transmission``.
_GLASS_FACE_RGBA = (180, 220, 240, 80)
# Frame-Material: ANTHRAZIT — high contrast zur Wand (sonst verschmelzen
# warmweisse Frames mit der hellen Wand-Vertex-Color im PBR-Renderer).
# Echte moderne Aluminium-Fensterprofile in DB-703 (Anthrazitgrau).
_FRAME_FACE_RGBA = (45, 48, 52, 255)


def _extrude_polygon_in_plan(
    polygon_xy_m: Sequence[Tuple[float, float]],
    *,
    base_y_m: float,
    height_y_m: float,
    flip_y_for_pdf: bool = True,
):
    """Extrudiere ein 2D-Plan-Polygon zu einem 3D-Box-Mesh."""
    import trimesh
    from shapely.geometry import Polygon

    if height_y_m <= 1e-6:
        return None
    if len(polygon_xy_m) < 3:
        return None

    pts = polygon_xy_m
    if flip_y_for_pdf:
        pts = [(x, -y) for x, y in pts]

    poly = Polygon(pts).buffer(0)
    if poly.is_empty or not poly.is_valid:
        return None
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)

    mesh = trimesh.creation.extrude_polygon(poly, height_y_m)
    R = trimesh.transformations.rotation_matrix(-math.pi / 2.0, [1, 0, 0])
    mesh.apply_transform(R)
    mesh.apply_translation([0.0, base_y_m, 0.0])
    return mesh


def _set_face_color(mesh, rgba: Tuple[int, int, int, int]) -> None:
    if mesh is None:
        return
    try:
        n_faces = len(mesh.faces)
        mesh.visual.face_colors = [list(rgba)] * n_faces
    except Exception:
        pass


def _set_pbr_material(mesh, *, name: str,
                     rgba: Tuple[int, int, int, int],
                     metallic: float = 0.0,
                     roughness: float = 0.85) -> None:
    """Set a proper GLTF PBR material on the mesh (so model-viewer renders
    the colour correctly, not as plain vertex-color fallback).
    """
    if mesh is None:
        return
    try:
        import trimesh.visual.material as _mat
        r, g, b, a = rgba
        mat = _mat.PBRMaterial(
            baseColorFactor=[r / 255.0, g / 255.0, b / 255.0, a / 255.0],
            metallicFactor=float(metallic),
            roughnessFactor=float(roughness),
            name=name,
            doubleSided=True,
        )
        mesh.visual = mesh.visual.copy() if hasattr(mesh.visual, "copy") else mesh.visual
        try:
            mesh.visual.material = mat
        except Exception:
            # Some visual subclasses dont support direct material assignment;
            # convert to TextureVisuals first.
            import trimesh
            mesh.visual = trimesh.visual.TextureVisuals(material=mat)
    except Exception:
        # Material is non-essential; fall back to face_colors.
        _set_face_color(mesh, rgba)


def _polygon_from_axis(
    s_lo: float, s_hi: float,
    perp_lo: float, perp_hi: float,
    origin: Tuple[float, float],
    tangent: Tuple[float, float],
    perp: Tuple[float, float],
) -> List[Tuple[float, float]]:
    ox, oy = origin
    tx, ty = tangent
    px, py = perp

    def _w(s: float, t: float) -> Tuple[float, float]:
        return (ox + s * tx + t * px, oy + s * ty + t * py)

    return [
        _w(s_lo, perp_lo),
        _w(s_hi, perp_lo),
        _w(s_hi, perp_hi),
        _w(s_lo, perp_hi),
    ]


def _concat(meshes: List) -> Optional[object]:
    """Concatenate a list of trimesh meshes; None / leere → None."""
    valid = [m for m in meshes if m is not None and len(m.vertices) > 0]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    import trimesh
    return trimesh.util.concatenate(valid)


def build_window_meshes(
    placed: PlacedWindow,
    *,
    wall_height_m: float = 2.80,
    sill_m: float = DEFAULT_SILL_M,
    head_m: float = DEFAULT_HEAD_M,
    frame_width_m: float = DEFAULT_FRAME_WIDTH_M,
    frame_thickness_m: float = DEFAULT_FRAME_THICKNESS_M,
    wall_overlap_m: float = WALL_OVERLAP_M,
    flip_y_for_pdf: bool = True,
) -> OpeningMeshes:
    """Pro PlacedWindow vollstaendiges Fenster-Mesh-Set.

    Komponenten:
      * Parapet (extended perp) — Brüstung fest an Wand
      * Lintel  (extended perp) — Sturz fest an Wand
      * Frame   (extended perp) — 4 Riegel um das Glas, in Holz/weiss
      * Glass   (inset perp + frame-inset s/y) — transparent, im Glasband

    Klassen ``window`` und ``window_large`` werden geometrisch identisch
    behandelt; Differenzierung kommt über die IFC-Klasse + Semantics.
    """
    sill = max(0.0, float(sill_m))
    head_capped = min(float(head_m), wall_height_m - 0.01)
    head = max(sill + 0.01, head_capped)

    # Wenn Achsen-Info fehlt (Pre-V2 PlacedWindow): Fallback auf das
    # gespeicherte Polygon ohne perp-Extension/Frame.
    if (placed.axis_perp_hi_m == placed.axis_perp_lo_m
            and placed.axis_tangent == (1.0, 0.0)
            and placed.axis_perp == (0.0, 1.0)):
        return _build_legacy(
            placed, wall_height_m=wall_height_m,
            sill=sill, head=head, flip_y_for_pdf=flip_y_for_pdf,
        )

    s_lo = placed.axis_s_lo_m
    s_hi = placed.axis_s_hi_m
    perp_lo = placed.axis_perp_lo_m
    perp_hi = placed.axis_perp_hi_m
    perp_lo_ext = perp_lo - wall_overlap_m
    perp_hi_ext = perp_hi + wall_overlap_m

    origin = placed.axis_origin_m
    tangent = placed.axis_tangent
    perp = placed.axis_perp

    # Polygon "extended" = mit Wand-Lueckenschluss (parapet/lintel/frame).
    poly_ext = _polygon_from_axis(
        s_lo, s_hi, perp_lo_ext, perp_hi_ext, origin, tangent, perp,
    )

    meshes = OpeningMeshes()

    # Wand-Material fuer Parapet/Lintel: exakt die dominante Wand-Vertex-Color
    # aus der door-BIM (RGBA 220/220/220/255). Wir setzen das ueber
    # face_colors statt PBR-Material — sonst rendert model-viewer das
    # PBR-Material durch andere Shading-Pfade (Roughness/Metallic) als
    # die unmaterialisierte Wand mit Vertex-Color, und Brüstung/Sturz
    # erscheinen leicht heller/anders als die Wand obwohl die RGB-Werte
    # identisch sind.
    _WALL_RGBA = (220, 220, 220, 255)

    # 1. Parapet (0 -> sill) mit extended perp -> luckenlos zur Wand.
    if sill > 1e-3:
        meshes.parapet = _extrude_polygon_in_plan(
            poly_ext, base_y_m=0.0, height_y_m=sill,
            flip_y_for_pdf=flip_y_for_pdf,
        )
        if meshes.parapet is not None:
            meshes.parapet.visual.face_colors = _WALL_RGBA

    # 2. Lintel (head -> wall_height) mit extended perp.
    lintel_height = max(0.0, wall_height_m - head)
    if lintel_height > 1e-3:
        meshes.lintel = _extrude_polygon_in_plan(
            poly_ext, base_y_m=head, height_y_m=lintel_height,
            flip_y_for_pdf=flip_y_for_pdf,
        )
        if meshes.lintel is not None:
            meshes.lintel.visual.face_colors = _WALL_RGBA

    # 3. Frame: 4 Riegel um das Glasband.
    glass_height = head - sill
    width_along_s = s_hi - s_lo
    fw = min(frame_width_m, max(1e-3, width_along_s * 0.20))
    ft = min(frame_thickness_m, max(1e-3, glass_height * 0.20))

    frame_parts = []
    # Half-jambs an internen Block-Grenzen: zwei benachbarte Sub-Fenster
    # sharen einen Mullion. Das erste Sub-Fenster im Block bekommt einen
    # vollen linken Jamb (auf Block-Edge), das letzte einen vollen rechten.
    # Sub-Fenster in der Mitte: nur halb-jambs an beiden Seiten — zwei
    # half-jambs benachbarter Sub-Fenster ergeben einen 60mm Mullion.
    left_jamb_w = fw if placed.is_first_in_block else fw * 0.5
    right_jamb_w = fw if placed.is_last_in_block else fw * 0.5

    # Linker Jamb
    if left_jamb_w > 1e-3 and glass_height > 1e-3:
        poly_lj = _polygon_from_axis(
            s_lo, s_lo + left_jamb_w, perp_lo_ext, perp_hi_ext,
            origin, tangent, perp,
        )
        frame_parts.append(_extrude_polygon_in_plan(
            poly_lj, base_y_m=sill, height_y_m=glass_height,
            flip_y_for_pdf=flip_y_for_pdf,
        ))

    # Rechter Jamb
    if right_jamb_w > 1e-3 and glass_height > 1e-3:
        poly_rj = _polygon_from_axis(
            s_hi - right_jamb_w, s_hi, perp_lo_ext, perp_hi_ext,
            origin, tangent, perp,
        )
        frame_parts.append(_extrude_polygon_in_plan(
            poly_rj, base_y_m=sill, height_y_m=glass_height,
            flip_y_for_pdf=flip_y_for_pdf,
        ))

    # Unterer + Oberer Riegel zwischen Jambs
    rail_s_lo = s_lo + left_jamb_w
    rail_s_hi = s_hi - right_jamb_w
    if ft > 1e-3 and rail_s_hi > rail_s_lo:
        poly_br = _polygon_from_axis(
            rail_s_lo, rail_s_hi, perp_lo_ext, perp_hi_ext,
            origin, tangent, perp,
        )
        frame_parts.append(_extrude_polygon_in_plan(
            poly_br, base_y_m=sill, height_y_m=ft,
            flip_y_for_pdf=flip_y_for_pdf,
        ))

        # Oberer Riegel
        frame_parts.append(_extrude_polygon_in_plan(
            poly_br, base_y_m=head - ft, height_y_m=ft,
            flip_y_for_pdf=flip_y_for_pdf,
        ))

    frame_mesh = _concat(frame_parts)
    if frame_mesh is not None:
        _set_pbr_material(frame_mesh, name="window_frame",
                          rgba=_FRAME_FACE_RGBA, roughness=0.55)
    meshes.frame = frame_mesh

    # 4. Glas: inset im s und y, im echten perp (damit es aussieht als
    # saesse die Verglasung mittig in der Wand).
    inner_s_lo = s_lo + left_jamb_w
    inner_s_hi = s_hi - right_jamb_w
    inner_y_lo = sill + ft
    inner_y_hi = head - ft
    if inner_s_hi - inner_s_lo > 1e-3 and inner_y_hi - inner_y_lo > 1e-3:
        glass_inset_perp = 0.005
        poly_glass = _polygon_from_axis(
            inner_s_lo, inner_s_hi,
            perp_lo + glass_inset_perp, perp_hi - glass_inset_perp,
            origin, tangent, perp,
        )
        glass = _extrude_polygon_in_plan(
            poly_glass, base_y_m=inner_y_lo,
            height_y_m=inner_y_hi - inner_y_lo,
            flip_y_for_pdf=flip_y_for_pdf,
        )
        if glass is not None:
            _set_pbr_material(glass, name="window_glass",
                              rgba=_GLASS_FACE_RGBA,
                              metallic=0.0, roughness=0.05)
            meshes.materials.append("glass_transmissive")
        meshes.fill = glass

    return meshes


def _build_legacy(
    placed: PlacedWindow,
    *,
    wall_height_m: float,
    sill: float,
    head: float,
    flip_y_for_pdf: bool,
) -> OpeningMeshes:
    """Fallback ohne Frame fuer alte PlacedWindow-Records ohne axis-info."""
    polygon_xy = list(placed.polygon_m)
    meshes = OpeningMeshes()
    if sill > 1e-3:
        meshes.parapet = _extrude_polygon_in_plan(
            polygon_xy, base_y_m=0.0, height_y_m=sill,
            flip_y_for_pdf=flip_y_for_pdf,
        )
    fill_height = head - sill
    if fill_height > 1e-3:
        fill = _extrude_polygon_in_plan(
            polygon_xy, base_y_m=sill, height_y_m=fill_height,
            flip_y_for_pdf=flip_y_for_pdf,
        )
        _set_face_color(fill, _GLASS_FACE_RGBA)
        meshes.fill = fill
        meshes.materials.append("glass_transmissive")
    lintel_height = max(0.0, wall_height_m - head)
    if lintel_height > 1e-3:
        meshes.lintel = _extrude_polygon_in_plan(
            polygon_xy, base_y_m=head, height_y_m=lintel_height,
            flip_y_for_pdf=flip_y_for_pdf,
        )
    return meshes
