"""Advisory native PDF polygons; calibrated measurements retain visual checks.

Only simple, single-ring, filled straight-line paths are accepted. Curves,
compound paths/holes, invalid polygons and clipped-off-page shapes are excluded
instead of being approximated or silently filled. A path is not a room label.
"""

import pymupdf
from pydantic import Field
from shapely.geometry import Point, Polygon

from measurements import Anchor, AreaUnit, Box, Input, _result, measure_polygon


def ring_from_path(path):
    points = []
    for item in path["items"]:
        if item[0] == "l":
            a, b = tuple(item[1]), tuple(item[2])
            if points and sum((points[-1][i] - a[i]) ** 2 for i in (0, 1)) > 1e-8:
                return None  # Disconnected/compound fill: do not erase a hole.
            if not points:
                points.append(a)
            points.append(b)
        elif item[0] == "re" and len(path["items"]) == 1:
            r = item[1]
            points = [tuple(r.tl), tuple(r.tr), tuple(r.br), tuple(r.bl)]
        elif item[0] == "qu" and len(path["items"]) == 1:
            q = item[1]
            points = [tuple(q.ul), tuple(q.ur), tuple(q.lr), tuple(q.ll)]
        else:
            return None
    if points and points[-1] == points[0]:
        points.pop()
    return points if 3 <= len(points) <= 64 else None


def extract_regions(page, pixmap, scale, text_lines):
    result = []
    paths = page.get_drawings()
    if len(paths) > 20_000:
        return ()
    transform = page.rotation_matrix * pymupdf.Matrix(scale, scale)
    frame_area = pixmap.width * pixmap.height
    for index, path in enumerate(paths):
        if "f" not in path.get("type", "") or not path.get("fill_opacity", 0):
            continue
        ring = ring_from_path(path)
        if ring is None:
            continue
        pixels = []
        for x, y in ring:
            p = pymupdf.Point(x, y) * transform
            pixels.append([p.x - pixmap.x, p.y - pixmap.y])
        shape = Polygon(pixels)
        if (not shape.is_valid or not 0.00005 * frame_area < shape.area < 0.85 * frame_area
            or any(x < 0 or y < 0 or x > pixmap.width or y > pixmap.height for x, y in pixels)):
            continue
        polygon = [[x * 1000 / pixmap.width, y * 1000 / pixmap.height] for x, y in pixels]
        normalized = Polygon(polygon)
        labels = [line["text"] for line in text_lines if normalized.covers(Point(
            (line["bbox"][0] + line["bbox"][2]) / 2,
            (line["bbox"][1] + line["bbox"][3]) / 2))][:30]
        result.append({"id": f"V{index + 1}", "polygon": polygon,
                       "bbox": list(normalized.bounds), "labels": labels,
                       "area_px2": shape.area, "fill_rgb": path.get("fill")})
        if len(result) >= 500:
            break
    return tuple(result)


def locate_regions(regions, args):
    from native_text import search_lines
    # Reuse strict literal/bounds validation. Query applies to enclosed text hints.
    search_lines((), args)
    query = args.get("query", "").casefold()
    bbox = args.get("bbox", [0, 0, 1000, 1000])
    selected = [r for r in regions if query in " ".join(r["labels"]).casefold()
                and r["bbox"][0] < bbox[2] and r["bbox"][2] > bbox[0]
                and r["bbox"][1] < bbox[3] and r["bbox"][3] > bbox[1]]
    return {"status": "candidates_only", "coordinate_frame": "original_page_normalized_0_1000",
            "regions": [{k: v for k, v in r.items() if k != "polygon"} for r in selected[:40]],
            "matched": len(selected), "truncated": len(selected) > 40,
            "instruction": "Filled vector paths are candidate boundaries, not classified rooms. Some walls/furniture/overlays can be included and curved/compound/clipped paths are omitted. Visually verify the selected region and its use with Nemotron. measure_vector_area can use these registered IDs with cited scale anchors; no scale or physical area is inferred here."}


class VectorSelection(Input):
    region_id: str = Field(min_length=1, max_length=30)
    image_id: str = Field(min_length=1, max_length=80)


class VectorAreaRequest(Input):
    target: str = Field(min_length=1, max_length=200)
    output_unit: AreaUnit
    boundary_basis: str = Field(min_length=1, max_length=600)
    view_bbox: Box
    regions: list[VectorSelection] = Field(min_length=1, max_length=40)
    anchors: list[Anchor] = Field(min_length=2, max_length=6)


def measure_regions(args, regions, delivered, images, page_size):
    request = VectorAreaRequest.model_validate(args)
    ids = [r.region_id for r in request.regions]
    if len(set(ids)) != len(ids) or not set(ids) <= delivered:
        raise ValueError("Select distinct region IDs supplied by get_vector_regions")
    registry = {r["id"]: r for r in regions}
    image_registry = {i.image_id: i for i in images}
    polygons, parts, citations = [], [], []
    width, height = page_size
    for selection in request.regions:
        region = registry[selection.region_id]
        shape = Polygon(region["polygon"])
        if any(shape.intersection(other).area > 1e-6 for other in polygons):
            raise ValueError("Selected vector regions overlap; select disjoint boundaries")
        polygons.append(shape)
        image = image_registry.get(selection.image_id)
        if image is None:
            raise ValueError("Select a supplied crop covering the entire vector region")
        x0, y0, x1, y1 = image.metadata["canonical_bbox"]
        sent_w, sent_h = image.metadata["sent_size"]
        b = [v * (width if i % 2 == 0 else height) / 1000 for i, v in enumerate(region["bbox"])]
        if b[0] < x0 - .01 or b[1] < y0 - .01 or b[2] > x1 + .01 or b[3] > y1 + .01:
            raise ValueError("The source crop must cover the entire selected vector region")
        citation = {"image_id": image.image_id, "bbox": [
            max(0, (b[0] - x0) * sent_w / (x1 - x0)),
            max(0, (b[1] - y0) * sent_h / (y1 - y0)),
            min(sent_w, (b[2] - x0) * sent_w / (x1 - x0)),
            min(sent_h, (b[3] - y0) * sent_h / (y1 - y0))]}
        part = measure_polygon({"target": request.target, "output_unit": request.output_unit,
            "boundary_basis": request.boundary_basis, "region_citation": citation,
            "view_bbox": request.view_bbox, "polygon": region["polygon"],
            "anchors": [a.model_dump() for a in request.anchors]}, images, page_size)
        parts.append({"region_id": selection.region_id, **part})
        citations.extend(part["citations"])
    return _result(request, sum(p["area_m2"] for p in parts), citations, images,
                   method="scaled_vector_regions", approximate=True,
                   formula="Sum of disjoint source vector polygon areas calibrated by cited lengths",
                   components=parts,
                   geometry_source="Native PDF single-ring filled straight-line paths; visual boundary selection and scale readings remain advisory.")
