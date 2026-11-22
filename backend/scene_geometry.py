"""Explicit straight-line PDF boundaries in canonical page pixels.

Paths are geometry proposals, never room labels. Open strokes, Bezier curves,
holes, clipped boundaries and self intersections are not silently filled.
"""
import json
import math

import pymupdf
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union


BOUNDARY_LIMITS = {'paths': 20000, 'segments': 12000, 'faces': 1000, 'candidates': 1000}


class BoundaryBudgetExceeded(ValueError):
    """An intentional bounded-reader refusal, not an unknown geometry failure."""
    def __init__(self, resource, observed):
        if resource not in BOUNDARY_LIMITS or type(observed) is not int or observed <= BOUNDARY_LIMITS[resource]:
            raise ValueError('Invalid native boundary budget diagnostic')
        self.resource = resource
        self.observed = observed
        self.limit = BOUNDARY_LIMITS[resource]
        self.reason_code = 'native_boundary_' + resource + '_budget_exceeded'
        super().__init__(self.reason_code)

    def diagnostic(self):
        return {'schema': 'planproof-native-boundary-diagnostic-v1', 'status': 'unavailable',
                'reason_code': self.reason_code, 'resource': self.resource,
                'observed': self.observed, 'limit': self.limit}


def transform(page, pdf):
    return pdf.rotation_matrix * pymupdf.Matrix(page.width / pdf.rect.width,
                                                page.height / pdf.rect.height)


def _same(a, b):
    return math.dist(a, b) < 1e-5


def _path_rings(path):
    rings, current = [], []
    def close(points, final=False):
        if not _same(points[0], points[-1]):
            if not (final and path.get('closePath')) and 'f' not in path.get('type', ''):
                return None
            points = [*points, points[0]]
        return points
    for item in path['items']:
        kind = item[0]
        if kind in ('re', 'qu'):
            if current:
                current = close(current)
                if current is None:
                    return []
                rings.append(current)
                current = []
            if kind == 're':
                rect = item[1]
                rings.append([tuple(rect.tl), tuple(rect.tr), tuple(rect.br), tuple(rect.bl), tuple(rect.tl)])
            else:
                quad = item[1]
                rings.append([tuple(quad.ul), tuple(quad.ur), tuple(quad.lr), tuple(quad.ll), tuple(quad.ul)])
        elif kind == 'l':
            first, last = tuple(item[1]), tuple(item[2])
            if current and not _same(current[-1], first):
                current = close(current)
                if current is None:
                    return []
                rings.append(current)
                current = []
            if not current:
                current.append(first)
            current.append(last)
        else:
            return []
    if current:
        current = close(current, final=True)
        if current is None:
            return []
        rings.append(current)
    return rings


def _canonical_shape(ring, matrix, page):
    if not 4 <= len(ring) <= 1025:
        return None
    points = [tuple(pymupdf.Point(*point) * matrix) for point in ring]
    if any(not math.isfinite(value) for point in points for value in point):
        return None
    if any(x < -1e-5 or y < -1e-5 or x > page.width + 1e-5 or y > page.height + 1e-5 for x, y in points):
        return None
    points = [[min(page.width, max(0, x)), min(page.height, max(0, y))] for x, y in points]
    shape = Polygon(points)
    if not shape.is_valid or not 16 < shape.area < page.width * page.height * .98:
        return None
    return shape


def _clip_shape(path, matrix):
    """Full clip geometry, not its bounding rectangle or a repaired polygon.

    PDF clipping implicitly closes subpaths. Clip bounds may exceed the crop;
    unlike room polygons they must not be rejected for covering the whole page.
    Unknown, curved, compound or holed clips remain unsupported.
    """
    rings = _path_rings({**path, 'type': 'f'})
    if len(rings) != 1 or not 4 <= len(rings[0]) <= 1025:
        return None
    points = [tuple(pymupdf.Point(*point) * matrix) for point in rings[0]]
    if any(not math.isfinite(value) for point in points for value in point):
        return None
    shape = Polygon(points)
    return shape if shape.is_valid and not shape.interiors and shape.area > 0 else None


def _clip_metadata(clips):
    if not clips:
        return {}
    return {'clip_containment': 'Whole source polygon contained in every active straight single-ring PDF clip',
            'clip_bounds_json': json.dumps([{'level': level, 'bbox': [round(value, 8) for value in shape.bounds]}
                                           for level, shape in clips], sort_keys=True, separators=(',', ':'))}


class BoundaryRegions(list):
    """Complete explicit paths, with bounded fallback availability recorded."""
    def __init__(self):
        super().__init__()
        self.diagnostics = []


def boundary_regions(page):
    """Return bounded, deduplicated original-file polygon proposals.

    Separate closed strokes can form an exact face; no snapping or inferred line
    extension closes a door gap. Nested subpaths are excluded as possible holes.
    """
    if not getattr(page, 'pdf_bytes', b''):
        return []
    with pymupdf.open(stream=page.pdf_bytes, filetype='pdf') as doc:
        pdf = doc[page.page_index]
        matrix = transform(page, pdf)
        paths = pdf.get_drawings(extended=True)
        if len(paths) > BOUNDARY_LIMITS['paths']:
            raise BoundaryBudgetExceeded('paths', len(paths))
        shapes, segments, clips = [], [], []
        for path in paths:
            level = path.get('level', 0)
            clips = [(clip_level, shape) for clip_level, shape in clips if clip_level < level]
            if path.get('type') == 'clip':
                clips.append((level, _clip_shape(path, matrix)))
                continue
            if path.get('type') == 'group' or len(clips) > 32 or any(shape is None for _, shape in clips):
                continue
            opacity = path.get('fill_opacity') if 'f' in path.get('type', '') else path.get('stroke_opacity')
            if not isinstance(opacity, (float, int)) or opacity <= 0:
                continue
            rings = _path_rings(path)
            members = [shape for ring in rings if (shape := _canonical_shape(ring, matrix, page)) is not None]
            if rings and len(members) != len(rings):
                continue  # A clipped/invalid component cannot erase a possible hole.
            if any(not clip.covers(shape) for shape in members for _, clip in clips):
                continue  # Never intersect a cut path into a new room boundary.
            nested = any(shape is not other and (shape.covers(other) or other.covers(shape))
                         for shape in members for other in members)
            # Our scene polygon field is a single ring: retaining only the
            # exterior of a holed path would silently erase excluded floor area.
            for shape in members:
                if any(shape is not other and (shape.covers(other) or other.covers(shape)) for other in members):
                    continue
                shapes.append((shape, 'explicit closed PDF path', _clip_metadata(clips)))
            if nested:
                continue
            if any(item[0] not in ('l', 're', 'qu') for item in path['items']):
                continue  # Never polygonize the straight fragments of a curved path.
            if 's' in path.get('type', ''):
                for item in path['items']:
                    if item[0] != 'l':
                        continue
                    first, last = item[1] * matrix, item[2] * matrix
                    if first != last and all(0 <= p.x <= page.width and 0 <= p.y <= page.height for p in (first, last)):
                        segment = LineString([(first.x, first.y), (last.x, last.y)])
                        if all(clip.covers(segment) for _, clip in clips):
                            segments.append(segment)
        diagnostics = []
        if len(segments) > BOUNDARY_LIMITS['segments']:
            diagnostics.append(BoundaryBudgetExceeded('segments', len(segments)).diagnostic())
        elif segments:
            faces = list(polygonize(unary_union(segments)))
            if len(faces) > BOUNDARY_LIMITS['faces']:
                diagnostics.append(BoundaryBudgetExceeded('faces', len(faces)).diagnostic())
                faces = []
            for shape in faces:
                if not shape.interiors and shape.is_valid and 16 < shape.area < page.width * page.height * .98:
                    shapes.append((shape, 'closed face from explicit PDF line segments',
                                   {'clip_containment': 'Every original boundary segment wholly contained in its active PDF clips'}))
        result, seen = BoundaryRegions(), set()
        result.diagnostics = diagnostics
        for shape, basis, clip_attributes in shapes:
            key = tuple(round(value, 4) for point in shape.normalize().exterior.coords for value in point)
            if key in seen:
                continue
            seen.add(key)
            result.append({'polygon': [[x * 1000 / page.width, y * 1000 / page.height] for x, y in shape.exterior.coords],
                           'boundary_basis': basis, **clip_attributes})
        return result
