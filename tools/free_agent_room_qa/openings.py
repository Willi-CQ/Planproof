"""Bind printed labels to physical openings using the visible vector strokes of one PDF page."""
from collections import defaultdict
import math

from shapely.geometry import Point, Polygon

from .geometry import center


def _complex(point):
    return complex(float(point[0]), float(point[1]))


def _xy(value):
    return [value.real, value.imag]


def _walls(page):
    matrix = page.rotation_matrix
    segments = []
    for path in page.get_drawings():
        color = path.get('color')
        width = float(path.get('width') or 0)
        if not color or max(color) > .55 or not width or (path.get('stroke_opacity') or 0) < .65:
            continue
        for item in path['items']:
            if item[0] == 'l':
                a, b = (_complex(p * matrix) for p in item[1:3])
                if abs(b - a) > .1:
                    segments.append((a, b, width))
    return segments


def _wall_support(mid, tangent, length, depth, segments, min_pen):
    """Find visible wall continuations just outside both ends, along the opening's axis."""
    found = {}
    normal_limit = depth / 2 + .09 * length
    for a, b, pen in segments:
        if pen < min_pen or abs(b-a) < .2 * length:
            continue
        local_a, local_b = (a-mid)/tangent, (b-mid)/tangent
        if abs((local_b-local_a).imag) > .18 * abs(b-a):
            continue
        for sign in (-1, 1):
            station = sign * .62 * length
            lo, hi = sorted((local_a.real, local_b.real))
            if not lo <= station <= hi:
                continue
            fraction = (station-local_a.real)/(local_b.real-local_a.real)
            offset = local_a.imag + fraction*(local_b-local_a).imag
            if abs(offset) <= normal_limit and (sign not in found or abs(offset) < abs(found[sign][0])):
                found[sign] = (offset, _xy(a + fraction * (b-a)), [a, b])
    return found


def _swing_matches(mid, tangent, length, depth, swings):
    result = []
    for swing in swings:
        hinge, jamb = _complex(swing['hinge']), _complex(swing['jamb'])
        span = abs(jamb-hinge)
        if not .67 * length <= span <= 1.18 * length:
            continue
        along = (jamb-hinge)/tangent
        offset = ((hinge+jamb)/2-mid)/tangent
        if abs(along.imag) > .2 * span or abs(offset.real) > .28 * length:
            continue
        if abs(offset.imag) <= depth/2 + .1 * length:
            result.append(swing)
    return result


def _broken_wall_frames(swings, walls, profiles):
    """A swing without an opening profile needs a wall terminating at each physical jamb."""
    found = []
    for swing in swings:
        if any(swing in f['swings'] for f in profiles):
            continue
        a, b = _complex(swing['hinge']), _complex(swing['jamb'])
        length = abs(b-a)
        if length < 3:
            continue
        tangent = (b-a)/length
        mid = (a+b)/2
        supports = _wall_support(mid, tangent, length, .12*length, walls, max(.55, .025*length))
        if len(supports) != 2:
            continue
        # A continuous wall or an arc/leaf drawn in front of one does not establish a passage.
        if any(min(abs(point-jamb) for point in supports[sign][2]) > .23*length
               for sign, jamb in ((-1, a), (1, b))):
            continue
        if any(min(((p-mid)/tangent).real for p in (u,v)) < -.12*length and
               max(((p-mid)/tangent).real for p in (u,v)) > .12*length and
               max(abs(((p-mid)/tangent).imag) for p in (u,v)) < .09*length and pen >= max(.55,.025*length)
               for u,v,pen in walls):
            continue
        half_depth = max(.3, .04*length)
        outline = [end + 1j*tangent*sign*half_depth for end, sign in ((a,-1),(b,-1),(b,1),(a,1))]
        found.append(dict(mid=mid, tangent=tangent, length=length, depth=2*half_depth, ends=[a,b], passage=True,
                          outline=outline, intervals=[supports[s][1] for s in (-1,1)],
                          swings=[swing], visible_profile=[]))
    return found


def _profiles(page, swings):
    from mark_frames import components
    import pymupdf

    walls = _walls(page)
    scale = min(2., 2200/max(page.rect.width, page.rect.height))
    image = page.get_pixmap(matrix=pymupdf.Matrix(scale,scale), colorspace=pymupdf.csGRAY, alpha=False)
    def brightness(point):
        x,y=round(point.real*scale),round(point.imag*scale)
        if not (0<=x<image.width and 0<=y<image.height):
            return None
        return image.pixel(x,y)[0]

    result = []
    for part in components(page):
        corners = [_complex(p) for p in part.pts]
        length, depth = float(part.long), float(part.short)
        if not (1.05 <= length / depth <= 40 and 3 <= length <= .25 * min(page.rect.width, page.rect.height)):
            continue
        sides = [b-a for a, b in zip(corners, corners[1:] + corners[:1])]
        side = max(sides, key=abs)
        tangent = side / abs(side)
        mid = _complex(part.center)
        confirmed = _swing_matches(mid, tangent, length, depth, swings)
        if length < 2*depth and not confirmed:
            continue
        supports = _wall_support(mid, tangent, length, depth, walls, max(1.12*part.pen, .032*length))
        if len(supports) != 2 or len(confirmed) > 1:
            continue
        offsets = [v[0] for v in supports.values()]
        if abs(offsets[0]-offsets[1]) > max(.25*depth, .08*length):
            continue
        # Use the wall axis, even when a deep fixture's label is offset towards the door swing.
        shift = sum(offsets) / 2
        if abs(shift) > depth/2 + .035*length:
            continue
        axis_mid = mid + 1j*tangent*shift
        # A white profile must actually interrupt the rendered wall. A shape below a
        # solid dark wall in PDF paint order cannot prove a passage through it.
        outside = [brightness(_complex(v[1])) for v in supports.values()]
        wall_visible = all(value is not None and value <= 174 for value in outside)
        clear = []
        for sign in (-1,1):
            samples = [brightness(axis_mid + length*tangent*complex(sign*t, n))
                       for t in (.18,.27,.36) for n in (-.012,0,.012)]
            clear.append(sum(value is not None and value>=185 for value in samples))
        ends = [axis_mid + sign*.5*length*tangent for sign in (-1, 1)]
        jamb_depth = max(.25, min(depth*.5, .09*length))
        outline = [end + sign*1j*tangent*jamb_depth for end, sign in
                   ((ends[0], -1), (ends[1], -1), (ends[1], 1), (ends[0], 1))]
        result.append(dict(mid=mid, tangent=tangent, length=length, depth=depth, ends=ends, outline=outline,
                           passage=wall_visible and min(clear)>=3,
                           intervals=[v[1] for _, v in sorted(supports.items())], swings=confirmed,
                           visible_profile=[_xy(p) for p in corners]))
    return result + _broken_wall_frames(swings, walls, result)


def marked_vector_profiles(page, marks, swings, existing):
    """Return confirmed passages and separately located, unresolved structural profiles."""
    profiles = _profiles(page, swings)
    candidates = defaultdict(list)
    inverse = defaultdict(list)
    for name, labels in marks.items():
        if len(labels) != 1:
            continue
        label = labels[0]
        anchor = _complex(center(label['bbox']))
        box = label['bbox']
        em = min(box[2]-box[0], box[3]-box[1])
        for i, frame in enumerate(profiles):
            length, depth = frame['length'], frame['depth']
            if length < 1.2*em:
                continue
            position = (anchor-frame['mid'])/frame['tangent']
            if abs(position.real) > .58 * length or abs(position.imag) > depth/2 + min(1.4*em, .18*length):
                continue
            # An unconfirmed profile supplies position only for a mark centered inside
            # the fixture. A nearby label or leader endpoint cannot exclude a remote room.
            if not frame['passage'] and (abs(position.real)>.2*length or abs(position.imag)>.42*depth):
                continue
            cost = abs(position.real)/length + max(0., abs(position.imag)-depth/2)/length
            candidates[name].append((cost, i))
            inverse[i].append((cost, name))

    def winner(ranked):
        ranked = sorted(ranked)
        return ranked[0][1] if len(ranked)==1 or ranked[1][0]-ranked[0][0] > .20 else None

    confirmed, located = {}, {}
    for name, options in candidates.items():
        i = winner(options)
        if name in existing or i is None or winner(inverse[i]) != name:
            continue
        frame = profiles[i]
        points = frame['outline']
        bearing = math.degrees(math.atan2(frame['tangent'].imag, frame['tangent'].real))
        target = confirmed if frame['passage'] else located
        status = ('supported_swing_and_wall_opening' if frame['swings'] else 'supported_marked_wall_opening'
                  ) if frame['passage'] else 'centered_mark_in_structural_profile'
        target[name] = dict(mark=name, bbox=[min(p.real for p in points), min(p.imag for p in points),
                                             max(p.real for p in points), max(p.imag for p in points)],
                            length_pt=frame['length'], bearing_deg=bearing,
                            jamb_centers=[_xy(p) for p in frame['ends']],
                            wall_support_points=frame['intervals'], opening_profile=frame['visible_profile'],
                            swing_evidence=frame['swings'],
                            status=status)
    return confirmed,located


def polygon_sides(frame, rooms):
    """Require several pairs of locally opposite, uniquely named interiors along the jamb axis."""
    a, b = map(_complex, frame['jamb_centers'])
    if abs(b - a) < .1:
        return None
    tangent = (b-a)/abs(b-a)
    mid = (a+b)/2
    length = frame['length_pt']
    shapes = {name:Polygon(room['polygon']) for name, room in rooms.items() if room.get('polygon')}
    pairs = defaultdict(list)
    for offset in (.11, .20, .32, .44):
        for lateral in (-.17, 0, .17):
            support = {}
            for sign in (-1, 1):
                p = mid + length * tangent * complex(lateral, sign*offset)
                coord = _xy(p)
                point = Point(coord)
                owners = [name for name, poly in shapes.items() if poly.contains(point)]
                if len(owners) == 1:
                    support[owners[0]] = coord
            if len(support) == 2:
                pairs[tuple(sorted(support))].append((offset, lateral, support))
    if len(pairs) != 1:
        return None
    (pair, votes), = pairs.items()
    if len({offset for offset, _, _ in votes}) < 2 or len({lateral for _, lateral, _ in votes}) < 2:
        return None
    best = min(votes, key=lambda vote:(abs(vote[1]), vote[0]))
    return dict(rooms=list(pair), probes=best[2], basis='opposite room interiors on a confirmed physical opening')


def touches_scope(frame, polygons):
    """Exclude a registered opening only when its known axis and both near sides miss a requested room."""
    if not polygons:
        return True
    a,b = map(_complex, frame['jamb_centers'])
    if abs(b-a) < .1:
        return True
    tangent=(b-a)/abs(b-a)
    mid=(a+b)/2
    reach=frame['length_pt']
    region=Polygon([_xy(mid+tangent*reach*complex(t,n)) for t,n in
                    [(-.53,-.51),(.53,-.51),(.53,.51),(-.53,.51)]])
    return all(p.intersects(region) for p in polygons)
