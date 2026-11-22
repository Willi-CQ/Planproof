"""Locate a named floor outline and verify a drawn graphic scale independently."""
import csv
import math
import re
import subprocess

import cv2
import numpy as np
from PIL import Image
from shapely.affinity import translate
from shapely.geometry import Point, box as rectangle

from . import ocr
from .geometry import center, white_enclosures
from .reader import clean, evidence
from .room_intent import NAMES


SCALE_NUMBER = re.compile(r"(\d+(?:[.,]\d+)?)(m|ft|'|′)?", re.I)


def room_labels(lines, target):
    aliases = NAMES.get(target, (target,))
    is_room_code = re.fullmatch(r"r[-.]?[0-9]{1,4}", target)
    anchors = []
    for i, line in enumerate(lines):
        label = clean(line["text"]).lower().strip(" .:;")
        if label in aliases:
            anchors.append(evidence("room_label", line["bbox"], line["text"], f"geom-label-{i}"))
        elif is_room_code:
            for j, word in enumerate(line.get("words", [])):
                if clean(word["text"]).lower().strip(" .:;") == target:
                    anchors.append(evidence("room_label",word["bbox"],word["text"],f"geom-label-{i}-{j}"))
    exact = [a for a in anchors if clean(a["text"]).lower().strip(" .:;") == target]
    return exact if exact else anchors


def grayscale(image):
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        gray = Image.new("L", image.size, 255)
        gray.paste(rgba.convert("L"), mask=rgba.getchannel("A"))
        return gray
    return image.convert("L")


def crop_outline(image, anchor, labels):
    x, y = center(anchor["bbox"])
    width, height = image.size
    left, top = max(0, round(x) - 1750), max(0, round(y) - 1750)
    right, bottom = min(width, round(x) + 1750), min(height, round(y) + 1750)
    gray = np.asarray(grayscale(image.crop((left, top, right, bottom))))
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    local = {}
    for item in labels:
        x0, y0, x1, y1 = item["bbox"]
        if left <= x0 < x1 <= right and top <= y0 < y1 <= bottom:
            local[item["id"]] = {"bbox": [x0-left, y0-top, x1-left, y1-top]}
    chosen = local.get(anchor["id"])
    if not chosen:
        return None
    xx, yy, rr, bb = chosen["bbox"]
    label_height = bb-yy
    if label_height < 2 or label_height > min(gray.shape) / 6:
        return None
    # Remove thin type/swing marks and connect facing ends of the heavier,
    # collinear wall strokes across short door gaps. Test a second gap limit.
    thick = max(5, min(19, round(label_height * .27))) | 1
    walls = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((thick, thick), np.uint8))
    limits = [max(4, round(min(min(gray.shape)*.15, label_height*f))) for f in (4.5, 5.5)]
    candidates = []
    for limit in limits:
        horizontal = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, np.ones((1, limit|1), np.uint8))
        vertical = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, np.ones((limit|1, 1), np.uint8))
        shape = white_enclosures(255 - cv2.bitwise_or(horizontal, vertical), local, 1, 1).get(anchor["id"])
        if shape is None or not shape.covers(Point((xx+rr)/2, (yy+bb)/2)):
            return None
        x0,y0,x1,y1 = shape.bounds
        if x1-x0 < 1.35*(rr-xx) or y1-y0 < 2.2*label_height:
            return None
        candidates.append(shape)
    a, b = candidates
    if a.intersection(b).area < .97 * a.union(b).area:
        return None
    # OpenCV traces the centres of the white boundary pixels. Their occupied
    # extent reaches half a pixel farther toward the actual inner wall faces.
    return translate(a, xoff=left+.5, yoff=top+.5).buffer(.5, join_style=2)


def runs_of_true(row):
    changes = np.flatnonzero(np.diff(np.pad(row.astype(np.int8),(1,1))))
    return list(zip(changes[::2],changes[1::2]))


def filled_strip(image, point):
    """Find a straight strip of uniform nonwhite floor fill between changes in width or fill."""
    x,y = (round(n) for n in point)
    left,top = max(0,x-1750),max(0,y-1750)
    right,bottom = min(image.width,x+1751),min(image.height,y+1751)
    rgb = np.asarray(image.crop((left,top,right,bottom)).convert("RGB"))
    ix,iy = x-left,y-top
    if not (0 <= ix < rgb.shape[1] and 0 <= iy < rgb.shape[0]):
        return None
    fill = np.median(rgb[max(0,iy-2):iy+3,max(0,ix-2):ix+3],axis=(0,1))
    if max(fill) >= 249 or min(fill) <= 95:
        return None
    mask = cv2.inRange(rgb, tuple(fill-7), tuple(fill+7)) > 0
    found = []
    for vertical in (True,False):
        field, row, col = (mask,iy,ix) if vertical else (mask.T,ix,iy)
        h,w = field.shape
        core = field[max(0,row-6):row+7].mean(axis=0) > .72
        cross = next(((l,r) for l,r in runs_of_true(core) if l<=col<r),None)
        if not cross:
            continue
        a,b = cross
        if a<3 or b>w-3 or b-a<12:
            continue
        span = b-a
        support = field[:,a:b].mean(axis=1) >= .83
        gap = min(9,max(3,round(span*.045)))|1
        stable = cv2.morphologyEx(support.astype(np.uint8).reshape(-1,1),cv2.MORPH_CLOSE,
                                  np.ones((gap,1), np.uint8)).ravel()
        reach = next(((l,r) for l,r in runs_of_true(stable) if l<=row<r),None)
        if not reach:
            continue
        p,q = reach
        if p<3 or q>h-3 or q-p<2.5*span:
            continue
        if (support[p:q].mean()<.91 or field[p:q,a-3:a].mean()>.25 or
                field[p:q,b:b+3].mean()>.25):
            continue
        found.append(rectangle(left+a,top+p,left+b,top+q) if vertical
                     else rectangle(left+p,top+a,left+q,top+b))
    return found[0] if len(found)==1 else None


def point_outline(image, point):
    strip = filled_strip(image, point)
    if strip is not None:
        return strip
    for half_height in (10,15,23):
        if not all(half_height <= c < limit-half_height for c,limit in zip(point,image.size)):
            continue
        x,y = point
        anchor = {"id":"selected","bbox":[x-half_height,y-half_height,x+half_height,y+half_height]}
        shape = crop_outline(image,anchor,[anchor])
        if shape is not None:
            return shape
    return None


def bar_candidates(image):
    """Return long, straight runs crossed by three or more projecting ticks."""
    scale = min(1., 2500/max(image.size))
    size = tuple(max(1, round(n*scale)) for n in image.size)
    thumb = image.resize(size, Image.Resampling.LANCZOS) if size != image.size else image
    gray = np.asarray(grayscale(thumb))
    ink = cv2.inRange(gray, 0, 145)
    height, width = gray.shape
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1,max(28,width//24)|1), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(horizontal, connectivity=8)
    bars = []
    for x,y,w,h,_ in stats[1:]:
        if w < 34 or w > width*.43 or h > max(6,w*.035) or min(x,y,width-x-w,height-y-h) < 3:
            continue
        look = max(5, int(w*.055))
        if y < look or y+h+look > height:
            continue
        crossing = (ink[y-look:y, x:x+w] > 0).sum(axis=0) >= max(2, look//3)
        crossing &= (ink[y+h:y+h+look, x:x+w] > 0).sum(axis=0) >= max(2, look//3)
        runs = np.split(np.flatnonzero(crossing), np.where(np.diff(np.flatnonzero(crossing)) > 1)[0]+1)
        ticks = [float(np.mean(r)) for r in runs if 0 < len(r) <= max(6,w*.04)]
        if not 3 <= len(ticks) <= 21 or ticks[0] > .11*w or ticks[-1] < .89*w:
            continue
        spans = np.diff(ticks)
        if min(spans) < .027*w or max(spans) > 4.1*min(spans):
            continue
        bars.append((x+ticks[0], x+ticks[-1], y+h/2, look, y+h))
    sx, sy = image.width/width, image.height/height
    return [(left*sx, right*sx, y*sy, look*sy, baseline*sy)
            for left,right,y,look,baseline in sorted(bars, key=lambda b:b[1]-b[0], reverse=True)[:10]]


def checkerboard_bars(image):
    factor = min(1.,3000/max(image.size))
    w,h = (max(1,round(n*factor)) for n in image.size)
    ink = cv2.inRange(np.asarray(grayscale(image.resize((w,h),Image.Resampling.LANCZOS))),0,163)
    merged = cv2.morphologyEx(ink,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    _,_,stats,_ = cv2.connectedComponentsWithStats(merged,connectivity=8)
    found = []
    for x,y,width,height,area in stats[1:]:
        if width < w*.047 or not .022*width <= height <= .13*width or not .27 < area/(width*height) < .82:
            continue
        if min(x,y,w-x-width,h-y-height)<3:
            continue
        top = ink[y+height//4,x:x+width]>0
        bottom = ink[y+3*height//4,x:x+width]>0
        dark_counts = [sum(b-a>=width*.065 for a,b in runs_of_true(row)) for row in (top,bottom)]
        if (top != bottom).mean() < .81 or min(dark_counts)<1 or not 3 <= sum(dark_counts) <= 16:
            continue
        endpoints = (x*image.width/w,(x+width-1)*image.width/w)
        found.append((*endpoints,(y+height/2)*image.height/h,height*image.height/h,y*image.height/h))
    return sorted(found,key=lambda b:b[1]-b[0],reverse=True)[:6]


def read_scale(image, bars, *, labels_above=False):
    verified = []
    for x0,x1,y,look,baseline in bars:
        span = x1-x0
        left,right = max(0, math.floor(x0 - .08*span)), min(image.width, math.ceil(x1 + .2*span))
        if labels_above:
            top = max(0,math.floor(baseline - max(2.2*look,.11*span)))
            bottom = max(0,math.floor(baseline - max(3,.17*look)))
        else:
            top = min(image.height, math.floor(baseline + .52*look))
            bottom = min(image.height, math.ceil(baseline + max(3.5*look, .14*span)))
        if bottom-top < 6:
            continue
        chip = grayscale(image.crop((left,top,right,bottom)))
        zoom = min(4, max(1, math.ceil(85 / chip.height)))
        chip = chip.resize((chip.width*zoom,chip.height*zoom), Image.Resampling.LANCZOS)
        try:
            # Labels above a segmented scale can be hundreds of pixels apart.
            # Sparse-text recognition reads them individually across the bar.
            tsv = ocr.read_tsv(chip, 11 if labels_above else 7, 2, {"tessedit_char_whitelist":"0123456789.,mft'′ ",
                                            "preserve_interword_spaces":"1"})
        except (OSError, subprocess.TimeoutExpired):
            continue
        words = [row for row in csv.DictReader(tsv.splitlines(),delimiter="\t")
                 if row.get("level") == "5" and row.get("text","").strip()]
        readings = [(match, left + (float(row["left"])+float(row["width"])/2)/zoom)
                    for row in words if (match:=SCALE_NUMBER.fullmatch(row["text"].strip()))]
        if len(readings)<3:
            continue
        zero,lo = readings[0]
        match,hi = readings[-1]
        if float(zero[1].replace(",",".")) != 0:
            continue
        if abs(lo-x0) > .14*span or abs(hi-x1) > .15*span:
            continue
        unit = (match[2] or words[-1]["text"]).lower()
        if unit not in ("m","ft","'","′"):
            continue
        value = float(match[1].replace(",","."))
        if value <= 0 or any(abs(float(mid[1].replace(",","."))/value - (x-x0)/span)>.09
                             for mid,x in readings[1:-1]):
            continue
        distance = value * (1 if unit=="m" else .3048)
        if distance > 0:
            verified.append((distance/span, (left, min(top,max(0,y-look)), right, min(image.height,max(bottom,y+look))),
                             [[x0,y],[x1,y]], match[1]+" "+("m" if unit=="m" else "ft")))
    if not verified or max(a[0] for a in verified) / min(a[0] for a in verified) > 1.04:
        return None
    return verified[0]


def inspect_room(image, scene, program, *, frame_scale=(1.,1.), room_point=None):
    args = program["args"]
    de = args["lang"] == "de"
    locate = program["op"] == "locate_room"
    sx, sy = frame_scale
    result = dict(status="unresolved", value=None, unit=None, evidence=[], calculation=None, reason="")
    label = None
    if room_point is not None:
        polygon = point_outline(image, tuple(v*n for v,n in zip(room_point,image.size)))
    else:
        hits = room_labels(scene["lines"], args["room"])
        if len(hits) != 1:
            return {**result, "reason_code":"room_target_ambiguous" if hits else "room_target_not_found",
                    "reason":("Mehrere Räume passen zu deiner Frage. Wähle die gewünschte Stelle direkt im Plan."
                              if hits else "Wähle die gewünschte Stelle direkt im Plan. Dann kann ich die Kontur prüfen.")
                             if de else ("Several rooms match. Select the one you mean directly on the drawing."
                                         if hits else "Select the room on the drawing so I can check its outline.")}
        label = hits[0]
        other = [item["label"] for item in scene.get("rooms", {}).values() if item["label"]["bbox"] != label["bbox"]]
        labels = [{**item,"bbox":[v*s for v,s in zip(item["bbox"],(sx,sy,sx,sy))]} for item in [label,*other]]
        polygon = crop_outline(image,labels[0],labels)
    name = label["text"] if label else ("Ausgewählter Bereich" if de else "Selected area")
    if polygon is None:
        if not label:
            return {**result,"reason_code":"room_boundary_unverified",
                    "reason":"Um diesen Punkt ließ sich keine eindeutige Bodenkontur bestätigen."
                        if de else "I could not verify a floor boundary around that point."}
        return {**result, "status":"partial", "value":name, "evidence":[label],
                "reason_code":"room_boundary_unverified",
                "reason":"Die Beschriftung ist markiert. Die zugehörige Bodenkontur ist nicht eindeutig."
                    if de else "The label is marked. I could not verify the corresponding floor outline."}
    outline = evidence("room_polygon",[v/s for v,s in zip(polygon.bounds,(sx,sy,sx,sy))],name,"room-outline")
    outline["polygon"] = [[x/sx,y/sy] for x,y in polygon.exterior.coords]
    result["evidence"] = [outline, label] if label else [outline]
    if locate:
        return {**result, "status":"answered", "value":name,
                "calculation":"Die markierte Kontur folgt den Innenkanten und schließt kurze Öffnungen."
                    if de else "Marked the floor outline along the inside of the walls, closing short openings."}
    scale = read_scale(image, bar_candidates(image))
    if scale is None:
        scale = read_scale(image, checkerboard_bars(image), labels_above=True)
    if scale is None:
        return {**result, "status":"partial", "value":name, "reason_code":"room_scale_unverified",
                "reason":"Die Kontur ist markiert. Für die Fläche fehlt ein eindeutiger grafischer Maßstab."
                    if de else "The outline is marked. I need a verifiable graphic scale to give its area."}
    metres, box, ends, span = scale
    unit = args["unit"]
    value = polygon.area * metres**2 / (.3048**2 if unit == "ft²" else 1)
    marker = evidence("graphic_scale",[v/s for v,s in zip(box,(sx,sy,sx,sy))],span,"room-scale")
    marker["line"] = [[x/sx,y/sy] for x,y in ends]
    return {**result,"status":"answered","value":round(value,2), "unit":unit, "approximate":True,
            "evidence":[outline,marker],
            "calculation":("Ungefähre Bodenfläche aus markierter Innenkontur und grafischem Maßstab ("+span+").")
                if de else ("Approximate floor area from the marked inside boundary and the graphic scale ("+span+").")}
