"""Locate a named floor outline and verify a drawn graphic scale independently."""
import csv
import math
import re
import subprocess

import cv2
import numpy as np
from PIL import Image
from shapely.affinity import translate
from shapely.geometry import Point

from . import ocr
from .geometry import center, white_enclosures
from .reader import clean, evidence
from .room_intent import NAMES


SCALE_VALUE = re.compile(r"^0(?:[ ,]+[0-9]+(?:[.,][0-9]+)?)*?\s+(\d+(?:[.,]\d+)?)\s*(m|ft|'|′)$", re.I)


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


def read_scale(image, bars):
    verified = []
    for x0,x1,y,look,baseline in bars:
        span = x1-x0
        left,right = max(0, math.floor(x0 - .08*span)), min(image.width, math.ceil(x1 + .2*span))
        top = min(image.height, math.floor(baseline + .52*look))
        bottom = min(image.height, math.ceil(baseline + max(3.5*look, .14*span)))
        if bottom-top < 6:
            continue
        chip = grayscale(image.crop((left,top,right,bottom)))
        zoom = min(4, max(1, math.ceil(85 / chip.height)))
        chip = chip.resize((chip.width*zoom,chip.height*zoom), Image.Resampling.LANCZOS)
        try:
            tsv = ocr.read_tsv(chip, 7, 2, {"tessedit_char_whitelist":"0123456789.,mft'′ ",
                                            "preserve_interword_spaces":"1"})
        except (OSError, subprocess.TimeoutExpired):
            continue
        words = [row for row in csv.DictReader(tsv.splitlines(),delimiter="\t")
                 if row.get("level") == "5" and row.get("text","").strip()]
        reading = " ".join(row["text"].strip() for row in words)
        match = SCALE_VALUE.fullmatch(reading)
        if not match or len(words) < 2 or not words[0]["text"].startswith("0"):
            continue
        lo = left + float(words[0]["left"])/zoom
        last_number = next((w for w in reversed(words) if re.search(r"\d",w["text"])), words[-1])
        hi = left + (float(last_number["left"])+float(last_number["width"])/2)/zoom
        if abs(lo-x0) > .14*span or abs(hi-x1) > .15*span:
            continue
        distance = float(match[1].replace(",", ".")) * (1 if match[2].lower()=="m" else .3048)
        if distance > 0:
            verified.append((distance/span, (left, max(0,y-look), right, bottom), [[x0,y],[x1,y]], match[1]+" "+match[2]))
    if not verified or max(a[0] for a in verified) / min(a[0] for a in verified) > 1.04:
        return None
    return verified[0]


def inspect_room(image, scene, program, *, frame_scale=(1.,1.)):
    args = program["args"]
    de = args["lang"] == "de"
    locate = program["op"] == "locate_room"
    sx, sy = frame_scale
    hits = room_labels(scene["lines"], args["room"])
    result = dict(status="unresolved", value=None, unit=None, evidence=[], calculation=None, reason="")
    if len(hits) != 1:
        return {**result, "reason_code":"room_target_ambiguous" if hits else "room_target_not_found",
                "reason":("Ich finde mehrere passende Räume. Nenne bitte die Raumkennung oder frage nach einem eindeutigen Ausschnitt."
                          if hits else "Ich konnte die Raumbezeichnung auf dieser Seite nicht eindeutig finden.")
                         if de else ("I found several matching rooms. Give the room ID or ask about a clearer detail."
                                     if hits else "I could not find a matching room label on this page.")}
    label = hits[0]
    other = [item["label"] for item in scene.get("rooms", {}).values() if item["label"]["bbox"] != label["bbox"]]
    labels = [{**item,"bbox":[v*s for v,s in zip(item["bbox"],(sx,sy,sx,sy))]} for item in [label,*other]]
    polygon = crop_outline(image,labels[0],labels)
    if polygon is None:
        return {**result, "status":"partial", "value":label["text"], "evidence":[label],
                "reason_code":"room_boundary_unverified",
                "reason":"Die Beschriftung ist markiert. Die zugehörige Bodenkontur ist nicht eindeutig."
                    if de else "The label is marked. I could not verify the corresponding floor outline."}
    outline = evidence("room_polygon",[v/s for v,s in zip(polygon.bounds,(sx,sy,sx,sy))],label["text"],"room-outline")
    outline["polygon"] = [[x/sx,y/sy] for x,y in polygon.exterior.coords]
    result["evidence"] = [outline, label]
    if locate:
        return {**result, "status":"answered", "value":label["text"],
                "calculation":"Die markierte Kontur folgt den Innenkanten und schließt kurze Öffnungen."
                    if de else "Marked the floor outline along the inside of the walls, closing short openings."}
    scale = read_scale(image, bar_candidates(image))
    if scale is None:
        return {**result, "status":"partial", "value":label["text"], "reason_code":"room_scale_unverified",
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
