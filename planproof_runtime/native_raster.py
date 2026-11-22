"""Read text on large sheets in overlapping native-pixel tiles."""
from collections import defaultdict
import csv
from io import StringIO
from time import monotonic

import cv2
import numpy as np
from PIL import Image

from tools.free_agent_room_qa import ocr
from tools.free_agent_room_qa.reader import label_reads


TILE_WIDTH, TILE_HEIGHT = 1800, 1500
MARGIN = 112
OCR_SECONDS = 120
TILE_SECONDS = 8
MAX_LINES = 12_000


def empty_scene(digest, page, size, pdf=False):
    return dict(source_sha256=digest, page=page, width=size[0], height=size[1],
                coordinates="displayed_pdf_points" if pdf else "image_pixels",
                reader="Question routing", native_text_complete=False, lines=[], rooms={},
                marks={}, frames={}, relations={}, located_profiles={}, suggestions=[], swing_evidence=[])


def tiles(width, height):
    for y in range(0, height, TILE_HEIGHT):
        for x in range(0, width, TILE_WIDTH):
            core = (x, y, min(width, x + TILE_WIDTH), min(height, y + TILE_HEIGHT))
            crop = (max(0, x - MARGIN), max(0, y - MARGIN),
                    min(width, core[2] + MARGIN), min(height, core[3] + MARGIN))
            yield core, crop


def text_tile(image, bounds):
    tile = image.crop(bounds)
    if "A" in tile.getbands() or "transparency" in image.info:
        rgba = tile.convert("RGBA")
        gray = Image.new("L", tile.size, 255)
        gray.paste(rgba.convert("L"), mask=rgba.getchannel("A"))
    elif tile.mode.startswith("I;16"):
        gray = Image.fromarray((np.asarray(tile, dtype=np.uint16) >> 8).astype(np.uint8))
    else:
        gray = tile.convert("L")
    low, high = gray.getextrema()
    if high - low < 4:
        return None
    # Long grid and wall strokes obscure sparse words. Erase only extended
    # straight runs and preserve the grayscale character shapes elsewhere.
    pixels = np.array(gray)
    ink = cv2.inRange(pixels, 0, 185)
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, 123), np.uint8))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((123, 1), np.uint8))
    pixels[(horizontal | vertical) != 0] = 255
    return Image.fromarray(pixels)


def lines_from_tsv(tsv, core, crop):
    if not tsv.startswith("level\t"):
        tsv = ocr.TSV_HEADER + "\n" + tsv
    lines = defaultdict(list)
    for row in csv.DictReader(StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        text = (row.get("text") or "").strip()
        if row.get("level") != "5" or not text:
            continue
        confidence = float(row["conf"])
        if confidence < 45:
            continue
        x, y = int(row["left"]) + crop[0], int(row["top"]) + crop[1]
        right, bottom = min(crop[2], x + int(row["width"])), min(crop[3], y + int(row["height"]))
        if not (crop[0] <= x < right and crop[1] <= y < bottom):
            continue
        key = tuple(row[k] for k in ("block_num", "par_num", "line_num"))
        lines[key].append(dict(text=text, confidence=confidence, bbox=[x, y, right, bottom]))
    for words in lines.values():
        boxes = [word["bbox"] for word in words]
        box = [min(b[0] for b in boxes), min(b[1] for b in boxes),
               max(b[2] for b in boxes), max(b[3] for b in boxes)]
        # Overlap retains context at the crop edges. Assign the result to
        # the interior containing its centre to avoid duplicate observations.
        if core[0] <= (box[0] + box[2]) / 2 < core[2] and core[1] <= (box[1] + box[3]) / 2 < core[3]:
            yield dict(text=" ".join(w["text"] for w in words), words=words, bbox=box)


def read_native(image, digest):
    from tesserocr import PyTessBaseAPI
    scene = empty_scene(digest, 1, image.size)
    scene["reader"] = "Native-resolution tiled OCR"
    regions = list(tiles(*image.size))
    coverage = dict(tiles_total=len(regions), tiles_read=0, tiles_blank=0, tiles_timed_out=0,
                    deadline_reached=False, line_limit_reached=False)
    scene["raster_coverage"] = coverage
    deadline = monotonic() + OCR_SECONDS
    with PyTessBaseAPI(path=str(ocr.ASSETS), lang="eng", psm=11) as api:
        for core, crop in regions:
            if monotonic() >= deadline:
                coverage["deadline_reached"] = True
                break
            tile = text_tile(image, crop)
            if tile is None:
                coverage["tiles_blank"] += 1
                continue
            api.SetImage(tile)
            api.SetSourceResolution(300)
            timeout = max(1, int(min(TILE_SECONDS, deadline - monotonic()) * 1000))
            if not api.Recognize(timeout=timeout):
                coverage["tiles_timed_out"] += 1
                continue
            new = list(lines_from_tsv(api.GetTSVText(0), core, crop))
            capacity = MAX_LINES - len(scene["lines"])
            scene["lines"].extend(new[:capacity])
            coverage["tiles_read"] += 1
            if len(new) > capacity:
                coverage["line_limit_reached"] = True
                break
    labels, scene["marks"] = label_reads(scene["lines"])
    scene["rooms"] = {name: {"label": label} for name, label in labels.items()}
    return scene
