"""Read compact printed identifiers from enlarged, oriented raster word crops."""
import csv
from functools import lru_cache
import importlib.util
import math
from pathlib import Path
import re
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from . import ocr


TAG = re.compile(r'(?:D(?:R|OOR)?|R)[.-]?[0-9]{1,5}[A-Z]?')
MAX_CROPS = 160
SLOT_W, SLOT_H = 320, 64


def overlap(a, b):
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return area / smaller if smaller else 0.


def word_boxes(gray, proposals):
    """Group short text components horizontally and vertically, with and without a door proposal."""
    height, width = gray.shape
    found = {}
    for threshold in (145, 200):
        mask = (gray < threshold).astype(np.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        w, h, area = stats[:, 2], stats[:, 3], stats[:, 4]
        glyph = (w >= 1) & (h >= 1) & (area >= 3) & (w <= 64) & (h <= 64) & (area <= 900)
        glyph[0] = False
        ink = glyph[labels].astype(np.uint8)
        for vertical in (False, True):
            for gap in (2, 4, 7):
                kernel = np.ones((2 * gap + 1, 1) if vertical else (1, 2 * gap + 1), np.uint8)
                groups = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)
                _, _, groups_stats, _ = cv2.connectedComponentsWithStats(groups)
                for x, y, ww, hh, count in groups_stats[1:]:
                    length, high = (hh, ww) if vertical else (ww, hh)
                    if not (4 <= high <= 40 and 1.35 * high <= length <= 7.7 * high and count >= 9):
                        continue
                    region = ink[y:y + hh, x:x + ww]
                    ink_fraction = np.count_nonzero(region) / (ww * hh)
                    if not .12 <= ink_fraction <= .76:
                        continue
                    # A word contains several strokes with aligned relative heights.
                    projection = np.any(region, axis=1 if vertical else 0)
                    starts = np.count_nonzero(projection & ~np.r_[False, projection[:-1]])
                    if starts > 9:
                        continue
                    bounds = tuple(int(v) for v in (x, y, x + ww, y + hh))
                    cx, cy = x + ww / 2, y + hh / 2
                    distance = min((math.hypot(max(p[0] - cx, cx - p[2], 0),
                                               max(p[1] - cy, cy - p[3], 0)) /
                                    max(12., p[2] - p[0], p[3] - p[1]) for p in proposals),
                                   default=float('inf'))
                    rank = (distance > .8, abs(length / high - 2.6), -min(starts, 4), min(distance, 4))
                    if bounds not in found or rank < found[bounds][0]:
                        found[bounds] = (rank, vertical)
    result = []
    for bounds, (_, vertical) in sorted(found.items(), key=lambda item: item[1][0]):
        if any(overlap(bounds, other) > .88 and
               abs((bounds[2] - bounds[0]) - (other[2] - other[0])) < .28 * (bounds[2] - bounds[0]) and
               abs((bounds[3] - bounds[1]) - (other[3] - other[1])) < .28 * (bounds[3] - bounds[1])
               for other, _ in result):
            continue
        result.append((bounds, vertical))
        if len(result) == MAX_CROPS:
            break
    return result


def read_slots(views, mode, digits=False):
    """Run the same alphabet-unconstrained OCR on a batch of separated word cells."""
    sheet = Image.new('L', (SLOT_W, SLOT_H * len(views)), 255)
    for index, (_, tile) in enumerate(views):
        target = min(35 / tile.height, (SLOT_W - 16) / tile.width)
        enlarged = tile.resize((max(1, round(tile.width * target)), max(1, round(tile.height * target))),
                               Image.Resampling.LANCZOS)
        sheet.paste(enlarged, (15, index * SLOT_H + (SLOT_H - enlarged.height) // 2))
    variables = {'load_system_dawg': '0', 'load_freq_dawg': '0'}
    if digits:
        variables['tessedit_char_whitelist'] = 'DR0123456789.-'
    try:
        tsv = ocr.read_tsv(sheet, mode, 12, variables)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    rows = {}
    for word in csv.DictReader(tsv.splitlines(), delimiter='\t', quoting=csv.QUOTE_NONE):
        if word.get('level') != '5' or not word['text'].strip():
            continue
        y, h = int(word['top']), int(word['height'])
        slot = (y + h // 2) // SLOT_H
        if not (0 <= slot < len(views) and y // SLOT_H == (y + h - 1) // SLOT_H):
            continue
        rows.setdefault(slot, []).append(word)
    accepted = {}
    for slot, words in rows.items():
        if len(words) == 1:
            token = words[0]['text'].strip('()[]:;.,')
            if re.fullmatch(r'[A-Za-z0-9|.-]{2,12}', token):
                accepted[slot] = (token, float(words[0]['conf']))
    return accepted


@lru_cache(maxsize=1)
def fonts():
    names = ['DejaVuSans.ttf', 'LiberationSans-Regular.ttf', 'Arial.ttf']
    if ocr.bundled():
        names = [str(ocr.ASSETS / 'DejaVuSans.ttf')]
    mpl = importlib.util.find_spec('matplotlib')
    if mpl and mpl.origin:
        names.append(str(Path(mpl.origin).parent / 'mpl-data/fonts/ttf/DejaVuSans.ttf'))
    found = []
    for name in names:
        try:
            font = ImageFont.truetype(name, 12)
        except OSError:
            continue
        if font.path not in found:
            found.append(font.path)
    return found


def signature(gray):
    ink = np.asarray(gray, dtype=np.float32)
    ink = np.clip((np.percentile(ink, 95) - ink) / max(30., np.percentile(ink, 95) - ink.min()), 0, 1)
    ys, xs = np.nonzero(ink > .4)
    if not len(xs):
        return None
    cropped = ink[ys.min():ys.max()+1, xs.min():xs.max()+1]
    return cv2.resize(cropped, (64, 24), interpolation=cv2.INTER_LINEAR), cropped.shape[1] / cropped.shape[0]


@lru_cache(maxsize=192)
def templates(token, height):
    variants = []
    for name in fonts():
        for size in range(max(5, int(height * 1.05)), min(54, int(height * 1.6) + 3)):
            font = ImageFont.truetype(name, size)
            tile = Image.new('L', (max(16, math.ceil(font.getlength(token)) + 8), size * 2 + 8), 255)
            ImageDraw.Draw(tile).text((3, 2), token, font=font, fill=0)
            match = signature(tile)
            if match:
                variants.append(match)
    return variants


def shape_distance(crop, token):
    actual = signature(crop)
    if actual is None:
        return float('inf')
    ink, aspect = actual
    return min((float(np.mean((ink - other) ** 2)) + .2 * abs(math.log(aspect / ratio))
                for other, ratio in templates(token, crop.height)), default=float('inf'))


def has_continuation(gray, box, vertical):
    if vertical:
        gray = gray.T
        box = (box[1], box[0], box[3], box[2])
    x, y, right, bottom = box
    height = bottom - y
    left = max(0, x - 2 * height)
    top = max(0, y - height)
    tile = gray[top:bottom + height, left:right + 2 * height]
    mask = (tile < 165).astype(np.uint8)
    _, _, components, _ = cv2.connectedComponentsWithStats(mask)
    for u, v, w, h, _ in components[1:]:
        if not (.55 * height <= h <= 1.4 * height and 1 <= w <= 1.6 * height):
            continue
        if abs(top + v + h / 2 - (y + bottom) / 2) > .28 * height:
            continue
        # A similarly aligned following or preceding glyph inside normal character spacing
        # makes the candidate a substring of a longer printed label.
        gap = max(x - (left + u + w), left + u - right)
        if 0 <= gap <= .45 * height:
            return True
    return False


def read_token(plain, sparse, limited, tile):
    """Digit-constrained OCR can suggest a repair. The unrestricted prefix and visual glyphs must agree."""
    raw, confidence = plain
    upper = raw.upper()
    if TAG.fullmatch(upper) and sparse and sparse[0].upper() == upper and max(confidence, sparse[1]) >= 42:
        return upper, min(confidence, sparse[1])
    if not re.fullmatch(r'(?:DR|D|R)[.-]?[0-9OIL|S]{1,5}', upper):
        return None
    prefix = 'DR' if upper.startswith('DR') else upper[0]
    candidate = prefix + upper[len(prefix):].translate(str.maketrans({'O':'0','I':'1','L':'1','|':'1','S':'5'}))
    if not limited or not TAG.fullmatch(candidate):
        return None
    if limited[0] != candidate and not (candidate != upper and
            re.fullmatch(re.escape(prefix) + r'[0-9]{1,5}', limited[0]) and
            len(limited[0]) < len(candidate) and candidate.replace('0', '') == limited[0].replace('0', '')):
        return None
    target = shape_distance(tile, candidate)
    if candidate == upper:
        return (candidate, confidence) if target < .095 else None
    # Consider the actual unconstrained reading including O vs 0 and S vs 5.
    raw_distance = min(shape_distance(tile, raw), shape_distance(tile, upper))
    competitor = min(raw_distance, shape_distance(tile, limited[0])) if limited[0] != candidate else raw_distance
    if target + .006 < competitor and target < (.085 if limited[0] != candidate else .105):
        return candidate, max(0., 100. * (1 - target))
    return None


def read_raster_tags(image, proposals, width, height, existing):
    """Return extra source-coordinate OCR lines. The caller retains its incomplete-inventory status."""
    if not ocr.available():
        return []
    gray = ImageOps.grayscale(image)
    scale = min(1., 2400 / max(gray.size))
    search = gray.resize((max(1, round(gray.width * scale)), max(1, round(gray.height * scale))),
                         Image.Resampling.LANCZOS) if scale < 1 else gray
    sx, sy = search.width / width, search.height / height
    local = [[x * factor for x, factor in zip(p['bbox'], (sx, sy, sx, sy))]
             for p in proposals if p.get('bbox') and len(p['bbox']) == 4]
    pixels = np.asarray(search)
    boxes = word_boxes(pixels, local)
    views = []
    for i, (b, vertical) in enumerate(boxes):
        coords = tuple(int(round(n * f)) for n, f in zip(b, (gray.width / search.width, gray.height / search.height) * 2))
        crop = gray.crop(coords)
        for angle in ((90, 270) if vertical else (0, 180)):
            tile = crop.rotate(angle, expand=True)
            views.append(((i, angle), tile))
    hits = []
    for start in range(0, len(views), 40):
        batch = views[start:start + 40]
        first = read_slots(batch, 6)
        eligible = [slot for slot, (word, _) in first.items() if word.upper().startswith(('D','R'))]
        if not eligible:
            continue
        second = read_slots(batch, 11)
        limited = read_slots(batch, 6, digits=True)
        for slot in eligible:
            accepted = read_token(first[slot], second.get(slot), limited.get(slot), batch[slot][1])
            if not accepted:
                continue
            token, confidence = accepted
            index, angle = batch[slot][0]
            if has_continuation(pixels, *boxes[index]):
                continue
            b = [float(n / f) for n, f in zip(boxes[index][0], (sx, sy, sx, sy))]
            hits.append((token, b, angle, confidence))
    known = [(w['text'].strip('()[]:;.,').upper(), w['bbox']) for line in existing for w in line['words']]
    result = []
    for token, box, angle, confidence in sorted(hits, key=lambda item: -item[3]):
        if any(overlap(box, b) > .5 for name, b in known if name == token):
            continue
        if any(name != token and overlap(box, b) > .6 for name, b, _, _ in hits):
            continue
        if any(name != token and TAG.fullmatch(name) and overlap(box, b) > .6 for name, b in known):
            continue
        word = dict(text=token, bbox=box)
        result.append(dict(**word, words=[word.copy()], reader='tesseract-word-crop',
                           rotation_degrees=angle))
        known.append((token, box))
    return result
