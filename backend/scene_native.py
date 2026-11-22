"""Visible native text/table extraction, separate from frozen legacy PDF hints."""

from io import BytesIO
import math
import re
import unicodedata

from PIL import Image
import pymupdf
from shapely.geometry import box as shape_box
from shapely.strtree import STRtree


def _character_runs(chars, direction, size, words=False):
    """Split real whitespace/position gaps without guessing characters."""
    dx, dy = direction
    runs, current = [], []
    for char in chars:
        if words and chr(char[0]).isspace():
            if current:
                runs.append(current)
                current = []
            continue
        if current:
            previous = current[-1]
            x0, y0, x1, y1 = previous[3]
            tail = max(x * dx + y * dy for x, y in ((x0, y0), (x0, y1), (x1, y0), (x1, y1)))
            gap = char[2][0] * dx + char[2][1] * dy - tail
            backwards = (char[2][0] - previous[2][0]) * dx + (char[2][1] - previous[2][1]) * dy
            if gap > max(1, size * (.45 if words else 1.5)) or backwards < -size:
                runs.append(current)
                current = []
        current.append(char)
    if current:
        runs.append(current)
    return runs


def _word_lines(page, chars, matrix, direction, size, raster):
    words = []
    for run in _character_runs(chars, direction, size, words=True):
        text = ''.join(chr(char[0]) for char in run).strip()
        original = pymupdf.Rect(run[0][3])
        for char in run[1:]:
            original |= pymupdf.Rect(char[3])
        rect = original * matrix
        if text and 0 <= rect.x0 < rect.x1 <= page.width and 0 <= rect.y0 < rect.y1 <= page.height:
            crop = raster.crop((math.floor(rect.x0), math.floor(rect.y0), math.ceil(rect.x1), math.ceil(rect.y1)))
            if max(high - low for low, high in crop.getextrema()) < 4:
                continue
            words.append({'text': text, 'bbox': [value * 1000 / (page.width if index % 2 == 0 else page.height)
                                               for index, value in enumerate(rect)]})
    return words


def _oriented_bounds(bounds, direction):
    dx, dy = direction
    x0, y0, x1, y1 = bounds
    corners = ((x0, y0), (x0, y1), (x1, y0), (x1, y1))
    along = [x * dx + y * dy for x, y in corners]
    normal = [-x * dy + y * dx for x, y in corners]
    return [min(along), min(normal), max(along), max(normal)]


def _quantity_lines(page, lines):
    """Join only spatially adjacent numeric/unit glyph spans, not room prose."""
    used, result = set(), []
    for index, line in enumerate(lines):
        if index in used:
            continue
        text = unicodedata.normalize('NFKC', line['text']).strip()
        number = re.fullmatch(r'\d+(?:[.,]\d+)?', text)
        squared = re.fullmatch(r'\d+(?:[.,]\d+)?\s*m', text, re.I)
        if not number and not squared:
            continue
        bounds = [v * (page.width if i % 2 == 0 else page.height) / 1000 for i, v in enumerate(line['bbox'])]
        direction = line.get('direction', (1, 0))
        oriented = _oriented_bounds(bounds, direction)
        height = oriented[3] - oriented[1]
        choices = []
        for other_index, other in enumerate(lines):
            if other_index == index or other_index in used:
                continue
            unit = unicodedata.normalize('NFKC', other['text']).strip()
            if number and not re.fullmatch(r'm\s*2|ft\s*2|sq\.?\s*ft|m|cm|mm|ft', unit, re.I):
                continue
            if squared and unit != '2':
                continue
            other_direction = other.get('direction', (1, 0))
            if sum(first * second for first, second in zip(direction, other_direction)) < .99:
                continue  # Numeric/unit text must share the same reading direction.
            target = [v * (page.width if i % 2 == 0 else page.height) / 1000 for i, v in enumerate(other['bbox'])]
            target_oriented = _oriented_bounds(target, direction)
            gap = target_oriented[0] - oriented[2]
            overlap = min(target_oriented[3], oriented[3]) - max(target_oriented[1], oriented[1])
            minimum_height = min(height, target_oriented[3] - target_oriented[1])
            if not -height * .1 <= gap <= height * .8 or overlap < minimum_height * .25:
                continue
            if squared and (target_oriented[1] + target_oriented[3]) / 2 >= (oriented[1] + oriented[3]) / 2 - height * .1:
                continue
            choices.append((other_index, other, target, unit))
        if len(choices) != 1:
            continue
        other_index, other, target, unit = choices[0]
        union = [min(bounds[0], target[0]), min(bounds[1], target[1]), max(bounds[2], target[2]), max(bounds[3], target[3])]
        result.append({**line, 'text': text + ('' if squared else ' ') + unit,
                       'words': [*line.get('words', []), *other.get('words', [])],
                       'text_truncated': bool(line.get('text_truncated') or other.get('text_truncated')),
                       'source_clipped': bool(line.get('source_clipped') or other.get('source_clipped')),
                       'bbox': [v * 1000 / (page.width if i % 2 == 0 else page.height) for i, v in enumerate(union)]})
        used.update((index, other_index))
    return tuple([line for index, line in enumerate(lines) if index not in used] + result)


def visible_lines(page):
    if not getattr(page, "pdf_bytes", b""):
        return _quantity_lines(page, [line for line in page.text_lines if line.get("visible") is True])
    with pymupdf.open(stream=page.pdf_bytes, filetype="pdf") as doc:
        pdf = doc[page.page_index]
        sx, sy = page.width / pdf.rect.width, page.height / pdf.rect.height
        matrix = pdf.rotation_matrix * pymupdf.Matrix(sx, sy)
        traces=pdf.get_texttrace()
        painting=pdf.get_bboxlog()
        if len(painting)>20000:
            return ()  # Do not silently truncate a visibility audit.
        geometries=[]
        sequences=[]
        for seq,(kind,bounds,*_) in enumerate(painting):
            if kind!='ignore-text' and bounds[0]<bounds[2] and bounds[1]<bounds[3]:
                sequences.append(seq)
                geometries.append(shape_box(*bounds))
        tree=STRtree(geometries)
        signatures={s['seqno']:tuple((c[0],c[2],c[3]) for c in s['chars']) for s in traces}
        raster=Image.open(BytesIO(page.png)).convert('RGB')
        lines = []
        for span in traces:
            if span.get("type") == 3 or span.get("opacity", 1) <= 0:
                continue
            # A trace span can cross several text baselines (e.g. Kitchen\n24m2).
            # Preserve baselines before associating room labels and measurements.
            dx,dy=span.get('dir',(1,0))
            origin = pymupdf.Point(0, 0) * matrix
            endpoint = pymupdf.Point(dx, dy) * matrix
            length = math.hypot(endpoint.x - origin.x, endpoint.y - origin.y)
            if not math.isfinite(length) or length <= 0:
                continue
            canonical_direction = [(endpoint.x - origin.x) / length, (endpoint.y - origin.y) / length]
            groups={}
            for char in span['chars']:
                if char[0] >= 0:
                    baseline=round(-dy*char[2][0]+dx*char[2][1],2)
                    groups.setdefault(baseline,[]).append(char)
            size = max(1, float(span.get('size') or 1))
            runs = [run for chars in groups.values() for run in _character_runs(chars, (dx, dy), size)]
            for chars in runs:
                text=''.join(chr(c[0]) for c in chars).strip()
                original=pymupdf.Rect(chars[0][3])
                for char in chars[1:]:
                    original |= pymupdf.Rect(char[3])
                hidden=False
                for index in tree.query(shape_box(*original)):
                    seq=sequences[int(index)]
                    if seq<=span['seqno']:
                        continue
                    # Fill+stroke text can appear twice with identical glyphs.
                    if signatures.get(seq)==signatures.get(span['seqno']):
                        continue
                    overlap=pymupdf.Rect(painting[seq][1])&original
                    if overlap.get_area()>max(.01,original.get_area()*.01):
                        hidden=True
                        break
                if hidden:
                    continue
                rect=original*matrix
                box = [max(0, rect.x0), max(0, rect.y0), min(page.width, rect.x1), min(page.height, rect.y1)]
                if text and box[0] < box[2] and box[1] < box[3]:
                    crop=raster.crop((math.floor(box[0]),math.floor(box[1]),math.ceil(box[2]),math.ceil(box[3])))
                    if max(high-low for low,high in crop.getextrema())<4:
                        continue  # White-on-white and other flat, invisible text.
                    lines.append({"text": text[:1000], "bbox": [v * 1000 / (page.width if i % 2 == 0 else page.height) for i, v in enumerate(box)],
                                  "visible": True, 'words': _word_lines(page, chars, matrix, (dx, dy), size, raster),
                                  'direction': canonical_direction,
                                  'text_truncated': len(text) > 1000,
                                  'source_clipped': not (0 <= rect.x0 < rect.x1 <= page.width and 0 <= rect.y0 < rect.y1 <= page.height)})
        # Rendering modes can duplicate text, including separate fill/stroke
        # spans. Deduplicate only coincident text, never two real repeated labels.
        unique = []
        for line in lines:
            same = ''.join(unicodedata.normalize('NFKC', line['text']).casefold().split())
            if any(same == ''.join(unicodedata.normalize('NFKC', other['text']).casefold().split())
                   and max(abs(a - b) for a, b in zip(line['bbox'], other['bbox'])) < .5 for other in unique):
                continue
            unique.append(line)
        # No silently truncated inventory: callers can mark coverage partial.
        return _quantity_lines(page, unique)


def tables(page, visible=None):
    if not getattr(page, "pdf_bytes", b""):
        return []
    visible=visible_lines(page) if visible is None else visible
    with pymupdf.open(stream=page.pdf_bytes, filetype="pdf") as doc:
        pdf = doc[page.page_index]
        matrix = pdf.rotation_matrix * pymupdf.Matrix(page.width / pdf.rect.width, page.height / pdf.rect.height)
        result = []
        for table in pdf.find_tables().tables:
            for row_index, row in enumerate(table.extract()):
                cells = table.rows[row_index].cells
                for col_index, text in enumerate(row):
                    if not text or cells[col_index] is None:
                        continue
                    rect = pymupdf.Rect(cells[col_index]) * matrix
                    box = [max(0, rect.x0), max(0, rect.y0), min(page.width, rect.x1), min(page.height, rect.y1)]
                    if box[0] < box[2] and box[1] < box[3]:
                        checked=[]
                        for line in visible:
                            lb=[v*(page.width if i%2==0 else page.height)/1000 for i,v in enumerate(line['bbox'])]
                            if box[0]<=lb[0] and box[1]<=lb[1] and box[2]>=lb[2] and box[3]>=lb[3]:
                                checked.append(line['text'])
                        if ''.join(text.split())!=''.join(''.join(checked).split()):
                            continue
                        result.append({"bbox": box, "label": text, "row": row_index, "column": col_index})
        return result


def dimension_lines(page, walls=False):
    """Only explicit thin straight segments; never synthesize a scale from pixels."""
    if not getattr(page,'pdf_bytes',b''):
        return []
    with pymupdf.open(stream=page.pdf_bytes,filetype='pdf') as doc:
        pdf=doc[page.page_index]
        matrix=pdf.rotation_matrix*pymupdf.Matrix(page.width/pdf.rect.width,page.height/pdf.rect.height)
        paths=pdf.get_drawings()
        if len(paths)>20000:
            return []
        result=[]
        for path in paths:
            # PyMuPDF reports width/opacity=None for fill-only paths. Those
            # polygons remain room evidence, but are never dimension/wall strokes.
            if 's' not in path.get('type',''):
                continue
            width,opacity=path.get('width'),path.get('stroke_opacity')
            if (not isinstance(width,(int,float)) or not isinstance(opacity,(int,float))
                or not math.isfinite(width) or not math.isfinite(opacity)
                or width<=0 or opacity<=0 or (width<2 if walls else width>1.5)):
                continue
            for item in path['items']:
                if item[0]=='l':
                    a,b=item[1]*matrix,item[2]*matrix
                    result.append([[a.x,a.y],[b.x,b.y]])
        return result
