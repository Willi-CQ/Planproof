"""Read a door-specific printed property from a labelled note or a single table row."""
from decimal import Decimal
import re

from .reader import DOOR, clean, evidence


MARK = re.compile(r'(?<![\w.-])(?:DOOR|DR|D)[.-]?\d{1,5}[A-Z]?(?![\w.-])', re.I)
FIRE = re.compile(r'\b(?:fire[- ]resistance(?: rating)?|fire rating|feuerwiderstand(?:sdauer)?|brandwiderstand)\b', re.I)
AS_BUILT = re.compile(r'\b(?:as[- ]built|field[- ]measured|site[- ]measured|aufgemessen\w*|aufmaß|bestandsaufmaß)\b', re.I)
CLEAR_WIDTH = re.compile(r'\b(?:clear (?:opening )?width|lichte\w* (?:öffnungs|durchgangs)?breite|durchgangsbreite)\b', re.I)
CERTIFIED = re.compile(r'\b(?:certified|zertifiziert\w*|nachgewiesen\w*)\b', re.I)
UNVERIFIED = re.compile(r'\b(?:not|no|none|nicht|kein\w*|unknown|unverified|uncertified|tbc|tbd|pending|'
                        r'proposed|required|recommended|target|design|nominal|planned|soll\w*|vorgesehen\w*|'
                        r'erforderlich\w*|minimum|maximum|at least|at most|approx\w*|ca)\b'
                        r'|[<>≤≥~±]', re.I)
MARK_HEADER = re.compile(r'(?:door(?: (?:mark|id|no\.?|number|tag))?|tür(?:kennung|nummer|[- ]?nr\.?)?|mark|kennzeichen)', re.I)
UNITS = {'min': ('fire_rating', 1), 'mm': ('as_built_width', 1),
         'cm': ('as_built_width', 10), 'm': ('as_built_width', 1000)}
UNIT = r'min(?:ute(?:s|n)?)?\.?|mm|cm|m'
QUANTITY = re.compile(rf'(?P<number>\d+(?:[.,]\d+)?)\s*(?P<unit>{UNIT})?', re.I)


def property_field(text, op):
    if op == 'fire_rating':
        return FIRE.search(text)
    return AS_BUILT.search(text) and CLEAR_WIDTH.search(text)


def canonical_unit(unit):
    return 'min' if unit.lower().startswith('min') else unit.lower()


def quantity(text, op, header_unit=''):
    match = QUANTITY.fullmatch(text.strip(' \t.:'))
    if not match:
        return None
    unit = canonical_unit(match['unit'] or header_unit)
    if unit not in UNITS or UNITS[unit][0] != op:
        return None
    number = Decimal(match['number'].replace(',', '.')) * UNITS[unit][1]
    return number if number > 0 else None


def box_for(words):
    boxes = [word['bbox'] for word in words]
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def visual_rows(lines):
    """Recover horizontal rows and split separate cells by their actual glyph gap."""
    words = [w for line in lines for w in line.get('words', ()) if w['text'].strip()]
    rows = []
    for word in sorted(words, key=lambda w: (sum(w['bbox'][1::2])/2, w['bbox'][0])):
        b = word['bbox']
        cy = (b[1]+b[3])/2
        if not rows or abs(cy - rows[-1][0]) > .45 * min(rows[-1][1], b[3]-b[1]):
            rows.append((cy, b[3]-b[1], []))
        rows[-1][2].append(word)
    for cy, height, row in rows:
        cells = []
        for word in sorted(row, key=lambda w: w['bbox'][0]):
            if not cells or word['bbox'][0] - cells[-1][-1]['bbox'][2] > 1.75 * height:
                cells.append([])
            cells[-1].append(word)
        yield cy, height, [dict(text=clean(' '.join(w['text'] for w in group)), bbox=box_for(group))
                            for group in cells]


def table_values(lines, op, mark, certified):
    header = None
    previous_y = None
    for row_index, (y, height, cells) in enumerate(visual_rows(lines)):
        ids = [i for i, cell in enumerate(cells) if MARK_HEADER.fullmatch(cell['text'])]
        fields = [i for i, cell in enumerate(cells) if property_field(cell['text'], op)]
        if ids or fields:
            header = None
            if len(ids) == len(fields) == 1:
                field = cells[fields[0]]
                if UNVERIFIED.search(field['text']) or (certified and not CERTIFIED.search(field['text'])):
                    continue
                bounds = [(a['bbox'][2]+b['bbox'][0])/2 for a, b in zip(cells, cells[1:])]
                unit_match = re.search(rf'[\[(]\s*({UNIT})\s*[\])]\s*$', field['text'], re.I)
                header = (bounds, ids[0], fields[0], field, unit_match[1] if unit_match else '', row_index)
                previous_y = y
            continue
        if header is None:
            continue
        bounds, id_col, value_col, field, header_unit, header_row = header
        if y-previous_y > 4*height:
            header = None
            continue
        by_col = {}
        for cell in cells:
            cx = (cell['bbox'][0]+cell['bbox'][2])/2
            col = sum(cx > boundary for boundary in bounds)
            by_col.setdefault(col, []).append(cell)
        door = by_col.get(id_col, [])
        if len(door) != 1 or not DOOR.fullmatch(door[0]['text']):
            header = None
            continue
        previous_y = y
        if door[0]['text'].upper() != mark:
            continue
        values = by_col.get(value_col, [])
        if not values:
            continue
        # Keep both the column meaning and the row identity with the value.
        text = ' '.join(c['text'] for c in values)
        yield quantity(text, op, header_unit), [
            evidence('schedule_field', field['bbox'], field['text'], f'schedule-column-{header_row}-{value_col}'),
            evidence('schedule_mark', door[0]['bbox'], mark, f'schedule-door-{row_index}'),
            evidence('schedule_value', box_for(values), text, f'schedule-value-{row_index}-{value_col}')]


def printed_property(scene, op, mark, certified=False):
    lines = scene['lines']
    if not any(property_field(clean(line['text']), op) for line in lines):
        return None, [], 'missing'
    records = []
    for i, line in enumerate(lines):
        text = clean(line['text'])
        marks = list(MARK.finditer(text))
        if len(marks) != 1 or marks[0].group().upper() != mark or not property_field(text, op):
            continue
        if certified and not CERTIFIED.search(text):
            continue
        # A labelled note must bind a single, explicit quantity to one door.
        # A second field, an interval or a design qualifier is ambiguous.
        body = text[:marks[0].start()] + ' ' + text[marks[0].end():]
        nums = list(re.finditer(rf'(?<![\w.+,-])\d+(?:[.,]\d+)?\s*(?:{UNIT})(?!\w)', body, re.I))
        has_other_number = len(re.findall(r'\d+(?:[.,]\d+)?', body)) != 1
        value = None
        if len(nums) == 1 and not has_other_number and not UNVERIFIED.search(body):
            label_end = FIRE.search(body).end() if op == 'fire_rating' else max(
                CLEAR_WIDTH.search(body).end(), AS_BUILT.search(body).end())
            number = nums[0]
            link = body[label_end:number.start()].strip(' \t:.=()')
            if label_end <= number.start() and link.lower() in ('', 'is', 'ist', 'beträgt'):
                value = quantity(number.group(), op)
        records.append((value, [evidence('schedule_note', line['bbox'], text, f'schedule-note-{i}')]))
    records.extend(table_values(lines, op, mark, certified))
    values = {v for v, _ in records}
    anchors = [item for _, items in records for item in items]
    if not records:
        return None, [], 'missing'
    if len(values) != 1 or None in values:
        return None, anchors, 'ambiguous'
    return values.pop(), anchors, 'documented'
