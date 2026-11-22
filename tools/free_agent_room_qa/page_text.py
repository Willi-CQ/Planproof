"""Ground literal sheet questions in printed words and their source coordinates."""
from bisect import bisect_left, bisect_right
import math
import re
import unicodedata


TEXT_OPS = {'locate_text', 'count_text', 'count_room_labels', 'count_tag_prefixes', 'inspect_object_count'}
QUOTES = re.compile(r'''"([^"\n]+)"|„([^“”\n]+)[“”]|“([^”\n]+)”|'([^'\n]+)'|‘([^’\n]+)’|«([^»\n]+)»''')
COUNT = r'(?:how many|count(?: all| the)?|wie viele|zähle(?: alle| die)?)'
PREFIX_LIST = r'[a-z]{1,8}-(?:\s*(?:,\s*(?:or\s+)?|\bor\b|\boder\b|\bund\b|\band\b)\s*[a-z]{1,8}-){0,5}'
EVIDENCE_LIMIT = 60
WORD = re.compile(r"[^\W_]+(?:[-/'][^\W_]+)*", re.UNICODE)
DASHES = str.maketrans('‐‑‒–—−', '------')


def tokens(text):
    return WORD.findall(unicodedata.normalize('NFKC', text).translate(DASHES).casefold())


def valid_literal(text):
    return isinstance(text, str) and 0 < len(text) <= 90 and 0 < len(tokens(text)) <= 12


def _scope(form):
    """Keep named sheet qualifiers explicit. Other restrictions must match the full grammar."""
    generic = re.search(r' (?:on|in) (?:this|the) (?:sheet|drawing|floor plan|plan)$|'
                        r' (?:auf|in) (?:diesem|dem|dieser|der) (?:blatt|plan|zeichnung|grundriss)$', form, re.I)
    if generic:
        return form[:generic.start()], None
    named = re.search(r' (?:on the|auf dem|im) ([\w -]{1,75})$', form, re.I)
    if named and re.search(r'\b(?:plan|level|floor|geschoss|ebene)\b', named[1], re.I):
        return form[:named.start()], named[1].strip()
    return form, None


def compile_text(question):
    """Return a typed literal operation or leave the question to the original compiler."""
    form = question.strip(' .!?').translate(DASHES)
    quotes = list(QUOTES.finditer(form))
    if len(quotes) == 1:
        quote = quotes[0]
        target = next(s for s in quote.groups() if s is not None).strip()
        form = form[:quote.start()] + '@' + form[quote.end():]
    else:
        target = None
    stem, title = _scope(form)
    if target is None and not quotes:
        # Bare targets are limited to explicit text-search requests; qualifiers stay outside the target.
        plain = re.fullmatch(r'((?:find|locate|search for|show me) (?:the )?(?:label|text|phrase)|'
                             r'(?:suche|finde|zeige(?: mir)?) (?:den text|die beschriftung))'
                             r' ([\w][\w /-]{0,75})', stem, re.I)
        if plain:
            target = plain[2].strip()
            stem = plain[1] + ' @'
    if valid_literal(target):
        locate = (r'(?:find|locate|search for|show me)(?: (?:the )?(?:label|text|phrase|word))? @|'
                  r'where (?:is|does) (?:the )?(?:(?:label|text|phrase|word) )?@(?: (?:appear|occur|printed|shown))?|'
                  r'(?:finde|suche|zeige(?: mir)?)(?: (?:nach )?(?:dem |den |der |die |das )?(?:text|wort|beschriftung))? @|'
                  r'wo (?:steht|finde ich|ist) (?:die beschriftung |der text |das wort )?@(?: (?:gedruckt|beschriftet))?')
        occurrences = (r'how many times (?:is @ (?:printed|written|shown)|does @ (?:appear|occur))|'
                       r'count (?:all |the )?(?:occurrences|instances|labels) (?:of )?@|'
                       r'wie (?:oft|häufig) (?:steht|erscheint|findet sich) @|wie oft (?:ist|wird) @ (?:gedruckt|angezeigt)')
        rooms = (r'(?:how many rooms are|count (?:the |all )?rooms) (?:explicitly )?(?:labelled|labeled|named|marked) @|'
                 r'wie viele räume (?:sind (?:ausdrücklich |explizit )?(?:(?:mit|als) )?@ (?:beschriftet|bezeichnet)|'
                 r'(?:heißen|tragen die (?:bezeichnung|beschriftung)) @)|'
                 r'zähle (?:die |alle )?räume mit der (?:bezeichnung|beschriftung) @')
        for op, pattern in [('locate_text', locate), ('count_text', occurrences), ('count_room_labels', rooms)]:
            if re.fullmatch(pattern, stem, re.I):
                program = {'op': op, 'args': {'text': target}}
                if title:
                    program['scope_title'] = title
                return program
        return None
    if quotes:
        return None
    fields = re.fullmatch(COUNT + r' (?:(?:distinct|different|unique|verschiedene) )?'
                          r'(?:equipment (?:identifiers|tags|ids)|(?:geräte|anlagen)kennungen|tags) '
                          r'(?:begin(?:ning)?(?: with)?|start(?:ing)? with|(?:beginnen|starten) mit|mit (?:dem )?präfix) (' + PREFIX_LIST + r')'
                          r'(?: (?:are (?:shown|printed|visible)|sind (?:gedruckt|sichtbar)))?', stem, re.I)
    if fields:
        prefixes = list(dict.fromkeys(re.findall(r'[a-z]{1,8}-(?!\w)', fields[1].upper(), re.I)))
        result = {'op':'count_tag_prefixes', 'args':{'prefixes':prefixes}}
        if title:
            result['scope_title'] = title
        return result
    # Object requests need geometry. The evidence reader can show text leads while explicitly
    # retaining that missing step, including qualifiers such as legend exclusions or plan views.
    if not re.match(COUNT + r'\b', form, re.I):
        return None
    if re.match(COUNT + r' (?:(?:separate|distinct|individual|steel|different|separaten?|'
                r'verschiedene[nr]?|einzelnen?) )*(?:stairs?|staircases?|stairways?|stairwells?|'
                r'(?:stahl)?treppen(?:häuser|kerne|läufe|standorte)?)\b', form, re.I):
        return {'op':'inspect_object_count', 'args':{'kind':'stairs'}}
    tagged = re.search(r'\b(?:identifiers?|tags?|ids?|kennungen)\s+(?:begin(?:ning)?(?: with)?|'
                       r'start(?:ing)? with|mit (?:dem )?präfix)\s+(' + PREFIX_LIST + r')', form, re.I)
    if tagged:
        prefixes = list(dict.fromkeys(re.findall(r'[A-Z]{1,8}-', tagged[1].upper())))
        return {'op':'inspect_object_count', 'args':{'kind':'tagged_objects','prefixes':prefixes}}
    return None


def union_box(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _word_runs(scene):
    runs = []
    for row in scene['lines']:
        run = []
        for word in row.get('words', []):
            box = word.get('bbox')
            if (not isinstance(box, (tuple, list)) or len(box) != 4 or
                    not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box) or
                    not (0 <= box[0] < box[2] <= scene['width'] and 0 <= box[1] < box[3] <= scene['height'])):
                continue
            if run:
                prior = run[-1]['bbox']
                height = max(box[3]-box[1], prior[3]-prior[1])
                if box[0] - prior[2] > 2.5 * height:
                    runs.append(run)
                    run = []
            run.append(word)
        if run:
            runs.append(run)
    return runs


def _overlaps(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return intersection >= .55 * min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))


def _matches(runs, target, standalone):
    """Find contiguous words. A stacked label may span two closely aligned short lines."""
    rows = []
    found = []
    for run in runs:
        units = [(part, word) for word in run for part in tokens(word['text'])]
        names = [name for name, _ in units]
        if not names:
            continue
        for i in range(len(names) - len(target) + 1):
            if names[i:i + len(target)] != target or (standalone and names != target):
                continue
            found.append(union_box([word['bbox'] for _, word in units[i:i+len(target)]]))
        if len(names) < len(target) and (target[:len(names)] == names or target[-len(names):] == names):
            rows.append((names, union_box([word['bbox'] for _, word in units])))
    rows.sort(key=lambda row: row[1][1])
    tops = [row[1][1] for row in rows]
    for names, box in rows:
        if names != target[:len(names)]:
            continue
        height = box[3] - box[1]
        candidates = []
        for tail, other in rows[bisect_left(tops, box[3]-.15*height):bisect_right(tops, box[3]+1.1*height)]:
            if (names + tail == target and
                    .6 * height <= other[3]-other[1] <= 1.6 * height and
                    min(abs(box[0]-other[0]), abs((box[0]+box[2]-other[0]-other[2])/2)) <= .8 * height):
                candidates.append(other)
        if len(candidates) == 1:
            found.append(union_box([box, candidates[0]]))
    return found


def _hits(runs, args, op):
    if 'text' in args:
        return [(args['text'], box) for box in _matches(runs, tokens(args['text']), op == 'count_room_labels')]
    result = []
    prefixes = args.get('prefixes', [])
    tag = re.compile(r'(?:' + '|'.join(map(re.escape, prefixes)) + r')\d+[A-Z0-9]*(?:-[A-Z0-9]+)*', re.I)
    stairs = {'stair','stairs','staircase','stairwell','stairway','treppe','treppenhaus','treppenkern'}
    for run in runs:
        for index, word in enumerate(run):
            raw = word['text'].translate(DASHES).strip('.,;:()[]').upper()
            if args.get('kind') == 'stairs':
                if raw.casefold() in stairs:
                    result.append((raw, word['bbox']))
            elif tag.fullmatch(raw):
                result.append((raw, word['bbox']))
            elif raw in prefixes or raw + '-' in prefixes:
                combined, boxes = raw, [word['bbox']]
                for right in run[index+1:index+3]:
                    gap = right['bbox'][0] - boxes[-1][2]
                    if not -.2 <= gap/(word['bbox'][3]-word['bbox'][1]) <= .65:
                        break
                    combined += right['text'].translate(DASHES).strip('.,;:()[]').upper()
                    boxes.append(right['bbox'])
                    if tag.fullmatch(combined):
                        result.append((combined, union_box(boxes)))
                        break
                    if combined not in prefixes:
                        break
    return result


def text_answer(scene, program):
    op, args = program['op'], program.get('args')
    error = {'reason_code':'unsupported_question', 'reason':'A literal sheet label or identifier prefix is required.'}
    if not isinstance(args, dict):
        return error
    if op in ('locate_text', 'count_text', 'count_room_labels'):
        if set(args) != {'text'} or not valid_literal(args['text']):
            return error
    elif op == 'inspect_object_count' and args == {'kind':'stairs'}:
        pass
    else:
        keys = {'prefixes'} if op == 'count_tag_prefixes' else {'kind','prefixes'}
        if (set(args) != keys or ('kind' in args and args['kind'] != 'tagged_objects') or
                not isinstance(args['prefixes'], list) or not 0 < len(args['prefixes']) <= 6 or
                not all(isinstance(p, str) and re.fullmatch(r'[A-Z]{1,8}-', p) for p in args['prefixes'])):
            return error
    runs = _word_runs(scene)
    items = []
    clipped = False
    for text, box in _hits(runs, args, op):
        if any(text.casefold() == old['text'].casefold() and _overlaps(box, old['bbox']) for old in items):
            continue
        if len(items) == EVIDENCE_LIMIT:
            clipped = True
            break
        items.append({'id':f'printed-{len(items)+1}', 'kind':'printed_label', 'bbox':box, 'text':text})
    if op == 'inspect_object_count':
        reason = 'Counting physical objects in the requested view requires symbol and boundary evidence. A printed label may also occur in a note or legend.'
        result = {'reason_code':'visual_count_required', 'reason':reason, 'items':items}
        if items:
            result.update(status='partial', value=f'{len(items)} printed label locations',
                          calculation='Located related text for inspection. The number shown describes text locations only.')
        return result
    if not items:
        return {'reason_code':'text_not_found', 'reason':'No matching printed text was verified on this sheet. Its absence is unconfirmed.'}
    scope = program.get('scope_title')
    if scope:
        if not valid_literal(scope):
            return error
        anchors = _matches(runs, tokens(scope), True)
        if not anchors:
            return {'status':'partial', 'value':f'{len(items)} matching label locations on uploaded sheet',
                    'items':items, 'calculation':'Located the matching printed text on the supplied page.',
                    'reason_code':'scope_unverified', 'reason':'The named plan or level could not be verified. The marked text locations belong to the uploaded page.'}
    total = len({i['text'].upper() for i in items}) if op == 'count_tag_prefixes' else len(items)
    if scope:
        items.append({'id':'scope-title', 'kind':'sheet_title', 'bbox':anchors[0], 'text':scope})
    if op == 'locate_text':
        return {'status':'answered', 'value':args['text'], 'items':items,
                'calculation':'Located the requested printed phrase in the displayed sheet coordinates.'}
    partial = clipped or not scene['native_text_complete'] or op == 'count_room_labels'
    basis = {'count_text':'Counted spatially distinct occurrences of the requested printed text.',
             'count_room_labels':'Counted distinct locations bearing the requested room label.',
             'count_tag_prefixes':'Counted distinct printed identifiers with a requested prefix.'}[op]
    result = {'status':'partial' if partial else 'answered', 'value':total, 'unit':'count', 'items':items, 'calculation':basis}
    if partial:
        reasons = []
        if not scene['native_text_complete']:
            reasons.append('OCR may miss labels on the sheet.')
        if op == 'count_room_labels':
            reasons.append('A physical room total requires confirming room boundaries and excluding legends and schedules.')
        if clipped:
            reasons.append('The evidence limit was reached.')
        result.update(reason_code='incomplete_coverage', reason=' '.join(reasons))
    return result
