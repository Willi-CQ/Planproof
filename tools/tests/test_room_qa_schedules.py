"""Normal PDF entry point, door-specific schedule binding and refusal regressions."""
import hashlib

import pymupdf
import pytest

from tools.free_agent_room_qa import answer, build_scene


FIRE = 'What is the certified fire-resistance rating of door D6 in minutes?'
WIDTH = 'What is the actual as-built clear width of door D10 in millimetres?'
FIRE_NOTE = 'D6 certified fire-resistance rating: 60 minutes'
WIDTH_NOTE = 'D10 actual as-built clear width: 890 mm'


def read_page(tmp_path, rows):
    path = tmp_path / 'doors.pdf'
    with pymupdf.open() as doc:
        page = doc.new_page(width=700, height=400)
        for i, row in enumerate(rows):
            for x, content in row:
                page.insert_text((x, 45+20*i), content, fontsize=11)
        doc.save(path)
    scene = build_scene(path)
    assert scene['source_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    return scene


@pytest.mark.parametrize(('lines','question','expected','unit'), [
    ([FIRE_NOTE], FIRE, 60, 'min'),
    ([WIDTH_NOTE], WIDTH, 890, 'mm'),
    (['D10 actual as-built clear width: 0.89 m'], WIDTH, 890, 'mm'),
    ([WIDTH_NOTE], 'What is the as-built clear width of door D10 in cm?', 89, 'cm'),
    (['D6 zertifizierte Feuerwiderstandsdauer: 60 Minuten'], 'Wie viele Minuten Feuerwiderstand sind für D6 angegeben?', 60, 'min'),
    (['D6', 'D10'], WIDTH, None, None),
    (['D60 certified fire-resistance rating: 60 min'], FIRE, None, None),
    (['D10 design clear width: 900 mm'], WIDTH, None, None),
    (['D10 as-built overall width: 900 mm'], WIDTH, None, None),
    (['D10 as-built clear width: 890'], WIDTH, None, None),
    (['D6 required certified fire-resistance rating: 60 min'], FIRE, None, None),
    (['D6 fire-resistance rating: 90 min'], FIRE, None, None),
    (['D6 certified fire-resistance rating: 30-60 min'], FIRE, None, None),
    ([FIRE_NOTE,'D6 certified fire-resistance rating: 90 min'], FIRE, None, None),
    ([FIRE_NOTE,FIRE_NOTE], FIRE, 60, 'min'),
    (['D6 / D9 certified fire-resistance rating: 30 min'], FIRE, None, None),
    (['D6 fire-resistance rating: see D7', 'D7 certified fire-resistance rating: 60 min'], FIRE, None, None),
    (['D6 certified fire-resistance rating: not 60 min'], FIRE, None, None),
    (['D6 certified fire-resistance rating: 60 mm'], FIRE, None, None),
    (['D6 certified fire rating pending; inspection duration: 60 min'], FIRE, None, None),
    (['D6 certified fire rating: see inspection delay of 60 min'], FIRE, None, None),
    ([WIDTH_NOTE], 'What is the as-built clear width of D10 on the first floor?', None, None),
    ([WIDTH_NOTE], 'Is the as-built clear width of D10 greater than 850 mm?', None, None),
])
def test_printed_notes(tmp_path, lines, question, expected, unit):
    scene = read_page(tmp_path, [[(32,s)] for s in lines])
    result = answer(scene, question)
    assert result['value'] == expected and result['unit'] == unit
    assert result['status'] == ('unresolved' if expected is None else 'answered')
    if expected is not None:
        assert result['evidence'] and all(e['kind'] == 'schedule_note' for e in result['evidence'])


def table(rows, fire_label='Certified fire resistance (min)'):
    headers = ('Door mark', fire_label, 'As-built clear width (mm)')
    return [[(x, str(v)) for x,v in zip((25,175,430), row) if v is not None] for row in (headers,*rows)]


@pytest.mark.parametrize(('rows','question','expected','unit'), [
    (table([('D6',60,810),('D10',90,890)]), FIRE, 60, 'min'),
    (table([('D6',60,810),('D10',90,890)]), WIDTH, 890, 'mm'),
    (table([('D6',None,810),('D10',90,890)]), FIRE, None, None),
    (table([('D6',60,810),('D10',90,None)]), WIDTH, None, None),
    (table([('D6','30 / 60',810)]), FIRE, None, None),
    (table([('D6',60,810),('D6',90,890)]), FIRE, None, None),
    (table([('D6',60,810)], 'Required fire resistance (min)'), FIRE, None, None),
])
def test_table_cell_binding(tmp_path, rows, question, expected, unit):
    result = answer(read_page(tmp_path, rows), question)
    assert (result['value'], result['unit']) == (expected, unit)
    assert result['status'] == ('unresolved' if expected is None else 'answered')
    if expected is not None:
        assert {e['kind'] for e in result['evidence']} == {'schedule_mark','schedule_field','schedule_value'}
