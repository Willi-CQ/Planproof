"""Regression checks for German/English sheet queries and cited text sightings."""
import pytest

from tools.free_agent_room_qa import answer, answer_program, compile_question


def line(text, x, y):
    words = []
    for part in text.split():
        width = len(part) * 6
        words.append({'text':part, 'bbox':[x, y, x+width, y+10]})
        x += width + 3
    return {'text':text, 'words':words, 'bbox':[words[0]['bbox'][0], y, x-3, y+10]}


def scene(*lines, native=False):
    return {'page':1, 'width':500, 'height':500, 'coordinates':'image_pixels',
            'source_sha256':'unit-fixture', 'lines':list(lines), 'native_text_complete':native}


@pytest.mark.parametrize('question', [
    'Find the label "North Hall".', 'Wo steht „North Hall“?', 'Locate the text North Hall.',
    'Zeige mir die Beschriftung „North Hall“ auf diesem Plan.',
])
def test_source_bound_term_locations(question):
    data = scene(line('North Hall', 45, 20), line('North Hallway', 45, 90))
    result = answer(data, question)
    assert result['program']['op'] == 'locate_text' and result['status'] == 'answered'
    assert result['value'] == 'North Hall' and len(result['evidence']) == 1
    assert result['evidence'][0]['bbox'] == [45, 20, 102, 30]


def test_distinct_stacked_room_labels_text_and_explicit_partial():
    data = scene(line('Group', 30, 40), line('Room', 32, 53), line('GROUP ROOM', 180, 50),
                 line('Group Room', 181, 50), line('Group Room Storage', 270, 70))
    result = answer(data, 'How many rooms are explicitly labeled "Group Room" on this plan?')
    assert result['program']['op'] == 'count_room_labels'
    assert (result['status'], result['value'], len(result['evidence'])) == ('partial', 2, 2)
    assert result['reason_code'] == 'incomplete_coverage'


@pytest.mark.parametrize('question', ['Wie viele Räume sind als „Ruheraum“ beschriftet?',
                                     'Wie viele Räume heißen „Ruheraum“?'])
def test_german_room_labels(question):
    result = answer(scene(line('Ruheraum', 10, 10)), question)
    assert (result['program']['op'], result['status'], result['value']) == ('count_room_labels','partial',1)


def test_unverified_named_scope_never_yields_numeric_room_count():
    question = 'How many rooms are explicitly labeled "Quiet Room" on the Learning Plan Level 02?'
    data = scene(line('Quiet Room', 35, 80), line('Learning Plan Level 00', 5, 12))
    result = answer(data, question)
    assert result['reason_code'] == 'scope_unverified' and isinstance(result['value'], str)
    data['lines'].append(line('Learning Plan Level 02', 35, 320))
    result = answer(data, question)
    assert result['status'] == 'partial' and result['value'] == 1
    assert any(item['kind'] == 'sheet_title' for item in result['evidence'])


@pytest.mark.parametrize(('native', 'status'), [(True, 'answered'), (False, 'partial')])
def test_text_count_precision(native, status):
    result = answer(scene(line('Mix A Mix', 5, 15), line('Mixture', 25, 105), native=native),
                    'Wie oft steht „Mix“ auf diesem Blatt?')
    assert (result['status'], result['value'], len(result['evidence'])) == (status, 2, 2)


def test_tag_count_uses_complete_printed_identifiers_and_deduplicates_references():
    data = scene(line('PB-72 PB-73', 20, 30), line('PB-72', 80, 80), line('PB- XX-10 XPB-9', 70, 160))
    result = answer(data, 'How many distinct equipment tags beginning with PB- are printed on this sheet?')
    assert (result['status'], result['value']) == ('partial', 2)
    assert [item['text'] for item in result['evidence']] == ['PB-72', 'PB-73', 'PB-72']
    result = answer(data, 'How many pump units with equipment identifiers beginning PB- are drawn in the plant plan? Exclude diagram references.')
    assert result['reason_code'] == 'visual_count_required' and isinstance(result['value'], str)


def test_unseen_term_and_unresolved_object_scope_never_imply_zero():
    data = scene(line('East Hall', 60, 16))
    assert answer(data, 'Find the label "West Hall".')['value'] is None
    result = answer(data, 'How many separate stair locations are shown on the upper floor? Exclude its legend.')
    assert result['status'] == 'unresolved' and result['reason_code'] == 'visual_count_required'
    data['lines'].append(line('STAIR', 190, 215))
    result = answer(data, 'How many distinct steel stair locations appear in this detail?')
    assert result['status'] == 'partial' and isinstance(result['value'], str)
    assert result['reason_code'] == 'visual_count_required'


def test_semantic_constraints_and_invalid_programs_fail_closed():
    assert compile_question('How many doors directly connect to STAIRS?')['op'] == 'room_door_count'
    assert compile_question('How many stairs should we remove?')['op'] == 'unsupported'
    assert compile_question('Is "North Hall" safe for a 50-tonne machine?')['op'] == 'unsupported'
    assert compile_question('How many rooms are labeled "North Hall" except on the second floor?')['op'] == 'unsupported'
    data = scene()
    result = answer_program(data, {'op':'count_tag_prefixes', 'args':{'prefixes':['.*']}})
    assert result['value'] is None and result['status'] == 'unresolved'
