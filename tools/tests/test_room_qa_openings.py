"""Physical opening evidence, ambiguous assignments and sidedness."""
import sys
import unittest
from pathlib import Path

import pymupdf

from shapely.geometry import box
from tools.free_agent_room_qa.openings import marked_vector_profiles, polygon_sides, touches_scope

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'models/sheet-doors-v1'))
from vector_doors import inspect_page


def frame_from_drawing(walls='broken', profile=True, duplicate=False, swing=True, rotation=0, located=False):
    with pymupdf.open() as doc:
        page = doc.new_page(width=300, height=300)
        if walls == 'broken':
            for a,b in [((35,150),(115,150)),((165,150),(245,150))]:
                page.draw_line(a, b, width=4)
        elif walls == 'continuous':
            page.draw_line((35,150), (245,150), width=4)
        if profile:
            page.draw_rect((113,146,167,154), color=(.3,.3,.3), fill=(1,1,1), width=.7)
        if walls == 'overpainted':
            page.draw_line((35,150), (245,150), width=4)
        if swing:
            page.draw_bezier((165,150),(165,122.39),(142.61,100),(115,100),width=.85)
            page.draw_line((115,150),(115,100),width=.9)
        marks = {'D7':[dict(bbox=[132,146,144,154])]}
        if duplicate:
            marks['D9'] = [dict(bbox=[136,146,148,154])]
        if rotation:
            page.set_rotation(rotation)
            for labels in marks.values():
                for label in labels:
                    label['bbox'] = list(pymupdf.Rect(label['bbox']) * page.rotation_matrix)
        passages,profiles=marked_vector_profiles(page, marks, inspect_page(page)['swings'], {})
        return profiles if located else passages


class OpeningEvidenceTests(unittest.TestCase):
    def test_confirmed_swing_and_structure(self):
        frame = frame_from_drawing()['D7']
        self.assertEqual(frame['status'], 'supported_swing_and_wall_opening')
        self.assertEqual(len(frame['wall_support_points']), 2)

    def test_wall_gap_with_no_opaque_opening_rectangle(self):
        frame = frame_from_drawing(profile=False)['D7']
        self.assertTrue(frame['swing_evidence'])
        self.assertEqual(frame['opening_profile'], [])

    def test_swing_on_an_unbroken_wall_without_profile(self):
        self.assertFalse(frame_from_drawing(walls='continuous', profile=False))

    def test_no_wall_continuations(self):
        self.assertFalse(frame_from_drawing(walls='absent'))

    def test_wall_painted_over_the_opening(self):
        self.assertFalse(frame_from_drawing(walls='overpainted'))
        self.assertEqual(frame_from_drawing(walls='overpainted',located=True)['D7']['status'],
                         'centered_mark_in_structural_profile')

    def test_rotated_native_sheet_uses_displayed_coordinates(self):
        frame = frame_from_drawing(rotation=90)['D7']
        self.assertAlmostEqual(abs(frame['bearing_deg'])%180, 90)
        self.assertGreater(frame['bbox'][3]-frame['bbox'][1], frame['bbox'][2]-frame['bbox'][0])

    def test_scope_excludes_only_physically_distant_rooms(self):
        frame = frame_from_drawing()['D7']
        self.assertTrue(touches_scope(frame,[box(70,90,138,147), box(120,154,200,220)]))
        self.assertFalse(touches_scope(frame,[box(70,90,138,147), box(215,190,260,240)]))

    def test_two_labels_compete_for_one_opening(self):
        self.assertFalse(frame_from_drawing(duplicate=True))

    def test_two_distinct_interior_probes(self):
        frame = frame_from_drawing()['D7']
        rooms = {'A': {'polygon':[(20,20),(260,20),(260,146),(20,146)]},
                 'B': {'polygon':[(20,154),(260,154),(260,260),(20,260)]}}
        pair = polygon_sides(frame, rooms)
        self.assertEqual(pair['rooms'], ['A','B'])
        self.assertLess(pair['probes']['A'][1], 146)
        self.assertGreater(pair['probes']['B'][1], 154)
        self.assertIsNone(polygon_sides(frame, {'A':rooms['A']}))


if __name__ == '__main__':
    unittest.main()
