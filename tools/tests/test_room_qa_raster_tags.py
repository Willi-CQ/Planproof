import numpy as np
from PIL import Image

from tools.free_agent_room_qa import raster_tags


def test_truncated_words_on_either_axis():
    page = np.full((60, 90), 255, dtype=np.uint8)
    page[20:32, 15:20] = 0
    page[20:32, 23:28] = 0
    page[20:32, 31:37] = 0
    assert raster_tags.has_continuation(page, (23, 20, 37, 32), False)
    assert raster_tags.has_continuation(page.T, (20, 23, 32, 37), True)
    assert not raster_tags.has_continuation(page, (15, 20, 37, 32), False)


def test_digit_repair_needs_visible_prefix_and_shape_advantage(monkeypatch):
    distances = {'D06': .03, 'DO6': .11, 'D6': .13}
    monkeypatch.setattr(raster_tags, 'shape_distance', lambda _, value: distances[value])
    assert raster_tags.read_token(('DO6', 80), None, ('D6', 30), None)[0] == 'D06'
    assert raster_tags.read_token(('006', 95), None, ('D06', 95), None) is None
    distances['DO6'] = .031
    assert raster_tags.read_token(('DO6', 80), None, ('D06', 85), None) is None
    assert raster_tags.read_token(('DO6', 80), None, ('D07', 85), None) is None


def test_no_ocr_binary(monkeypatch):
    monkeypatch.setattr(raster_tags.shutil, 'which', lambda _: None)
    assert raster_tags.read_raster_tags(Image.new('L', (20, 20), 255), [], 20, 20, []) == []
