"""Lossless quarter-turn inspection views with inverse source-local coordinates."""

import base64
from dataclasses import replace
from io import BytesIO

from PIL import Image


def validate_rotation(degrees):
    if type(degrees) is not int or degrees not in (0, 90, 180, 270):
        raise ValueError("Inspection rotation must be 0, 90, 180 or 270 clockwise degrees")


def rotated_views(images, degrees):
    validate_rotation(degrees)
    if not degrees:
        return images
    operation = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180,
                 270: Image.Transpose.ROTATE_90}[degrees]
    views = []
    for image in images:
        with Image.open(BytesIO(base64.b64decode(image.data_url.split(",", 1)[1], validate=True))) as pixels:
            turned = pixels.transpose(operation)
            stream = BytesIO()
            turned.save(stream, format="PNG")
        views.append(replace(image, data_url="data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii"),
                             metadata={**image.metadata, "sent_size": list(turned.size)}))
    return views


def original_box(box, degrees):
    """Map normalized displayed-view edges to normalized original-crop edges."""
    validate_rotation(degrees)
    x0, y0, x1, y1 = box
    if degrees == 90:
        return [y0, 1000 - x1, y1, 1000 - x0]
    if degrees == 180:
        return [1000 - x1, 1000 - y1, 1000 - x0, 1000 - y0]
    if degrees == 270:
        return [1000 - y1, x0, 1000 - y0, x1]
    return list(box)
