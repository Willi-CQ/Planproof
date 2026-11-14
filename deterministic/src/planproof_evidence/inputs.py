"""PNG and bounded PDF render/native extraction with crop/rotation-aware transforms."""

from __future__ import annotations

import ctypes
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .contracts import PageFrame, rgb_hash, sha256, transform_points


def _pdf_matrix(page, width, height, raw):
    """Derive affine from exact canvas basis rather than rounded page-to-device points."""
    points = []
    for x, y in ((0, 0), (width, 0), (0, height)):
        px, py = ctypes.c_double(), ctypes.c_double()
        if not raw.FPDF_DeviceToPage(page, 0, 0, width, height, 0, x, y, px, py):
            raise ValueError("PDF coordinate conversion failed")
        points.append([px.value, py.value])
    p, x, y = np.asarray(points)
    inverse = np.array(
        [
            [(x[0] - p[0]) / width, (y[0] - p[0]) / height, p[0]],
            [(x[1] - p[1]) / width, (y[1] - p[1]) / height, p[1]],
            [0, 0, 1],
        ]
    )
    return np.linalg.inv(inverse)


def _native_paths(page, matrix, size, raw, budget=10000):
    from shapely.geometry import LineString, box

    segments, curves, clipped = [], 0, 0
    bounds = box(0, 0, *size)

    def walk(parent, parent_matrix, depth=0, form=False):
        nonlocal curves, clipped
        if depth > 8:
            clipped += 1
            return
        count = raw.FPDFFormObj_CountObjects(parent) if form else raw.FPDFPage_CountObjects(parent)
        for i in range(count):
            if len(segments) >= budget:
                clipped += 1
                return
            obj = (
                raw.FPDFFormObj_GetObject(parent, i) if form else raw.FPDFPage_GetObject(parent, i)
            )
            objtype = raw.FPDFPageObj_GetType(obj)
            ctm = raw.FS_MATRIX()
            if not raw.FPDFPageObj_GetMatrix(obj, ctm):
                clipped += 1
                continue
            own = np.array([[ctm.a, ctm.c, ctm.e], [ctm.b, ctm.d, ctm.f], [0, 0, 1]])
            combined = parent_matrix @ own
            if objtype == raw.FPDF_PAGEOBJ_FORM:
                walk(obj, combined, depth + 1, True)
            elif objtype == raw.FPDF_PAGEOBJ_PATH:
                previous, start = None, None
                for j in range(raw.FPDFPath_CountSegments(obj)):
                    if len(segments) >= budget:
                        clipped += 1
                        return
                    segment = raw.FPDFPath_GetPathSegment(obj, j)
                    x, y = ctypes.c_float(), ctypes.c_float()
                    if not raw.FPDFPathSegment_GetPoint(segment, x, y):
                        continue
                    point = transform_points([[x.value, y.value]], matrix @ combined)[0]
                    kind = raw.FPDFPathSegment_GetType(segment)
                    if kind == raw.FPDF_SEGMENT_MOVETO:
                        start = point
                    elif kind == raw.FPDF_SEGMENT_LINETO and previous is not None:
                        line = LineString([previous, point]).intersection(bounds)
                        if line.geom_type == "LineString" and line.length > 0.1:
                            segments.append([list(line.coords[0]), list(line.coords[-1])])
                    elif kind == raw.FPDF_SEGMENT_BEZIERTO:
                        curves += 1
                    previous = point
                    if raw.FPDFPathSegment_GetClose(segment) and start is not None:
                        line = LineString([point, start]).intersection(bounds)
                        if line.geom_type == "LineString" and line.length > 0.1:
                            segments.append([list(line.coords[0]), list(line.coords[-1])])

    walk(page, np.eye(3))
    return segments, {
        "bezier_control_points_skipped": curves,
        "truncated_objects": clipped,
        "coverage": "native straight paths; raster handles curves/images; clipping paths not semantically interpreted",
    }


def _native_text(page, matrix, size):
    spans, text, boxes = [], [], []
    textpage = page.get_textpage()

    def flush():
        if text and boxes:
            a = np.asarray(boxes)
            bounds = [
                max(0.0, float(a[:, 0].min())),
                max(0.0, float(a[:, 1].min())),
                min(float(size[0]), float(a[:, 2].max())),
                min(float(size[1]), float(a[:, 3].max())),
            ]
            if bounds[2] > bounds[0] and bounds[3] > bounds[1]:
                spans.append(
                    {
                        "id": f"native-text-{len(spans):05d}",
                        "text": "".join(text),
                        "bbox": bounds,
                        "source_family": "native_pdf",
                    }
                )
        text.clear()
        boxes.clear()

    try:
        for i in range(min(textpage.count_chars(), 100000)):
            char = textpage.get_text_range(i, 1)
            if char.isspace() or not char:
                flush()
                continue
            x0, y0, x1, y1 = textpage.get_charbox(i)
            corners = np.asarray(transform_points([[x0, y0], [x1, y1], [x0, y1], [x1, y0]], matrix))
            boxes.append([*corners.min(axis=0), *corners.max(axis=0)])
            text.append(char)
        flush()
    finally:
        textpage.close()
    return spans


def load_pages(input_path, config):
    path = Path(input_path)
    source_hash = sha256(path)
    document_id = "sha256:" + source_hash
    if path.suffix.lower() == ".png":
        with Image.open(path) as source:
            # Preserve raw stored dimensions; never silently EXIF-transpose after ML binding.
            image = np.asarray(source.convert("RGB")).copy()
        identity = np.eye(3).tolist()
        yield (
            image,
            PageFrame(
                document_id,
                0,
                source_hash,
                rgb_hash(image),
                [image.shape[1], image.shape[0]],
                identity,
                identity,
                {"format": "PNG", "source_units": "pixels", "exif_orientation_applied": False},
            ),
            [],
            [],
            {},
        )
        return
    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PNG and PDF inputs are supported")
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw

    pdf = pdfium.PdfDocument(str(path))
    total_pixels = 0
    try:
        for index in range(len(pdf)):
            page = pdf[index]
            width, height = page.get_size()
            scale = min(
                config.pdf_dpi / 72, math.sqrt(config.pdf_max_pixels / max(width * height, 1))
            )
            estimated_pixels = math.ceil(width * scale) * math.ceil(height * scale)
            if total_pixels + estimated_pixels > config.pdf_total_pixel_budget:
                raise ValueError("PDF total rendering pixel budget exceeded; split into page jobs")
            total_pixels += estimated_pixels
            bitmap = page.render(scale=scale)
            image = np.asarray(bitmap.to_pil().convert("RGB")).copy()
            size = [image.shape[1], image.shape[0]]
            matrix = _pdf_matrix(page, *size, raw)
            lines, coverage = _native_paths(page, matrix, size, raw)
            text = _native_text(page, matrix, size)
            metadata = {
                "format": "PDF",
                "source_units": "PDF_points",
                "rotation": page.get_rotation(),
                "cropbox": list(page.get_cropbox()),
                "mediabox": list(page.get_mediabox()),
                "render_scale": scale,
                "renderer": "pypdfium2-4.30.0",
                "native_coverage": coverage,
            }
            frame = PageFrame(
                document_id,
                index,
                source_hash,
                rgb_hash(image),
                size,
                matrix.tolist(),
                np.linalg.inv(matrix).tolist(),
                metadata,
            )
            bitmap.close()
            page.close()
            yield image, frame, lines, text, coverage
    finally:
        pdf.close()
