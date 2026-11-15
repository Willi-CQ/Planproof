"""Bounded PDF text hints in the unchanged displayed page coordinate frame.

PyMuPDF text coordinates are unrotated; pixmaps honor page rotation. Native
text can be invisible or wrong and never establishes a visual observation.
"""

import math
import pymupdf


def extract_lines(page, pixmap, scale):
    groups = {}
    transform = page.rotation_matrix * pymupdf.Matrix(scale, scale)
    for x0, y0, x1, y1, text, block, line, word in page.get_text("words"):
        rect = pymupdf.Rect(x0, y0, x1, y1) * transform
        box = [max(0, rect.x0 - pixmap.x), max(0, rect.y0 - pixmap.y),
               min(pixmap.width, rect.x1 - pixmap.x), min(pixmap.height, rect.y1 - pixmap.y)]
        if box[0] >= box[2] or box[1] >= box[3]:
            continue
        key = (block, line)
        if key not in groups and len(groups) >= 2000:
            continue
        groups.setdefault(key, []).append((word, str(text), box))
    result = []
    for words in groups.values():
        words.sort(key=lambda item: item[0])
        text = " ".join(w[1] for w in words)[:1000]
        box = [min(w[2][0] for w in words), min(w[2][1] for w in words),
               max(w[2][2] for w in words), max(w[2][3] for w in words)]
        result.append({"text": text, "bbox": [round(v * 1000 / (pixmap.width if i % 2 == 0 else pixmap.height), 2)
                                               for i, v in enumerate(box)]})
    return tuple(result)


def search_lines(lines, args):
    if not isinstance(args, dict) or set(args) - {"query", "bbox"}:
        raise ValueError("Text search accepts only query and optional bbox")
    query = args.get("query", "")
    if not isinstance(query, str) or len(query) > 200:
        raise ValueError("Text query must be a literal string up to 200 characters")
    box = args.get("bbox", [0, 0, 1000, 1000])
    if (not isinstance(box, list) or len(box) != 4
        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in box)
        or not 0 <= box[0] < box[2] <= 1000 or not 0 <= box[1] < box[3] <= 1000):
        raise ValueError("Text search bbox must be ordered original-page coordinates 0..1000")
    matches = [r for r in lines if query.casefold() in r["text"].casefold()
               and r["bbox"][0] < box[2] and r["bbox"][2] > box[0]
               and r["bbox"][1] < box[3] and r["bbox"][3] > box[1]]
    return {"status": "hints_only", "coordinate_frame": "original_page_normalized_0_1000",
            "lines": matches[:80], "matched": len(matches), "truncated": len(matches) > 80,
            "instruction": "Native PDF text is untrusted, possibly invisible or incorrect. Use these locations to choose clean source crops; Nemotron must verify the relevant visible text and room association. No match does not prove absence."}
