from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class InputDetection:
    kind: str               # "vector_pdf" | "raster_pdf" | "image" | "unknown"
    quality: str            # "high" | "medium" | "low"
    reason: str
    meta: Dict[str, Any]


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _pdf_image_area_ratio(pdf, page_index: int) -> float:
    """Best-effort: Anteil der sichtbaren Bildfläche an der Pagefläche."""
    try:
        breite, hoehe = pdf.seitengroesse(page_index)
        page_area = float(breite * hoehe) or 1.0
        img_area = 0.0
        for x0, y0, x1, y1 in pdf.bildrahmen(page_index):
            img_area += max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
        return float(img_area / page_area)
    except Exception:
        return 0.0


def detect_plan_input(
    path: str,
    *,
    max_pages: int = 1,
    vector_min_drawings: int = 1,
    vector_min_text_chars: int = 10,
    raster_min_img_area_ratio: float = 0.40,
) -> InputDetection:
    p = Path(path)
    ext = p.suffix.lower()

    if ext in IMAGE_EXTS:
        return InputDetection(
            kind="image",
            quality="medium",
            reason=f"extension={ext}",
            meta={"ext": ext},
        )

    if ext != ".pdf":
        return InputDetection(kind="unknown", quality="low", reason=f"unsupported_ext={ext}", meta={"ext": ext})

    # pdfium statt PyMuPDF (AGPL), 14.09.2026 - siehe planproof_geometry.pdf.
    try:
        from planproof_geometry.pdf import PdfDatei
    except Exception as e:
        return InputDetection(kind="unknown", quality="low", reason="missing_pdfium", meta={"error": str(e)})

    with PdfDatei(p) as pdf:
        page_count = min(pdf.seitenzahl, max_pages)
        drawings_total = 0
        text_chars_total = 0
        img_ratio_max = 0.0

        for i in range(page_count):
            try:
                drawings_total += pdf.linien(i).pfade
            except Exception:
                pass

            try:
                text_chars_total += len(pdf.seitentext(i).strip())
            except Exception:
                pass

            img_ratio_max = max(img_ratio_max, _pdf_image_area_ratio(pdf, i))

        # Klassifikation
        if drawings_total >= vector_min_drawings or text_chars_total >= vector_min_text_chars:
            return InputDetection(
                kind="vector_pdf",
                quality="high",
                reason=f"drawings={drawings_total}, text_chars={text_chars_total}",
                meta={"drawings": drawings_total, "text_chars": text_chars_total, "img_ratio_max": img_ratio_max},
            )

        if img_ratio_max >= raster_min_img_area_ratio:
            return InputDetection(
                kind="raster_pdf",
                quality="medium",
                reason=f"img_ratio_max={img_ratio_max:.2f}, drawings={drawings_total}, text_chars={text_chars_total}",
                meta={"drawings": drawings_total, "text_chars": text_chars_total, "img_ratio_max": img_ratio_max},
            )

        return InputDetection(
            kind="unknown",
            quality="low",
            reason=f"ambiguous: drawings={drawings_total}, text_chars={text_chars_total}, img_ratio_max={img_ratio_max:.2f}",
            meta={"drawings": drawings_total, "text_chars": text_chars_total, "img_ratio_max": img_ratio_max},
        )
