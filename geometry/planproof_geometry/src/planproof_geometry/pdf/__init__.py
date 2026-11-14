"""PDF-Zugriff fuer API und Worker - ueber pdfium, nicht mehr PyMuPDF (14.09.2026).

Die Begruendung und die gemessene Semantik stehen in :mod:`.pdfium_adapter`.
"""

from .pdfium_adapter import (
    BEZIER_SCHRITTE,
    Gerendert,
    Linien,
    PdfDatei,
    PdfQuelle,
    Rahmen,
    TextSpan,
    png_aus_feld,
    seitenzahl,
)

__all__ = [
    "BEZIER_SCHRITTE",
    "Gerendert",
    "Linien",
    "PdfDatei",
    "PdfQuelle",
    "Rahmen",
    "TextSpan",
    "png_aus_feld",
    "seitenzahl",
]
