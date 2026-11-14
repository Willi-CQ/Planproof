from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

BBox = Tuple[float, float, float, float]  # x0,y0,x1,y1 (PDF coords)


@dataclass
class TextSpan:
    id: str
    text: str
    bbox: BBox
    page: int = 0
    rotation: float = 0.0
    conf: float = -1.0  # OCR confidence (0..100) or -1 if unknown


@dataclass
class LineSeg:
    id: str
    a: Tuple[float, float]
    b: Tuple[float, float]
    page: int = 0


@dataclass
class PlanExtraction:
    doc: Dict[str, Any]
    texts: List[TextSpan]
    lines: List[LineSeg]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc": self.doc,
            "texts": [asdict(t) for t in self.texts],
            "geometry": [
                {"id": linie.id, "type": "line", "page": linie.page, "a": linie.a, "b": linie.b}
                for linie in self.lines
            ],
        }


def extract_pdf_vector(pdf_path: str, max_pages: int = 1) -> PlanExtraction:
    """
    Vektorbasierte Extraktion (kein OCR):
    - Text inkl. Bounding-Boxes (je Span)
    - Liniensegmente aus allen sichtbaren Pfaden: Linien, Rechtecke, Quads und
      kubische Kurven (8 Stuetzstellen je Kurve)

    Seit 14.09.2026 ueber pdfium (``planproof_geometry.pdf``) statt PyMuPDF (AGPL).
    Koordinaten wie bisher: ungedrehter Seitenraum, Ursprung oben links.
    """
    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(pdf_path)

    from planproof_geometry.pdf import PdfDatei

    texts: list[TextSpan] = []
    lines: list[LineSeg] = []

    tid = 1
    gid = 1

    with PdfDatei(p) as pdf:
        page_count = min(pdf.seitenzahl, max_pages)
        doc_meta: dict[str, Any] = {
            "source_file": str(p),
            "pages": page_count,
            "coordinate_system": "pdf_points",
            "page_size": {"w": 0.0, "h": 0.0},
        }
        for page_index in range(page_count):
            breite, hoehe = pdf.seitengroesse(page_index)

            # 1) Text
            for span in pdf.textspans(page_index):
                text = span.text.strip()
                if not text:
                    continue
                # rotation ist in PDFs tricky; für v0 lassen wir 0
                texts.append(TextSpan(
                    id=f"t{tid}",
                    text=text,
                    bbox=(float(span.x0), float(span.y0), float(span.x1), float(span.y1)),
                    page=page_index,
                    rotation=0.0,
                ))
                tid += 1

            # 2) Vektor-Linien. CAD/Architektur-PDFs nutzen oft Rechtecke u. Kurven
            # statt einfacher Linien - der Adapter zerlegt alle in Segmente.
            try:
                for ax, ay, bx, by in pdf.linien(page_index).segmente:
                    lines.append(LineSeg(id=f"g{gid}", a=(ax, ay), b=(bx, by), page=page_index))
                    gid += 1
            except Exception:
                pass

            doc_meta["page_size"] = {"w": float(breite), "h": float(hoehe)}

    return PlanExtraction(doc=doc_meta, texts=texts, lines=lines)
