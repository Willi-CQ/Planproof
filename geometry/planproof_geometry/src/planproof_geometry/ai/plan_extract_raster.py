from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union
import math

import numpy as np

from planproof_geometry.ai.plan_extract_pdf import PlanExtraction, TextSpan, LineSeg  # reuse dataclasses
from planproof_geometry.ai.cv_compat import hough_lines_p
from planproof_geometry.ai.ocr_setup import configure_tesseract

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _cv2_gray_from_path(image_path: str) -> np.ndarray:
    import cv2
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    return img


def _render_pdf_page_gray(pdf_path: str, page_index: int = 0, target_max_px: int = 3500) -> Tuple[np.ndarray, Dict[str, Any]]:
    # pdfium statt PyMuPDF (AGPL), 14.09.2026 - gleiche Groesse, siehe planproof_geometry.pdf.
    from planproof_geometry.pdf import PdfDatei

    with PdfDatei(pdf_path) as pdf:
        w_pt, h_pt = pdf.seitengroesse(page_index)
        max_pt = max(w_pt, h_pt) or 1.0
        zoom = float(target_max_px / max_pt)
        zoom = max(1.5, min(zoom, 8.0))  # clamp
        img = pdf.rendern_grau(page_index, zoom)

    h_px, w_px = img.shape[:2]
    meta = {
        "page_index": page_index,
        "page_size_pt": {"w": w_pt, "h": h_pt},
        "render": {"zoom": zoom, "width_px": int(w_px), "height_px": int(h_px)},
        "px_per_pt": float(w_px / w_pt) if w_pt else float(zoom),
    }
    return img, meta


def _preprocess(gray: np.ndarray) -> np.ndarray:
    import cv2
    # leicht glätten + Kontrast verbessern (robuste Defaults)
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    return g

def _preprocess_for_ocr(gray: np.ndarray) -> np.ndarray:
    import cv2
    g = cv2.GaussianBlur(gray, (3, 3), 0)

    # Kontrast: leichte Normalisierung
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)

    # Adaptive Threshold (Pläne profitieren stark)
    th = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 25, 11
    )

    # Morphological close: reconnect broken characters in raster scans
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel_close)

    # Slight horizontal dilation to thicken thin strokes without merging lines
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    th = cv2.dilate(th, kernel_dilate, iterations=1)

    return th


def _detect_lines(
    gray: np.ndarray,
    *,
    min_len_px: int = 60,
    canny_low: int = 50,
    canny_high: int = 150,
) -> List[Tuple[float, float, float, float]]:
    import cv2

    g = _preprocess(gray)

    # 1) LSD (wenn verfügbar) -> oft besser für Pläne
    lines_out: List[Tuple[float, float, float, float]] = []
    try:
        lsd = cv2.createLineSegmentDetector(0)
        detected = lsd.detect(g)[0]
        if detected is not None:
            for seg in detected.reshape(-1, 4):
                x1, y1, x2, y2 = map(float, seg)
                if math.hypot(x2 - x1, y2 - y1) >= min_len_px:
                    lines_out.append((x1, y1, x2, y2))
    except Exception:
        pass

    # 2) Fallback HoughP
    if not lines_out:
        edges = cv2.Canny(g, canny_low, canny_high, L2gradient=True)
        # (N, 4) in OpenCV 4 wie 5 - siehe planproof_geometry.ai.cv_compat.
        h = hough_lines_p(edges, 1, np.pi / 180.0, threshold=80,
                          minLineLength=min_len_px, maxLineGap=12)
        for x1, y1, x2, y2 in h:
            lines_out.append((float(x1), float(y1), float(x2), float(y2)))

    # Dedup (grob): Endpunkte runden + normalisieren
    snap = 2.0
    seen = set()
    deduped: List[Tuple[float, float, float, float]] = []
    for x1, y1, x2, y2 in lines_out:
        ax, ay, bx, by = (round(x1 / snap) * snap, round(y1 / snap) * snap,
                          round(x2 / snap) * snap, round(y2 / snap) * snap)
        key = tuple(sorted([(ax, ay), (bx, by)]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((x1, y1, x2, y2))

    return deduped


def _ocr_text_lines_paddle(gray: np.ndarray, *, lang: str = "deu+eng") -> Tuple[List[Tuple[str, Tuple[int,int,int,int], float]], Dict[str, Any]]:
    """
    OCR using PaddleOCR - better for technical drawings.
    Returns: list of (text, bbox_px(x0,y0,x1,y1), confidence)
    """
    try:
        from planproof_geometry.ai.ocr_paddle import ocr_paddle
    except ImportError as e:
        logger.warning(f"PaddleOCR module not available: {e}")
        return [], {"ocr_available": False, "ocr_engine": "paddleocr", "error": str(e)}

    # PaddleOCR works better with preprocessed images
    img_for_ocr = _preprocess_for_ocr(gray)

    result = ocr_paddle(img_for_ocr, lang=lang, use_gpu=False, min_confidence=0.3)

    if not result.meta.get("available", False):
        return [], result.meta

    out: List[Tuple[str, Tuple[int,int,int,int], float]] = []
    for line in result.lines:
        # Confidence in PaddleOCR is 0-1, convert to 0-100 for consistency
        conf = line.confidence * 100.0
        out.append((line.text, line.bbox, conf))

    meta = {
        "ocr_available": True,
        "ocr_engine": "paddleocr",
        "lines": len(out),
        "avg_conf": result.meta.get("avg_confidence", 0.0),
        "lang": lang,
        **{k: v for k, v in result.meta.items() if k not in ("lines_detected", "avg_confidence")}
    }
    return out, meta


def _ocr_text_lines_tesseract(gray: np.ndarray, *, lang: str = "deu+eng") -> Tuple[List[Tuple[str, Tuple[int,int,int,int], float]], Dict[str, Any]]:
    """
    OCR using Tesseract - fallback option.
    Returns: list of (text, bbox_px(x0,y0,x1,y1), avg_conf)
    Gruppiert Words zu Text-Zeilen über tesseract line_num.
    """
    try:
        import pytesseract
    except Exception as e:
        return [], {"ocr_available": False, "ocr_engine": "tesseract", "error": f"pytesseract import failed: {e!r}"}

    # Version defensiv ermitteln
    try:
        tess_ver = str(pytesseract.get_tesseract_version())
    except Exception:
        tess_ver = None

    setup = configure_tesseract()
    if not setup.get("ok", False):
        return [], {"ocr_available": False, "ocr_engine": "tesseract", "error": setup}

    # Sprache automatisch auf vorhandene traineddata einschränken
    available = set((setup.get("available_langs") or ["eng"]))
    requested = [p.strip() for p in (lang or "eng").split("+") if p.strip()]
    chosen = [p for p in requested if p in available]
    if not chosen:
        chosen = ["eng"]
    lang = "+".join(chosen)

    cfg = "--oem 3 --psm 6 -c preserve_interword_spaces=1"
    img_for_ocr = _preprocess_for_ocr(gray)

    try:
        data = pytesseract.image_to_data(
            img_for_ocr,
            lang=lang,
            config=cfg,
            output_type=pytesseract.Output.DICT,
        )
    except Exception as e:
        return [], {
            "ocr_available": False,
            "ocr_engine": "tesseract",
            "error": f"image_to_data failed: {e!r}",
            "setup": setup,
            "tesseract_version": tess_ver,
            "cfg": cfg,
            "lang": lang,
        }

    n = len(data.get("text", []))
    groups: Dict[Tuple[int,int,int,int], Dict[str, Any]] = {}
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        conf_str = str(data.get("conf", ["-1"])[i])
        try:
            conf = float(conf_str)
        except Exception:
            conf = -1.0
        if not txt or conf < 0:
            continue

        key = (
            int(data.get("page_num", [1])[i]),
            int(data.get("block_num", [0])[i]),
            int(data.get("par_num", [0])[i]),
            int(data.get("line_num", [0])[i]),
        )

        x, y = int(data["left"][i]), int(data["top"][i])
        w, h = int(data["width"][i]), int(data["height"][i])

        g = groups.setdefault(key, {"words": [], "bbox": [10**9, 10**9, -10**9, -10**9], "confs": []})
        g["words"].append((x, txt))
        g["bbox"][0] = min(g["bbox"][0], x)
        g["bbox"][1] = min(g["bbox"][1], y)
        g["bbox"][2] = max(g["bbox"][2], x + w)
        g["bbox"][3] = max(g["bbox"][3], y + h)
        g["confs"].append(conf)

    out = []
    confs_all = []
    for _, g in groups.items():
        words = [w for _, w in sorted(g["words"], key=lambda t: t[0])]
        text_line = " ".join(words).strip()
        x0, y0, x1, y1 = map(int, g["bbox"])
        avg_conf = float(sum(g["confs"]) / max(1, len(g["confs"])))
        out.append((text_line, (x0, y0, x1, y1), avg_conf))
        confs_all.append(avg_conf)

    meta = {
        "ocr_available": True,
        "ocr_engine": "tesseract",
        "tesseract_version": tess_ver,
        "lines": len(out),
        "avg_conf": float(sum(confs_all) / max(1, len(confs_all))) if confs_all else 0.0,
        "lang": lang,
        "cfg": cfg,
        "tessdata_dir": str(setup.get("tessdata", "")),
    }
    return out, meta


def _ocr_text_lines(
    gray: np.ndarray,
    *,
    lang: str = "deu+eng",
    prefer_paddle: bool = True,
    require_paddle: bool = False,
) -> Tuple[List[Tuple[str, Tuple[int,int,int,int], float]], Dict[str, Any]]:
    """
    Run OCR on image with PaddleOCR (best for technical drawings + best.pt scaling).

    - prefer_paddle=True: Use PaddleOCR, fallback to Tesseract if unavailable.
    - require_paddle=True: Use only PaddleOCR; raise RuntimeError if not available
      (for best.pt mode where consistent scaling is required).

    Returns: list of (text, bbox_px(x0,y0,x1,y1), confidence 0-100)
    """
    lines, meta = _ocr_text_lines_paddle(gray, lang=lang)

    if meta.get("ocr_available", False) and not meta.get("error"):
        logger.info(f"PaddleOCR: detected {len(lines)} text lines, avg_conf={meta.get('avg_conf', 0):.1f}")
        return lines, meta

    # Paddle fehlgeschlagen → Tesseract-Fallback (wichtig für Azure bei wenig RAM)
    if prefer_paddle or require_paddle:
        err = meta.get("error", "PaddleOCR not available")
        logger.warning(
            f"PaddleOCR failed ({err}), falling back to Tesseract for scale inference. "
            "For Azure with limited RAM, Tesseract uses less memory."
        )

    lines, meta = _ocr_text_lines_tesseract(gray, lang=lang)
    logger.info(f"Tesseract: detected {len(lines)} text lines, avg_conf={meta.get('avg_conf', 0):.1f}")
    return lines, meta

def extract_raster_plan(
    path: str,
    *,
    max_pages: int = 1,
    assumed_image_dpi: int = 300,
    target_pdf_render_max_px: int = 3500,
    min_line_len_px: int = 60,
    canny_low: int = 50,
    canny_high: int = 150,
    ocr_lang: str = "deu+eng",
    return_image: bool = False,
    require_paddle_ocr: bool = False,
) -> Union[PlanExtraction, Tuple[PlanExtraction, np.ndarray]]:
    """
    Raster-PDF/Bild -> PlanExtraction in pdf_points Koordinaten.
    - PDF: Render->px, dann px->pt via px_per_pt aus Render meta
    - Image: px->pt via dpi (pt = px * 72 / dpi)
    """
    p = Path(path)
    ext = p.suffix.lower()

    # Load gray + mapping px->pt
    doc_meta: Dict[str, Any] = {
        "source_file": str(p),
        "pages": 1,
        "coordinate_system": "pdf_points",
    }

    if ext == ".pdf":
        from planproof_geometry.pdf import seitenzahl as _seitenzahl

        actual_page_count = min(_seitenzahl(p), max_pages)

        gray, meta = _render_pdf_page_gray(str(p), page_index=0, target_max_px=target_pdf_render_max_px)
        px_per_pt = float(meta["px_per_pt"])
        page_w_pt = float(meta["page_size_pt"]["w"])
        page_h_pt = float(meta["page_size_pt"]["h"])
        doc_meta["page_size"] = {"w": page_w_pt, "h": page_h_pt}
        doc_meta["pages"] = actual_page_count
        doc_meta["input_type"] = "raster_pdf"
        doc_meta["raster"] = {"pdf_render": meta}
    elif ext in IMAGE_EXTS:
        actual_page_count = 1
        gray = _cv2_gray_from_path(str(p))
        dpi = float(assumed_image_dpi)
        px_per_pt = float(dpi / 72.0)  # pt = px / (dpi/72)
        h_px, w_px = gray.shape[:2]
        page_w_pt = float(w_px / px_per_pt)
        page_h_pt = float(h_px / px_per_pt)
        doc_meta["page_size"] = {"w": page_w_pt, "h": page_h_pt}
        doc_meta["input_type"] = "image"
        doc_meta["raster"] = {"assumed_dpi": int(assumed_image_dpi), "size_px": {"w": int(w_px), "h": int(h_px)}}
    else:
        raise RuntimeError(f"Unsupported raster input: {path}")

    texts: List[TextSpan] = []
    segs: List[LineSeg] = []
    tid = 1
    gid = 1
    first_page_image = gray  # keep page-0 image for return_image

    for page_idx in range(actual_page_count):
        if page_idx == 0:
            page_gray = gray
            page_px_per_pt = px_per_pt
        else:
            page_gray, page_meta = _render_pdf_page_gray(str(p), page_index=page_idx, target_max_px=target_pdf_render_max_px)
            page_px_per_pt = float(page_meta["px_per_pt"])

        # Detect lines in px
        lines_px = _detect_lines(
            page_gray,
            min_len_px=min_line_len_px,
            canny_low=canny_low,
            canny_high=canny_high,
        )

        # OCR text lines (px bboxes)
        ocr_lines, ocr_meta_page = _ocr_text_lines(page_gray, lang=ocr_lang, require_paddle=require_paddle_ocr)
        if page_idx == 0:
            doc_meta.setdefault("raster", {})["ocr"] = ocr_meta_page

        for txt, (x0, y0, x1, y1), conf in ocr_lines:
            bb = (float(x0 / page_px_per_pt), float(y0 / page_px_per_pt),
                  float(x1 / page_px_per_pt), float(y1 / page_px_per_pt))
            texts.append(TextSpan(
                id=f"t{tid}", text=txt, bbox=bb,
                page=page_idx, rotation=0.0, conf=float(conf),
            ))
            tid += 1

        for x1, y1, x2, y2 in lines_px:
            a = (float(x1 / page_px_per_pt), float(y1 / page_px_per_pt))
            b = (float(x2 / page_px_per_pt), float(y2 / page_px_per_pt))
            segs.append(LineSeg(id=f"g{gid}", a=a, b=b, page=page_idx))
            gid += 1

    # Qualitätsheuristik (based on page 0)
    page0_lines = [(float(s.a[0] * px_per_pt), float(s.a[1] * px_per_pt),
                     float(s.b[0] * px_per_pt), float(s.b[1] * px_per_pt))
                    for s in segs if s.page == 0]
    total_len_px = sum(math.hypot(x2-x1, y2-y1) for x1,y1,x2,y2 in page0_lines) if page0_lines else 0.0
    first_ocr_meta = doc_meta.get("raster", {}).get("ocr", {})
    doc_meta["quality"] = _quality_label(
        n_lines=len(segs),
        total_len_px=total_len_px,
        ocr_avg_conf=float(first_ocr_meta.get("avg_conf", 0.0) or 0.0),
        rendered_max_px=int(max(first_page_image.shape[:2])),
    )

    extraction = PlanExtraction(doc=doc_meta, texts=texts, lines=segs)
    if return_image:
        import cv2
        bgr = cv2.cvtColor(first_page_image, cv2.COLOR_GRAY2BGR)
        return extraction, bgr
    return extraction


def _quality_label(*, n_lines: int, total_len_px: float, ocr_avg_conf: float, rendered_max_px: int) -> str:
    # sehr simple Ampel: genug Geometrie + halbwegs OCR + ausreichende Auflösung
    if rendered_max_px < 1400 or n_lines < 80 or total_len_px < 25000:
        return "low"
    if ocr_avg_conf < 45.0:
        return "medium"
    return "high"
