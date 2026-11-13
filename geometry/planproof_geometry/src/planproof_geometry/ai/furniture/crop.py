"""Crop-Extraktion pro Moebel-Bbox.

Der entscheidende Trick gegen die dokumentierte LLM-Schwaeche (Vision-LLMs sind
schwach in Lokalisierung/Geometrie): Wir geben dem LLM **nicht den ganzen Plan**,
sondern einen **hochaufloesenden Ausschnitt** der bereits lokalisierten Bbox
(+ etwas Kontextrand). Damit muss das LLM nie selbst lokalisieren — es muss den
Crop nur noch **benennen**, und das ist seine Staerke.

Eingabe-Bild ist das vom Pipeline gerenderte Plan-Bild (grayscale oder RGB,
numpy). Bbox-Koordinaten sind in **Render-Pixeln** (plan-px * scale_xy).
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Optional, Tuple

# Kontextrand um die Bbox, damit das LLM benachbarte Hinweise sieht
# (Wandanschluss, Tuer daneben). Anteilig an der Bbox-Groesse + Mindestwert.
DEFAULT_MARGIN_FRAC = 0.25
DEFAULT_MIN_MARGIN_PX = 24
# Upscale kleiner Crops, damit duenne Symbol-Linien fuer das LLM lesbar bleiben.
MIN_CROP_LONG_EDGE_PX = 256
MAX_CROP_LONG_EDGE_PX = 768


@dataclass
class Crop:
    """Ein extrahierter Bildausschnitt einer Moebel-Bbox."""

    png_bytes: bytes
    width: int
    height: int

    def to_base64(self) -> str:
        return base64.b64encode(self.png_bytes).decode("ascii")

    def to_data_url(self) -> str:
        return "data:image/png;base64," + self.to_base64()


def _bbox_corners_px(
    bbox_px: Tuple[float, float, float, float],
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox_px
    return (
        int(min(x0, x1)),
        int(min(y0, y1)),
        int(max(x0, x1)),
        int(max(y0, y1)),
    )


def extract_crop(
    image,
    bbox_px: Tuple[float, float, float, float],
    *,
    margin_frac: float = DEFAULT_MARGIN_FRAC,
    min_margin_px: int = DEFAULT_MIN_MARGIN_PX,
) -> Optional[Crop]:
    """PNG-Crop um eine Bbox extrahieren (mit Kontextrand, ggf. hochskaliert).

    Gibt None zurueck, wenn cv2 fehlt, das Bild leer ist oder der Crop
    degeneriert. Soft-fail-freundlich: der Aufrufer behandelt None als
    "kein Crop -> Klassifikation ohne Bild moeglich (nur Raum-Prior)".
    """
    try:
        import cv2
    except ImportError:
        return None
    if image is None or getattr(image, "size", 0) == 0:
        return None

    h, w = image.shape[:2]
    bx0, by0, bx1, by1 = _bbox_corners_px(bbox_px)
    bw = max(1, bx1 - bx0)
    bh = max(1, by1 - by0)
    margin = max(min_margin_px, int(round(margin_frac * max(bw, bh))))

    x0 = max(0, bx0 - margin)
    y0 = max(0, by0 - margin)
    x1 = min(w, bx1 + margin)
    y1 = min(h, by1 + margin)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None

    patch = image[y0:y1, x0:x1].copy()

    # Upscale/Downscale auf eine LLM-freundliche Aufloesung (lange Kante).
    long_edge = max(patch.shape[0], patch.shape[1])
    target = min(MAX_CROP_LONG_EDGE_PX, max(MIN_CROP_LONG_EDGE_PX, long_edge))
    if long_edge > 0 and target != long_edge:
        scale = target / float(long_edge)
        new_w = max(1, int(round(patch.shape[1] * scale)))
        new_h = max(1, int(round(patch.shape[0] * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        patch = cv2.resize(patch, (new_w, new_h), interpolation=interp)

    ok, buffer = cv2.imencode(".png", patch)
    if not ok:
        return None
    return Crop(
        png_bytes=buffer.tobytes(),
        width=int(patch.shape[1]),
        height=int(patch.shape[0]),
    )
