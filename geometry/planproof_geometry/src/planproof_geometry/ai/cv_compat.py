"""OpenCV-Kompatibilitaet fuer Rueckgabeformen, die sich zwischen Hauptversionen aendern.

Befund 08.09.2026: ``cv2.HoughLinesP`` liefert in OpenCV 4.x ein Array der
Form ``(N, 1, 4)``, in OpenCV 5.0 (PyPI seit 02.07.2026) dagegen ``(N, 4)``.
Aufrufer, die ``x1, y1, x2, y2 = line[0]`` entpacken, sterben unter 5.0 mit
``TypeError: cannot unpack non-iterable numpy.int32 object``. Genau so ist die
Plan-zu-BIM-Pipeline im Worker-Image ``prod-20260708`` bei jeder
Tuerplatzierung gestorben: das Dockerfile installiert mit ``pip`` ohne Lock,
und sechs Tage vor dem Build war OpenCV 5.0.0.93 erschienen. Lokal (OpenCV
4.10) lief derselbe Plan durch.

Regel: kein direkter ``cv2.HoughLinesP``-Aufruf im Paket - immer
:func:`hough_lines_p`, das in beiden Welten ``(N, 4)`` liefert (leer: ``(0, 4)``).
Ein Test (``test_cv_compat_hough_2026_09_08.py``) wacht ueber die Regel.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def normalize_hough_lines(lines: Any | None) -> np.ndarray:
    """Bringt eine ``HoughLinesP``-Rueckgabe auf die Form ``(N, 4)``.

    * ``None`` (kein Treffer) und leere Arrays werden zu ``(0, 4)`` int32.
    * ``(N, 1, 4)`` (OpenCV 4) und ``(N, 4)`` (OpenCV 5) werden zu ``(N, 4)``.
    * Alles, dessen letzte Achse nicht 4 Werte hat, ist ein Programmfehler und
      wird als ``ValueError`` gemeldet statt still umgedeutet.
    """
    if lines is None:
        return np.zeros((0, 4), dtype=np.int32)
    arr = np.asarray(lines)
    if arr.size == 0:
        dtype = arr.dtype if arr.dtype.kind in "iuf" else np.int32
        return np.zeros((0, 4), dtype=dtype)
    if arr.ndim == 0 or arr.shape[-1] != 4:
        raise ValueError(
            f"HoughLinesP-Rueckgabe hat Form {arr.shape}, erwartet (N, 4) oder (N, 1, 4)"
        )
    return arr.reshape(-1, 4)


def hough_lines_p(
    image: np.ndarray,
    rho: float,
    theta: float,
    threshold: int,
    *,
    minLineLength: float = 0,
    maxLineGap: float = 0,
) -> np.ndarray:
    """``cv2.HoughLinesP`` mit versionsunabhaengiger Rueckgabe ``(N, 4)``.

    Signatur wie das Original (``rho``, ``theta``, ``threshold`` positional
    oder benannt, ``minLineLength``/``maxLineGap`` benannt). Kein Treffer
    liefert ein leeres ``(0, 4)``-Array, nie ``None`` - Aufrufer iterieren
    direkt: ``for x1, y1, x2, y2 in hough_lines_p(...)``.
    """
    import cv2

    lines = cv2.HoughLinesP(
        image, rho, theta, threshold,
        minLineLength=minLineLength, maxLineGap=maxLineGap,
    )
    return normalize_hough_lines(lines)


__all__ = ["hough_lines_p", "normalize_hough_lines"]
