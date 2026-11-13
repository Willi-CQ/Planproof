"""Pixel-Budget fuer das Rastern untrusted PDFs im Core (Issue #47).

Die vier produktiven Renderer der Plan-zu-BIM-Pipeline
(``diagnose_door_placement``, ``merge_doors_and_windows_bim``,
``add_stairs_to_bim``, ``plan_to_bim_v3_pipeline``) rasterten mit einem FESTEN
Faktor 4,0 und lasen ``page.rect`` nie. Die Seitengeometrie ist aber ein Wert
aus der hochgeladenen Datei: eine ~2 KB grosse Seite von 10800 x 10800 pt ergibt
bei Faktor 4 ein Bild von 43200 x 43200 Pixeln. Dazu haelt der ``_render_cache``
das Ergebnis fest und gibt jedem der drei Aufrufer eine ``.copy()`` — die Spitze
liegt also bei mehreren Vollbildern gleichzeitig.

Der Operator-Knopf ``target_pdf_render_max_px`` (gesetzt ueber
``PLANPROOF_LOW_RESOURCES``) existierte bereits, wurde von ``perfect_bim`` aber
nirgends gelesen — auf dem Produktions-Default-Pfad war er wirkungslos.

Das Muster stammt aus ``plan_extract_raster.py``, das den Zoom seit jeher aus
``page.rect`` ableitet und clampt. Es fehlte nur auf den neueren Pfaden.

Zwilling in der API: ``planproof_api/services/rag/pdf_budget.py`` traegt
dieselbe Rechnung fuer OCR und Vision. Bewusst zwei Dateien —
``planproof_geometry`` ist ein eigenstaendiges Paket ohne Abhaengigkeit auf die
API. Beide werden auf dieselben Zahlen getestet.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except (TypeError, ValueError):
        return default


#: Obergrenze der Bildflaeche in Pixeln. Bewusst so gewaehlt, dass ein A0-Plan
#: (2384 x 3370 pt) beim Produktions-Faktor 4,0 GENAU SO gerendert wird wie
#: bisher — 9536 x 13480 = 128,5 MPx. Ein Deckel, der reale Plaene groeber
#: macht, waere ein Qualitaetsverlust ohne Sicherheitsgewinn; die Angriffsseite
#: will das Vierzehnfache (10800 x 10800 pt bei Faktor 4 = 1866 MPx = 1,9 GB).
#: Grosszuegiger als die API (40 MPx), weil der Worker 8 GiB hat und hier
#: Planqualitaet das Produkt ist — aber der Render-Cache haelt Vollbilder und
#: jeder der drei Aufrufer kopiert, also bleibt es bei einer harten Grenze.
MAX_RENDER_PIXELS: int = _env_int("PLANPROOF_MAX_RENDER_PIXELS", 200_000_000)

#: Unter diesem Zoom ist ein Bauplan nicht mehr auswertbar. Passt die Seite
#: selbst hier nicht, ist die Datei fuer diese Pipeline ungeeignet — dann ist
#: ein klarer Fehler besser als ein OOM-Kill des Subprozesses, der nur
#: "Pipeline hat keine Ausgabedateien erzeugt" hinterlaesst.
MIN_RENDER_ZOOM: float = float(os.getenv("PLANPROOF_MIN_RENDER_ZOOM", "") or 0.5)


class PageTooLargeError(ValueError):
    """Die Seite passt selbst beim Mindest-Zoom nicht in das Pixel-Budget."""


def fit_zoom(width_pt: float, height_pt: float, wunsch_zoom: float) -> float:
    """Der groesste Zoom <= ``wunsch_zoom``, der ins Pixel-Budget passt.

    Wirft ``PageTooLargeError``, wenn selbst ``MIN_RENDER_ZOOM`` nicht reicht.
    Der Aufrufer soll den Job dann mit klarer Meldung beenden — im Gegensatz zum
    OOM-Kill ist das ein Ergebnis, das jemand lesen kann.
    """
    try:
        w = float(width_pt)
        h = float(height_pt)
    except (TypeError, ValueError):
        return wunsch_zoom
    if not (w > 0 and h > 0):
        return wunsch_zoom

    # pixel = (w * zoom) * (h * zoom)  ->  zoom = sqrt(budget / (w * h))
    flaeche = w * h
    passend = (MAX_RENDER_PIXELS / flaeche) ** 0.5
    if passend >= wunsch_zoom:
        return wunsch_zoom
    if passend < MIN_RENDER_ZOOM:
        raise PageTooLargeError(
            f"PDF-Seite {w:.0f} x {h:.0f} pt ist zu gross zum Rastern: selbst bei Zoom "
            f"{MIN_RENDER_ZOOM} entstuenden mehr als {MAX_RENDER_PIXELS} Pixel."
        )
    logger.info(
        "PDF-Rasterung gedrosselt: %.0f x %.0f pt -> Zoom %.3f statt %.3f (Budget %d px)",
        w,
        h,
        passend,
        wunsch_zoom,
        MAX_RENDER_PIXELS,
    )
    return passend


__all__ = ["MAX_RENDER_PIXELS", "MIN_RENDER_ZOOM", "PageTooLargeError", "fit_zoom"]
