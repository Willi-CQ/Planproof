"""EU-Vision-LLM-Klassifikation der Moebel-Bboxes.

Was das LLM macht (und NUR das): es **benennt**. Pro Bbox bekommt es
  - einen hochaufloesenden Crop (siehe crop.py),
  - den Raum-Kontext ("dieser Crop liegt im Bad"),
  - den geschlossenen Katalog (catalog.py)
und gibt strukturiertes JSON zurueck: welche Katalog-Items, in welcher
Reihenfolge, plus ein grobes Facing. Geometrie/Position/Skalierung entstehen
danach deterministisch in placement.py.

Transport-agnostisch: dieses Modul kennt keinen HTTP-Client. Es baut die
Messages und parst die Antwort; den eigentlichen Call macht ein injizierter
:class:`FurnitureClassifier`. So bleibt der Kern testbar (Fake-Classifier) und
der Egress-Pfad austauschbar — produktiv der **EU-Chokepoint** (Mistral Large 3
auf Azure Foundry, via internem API-Endpunkt), nie direkt OpenRouter/US.

Soft-fail: ohne Classifier (oder bei jedem Fehler) faellt jede Bbox auf das
generische Platzhalter-Item zurueck — die Pipeline produziert weiter Geometrie.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from .catalog import GENERIC_ID, UNKNOWN_ID, catalog_enum, catalog_prompt_table
from .crop import Crop
from .rooms import RoomRegion

logger = logging.getLogger(__name__)


# ============================================================================
# Datenklassen
# ============================================================================

@dataclass
class FurnitureContent:
    """Ein erkanntes Einzel-Item innerhalb einer Bbox (geordnet)."""

    catalog_id: str          # immer eine gueltige Katalog-ID (oder GENERIC_ID)
    order: int               # 0-basiert, Reihenfolge entlang der Zeile
    confidence: float        # 0..1


@dataclass
class FurnitureClassification:
    """LLM-Ergebnis fuer eine Moebel-Bbox."""

    bbox_id: str
    zone_type: str                       # "single_object" | "kitchen_run" | "bathroom" | ...
    contents: List[FurnitureContent]     # geordnete Liste (>= 1)
    facing_hint: str                     # "" | "toward_room" | "against_wall" | "north"...
    room_type: str                       # was wir dem LLM als Raum gegeben haben
    low_confidence: bool
    source: str                          # "llm" | "fallback"
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_run(self) -> bool:
        return len(self.contents) > 1


class FurnitureClassifier(Protocol):
    """Transport fuer einen Vision-Klassifikations-Call.

    Implementierungen: der EU-Internal-API-Client (produktiv) oder ein
    Fake/OpenRouter-Client (Test/Dev). Vertrag = OpenAI-Chat-Completion-Format
    (``{"choices": [{"message": {"content": "..."}}]}``).
    """

    def classify(self, *, messages: List[Dict[str, Any]], request_id: str) -> Dict[str, Any]:
        ...


# ============================================================================
# Prompt + Schema
# ============================================================================

VALID_CATALOG_IDS = set(catalog_enum())

SYSTEM_PROMPT = """\
Du bist Experte fuer Architektur-Grundrisse und Innenausstattung. Du
analysierst einen kleinen Bildausschnitt aus einem Bauplan, auf dem ein
moebliertes Symbol markiert ist (Moebel, Sanitaer- oder Kuechenobjekt).

Deine EINZIGE Aufgabe: benenne, WAS im Ausschnitt steht — anhand eines
GESCHLOSSENEN Katalogs. Du bestimmst KEINE Koordinaten, Masse oder Winkel;
das macht ein nachgelagerter Algorithmus.

REGELN:
- Antworte NUR mit JSON im vorgegebenen Schema, ohne Markdown-Codeblock.
- Jede catalog_id MUSS EXAKT aus der Katalog-Liste stammen. Erfinde nichts.
  Bist du unsicher, nutze "{generic}" (generisches Moebel) oder "{unknown}".
- Ein Ausschnitt kann MEHRERE Objekte enthalten (z.B. eine Kuechenzeile:
  Herd, Spuele, Geschirrspueler nebeneinander). Liste sie dann in
  "contents" in der raeumlichen Reihenfolge (links->rechts bzw. oben->unten,
  wie sie entlang der laengeren Bbox-Kante stehen).
- Ein einzelnes Objekt -> "contents" mit genau einem Eintrag,
  zone_type "single_object".
- RAUM-KONTEXT ist VERBINDLICH (separat ermittelt, unten angegeben): waehle nur
  Items, die in diesem Raum plausibel sind. In "Kueche"/"Pantry" KEINE
  Sanitaerobjekte ausser Spuele; in "Schlafzimmer" KEINE Kueche/Sanitaer; in
  "Wohnzimmer"/"Esszimmer" Sofa/Tisch/Stuehle/TV; "Bad" siehe Sanitaer-Regel.
- SANITAER NICHT pauschal als Badewanne raten — unterscheide nach Form/Groesse:
  * WC / Wand-WC: kleines Becken, meist mit Spuelkasten-Rechteck dahinter (~0.4 m).
  * Waschbecken: kleines rundes/ovales Becken an der Wand (~0.5-0.6 m).
  * Dusche: QUADRATISCH ~0.9x0.9 m, oft in einer Raumecke (Ablauf/Diagonale).
  * Badewanne: NUR bei klar LAENGLICHEM Rechteck/Oval ~1.7x0.75 m.
  * Bidet/Urinal: kleines Becken neben dem WC.
  Im Zweifel das KLEINERE/haeufigere Objekt (WC/Waschbecken) statt Badewanne.
- ERFINDE KEINE Objekte: liste nur, was im Ausschnitt klar erkennbar ist. Lieber
  "{generic}" + low_confidence=true als ein falsches spezifisches Objekt. Eine
  einzelne Bbox ist meist EIN Objekt; mehrere nur bei klar getrennten Symbolen.
- facing_hint: grobe Ausrichtung, falls erkennbar ("against_wall" wenn das
  Objekt an einer Wand sitzt, "toward_room" sonst). Keine Winkel.
- low_confidence=true, wenn der Ausschnitt zu mehrdeutig fuer eine sichere
  Zuordnung ist.

KATALOG (id | label | nennmass WxTxH):
{catalog}
"""


def _response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "zone_type": {"type": "string"},
            "facing_hint": {"type": "string"},
            "low_confidence": {"type": "boolean"},
            "contents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "catalog_id": {"type": "string", "enum": catalog_enum()},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["catalog_id"],
                },
            },
        },
        "required": ["contents"],
    }


def build_system_prompt() -> str:
    return SYSTEM_PROMPT.format(
        generic=GENERIC_ID,
        unknown=UNKNOWN_ID,
        catalog=catalog_prompt_table(),
    )


def build_classification_messages(
    *,
    crop: Optional[Crop],
    room: Optional[RoomRegion],
    bbox_long_edge_m: float,
) -> List[Dict[str, Any]]:
    """OpenAI-kompatible Messages fuer einen Bbox-Klassifikations-Call bauen.

    Bild (Crop) + Text (Raum-Kontext, Bbox-Groesse, Schema). Ohne Crop wird
    rein textuell klassifiziert (nur Raum-Prior) — Soft-fall fuer fehlendes cv2.
    """
    room_desc = room.describe() if room else "unbekannt (kein Raum-Label in der Naehe)"
    user_text = (
        f"ERKANNTER RAUM (verbindlich, separat per OCR/Label ermittelt): {room_desc}\n"
        f"Waehle nur Items, die in DIESEM Raum plausibel sind.\n"
        f"Bbox-Laenge (laengere Kante): {bbox_long_edge_m:.2f} m "
        f"(Hinweis: eine lange Bbox in Kueche/Bad ist meist eine Zeile mehrerer Objekte; "
        f"eine kurze/quadratische Bbox ist meist EIN Objekt).\n\n"
        f"Antworte im JSON-Schema:\n{json.dumps(_response_schema(), ensure_ascii=False)}"
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt()},
    ]
    if crop is not None:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": crop.to_data_url(), "detail": "high"}},
                {"type": "text", "text": user_text},
            ],
        })
    else:
        messages.append({"role": "user", "content": user_text})
    return messages


# ============================================================================
# Parsing + Validierung
# ============================================================================

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    md = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if md:
        text = md.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def parse_classification_response(
    response_text: str,
    *,
    bbox_id: str,
    room_type: str = "",
) -> Optional[FurnitureClassification]:
    """LLM-Antworttext zu :class:`FurnitureClassification` parsen+validieren.

    None nur bei voellig unparsbarer Antwort (Aufrufer macht dann Fallback).
    Ungueltige/halluzinierte catalog_ids werden auf das generische Item
    gemappt statt verworfen — so geht kein Objekt verloren.
    """
    data = _extract_json(response_text)
    if data is None:
        return None

    raw_contents = data.get("contents")
    if not isinstance(raw_contents, list):
        raw_contents = []

    contents: List[FurnitureContent] = []
    for i, entry in enumerate(raw_contents):
        if not isinstance(entry, dict):
            continue
        cid = str(entry.get("catalog_id", "")).strip().lower()
        if cid in ("", UNKNOWN_ID) or cid not in VALID_CATALOG_IDS:
            cid = GENERIC_ID
        try:
            conf = float(entry.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        contents.append(FurnitureContent(
            catalog_id=cid,
            order=len(contents),
            confidence=max(0.0, min(1.0, conf)),
        ))

    if not contents:
        contents = [FurnitureContent(GENERIC_ID, 0, 0.3)]

    zone_type = str(data.get("zone_type", "")).strip() or (
        "single_object" if len(contents) == 1 else "run"
    )
    facing = str(data.get("facing_hint", "")).strip().lower()
    low_conf = bool(data.get("low_confidence", False))
    # Auch ohne explizites Flag: alle Items generisch -> niedrige Confidence.
    if all(c.catalog_id == GENERIC_ID for c in contents):
        low_conf = True

    return FurnitureClassification(
        bbox_id=bbox_id,
        zone_type=zone_type,
        contents=contents,
        facing_hint=facing,
        room_type=room_type,
        low_confidence=low_conf,
        source="llm",
        raw=data,
    )


def fallback_classification(
    bbox_id: str,
    *,
    room_type: str = "",
) -> FurnitureClassification:
    """Generischer Platzhalter, wenn kein LLM/Crop verfuegbar ist."""
    return FurnitureClassification(
        bbox_id=bbox_id,
        zone_type="single_object",
        contents=[FurnitureContent(GENERIC_ID, 0, 0.0)],
        facing_hint="",
        room_type=room_type,
        low_confidence=True,
        source="fallback",
    )


# ============================================================================
# Orchestrierung
# ============================================================================

@dataclass
class ClassifyInput:
    """Eine zu klassifizierende Bbox mit Crop + Raum + Geometrie-Hinweis."""

    bbox_id: str
    crop: Optional[Crop]
    room: Optional[RoomRegion]
    bbox_long_edge_m: float


def _cache_key(inp: ClassifyInput) -> str:
    h = hashlib.sha1()
    if inp.crop is not None:
        h.update(inp.crop.png_bytes)
    h.update((inp.room.describe() if inp.room else "").encode("utf-8"))
    h.update(f"{inp.bbox_long_edge_m:.2f}".encode("ascii"))
    return h.hexdigest()


def classify_furniture(
    inputs: Sequence[ClassifyInput],
    *,
    classifier: Optional[FurnitureClassifier],
    request_id: str = "furniture",
    cache: Optional[Dict[str, FurnitureClassification]] = None,
    usage_sink: Optional[Dict[str, Any]] = None,
) -> Dict[str, FurnitureClassification]:
    """Alle Bboxes klassifizieren. Soft-fail pro Bbox -> Fallback.

    ``cache`` (optional, keyed by Crop+Raum-Hash) macht Re-Runs deterministisch
    und spart Tokens. Ohne ``classifier`` wird ueberall der Fallback genutzt.
    ``usage_sink`` (optional): akkumuliert die echte Token-Usage des LLM
    (prompt_tokens/completion_tokens/calls/model/cost_usd) fuer Kostentransparenz.
    """
    results: Dict[str, FurnitureClassification] = {}

    # Pacing zwischen LLM-Calls (Provider-Rate-Limit, z.B. Azure-EU 20/min). Per
    # Env P2B_FURNITURE_CLASSIFY_DELAY_S sekunden-Delay vor jedem echten Call
    # (ausser dem ersten). 0 = aus (Default). Zusammen mit dem 429-Retry im
    # Adapter haelt das den Moebel-Batch sauber unter dem Limit.
    try:
        pacing_s = float(os.environ.get("P2B_FURNITURE_CLASSIFY_DELAY_S", "0") or 0)
    except ValueError:
        pacing_s = 0.0
    made_call = False

    for inp in inputs:
        room_type = inp.room.room_type if inp.room else ""

        if classifier is None:
            results[inp.bbox_id] = fallback_classification(inp.bbox_id, room_type=room_type)
            continue

        key = _cache_key(inp)
        if cache is not None and key in cache:
            cached = cache[key]
            results[inp.bbox_id] = FurnitureClassification(
                bbox_id=inp.bbox_id,
                zone_type=cached.zone_type,
                contents=[FurnitureContent(c.catalog_id, c.order, c.confidence)
                          for c in cached.contents],
                facing_hint=cached.facing_hint,
                room_type=room_type,
                low_confidence=cached.low_confidence,
                source=cached.source,
                raw=cached.raw,
            )
            continue

        if pacing_s > 0 and made_call:
            time.sleep(pacing_s)
        made_call = True

        try:
            messages = build_classification_messages(
                crop=inp.crop, room=inp.room,
                bbox_long_edge_m=inp.bbox_long_edge_m,
            )
            response = classifier.classify(
                messages=messages,
                request_id=f"{request_id}-{inp.bbox_id}",
            )
            if usage_sink is not None:
                u = response.get("usage") or {}
                usage_sink["prompt_tokens"] = usage_sink.get("prompt_tokens", 0) + int(u.get("prompt_tokens") or 0)
                usage_sink["completion_tokens"] = usage_sink.get("completion_tokens", 0) + int(u.get("completion_tokens") or 0)
                usage_sink["calls"] = usage_sink.get("calls", 0) + 1
                if response.get("model"):
                    usage_sink["model"] = response.get("model")
                if response.get("cost_usd") is not None:
                    usage_sink["cost_usd"] = round(usage_sink.get("cost_usd", 0.0) + float(response.get("cost_usd") or 0), 6)
            text = (response.get("choices") or [{}])[0].get("message", {}).get("content", "")
            parsed = parse_classification_response(text, bbox_id=inp.bbox_id, room_type=room_type)
        except Exception as exc:  # noqa: BLE001 — darf die Pipeline nie brechen
            logger.warning("furniture classify failed for %s: %s", inp.bbox_id, exc)
            parsed = None

        if parsed is None:
            parsed = fallback_classification(inp.bbox_id, room_type=room_type)

        results[inp.bbox_id] = parsed
        if cache is not None:
            cache[key] = parsed

    return results
