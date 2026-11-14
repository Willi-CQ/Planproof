"""EU-Egress-Transport fuer die Moebel-Klassifikation.

Der DSGVO-korrekte Pfad (Art. 44): NICHT direkt OpenRouter/US, sondern ueber
den internen API-Endpunkt ``/internal/pipeline/classify-furniture``, der durch
``Supervisor.complete()`` -> ``get_provider_adapter()`` -> ``AzureAIAdapter``
routet. In der ``eu_only``-Posture landet der Call automatisch EU-resident auf
Mistral-Large-3 (Azure Foundry) — genau der Pfad, den auch Arqon nutzt.

Spiegelt das Worker->API-Muster aus plan_llm_refine.py:
``PLANPROOF_API_URL`` + ``API_KEY`` (Header ``X-API-Key``).

Soft-fail: ohne konfigurierte API gibt :func:`resolve_default_classifier`
``None`` zurueck -> die Pipeline setzt generische Platzhalter statt zu brechen.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_ENDPOINT = "/internal/pipeline/classify-furniture"


class InternalApiFurnitureClassifier:
    """:class:`~.classify.FurnitureClassifier`, der ueber den EU-Chokepoint geht."""

    def __init__(self, api_url: str, api_key: str, *, timeout_s: float = 60.0,
                 max_tokens: int = 512) -> None:
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._max_tokens = max_tokens

    def classify(self, *, messages: List[Dict[str, Any]], request_id: str) -> Dict[str, Any]:
        """Messages durch den EU-Endpunkt schicken; OpenAI-Format zurueckgeben."""
        import httpx

        url = f"{self._api_url}{_ENDPOINT}"
        headers = {"X-API-Key": self._api_key, "Content-Type": "application/json"}
        payload = {
            "request_id": request_id,
            "messages": messages,
            "max_tokens": self._max_tokens,
        }
        with httpx.Client(timeout=self._timeout_s) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        # Endpoint liefert {content, model_used, usage, cost_usd} -> zurueck ins
        # OpenAI-Format (+ usage), damit classify_furniture content liest und die
        # Token-Usage aggregieren kann.
        return {
            "choices": [{"message": {"content": data.get("content") or ""}}],
            "model": data.get("model_used"),
            "usage": data.get("usage") or {},
            "cost_usd": data.get("cost_usd"),
        }


def resolve_default_classifier(
    config: Optional[Dict[str, Any]] = None,
) -> Optional[InternalApiFurnitureClassifier]:
    """EU-Classifier aus Config/Env bauen, oder None wenn nicht konfiguriert.

    Spiegelt die Gate-Logik von LLMRefineStep: ohne ``PLANPROOF_API_URL`` +
    ``API_KEY`` kein Egress -> None (Aufrufer faellt auf Platzhalter zurueck).
    """
    config = config or {}
    # Pilot-Notaus respektieren (wie llm_voting): externe KI deaktiviert?
    if os.environ.get("ARQON_LLM_EGRESS_DISABLED", "").strip().lower() in ("1", "true", "yes"):
        logger.info("furniture: ARQON_LLM_EGRESS_DISABLED set — no LLM classifier")
        return None

    api_url = config.get("api_url") or os.environ.get("PLANPROOF_API_URL")
    api_key = config.get("api_key") or os.environ.get("API_KEY")
    if not api_url or not api_key:
        logger.info("furniture: API not configured (PLANPROOF_API_URL / API_KEY) — no LLM classifier")
        return None
    return InternalApiFurnitureClassifier(api_url, api_key)
