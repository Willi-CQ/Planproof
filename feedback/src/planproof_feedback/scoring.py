"""Evaluator-only answer comparison; never imported by the model service."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation


def number(value):
    if isinstance(value, bool) or value is None:
        raise ValueError("Not a numeric answer")
    text = str(value).strip()
    # One decimal separator only; no guessing locale-specific thousands separators or units.
    if not re.fullmatch(r"[+-]?(?:\d+(?:[.,]\d+)?|\.\d+)(?:[eE][+-]?\d+)?", text):
        raise ValueError("Not a bare numeric answer")
    try:
        result = Decimal(text.replace(",", "."))
    except InvalidOperation:
        raise ValueError("Not a numeric answer") from None
    if not result.is_finite():
        raise ValueError("Nonfinite answer")
    return result


def text_key(value):
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def unit_key(value):
    return text_key(value or "").replace("²", "2").replace("³", "3").replace("^", "")


UNITS = {
    "m": ("length", Decimal(1)),
    "mm": ("length", Decimal("0.001")),
    "cm": ("length", Decimal("0.01")),
    "ft": ("length", Decimal("0.3048")),
    "m2": ("area", Decimal(1)),
    "sqm": ("area", Decimal(1)),
    "ft2": ("area", Decimal("0.09290304")),
    "sqft": ("area", Decimal("0.09290304")),
    "m3": ("volume", Decimal(1)),
    "ft3": ("volume", Decimal("0.028316846592")),
}


def score_answer(reference: dict, prediction: dict | None):
    kind = reference["type"]
    if kind not in ("count", "area", "length", "volume", "text"):
        return {"scorable": False, "correct": None, "reason": "unsupported_reference_type"}
    if not prediction or prediction.get("status") != "answered":
        return {"scorable": True, "correct": False, "reason": "abstained_or_failed"}
    if kind == "text":
        correct = text_key(reference["answer"]) == text_key(prediction["answer"])
        return {"scorable": True, "correct": correct, "reason": "normalized_exact_text_surrogate"}
    try:
        expected, actual = number(reference["answer"]), number(prediction["answer"])
    except ValueError:
        return {"scorable": True, "correct": False, "reason": "invalid_numeric_answer"}
    ref_unit, pred_unit = unit_key(reference.get("unit")), unit_key(prediction.get("unit"))
    if kind == "count":
        if (
            pred_unit not in ("count", "counts", "")
            or actual < 0
            or actual != actual.to_integral_value()
        ):
            return {"scorable": True, "correct": False, "reason": "invalid_count_or_unit"}
        return {
            "scorable": True,
            "correct": actual == expected,
            "reason": "exact_count",
            "absolute_error": float(abs(actual - expected)),
        }
    if (
        ref_unit not in UNITS
        or pred_unit not in UNITS
        or UNITS[ref_unit][0] != kind
        or UNITS[pred_unit][0] != kind
    ):
        return {"scorable": True, "correct": False, "reason": "missing_or_incompatible_unit"}
    actual = actual * UNITS[pred_unit][1] / UNITS[ref_unit][1]
    absolute_error = abs(actual - expected)
    relative_error = absolute_error / abs(expected) if expected else None
    return {
        "scorable": True,
        "correct": absolute_error == 0 if expected == 0 else relative_error <= Decimal("0.02"),
        "reason": "within_2_percent" if expected else "zero_reference_requires_exact_zero",
        "absolute_error": float(absolute_error),
        "relative_error": float(relative_error) if relative_error is not None else None,
    }


def summarize(records: list[dict]):
    def group(items):
        scorable = [x for x in items if x["score"]["scorable"]]
        correct = sum(x["score"]["correct"] is True for x in scorable)
        answered = sum(x.get("answer", {}).get("status") == "answered" for x in items)
        return {
            "cases": len(items),
            "scorable": len(scorable),
            "correct": correct,
            "accuracy": correct / len(scorable) if scorable else None,
            "answered": answered,
            "answer_coverage": answered / len(items) if items else None,
            "failures": sum(x["status"] != "ok" for x in items),
            "stage_failures": sum(
                call["status"] != "ok"
                for x in items
                for call in x.get("trace", {}).get("calls", [])
            ),
            "workflow_stops": dict(
                Counter(x.get("trace", {}).get("stop_reason", "error") for x in items)
            ),
            "latency_seconds": sum(x.get("latency_seconds", 0) for x in items),
            "tokens": {
                k: sum(x.get("usage", {}).get(k, 0) for x in items)
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            },
            "cases_missing_token_fields": {
                k: sum(k not in x.get("usage", {}) for x in items)
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            },
            "cases_missing_usage": sum(not x.get("usage") for x in items),
        }

    result = {"overall": group(records)}
    for field in ("source", "type", "mode"):
        groups = defaultdict(list)
        for record in records:
            groups[record[field]].append(record)
        result["by_" + field] = {k: group(v) for k, v in sorted(groups.items())}
    result["cost_usd"] = None
    result["cost_note"] = (
        "No verified endpoint price; token counts are reported without invented cost."
    )
    result["citation_note"] = (
        "Citation identities and bounds checked; semantic grounding needs human review."
    )
    return result
