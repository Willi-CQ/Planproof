"""Keep unqualified semantic changes explicitly opt-in at the deployment boundary."""

from copy import deepcopy
import os


QUALIFIED = "qualified-v6"
CANDIDATE = "scope-count-candidate"


def active_profile():
    profile = os.getenv("PLANPROOF_AGENT_PROFILE", QUALIFIED).strip()
    if profile not in (QUALIFIED, CANDIDATE):
        raise ValueError("Unknown PLANPROOF_AGENT_PROFILE")
    return profile


def candidate_enabled():
    return active_profile() == CANDIDATE


def tool_schema(model, candidate):
    """Hide candidate-only fields and definitions from qualified model requests."""
    schema = deepcopy(model.model_json_schema())
    if candidate:
        return schema
    schema.get("properties", {}).pop("task_scope", None)
    definitions = schema.get("$defs", {})
    definitions.get("CountBasis", {}).get("properties", {}).pop("exclusions", None)
    for name in ("TaskScope", "ScopeExclusion", "ScopeCitation"):
        definitions.pop(name, None)
    return schema
