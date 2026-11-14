"""Explicit tool boundary for a later agent; the host application registers allowed regions."""

from __future__ import annotations

from pathlib import Path

from .regions import load_region


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "region_id": {"type": "string", "description": "Host-registered clean crop ID"},
                    "question": {"type": "string"},
                    argument: {"type": "string", "description": "Proposed answer and rationale"},
                },
                "required": ["region_id", "question", argument],
                "additionalProperties": False,
            },
        },
    }
    for name, argument, description in (
        (
            "nemotron_visual_feedback",
            "candidate",
            "Review an answer against a registered drawing crop; advisory feedback.",
        ),
        (
            "nemotron_self_feedback",
            "draft",
            "Critique the agent's draft against a registered crop; correlated feedback.",
        ),
    )
]


class FeedbackTools:
    def __init__(self, service, image_config: dict, regions: dict[str, Path]):
        self.service = service
        self.image_config = image_config
        self.regions = dict(regions)

    def call(self, name: str, arguments: dict):
        if name not in {tool["function"]["name"] for tool in TOOL_DEFINITIONS}:
            raise ValueError("Unknown feedback tool")
        draft_key = "candidate" if name == "nemotron_visual_feedback" else "draft"
        if not isinstance(arguments, dict) or set(arguments) != {
            "region_id",
            "question",
            draft_key,
        }:
            raise ValueError("Feedback tool arguments do not match the schema")
        if not all(isinstance(value, str) and value.strip() for value in arguments.values()):
            raise ValueError("Feedback tool arguments must be nonempty strings")
        region = self.regions.get(arguments["region_id"])
        if region is None:
            raise ValueError("The calling application has not registered this region ID")
        images, evidence = load_region(region, self.image_config)
        if name == "nemotron_visual_feedback":
            return self.service.review(
                arguments["question"], images, candidate=arguments["candidate"], evidence=evidence
            )
        return self.service.critique_draft(
            arguments["question"], images, draft=arguments["draft"], evidence=evidence
        )
