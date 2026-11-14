"""Optional readable answer metadata; never changes answer values or evidence."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class BriefingStatement(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    section: Literal["answer", "basis", "limitation"]
    text: str = Field(min_length=1, max_length=900)
    citations: list[int] = Field(default_factory=list, max_length=100)


class CitationDetail(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    citation: int = Field(ge=1, le=100)
    kind: Literal["text", "detection", "measurement"]
    label: str = Field(min_length=1, max_length=100)


def attach_presentation(result, original_citations, raw):
    """Map model's 1-based references after host measurement citations are added.

    Invalid optional prose is omitted rather than rejecting a valid core answer.
    Bounds/identity validation still applies to the original final citations.
    """
    result["briefing"], result["citation_details"] = [], []
    mapping = {}
    for index, citation in enumerate(original_citations, 1):
        if citation in result["citations"]:
            mapping[index] = result["citations"].index(citation) + 1
    statements = raw.get("briefing")
    if isinstance(statements, list):
        for value in statements[:8]:
            try:
                statement = BriefingStatement.model_validate(value)
                if not statement.text.strip():
                    continue
                if any(ref not in mapping for ref in statement.citations):
                    continue
                if statement.section != "limitation" and not statement.citations:
                    continue
                result["briefing"].append(
                    {
                        **statement.model_dump(),
                        "citations": list(
                            dict.fromkeys(mapping[i] for i in statement.citations)
                        ),
                    }
                )
            except (ValidationError, TypeError):
                continue
    details = raw.get("citation_details")
    if isinstance(details, list):
        seen = set()
        for value in details[:100]:
            try:
                detail = CitationDetail.model_validate(value)
                if detail.citation not in mapping or detail.citation in seen:
                    continue
                seen.add(detail.citation)
                result["citation_details"].append(
                    {**detail.model_dump(), "citation": mapping[detail.citation]}
                )
            except (ValidationError, TypeError):
                continue
    return result
