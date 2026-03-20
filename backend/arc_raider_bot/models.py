"""Pydantic models shared across the application."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Agent output models
# ---------------------------------------------------------------------------

class ArcRaidersResponse(BaseModel):
    """Structured response returned by the ARC-RAIDERS wikibot agent."""

    answer: str = Field(
        description=(
            "A concise, wiki-style answer to the user's question about ARC Raiders. "
            "Must contain ONLY the explanatory text — NEVER include URLs, links, or "
            "references here. All URLs go in the 'sources' field instead."
        ),
    )
    sources: list[str] = Field(
        default_factory=list,
        description=(
            "A list of URLs from the search results that were used to compose the answer. "
            "Each entry must be a plain URL string (e.g. 'https://example.com/page'). "
            "Do NOT put URLs in the 'answer' field — put them here."
        ),
    )


class ValidationVerdict(BaseModel):
    """Result of the validator agent's review."""

    is_valid: bool = Field(
        description="True if the response passes all checks, False if it needs correction.",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="List of specific issues found. Empty when is_valid is True.",
    )
    corrected_answer: str | None = Field(
        default=None,
        description=(
            "If is_valid is False, a corrected version of the answer text "
            "(still no URLs). None when is_valid is True."
        ),
    )
    corrected_sources: list[str] | None = Field(
        default=None,
        description="If sources were wrong, a corrected list. None when sources are fine.",
    )


# ---------------------------------------------------------------------------
# RAG cache model
# ---------------------------------------------------------------------------

class CachedQA(BaseModel):
    """A previously answered question retrieved from the knowledge cache."""

    question: str
    answer: str
    sources: list[str] = Field(default_factory=list)
    similarity: float = 0.0


# ---------------------------------------------------------------------------
# API request / response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    """Incoming question payload."""

    question: str
    session_id: str | None = Field(
        default=None,
        description="Session ID for conversation continuity",
    )


class AskResponse(BaseModel):
    """Outgoing answer payload."""

    answer: str
    sources: list[str] = Field(default_factory=list)
    session_id: str
