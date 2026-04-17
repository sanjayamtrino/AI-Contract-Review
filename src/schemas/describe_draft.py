"""Schemas for the Describe & Draft Agent."""

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class DescribeDraftRequest(BaseModel):
    """Request body — only prompt; session_id arrives via X-Session-ID header."""

    prompt: str = Field(
        description="Free-text drafting request (e.g. 'draft an NDA' or 'draft a liquidity cap clause')",
        min_length=1,
        max_length=2000,
    )


# --- LLM internal schemas (not exposed in API response) ---


class IntentClassification(BaseModel):
    """Output of the intent-classifier LLM call."""

    mode: Literal["list_of_clauses", "single_clause", "clarification"]
    detected_agreement_type: Optional[str] = None  # e.g. "NDA", "SaaS Agreement"
    clarification_question: Optional[str] = None  # populated when mode == "clarification"


class ClauseVersion(BaseModel):
    """One of the 5 versions returned by the generation call."""

    title: str = Field(description="Clause name or title")
    summary: str = Field(description="One-sentence summary of what this clause covers")
    drafted_clause: str = Field(
        description="Full drafted clause text. Empty string for list_of_clauses mode versions."
    )


class DescribeDraftLLMResponse(BaseModel):
    """Raw LLM output: exactly 5 ClauseVersions."""

    versions: List[ClauseVersion] = Field(
        min_length=5,
        max_length=5,
        description="Exactly 5 versions — validated post-generation",
    )


# --- API response schema ---


class DescribeDraftErrorType(str, Enum):
    """Typed error categories for the describe-draft agent."""

    CLARIFICATION_NEEDED = "clarification_needed"
    VALIDATION_FAILED = "validation_failed"
    LLM_FAILED = "llm_failed"
    RATE_LIMITED = "rate_limited"


class DescribeDraftResponse(BaseModel):
    """Public API response from the describe-draft endpoint."""

    session_id: str
    mode: Literal["list_of_clauses", "single_clause", "clarification"]
    status: Literal["ok", "error"]
    disclaimer: Optional[str] = "AI-generated draft. Subject to attorney review before use."
    clarification_question: Optional[str] = None
    versions: List[ClauseVersion] = Field(default_factory=list)
    error_type: Optional[DescribeDraftErrorType] = None
    error_message: Optional[str] = None
