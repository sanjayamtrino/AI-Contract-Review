"""Schemas for the Describe & Draft Agent."""

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class DescribeDraftRequest(BaseModel):
    """Request body — prompt and optional regenerate flag; session_id arrives via X-Session-ID header."""

    prompt: str = Field(
        description="Free-text drafting request (e.g. 'draft an NDA' or 'draft a liquidity cap clause')",
        min_length=1,
        max_length=2000,
    )
    regenerate: bool = Field(
        default=False,
        description=(
            "If true and the session has a prior single-clause draft, produce an improved "
            "variation instead of a fresh draft. Ignored for list_of_clauses / clarification."
        ),
    )


# --- LLM internal schemas (not exposed in API response) ---


class IntentClassification(BaseModel):
    """Output of the intent-classifier LLM call."""

    mode: Literal["list_of_clauses", "single_clause", "clarification"]
    detected_agreement_type: Optional[str] = None  # e.g. "NDA", "SaaS Agreement"
    clarification_question: Optional[str] = None  # populated when mode == "clarification"


class ClauseListEntry(BaseModel):
    """One clause entry in the agreement's full clause list (list_of_clauses mode)."""

    title: str = Field(description="Clause name (e.g. 'Confidentiality Obligations')")
    summary: str = Field(
        description="One-sentence description of what this clause covers"
    )


class ClauseVersion(BaseModel):
    """One drafted version of a single clause (single_clause mode).

    A single_clause call returns one ClauseVersion. Clicking regenerate produces
    another improved ClauseVersion for the same clause.
    """

    title: str = Field(description="Clause name or title")
    summary: str = Field(description="One-sentence summary of this version's approach")
    drafted_clause: str = Field(description="The full drafted clause text")


class ClauseListLLMResponse(BaseModel):
    """Raw LLM output for list_of_clauses mode — one complete clause list."""

    clauses: List[ClauseListEntry] = Field(
        min_length=8,
        description=(
            "Complete list of clauses for the requested agreement type. "
            "Minimum 8; prompt requests 12+ for most agreements."
        ),
    )


class DuplicateCheckResult(BaseModel):
    """LLM output for duplicate-clause detection against an uploaded document."""

    is_duplicate: bool = Field(
        description=(
            "True if the candidate text is already a clause on the same topic "
            "as the user's drafting request."
        )
    )
    matched_title: Optional[str] = Field(
        default=None,
        description="Heading/title of the existing clause, if one is discernible.",
    )


class ExistingClauseMatch(BaseModel):
    """Describes a clause already present in the uploaded document that matches the request."""

    title: Optional[str] = Field(
        default=None,
        description="Heading or inferred title of the existing clause.",
    )
    excerpt: str = Field(
        description="Verbatim text of the existing clause as pulled from the document."
    )
    similarity_score: float = Field(
        description="Retrieval similarity score of the matched chunk."
    )


class DescribeDraftLLMResponse(BaseModel):
    """Raw LLM output for single_clause mode — exactly 1 ClauseVersion per call.

    Additional versions are produced by calling the endpoint again with regenerate=true.
    """

    versions: List[ClauseVersion] = Field(
        min_length=1,
        max_length=1,
        description="Exactly 1 version per call — validated post-generation",
    )


# --- API response schema ---


class DescribeDraftErrorType(str, Enum):
    """Typed error categories for the describe-draft agent."""

    CLARIFICATION_NEEDED = "clarification_needed"
    VALIDATION_FAILED = "validation_failed"
    LLM_FAILED = "llm_failed"
    RATE_LIMITED = "rate_limited"


class DescribeDraftResponse(BaseModel):
    """Public API response from the describe-draft endpoint.

    Which field is populated depends on `mode`:
      - list_of_clauses      → `clauses` (the full clause list for the agreement type)
      - single_clause        → `versions` (exactly 1 drafted version per call; regenerate for more)
      - single_clause_exists → `existing_clause` (the matching clause already in the uploaded doc)
      - clarification        → `clarification_question` (no generation)
    """

    session_id: str
    mode: Literal[
        "list_of_clauses",
        "single_clause",
        "single_clause_exists",
        "clarification",
    ]
    status: Literal["ok", "error"]
    disclaimer: Optional[str] = "AI-generated draft. Subject to attorney review before use."
    clarification_question: Optional[str] = None
    clauses: List[ClauseListEntry] = Field(
        default_factory=list,
        description="Populated only in list_of_clauses mode.",
    )
    versions: List[ClauseVersion] = Field(
        default_factory=list,
        description="Populated only in single_clause mode — 1 entry per call.",
    )
    existing_clause: Optional[ExistingClauseMatch] = Field(
        default=None,
        description=(
            "Populated only in single_clause_exists mode — the matching clause "
            "already present in the uploaded document."
        ),
    )
    regenerated: bool = Field(
        default=False,
        description="True when this response was produced by a regenerate call.",
    )
    grounded_in_document: bool = Field(
        default=False,
        description=(
            "True when a document was attached to the session and the draft was grounded "
            "in it (parties, governing law, and relevant existing clauses)."
        ),
    )
    error_type: Optional[DescribeDraftErrorType] = None
    error_message: Optional[str] = None
