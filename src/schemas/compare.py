from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# --- Request model ---


class CompareRequest(BaseModel):
    """Request to compare two contract documents within the same session."""

    session_id: str = Field(description="Session containing both ingested documents")
    document_id_a: str = Field(
        description="Document ID of the original/baseline contract"
    )
    document_id_b: str = Field(
        description="Document ID of the revised contract"
    )


# --- LLM structured output model ---


class ClauseComparisonResult(BaseModel):
    """Structured output the LLM returns for a single clause-pair comparison.

    Used as the response_model parameter in AzureOpenAIModel.generate().
    """

    change_type: Literal["modified", "reordered"] = Field(
        description="Type of change: 'modified' if legal meaning changed, "
        "'reordered' if only position or formatting changed"
    )
    old_text: str = Field(
        description="Exact quoted text from the original clause that changed"
    )
    new_text: str = Field(
        description="Exact quoted text from the revised clause that changed"
    )
    risk_level: Literal["high", "medium", "low"] = Field(
        description="Risk level assessed from the reviewing party's perspective"
    )
    risk_justification: str = Field(
        description="1-2 sentence explanation of why this risk level was assigned"
    )
    affected_party: Literal["party_a", "party_b", "neutral"] = Field(
        description="Which party the change favors: party_a (original drafter), "
        "party_b (reviser), or neutral"
    )
    is_substantive: bool = Field(
        description="True if the change alters legal obligations, rights, or terms; "
        "false if cosmetic (formatting, whitespace, reordering)"
    )
    legal_implication: str = Field(
        description="1 sentence describing the practical legal impact of this change"
    )


# --- Internal pipeline models (not for LLM) ---


class ClauseUnit(BaseModel):
    """A single clause/chunk extracted from a document."""

    chunk_index: int = Field(description="Index of the chunk in the FAISS store")
    content: str = Field(description="Full text content of the clause")
    section_heading: Optional[str] = Field(
        default=None, description="Section heading this clause belongs to"
    )
    document_id: str = Field(description="ID of the source document")
    position: int = Field(
        description="Ordinal position of this clause within the document"
    )


class ClauseMatch(BaseModel):
    """A matched pair of clauses across two document versions."""

    clause_a: ClauseUnit = Field(description="Clause from the original document")
    clause_b: ClauseUnit = Field(description="Clause from the revised document")
    similarity: float = Field(
        description="Embedding similarity score between the two clauses"
    )


class MatchResult(BaseModel):
    """Container for all matching results from the clause-pairing step."""

    paired: List[ClauseMatch] = Field(
        default_factory=list,
        description="Clause pairs matched above the similarity threshold",
    )
    unmatched_a: List[ClauseUnit] = Field(
        default_factory=list,
        description="Clauses from document A with no match (removed clauses)",
    )
    unmatched_b: List[ClauseUnit] = Field(
        default_factory=list,
        description="Clauses from document B with no match (added clauses)",
    )


# --- Response models ---


class ChangeEntry(BaseModel):
    """A single change in the comparison diff output."""

    change_type: Literal["added", "removed", "modified", "reordered"] = Field(
        description="Type of change detected between the two document versions"
    )
    clause_id: Optional[str] = Field(
        default=None, description="Original clause number or identifier if available"
    )
    clause_heading: Optional[str] = Field(
        default=None, description="Section heading the clause belongs to"
    )
    old_text: Optional[str] = Field(
        default=None,
        description="Full clause text from original for removed/modified; None for added",
    )
    new_text: Optional[str] = Field(
        default=None,
        description="Full clause text from revised for added/modified; None for removed",
    )
    risk_level: Optional[Literal["high", "medium", "low"]] = Field(
        default=None,
        description="Risk level of the change; None for reordered clauses",
    )
    risk_justification: Optional[str] = Field(
        default=None, description="Brief explanation of the assigned risk level"
    )
    affected_party: Optional[Literal["party_a", "party_b", "neutral"]] = Field(
        default=None,
        description="Which party the change favors; None for reordered clauses",
    )
    is_substantive: Optional[bool] = Field(
        default=None,
        description="Whether the change alters legal meaning; None for reordered",
    )
    legal_implication: Optional[str] = Field(
        default=None,
        description="Practical legal impact of the change; None for reordered",
    )


class SectionDiff(BaseModel):
    """Changes grouped by contract section or heading."""

    section_heading: str = Field(
        description="Name of the contract section these changes belong to"
    )
    changes: List[ChangeEntry] = Field(
        default_factory=list,
        description="List of individual changes within this section",
    )


class ComparisonMetadata(BaseModel):
    """Metadata about a document comparison operation."""

    document_name_a: Optional[str] = Field(
        default=None, description="Display name of the original document"
    )
    document_name_b: Optional[str] = Field(
        default=None, description="Display name of the revised document"
    )
    session_id: str = Field(description="Session ID containing both documents")
    document_id_a: str = Field(description="Document ID of the original")
    document_id_b: str = Field(description="Document ID of the revised")
    comparison_timestamp: str = Field(
        description="ISO 8601 timestamp of when the comparison was performed"
    )
    total_changes: int = Field(
        description="Total number of changes detected across all sections"
    )


class CompareResponse(BaseModel):
    """Top-level response from the Compare Agent."""

    sections: List[SectionDiff] = Field(
        default_factory=list,
        description="Changes grouped by contract section/heading",
    )
    metadata: Optional[ComparisonMetadata] = Field(
        default=None, description="Comparison metadata including document identifiers"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if comparison could not be completed",
    )
