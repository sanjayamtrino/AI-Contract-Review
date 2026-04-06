from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class ChangeType(Enum):
    ADDED = "added"
    MODIFIED = "modified"
    REMOVED = "removed"


class RiskImpact(Enum):
    INCREASED = "increased"
    DECREASED = "decreased"
    NEUTRAL = "neutral"


class ClauseChange(BaseModel):
    clause_title: Optional[str] = Field(None, description="Clause name (e.g., 'Termination', 'Confidentiality Duration')")
    change_type: ChangeType = Field(..., description="Type of change: added, modified, removed")
    original_text: Optional[str] = Field(None, description="Verbatim from Doc A (null only if 'added')")
    revised_text: Optional[str] = Field(None, description="Verbatim from Doc B (null only if 'removed')")
    change_summary: str = Field(..., description="(1) What changed, (2) legal/commercial impact, (3) which party it favors")
    risk_impact: RiskImpact = Field(..., description="Assessment of the risk impact: increased, decreased, neutral")
    significance: str = Field(..., description="high, medium, or low significance")


# class DocumentComparisonResult(BaseModel):
#     summary: str = Field(..., description="High level summary of all changes")
#     changes: List[ClauseChange] = Field(..., description="List of specific changes identified in the document comparison")


class MissingClauses(BaseModel):
    """List of important clauses that are present in one document but missing in the other, along with potential risks associated with their absence."""

    clause_title: str = Field(..., description="Name of the missing clause (e.g., 'Indemnification', 'Limitation of Liability')")
    missing_in: str = Field(..., description="Indicates which document is missing the clause: 'Doc A' or 'Doc B'")
    clause_text: str = Field(..., description="Verbatim text of the missing clause from the document where it is present")
    impact_summary: str = Field(..., description="Summary of the potential legal and commercial risks associated with the absence of this clause in the other document")
    significance: str = Field(..., description="high, medium, or low significance based on the potential risks")


class DocumentComparisonResult(BaseModel):
    """Result of comparing two versions of a document, including a summary and detailed changes."""

    # executive_summary: str = Field(..., description="3-5 sentences — total changes found, the 2-3 most impactful, overall risk direction and which party it favors")
    # overall_risk_impact: str = Field(..., description="Net effect of all changes combined")
    changes: List[ClauseChange] = Field(..., description="List of specific changes identified in the document comparison")
    missing_clauses: List[MissingClauses] = Field(
        ..., description="List of important clauses that are present in one document but missing in the other, along with potential risks associated with their absence"
    )


class ClauseUnit(BaseModel):
    """A single clause extracted from a document's chunks."""

    clause_id: str = Field(..., description="Unique identifier for this clause unit")
    heading: Optional[str] = Field(None, description="The section heading this clause belongs to, if any")
    content: str = Field(..., description="The full text content of this clause")
    position: int = Field(..., description="The position of this clause in the original document (e.g., clause number or order index)")
    doc_order: int = Field(0, description="The order of this clause in the document, used for reordering detection")
    embedding: List[float] = Field(default_factory=list)


class MatchResult(BaseModel):
    """Output of the clause matching stage."""

    matched_pairs: List[Tuple[int, int, float]] = Field(
        ..., description="List of matched clause pairs with their similarity scores. Each tuple contains (index_in_doc_a, index_in_doc_b, similarity_score)"
    )
    unmatched_a: List[int] = Field(..., description="List of clause indices in Document A that were not matched to any clause in Document B")
    unmatched_b: List[int] = Field(..., description="List of clause indices in Document B that were not matched to any clause in Document A")


class CompareRequest(BaseModel):
    """Request body for the compare endpoint."""

    session_id: str = Field(..., description="Unique identifier for the user's session")
    document_id_a: str = Field(..., description="Unique identifier for Document A (e.g., version 1)")
    document_id_b: str = Field(..., description="Unique identifier for Document B (e.g., version 2)")


class ClauseComparisonLLMResponse(BaseModel):
    """Structured output schema for the per-pair LLM comparison call."""

    change_type: str = Field(description="One of: modified, reordered")
    modification_type: Optional[str] = Field(None, description="Sub-type: value, language, scope, structural, rewritten")
    risk_level: str = Field(description="One of: high, medium, low")
    affected_party: Optional[str] = Field(None, description="Which party is affected by this change")
    old_text: str = Field(description="Exact text from Document A that changed")
    new_text: str = Field(description="Exact text from Document B that changed")
    legal_implication: str = Field(description="Brief note on legal meaning of this change")
    is_substantive: bool = Field(description="Whether this is a meaningful change vs cosmetic")


class ChangeEntry(BaseModel):
    """A single change between two document clauses."""

    clause_name: str = Field(description="Name of the clause or section where the change occurred")
    change_type: str = Field(description="One of: added, removed, modified, reordered")
    modification_type: Optional[str] = Field(None, description="Sub-type: value, language, scope, structural, rewritten")
    risk_level: str = Field(description="One of: high, medium, low")
    affected_party: Optional[str] = Field(None, description="Which party is affected by this change")
    confidence: str = Field("high", description="Confidence level of the change detection")
    text_from_doc_a: Optional[str] = Field(None, description="Text from Document A related to this change")
    text_from_doc_b: Optional[str] = Field(None, description="Text from Document B related to this change")
    legal_implication: Optional[str] = Field(None, description="Brief note on legal meaning of this change")
    is_substantive: bool = Field(True, description="Whether this is a meaningful change vs cosmetic")


class SectionGroup(BaseModel):
    """A group of changes under the same section heading."""

    section_name: str = Field(description="Name of the section or heading that groups these changes")
    changes: List[ChangeEntry] = Field(description="List of changes that fall under this section")


class CompareSummary(BaseModel):
    """Summary statistics for the comparison."""

    total_changes: int = Field(description="Total number of changes detected between the two documents")
    added: int = Field(description="Number of clauses that were added in Document B compared to Document A")
    removed: int = Field(description="Number of clauses that were removed in Document B compared to Document A")
    modified: int = Field(description="Number of clauses that were modified between Document A and Document B")
    reordered: int = Field(description="Number of clauses that were reordered between Document A and Document B")
    overall_risk: str = Field(description="Overall risk level based on the changes")
    high_risk_count: int = Field(description="Number of changes classified as high risk")
    llm_calls_made: int = Field(description="Number of calls made to the LLM for detailed change analysis")
    llm_calls_skipped: int = Field(default=0, description="Number of changes for which LLM analysis was skipped due to low similarity or other heuristics")


class CompareResponse(BaseModel):
    """Top-level response for the compare endpoint."""

    success: bool = Field(description="Indicates whether the comparison was successful")
    error: Optional[str] = Field(None, description="Error message if the comparison failed")
    message: Optional[str] = Field(None, description="Additional information about the comparison")
    summary: Optional[CompareSummary] = Field(None, description="Summary statistics for the comparison")
    sections: List[SectionGroup] = Field(description="List of section groups containing the changes")
    document_id_a: Optional[str] = Field(None, description="Unique identifier for Document A")
    document_id_b: Optional[str] = Field(None, description="Unique identifier for Document B")
