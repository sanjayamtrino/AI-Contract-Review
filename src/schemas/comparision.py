from enum import Enum
from typing import List, Optional

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
