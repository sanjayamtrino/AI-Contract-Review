from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────


class ChangeType(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"


class RiskImpact(str, Enum):
    INCREASED = "increased"
    DECREASED = "decreased"
    NEUTRAL = "neutral"


class Significance(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ── LLM Response Models ─────────────────────────────────────────
# Passed as response_model to AzureOpenAIModel.generate()


class ClauseChange(BaseModel):
    """A single clause-level change between two contract versions."""

    model_config = {"populate_by_name": True}

    reasoning: str = Field(
        ...,
        alias="_reasoning",
        description="Step-by-step analysis of how this change was identified",
    )
    clause_title: str = Field(
        ...,
        description="Title or subject of the clause (e.g. 'Termination', 'Liability Cap')",
    )
    change_type: ChangeType = Field(
        ..., description="Type of change: added, removed, or modified"
    )
    original_text: Optional[str] = Field(
        None,
        description="Exact verbatim text from Document A (original). Null if clause was added.",
    )
    revised_text: Optional[str] = Field(
        None,
        description="Exact verbatim text from Document B (revised). Null if clause was removed.",
    )
    change_summary: str = Field(
        ...,
        description="2-3 sentence plain-language explanation of what changed and why it matters",
    )
    risk_impact: RiskImpact = Field(
        ...,
        description="Whether the change increases, decreases, or has neutral impact on risk",
    )
    significance: Significance = Field(
        ..., description="How significant the change is: high, medium, or low"
    )


class MissingClauseInVersion(BaseModel):
    """A clause present in one document version but entirely absent in the other."""

    clause_title: str = Field(
        ...,
        description="Name or title of the missing clause (e.g. 'Force Majeure', 'Non-Compete')",
    )
    missing_in: Literal["document_a", "document_b"] = Field(
        ...,
        description=(
            "'document_a' = clause exists in Document B (revised) but is missing in Document A (original). "
            "'document_b' = clause exists in Document A (original) but is missing in Document B (revised)."
        ),
    )
    clause_text: str = Field(
        ...,
        description="Verbatim text of the clause from the document that contains it",
    )
    impact_summary: str = Field(
        ...,
        description=(
            "2-3 sentence explanation of why this clause is missing, "
            "what risk it creates, and which party is affected"
        ),
    )
    significance: Significance = Field(
        ..., description="How significant the absence is: high, medium, or low"
    )


class VersionCompareResult(BaseModel):
    """LLM-generated structured comparison of two contract versions.

    This is the response_model passed to AzureOpenAIModel.generate().
    """

    model_config = {"populate_by_name": True}

    analysis_reasoning: str = Field(
        ...,
        alias="_analysis_reasoning",
        description="High-level chain-of-thought before listing individual changes",
    )
    changes: List[ClauseChange] = Field(
        ...,
        description="List of all clause-level changes between the two versions",
    )
    missing_clauses: List[MissingClauseInVersion] = Field(
        ...,
        description=(
            "Clauses that exist in one document version but are entirely absent "
            "in the other. Do NOT include clauses that were modified — only those "
            "with no counterpart at all. Return an empty array [] if no clauses are missing."
        ),
    )
    executive_summary: str = Field(
        ...,
        description=(
            "3-5 sentence executive summary of the key differences, "
            "highlighting the most impactful changes"
        ),
    )
    overall_risk_impact: RiskImpact = Field(
        ..., description="Net risk impact of all changes combined"
    )


# ── Deterministic Report Model (built in Python, NOT by LLM) ───


class CompareStatistics(BaseModel):
    """Aggregate statistics for the comparison."""

    total_changes: int = Field(..., description="Total number of clause changes found")
    clauses_added: int = Field(0, description="Number of new clauses in revised version")
    clauses_removed: int = Field(
        0, description="Number of clauses removed from original"
    )
    clauses_modified: int = Field(
        0, description="Number of clauses with modifications"
    )
    high_significance: int = Field(
        0, description="Number of high-significance changes"
    )
    medium_significance: int = Field(
        0, description="Number of medium-significance changes"
    )
    low_significance: int = Field(0, description="Number of low-significance changes")
    clauses_missing_in_original: int = Field(
        0, description="Clauses present in revised but missing in original"
    )
    clauses_missing_in_revised: int = Field(
        0, description="Clauses present in original but missing in revised"
    )


class VersionCompareReport(BaseModel):
    """Final comparison report. Statistics built in Python, NOT by LLM."""

    session_id: str
    document_a_id: str = Field(
        ..., description="Document ID of the original (first ingested) version"
    )
    document_b_id: str = Field(
        ..., description="Document ID of the revised (second ingested) version"
    )
    statistics: CompareStatistics
    executive_summary: str
    overall_risk_impact: RiskImpact
    changes: List[ClauseChange]
    changes_by_type: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Clause titles grouped by change type for quick access",
    )
    high_risk_changes: List[str] = Field(
        default_factory=list,
        description="Clause titles where risk_impact == increased AND significance == high",
    )
    missing_clauses: List[MissingClauseInVersion] = Field(
        default_factory=list,
        description="Clauses present in one version but entirely absent in the other",
    )
