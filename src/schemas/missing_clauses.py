"""Schemas for the missing clauses analysis endpoint."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class MissingClause(BaseModel):
    """A single missing, incomplete, or ambiguous clause identified in the contract."""

    clause_name: str = Field(
        ...,
        description="Standard legal name of the clause (e.g., 'Indemnification', 'Force Majeure').",
    )
    status: Literal["absent", "incomplete", "ambiguous"] = Field(
        ...,
        description=(
            "'absent' = clause is entirely missing; "
            "'incomplete' = clause exists but omits critical sub-provisions; "
            "'ambiguous' = clause exists but language is vague or open to interpretation."
        ),
    )
    importance: Literal["high", "medium", "low"] = Field(
        ...,
        description=(
            "'high' = missing clause creates significant legal/financial risk; "
            "'medium' = missing clause may cause disputes but is not critical; "
            "'low' = best-practice clause whose absence is unlikely to cause immediate harm."
        ),
    )
    explanation: str = Field(
        ...,
        description="Why this clause matters for this contract type and what risk its absence or deficiency creates.",
    )
    draft_clause: str = Field(
        ...,
        description="Draft clause language the user could insert into the contract.",
    )


class MissingClausesRequest(BaseModel):
    """Request schema for the missing clauses analysis."""

    contract_text: str = Field(
        ...,
        description="The full contract or document text to analyze for missing clauses.",
    )


class ClauseCheckItem(BaseModel):
    """A single item in the clause-by-clause checklist audit."""

    clause_name: str = Field(
        ...,
        description="Standard clause name being checked.",
    )
    verdict: Literal["covered", "absent", "incomplete", "ambiguous"] = Field(
        ...,
        description="'covered' = found in document; 'absent' = missing; 'incomplete' = partially covered; 'ambiguous' = vague.",
    )
    reference: Optional[str] = Field(
        None,
        description="If covered: brief note of where/how. If not covered: null.",
    )


class MissingClausesLLMResponse(BaseModel):
    """Structured LLM response for the missing clauses analysis."""

    contract_type: str = Field(
        ...,
        description=(
            "Detected contract type (e.g., 'Non-Disclosure Agreement', 'Master Service Agreement'). "
            "Return 'Unidentified' if the document type cannot be determined."
        ),
    )
    clause_checklist: List[ClauseCheckItem] = Field(
        ...,
        description=(
            "Complete clause-by-clause audit. List EVERY standard clause checked — "
            "both covered and missing — so the audit trail is visible."
        ),
    )
    missing_clauses: List[MissingClause] = Field(
        default_factory=list,
        description="List of all missing, incomplete, or ambiguous clauses from the checklist.",
    )
    total_missing: int = Field(
        ...,
        description="Total count of items in missing_clauses.",
    )
    summary: str = Field(
        ...,
        description="Brief overall assessment of the contract's clause completeness and key risks.",
    )
