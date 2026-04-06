"""Schemas for the playbook review pipeline (rules, evaluations, reports)."""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_FOUND = "NOT FOUND"


class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


# ── Layer 1: Playbook Rule (input from JSON) ────────────────────


class PlaybookRule(BaseModel):
    """A single playbook rule loaded and normalized from JSON."""

    title: str = Field(..., description="Rule title / name")
    instruction: str = Field(..., description="What to check for in the contract")
    description: str = Field(
        ..., description="The expected standard position or detailed rule text"
    )
    category: Optional[str] = Field(
        None, description="Rule category (liability, termination, etc.)"
    )
    standard_position: Optional[str] = None
    fallback_position: Optional[str] = None
    canned_response: Optional[str] = None
    order: Optional[int] = None


# ── Layer 2: Per-Rule LLM Evaluation Output ─────────────────────
# Matches the JSON structure in rule_evaluation_v2_prompt.mustache


class EvaluatedParagraph(BaseModel):
    """LLM evaluation of a single paragraph against a rule.

    .. deprecated::
        Replaced by ``para_identifiers`` on :class:`RuleEvaluation` in the V2
        pipeline.  Kept for backward-compatibility with serialised V1 reports.
    """

    paragraph_id: str = Field(
        ..., description="Chunk index or identifier from FAISS"
    )
    paragraph_text: str = Field(
        ..., description="Exact verbatim text from the document chunk"
    )
    relevance: str = Field(
        ..., description="Why this paragraph is or is not relevant — cite specific words"
    )
    verdict: Verdict
    evidence: str = Field(
        ..., description="Direct quote from the paragraph supporting the verdict"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)


class RuleEvaluation(BaseModel):
    """LLM output for evaluating one rule against retrieved paragraphs.

    This is the response_model passed to AzureOpenAIModel.generate().
    Field names and aliases match the JSON format in rule_evaluation_v2_prompt.mustache.
    """

    model_config = {"populate_by_name": True}

    # Internal: chain-of-thought (improves accuracy, hidden from user)
    reasoning: str = Field(
        ...,
        alias="_reasoning",
        description="Step-by-step chain-of-thought analysis",
    )

    # User-facing core fields
    rule_title: str
    rule_instruction: str = Field(
        ..., description="Echo-back of the exact rule instruction"
    )
    rule_description: str = Field(
        ..., description="Echo-back of the exact rule description"
    )
    para_identifiers: List[str] = Field(
        default_factory=list,
        description="Matched chunk indices as strings",
    )
    status: Verdict
    reason: str = Field(
        ..., description="2-4 sentence explanation with exact text quotes"
    )
    suggestion: str = Field(
        default="",
        description="Remediation guidance (if FAIL/NOT FOUND), empty string if PASS",
    )
    suggested_fix: str = Field(
        default="",
        description="Full corrected clause text, or empty string if PASS",
    )

    # Internal: needed for report sorting/grouping/risk computation
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_level: RiskLevel


# ── Layer 3: Deterministic Report (NO LLM) ──────────────────────


class RuleResult(BaseModel):
    """A single rule's result in the final report.
    Combines the rule definition with its evaluation."""

    rule_title: str
    rule_instruction: str = Field(
        default="", description="The rule instruction that was evaluated"
    )
    rule_description: str = Field(
        default="", description="The rule description that was evaluated"
    )
    para_identifiers: List[str] = Field(
        default_factory=list,
        description="Matched chunk indices as strings",
    )
    category: Optional[str] = None
    status: Verdict
    risk_level: RiskLevel
    confidence: float
    reason: str
    suggestion: str = Field(
        default="",
        description="Remediation guidance (if FAIL/NOT FOUND)",
    )
    suggested_fix: str = Field(
        default="",
        description="Full corrected clause text, or empty string if PASS",
    )
    paragraphs_retrieved: int = Field(
        ..., description="How many paragraphs were retrieved from FAISS"
    )


class ReportStatistics(BaseModel):
    """Aggregate statistics for the review report."""

    total_rules: int
    rules_passed: int
    rules_failed: int
    rules_not_found: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int


class PlaybookReviewReport(BaseModel):
    """Final deterministic report. Built in Python, NOT by LLM."""

    session_id: str
    playbook_source: str = Field(
        ..., description="Which playbook JSON file was used"
    )
    statistics: ReportStatistics
    overall_risk_level: RiskLevel
    rule_results: List[RuleResult]
    rules_by_risk: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Rule titles grouped by risk level for quick access",
    )
    missing_clauses: List[str] = Field(
        default_factory=list,
        description="Rule titles where status == NOT_FOUND",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Any rules that failed evaluation due to errors",
    )
