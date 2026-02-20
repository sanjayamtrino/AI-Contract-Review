from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ParaSimilarity(BaseModel):
    """Paragraph Similarity Schema."""

    paragraph: str = Field(..., description="Paragraph of the document.")
    Similarity: float = Field(..., description="Similarity score of the given para to the rule.")


class RuleInfo(BaseModel):
    title: str = Field(..., description="Title of the rule")
    instruction: str = Field(..., description="Instruction for the rule")
    description: str
    tags: Optional[List[str]] = None


class TextInfo(BaseModel):
    text: str = Field(..., description="Text content of the paragraph")
    paraindetifier: str = Field(..., description="Identifier for the paragraph")


class RuleCheckRequest(BaseModel):
    rulesinformation: List[RuleInfo] = Field(..., description="List of rules to check against")
    textinformation: List[TextInfo] = Field(..., description="List of text paragraphs to check")


class Match(BaseModel):
    rule_title: str = Field(..., description="Title of the rule that was matched")
    para_identifier: str = Field(..., description="Identifier of the matching paragraph")
    reasoning: Optional[str] = Field(None, description="Brief explanation of why it matches")


class RuleCheckResponse(BaseModel):
    matches: List[Match] = Field(..., description="List of matches found")


class RuleResult(BaseModel):
    title: str
    instruction: str
    description: str
    paragraphidentifier: str
    paragraphcontext: str
    similarity_scores: List[float] = []


class ResponseStatus(str, Enum):
    """Response status for the AI Review."""

    # CRITICAL = "critical"
    # MEDIUM = "medium"
    # LOW = "low"
    # GOOD = "good"

    PASS = "PASS"
    FAIL = "FAIL"
    NOTFOUND = "NOT FOUND"


class PlayBookReviewRequest(BaseModel):
    """Schema for the AI review request."""

    rule_title: str = Field(..., description="Title of the rule to be compared.")
    # instruction: str = Field(..., description="Instructions for the given rule")
    description: str = Field(..., description="Detailed description of the given rule.")
    paragraphs: List[str] = Field(..., description="list of retrieved paragraphs to validate with.")


class PlayBookReviewResponse(BaseModel):
    """Schema for AI review reponse for the given rules and paras."""

    rule_title: str = Field(..., description="Title of the rule that was considered.")
    rule_instruction: str = Field(..., description="Rule Instruction that was considered.")
    rule_description: str = Field(..., description="Rule Description that was considered.")
    para_identifiers: List[str] = Field(..., description="List of Paragraphs that matched the rule.")
    status: ResponseStatus = Field(..., description="Status of the given rules and para (critical, medium, low, good)")
    reason: str = Field(..., description="Reason of the Review either good or bad,")
    suggestion: str = Field(..., description="A brief suggestion of the paragraphs over the rule.")
    suggested_fix: str = Field(..., description="Suggested fix for the given rule and the paragraph.")


class MatchedParagraph(BaseModel):
    paragraph_id: str = Field(..., description="Exact ID from input")
    paragraph_text: str = Field(..., description="Exact text from input")
    reason: str = Field(..., description="Why this paragraph is relevant — cite specific words")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in paragraph relevance")


class MatchedClause(BaseModel):
    clause_number: str = Field(..., description="e.g., 1, 1.1, 2.3")
    clause_title: str = Field(..., description="Descriptive title")
    clause_text: str = Field(..., description="EXACT verbatim text from the paragraph")
    reason: str = Field(..., description="Why this clause relates to the rule — cite specific words")
    risk_level: str = Field(..., description="Risk level: Low | Medium | High | Critical")
    deviation_from_playbook: str = Field(..., description="Exact deviation or 'None'")
    suggested_revision: str = Field(..., description="Concrete corrected clause text or 'No revision needed'")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in clause evaluation")


class RuleAnalysis(BaseModel):
    analysis_reasoning: str = Field(..., description="2-4 sentences reasoning with PASS/FAIL/NOT FOUND")
    rule_title: str = Field(..., description="Exact rule title from input")
    matched_paragraphs: List[MatchedParagraph] = Field(default_factory=list)
    matched_clauses: List[MatchedClause] = Field(default_factory=list)
    overall_rule_risk: str = Field(..., description="Low | Medium | High | Critical")
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score for this rule")
    summary: str = Field(..., description="PASS: ... | FAIL: ... | NOT FOUND: ... — concise verdict with key evidence")


class PlaybookAnalysisResponse(BaseModel):
    rules_analysis: List[RuleAnalysis] = Field(..., description="List of rule evaluations, one per rule")


class ListPlaybookAnalysisResponse(BaseModel):
    analysis: List[PlaybookAnalysisResponse]
