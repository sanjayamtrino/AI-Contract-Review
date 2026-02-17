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
    tags: List[str]


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

    CRITICAL = "critical"
    MEDIUM = "medium"
    LOW = "low"
    GOOD = "good"


class PlayBookReviewRequest(BaseModel):
    """Schema for the AI review request."""

    rule_title: str = Field(..., description="Title of the rule to be compared.")
    # instruction: str = Field(..., description="Instructions for the given rule")
    description: str = Field(..., description="Detailed description of the given rule.")
    paragraphs: List[str] = Field(..., description="list of retrieved paragraphs to validate with.")


class PlayBookReviewResponse(BaseModel):
    """Schema for AI review reponse for the given rules and paras."""

    rule_title: str = Field(..., description="Title of the rule that was considered.")
    status: ResponseStatus = Field(..., description="Status of the given rules and para (critical, medium, low, good)")
    reason: str = Field(..., description="Reason of the Review either good or bad,")
    suggested_fix: str = Field(..., description="Suggested fix for the given rule and the paragraph.")
