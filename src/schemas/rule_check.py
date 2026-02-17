from typing import List, Optional
from pydantic import BaseModel, Field

class RuleInfo(BaseModel):
    title: str = Field(..., description="Title of the rule")
    instruction: str = Field(..., description="Instruction for the rule")

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
