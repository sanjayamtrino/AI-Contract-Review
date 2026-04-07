from pydantic import BaseModel, Field


class GeneralReviewResponse(BaseModel):
    reason: str = Field(..., description="Reason for the review result, explaining why the paragraph does or does not comply with the rule.")
    suggested_fix: str = Field(..., description="Suggested fix or improvement to make the paragraph comply with the rule, if applicable.")


class GeneralReviewRequest(BaseModel):
    paragraph: str = Field(..., description="The specific paragraph from the contract that needs to be reviewed against the rule.")
    rule: str = Field(..., description="The specific rule or guideline that the paragraph is being reviewed against (e.g., 'Confidentiality clause must specify duration of at least 3 years').")
