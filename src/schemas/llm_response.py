from pydantic import BaseModel, Field


class QueryLLmResponse(BaseModel):
    """Schema for the Response for the given user query."""

    response: str = Field(..., description="Detailed response for the user query.")
