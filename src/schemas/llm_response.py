"""Schema for LLM query responses."""

from pydantic import BaseModel, Field


class QueryLLmResponse(BaseModel):
    """Response schema for user document queries."""

    response: str = Field(..., description="Detailed response for the user query.")
