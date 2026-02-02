from typing import List

from pydantic import BaseModel, Field


class QueryRewriterResponse(BaseModel):
    """Response schema for the query rewriter."""

    queries: List[str] = Field(..., description="Rewritten queries as a list of strings.")
