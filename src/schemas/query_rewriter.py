from typing import List

from pydantic import BaseModel, Field, field_validator


class Query(BaseModel):
    """Individual query object."""

    query: str = Field(..., description="A rewritten query string.")


class QueryRewriterResponse(BaseModel):
    """Response schema for the query rewriter."""

    queries: List[Query] = Field(..., description="Rewritten queries as a list of query objects.")

    @field_validator("queries", mode="before")
    @classmethod
    def normalize_queries(cls, v: list) -> list:
        """Normalize queries to ensure they are Query objects."""
        if isinstance(v, list):
            return [Query(query=item) if isinstance(item, str) else item for item in v]
        return v
