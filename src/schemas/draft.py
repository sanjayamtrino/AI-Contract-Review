"""Schemas for the Describe & Draft endpoint."""

from typing import List, Optional

from pydantic import BaseModel, Field


class DraftRequest(BaseModel):
    """Request body for the draft endpoint."""

    session_id: str
    user_prompt: str = Field(description="What the user wants drafted")


class DraftVersion(BaseModel):
    """A single draft alternative with a style label."""

    version: int = Field(description="1, 2, or 3")
    label: str = Field(description="Formal, Plain English, or Concise")
    text: str = Field(description="The drafted clause/agreement text")


class DraftLLMResponse(BaseModel):
    """Internal model for structured LLM output."""

    summary: str = Field(
        description="Clean restatement of the user's request in 1-2 sentences"
    )
    drafts: List[DraftVersion] = Field(min_length=3, max_length=3)


class DraftResponse(BaseModel):
    """API response for the draft endpoint."""

    session_id: str
    status: str = Field(default="ok")
    summary: str = Field(
        default="", description="Clean restatement of user's request"
    )
    note: Optional[str] = Field(
        default=None,
        description="Warning if similar clause exists in document",
    )
    drafts: List[DraftVersion] = Field(default_factory=list)
    error: Optional[str] = Field(default=None)
