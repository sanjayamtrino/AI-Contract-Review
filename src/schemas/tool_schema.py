from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field


class SummaryToolRequest(BaseModel):
    """Request Schema for the Summary Tool."""

    session_id: Optional[str] = Field(..., description="Session Id of the data.")


class ToolResponse(BaseModel):
    """Response Schema for the tool responses."""

    tool_id: Union[str, None] = Field(None, description="Unique id for the tool.")
    status: bool = Field(..., description="Response status of the tool.")
    response: Dict[str, Any] = Field(..., description="Response content of the tool.")
    metadata: Dict[str, Any] = Field(..., description="Tool Response Metadata.")
    response_time: str = Field(..., description="Time taken for the response.")
