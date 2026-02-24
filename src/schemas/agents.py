from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """Typed response from any agent. Replaces plain Dict[str, Any] returns."""

    agent: str = Field(description="Name of the agent that handled the request")
    tools_called: List[str] = Field(
        default_factory=list, description="Tools invoked by the agent"
    )
    tool_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Results from each tool keyed by tool name",
    )
    response: Optional[str] = Field(
        default=None,
        description="Direct text response when no tools were called (e.g., clarification)",
    )
