from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class ErrorType(str, Enum):
    """Categorizes errors for programmatic handling by API consumers."""

    LLM_FAILURE = "llm_failure"
    LLM_TIMEOUT = "llm_timeout"
    SESSION_NOT_FOUND = "session_not_found"
    NO_DOCUMENT = "no_document"
    UNKNOWN_AGENT = "unknown_agent"
    VALIDATION_ERROR = "validation_error"
    PARSE_ERROR = "parse_error"
    INTERNAL_ERROR = "internal_error"


class AgentError(BaseModel):
    """Structured error from an agent or tool. NEVER confused with domain results."""

    error_type: ErrorType
    message: str = Field(description="Human-readable error description")
    recoverable: bool = Field(
        description="True if the operation could succeed on retry (e.g., LLM timeout)"
    )


class OrchestratorResponse(BaseModel):
    """Standard response envelope from the orchestrator. Either response or error is set, never both."""

    agent: Optional[str] = Field(default=None, description="Agent that handled the request")
    tools_called: List[str] = Field(default_factory=list, description="Tools invoked by the agent")
    response: Optional[Any] = Field(default=None, description="Agent result payload")
    error: Optional[AgentError] = Field(default=None, description="Structured error if request failed")
