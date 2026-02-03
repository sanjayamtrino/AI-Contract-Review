"""
Orchestrator Exceptions

Custom exceptions for the orchestrator system.
"""


class OrchestratorError(Exception):
    """Base exception for orchestrator errors."""
    pass


class OrchestratorInitializationError(OrchestratorError):
    """Raised when orchestrator fails to initialize."""
    pass


class ClassificationError(OrchestratorError):
    """Exception raised during intent classification."""
    pass


class RoutingError(OrchestratorError):
    """Raised when routing decision fails."""
    pass


class UnknownAgentError(OrchestratorError):
    """Raised when an unknown agent type is requested."""
    pass


class AgentTimeoutError(OrchestratorError):
    """Raised when agent execution times out."""
    pass


class ChainTimeoutError(OrchestratorError):
    """Raised when agent chain execution times out."""
    pass


class ChainDepthExceededError(OrchestratorError):
    """Raised when chain depth exceeds maximum."""
    pass


class AgentExecutionError(OrchestratorError):
    """Exception raised during agent execution."""
    pass


class FallbackActivationError(OrchestratorError):
    """Raised when fallback is activated."""
    pass
