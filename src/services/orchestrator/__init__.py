"""
Orchestrator Service for Multi-Agent Legal Document Processing

This module provides the central orchestration layer that routes user queries
to appropriate specialized agents using GenAI (Gemini) for classification.
"""

from src.services.orchestrator.orchestrator import AgentOrchestrator
from src.services.orchestrator.classifier import IntentClassifier
from src.services.orchestrator.router import AgentRouter
from src.services.orchestrator.context_manager import ContextManager
from src.services.orchestrator.models import (
    AgentType,
    ClassificationResult,
    RoutingDecision,
    OrchestratorContext,
    AgentResponse,
    ConfidenceLevel,
    SubIntent,
    RequirementType,
)
from src.services.orchestrator.exceptions import (
    OrchestratorError,
    ClassificationError,
    RoutingError,
    AgentTimeoutError,
    ChainDepthExceededError,
)

__all__ = [
    "AgentOrchestrator",
    "IntentClassifier",
    "AgentRouter",
    "ContextManager",
    "AgentType",
    "ClassificationResult",
    "RoutingDecision",
    "OrchestratorContext",
    "AgentResponse",
    "ConfidenceLevel",
    "SubIntent",
    "RequirementType",
    "OrchestratorError",
    "ClassificationError",
    "RoutingError",
    "AgentTimeoutError",
    "ChainDepthExceededError",
]
