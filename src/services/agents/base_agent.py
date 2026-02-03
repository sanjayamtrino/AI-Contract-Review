"""
Base Agent Class

Provides the foundation for all specialized agents in the system.
Implements common functionality including:
- Prompt loading and management
- GenAI (Gemini) interaction
- Response validation
- Token tracking
- Error handling
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4

from src.services.llm.base_model import BaseLLMModel
from src.config.logging import get_logger
from src.services.orchestrator.models import OrchestratorContext, SubIntent

logger = get_logger(__name__)


@dataclass
class AgentResult:
    """Standard result structure for all agents."""
    content: str
    agent_name: str
    sub_intent: Optional[str] = None
    tokens_used: Dict[str, int] = field(default_factory=dict)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    referenced_documents: List[UUID] = field(default_factory=list)
    referenced_sections: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    processing_time_ms: int = 0
    follow_up_questions: List[str] = field(default_factory=list)
    related_actions: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    All specialized agents must inherit from this class and implement
    the abstract methods. The base class provides:
    
    - Prompt template management
    - LLM service integration
    - Response processing
    - Error handling
    - Token tracking
    
    Usage:
        class MyAgent(BaseAgent):
            @property
            def agent_type(self) -> str:
                return "my_agent"
            
            @property
            def system_prompt(self) -> str:
                return "..."
            
            async def process(self, query, sub_intent, parameters, context):
                # Implementation
                pass
    """
    
    def __init__(self, llm_model: BaseLLMModel):
        self.llm = llm_model
    
    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Return the agent type identifier."""
        pass
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the agent's system prompt."""
        pass
    
    @abstractmethod
    async def process(
        self,
        query: str,
        sub_intent: Optional[SubIntent],
        parameters: Dict[str, Any],
        context: OrchestratorContext,
    ) -> AgentResult:
        """
        Process a query and return results.
        
        Args:
            query: The user's query
            sub_intent: Specific sub-intent if applicable
            parameters: Additional parameters from routing
            context: Shared orchestrator context
            
        Returns:
            AgentResult with content and metadata
        """
        pass
    
    async def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 4000,
    ) -> Dict[str, Any]:
        """
        Call the LLM with proper error handling and token tracking.
        
        Args:
            prompt: User query/content
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict with content and metadata
        """
        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Estimate token usage
            tokens_used = self._estimate_tokens(prompt, response)
            
            return {
                "content": response,
                "tokens_used": tokens_used,
                "success": True,
            }
            
        except Exception as e:
            logger.error(
                f"LLM call failed for {self.agent_type}",
                error=str(e),
            )
            raise AgentExecutionError(f"LLM call failed: {str(e)}") from e
    
    def _estimate_tokens(
        self,
        prompt: str,
        response: str,
    ) -> Dict[str, int]:
        """
        Estimate token usage.
        
        Uses approximate token count (4 chars per token).
        """
        prompt_tokens = len(prompt) // 4
        completion_tokens = len(response) // 4
        
        return {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": prompt_tokens + completion_tokens,
        }
    
    def _build_citations(
        self,
        context: OrchestratorContext,
        sections: List[str],
    ) -> List[Dict[str, Any]]:
        """Build citation objects for referenced sections."""
        citations = []
        
        for section in sections:
            citation = {
                "section": section,
                "document_id": str(context.active_documents[0].document_id) if context.active_documents else None,
                "document_name": context.active_documents[0].filename if context.active_documents else None,
            }
            citations.append(citation)
        
        return citations
    
    def _extract_sections_from_response(self, response: str) -> List[str]:
        """Extract section references from agent response."""
        import re
        
        # Look for patterns like "Section X.Y" or "Section X"
        pattern = r'Section\s+(\d+(?:\.\d+)*)'
        matches = re.findall(pattern, response, re.IGNORECASE)
        
        return list(set(matches))  # Deduplicate
    
    def _validate_response(self, response: str) -> bool:
        """Validate that response meets quality standards."""
        # Check minimum length
        if len(response) < 50:
            logger.warning(f"Response too short: {len(response)} chars")
            return False
        
        return True
    
    def _handle_error(self, error: Exception, query: str) -> AgentResult:
        """Create error result."""
        logger.error(f"Agent {self.agent_type} error", error=str(error))
        
        return AgentResult(
            content=(
                f"I apologize, but I encountered an error while processing your request. "
                f"Please try again or rephrase your question."
            ),
            agent_name=self.agent_type,
            confidence_score=0.0,
            metadata={"error": str(error), "error_type": type(error).__name__},
        )


class AgentExecutionError(Exception):
    """Exception raised during agent execution."""
    pass
