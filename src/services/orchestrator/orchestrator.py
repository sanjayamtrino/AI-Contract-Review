"""
Agent Orchestrator - Main Entry Point

Central orchestration engine that coordinates all agents using GenAI (Gemini)
for classification and FAISS for vector storage.
"""

import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime
from time import perf_counter
from uuid import uuid4

from src.config.logging import get_logger
from src.config.settings import get_settings
from src.services.orchestrator.models import (
    AgentType,
    ClassificationResult,
    RoutingDecision,
    OrchestratorContext,
    OrchestratorResponse,
    AgentResponse,
)
from src.services.orchestrator.classifier import IntentClassifier
from src.services.orchestrator.router import AgentRouter
from src.services.orchestrator.context_manager import ContextManager
from src.services.orchestrator.exceptions import (
    OrchestratorInitializationError,
    AgentTimeoutError,
    ChainDepthExceededError,
    UnknownAgentError,
)
from src.services.llm.base_model import BaseLLMModel
from src.services.vector_store.embeddings.embedding_service import EmbeddingService

logger = get_logger(__name__)
settings = get_settings()


class AgentOrchestrator:
    """
    Main orchestrator for the Legal AI system.
    
    Coordinates:
    - Intent classification using GenAI (Gemini)
    - Agent routing with requirement validation
    - Context management with FAISS
    - Multi-agent chain execution
    - Fallback handling
    
    Usage:
        orchestrator = AgentOrchestrator(llm_model, embedding_service)
        await orchestrator.initialize()
        response = await orchestrator.process_query(
            query="Summarize this contract",
            context=orchestrator_context
        )
    """
    
    def __init__(
        self,
        llm_model: BaseLLMModel,
        embedding_service: EmbeddingService,
    ):
        self.llm = llm_model
        self.embedding_service = embedding_service
        
        # Sub-components
        self.classifier: Optional[IntentClassifier] = None
        self.router: Optional[AgentRouter] = None
        self.context_manager: Optional[ContextManager] = None
        
        # Agent registry
        self._agents: Dict[AgentType, Any] = {}
        
        # Performance tracking
        self._execution_stats: Dict[str, Any] = {
            "total_queries": 0,
            "successful_routings": 0,
            "fallback_activations": 0,
            "chain_executions": 0,
            "average_latency_ms": 0.0,
        }
        
        self._initialized = False
    
    async def initialize(self) -> "AgentOrchestrator":
        """
        Initialize the orchestrator with all components.
        
        Returns:
            Self for method chaining
        """
        if self._initialized:
            logger.warning("Orchestrator already initialized")
            return self
        
        logger.info("Initializing Agent Orchestrator")
        
        try:
            # Initialize sub-components
            self.classifier = IntentClassifier(self.llm)
            self.router = AgentRouter()
            self.context_manager = ContextManager(self.embedding_service)
            
            self._initialized = True
            logger.info("Agent Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize orchestrator", error=str(e))
            raise OrchestratorInitializationError(f"Initialization failed: {str(e)}") from e
        
        return self
    
    async def process_query(
        self,
        query: str,
        context: OrchestratorContext,
        correlation_id: Optional[str] = None,
    ) -> OrchestratorResponse:
        """
        Process a user query through the orchestration pipeline.
        
        Args:
            query: User's natural language query
            context: Orchestrator context with documents, history, etc.
            correlation_id: Optional request tracing ID
            
        Returns:
            OrchestratorResponse with content and metadata
        """
        start_time = perf_counter()
        
        logger.info(
            "Processing query",
            query=query[:100],
            session_id=str(context.session_id),
            correlation_id=correlation_id or "none",
        )
        
        try:
            # Step 1: Classify intent
            classification = await self.classifier.classify(query, context)
            
            # Step 2: Make routing decision
            routing_decision = await self.router.route(classification, context)
            
            # Step 3: Execute based on routing
            if routing_decision.clarification_needed:
                response = await self._handle_clarification(routing_decision, context)
            elif routing_decision.execution_mode == "chain":
                response = await self._execute_chain(
                    routing_decision, query, context
                )
            else:
                response = await self._execute_single_agent(
                    routing_decision, query, context
                )
            
            # Update statistics
            self._update_stats(success=True, chained=routing_decision.execution_mode == "chain")
            
            # Calculate processing time
            processing_time_ms = int((perf_counter() - start_time) * 1000)
            
            # Build final response
            return OrchestratorResponse(
                content=response.content,
                agent_used=routing_decision.target_agent,
                confidence_score=classification.confidence,
                processing_time_ms=processing_time_ms,
                tokens_used=response.tokens_used,
                citations=response.citations,
                referenced_documents=response.referenced_documents,
                referenced_sections=response.referenced_sections,
                execution_trace=context.execution_trace,
                was_chained=routing_decision.execution_mode == "chain",
                follow_up_questions=response.follow_up_questions,
                related_actions=response.related_actions,
            )
            
        except Exception as e:
            logger.error("Query processing failed", error=str(e), exc_info=True)
            
            # Update statistics
            self._update_stats(success=False)
            
            # Attempt fallback
            return await self._handle_error_fallback(query, context, e)
    
    async def _execute_single_agent(
        self,
        routing: RoutingDecision,
        query: str,
        context: OrchestratorContext,
    ) -> AgentResponse:
        """Execute a single agent."""
        agent_type = routing.target_agent
        
        logger.info(
            "Executing single agent",
            agent=agent_type.value,
            sub_intent=routing.classification.sub_intent,
        )
        
        # Get agent instance
        agent = self._get_agent(agent_type)
        
        # Execute with timeout
        try:
            timeout = getattr(settings, 'ORCHESTRATOR_AGENT_TIMEOUT_SECONDS', 120)
            result = await asyncio.wait_for(
                agent.process(
                    query=query,
                    sub_intent=routing.classification.sub_intent,
                    parameters=routing.parameters,
                    context=context,
                ),
                timeout=timeout,
            )
            
            return result
            
        except asyncio.TimeoutError:
            logger.error("Agent execution timed out", agent=agent_type.value)
            raise AgentTimeoutError(f"Agent {agent_type.value} timed out")
    
    async def _execute_chain(
        self,
        routing: RoutingDecision,
        query: str,
        context: OrchestratorContext,
    ) -> AgentResponse:
        """Execute multiple agents in a chain."""
        logger.info(
            "Executing agent chain",
            agents=[a.value for a in routing.classification.chain_order],
        )
        
        # Check chain depth
        if not self.context_manager.check_chain_depth(context):
            max_depth = getattr(settings, 'ORCHESTRATOR_MAX_CHAIN_DEPTH', 3)
            raise ChainDepthExceededError(
                f"Maximum chain depth ({max_depth}) exceeded"
            )
        
        self.context_manager.increment_chain_depth(context)
        
        # Execute each agent in sequence
        last_result = None
        
        for i, agent_type in enumerate(routing.classification.chain_order):
            logger.info(f"Chain step {i+1}: {agent_type.value}")
            
            # Transform query based on previous results
            step_query = query if i == 0 else self._transform_for_chain(last_result, agent_type)
            
            # Get and execute agent
            agent = self._get_agent(agent_type)
            
            last_result = await agent.process(
                query=step_query,
                context=context,
            )
            
            # Store intermediate result
            self.context_manager.add_intermediate_result(
                context,
                f"step_{i+1}_{agent_type.value}",
                last_result.content,
            )
        
        return last_result
    
    def _transform_for_chain(
        self,
        previous_result: AgentResponse,
        target_agent: AgentType,
    ) -> str:
        """Transform previous agent output for next agent in chain."""
        transformations = {
            (AgentType.DOCUMENT_INFORMATION, AgentType.RISK_COMPLIANCE):
                lambda x: f"Based on this document summary, identify risks:\n\n{x.content}",
            (AgentType.DOCUMENT_INFORMATION, AgentType.PLAYBOOK_REVIEW):
                lambda x: f"Review this document summary against our playbook:\n\n{x.content}",
        }
        
        # Find matching transformation
        for (from_agent, to_agent), transform in transformations.items():
            if to_agent == target_agent:
                return transform(previous_result)
        
        return f"Previous analysis:\n{previous_result.content[:2000]}\n\nPlease continue the analysis."
    
    async def _handle_clarification(
        self,
        routing: RoutingDecision,
        context: OrchestratorContext,
    ) -> AgentResponse:
        """Handle cases where clarification is needed."""
        logger.info("Requesting user clarification")
        
        return AgentResponse(
            content=routing.clarification_question or 
                    "I'm not sure I understand. Could you clarify what you'd like me to do?",
            agent_name="orchestrator",
            follow_up_questions=routing.suggested_queries,
            related_actions=[
                {"label": "Summarize document", "query": "Summarize this document"},
                {"label": "Check for risks", "query": "What are the risks in this document?"},
            ],
        )
    
    async def _handle_error_fallback(
        self,
        query: str,
        context: OrchestratorContext,
        error: Exception,
    ) -> OrchestratorResponse:
        """Handle errors with fallback response."""
        logger.info("Activating error fallback")
        
        return OrchestratorResponse(
            content=(
                "I apologize, but I encountered an issue while processing your request. "
                "Please try again or contact support if the problem persists."
            ),
            agent_used=AgentType.DOC_CHAT,
            confidence_score=0.5,
            processing_time_ms=0,
            partial_response=True,
            error_info={
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )
    
    def _get_agent(self, agent_type: AgentType) -> Any:
        """Get or create agent instance."""
        if agent_type not in self._agents:
            self._agents[agent_type] = self._create_agent(agent_type)
        return self._agents[agent_type]
    
    def _create_agent(self, agent_type: AgentType) -> Any:
        """Factory method to create agent instances."""
        from src.services.agents import AGENT_REGISTRY
        
        agent_class = AGENT_REGISTRY.get(agent_type)
        if not agent_class:
            raise UnknownAgentError(f"Unknown agent type: {agent_type}")
        
        return agent_class(self.llm)
    
    def _update_stats(self, success: bool, chained: bool = False) -> None:
        """Update execution statistics."""
        self._execution_stats["total_queries"] += 1
        
        if success:
            self._execution_stats["successful_routings"] += 1
        else:
            self._execution_stats["fallback_activations"] += 1
        
        if chained:
            self._execution_stats["chain_executions"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current execution statistics."""
        return self._execution_stats.copy()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on orchestrator components."""
        health = {
            "orchestrator": "healthy" if self._initialized else "not_initialized",
            "classifier": "healthy" if self.classifier else "missing",
            "router": "healthy" if self.router else "missing",
            "context_manager": "healthy" if self.context_manager else "missing",
            "agents_registered": len(self._agents),
        }
        
        health["overall"] = "healthy" if all(
            v == "healthy" for k, v in health.items() 
            if k not in ["agents_registered"]
        ) else "degraded"
        
        return health
