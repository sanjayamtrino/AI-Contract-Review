"""
Agent Router with Requirement Validation

Makes final routing decisions based on classification results and
validates that all requirements are met before execution.
"""

from typing import Any, Dict, List

from src.config.logging import get_logger
from src.services.orchestrator.models import (
    AgentType,
    ClassificationResult,
    RoutingDecision,
    RequirementType,
    RequirementCheck,
    ConfidenceLevel,
    OrchestratorContext,
)
from src.services.orchestrator.exceptions import RoutingError
from src.config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()


class AgentRouter:
    """
    Routes queries to appropriate agents with requirement validation.
    
    The router:
    1. Validates classification confidence
    2. Checks all requirements are met
    3. Determines execution mode (single/chain/parallel)
    4. Selects fallback if needed
    5. Generates clarification questions when ambiguous
    """
    
    # Agent requirements mapping
    AGENT_REQUIREMENTS: Dict[AgentType, List[RequirementType]] = {
        AgentType.DOCUMENT_INFORMATION: [RequirementType.DOCUMENT],
        AgentType.PLAYBOOK_REVIEW: [RequirementType.DOCUMENT, RequirementType.PLAYBOOK],
        AgentType.GENERAL_REVIEW: [RequirementType.DOCUMENT],
        AgentType.RISK_COMPLIANCE: [RequirementType.DOCUMENT],
        AgentType.VERSION_COMPARISON: [RequirementType.MULTIPLE_DOCUMENTS],
        AgentType.STANDARD_CLAUSE: [RequirementType.DOCUMENT],
        AgentType.DOC_CHAT: [RequirementType.DOCUMENT],
        AgentType.PLAYBOOK_RULES: [RequirementType.PLAYBOOK],
        AgentType.INTERACTIVE_AI: [],  # Can work with or without documents
    }
    
    def __init__(self):
        self.fallback_enabled = getattr(settings, 'ORCHESTRATOR_FALLBACK_ENABLED', True)
        self.classification_threshold = getattr(settings, 'ORCHESTRATOR_CLASSIFICATION_THRESHOLD', 0.75)
    
    async def route(
        self,
        classification: ClassificationResult,
        context: OrchestratorContext,
    ) -> RoutingDecision:
        """
        Make routing decision based on classification.
        
        Args:
            classification: Intent classification result
            context: Current orchestrator context
            
        Returns:
            RoutingDecision with validated routing information
        """
        logger.info(
            "Making routing decision",
            agent=classification.primary_agent.value,
            confidence=classification.confidence,
        )
        
        # Check confidence threshold
        if classification.confidence < self.classification_threshold:
            if self.fallback_enabled:
                logger.info("Low confidence, using fallback", confidence=classification.confidence)
                return self._create_fallback_routing(classification)
            else:
                return self._create_clarification_routing(classification)
        
        # Check requirements
        requirements_met, checks, missing = self._validate_requirements(
            classification.primary_agent, context
        )
        
        # Determine execution mode
        execution_mode = self._determine_execution_mode(classification)
        
        # Build parameters
        parameters = self._build_parameters(classification, context)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            classification, requirements_met, execution_mode
        )
        
        decision = RoutingDecision(
            classification=classification,
            requirements_met=requirements_met,
            requirement_checks=checks,
            missing_requirements=missing,
            target_agent=classification.primary_agent,
            execution_mode=execution_mode,
            fallback_agent=AgentType.DOC_CHAT if not requirements_met else None,
            clarification_needed=not requirements_met and len(missing) > 0,
            clarification_question=self._generate_clarification_question(missing) if missing else None,
            suggested_queries=self._generate_suggested_queries(classification, missing),
            parameters=parameters,
            decision_reasoning=reasoning,
        )
        
        logger.info(
            "Routing decision made",
            target=decision.target_agent.value,
            mode=decision.execution_mode,
            requirements_met=decision.requirements_met,
        )
        
        return decision
    
    def _validate_requirements(
        self,
        agent_type: AgentType,
        context: OrchestratorContext,
    ) -> tuple[bool, List[RequirementCheck], List[RequirementType]]:
        """Validate that all requirements for an agent are met."""
        required = self.AGENT_REQUIREMENTS.get(agent_type, [])
        checks = []
        missing = []
        
        for req in required:
            check = self._check_requirement(req, context)
            checks.append(check)
            if not check.available:
                missing.append(req)
        
        return len(missing) == 0, checks, missing
    
    def _check_requirement(
        self,
        requirement: RequirementType,
        context: OrchestratorContext,
    ) -> RequirementCheck:
        """Check if a specific requirement is satisfied."""
        
        if requirement == RequirementType.DOCUMENT:
            available = len(context.active_documents) > 0
            return RequirementCheck(
                requirement_type=requirement,
                required=True,
                available=available,
                details=f"{len(context.active_documents)} document(s) available",
            )
        
        elif requirement == RequirementType.MULTIPLE_DOCUMENTS:
            available = len(context.active_documents) >= 2
            return RequirementCheck(
                requirement_type=requirement,
                required=True,
                available=available,
                details=f"{len(context.active_documents)} document(s) available (need 2+)",
            )
        
        elif requirement == RequirementType.PLAYBOOK:
            available = context.active_playbook is not None
            return RequirementCheck(
                requirement_type=requirement,
                required=True,
                available=available,
                details=f"Playbook: {context.active_playbook.name if context.active_playbook else 'None'}",
            )
        
        elif requirement == RequirementType.PREVIOUS_VERSION:
            return RequirementCheck(
                requirement_type=requirement,
                required=True,
                available=False,
                details="Previous version check not implemented",
            )
        
        elif requirement == RequirementType.USER_PERMISSION:
            return RequirementCheck(
                requirement_type=requirement,
                required=True,
                available=True,
                details="Permission check passed",
            )
        
        return RequirementCheck(
            requirement_type=requirement,
            required=False,
            available=True,
            details="Not applicable",
        )
    
    def _determine_execution_mode(
        self,
        classification: ClassificationResult,
    ) -> str:
        """Determine how agents should be executed."""
        if classification.requires_chaining and len(classification.chain_order) > 1:
            return "chain"
        elif len(classification.secondary_agents) > 0:
            return "single"  # For now, use primary agent
        else:
            return "single"
    
    def _build_parameters(
        self,
        classification: ClassificationResult,
        context: OrchestratorContext,
    ) -> Dict[str, Any]:
        """Build execution parameters for the agent."""
        params = {
            "sub_intent": classification.sub_intent.value if classification.sub_intent else None,
            "extracted_entities": classification.extracted_entities,
            "document_references": classification.document_references,
            "section_references": classification.section_references,
        }
        
        # Add document IDs if available
        if context.active_documents:
            params["document_ids"] = [str(d.document_id) for d in context.active_documents]
        
        # Add playbook info if relevant
        if classification.primary_agent in [AgentType.PLAYBOOK_REVIEW, AgentType.PLAYBOOK_RULES]:
            if context.active_playbook:
                params["playbook_id"] = str(context.active_playbook.playbook_id)
        
        return params
    
    def _generate_reasoning(
        self,
        classification: ClassificationResult,
        requirements_met: bool,
        execution_mode: str,
    ) -> str:
        """Generate human-readable reasoning for the routing decision."""
        parts = [
            f"Primary classification: {classification.primary_agent.value}",
            f"Confidence: {classification.confidence} ({classification.confidence_level.value})",
            f"Requirements met: {requirements_met}",
            f"Execution mode: {execution_mode}",
        ]
        
        if classification.sub_intent:
            parts.append(f"Sub-intent: {classification.sub_intent.value}")
        
        if classification.secondary_agents:
            parts.append(f"Secondary agents: {[a.value for a in classification.secondary_agents]}")
        
        return "; ".join(parts)
    
    def _create_fallback_routing(
        self,
        classification: ClassificationResult,
    ) -> RoutingDecision:
        """Create routing decision for fallback case."""
        return RoutingDecision(
            classification=classification,
            requirements_met=True,
            requirement_checks=[],
            missing_requirements=[],
            target_agent=AgentType.DOC_CHAT,
            execution_mode="single",
            fallback_agent=None,
            clarification_needed=False,
            parameters={},
            decision_reasoning=f"Low confidence ({classification.confidence}) - using fallback agent",
        )
    
    def _create_clarification_routing(
        self,
        classification: ClassificationResult,
    ) -> RoutingDecision:
        """Create routing decision requesting clarification."""
        return RoutingDecision(
            classification=classification,
            requirements_met=False,
            requirement_checks=[],
            missing_requirements=[],
            target_agent=classification.primary_agent,
            execution_mode="single",
            fallback_agent=AgentType.DOC_CHAT,
            clarification_needed=True,
            clarification_question=self._generate_ambiguity_clarification(classification),
            suggested_queries=self._generate_suggested_queries(classification, []),
            parameters={},
            decision_reasoning="Ambiguous classification - clarification needed",
        )
    
    def _generate_clarification_question(
        self,
        missing: List[RequirementType],
    ) -> str:
        """Generate question for missing requirements."""
        if RequirementType.DOCUMENT in missing:
            return "I'd be happy to help, but I don't see any documents. Please upload a document first."
        
        if RequirementType.MULTIPLE_DOCUMENTS in missing:
            return "To compare versions, I need at least two documents. Please upload another version."
        
        if RequirementType.PLAYBOOK in missing:
            return "I don't see an active playbook for your organization. Please set up your playbook first."
        
        return "I need some additional information to help you. Could you provide more details?"
    
    def _generate_ambiguity_clarification(
        self,
        classification: ClassificationResult,
    ) -> str:
        """Generate clarification for ambiguous classification."""
        if classification.ambiguity_flags:
            return (
                f"I'm not entirely sure what you'd like me to do. "
                f"Possible interpretations: {', '.join(classification.ambiguity_flags)}. "
                f"Could you clarify?"
            )
        
        return "I'm not sure I understand. Could you be more specific about what you'd like me to do?"
    
    def _generate_suggested_queries(
        self,
        classification: ClassificationResult,
        missing: List[RequirementType],
    ) -> List[str]:
        """Generate suggested queries based on context."""
        suggestions = []
        
        # Based on primary agent
        agent_suggestions = {
            AgentType.DOCUMENT_INFORMATION: [
                "Summarize this document",
                "What are the key terms?",
                "Extract all dates and amounts",
            ],
            AgentType.PLAYBOOK_REVIEW: [
                "Check this against our playbook",
                "What clauses are missing per our standards?",
            ],
            AgentType.RISK_COMPLIANCE: [
                "What are the risks in this document?",
                "Check GDPR compliance",
            ],
            AgentType.GENERAL_REVIEW: [
                "Review this document for quality",
                "Is this document complete?",
            ],
            AgentType.VERSION_COMPARISON: [
                "What changed from the previous version?",
                "Compare these two documents",
            ],
            AgentType.DOC_CHAT: [
                "What does section 3 say?",
                "Explain the termination clause",
            ],
        }
        
        if classification.primary_agent in agent_suggestions:
            suggestions.extend(agent_suggestions[classification.primary_agent])
        
        # Add generic suggestions
        suggestions.extend([
            "Help me understand this document",
            "What should I focus on?",
        ])
        
        return suggestions[:5]
