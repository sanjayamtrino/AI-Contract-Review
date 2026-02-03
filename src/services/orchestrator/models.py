"""
Data Models for Orchestrator System

Pydantic models defining the structure of all data flowing through the orchestrator.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, ConfigDict


class ConfidenceLevel(str, Enum):
    """Confidence level classifications for routing decisions."""
    VERY_HIGH = "very_high"      # 0.90 - 1.00
    HIGH = "high"                # 0.80 - 0.89
    MEDIUM = "medium"            # 0.70 - 0.79
    LOW = "low"                  # 0.50 - 0.69
    VERY_LOW = "very_low"        # 0.00 - 0.49


class AgentType(str, Enum):
    """Available agent types in the system."""
    DOCUMENT_INFORMATION = "document_information"
    PLAYBOOK_REVIEW = "playbook_review"
    GENERAL_REVIEW = "general_review"
    RISK_COMPLIANCE = "risk_compliance"
    VERSION_COMPARISON = "version_comparison"
    STANDARD_CLAUSE = "standard_clause"
    DOC_CHAT = "doc_chat"
    PLAYBOOK_RULES = "playbook_rules"
    INTERACTIVE_AI = "interactive_ai"


class SubIntent(str, Enum):
    """Sub-intents for more granular routing."""
    # Document Information
    SUMMARY = "summary"
    KEY_DETAILS = "key_details"
    ENTITY_EXTRACTION = "entity_extraction"
    STRUCTURE_ANALYSIS = "structure_analysis"
    
    # Playbook Review
    CLAUSE_REVIEW = "clause_review"
    MISSING_CLAUSES = "missing_clauses"
    DEVIATION_ANALYSIS = "deviation_analysis"
    
    # General Review
    QUALITY_ASSESSMENT = "quality_assessment"
    GRAMMAR_CLARITY = "grammar_clarity"
    FORMATTING = "formatting"
    COMPLETENESS = "completeness"
    
    # Risk & Compliance
    RISK_IDENTIFICATION = "risk_identification"
    COMPLIANCE_CHECK = "compliance_check"
    LIABILITY_ASSESSMENT = "liability_assessment"
    REGULATORY_MAPPING = "regulatory_mapping"
    
    # Version Comparison
    DIFF_ANALYSIS = "diff_analysis"
    IMPACT_ASSESSMENT = "impact_assessment"
    CHANGE_SUMMARY = "change_summary"
    
    # Standard Clause
    BENCHMARK_COMPARISON = "benchmark_comparison"
    MARKET_ANALYSIS = "market_analysis"
    GAP_ANALYSIS = "gap_analysis"
    
    # Doc Chat
    DIRECT_ANSWER = "direct_answer"
    EXPLANATION = "explanation"
    GUIDANCE = "guidance"
    
    # Playbook Rules
    RULE_DISPLAY = "rule_display"
    PLAYBOOK_SUMMARY = "playbook_summary"
    RULE_EXPLANATION = "rule_explanation"
    
    # Interactive AI
    DRAFTING = "drafting"
    IMPROVEMENT = "improvement"
    NEGOTIATION_SIM = "negotiation_simulation"
    ALTERNATIVES = "alternatives"


class RequirementType(str, Enum):
    """Types of requirements that may be needed."""
    DOCUMENT = "document"
    PLAYBOOK = "playbook"
    MULTIPLE_DOCUMENTS = "multiple_documents"
    PREVIOUS_VERSION = "previous_version"
    USER_PERMISSION = "user_permission"


class ClassificationResult(BaseModel):
    """Result of intent classification."""
    model_config = ConfigDict(use_enum_values=True)
    
    query: str = Field(description="Original user query")
    primary_agent: AgentType = Field(description="Primary agent for handling")
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence")
    confidence_level: ConfidenceLevel = Field(description="Categorized confidence level")
    sub_intent: Optional[SubIntent] = Field(default=None, description="Specific sub-intent")
    
    # Multi-agent detection
    secondary_agents: List[AgentType] = Field(default_factory=list, description="Additional agents needed")
    requires_chaining: bool = Field(default=False, description="Whether multiple agents must execute in sequence")
    chain_order: List[AgentType] = Field(default_factory=list, description="Execution order if chained")
    
    # Entity extraction
    extracted_entities: Dict[str, Any] = Field(default_factory=dict, description="Entities extracted from query")
    document_references: List[str] = Field(default_factory=list, description="Document IDs or names mentioned")
    section_references: List[str] = Field(default_factory=list, description="Specific sections mentioned")
    
    # Classification metadata
    reasoning: str = Field(description="LLM's reasoning for classification")
    alternative_agents: List[Dict[str, Any]] = Field(default_factory=list, description="Other possible agents with scores")
    ambiguity_flags: List[str] = Field(default_factory=list, description="Potential ambiguities detected")
    
    @field_validator("confidence_level", mode="before")
    @classmethod
    def calculate_confidence_level(cls, v, info) -> ConfidenceLevel:
        """Calculate confidence level from score."""
        confidence = info.data.get("confidence", 0.0)
        if confidence >= 0.90:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.80:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.70:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.50:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


class RequirementCheck(BaseModel):
    """Check for a specific requirement."""
    requirement_type: RequirementType
    required: bool
    available: bool
    details: Optional[str] = None


class RoutingDecision(BaseModel):
    """Final routing decision after validation."""
    model_config = ConfigDict(use_enum_values=True)
    
    classification: ClassificationResult = Field(description="Original classification")
    
    # Validation results
    requirements_met: bool = Field(description="Whether all requirements are satisfied")
    requirement_checks: List[RequirementCheck] = Field(default_factory=list)
    missing_requirements: List[RequirementType] = Field(default_factory=list)
    
    # Routing decision
    target_agent: AgentType = Field(description="Agent to execute")
    execution_mode: str = Field(default="single")  # single, chain, parallel
    fallback_agent: Optional[AgentType] = Field(default=AgentType.DOC_CHAT)
    
    # Clarification
    clarification_needed: bool = Field(default=False)
    clarification_question: Optional[str] = None
    suggested_queries: List[str] = Field(default_factory=list)
    
    # Execution parameters
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for agent execution")
    
    # Decision metadata
    decision_timestamp: datetime = Field(default_factory=datetime.utcnow)
    decision_reasoning: str = Field(description="Why this routing decision was made")


class DocumentReference(BaseModel):
    """Reference to a document in the system."""
    document_id: UUID
    filename: str
    document_type: str
    upload_timestamp: datetime
    extracted_summary: Optional[str] = None


class PlaybookReference(BaseModel):
    """Reference to an organization's playbook."""
    playbook_id: UUID
    organization_id: UUID
    name: str
    version: str
    rules_summary: Optional[str] = None


class ConversationMessage(BaseModel):
    """A message in the conversation history."""
    message_id: UUID = Field(default_factory=uuid4)
    role: str = Field(pattern="^(user|assistant|system)$")
    content: str
    agent_name: Optional[AgentType] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OrchestratorContext(BaseModel):
    """Shared context for agent execution."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Session identification
    session_id: UUID = Field(default_factory=uuid4)
    conversation_id: UUID = Field(default_factory=uuid4)
    user_id: Optional[UUID] = None
    organization_id: Optional[UUID] = None
    
    # Document context
    active_documents: List[DocumentReference] = Field(default_factory=list)
    document_contents: Dict[UUID, str] = Field(default_factory=dict)
    
    # Playbook context
    active_playbook: Optional[PlaybookReference] = None
    playbook_rules: Dict[str, Any] = Field(default_factory=dict)
    
    # Conversation context
    conversation_history: List[ConversationMessage] = Field(default_factory=list)
    max_history_messages: int = Field(default=20)
    
    # Execution context
    current_agent: Optional[AgentType] = None
    agent_chain_depth: int = Field(default=0, ge=0, le=10)
    execution_trace: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Intermediate results (for chaining)
    intermediate_results: Dict[str, Any] = Field(default_factory=dict)
    
    # User preferences
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    
    # Citation tracking
    citations_used: List[Dict[str, Any]] = Field(default_factory=list)
    
    def add_message(self, role: str, content: str, agent_name: Optional[AgentType] = None, metadata: Optional[Dict] = None):
        """Add a message to conversation history."""
        message = ConversationMessage(
            role=role,
            content=content,
            agent_name=agent_name,
            metadata=metadata or {}
        )
        self.conversation_history.append(message)
        
        # Trim history if needed
        if len(self.conversation_history) > self.max_history_messages:
            self.conversation_history = self.conversation_history[-self.max_history_messages:]
    
    def get_formatted_history(self, n: Optional[int] = None) -> str:
        """Get formatted conversation history for prompts."""
        messages = self.conversation_history
        if n:
            messages = messages[-n:]
        
        formatted = []
        for msg in messages:
            prefix = f"[{msg.agent_name.value}]" if msg.agent_name else f"[{msg.role.upper()}]"
            formatted.append(f"{prefix}: {msg.content}")
        
        return "\n".join(formatted)
    
    def add_intermediate_result(self, key: str, value: Any):
        """Store intermediate result for chaining."""
        self.intermediate_results[key] = value
    
    def get_intermediate_result(self, key: str) -> Optional[Any]:
        """Retrieve intermediate result."""
        return self.intermediate_results.get(key)


class AgentResponse(BaseModel):
    """Standard response from any agent."""
    content: str
    agent_name: str
    sub_intent: Optional[str] = None
    tokens_used: Dict[str, int] = Field(default_factory=dict)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    referenced_documents: List[UUID] = Field(default_factory=list)
    referenced_sections: List[str] = Field(default_factory=list)
    confidence_score: float = 0.0
    processing_time_ms: int = 0
    follow_up_questions: List[str] = Field(default_factory=list)
    related_actions: List[Dict[str, str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OrchestratorResponse(BaseModel):
    """Final response from the orchestrator."""
    model_config = ConfigDict(use_enum_values=True)
    
    # Response content
    content: str = Field(description="Main response content")
    
    # Execution metadata
    agent_used: AgentType = Field(description="Primary agent that generated response")
    confidence_score: float = Field(ge=0.0, le=1.0)
    processing_time_ms: int = Field(description="Total processing time in milliseconds")
    tokens_used: Dict[str, int] = Field(default_factory=dict, description="Token usage by component")
    
    # Citations and references
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="Document citations")
    referenced_documents: List[UUID] = Field(default_factory=list)
    referenced_sections: List[str] = Field(default_factory=list)
    
    # Execution details
    execution_trace: List[Dict[str, Any]] = Field(default_factory=list, description="Step-by-step execution")
    was_chained: bool = Field(default=False, description="Whether multiple agents were involved")
    chain_details: Optional[List[Dict[str, Any]]] = None
    
    # Suggestions
    follow_up_questions: List[str] = Field(default_factory=list)
    related_actions: List[Dict[str, str]] = Field(default_factory=list)
    
    # Response metadata
    response_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Error handling
    partial_response: bool = Field(default=False, description="Whether response is incomplete due to error")
    error_info: Optional[Dict[str, Any]] = None
