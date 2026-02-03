"""
Intent Classification using GenAI (Gemini)

Uses Google's Gemini for sophisticated intent classification with few-shot examples
and confidence calibration.
"""

import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from src.services.llm.base_model import BaseLLMModel
from src.config.logging import get_logger
from src.services.orchestrator.models import (
    AgentType,
    ClassificationResult,
    ConfidenceLevel,
    OrchestratorContext,
    SubIntent,
)
from src.services.orchestrator.exceptions import ClassificationError

logger = get_logger(__name__)


@dataclass
class ClassificationExample:
    """Few-shot example for intent classification."""
    query: str
    classification: Dict[str, Any]
    reasoning: str


class IntentClassifier:
    """
    Advanced intent classifier using GenAI (Gemini).
    
    Features:
    - Few-shot learning with curated examples
    - Chain-of-thought reasoning
    - Confidence calibration
    - Multi-label classification support
    - Ambiguity detection
    """
    
    # Classification prompt template
    SYSTEM_PROMPT = """You are the Central Intelligence Classification Engine for a Legal Document AI System.
Your role is to analyze user queries with extreme precision and classify them into one of nine specialized agent categories.

=== AGENT DEFINITIONS ===

[AGENT_01_DOCUMENT_INFORMATION]
Purpose: Extract, summarize, and present information from legal documents
Triggers: "summarize", "what does it say", "key points", "extract", "find mentions of", "what are the main terms", "give me an overview"
Sub-tasks:
- summary: General document overview
- key_details: Specific fact extraction (dates, amounts, parties, clauses)
- entity_extraction: Identify people, companies, locations
- structure_analysis: Document organization and sections

[AGENT_02_PLAYBOOK_REVIEW]
Purpose: Check document against organization's custom playbook/rules
Triggers: "check against our playbook", "does this meet our standards", "playbook compliance", "our company policy", "internal requirements"
Requirements: Requires active playbook loaded for the organization
Sub-tasks:
- clause_review: Check specific clauses against rules
- missing_clauses: Identify what's missing per playbook
- deviation_analysis: Find differences from standards

[AGENT_03_GENERAL_REVIEW]
Purpose: Overall document quality assessment
Triggers: "review this document", "general feedback", "overall assessment", "quality check", "how good is this"
Sub-tasks:
- quality_assessment: Overall document quality score
- grammar_clarity: Language quality
- formatting: Structure and presentation
- completeness: Is the document complete?
- coherence: Does it make sense?

[AGENT_04_RISK_COMPLIANCE]
Purpose: Identify legal and compliance risks
Triggers: "risks", "compliance", "regulatory issues", "legal exposure", "problems", "concerns", "red flags", "GDPR", "HIPAA", "violation", "liability"
Sub-tasks:
- risk_identification: Find potential risks
- compliance_check: Regulatory compliance
- liability_assessment: Legal exposure analysis
- regulatory_mapping: Map to specific regulations

[AGENT_05_VERSION_COMPARISON]
Purpose: Compare document versions
Triggers: "what changed", "compare with previous", "differences from last version", "version comparison", "what's new"
Requirements: Requires 2+ document versions uploaded
Sub-tasks:
- diff_analysis: Line-by-line changes
- impact_assessment: Significance of changes
- change_summary: High-level comparison

[AGENT_06_STANDARD_CLAUSE]
Purpose: Compare against industry/market standards
Triggers: "standard clause", "industry benchmark", "market standard", "typical language", "common practice", "how does this compare to market"
Sub-tasks:
- benchmark_comparison: Compare to standards
- market_analysis: Industry norms
- gap_analysis: What's different from standard

[AGENT_07_DOC_CHAT]
Purpose: Conversational Q&A about document content
Triggers: "what is", "explain", "help me understand", "question about", "what does section mean", "clarify", "I'm confused about"
Sub-tasks:
- direct_answer: Answer specific questions
- explanation: Explain complex sections
- guidance: Help navigate document

[AGENT_08_PLAYBOOK_RULES]
Purpose: Show/explain the playbook itself (not checking documents)
Triggers: "show me the playbook", "what are our rules", "playbook standards", "our company policy", "internal guidelines"
Sub-tasks:
- rule_display: Show specific rules
- playbook_summary: Overview of standards
- rule_explanation: Explain why rules exist

[AGENT_09_INTERACTIVE_AI]
Purpose: Collaborative editing and drafting assistance
Triggers: "help me draft", "suggest improvements", "collaborate", "interactive editing", "help me write", "revise this", "make it better"
Sub-tasks:
- drafting: Help write content
- improvement: Enhance existing text
- negotiation_simulation: Practice negotiations
- alternative_language: Provide different wording options

=== CLASSIFICATION RULES ===

1. Confidence Scoring:
   - 0.90-1.00: VERY_HIGH - direct routing
   - 0.80-0.89: HIGH - direct routing with monitoring
   - 0.70-0.79: MEDIUM - proceed with caution
   - 0.50-0.69: LOW - request clarification
   - <0.50: VERY_LOW - strong clarification needed

2. Multi-Agent Detection:
   - If query contains "and" or "then" connecting different tasks → requires_chaining = true
   - Example: "Summarize and check risks" → Chain [Document_Info, Risk_Compliance]

3. Context Requirements:
   - Any comparison agent needs multiple documents
   - Playbook agents need active playbook
   - If requirements missing → return error_code

4. Ambiguity Resolution:
   - "Review" alone → Ask: "General review or against playbook?"
   - "Check" alone → Ask: "Check for risks or playbook compliance?"

=== OUTPUT FORMAT ===

Respond with valid JSON:
{
  "primary_agent": "agent_type",
  "confidence": 0.95,
  "confidence_level": "very_high",
  "sub_intent": "specific_task",
  "secondary_agents": [],
  "requires_chaining": false,
  "chain_order": [],
  "extracted_entities": {},
  "document_references": [],
  "section_references": [],
  "reasoning": "explanation",
  "alternative_agents": [],
  "ambiguity_flags": []
}"""
    
    def __init__(self, llm_model: BaseLLMModel):
        self.llm = llm_model
        self._examples = self._load_examples()
    
    async def classify(
        self,
        query: str,
        context: OrchestratorContext,
    ) -> ClassificationResult:
        """
        Classify user query intent.
        
        Args:
            query: User's natural language query
            context: Current orchestrator context
            
        Returns:
            ClassificationResult with agent routing information
        """
        logger.info("Classifying intent", query=query[:100])
        
        try:
            # Build classification prompt with context
            prompt = self._build_classification_prompt(query, context)
            
            # Call LLM
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=2000,
            )
            
            # Parse and validate result
            classification = self._parse_classification_result(query, response)
            
            # Apply confidence calibration
            classification = self._calibrate_confidence(classification)
            
            # Detect potential ambiguities
            classification = self._detect_ambiguities(classification)
            
            logger.info(
                "Classification complete",
                agent=classification.primary_agent.value,
                confidence=classification.confidence,
            )
            
            return classification
            
        except Exception as e:
            logger.error("Classification failed", error=str(e))
            return self._create_fallback_classification(query, str(e))
    
    def _build_classification_prompt(
        self,
        query: str,
        context: OrchestratorContext,
    ) -> str:
        """Build the complete classification prompt."""
        
        prompt_parts = [
            "## CLASSIFICATION TASK",
            "",
            f"**User Query:** {query}",
            "",
            "## CONTEXT INFORMATION",
        ]
        
        # Add active documents context
        if context.active_documents:
            prompt_parts.append(f"**Active Documents:** {len(context.active_documents)}")
            for doc in context.active_documents:
                prompt_parts.append(f"  - {doc.filename} (Type: {doc.document_type})")
        else:
            prompt_parts.append("**Active Documents:** None")
        
        # Add playbook context
        if context.active_playbook:
            prompt_parts.append(f"**Active Playbook:** {context.active_playbook.name}")
        else:
            prompt_parts.append("**Active Playbook:** None")
        
        # Add conversation history context
        if context.conversation_history:
            prompt_parts.append("**Recent Conversation:**")
            recent = context.conversation_history[-3:]
            for msg in recent:
                agent_info = f" [{msg.agent_name.value}]" if msg.agent_name else ""
                prompt_parts.append(f"  {msg.role}{agent_info}: {msg.content[:100]}...")
        
        # Add few-shot examples
        prompt_parts.extend([
            "",
            "## CLASSIFICATION EXAMPLES",
            "",
        ])
        
        for i, example in enumerate(self._examples[:5], 1):
            prompt_parts.extend([
                f"### Example {i}",
                f"**Query:** {example.query}",
                f"**Reasoning:** {example.reasoning}",
                f"**Classification:** {json.dumps(example.classification, indent=2)}",
                "",
            ])
        
        prompt_parts.extend([
            "## YOUR CLASSIFICATION",
            "",
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_classification_result(
        self,
        query: str,
        result: str,
    ) -> ClassificationResult:
        """Parse and validate classification result."""
        try:
            # Clean up result
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()
            
            data = json.loads(result)
            
            # Validate agent type
            primary_agent = AgentType(data["primary_agent"])
            
            # Parse secondary agents
            secondary_agents = [
                AgentType(a) for a in data.get("secondary_agents", [])
            ]
            
            # Parse chain order
            chain_order = [
                AgentType(a) for a in data.get("chain_order", [])
            ]
            
            # Parse sub-intent
            sub_intent = None
            if data.get("sub_intent"):
                try:
                    sub_intent = SubIntent(data["sub_intent"])
                except ValueError:
                    logger.warning(f"Unknown sub-intent: {data['sub_intent']}")
            
            return ClassificationResult(
                query=query,
                primary_agent=primary_agent,
                confidence=float(data.get("confidence", 0.5)),
                confidence_level=ConfidenceLevel(data.get("confidence_level", "low")),
                sub_intent=sub_intent,
                secondary_agents=secondary_agents,
                requires_chaining=data.get("requires_chaining", False),
                chain_order=chain_order,
                extracted_entities=data.get("extracted_entities", {}),
                document_references=data.get("document_references", []),
                section_references=data.get("section_references", []),
                reasoning=data.get("reasoning", ""),
                alternative_agents=data.get("alternative_agents", []),
                ambiguity_flags=data.get("ambiguity_flags", []),
            )
            
        except json.JSONDecodeError as e:
            logger.error("Failed to parse classification result", result=result[:200])
            raise ClassificationError(f"Invalid JSON: {e}")
        except (KeyError, ValueError) as e:
            logger.error("Invalid classification structure", error=str(e))
            raise ClassificationError(f"Invalid structure: {e}")
    
    def _calibrate_confidence(self, classification: ClassificationResult) -> ClassificationResult:
        """Apply confidence calibration to avoid overconfidence."""
        raw_confidence = classification.confidence
        
        # Apply calibration curve
        if raw_confidence > 0.95:
            calibrated = 0.90 + (raw_confidence - 0.95) * 0.5
        elif raw_confidence > 0.80:
            calibrated = raw_confidence * 0.95
        else:
            calibrated = raw_confidence
        
        classification.confidence = round(calibrated, 3)
        classification.confidence_level = self._confidence_to_level(calibrated)
        
        return classification
    
    def _detect_ambiguities(self, classification: ClassificationResult) -> ClassificationResult:
        """Detect potential ambiguities in classification."""
        ambiguities = list(classification.ambiguity_flags)
        query_lower = classification.query.lower()
        
        # Check for ambiguous terms
        ambiguous_terms = {
            "review": [AgentType.GENERAL_REVIEW, AgentType.PLAYBOOK_REVIEW],
            "check": [AgentType.RISK_COMPLIANCE, AgentType.PLAYBOOK_REVIEW],
            "analyze": [AgentType.DOCUMENT_INFORMATION, AgentType.RISK_COMPLIANCE],
            "compare": [AgentType.VERSION_COMPARISON, AgentType.STANDARD_CLAUSE],
        }
        
        for term, possible_agents in ambiguous_terms.items():
            if term in query_lower and classification.primary_agent in possible_agents:
                if len(possible_agents) > 1:
                    ambiguities.append(
                        f"Query contains '{term}' which could indicate "
                        f"{', '.join(a.value for a in possible_agents)}"
                    )
        
        # Check for low confidence with multiple alternatives
        if classification.confidence < 0.70 and len(classification.alternative_agents) > 1:
            ambiguities.append("Low confidence with multiple viable alternatives")
        
        classification.ambiguity_flags = ambiguities
        return classification
    
    def _create_fallback_classification(
        self,
        query: str,
        error_message: str,
    ) -> ClassificationResult:
        """Create safe fallback classification on error."""
        return ClassificationResult(
            query=query,
            primary_agent=AgentType.DOC_CHAT,
            confidence=0.5,
            confidence_level=ConfidenceLevel.LOW,
            sub_intent=SubIntent.DIRECT_ANSWER,
            reasoning=f"Classification failed: {error_message}. Falling back to general chat.",
            ambiguity_flags=["classification_error"],
        )
    
    def _confidence_to_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to level."""
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
    
    def _load_examples(self) -> List[ClassificationExample]:
        """Load few-shot classification examples."""
        return [
            ClassificationExample(
                query="Summarize this contract and tell me the key terms",
                classification={
                    "primary_agent": "document_information",
                    "confidence": 0.96,
                    "confidence_level": "very_high",
                    "sub_intent": "summary",
                    "requires_chaining": False,
                },
                reasoning="Query explicitly asks for document summary",
            ),
            ClassificationExample(
                query="Does this agreement meet our company's legal standards?",
                classification={
                    "primary_agent": "playbook_review",
                    "confidence": 0.92,
                    "confidence_level": "very_high",
                    "sub_intent": "clause_review",
                    "requires_chaining": False,
                },
                reasoning="Query asks about compliance with organizational standards",
            ),
            ClassificationExample(
                query="What are the risks and compliance issues in this document?",
                classification={
                    "primary_agent": "risk_compliance",
                    "confidence": 0.94,
                    "confidence_level": "very_high",
                    "sub_intent": "risk_identification",
                    "requires_chaining": False,
                },
                reasoning="Query explicitly asks about risks and compliance",
            ),
            ClassificationExample(
                query="Summarize this contract and tell me if there are any risks",
                classification={
                    "primary_agent": "document_information",
                    "confidence": 0.88,
                    "confidence_level": "high",
                    "sub_intent": "summary",
                    "secondary_agents": ["risk_compliance"],
                    "requires_chaining": True,
                    "chain_order": ["document_information", "risk_compliance"],
                },
                reasoning="Query contains two distinct tasks requiring chaining",
            ),
            ClassificationExample(
                query="Review this",
                classification={
                    "primary_agent": "general_review",
                    "confidence": 0.55,
                    "confidence_level": "medium",
                    "sub_intent": "quality_assessment",
                    "ambiguity_flags": ["Generic 'review' could indicate multiple agent types"],
                },
                reasoning="Query is ambiguous - 'review' could mean multiple things",
            ),
        ]
