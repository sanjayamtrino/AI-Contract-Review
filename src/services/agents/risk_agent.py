"""
Risk & Compliance Agent

Responsible for identifying risks, compliance issues, and liability assessments in contracts.
"""
from typing import Any, Dict, Optional

from src.services.agents.base_agent import BaseAgent, AgentResult
from src.services.orchestrator.models import OrchestratorContext, SubIntent, AgentType

class RiskComplianceAgent(BaseAgent):
    """Agent for risk and compliance analysis."""
    
    @property
    def agent_type(self) -> str:
        return AgentType.RISK_COMPLIANCE.value
    
    @property
    def system_prompt(self) -> str:
        return """You are a senior risk compliance officer.
        Your goal is to identify potential risks, non-compliance with standard regulations, and unfavourable liability clauses in contracts.
        Highlight high-risk areas and suggest mitigations where possible.
        """
    
    async def process(
        self,
        query: str,
        sub_intent: Optional[SubIntent],
        parameters: Dict[str, Any],
        context: OrchestratorContext,
    ) -> AgentResult:
        """Process risk and compliance requests."""
        
         # Determine the specific task based on sub-intent
        prompt = query
        if sub_intent == SubIntent.RISK_IDENTIFICATION:
            prompt = f"Identify the top risks in this contract. Query: {query}"
        
        # Add context from active documents
        if context.active_documents:
             doc_content = context.document_contents.get(context.active_documents[0].document_id, "")
             if doc_content:
                prompt += f"\n\nDocument Content:\n{doc_content[:10000]}"

        # Call LLM
        response_data = await self._call_llm(prompt)
        
        return AgentResult(
            content=response_data["content"],
            agent_name=self.agent_type,
            sub_intent=sub_intent,
            tokens_used=response_data["tokens_used"],
            confidence_score=0.9,
             processing_time_ms=0
        )
