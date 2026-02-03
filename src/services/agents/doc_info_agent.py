"""
Document Information Agent

Responsible for extracting summaries, key details, and structured information from documents.
"""
from typing import Any, Dict, Optional

from src.services.agents.base_agent import BaseAgent, AgentResult
from src.services.orchestrator.models import OrchestratorContext, SubIntent, AgentType

class DocumentInformationAgent(BaseAgent):
    """Agent for extracting document information."""
    
    @property
    def agent_type(self) -> str:
        return AgentType.DOCUMENT_INFORMATION.value
    
    @property
    def system_prompt(self) -> str:
        return """You are an expert legal document analyst. 
        Your goal is to extract key information, summarize content, and identify the structure of legal documents.
        Focus on accuracy, clarity, and capturing essential details like parties, dates, and obligations.
        """
    
    async def process(
        self,
        query: str,
        sub_intent: Optional[SubIntent],
        parameters: Dict[str, Any],
        context: OrchestratorContext,
    ) -> AgentResult:
        """Process document information requests."""
        
        # Determine the specific task based on sub-intent
        prompt = query
        if sub_intent == SubIntent.SUMMARY:
            prompt = f"Please provide a comprehensive summary of this document. Query: {query}"
        elif sub_intent == SubIntent.KEY_DETAILS:
            prompt = f"Extract the key details (Parties, Effective Date, term, etc.) from this document. Query: {query}"
        
        # Add context from active documents
        if context.active_documents:
            # In a real implementation, we would fetch the document content here
            # For now, we assume the content might be passed or we use a placeholder
            doc_content = context.document_contents.get(context.active_documents[0].document_id, "")
            if doc_content:
                prompt += f"\n\nDocument Content:\n{doc_content[:10000]}" # Limit context for now
        
        # Call LLM
        response_data = await self._call_llm(prompt)
        
        return AgentResult(
            content=response_data["content"],
            agent_name=self.agent_type,
            sub_intent=sub_intent,
            tokens_used=response_data["tokens_used"],
            confidence_score=0.9, # Placeholder
            processing_time_ms=0
        )
