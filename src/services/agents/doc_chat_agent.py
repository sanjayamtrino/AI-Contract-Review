"""
Document Chat Agent

Responsible for general Q&A and conversational interaction with the document.
"""
from typing import Any, Dict, Optional

from src.services.agents.base_agent import BaseAgent, AgentResult
from src.services.orchestrator.models import OrchestratorContext, SubIntent, AgentType

class DocChatAgent(BaseAgent):
    """Agent for general document chat."""
    
    @property
    def agent_type(self) -> str:
        return AgentType.DOC_CHAT.value
    
    @property
    def system_prompt(self) -> str:
        return """You are a helpful assistant for reviewing contracts.
        Answer the user's questions based strictly on the provided document content.
        If the answer is not in the document, state that clearly.
        """
    
    async def process(
        self,
        query: str,
        sub_intent: Optional[SubIntent],
        parameters: Dict[str, Any],
        context: OrchestratorContext,
    ) -> AgentResult:
        """Process chat requests."""
        
        prompt = query
        
        # Add conversation history
        history = context.get_formatted_history(n=5)
        if history:
            prompt = f"Conversation History:\n{history}\n\nCurrent Query: {query}"
            
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
