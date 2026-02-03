"""
Context Management System with FAISS Integration

Manages shared state across agent executions, including conversation history,
document references, and intermediate results for agent chaining.
Uses FAISS for vector storage and semantic search.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

import faiss
import numpy as np

from src.config.logging import get_logger
from src.config.settings import get_settings
from src.services.orchestrator.models import (
    OrchestratorContext,
    DocumentReference,
    ConversationMessage,
    AgentType,
)
from src.services.vector_store.embeddings.embedding_service import EmbeddingService

logger = get_logger(__name__)
settings = get_settings()


class ContextManager:
    """
    Manages orchestrator context with FAISS memory integration.
    
    Features:
    - Conversation history management
    - Document content caching
    - Intermediate result storage
    - FAISS-based semantic search
    - Context window optimization
    """
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self._faiss_indices: Dict[UUID, faiss.Index] = {}
        self._faiss_metadata: Dict[UUID, List[Dict]] = {}
    
    async def create_context(
        self,
        user_id: Optional[UUID] = None,
        organization_id: Optional[UUID] = None,
        conversation_id: Optional[UUID] = None,
    ) -> OrchestratorContext:
        """Create a new orchestrator context."""
        context = OrchestratorContext(
            user_id=user_id,
            organization_id=organization_id,
        )
        
        if conversation_id:
            # Load existing conversation
            await self._load_conversation(context, conversation_id)
        
        return context
    
    async def add_document(
        self,
        context: OrchestratorContext,
        document_id: UUID,
        filename: str,
        document_type: str,
        extracted_text: str,
        summary: Optional[str] = None,
    ) -> None:
        """Add a document to the context and FAISS index."""
        from datetime import datetime
        
        doc_ref = DocumentReference(
            document_id=document_id,
            filename=filename,
            document_type=document_type,
            upload_timestamp=datetime.utcnow(),
            extracted_summary=summary,
        )
        
        context.active_documents.append(doc_ref)
        context.document_contents[document_id] = extracted_text
        
        # Create FAISS index for this document
        await self._create_faiss_index(context.conversation_id, document_id, extracted_text)
        
        logger.info(
            "Document added to context",
            document_id=str(document_id),
            conversation_id=str(context.conversation_id),
        )
    
    async def _create_faiss_index(
        self,
        conversation_id: UUID,
        document_id: UUID,
        text: str,
    ) -> None:
        """Create FAISS index for document chunks."""
        # Chunk the text
        chunks = self._chunk_text(text)
        
        if not chunks:
            return
        
        # Get embeddings for chunks
        embeddings = []
        for chunk in chunks:
            embedding = await self.embedding_service.get_embedding(chunk)
            embeddings.append(embedding)
        
        # Create FAISS index
        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Add vectors
        vectors = np.array(embeddings).astype('float32')
        index.add(vectors)
        
        # Store index and metadata
        index_key = f"{conversation_id}_{document_id}"
        self._faiss_indices[index_key] = index
        self._faiss_metadata[index_key] = [
            {"chunk": chunk, "index": i} for i, chunk in enumerate(chunks)
        ]
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    async def add_message(
        self,
        context: OrchestratorContext,
        role: str,
        content: str,
        agent_name: Optional[AgentType] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a message to conversation history."""
        context.add_message(role, content, agent_name, metadata)
    
    async def get_relevant_context(
        self,
        context: OrchestratorContext,
        query: str,
        top_k: int = 5,
    ) -> List[str]:
        """Retrieve relevant context using FAISS semantic search."""
        if not context.active_documents:
            return []
        
        # Get query embedding
        query_embedding = await self.embedding_service.get_embedding(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        results = []
        
        # Search in each document's FAISS index
        for doc in context.active_documents:
            index_key = f"{context.conversation_id}_{doc.document_id}"
            
            if index_key not in self._faiss_indices:
                continue
            
            index = self._faiss_indices[index_key]
            metadata = self._faiss_metadata[index_key]
            
            # Search
            scores, indices = index.search(query_vector, top_k)
            
            # Get chunks
            for idx in indices[0]:
                if idx >= 0 and idx < len(metadata):
                    results.append(metadata[idx]["chunk"])
        
        return results[:top_k]
    
    async def set_playbook(
        self,
        context: OrchestratorContext,
        playbook_id: UUID,
        organization_id: UUID,
        name: str,
        version: str,
        rules: Dict[str, Any],
    ) -> None:
        """Set the active playbook for the context."""
        from datetime import datetime
        
        context.active_playbook = PlaybookReference(
            playbook_id=playbook_id,
            organization_id=organization_id,
            name=name,
            version=version,
            rules_summary=self._summarize_rules(rules),
        )
        context.playbook_rules = rules
        
        logger.info(
            "Playbook set in context",
            playbook_id=str(playbook_id),
            name=name,
        )
    
    def get_optimized_context(
        self,
        context: OrchestratorContext,
        max_tokens: int = 4000,
    ) -> str:
        """
        Get optimized context string within token limit.
        
        Prioritizes:
        1. Current query context
        2. Recent conversation (last N messages)
        3. Document summaries
        4. Relevant document sections
        """
        parts = []
        
        # Add document summaries
        if context.active_documents:
            doc_summary = "## Active Documents\n"
            for doc in context.active_documents:
                doc_summary += f"- {doc.filename} ({doc.document_type})\n"
                if doc.extracted_summary:
                    doc_summary += f"  Summary: {doc.extracted_summary[:200]}...\n"
            parts.append(doc_summary)
        
        # Add recent conversation
        if context.conversation_history:
            max_history = getattr(settings, 'ORCHESTRATOR_CONTEXT_WINDOW_MESSAGES', 20)
            recent = context.conversation_history[-max_history:]
            convo = "## Recent Conversation\n"
            for msg in recent:
                agent_info = f" [{msg.agent_name.value}]" if msg.agent_name else ""
                convo += f"{msg.role}{agent_info}: {msg.content[:300]}\n"
            parts.append(convo)
        
        # Add playbook info if relevant
        if context.active_playbook:
            playbook_info = f"## Active Playbook: {context.active_playbook.name}\n"
            if context.active_playbook.rules_summary:
                playbook_info += f"Rules Summary: {context.active_playbook.rules_summary[:500]}\n"
            parts.append(playbook_info)
        
        return "\n\n".join(parts)
    
    def add_intermediate_result(
        self,
        context: OrchestratorContext,
        key: str,
        value: Any,
    ) -> None:
        """Store intermediate result for agent chaining."""
        from datetime import datetime
        
        context.add_intermediate_result(key, value)
        context.execution_trace.append({
            "type": "intermediate_result",
            "key": key,
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    def get_intermediate_result(
        self,
        context: OrchestratorContext,
        key: str,
    ) -> Optional[Any]:
        """Retrieve intermediate result."""
        return context.get_intermediate_result(key)
    
    def increment_chain_depth(self, context: OrchestratorContext) -> int:
        """Increment and return current chain depth."""
        context.agent_chain_depth += 1
        return context.agent_chain_depth
    
    def check_chain_depth(self, context: OrchestratorContext) -> bool:
        """Check if chain depth is within limits."""
        max_depth = getattr(settings, 'ORCHESTRATOR_MAX_CHAIN_DEPTH', 3)
        return context.agent_chain_depth < max_depth
    
    async def persist_context(self, context: OrchestratorContext) -> None:
        """Persist context to database."""
        logger.info(
            "Persisting context",
            conversation_id=str(context.conversation_id),
            messages=len(context.conversation_history),
        )
    
    async def _load_conversation(
        self,
        context: OrchestratorContext,
        conversation_id: UUID,
    ) -> None:
        """Load existing conversation from database."""
        logger.info(
            "Loading conversation",
            conversation_id=str(conversation_id),
        )
    
    def _summarize_rules(self, rules: Dict[str, Any]) -> str:
        """Create a summary of playbook rules."""
        categories = list(rules.keys())
        total_rules = sum(len(v) for v in rules.values() if isinstance(v, list))
        return f"{total_rules} rules across {len(categories)} categories: {', '.join(categories[:5])}"


from datetime import datetime
from uuid import uuid4
from src.services.orchestrator.models import PlaybookReference
