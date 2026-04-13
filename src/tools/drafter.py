"""
Drafter tool -- business logic for the Describe & Draft agent.

Extracts optional document context and similar-clause notes, then calls
the LLM with a Mustache prompt to generate 3 draft alternatives.
"""

import logging
from typing import Any, Optional

import numpy as np

from src.dependencies import get_service_container
from src.schemas.draft import DraftLLMResponse, DraftResponse
from src.services.clause_extractor import ClauseUnit, extract_all_clauses
from src.services.prompts.v1 import load_prompt

logger = logging.getLogger(__name__)

SIMILAR_CLAUSE_THRESHOLD = 0.60


def _extract_document_context(session: Any) -> Optional[dict]:
    """Gather document context from the first document in the session.

    Returns a dict with ``document_excerpt`` or None if unavailable.
    Best-effort: never raises.
    """
    try:
        documents = getattr(session, "documents", None) or {}
        if not documents:
            return None

        first_doc_id = next(iter(documents))
        doc_info = documents[first_doc_id]
        chunk_indices = doc_info.get("chunk_indices", [])
        if not chunk_indices:
            return None

        chunk_store = getattr(session, "chunk_store", None) or {}
        texts = []
        for idx in chunk_indices[:5]:
            chunk = chunk_store.get(idx)
            if chunk is None:
                continue
            content = (getattr(chunk, "content", None) or "").strip()
            if content:
                texts.append(content)

        if not texts:
            return None

        excerpt = "\n\n".join(texts)[:4000]
        return {"document_excerpt": excerpt}
    except Exception:
        logger.warning("Failed to extract document context", exc_info=True)
        return None


async def _find_similar_clause(session: Any, user_prompt: str) -> Optional[str]:
    """Check if the document already contains a clause similar to the request.

    Returns a human-readable note string if similarity >= threshold,
    otherwise None. Never raises -- failures are logged and swallowed.
    """
    try:
        clauses = extract_all_clauses(session)
        if not clauses:
            return None

        container = get_service_container()
        embedding_service = container.embedding_service

        prompt_embedding = await embedding_service.generate_embeddings(user_prompt)
        if not prompt_embedding:
            return None

        # Ensure each clause has an embedding
        for clause in clauses:
            if not clause.embedding or len(clause.embedding) == 0:
                clause.embedding = await embedding_service.generate_embeddings(
                    clause.content
                )

        # Compute cosine similarity (following compare.py pattern)
        query = np.array(prompt_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        query = query / max(query_norm, 1e-10)

        best_score = 0.0
        best_clause: Optional[ClauseUnit] = None

        for clause in clauses:
            if not clause.embedding:
                continue
            vec = np.array(clause.embedding, dtype=np.float32)
            vec_norm = np.linalg.norm(vec)
            vec = vec / max(vec_norm, 1e-10)
            score = float(np.dot(query, vec))
            if score > best_score:
                best_score = score
                best_clause = clause

        if best_score >= SIMILAR_CLAUSE_THRESHOLD and best_clause is not None:
            heading = best_clause.heading or f"Clause at position {best_clause.doc_order + 1}"
            return (
                f'A similar clause already exists in the document: "{heading}" '
                f"(similarity: {best_score:.0%}). The drafts below are provided regardless."
            )

        return None
    except Exception:
        logger.warning("Similar clause detection failed", exc_info=True)
        return None


async def generate_drafts(session_id: str, user_prompt: str) -> DraftResponse:
    """Generate 3 draft alternatives from a user description.

    Handles: no session, no document, document with context, similar clause
    found/not found, and LLM failure.
    """
    container = get_service_container()
    session = container.session_manager.get_session(session_id)

    if session is None:
        return DraftResponse(
            session_id=session_id, status="error", error="Session not found"
        )

    # Optional enrichment
    doc_context = _extract_document_context(session)
    similar_clause_note = await _find_similar_clause(session, user_prompt)

    # Build prompt context
    context = {
        "user_prompt": user_prompt,
        "has_document_context": doc_context is not None,
        "has_similar_clause": similar_clause_note is not None,
    }
    if doc_context:
        context["document_excerpt"] = doc_context["document_excerpt"]
    if similar_clause_note:
        context["similar_clause_note"] = similar_clause_note

    # Load prompt and call LLM
    prompt = load_prompt("describe_draft_prompt")
    llm_client = container.azure_openai_model

    try:
        llm_response: DraftLLMResponse = await llm_client.generate(
            prompt=prompt,
            context=context,
            response_model=DraftLLMResponse,
            temperature=0.7,
        )

        # Safety check (Pydantic min_length/max_length should enforce this)
        if len(llm_response.drafts) != 3:
            return DraftResponse(
                session_id=session_id,
                status="error",
                error=f"Expected 3 drafts, got {len(llm_response.drafts)}",
            )

        return DraftResponse(
            session_id=session_id,
            status="ok",
            summary=llm_response.summary,
            note=similar_clause_note,
            drafts=llm_response.drafts,
        )
    except Exception as e:
        logger.error("Draft generation failed: %s", e, exc_info=True)
        return DraftResponse(
            session_id=session_id, status="error", error=str(e)
        )
