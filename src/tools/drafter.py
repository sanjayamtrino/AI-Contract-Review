"""
Drafter tool -- business logic for the Describe & Draft agent.

Enriches draft requests with (1) structured document metadata (parties,
governing law, contract type), (2) semantically relevant sections from
the loaded document, and (3) a similar-clause note, then calls the LLM
with a Mustache prompt to generate 3 draft alternatives.
"""

import logging
from typing import Any, List, Optional

import numpy as np

from src.dependencies import get_service_container
from src.schemas.draft import DraftLLMResponse, DraftResponse
from src.services.clause_extractor import ClauseUnit, extract_all_clauses
from src.services.prompts.v1 import load_prompt
from src.tools.key_details import get_key_details

logger = logging.getLogger(__name__)

SIMILAR_CLAUSE_THRESHOLD = 0.60
RELEVANT_CHUNKS_TOP_K = 5


async def _extract_document_metadata(session_id: str) -> Optional[dict]:
    """Extract structured document metadata (parties, governing law, type).

    Returns a dict ready for the prompt or None if unavailable.
    Best-effort: never raises.
    """
    try:
        details = await get_key_details(session_id=session_id)
    except Exception:
        logger.warning("Failed to extract document metadata", exc_info=True)
        return None

    parties_lines: List[str] = []
    for party in (details.parties or []):
        name = party.name or "(unnamed)"
        role = party.role_description or party.role or ""
        parties_lines.append(f"- {name} ({role})".strip())

    doc_type = ""
    try:
        doc_type = details.extraction_metadata.document_type_detected or ""
    except Exception:
        pass

    duration_text = ""
    try:
        if details.duration and details.duration.raw_text:
            duration_text = details.duration.raw_text
    except Exception:
        pass

    effective_date = ""
    try:
        if details.effective_date and details.effective_date.raw_text:
            effective_date = details.effective_date.raw_text
    except Exception:
        pass

    metadata = {
        "document_type": doc_type,
        "parties": "\n".join(parties_lines) if parties_lines else "",
        "duration": duration_text,
        "effective_date": effective_date,
    }

    if not any(metadata.values()):
        return None
    return metadata


async def _get_relevant_sections(
    session: Any, user_prompt: str
) -> Optional[str]:
    """Retrieve semantically relevant chunks from the document for style matching.

    Returns a joined string of top-k relevant sections or None.
    Best-effort: never raises.
    """
    try:
        container = get_service_container()
        retrieval_service = container.retrieval_service

        result = await retrieval_service.retrieve_data(
            query=user_prompt,
            top_k=RELEVANT_CHUNKS_TOP_K,
            dynamic_k=False,
            session_data=session,
        )

        chunks = result.get("chunks") or []
        if not chunks:
            return None

        texts = []
        for chunk in chunks:
            content = (chunk.get("content") or "").strip()
            if content:
                texts.append(content)

        if not texts:
            return None

        return "\n\n---\n\n".join(texts)[:6000]
    except Exception:
        logger.warning("Failed to retrieve relevant sections", exc_info=True)
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

        for clause in clauses:
            if not clause.embedding or len(clause.embedding) == 0:
                clause.embedding = await embedding_service.generate_embeddings(
                    clause.content
                )

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


async def generate_drafts(session_id: Optional[str], user_prompt: str) -> DraftResponse:
    """Generate 3 draft alternatives from a user description.

    When a session with a document is available, enriches the prompt with:
      * structured document metadata (parties, contract type, duration)
      * semantically relevant existing sections (for tone/style matching)
      * a similar-clause note (if a close match already exists)

    Without a session or document, falls back to generic placeholder drafting.
    """
    container = get_service_container()

    metadata: Optional[dict] = None
    relevant_sections: Optional[str] = None
    similar_clause_note: Optional[str] = None

    if session_id:
        session = container.session_manager.get_session(session_id)
        if session is None:
            return DraftResponse(
                session_id=session_id, status="error", error="Session not found"
            )

        metadata = await _extract_document_metadata(session_id)
        relevant_sections = await _get_relevant_sections(session, user_prompt)
        similar_clause_note = await _find_similar_clause(session, user_prompt)

    has_document_context = metadata is not None or relevant_sections is not None

    context = {
        "user_prompt": user_prompt,
        "has_document_context": has_document_context,
        "has_metadata": metadata is not None,
        "has_relevant_sections": relevant_sections is not None,
        "has_similar_clause": similar_clause_note is not None,
    }
    if metadata:
        context.update(
            {
                "document_type": metadata.get("document_type", ""),
                "parties": metadata.get("parties", ""),
                "duration": metadata.get("duration", ""),
                "effective_date": metadata.get("effective_date", ""),
            }
        )
    if relevant_sections:
        context["relevant_sections"] = relevant_sections
    if similar_clause_note:
        context["similar_clause_note"] = similar_clause_note

    prompt = load_prompt("describe_draft_prompt")
    llm_client = container.azure_openai_model

    try:
        llm_response: DraftLLMResponse = await llm_client.generate(
            prompt=prompt,
            context=context,
            response_model=DraftLLMResponse,
            temperature=0.7,
        )

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
