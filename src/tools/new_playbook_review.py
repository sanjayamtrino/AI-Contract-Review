import asyncio
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from docx.document import Document

from src.config.logging import get_logger
from src.dependencies import get_service_container
from src.schemas.playbook_review import (
    PlayBookReviewFinalResponse,
    PlayBookReviewLLMResponse,
    PlayBookReviewResponse,
    RuleCheckRequest,
    RuleInfo,
)
from src.services.session_manager import SessionData

logger = get_logger(__name__)


AGENT_NAME = "playbook_review_agent"

# Hybrid title matching thresholds. Tuned permissive — when neither layer
# reaches the threshold we still surface the top embedding candidate so the
# LLM can verify the match itself (the prompt has a title-alignment guard
# that returns "Not Found" if the candidate is the wrong topic).
FUZZY_TITLE_THRESHOLD = 0.50
TITLE_EMBEDDING_THRESHOLD = 0.40
TITLE_EMBEDDING_BEST_EFFORT_FLOOR = 0.20

# Rule-title content words that should not influence fuzzy token-overlap.
_TITLE_STOP_WORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at",
    "by", "with", "from", "this", "that", "these", "those",
}

_TITLE_WORD_PATTERN = re.compile(r"\b[a-z]{3,}\b")


def normalize(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _content_tokens(text: str) -> set:
    """Lowercase content tokens (>=3 chars), with stop-words removed."""
    return {
        t for t in _TITLE_WORD_PATTERN.findall(text.lower())
        if t not in _TITLE_STOP_WORDS
    }


def _fuzzy_title_ratio(rule_title: str, clause_heading: str) -> float:
    """[0, 1] similarity between a rule title and a clause heading.

    Combines token-overlap (every content word of the rule appearing in the
    heading) with character-level SequenceMatcher. Returns the max so single-
    token rules and multi-word rules both behave well.
    """
    if not rule_title or not clause_heading:
        return 0.0
    a = normalize(rule_title)
    b = normalize(clause_heading)
    if not a or not b:
        return 0.0
    rule_tokens = _content_tokens(a)
    head_tokens = _content_tokens(b)
    token_overlap = (
        len(rule_tokens & head_tokens) / len(rule_tokens) if rule_tokens else 0.0
    )
    seq_ratio = SequenceMatcher(None, a, b).ratio()
    return max(token_overlap, seq_ratio)


async def _find_matching_chunk(
    rule: RuleInfo,
    chunks,
    rule_embedding: Optional[np.ndarray],
    head_embeddings: Optional[np.ndarray],
    headed_chunks: List,
) -> Tuple[Optional[object], str]:
    """Pick the best chunk for a rule using fuzzy then embedding fallback.

    Returns (chunk_or_None, strategy). strategy is one of
    "title_fuzzy", "title_embedding", or "" when nothing matched.
    """
    # 1) Fuzzy
    best_chunk = None
    best_ratio = 0.0
    for chunk in chunks:
        heading = chunk.metadata.get("section_heading", "") if chunk.metadata else ""
        if not heading:
            continue
        ratio = _fuzzy_title_ratio(rule.title, heading)
        if ratio > best_ratio:
            best_ratio = ratio
            best_chunk = chunk

    if best_ratio >= FUZZY_TITLE_THRESHOLD and best_chunk is not None:
        logger.info(
            "Title match (fuzzy %.2f): rule '%s' -> clause '%s'",
            best_ratio, rule.title,
            best_chunk.metadata.get("section_heading", ""),
        )
        return best_chunk, "title_fuzzy"

    # 2) Embedding fallback on titles only
    if (
        rule_embedding is None
        or head_embeddings is None
        or head_embeddings.size == 0
        or not headed_chunks
    ):
        return None, ""

    rule_norm = rule_embedding / (np.linalg.norm(rule_embedding) + 1e-10)
    scores = head_embeddings @ rule_norm
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score >= TITLE_EMBEDDING_THRESHOLD:
        chunk = headed_chunks[best_idx]
        logger.info(
            "Title match (embedding %.2f): rule '%s' -> clause '%s'",
            best_score, rule.title,
            chunk.metadata.get("section_heading", ""),
        )
        return chunk, "title_embedding"

    # Best-effort fallback: if even the embedding match is weak, still surface
    # the top candidate so the LLM gets a chance. The prompt's title-alignment
    # guard will return "Not Found" if the candidate is genuinely off-topic.
    # This avoids the failure mode where every borderline rule returns
    # Not Found purely because of a sub-threshold score.
    if best_score >= TITLE_EMBEDDING_BEST_EFFORT_FLOOR and best_idx >= 0:
        chunk = headed_chunks[best_idx]
        logger.info(
            "Title match (best-effort %.2f): rule '%s' -> clause '%s' "
            "(LLM will verify topic alignment).",
            best_score, rule.title,
            chunk.metadata.get("section_heading", ""),
        )
        return chunk, "title_best_effort"

    return None, ""


async def playbook_review_service(
    document: Document,
    request: RuleCheckRequest,
    session_data: SessionData,
) -> PlayBookReviewFinalResponse:
    """Review a docx against a list of rules using clause-title matching."""
    service_container = get_service_container()
    registry = service_container.ingestion_service.registry
    parser = registry.get_parser()
    embedding_service = service_container.embedding_service

    # Step 1: parse document into clauses (each chunk has metadata.section_heading)
    parse_result = await parser.parse_document(document, session_data)
    if not parse_result.success or not parse_result.chunks:
        logger.error("Failed to parse document: %s", parse_result.error_message)
        return PlayBookReviewFinalResponse(rules_review=[], missing_clauses=None)

    # Cache chunks on the session when one is available. We do not require a
    # pre-existing session — review can still run end-to-end from just the
    # uploaded document, the session is only used to speed up repeat calls.
    if session_data is not None:
        for chunk in parse_result.chunks:
            session_data.chunk_store[chunk.chunk_index] = chunk
    else:
        logger.info(
            "No session_data attached (likely an unknown X-Session-Id). "
            "Continuing without session-side caching."
        )

    logger.info(
        "Parsed document with %d clause chunks", len(parse_result.chunks),
    )

    # Step 2: precompute embeddings once per request (cheap parallel batch).
    headed_chunks = [
        c for c in parse_result.chunks
        if c.metadata and c.metadata.get("section_heading")
    ]
    head_embeddings: Optional[np.ndarray] = None
    if headed_chunks:
        head_embs_raw = await asyncio.gather(
            *[
                embedding_service.generate_embeddings(c.metadata["section_heading"])
                for c in headed_chunks
            ]
        )
        head_matrix = np.array(head_embs_raw)
        head_embeddings = head_matrix / (
            np.linalg.norm(head_matrix, axis=1, keepdims=True) + 1e-10
        )

    rule_embeddings_raw = await asyncio.gather(
        *[embedding_service.generate_embeddings(r.title) for r in request.rulesinformation]
    )

    # Step 3: per-rule match + LLM evaluation
    rules_review: List[PlayBookReviewResponse] = []

    for rule, rule_emb in zip(request.rulesinformation, rule_embeddings_raw):
        rule_emb_arr = np.array(rule_emb)

        matching_chunk, strategy = await _find_matching_chunk(
            rule=rule,
            chunks=parse_result.chunks,
            rule_embedding=rule_emb_arr,
            head_embeddings=head_embeddings,
            headed_chunks=headed_chunks,
        )

        if not matching_chunk:
            logger.warning(
                "No clause matched rule '%s' (fuzzy < %.2f, embedding < %.2f).",
                rule.title, FUZZY_TITLE_THRESHOLD, TITLE_EMBEDDING_THRESHOLD,
            )
            llm_response = PlayBookReviewLLMResponse(
                para_identifiers=[],
                status="Not Found",
                reason=(
                    f"No clause in the document was a close enough title match "
                    f"for rule '{rule.title}'."
                ),
                suggestion="",
                suggested_fix="",
            )
        else:
            llm_response = await validate_clause_against_rule(
                clause_content=matching_chunk.content,
                clause_heading=matching_chunk.metadata.get("section_heading", ""),
                rule_title=rule.title,
                rule_description=rule.description,
                rule_instruction=rule.instruction,
            )

        rules_review.append(
            PlayBookReviewResponse(
                rule_title=rule.title,
                rule_instruction=rule.instruction,
                rule_description=rule.description,
                content=llm_response,
            )
        )

    return PlayBookReviewFinalResponse(rules_review=rules_review, missing_clauses=None)


async def validate_clause_against_rule(
    clause_content: str,
    clause_heading: str,
    rule_title: str,
    rule_description: str,
    rule_instruction: str,
) -> PlayBookReviewLLMResponse:
    """Send the matched clause + rule to the LLM for compliance evaluation."""
    container = get_service_container()
    llm_model = container.azure_openai_model

    prompt = Path(
        r"src\services\prompts\v1\ai_review_prompt_v2.mustache"
    ).read_text(encoding="utf-8")

    para_id = clause_heading.strip() or "matched_clause"
    paragraphs_block = f"PARA_ID: {para_id}\nTEXT: {clause_content}"

    context = {
        "rule_title": rule_title,
        "rule_instruction": rule_instruction,
        "rule_description": rule_description,
        "paragraphs": paragraphs_block,
    }

    try:
        response = await llm_model.generate(
            prompt=prompt,
            context=context,
            response_model=PlayBookReviewLLMResponse,
        )
        return response
    except Exception as exc:
        logger.exception("LLM validation failed for rule '%s'.", rule_title)
        return PlayBookReviewLLMResponse(
            para_identifiers=[para_id],
            status="Error",
            reason=f"LLM validation failed: {exc}",
            suggestion="",
            suggested_fix="",
        )
