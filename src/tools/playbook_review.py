import asyncio
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from cachetools import LRUCache

from src.config.logging import get_logger
from src.dependencies import get_service_container
from src.schemas.playbook_review import (
    MissingClausesLLMResponse,
    PlayBookReviewFinalResponse,
    PlayBookReviewLLMResponse,
    PlayBookReviewResponse,
    RuleCheckRequest,
    RuleInfo,
    RuleResult,
    TextInfo,
)

logger = get_logger(__name__)

# ==============================
# Configuration
# ==============================

AGENT_NAME = "playbook_review_agent"

# FIX 1: Lowered threshold from 0.5 → 0.25
# Previous value was silently dropping relevant paragraphs before the LLM ever saw them.
SIMILARITY_THRESHOLD = 0.25

# FIX 2: Increased TOP_K from 5 → 15
# With 33 paragraphs, TOP_K=5 was cutting off multi-paragraph clauses
# (e.g. Compensation spans P0004, P0005, P0025, P0026, P0027 — all 5 could be missed).
TOP_K = 15

# FIX 3: Small-doc threshold — if the document has fewer paragraphs than this,
# skip embedding filtering entirely and pass all paragraphs to the LLM.
# This prevents the retrieval layer from gatekeeping on small contracts.
SMALL_DOC_THRESHOLD = 50

EMBEDDING_CACHE_SIZE = 10_000

SIMILARITY_PROMPT = Path(r"src\services\prompts\v1\ai_review_prompt_v2.mustache").read_text(encoding="utf-8")

MISSING_CLAUSES_PROMPT = Path(r"src\services\prompts\v1\missing_clauses.mustache").read_text(encoding="utf-8")

# ==============================
# Embedding Cache (LRU + Safe)
# ==============================

_embedding_cache: LRUCache = LRUCache(maxsize=EMBEDDING_CACHE_SIZE)
_embedding_locks: Dict[str, asyncio.Lock] = {}
_embedding_locks_guard = asyncio.Lock()


def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


async def _get_lock_for_key(key: str) -> asyncio.Lock:
    async with _embedding_locks_guard:
        if key not in _embedding_locks:
            _embedding_locks[key] = asyncio.Lock()
        return _embedding_locks[key]


async def get_embedding(embedding_model, text: str) -> np.ndarray:
    key = _hash(text)

    if key in _embedding_cache:
        return _embedding_cache[key]

    lock = await _get_lock_for_key(key)

    async with lock:
        if key not in _embedding_cache:
            embedding = await embedding_model.generate_embeddings(text)
            _embedding_cache[key] = embedding
        return _embedding_cache[key]


# ==============================
# Text Processing
# ==============================

TOKEN_PATTERN = re.compile(r"\b[a-zA-Z]{3,}\b")


def tokenize(text: str) -> set:
    return set(TOKEN_PATTERN.findall(text.lower()))


def keyword_score(rule_text: str, para_text: str) -> float:
    rule_tokens = tokenize(rule_text)
    if not rule_tokens:
        return 0.0
    para_tokens = tokenize(para_text)
    return len(rule_tokens & para_tokens) / len(rule_tokens)


def hybrid_score(cosine: float, keyword: float) -> float:
    return 0.75 * cosine + 0.25 * keyword


def normalize_embeddings(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    return matrix / norms


# ==============================
# Missing Clauses
# ==============================


def _build_reviewed_rules_summary(reviewed: Dict[str, PlayBookReviewResponse]) -> str:
    lines = []
    for title, review in reviewed.items():
        para_ids = ", ".join(review.content.para_identifiers) or "none"
        lines.append(f"RULE: {title} | STATUS: {review.content.status} | PARAS: {para_ids}")
    return "\n".join(lines) if lines else "None"


async def get_missing_clauses(
    llm_model,
    full_text: str,
    reviewed_rules_summary: str,
) -> MissingClausesLLMResponse:

    try:
        response = await llm_model.generate(
            prompt=MISSING_CLAUSES_PROMPT,
            context={
                "data": full_text,
                "reviewed_rules_summary": reviewed_rules_summary,
            },
            response_model=MissingClausesLLMResponse,
        )
        logger.info("Missing clauses identified: %d", len(response.missing_clauses))
        return response

    except Exception as exc:
        logger.exception("Missing clauses evaluation failed.")
        return MissingClausesLLMResponse(
            missing_clauses=[],
            total_missing=0,
            summary=f"LLM error: {exc}",
        )


# ==============================
# Rule Processing
# ==============================


def _select_paragraphs_for_rule(
    rule_text: str,
    rule_norm: np.ndarray,
    normalized_para_embeddings: np.ndarray,
    request: RuleCheckRequest,
) -> List[Tuple[TextInfo, float]]:
    """
    FIX 4: Unified paragraph selection strategy.

    For small documents (≤ SMALL_DOC_THRESHOLD paragraphs):
        → Skip embedding filtering entirely. Pass ALL paragraphs to LLM.
          The LLM is perfectly capable of reading 30–50 short paragraphs and
          deciding relevance itself. The embedding layer was the one making mistakes.

    For large documents:
        → Use TOP_K + hybrid score + SIMILARITY_THRESHOLD as before,
          BUT fall back to top-5 by cosine if nothing clears the threshold.
          This prevents the "empty context" bug where the LLM was forced
          to say "Not Found" because it received zero paragraphs.
    """
    num_paras = len(request.textinformation)

    # Small doc: pass everything to the LLM
    if num_paras <= SMALL_DOC_THRESHOLD:
        cosine_scores = normalized_para_embeddings @ rule_norm
        result = []
        for idx, para in enumerate(request.textinformation):
            kw = keyword_score(rule_text, para.text)
            score = hybrid_score(float(cosine_scores[idx]), kw)
            result.append((para, score))
        # Sort by score descending so LLM context is ordered by relevance
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    # Large doc: use embedding filtering
    cosine_scores = normalized_para_embeddings @ rule_norm
    top_indices = np.argsort(cosine_scores)[::-1][:TOP_K]

    matched: List[Tuple[TextInfo, float]] = []
    for idx in top_indices:
        para = request.textinformation[idx]
        kw = keyword_score(rule_text, para.text)
        score = hybrid_score(float(cosine_scores[idx]), kw)
        if score >= SIMILARITY_THRESHOLD:
            matched.append((para, score))

    # FIX 5: Fallback — never send empty context to the LLM.
    # If nothing clears the threshold, send the top-5 by cosine anyway
    # so the LLM can make the final call instead of defaulting to "Not Found".
    if not matched:
        logger.warning(
            "No paragraphs cleared threshold %.2f for rule. Falling back to top-5 by cosine.",
            SIMILARITY_THRESHOLD,
        )
        fallback_indices = np.argsort(cosine_scores)[::-1][:5]
        for idx in fallback_indices:
            para = request.textinformation[idx]
            kw = keyword_score(rule_text, para.text)
            score = hybrid_score(float(cosine_scores[idx]), kw)
            matched.append((para, score))

    return matched


async def _process_rule(
    rule: RuleInfo,
    normalized_para_embeddings: np.ndarray,
    request: RuleCheckRequest,
    embedding_model,
    llm_model,
) -> Tuple[str, PlayBookReviewResponse]:

    rule_text = f"TITLE: {rule.title}\n" f"INSTRUCTION: {rule.instruction}\n" f"DESCRIPTION: {rule.description}\n"

    rule_emb = await get_embedding(embedding_model, rule_text)
    rule_norm = rule_emb / (np.linalg.norm(rule_emb) + 1e-10)

    # FIX 4+5 applied here
    matched = _select_paragraphs_for_rule(rule_text, rule_norm, normalized_para_embeddings, request)

    result = RuleResult(
        title=rule.title,
        instruction=rule.instruction,
        description=rule.description,
        paragraphidentifier=",".join(p.paraindetifier for p, _ in matched),
        paragraphcontext="\n\n".join(f"PARA_ID: {p.paraindetifier}\nTEXT: {p.text.strip()}" for p, _ in matched),
        similarity_scores=[score for _, score in matched],
    )

    try:
        llm_response: PlayBookReviewLLMResponse = await llm_model.generate(
            prompt=SIMILARITY_PROMPT,
            context={
                "rule_title": result.title,
                "rule_instruction": result.instruction,
                "rule_description": result.description,
                "paragraphs": result.paragraphcontext,
            },
            response_model=PlayBookReviewLLMResponse,
        )

    except Exception as exc:
        logger.exception("LLM rule evaluation failed.")
        llm_response = PlayBookReviewLLMResponse(
            para_identifiers=[],
            status="Error",
            reason=str(exc),
            suggestion="",
            suggested_fix="",
        )

    return rule.title, PlayBookReviewResponse(
        rule_title=rule.title,
        rule_instruction=rule.instruction,
        rule_description=rule.description,
        content=llm_response,
    )


# ==============================
# Main Entry
# ==============================


async def review_document(
    session_id: str,
    request: RuleCheckRequest,
    force_update_rules: Optional[List[str]] = None,
) -> PlayBookReviewFinalResponse:

    force_update_rules = force_update_rules or []

    container = get_service_container()
    embedding_model = container.embedding_service
    llm_model = container.azure_openai_model

    session_data = container.session_manager.get_session(session_id)
    if not session_data:
        return PlayBookReviewFinalResponse(
            rules_review=[],
            missing_clauses=None,
        )

    agent_cache = session_data.tool_results.get(AGENT_NAME, {})
    cached_reviews: Dict[str, PlayBookReviewResponse] = {r.rule_title: r for r in agent_cache.get("rules_review", [])}

    # Determine stale rules
    rules_to_update = []
    for rule in request.rulesinformation:
        cached = cached_reviews.get(rule.title)
        if not cached or rule.title in force_update_rules or cached.rule_description != rule.description or cached.rule_instruction != rule.instruction:
            rules_to_update.append(rule)

    # If nothing changed, return cache
    if not rules_to_update:
        return PlayBookReviewFinalResponse(
            rules_review=list(cached_reviews.values()),
            missing_clauses=agent_cache.get("missing_clauses"),
        )

    # Precompute normalized paragraph embeddings once
    para_embeddings = np.array(await asyncio.gather(*[get_embedding(embedding_model, p.text) for p in request.textinformation]))
    normalized_para_embeddings = normalize_embeddings(para_embeddings)

    # Process rules concurrently
    updates = await asyncio.gather(
        *[
            _process_rule(
                rule,
                normalized_para_embeddings,
                request,
                embedding_model,
                llm_model,
            )
            for rule in rules_to_update
        ]
    )

    cached_reviews.update(dict(updates))

    # Recompute missing clauses if doc OR rules changed
    full_text = "\n\n".join(f"PARA_ID: {p.paraindetifier}\nTEXT: {p.text}" for p in request.textinformation)

    doc_hash = _hash(full_text)
    rules_hash = _hash("".join(r.title + r.description for r in request.rulesinformation))

    cached_doc_hash = agent_cache.get("doc_hash")
    cached_rules_hash = agent_cache.get("rules_hash")

    if doc_hash != cached_doc_hash or rules_hash != cached_rules_hash:
        logger.info("Re-evaluating missing clauses.")
        reviewed_summary = _build_reviewed_rules_summary(cached_reviews)
        missing_clauses = await get_missing_clauses(llm_model, full_text, reviewed_summary)
    else:
        missing_clauses = agent_cache.get("missing_clauses")

    session_data.tool_results[AGENT_NAME] = {
        "rules_review": list(cached_reviews.values()),
        "missing_clauses": missing_clauses,
        "doc_hash": doc_hash,
        "rules_hash": rules_hash,
    }

    return PlayBookReviewFinalResponse(
        rules_review=list(cached_reviews.values()),
        missing_clauses=missing_clauses,
    )


# import asyncio
# import hashlib
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple

# import numpy as np

# from src.config.logging import get_logger
# from src.dependencies import get_service_container
# from src.schemas.playbook_review import (
#     MissingClausesLLMResponse,
#     PlayBookReviewFinalResponse,
#     PlayBookReviewLLMResponse,
#     PlayBookReviewResponse,
#     RuleCheckRequest,
#     RuleInfo,
#     RuleResult,
#     TextInfo,
# )

# logger = get_logger(__name__)

# similarity_prompt_template = Path(r"src\services\prompts\v1\ai_review_prompt_v2.mustache").read_text(encoding="utf-8")

# AGENT_NAME = "playbook_review_agent"
# SIMILARITY_THRESHOLD = 0.35

# # Protect shared cache against async race conditions
# embedding_cache: Dict[str, np.ndarray] = {}
# _embedding_locks: Dict[str, asyncio.Lock] = {}


# def _hash(text: str) -> str:
#     return hashlib.md5(text.encode("utf-8")).hexdigest()


# def _doc_hash(text_items: List[TextInfo]) -> str:
#     """Stable hash of the full document paragraph list."""
#     combined = "".join(item.text for item in text_items)
#     return _hash(combined)


# def _build_reviewed_rules_summary(cached_rules_review: Dict[str, PlayBookReviewResponse]) -> str:
#     """Build a summary string of the reviewed rules to pass to the missing clauses LLM, so it knows which topics have already been reviewed and should not be re-flagged as missing."""

#     lines: List[str] = []
#     for title, review in cached_rules_review.items():
#         para_ids = ", ".join(review.content.para_identifiers) or "none"
#         lines.append(f"RULE: {title} | STATUS: {review.content.status} | PARAS: {para_ids}")
#     return "\n".join(lines) if lines else "None"


# async def get_missing_clauses(data: str, reviewed_rules_summary: str) -> MissingClausesLLMResponse:
#     """Evaluate the document for genuinely absent clauses."""

#     service_container = get_service_container()
#     llm_model = service_container.azure_openai_model

#     prompt = Path(r"src\services\prompts\v1\missing_clauses.mustache").read_text(encoding="utf-8")

#     try:
#         response: MissingClausesLLMResponse = await llm_model.generate(
#             prompt=prompt,
#             context={
#                 "data": data,
#                 "reviewed_rules_summary": reviewed_rules_summary,
#             },
#             response_model=MissingClausesLLMResponse,
#         )
#         logger.info(f"Identified {len(response.missing_clauses)} missing clauses.")
#         return response
#     except Exception as exc:
#         logger.error(f"Missing clauses LLM call failed: {exc}")
#         return MissingClausesLLMResponse(
#             missing_clauses=[],
#             total_missing=0,
#             summary="Could not evaluate missing clauses due to an LLM error.",
#         )

#     # {
#     #   "text": "Confidential Information does not include any information, however designated, that:  (i) is or subsequently becomes publicly available without Receiving Party’s breach of any obligation owed Disclosing Party; (ii) was known by Receiving Party prior to Disclosing Party’s disclosure of such information to Receiving Party pursuant to the terms of this Agreement; (iii) became known to Receiving Party from a source other than Disclosing Party other than by the breach of an obligation of confidentiality owed to Disclosing Party; or (iv) is independently developed by Receiving Party.  \r",
#     #   "paraindetifier": "P0005"
#     # },


# async def get_embedding(embedding_model, text: str) -> np.ndarray:
#     """Task-safe embedding lookup with per-key lock to prevent duplicate fetches."""

#     key = _hash(text)
#     if key in embedding_cache:
#         return embedding_cache[key]
#     if key not in _embedding_locks:
#         _embedding_locks[key] = asyncio.Lock()
#     async with _embedding_locks[key]:
#         if key not in embedding_cache:
#             embedding_cache[key] = await embedding_model.generate_embeddings(text)
#     return embedding_cache[key]


# def keyword_score(rule: str, para: str) -> float:
#     rule_tokens = set(rule.lower().split())
#     para_tokens = set(para.lower().split())
#     if not rule_tokens:
#         return 0.0
#     return len(rule_tokens & para_tokens) / len(rule_tokens)


# def hybrid_score(cosine: float, keyword: float) -> float:
#     return 0.75 * cosine + 0.25 * keyword


# async def get_matching_pairs_faiss(request: RuleCheckRequest) -> List[RuleResult]:
#     """Get the matching pairs for the given rules with FAISS."""

#     service_container = get_service_container()
#     faiss_db = service_container.faiss_store
#     embedding_model = service_container.embedding_service

#     faiss_db.reset_index()

#     para_map: List[TextInfo] = []
#     for item in request.textinformation:
#         emb = await get_embedding(embedding_model, item.text)
#         await faiss_db.index_embedding(emb, metadata=item.paraidentifier)
#         para_map.append(item)

#     results: List[RuleResult] = []
#     for rule in request.rulesinformation:
#         rule_text = f"TITLE: {rule.title}\n" f"INSTRUCTION: {rule.instruction}\n" f"DESCRIPTION: {rule.description}\n" f"TAGS: {', '.join(rule.tags)}"

#         rule_emb = await get_embedding(embedding_model, rule_text)
#         faiss_result = await faiss_db.search_index(rule_emb, top_k=5)

#         indices = faiss_result.get("indices", [])
#         scores = faiss_result.get("scores", [])

#         matched_pairs: List[Tuple[TextInfo, float]] = []
#         for idx, score in zip(indices, scores):
#             if idx == -1 or idx >= len(para_map):
#                 continue
#             para = para_map[idx]
#             kw = keyword_score(rule_text, para.text)
#             final_score = hybrid_score(float(score), kw)
#             if final_score >= SIMILARITY_THRESHOLD:
#                 matched_pairs.append((para, final_score))

#         if not matched_pairs:
#             results.append(
#                 RuleResult(
#                     title=rule.title,
#                     instruction=rule.instruction,
#                     description="No relevant contract paragraphs found.",
#                     paragraphidentifier="",
#                     paragraphcontext="",
#                     similarity_scores=[],
#                 )
#             )
#             continue

#         results.append(
#             RuleResult(
#                 title=rule.title,
#                 instruction=rule.instruction,
#                 description=rule.description,
#                 paragraphidentifier=",".join(p.paraidentifier for p, _ in matched_pairs),
#                 paragraphcontext="\n\n".join(f"PARA_ID: {p.paraidentifier}\nTEXT: {p.text.strip()}" for p, _ in matched_pairs),
#                 similarity_scores=[s for _, s in matched_pairs],
#             )
#         )

#     return results


# async def _process_single_rule(rule: RuleInfo, para_embeddings: np.ndarray, request: RuleCheckRequest, llm_model, embedding_model) -> Tuple[str, PlayBookReviewResponse]:
#     """Process a single rule: compute similarities, find matches, and call LLM for review. Returns the rule title and its corresponding PlayBookReviewResponse."""

#     rule_text = f"TITLE: {rule.title}\n" f"INSTRUCTION: {rule.instruction}\n" f"DESCRIPTION: {rule.description}\n"  # f"TAGS: {', '.join(rule.tags)}"

#     rule_emb = await get_embedding(embedding_model, rule_text)

#     norms_rule = rule_emb / (np.linalg.norm(rule_emb) + 1e-10)
#     norms_para = para_embeddings / (np.linalg.norm(para_embeddings, axis=1, keepdims=True) + 1e-10)
#     cosine_scores = norms_para @ norms_rule

#     top_idx = np.argsort(cosine_scores)[::-1][:5]

#     matched: List[Tuple[TextInfo, float]] = []
#     for idx in top_idx:
#         para = request.textinformation[idx]
#         kw = keyword_score(rule_text, para.text)
#         score = hybrid_score(float(cosine_scores[idx]), kw)
#         if score >= SIMILARITY_THRESHOLD:
#             matched.append((para, score))

#     if not matched:
#         result = RuleResult(
#             title=rule.title,
#             instruction=rule.instruction,
#             description="No relevant contract paragraphs found.",
#             paragraphidentifier="",
#             paragraphcontext="",
#             similarity_scores=[],
#         )
#     else:
#         result = RuleResult(
#             title=rule.title,
#             instruction=rule.instruction,
#             description=rule.description,
#             paragraphidentifier=",".join(p.paraindetifier for p, _ in matched),
#             paragraphcontext="\n\n".join(f"PARA_ID: {p.paraindetifier}\nTEXT: {p.text.strip()}" for p, _ in matched),
#             similarity_scores=[s for _, s in matched],
#         )

#     context = {
#         "rule_title": result.title,
#         "rule_instruction": result.instruction,
#         "rule_description": result.description,
#         "paragraphs": result.paragraphcontext,
#     }

#     try:
#         llm_result: PlayBookReviewLLMResponse = await llm_model.generate(
#             prompt=similarity_prompt_template,
#             context=context,
#             response_model=PlayBookReviewLLMResponse,
#         )
#     except Exception as exc:
#         logger.error(f"LLM review failed for rule '{rule.title}': {exc}")
#         llm_result = PlayBookReviewLLMResponse(
#             para_identifiers=[p.paraidentifier for p, _ in matched],
#             status="Error",
#             reason=f"LLM call failed: {exc}",
#             suggestion="",
#             suggested_fix="",
#         )

#     return rule.title, PlayBookReviewResponse(
#         rule_title=result.title,
#         rule_instruction=result.instruction,
#         rule_description=result.description,
#         content=llm_result,
#     )


# async def review_document(session_id: str, request: RuleCheckRequest, force_update_rules: Optional[List[str]] = None) -> PlayBookReviewFinalResponse:
#     """Run playbook review for the given document and rules, with optional force update for specific rules."""

#     force_update_rules = force_update_rules or []

#     service_container = get_service_container()
#     embedding_model = service_container.embedding_service
#     llm_model = service_container.azure_openai_model

#     session_data = service_container.session_manager.get_session(session_id)
#     if not session_data:
#         return PlayBookReviewFinalResponse(rules_review=[], missing_clauses=None)

#     agent_cache = session_data.tool_results.get(AGENT_NAME, {})
#     cached_rules_review: Dict[str, PlayBookReviewResponse] = {r.rule_title: r for r in agent_cache.get("rules_review", [])}

#     # Determine which rules need (re)processing
#     rules_to_update: List[RuleInfo] = []
#     for rule in request.rulesinformation:
#         cached = cached_rules_review.get(rule.title)
#         if force_update_rules and rule.title in force_update_rules:
#             rules_to_update.append(rule)
#             continue
#         if not cached:
#             rules_to_update.append(rule)
#             continue
#         if cached.rule_description != rule.description or cached.rule_instruction != rule.instruction:
#             rules_to_update.append(rule)

#     # Return fully-cached result when nothing changed
#     if not rules_to_update:
#         return PlayBookReviewFinalResponse(
#             rules_review=list(cached_rules_review.values()),
#             missing_clauses=agent_cache.get("missing_clauses"),
#         )

#     # Precompute all paragraph embeddings once, concurrently
#     para_embeddings = np.array(await asyncio.gather(*[get_embedding(embedding_model, item.text) for item in request.textinformation]))

#     # Process all stale rules concurrently
#     rule_results = await asyncio.gather(*[_process_single_rule(rule, para_embeddings, request, llm_model, embedding_model) for rule in rules_to_update])
#     cached_rules_review.update(dict(rule_results))

#     # Use ALL paragraph text — not just matched ones — so nothing is
#     # invisible to the missing-clauses LLM
#     all_text = " ".join(p.text for p in request.textinformation)

#     # Only re-run missing-clauses when the document itself changed
#     current_doc_hash = _doc_hash(request.textinformation)
#     cached_doc_hash = agent_cache.get("doc_hash")

#     if current_doc_hash != cached_doc_hash:
#         logger.info("Document changed — re-evaluating missing clauses.")

#         # Pass the full rules-review summary so the missing-clauses LLM doesn't
#         # double-flag topics already handled by the rules review
#         reviewed_rules_summary = _build_reviewed_rules_summary(cached_rules_review)

#         missing_clauses = await get_missing_clauses(
#             data=all_text,
#             reviewed_rules_summary=reviewed_rules_summary,
#         )
#     else:
#         logger.info("Document unchanged — reusing cached missing clauses.")
#         missing_clauses = agent_cache.get("missing_clauses")

#     session_data.tool_results[AGENT_NAME] = {
#         "rules_review": list(cached_rules_review.values()),
#         "missing_clauses": missing_clauses,
#         "doc_hash": current_doc_hash,
#     }

#     return PlayBookReviewFinalResponse(
#         rules_review=list(cached_rules_review.values()),
#         missing_clauses=missing_clauses,
#     )
