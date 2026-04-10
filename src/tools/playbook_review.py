# import asyncio
# import hashlib
# from pathlib import Path
# from typing import Any, Dict, List

# import numpy as np

# from src.config.logging import get_logger
# from src.dependencies import get_service_container
# from src.schemas.playbook_review import (
#     MissingClausesLLMResponse,
#     ParaSimilarity,
#     PlayBookReviewFinalResponse,
#     PlayBookReviewLLMResponse,
#     PlayBookReviewResponse,
#     RuleCheckRequest,
#     RuleResult,
#     TextInfo,
# )

# logger = get_logger(__name__)

# similarity_prompt_template = Path(r"src\services\prompts\v1\ai_review_prompt_v2.mustache").read_text(encoding="utf-8")

# AGENT_NAME = "playbook_review_agent"


# def _hash(text: str) -> str:
#     return hashlib.md5(text.encode("utf-8")).hexdigest()


# async def get_missing_clauses(data: str) -> MissingClausesLLMResponse:
#     """Get the missing clauses for the given contract text."""

#     service_container = get_service_container()
#     llm_model = service_container.azure_openai_model

#     prompt = Path(r"src\services\prompts\v1\missing_clauses.mustache").read_text(encoding="utf-8")
#     context = {"data": data}
#     response: MissingClausesLLMResponse = await llm_model.generate(
#         prompt=prompt,
#         context=context,
#         response_model=MissingClausesLLMResponse,
#     )

#     # Extract missing clauses from the structured response
#     missing_clauses = response.missing_clauses

#     logger.info(f"Identified {len(missing_clauses)} missing clauses.")

#     return response


# async def get_matching_pairs_faiss(request: RuleCheckRequest) -> List[RuleResult]:
#     """Get the matching pairs for the given rules with FAISS."""

#     service_container = get_service_container()
#     faiss_db = service_container.faiss_store
#     embedding_model = service_container.embedding_service

#     # Index all paragraph embeddings into FAISS
#     for item in request.textinformation:
#         embedd_vector = await embedding_model.generate_embeddings(item.text)
#         logger.info(f"Indexing paragraph {item.paraindetifier} into FAISS.")
#         await faiss_db.index_embedding(embedd_vector)

#     results: List[RuleResult] = []

#     for rule in request.rulesinformation:
#         rule_text = f"title: {rule.title}. " f"description: {rule.description}. "  #  f"tags: {', '.join(rule.tags)}
#         logger.info(f"Generating embedding for rule '{rule.title}'.")
#         rule_embedds = await embedding_model.generate_embeddings(rule_text)
#         logger.info(f"Searching for similar paragraphs in FAISS for rule '{rule.title}'.")
#         faiss_result: Dict[str, Any] = await faiss_db.search_index(rule_embedds, top_k=3)

#         indices = faiss_result.get("indices", [])
#         scores = faiss_result.get("scores", [])

#         # Filter out invalid FAISS indices (-1 means no result found)
#         matched_pairs: List[ParaSimilarity] = [(idx, score) for idx, score in zip(indices, scores) if idx != -1 and idx < len(request.textinformation)]

#         if not matched_pairs:
#             logger.info(f"No relevant paragraphs found in FAISS for rule '{rule.title}'.")
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

#         logger.info(f"Found {len(matched_pairs)} similar paragraphs in FAISS for rule '{rule.title}'.")

#         matched_paras = [request.textinformation[idx] for idx, _ in matched_pairs]
#         similarity_scores = [float(score) for _, score in matched_pairs]

#         para_ids = ",".join(p.paraindetifier for p in matched_paras)
#         para_context = "\n\n".join(f"[{p.paraindetifier}] {p.text.strip()}" for p in matched_paras)

#         results.append(
#             RuleResult(
#                 title=rule.title,
#                 instruction=rule.instruction,
#                 description=rule.description,
#                 paragraphidentifier=para_ids,
#                 paragraphcontext=para_context,
#                 similarity_scores=similarity_scores,
#             )
#         )

#     return results


# def find_similarity(rule_embedd: np.ndarray, para_embedds: np.ndarray, para_items: List[TextInfo], top_k: int = 3, threshold: float = 0.30) -> List[ParaSimilarity]:
#     """get the similar paragraphs for the given rules."""

#     logger.info("Normalizing embeddings for similarity computation.")
#     # Normalize safely
#     rules_norm = rule_embedd / (np.linalg.norm(rule_embedd) + 1e-10)
#     para_norms = para_embedds / (np.linalg.norm(para_embedds, axis=1, keepdims=True) + 1e-10)

#     logger.info("Computing cosine similarity between rule and paragraph embeddings.")
#     # compute cosine similarity
#     scores = para_norms @ rules_norm

#     logger.info("Sorting similarity scores and filtering based on threshold.")
#     # Sort indices descending
#     top_indices = np.argsort(scores)[::-1][:top_k]

#     results: List[ParaSimilarity] = []
#     for idx in top_indices:
#         if scores[idx] >= threshold:
#             results.append({"paragraph": para_items[idx], "similarity": float(scores[idx])})

#     logger.info(f"Found {len(results)} paragraphs with similarity above the threshold of {threshold}.")

#     return results


# async def review_document(session_id: str, request: RuleCheckRequest, force_update_rules: List[str] = None) -> PlayBookReviewFinalResponse:
#     """Run playbook review for the given document and rules, with optional force update for specific rules."""

#     force_update_rules = force_update_rules or []

#     service_container = get_service_container()
#     embedding_model = service_container.embedding_service
#     llm_model = service_container.azure_openai_model

#     session_data = service_container.session_manager.get_session(session_id)
#     if not session_data:
#         return PlayBookReviewFinalResponse(rules_review=[], missing_clauses=None)

#     # Load cached rules for this agent
#     agent_cache = session_data.tool_results.get(AGENT_NAME, {})
#     cached_rules_review: Dict[str, PlayBookReviewResponse] = {r.rule_title: r for r in agent_cache.get("rules_review", [])}

#     # Determine which rules need to be updated
#     rules_to_update: List[RuleCheckRequest] = []

#     for rule in request.rulesinformation:
#         cached_rule = cached_rules_review.get(rule.title)

#         # Force update if explicitly requested
#         if force_update_rules and rule.title in force_update_rules:
#             rules_to_update.append(rule)
#             continue

#         # Update if rule is new
#         if not cached_rule:
#             rules_to_update.append(rule)
#             continue

#         # Update if description or instruction has changed
#         if cached_rule.rule_description != rule.description or cached_rule.rule_instruction != rule.instruction:
#             rules_to_update.append(rule)

#     if not rules_to_update:
#         # Nothing to update, return cached results
#         return PlayBookReviewFinalResponse(
#             rules_review=list(cached_rules_review.values()),
#             missing_clauses=agent_cache.get("missing_clauses"),
#         )

#     # Generate embeddings for rules to update and all paragraphs
#     rule_texts = [f"title: {rule.title}. description: {rule.description}. tags: {', '.join(rule.tags)}" for rule in rules_to_update]
#     rule_embeddings = np.array(await asyncio.gather(*[embedding_model.generate_embeddings(text) for text in rule_texts]))
#     para_embeddings = np.array(await asyncio.gather(*[embedding_model.generate_embeddings(item.text) for item in request.textinformation]))

#     # Process each rule to update
#     for rule, rule_emb in zip(rules_to_update, rule_embeddings):
#         matched: List[ParaSimilarity] = find_similarity(
#             rule_embedd=rule_emb,
#             para_embedds=para_embeddings,
#             para_items=request.textinformation,
#             top_k=3,
#             threshold=0.30,
#         )

#         if not matched:
#             result = RuleResult(
#                 title=rule.title,
#                 instruction=rule.instruction,
#                 description="No relevant contract paragraphs found.",
#                 paragraphidentifier="",
#                 paragraphcontext="",
#                 similarity_scores=[],
#             )
#         else:
#             para_ids = ",".join(m["paragraph"].paraindetifier for m in matched)
#             para_context = "\n\n".join(f"[{m['paragraph'].paraindetifier}] {m['paragraph'].text.strip()}" for m in matched)
#             similarity_scores = [m["similarity"] for m in matched]
#             result = RuleResult(
#                 title=rule.title,
#                 instruction=rule.instruction,
#                 description=rule.description,
#                 paragraphidentifier=para_ids,
#                 paragraphcontext=para_context,
#                 similarity_scores=similarity_scores,
#             )

#         # Call LLM for this updated rule
#         context: Dict[str, Any] = {
#             "rule_title": result.title,
#             "rule_instruction": result.instruction,
#             "rule_description": result.description,
#             "paragraphs": result.paragraphcontext,
#         }

#         llm_result: PlayBookReviewLLMResponse = await llm_model.generate(
#             prompt=similarity_prompt_template,
#             context=context,
#             response_model=PlayBookReviewLLMResponse,
#         )

#         response_item = PlayBookReviewResponse(
#             rule_title=result.title,
#             rule_instruction=result.instruction,
#             rule_description=result.description,
#             content=llm_result,
#         )

#         # Update cached review
#         cached_rules_review[rule.title] = response_item

#     # Update missing clauses using all paragraph identifiers in cached rules
#     all_paragraph_data = " ".join(pid for r in cached_rules_review.values() if r.content.para_identifiers for pid in r.content.para_identifiers)
#     missing_clauses: MissingClausesLLMResponse = await get_missing_clauses(data=all_paragraph_data)

#     # Save back to session
#     session_data.tool_results[AGENT_NAME] = {
#         "rules_review": list(cached_rules_review.values()),
#         "missing_clauses": missing_clauses,
#     }

#     return PlayBookReviewFinalResponse(
#         rules_review=list(cached_rules_review.values()),
#         missing_clauses=missing_clauses,
#     )

import asyncio
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

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

similarity_prompt_template = Path(r"src\services\prompts\v1\ai_review_prompt_v2.mustache").read_text(encoding="utf-8")

AGENT_NAME = "playbook_review_agent"
SIMILARITY_THRESHOLD = 0.35

# Protect shared cache against async race conditions
embedding_cache: Dict[str, np.ndarray] = {}
_embedding_locks: Dict[str, asyncio.Lock] = {}


def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _doc_hash(text_items: List[TextInfo]) -> str:
    """Stable hash of the full document paragraph list."""
    combined = "".join(item.text for item in text_items)
    return _hash(combined)


def _build_reviewed_rules_summary(cached_rules_review: Dict[str, PlayBookReviewResponse]) -> str:
    """Build a summary string of the reviewed rules to pass to the missing clauses LLM, so it knows which topics have already been reviewed and should not be re-flagged as missing."""

    lines: List[str] = []
    for title, review in cached_rules_review.items():
        para_ids = ", ".join(review.content.para_identifiers) or "none"
        lines.append(f"RULE: {title} | STATUS: {review.content.status} | PARAS: {para_ids}")
    return "\n".join(lines) if lines else "None"


async def get_missing_clauses(data: str, reviewed_rules_summary: str) -> MissingClausesLLMResponse:
    """Evaluate the document for genuinely absent clauses."""

    service_container = get_service_container()
    llm_model = service_container.azure_openai_model

    prompt = Path(r"src\services\prompts\v1\missing_clauses.mustache").read_text(encoding="utf-8")

    try:
        response: MissingClausesLLMResponse = await llm_model.generate(
            prompt=prompt,
            context={
                "data": data,
                "reviewed_rules_summary": reviewed_rules_summary,
            },
            response_model=MissingClausesLLMResponse,
        )
        logger.info(f"Identified {len(response.missing_clauses)} missing clauses.")
        return response
    except Exception as exc:
        logger.error(f"Missing clauses LLM call failed: {exc}")
        return MissingClausesLLMResponse(
            missing_clauses=[],
            total_missing=0,
            summary="Could not evaluate missing clauses due to an LLM error.",
        )

    # {
    #   "text": "Confidential Information does not include any information, however designated, that:  (i) is or subsequently becomes publicly available without Receiving Party’s breach of any obligation owed Disclosing Party; (ii) was known by Receiving Party prior to Disclosing Party’s disclosure of such information to Receiving Party pursuant to the terms of this Agreement; (iii) became known to Receiving Party from a source other than Disclosing Party other than by the breach of an obligation of confidentiality owed to Disclosing Party; or (iv) is independently developed by Receiving Party.  \r",
    #   "paraindetifier": "P0005"
    # },


async def get_embedding(embedding_model, text: str) -> np.ndarray:
    """Task-safe embedding lookup with per-key lock to prevent duplicate fetches."""

    key = _hash(text)
    if key in embedding_cache:
        return embedding_cache[key]
    if key not in _embedding_locks:
        _embedding_locks[key] = asyncio.Lock()
    async with _embedding_locks[key]:
        if key not in embedding_cache:
            embedding_cache[key] = await embedding_model.generate_embeddings(text)
    return embedding_cache[key]


def keyword_score(rule: str, para: str) -> float:
    rule_tokens = set(rule.lower().split())
    para_tokens = set(para.lower().split())
    if not rule_tokens:
        return 0.0
    return len(rule_tokens & para_tokens) / len(rule_tokens)


def hybrid_score(cosine: float, keyword: float) -> float:
    return 0.75 * cosine + 0.25 * keyword


async def get_matching_pairs_faiss(request: RuleCheckRequest) -> List[RuleResult]:
    """Get the matching pairs for the given rules with FAISS."""

    service_container = get_service_container()
    faiss_db = service_container.faiss_store
    embedding_model = service_container.embedding_service

    faiss_db.reset_index()

    para_map: List[TextInfo] = []
    for item in request.textinformation:
        emb = await get_embedding(embedding_model, item.text)
        await faiss_db.index_embedding(emb, metadata=item.paraidentifier)
        para_map.append(item)

    results: List[RuleResult] = []
    for rule in request.rulesinformation:
        rule_text = f"TITLE: {rule.title}\n" f"INSTRUCTION: {rule.instruction}\n" f"DESCRIPTION: {rule.description}\n" f"TAGS: {', '.join(rule.tags)}"

        rule_emb = await get_embedding(embedding_model, rule_text)
        faiss_result = await faiss_db.search_index(rule_emb, top_k=5)

        indices = faiss_result.get("indices", [])
        scores = faiss_result.get("scores", [])

        matched_pairs: List[Tuple[TextInfo, float]] = []
        for idx, score in zip(indices, scores):
            if idx == -1 or idx >= len(para_map):
                continue
            para = para_map[idx]
            kw = keyword_score(rule_text, para.text)
            final_score = hybrid_score(float(score), kw)
            if final_score >= SIMILARITY_THRESHOLD:
                matched_pairs.append((para, final_score))

        if not matched_pairs:
            results.append(
                RuleResult(
                    title=rule.title,
                    instruction=rule.instruction,
                    description="No relevant contract paragraphs found.",
                    paragraphidentifier="",
                    paragraphcontext="",
                    similarity_scores=[],
                )
            )
            continue

        results.append(
            RuleResult(
                title=rule.title,
                instruction=rule.instruction,
                description=rule.description,
                paragraphidentifier=",".join(p.paraidentifier for p, _ in matched_pairs),
                paragraphcontext="\n\n".join(f"PARA_ID: {p.paraidentifier}\nTEXT: {p.text.strip()}" for p, _ in matched_pairs),
                similarity_scores=[s for _, s in matched_pairs],
            )
        )

    return results


async def _process_single_rule(rule: RuleInfo, para_embeddings: np.ndarray, request: RuleCheckRequest, llm_model, embedding_model) -> Tuple[str, PlayBookReviewResponse]:
    """Process a single rule: compute similarities, find matches, and call LLM for review. Returns the rule title and its corresponding PlayBookReviewResponse."""

    rule_text = f"TITLE: {rule.title}\n" f"INSTRUCTION: {rule.instruction}\n" f"DESCRIPTION: {rule.description}\n"  # f"TAGS: {', '.join(rule.tags)}"

    rule_emb = await get_embedding(embedding_model, rule_text)

    norms_rule = rule_emb / (np.linalg.norm(rule_emb) + 1e-10)
    norms_para = para_embeddings / (np.linalg.norm(para_embeddings, axis=1, keepdims=True) + 1e-10)
    cosine_scores = norms_para @ norms_rule

    top_idx = np.argsort(cosine_scores)[::-1][:5]

    matched: List[Tuple[TextInfo, float]] = []
    for idx in top_idx:
        para = request.textinformation[idx]
        kw = keyword_score(rule_text, para.text)
        score = hybrid_score(float(cosine_scores[idx]), kw)
        if score >= SIMILARITY_THRESHOLD:
            matched.append((para, score))

    if not matched:
        result = RuleResult(
            title=rule.title,
            instruction=rule.instruction,
            description="No relevant contract paragraphs found.",
            paragraphidentifier="",
            paragraphcontext="",
            similarity_scores=[],
        )
    else:
        result = RuleResult(
            title=rule.title,
            instruction=rule.instruction,
            description=rule.description,
            paragraphidentifier=",".join(p.paraindetifier for p, _ in matched),
            paragraphcontext="\n\n".join(f"PARA_ID: {p.paraindetifier}\nTEXT: {p.text.strip()}" for p, _ in matched),
            similarity_scores=[s for _, s in matched],
        )

    context = {
        "rule_title": result.title,
        "rule_instruction": result.instruction,
        "rule_description": result.description,
        "paragraphs": result.paragraphcontext,
    }

    try:
        llm_result: PlayBookReviewLLMResponse = await llm_model.generate(
            prompt=similarity_prompt_template,
            context=context,
            response_model=PlayBookReviewLLMResponse,
        )
    except Exception as exc:
        logger.error(f"LLM review failed for rule '{rule.title}': {exc}")
        llm_result = PlayBookReviewLLMResponse(
            para_identifiers=[p.paraidentifier for p, _ in matched],
            status="Error",
            reason=f"LLM call failed: {exc}",
            suggestion="",
            suggested_fix="",
        )

    return rule.title, PlayBookReviewResponse(
        rule_title=result.title,
        rule_instruction=result.instruction,
        rule_description=result.description,
        content=llm_result,
    )


async def review_document(session_id: str, request: RuleCheckRequest, force_update_rules: Optional[List[str]] = None) -> PlayBookReviewFinalResponse:
    """Run playbook review for the given document and rules, with optional force update for specific rules."""

    force_update_rules = force_update_rules or []

    service_container = get_service_container()
    embedding_model = service_container.embedding_service
    llm_model = service_container.azure_openai_model

    session_data = service_container.session_manager.get_session(session_id)
    if not session_data:
        return PlayBookReviewFinalResponse(rules_review=[], missing_clauses=None)

    agent_cache = session_data.tool_results.get(AGENT_NAME, {})
    cached_rules_review: Dict[str, PlayBookReviewResponse] = {r.rule_title: r for r in agent_cache.get("rules_review", [])}

    # Determine which rules need (re)processing
    rules_to_update: List[RuleInfo] = []
    for rule in request.rulesinformation:
        cached = cached_rules_review.get(rule.title)
        if force_update_rules and rule.title in force_update_rules:
            rules_to_update.append(rule)
            continue
        if not cached:
            rules_to_update.append(rule)
            continue
        if cached.rule_description != rule.description or cached.rule_instruction != rule.instruction:
            rules_to_update.append(rule)

    # Return fully-cached result when nothing changed
    if not rules_to_update:
        return PlayBookReviewFinalResponse(
            rules_review=list(cached_rules_review.values()),
            missing_clauses=agent_cache.get("missing_clauses"),
        )

    # Precompute all paragraph embeddings once, concurrently
    para_embeddings = np.array(await asyncio.gather(*[get_embedding(embedding_model, item.text) for item in request.textinformation]))

    # Process all stale rules concurrently
    rule_results = await asyncio.gather(*[_process_single_rule(rule, para_embeddings, request, llm_model, embedding_model) for rule in rules_to_update])
    cached_rules_review.update(dict(rule_results))

    # Use ALL paragraph text — not just matched ones — so nothing is
    # invisible to the missing-clauses LLM
    all_text = " ".join(p.text for p in request.textinformation)

    # Only re-run missing-clauses when the document itself changed
    current_doc_hash = _doc_hash(request.textinformation)
    cached_doc_hash = agent_cache.get("doc_hash")

    if current_doc_hash != cached_doc_hash:
        logger.info("Document changed — re-evaluating missing clauses.")

        # Pass the full rules-review summary so the missing-clauses LLM doesn't
        # double-flag topics already handled by the rules review
        reviewed_rules_summary = _build_reviewed_rules_summary(cached_rules_review)

        missing_clauses = await get_missing_clauses(
            data=all_text,
            reviewed_rules_summary=reviewed_rules_summary,
        )
    else:
        logger.info("Document unchanged — reusing cached missing clauses.")
        missing_clauses = agent_cache.get("missing_clauses")

    session_data.tool_results[AGENT_NAME] = {
        "rules_review": list(cached_rules_review.values()),
        "missing_clauses": missing_clauses,
        "doc_hash": current_doc_hash,
    }

    return PlayBookReviewFinalResponse(
        rules_review=list(cached_rules_review.values()),
        missing_clauses=missing_clauses,
    )
