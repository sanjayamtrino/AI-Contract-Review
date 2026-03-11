import asyncio
from pathlib import Path
from typing import Any, Dict, List
from urllib import response

import numpy as np

from src.config.logging import get_logger
from src.dependencies import get_service_container
from src.schemas.rule_check import MissingClausesLLMResponse, ParaSimilarity, RuleCheckRequest, RuleResult, TextInfo

logger = get_logger(__name__)


async def get_missing_clauses(text_items: List[TextInfo], rule_titles: List[str]) -> MissingClausesLLMResponse:
    """
    Detect missing clauses — hybrid approach.
    Get the missing clauses for the given contract text.
    """
    service_container = get_service_container()
    llm_model = service_container.azure_openai_model

    # Label every paragraph with its ID so LLM returns accurate para_identifiers
    labeled_paragraphs = "\n\n".join(f"[{item.paraindetifier}] {item.text.strip()}" for item in text_items if item.text.strip())

    # Format playbook titles as numbered list for the prompt
    playbook_clauses = "\n".join(f"{i+1}. {title}" for i, title in enumerate(rule_titles))

    prompt = Path(r"src\services\prompts\v1\missing_clauses.mustache").read_text(encoding="utf-8")
    context: Dict[str, Any] = {
        "data": labeled_paragraphs,
        "playbook_clauses": playbook_clauses,
    }

    response: MissingClausesLLMResponse = await llm_model.generate(
        prompt=prompt,
        context=context,
        response_model=MissingClausesLLMResponse,
    )
    # Extract missing clauses from the structured response

    truly_missing = [c for c in response.missing_clauses if c.is_missing]

    logger.info(f"Missing clause check complete — " f"{len(truly_missing)} missing clauses detected: " f"{[c.clause_name for c in truly_missing] or 'none'}")

    response.missing_clauses = truly_missing
    response.total_missing = len(truly_missing)

    return response


"""
    truly_missing = [c for c in response.missing_clauses if c.is_missing]

    logger.info(f"Missing clause check complete — " f"{len(truly_missing)} absent out of {len(response.missing_clauses)} clauses: " f"{[c.clause_name for c in truly_missing] or 'none'}")

    return response
"""


async def get_matching_pairs_faiss(request: RuleCheckRequest) -> List[RuleResult]:
    """Get the matching pairs for the given rules with FAISS."""

    service_container = get_service_container()
    faiss_db = service_container.faiss_store
    embedding_model = service_container.embedding_service
    # Index all paragraph embeddings into FAISS
    for item in request.textinformation:
        embedd_vector = await embedding_model.generate_embeddings(item.text)
        logger.info(f"Indexing paragraph {item.paraindetifier} into FAISS.")
        await faiss_db.index_embedding(embedd_vector)

    results: List[RuleResult] = []

    for rule in request.rulesinformation:
        rule_text = f"title: {rule.title}. " f"description: {rule.description}. "
        logger.info(f"Generating embedding for rule '{rule.title}'.")
        rule_embedds = await embedding_model.generate_embeddings(rule_text)
        logger.info(f"Searching for similar paragraphs in FAISS for rule '{rule.title}'.")
        faiss_result: Dict[str, Any] = await faiss_db.search_index(rule_embedds, top_k=3)

        indices = faiss_result.get("indices", [])
        scores = faiss_result.get("scores", [])
        # Filter out invalid FAISS indices (-1 means no result found)
        matched_pairs: List[ParaSimilarity] = [(idx, score) for idx, score in zip(indices, scores) if idx != -1 and idx < len(request.textinformation)]

        if not matched_pairs:
            logger.info(f"No relevant paragraphs found in FAISS for rule '{rule.title}'.")
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

        matched_paras = [request.textinformation[idx] for idx, _ in matched_pairs]
        similarity_scores = [float(score) for _, score in matched_pairs]
        para_ids = ",".join(p.paraindetifier for p in matched_paras)
        para_context = "\n\n".join(f"[{p.paraindetifier}] {p.text.strip()}" for p in matched_paras)

        results.append(
            RuleResult(
                title=rule.title,
                instruction=rule.instruction,
                description=rule.description,
                paragraphidentifier=para_ids,
                paragraphcontext=para_context,
                similarity_scores=similarity_scores,
            )
        )

    return results


def find_similarity(
    rule_embedd: np.ndarray,
    para_embedds: np.ndarray,
    para_items: List[TextInfo],
    top_k: int = 4,
    threshold: float = 0.25,
) -> List[ParaSimilarity]:
    """Get the similar paragraphs for the given rule.

    Evaluates ALL paragraphs against threshold first.
    top_k only limits the final output — not which paragraphs are evaluated.
    """
    logger.info("Normalizing embeddings for similarity computation.")
    # Normalize safely
    rules_norm = rule_embedd / (np.linalg.norm(rule_embedd) + 1e-10)
    para_norms = para_embedds / (np.linalg.norm(para_embedds, axis=1, keepdims=True) + 1e-10)

    logger.info("Computing cosine similarity between rule and paragraph embeddings.")
    # compute cosine similarity
    scores = para_norms @ rules_norm

    logger.info("Sorting similarity scores and filtering based on threshold.")
    # Sort indices descending
    all_indices_sorted = np.argsort(scores)[::-1]  # ALL paragraphs — no rank cutoff

    results: List[ParaSimilarity] = []
    for idx in all_indices_sorted:
        if len(results) >= top_k:
            break
        if scores[idx] >= threshold:
            results.append(
                {
                    "paragraph": para_items[idx],
                    "similarity": float(scores[idx]),
                }
            )
        else:
            break  # sorted descending — nothing below will pass

    logger.info(f"Found {len(results)} paragraphs with similarity above threshold of {threshold}.")
    return results


async def get_matching_paras(request: RuleCheckRequest) -> List[RuleResult]:
    """Get the matching paras for the given rules."""

    service_container = get_service_container()
    embedding_model = service_container.embedding_service

    rule_texts = [f"title: {rule.title}. " f"instruction: {rule.instruction}. " f"description: {rule.description}. " f"tags: {', '.join(rule.tags or [])}" for rule in request.rulesinformation]

    logger.info("Generating embeddings for rules and paragraphs.")
    rule_embeddings = np.array(await asyncio.gather(*[embedding_model.generate_embeddings(text) for text in rule_texts]))
    para_embeddings = np.array(await asyncio.gather(*[embedding_model.generate_embeddings(item.text) for item in request.textinformation]))

    results: List[RuleResult] = []

    for rule, rule_emb in zip(request.rulesinformation, rule_embeddings):
        logger.info(f"Finding similar paragraphs for rule '{rule.title}'.")

        matched: List[ParaSimilarity] = find_similarity(
            rule_embedd=rule_emb,
            para_embedds=para_embeddings,
            para_items=request.textinformation,
            top_k=3,
            threshold=0.30,
        )

        if not matched:
            logger.info(f"No relevant paragraphs found for rule '{rule.title}'.")
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

        para_ids = ",".join(m["paragraph"].paraindetifier for m in matched)
        para_context = "\n\n".join(f"[{m['paragraph'].paraindetifier}] {m['paragraph'].text.strip()}" for m in matched)
        similarity_scores = [m["similarity"] for m in matched]

        results.append(
            RuleResult(
                title=rule.title,
                instruction=rule.instruction,
                description=rule.description,
                paragraphidentifier=para_ids,
                paragraphcontext=para_context,
                similarity_scores=similarity_scores,
            )
        )

    return results
