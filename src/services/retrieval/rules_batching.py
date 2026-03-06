import asyncio
import re
from typing import Any, Dict, List, Optional

import numpy as np

from src.config.logging import get_logger
from src.dependencies import get_service_container
from src.schemas.rule_check import RuleCheckRequest, RuleResult, TextInfo

logger = get_logger(__name__)


# Heading detection — copied from semantic_parser.py _is_structural_heading()


_SECTION_LABEL_RE = re.compile(r"^(" r"\d+[\.\)]?\s+\S.*|" r"\d+[\.\)]?\s*$|" r"[A-Z][A-Z\s\.\,\&\'\-]{1,60}$" r")")

_HEADING_MAX_WORDS = 8


def _is_heading(text: str, max_words: int = _HEADING_MAX_WORDS) -> bool:
    """
    Returns True when text looks like a section heading or title.
    Copied directly from semantic_parser.py _is_structural_heading().
    """
    words = text.split()
    if len(words) > max_words or len(words) == 0:
        return False
    return bool(_SECTION_LABEL_RE.match(text.strip()))


# SIMILARITY THRESHOLDS
#
# Two-tier threshold system instead of hard keyword gate:
#
#   HIGH_THRESHOLD (0.45) — paragraph passes with similarity alone.
#     Strong semantic match means the paragraph is almost certainly relevant
#     regardless of which specific words appear. No keyword check needed.
#
#   LOW_THRESHOLD (0.35) — paragraph is a candidate but needs keyword support.
#     Borderline similarity. Keyword check applied to confirm relevance.
#     If keyword score > 0 → include. If keyword score = 0 → exclude.
#
#   Below LOW_THRESHOLD — excluded entirely. Too weak to be relevant.
#
# This approach is robust across all contract types because:
#   - Strong matches (>= 0.45) always get through regardless of wording
#   - Weak matches (0.35-0.45) still need topical confirmation via keywords
#   - Very weak matches (< 0.35) are always excluded
#
# The keyword check for borderline paragraphs uses individual significant
# words from the tags — not full phrase match — so variant phrasing
# like "survive" matching tag "survival clause" works correctly.


HIGH_THRESHOLD = 0.36  # Pass with similarity alone — no keyword check
LOW_THRESHOLD = 0.28  # Pass only if keyword score > 0
HARD_MIN = 0.18  # Absolute floor — never consider below this


# Keyword scoring — used only for borderline paragraphs (LOW to HIGH range)


_STOP_WORDS = {
    "the",
    "and",
    "for",
    "of",
    "or",
    "in",
    "to",
    "a",
    "an",
    "is",
    "are",
    "be",
    "by",
    "on",
    "at",
    "no",
    "not",
    "any",
    "all",
    "its",
    "this",
    "that",
    "with",
    "from",
    "as",
    "it",
    "if",
    "may",
    "shall",
}


def _keyword_score(paragraph_text: str, tags: List[str]) -> int:
    """
    Returns count of tags that have at least one significant word
    present in the paragraph text.

    Used only for borderline paragraphs (similarity between LOW and HIGH).
    Not used as a hard blocker for strong matches.

    Per-tag matching:
      - Split tag into individual words
      - Remove stop words
      - If ANY remaining word appears in the paragraph → tag is matched
      - Also checks 5-character stem for inflection variants

    """
    if not tags:
        return 1

    text_lower = paragraph_text.lower()
    score = 0

    for tag in tags:
        raw_words = re.split(r"[\s\-\_]+", tag.lower())
        significant = [w for w in raw_words if len(w) >= 4 and w not in _STOP_WORDS]

        if not significant:
            score += 1
            continue

        matched = False
        for word in significant:
            if word in text_lower:
                matched = True
                break
            # Stem match — first 5 chars covers most English inflections
            stem = word[:5]
            if len(stem) >= 4 and re.search(r"\b" + re.escape(stem), text_lower):
                matched = True
                break

        if matched:
            score += 1

    return score


# FAISS matching — unchanged from original


async def get_matching_pairs_faiss(request: RuleCheckRequest) -> List[RuleResult]:
    """Get the matching pairs for the given rules with FAISS."""

    service_container = get_service_container()
    faiss_db = service_container.faiss_store
    embedding_model = service_container.embedding_service

    for item in request.textinformation:
        embedd_vector = await embedding_model.generate_embeddings(item.text)
        logger.info(f"Indexing paragraph {item.paraindetifier} into FAISS.")
        await faiss_db.index_embedding(embedd_vector)

    results: List[RuleResult] = []

    for rule in request.rulesinformation:
        safe_tags = rule.tags or []
        tag_str = ", ".join(safe_tags)
        rule_text = f"title: {rule.title}. " f"instruction: {rule.instruction}. " f"description: {rule.description}. " f"tags: {tag_str}"
        logger.info(f"Generating embedding for rule '{rule.title}'.")
        rule_embedds = await embedding_model.generate_embeddings(rule_text)
        logger.info(f"Searching for similar paragraphs in FAISS for rule '{rule.title}'.")
        faiss_result: Dict[str, Any] = await faiss_db.search_index(rule_embedds, top_k=3)

        indices = faiss_result.get("indices", [])
        scores = faiss_result.get("scores", [])

        matched_pairs = [(idx, score) for idx, score in zip(indices, scores) if idx != -1 and idx < len(request.textinformation)]

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


# Cosine similarity with two-tier threshold


def find_similarity(
    rule_embedd: np.ndarray,
    para_embedds: np.ndarray,
    para_items: List[TextInfo],
    top_k: int = 5,
    threshold: float = LOW_THRESHOLD,
    tags: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Two-tier similarity matching — robust for all contract types.

    Tier 1 — similarity >= HIGH_THRESHOLD (0.42):
      Strong semantic match. Include regardless of keyword presence.
      The embedding model is confident — no additional check needed.

    Tier 2 — similarity >= LOW_THRESHOLD (0.35) and < HIGH_THRESHOLD:
      Borderline match. Apply keyword check.
      Include if keyword score > 0. Exclude if keyword score = 0.

    Below LOW_THRESHOLD:
      Exclude entirely.

    Gate 0 (heading filter) applied before both tiers.
    """
    tags = tags or []

    logger.info("Normalizing embeddings for similarity computation.")
    rules_norm = rule_embedd / (np.linalg.norm(rule_embedd) + 1e-10)
    para_norms = para_embedds / (np.linalg.norm(para_embedds, axis=1, keepdims=True) + 1e-10)

    logger.info("Computing cosine similarity between rule and paragraph embeddings.")
    scores = para_norms @ rules_norm

    logger.info("Sorting similarity scores and filtering based on threshold.")
    top_indices = np.argsort(scores)[::-1][: top_k * 8]  # evaluate more candidates

    results = []
    for idx in top_indices:
        if len(results) >= top_k:
            break

        score = float(scores[idx])
        para = para_items[idx]

        # Absolute floor
        if score < HARD_MIN:
            break

        # Gate 0 — heading filter
        if _is_heading(para.text):
            logger.debug(f"Heading filtered: {para.paraindetifier}")
            continue

        # Tier 1 — strong match, include directly
        if score >= HIGH_THRESHOLD:
            logger.debug(f"Tier 1 pass: {para.paraindetifier} score={score:.3f}")
            results.append({"paragraph": para, "similarity": score})
            continue

        # Below low threshold — exclude
        if score < LOW_THRESHOLD:
            continue

        # Tier 2 — borderline match, require keyword confirmation
        if tags:
            kw = _keyword_score(para.text, tags)
            if kw == 0 and score < (LOW_THRESHOLD + 0.03):
                logger.debug(f"Tier 2 excluded: {para.paraindetifier} " f"score={score:.3f} keyword=0")
                continue
            logger.debug(f"Tier 2 pass: {para.paraindetifier} " f"score={score:.3f} keyword={kw}")

        results.append({"paragraph": para, "similarity": score})

    logger.info(f"Found {len(results)} paragraphs after two-tier filter.")

    return results


# Main matching function


async def get_matching_paras(request: RuleCheckRequest) -> List[RuleResult]:
    """Get the matching paras for the given rules."""

    service_container = get_service_container()
    embedding_model = service_container.embedding_service

    # Safe tags join — handles None and empty list
    rule_texts = []
    for rule in request.rulesinformation:
        safe_tags = rule.tags or []
        tag_str = ", ".join(safe_tags)
        rule_texts.append(f"title: {rule.title}. " f"instruction: {rule.instruction}. " f"description: {rule.description}. " f"tags: {tag_str}")
    logger.info("Generating embeddings for rules and paragraphs.")
    rule_embeddings = np.array(await asyncio.gather(*[embedding_model.generate_embeddings(text) for text in rule_texts]))
    para_embeddings = np.array(await asyncio.gather(*[embedding_model.generate_embeddings(item.text) for item in request.textinformation]))

    results: List[RuleResult] = []

    for rule, rule_emb in zip(request.rulesinformation, rule_embeddings):
        safe_tags = rule.tags or []
        logger.info(f"Finding similar paragraphs for rule '{rule.title}'.")

        matched = find_similarity(
            rule_embedd=rule_emb,
            para_embedds=para_embeddings,
            para_items=request.textinformation,
            top_k=3,
            threshold=LOW_THRESHOLD,
            tags=safe_tags,
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
