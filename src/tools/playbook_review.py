import asyncio
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from src.services.llm.azure_openai_model import AzureOpenAIModel

logger = get_logger(__name__)


AGENT_NAME = "playbook_review_agent"

SIMILARITY_PROMPT = Path(r"src\services\prompts\v1\ai_review_prompt_v2.mustache").read_text(encoding="utf-8")

MISSING_CLAUSES_PROMPT = Path(r"src\services\prompts\v1\missing_clauses.mustache").read_text(encoding="utf-8")


def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _normalize(text: str) -> str:
    """Lowercase and strip all punctuation/whitespace for fuzzy matching."""
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


# Strips fallback markers like " - Fallback 1", " – fallback2", " - Fallback Special"
# off the end of a rule title. Permissive on dash type (-, –, —), spacing, and
# what follows "Fallback" (digits, words, etc.) so playbook authors can use
# whatever convention they like.
_FALLBACK_SUFFIX_PATTERN = re.compile(
    r"\s*[-–—]\s*[Ff]allback\b.*$"
)


def _canonical_title(title: str) -> str:
    """Return the rule title with any '- Fallback N' suffix stripped.

    Primary + fallback rule variants of the same clause share a canonical
    title so they match the SAME contract paragraphs. Without this, a rule
    titled 'Term/Termination - Fallback 1' would never match the contract's
    'Term/Termination' heading and would fall through to full-document mode,
    where the LLM picks paragraphs at its own discretion (non-deterministic).

    Authors can also write fallbacks with the same bare title as the primary
    (distinguishing them only by rule_type). In that case there is no suffix
    to strip and this function is a no-op.
    """
    return _FALLBACK_SUFFIX_PATTERN.sub("", title).strip()


def extract_clauses_from_paragraphs(textinformation: List[TextInfo], rule_titles: List[str]) -> Dict[str, List[TextInfo]]:
    """Extract paragraphs grouped under each rule's canonical clause heading.

    The returned map is keyed by *canonical title* (fallback suffix stripped),
    so a primary rule and any number of its fallback variants resolve to the
    SAME paragraph list when looked up.
    """

    # canonical_title → list of original rule titles (for the warn log)
    canonical_titles: Dict[str, List[str]] = {}
    for t in rule_titles:
        canonical_titles.setdefault(_canonical_title(t), []).append(t)

    # normalized canonical title → canonical title (used for paragraph matching)
    normalized_titles: Dict[str, str] = {
        _normalize(canonical): canonical for canonical in canonical_titles
    }

    # Initialize empty lists per canonical title so callers always get a key
    clause_map: Dict[str, List[TextInfo]] = {canonical: [] for canonical in canonical_titles}
    current_clause: Optional[str] = None

    for para in textinformation:
        para_norm = _normalize(para.text)
        matched_title: Optional[str] = None

        for norm_title, canonical in normalized_titles.items():
            # Exact match — standalone header paragraph (Format B)
            if para_norm == norm_title:
                matched_title = canonical
                break

            # Para starts with title — merged header+content (Format A)
            if para_norm.startswith(norm_title):
                matched_title = canonical
                break

        if matched_title:
            # This paragraph opens a new clause; include it (it may carry content)
            current_clause = matched_title
            clause_map[current_clause].append(para)
        elif current_clause is not None:
            # Body paragraph — belongs to the currently active clause
            clause_map[current_clause].append(para)
        # else: paragraph appears before any recognized clause header — skip

    # Warn on any canonical title whose clause was never found in the document
    for canonical, paras in clause_map.items():
        if not paras:
            originals = canonical_titles.get(canonical, [canonical])
            logger.warning(
                "Clause extraction: no paragraphs found for canonical title '%s' "
                "(rules: %s). These rules will be evaluated against the full "
                "document as fallback.",
                canonical, originals,
            )

    return clause_map


def _build_reviewed_rules_summary(reviewed: Dict[Tuple[str, str], PlayBookReviewResponse]) -> str:
    """Builds a concise summary of reviewed rules and their statuses for the missing clauses evaluation."""

    lines = []
    for (title, rule_type), review in reviewed.items():
        para_ids = ", ".join(review.content.para_identifiers) or "none"
        lines.append(f"RULE: {title} ({rule_type}) | STATUS: {review.content.status} | PARAS: {para_ids}")
    return "\n".join(lines) if lines else "None"


async def get_missing_clauses(llm_model: AzureOpenAIModel, full_text: str, reviewed_rules_summary: str) -> MissingClausesLLMResponse:
    """Gets missing clauses from the LLM based on the full document text and a summary of reviewed rules."""

    try:
        response: MissingClausesLLMResponse = await llm_model.generate(
            prompt=MISSING_CLAUSES_PROMPT,
            context={
                "data": full_text,
                "reviewed_rules_summary": reviewed_rules_summary,
            },
            response_model=MissingClausesLLMResponse,
        )
        logger.info(f"Missing clauses identified: {len(response.missing_clauses)}")

        return response

    except Exception as exc:
        logger.exception("Missing clauses evaluation failed.")
        return MissingClausesLLMResponse(missing_clauses=[], total_missing=0, summary=f"LLM error: {exc}")


async def _process_rule(rule: RuleInfo, clause_map: Dict[str, List[TextInfo]], full_document: List[TextInfo], llm_model: AzureOpenAIModel) -> Tuple[Tuple[str, str], PlayBookReviewResponse]:
    """Evaluates a single rule against its extracted clause paragraphs.

    Lookup is by canonical title so that primary + fallback1 + fallback2 + ...
    of the same clause all evaluate against the SAME paragraph list. They
    differ only in the LLM instruction sent for evaluation.
    """

    matched_paras: List[TextInfo] = clause_map.get(_canonical_title(rule.title), [])

    if not matched_paras:
        logger.warning(f"No extracted paragraphs for rule {rule.title}. Falling back to full document.")
        matched_paras = full_document

    paragraph_context = "\n\n".join(f"PARA_ID: {p.paraindetifier}\nTEXT: {p.text.strip()}" for p in matched_paras)

    result = RuleResult(
        title=rule.title,
        instruction=rule.instruction,
        description=rule.description,
        paragraphidentifier=",".join(p.paraindetifier for p in matched_paras),
        paragraphcontext=paragraph_context,
        similarity_scores=[],
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
        logger.exception("LLM rule evaluation failed for rule '%s'.", rule.title)
        llm_response = PlayBookReviewLLMResponse(
            para_identifiers=[],
            status="Error",
            reason=str(exc),
            suggestion="",
            suggested_fix="",
        )

    return (rule.title, rule.rule_type or "primary"), PlayBookReviewResponse(
        rule_type=rule.rule_type or "primary",
        rule_title=rule.title,
        rule_instruction=rule.instruction,
        rule_description=rule.description,
        content=llm_response,
    )


async def review_document(session_id: str, request: RuleCheckRequest, force_update_rules: Optional[List[str]] = None) -> PlayBookReviewFinalResponse:
    """Main entry point for reviewing a document against a set of rules. Supports caching and selective re-evaluation of rules based on changes in rule text or document content."""

    force_update_rules = force_update_rules or []

    container = get_service_container()
    llm_model = container.azure_openai_model  # Embeddings no longer needed

    session_data = container.session_manager.get_session(session_id)
    if not session_data:
        return PlayBookReviewFinalResponse(
            rules_review=[],
            missing_clauses=None,
        )

    agent_cache = session_data.tool_results.get(AGENT_NAME, {})
    # # Key by (title, rule_type) to preserve multiple rule variants (primary, fallback, fallback2, etc.)
    # cached_reviews: Dict[Tuple[str, str], PlayBookReviewResponse] = {(r.rule_title, r.rule_type): r for r in agent_cache.get("rules_review", [])}

    # # Determine which rules are stale and need re-evaluation
    # rules_to_update: List[RuleInfo] = []
    # for rule in request.rulesinformation:
    #     cache_key = (rule.title, rule.rule_type or "primary")
    #     cached = cached_reviews.get(cache_key)
    #     if not cached or rule.title in force_update_rules or cached.rule_description != rule.description or cached.rule_instruction != rule.instruction:
    #         rules_to_update.append(rule)

    # # Nothing changed — serve from cache
    # if not rules_to_update:
    #     logger.info("All rules up to date in cache. Returning cached results.")
    #     return PlayBookReviewFinalResponse(
    #         rules_review=list(cached_reviews.values()),
    #         missing_clauses=agent_cache.get("missing_clauses"),
    #     )

    # Process all rules (cache disabled)
    rules_to_update: List[RuleInfo] = request.rulesinformation
    cached_reviews: Dict[Tuple[str, str], PlayBookReviewResponse] = {}

    # Extract clauses once for all stale rules. The map is keyed by canonical
    # title so that primary + fallback variants of one clause share paragraphs.
    rule_titles = [rule.title for rule in rules_to_update]
    clause_map = extract_clauses_from_paragraphs(request.textinformation, rule_titles)

    matched_canonical = sum(1 for paras in clause_map.values() if paras)
    logger.info(
        "Clause extraction complete. %d/%d canonical clause titles have matched paragraphs (%d rules total).",
        matched_canonical, len(clause_map), len(rule_titles),
    )

    # Process stale rules concurrently
    updates: List[Tuple[Tuple[str, str], PlayBookReviewResponse]] = await asyncio.gather(
        *[
            _process_rule(
                rule,
                clause_map,
                request.textinformation,  # fallback
                llm_model,
            )
            for rule in rules_to_update
        ]
    )

    cached_reviews.update(dict(updates))

    # # Recompute missing clauses if document or rules changed
    # full_text = "\n\n".join(f"PARA_ID: {p.paraindetifier}\nTEXT: {p.text}" for p in request.textinformation)
    #
    # doc_hash = _hash(full_text)
    # rules_hash = _hash("".join(r.title + r.description for r in request.rulesinformation))
    #
    # cached_doc_hash = agent_cache.get("doc_hash")
    # cached_rules_hash = agent_cache.get("rules_hash")
    #
    # if doc_hash != cached_doc_hash or rules_hash != cached_rules_hash:
    #     logger.info("Document or rules changed. Re-evaluating missing clauses.")
    #     reviewed_summary = _build_reviewed_rules_summary(cached_reviews)
    #     missing_clauses = await get_missing_clauses(llm_model, full_text, reviewed_summary)
    # else:
    #     missing_clauses = agent_cache.get("missing_clauses")

    # # Persist updated results to session cache
    # session_data.tool_results[AGENT_NAME] = {
    #     "rules_review": list(cached_reviews.values()),
    #     "missing_clauses": missing_clauses,
    #     "doc_hash": doc_hash,
    #     "rules_hash": rules_hash,
    # }

    # Cache disabled - not persisting results
    missing_clauses = None

    return PlayBookReviewFinalResponse(
        rules_review=list(cached_reviews.values()),
        missing_clauses=missing_clauses,
    )
