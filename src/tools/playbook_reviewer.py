"""
Playbook Review Tool — evaluates a contract against playbook rules.

Pipeline:
  1. Load rules from JSON
  2. Validate session has an ingested document
  3. For each rule: embed rule text -> search FAISS -> retrieve paragraphs (no LLM)
  4. For each rule (parallel): LLM evaluates rule against paragraphs
  5. Aggregate results deterministically into a report (no LLM)

Total LLM calls: N (one per rule, parallelized with asyncio.gather)
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.dependencies import get_service_container
from src.schemas.playbook import (
    PlaybookReviewReport,
    PlaybookRule,
    ReportStatistics,
    RiskLevel,
    RuleEvaluation,
    RuleResult,
    Verdict,
)
from src.services.playbook_loader import load_playbook_rules
from src.services.session_manager import SessionData

# Constants
PARAGRAPHS_PER_RULE = 8
SIMILARITY_THRESHOLD = 0.25
MAX_CONCURRENT_EVALS = 5

_PROMPT_PATH = Path("src/services/prompts/v1/rule_evaluation_v2_prompt.mustache")

_SYSTEM_MESSAGE = (
    "You are an expert Contract Review Analyst. Follow ALL instructions precisely. "
    "Evaluate the rule against the provided paragraphs. Return ONLY valid JSON "
    "matching the schema. Every claim must be grounded in exact document text."
)


# --- Session Validation ---


def _get_session(session_id: str) -> SessionData:
    """Retrieve session data or raise ValueError."""
    container = get_service_container()
    session = container.session_manager.get_session(session_id)
    if not session:
        raise ValueError(f"Session '{session_id}' not found or expired.")
    if len(session.chunk_store) == 0:
        raise ValueError("No document ingested in this session.")
    return session


# --- Paragraph Retrieval (no LLM) ---


async def _retrieve_paragraphs_for_rule(
    rule: PlaybookRule,
    session: SessionData,
    top_k: int = PARAGRAPHS_PER_RULE,
    threshold: float = SIMILARITY_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Embed rule text, search session FAISS, return relevant paragraphs."""
    container = get_service_container()
    embedding_service = container.embedding_service

    search_text = (
        f"Rule: {rule.title}\n"
        f"Instruction: {rule.instruction}\n"
        f"Description: {rule.description}"
    )

    query_embedding = await embedding_service.generate_embeddings(text=search_text, task="retrieval.query")
    search_result = await session.vector_store.search_index(query_embedding, top_k=top_k)

    indices = search_result.get("indices", [])
    scores = search_result.get("scores", [])

    paragraphs = []
    for idx, score in zip(indices, scores):
        if idx < 0:  # FAISS returns -1 for unfilled slots
            continue
        if score < threshold:
            continue
        chunk = session.chunk_store.get(idx)
        if chunk and chunk.content:
            paragraphs.append({
                "chunk_index": idx,
                "content": chunk.content,
                "similarity_score": float(score),
            })

    return paragraphs


# --- Per-Rule LLM Evaluation ---


def _sync_generate(llm: Any, prompt_template: str, context: Dict[str, Any]) -> RuleEvaluation:
    """Synchronous LLM call for use with asyncio.to_thread() (parallel execution)."""
    import json as _json

    rendered_prompt = llm.render_prompt_template(prompt=prompt_template, context=context)

    response = llm.client.chat.completions.create(
        model=llm.deployment_name,
        messages=[
            {"role": "system", "content": _SYSTEM_MESSAGE},
            {"role": "user", "content": rendered_prompt},
        ],
        temperature=0.0,
        max_tokens=16384,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": RuleEvaluation.__name__,
                "schema": RuleEvaluation.model_json_schema(),
                "strict": False,
            },
        },
    )

    response_text = response.choices[0].message.content
    if response_text is None:
        raise ValueError("Empty response from LLM model.")

    return RuleEvaluation.model_validate(_json.loads(response_text))


def _format_paragraphs_for_prompt(paragraphs: List[Dict[str, Any]]) -> str:
    """Format retrieved paragraphs into the prompt template text block."""
    parts = []
    for p in paragraphs:
        parts.append(
            f"--- Paragraph ID: {p['chunk_index']} "
            f"(similarity: {p['similarity_score']:.3f}) ---\n"
            f"{p['content']}"
        )
    return "\n\n".join(parts)


async def _evaluate_rule(
    rule: PlaybookRule,
    paragraphs: List[Dict[str, Any]],
) -> Tuple[RuleEvaluation, Optional[str]]:
    """Evaluate one rule against retrieved paragraphs using the LLM.

    Returns (RuleEvaluation, error_message_or_None).
    """
    # No paragraphs → NOT FOUND without LLM call
    if not paragraphs:
        return (
            RuleEvaluation(
                _reasoning="No paragraphs retrieved from the document for this rule.",
                rule_title=rule.title,
                rule_instruction=rule.instruction,
                rule_description=rule.description,
                para_identifiers=[],
                status=Verdict.NOT_FOUND,
                reason=f"No relevant paragraphs found for rule '{rule.title}'.",
                suggestion=f"Add a clause addressing: {rule.instruction}",
                suggested_fix=f"Consider adding a clause addressing: {rule.description[:300]}",
                confidence=0.0,
                risk_level=RiskLevel.MEDIUM,
            ),
            None,
        )

    container = get_service_container()
    llm = container.azure_openai_model
    prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")

    context = {
        "rule_title": rule.title,
        "rule_instruction": rule.instruction,
        "rule_description": rule.description,
        "paragraphs_text": _format_paragraphs_for_prompt(paragraphs),
    }

    try:
        evaluation = await asyncio.to_thread(_sync_generate, llm, prompt_template, context)
        return evaluation, None
    except Exception as e:
        return (
            RuleEvaluation(
                _reasoning=f"Evaluation failed: {str(e)}",
                rule_title=rule.title,
                rule_instruction=rule.instruction,
                rule_description=rule.description,
                para_identifiers=[],
                status=Verdict.NOT_FOUND,
                reason=f"Evaluation error for rule '{rule.title}': {str(e)}",
                suggestion="",
                suggested_fix="",
                confidence=0.0,
                risk_level=RiskLevel.MEDIUM,
            ),
            str(e),
        )


# --- Deterministic Report Assembly ---


def _compute_overall_risk(results: List[RuleResult]) -> RiskLevel:
    """Determine overall risk: CRITICAL > HIGH > 3+ MEDIUM → HIGH > MEDIUM > LOW."""
    risk_counts = {r: 0 for r in RiskLevel}
    for result in results:
        risk_counts[result.risk_level] += 1

    if risk_counts[RiskLevel.CRITICAL] > 0:
        return RiskLevel.CRITICAL
    if risk_counts[RiskLevel.HIGH] > 0:
        return RiskLevel.HIGH
    if risk_counts[RiskLevel.MEDIUM] >= 3:
        return RiskLevel.HIGH
    if risk_counts[RiskLevel.MEDIUM] > 0:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _build_report(
    session_id: str,
    playbook_name: str,
    rule_results: List[RuleResult],
    errors: List[str],
) -> PlaybookReviewReport:
    """Assemble the final report deterministically (no LLM call)."""
    stats = ReportStatistics(
        total_rules=len(rule_results),
        rules_passed=sum(1 for r in rule_results if r.status == Verdict.PASS),
        rules_failed=sum(1 for r in rule_results if r.status == Verdict.FAIL),
        rules_not_found=sum(1 for r in rule_results if r.status == Verdict.NOT_FOUND),
        critical_count=sum(1 for r in rule_results if r.risk_level == RiskLevel.CRITICAL),
        high_count=sum(1 for r in rule_results if r.risk_level == RiskLevel.HIGH),
        medium_count=sum(1 for r in rule_results if r.risk_level == RiskLevel.MEDIUM),
        low_count=sum(1 for r in rule_results if r.risk_level == RiskLevel.LOW),
    )

    rules_by_risk: Dict[str, List[str]] = {}
    for level in RiskLevel:
        titles = [r.rule_title for r in rule_results if r.risk_level == level]
        if titles:
            rules_by_risk[level.value] = titles

    missing_clauses = [r.rule_title for r in rule_results if r.status == Verdict.NOT_FOUND]

    # Sort: CRITICAL first, then HIGH, MEDIUM, LOW
    risk_order = {RiskLevel.CRITICAL: 0, RiskLevel.HIGH: 1, RiskLevel.MEDIUM: 2, RiskLevel.LOW: 3}
    sorted_results = sorted(rule_results, key=lambda r: risk_order[r.risk_level])

    return PlaybookReviewReport(
        session_id=session_id,
        playbook_source=playbook_name,
        statistics=stats,
        overall_risk_level=_compute_overall_risk(rule_results),
        rule_results=sorted_results,
        rules_by_risk=rules_by_risk,
        missing_clauses=missing_clauses,
        errors=errors,
    )


# --- Main Entry Points ---


async def full_playbook_review(
    session_id: str,
    playbook_name: str = "v3",
) -> PlaybookReviewReport:
    """Run a full playbook review of the contract in the given session.

    Pipeline: load rules -> validate session -> retrieve paragraphs (parallel)
    -> evaluate rules (parallel LLM) -> assemble report (deterministic).
    """
    rules = load_playbook_rules(playbook_name)
    session = _get_session(session_id)

    # Retrieve paragraphs for each rule (parallel, no LLM)
    retrieval_tasks = [_retrieve_paragraphs_for_rule(rule, session) for rule in rules]
    all_paragraphs = await asyncio.gather(*retrieval_tasks)

    # Evaluate each rule (parallel LLM, with concurrency limit)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_EVALS)

    async def _eval_with_limit(rule: PlaybookRule, paragraphs: List[Dict[str, Any]]) -> Tuple[RuleEvaluation, Optional[str]]:
        async with semaphore:
            return await _evaluate_rule(rule, paragraphs)

    eval_tasks = [_eval_with_limit(rule, paras) for rule, paras in zip(rules, all_paragraphs)]
    eval_results = await asyncio.gather(*eval_tasks)

    # Build results and collect errors
    rule_results: List[RuleResult] = []
    errors: List[str] = []

    for rule, paragraphs, (evaluation, error) in zip(rules, all_paragraphs, eval_results):
        if error:
            errors.append(f"{rule.title}: {error}")

        rule_results.append(
            RuleResult(
                rule_title=evaluation.rule_title,
                rule_instruction=evaluation.rule_instruction,
                rule_description=evaluation.rule_description,
                para_identifiers=evaluation.para_identifiers,
                category=rule.category,
                status=evaluation.status,
                risk_level=evaluation.risk_level,
                confidence=evaluation.confidence,
                reason=evaluation.reason,
                suggestion=evaluation.suggestion,
                suggested_fix=evaluation.suggested_fix,
                paragraphs_retrieved=len(paragraphs),
            )
        )

    return _build_report(session_id, playbook_name, rule_results, errors)


async def single_rule_review(
    session_id: str,
    rule_title: str,
    playbook_name: str = "v3",
) -> RuleResult:
    """Review a single rule against the contract (case-insensitive partial match)."""
    rules = load_playbook_rules(playbook_name)

    matching_rule = None
    for rule in rules:
        if rule_title.lower() in rule.title.lower():
            matching_rule = rule
            break

    if not matching_rule:
        raise ValueError(f"Rule '{rule_title}' not found. Available: {[r.title for r in rules]}")

    session = _get_session(session_id)
    paragraphs = await _retrieve_paragraphs_for_rule(matching_rule, session)
    evaluation, error = await _evaluate_rule(matching_rule, paragraphs)

    return RuleResult(
        rule_title=evaluation.rule_title,
        rule_instruction=evaluation.rule_instruction,
        rule_description=evaluation.rule_description,
        para_identifiers=evaluation.para_identifiers,
        category=matching_rule.category,
        status=evaluation.status,
        risk_level=evaluation.risk_level,
        confidence=evaluation.confidence,
        reason=evaluation.reason,
        suggestion=evaluation.suggestion,
        suggested_fix=evaluation.suggested_fix,
        paragraphs_retrieved=len(paragraphs),
    )
