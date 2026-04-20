"""
Describe & Draft tool — classify intent, generate 5 versions, validate, write session memory.

All business logic for the Describe & Draft Agent lives here. The agent module
(src/agents/describe_draft.py) is a thin dispatcher that delegates to this module.
"""
import logging
import time
from typing import List, Optional

from src.dependencies import get_service_container
from src.schemas.describe_draft import (
    ClauseListEntry,
    ClauseListLLMResponse,
    ClauseVersion,
    DescribeDraftErrorType,
    DescribeDraftLLMResponse,
    DescribeDraftResponse,
    IntentClassification,
)
from src.services.prompts.v1 import load_prompt

logger = logging.getLogger(__name__)

# --- Banned phrase list for post-generation validator ---
_BANNED_PHRASES = [
    "witnesseth",
    "party of the first part",
    "party of the second part",
    "in witness whereof",
    "now therefore",
    "know all men by these presents",
]

# Axis-label patterns that must not leak into titles or summaries (case-insensitive
# substring match). These target phrases the LLM uses when it labels a VERSION
# by its axis, not legitimate legal vocabulary.
_BANNED_TITLE_SUMMARY_WORDS = [
    "party a-focused",
    "party b-focused",
    "party a-weighted",
    "party b-weighted",
    "weighted toward party",
    "version 1",
    "version 2",
    "version 3",
    "version 4",
    "version 5",
    "balanced version",
    "protective version",
    "plain-english version",
    "plain english version",
    "exhaustive version",
    "comprehensive version",
    "minimal version",
    "essential version",
    "belt-and-suspenders",
]

# Max prompt length (must match schema Field max_length)
_MAX_PROMPT_LENGTH = 2000

# Injection denylist — checked as case-insensitive substring matches
_INJECTION_PATTERNS = [
    "ignore all instructions",
    "ignore previous instructions",
    "disregard previous",
    "forget your instructions",
    "system prompt:",
    "system: ",
]


def _sanitize_prompt(prompt: str) -> str:
    """Raise ValueError if prompt contains injection patterns; return stripped prompt."""
    p = prompt.strip()
    p_lower = p.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern in p_lower:
            raise ValueError(f"Prompt contains disallowed pattern: '{pattern}'")
    return p


def _validate_draft_response(response: DescribeDraftLLMResponse) -> None:
    """Validate single_clause mode output: exactly 5 versions, each a full clause.

    Checks:
      - exactly 5 versions
      - non-empty title and summary on every version
      - drafted_clause is at least 50 chars and contains no banned phrases or axis labels
    """
    if len(response.versions) != 5:
        raise ValueError(f"Expected 5 versions, got {len(response.versions)}")
    for i, version in enumerate(response.versions):
        idx = i + 1
        if not version.title or not version.title.strip():
            raise ValueError(f"Version {idx}: title is empty")
        if not version.summary or not version.summary.strip():
            raise ValueError(f"Version {idx}: summary is empty")

        # Axis-label leakage check — titles and summaries must describe content, not style
        title_lower = version.title.lower()
        summary_lower = version.summary.lower()
        for word in _BANNED_TITLE_SUMMARY_WORDS:
            if word in title_lower:
                raise ValueError(
                    f"Version {idx}: title contains forbidden axis label '{word}'"
                )
            if word in summary_lower:
                raise ValueError(
                    f"Version {idx}: summary contains forbidden axis label '{word}'"
                )

        if not version.drafted_clause.strip():
            raise ValueError(f"Version {idx}: drafted_clause is empty")
        if len(version.drafted_clause) < 50:
            raise ValueError(
                f"Version {idx}: drafted_clause suspiciously short "
                f"({len(version.drafted_clause)} chars)"
            )
        lower = version.drafted_clause.lower()
        for phrase in _BANNED_PHRASES:
            if phrase in lower:
                raise ValueError(
                    f"Version {idx}: banned phrase '{phrase}' found in drafted_clause"
                )


def _validate_clause_list(response: ClauseListLLMResponse) -> None:
    """Validate list_of_clauses mode output: one complete clause list (≥12 clauses)."""
    if len(response.clauses) < 12:
        raise ValueError(
            f"Expected at least 12 clauses for a complete agreement, "
            f"got {len(response.clauses)}"
        )
    seen_titles: set = set()
    for i, clause in enumerate(response.clauses):
        idx = i + 1
        if not clause.title or not clause.title.strip():
            raise ValueError(f"Clause {idx}: title is empty")
        if not clause.summary or not clause.summary.strip():
            raise ValueError(f"Clause {idx}: summary is empty")

        title_norm = clause.title.strip().lower()
        if title_norm in seen_titles:
            raise ValueError(f"Clause {idx}: duplicate title '{clause.title}'")
        seen_titles.add(title_norm)

        # No archaic legalese in summaries
        summary_lower = clause.summary.lower()
        for phrase in _BANNED_PHRASES:
            if phrase in summary_lower:
                raise ValueError(
                    f"Clause {idx}: banned phrase '{phrase}' found in summary"
                )


async def _classify_intent(prompt: str) -> IntentClassification:
    container = get_service_container()
    llm = container.azure_openai_model
    rendered = load_prompt(
        "describe_draft_classifier_prompt", context={"user_prompt": prompt}
    )
    return await llm.generate(
        prompt=rendered,
        context={},
        response_model=IntentClassification,
        system_message="Classify the user's drafting intent. Return ONLY valid JSON.",
        temperature=0.0,
    )


async def _generate_clause_versions(
    prompt: str,
    agreement_type: Optional[str],
    prior_clauses: List[str],
) -> DescribeDraftLLMResponse:
    """single_clause mode: generate exactly 5 versions of one clause."""
    container = get_service_container()
    llm = container.azure_openai_model
    mode_instruction = (
        f"Draft a {agreement_type or 'legal'} clause as requested by the user."
    )
    context = {
        "user_prompt": prompt,
        "mode": "single_clause",
        "mode_instruction": mode_instruction,
        "is_single_clause": True,
        "is_list_of_clauses": False,
        "agreement_type": agreement_type or "",
        "has_agreement_type": bool(agreement_type),
        "prior_clauses": "\n".join(prior_clauses) if prior_clauses else "",
        "has_prior_clauses": bool(prior_clauses),
    }
    rendered = load_prompt("describe_draft_generation_prompt", context=context)
    return await llm.generate(
        prompt=rendered,
        context={},
        response_model=DescribeDraftLLMResponse,
        system_message=(
            "You are an expert legal drafter. "
            "Never invent case law. Never cite statutes unless the user named them. "
            "No archaic legalese. Consistent defined-term capitalization within each version. "
            "Return ONLY valid JSON matching the schema."
        ),
        temperature=0.15,
    )


async def _generate_clause_list(
    prompt: str,
    agreement_type: Optional[str],
    prior_clauses: List[str],
) -> ClauseListLLMResponse:
    """list_of_clauses mode: return ONE comprehensive clause list for the agreement type."""
    container = get_service_container()
    llm = container.azure_openai_model
    mode_instruction = (
        f"List all clauses that should appear in a "
        f"{agreement_type or 'legal agreement'} as requested by the user."
    )
    context = {
        "user_prompt": prompt,
        "mode": "list_of_clauses",
        "mode_instruction": mode_instruction,
        "is_single_clause": False,
        "is_list_of_clauses": True,
        "agreement_type": agreement_type or "",
        "has_agreement_type": bool(agreement_type),
        "prior_clauses": "\n".join(prior_clauses) if prior_clauses else "",
        "has_prior_clauses": bool(prior_clauses),
    }
    rendered = load_prompt("describe_draft_generation_prompt", context=context)
    return await llm.generate(
        prompt=rendered,
        context={},
        response_model=ClauseListLLMResponse,
        system_message=(
            "You are an expert legal drafter. Return ONE complete clause list for the "
            "requested agreement type. Do NOT return multiple versions. "
            "Return ONLY valid JSON matching the schema."
        ),
        temperature=0.1,
    )


def _read_session_context(session_id: str) -> dict:
    container = get_service_container()
    session = container.session_manager.get_or_create_session(session_id)
    return {
        "agreement_type": session.metadata.get("draft_agreement_type"),
        "prior_clauses": session.metadata.get("draft_prior_clauses", []),
    }


def _write_session_context(
    session_id: str,
    agreement_type: Optional[str],
    new_clause_titles: List[str],
    clear_prior: bool = False,
) -> None:
    container = get_service_container()
    session = container.session_manager.get_or_create_session(session_id)
    if agreement_type:
        session.metadata["draft_agreement_type"] = agreement_type
    if clear_prior:
        session.metadata["draft_prior_clauses"] = []
    if new_clause_titles:
        prior = session.metadata.get("draft_prior_clauses", [])
        session.metadata["draft_prior_clauses"] = prior + new_clause_titles


async def generate_describe_draft(prompt: str, session_id: str) -> DescribeDraftResponse:
    """
    Main entry point for the describe-draft agent.

    Flow:
      1. Sanitize input.
      2. Read session memory.
      3. Classify intent (temp 0.0).
      4. On clarification: return immediately with no generation.
      5. On list/single: generate 5 versions (temp 0.1/0.15), validate, retry once on failure.
      6. Write session memory.
      7. Emit audit log.
    """
    start_time = time.time()

    # 1. Sanitize input
    try:
        clean_prompt = _sanitize_prompt(prompt)
    except ValueError as e:
        return DescribeDraftResponse(
            session_id=session_id,
            mode="clarification",
            status="error",
            disclaimer=None,
            error_type=DescribeDraftErrorType.VALIDATION_FAILED,
            error_message=str(e),
        )

    # 2. Read session memory
    ctx = _read_session_context(session_id)
    stored_agreement_type: Optional[str] = ctx["agreement_type"]
    prior_clauses: List[str] = ctx["prior_clauses"]

    # 3. Classify intent
    try:
        classification = await _classify_intent(clean_prompt)
    except Exception as e:
        logger.error(
            "describe_draft classify error session=%s error=%s", session_id, str(e)
        )
        return DescribeDraftResponse(
            session_id=session_id,
            mode="clarification",
            status="error",
            disclaimer=None,
            error_type=DescribeDraftErrorType.LLM_FAILED,
            error_message=f"Intent classification failed: {str(e)}",
        )

    mode = classification.mode
    detected_agreement_type = classification.detected_agreement_type

    # 4. Clarification path — no generation
    if mode == "clarification":
        return DescribeDraftResponse(
            session_id=session_id,
            mode="clarification",
            status="ok",
            clarification_question=classification.clarification_question,
            versions=[],
            error_type=None,
        )

    # Determine effective agreement type and whether to clear prior clauses
    effective_agreement_type = detected_agreement_type or stored_agreement_type
    clear_prior = (
        detected_agreement_type is not None
        and stored_agreement_type is not None
        and detected_agreement_type.lower() != stored_agreement_type.lower()
    )

    # 5. Generate — branch on mode
    validation_error: Optional[str] = None
    clauses_out: List[ClauseListEntry] = []
    versions_out: List[ClauseVersion] = []
    units_generated = 0

    if mode == "list_of_clauses":
        list_response: Optional[ClauseListLLMResponse] = None
        for attempt in range(2):
            try:
                raw_list = await _generate_clause_list(
                    prompt=clean_prompt,
                    agreement_type=effective_agreement_type,
                    prior_clauses=prior_clauses if not clear_prior else [],
                )
                _validate_clause_list(raw_list)
                list_response = raw_list
                break
            except ValueError as ve:
                validation_error = str(ve)
                logger.warning(
                    "describe_draft list validation failed session=%s attempt=%d error=%s",
                    session_id, attempt + 1, validation_error,
                )
            except Exception as e:
                error_msg = str(e)
                logger.error(
                    "describe_draft list generation error session=%s attempt=%d error=%s",
                    session_id, attempt + 1, error_msg,
                )
                error_type = (
                    DescribeDraftErrorType.RATE_LIMITED
                    if "rate" in error_msg.lower()
                    else DescribeDraftErrorType.LLM_FAILED
                )
                return DescribeDraftResponse(
                    session_id=session_id,
                    mode=mode,
                    status="error",
                    disclaimer=None,
                    error_type=error_type,
                    error_message=f"LLM generation failed: {error_msg}",
                )

        if list_response is None:
            return DescribeDraftResponse(
                session_id=session_id,
                mode=mode,
                status="error",
                disclaimer=None,
                error_type=DescribeDraftErrorType.VALIDATION_FAILED,
                error_message=(
                    f"Clause-list validation failed after 2 attempts: {validation_error}"
                ),
            )

        clauses_out = list_response.clauses
        units_generated = len(clauses_out)
    else:
        # single_clause mode — 5 versions with retry
        versions_response: Optional[DescribeDraftLLMResponse] = None
        for attempt in range(2):
            try:
                raw = await _generate_clause_versions(
                    prompt=clean_prompt,
                    agreement_type=effective_agreement_type,
                    prior_clauses=prior_clauses if not clear_prior else [],
                )
                _validate_draft_response(raw)
                versions_response = raw
                break
            except ValueError as ve:
                validation_error = str(ve)
                logger.warning(
                    "describe_draft validation failed session=%s attempt=%d error=%s",
                    session_id, attempt + 1, validation_error,
                )
            except Exception as e:
                error_msg = str(e)
                logger.error(
                    "describe_draft generation error session=%s attempt=%d error=%s",
                    session_id, attempt + 1, error_msg,
                )
                error_type = (
                    DescribeDraftErrorType.RATE_LIMITED
                    if "rate" in error_msg.lower()
                    else DescribeDraftErrorType.LLM_FAILED
                )
                return DescribeDraftResponse(
                    session_id=session_id,
                    mode=mode,
                    status="error",
                    disclaimer=None,
                    error_type=error_type,
                    error_message=f"LLM generation failed: {error_msg}",
                )

        if versions_response is None:
            return DescribeDraftResponse(
                session_id=session_id,
                mode=mode,
                status="error",
                disclaimer=None,
                error_type=DescribeDraftErrorType.VALIDATION_FAILED,
                error_message=(
                    f"Generation validation failed after 2 attempts: {validation_error}"
                ),
            )

        versions_out = versions_response.versions
        units_generated = len(versions_out)

    # 6. Write session memory
    if mode == "single_clause" and versions_out:
        # Track the clause the user just drafted (first version's title)
        new_titles = [versions_out[0].title]
    else:
        # list_of_clauses: don't add all clauses to prior (noisy); just record agreement_type
        new_titles = []
    _write_session_context(
        session_id=session_id,
        agreement_type=effective_agreement_type,
        new_clause_titles=new_titles,
        clear_prior=clear_prior,
    )

    # 7. Audit log (per-call token usage is emitted by the LLM client)
    latency_ms = int((time.time() - start_time) * 1000)
    logger.info(
        "describe_draft_audit session=%s mode=%s agreement_type=%s "
        "units_generated=%d latency_ms=%d",
        session_id,
        mode,
        effective_agreement_type or "unknown",
        units_generated,
        latency_ms,
    )

    return DescribeDraftResponse(
        session_id=session_id,
        mode=mode,
        status="ok",
        clauses=clauses_out,
        versions=versions_out,
    )
