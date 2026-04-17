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
    """Raise ValueError with a specific message on any validation failure.

    Checks:
      - exactly 5 versions
      - non-empty title and summary on every version
      - drafted_clause (when present) is at least 50 chars and contains no banned phrases
    """
    if len(response.versions) != 5:
        raise ValueError(f"Expected 5 versions, got {len(response.versions)}")
    for i, version in enumerate(response.versions):
        idx = i + 1
        if not version.title or not version.title.strip():
            raise ValueError(f"Version {idx}: title is empty")
        if not version.summary or not version.summary.strip():
            raise ValueError(f"Version {idx}: summary is empty")
        # drafted_clause may be empty for list_of_clauses mode — check only if present
        if version.drafted_clause.strip():
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


async def _generate_versions(
    prompt: str,
    mode: str,
    agreement_type: Optional[str],
    prior_clauses: List[str],
) -> DescribeDraftLLMResponse:
    container = get_service_container()
    llm = container.azure_openai_model
    is_single = mode == "single_clause"
    temperature = 0.15 if is_single else 0.1
    mode_instruction = (
        f"Draft a {agreement_type or 'legal'} clause as requested by the user."
        if is_single
        else f"List the clauses that should appear in a "
             f"{agreement_type or 'legal agreement'} as requested by the user."
    )
    context = {
        "user_prompt": prompt,
        "mode": mode,
        "mode_instruction": mode_instruction,
        "is_single_clause": is_single,
        "is_list_of_clauses": not is_single,
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
        temperature=temperature,
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

    # 5. Generate 5 versions — with one retry on validation failure
    llm_response: Optional[DescribeDraftLLMResponse] = None
    validation_error: Optional[str] = None
    for attempt in range(2):
        try:
            raw = await _generate_versions(
                prompt=clean_prompt,
                mode=mode,
                agreement_type=effective_agreement_type,
                prior_clauses=prior_clauses if not clear_prior else [],
            )
            _validate_draft_response(raw)
            llm_response = raw
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

    if llm_response is None:
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

    # 6. Write session memory (store first version's title as representative)
    new_titles = [v.title for v in llm_response.versions[:1]]
    _write_session_context(
        session_id=session_id,
        agreement_type=effective_agreement_type,
        new_clause_titles=new_titles,
        clear_prior=clear_prior,
    )

    # 7. Audit log
    latency_ms = int((time.time() - start_time) * 1000)
    logger.info(
        "describe_draft_audit session=%s mode=%s agreement_type=%s "
        "versions_generated=%d input_tokens=%d output_tokens=%d latency_ms=%d",
        session_id,
        mode,
        effective_agreement_type or "unknown",
        len(llm_response.versions),
        0,  # input_tokens not exposed by current llm.generate() return value
        0,  # output_tokens not exposed by current llm.generate() return value
        latency_ms,
    )

    return DescribeDraftResponse(
        session_id=session_id,
        mode=mode,
        status="ok",
        versions=llm_response.versions,
    )
