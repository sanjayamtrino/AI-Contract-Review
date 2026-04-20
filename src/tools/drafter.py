"""
Describe & Draft tool — classify intent, generate a draft, validate, write session memory.

single_clause mode returns one draft per call; the caller regenerates by invoking
the endpoint again with regenerate=true, which asks the LLM to improve on the
previous draft stored in session metadata.

All business logic for the Describe & Draft Agent lives here. The agent module
(src/agents/describe_draft.py) is a thin dispatcher that delegates to this module.
"""
import logging
import time
from typing import Any, Dict, List, Optional

from src.dependencies import get_service_container
from src.schemas.describe_draft import (
    ClauseListEntry,
    ClauseListLLMResponse,
    ClauseVersion,
    DescribeDraftErrorType,
    DescribeDraftLLMResponse,
    DescribeDraftResponse,
    DuplicateCheckResult,
    ExistingClauseMatch,
    IntentClassification,
)
from src.services.prompts.v1 import load_prompt

logger = logging.getLogger(__name__)

# Retrieval settings for doc-grounded drafting
_RETRIEVAL_TOP_K = 4
# Minimum similarity score on the top chunk to trigger LLM duplicate confirmation.
# FAISS with normalized embeddings produces cosine similarities in roughly [0, 1].
# 0.55 catches obvious duplicates without flagging tangentially related text.
_DUPLICATE_SIMILARITY_GATE = 0.55

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
# substring match). These target phrases the LLM uses when it labels a draft by
# stylistic axis instead of clause content, not legitimate legal vocabulary.
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
    "regenerated version",
    "improved version",
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
    """Validate single_clause mode output: exactly 1 version containing a full clause.

    Checks:
      - exactly 1 version
      - non-empty title and summary
      - drafted_clause is at least 50 chars and contains no banned phrases or axis labels
    """
    if len(response.versions) != 1:
        raise ValueError(f"Expected 1 version, got {len(response.versions)}")
    version = response.versions[0]

    if not version.title or not version.title.strip():
        raise ValueError("Version: title is empty")
    if not version.summary or not version.summary.strip():
        raise ValueError("Version: summary is empty")

    # Axis-label leakage check — titles and summaries must describe content, not style
    title_lower = version.title.lower()
    summary_lower = version.summary.lower()
    for word in _BANNED_TITLE_SUMMARY_WORDS:
        if word in title_lower:
            raise ValueError(
                f"Version: title contains forbidden axis label '{word}'"
            )
        if word in summary_lower:
            raise ValueError(
                f"Version: summary contains forbidden axis label '{word}'"
            )

    if not version.drafted_clause.strip():
        raise ValueError("Version: drafted_clause is empty")
    if len(version.drafted_clause) < 50:
        raise ValueError(
            f"Version: drafted_clause suspiciously short "
            f"({len(version.drafted_clause)} chars)"
        )
    lower = version.drafted_clause.lower()
    for phrase in _BANNED_PHRASES:
        if phrase in lower:
            raise ValueError(
                f"Version: banned phrase '{phrase}' found in drafted_clause"
            )


def _validate_regenerated_draft_differs(
    new_version: ClauseVersion, prior_version: ClauseVersion
) -> None:
    """Regenerate must produce a draft meaningfully different from the prior one."""
    if (
        new_version.drafted_clause.strip() == prior_version.drafted_clause.strip()
        or new_version.summary.strip() == prior_version.summary.strip()
    ):
        raise ValueError(
            "Regenerated version is identical to the prior draft — not a meaningful variation"
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


def _session_has_document(session) -> bool:
    """True when the session has at least one ingested document."""
    docs = getattr(session, "documents", None)
    if docs:
        return True
    chunk_store = getattr(session, "chunk_store", None)
    return bool(chunk_store)


async def _get_doc_grounding(session_id: str) -> Optional[Dict[str, Any]]:
    """Extract (or reuse cached) parties + governing_law for the session's document.

    Cached in session.metadata["draft_doc_grounding"] so regenerate calls don't
    re-run the extraction LLM call. Returns None if extraction fails or yields nothing.
    """
    from src.tools.key_information import get_key_information  # local import to avoid cycles

    container = get_service_container()
    session = container.session_manager.get_or_create_session(session_id)
    cached = session.metadata.get("draft_doc_grounding")
    if isinstance(cached, dict) and cached.get("parties"):
        return cached

    try:
        payload = await get_key_information(session_id=session_id, response_format="JSON")
    except Exception as e:
        logger.warning(
            "doc grounding extraction failed session=%s error=%s", session_id, str(e)
        )
        return None

    if not isinstance(payload, dict):
        return None

    parties_raw = payload.get("parties") or []
    parties: List[Dict[str, Optional[str]]] = []
    for p in parties_raw:
        if not isinstance(p, dict):
            continue
        name = (p.get("name") or "").strip()
        if not name:
            continue
        parties.append({
            "name": name,
            "role": (p.get("role") or "").strip() or None,
        })

    gov_law = payload.get("governing_law") or {}
    gov_law_str = ""
    if isinstance(gov_law, dict):
        gov_law_str = (gov_law.get("value") or gov_law.get("information") or "").strip()

    if not parties and not gov_law_str:
        return None

    grounding = {"parties": parties, "governing_law": gov_law_str}
    session.metadata["draft_doc_grounding"] = grounding
    return grounding


async def _retrieve_relevant_chunks(
    session_id: str, query: str, top_k: int = _RETRIEVAL_TOP_K
) -> List[Dict[str, Any]]:
    """Retrieve top-K document chunks matching the drafting prompt. Empty list on failure."""
    container = get_service_container()
    session = container.session_manager.get_or_create_session(session_id)
    if not _session_has_document(session):
        return []
    try:
        result = await container.retrieval_service.retrieve_data(
            query=query,
            top_k=top_k,
            session_data=session,
        )
    except Exception as e:
        logger.warning(
            "doc retrieval failed session=%s error=%s", session_id, str(e)
        )
        return []
    chunks = result.get("chunks") or []
    return chunks


async def _check_duplicate_clause(
    user_prompt: str, chunks: List[Dict[str, Any]]
) -> Optional[ExistingClauseMatch]:
    """Decide whether the top retrieved chunk is already a clause on the user's topic.

    Two-stage gate: (1) similarity score must exceed _DUPLICATE_SIMILARITY_GATE, and
    (2) a cheap LLM confirmation call must agree. Only returns a match when both pass.
    """
    if not chunks:
        return None
    top = chunks[0]
    score = float(top.get("similarity_score") or 0.0)
    if score < _DUPLICATE_SIMILARITY_GATE:
        return None

    candidate_text = (top.get("content") or "").strip()
    if not candidate_text:
        return None

    container = get_service_container()
    rendered = load_prompt(
        "describe_draft_duplicate_check_prompt",
        context={"user_request": user_prompt, "candidate_text": candidate_text},
    )
    try:
        result = await container.azure_openai_model.generate(
            prompt=rendered,
            context={},
            response_model=DuplicateCheckResult,
            system_message=(
                "You decide whether a candidate passage already covers a specific "
                "drafting request. Be strict. Return ONLY valid JSON."
            ),
            temperature=0.0,
        )
    except Exception as e:
        logger.warning("duplicate-check LLM call failed error=%s", str(e))
        return None

    if not result.is_duplicate:
        return None

    inferred_title = result.matched_title
    if not inferred_title:
        metadata = top.get("metadata") or {}
        inferred_title = metadata.get("section_heading") if isinstance(metadata, dict) else None

    return ExistingClauseMatch(
        title=inferred_title,
        excerpt=candidate_text,
        similarity_score=score,
    )


def _format_parties_block(parties: List[Dict[str, Optional[str]]]) -> str:
    if not parties:
        return ""
    lines = []
    for p in parties:
        role = p.get("role")
        if role:
            lines.append(f"- {p['name']} (role: {role})")
        else:
            lines.append(f"- {p['name']}")
    return "\n".join(lines)


def _format_relevant_chunks(chunks: List[Dict[str, Any]], limit: int = 3) -> str:
    if not chunks:
        return ""
    pieces = []
    for c in chunks[:limit]:
        content = (c.get("content") or "").strip()
        if content:
            pieces.append(content)
    return "\n\n---\n\n".join(pieces)


async def _generate_clause_draft(
    prompt: str,
    agreement_type: Optional[str],
    prior_clauses: List[str],
    prior_draft: Optional[ClauseVersion] = None,
    doc_grounding: Optional[Dict[str, Any]] = None,
    relevant_chunks: Optional[List[Dict[str, Any]]] = None,
) -> DescribeDraftLLMResponse:
    """single_clause mode: generate exactly 1 draft of the requested clause.

    If `prior_draft` is given, the prompt asks the LLM for a meaningfully
    different and improved variation (regenerate flow); temperature is
    nudged up to encourage variation.

    If `doc_grounding` is given, injects party names + governing law and
    (optionally) relevant chunks from the uploaded document so the draft is
    drop-in ready for that contract.
    """
    container = get_service_container()
    llm = container.azure_openai_model
    mode_instruction = (
        f"Draft a {agreement_type or 'legal'} clause as requested by the user."
    )
    is_regenerate = prior_draft is not None
    has_grounding = bool(doc_grounding and doc_grounding.get("parties"))
    parties_block = _format_parties_block(doc_grounding["parties"]) if has_grounding else ""
    governing_law = (doc_grounding or {}).get("governing_law") or ""
    chunks_text = _format_relevant_chunks(relevant_chunks or [])

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
        "is_regenerate": is_regenerate,
        "prior_draft_title": prior_draft.title if prior_draft else "",
        "prior_draft_clause": prior_draft.drafted_clause if prior_draft else "",
        "has_doc_grounding": has_grounding,
        "doc_parties_block": parties_block,
        "doc_governing_law": governing_law or "(not specified in document)",
        "has_relevant_chunks": bool(chunks_text),
        "doc_relevant_chunks": chunks_text,
    }
    rendered = load_prompt("describe_draft_generation_prompt", context=context)
    return await llm.generate(
        prompt=rendered,
        context={},
        response_model=DescribeDraftLLMResponse,
        system_message=(
            "You are an expert legal drafter. "
            "Never invent case law. Never cite statutes unless the user named them. "
            "No archaic legalese. Consistent defined-term capitalization. "
            "When party names are provided, use them exactly — no placeholders. "
            "Return ONLY valid JSON matching the schema."
        ),
        temperature=0.35 if is_regenerate else 0.15,
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
    last_raw = session.metadata.get("draft_last_version")
    last_version: Optional[ClauseVersion] = None
    if isinstance(last_raw, dict):
        try:
            last_version = ClauseVersion.model_validate(last_raw)
        except Exception:
            last_version = None
    return {
        "agreement_type": session.metadata.get("draft_agreement_type"),
        "prior_clauses": session.metadata.get("draft_prior_clauses", []) or [],
        "last_version": last_version,
    }


def _write_session_context(
    session_id: str,
    agreement_type: Optional[str],
    new_clause_titles: List[str],
    clear_prior: bool = False,
    last_version: Optional[ClauseVersion] = None,
    clear_last_version: bool = False,
) -> None:
    container = get_service_container()
    session = container.session_manager.get_or_create_session(session_id)
    if agreement_type:
        session.metadata["draft_agreement_type"] = agreement_type
    if clear_prior:
        session.metadata["draft_prior_clauses"] = []
    if new_clause_titles:
        prior = session.metadata.get("draft_prior_clauses", []) or []
        session.metadata["draft_prior_clauses"] = prior + new_clause_titles
    if clear_last_version:
        session.metadata.pop("draft_last_version", None)
    elif last_version is not None:
        session.metadata["draft_last_version"] = last_version.model_dump()


async def generate_describe_draft(
    prompt: str,
    session_id: str,
    regenerate: bool = False,
) -> DescribeDraftResponse:
    """
    Main entry point for the describe-draft agent.

    Flow:
      1. Sanitize input.
      2. Read session memory (including last single-clause draft if any).
      3. Classify intent (temp 0.0).
      4. On clarification: return immediately with no generation.
      5. On list_of_clauses: generate one clause list, validate, retry once.
         On single_clause: generate ONE draft (regenerate=True asks for an improved
         variation of the prior draft), validate, retry once.
      6. Write session memory — stash the latest single-clause draft for later regenerate.
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
    stored_last_version: Optional[ClauseVersion] = ctx["last_version"]

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
    grounded_in_doc = False

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
        # single_clause mode — generate ONE draft (regenerate asks for an improved
        # variation of the prior draft stored in session).
        effective_regenerate = regenerate and stored_last_version is not None
        prior_draft_for_prompt = stored_last_version if effective_regenerate else None

        # Doc-grounded path: only when a document is attached to the session.
        container = get_service_container()
        session_obj = container.session_manager.get_or_create_session(session_id)
        has_document = _session_has_document(session_obj)
        doc_grounding: Optional[Dict[str, Any]] = None
        relevant_chunks: List[Dict[str, Any]] = []

        if has_document:
            doc_grounding = await _get_doc_grounding(session_id)
            relevant_chunks = await _retrieve_relevant_chunks(session_id, clean_prompt)
            grounded_in_doc = bool(doc_grounding and doc_grounding.get("parties"))

            # Duplicate detection — skip when regenerate is active (user already
            # saw the prior draft; they want a new one regardless).
            if not effective_regenerate and relevant_chunks:
                try:
                    existing = await _check_duplicate_clause(clean_prompt, relevant_chunks)
                except Exception as e:
                    logger.warning(
                        "duplicate check errored session=%s error=%s", session_id, str(e)
                    )
                    existing = None
                if existing is not None:
                    _write_session_context(
                        session_id=session_id,
                        agreement_type=effective_agreement_type,
                        new_clause_titles=[],
                        clear_prior=clear_prior,
                    )
                    latency_ms = int((time.time() - start_time) * 1000)
                    logger.info(
                        "describe_draft_audit session=%s mode=single_clause_exists "
                        "agreement_type=%s similarity=%.3f latency_ms=%d",
                        session_id,
                        effective_agreement_type or "unknown",
                        existing.similarity_score,
                        latency_ms,
                    )
                    return DescribeDraftResponse(
                        session_id=session_id,
                        mode="single_clause_exists",
                        status="ok",
                        existing_clause=existing,
                        grounded_in_document=True,
                    )

        versions_response: Optional[DescribeDraftLLMResponse] = None
        for attempt in range(2):
            try:
                raw = await _generate_clause_draft(
                    prompt=clean_prompt,
                    agreement_type=effective_agreement_type,
                    prior_clauses=prior_clauses if not clear_prior else [],
                    prior_draft=prior_draft_for_prompt,
                    doc_grounding=doc_grounding,
                    relevant_chunks=relevant_chunks,
                )
                _validate_draft_response(raw)
                if effective_regenerate:
                    _validate_regenerated_draft_differs(
                        raw.versions[0], stored_last_version  # type: ignore[arg-type]
                    )
                versions_response = raw
                break
            except ValueError as ve:
                validation_error = str(ve)
                logger.warning(
                    "describe_draft validation failed session=%s attempt=%d regenerate=%s error=%s",
                    session_id, attempt + 1, effective_regenerate, validation_error,
                )
            except Exception as e:
                error_msg = str(e)
                logger.error(
                    "describe_draft generation error session=%s attempt=%d regenerate=%s error=%s",
                    session_id, attempt + 1, effective_regenerate, error_msg,
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
    is_regenerate_hit = (
        mode == "single_clause"
        and regenerate
        and stored_last_version is not None
    )
    if mode == "single_clause" and versions_out and not is_regenerate_hit:
        # Fresh draft: track the clause title in prior_clauses for future context.
        new_titles = [versions_out[0].title]
    else:
        # Regenerate (same clause) or list_of_clauses: don't append to prior_clauses.
        new_titles = []
    _write_session_context(
        session_id=session_id,
        agreement_type=effective_agreement_type,
        new_clause_titles=new_titles,
        clear_prior=clear_prior,
        last_version=versions_out[0] if (mode == "single_clause" and versions_out) else None,
        clear_last_version=(mode == "list_of_clauses"),
    )

    # 7. Audit log (per-call token usage is emitted by the LLM client)
    latency_ms = int((time.time() - start_time) * 1000)
    logger.info(
        "describe_draft_audit session=%s mode=%s agreement_type=%s "
        "units_generated=%d regenerate=%s grounded=%s latency_ms=%d",
        session_id,
        mode,
        effective_agreement_type or "unknown",
        units_generated,
        is_regenerate_hit,
        grounded_in_doc,
        latency_ms,
    )

    return DescribeDraftResponse(
        session_id=session_id,
        mode=mode,
        status="ok",
        clauses=clauses_out,
        versions=versions_out,
        regenerated=is_regenerate_hit,
        grounded_in_document=grounded_in_doc,
    )
