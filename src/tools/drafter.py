"""
Describe & Draft tool — classify intent, generate a draft, validate, write session memory.

Two modes, gated by the caller's "Use Document Context" toggle (use_document_context):
  - Use Document Context ON: the draft is grounded in the document opened on this
    session — its extracted parties and governing law are injected so the clause is
    drop-in ready for that contract, and relevant existing text is supplied for
    style / defined-term consistency.
  - Use Document Context OFF (default): any attached document is ignored. Drafts are
    emitted as reusable templates with [PLACEHOLDER] tokens (ALL CAPS in square
    brackets) that the frontend can substitute later.

Each call returns either one drafted clause (single_clause mode) or one complete
clause list with drafted bodies (list_of_clauses mode), each carrying a summary.

All business logic for the Describe & Draft Agent lives here. The agent module
(src/agents/describe_draft.py) is a thin dispatcher that delegates to this module.
"""
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from src.dependencies import get_service_container
from src.schemas.describe_draft import (
    ClauseListEntry,
    ClauseListLLMResponse,
    DescribeDraftErrorType,
    DescribeDraftLLMResponse,
    DescribeDraftResponse,
    DraftedClause,
    IntentClassification,
)
from src.services.prompts.v1 import load_prompt

logger = logging.getLogger(__name__)

# Number of document chunks retrieved to give the drafter style / defined-term
# context when Use Document Context is on.
_RETRIEVAL_TOP_K = 4


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

# Placeholder token format: [ALL CAPS + SPACES + DIGITS], e.g. [PARTY A], [EFFECTIVE DATE]
_PLACEHOLDER_PATTERN = re.compile(r"\[([A-Z][A-Z0-9 /\-&]{1,60})\]")

# Placeholder token-name substrings that MUST NOT appear when Use Document Context is on.
# Document grounding always supplies the parties and the governing law / forum, so a
# `[PARTY A]` or `[GOVERNING STATE]` token in a doc-grounded draft is a real violation.
# Factual placeholders the document does NOT supply (amounts, dates, durations, cure /
# notice periods) are allowed even in doc-grounded mode — the user must fill those in
# either way. Substring match is case-insensitive against the token name.
_GROUNDED_FORBIDDEN_PLACEHOLDER_SUBSTRINGS = [
    # party-identity tokens
    "PARTY",
    "TENANT",
    "LANDLORD",
    "CUSTOMER",
    "CLIENT",
    "VENDOR",
    "SUPPLIER",
    "EMPLOYER",
    "EMPLOYEE",
    "CONTRACTOR",
    "DISCLOSING",
    "RECEIVING",
    "COMPANY",
    "CORPORATION",
    "BUYER",
    "SELLER",
    "LICENSOR",
    "LICENSEE",
    "INDEMNIFIER",
    "INDEMNITEE",
    # governing-law / forum tokens
    "GOVERNING LAW",
    "GOVERNING STATE",
    "JURISDICTION",
    "VENUE",
    "FORUM",
]
# Minimum body length for an individual drafted clause body in list mode.
# Intentionally permissive: a complete agreement contains some legitimately
# short boilerplate clauses (Headings, Construction, Counterparts, basic
# Severability) that can land at 70-110 chars and still be complete. The 60-char
# per-clause floor only catches truly empty entries ("TBD", "see attached", or
# the LLM accidentally returning the title as the body). Real list-quality
# enforcement happens via the aggregate average gate below — that's what
# guarantees the heavyweight clauses (Indemnification, Limitation of Liability,
# Term & Termination) carry real depth.
_MIN_LIST_CLAUSE_BODY_LEN = 60
# Aggregate floor for list mode: average drafted_clause length across the whole
# list must be ≥ this. This is the PRIMARY quality gate for list mode — it
# ensures the agreement as a whole is substantive without false-positiving the
# legitimately short boilerplate clauses that appear in every well-drafted
# agreement.
_MIN_LIST_AVG_CLAUSE_BODY_LEN = 280
# Single-clause mode: the user is asking for ONE clause and expects industry-grade
# depth. 300 chars is the floor — a complete Governing Law clause with choice of
# law, forum, jury waiver, and CISG carve-out is comfortably > 600 chars.
_MIN_SINGLE_CLAUSE_BODY_LEN = 300
# Summary spec for SINGLE-CLAUSE mode: 2–3 sentences explaining scope, allocation
# of risk, and notable carve-outs. A single-line label fails to brief the reader.
_MIN_SUMMARY_LEN = 80
# Summary floor for LIST mode: the list view is scannable — short one-sentence
# descriptors are appropriate. The body carries the depth, not the summary. 40
# chars catches truly empty labels ("LoL clause") without rejecting useful
# one-liners ("Limits each party's liability for indirect damages.").
_MIN_LIST_SUMMARY_LEN = 40

# Temperature for drafts — low so output is stable and on-spec.
_DRAFT_TEMPERATURE = 0.15


def _sanitize_prompt(prompt: str) -> str:
    """Raise ValueError if prompt contains injection patterns; return stripped prompt."""
    p = prompt.strip()
    p_lower = p.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern in p_lower:
            raise ValueError(f"Prompt contains disallowed pattern: '{pattern}'")
    return p


def _extract_placeholders(text: str) -> List[str]:
    """Return distinct `[ALL CAPS]` placeholder tokens found in text, in first-seen order."""
    seen: List[str] = []
    for match in _PLACEHOLDER_PATTERN.finditer(text or ""):
        token = f"[{match.group(1)}]"
        if token not in seen:
            seen.append(token)
    return seen


def _grounded_forbidden_placeholders(placeholders: List[str]) -> List[str]:
    """Return placeholders whose token name names a party or the governing law.

    These are the only placeholders that are real violations in doc-grounded mode —
    the document already supplied those values. Factual placeholders for facts the
    document does not contain (amounts, dates, durations, cure / notice periods)
    are kept out of the returned list and treated as acceptable.
    """
    forbidden: List[str] = []
    for tok in placeholders:
        name = tok.strip("[]").upper()
        if any(needle in name for needle in _GROUNDED_FORBIDDEN_PLACEHOLDER_SUBSTRINGS):
            forbidden.append(tok)
    return forbidden


def _validate_draft_response(
    response: DescribeDraftLLMResponse,
    *,
    require_placeholders: bool = False,
    forbid_placeholders: bool = False,
) -> None:
    """Validate single_clause mode output: one full drafted clause.

    Checks:
      - a clause is present
      - non-empty title and a substantive (≥80 char) summary
      - drafted_clause is at least 300 chars and contains no banned phrases or axis labels
      - if forbid_placeholders: drafted_clause contains NO party/governing-law tokens

    `require_placeholders` is a soft preference (logged, not enforced) — some clauses
    are pure boilerplate with no fillable facts.

    After validation, `clause.placeholders` is rewritten to the authoritative
    list of tokens found in `drafted_clause` — the LLM's own list is advisory.
    """
    clause = response.clause
    if clause is None:
        raise ValueError("No clause was generated")

    if not clause.title or not clause.title.strip():
        raise ValueError("Clause: title is empty")
    if not clause.summary or not clause.summary.strip():
        raise ValueError("Clause: summary is empty")
    if len(clause.summary.strip()) < _MIN_SUMMARY_LEN:
        raise ValueError(
            f"Clause: summary is a one-line label ({len(clause.summary.strip())} chars); "
            f"the spec requires a 2-3 sentence brief covering scope, allocation of risk, "
            f"and notable carve-outs (≥{_MIN_SUMMARY_LEN} chars)"
        )

    # Axis-label leakage check — titles and summaries must describe content, not style
    title_lower = clause.title.lower()
    summary_lower = clause.summary.lower()
    for word in _BANNED_TITLE_SUMMARY_WORDS:
        if word in title_lower:
            raise ValueError(
                f"Clause: title contains forbidden axis label '{word}'"
            )
        if word in summary_lower:
            raise ValueError(
                f"Clause: summary contains forbidden axis label '{word}'"
            )

    if not clause.drafted_clause.strip():
        raise ValueError("Clause: drafted_clause is empty")
    if len(clause.drafted_clause.strip()) < _MIN_SINGLE_CLAUSE_BODY_LEN:
        raise ValueError(
            f"Clause: drafted_clause is too short for an industry-grade clause "
            f"({len(clause.drafted_clause.strip())} chars; ≥{_MIN_SINGLE_CLAUSE_BODY_LEN} required). "
            f"The clause must satisfy the QUALITY BAR — operative rule plus ancillary "
            f"provisions (notice, cure, exceptions, remedies, survival)."
        )
    lower = clause.drafted_clause.lower()
    for phrase in _BANNED_PHRASES:
        if phrase in lower:
            raise ValueError(
                f"Clause: banned phrase '{phrase}' found in drafted_clause"
            )

    found_placeholders = _extract_placeholders(clause.drafted_clause)
    if require_placeholders and not found_placeholders:
        # Soft preference: most clauses benefit from [PLACEHOLDER] tokens so the
        # frontend can do find-and-replace. But some clauses (Severability,
        # Entire Agreement, Counterparts, basic Force Majeure, basic Waiver) have
        # no user-fillable facts — failing them is a worse UX than accepting a
        # placeholder-free template. Log so we can see how often this happens
        # without blocking the response.
        logger.info(
            "describe_draft single_clause draft contains no [PLACEHOLDER] tokens "
            "in no-doc mode (title=%r) — accepting as boilerplate-style clause",
            clause.title,
        )
    if forbid_placeholders and found_placeholders:
        grounded_forbidden = _grounded_forbidden_placeholders(found_placeholders)
        if grounded_forbidden:
            raise ValueError(
                f"Clause: drafted_clause contains party-identity or governing-law "
                f"[PLACEHOLDER] tokens that must come from the attached document "
                f"(found {grounded_forbidden[:3]}). Factual placeholders for values "
                f"the document does not supply (amounts, dates, durations) are allowed."
            )
    clause.placeholders = found_placeholders


def _validate_clause_list(
    response: ClauseListLLMResponse,
    *,
    require_placeholders: bool = False,
    forbid_placeholders: bool = False,
) -> None:
    """Validate list_of_clauses mode output: one complete clause list (≥12 clauses).

    Every entry must have a non-empty drafted body; no banned phrases. The
    placeholder rule for no-doc mode is a soft preference (logged, not enforced).
    For doc-grounded lists, NO clause may contain party-identity or governing-law
    placeholder tokens.
    """
    if len(response.clauses) < 12:
        raise ValueError(
            f"Expected at least 12 clauses for a complete agreement, "
            f"got {len(response.clauses)}"
        )
    # Agreement summary is the orienting overview shown at the top of the list.
    # Must be non-empty and at least minimally substantive (a one-word stub
    # like "Agreement." is rejected).
    if not response.agreement_summary or not response.agreement_summary.strip():
        raise ValueError("agreement_summary is empty")
    if len(response.agreement_summary.strip()) < 60:
        raise ValueError(
            f"agreement_summary is too short to orient the reader "
            f"({len(response.agreement_summary.strip())} chars; ≥60 required). "
            f"It should be 3-5 sentences covering purpose, parties, core "
            f"exchange, and notable structural features."
        )
    seen_titles: set = set()
    clauses_with_placeholders = 0
    for i, clause in enumerate(response.clauses):
        idx = i + 1
        if not clause.title or not clause.title.strip():
            raise ValueError(f"Clause {idx}: title is empty")
        if not clause.summary or not clause.summary.strip():
            raise ValueError(f"Clause {idx}: summary is empty")
        if len(clause.summary.strip()) < _MIN_LIST_SUMMARY_LEN:
            raise ValueError(
                f"Clause {idx} ('{clause.title}'): summary is too short to be useful "
                f"({len(clause.summary.strip())} chars; ≥{_MIN_LIST_SUMMARY_LEN} required for list mode). "
                f"A short descriptive sentence is fine — the body carries the depth."
            )

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

        # Drafted body checks
        if not clause.drafted_clause or not clause.drafted_clause.strip():
            raise ValueError(f"Clause {idx}: drafted_clause is empty")
        if len(clause.drafted_clause.strip()) < _MIN_LIST_CLAUSE_BODY_LEN:
            raise ValueError(
                f"Clause {idx}: drafted_clause suspiciously short "
                f"({len(clause.drafted_clause.strip())} chars)"
            )
        body_lower = clause.drafted_clause.lower()
        for phrase in _BANNED_PHRASES:
            if phrase in body_lower:
                raise ValueError(
                    f"Clause {idx}: banned phrase '{phrase}' found in drafted_clause"
                )

        found_placeholders = _extract_placeholders(clause.drafted_clause)
        if forbid_placeholders and found_placeholders:
            grounded_forbidden = _grounded_forbidden_placeholders(found_placeholders)
            if grounded_forbidden:
                raise ValueError(
                    f"Clause {idx} ('{clause.title}'): drafted_clause contains "
                    f"party-identity or governing-law [PLACEHOLDER] tokens that "
                    f"must come from the attached document "
                    f"(found {grounded_forbidden[:3]}). Factual placeholders for "
                    f"values the document does not supply are allowed."
                )
        if found_placeholders:
            clauses_with_placeholders += 1
        clause.placeholders = found_placeholders

    if require_placeholders:
        total = len(response.clauses)
        recommended_min = max(4, int(total * 0.6))
        if clauses_with_placeholders < recommended_min:
            # Log only — do not reject. Even when the LLM under-uses placeholders,
            # the drafted clauses are still usable: role labels (Tenant, Vendor,
            # Employee) make the agreement readable, and the user can find-and-
            # replace specific names afterward.
            logger.info(
                "describe_draft list mode below recommended placeholder coverage: "
                "%d/%d clauses have [PLACEHOLDER] tokens (recommended ≥%d). "
                "Returning anyway — user can find-and-replace.",
                clauses_with_placeholders, total, recommended_min,
            )

    # Aggregate depth — log only, do not reject. A "thin" list is still useful
    # output the user can act on; rejecting it returns nothing, which is worse.
    total_body_chars = sum(len(c.drafted_clause.strip()) for c in response.clauses)
    avg_body_len = total_body_chars / len(response.clauses)
    if avg_body_len < _MIN_LIST_AVG_CLAUSE_BODY_LEN:
        logger.info(
            "describe_draft list mode below recommended depth: avg=%.0f chars "
            "across %d clauses (recommended ≥%d). Returning anyway.",
            avg_body_len, len(response.clauses), _MIN_LIST_AVG_CLAUSE_BODY_LEN,
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

    Cached in session.metadata["draft_doc_grounding"] so repeat calls don't
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
    doc_grounding: Optional[Dict[str, Any]] = None,
    relevant_chunks: Optional[List[Dict[str, Any]]] = None,
) -> DescribeDraftLLMResponse:
    """single_clause mode: generate exactly 1 draft of the requested clause.

    If `doc_grounding` is given, injects party names + governing law and
    (optionally) relevant chunks from the opened document so the draft is
    drop-in ready for that contract. Otherwise the prompt instructs the LLM
    to use `[PLACEHOLDER]` tokens.
    """
    container = get_service_container()
    llm = container.azure_openai_model
    mode_instruction = (
        f"Draft a {agreement_type or 'legal'} clause as requested by the user."
    )
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
            "When no document is attached, use [ALL CAPS] square-bracket placeholders. "
            "Return ONLY valid JSON matching the schema."
        ),
        temperature=_DRAFT_TEMPERATURE,
    )


async def _generate_clause_list(
    prompt: str,
    agreement_type: Optional[str],
    prior_clauses: List[str],
    doc_grounding: Optional[Dict[str, Any]] = None,
) -> ClauseListLLMResponse:
    """list_of_clauses mode: return ONE comprehensive clause list with drafted bodies."""
    container = get_service_container()
    llm = container.azure_openai_model
    mode_instruction = (
        f"List all clauses that should appear in a "
        f"{agreement_type or 'legal agreement'} as requested by the user, "
        f"and draft the body of each one."
    )
    has_grounding = bool(doc_grounding and doc_grounding.get("parties"))
    parties_block = _format_parties_block(doc_grounding["parties"]) if has_grounding else ""
    governing_law = (doc_grounding or {}).get("governing_law") or ""

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
        "has_doc_grounding": has_grounding,
        "doc_parties_block": parties_block,
        "doc_governing_law": governing_law or "(not specified in document)",
    }
    rendered = load_prompt("describe_draft_generation_prompt", context=context)
    return await llm.generate(
        prompt=rendered,
        context={},
        response_model=ClauseListLLMResponse,
        system_message=(
            "You are an expert legal drafter. Return ONE complete clause list for the "
            "requested agreement type, with a drafted body for every clause. "
            "When a document is attached, use the exact party names and governing law provided. "
            "When no document is attached, use [ALL CAPS] square-bracket placeholders "
            "consistently across every clause. Return ONLY valid JSON matching the schema."
        ),
        temperature=_DRAFT_TEMPERATURE,
    )


def _read_session_context(session_id: str) -> dict:
    container = get_service_container()
    session = container.session_manager.get_or_create_session(session_id)
    return {
        "agreement_type": session.metadata.get("draft_agreement_type"),
        "prior_clauses": session.metadata.get("draft_prior_clauses", []) or [],
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
        prior = session.metadata.get("draft_prior_clauses", []) or []
        session.metadata["draft_prior_clauses"] = prior + new_clause_titles


def _error_response(
    session_id: str,
    mode: str,
    error_type: DescribeDraftErrorType,
    message: str,
) -> DescribeDraftResponse:
    return DescribeDraftResponse(
        session_id=session_id,
        mode=mode,  # type: ignore[arg-type]
        status="error",
        disclaimer=None,
        error_type=error_type,
        error_message=message,
    )


async def _run_single_clause_generation(
    session_id: str,
    clean_prompt: str,
    effective_agreement_type: Optional[str],
    prior_clauses: List[str],
    doc_grounding: Optional[Dict[str, Any]],
    relevant_chunks: List[Dict[str, Any]],
) -> Tuple[Optional[DraftedClause], Optional[DescribeDraftResponse]]:
    """Single-clause generation + validation with one retry.

    Returns (clause, None) on success, or (None, error_response) on failure.
    """
    has_grounding = bool(doc_grounding and doc_grounding.get("parties"))
    validation_error: Optional[str] = None

    for attempt in range(2):
        try:
            raw = await _generate_clause_draft(
                prompt=clean_prompt,
                agreement_type=effective_agreement_type,
                prior_clauses=prior_clauses,
                doc_grounding=doc_grounding,
                relevant_chunks=relevant_chunks,
            )
            _validate_draft_response(
                raw,
                require_placeholders=not has_grounding,
                forbid_placeholders=has_grounding,
            )
            return raw.clause, None
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
            return None, _error_response(
                session_id,
                "single_clause",
                error_type,
                f"LLM generation failed: {error_msg}",
            )

    return None, _error_response(
        session_id,
        "single_clause",
        DescribeDraftErrorType.VALIDATION_FAILED,
        f"Generation validation failed after 2 attempts: {validation_error}",
    )


async def generate_describe_draft(
    prompt: Optional[str],
    session_id: str,
    use_document_context: bool = False,
) -> DescribeDraftResponse:
    """
    Main entry point for the describe-draft agent.

    Flow:
      1. Sanitize input, classify intent (single_clause vs list_of_clauses).
      2. Resolve document grounding from `use_document_context` (the "Use Document
         Context" checkbox):
           - OFF: ignore any attached document, draft a [PLACEHOLDER] template.
           - ON: require a document on the session (else DOCUMENT_REQUIRED), then
             ground the draft in its parties + governing law (+ relevant text).
      3. list_of_clauses: generate the clause list (with drafted bodies), validate, retry once.
         single_clause: generate ONE draft, validate, retry once.
      4. Write session memory (agreement type + drafted clause titles).
      5. Emit audit log.
    """
    start_time = time.time()

    # --- Sanitize input ---
    raw_prompt = prompt or ""
    if not raw_prompt.strip():
        return _error_response(
            session_id,
            "single_clause",
            DescribeDraftErrorType.VALIDATION_FAILED,
            "Prompt must not be empty.",
        )
    try:
        clean_prompt = _sanitize_prompt(raw_prompt)
    except ValueError as e:
        return _error_response(
            session_id,
            "single_clause",
            DescribeDraftErrorType.VALIDATION_FAILED,
            str(e),
        )

    # Read session memory
    ctx = _read_session_context(session_id)
    stored_agreement_type: Optional[str] = ctx["agreement_type"]
    prior_clauses: List[str] = ctx["prior_clauses"]

    # "Use Document Context" is on but nothing is open → tell the user up front,
    # before spending an LLM call on classification.
    container = get_service_container()
    session_obj = container.session_manager.get_or_create_session(session_id)
    if use_document_context and not _session_has_document(session_obj):
        return _error_response(
            session_id,
            "single_clause",
            DescribeDraftErrorType.DOCUMENT_REQUIRED,
            "Use Document Context is on but no document is open on this session. "
            "Open a document first, or turn off Use Document Context to draft a "
            "template.",
        )

    # Classify intent
    try:
        classification = await _classify_intent(clean_prompt)
    except Exception as e:
        logger.error(
            "describe_draft classify error session=%s error=%s", session_id, str(e)
        )
        return _error_response(
            session_id,
            "single_clause",
            DescribeDraftErrorType.LLM_FAILED,
            f"Intent classification failed: {str(e)}",
        )

    mode = classification.mode
    detected_agreement_type = classification.detected_agreement_type

    effective_agreement_type = detected_agreement_type or stored_agreement_type
    clear_prior = (
        detected_agreement_type is not None
        and stored_agreement_type is not None
        and detected_agreement_type.lower() != stored_agreement_type.lower()
    )

    # Resolve document grounding from the "Use Document Context" toggle.
    doc_grounding: Optional[Dict[str, Any]] = None
    if use_document_context:
        doc_grounding = await _get_doc_grounding(session_id)
    has_document = use_document_context
    grounded_in_doc = bool(doc_grounding and doc_grounding.get("parties"))

    clauses_out: List[ClauseListEntry] = []
    clause_out: Optional[DraftedClause] = None
    agreement_summary_out: Optional[str] = None
    units_generated = 0
    validation_error: Optional[str] = None

    if mode == "list_of_clauses":
        list_response: Optional[ClauseListLLMResponse] = None
        for attempt in range(2):
            try:
                raw_list = await _generate_clause_list(
                    prompt=clean_prompt,
                    agreement_type=effective_agreement_type,
                    prior_clauses=prior_clauses if not clear_prior else [],
                    doc_grounding=doc_grounding,
                )
                _validate_clause_list(
                    raw_list,
                    require_placeholders=not grounded_in_doc,
                    forbid_placeholders=grounded_in_doc,
                )
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
                return _error_response(
                    session_id,
                    mode,
                    error_type,
                    f"LLM generation failed: {error_msg}",
                )

        if list_response is None:
            return _error_response(
                session_id,
                mode,
                DescribeDraftErrorType.VALIDATION_FAILED,
                f"Clause-list validation failed after 2 attempts: {validation_error}",
            )

        clauses_out = list_response.clauses
        agreement_summary_out = list_response.agreement_summary
        units_generated = len(clauses_out)
    else:
        # single_clause mode — generate ONE draft.
        relevant_chunks: List[Dict[str, Any]] = []
        if has_document:
            relevant_chunks = await _retrieve_relevant_chunks(session_id, clean_prompt)

        version, error_resp = await _run_single_clause_generation(
            session_id=session_id,
            clean_prompt=clean_prompt,
            effective_agreement_type=effective_agreement_type,
            prior_clauses=prior_clauses if not clear_prior else [],
            doc_grounding=doc_grounding,
            relevant_chunks=relevant_chunks,
        )
        if error_resp is not None:
            return error_resp
        clause_out = version
        units_generated = 1

    # --- Write session memory ---
    new_titles = (
        [clause_out.title]
        if (mode == "single_clause" and clause_out is not None)
        else []
    )
    _write_session_context(
        session_id=session_id,
        agreement_type=effective_agreement_type,
        new_clause_titles=new_titles,
        clear_prior=clear_prior,
    )

    # --- Audit log ---
    latency_ms = int((time.time() - start_time) * 1000)
    logger.info(
        "describe_draft_audit session=%s mode=%s agreement_type=%s "
        "units_generated=%d grounded=%s latency_ms=%d",
        session_id,
        mode,
        effective_agreement_type or "unknown",
        units_generated,
        grounded_in_doc,
        latency_ms,
    )

    return DescribeDraftResponse(
        session_id=session_id,
        mode=mode,
        status="ok",
        agreement_summary=agreement_summary_out,
        clauses=clauses_out,
        clause=clause_out,
        grounded_in_document=grounded_in_doc,
    )
