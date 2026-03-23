"""Version Compare Tool — compares two contract versions in the same session.

Pipeline:
  1. Validate session has 2+ ingested documents
  2. Reconstruct full text per document from session chunks
  3. Pass 1: LLM identifies changes (structured JSON output)
  4. Pass 2: Verification LLM call finds missed changes
  5. Merge results and deduplicate
  6. Assemble deterministic report with computed statistics (NO LLM)

Total LLM calls: 2
"""

import asyncio
import json as _json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.dependencies import get_service_container
from src.schemas.compare import (
    ChangeType,
    CompareStatistics,
    MissingClauseInVersion,
    RiskImpact,
    Significance,
    VersionCompareReport,
    VersionCompareResult,
)
from src.services.session_manager import SessionData

# ── Constants ────────────────────────────────────────────────────

_PROMPT_PATH = Path("src/services/prompts/v1/version_compare_prompt.mustache")

_SYSTEM_MESSAGE = (
    "You are an expert Contract Comparison Analyst. "
    "Find EVERY difference between the two contract versions — recall is critical. "
    "Check every clause sentence by sentence, all numbers/dates/durations, "
    "term and termination clauses, footers, and headers. "
    "A single clause may have multiple changes — capture all of them. "
    "IMPORTANT: After finding modifications, identify clauses that exist in one document "
    "but are ENTIRELY ABSENT in the other — report these in the `missing_clauses` array. "
    "If a clause is in Document A but not in Document B, set missing_in to 'document_b'. "
    "If a clause is in Document B but not in Document A, set missing_in to 'document_a'. "
    "Return ONLY valid JSON. Use exact verbatim quotes from the documents."
)

_VERIFY_SYSTEM_MESSAGE = (
    "You are a senior Contract Comparison Reviewer. "
    "Another analyst already found some changes. Your job is to find changes they MISSED. "
    "Focus on intra-clause modifications, numerical changes, dates, new sentences "
    "inserted within existing clauses, AND any missing clauses (clauses present in one "
    "document but entirely absent in the other). "
    "Return ONLY valid JSON. Only report NEW changes and missing clauses not already in the list."
)


# ── Step 1: Session Validation ───────────────────────────────────


def _get_session(session_id: str) -> SessionData:
    """Retrieve session data or raise ValueError."""
    container = get_service_container()
    session = container.session_manager.get_session(session_id)
    if not session:
        raise ValueError(f"Session '{session_id}' not found or expired.")
    if len(session.chunk_store) == 0:
        raise ValueError("No document ingested in this session.")
    return session


# ── Step 2: Document Text Reconstruction ─────────────────────────


def _get_document_texts(session: SessionData) -> List[Tuple[str, str]]:
    """Reconstruct full text per document from session chunks.

    Returns:
        List of (document_id, full_text) tuples, ordered by first chunk index
        (first ingested document = Document A).

    Raises:
        ValueError: If fewer than 2 documents found in session.
    """
    if len(session.documents) < 2:
        raise ValueError(
            f"Version compare requires at least 2 documents in the session. "
            f"Found {len(session.documents)}. "
            f"Please ingest two contract versions first."
        )

    doc_texts: List[Tuple[str, str, int]] = []  # (doc_id, full_text, min_chunk_idx)

    for doc_id, doc_info in session.documents.items():
        chunk_indices = doc_info.get("chunk_indices", [])
        if not chunk_indices:
            continue

        # Sort by chunk index to preserve document order
        sorted_indices = sorted(chunk_indices)

        # Reconstruct full text from chunks
        parts = []
        for idx in sorted_indices:
            chunk = session.chunk_store.get(idx)
            if chunk and chunk.content:
                parts.append(chunk.content)

        if parts:
            full_text = "\n\n".join(parts)
            min_idx = sorted_indices[0]
            doc_texts.append((doc_id, full_text, min_idx))

    if len(doc_texts) < 2:
        raise ValueError(
            f"Version compare requires at least 2 documents with content. "
            f"Found {len(doc_texts)} documents with text."
        )

    # Sort by min chunk index so first ingested = Document A
    doc_texts.sort(key=lambda x: x[2])

    return [(doc_id, text) for doc_id, text, _ in doc_texts]


# ── Step 3: LLM Calls ───────────────────────────────────────────


def _sync_llm_call(
    llm: Any,
    prompt_template: str,
    context: Dict[str, Any],
    system_message: str,
) -> VersionCompareResult:
    """Synchronous LLM call for use with asyncio.to_thread().

    The OpenAI client is synchronous, so this runs in a thread pool
    to avoid blocking the event loop.
    """
    rendered_prompt = llm.render_prompt_template(
        prompt=prompt_template, context=context
    )

    response = llm.client.chat.completions.create(
        model=llm.deployment_name,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": rendered_prompt},
        ],
        temperature=0.0,
        max_tokens=16384,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": VersionCompareResult.__name__,
                "schema": VersionCompareResult.model_json_schema(),
                "strict": False,
            },
        },
    )

    response_text = response.choices[0].message.content
    if response_text is None:
        raise ValueError("Empty response from LLM model.")

    response_data = _json.loads(response_text)
    return VersionCompareResult.model_validate(response_data)


def _format_existing_changes(result: VersionCompareResult) -> str:
    """Format pass-1 changes into a readable list for the verification prompt."""
    lines = []
    for i, change in enumerate(result.changes, 1):
        lines.append(f"{i}. [{change.change_type.value.upper()}] {change.clause_title}")
        lines.append(f"   Summary: {change.change_summary}")
        if change.original_text:
            # Truncate long quotes for the verification context
            orig = change.original_text[:200] + "..." if len(change.original_text) > 200 else change.original_text
            lines.append(f"   Original: \"{orig}\"")
        if change.revised_text:
            rev = change.revised_text[:200] + "..." if len(change.revised_text) > 200 else change.revised_text
            lines.append(f"   Revised: \"{rev}\"")
        lines.append("")
    # Include missing clauses found in pass 1
    if result.missing_clauses:
        lines.append("--- MISSING CLAUSES ALREADY FOUND ---")
        for j, mc in enumerate(result.missing_clauses, 1):
            lines.append(f"{j}. [{mc.missing_in.upper()}] {mc.clause_title}")
            lines.append(f"   Impact: {mc.impact_summary}")
            lines.append("")

    return "\n".join(lines) if lines else "No changes found in first pass."


def _merge_results(
    pass1: VersionCompareResult, pass2: VersionCompareResult
) -> VersionCompareResult:
    """Merge pass-1 and pass-2 results, deduplicating by clause_title + change_type."""
    seen = set()
    merged_changes = []

    for change in pass1.changes:
        key = (change.clause_title.lower().strip(), change.change_type)
        if key not in seen:
            seen.add(key)
            merged_changes.append(change)

    for change in pass2.changes:
        key = (change.clause_title.lower().strip(), change.change_type)
        if key not in seen:
            seen.add(key)
            merged_changes.append(change)

    # Merge missing clauses, dedup by clause_title + missing_in
    seen_missing = set()
    merged_missing: List[MissingClauseInVersion] = []

    for mc in pass1.missing_clauses:
        key = (mc.clause_title.lower().strip(), mc.missing_in)
        if key not in seen_missing:
            seen_missing.add(key)
            merged_missing.append(mc)

    for mc in pass2.missing_clauses:
        key = (mc.clause_title.lower().strip(), mc.missing_in)
        if key not in seen_missing:
            seen_missing.add(key)
            merged_missing.append(mc)

    # Build executive summary
    extras = len(pass2.changes) + len(pass2.missing_clauses)
    if extras:
        total = len(merged_changes)
        total_missing = len(merged_missing)
        executive_summary = (
            f"{pass1.executive_summary} "
            f"A verification pass identified {extras} additional finding(s) "
            f"for a total of {total} changes and {total_missing} missing clause(s)."
        )
    else:
        executive_summary = pass1.executive_summary

    return VersionCompareResult(
        _analysis_reasoning=pass1.analysis_reasoning,
        changes=merged_changes,
        missing_clauses=merged_missing,
        executive_summary=executive_summary,
        overall_risk_impact=pass1.overall_risk_impact,
    )


# ── Step 4: Deterministic Report Assembly ────────────────────────


def _build_report(
    session_id: str,
    doc_a_id: str,
    doc_b_id: str,
    result: VersionCompareResult,
) -> VersionCompareReport:
    """Assemble the final report deterministically. No LLM call."""

    changes = result.changes

    missing = result.missing_clauses

    # Compute statistics
    stats = CompareStatistics(
        total_changes=len(changes),
        clauses_added=sum(1 for c in changes if c.change_type == ChangeType.ADDED),
        clauses_removed=sum(1 for c in changes if c.change_type == ChangeType.REMOVED),
        clauses_modified=sum(
            1 for c in changes if c.change_type == ChangeType.MODIFIED
        ),
        high_significance=sum(
            1 for c in changes if c.significance == Significance.HIGH
        ),
        medium_significance=sum(
            1 for c in changes if c.significance == Significance.MEDIUM
        ),
        low_significance=sum(
            1 for c in changes if c.significance == Significance.LOW
        ),
        clauses_missing_in_original=sum(
            1 for mc in missing if mc.missing_in == "document_a"
        ),
        clauses_missing_in_revised=sum(
            1 for mc in missing if mc.missing_in == "document_b"
        ),
    )

    # Group clause titles by change type
    changes_by_type: Dict[str, List[str]] = {}
    for change_type in ChangeType:
        titles = [c.clause_title for c in changes if c.change_type == change_type]
        if titles:
            changes_by_type[change_type.value] = titles

    # Identify high-risk changes (risk increased AND high significance)
    high_risk_changes = [
        c.clause_title
        for c in changes
        if c.risk_impact == RiskImpact.INCREASED
        and c.significance == Significance.HIGH
    ]

    return VersionCompareReport(
        session_id=session_id,
        document_a_id=doc_a_id,
        document_b_id=doc_b_id,
        statistics=stats,
        executive_summary=result.executive_summary,
        overall_risk_impact=result.overall_risk_impact,
        changes=changes,
        changes_by_type=changes_by_type,
        high_risk_changes=high_risk_changes,
        missing_clauses=missing,
    )


# ── Main Entry Point ────────────────────────────────────────────


async def compare_versions(session_id: str) -> VersionCompareReport:
    """Compare two contract versions in the given session.

    Pipeline:
      1. Validate session has 2+ documents
      2. Reconstruct full text per document
      3. Pass 1: LLM identifies changes
      4. Pass 2: Verification LLM finds missed changes
      5. Merge and deduplicate
      6. Assemble report deterministically

    Args:
        session_id: Active session with two ingested contract versions.

    Returns:
        VersionCompareReport with all clause changes and statistics.
    """
    # Step 1: Validate session
    session = _get_session(session_id)

    # Step 2: Get document texts (ordered by ingestion time)
    doc_texts = _get_document_texts(session)
    doc_a_id, doc_a_text = doc_texts[0]
    doc_b_id, doc_b_text = doc_texts[1]

    container = get_service_container()
    llm = container.azure_openai_model

    # Step 3: Pass 1 — Initial comparison
    prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
    context = {
        "document_a_text": doc_a_text,
        "document_b_text": doc_b_text,
    }

    pass1_result = await asyncio.to_thread(
        _sync_llm_call, llm, prompt_template, context, _SYSTEM_MESSAGE
    )

    # Step 4: Pass 2 — Verification (find missed changes)
    verify_context = {
        "is_verification": True,
        "existing_changes": _format_existing_changes(pass1_result),
        "document_a_text": doc_a_text,
        "document_b_text": doc_b_text,
    }

    pass2_result = await asyncio.to_thread(
        _sync_llm_call, llm, prompt_template, verify_context, _VERIFY_SYSTEM_MESSAGE
    )

    # Step 5: Merge results
    merged_result = _merge_results(pass1_result, pass2_result)

    # Step 6: Build deterministic report
    return _build_report(session_id, doc_a_id, doc_b_id, merged_result)
