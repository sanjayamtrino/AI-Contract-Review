"""Compare Agent -- Extract-Match-Compare pipeline for contract document comparison.

Deterministic pipeline (no LLM routing):
1. Extract clauses from two documents via session chunk stores
2. Match clauses using embedding cosine similarity (greedy 1:1 + one-to-many scan)
3. Compare each matched pair via focused LLM call
4. Assemble section-grouped diff with risk assessment
"""

import logging
from datetime import datetime, timezone
from typing import List

import numpy as np

from src.dependencies import get_service_container
from src.schemas.compare import (
    ChangeEntry,
    ClauseMatch,
    ClauseUnit,
    CompareResponse,
    ComparisonMetadata,
    ClauseComparisonResult,
    MatchResult,
    SectionDiff,
)
from src.services.prompts.v1 import load_prompt

logger = logging.getLogger("AI_Contract.CompareAgent")

# --- Constants ---

SIMILARITY_THRESHOLD = 0.72  # Below this, clauses are unmatched
IDENTICAL_THRESHOLD = 0.98   # Above this for ALL pairs, short-circuit as identical
MAX_LLM_CALLS = 50           # Cap on per-comparison LLM calls


# --- Stage 1: Clause Extraction ---


def extract_clauses(session, document_id: str) -> List[ClauseUnit]:
    """Extract clause units from a document's chunks in the session.

    Args:
        session: SessionData containing chunk_store and documents.
        document_id: ID of the document to extract clauses from.

    Returns:
        List of ClauseUnit, one per chunk belonging to the document.

    Raises:
        ValueError: If document_id is not found in the session.
    """
    doc_info = session.documents.get(document_id)
    if doc_info is None:
        raise ValueError(
            f"Document '{document_id}' not found in session '{session.session_id}'"
        )

    chunk_indices = doc_info.get("chunk_indices", [])
    clauses: List[ClauseUnit] = []

    for position, chunk_idx in enumerate(chunk_indices):
        chunk = session.chunk_store.get(chunk_idx)
        if chunk is None:
            logger.warning(
                "Chunk index %d not found in chunk_store for document %s",
                chunk_idx,
                document_id,
            )
            continue

        clause = ClauseUnit(
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            section_heading=chunk.metadata.get("section_heading"),
            document_id=document_id,
            position=position,
        )
        clauses.append(clause)

    return clauses


# --- Stage 2: Embedding-based Matching ---


def cosine_similarity(vec1, vec2) -> float:
    """Compute cosine similarity between two vectors.

    Handles zero-vector edge case by returning 0.0.
    """
    a = np.array(vec1, dtype=np.float64)
    b = np.array(vec2, dtype=np.float64)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


async def match_clauses(clauses_a, clauses_b, embedding_service) -> MatchResult:
    """Match clauses across two documents using embedding cosine similarity.

    Performs greedy 1:1 matching followed by a one-to-many scan for split detection.

    Args:
        clauses_a: Clauses from document A (original).
        clauses_b: Clauses from document B (revised).
        embedding_service: Service to generate embeddings.

    Returns:
        MatchResult with paired, unmatched_a, and unmatched_b lists.
    """
    if not clauses_a or not clauses_b:
        return MatchResult(
            paired=[],
            unmatched_a=list(clauses_a),
            unmatched_b=list(clauses_b),
        )

    # Generate embeddings for all clauses
    embeddings_a = []
    for clause in clauses_a:
        emb = await embedding_service.generate_embeddings(clause.content)
        embeddings_a.append(emb)

    embeddings_b = []
    for clause in clauses_b:
        emb = await embedding_service.generate_embeddings(clause.content)
        embeddings_b.append(emb)

    # Compute NxM similarity matrix
    n = len(clauses_a)
    m = len(clauses_b)
    sim_matrix = [[0.0] * m for _ in range(n)]

    for i in range(n):
        for j in range(m):
            sim_matrix[i][j] = cosine_similarity(embeddings_a[i], embeddings_b[j])

    # Short-circuit check: if all best-match similarities > IDENTICAL_THRESHOLD,
    # documents are effectively identical
    best_matches_above_identical = True
    for i in range(n):
        best_sim = max(sim_matrix[i]) if m > 0 else 0.0
        if best_sim <= IDENTICAL_THRESHOLD:
            best_matches_above_identical = False
            break

    if best_matches_above_identical and n > 0:
        # Also check from B's perspective
        for j in range(m):
            best_sim = max(sim_matrix[i][j] for i in range(n)) if n > 0 else 0.0
            if best_sim <= IDENTICAL_THRESHOLD:
                best_matches_above_identical = False
                break

    if best_matches_above_identical and n > 0 and m > 0:
        logger.info("All clause similarities above %.2f -- documents are identical", IDENTICAL_THRESHOLD)
        return MatchResult(paired=[], unmatched_a=[], unmatched_b=[])

    # Greedy 1:1 matching: for each A clause, find best B match above threshold
    paired: List[ClauseMatch] = []
    used_b = set()

    # Build sorted list of (similarity, i, j) for greedy assignment
    all_pairs = []
    for i in range(n):
        for j in range(m):
            if sim_matrix[i][j] >= SIMILARITY_THRESHOLD:
                all_pairs.append((sim_matrix[i][j], i, j))

    # Sort by similarity descending for best-first greedy matching
    all_pairs.sort(key=lambda x: x[0], reverse=True)

    used_a = set()
    for sim, i, j in all_pairs:
        if i in used_a or j in used_b:
            continue
        paired.append(
            ClauseMatch(
                clause_a=clauses_a[i],
                clause_b=clauses_b[j],
                similarity=sim,
            )
        )
        used_a.add(i)
        used_b.add(j)

    # One-to-many scan: check remaining unmatched B clauses against ALL A clauses
    # (including already-matched) for potential splits
    remaining_b = [j for j in range(m) if j not in used_b]
    for j in remaining_b:
        best_sim = 0.0
        best_i = -1
        for i in range(n):
            if sim_matrix[i][j] > best_sim:
                best_sim = sim_matrix[i][j]
                best_i = i
        if best_sim >= SIMILARITY_THRESHOLD and best_i >= 0:
            paired.append(
                ClauseMatch(
                    clause_a=clauses_a[best_i],
                    clause_b=clauses_b[j],
                    similarity=best_sim,
                )
            )
            used_b.add(j)

    # Determine unmatched clauses
    unmatched_a = [clauses_a[i] for i in range(n) if i not in used_a]
    unmatched_b = [clauses_b[j] for j in range(m) if j not in used_b]

    logger.info(
        "Matching complete: %d paired, %d removed (unmatched A), %d added (unmatched B)",
        len(paired),
        len(unmatched_a),
        len(unmatched_b),
    )

    return MatchResult(paired=paired, unmatched_a=unmatched_a, unmatched_b=unmatched_b)


# --- Stage 3: LLM Comparison ---


async def compare_clause_pair(
    clause_a: ClauseUnit,
    clause_b: ClauseUnit,
    llm_client,
) -> ClauseComparisonResult:
    """Compare a matched clause pair via a focused LLM call.

    Each call sees only the two clause texts -- no cross-contamination.

    Args:
        clause_a: Clause from the original document.
        clause_b: Clause from the revised document.
        llm_client: AzureOpenAIModel instance.

    Returns:
        ClauseComparisonResult with change_type, risk_level, etc.
    """
    prompt = load_prompt("clause_comparison_prompt")
    context = {
        "clause_a_text": clause_a.content,
        "clause_b_text": clause_b.content,
    }

    result = await llm_client.generate(
        prompt=prompt,
        context=context,
        response_model=ClauseComparisonResult,
        system_message=(
            "You are comparing two contract clause versions. "
            "Quote exact text differences. Do not paraphrase."
        ),
        temperature=0.1,
    )
    return result


# --- Stage 4: Change Entry Builders ---


def build_change_entry(match: ClauseMatch, comparison: ClauseComparisonResult) -> ChangeEntry:
    """Convert a matched pair's LLM comparison result into a ChangeEntry.

    For reordered clauses (similarity > 0.98), risk_level is left as None
    (informational only).
    """
    clause_id = (
        match.clause_a.section_heading
        or f"clause-{match.clause_a.position}"
    )

    # Reordered clauses are informational -- no risk level
    if comparison.change_type == "reordered":
        return ChangeEntry(
            change_type="reordered",
            clause_id=clause_id,
            clause_heading=match.clause_a.section_heading,
            old_text=match.clause_a.content,
            new_text=match.clause_b.content,
            risk_level=None,
            risk_justification=None,
            affected_party=None,
            is_substantive=False,
            legal_implication=comparison.legal_implication,
        )

    return ChangeEntry(
        change_type=comparison.change_type,
        clause_id=clause_id,
        clause_heading=match.clause_a.section_heading,
        old_text=match.clause_a.content,
        new_text=match.clause_b.content,
        risk_level=comparison.risk_level,
        risk_justification=comparison.risk_justification,
        affected_party=comparison.affected_party,
        is_substantive=comparison.is_substantive,
        legal_implication=comparison.legal_implication,
    )


def build_added_entry(clause: ClauseUnit) -> ChangeEntry:
    """Build a ChangeEntry for a clause present only in the revised document."""
    return ChangeEntry(
        change_type="added",
        clause_id=clause.section_heading or f"clause-{clause.position}",
        clause_heading=clause.section_heading,
        old_text=None,
        new_text=clause.content,
        risk_level="high",
        risk_justification="New clause not present in original document -- requires review",
        affected_party=None,
        is_substantive=True,
        legal_implication="New clause introduces terms not in the original agreement",
    )


def build_removed_entry(clause: ClauseUnit) -> ChangeEntry:
    """Build a ChangeEntry for a clause present only in the original document."""
    return ChangeEntry(
        change_type="removed",
        clause_id=clause.section_heading or f"clause-{clause.position}",
        clause_heading=clause.section_heading,
        old_text=clause.content,
        new_text=None,
        risk_level="high",
        risk_justification="Clause removed from original document -- protection may be lost",
        affected_party=None,
        is_substantive=True,
        legal_implication="Removal of this clause may eliminate previously agreed protections",
    )


# --- Stage 5: Section Grouping ---


def group_by_section(changes: List[ChangeEntry]) -> List[SectionDiff]:
    """Group change entries by their clause_heading into SectionDiff objects.

    Clauses with no heading are grouped under 'General / Ungrouped'.
    """
    from collections import OrderedDict

    sections: OrderedDict[str, List[ChangeEntry]] = OrderedDict()
    for change in changes:
        heading = change.clause_heading or "General / Ungrouped"
        if heading not in sections:
            sections[heading] = []
        sections[heading].append(change)

    return [
        SectionDiff(section_heading=heading, changes=section_changes)
        for heading, section_changes in sections.items()
    ]


# --- Pipeline Orchestrator ---


async def run(session_id: str, document_id_a: str, document_id_b: str) -> CompareResponse:
    """Compare two documents within a session.

    Pipeline: extract clauses -> match via embeddings -> compare pairs via LLM
    -> group by section -> return structured diff.

    Args:
        session_id: Session containing both ingested documents.
        document_id_a: Document ID of the original/baseline contract.
        document_id_b: Document ID of the revised contract.

    Returns:
        CompareResponse with sections, metadata, or error.
    """
    try:
        container = get_service_container()
        session_manager = container.session_manager
        embedding_service = container.embedding_service
        llm_client = container.azure_openai_model

        # Validate session
        session = session_manager.get_session(session_id)
        if session is None:
            return CompareResponse(error=f"Session '{session_id}' not found")

        # Validate documents exist
        if document_id_a not in session.documents:
            return CompareResponse(
                error=f"Document '{document_id_a}' not found in session"
            )
        if document_id_b not in session.documents:
            return CompareResponse(
                error=f"Document '{document_id_b}' not found in session"
            )

        # Stage 1: Extract clauses
        clauses_a = extract_clauses(session, document_id_a)
        clauses_b = extract_clauses(session, document_id_b)
        logger.info(
            "Extracted clauses: doc_a=%d, doc_b=%d", len(clauses_a), len(clauses_b)
        )

        # Stage 2: Match clauses via embedding similarity
        match_result = await match_clauses(clauses_a, clauses_b, embedding_service)

        # Build metadata
        doc_a_meta = session.documents[document_id_a].get("metadata", {})
        doc_b_meta = session.documents[document_id_b].get("metadata", {})

        # Short-circuit: identical documents (empty match result with no unmatched)
        if (
            not match_result.paired
            and not match_result.unmatched_a
            and not match_result.unmatched_b
        ):
            logger.info("Documents are identical -- no changes detected")
            metadata = ComparisonMetadata(
                document_name_a=doc_a_meta.get("filename"),
                document_name_b=doc_b_meta.get("filename"),
                session_id=session_id,
                document_id_a=document_id_a,
                document_id_b=document_id_b,
                comparison_timestamp=datetime.now(timezone.utc).isoformat(),
                total_changes=0,
            )
            return CompareResponse(sections=[], metadata=metadata)

        # Stage 3: Compare each matched pair via LLM (capped at MAX_LLM_CALLS)
        all_changes: List[ChangeEntry] = []
        llm_calls_made = 0

        for match in match_result.paired:
            if llm_calls_made >= MAX_LLM_CALLS:
                logger.warning(
                    "LLM call cap reached (%d). Remaining %d pairs skipped.",
                    MAX_LLM_CALLS,
                    len(match_result.paired) - llm_calls_made,
                )
                break

            comparison = await compare_clause_pair(
                match.clause_a, match.clause_b, llm_client
            )
            entry = build_change_entry(match, comparison)
            all_changes.append(entry)
            llm_calls_made += 1

        logger.info("LLM comparison calls made: %d", llm_calls_made)

        # Build added/removed entries (no LLM needed)
        for clause in match_result.unmatched_b:
            all_changes.append(build_added_entry(clause))

        for clause in match_result.unmatched_a:
            all_changes.append(build_removed_entry(clause))

        # Stage 4: Group by section
        sections = group_by_section(all_changes)

        # Build metadata
        metadata = ComparisonMetadata(
            document_name_a=doc_a_meta.get("filename"),
            document_name_b=doc_b_meta.get("filename"),
            session_id=session_id,
            document_id_a=document_id_a,
            document_id_b=document_id_b,
            comparison_timestamp=datetime.now(timezone.utc).isoformat(),
            total_changes=len(all_changes),
        )

        return CompareResponse(sections=sections, metadata=metadata)

    except ValueError as e:
        logger.error("Validation error in compare agent: %s", str(e))
        return CompareResponse(error=str(e))
    except Exception as e:
        logger.error("Internal error in compare agent: %s", str(e), exc_info=True)
        return CompareResponse(error="Internal comparison error")
