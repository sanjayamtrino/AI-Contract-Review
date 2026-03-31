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


# --- Stage 3: LLM Comparison (stub for Task 2) ---

# compare_clause_pair, build_change_entry, build_added_entry, build_removed_entry,
# group_by_section, and run() will be implemented in Task 2.


async def run(session_id: str, document_id_a: str, document_id_b: str) -> CompareResponse:
    """Compare two documents within a session. Stub -- Task 2 completes this."""
    raise NotImplementedError("run() will be completed in Task 2")
