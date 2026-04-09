import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from docx.document import Document

from src.config.logging import Logger
from src.dependencies import get_service_container
from src.schemas.comparision import (
    ChangeEntry,
    ClauseComparisonLLMResponse,
    ClauseUnit,
    CompareResponse,
    CompareSummary,
    MatchResult,
    SectionGroup,
)
from src.schemas.registry import ParseResult
from src.services.registry.registry import ParserRegistry

# Similarity thresholds
SIMILARITY_THRESHOLD = 0.72
IDENTICAL_THRESHOLD = 0.98
SPLIT_MERGE_THRESHOLD = 0.75
MAX_LLM_CALLS = 50

AGENT_NAME = "document_comparison_agent"

logger = Logger().logger
registry = None


def get_registry():
    global registry
    if registry is None:
        registry = ParserRegistry()
    return registry


parser = None


def get_parser():
    global parser
    if parser is None:
        parser = get_registry().get_parser()
    return parser


# --- Stage 1: Clause Extraction ---


def _extract_heading_fallback(content: str) -> Optional[str]:
    """Derive heading from first line when metadata lacks section_heading."""

    first_line = content.strip().split("\n")[0].strip()
    if re.match(r"^(\d+\.[\d.]*\s|Section\s|ARTICLE\s|[A-Z][A-Z\s]{4,}$)", first_line):
        return first_line

    # ALL CAPS heading on its own line
    if re.match(r"^[A-Z][A-Z\s]{4,}$", first_line):
        return first_line

    # Title-case phrase before first ". " — e.g. "Audit Rights. Content..."
    m = re.match(r"^([A-Z][A-Za-z\s/&,-]{2,60})\.\s", first_line)
    if m:
        return m.group(1).strip()

    return None


def extract_clauses(document: ParseResult) -> List[ClauseUnit]:
    """Extract clauses from a document's chunks, using metadata and fallback heuristics for headings."""

    clauses: List[ClauseUnit] = []
    order = 0

    for chunk in document.chunks:
        if chunk is None:
            continue

        content = chunk.content.strip()
        if not content:
            continue

        heading = chunk.metadata.get("section_heading")
        if not heading:
            heading = _extract_heading_fallback(content) or chunk.metadata.get("section_heading")

        clauses.append(ClauseUnit(clause_id=f"{chunk.chunk_id}", heading=heading, content=content, position=chunk.chunk_index, doc_order=order, embedding=chunk.embedding_vector or []))
        order += 1

    return clauses


# --- Stage 2: Clause Matching ---


def _compute_similarity_matrix(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix between two embedding sets."""

    norms_a = np.linalg.norm(emb_a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(emb_b, axis=1, keepdims=True)
    emb_a_norm = emb_a / np.maximum(norms_a, 1e-10)
    emb_b_norm = emb_b / np.maximum(norms_b, 1e-10)
    return emb_a_norm @ emb_b_norm.T


def _cosine_similarity(emb_a: List[float], emb_b: List[float]) -> float:
    """Cosine similarity between two single embedding vectors."""

    a = np.array(emb_a, dtype=np.float32).reshape(1, -1)
    b = np.array(emb_b, dtype=np.float32).reshape(1, -1)
    return float(_compute_similarity_matrix(a, b)[0][0])


def _greedy_match(sim_matrix: np.ndarray, threshold: float = SIMILARITY_THRESHOLD) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Greedy 1:1 matching — pick best pair, remove both, repeat."""

    n, m = sim_matrix.shape
    pairs: List[Tuple[int, int, float]] = []
    used_a: set = set()
    used_b: set = set()

    flat = [(float(sim_matrix[i][j]), i, j) for i in range(n) for j in range(m)]
    flat.sort(reverse=True)

    for score, i, j in flat:
        if score < threshold:
            break
        if i in used_a or j in used_b:
            continue
        pairs.append((i, j, score))
        used_a.add(i)
        used_b.add(j)

    unmatched_a = [i for i in range(n) if i not in used_a]
    unmatched_b = [j for j in range(m) if j not in used_b]
    return pairs, unmatched_a, unmatched_b


async def _ensure_embeddings(clauses_a: List[ClauseUnit], clauses_b: List[ClauseUnit], indices_a: List[int], indices_b: List[int], embedding_service) -> None:
    """Generate embeddings for clauses that don't have them yet."""

    for idx in indices_a:
        if not clauses_a[idx].embedding:
            clauses_a[idx].embedding = await embedding_service.generate_embeddings(clauses_a[idx].content)
    for idx in indices_b:
        if not clauses_b[idx].embedding:
            clauses_b[idx].embedding = await embedding_service.generate_embeddings(clauses_b[idx].content)


async def match_clauses(clauses_a: List[ClauseUnit], clauses_b: List[ClauseUnit], embedding_service) -> MatchResult:
    """Match clauses between two documents using a combination of heading-based and embedding-based similarity."""

    n = len(clauses_a)
    m = len(clauses_b)

    # Step 1: heading-based matching (unique headings only)
    heading_pairs: List[Tuple[int, int, float]] = []
    used_a: set = set()
    used_b: set = set()

    heading_map_a: Dict[str, List[int]] = {}
    for i, clause in enumerate(clauses_a):
        if clause.heading:
            key = clause.heading.strip().lower()
            heading_map_a.setdefault(key, []).append(i)

    heading_map_b: Dict[str, List[int]] = {}
    for j, clause in enumerate(clauses_b):
        if clause.heading:
            key = clause.heading.strip().lower()
            heading_map_b.setdefault(key, []).append(j)

    heading_matched_indices: List[Tuple[int, int]] = []

    for key, indices_a_list in heading_map_a.items():
        indices_b_list = heading_map_b.get(key, [])
        if len(indices_a_list) == 1 and len(indices_b_list) == 1:
            i, j = indices_a_list[0], indices_b_list[0]
            heading_matched_indices.append((i, j))
            used_a.add(i)
            used_b.add(j)
            logger.debug(f"Heading match: '{key}' -> A[{i}] <-> B[{j}]")

    # Generate embeddings for all clauses
    await _ensure_embeddings(clauses_a, clauses_b, list(range(n)), list(range(m)), embedding_service)

    # Compute content similarity for heading-matched pairs
    for i, j in heading_matched_indices:
        sim = _cosine_similarity(clauses_a[i].embedding, clauses_b[j].embedding)
        heading_pairs.append((i, j, sim))
        logger.info(f"Heading match: '{clauses_a[i].heading}' — similarity: {sim:.4f}")

    # Step 2: embedding similarity for remaining clauses
    remaining_a = [i for i in range(n) if i not in used_a]
    remaining_b = [j for j in range(m) if j not in used_b]

    embedding_pairs: List[Tuple[int, int, float]] = []

    if remaining_a and remaining_b:
        emb_a = np.array([clauses_a[i].embedding for i in remaining_a], dtype=np.float32)
        emb_b = np.array([clauses_b[j].embedding for j in remaining_b], dtype=np.float32)

        sim_matrix = _compute_similarity_matrix(emb_a, emb_b)
        local_pairs, local_unmatched_a, local_unmatched_b = _greedy_match(sim_matrix)

        for li, lj, score in local_pairs:
            embedding_pairs.append((remaining_a[li], remaining_b[lj], score))

        final_unmatched_a = [remaining_a[li] for li in local_unmatched_a]
        final_unmatched_b = [remaining_b[lj] for lj in local_unmatched_b]
    else:
        final_unmatched_a = remaining_a
        final_unmatched_b = remaining_b

    return MatchResult(
        matched_pairs=heading_pairs + embedding_pairs,
        unmatched_a=final_unmatched_a,
        unmatched_b=final_unmatched_b,
    )


# --- Stage 2.5: Split/Merge Detection ---


def _detect_splits_and_merges(match_result: MatchResult, clauses_a: List[ClauseUnit], clauses_b: List[ClauseUnit]) -> Tuple[List[ChangeEntry], List[int], List[int]]:
    """Detect splits (1 A -> many B) and merges (many A -> 1 B) among unmatched clauses."""

    entries: List[ChangeEntry] = []
    explained_a: set = set()
    explained_b: set = set()

    matched_a_indices = {idx_a for idx_a, _, _ in match_result.matched_pairs}
    matched_b_indices = {idx_b for _, idx_b, _ in match_result.matched_pairs}

    # Splits: unmatched B clause similar to a matched A clause
    for j in match_result.unmatched_b:
        best_sim, best_a_idx = 0.0, -1
        for idx_a in matched_a_indices:
            sim = _cosine_similarity(clauses_a[idx_a].embedding, clauses_b[j].embedding)
            if sim > best_sim:
                best_sim, best_a_idx = sim, idx_a

        if best_sim >= SPLIT_MERGE_THRESHOLD and best_a_idx >= 0:
            clause_a, clause_b = clauses_a[best_a_idx], clauses_b[j]
            entries.append(
                ChangeEntry(
                    clause_title=clause_a.heading or clause_b.heading or f"Clause at position {clause_a.doc_order + 1}",
                    change_type="modified",
                    change_summary="Clause split into multiple clauses. Content preserved but restructured.",
                    risk_level="low",
                    risk_impact="Low risk — structural reorganisation only, no substantive change.",
                    original_text=clause_a.content,
                    revised_text=clause_b.content,
                )
            )
            explained_b.add(j)
            logger.info(f"Split detected: A[{best_a_idx}] -> B[{j}] (sim={best_sim:.4f})")

    # Merges: unmatched A clause similar to a matched B clause
    for i in match_result.unmatched_a:
        best_sim, best_b_idx = 0.0, -1
        for idx_b in matched_b_indices:
            sim = _cosine_similarity(clauses_a[i].embedding, clauses_b[idx_b].embedding)
            if sim > best_sim:
                best_sim, best_b_idx = sim, idx_b

        if best_sim >= SPLIT_MERGE_THRESHOLD and best_b_idx >= 0:
            clause_a, clause_b = clauses_a[i], clauses_b[best_b_idx]
            entries.append(
                ChangeEntry(
                    clause_title=clause_a.heading or clause_b.heading or f"Clause at position {clause_a.doc_order + 1}",
                    change_type="modified",
                    change_summary="Clause merged with another clause. Content preserved but combined.",
                    risk_level="low",
                    risk_impact="Low risk — structural reorganisation only, no substantive change.",
                    original_text=clause_a.content,
                    revised_text=clause_b.content,
                )
            )
            explained_a.add(i)
            logger.info(f"Merge detected: A[{i}] -> B[{best_b_idx}] (sim={best_sim:.4f})")

    remaining_a = [i for i in match_result.unmatched_a if i not in explained_a]
    remaining_b = [j for j in match_result.unmatched_b if j not in explained_b]

    return entries, remaining_a, remaining_b


# --- Stage 2.75: Containment Reconciliation ---


def _normalize_for_containment(text: str) -> str:
    """Collapse whitespace for robust substring containment checks."""

    return " ".join(text.split()).lower()


def _reconcile_containment(unmatched_a: List[int], unmatched_b: List[int], clauses_a: List[ClauseUnit], clauses_b: List[ClauseUnit]) -> Tuple[List[ChangeEntry], List[int], List[int]]:
    """Detect cases where an unmatched clause from one document is fully contained within an unmatched clause from the other document, indicating an addition or removal rather than a modification. Returns new change entries and updated unmatched lists."""

    entries: List[ChangeEntry] = []
    explained_a: set = set()
    explained_b: set = set()

    # Check if removed (A) clause text is contained in an added (B) clause
    for i in unmatched_a:
        norm_a = _normalize_for_containment(clauses_a[i].content)
        if len(norm_a) < 20:
            continue

        for j in unmatched_b:
            if j in explained_b:
                continue
            norm_b = _normalize_for_containment(clauses_b[j].content)

            if norm_a in norm_b:
                clause_a, clause_b = clauses_a[i], clauses_b[j]
                entries.append(
                    ChangeEntry(
                        clause_title=clause_b.heading or clause_a.heading or f"Clause at position {clause_b.doc_order + 1}",
                        change_type="added",
                        change_summary="New content added alongside existing clause text in Document B.",
                        risk_level="medium",
                        risk_impact="New clause language introduced — review for additional obligations or rights.",
                        revised_text=clause_b.content,
                    )
                )
                explained_a.add(i)
                explained_b.add(j)
                logger.info(f"Containment: A[{i}] found inside B[{j}] — reporting as addition")
                break

    # Reverse: B text contained in A (content was trimmed)
    for j in unmatched_b:
        if j in explained_b:
            continue
        norm_b = _normalize_for_containment(clauses_b[j].content)
        if len(norm_b) < 20:
            continue

        for i in unmatched_a:
            if i in explained_a:
                continue
            norm_a = _normalize_for_containment(clauses_a[i].content)

            if norm_b in norm_a:
                clause_a, clause_b = clauses_a[i], clauses_b[j]
                entries.append(
                    ChangeEntry(
                        clause_title=clause_a.heading or clause_b.heading or f"Clause at position {clause_a.doc_order + 1}",
                        change_type="removed",
                        change_summary="Content removed from this clause in Document B.",
                        risk_level="medium",
                        risk_impact="Clause content trimmed — verify no obligations or protections were lost.",
                        original_text=clause_a.content,
                    )
                )
                explained_a.add(i)
                explained_b.add(j)
                logger.info(f"Containment: B[{j}] found inside A[{i}] — reporting as removal")
                break

    remaining_a = [i for i in unmatched_a if i not in explained_a]
    remaining_b = [j for j in unmatched_b if j not in explained_b]

    return entries, remaining_a, remaining_b


# --- Stage 3: Per-Pair LLM Comparison ---


async def _compare_single_pair(clause_a: ClauseUnit, clause_b: ClauseUnit, llm_client) -> ClauseComparisonLLMResponse:
    """Send one clause pair to the LLM for detailed comparison."""

    prompt = Path(r"src\services\prompts\v1\clause_comparison_prompt.mustache").read_text(encoding="utf-8")
    context = {
        "clause_heading": clause_a.heading or clause_b.heading or "Unnamed Clause",
        "clause_a_text": clause_a.content,
        "clause_b_text": clause_b.content,
    }
    return await llm_client.generate(
        prompt=prompt,
        context=context,
        response_model=ClauseComparisonLLMResponse,
    )


def _build_change_entry(clause_a: ClauseUnit, clause_b: ClauseUnit, comparison: ClauseComparisonLLMResponse) -> ChangeEntry:
    """Convert an LLM comparison result into a ChangeEntry."""

    return ChangeEntry(
        clause_title=clause_a.heading or clause_b.heading or f"Clause at position {clause_a.doc_order + 1}",
        change_type=comparison.change_type,
        change_summary=comparison.change_summary,
        risk_level=comparison.risk_level,
        risk_impact=comparison.risk_impact,
        original_text=comparison.original_text,
        revised_text=comparison.revised_text,
    )


def _make_skipped_entry(clause_a: ClauseUnit, clause_b: ClauseUnit) -> ChangeEntry:
    """Placeholder entry when LLM call limit is exceeded."""

    return ChangeEntry(
        clause_title=clause_a.heading or clause_b.heading or f"Clause at position {clause_a.doc_order + 1}",
        change_type="modified",
        change_summary="Comparison skipped — LLM call limit exceeded or failed.",
        risk_level="medium",
        risk_impact="Unable to assess — manual review recommended.",
        original_text=clause_a.content,
        revised_text=clause_b.content,
    )


async def compare_matched_pairs(pairs: List[Tuple[int, int, float]], clauses_a: List[ClauseUnit], clauses_b: List[ClauseUnit], llm_client) -> Tuple[List[ChangeEntry], int, int]:
    """For each matched pair of clauses, determine if they are truly unchanged or if they have meaningful modifications, using the LLM for detailed analysis when needed. Returns a list of ChangeEntry objects along with counts of LLM calls made and skipped."""

    results: List[ChangeEntry] = []
    llm_calls = 0
    llm_skipped = 0

    for idx_a, idx_b, similarity in pairs:
        clause_a = clauses_a[idx_a]
        clause_b = clauses_b[idx_b]

        # Exact text match — truly unchanged
        if clause_a.content == clause_b.content:
            continue

        if llm_calls >= MAX_LLM_CALLS:
            results.append(_make_skipped_entry(clause_a, clause_b))
            llm_skipped += 1
            continue

        try:
            comparison = await _compare_single_pair(clause_a, clause_b, llm_client)
            results.append(_build_change_entry(clause_a, clause_b, comparison))
            llm_calls += 1
        except Exception as e:
            logger.error(f"LLM comparison failed for {clause_a.clause_id} vs {clause_b.clause_id}: {e}")
            results.append(_make_skipped_entry(clause_a, clause_b))
            llm_skipped += 1

    return results, llm_calls, llm_skipped


# --- Stage 4: Unmatched Entries & Grouping ---


def _build_unmatched_entries(unmatched_a: List[int], unmatched_b: List[int], clauses_a: List[ClauseUnit], clauses_b: List[ClauseUnit]) -> List[ChangeEntry]:
    """Create ChangeEntry objects for clauses only present in one document."""

    entries: List[ChangeEntry] = []

    for idx in unmatched_a:
        clause = clauses_a[idx]
        entries.append(
            ChangeEntry(
                clause_title=clause.heading or f"Clause at position {clause.doc_order + 1}",
                change_type="removed",
                change_summary="This clause has been removed from Document B.",
                risk_level="medium",
                risk_impact="Clause removed — verify no obligations or protections were lost.",
                original_text=clause.content,
            )
        )

    for idx in unmatched_b:
        clause = clauses_b[idx]
        entries.append(
            ChangeEntry(
                clause_title=clause.heading or f"Clause at position {clause.doc_order + 1}",
                change_type="added",
                change_summary="This clause has been added in Document B.",
                risk_level="medium",
                risk_impact="New clause introduced — review for additional obligations or rights.",
                revised_text=clause.content,
            )
        )

    return entries


def group_by_section(changes: List[ChangeEntry]) -> List[SectionGroup]:
    """Group change entries by their section heading."""

    section_map: Dict[str, List[ChangeEntry]] = {}
    for change in changes:
        section = change.clause_title or "General / Ungrouped"
        section_map.setdefault(section, []).append(change)

    return [SectionGroup(section_name=name, changes=entries) for name, entries in section_map.items()]


def _compute_summary(changes: List[ChangeEntry], llm_calls_made: int, llm_calls_skipped: int) -> CompareSummary:
    """Compute aggregate statistics from all detected changes."""

    added = sum(1 for c in changes if c.change_type == "added")
    removed = sum(1 for c in changes if c.change_type == "removed")
    modified = sum(1 for c in changes if c.change_type == "modified")
    reordered = sum(1 for c in changes if c.change_type == "reordered")
    high_risk = sum(1 for c in changes if c.risk_level == "high")

    if high_risk > 0:
        overall_risk = "high"
    elif any(c.risk_level == "medium" for c in changes):
        overall_risk = "medium"
    else:
        overall_risk = "low"

    return CompareSummary(
        total_changes=len(changes),
        added=added,
        removed=removed,
        modified=modified,
        reordered=reordered,
        overall_risk=overall_risk,
        high_risk_count=high_risk,
        llm_calls_made=llm_calls_made,
        llm_calls_skipped=llm_calls_skipped,
    )


def _zero_changes_summary() -> CompareSummary:
    """Summary for identical or same-document comparisons."""

    return CompareSummary(
        total_changes=0,
        added=0,
        removed=0,
        modified=0,
        reordered=0,
        overall_risk="low",
        high_risk_count=0,
        llm_calls_made=0,
        llm_calls_skipped=0,
    )


# --- Stage 5: Full Pipeline ---


async def extract_text(document: Document) -> str:
    """Extract full text from a docx Document object."""

    paragraphs = [para.text for para in document.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


async def run(session_id: str, document_a: Document, document_b: Document) -> CompareResponse:
    """Execute the full document comparison pipeline."""

    container = get_service_container()
    embedding_service = container.embedding_service
    llm_client = container.azure_openai_model

    parser = get_parser()

    # load cached data for this agent
    session_data = container.session_manager.get_session(session_id=session_id)
    cached_data: Dict[str, CompareResponse] = session_data.tool_results[AGENT_NAME] if AGENT_NAME in session_data.tool_results else []
    if cached_data:
        logger.info(f"Loaded cached data for session {session_id} and agent {AGENT_NAME}")
        return cached_data

    doc_a: ParseResult = await parser.parse_document(document_a)
    doc_b: ParseResult = await parser.parse_document(document_b)

    doc_text_a = await extract_text(document_a)
    doc_text_b = await extract_text(document_b)

    # Guard: same document
    if hash(doc_text_a) == hash(doc_text_b):
        return CompareResponse(
            success=True,
            message="Both document IDs are the same. Provide two different documents to compare.",
            summary=_zero_changes_summary(),
            sections=[],
        )

    # Stage 1: Extract clauses
    logger.info("Extracting clauses for the doccuments...")
    clauses_a = extract_clauses(doc_a)
    clauses_b = extract_clauses(doc_b)

    # Edge case: empty documents
    if not clauses_a and not clauses_b:
        return CompareResponse(success=True, summary=_zero_changes_summary(), sections=[])

    if not clauses_a:
        entries = _build_unmatched_entries([], list(range(len(clauses_b))), [], clauses_b)
        return CompareResponse(
            success=True,
            summary=_compute_summary(entries, 0, 0),
            sections=group_by_section(entries),
        )

    if not clauses_b:
        entries = _build_unmatched_entries(list(range(len(clauses_a))), [], clauses_a, [])
        return CompareResponse(
            success=True,
            summary=_compute_summary(entries, 0, 0),
            sections=group_by_section(entries),
        )

    # Stage 2: Match clauses
    logger.info(f"Matching {len(clauses_a)} clauses (A) against {len(clauses_b)} clauses (B)")
    match_result = await match_clauses(clauses_a, clauses_b, embedding_service)
    logger.info(f"Matched: {len(match_result.matched_pairs)}, " f"Unmatched A: {len(match_result.unmatched_a)}, " f"Unmatched B: {len(match_result.unmatched_b)}")

    # Stage 3: Split/merge detection
    split_merge_entries, remaining_a, remaining_b = _detect_splits_and_merges(match_result, clauses_a, clauses_b)

    # Stage 3.5: Containment reconciliation
    containment_entries, remaining_a, remaining_b = _reconcile_containment(remaining_a, remaining_b, clauses_a, clauses_b)

    # Stage 4: LLM comparison for content differences
    llm_changes, llm_calls_made, llm_calls_skipped = await compare_matched_pairs(
        match_result.matched_pairs,
        clauses_a,
        clauses_b,
        llm_client,
    )

    # Stage 5: Build unmatched entries
    unmatched_entries = _build_unmatched_entries(remaining_a, remaining_b, clauses_a, clauses_b)

    # Stage 6: Combine, group, summarize
    all_changes = llm_changes + split_merge_entries + containment_entries + unmatched_entries
    sections = group_by_section(all_changes)
    summary = _compute_summary(all_changes, llm_calls_made, llm_calls_skipped)

    logger.info(
        f"Compare complete: {summary.total_changes} changes "
        f"({summary.added}A/{summary.removed}R/{summary.modified}M/{summary.reordered}O), "
        f"LLM calls: {llm_calls_made} made, {llm_calls_skipped} skipped"
    )

    message = "Both documents are identical. No differences found." if summary.total_changes == 0 else None

    # Store data in cache for this session and agent
    session_data.tool_results[AGENT_NAME] = {
        "success": True,
        "message": message,
        "summary": summary,
        "sections": sections,
    }

    return CompareResponse(
        success=True,
        message=message,
        summary=summary,
        sections=sections,
    )
