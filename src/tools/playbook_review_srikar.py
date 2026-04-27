import asyncio
import hashlib
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from cachetools import LRUCache

from src.config.logging import get_logger
from src.dependencies import get_service_container
from src.schemas.playbook_review_srikar import (
    MissingClausesLLMResponse,
    PlayBookReviewFinalResponse,
    PlayBookReviewLLMResponse,
    PlayBookReviewResponse,
    RuleCheckRequest,
    RuleInfo,
    RuleResult,
    TextInfo,
)

logger = get_logger(__name__)

# ==============================
# Configuration
# ==============================

AGENT_NAME = "playbook_review_srikar_agent"

# FIX 1: Lowered threshold from 0.5 → 0.25
# Previous value was silently dropping relevant paragraphs before the LLM ever saw them.
SIMILARITY_THRESHOLD = 0.25

# FIX 2: Increased TOP_K from 5 → 15
# With 33 paragraphs, TOP_K=5 was cutting off multi-paragraph clauses
# (e.g. Compensation spans P0004, P0005, P0025, P0026, P0027 — all 5 could be missed).
TOP_K = 15

# FIX 3: Small-doc threshold — if the document has fewer paragraphs than this,
# skip embedding filtering entirely and pass all paragraphs to the LLM.
# This prevents the retrieval layer from gatekeeping on small contracts.
SMALL_DOC_THRESHOLD = 50

EMBEDDING_CACHE_SIZE = 10_000

# Title matching thresholds (clause-title-based selection layer).
FUZZY_TITLE_THRESHOLD = 0.70
TITLE_EMBEDDING_FALLBACK_THRESHOLD = 0.60

# Per-rule LLM evaluation timeout (seconds). A single hung rule must not
# block the whole review — asyncio.gather collects errors per coroutine.
RULE_LLM_TIMEOUT_SECONDS = 60.0
MISSING_CLAUSES_LLM_TIMEOUT_SECONDS = 90.0

SIMILARITY_PROMPT = Path(r"src\services\prompts\v1\ai_review_prompt_v2_srikar.mustache").read_text(encoding="utf-8")

MISSING_CLAUSES_PROMPT = Path(r"src\services\prompts\v1\missing_clauses_srikar.mustache").read_text(encoding="utf-8")

# ==============================
# Embedding Cache (LRU + Safe)
# ==============================

_embedding_cache: LRUCache = LRUCache(maxsize=EMBEDDING_CACHE_SIZE)
_embedding_locks: Dict[str, asyncio.Lock] = {}
_embedding_locks_guard = asyncio.Lock()


def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


async def _get_lock_for_key(key: str) -> asyncio.Lock:
    async with _embedding_locks_guard:
        if key not in _embedding_locks:
            _embedding_locks[key] = asyncio.Lock()
        return _embedding_locks[key]


async def get_embedding(embedding_model, text: str) -> np.ndarray:
    key = _hash(text)

    if key in _embedding_cache:
        return _embedding_cache[key]

    lock = await _get_lock_for_key(key)

    async with lock:
        if key not in _embedding_cache:
            embedding = await embedding_model.generate_embeddings(text)
            _embedding_cache[key] = embedding
        return _embedding_cache[key]


# ==============================
# Text Processing
# ==============================

TOKEN_PATTERN = re.compile(r"\b[a-zA-Z]{3,}\b")


def tokenize(text: str) -> set:
    return set(TOKEN_PATTERN.findall(text.lower()))


def keyword_score(rule_text: str, para_text: str) -> float:
    rule_tokens = tokenize(rule_text)
    if not rule_tokens:
        return 0.0
    para_tokens = tokenize(para_text)
    return len(rule_tokens & para_tokens) / len(rule_tokens)


def hybrid_score(cosine: float, keyword: float) -> float:
    return 0.75 * cosine + 0.25 * keyword


def normalize_embeddings(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    return matrix / norms


# ==============================
# Missing Clauses
# ==============================


def _build_reviewed_rules_summary(reviewed: Dict[str, PlayBookReviewResponse]) -> str:
    lines = []
    for title, review in reviewed.items():
        para_ids = ", ".join(review.content.para_identifiers) or "none"
        lines.append(f"RULE: {title} | STATUS: {review.content.status} | PARAS: {para_ids}")
    return "\n".join(lines) if lines else "None"


async def get_missing_clauses(
    llm_model,
    full_text: str,
    reviewed_rules_summary: str,
) -> MissingClausesLLMResponse:

    try:
        response = await asyncio.wait_for(
            llm_model.generate(
                prompt=MISSING_CLAUSES_PROMPT,
                context={
                    "data": full_text,
                    "reviewed_rules_summary": reviewed_rules_summary,
                },
                response_model=MissingClausesLLMResponse,
            ),
            timeout=MISSING_CLAUSES_LLM_TIMEOUT_SECONDS,
        )
        logger.info("Missing clauses identified: %d", len(response.missing_clauses))
        return response

    except asyncio.TimeoutError:
        logger.error(
            "Missing clauses evaluation timed out after %.1fs.",
            MISSING_CLAUSES_LLM_TIMEOUT_SECONDS,
        )
        return MissingClausesLLMResponse(
            missing_clauses=[],
            total_missing=0,
            summary=(
                f"Missing clauses evaluation timed out after "
                f"{MISSING_CLAUSES_LLM_TIMEOUT_SECONDS:.0f}s."
            ),
        )
    except Exception as exc:
        logger.exception("Missing clauses evaluation failed.")
        return MissingClausesLLMResponse(
            missing_clauses=[],
            total_missing=0,
            summary=f"LLM error: {exc}",
        )


# ==============================
# Clause Title Detection & Matching
# ==============================

_HEADING_FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "this", "that", "these", "those", "and",
    "or", "if", "but", "for", "it", "we", "you", "he", "she", "they", "all",
    "any", "each", "please", "however", "furthermore", "moreover", "also",
    "in", "on", "at", "to", "of", "by", "with", "from",
}

_HEADING_LEADING_NUMBERING = re.compile(
    r"^(\d+(?:\.\d+)*\.?\s+|section\s+\w+\s*[-:.]?\s*|article\s+\w+\s*[-:.]?\s*)",
    flags=re.IGNORECASE,
)


def _extract_heading_from_text(content: str) -> Optional[str]:
    """Derive a clause heading from a paragraph's first line.

    Returns just the heading title (the body that follows is dropped). Examples:
      "1.2 Termination. Either party..."  -> "1.2 Termination"
      "Audit Rights. Epit shall..."        -> "Audit Rights"
      "Section 5 - Liability. Each..."     -> "Section 5 - Liability"
      "ARTICLE III"                        -> "ARTICLE III"
    Returns None when the paragraph is body text rather than a clause start.
    """
    first_line = content.strip().split("\n")[0].strip()
    if not first_line:
        return None

    # Numbered sections: "1.2 Termination. ..." -> "1.2 Termination"
    m = re.match(
        r"^(\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z][A-Za-z\s/&,'-]{0,80}?)(?=\.(?:\s|$))",
        first_line,
    )
    if m:
        return m.group(1).strip()

    # Section / Article markers with optional dash-separated title
    m = re.match(
        r"^((?:Section|SECTION|Article|ARTICLE)\s+[\w]+(?:\s*[-:]\s*[A-Za-z][A-Za-z\s/&,'-]{0,60})?)(?=\.(?:\s|$)|$)",
        first_line,
    )
    if m:
        return m.group(1).strip()

    # ALL CAPS standalone heading (allows hyphens and apostrophes)
    if re.match(r"^[A-Z][A-Z\s'\-]{4,}$", first_line):
        return first_line

    # Title-case prefix before ". " — "Audit Rights. Body text..."
    m = re.match(r"^([A-Z][A-Za-z\s/&,'-]{2,60})\.\s", first_line)
    if m:
        candidate = m.group(1).strip()
        words = candidate.lower().split()
        content_words = [w for w in words if w not in _HEADING_FUNCTION_WORDS]
        if len(content_words) >= 1 and len(words) <= 8:
            return candidate

    return None


# Inline numbered headings appearing mid-paragraph. Designed to be conservative:
# requires a numbering prefix (1., 1.2, 1.2.3, etc.), Section, or ARTICLE marker
# followed by a Title-Cased phrase ending in ". " — matches the standard contract
# convention without false-firing on prose that happens to start with a number.
_INLINE_NUMBERED_HEADING = re.compile(
    r"(?:^|(?<=[.!?]\s)|(?<=\n))"
    r"(?P<heading>"
    r"(?:"
    r"\d+(?:\.\d+)*\.?\s+|"
    r"Section\s+\w+\s*[-:]?\s*|"
    r"SECTION\s+\w+\s*[-:]?\s*|"
    r"Article\s+\w+\s*[-:]?\s*|"
    r"ARTICLE\s+\w+\s*[-:]?\s*"
    r")"
    r"[A-Z][A-Za-z][A-Za-z\s/&,'\-]{1,80}?"
    r")\.(?=\s)",
    flags=re.MULTILINE,
)

# Suffix applied to a paragraph identifier when one input chunk is split into
# multiple synthetic segments. Stripped from the LLM's para_identifiers before
# the final response goes back to the frontend, so external IDs stay clean.
_SPLIT_ID_SEPARATOR = "#"


def _strip_split_suffix(para_id: str) -> str:
    """Remove the synthetic split suffix ("#N") from a paragraph id."""
    return para_id.split(_SPLIT_ID_SEPARATOR, 1)[0] if para_id else para_id


def _split_paragraph_at_inline_headings(para: TextInfo) -> List[TextInfo]:
    """Break one coarse paragraph into per-clause segments at inline headings.

    Real ingested NDAs frequently arrive as a small number of large chunks where
    multiple numbered sections live inside one paragraph (e.g. "6. Term...
    7. Remedies... 8. Assignment..."). The heading extractor only inspects the
    very first line of each paragraph, so without splitting, all internal clause
    headings are invisible to the title-matching layer.

    On any unexpected failure the original paragraph is returned unchanged so
    the pipeline cannot be broken by malformed input.
    """
    text = (para.text or "").strip()
    if not text:
        return [para]

    matches = list(_INLINE_NUMBERED_HEADING.finditer(text))
    if len(matches) <= 1:
        # Zero or one inline heading -> nothing to gain by splitting.
        return [para]

    boundaries: List[int] = []
    if matches[0].start() > 0:
        boundaries.append(0)  # preserve any preamble before the first heading
    boundaries.extend(m.start() for m in matches)
    boundaries.append(len(text))

    segments: List[TextInfo] = []
    seg_index = 0
    for i in range(len(boundaries) - 1):
        seg = text[boundaries[i]:boundaries[i + 1]].strip()
        if not seg:
            continue
        seg_index += 1
        synthetic_id = f"{para.paraindetifier}{_SPLIT_ID_SEPARATOR}{seg_index}"
        segments.append(TextInfo(paraindetifier=synthetic_id, text=seg))

    return segments if segments else [para]


def _expand_paragraphs_with_inline_headings(
    paragraphs: List[TextInfo],
) -> List[TextInfo]:
    """Apply the inline-heading splitter to every paragraph and flatten.

    Defensive: any per-paragraph failure logs and keeps the original.
    """
    expanded: List[TextInfo] = []
    for para in paragraphs:
        try:
            expanded.extend(_split_paragraph_at_inline_headings(para))
        except Exception:  # noqa: BLE001 — never let a malformed para break review
            logger.exception(
                "Inline-heading splitter failed for paragraph '%s'; "
                "using original paragraph as-is.",
                getattr(para, "paraindetifier", "?"),
            )
            expanded.append(para)
    return expanded


def _group_paragraphs_into_clauses(
    paragraphs: List[TextInfo],
) -> List[Tuple[Optional[str], List[TextInfo]]]:
    """Group paragraphs into clauses using extracted headings as boundaries.

    Each new heading-bearing paragraph starts a new clause; following paragraphs
    without their own heading belong to the most recent clause. Paragraphs that
    appear before any heading are grouped under heading=None (preamble).
    """
    if not paragraphs:
        return []

    clauses: List[Tuple[Optional[str], List[TextInfo]]] = []
    current_heading: Optional[str] = None
    current_paras: List[TextInfo] = []

    for para in paragraphs:
        heading = _extract_heading_from_text(para.text)
        if heading:
            if current_paras:
                clauses.append((current_heading, current_paras))
            current_heading = heading
            current_paras = [para]
        else:
            current_paras.append(para)

    if current_paras:
        clauses.append((current_heading, current_paras))

    return clauses


def _normalize_heading_for_match(heading: str) -> str:
    """Lowercase + strip leading numbering ("1.2", "Section 5 -", etc.)."""
    cleaned = _HEADING_LEADING_NUMBERING.sub("", heading.strip(), count=1)
    return cleaned.strip().lower()


_TOKENIZE_PATTERN = re.compile(r"\b[a-z]{3,}\b")


def _content_tokens(text: str) -> set:
    """Lowercase tokens of length >= 3, with stop-words removed."""
    return {t for t in _TOKENIZE_PATTERN.findall(text.lower()) if t not in _HEADING_FUNCTION_WORDS}


def _fuzzy_title_ratio(rule_title: str, clause_heading: str) -> float:
    """Similarity in [0, 1] between a rule title and a clause heading.

    Combines token-overlap (every content word of the rule appears in the
    heading) with character-level SequenceMatcher ratio. Returns the max of
    the two so single-token rules and multi-word rules both behave well.
    Avoids substring shortcuts that false-positive on body text.
    """
    if not rule_title or not clause_heading:
        return 0.0

    rule_norm = rule_title.strip().lower()
    head_norm = _normalize_heading_for_match(clause_heading)
    if not rule_norm or not head_norm:
        return 0.0

    rule_tokens = _content_tokens(rule_norm)
    head_tokens = _content_tokens(head_norm)
    token_overlap = (
        len(rule_tokens & head_tokens) / len(rule_tokens) if rule_tokens else 0.0
    )

    seq_ratio = SequenceMatcher(None, rule_norm, head_norm).ratio()

    return max(token_overlap, seq_ratio)


def _best_clause_by_fuzzy_title(
    rule_title: str,
    clauses: List[Tuple[Optional[str], List[TextInfo]]],
) -> Tuple[int, float]:
    """Return (clause_index, ratio) of the best fuzzy match. (-1, 0.0) if none."""
    best_idx = -1
    best_ratio = 0.0
    for idx, (heading, _paras) in enumerate(clauses):
        if not heading:
            continue
        ratio = _fuzzy_title_ratio(rule_title, heading)
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = idx
    return best_idx, best_ratio


async def _best_clause_by_title_embedding(
    rule_title: str,
    clauses: List[Tuple[Optional[str], List[TextInfo]]],
    embedding_model,
) -> Tuple[int, float]:
    """Cosine similarity between rule_title and each clause heading.

    Returns (clause_index, score). (-1, 0.0) if no clause has a heading.
    """
    headed: List[Tuple[int, str]] = [
        (i, h) for i, (h, _) in enumerate(clauses) if h
    ]
    if not headed:
        return -1, 0.0

    rule_emb = await get_embedding(embedding_model, rule_title)
    rule_norm = rule_emb / (np.linalg.norm(rule_emb) + 1e-10)

    head_embs = await asyncio.gather(
        *[get_embedding(embedding_model, h) for _, h in headed]
    )
    head_matrix = np.array(head_embs)
    head_norm = head_matrix / (np.linalg.norm(head_matrix, axis=1, keepdims=True) + 1e-10)
    scores = head_norm @ rule_norm

    best_pos = int(np.argmax(scores))
    return headed[best_pos][0], float(scores[best_pos])


# ==============================
# Rule Processing
# ==============================


def _select_paragraphs_by_content(
    rule_text: str,
    rule_norm: np.ndarray,
    normalized_para_embeddings: np.ndarray,
    request: RuleCheckRequest,
) -> List[Tuple[TextInfo, float]]:
    """Content-based paragraph selection (cosine + keyword hybrid).

    Used as a safety-net when title-based matching cannot anchor a rule
    (e.g. contracts that arrive with no detectable clause headings).
    """
    num_paras = len(request.textinformation)

    if num_paras <= SMALL_DOC_THRESHOLD:
        cosine_scores = normalized_para_embeddings @ rule_norm
        result = []
        for idx, para in enumerate(request.textinformation):
            kw = keyword_score(rule_text, para.text)
            score = hybrid_score(float(cosine_scores[idx]), kw)
            result.append((para, score))
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    cosine_scores = normalized_para_embeddings @ rule_norm
    top_indices = np.argsort(cosine_scores)[::-1][:TOP_K]

    matched: List[Tuple[TextInfo, float]] = []
    for idx in top_indices:
        para = request.textinformation[idx]
        kw = keyword_score(rule_text, para.text)
        score = hybrid_score(float(cosine_scores[idx]), kw)
        if score >= SIMILARITY_THRESHOLD:
            matched.append((para, score))

    if not matched:
        logger.warning(
            "Content fallback: no paragraphs cleared %.2f for rule '%s'. Using top-5 by cosine.",
            SIMILARITY_THRESHOLD,
            rule_text.split("\n", 1)[0][:80],
        )
        fallback_indices = np.argsort(cosine_scores)[::-1][:5]
        for idx in fallback_indices:
            para = request.textinformation[idx]
            kw = keyword_score(rule_text, para.text)
            score = hybrid_score(float(cosine_scores[idx]), kw)
            matched.append((para, score))

    return matched


async def _select_paragraphs_for_rule(
    rule: RuleInfo,
    rule_text: str,
    rule_norm: np.ndarray,
    clauses: List[Tuple[Optional[str], List[TextInfo]]],
    normalized_para_embeddings: np.ndarray,
    request: RuleCheckRequest,
    embedding_model,
) -> Tuple[List[Tuple[TextInfo, float]], List[str], str]:
    """Pick paragraphs for a rule using clause-title matching first.

    Strategy:
      1. Fuzzy match rule.title against extracted clause headings.
      2. If no fuzzy hit, try cosine similarity between rule.title and each
         clause heading (titles only — much sharper than full-paragraph cosine).
      3. If neither title strategy hits, fall back to content-based hybrid
         scoring over the whole document so headingless contracts still work.

    Returns (paragraphs_with_scores, matched_clause_titles, match_strategy).
    """
    # 1) Fuzzy title match
    fuzzy_idx, fuzzy_ratio = _best_clause_by_fuzzy_title(rule.title, clauses)
    if fuzzy_ratio >= FUZZY_TITLE_THRESHOLD and fuzzy_idx >= 0:
        heading, paras = clauses[fuzzy_idx]
        logger.info(
            "Title match (fuzzy %.2f): rule '%s' -> clause '%s'",
            fuzzy_ratio, rule.title, heading,
        )
        return (
            [(p, fuzzy_ratio) for p in paras],
            [heading] if heading else [],
            "title_fuzzy",
        )

    # 2) Title-only embedding fallback
    emb_idx, emb_score = await _best_clause_by_title_embedding(
        rule.title, clauses, embedding_model,
    )
    if emb_score >= TITLE_EMBEDDING_FALLBACK_THRESHOLD and emb_idx >= 0:
        heading, paras = clauses[emb_idx]
        logger.info(
            "Title match (embedding %.2f): rule '%s' -> clause '%s'",
            emb_score, rule.title, heading,
        )
        return (
            [(p, emb_score) for p in paras],
            [heading] if heading else [],
            "title_embedding",
        )

    # 3) Content fallback (headingless docs or genuinely off-spec rules)
    logger.info(
        "Title match failed (best fuzzy=%.2f, best emb=%.2f) for rule '%s'. "
        "Falling back to content scoring.",
        fuzzy_ratio, emb_score, rule.title,
    )
    content_matches = _select_paragraphs_by_content(
        rule_text, rule_norm, normalized_para_embeddings, request,
    )
    return content_matches, [], "content_fallback"


async def _process_rule(
    rule: RuleInfo,
    clauses: List[Tuple[Optional[str], List[TextInfo]]],
    normalized_para_embeddings: np.ndarray,
    request: RuleCheckRequest,
    embedding_model,
    llm_model,
) -> Tuple[str, PlayBookReviewResponse]:

    rule_text = f"TITLE: {rule.title}\n" f"INSTRUCTION: {rule.instruction}\n" f"DESCRIPTION: {rule.description}\n"

    rule_emb = await get_embedding(embedding_model, rule_text)
    rule_norm = rule_emb / (np.linalg.norm(rule_emb) + 1e-10)

    matched, matched_clause_titles, match_strategy = await _select_paragraphs_for_rule(
        rule=rule,
        rule_text=rule_text,
        rule_norm=rule_norm,
        clauses=clauses,
        normalized_para_embeddings=normalized_para_embeddings,
        request=request,
        embedding_model=embedding_model,
    )

    result = RuleResult(
        title=rule.title,
        instruction=rule.instruction,
        description=rule.description,
        paragraphidentifier=",".join(p.paraindetifier for p, _ in matched),
        paragraphcontext="\n\n".join(f"PARA_ID: {p.paraindetifier}\nTEXT: {p.text.strip()}" for p, _ in matched),
        similarity_scores=[score for _, score in matched],
    )

    try:
        llm_response: PlayBookReviewLLMResponse = await asyncio.wait_for(
            llm_model.generate(
                prompt=SIMILARITY_PROMPT,
                context={
                    "rule_title": result.title,
                    "rule_instruction": result.instruction,
                    "rule_description": result.description,
                    "paragraphs": result.paragraphcontext,
                },
                response_model=PlayBookReviewLLMResponse,
            ),
            timeout=RULE_LLM_TIMEOUT_SECONDS,
        )

    except asyncio.TimeoutError:
        logger.error(
            "LLM rule evaluation timed out after %.1fs for rule '%s'.",
            RULE_LLM_TIMEOUT_SECONDS, rule.title,
        )
        llm_response = PlayBookReviewLLMResponse(
            para_identifiers=[],
            status="Error",
            reason=f"Rule evaluation timed out after {RULE_LLM_TIMEOUT_SECONDS:.0f}s.",
            suggestion="",
            suggested_fix="",
        )
    except Exception as exc:
        logger.exception("LLM rule evaluation failed.")
        llm_response = PlayBookReviewLLMResponse(
            para_identifiers=[],
            status="Error",
            reason=str(exc),
            suggestion="",
            suggested_fix="",
        )

    # If we split a coarse paragraph into synthetic segments, strip the suffix
    # from every paragraph id in the LLM response so the frontend only ever
    # sees the original chunk identifiers it expects.
    cleaned_para_ids: List[str] = []
    for pid in llm_response.para_identifiers:
        cleaned = _strip_split_suffix(pid)
        if cleaned and cleaned not in cleaned_para_ids:
            cleaned_para_ids.append(cleaned)
    llm_response.para_identifiers = cleaned_para_ids

    # If the title-matching layer couldn't anchor a heading and the LLM
    # supplied a concise clause name, use that as the matched clause label.
    # This is the last-resort fallback — the LLM has actually read the
    # paragraphs and can name the topic even when no heading is detectable.
    if not matched_clause_titles:
        llm_label = (getattr(llm_response, "matched_clause_name", "") or "").strip()
        if llm_label:
            matched_clause_titles = [llm_label]
            if match_strategy == "content_fallback":
                match_strategy = "llm_label"

    return rule.title, PlayBookReviewResponse(
        rule_title=rule.title,
        rule_instruction=rule.instruction,
        rule_description=rule.description,
        matched_clause_titles=matched_clause_titles,
        match_strategy=match_strategy,
        content=llm_response,
    )


# ==============================
# Main Entry
# ==============================


def _resolve_paragraphs(
    request: RuleCheckRequest,
    session_data,
) -> List[TextInfo]:
    """Return the paragraphs to review.

    Preference order:
      1. request.textinformation (frontend-supplied paragraphs)
      2. session_data.chunk_store (paragraphs already ingested into the session)

    When falling back to the session, paragraph IDs are renumbered to the
    P0001/P0002/... pattern so downstream consumers anchoring on those IDs
    behave the same regardless of source.
    """
    if request.textinformation:
        return list(request.textinformation)

    chunk_store = getattr(session_data, "chunk_store", None) or {}
    if not chunk_store:
        return []

    ordered = sorted(
        chunk_store.items(),
        key=lambda kv: getattr(kv[1], "chunk_index", kv[0]),
    )

    paragraphs: List[TextInfo] = []
    position = 1
    for _idx, chunk in ordered:
        content = (getattr(chunk, "content", None) or "").strip()
        if not content:
            continue

        # If the parser tagged this chunk with a section heading and that
        # heading isn't already the first line of the content, surface it as
        # a leading line so the title-matching layer can pick it up.
        metadata = getattr(chunk, "metadata", None)
        section_heading = None
        if isinstance(metadata, dict):
            raw = metadata.get("section_heading")
            if isinstance(raw, str) and raw.strip():
                section_heading = raw.strip()

        if section_heading:
            first_line = content.split("\n", 1)[0].strip()
            if section_heading.lower() not in first_line.lower():
                content = f"{section_heading}. {content}"

        paragraphs.append(
            TextInfo(paraindetifier=f"P{position:04d}", text=content)
        )
        position += 1

    return paragraphs


async def review_document(
    session_id: str,
    request: RuleCheckRequest,
    force_update_rules: Optional[List[str]] = None,
) -> PlayBookReviewFinalResponse:

    force_update_rules = force_update_rules or []

    container = get_service_container()
    embedding_model = container.embedding_service
    llm_model = container.azure_openai_model

    session_data = container.session_manager.get_session(session_id)
    if not session_data:
        return PlayBookReviewFinalResponse(
            rules_review=[],
            missing_clauses=None,
        )

    # Resolve paragraphs: prefer request.textinformation; otherwise pull from
    # the session's chunk_store (populated by the ingestion endpoint).
    paragraphs = _resolve_paragraphs(request, session_data)
    if not paragraphs:
        logger.warning(
            "No paragraphs to review for session '%s' "
            "(empty textinformation and empty chunk_store).",
            session_id,
        )
        return PlayBookReviewFinalResponse(
            rules_review=[],
            missing_clauses=None,
        )

    # Coarse parsers often pack multiple clauses into one chunk. Split any
    # paragraph that contains inline numbered headings so each clause becomes
    # its own segment — title-based matching can then fire on every section.
    paragraphs = _expand_paragraphs_with_inline_headings(paragraphs)
    logger.info(
        "Paragraphs after inline-heading expansion: %d",
        len(paragraphs),
    )

    # Mutate the request so the rest of the pipeline (which reads
    # request.textinformation) works uniformly regardless of source.
    request.textinformation = paragraphs

    agent_cache = session_data.tool_results.get(AGENT_NAME, {})
    cached_reviews: Dict[str, PlayBookReviewResponse] = {r.rule_title: r for r in agent_cache.get("rules_review", [])}

    # Determine stale rules
    rules_to_update = []
    for rule in request.rulesinformation:
        cached = cached_reviews.get(rule.title)
        if not cached or rule.title in force_update_rules or cached.rule_description != rule.description or cached.rule_instruction != rule.instruction:
            rules_to_update.append(rule)

    # If nothing changed, return cache
    if not rules_to_update:
        return PlayBookReviewFinalResponse(
            rules_review=list(cached_reviews.values()),
            missing_clauses=agent_cache.get("missing_clauses"),
        )

    # Precompute normalized paragraph embeddings once (used by content fallback).
    para_embeddings = np.array(await asyncio.gather(*[get_embedding(embedding_model, p.text) for p in request.textinformation]))
    normalized_para_embeddings = normalize_embeddings(para_embeddings)

    # Group paragraphs into clauses by detected headings — used for the
    # title-first matching layer.
    clauses = _group_paragraphs_into_clauses(request.textinformation)
    headed_count = sum(1 for h, _ in clauses if h)
    logger.info(
        "Document grouped into %d clause(s); %d have detected headings.",
        len(clauses), headed_count,
    )

    # Process rules concurrently
    updates = await asyncio.gather(
        *[
            _process_rule(
                rule,
                clauses,
                normalized_para_embeddings,
                request,
                embedding_model,
                llm_model,
            )
            for rule in rules_to_update
        ]
    )

    cached_reviews.update(dict(updates))

    # Recompute missing clauses if doc OR rules changed
    full_text = "\n\n".join(f"PARA_ID: {p.paraindetifier}\nTEXT: {p.text}" for p in request.textinformation)

    doc_hash = _hash(full_text)
    rules_hash = _hash("".join(r.title + r.description for r in request.rulesinformation))

    cached_doc_hash = agent_cache.get("doc_hash")
    cached_rules_hash = agent_cache.get("rules_hash")

    if doc_hash != cached_doc_hash or rules_hash != cached_rules_hash:
        logger.info("Re-evaluating missing clauses.")
        reviewed_summary = _build_reviewed_rules_summary(cached_reviews)
        missing_clauses = await get_missing_clauses(llm_model, full_text, reviewed_summary)
    else:
        missing_clauses = agent_cache.get("missing_clauses")

    session_data.tool_results[AGENT_NAME] = {
        "rules_review": list(cached_reviews.values()),
        "missing_clauses": missing_clauses,
        "doc_hash": doc_hash,
        "rules_hash": rules_hash,
    }

    return PlayBookReviewFinalResponse(
        rules_review=list(cached_reviews.values()),
        missing_clauses=missing_clauses,
    )


# import asyncio
# import hashlib
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple

# import numpy as np

# from src.config.logging import get_logger
# from src.dependencies import get_service_container
# from src.schemas.playbook_review import (
#     MissingClausesLLMResponse,
#     PlayBookReviewFinalResponse,
#     PlayBookReviewLLMResponse,
#     PlayBookReviewResponse,
#     RuleCheckRequest,
#     RuleInfo,
#     RuleResult,
#     TextInfo,
# )

# logger = get_logger(__name__)

# similarity_prompt_template = Path(r"src\services\prompts\v1\ai_review_prompt_v2.mustache").read_text(encoding="utf-8")

# AGENT_NAME = "playbook_review_agent"
# SIMILARITY_THRESHOLD = 0.35

# # Protect shared cache against async race conditions
# embedding_cache: Dict[str, np.ndarray] = {}
# _embedding_locks: Dict[str, asyncio.Lock] = {}


# def _hash(text: str) -> str:
#     return hashlib.md5(text.encode("utf-8")).hexdigest()


# def _doc_hash(text_items: List[TextInfo]) -> str:
#     """Stable hash of the full document paragraph list."""
#     combined = "".join(item.text for item in text_items)
#     return _hash(combined)


# def _build_reviewed_rules_summary(cached_rules_review: Dict[str, PlayBookReviewResponse]) -> str:
#     """Build a summary string of the reviewed rules to pass to the missing clauses LLM, so it knows which topics have already been reviewed and should not be re-flagged as missing."""

#     lines: List[str] = []
#     for title, review in cached_rules_review.items():
#         para_ids = ", ".join(review.content.para_identifiers) or "none"
#         lines.append(f"RULE: {title} | STATUS: {review.content.status} | PARAS: {para_ids}")
#     return "\n".join(lines) if lines else "None"


# async def get_missing_clauses(data: str, reviewed_rules_summary: str) -> MissingClausesLLMResponse:
#     """Evaluate the document for genuinely absent clauses."""

#     service_container = get_service_container()
#     llm_model = service_container.azure_openai_model

#     prompt = Path(r"src\services\prompts\v1\missing_clauses.mustache").read_text(encoding="utf-8")

#     try:
#         response: MissingClausesLLMResponse = await llm_model.generate(
#             prompt=prompt,
#             context={
#                 "data": data,
#                 "reviewed_rules_summary": reviewed_rules_summary,
#             },
#             response_model=MissingClausesLLMResponse,
#         )
#         logger.info(f"Identified {len(response.missing_clauses)} missing clauses.")
#         return response
#     except Exception as exc:
#         logger.error(f"Missing clauses LLM call failed: {exc}")
#         return MissingClausesLLMResponse(
#             missing_clauses=[],
#             total_missing=0,
#             summary="Could not evaluate missing clauses due to an LLM error.",
#         )

#     # {
#     #   "text": "Confidential Information does not include any information, however designated, that:  (i) is or subsequently becomes publicly available without Receiving Party’s breach of any obligation owed Disclosing Party; (ii) was known by Receiving Party prior to Disclosing Party’s disclosure of such information to Receiving Party pursuant to the terms of this Agreement; (iii) became known to Receiving Party from a source other than Disclosing Party other than by the breach of an obligation of confidentiality owed to Disclosing Party; or (iv) is independently developed by Receiving Party.  \r",
#     #   "paraindetifier": "P0005"
#     # },


# async def get_embedding(embedding_model, text: str) -> np.ndarray:
#     """Task-safe embedding lookup with per-key lock to prevent duplicate fetches."""

#     key = _hash(text)
#     if key in embedding_cache:
#         return embedding_cache[key]
#     if key not in _embedding_locks:
#         _embedding_locks[key] = asyncio.Lock()
#     async with _embedding_locks[key]:
#         if key not in embedding_cache:
#             embedding_cache[key] = await embedding_model.generate_embeddings(text)
#     return embedding_cache[key]


# def keyword_score(rule: str, para: str) -> float:
#     rule_tokens = set(rule.lower().split())
#     para_tokens = set(para.lower().split())
#     if not rule_tokens:
#         return 0.0
#     return len(rule_tokens & para_tokens) / len(rule_tokens)


# def hybrid_score(cosine: float, keyword: float) -> float:
#     return 0.75 * cosine + 0.25 * keyword


# async def get_matching_pairs_faiss(request: RuleCheckRequest) -> List[RuleResult]:
#     """Get the matching pairs for the given rules with FAISS."""

#     service_container = get_service_container()
#     faiss_db = service_container.faiss_store
#     embedding_model = service_container.embedding_service

#     faiss_db.reset_index()

#     para_map: List[TextInfo] = []
#     for item in request.textinformation:
#         emb = await get_embedding(embedding_model, item.text)
#         await faiss_db.index_embedding(emb, metadata=item.paraidentifier)
#         para_map.append(item)

#     results: List[RuleResult] = []
#     for rule in request.rulesinformation:
#         rule_text = f"TITLE: {rule.title}\n" f"INSTRUCTION: {rule.instruction}\n" f"DESCRIPTION: {rule.description}\n" f"TAGS: {', '.join(rule.tags)}"

#         rule_emb = await get_embedding(embedding_model, rule_text)
#         faiss_result = await faiss_db.search_index(rule_emb, top_k=5)

#         indices = faiss_result.get("indices", [])
#         scores = faiss_result.get("scores", [])

#         matched_pairs: List[Tuple[TextInfo, float]] = []
#         for idx, score in zip(indices, scores):
#             if idx == -1 or idx >= len(para_map):
#                 continue
#             para = para_map[idx]
#             kw = keyword_score(rule_text, para.text)
#             final_score = hybrid_score(float(score), kw)
#             if final_score >= SIMILARITY_THRESHOLD:
#                 matched_pairs.append((para, final_score))

#         if not matched_pairs:
#             results.append(
#                 RuleResult(
#                     title=rule.title,
#                     instruction=rule.instruction,
#                     description="No relevant contract paragraphs found.",
#                     paragraphidentifier="",
#                     paragraphcontext="",
#                     similarity_scores=[],
#                 )
#             )
#             continue

#         results.append(
#             RuleResult(
#                 title=rule.title,
#                 instruction=rule.instruction,
#                 description=rule.description,
#                 paragraphidentifier=",".join(p.paraidentifier for p, _ in matched_pairs),
#                 paragraphcontext="\n\n".join(f"PARA_ID: {p.paraidentifier}\nTEXT: {p.text.strip()}" for p, _ in matched_pairs),
#                 similarity_scores=[s for _, s in matched_pairs],
#             )
#         )

#     return results


# async def _process_single_rule(rule: RuleInfo, para_embeddings: np.ndarray, request: RuleCheckRequest, llm_model, embedding_model) -> Tuple[str, PlayBookReviewResponse]:
#     """Process a single rule: compute similarities, find matches, and call LLM for review. Returns the rule title and its corresponding PlayBookReviewResponse."""

#     rule_text = f"TITLE: {rule.title}\n" f"INSTRUCTION: {rule.instruction}\n" f"DESCRIPTION: {rule.description}\n"  # f"TAGS: {', '.join(rule.tags)}"

#     rule_emb = await get_embedding(embedding_model, rule_text)

#     norms_rule = rule_emb / (np.linalg.norm(rule_emb) + 1e-10)
#     norms_para = para_embeddings / (np.linalg.norm(para_embeddings, axis=1, keepdims=True) + 1e-10)
#     cosine_scores = norms_para @ norms_rule

#     top_idx = np.argsort(cosine_scores)[::-1][:5]

#     matched: List[Tuple[TextInfo, float]] = []
#     for idx in top_idx:
#         para = request.textinformation[idx]
#         kw = keyword_score(rule_text, para.text)
#         score = hybrid_score(float(cosine_scores[idx]), kw)
#         if score >= SIMILARITY_THRESHOLD:
#             matched.append((para, score))

#     if not matched:
#         result = RuleResult(
#             title=rule.title,
#             instruction=rule.instruction,
#             description="No relevant contract paragraphs found.",
#             paragraphidentifier="",
#             paragraphcontext="",
#             similarity_scores=[],
#         )
#     else:
#         result = RuleResult(
#             title=rule.title,
#             instruction=rule.instruction,
#             description=rule.description,
#             paragraphidentifier=",".join(p.paraindetifier for p, _ in matched),
#             paragraphcontext="\n\n".join(f"PARA_ID: {p.paraindetifier}\nTEXT: {p.text.strip()}" for p, _ in matched),
#             similarity_scores=[s for _, s in matched],
#         )

#     context = {
#         "rule_title": result.title,
#         "rule_instruction": result.instruction,
#         "rule_description": result.description,
#         "paragraphs": result.paragraphcontext,
#     }

#     try:
#         llm_result: PlayBookReviewLLMResponse = await llm_model.generate(
#             prompt=similarity_prompt_template,
#             context=context,
#             response_model=PlayBookReviewLLMResponse,
#         )
#     except Exception as exc:
#         logger.error(f"LLM review failed for rule '{rule.title}': {exc}")
#         llm_result = PlayBookReviewLLMResponse(
#             para_identifiers=[p.paraidentifier for p, _ in matched],
#             status="Error",
#             reason=f"LLM call failed: {exc}",
#             suggestion="",
#             suggested_fix="",
#         )

#     return rule.title, PlayBookReviewResponse(
#         rule_title=result.title,
#         rule_instruction=result.instruction,
#         rule_description=result.description,
#         content=llm_result,
#     )


# async def review_document(session_id: str, request: RuleCheckRequest, force_update_rules: Optional[List[str]] = None) -> PlayBookReviewFinalResponse:
#     """Run playbook review for the given document and rules, with optional force update for specific rules."""

#     force_update_rules = force_update_rules or []

#     service_container = get_service_container()
#     embedding_model = service_container.embedding_service
#     llm_model = service_container.azure_openai_model

#     session_data = service_container.session_manager.get_session(session_id)
#     if not session_data:
#         return PlayBookReviewFinalResponse(rules_review=[], missing_clauses=None)

#     agent_cache = session_data.tool_results.get(AGENT_NAME, {})
#     cached_rules_review: Dict[str, PlayBookReviewResponse] = {r.rule_title: r for r in agent_cache.get("rules_review", [])}

#     # Determine which rules need (re)processing
#     rules_to_update: List[RuleInfo] = []
#     for rule in request.rulesinformation:
#         cached = cached_rules_review.get(rule.title)
#         if force_update_rules and rule.title in force_update_rules:
#             rules_to_update.append(rule)
#             continue
#         if not cached:
#             rules_to_update.append(rule)
#             continue
#         if cached.rule_description != rule.description or cached.rule_instruction != rule.instruction:
#             rules_to_update.append(rule)

#     # Return fully-cached result when nothing changed
#     if not rules_to_update:
#         return PlayBookReviewFinalResponse(
#             rules_review=list(cached_rules_review.values()),
#             missing_clauses=agent_cache.get("missing_clauses"),
#         )

#     # Precompute all paragraph embeddings once, concurrently
#     para_embeddings = np.array(await asyncio.gather(*[get_embedding(embedding_model, item.text) for item in request.textinformation]))

#     # Process all stale rules concurrently
#     rule_results = await asyncio.gather(*[_process_single_rule(rule, para_embeddings, request, llm_model, embedding_model) for rule in rules_to_update])
#     cached_rules_review.update(dict(rule_results))

#     # Use ALL paragraph text — not just matched ones — so nothing is
#     # invisible to the missing-clauses LLM
#     all_text = " ".join(p.text for p in request.textinformation)

#     # Only re-run missing-clauses when the document itself changed
#     current_doc_hash = _doc_hash(request.textinformation)
#     cached_doc_hash = agent_cache.get("doc_hash")

#     if current_doc_hash != cached_doc_hash:
#         logger.info("Document changed — re-evaluating missing clauses.")

#         # Pass the full rules-review summary so the missing-clauses LLM doesn't
#         # double-flag topics already handled by the rules review
#         reviewed_rules_summary = _build_reviewed_rules_summary(cached_rules_review)

#         missing_clauses = await get_missing_clauses(
#             data=all_text,
#             reviewed_rules_summary=reviewed_rules_summary,
#         )
#     else:
#         logger.info("Document unchanged — reusing cached missing clauses.")
#         missing_clauses = agent_cache.get("missing_clauses")

#     session_data.tool_results[AGENT_NAME] = {
#         "rules_review": list(cached_rules_review.values()),
#         "missing_clauses": missing_clauses,
#         "doc_hash": current_doc_hash,
#     }

#     return PlayBookReviewFinalResponse(
#         rules_review=list(cached_rules_review.values()),
#         missing_clauses=missing_clauses,
#     )
