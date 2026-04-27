from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from src.schemas.playbook_review_srikar import (
    PlayBookReviewLLMResponse,
    RuleCheckRequest,
    RuleInfo,
    TextInfo,
)
from src.tools import playbook_review_srikar as agent
from src.tools.playbook_review_srikar import (
    _best_clause_by_fuzzy_title,
    _expand_paragraphs_with_inline_headings,
    _extract_heading_from_text,
    _fuzzy_title_ratio,
    _group_paragraphs_into_clauses,
    _split_paragraph_at_inline_headings,
    _strip_split_suffix,
    review_document,
)


# ─────────────────────────────────────────────────────────────
# Heading extraction
# ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1. Confidential Information. The term means any non-public information.", "1. Confidential Information"),
        ("1.2 Termination. Either party may terminate.", "1.2 Termination"),
        ("Section 5 - Liability. Each party is liable.", "Section 5 - Liability"),
        ("ARTICLE III", "ARTICLE III"),
        ("Audit Rights. Each party may audit annually.", "Audit Rights"),
        ("This Agreement is made on April 27, 2026 between Acme and Beta.", None),
        ("", None),
        ("MUTUAL NON-DISCLOSURE AGREEMENT", "MUTUAL NON-DISCLOSURE AGREEMENT"),
    ],
)
def test_extract_heading_from_text(text, expected):
    assert _extract_heading_from_text(text) == expected


def test_group_paragraphs_into_clauses_groups_under_headings():
    paras = [
        TextInfo(text="Preamble line.", paraindetifier="P1"),
        TextInfo(text="1. Confidentiality. The receiving party shall keep confidential.", paraindetifier="P2"),
        TextInfo(text="No confidential information is to be disclosed.", paraindetifier="P3"),
        TextInfo(text="2. Termination. Either party may terminate.", paraindetifier="P4"),
    ]

    clauses = _group_paragraphs_into_clauses(paras)

    assert [h for h, _ in clauses] == [None, "1. Confidentiality", "2. Termination"]
    assert [[p.paraindetifier for p in ps] for _, ps in clauses] == [["P1"], ["P2", "P3"], ["P4"]]


# ─────────────────────────────────────────────────────────────
# Fuzzy title matching
# ─────────────────────────────────────────────────────────────


def test_fuzzy_title_ratio_token_overlap_beats_short_word_false_positive():
    # "Term" should NOT 1.0-match a paragraph that merely contains the word "term"
    # in body text — heading is what matters.
    short_match = _fuzzy_title_ratio("Indemnification", "Confidential Information")
    assert short_match < 0.70

    # But a single-token rule that matches a clause heading via token overlap
    # should still pass (e.g. "Liability" → "Limitation of Liability").
    assert _fuzzy_title_ratio("Liability", "Section 4 - Limitation of Liability") >= 0.70


def test_fuzzy_title_ratio_handles_synonyms_below_threshold():
    # Pure synonym pairs (different lemma) sit below the fuzzy threshold so
    # the embedding fallback gets a chance.
    assert _fuzzy_title_ratio("Force Majeure", "Governing Law") < 0.70


def test_best_clause_by_fuzzy_title_picks_correct_clause():
    paras = [
        TextInfo(text="1. Confidentiality. Receiving party shall...", paraindetifier="P1"),
        TextInfo(text="2. Termination. Either party may...", paraindetifier="P2"),
        TextInfo(text="3. Governing Law. This Agreement is governed by...", paraindetifier="P3"),
    ]
    clauses = _group_paragraphs_into_clauses(paras)

    idx, ratio = _best_clause_by_fuzzy_title("Termination", clauses)
    assert clauses[idx][0] == "2. Termination"
    assert ratio >= 0.70


# ─────────────────────────────────────────────────────────────
# Full pipeline (mocked LLM + embedding service)
# ─────────────────────────────────────────────────────────────


def _mock_llm_response_good() -> PlayBookReviewLLMResponse:
    return PlayBookReviewLLMResponse(
        para_identifiers=["P2"],
        status="Good",
        reason="Clause meets the rule.",
        suggestion="",
        suggested_fix="",
    )


def _build_container_mock(
    llm_response: PlayBookReviewLLMResponse,
    chunk_store: dict = None,
):
    container = MagicMock()

    # Embedding service: deterministic per-text vectors
    async def _gen_emb(text):
        return np.array([float(len(text) % 7), 1.0, 2.0])

    container.embedding_service.generate_embeddings = AsyncMock(side_effect=_gen_emb)

    # LLM
    container.azure_openai_model.generate = AsyncMock(return_value=llm_response)

    # Session — chunk_store populated only when caller passes one
    session = MagicMock()
    session.tool_results = {}
    session.chunk_store = chunk_store if chunk_store is not None else {}
    container.session_manager.get_session = MagicMock(return_value=session)

    return container


@pytest.mark.asyncio
async def test_review_document_title_match_populates_matched_clause_titles(monkeypatch):
    """When a rule title matches a clause heading via fuzzy match,
    matched_clause_titles is set and match_strategy is 'title_fuzzy'."""
    container = _build_container_mock(_mock_llm_response_good())
    monkeypatch.setattr(agent, "get_service_container", lambda: container)

    # Reset the module-level embedding cache so this test is hermetic.
    agent._embedding_cache.clear()

    request = RuleCheckRequest(
        rulesinformation=[
            RuleInfo(title="Confidentiality", instruction="...", description="..."),
        ],
        textinformation=[
            TextInfo(text="Preamble line of the contract.", paraindetifier="P1"),
            TextInfo(text="1. Confidentiality. Receiving party shall keep information secret.", paraindetifier="P2"),
            TextInfo(text="2. Termination. Either party may terminate.", paraindetifier="P3"),
        ],
    )

    result = await review_document(session_id="s1", request=request)

    assert len(result.rules_review) == 1
    review = result.rules_review[0]
    assert review.rule_title == "Confidentiality"
    assert review.matched_clause_titles == ["1. Confidentiality"]
    assert review.match_strategy in {"title_fuzzy", "title_embedding"}
    assert review.content.status == "Good"


@pytest.mark.asyncio
async def test_review_document_no_heading_falls_back_to_content(monkeypatch):
    """A document with no detectable headings should still produce a review
    via the content-based fallback, with empty matched_clause_titles."""
    container = _build_container_mock(_mock_llm_response_good())
    monkeypatch.setattr(agent, "get_service_container", lambda: container)
    agent._embedding_cache.clear()

    request = RuleCheckRequest(
        rulesinformation=[
            RuleInfo(title="Confidentiality", instruction="...", description="..."),
        ],
        textinformation=[
            TextInfo(text="this contract has no obvious headings just running text.", paraindetifier="P1"),
            TextInfo(text="parties shall keep information confidential at all times.", paraindetifier="P2"),
        ],
    )

    result = await review_document(session_id="s1", request=request)

    review = result.rules_review[0]
    assert review.match_strategy == "content_fallback"
    assert review.matched_clause_titles == []


# ─────────────────────────────────────────────────────────────
# Session-chunk fallback (no textinformation in request)
# ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_review_document_falls_back_to_session_chunk_store(monkeypatch):
    """When the request omits textinformation, paragraphs are pulled from
    the session's chunk_store (populated by the ingestion endpoint)."""
    chunk_store = {
        0: SimpleNamespace(
            chunk_index=0,
            content="Preamble. This Agreement is entered into between Acme and Beta.",
        ),
        1: SimpleNamespace(
            chunk_index=1,
            content="1. Confidentiality. Receiving party shall keep information secret.",
        ),
        2: SimpleNamespace(
            chunk_index=2,
            content="2. Termination. Either party may terminate.",
        ),
    }
    container = _build_container_mock(_mock_llm_response_good(), chunk_store=chunk_store)
    monkeypatch.setattr(agent, "get_service_container", lambda: container)
    agent._embedding_cache.clear()

    # textinformation deliberately omitted — should be pulled from session.
    request = RuleCheckRequest(
        rulesinformation=[
            RuleInfo(title="Confidentiality", instruction="...", description="..."),
        ],
    )

    result = await review_document(session_id="s1", request=request)

    assert len(result.rules_review) == 1
    review = result.rules_review[0]
    # Heading-based match should still anchor on the session-derived paragraphs.
    assert review.matched_clause_titles == ["1. Confidentiality"]
    assert review.match_strategy in {"title_fuzzy", "title_embedding"}
    # Confirm the request was hydrated from session chunks with P####-style ids.
    assert request.textinformation is not None
    assert [p.paraindetifier for p in request.textinformation] == ["P0001", "P0002", "P0003"]


@pytest.mark.asyncio
async def test_review_document_returns_empty_when_no_paragraphs_anywhere(monkeypatch):
    """No textinformation in the request and no chunks in the session →
    the agent returns an empty review without calling the LLM."""
    container = _build_container_mock(_mock_llm_response_good())  # chunk_store={}
    monkeypatch.setattr(agent, "get_service_container", lambda: container)
    agent._embedding_cache.clear()

    request = RuleCheckRequest(
        rulesinformation=[
            RuleInfo(title="Confidentiality", instruction="...", description="..."),
        ],
    )

    result = await review_document(session_id="s1", request=request)

    assert result.rules_review == []
    assert result.missing_clauses is None
    container.azure_openai_model.generate.assert_not_called()


# ─────────────────────────────────────────────────────────────
# Inline heading splitter (coarse-chunk safety)
# ─────────────────────────────────────────────────────────────


def test_split_paragraph_no_inline_headings_returns_original():
    para = TextInfo(
        paraindetifier="P0001",
        text="This is preamble text without any clause headings.",
    )
    out = _split_paragraph_at_inline_headings(para)
    assert out == [para]


def test_split_paragraph_single_heading_returns_original():
    para = TextInfo(
        paraindetifier="P0002",
        text="1. Confidentiality. The receiving party shall keep all information secret.",
    )
    out = _split_paragraph_at_inline_headings(para)
    # Only one heading -> nothing to split
    assert out == [para]


def test_split_paragraph_multi_clause_breaks_into_segments():
    para = TextInfo(
        paraindetifier="P0007",
        text=(
            "6. Term. This Agreement remains in effect for one year. "
            "7. Remedies. Disclosing party may seek injunctive relief. "
            "8. Assignment. Neither party may assign without consent. "
            "9. Governing Law. This Agreement is governed by Delaware law."
        ),
    )
    segments = _split_paragraph_at_inline_headings(para)
    assert len(segments) == 4
    assert all(s.paraindetifier.startswith("P0007#") for s in segments)
    # Each segment should start with its own heading
    assert segments[0].text.startswith("6. Term")
    assert segments[1].text.startswith("7. Remedies")
    assert segments[2].text.startswith("8. Assignment")
    assert segments[3].text.startswith("9. Governing Law")


def test_split_paragraph_preserves_preamble_before_first_heading():
    para = TextInfo(
        paraindetifier="P0007",
        text=(
            "Some preamble explaining the section. "
            "1. Confidentiality. Receiving party shall keep secret. "
            "2. Term. Effective for one year."
        ),
    )
    segments = _split_paragraph_at_inline_headings(para)
    assert len(segments) == 3
    assert "preamble" in segments[0].text.lower()
    assert segments[1].text.startswith("1. Confidentiality")
    assert segments[2].text.startswith("2. Term")


def test_strip_split_suffix_handles_both_forms():
    assert _strip_split_suffix("P0007#3") == "P0007"
    assert _strip_split_suffix("P0001") == "P0001"  # untouched
    assert _strip_split_suffix("") == ""  # safe on empty


def test_expand_paragraphs_with_inline_headings_handles_failure_gracefully(monkeypatch):
    """If the splitter raises for one paragraph, the original is kept and
    other paragraphs continue processing — pipeline never breaks."""
    def _broken_splitter(para):
        if para.paraindetifier == "P0002":
            raise RuntimeError("simulated parser failure")
        return [para]

    monkeypatch.setattr(
        "src.tools.playbook_review_srikar._split_paragraph_at_inline_headings",
        _broken_splitter,
    )

    paras = [
        TextInfo(paraindetifier="P0001", text="alpha"),
        TextInfo(paraindetifier="P0002", text="beta"),  # this one will raise
        TextInfo(paraindetifier="P0003", text="gamma"),
    ]
    out = _expand_paragraphs_with_inline_headings(paras)
    assert [p.paraindetifier for p in out] == ["P0001", "P0002", "P0003"]


@pytest.mark.asyncio
async def test_review_document_title_match_works_after_inline_split(monkeypatch):
    """End-to-end: a single coarse chunk containing 4 clauses should expand,
    title-match correctly per rule, and return clean (un-suffixed) para IDs."""
    container = _build_container_mock(
        PlayBookReviewLLMResponse(
            para_identifiers=["P0007#3"],  # synthetic id from the splitter
            status="Good",
            reason="Matches rule.",
            suggestion="",
            suggested_fix="",
        ),
    )
    monkeypatch.setattr(agent, "get_service_container", lambda: container)
    agent._embedding_cache.clear()

    request = RuleCheckRequest(
        rulesinformation=[
            RuleInfo(title="Assignment", instruction="...", description="..."),
        ],
        textinformation=[
            TextInfo(
                paraindetifier="P0007",
                text=(
                    "6. Term. This Agreement remains in effect for one year. "
                    "7. Remedies. Disclosing party may seek injunctive relief. "
                    "8. Assignment. Neither party may assign without consent. "
                    "9. Governing Law. Governed by Delaware law."
                ),
            ),
        ],
    )

    result = await review_document(session_id="s1", request=request)

    review = result.rules_review[0]
    # Title matching should now anchor on the per-clause segment "8. Assignment"
    assert review.matched_clause_titles == ["8. Assignment"]
    assert review.match_strategy in {"title_fuzzy", "title_embedding"}
    # Synthetic suffix must be stripped before reaching the frontend
    assert review.content.para_identifiers == ["P0007"]


# ─────────────────────────────────────────────────────────────
# 3-tier clause-name resolution: regex / parser metadata / LLM label
# ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_review_document_uses_llm_label_when_title_match_fails(monkeypatch):
    """When title matching can't anchor a clause heading and the LLM supplies
    matched_clause_name, the agent surfaces that name and changes
    match_strategy to 'llm_label'."""
    container = _build_container_mock(
        PlayBookReviewLLMResponse(
            para_identifiers=["P0001"],
            matched_clause_name="Confidential Information",
            status="Good",
            reason="Clause matches.",
            suggestion="",
            suggested_fix="",
        ),
    )
    monkeypatch.setattr(agent, "get_service_container", lambda: container)
    agent._embedding_cache.clear()

    # Lower-case body text only — no detectable heading anywhere.
    request = RuleCheckRequest(
        rulesinformation=[
            RuleInfo(title="Confidentiality", instruction="...", description="..."),
        ],
        textinformation=[
            TextInfo(text="this contract has no obvious headings just running text.", paraindetifier="P0001"),
            TextInfo(text="parties shall keep information confidential at all times.", paraindetifier="P0002"),
        ],
    )

    result = await review_document(session_id="s1", request=request)

    review = result.rules_review[0]
    assert review.matched_clause_titles == ["Confidential Information"]
    assert review.match_strategy == "llm_label"


@pytest.mark.asyncio
async def test_review_document_title_match_wins_over_llm_label(monkeypatch):
    """Even if the LLM provides matched_clause_name, the regex/metadata title
    match takes precedence — that label is the most accurate."""
    container = _build_container_mock(
        PlayBookReviewLLMResponse(
            para_identifiers=["P0002"],
            matched_clause_name="Some LLM-Named Topic",
            status="Good",
            reason="ok",
            suggestion="",
            suggested_fix="",
        ),
    )
    monkeypatch.setattr(agent, "get_service_container", lambda: container)
    agent._embedding_cache.clear()

    request = RuleCheckRequest(
        rulesinformation=[
            RuleInfo(title="Confidentiality", instruction="...", description="..."),
        ],
        textinformation=[
            TextInfo(text="Preamble", paraindetifier="P0001"),
            TextInfo(text="1. Confidentiality. Receiving party shall keep secret.", paraindetifier="P0002"),
        ],
    )

    result = await review_document(session_id="s1", request=request)

    review = result.rules_review[0]
    # Heading from text wins, not the LLM-supplied label.
    assert review.matched_clause_titles == ["1. Confidentiality"]
    assert review.match_strategy in {"title_fuzzy", "title_embedding"}


@pytest.mark.asyncio
async def test_review_document_uses_parser_metadata_section_heading(monkeypatch):
    """When pulling from session chunk_store, parser-supplied section_heading
    in chunk metadata is prepended to content so title matching can use it."""
    chunk_store = {
        0: SimpleNamespace(
            chunk_index=0,
            content="Receiving party shall keep all information secret.",
            metadata={"section_heading": "Confidentiality"},
        ),
        1: SimpleNamespace(
            chunk_index=1,
            content="Either party may terminate.",
            metadata={"section_heading": "Termination"},
        ),
    }
    container = _build_container_mock(
        PlayBookReviewLLMResponse(
            para_identifiers=["P0001"],
            matched_clause_name="",
            status="Good",
            reason="ok",
            suggestion="",
            suggested_fix="",
        ),
        chunk_store=chunk_store,
    )
    monkeypatch.setattr(agent, "get_service_container", lambda: container)
    agent._embedding_cache.clear()

    request = RuleCheckRequest(
        rulesinformation=[
            RuleInfo(title="Confidentiality", instruction="...", description="..."),
        ],
    )

    result = await review_document(session_id="s1", request=request)

    review = result.rules_review[0]
    # Parser metadata heading made the chunk look like "Confidentiality. ..."
    # to the heading extractor — so title fuzzy matching anchored cleanly.
    assert review.matched_clause_titles == ["Confidentiality"]
    assert review.match_strategy in {"title_fuzzy", "title_embedding"}
