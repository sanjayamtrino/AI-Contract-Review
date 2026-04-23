"""
Integration tests for doc-grounded single_clause drafting.

Covers:
- Session with no document → behaves like a generic draft (unchanged path).
- Session with document → drafter invokes key_information extraction + retrieval,
  injects parties + governing law into the prompt, and flags grounded_in_document=True.
- Session with document where retrieval returns a matching clause + LLM confirms →
  response mode is single_clause_exists with the existing clause body in
  drafted_clause.
- Regenerate with document → duplicate check is skipped.
"""
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.schemas.describe_draft import (
    ClauseListEntry,
    ClauseListLLMResponse,
    ClauseVersion,
    DescribeDraftLLMResponse,
    DuplicateCheckResult,
    IntentClassification,
)
from src.tools.drafter import generate_describe_draft


def _make_classification(mode: str, agreement_type: Optional[str] = "NDA") -> IntentClassification:
    return IntentClassification(
        mode=mode,
        detected_agreement_type=agreement_type if mode != "clarification" else None,
        clarification_question=None,
    )


def _make_draft_response(title: str = "Confidentiality Obligations") -> DescribeDraftLLMResponse:
    return DescribeDraftLLMResponse(
        versions=[
            ClauseVersion(
                title=title,
                summary="A mutual confidentiality undertaking.",
                drafted_clause=(
                    "Acme Holdings Inc. and Beacon Ventures LLC each agree to keep in strict "
                    "confidence all Confidential Information disclosed by the other party. " * 2
                ),
            )
        ]
    )


def _build_container(
    classification: IntentClassification,
    draft_response: Optional[DescribeDraftLLMResponse],
    has_document: bool,
    key_info_payload: Optional[dict] = None,
    retrieval_chunks: Optional[list] = None,
    duplicate_result: Optional[DuplicateCheckResult] = None,
) -> MagicMock:
    """
    Build a mock service container.

    LLM call order (depends on flow):
      1. classifier
      2. (if retrieval has a hot chunk) duplicate_check LLM call → DuplicateCheckResult
      3. generation → DescribeDraftLLMResponse  (skipped if duplicate_result.is_duplicate=True)
    """
    container = MagicMock()

    session = MagicMock()
    session.metadata = {}
    session.documents = {"doc-1": {}} if has_document else {}
    session.chunk_store = {0: MagicMock(content="...")} if has_document else {}
    container.session_manager.get_or_create_session.return_value = session
    container.session_manager.get_session.return_value = session

    # LLM sequencing
    call_log: list = []

    async def generate_side_effect(*args, **kwargs):
        model = kwargs.get("response_model")
        call_log.append(model)
        if model is IntentClassification:
            return classification
        if model is DuplicateCheckResult:
            return duplicate_result or DuplicateCheckResult(is_duplicate=False, matched_title=None)
        if model is DescribeDraftLLMResponse:
            if draft_response is None:
                raise ValueError("no draft response configured")
            return draft_response
        if model is ClauseListLLMResponse:
            if draft_response is None:
                raise ValueError("no list response configured")
            return draft_response
        # key_information uses KeyInformationToolResponse; return the raw dict via patched get_key_information instead.
        raise AssertionError(f"Unexpected response_model: {model}")

    container.azure_openai_model.generate = AsyncMock(side_effect=generate_side_effect)

    # Retrieval
    retrieval_payload = {
        "chunks": retrieval_chunks or [],
        "num_results": len(retrieval_chunks or []),
    }
    container.retrieval_service.retrieve_data = AsyncMock(return_value=retrieval_payload)

    container._call_log = call_log
    container._session = session
    return container, key_info_payload


@pytest.mark.asyncio
async def test_no_document_attached_skips_grounding():
    """Session without a document → no key_information call, no retrieval, grounded_in_document=False."""
    classification = _make_classification("single_clause", "NDA")
    # No-doc path requires placeholder tokens in the drafted body.
    draft = DescribeDraftLLMResponse(
        versions=[
            ClauseVersion(
                title="Confidentiality Obligations",
                summary="A mutual confidentiality undertaking.",
                drafted_clause=(
                    "[PARTY A] and [PARTY B] each agree to keep in strict confidence all "
                    "Confidential Information disclosed by the other party until the "
                    "[EXPIRATION DATE]. " * 2
                ),
            )
        ]
    )
    container, _ = _build_container(
        classification=classification,
        draft_response=draft,
        has_document=False,
    )

    with patch("src.tools.drafter.get_service_container", return_value=container), \
         patch("src.tools.key_information.get_key_information", new=AsyncMock()) as mock_key_info:
        response = await generate_describe_draft(
            prompt="Draft a confidentiality clause",
            session_id="test-grounding-nodoc",
        )

    assert response.status == "ok"
    assert response.mode == "single_clause"
    assert response.grounded_in_document is False
    mock_key_info.assert_not_called()
    container.retrieval_service.retrieve_data.assert_not_called()


@pytest.mark.asyncio
async def test_document_attached_grounds_draft_with_parties_and_governing_law():
    """With a doc attached, key_information + retrieval are called; response flags grounded_in_document=True."""
    classification = _make_classification("single_clause", "NDA")
    draft = _make_draft_response()

    key_info_payload = {
        "parties": [
            {"name": "Acme Holdings Inc.", "role": "Disclosing Party"},
            {"name": "Beacon Ventures LLC", "role": "Receiving Party"},
        ],
        "governing_law": {"value": "State of Delaware", "information": "Delaware law governs."},
    }
    retrieval_chunks = [
        {
            "content": "Some unrelated boilerplate paragraph from the uploaded contract.",
            "similarity_score": 0.30,
            "metadata": {"section_heading": "Preamble"},
        }
    ]

    container, _ = _build_container(
        classification=classification,
        draft_response=draft,
        has_document=True,
        key_info_payload=key_info_payload,
        retrieval_chunks=retrieval_chunks,
    )

    with patch("src.tools.drafter.get_service_container", return_value=container), \
         patch("src.tools.key_information.get_key_information",
               new=AsyncMock(return_value=key_info_payload)):
        response = await generate_describe_draft(
            prompt="Draft a confidentiality clause",
            session_id="test-grounding-doc",
        )

    assert response.status == "ok"
    assert response.mode == "single_clause"
    assert response.grounded_in_document is True
    container.retrieval_service.retrieve_data.assert_awaited_once()
    # Grounding cached in session metadata for later calls / regenerate.
    assert "draft_doc_grounding" in container._session.metadata
    assert container._session.metadata["draft_doc_grounding"]["parties"][0]["name"] == "Acme Holdings Inc."


@pytest.mark.asyncio
async def test_duplicate_clause_detected_returns_single_clause_exists_mode():
    """When retrieval hits a matching chunk and the LLM confirms duplicate, no draft is produced."""
    classification = _make_classification("single_clause", "NDA")

    key_info_payload = {
        "parties": [{"name": "Acme Holdings Inc.", "role": "Disclosing Party"}],
        "governing_law": {"value": "State of Delaware", "information": ""},
    }
    matching_chunk_text = (
        "Confidentiality. Each party agrees to maintain in strict confidence all "
        "Confidential Information received from the other party and shall not disclose "
        "it to any third party without prior written consent."
    )
    retrieval_chunks = [
        {
            "index": 3,
            "content": matching_chunk_text,
            "similarity_score": 0.82,
            "metadata": {"section_heading": "Confidentiality", "page_number": 7},
        }
    ]
    duplicate_yes = DuplicateCheckResult(is_duplicate=True, matched_title="Confidentiality")

    container, _ = _build_container(
        classification=classification,
        draft_response=None,  # drafter should NOT call generation
        has_document=True,
        key_info_payload=key_info_payload,
        retrieval_chunks=retrieval_chunks,
        duplicate_result=duplicate_yes,
    )

    with patch("src.tools.drafter.get_service_container", return_value=container), \
         patch("src.tools.key_information.get_key_information",
               new=AsyncMock(return_value=key_info_payload)):
        response = await generate_describe_draft(
            prompt="Draft a confidentiality clause",
            session_id="test-grounding-dup",
        )

    assert response.status == "ok"
    assert response.mode == "single_clause_exists"
    assert response.existing_clause is not None
    assert response.existing_clause.title == "Confidentiality"
    assert "Confidentiality" in response.existing_clause.drafted_clause
    assert response.existing_clause.similarity_score == pytest.approx(0.82)
    assert response.grounded_in_document is True
    assert response.versions == []
    # Location must be populated from the chunk's top-level index + metadata.
    loc = response.existing_clause.location
    assert loc is not None
    assert loc.chunk_index == 3
    assert loc.section_heading == "Confidentiality"
    assert loc.page_number == 7
    # LLM calls: classifier + duplicate check = 2 (NO generation call)
    models_called = container._call_log
    assert IntentClassification in models_called
    assert DuplicateCheckResult in models_called
    assert DescribeDraftLLMResponse not in models_called


@pytest.mark.asyncio
async def test_low_similarity_does_not_trigger_duplicate_llm():
    """When top chunk similarity is below the gate, the LLM duplicate check is skipped entirely."""
    classification = _make_classification("single_clause", "NDA")
    draft = _make_draft_response()

    key_info_payload = {
        "parties": [{"name": "Acme Holdings Inc.", "role": "Disclosing Party"}],
        "governing_law": {"value": "", "information": ""},
    }
    retrieval_chunks = [
        {
            "content": "Some unrelated clause about payment terms.",
            "similarity_score": 0.20,
            "metadata": {},
        }
    ]

    container, _ = _build_container(
        classification=classification,
        draft_response=draft,
        has_document=True,
        key_info_payload=key_info_payload,
        retrieval_chunks=retrieval_chunks,
    )

    with patch("src.tools.drafter.get_service_container", return_value=container), \
         patch("src.tools.key_information.get_key_information",
               new=AsyncMock(return_value=key_info_payload)):
        response = await generate_describe_draft(
            prompt="Draft a confidentiality clause",
            session_id="test-grounding-lowsim",
        )

    assert response.status == "ok"
    assert response.mode == "single_clause"
    # Duplicate-check LLM should NOT have been called.
    assert DuplicateCheckResult not in container._call_log


@pytest.mark.asyncio
async def test_regenerate_with_document_skips_duplicate_check():
    """Regenerate flow should not block on duplicate detection — user explicitly wants a new draft."""
    classification = _make_classification("single_clause", "NDA")
    new_draft = DescribeDraftLLMResponse(
        versions=[
            ClauseVersion(
                title="Confidentiality Obligations",
                summary="A tighter variant.",
                drafted_clause=(
                    "Receiving Party shall hold Confidential Information in strict confidence. "
                    "Permitted Purpose. Carve-outs apply. " * 3
                ),
            )
        ]
    )

    key_info_payload = {
        "parties": [{"name": "Acme Holdings Inc.", "role": "Disclosing Party"}],
        "governing_law": {"value": "State of Delaware", "information": ""},
    }
    retrieval_chunks = [
        {
            "content": "Confidentiality. Each party agrees to keep information confidential.",
            "similarity_score": 0.90,  # Would normally trigger duplicate check
            "metadata": {"section_heading": "Confidentiality"},
        }
    ]

    container, _ = _build_container(
        classification=classification,
        draft_response=new_draft,
        has_document=True,
        key_info_payload=key_info_payload,
        retrieval_chunks=retrieval_chunks,
        duplicate_result=DuplicateCheckResult(is_duplicate=True, matched_title="Confidentiality"),
    )
    # Seed a prior draft so regenerate is effective.
    container._session.metadata["draft_last_version"] = {
        "title": "Confidentiality Obligations",
        "summary": "An earlier broad version.",
        "drafted_clause": (
            "The Receiving Party shall keep information confidential. " * 5
        ),
    }

    with patch("src.tools.drafter.get_service_container", return_value=container), \
         patch("src.tools.key_information.get_key_information",
               new=AsyncMock(return_value=key_info_payload)):
        response = await generate_describe_draft(
            prompt="Draft a confidentiality clause",
            session_id="test-grounding-regen",
            regenerate=True,
        )

    assert response.status == "ok"
    assert response.mode == "single_clause"
    assert response.regenerated is True
    assert response.grounded_in_document is True
    # Duplicate-check LLM must NOT have been called on the regenerate path.
    assert DuplicateCheckResult not in container._call_log


@pytest.mark.asyncio
async def test_duplicate_match_with_sparse_metadata_still_returns_some_location():
    """When the chunk has no metadata keys, location can fall back to the top-level index alone."""
    classification = _make_classification("single_clause", "NDA")

    key_info_payload = {
        "parties": [{"name": "Acme Holdings Inc.", "role": "Disclosing Party"}],
        "governing_law": {"value": "State of Delaware", "information": ""},
    }
    retrieval_chunks = [
        {
            "index": 11,
            "content": (
                "Each party agrees that all Confidential Information received from the "
                "other shall be held in strict confidence and used solely for the "
                "Permitted Purpose."
            ),
            "similarity_score": 0.77,
            "metadata": {},  # no section heading or page number
        }
    ]
    duplicate_yes = DuplicateCheckResult(is_duplicate=True, matched_title=None)

    container, _ = _build_container(
        classification=classification,
        draft_response=None,
        has_document=True,
        key_info_payload=key_info_payload,
        retrieval_chunks=retrieval_chunks,
        duplicate_result=duplicate_yes,
    )

    with patch("src.tools.drafter.get_service_container", return_value=container), \
         patch("src.tools.key_information.get_key_information",
               new=AsyncMock(return_value=key_info_payload)):
        response = await generate_describe_draft(
            prompt="Draft a confidentiality clause",
            session_id="test-grounding-loc-sparse",
        )

    assert response.mode == "single_clause_exists"
    assert response.existing_clause is not None
    loc = response.existing_clause.location
    assert loc is not None
    assert loc.chunk_index == 11
    assert loc.section_heading is None
    assert loc.page_number is None


@pytest.mark.asyncio
async def test_list_of_clauses_with_document_uses_real_party_names_and_no_placeholders():
    """Doc-grounded list mode: each drafted body uses real party names, no [PLACEHOLDER] tokens."""
    classification = _make_classification("list_of_clauses", "NDA")

    key_info_payload = {
        "parties": [
            {"name": "Acme Holdings Inc.", "role": "Disclosing Party"},
            {"name": "Beacon Ventures LLC", "role": "Receiving Party"},
        ],
        "governing_law": {"value": "State of Delaware", "information": "Delaware law governs."},
    }

    # Build a 13-clause list where each body uses the real party names.
    def _body(i: int) -> str:
        return (
            f"For purposes of Section {i + 1}, Acme Holdings Inc. and Beacon Ventures LLC "
            f"agree that the terms set forth apply from the effective date of this Agreement. "
            f"The Agreement is governed by the laws of the State of Delaware. " * 2
        )

    list_response = ClauseListLLMResponse(
        clauses=[
            ClauseListEntry(
                title="Definitions" if i == 0 else f"Section {i + 1} — Topic {i}",
                summary=f"A one-sentence description for section {i + 1}.",
                drafted_clause=_body(i),
            )
            for i in range(13)
        ]
    )

    container, _ = _build_container(
        classification=classification,
        draft_response=list_response,
        has_document=True,
        key_info_payload=key_info_payload,
        retrieval_chunks=[],  # list mode does not retrieve chunks
    )

    with patch("src.tools.drafter.get_service_container", return_value=container), \
         patch("src.tools.key_information.get_key_information",
               new=AsyncMock(return_value=key_info_payload)):
        response = await generate_describe_draft(
            prompt="Draft an NDA",
            session_id="test-grounding-list",
        )

    assert response.status == "ok"
    assert response.mode == "list_of_clauses"
    assert len(response.clauses) == 13
    assert response.grounded_in_document is True
    for entry in response.clauses:
        assert "Acme Holdings Inc." in entry.drafted_clause
        assert "Beacon Ventures LLC" in entry.drafted_clause
        assert entry.placeholders == []  # doc-grounded → no placeholders


@pytest.mark.asyncio
async def test_doc_grounded_draft_containing_placeholders_triggers_validation_error():
    """Doc-grounded single-clause drafts must not contain [PLACEHOLDER] tokens."""
    classification = _make_classification("single_clause", "NDA")

    key_info_payload = {
        "parties": [{"name": "Acme Holdings Inc.", "role": "Disclosing Party"}],
        "governing_law": {"value": "State of Delaware", "information": ""},
    }

    # Bad LLM response: pretends to be doc-grounded but still emits [PARTY A].
    bad_draft = DescribeDraftLLMResponse(versions=[
        ClauseVersion(
            title="Confidentiality Obligations",
            summary="Cross-wired draft that still uses placeholders.",
            drafted_clause=(
                "[PARTY A] and [PARTY B] agree to maintain all Confidential Information "
                "in strict confidence. The obligations survive termination. " * 2
            ),
        )
    ])

    container, _ = _build_container(
        classification=classification,
        draft_response=bad_draft,
        has_document=True,
        key_info_payload=key_info_payload,
        retrieval_chunks=[],
    )

    with patch("src.tools.drafter.get_service_container", return_value=container), \
         patch("src.tools.key_information.get_key_information",
               new=AsyncMock(return_value=key_info_payload)):
        response = await generate_describe_draft(
            prompt="Draft a confidentiality clause",
            session_id="test-grounding-forbid",
        )

    assert response.status == "error"
    assert response.error_message is not None
    assert "placeholder" in response.error_message.lower()
