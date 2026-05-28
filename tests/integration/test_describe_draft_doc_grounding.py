"""
Integration tests for document-grounded drafting (Use Document Context = on).

Covers:
- use_document_context=False → document is ignored, no key_information / retrieval calls.
- use_document_context=True with a document → drafter invokes key_information extraction
  + retrieval, injects parties + governing law, and flags grounded_in_document=True.
- Doc-grounded list mode uses the real party names and forbids [PLACEHOLDER] tokens.
- A doc-grounded draft that leaks party-identity placeholders fails validation.
"""
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.schemas.describe_draft import (
    ClauseListEntry,
    ClauseListLLMResponse,
    DescribeDraftLLMResponse,
    DraftedClause,
    IntentClassification,
)
from src.tools.drafter import generate_describe_draft


_GOOD_SUMMARY = (
    "A mutual confidentiality undertaking that binds both parties, allocates the "
    "disclosure risk to the receiving party, and carves out information that becomes "
    "public through no fault of the recipient."
)

_GOOD_AGREEMENT_SUMMARY = (
    "This mutual non-disclosure agreement governs how the parties exchange and protect "
    "confidential information, sets out permitted uses and recipients, and describes the "
    "survival period and the principal carve-outs each side should know about."
)


def _make_classification(mode: str, agreement_type: Optional[str] = "NDA") -> IntentClassification:
    return IntentClassification(mode=mode, detected_agreement_type=agreement_type)


def _make_draft_response(title: str = "Confidentiality Obligations") -> DescribeDraftLLMResponse:
    return DescribeDraftLLMResponse(
        clause=DraftedClause(
            title=title,
            summary=_GOOD_SUMMARY,
            drafted_clause=(
                "Acme Holdings Inc. and Beacon Ventures LLC each agree to keep in strict "
                "confidence all Confidential Information disclosed by the other party. " * 3
            ),
        )
    )


def _build_container(
    classification: IntentClassification,
    draft_response: Optional[DescribeDraftLLMResponse],
    has_document: bool,
    retrieval_chunks: Optional[list] = None,
) -> MagicMock:
    """
    Build a mock service container.

    LLM call order:
      1. classifier → IntentClassification
      2. generation → DescribeDraftLLMResponse / ClauseListLLMResponse
    """
    container = MagicMock()

    session = MagicMock()
    session.metadata = {}
    session.documents = {"doc-1": {}} if has_document else {}
    session.chunk_store = {0: MagicMock(content="...")} if has_document else {}
    container.session_manager.get_or_create_session.return_value = session
    container.session_manager.get_session.return_value = session

    call_log: list = []

    async def generate_side_effect(*args, **kwargs):
        model = kwargs.get("response_model")
        call_log.append(model)
        if model is IntentClassification:
            return classification
        if model in (DescribeDraftLLMResponse, ClauseListLLMResponse):
            if draft_response is None:
                raise ValueError("no draft response configured")
            return draft_response
        raise AssertionError(f"Unexpected response_model: {model}")

    container.azure_openai_model.generate = AsyncMock(side_effect=generate_side_effect)

    retrieval_payload = {
        "chunks": retrieval_chunks or [],
        "num_results": len(retrieval_chunks or []),
    }
    container.retrieval_service.retrieve_data = AsyncMock(return_value=retrieval_payload)

    container._call_log = call_log
    container._session = session
    return container


@pytest.mark.asyncio
async def test_use_document_context_off_skips_grounding():
    """Box unchecked → no key_information call, no retrieval, grounded_in_document=False."""
    classification = _make_classification("single_clause", "NDA")
    draft = DescribeDraftLLMResponse(
        clause=DraftedClause(
            title="Confidentiality Obligations",
            summary=_GOOD_SUMMARY,
            drafted_clause=(
                "[PARTY A] and [PARTY B] each agree to keep in strict confidence all "
                "Confidential Information disclosed by the other party until the "
                "[EXPIRATION DATE]. " * 2
            ),
        )
    )
    container = _build_container(
        classification=classification,
        draft_response=draft,
        has_document=True,  # a document IS attached, but the box is unchecked
    )

    with patch("src.tools.drafter.get_service_container", return_value=container), \
         patch("src.tools.key_information.get_key_information", new=AsyncMock()) as mock_key_info:
        response = await generate_describe_draft(
            prompt="Draft a confidentiality clause",
            session_id="test-grounding-off",
            use_document_context=False,
        )

    assert response.status == "ok"
    assert response.mode == "single_clause"
    assert response.grounded_in_document is False
    mock_key_info.assert_not_called()
    container.retrieval_service.retrieve_data.assert_not_called()


@pytest.mark.asyncio
async def test_document_context_on_grounds_draft_with_parties_and_governing_law():
    """Box checked + doc attached → key_information + retrieval run; grounded_in_document=True."""
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

    container = _build_container(
        classification=classification,
        draft_response=draft,
        has_document=True,
        retrieval_chunks=retrieval_chunks,
    )

    with patch("src.tools.drafter.get_service_container", return_value=container), \
         patch("src.tools.key_information.get_key_information",
               new=AsyncMock(return_value=key_info_payload)):
        response = await generate_describe_draft(
            prompt="Draft a confidentiality clause",
            session_id="test-grounding-on",
            use_document_context=True,
        )

    assert response.status == "ok"
    assert response.mode == "single_clause"
    assert response.grounded_in_document is True
    container.retrieval_service.retrieve_data.assert_awaited_once()
    # Grounding cached in session metadata for later calls.
    assert "draft_doc_grounding" in container._session.metadata
    assert container._session.metadata["draft_doc_grounding"]["parties"][0]["name"] == "Acme Holdings Inc."


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

    def _body(i: int) -> str:
        return (
            f"For purposes of Section {i + 1}, Acme Holdings Inc. and Beacon Ventures LLC "
            f"agree that the terms set forth apply from the effective date of this Agreement. "
            f"The Agreement is governed by the laws of the State of Delaware. " * 2
        )

    list_response = ClauseListLLMResponse(
        agreement_summary=_GOOD_AGREEMENT_SUMMARY,
        clauses=[
            ClauseListEntry(
                title="Definitions" if i == 0 else f"Section {i + 1} — Topic {i}",
                summary=f"A concise one-sentence description for section {i + 1} of this agreement.",
                drafted_clause=_body(i),
            )
            for i in range(13)
        ],
    )

    container = _build_container(
        classification=classification,
        draft_response=list_response,
        has_document=True,
        retrieval_chunks=[],  # list mode does not retrieve chunks
    )

    with patch("src.tools.drafter.get_service_container", return_value=container), \
         patch("src.tools.key_information.get_key_information",
               new=AsyncMock(return_value=key_info_payload)):
        response = await generate_describe_draft(
            prompt="Draft an NDA",
            session_id="test-grounding-list",
            use_document_context=True,
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
async def test_doc_grounded_draft_containing_party_placeholders_triggers_validation_error():
    """Doc-grounded single-clause drafts must not wrap party identities in [PLACEHOLDER] tokens."""
    classification = _make_classification("single_clause", "NDA")

    key_info_payload = {
        "parties": [{"name": "Acme Holdings Inc.", "role": "Disclosing Party"}],
        "governing_law": {"value": "State of Delaware", "information": ""},
    }

    # Bad LLM response: pretends to be doc-grounded but still emits [PARTY A].
    bad_draft = DescribeDraftLLMResponse(
        clause=DraftedClause(
            title="Confidentiality Obligations",
            summary=(
                "A confidentiality undertaking that, despite a document being attached, "
                "incorrectly wraps the party identities in placeholder tokens instead of "
                "using the extracted names."
            ),
            drafted_clause=(
                "[PARTY A] and [PARTY B] agree to maintain all Confidential Information "
                "in strict confidence. The obligations survive termination. " * 3
            ),
        )
    )

    container = _build_container(
        classification=classification,
        draft_response=bad_draft,
        has_document=True,
        retrieval_chunks=[],
    )

    with patch("src.tools.drafter.get_service_container", return_value=container), \
         patch("src.tools.key_information.get_key_information",
               new=AsyncMock(return_value=key_info_payload)):
        response = await generate_describe_draft(
            prompt="Draft a confidentiality clause",
            session_id="test-grounding-forbid",
            use_document_context=True,
        )

    assert response.status == "error"
    assert response.error_message is not None
    assert "placeholder" in response.error_message.lower()
