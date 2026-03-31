import difflib
import hashlib
from pathlib import Path

from docx.document import Document
from pydantic import BaseModel

from src.config.logging import get_logger
from src.config.settings import get_settings
from src.dependencies import get_service_container
from src.schemas.comparision import DocumentComparisonResult

logger = get_logger(__name__)
settings = get_settings()
service_container = get_service_container()
comparision_prompt = Path(r"src\services\prompts\v1\comparision_prompt_v2.mustache").read_text()


class VersionComparisionResponse(BaseModel):
    """Response model for document version comparison."""

    clause_title: str
    change_type: str
    original_text: str | None
    revised_text: str | None
    change_summary: str
    risk_impact: str
    significance: str


class DocumentComparisonResult(BaseModel):
    """Result model for document comparison."""

    changes: list[VersionComparisionResponse]


def normalize(text: str) -> str:
    """Normalize text by removing extra whitespace and converting to lowercase."""
    return " ".join(text.strip().lower().split())


def hash_para(text: str) -> str:
    """Generate a hash for a paragraph of text."""

    normalized_text = normalize(text)
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


async def extract_paragraphs(doc: Document) -> list[str]:
    """Extract paragraphs from a document."""

    return [para.text.strip() for para in doc.paragraphs if para.text.strip()]


async def diff_paragraphs(paras1: list[str], paras2: list[str]) -> DocumentComparisonResult:
    """Diff two lists of paragraphs and return added, removed, and unchanged paragraphs."""

    doc1_hashes = [hash_para(para) for para in paras1]
    doc2_hashes = [hash_para(para) for para in paras2]

    container = get_service_container()
    llm_model = container.azure_openai_model

    matcher = difflib.SequenceMatcher(None, doc1_hashes, doc2_hashes)

    changes = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        elif tag == "delete":
            for i in range(i1, i2):
                changes.append({"type": "removed", "old": paras1[i], "new": None})

        elif tag == "insert":
            for j in range(j1, j2):
                changes.append({"type": "added", "old": None, "new": paras2[j]})

        elif tag == "replace":
            len_old = i2 - i1
            len_new = j2 - j1
            max_len = max(len_old, len_new)

            for k in range(max_len):
                old_para = paras1[i1 + k] if k < len_old else None
                new_para = paras2[j1 + k] if k < len_new else None

                changes.append({"type": "modified", "old": old_para, "new": new_para})

    # Pass the changes to the LLM one by one

    overall_changes = []

    for change in changes:
        change_type = change["type"]
        old_text = change.get("old")
        new_text = change.get("new")
        context = {
            "change_type": change_type,
            "old_text": old_text,
            "new_text": new_text,
        }
        response = await llm_model.generate(prompt=comparision_prompt, context=context, response_model=VersionComparisionResponse, mode="JSON")
        overall_changes.append(response)

    return overall_changes


async def compare_doc_versions(doc1: Document, doc2: Document) -> DocumentComparisonResult:
    """Compares two versions of a document and returns the differences."""
    logger.info("Comparing document versions...")

    doc1_data = await extract_paragraphs(doc1)
    doc2_data = await extract_paragraphs(doc2)

    changes = diff_paragraphs(doc1_data, doc2_data)

    modified_paras = [c for c in changes if c["type"] == "modified" and c["old"] and c["new"]]

    llm_result = []

    if modified_paras:
        llm_model = service_container.azure_openai_model

        for pair in modified_paras:
            context = {
                "document_a_text": pair["old"],
                "document_b_text": pair["new"],
            }

            response = await llm_model.generate(prompt=comparision_prompt, context=context, response_model=DocumentComparisonResult, mode="JSON")
            llm_result.append(response)

    final_result = {
        "added": [c["new_text"] for c in changes if c["type"] == "added" and c["new_text"]],
        "removed": [c["text"] for c in changes if c["type"] == "removed" and c["text"]],
        "modified": llm_result,
    }

    return final_result
