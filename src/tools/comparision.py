from enum import Enum
from pathlib import Path
from typing import List, Optional

from docx.document import Document
from pydantic import BaseModel, Field

from src.config.logging import get_logger
from src.config.settings import get_settings
from src.dependencies import get_service_container

logger = get_logger(__name__)
settings = get_settings()
service_container = get_service_container()
comparision_prompt = Path(r"src\services\prompts\v1\comparision_prompt.mustache").read_text()


class ChangeType(Enum):
    ADDED = "added"
    MODIFIED = "modified"
    REMOVED = "removed"


class RiskImpact(Enum):
    INCREASED = "increased"
    DECREASED = "decreased"
    NEUTRAL = "neutral"


class ClauseChange(BaseModel):
    clause_title: Optional[str] = Field(None, description="Clause name (e.g., 'Termination', 'Confidentiality Duration')")
    change_type: ChangeType = Field(..., description="Type of change: added, modified, removed")
    original_text: Optional[str] = Field(None, description="Verbatim from Doc A (null only if 'added')")
    revised_text: Optional[str] = Field(None, description="Verbatim from Doc B (null only if 'removed')")
    change_summary: str = Field(..., description="(1) What changed, (2) legal/commercial impact, (3) which party it favors")
    risk_impact: RiskImpact = Field(..., description="Assessment of the risk impact: increased, decreased, neutral")
    significance: str = Field(..., description="high, medium, or low significance")


# class DocumentComparisonResult(BaseModel):
#     summary: str = Field(..., description="High level summary of all changes")
#     changes: List[ClauseChange] = Field(..., description="List of specific changes identified in the document comparison")


class MissingClauses(BaseModel):
    """List of important clauses that are present in one document but missing in the other, along with potential risks associated with their absence."""

    clause_title: str = Field(..., description="Name of the missing clause (e.g., 'Indemnification', 'Limitation of Liability')")
    missing_in: str = Field(..., description="Indicates which document is missing the clause: 'Doc A' or 'Doc B'")
    clause_text: str = Field(..., description="Verbatim text of the missing clause from the document where it is present")
    impact_summary: str = Field(..., description="Summary of the potential legal and commercial risks associated with the absence of this clause in the other document")
    significance: str = Field(..., description="high, medium, or low significance based on the potential risks")


class DocumentComparisonResult(BaseModel):
    """Result of comparing two versions of a document, including a summary and detailed changes."""

    # executive_summary: str = Field(..., description="3-5 sentences — total changes found, the 2-3 most impactful, overall risk direction and which party it favors")
    # overall_risk_impact: str = Field(..., description="Net effect of all changes combined")
    changes: List[ClauseChange] = Field(..., description="List of specific changes identified in the document comparison")
    missing_clauses: List[MissingClauses] = Field(
        ..., description="List of important clauses that are present in one document but missing in the other, along with potential risks associated with their absence"
    )


async def extract_data_from_doc(doc: Document) -> str:
    """Extracts data from a document and returns it as a string."""
    logger.info("Extracting data from document...")
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])


async def compare_doc_versions(doc1: Document, doc2: Document) -> DocumentComparisonResult:
    """Compares two versions of a document and returns the differences."""
    logger.info("Comparing document versions...")

    doc1_data = await extract_data_from_doc(doc1)
    doc2_data = await extract_data_from_doc(doc2)

    context = {
        # "doc1_text": doc1_data,
        # "doc2_text": doc2_data,
        "document_a_text": doc1_data,
        "document_b_text": doc2_data,
    }

    # Get the LLM model
    llm_model = service_container.azure_openai_model

    # Call the LLM to compare the documents
    response = await llm_model.generate(prompt=comparision_prompt, context=context, response_model=DocumentComparisonResult, mode="JSON")
    return response
