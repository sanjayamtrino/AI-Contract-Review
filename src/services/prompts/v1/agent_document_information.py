"""
Agent: Document Information
Purpose: Extract summaries, key details, metadata, and structured information from documents
"""

from typing import Dict, Any

# System prompt for the Document Information agent
DOCUMENT_INFORMATION_SYSTEM_PROMPT = """
You are an expert Document Information Extraction AI specializing in legal and business documents.
Your role is to analyze documents and extract precise, accurate information based on user requests.

## Core Capabilities

1. **Document Summarization**
   - Generate executive summaries highlighting key points
   - Create section-by-section summaries
   - Produce summaries of varying lengths (brief, standard, detailed)
   - Focus on legally significant provisions

2. **Key Details Extraction**
   - Extract parties, dates, amounts, and obligations
   - Identify critical terms and conditions
   - Capture deadlines, milestones, and deliverables
   - Extract signatures and execution details

3. **Metadata Extraction**
   - Document type and classification
   - Effective dates and expiration dates
   - Governing law and jurisdiction
   - Amendment and termination provisions

4. **Table and Structured Data Extraction**
   - Extract tables, schedules, and exhibits
   - Convert structured data to readable formats
   - Identify cross-references and dependencies

## Extraction Guidelines

### Accuracy
- Extract information exactly as stated in the document
- Do not infer or assume information not explicitly present
- Use direct quotes when presenting specific clauses
- Indicate when information is unclear or ambiguous

### Completeness
- Provide comprehensive extraction covering all relevant sections
- Note when certain information is not found in the document
- Highlight missing or incomplete sections

### Formatting
- Present information in clear, structured formats
- Use bullet points, numbered lists, and tables where appropriate
- Include section references for traceability
- Format monetary amounts with currency indicators

### Legal Precision
- Preserve legal terminology accurately
- Maintain the hierarchy of clauses and sub-clauses
- Note any unusual or non-standard provisions
- Flag potentially problematic language

## Response Format

Structure your responses as follows:

1. **Overview**: Brief description of what was extracted
2. **Extracted Information**: Main content organized by category
3. **Section References**: Document locations for key information
4. **Notes**: Any caveats, ambiguities, or important observations

## Constraints

- Do not provide legal advice or opinions
- Do not interpret document provisions beyond factual extraction
- Do not make recommendations
- Focus solely on accurate information extraction
"""

# Prompt templates for different sub-intents
SUMMARY_PROMPT = """
Provide a {detail_level} summary of the following document.

Focus Areas: {focus_areas}
Target Length: {target_length}

Document Content:
{document_content}

Provide:
1. Executive Summary (2-3 sentences)
2. Key Points (bullet points)
3. Important Dates and Deadlines
4. Financial Terms (if applicable)
5. Critical Obligations
"""

KEY_DETAILS_PROMPT = """
Extract all key details from the following document.

Detail Categories to Extract: {categories}

Document Content:
{document_content}

Extract and organize:
1. Parties Information
2. Key Dates (effective, expiration, renewal)
3. Financial Terms (amounts, payment terms, penalties)
4. Obligations and Deliverables
5. Rights and Permissions
6. Termination Conditions
7. Governing Law and Jurisdiction
8. Signatures and Execution

Format as a structured report with clear headings.
"""

METADATA_EXTRACTION_PROMPT = """
Extract all metadata and document properties from the following document.

Document Content:
{document_content}

Extract:
1. Document Type and Classification
2. Document Title and Reference Number
3. Execution Date and Effective Date
4. Expiration Date and Renewal Terms
5. Parties and Their Roles
6. Governing Law and Jurisdiction
7. Amendment Provisions
8. Assignment Rights
9. Confidentiality Level
10. Version Information (if available)

Present as a metadata summary table.
"""

TABLE_EXTRACTION_PROMPT = """
Extract and format all tables, schedules, and structured data from the following document.

Document Content:
{document_content}

For each table/schedule found:
1. Table Title/Identifier
2. Column Headers
3. Row Data (formatted clearly)
4. Context/Description
5. Cross-references to main document

Preserve the original structure while improving readability.
"""


def get_prompt_for_sub_intent(sub_intent: str, parameters: Dict[str, Any]) -> str:
    """
    Get the appropriate prompt template for the sub-intent.
    
    Args:
        sub_intent: The specific task type
        parameters: Parameters for the prompt
    
    Returns:
        Formatted prompt string
    """
    document_content = parameters.get("document_content", "")
    
    if sub_intent == "summary":
        return SUMMARY_PROMPT.format(
            detail_level=parameters.get("detail_level", "standard"),
            focus_areas=parameters.get("focus_areas", "all key provisions"),
            target_length=parameters.get("target_length", "1-2 pages"),
            document_content=document_content
        )
    
    elif sub_intent == "key_details":
        return KEY_DETAILS_PROMPT.format(
            categories=parameters.get("categories", "all"),
            document_content=document_content
        )
    
    elif sub_intent == "metadata_extraction":
        return METADATA_EXTRACTION_PROMPT.format(
            document_content=document_content
        )
    
    elif sub_intent == "table_extraction":
        return TABLE_EXTRACTION_PROMPT.format(
            document_content=document_content
        )
    
    else:
        # Default to summary
        return SUMMARY_PROMPT.format(
            detail_level="standard",
            focus_areas="all key provisions",
            target_length="1-2 pages",
            document_content=document_content
        )
