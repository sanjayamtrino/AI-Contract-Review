"""
Agent: Custom Playbook Review
Purpose: Review documents against custom playbooks to identify missing clauses and deviations
"""

from typing import Dict, Any

# System prompt for the Custom Playbook Review agent
CUSTOM_PLAYBOOK_REVIEW_SYSTEM_PROMPT = """
You are an expert Contract Review AI specializing in playbook compliance analysis.
Your role is to compare documents against organizational playbooks and identify gaps, deviations, and compliance issues.

## Core Capabilities

1. **Clause-by-Clause Review**
   - Compare each clause against playbook standards
   - Identify missing required clauses
   - Flag non-compliant language
   - Assess acceptability of deviations

2. **Missing Clause Detection**
   - Identify clauses required by playbook but absent in document
   - Categorize missing clauses by criticality
   - Suggest standard language from playbook
   - Note conditional requirements

3. **Playbook Comparison**
   - Side-by-side comparison of document vs. playbook
   - Highlight acceptable vs. unacceptable deviations
   - Identify fallback position usage
   - Assess negotiation position strength

4. **Compliance Scoring**
   - Calculate overall compliance percentage
   - Rate individual clause compliance
   - Identify high-risk non-compliance areas
   - Provide compliance summary

## Review Guidelines

### Thoroughness
- Review every clause against playbook requirements
- Check for implicit requirements and dependencies
- Consider conditional and optional clauses
- Verify cross-references within the document

### Objectivity
- Apply playbook standards consistently
- Do not make assumptions beyond playbook provisions
- Distinguish between mandatory and preferred positions
- Acknowledge when playbook is silent on specific issues

### Clarity
- Clearly state what is missing or non-compliant
- Provide specific playbook references
- Quote relevant playbook language
- Explain why a deviation matters

### Practicality
- Prioritize issues by business impact
- Distinguish between deal-breakers and preferences
- Consider negotiation context
- Suggest practical alternatives

## Response Format

Structure your review as follows:

1. **Executive Summary**: Overall compliance score and key findings
2. **Compliance Matrix**: Detailed clause-by-clause comparison
3. **Missing Clauses**: List of required but absent provisions
4. **Deviations**: Non-compliant language with playbook references
5. **Recommendations**: Suggested actions and language

## Constraints

- Base all findings strictly on the provided playbook
- Do not invent playbook requirements
- Distinguish between mandatory and fallback positions
- Do not provide legal advice beyond playbook compliance
"""

# Prompt templates for different sub-intents
CLAUSE_REVIEW_PROMPT = """
Review the following document clauses against the organizational playbook.

Target Clauses: {target_clauses}
Review Depth: {review_depth}

Document Content:
{document_content}

Playbook Content:
{playbook_content}

For each clause reviewed:
1. Clause Location (section reference)
2. Current Document Language (summary)
3. Playbook Standard (reference and language)
4. Compliance Status (compliant / minor deviation / major deviation / missing)
5. Deviation Details (if applicable)
6. Recommended Action
7. Risk Level (low / medium / high)

Provide a summary table and detailed findings.
"""

MISSING_CLAUSES_PROMPT = """
Identify all missing clauses by comparing the document against the playbook requirements.

Document Content:
{document_content}

Playbook Content:
{playbook_content}

Document Type: {document_type}

For each missing clause:
1. Clause Name/Type
2. Playbook Reference (section/page)
3. Criticality Level (critical / important / optional)
4. Standard Language (from playbook)
5. Business Impact
6. Suggested Action

Categorize by:
- Must-have (deal-breaker if missing)
- Should-have (important but negotiable)
- Nice-to-have (preferred but not critical)

Provide a prioritized list with rationale.
"""

PLAYBOOK_COMPARISON_PROMPT = """
Perform a comprehensive comparison between the document and the organizational playbook.

Document Content:
{document_content}

Playbook Content:
{playbook_content}

Comparison Mode: {comparison_mode}

Provide:
1. Overall Compliance Score (percentage)
2. Compliance by Section/Category
3. Side-by-Side Comparison Table
   - Clause/Provision
   - Document Language
   - Playbook Standard
   - Match Status
4. Acceptable Deviations (with justification)
5. Unacceptable Deviations (requiring correction)
6. Missing Required Clauses
7. Strengths (where document exceeds playbook)
8. Recommendations for Improvement
"""

COMPLIANCE_CHECK_PROMPT = """
Conduct a compliance check of the document against playbook requirements.

Document Content:
{document_content}

Playbook Content:
{playbook_content}

Focus Areas: {focus_areas}

Provide:
1. Compliance Summary
   - Overall Score
   - Number of compliant clauses
   - Number of deviations
   - Number of missing clauses

2. Detailed Findings
   - By section/category
   - With specific references
   - Risk assessment for each issue

3. Priority Issues
   - High-risk non-compliance items
   - Deal-breaker issues
   - Must-fix before execution

4. Compliance Certificate
   - Statement of compliance level
   - Qualifications and exceptions
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
    playbook_content = parameters.get("playbook_content", "")
    
    if sub_intent == "clause_review":
        return CLAUSE_REVIEW_PROMPT.format(
            target_clauses=parameters.get("target_clauses", "all clauses"),
            review_depth=parameters.get("review_depth", "detailed"),
            document_content=document_content,
            playbook_content=playbook_content
        )
    
    elif sub_intent == "missing_clauses":
        return MISSING_CLAUSES_PROMPT.format(
            document_content=document_content,
            playbook_content=playbook_content,
            document_type=parameters.get("document_type", "contract")
        )
    
    elif sub_intent == "playbook_comparison":
        return PLAYBOOK_COMPARISON_PROMPT.format(
            document_content=document_content,
            playbook_content=playbook_content,
            comparison_mode=parameters.get("comparison_mode", "comprehensive")
        )
    
    elif sub_intent == "compliance_check":
        return COMPLIANCE_CHECK_PROMPT.format(
            document_content=document_content,
            playbook_content=playbook_content,
            focus_areas=parameters.get("focus_areas", "all areas")
        )
    
    else:
        # Default to playbook comparison
        return PLAYBOOK_COMPARISON_PROMPT.format(
            document_content=document_content,
            playbook_content=playbook_content,
            comparison_mode="comprehensive"
        )
