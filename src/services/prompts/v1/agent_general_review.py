"""
Agent: General Review
Purpose: Provide comprehensive general document review and analysis
"""

from typing import Dict, Any

# System prompt for the General Review agent
GENERAL_REVIEW_SYSTEM_PROMPT = """
You are an expert Document Review AI specializing in comprehensive legal document analysis.
Your role is to provide thorough, objective reviews of documents identifying issues, inconsistencies, and areas for improvement.

## Core Capabilities

1. **Full Document Review**
   - Complete end-to-end document analysis
   - Structural and organizational assessment
   - Content completeness verification
   - Logical flow and coherence check

2. **Section Review**
   - Detailed analysis of specific sections
   - Cross-section consistency check
   - Section-specific issue identification
   - Integration with overall document

3. **Language Review**
   - Grammar and syntax check
   - Clarity and readability assessment
   - Legal terminology accuracy
   - Ambiguity and vagueness detection
   - Redundancy identification

4. **Formatting Review**
   - Consistency of formatting
   - Numbering and cross-references
   - Heading hierarchy
   - Visual presentation

## Review Guidelines

### Comprehensiveness
- Review all aspects of the document
- Check for internal consistency
- Verify cross-references
- Assess document completeness

### Objectivity
- Provide balanced, unbiased assessment
- Support findings with specific examples
- Distinguish between issues and preferences
- Consider document context and purpose

### Constructiveness
- Identify problems clearly
- Suggest specific improvements
- Provide alternative language where helpful
- Prioritize issues by importance

### Professionalism
- Use appropriate professional tone
- Focus on document quality
- Avoid personal opinions
- Maintain confidentiality

## Response Format

Structure your review as follows:

1. **Executive Summary**: Overall assessment and key findings
2. **Detailed Findings**: Issues organized by category
3. **Specific Recommendations**: Actionable suggestions
4. **Priority Matrix**: Issues ranked by severity and effort

## Constraints

- Do not provide legal advice
- Do not rewrite entire documents
- Focus on review and analysis, not drafting
- Maintain professional objectivity
"""

# Prompt templates for different sub-intents
FULL_REVIEW_PROMPT = """
Conduct a comprehensive full review of the following document.

Review Focus: {review_focus}
Document Type: {document_type}

Document Content:
{document_content}

Provide a complete analysis covering:

1. **Executive Summary**
   - Overall quality rating
   - Key strengths
   - Primary concerns
   - Readiness assessment

2. **Structural Review**
   - Organization and flow
   - Section completeness
   - Logical progression
   - Cross-reference accuracy

3. **Content Review**
   - Completeness of provisions
   - Consistency of terms
   - Accuracy of information
   - Appropriateness of language

4. **Language Review**
   - Clarity and precision
   - Grammar and syntax
   - Legal terminology
   - Ambiguities and vagueness

5. **Formatting Review**
   - Visual consistency
   - Numbering systems
   - Heading hierarchy
   - Professional presentation

6. **Issues and Recommendations**
   - Critical issues (must fix)
   - Important issues (should fix)
   - Minor issues (could fix)
   - Suggested improvements

7. **Overall Assessment**
   - Document readiness
   - Risk level
   - Recommended next steps
"""

SECTION_REVIEW_PROMPT = """
Review the following specific section(s) of the document in detail.

Target Section(s): {target_sections}
Review Depth: {review_depth}

Document Content:
{document_content}

For each section reviewed:

1. **Section Overview**
   - Location and context
   - Purpose and scope
   - Relationship to other sections

2. **Content Analysis**
   - Completeness of provisions
   - Accuracy of information
   - Appropriateness for purpose

3. **Language Quality**
   - Clarity and precision
   - Legal terminology
   - Potential ambiguities
   - Suggested improvements

4. **Integration Check**
   - Consistency with other sections
   - Cross-reference accuracy
   - Terminology alignment
   - Logical flow

5. **Issues and Recommendations**
   - Specific problems identified
   - Suggested corrections
   - Alternative language options
   - Priority level

6. **Section Rating**
   - Quality score
   - Risk assessment
   - Recommended actions
"""

LANGUAGE_REVIEW_PROMPT = """
Conduct a detailed language review of the following document.

Review Scope: {review_scope}
Focus Areas: {focus_areas}

Document Content:
{document_content}

Analyze and report on:

1. **Clarity and Readability**
   - Sentence complexity
   - Paragraph structure
   - Overall readability score
   - Areas of confusion

2. **Grammar and Syntax**
   - Grammatical errors
   - Sentence structure issues
   - Punctuation problems
   - Style inconsistencies

3. **Legal Terminology**
   - Correct usage of terms
   - Consistency of terminology
   - Inappropriate language
   - Undefined terms

4. **Precision and Ambiguity**
   - Vague language
   - Ambiguous provisions
   - Unclear references
   - Missing definitions

5. **Redundancy and Conciseness**
   - Repetitive language
   - Unnecessary words
   - Overlapping provisions
   - Opportunities for consolidation

6. **Tone and Style**
   - Professional tone
   - Consistent voice
   - Appropriate formality
   - Audience appropriateness

7. **Recommendations**
   - Priority corrections
   - Suggested rewrites
   - Style improvements
   - Best practice alignment
"""

FORMATTING_REVIEW_PROMPT = """
Review the formatting and presentation of the following document.

Document Content:
{document_content}

Assess:

1. **Visual Consistency**
   - Font consistency
   - Spacing uniformity
   - Alignment consistency
   - Style application

2. **Numbering Systems**
   - Section numbering
   - Paragraph numbering
   - Page numbering
   - List formatting

3. **Heading Hierarchy**
   - Logical structure
   - Consistent levels
   - Clear distinction
   - Table of contents alignment

4. **Cross-References**
   - Accuracy of references
   - Consistent format
   - Working links (if electronic)
   - Clarity of references

5. **Lists and Tables**
   - Consistent formatting
   - Proper alignment
   - Clear presentation
   - Professional appearance

6. **White Space**
   - Appropriate margins
   - Paragraph spacing
   - Section breaks
   - Visual balance

7. **Formatting Recommendations**
   - Specific improvements
   - Professional standards
   - Industry best practices
   - Priority fixes
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
    
    if sub_intent == "full_review":
        return FULL_REVIEW_PROMPT.format(
            review_focus=parameters.get("review_focus", "comprehensive"),
            document_type=parameters.get("document_type", "legal document"),
            document_content=document_content
        )
    
    elif sub_intent == "section_review":
        return SECTION_REVIEW_PROMPT.format(
            target_sections=parameters.get("target_sections", "all sections"),
            review_depth=parameters.get("review_depth", "detailed"),
            document_content=document_content
        )
    
    elif sub_intent == "language_review":
        return LANGUAGE_REVIEW_PROMPT.format(
            review_scope=parameters.get("review_scope", "entire document"),
            focus_areas=parameters.get("focus_areas", "all language aspects"),
            document_content=document_content
        )
    
    elif sub_intent == "formatting_review":
        return FORMATTING_REVIEW_PROMPT.format(
            document_content=document_content
        )
    
    else:
        # Default to full review
        return FULL_REVIEW_PROMPT.format(
            review_focus="comprehensive",
            document_type="legal document",
            document_content=document_content
        )
