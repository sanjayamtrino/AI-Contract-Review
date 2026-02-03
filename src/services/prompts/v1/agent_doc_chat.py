"""
Agent: DocChat (Document Assistant)
Purpose: Interactive Q&A assistant for document queries and explanations
"""

from typing import Dict, Any

# System prompt for the DocChat agent
DOC_CHAT_SYSTEM_PROMPT = """
You are an expert Document Q&A AI Assistant specializing in helping users understand and navigate documents.
Your role is to provide accurate, helpful answers to questions about document content.

## Core Capabilities

1. **Specific Question Answering**
   - Answer precise questions about document content
   - Provide direct citations and references
   - Clarify ambiguous provisions
   - Confirm or correct user understanding

2. **Explanation Requests**
   - Explain complex legal language in simple terms
   - Define technical terms and jargon
   - Provide context for provisions
   - Illustrate concepts with examples

3. **Citation Lookup**
   - Find specific clauses or provisions
   - Locate referenced sections
   - Cross-reference related provisions
   - Verify citations

4. **Cross-Reference Navigation**
   - Help navigate document structure
   - Find related provisions
   - Explain dependencies
   - Map document organization

## Interaction Guidelines

### Accuracy
- Answer based solely on document content
- Cite specific sections when possible
- Distinguish between fact and interpretation
- Acknowledge when information is not in the document

### Clarity
- Use clear, accessible language
- Avoid unnecessary jargon
- Structure answers logically
- Use formatting for readability

### Helpfulness
- Anticipate follow-up questions
- Provide relevant context
- Offer related information
- Guide users to related sections

### Conciseness
- Be direct and to the point
- Avoid unnecessary elaboration
- Focus on the specific question
- Provide appropriate level of detail

## Response Format

Structure your responses as follows:

1. **Direct Answer**: Clear response to the question
2. **Supporting Evidence**: Citations and references
3. **Additional Context**: Helpful related information
4. **Follow-up Suggestions**: Related questions or sections

## Constraints

- Do not provide legal advice
- Do not interpret beyond document content
- Do not make assumptions not supported by text
- Stay within document scope
- Maintain professional tone
"""

# Prompt templates for different sub-intents
SPECIFIC_QUESTION_PROMPT = """
Answer the following specific question based on the document content.

Question: {question}

Document Content:
{document_content}

Conversation History:
{conversation_history}

Provide:

1. **Direct Answer**
   - Clear, concise answer to the question
   - Yes/No if applicable, with explanation
   - Specific information requested

2. **Document Citations**
   - Exact section references
   - Relevant quoted text
   - Page numbers if available
   - Context around the answer

3. **Supporting Details**
   - Related provisions
   - Conditions or qualifications
   - Exceptions or limitations
   - Definitions if relevant

4. **Confidence Assessment**
   - Certainty level
   - Any ambiguities
   - Areas requiring clarification
   - Assumptions made

5. **Related Information**
   - Connected provisions
   - Cross-references
   - Relevant sections
   - Suggested follow-up

If the answer is not in the document, clearly state this and explain what information would be needed.
"""

EXPLANATION_REQUEST_PROMPT = """
Provide a clear explanation of the requested document content.

Explanation Request: {explanation_request}
Target Content: {target_content}
Explanation Style: {explanation_style}
User Background: {user_background}

Document Content:
{document_content}

Provide:

1. **Simple Explanation**
   - Plain language explanation
   - Core concept in everyday terms
   - Avoid legal jargon
   - Use analogies if helpful

2. **Technical Explanation**
   - Precise legal meaning
   - Technical details
   - Legal implications
   - Standard interpretations

3. **Context and Purpose**
   - Why this provision exists
   - What it accomplishes
   - When it applies
   - How it fits in the document

4. **Key Terms Defined**
   - Important defined terms
   - Technical vocabulary
   - Industry-specific language
   - Cross-references to definitions

5. **Practical Examples**
   - Hypothetical scenarios
   - Real-world application
   - Common situations
   - Edge cases

6. **Related Provisions**
   - Connected sections
   - Dependencies
   - Cross-references
   - Supporting provisions

7. **Common Questions**
   - Frequently asked questions about this provision
   - Common misconceptions
   - Important clarifications
   - Practical tips
"""

CITATION_LOOKUP_PROMPT = """
Find and provide the exact citation for the requested content in the document.

Citation Request: {citation_request}
Search Scope: {search_scope}

Document Content:
{document_content}

Provide:

1. **Exact Location**
   - Section number/letter
   - Paragraph number
   - Page number (if available)
   - Heading/title

2. **Full Text**
   - Complete provision text
   - Surrounding context
   - Related subsections
   - Relevant definitions

3. **Citation Format**
   - Standard citation format
   - Multiple format options
   - Copy-paste ready
   - Reference format

4. **Related Citations**
   - Cross-referenced sections
   - Related provisions
   - Definitions section
   - Applicable schedules

5. **Citation Context**
   - How this provision fits
   - Related document structure
   - Navigation guidance
   - Related sections to review

6. **Verification**
   - Confirm this is the correct provision
   - Note any similar provisions
   - Flag potential confusion points
   - Suggest verification steps

If the exact citation cannot be found, provide the closest matches and explain the search performed.
"""

CROSS_REFERENCE_PROMPT = """
Help navigate cross-references and related provisions in the document.

Starting Point: {starting_point}
Navigation Goal: {navigation_goal}

Document Content:
{document_content}

Provide:

1. **Document Structure Overview**
   - Overall organization
   - Major sections
   - Navigation hierarchy
   - Key divisions

2. **Cross-Reference Map**
   - From the starting point:
     * Outgoing references (this section refers to)
     * Incoming references (sections referring here)
     * Related provisions
     * Dependencies

3. **Navigation Path**
   - Step-by-step navigation
   - Recommended reading order
   - Alternative paths
   - Shortcuts

4. **Related Provisions**
   - Topically related sections
   - Functionally connected provisions
   - Supporting provisions
   - Contrasting provisions

5. **Document Dependencies**
   - What this section depends on
   - What depends on this section
   - Circular references
   - Hierarchical relationships

6. **Navigation Tips**
   - How to find related content
   - Search strategies
   - Common navigation patterns
   - Document-specific guidance

7. **Suggested Reading List**
   - Priority sections to review
   - Logical reading order
   - Context-building sections
   - Deep-dive sections
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
    
    if sub_intent == "specific_question":
        return SPECIFIC_QUESTION_PROMPT.format(
            question=parameters.get("question", ""),
            document_content=document_content,
            conversation_history=parameters.get("conversation_history", "No previous context.")
        )
    
    elif sub_intent == "explanation_request":
        return EXPLANATION_REQUEST_PROMPT.format(
            explanation_request=parameters.get("explanation_request", ""),
            target_content=parameters.get("target_content", ""),
            explanation_style=parameters.get("explanation_style", "balanced"),
            user_background=parameters.get("user_background", "general"),
            document_content=document_content
        )
    
    elif sub_intent == "citation_lookup":
        return CITATION_LOOKUP_PROMPT.format(
            citation_request=parameters.get("citation_request", ""),
            search_scope=parameters.get("search_scope", "entire document"),
            document_content=document_content
        )
    
    elif sub_intent == "cross_reference":
        return CROSS_REFERENCE_PROMPT.format(
            starting_point=parameters.get("starting_point", "document beginning"),
            navigation_goal=parameters.get("navigation_goal", "understand document structure"),
            document_content=document_content
        )
    
    else:
        # Default to specific question
        return SPECIFIC_QUESTION_PROMPT.format(
            question=parameters.get("question", "Please provide document information."),
            document_content=document_content,
            conversation_history=parameters.get("conversation_history", "No previous context.")
        )
