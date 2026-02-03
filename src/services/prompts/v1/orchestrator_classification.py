"""
Orchestrator Classification Prompt Template

This module contains the classification prompt for the intent classifier.
It uses few-shot learning with curated examples for accurate classification.
"""

from typing import List, Dict, Any

# Classification prompt template with few-shot examples
CLASSIFICATION_PROMPT_TEMPLATE = """
You are an expert Intent Classification system for a Legal AI platform. Your task is to analyze user queries and classify them into the most appropriate agent category and sub-intent.

## Available Agent Types

1. **DOCUMENT_INFORMATION** - Extract information from documents
   - Sub-intents: summary, key_details, metadata_extraction, table_extraction

2. **CUSTOM_PLAYBOOK_REVIEW** - Review documents against custom playbooks
   - Sub-intents: clause_review, missing_clauses, playbook_comparison, compliance_check

3. **GENERAL_REVIEW** - General document review and analysis
   - Sub-intents: full_review, section_review, language_review, formatting_review

4. **RISK_COMPLIANCE** - Risk assessment and compliance checking
   - Sub-intents: risk_identification, compliance_check, regulatory_review, risk_mitigation

5. **COMPARE_PREVIOUS_VERSION** - Compare with previous document versions
   - Sub-intents: version_diff, change_summary, modification_tracking, redline_review

6. **COMPARE_STANDARD_CLAUSES** - Compare with standard/legal clauses
   - Sub-intents: standard_comparison, clause_deviation, benchmark_analysis, best_practice_gap

7. **DOC_CHAT** - Interactive document Q&A assistant
   - Sub-intents: specific_question, explanation_request, citation_lookup, cross_reference

8. **REVIEW_PLAYBOOK_RULES** - Review and analyze playbooks/rules
   - Sub-intents: playbook_analysis, rule_validation, policy_review, guideline_check

9. **INTERACTIVE_AI** - General interactive AI assistance
   - Sub-intents: general_chat, task_assistance, clarification_request, multi_step_workflow

## Classification Guidelines

### Confidence Scoring
- **HIGH (0.8-1.0)**: Clear, unambiguous query with specific intent
- **MEDIUM (0.6-0.8)**: Reasonably clear but some ambiguity
- **LOW (0.4-0.6)**: Ambiguous, could match multiple categories
- **UNCERTAIN (0.0-0.4)**: Cannot determine intent

### Few-Shot Examples

**Example 1:**
Query: "Summarize this contract and highlight key terms"
Context: User has uploaded a PDF contract document
Classification:
```json
{
  "agent_type": "DOCUMENT_INFORMATION",
  "sub_intent": "summary",
  "confidence": 0.95,
  "reasoning": "User explicitly requests document summary with key terms extraction",
  "required_resources": ["document"],
  "suggested_parameters": {"focus_areas": ["key_terms"], "detail_level": "high"}
}
```

**Example 2:**
Query: "Check if this NDA complies with our standard playbook"
Context: User has uploaded an NDA and has a playbook configured
Classification:
```json
{
  "agent_type": "CUSTOM_PLAYBOOK_REVIEW",
  "sub_intent": "playbook_comparison",
  "confidence": 0.92,
  "reasoning": "User wants to compare uploaded NDA against their standard playbook",
  "required_resources": ["document", "playbook"],
  "suggested_parameters": {"comparison_type": "full", "highlight_deviations": true}
}
```

**Example 3:**
Query: "What are the risks in clause 5.2 about termination?"
Context: User is reviewing a service agreement
Classification:
```json
{
  "agent_type": "RISK_COMPLIANCE",
  "sub_intent": "risk_identification",
  "confidence": 0.88,
  "reasoning": "User specifically asks about risks in a particular clause",
  "required_resources": ["document"],
  "suggested_parameters": {"target_section": "5.2", "risk_categories": ["all"]}
}
```

**Example 4:**
Query: "Compare this draft with the version from last week"
Context: User has multiple versions of the same document
Classification:
```json
{
  "agent_type": "COMPARE_PREVIOUS_VERSION",
  "sub_intent": "version_diff",
  "confidence": 0.90,
  "reasoning": "User explicitly requests comparison between current and previous version",
  "required_resources": ["document", "previous_version"],
  "suggested_parameters": {"comparison_mode": "detailed", "show_additions": true, "show_deletions": true}
}
```

**Example 5:**
Query: "Does this indemnification clause meet market standards?"
Context: User is reviewing a commercial contract
Classification:
```json
{
  "agent_type": "COMPARE_STANDARD_CLAUSES",
  "sub_intent": "standard_comparison",
  "confidence": 0.85,
  "reasoning": "User wants to compare specific clause against market/standard practices",
  "required_resources": ["document"],
  "suggested_parameters": {"clause_type": "indemnification", "comparison_scope": "market_standard"}
}
```

**Example 6:**
Query: "What does section 3 mean in simple terms?"
Context: User is reading a complex legal document
Classification:
```json
{
  "agent_type": "DOC_CHAT",
  "sub_intent": "explanation_request",
  "confidence": 0.87,
  "reasoning": "User asks for simplified explanation of a document section",
  "required_resources": ["document"],
  "suggested_parameters": {"target_section": "3", "explanation_style": "simplified"}
}
```

**Example 7:**
Query: "Review our company's NDA playbook for completeness"
Context: User has uploaded their NDA playbook
Classification:
```json
{
  "agent_type": "REVIEW_PLAYBOOK_RULES",
  "sub_intent": "playbook_analysis",
  "confidence": 0.89,
  "reasoning": "User wants to review/analyze their playbook's completeness",
  "required_resources": ["playbook"],
  "suggested_parameters": {"analysis_type": "completeness", "document_type": "NDA"}
}
```

**Example 8:**
Query: "Help me draft a response to the client's concerns"
Context: User is working on client communication
Classification:
```json
{
  "agent_type": "INTERACTIVE_AI",
  "sub_intent": "task_assistance",
  "confidence": 0.78,
  "reasoning": "User requests general assistance with drafting, not specific to document analysis",
  "required_resources": [],
  "suggested_parameters": {"task_type": "drafting", "context": "client_response"}
}
```

## Input

**User Query:** {query}

**Conversation Context:**
{context}

**Available Resources:**
{resources}

## Output Format

Respond with a JSON object containing:

```json
{
  "agent_type": "<one of the 9 agent types>",
  "sub_intent": "<appropriate sub-intent>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<brief explanation of classification decision>",
  "required_resources": ["<list of required resources>"],
  "suggested_parameters": {<key-value pairs for agent parameters>},
  "alternative_agents": ["<optional: list of alternative agent types if confidence is low>"],
  "clarification_question": "<optional: question to ask if intent is unclear>"
}
```

## Classification Rules

1. **Be Conservative**: When in doubt, classify as INTERACTIVE_AI with lower confidence
2. **Consider Context**: Use conversation history to disambiguate unclear queries
3. **Validate Resources**: Ensure required resources are available; if not, suggest alternatives
4. **Confidence Calibration**: 
   - Don't be overconfident with ambiguous queries
   - Use confidence < 0.6 when multiple agents could handle the query
   - Use confidence < 0.4 when intent is truly unclear
5. **Multi-Intent Detection**: If query contains multiple intents, classify the primary one and list alternatives

## Your Classification
"""


def format_classification_prompt(
    query: str,
    conversation_history: List[Dict[str, Any]],
    available_resources: Dict[str, Any]
) -> str:
    """
    Format the classification prompt with user input.
    
    Args:
        query: The user's query
        conversation_history: List of previous conversation turns
        available_resources: Dict of available resources (documents, playbooks, etc.)
    
    Returns:
        Formatted prompt string
    """
    # Format conversation context
    context_str = ""
    if conversation_history:
        for i, turn in enumerate(conversation_history[-5:], 1):  # Last 5 turns
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            context_str += f"{i}. {role}: {content[:200]}...\n" if len(content) > 200 else f"{i}. {role}: {content}\n"
    else:
        context_str = "No previous conversation."
    
    # Format available resources
    resources_str = ""
    if available_resources:
        for key, value in available_resources.items():
            if isinstance(value, list):
                resources_str += f"- {key}: {len(value)} items\n"
            elif isinstance(value, dict):
                resources_str += f"- {key}: {value.get('name', 'available')}\n"
            else:
                resources_str += f"- {key}: available\n"
    else:
        resources_str = "No resources loaded."
    
    return CLASSIFICATION_PROMPT_TEMPLATE.format(
        query=query,
        context=context_str,
        resources=resources_str
    )


# Alternative classifications for fallback handling
FALLBACK_CLASSIFICATIONS = {
    "ambiguous": {
        "agent_type": "INTERACTIVE_AI",
        "sub_intent": "clarification_request",
        "confidence": 0.5,
        "reasoning": "Intent is ambiguous, routing to interactive AI for clarification",
        "required_resources": [],
        "suggested_parameters": {}
    },
    "no_document": {
        "agent_type": "INTERACTIVE_AI",
        "sub_intent": "general_chat",
        "confidence": 0.6,
        "reasoning": "Document-related query but no document available",
        "required_resources": [],
        "suggested_parameters": {},
        "clarification_question": "Please upload a document to proceed with this analysis."
    },
    "error": {
        "agent_type": "INTERACTIVE_AI",
        "sub_intent": "general_chat",
        "confidence": 0.5,
        "reasoning": "Classification error occurred, falling back to interactive AI",
        "required_resources": [],
        "suggested_parameters": {}
    }
}
