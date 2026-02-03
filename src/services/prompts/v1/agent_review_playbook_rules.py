"""
Agent: Review Playbook & Rules
Purpose: Review and analyze playbooks, rules, policies, and guidelines
"""

from typing import Dict, Any

# System prompt for the Review Playbook & Rules agent
REVIEW_PLAYBOOK_RULES_SYSTEM_PROMPT = """
You are an expert Playbook and Rules Review AI specializing in policy analysis and guideline assessment.
Your role is to review organizational playbooks, rules, policies, and guidelines for completeness, consistency, and effectiveness.

## Core Capabilities

1. **Playbook Analysis**
   - Comprehensive playbook review
   - Structure and organization assessment
   - Coverage completeness check
   - Clarity and usability evaluation

2. **Rule Validation**
   - Rule consistency check
   - Conflict identification
   - Gap analysis
   - Redundancy detection

3. **Policy Review**
   - Policy completeness assessment
   - Alignment with objectives
   - Implementation feasibility
   - Enforcement mechanism review

4. **Guideline Check**
   - Guideline clarity assessment
   - Practical applicability
   - Consistency across guidelines
   - User-friendliness evaluation

## Review Guidelines

### Comprehensiveness
- Review all sections and provisions
- Check for internal consistency
- Verify coverage of key areas
- Assess logical flow

### Objectivity
- Evaluate based on best practices
- Identify strengths and weaknesses
- Provide balanced assessment
- Support findings with evidence

### Constructiveness
- Identify specific issues
- Suggest practical improvements
- Prioritize recommendations
- Consider implementation effort

### Practicality
- Assess real-world applicability
- Consider user perspective
- Evaluate enforcement feasibility
- Balance thoroughness with usability

## Response Format

Structure your review as follows:

1. **Executive Summary**: Overall assessment
2. **Structure Review**: Organization and flow
3. **Content Analysis**: Coverage and completeness
4. **Consistency Check**: Internal alignment
5. **Recommendations**: Prioritized improvements

## Constraints

- Do not rewrite entire playbooks
- Focus on review and analysis
- Consider organizational context
- Maintain professional objectivity
"""

# Prompt templates for different sub-intents
PLAYBOOK_ANALYSIS_PROMPT = """
Conduct a comprehensive analysis of the following playbook.

Playbook Content:
{playbook_content}

Playbook Type: {playbook_type}
Analysis Focus: {analysis_focus}

Provide:

1. **Executive Summary**
   - Playbook purpose and scope
   - Overall quality assessment
   - Key strengths
   - Primary concerns
   - Readiness for use

2. **Structure Assessment**
   - Organization and layout
   - Section hierarchy
   - Navigation ease
   - Logical flow
   - Table of contents adequacy

3. **Coverage Analysis**
   - Topics covered
   - Coverage gaps
   - Depth of coverage
   - Breadth of coverage
   - Comparison to best practices

4. **Content Quality**
   - Clarity of provisions
   - Specificity of guidance
   - Actionability
   - Examples and illustrations
   - Edge case coverage

5. **Clause Library Assessment**
   - Standard clauses included
   - Fallback positions
   - Alternative language
   - Clause organization
   - Searchability

6. **Decision Framework**
   - Decision trees included
   - Escalation procedures
   - Approval workflows
   - Exception handling
   - Guidance clarity

7. **Usability Review**
   - User-friendliness
   - Search functionality
   - Quick reference features
   - Visual aids
   - Accessibility

8. **Maintenance Considerations**
   - Update procedures
   - Version control
   - Change tracking
   - Review cycles
   - Ownership

9. **Recommendations**
   - Critical improvements
   - Important enhancements
   - Nice-to-have additions
   - Implementation priority
"""

RULE_VALIDATION_PROMPT = """
Validate the rules in the following playbook/policy for consistency, completeness, and conflicts.

Playbook/Policy Content:
{playbook_content}

Validation Scope: {validation_scope}
Rule Categories: {rule_categories}

Provide:

1. **Rule Inventory**
   - Complete list of rules
   - Categorized by type
   - Priority/severity levels
   - Cross-references

2. **Consistency Analysis**
   - Internal consistency check
   - Cross-rule alignment
   - Terminology consistency
   - Application consistency

3. **Conflict Detection**
   - Direct conflicts identified
   - Indirect conflicts
   - Priority conflicts
   - Exception conflicts
   - Recommended resolutions

4. **Gap Analysis**
   - Missing rules
   - Incomplete rules
   - Ambiguous rules
   - Unclear boundaries
   - Coverage gaps

5. **Redundancy Check**
   - Duplicate rules
   - Overlapping provisions
   - Unnecessary repetition
   - Consolidation opportunities

6. **Rule Quality Assessment**
   - Clarity of each rule
   - Measurability
   - Enforceability
   - Exceptions handling
   - Practicality

7. **Hierarchy Review**
   - Rule precedence
   - Override relationships
   - Exception hierarchy
   - Default rules
   - Special cases

8. **Validation Report**
   - Summary of issues
   - Priority ranking
   - Suggested fixes
   - Implementation guidance
"""

POLICY_REVIEW_PROMPT = """
Review the following policy for completeness, clarity, and effectiveness.

Policy Content:
{policy_content}

Policy Type: {policy_type}
Review Focus: {review_focus}

Provide:

1. **Policy Overview**
   - Policy purpose and objectives
   - Scope and applicability
   - Target audience
   - Effective date and version

2. **Structure Review**
   - Standard policy structure
   - Section completeness
   - Logical organization
   - Navigation and readability

3. **Content Completeness**
   - Required elements present:
     * Policy statement
     * Scope and applicability
     * Definitions
     * Policy provisions
     * Roles and responsibilities
     * Compliance requirements
     * Enforcement provisions
     * Exceptions
     * Related policies
     * Review and revision

4. **Clarity Assessment**
   - Language clarity
   - Jargon and technical terms
   - Ambiguity check
   - Consistency of terms
   - Readability score

5. **Implementation Review**
   - Implementation guidance
   - Responsibilities assigned
   - Resources required
   - Timeline specified
   - Training needs

6. **Enforcement Mechanisms**
   - Compliance monitoring
   - Violation consequences
   - Reporting procedures
   - Appeal processes
   - Remediation steps

7. **Alignment Check**
   - Organizational alignment
   - Regulatory compliance
   - Industry standards
   - Best practices
   - Consistency with other policies

8. **Effectiveness Evaluation**
   - Measurable objectives
   - Success indicators
   - Monitoring mechanisms
   - Review frequency
   - Feedback loops

9. **Recommendations**
   - Critical additions
   - Important clarifications
   - Structural improvements
   - Enhancement opportunities
"""

GUIDELINE_CHECK_PROMPT = """
Review the following guidelines for clarity, practicality, and consistency.

Guideline Content:
{guideline_content}

Guideline Type: {guideline_type}
Target Audience: {target_audience}

Provide:

1. **Guideline Overview**
   - Purpose and objectives
   - Scope and applicability
   - Intended users
   - Relationship to policies/procedures

2. **Clarity Review**
   - Language simplicity
   - Instruction clarity
   - Step-by-step guidance
   - Examples provided
   - Visual aids

3. **Practicality Assessment**
   - Real-world applicability
   - Resource requirements
   - Time considerations
   - Skill level needed
   - Common scenarios covered

4. **Completeness Check**
   - All steps included
   - Prerequisites stated
   - Expected outcomes defined
   - Troubleshooting included
   - FAQs addressed

5. **Consistency Review**
   - Internal consistency
   - Alignment with policies
   - Terminology consistency
   - Format consistency
   - Cross-reference accuracy

6. **User Experience**
   - Ease of use
   - Searchability
   - Quick reference value
   - Format and layout
   - Accessibility

7. **Scenarios Coverage**
   - Common situations
   - Edge cases
   - Exception handling
   - Alternative approaches
   - Decision guidance

8. **Maintenance Review**
   - Update frequency
   - Version control
   - Feedback mechanism
   - Improvement process
   - Ownership

9. **Improvement Recommendations**
   - Priority enhancements
   - Clarity improvements
   - Practical additions
   - Format suggestions
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
    content = parameters.get("content", parameters.get("playbook_content", parameters.get("policy_content", parameters.get("guideline_content", ""))))
    
    if sub_intent == "playbook_analysis":
        return PLAYBOOK_ANALYSIS_PROMPT.format(
            playbook_content=content,
            playbook_type=parameters.get("playbook_type", "general"),
            analysis_focus=parameters.get("analysis_focus", "comprehensive")
        )
    
    elif sub_intent == "rule_validation":
        return RULE_VALIDATION_PROMPT.format(
            playbook_content=content,
            validation_scope=parameters.get("validation_scope", "comprehensive"),
            rule_categories=parameters.get("rule_categories", "all")
        )
    
    elif sub_intent == "policy_review":
        return POLICY_REVIEW_PROMPT.format(
            policy_content=content,
            policy_type=parameters.get("policy_type", "general"),
            review_focus=parameters.get("review_focus", "comprehensive")
        )
    
    elif sub_intent == "guideline_check":
        return GUIDELINE_CHECK_PROMPT.format(
            guideline_content=content,
            guideline_type=parameters.get("guideline_type", "general"),
            target_audience=parameters.get("target_audience", "general users")
        )
    
    else:
        # Default to playbook analysis
        return PLAYBOOK_ANALYSIS_PROMPT.format(
            playbook_content=content,
            playbook_type="general",
            analysis_focus="comprehensive"
        )
