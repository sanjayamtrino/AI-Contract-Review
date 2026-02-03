"""
Agent: Risk & Compliance
Purpose: Identify risks, check compliance, and provide regulatory review
"""

from typing import Dict, Any

# System prompt for the Risk & Compliance agent
RISK_COMPLIANCE_SYSTEM_PROMPT = """
You are an expert Risk and Compliance AI specializing in legal and regulatory analysis.
Your role is to identify risks, assess compliance with regulations, and provide actionable risk mitigation strategies.

## Core Capabilities

1. **Risk Identification**
   - Legal risks (contractual, litigation, regulatory)
   - Financial risks (payment, penalty, exposure)
   - Operational risks (performance, delivery, execution)
   - Reputational risks (public, brand, relationship)
   - Strategic risks (business, market, competitive)

2. **Compliance Checking**
   - Regulatory compliance assessment
   - Industry standard alignment
   - Internal policy adherence
   - Jurisdiction-specific requirements
   - Cross-border compliance

3. **Regulatory Review**
   - Applicable regulation identification
   - Regulatory requirement mapping
   - Compliance gap analysis
   - Regulatory change impact
   - Enforcement trend awareness

4. **Risk Mitigation**
   - Mitigation strategy development
   - Control recommendation
   - Alternative approach suggestion
   - Risk transfer options
   - Monitoring mechanism design

## Analysis Guidelines

### Risk Assessment Framework
- **Likelihood**: Probability of risk occurrence (rare, unlikely, possible, likely, almost certain)
- **Impact**: Severity of consequences (insignificant, minor, moderate, major, catastrophic)
- **Risk Rating**: Combined score determining priority (low, medium, high, critical)

### Compliance Approach
- Identify all applicable regulations
- Map requirements to document provisions
- Note compliance gaps explicitly
- Consider jurisdictional variations
- Flag emerging regulatory issues

### Mitigation Strategy
- Propose practical, implementable controls
- Consider cost-benefit of mitigation
- Suggest risk transfer mechanisms
- Recommend monitoring approaches
- Prioritize by risk severity

## Response Format

Structure your analysis as follows:

1. **Executive Summary**: Key risks and compliance status
2. **Risk Register**: Detailed risk inventory with ratings
3. **Compliance Matrix**: Regulatory requirement mapping
4. **Gap Analysis**: Compliance shortfalls
5. **Mitigation Plan**: Recommended actions and controls
6. **Monitoring Framework**: Ongoing risk management

## Constraints

- Do not provide definitive legal conclusions
- Identify when specialist advice is needed
- Consider analysis as starting point, not final assessment
- Flag jurisdictional limitations
- Note regulatory uncertainty where applicable
"""

# Prompt templates for different sub-intents
RISK_IDENTIFICATION_PROMPT = """
Identify and assess all risks in the following document.

Risk Categories: {risk_categories}
Document Type: {document_type}
Industry Context: {industry_context}

Document Content:
{document_content}

Provide a comprehensive risk analysis:

1. **Risk Register**
   For each identified risk:
   - Risk ID and Name
   - Category (legal, financial, operational, reputational, strategic)
   - Description
   - Source/Location in document
   - Likelihood (1-5 scale)
   - Impact (1-5 scale)
   - Risk Rating (likelihood × impact)
   - Current Controls (if any)

2. **Risk Summary by Category**
   - Count and distribution
   - Highest rated risks per category
   - Category-specific concerns

3. **Critical Risks**
   - Top 5 highest rated risks
   - Detailed analysis of each
   - Immediate attention required

4. **Risk Interdependencies**
   - Related risks
   - Compound risk scenarios
   - Cascade effects

5. **Risk Trends**
   - Patterns across document
   - Concentration areas
   - Systemic issues
"""

COMPLIANCE_CHECK_PROMPT = """
Conduct a compliance check of the following document against applicable regulations and standards.

Regulatory Framework: {regulatory_framework}
Jurisdiction: {jurisdiction}
Industry: {industry}
Compliance Scope: {compliance_scope}

Document Content:
{document_content}

Applicable Regulations:
{regulations}

Provide:

1. **Compliance Overview**
   - Overall compliance status
   - Number of requirements checked
   - Compliance rate
   - Critical gaps

2. **Regulatory Mapping**
   For each applicable regulation:
   - Regulation name and reference
   - Specific requirements
   - Document provisions addressing them
   - Compliance status (full / partial / non-compliant / not addressed)
   - Gap details

3. **Compliance Gaps**
   - Missing provisions
   - Inadequate provisions
   - Incorrect provisions
   - Jurisdictional issues

4. **Compliance by Domain**
   - Data protection
   - Consumer protection
   - Financial regulations
   - Employment law
   - Environmental regulations
   - Industry-specific requirements

5. **Remediation Recommendations**
   - Priority actions
   - Suggested language
   - Implementation guidance
   - Timeline considerations
"""

REGULATORY_REVIEW_PROMPT = """
Perform a comprehensive regulatory review of the following document.

Jurisdiction: {jurisdiction}
Industry Sector: {industry_sector}
Document Type: {document_type}
Review Scope: {review_scope}

Document Content:
{document_content}

Provide:

1. **Applicable Regulations**
   - Complete list of applicable laws
   - Regulatory authorities
   - Recent regulatory changes
   - Pending regulations

2. **Requirement Analysis**
   - Mandatory requirements
   - Best practice standards
   - Industry guidelines
   - Enforcement trends

3. **Compliance Assessment**
   - Document provisions vs. requirements
   - Compliance gaps
   - Over-compliance areas
   - Regulatory risks

4. **Jurisdictional Considerations**
   - Multi-jurisdictional issues
   - Conflict of laws
   - Choice of law implications
   - Cross-border concerns

5. **Regulatory Trends**
   - Emerging issues
   - Enforcement priorities
   - Industry developments
   - Future considerations

6. **Recommendations**
   - Compliance improvements
   - Risk mitigation
   - Monitoring needs
   - Expert consultation areas
"""

RISK_MITIGATION_PROMPT = """
Develop risk mitigation strategies for the identified risks in the following document.

Risk Focus: {risk_focus}
Mitigation Approach: {mitigation_approach}
Resource Constraints: {resource_constraints}

Document Content:
{document_content}

Identified Risks:
{identified_risks}

Provide:

1. **Mitigation Strategy Overview**
   - Strategic approach
   - Key principles
   - Resource requirements
   - Implementation timeline

2. **Risk-Specific Mitigation**
   For each significant risk:
   - Risk description
   - Mitigation options
   - Recommended approach
   - Control mechanisms
   - Responsible party
   - Timeline
   - Success metrics

3. **Control Framework**
   - Preventive controls
   - Detective controls
   - Corrective controls
   - Control effectiveness assessment

4. **Risk Transfer Options**
   - Insurance possibilities
   - Indemnification strategies
   - Guarantee options
   - Contractual protections

5. **Monitoring and Reporting**
   - Key risk indicators
   - Monitoring frequency
   - Reporting structure
   - Escalation triggers

6. **Implementation Roadmap**
   - Phased approach
   - Quick wins
   - Long-term initiatives
   - Dependencies
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
    
    if sub_intent == "risk_identification":
        return RISK_IDENTIFICATION_PROMPT.format(
            risk_categories=parameters.get("risk_categories", "all"),
            document_type=parameters.get("document_type", "contract"),
            industry_context=parameters.get("industry_context", "general"),
            document_content=document_content
        )
    
    elif sub_intent == "compliance_check":
        return COMPLIANCE_CHECK_PROMPT.format(
            regulatory_framework=parameters.get("regulatory_framework", "general"),
            jurisdiction=parameters.get("jurisdiction", "general"),
            industry=parameters.get("industry", "general"),
            compliance_scope=parameters.get("compliance_scope", "comprehensive"),
            document_content=document_content,
            regulations=parameters.get("regulations", "all applicable")
        )
    
    elif sub_intent == "regulatory_review":
        return REGULATORY_REVIEW_PROMPT.format(
            jurisdiction=parameters.get("jurisdiction", "general"),
            industry_sector=parameters.get("industry_sector", "general"),
            document_type=parameters.get("document_type", "contract"),
            review_scope=parameters.get("review_scope", "comprehensive"),
            document_content=document_content
        )
    
    elif sub_intent == "risk_mitigation":
        return RISK_MITIGATION_PROMPT.format(
            risk_focus=parameters.get("risk_focus", "all significant"),
            mitigation_approach=parameters.get("mitigation_approach", "balanced"),
            resource_constraints=parameters.get("resource_constraints", "standard"),
            document_content=document_content,
            identified_risks=parameters.get("identified_risks", "to be identified")
        )
    
    else:
        # Default to risk identification
        return RISK_IDENTIFICATION_PROMPT.format(
            risk_categories="all",
            document_type="contract",
            industry_context="general",
            document_content=document_content
        )
