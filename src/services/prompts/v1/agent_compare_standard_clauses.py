"""
Agent: Compare Standard Clauses
Purpose: Compare document clauses with standard/legal clauses and industry benchmarks
"""

from typing import Dict, Any

# System prompt for the Compare Standard Clauses agent
COMPARE_STANDARD_CLAUSES_SYSTEM_PROMPT = """
You are an expert Standard Clause Comparison AI specializing in market practice and industry benchmark analysis.
Your role is to compare document clauses against standard provisions and identify deviations from market norms.

## Core Capabilities

1. **Standard Comparison**
   - Compare against industry-standard clauses
   - Reference standard form documents
   - Identify market practice norms
   - Assess deviation significance

2. **Clause Deviation Analysis**
   - Quantify deviations from standard
   - Categorize deviation types
   - Assess deviation impact
   - Recommend standardization

3. **Benchmark Analysis**
   - Compare against industry benchmarks
   - Reference peer practices
   - Assess competitiveness
   - Identify best-in-class provisions

4. **Best Practice Gap Analysis**
   - Identify gaps vs. best practices
   - Recommend improvements
   - Prioritize enhancements
   - Suggest implementation

## Comparison Guidelines

### Market Awareness
- Reference current market standards
- Consider industry-specific norms
- Account for regional variations
- Acknowledge evolving practices

### Objectivity
- Present factual comparisons
- Avoid value judgments
- Provide context for deviations
- Note when standards conflict

### Practicality
- Focus on meaningful deviations
- Distinguish material from minor
- Consider negotiation context
- Suggest practical alternatives

### Comprehensiveness
- Review all major clause types
- Check standard protections
- Verify market alignment
- Note innovative provisions

## Response Format

Structure your comparison as follows:

1. **Executive Summary**: Overall comparison results
2. **Clause-by-Clause Analysis**: Detailed comparisons
3. **Deviation Summary**: Key differences
4. **Benchmark Assessment**: Market position
5. **Recommendations**: Suggested improvements

## Constraints

- Base comparisons on recognized standards
- Acknowledge when standards vary
- Do not mandate specific language
- Consider context of deviations
- Note when bespoke provisions are appropriate
"""

# Prompt templates for different sub-intents
STANDARD_COMPARISON_PROMPT = """
Compare the following document clauses against standard industry provisions.

Document Content:
{document_content}

Standard Clauses Reference:
{standard_clauses}

Clause Types to Compare: {clause_types}
Industry: {industry}
Jurisdiction: {jurisdiction}

Provide:

1. **Comparison Overview**
   - Standard sources referenced
   - Comparison methodology
   - Overall alignment score
   - Key findings summary

2. **Clause-by-Clause Comparison**
   For each clause type:
   - Clause name/type
   - Document provision (summary)
   - Standard provision (reference)
   - Comparison result:
     * Aligned with standard
     * More favorable to party A
     * More favorable to party B
     * Missing
     * Non-standard
   - Deviation details
   - Market prevalence of deviation

3. **Standard Alignment Matrix**
   - Table showing all comparisons
   - Visual alignment indicators
   - Quick reference format

4. **Deviation Categories**
   - Market standard deviations
   - Industry-specific deviations
   - Jurisdictional variations
   - Bespoke provisions
   - Missing standard protections

5. **Contextual Analysis**
   - Industry context
   - Regional considerations
   - Document-specific factors
   - Negotiation history (if known)

6. **Comparison Notes**
   - Standards referenced
   - Limitations of comparison
   - Areas of uncertainty
   - Recommendations for further review
"""

CLAUSE_DEVIATION_PROMPT = """
Analyze deviations from standard clauses in the following document.

Document Content:
{document_content}

Standard Clauses:
{standard_clauses}

Deviation Focus: {deviation_focus}
Significance Threshold: {significance_threshold}

Provide:

1. **Deviation Inventory**
   List all deviations found:
   - Clause location
   - Standard provision
   - Actual provision
   - Nature of deviation
   - Deviation magnitude
   - Business impact
   - Risk level

2. **Deviation Classification**
   By type:
   - Favorable deviations (beneficial)
   - Unfavorable deviations (concerning)
   - Neutral deviations (cosmetic)
   - Missing provisions
   - Enhanced provisions

   By significance:
   - Critical (deal-breaker)
   - Major (material impact)
   - Minor (limited impact)
   - Cosmetic (no material impact)

3. **Deviation Analysis**
   For significant deviations:
   - Detailed comparison
   - Market context
   - Negotiation implications
   - Risk assessment
   - Recommendation

4. **Deviation Patterns**
   - Recurring deviation types
   - Systematic biases
   - Concentration areas
   - Thematic deviations

5. **Standard Recovery**
   - Suggested standard language
   - Fallback positions
   - Alternative approaches
   - Negotiation strategies

6. **Deviation Report**
   - Executive summary
   - Priority deviations
   - Action items
   - Timeline for resolution
"""

BENCHMARK_ANALYSIS_PROMPT = """
Perform a benchmark analysis comparing the document against industry standards and peer practices.

Document Content:
{document_content}

Benchmark Data:
{benchmark_data}

Benchmark Scope: {benchmark_scope}
Peer Group: {peer_group}

Provide:

1. **Benchmark Overview**
   - Benchmark sources
   - Peer group definition
   - Comparison methodology
   - Overall benchmark score

2. **Quantitative Benchmarks**
   - Financial terms comparison
   - Duration/term benchmarks
   - Threshold comparisons
   - Limit comparisons
   - Penalty/fee benchmarks

3. **Qualitative Benchmarks**
   - Protection level assessment
   - Risk allocation comparison
   - Flexibility assessment
   - Balance evaluation

4. **Peer Comparison**
   - Comparison to peer group
   - Percentile ranking
   - Best-in-class gaps
   - Worst-in-class risks

5. **Market Position**
   - Competitive positioning
   - Market standard alignment
   - Innovation areas
   - Lagging areas

6. **Benchmark Trends**
   - Market evolution
   - Emerging practices
   - Declining standards
   - Regional variations

7. **Benchmark Recommendations**
   - Areas for improvement
   - Competitive adjustments
   - Market alignment needs
   - Innovation opportunities
"""

BEST_PRACTICE_GAP_PROMPT = """
Identify gaps between the document and industry best practices.

Document Content:
{document_content}

Best Practices Reference:
{best_practices}

Gap Analysis Scope: {gap_analysis_scope}
Priority Areas: {priority_areas}

Provide:

1. **Best Practice Framework**
   - Best practice sources
   - Industry standards
   - Leading practices
   - Expert recommendations

2. **Gap Identification**
   For each best practice area:
   - Best practice standard
   - Document provision
   - Gap description
   - Gap severity
   - Business impact
   - Implementation difficulty

3. **Gap Categories**
   - Critical gaps (high impact, easy to fix)
   - Important gaps (high impact, hard to fix)
   - Quick wins (low impact, easy to fix)
   - Low priority (low impact, hard to fix)

4. **Gap Analysis by Clause Type**
   - Indemnification
   - Limitation of liability
   - Termination
   - Confidentiality
   - Intellectual property
   - Dispute resolution
   - Warranties
   - Representations
   - And other key clauses

5. **Best Practice Recommendations**
   - Specific improvements
   - Suggested language
   - Implementation guidance
   - Priority order

6. **Gap Closure Roadmap**
   - Phase 1: Critical gaps
   - Phase 2: Important gaps
   - Phase 3: Enhancement opportunities
   - Timeline and resources

7. **Gap Report**
   - Executive summary
   - Gap inventory
   - Priority actions
   - Resource requirements
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
    
    if sub_intent == "standard_comparison":
        return STANDARD_COMPARISON_PROMPT.format(
            document_content=document_content,
            standard_clauses=parameters.get("standard_clauses", "industry standards"),
            clause_types=parameters.get("clause_types", "all major clauses"),
            industry=parameters.get("industry", "general"),
            jurisdiction=parameters.get("jurisdiction", "general")
        )
    
    elif sub_intent == "clause_deviation":
        return CLAUSE_DEVIATION_PROMPT.format(
            document_content=document_content,
            standard_clauses=parameters.get("standard_clauses", "industry standards"),
            deviation_focus=parameters.get("deviation_focus", "all deviations"),
            significance_threshold=parameters.get("significance_threshold", "medium")
        )
    
    elif sub_intent == "benchmark_analysis":
        return BENCHMARK_ANALYSIS_PROMPT.format(
            document_content=document_content,
            benchmark_data=parameters.get("benchmark_data", "industry benchmarks"),
            benchmark_scope=parameters.get("benchmark_scope", "comprehensive"),
            peer_group=parameters.get("peer_group", "industry peers")
        )
    
    elif sub_intent == "best_practice_gap":
        return BEST_PRACTICE_GAP_PROMPT.format(
            document_content=document_content,
            best_practices=parameters.get("best_practices", "industry best practices"),
            gap_analysis_scope=parameters.get("gap_analysis_scope", "comprehensive"),
            priority_areas=parameters.get("priority_areas", "all areas")
        )
    
    else:
        # Default to standard comparison
        return STANDARD_COMPARISON_PROMPT.format(
            document_content=document_content,
            standard_clauses="industry standards",
            clause_types="all major clauses",
            industry="general",
            jurisdiction="general"
        )
