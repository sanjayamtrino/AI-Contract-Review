"""
Agent: Compare Previous Version
Purpose: Compare documents with previous versions to identify changes
"""

from typing import Dict, Any

# System prompt for the Compare Previous Version agent
COMPARE_PREVIOUS_VERSION_SYSTEM_PROMPT = """
You are an expert Document Comparison AI specializing in version control and change analysis.
Your role is to compare document versions and provide detailed, accurate analysis of all changes.

## Core Capabilities

1. **Version Diff Analysis**
   - Line-by-line comparison
   - Word-level change detection
   - Structural change identification
   - Formatting change tracking

2. **Change Summary**
   - High-level change overview
   - Categorized change list
   - Impact assessment
   - Significance ranking

3. **Modification Tracking**
   - Addition tracking
   - Deletion tracking
   - Modification tracking
   - Movement tracking (relocated content)

4. **Redline Review**
   - Traditional redline format
   - Track changes simulation
   - Clean and marked versions
   - Change acceptance workflow

## Comparison Guidelines

### Accuracy
- Identify every change, no matter how small
- Preserve context around changes
- Distinguish between substantive and cosmetic changes
- Note when changes affect meaning

### Clarity
- Present changes in readable format
- Use standard comparison notation
- Group related changes
- Provide clear before/after views

### Context
- Explain why changes matter
- Assess business/legal impact
- Note dependencies between changes
- Highlight critical modifications

### Completeness
- Review entire document
- Check all sections
- Verify cross-reference updates
- Confirm consistency of changes

## Response Format

Structure your comparison as follows:

1. **Executive Summary**: Overview of changes
2. **Change Statistics**: Quantified changes
3. **Detailed Comparison**: Section-by-section analysis
4. **Critical Changes**: High-impact modifications
5. **Recommendations**: Suggested actions

## Constraints

- Do not judge changes as good or bad
- Focus on factual comparison
- Note when context is needed to assess impact
- Flag potential issues for human review
"""

# Prompt templates for different sub-intents
VERSION_DIFF_PROMPT = """
Compare the following two document versions and identify all differences.

Current Version:
{current_version}

Previous Version:
{previous_version}

Comparison Mode: {comparison_mode}
Focus Areas: {focus_areas}

Provide:

1. **Change Summary**
   - Total number of changes
   - Additions (word/character count)
   - Deletions (word/character count)
   - Modifications (count)
   - Net change

2. **Detailed Diff**
   Section by section comparison showing:
   - Section reference
   - Previous text
   - Current text
   - Change type (added/deleted/modified)
   - Change significance

3. **Change Categories**
   - Substantive changes (affecting rights/obligations)
   - Administrative changes (dates, references)
   - Formatting changes
   - Language improvements
   - Error corrections

4. **Structural Changes**
   - Section additions/deletions
   - Reordering
   - Heading changes
   - Cross-reference updates

5. **Redline View**
   - Traditional redline format
   - Deletions marked with strikethrough
   - Additions marked with underline/highlight
   - Margin notes for changes

6. **Side-by-Side View**
   - Parallel comparison
   - Aligned sections
   - Visual diff indicators
"""

CHANGE_SUMMARY_PROMPT = """
Provide a comprehensive summary of changes between the two document versions.

Current Version:
{current_version}

Previous Version:
{previous_version}

Summary Focus: {summary_focus}
Audience: {audience}

Provide:

1. **Executive Overview**
   - Nature of changes
   - Overall impact
   - Key themes
   - Strategic significance

2. **Major Changes**
   - Top 10 most significant changes
   - Business impact of each
   - Legal implications
   - Action required

3. **Change Categories**
   - By section/topic
   - By type (substantive vs. administrative)
   - By party affected
   - By risk level

4. **Addition Highlights**
   - New provisions added
   - Expanded protections
   - New obligations
   - New procedures

5. **Deletion Highlights**
   - Removed provisions
   - Reduced protections
   - Eliminated obligations
   - Streamlined content

6. **Modification Highlights**
   - Changed terms
   - Adjusted thresholds
   - Modified procedures
   - Updated references

7. **Impact Assessment**
   - Business impact
   - Legal impact
   - Operational impact
   - Risk impact

8. **Recommendations**
   - Changes requiring approval
   - Changes needing clarification
   - Potential issues
   - Next steps
"""

MODIFICATION_TRACKING_PROMPT = """
Track all modifications between document versions with detailed attribution and analysis.

Current Version:
{current_version}

Previous Version:
{previous_version}

Tracking Focus: {tracking_focus}

Provide:

1. **Modification Log**
   Sequential list of all modifications:
   - Modification ID
   - Location (section/page)
   - Type (addition/deletion/modification/movement)
   - Previous text (summary)
   - Current text (summary)
   - Word count impact
   - Significance rating

2. **Addition Tracking**
   - All new content
   - Location of additions
   - Context of additions
   - Relationship to existing content

3. **Deletion Tracking**
   - All removed content
   - Location of deletions
   - Impact of removal
   - Whether deletion is clean or replaced

4. **Modification Tracking**
   - Changed content
   - Before/after comparison
   - Degree of change (minor/substantial)
   - Intent of change (if discernible)

5. **Movement Tracking**
   - Relocated content
   - Original location
   - New location
   - Any changes during relocation

6. **Change Patterns**
   - Recurring modification types
   - Concentrated change areas
   - Systematic changes
   - Thematic modifications

7. **Tracking Report**
   - Summary statistics
   - Trend analysis
   - Anomaly identification
   - Completeness verification
"""

REDLINE_REVIEW_PROMPT = """
Generate a comprehensive redline review of the document changes.

Current Version:
{current_version}

Previous Version:
{previous_version}

Redline Format: {redline_format}
Review Scope: {review_scope}

Provide:

1. **Redline Document**
   Traditional legal redline showing:
   - Deletions: ~~deleted text~~
   - Additions: __added text__ or highlighted
   - Margin notes indicating change type
   - Change numbering for reference

2. **Clean Version**
   - Current version without markup
   - For comparison purposes
   - Professional formatting

3. **Previous Version Reference**
   - Original text for reference
   - Clear section breaks
   - Easy navigation

4. **Change List**
   - Sequential list of all changes
   - Change numbers matching redline
   - Brief description of each
   - Section reference

5. **Review Notes**
   - Changes requiring attention
   - Potential issues
   - Inconsistencies
   - Questions for review

6. **Acceptance Tracking**
   - Space for approval marks
   - Reviewer annotations
   - Status indicators
   - Date/initial fields
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
    current_version = parameters.get("current_version", "")
    previous_version = parameters.get("previous_version", "")
    
    if sub_intent == "version_diff":
        return VERSION_DIFF_PROMPT.format(
            current_version=current_version,
            previous_version=previous_version,
            comparison_mode=parameters.get("comparison_mode", "detailed"),
            focus_areas=parameters.get("focus_areas", "all")
        )
    
    elif sub_intent == "change_summary":
        return CHANGE_SUMMARY_PROMPT.format(
            current_version=current_version,
            previous_version=previous_version,
            summary_focus=parameters.get("summary_focus", "comprehensive"),
            audience=parameters.get("audience", "legal team")
        )
    
    elif sub_intent == "modification_tracking":
        return MODIFICATION_TRACKING_PROMPT.format(
            current_version=current_version,
            previous_version=previous_version,
            tracking_focus=parameters.get("tracking_focus", "all modifications")
        )
    
    elif sub_intent == "redline_review":
        return REDLINE_REVIEW_PROMPT.format(
            current_version=current_version,
            previous_version=previous_version,
            redline_format=parameters.get("redline_format", "traditional"),
            review_scope=parameters.get("review_scope", "full document")
        )
    
    else:
        # Default to version diff
        return VERSION_DIFF_PROMPT.format(
            current_version=current_version,
            previous_version=previous_version,
            comparison_mode="detailed",
            focus_areas="all"
        )
