from typing import List, Optional

from src.config.logging import get_logger
from src.schemas.playbook_review import RuleCheckRequest
from src.services.clause_extractor import ClauseUnit, extract_all_clauses
from src.services.session_manager import SessionData

logger = get_logger(__name__)


async def find_clause_by_title(session: SessionData, rule_title: str) -> Optional[ClauseUnit]:
    """
    Find a clause in the session by its exact title.
    Handles case-insensitive matching and whitespace normalization.
    """
    clauses = extract_all_clauses(session)

    # Normalize the rule title for comparison
    normalized_rule_title = rule_title.strip().lower()

    for clause in clauses:
        if clause.heading:
            # Normalize and compare
            normalized_heading = clause.heading.strip().lower()
            # Remove trailing periods for matching
            normalized_heading = normalized_heading.rstrip(".")
            normalized_rule_title_clean = normalized_rule_title.rstrip(".")

            if normalized_heading == normalized_rule_title_clean:
                return clause

    logger.warning(f"No clause found matching title: {rule_title}")
    return None


async def validate_clause_against_rule(session: SessionData, rule_title: str, rule_description: str, rule_instruction: str) -> dict:
    """
    Extract paragraph by exact clause title and validate against rule.
    """
    # Find the clause matching the rule title
    clause = await find_clause_by_title(session, rule_title)

    if not clause:
        return {"status": "not_found", "error": f"No clause found with title: {rule_title}", "paragraph": None}

    # Now send clause content with rule description and instruction to LLM
    validation_input = {
        "clause_title": rule_title,
        "clause_content": clause.content,  # The paragraph
        "rule_description": rule_description,
        "rule_instruction": rule_instruction,
    }

    # Pass to your LLM validation logic here
    # ...

    return validation_input
