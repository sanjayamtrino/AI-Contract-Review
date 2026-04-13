"""
General Review Agent — deterministic dispatcher.

Routing is a simple Python branch, not an LLM call:

  - If the user selected a clause, route to ``clause_review`` (Mode 1).
    That tool runs a relevance gate first and either short-circuits with
    an alert or returns suggestions for the selected clause.

  - If nothing was selected, route to ``full_document_review`` (Mode 2).
    That tool extracts every clause from the session, matches clauses to
    the user's prompt via embedding similarity, and returns suggestions
    for the matched clauses.

Both tools return a fully-formed ``GeneralReviewResponse``, so this
dispatcher is intentionally thin — no response rewriting, no LLM routing,
no orchestrator coupling.
"""

from src.schemas.general_review import GeneralReviewResponse
from src.tools.general_reviewer import clause_review, full_document_review


async def run(
    prompt: str,
    session_id: str,
    selected_clause: str | None = None,
    clause_title: str | None = None,
) -> GeneralReviewResponse:
    """Run the general review for a session.

    Args:
        prompt: The user's review instruction.
        session_id: Active session with an ingested document.
        selected_clause: Text of the user-selected clause. ``None`` or
            blank means Mode 2 (full document review).
        clause_title: Optional heading for the selected clause.
    """
    if selected_clause and selected_clause.strip():
        return await clause_review(
            session_id=session_id,
            clause_text=selected_clause,
            user_prompt=prompt,
            clause_title=(clause_title or "Selected Clause").strip() or "Selected Clause",
        )

    return await full_document_review(
        session_id=session_id,
        user_prompt=prompt,
    )
