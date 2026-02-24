import json
from typing import Any, Callable, Dict, List

from src.agents import doc_information, playbook_review
from src.dependencies import get_service_container
from src.schemas.errors import AgentError, ErrorType, OrchestratorResponse
from src.services.prompts.v1 import load_prompt

# Agent registry — maps agent names to their run() functions
AGENT_REGISTRY: Dict[str, Callable] = {
    "doc_information_agent": doc_information.run,
    "playbook_review_agent": playbook_review.run,
}

# Agent definitions for OpenAI function calling
# The orchestrator sees agents as "tools" and routes to them
AGENT_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "doc_information_agent",
            "description": (
                "Handles all document information requests — summaries, key details, "
                "parties, dates, contract values, and general document understanding. "
                "Route here when the user asks anything about the content of their uploaded document."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's query, rewritten as a clear instruction.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "playbook_review_agent",
            "description": (
                "Reviews the uploaded contract against organizational playbook rules. "
                "Handles compliance checks, risk assessments, clause-by-clause reviews, "
                "and deviation analysis. Route here when the user asks to review the "
                "contract against playbook, check compliance, assess risk, or compare "
                "clauses against standard positions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's query, rewritten as a clear instruction.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


async def run_orchestrator(query: str, session_id: str) -> OrchestratorResponse:
    """Route the user query to the correct agent via the orchestrator LLM."""

    container = get_service_container()
    client = container.azure_openai_model

    # Session lookup
    session = container.session_manager.get_session(session_id)
    if session is None:
        return OrchestratorResponse(
            error=AgentError(
                error_type=ErrorType.SESSION_NOT_FOUND,
                message=f"Session '{session_id}' not found or expired",
                recoverable=False,
            )
        )

    orchestrator_prompt = load_prompt("orchestrator_prompt")

    # Build messages with conversation history
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": orchestrator_prompt},
    ]
    messages.extend(session.conversation_history[-10:])  # Last 5 turns (10 messages)
    messages.append(
        {
            "role": "user",
            "content": f"[The user has an uploaded legal document in session '{session_id}']\n\nUser query: {query}",
        }
    )

    # Step 1: Ask the LLM which agent to route to (retry-wrapped)
    try:
        response = await client.chat_completion(
            messages=messages,
            tools=AGENT_DEFINITIONS,
            tool_choice="auto",
            temperature=0.3,
        )
    except Exception as e:
        return OrchestratorResponse(
            error=AgentError(
                error_type=ErrorType.LLM_FAILURE,
                message=f"Orchestrator LLM call failed: {str(e)}",
                recoverable=True,
            )
        )

    assistant_message = response.choices[0].message

    # No tool calls — clarification or out-of-scope response
    if not assistant_message.tool_calls:
        content = assistant_message.content or ""
        session.add_turn("user", query)
        session.add_turn("assistant", content)
        return OrchestratorResponse(response={"message": content})

    # Step 2: Execute the routed agent
    for tool_call in assistant_message.tool_calls:
        agent_name = tool_call.function.name

        if agent_name not in AGENT_REGISTRY:
            return OrchestratorResponse(
                error=AgentError(
                    error_type=ErrorType.UNKNOWN_AGENT,
                    message=f"Unknown agent: {agent_name}",
                    recoverable=False,
                )
            )

        try:
            args = json.loads(tool_call.function.arguments)
            agent_query = args.get("query", query)

            agent_data = await AGENT_REGISTRY[agent_name](
                query=agent_query, session_id=session_id
            )
        except Exception as e:
            return OrchestratorResponse(
                error=AgentError(
                    error_type=ErrorType.INTERNAL_ERROR,
                    message=f"Agent '{agent_name}' failed: {str(e)}",
                    recoverable=True,
                )
            )

    # Successful result (agent_data is an AgentResponse Pydantic model)
    session.add_turn("user", query)
    session.add_turn("assistant", f"[Routed to {agent_name}]")
    return OrchestratorResponse(
        agent=agent_data.agent,
        tools_called=agent_data.tools_called,
        response=agent_data.tool_results if agent_data.tool_results else agent_data.response,
    )
