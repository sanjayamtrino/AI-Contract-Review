import json
from typing import Any, Callable, Dict, List

from src.agents import doc_information
from src.dependencies import get_service_container
from src.services.prompts.v1 import load_prompt

# Agent registry — maps agent names to their run() functions
AGENT_REGISTRY: Dict[str, Callable] = {
    "doc_information_agent": doc_information.run,
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
]


async def run_orchestrator(query: str, session_id: str) -> Dict[str, Any]:
    """Route the user query to the correct agent via the orchestrator LLM."""

    container = get_service_container()
    client = container.azure_openai_model

    orchestrator_prompt = load_prompt("orchestrator_prompt")

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": orchestrator_prompt},
        {"role": "user", "content": query},
    ]

    # Step 1: Ask the LLM which agent to route to
    response = client.client.chat.completions.create(
        model=client.deployment_name,
        messages=messages,
        temperature=0.3,
        tools=AGENT_DEFINITIONS,
        tool_choice="auto",
    )

    assistant_message = response.choices[0].message

    # No agent call — return text (clarification or out-of-scope)
    if not assistant_message.tool_calls:
        return {
            "response": assistant_message.content or "",
        }

    # Step 2: Execute the agent(s) and return raw results directly
    all_results: Dict[str, Any] = {}

    for tool_call in assistant_message.tool_calls:
        agent_name = tool_call.function.name

        if agent_name in AGENT_REGISTRY:
            try:
                args = json.loads(tool_call.function.arguments)
                agent_query = args.get("query", query)

                result = await AGENT_REGISTRY[agent_name](
                    query=agent_query, session_id=session_id
                )
                all_results[agent_name] = result

            except Exception as e:
                all_results[agent_name] = {"error": str(e)}
        else:
            all_results[agent_name] = {"error": f"Unknown agent: {agent_name}"}

    # Flatten: if only one agent was called, return its results directly
    if len(all_results) == 1:
        agent_data = next(iter(all_results.values()))
        return {
            "response": agent_data.get("tool_results", {}),
            "agent": agent_data.get("agent"),
            "tools_called": agent_data.get("tools_called", []),
        }

    return {
        "response": all_results,
    }
