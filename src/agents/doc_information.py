"""
Document Information Agent — selects and executes tools for
document summary and key details extraction using LLM tool-calling.
"""

import json
from typing import Any, Callable, Dict, List

from src.dependencies import get_service_container
from src.schemas.agents import AgentResponse
from src.services.prompts.v1 import load_prompt
from src.tools.key_details import get_key_details
from src.tools.summarizer import get_summary

# Tool registry and definitions
TOOL_REGISTRY: Dict[str, Callable] = {
    "get_summary": get_summary,
    "get_key_details": get_key_details,
}

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_summary",
            "description": (
                "Generate a comprehensive summary including document type, "
                "parties, key terms, critical dates, and risks."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_key_details",
            "description": (
                "Extract structured key details: parties, dates, "
                "contract value, duration, and payment terms."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


async def run(query: str, session_id: str) -> AgentResponse:
    """Decide which document info tool to call and execute it."""
    container = get_service_container()
    client = container.azure_openai_model
    agent_prompt = load_prompt("doc_information_agent_prompt")

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": agent_prompt},
        {"role": "user", "content": query},
    ]

    # Ask the LLM which tool(s) to call
    response = await client.chat_completion(
        messages=messages,
        tools=TOOL_DEFINITIONS,
        tool_choice="auto",
        temperature=0.3,
    )

    assistant_message = response.choices[0].message

    # No tool calls — return the LLM's clarification text
    if not assistant_message.tool_calls:
        return AgentResponse(
            agent="doc_information",
            response=assistant_message.content or "",
        )

    # Execute tool(s) and collect results
    tools_called: List[str] = []
    tool_results: Dict[str, Any] = {}

    for tool_call in assistant_message.tool_calls:
        func_name = tool_call.function.name

        if func_name in TOOL_REGISTRY:
            try:
                result = await TOOL_REGISTRY[func_name](session_id=session_id)

                if hasattr(result, "model_dump"):
                    tool_results[func_name] = result.model_dump()
                else:
                    tool_results[func_name] = {"result": str(result)}

                tools_called.append(func_name)

            except ValueError as e:
                tool_results[func_name] = {
                    "error": str(e),
                    "error_type": "tool_failure",
                    "recoverable": False,
                }
            except Exception as e:
                tool_results[func_name] = {
                    "error": str(e),
                    "error_type": "internal_error",
                    "recoverable": True,
                }

    return AgentResponse(
        agent="doc_information",
        tools_called=tools_called,
        tool_results=tool_results,
    )
