import json
from typing import Any, Callable, Dict, List

from src.dependencies import get_service_container
from src.services.prompts.v1 import load_prompt
from src.tools.key_details import get_key_details
from src.tools.summarizer import get_summary

# Tools owned by this agent
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
                "Get a comprehensive summary of the legal document including "
                "document type, parties involved, key terms, critical dates, and risks."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_key_details",
            "description": (
                "Extract structured key details from the legal document including "
                "parties, effective/expiration dates, contract value, duration, and payment terms."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


async def run(query: str, session_id: str) -> Dict[str, Any]:
    """Doc Information Agent — decides which tool to use and executes it."""

    container = get_service_container()
    client = container.azure_openai_model

    agent_prompt = load_prompt("doc_information_agent_prompt")

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": agent_prompt},
        {"role": "user", "content": query},
    ]

    # Step 1: Ask the LLM which tool(s) to call
    response = client.client.chat.completions.create(
        model=client.deployment_name,
        messages=messages,
        temperature=0.3,
        tools=TOOL_DEFINITIONS,
        tool_choice="auto",
    )

    assistant_message = response.choices[0].message

    # No tool calls — return the LLM's text response (clarification etc.)
    if not assistant_message.tool_calls:
        return {
            "agent": "doc_information",
            "response": assistant_message.content or "",
            "tools_called": [],
            "tool_results": {},
        }

    # Step 2: Execute the tool(s) and return raw results directly
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

            except Exception as e:
                tool_results[func_name] = {"error": str(e)}

    return {
        "agent": "doc_information",
        "tools_called": tools_called,
        "tool_results": tool_results,
    }
