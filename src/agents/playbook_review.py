"""
Playbook Review Agent — selects and executes the appropriate playbook tool
based on the user's query using LLM tool-calling.
"""

import json
from typing import Any, Callable, Dict, List

from src.dependencies import get_service_container
from src.schemas.agents import AgentResponse
from src.services.prompts.v1 import load_prompt
from src.tools.playbook_reviewer import full_playbook_review, single_rule_review

# Tool registry and definitions
TOOL_REGISTRY: Dict[str, Callable] = {
    "full_playbook_review": full_playbook_review,
    "single_rule_review": single_rule_review,
}

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "full_playbook_review",
            "description": (
                "Run a complete playbook review against ALL organizational rules. "
                "Returns risk assessment, verdicts, evidence, and suggested fixes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "playbook_name": {
                        "type": "string",
                        "description": "'v3' (NDA rules) or 'default' (general contract rules).",
                        "enum": ["v3", "default"],
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "single_rule_review",
            "description": (
                "Review the contract against a single specific rule "
                "(e.g. 'confidentiality', 'termination', 'liability')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_title": {
                        "type": "string",
                        "description": "The name/title of the rule to review.",
                    },
                    "playbook_name": {
                        "type": "string",
                        "description": "'v3' (NDA rules) or 'default' (general contract rules).",
                        "enum": ["v3", "default"],
                    },
                },
                "required": ["rule_title"],
            },
        },
    },
]


async def run(query: str, session_id: str) -> AgentResponse:
    """Decide which playbook tool to call and execute it."""
    container = get_service_container()
    client = container.azure_openai_model
    agent_prompt = load_prompt("playbook_review_v2_agent_prompt")

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

    # No tool calls — return the LLM's text response
    if not assistant_message.tool_calls:
        return AgentResponse(
            agent="playbook_review",
            response=assistant_message.content or "",
        )

    # Execute tool(s) and collect results
    tools_called: List[str] = []
    tool_results: Dict[str, Any] = {}

    for tool_call in assistant_message.tool_calls:
        func_name = tool_call.function.name

        if func_name in TOOL_REGISTRY:
            try:
                args = json.loads(tool_call.function.arguments)
                args["session_id"] = session_id

                result = await TOOL_REGISTRY[func_name](**args)

                if hasattr(result, "model_dump"):
                    tool_results[func_name] = result.model_dump()
                else:
                    tool_results[func_name] = {"result": str(result)}

                tools_called.append(func_name)

            except Exception as e:
                tool_results[func_name] = {"error": str(e)}

    return AgentResponse(
        agent="playbook_review",
        tools_called=tools_called,
        tool_results=tool_results,
    )
