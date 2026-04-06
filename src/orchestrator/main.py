"""
Custom OpenAI chat client for the agent framework with tool invocation support.

Bridges the agent_framework's BaseChatClient with the Azure OpenAI API,
handling message format conversion, tool execution, and response streaming.
"""

import inspect
import json
from pathlib import Path
from typing import Any, Dict, Optional

from agent_framework import (
    BaseChatClient,
    ChatAgent,
    ChatMessage,
    ChatResponse,
    ChatResponseUpdate,
    ToolMode,
)
from agent_framework._tools import FUNCTION_INVOKING_CHAT_CLIENT_MARKER

from src.dependencies import get_service_container, initialize_dependencies
from src.services.llm.azure_openai_model import AzureOpenAIModel
from src.tools.key_information import get_key_information
from src.tools.summarizer import get_summary


class OpenAIChat(BaseChatClient):
    """Adapts the Azure OpenAI API to the agent_framework's chat client interface."""

    _azure_model: Optional[Any] = None
    ToolMode.AUTO

    def __init__(self):
        super().__init__()
        setattr(self, FUNCTION_INVOKING_CHAT_CLIENT_MARKER, True)

    @property
    def client(self) -> AzureOpenAIModel:
        return get_service_container().azure_openai_model

    async def _inner_get_response(self, *, messages, chat_options, **kwargs):
        """Process messages, invoke tools if requested, and return final response."""

        # Build tool lookup from chat options
        tools: Dict[str, Any] = {}
        if chat_options and chat_options.tools:
            for tool in chat_options.tools:
                tools[tool.name] = tool

        max_iterations = 2
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Convert agent_framework messages to OpenAI format
            messages_list = []
            for msg in messages:
                for content in msg.contents:
                    if content.type == "text":
                        messages_list.append({
                            "role": msg.role.value,
                            "content": content.text,
                        })
                    elif content.type == "function_call":
                        messages_list.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": content.call_id,
                                "type": "function",
                                "function": {
                                    "name": content.name,
                                    "arguments": content.arguments if isinstance(content.arguments, str) else json.dumps(content.arguments),
                                },
                            }],
                        })
                    elif content.type == "function_result":
                        messages_list.append({
                            "role": "tool",
                            "tool_call_id": content.call_id,
                            "content": content.result,
                        })

            # Build OpenAI tool format
            tools_list = []
            if chat_options and chat_options.tools:
                for tool in chat_options.tools:
                    tools_list.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                        },
                    })

            # Call the LLM
            response = self.client.client.chat.completions.create(
                model=self.client.deployment_name,
                messages=messages_list,
                temperature=0.7,
                tools=tools_list if tools_list else None,
                tool_choice="auto" if tools_list else None,
            )

            assistant_message = response.choices[0].message

            # Handle tool calls
            if assistant_message.tool_calls:
                messages.append(
                    ChatMessage(
                        role="assistant",
                        contents=[{
                            "type": "function_call",
                            "call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        } for tool_call in assistant_message.tool_calls],
                    )
                )

                # Execute each tool and append results
                for tool_call in assistant_message.tool_calls:
                    func_name = tool_call.function.name

                    if func_name in tools:
                        try:
                            func = tools[func_name]
                            args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

                            if inspect.iscoroutinefunction(func):
                                result = await func(**args)
                            else:
                                result = func(**args)
                        except Exception as e:
                            raise ValueError("Unable to call the function.") from e
                    else:
                        result = f"Error: tool '{func_name}' not found."

                    messages.append(
                        ChatMessage(
                            role="tool",
                            contents=[{
                                "type": "function_result",
                                "call_id": tool_call.id,
                                "result": result,
                            }],
                        )
                    )
                continue

            # No tool calls — return final text response
            output = assistant_message.content or ""
            return ChatResponse(
                messages=[
                    ChatMessage(
                        role="assistant",
                        contents=[{"type": "text", "text": output}],
                    )
                ]
            )

    async def _inner_get_streaming_response(self, *, messages, chat_options, **kwargs):
        """Placeholder streaming response."""
        yield ChatResponseUpdate(
            role="assistant",
            contents=[{"type": "text", "text": "Hello"}],
        )


# Default agent setup
prompt = Path(r"src\services\prompts\v1\orchestrator_prompt.mustache").read_text()

agent = OpenAIChat().create_agent(
    name="Orchestrator Agent",
    instructions=prompt,
    tools=[get_summary],
)


async def get_azure_agent() -> ChatAgent:
    """Create an orchestrator agent with summary and key info tools."""
    return OpenAIChat().create_agent(
        name="Orchestrator Agent",
        instructions=prompt,
        tools=[get_summary, get_key_information],
    )


async def main():
    await initialize_dependencies()
    response = await agent.run("Summary")
    print(response.messages[0].contents[0].text)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
