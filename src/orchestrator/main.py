import json
from typing import Any, Dict, Optional

from agent_framework import (
    BaseChatClient,
    ChatMessage,
    ChatResponse,
    ChatResponseUpdate,
    ToolMode,
)
from agent_framework._tools import FUNCTION_INVOKING_CHAT_CLIENT_MARKER

from src.dependencies import get_service_container, initialize_dependencies
from src.services.llm.azure_openai_model import AzureOpenAIModel
from src.tools.summarizer import get_key_information, get_location, get_summary


class OpenAIChat(BaseChatClient):
    """Custom OpenAI Chat Client."""

    _azure_model: Optional[Any] = None
    ToolMode.AUTO

    def __init__(self):
        super().__init__()
        setattr(self, FUNCTION_INVOKING_CHAT_CLIENT_MARKER, True)

    @property
    def client(self) -> AzureOpenAIModel:
        """Get the Azure OpenAI model client, initializing if necessary."""
        return get_service_container().azure_openai_model()

    async def _inner_get_response(self, *, messages, chat_options, **kwargs):
        """The main function to return the response."""

        # Store tools in a Dict
        tools: Dict[str, Any] = {}
        if chat_options and chat_options.tools:
            for tool in chat_options.tools:
                tools[tool.name] = tool

        max_iterations = 2
        iteration = 0

        # Tool execution loop
        while iteration < max_iterations:
            iteration += 1

            # Store the message history into a list and pass to LLM
            messages_list = []
            for msg in messages:
                for content in msg.contents:
                    if content.type == "text":
                        messages_list.append(
                            {
                                "role": msg.role.value,
                                "content": content.text,
                            }
                        )
                    elif content.type == "function_call":
                        # Need to handle the function calling here
                        messages_list.append(
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": content.call_id,
                                        "type": "function",
                                        "function": {
                                            "name": content.name,
                                            "arguments": content.arguments if isinstance(content.arguments, str) else json.dumps(content.arguments),
                                        },
                                    }
                                ],
                            }
                        )
                    elif content.type == "function_result":
                        # tool results from function execution
                        result_content = content.result

                        messages_list.append(
                            {
                                "role": "tool",
                                "tool_call_id": content.call_id,
                                "content": result_content,
                            }
                        )

            # Convert tools to OpenAI format
            tools_list = []
            if chat_options and chat_options.tools:
                for tool in chat_options.tools:
                    tools_list.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description or "",
                            },
                        }
                    )

            # Call the llm model with tools
            response = self.client.client.chat.completions.create(
                model=self.client.deployment_name,
                messages=messages_list,
                temperature=0.7,
                tools=tools_list if tools_list else None,
                tool_choice="auto" if tools_list else None,
            )

            # Check if the model wants to call a tool
            assistant_message = response.choices[0].message

            # Check if the model wants to call any tools
            if assistant_message.tool_calls:
                messages.append(
                    ChatMessage(
                        role="assistant",
                        contents=[
                            {
                                "type": "function_call",
                                "call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            }
                            for tool_call in assistant_message.tool_calls
                        ],
                    )
                )

                # Execute tools and add results to messages
                for tool_call in assistant_message.tool_calls:
                    func_name = tool_call.function.name

                    if func_name in tools:
                        try:
                            func = tools[func_name]
                            result = func()
                        except Exception as e:
                            raise ValueError("Unable to call the function.") from e

                    else:
                        print("Tool not found")

                    messages.append(
                        ChatMessage(
                            role="tool",
                            contents=[
                                {
                                    "type": "function_result",
                                    "call_id": tool_call.id,
                                    "result": result,
                                }
                            ],
                        )
                    )

                continue

        # Extract the assistant text
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
        """Function used for streaming the response."""

        yield ChatResponseUpdate(
            role="assistant",
            contents=[
                {
                    "type": "text",
                    "text": "Hello",
                }
            ],
        )


agent = OpenAIChat().create_agent(
    name="Orchestrator Agent",
    instructions="You are an helpfull AI assistant.",
    tools=[get_summary, get_location, get_key_information],
)


async def main():
    # Initialize dependencies before running the agent
    initialize_dependencies()

    response = await agent.run("Summary")
    print(response.messages[0].contents[0].text)
    # print(response.text)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
