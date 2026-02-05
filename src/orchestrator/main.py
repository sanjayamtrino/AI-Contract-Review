from agent_framework import (
    BaseChatClient,
    ChatMessage,
    ChatResponse,
    ChatResponseUpdate,
)

from src.services.llm.azure_openai_model import AzureOpenAIModel


class OpenAIChat(BaseChatClient):
    """Custom OpenAI Chat Client."""

    client = AzureOpenAIModel()

    async def _inner_get_response(self, *, messages, chat_options, **kwargs):
        """The main function to return the response."""

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

        # Call the llm model
        response = self.client.client.responses.create(
            model=self.client.deployment_name,
            input=messages_list,
            temperature=0.7,
        )

        # Extract the assitant text
        output = ""
        for message in response.output:
            if message.role == "assistant":
                for content_item in message.content:
                    output += content_item.text

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
    instructions="You are an legal officer who knows everything in and out about the legal stuff.",
)


async def main():
    response = await agent.run("Tell me what NDA I should agree if the legal doc is about corporate.")
    print(response.messages[0].contents[0].text)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
