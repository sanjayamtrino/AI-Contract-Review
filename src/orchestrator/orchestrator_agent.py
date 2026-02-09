import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity.aio import AzureCliCredential

from src.tools.summarizer import get_key_information, get_location


async def main():
    async with AzureCliCredential() as credential:
        agent = AzureOpenAIResponsesClient(
            credential=credential,
        ).create_agent(
            name="AI Contract Review",
            instructions="You are an AI Assistant",
            tools=[get_location, get_key_information],
        )

        print(await agent.run("What is the location?"))


if __name__ == "__main__":
    asyncio.run(main())
