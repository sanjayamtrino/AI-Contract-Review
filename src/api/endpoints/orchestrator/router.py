from agent_framework import ChatAgent
from fastapi import APIRouter, File, Form, UploadFile

from tools.summarizer import summarizer

router = APIRouter(prefix="/orchestrator", tags=["Orchestrator"])

PROMPT = """
You are an AI Contract Review Orchestrator.

Rules:
- If the user asks to summarize, analyze, or review a contract, use the summarizer tool.
- If the question is general legal knowledge, answer directly.
- Base answers strictly on the provided contract.
"""

agent = ChatAgent(chat_client=AzureOpenAIChatClient(), instructions=PROMPT, tools=[summarizer])


@router.post("/analyze")
async def analyze_contract(query: str = Form(...), file: UploadFile = File(...)):
    contract_text = (await file.read()).decode("utf-8", errors="ignore")

    user_input = f"""
Contract:
{contract_text}

User Question:
{query}
"""

    result = await agent.run(user_input)

    return {"query": query, "response": str(result)}
