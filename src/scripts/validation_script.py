import asyncio
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config.settings import get_settings
from src.services.llm.gemini_model import GeminiModel
from src.services.orchestrator.classifier import IntentClassifier
from src.services.orchestrator.models import OrchestratorContext

async def test_llm():
    print("--- Testing GeminiModel ---")
    model = GeminiModel()
    
    print("1. Testing simple generation...")
    try:
        response = await model.generate("Hello, say 'Test Successful' if you can hear me.")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Simple generation failed: {e}")

    print("\n2. Testing with kwargs (Orchestrator style)...")
    try:
        response = await model.generate(
            prompt="Summarize the contract",
            system_prompt="You are a classifier. Output JSON.",
            temperature=0.1,
            max_tokens=100
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Kwargs generation failed: {e}")

async def test_classifier_flow():
    print("\n--- Testing IntentClassifier Flow ---")
    try:
        model = GeminiModel()
        classifier = IntentClassifier(model)
        
        context = OrchestratorContext(
            session_id="test_session",
            active_documents=[],
            conversation_history=[]
        )
        
        query = "Summarize the key terms of this document."
        print(f"Classifying query: '{query}'")
        
        result = await classifier.classify(query, context)
        print(f"Classification Result: {result}")
        
    except Exception as e:
        print(f"Classifier flow failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(test_llm())
        loop.run_until_complete(test_classifier_flow())
    finally:
        loop.close()
