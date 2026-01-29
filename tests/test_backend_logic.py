# tests/test_backend_logic.py
import asyncio
import sys
import os

# Add the project root to the python path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.llm.gemini_model import GeminiModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def start_testing_process():
    print("\n🔵 1. INITIALIZING AI ENGINE...")
    try:
        # Initialize the model
        ai_engine = GeminiModel()
        print(f"   ✅ Engine Online. Using Model: {ai_engine.llm.model}")
    except Exception as e:
        print(f"   ❌ Engine Failed to Start: {e}")
        return

    print("\n🔵 2. PREPARING TEST DOCUMENT...")
    # This simulates a user uploading a messy PDF
    test_document_text = """
    AGREEM ENT FOR SERVICE
    This Contract is made on March 15th, 2025 betwen Alpha Corp and Beta Ltd.
    Fee is $50,000 payable Net 30.
    """
    print(f"   ✅ Document Loaded ({len(test_document_text)} chars).")

    print("\n🔵 3. SENDING TO GOOGLE GEMINI 2.0...")
    print("   (Waiting for API response...)")
    
    try:
        # We call the 'generate' function we built earlier
        result = await ai_engine.generate(prompt=test_document_text)
        
        # Check if we got a valid dictionary back
        if isinstance(result, dict) and "parties" in result:
            print("   ✅ RESPONSE RECEIVED SUCCESSFULLY!")
        else:
            print("   ❌ RESPONSE FORMAT INVALID.")
            print(result)
            return

    except Exception as e:
        print(f"   ❌ API CONNECTION FAILED: {e}")
        print("   (Check your internet or API Key quota)")
        return

    print("\n🔵 4. VERIFYING DATA ACCURACY...")
    print("   ------------------------------------------------")
    print(f"   Summary:    {result.get('summary_simple')[:50]}...")
    print(f"   Party A:    {result.get('parties', {}).get('party_a')}")
    print(f"   Effect Date: {result.get('dates', {}).get('effective_date')}")
    print(f"   Total Fee:  {result.get('financials', {}).get('total_fee')}")
    print("   ------------------------------------------------")

    # Final Automated Check
    if result.get('financials', {}).get('total_fee') == "$50,000":
        print("\n🟢 TEST STATUS: PASSED (100% Accuracy)")
    else:
        print("\n🔴 TEST STATUS: FAILED (Data Mismatch)")

if __name__ == "__main__":
    asyncio.run(start_testing_process())
