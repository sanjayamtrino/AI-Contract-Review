from typing import Any, Dict,Optional
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


from src.config.logging import Logger
from src.config.settings import get_settings
from src.services.llm.base_model import BaseLLMModel

from src.schemas.contract_structure import DocumentReviewOutput
from src.services.prompts.v1.extraction_prompt import (
    SYSTEM_ROLE, 
    EXTRACTION_PROMPT_TEMPLATE, 
    REASONING_INSTRUCTIONS
)


class GeminiModel(BaseLLMModel, Logger):
    """Gemini LLM model for generating responsess."""
    

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",  
            google_api_key=self.settings.gemini_api_key,
            temperature=0.0,       
            convert_system_message_to_human=True
        )

        self.parser = PydanticOutputParser(pydantic_object=DocumentReviewOutput)

        self.logger.info("Gemini Model initialized with Pydantic Parser.")

    async def generate(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Executes the Smart Extraction Pipeline.
        
        Args:
            prompt (str): The raw text from the document (PDF/DOCX).
            context (dict): Optional context (not used heavily here, but good for future).
            
        Returns:
            Dict: A strictly typed dictionary containing Summary, Parties, Dates, etc.
        """
        try:
            document_text = prompt  # The input prompt is actually the document text here
            
            final_prompt_template = PromptTemplate(
                template=EXTRACTION_PROMPT_TEMPLATE.format(
                    system_role=SYSTEM_ROLE,
                    reasoning_instructions=REASONING_INSTRUCTIONS,
                    document_text="{document_text}"
                ) + "\n\n{format_instructions}",
                input_variables=["document_text"],
                partial_variables={"format_instructions": self.parser.get_format_instructions()}
            )

            chain = final_prompt_template | self.llm | self.parser

            self.logger.info("Executing Gemini Extraction Chain...")
            result = await chain.ainvoke({"document_text": document_text})
            return result.dict()

        except Exception as e:
            self.logger.error(f"Gemini Extraction Failed: {str(e)}")
            # Fallback: Return a structured error so the frontend doesn't crash
            return {
                "error": "Extraction Failed",
                "details": str(e),
                "summary_simple": "Error processing document.",
                "parties": {"party_a": "Unknown", "party_b": "Unknown"},
                "dates": {},
                "financials": {}
            }