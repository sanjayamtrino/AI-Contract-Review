from typing import Any, Dict, Optional, Union
import json

from google import genai
from pydantic import ValidationError

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
    """Gemini LLM model for generating responses using native Google GenAI SDK."""

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()
        
        # Initialize Google GenAI Client
        if not self.settings.gemini_api_key:
             raise ValueError("Gemini API Key is not configured.")
             
        self.client = genai.Client(api_key=self.settings.gemini_api_key)
        self.model_name = "gemini-2.0-flash" 

        self.logger.info("Gemini Model initialized with Native Google GenAI SDK.")

    async def generate(
        self, 
        prompt: str, 
        context: Dict[str, Any] = None,
        system_prompt: str = None,
        temperature: float = 0.0,
        max_tokens: int = 8192
    ) -> Union[Dict[str, Any], str]:
        """
        Executes the GenAI request.
        
        This method handles both:
        1. Structured Extraction (when prompt implies document extraction) - returning Dict
        2. General Chat/Agent interaction - returning str
        """
        try:
            # 1. Prepare Config
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }

            # 2. Determine if we are doing structured extraction (legacy behavior) 
            # or general agent interaction
            
            # If system_prompt is passed, it's likely a general agent call
            if system_prompt:
                self.logger.debug(f"Generating content with system prompt: {system_prompt[:50]}...")
                response = await self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        **generation_config,
                        "system_instruction": system_prompt
                    }
                )
                return response.text

            # 3. Else, we assume it's the specific specific Document Extraction task (legacy)
            # We construct the prompt manually as before
            document_text = prompt
            full_prompt = EXTRACTION_PROMPT_TEMPLATE.format(
                    system_role=SYSTEM_ROLE,
                    reasoning_instructions=REASONING_INSTRUCTIONS,
                    document_text=document_text
                )
            
            # Request JSON response for extraction
            generation_config["response_mime_type"] = "application/json"
            
            self.logger.info("Executing Gemini Extraction (Native SDK)...")
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config=generation_config
            )
            
            # Parse JSON result
            try:
                json_result = json.loads(response.text)
                
                # Validate with Pydantic
                validated_result = DocumentReviewOutput(**json_result)
                return validated_result.dict()
                
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON Decode Error: {e}")
                raise ValueError("LLM returned invalid JSON")
            except ValidationError as e:
                self.logger.error(f"Pydantic Validation Error: {e}")
                # Return the raw JSON even if validation fails slightly, or raise
                return json_result

        except Exception as e:
            self.logger.error(f"Gemini Generation Failed: {str(e)}")
            
            # Fallback for extraction
            if not system_prompt:
                 return {
                    "error": "Extraction Failed",
                    "details": str(e),
                    "summary_simple": "Error processing document.",
                    "parties": {"party_a": "Unknown", "party_b": "Unknown"},
                    "dates": {},
                    "financials": {}
                }
            
            # Fallback for chat
            return f"Error interacting with Gemini: {str(e)}"
