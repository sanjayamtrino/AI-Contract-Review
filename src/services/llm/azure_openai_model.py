"""
Azure OpenAI LLM client with retry logic, structured JSON responses,
and Mustache template rendering.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Type

import pystache
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from pydantic import ValidationError
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config.logging import Logger
from src.config.settings import get_settings
from src.exceptions.llm_exceptions import (
    APIKeyNotConfigured,
    BaseURLNotConfigured,
    DeploymentNotConfigured,
    EmptyResponseError,
    LLMModelError,
    ResponseParsingError,
)
from src.services.llm.base_model import BaseLLMModel

RETRYABLE_ERRORS = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)
_retry_logger = logging.getLogger("AI_Contract.AzureOpenAI")


class AzureOpenAIModel(BaseLLMModel, Logger):
    """Azure OpenAI client for structured JSON generation and chat completion."""

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()

        if not self.settings.azure_openai_api_key:
            raise APIKeyNotConfigured("Azure OpenAI API key is not configured.")
        if not self.settings.azure_openai_responses_deployment_name:
            raise DeploymentNotConfigured("Azure OpenAI deployment name is not configured.")
        if not self.settings.base_url:
            raise BaseURLNotConfigured("Azure Base URL is not configured.")

        self.client = OpenAI(
            base_url=self.settings.base_url,
            api_key=self.settings.azure_openai_api_key,
        )
        self.deployment_name = self.settings.azure_openai_responses_deployment_name

    def render_prompt_template(self, prompt: str, context: Dict[str, Any]) -> Any:
        """Render a Mustache template with HTML escaping disabled.

        Escaping is disabled because prompts are sent to the LLM, not rendered as HTML.
        """
        renderer = pystache.Renderer(escape=lambda u: u)
        return renderer.render(prompt, context)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        before_sleep=before_sleep_log(_retry_logger, logging.WARNING),
        reraise=True,
    )
    async def generate(
        self,
        prompt: str,
        context: Dict[str, Any],
        response_model: Type,
        system_message: str = "Extract the information and return valid JSON.",
        temperature: float = 0.2,
    ) -> Any:
        """Generate a structured JSON response validated against a Pydantic model."""
        if self.deployment_name is None:
            raise ValueError("Deployment name is not configured.")

        prompt = self.render_prompt_template(prompt=prompt, context=context)
        self.logger.debug(f"Rendered prompt for LLM: {prompt}")

        try:
            json_schema = response_model.model_json_schema()

            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=16384,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": json_schema,
                        "strict": False,
                    },
                },
            )

            finish_reason = response.choices[0].finish_reason
            usage = response.usage
            self.logger.info(f"LLM finish_reason={finish_reason}, tokens={usage}")
            if finish_reason == "length":
                self.logger.warning("Response truncated due to max_tokens limit!")

            response_text = response.choices[0].message.content
            if response_text is None:
                raise EmptyResponseError("Empty response from LLM model.")

            response_data = json.loads(response_text)
            return response_model.model_validate(response_data)

        except (json.JSONDecodeError, ValidationError) as e:
            self.logger.error(f"Failed to parse LLM response: {str(e)}")
            raise ResponseParsingError("Failed to parse the LLM response.") from e
        except (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError):
            raise  # Let tenacity handle retries
        except Exception as e:
            self.logger.error(f"LLM generation error: {str(e)}")
            raise LLMModelError("An error occurred while generating LLM response.") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        before_sleep=before_sleep_log(_retry_logger, logging.WARNING),
        reraise=True,
    )
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.3,
    ) -> Any:
        """Send a chat completion request with optional tool definitions."""
        kwargs: Dict[str, Any] = {
            "model": self.deployment_name,
            "messages": messages,
            "temperature": temperature,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        try:
            return self.client.chat.completions.create(**kwargs)
        except (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError):
            raise
        except Exception as e:
            self.logger.error(f"Chat completion error: {str(e)}")
            raise LLMModelError("An error occurred during chat completion.") from e
