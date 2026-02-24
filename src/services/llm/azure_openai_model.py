import json
import logging
from typing import Any, Dict, List, Optional, Type

import openai
import pystache
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError
from pydantic import ValidationError
from tenacity import before_sleep_log, retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config.logging import Logger
from src.config.settings import get_settings
from src.services.llm.base_model import BaseLLMModel

RETRYABLE_ERRORS = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)

_retry_logger = logging.getLogger("AI_Contract.AzureOpenAI")


class AzureOpenAIModel(BaseLLMModel, Logger):
    """Azure Open AI class for generating responses."""

    def __init__(self) -> None:
        """Initialize the agent."""

        super().__init__()
        self.settings = get_settings()

        if not self.settings.azure_openai_api_key:
            raise ValueError("Azure OpenAI API key is not configured.")

        if not self.settings.azure_openai_responses_deployment_name:
            raise ValueError("Azure OpenAI deployment name is not configured.")

        if not self.settings.base_url:
            raise ValueError("Azure Base URL cannot be empty.")

        self.client = OpenAI(
            base_url=self.settings.base_url,
            api_key=self.settings.azure_openai_api_key,
        )

        self.deployment_name = self.settings.azure_openai_responses_deployment_name

    def render_prompt_template(self, prompt: str, context: Dict[str, Any]) -> Any:
        """Mustache prompt template render function.

        Uses a custom Renderer with HTML escaping disabled since these
        templates produce LLM prompts, not HTML.  The default pystache
        behavior HTML-escapes {{var}} values (e.g. " -> &quot;),
        which corrupts contract text sent to the model.
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
    ) -> Any:
        """Main function to generate response."""

        if self.deployment_name is None:
            raise ValueError("Deployment name is not configured.")

        prompt = self.render_prompt_template(prompt=prompt, context=context)

        try:
            json_schema = response_model.model_json_schema()

            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
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

            # Extract and parse the JSON response
            response_text = response.choices[0].message.content
            if response_text is None:
                raise ValueError("Empty response from LLM model.")
            response_data = json.loads(response_text)

            # Validate and convert to Pydantic model
            validated_response = response_model.model_validate(response_data)
            return validated_response

        except json.JSONDecodeError as e:
            self.logger.error(f"Response parsing failed: {str(e)}.")
            raise ValueError(f"LLM returned invalid JSON: {str(e)}") from e
        except ValidationError as e:
            self.logger.error(f"Response parsing failed: {str(e)}.")
            raise ValueError(f"LLM response failed schema validation: {str(e)}") from e
        except ValueError:
            raise
        except Exception as e:
            self.logger.error(f"LLM generation failed: {str(e)}")
            raise ValueError(f"LLM generation failed: {str(e)}") from e

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
        """Send a chat completion request with optional tool calling.

        This wraps self.client.chat.completions.create() with retry logic
        for transient errors. Permanent errors (bad request, auth) are
        raised immediately.
        """

        return self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice if tools else openai.NOT_GIVEN,
        )
