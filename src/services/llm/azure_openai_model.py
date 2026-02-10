import json
from typing import Any, Dict, Type

import pystache
from openai import OpenAI
from pydantic import ValidationError

from src.config.logging import Logger
from src.config.settings import get_settings
from src.services.llm.base_model import BaseLLMModel


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
            # azure_endpoint=self.settings.base_url,
            # api_version=self.settings.azure_api_version,
        )

        self.deployment_name = self.settings.azure_openai_responses_deployment_name

    def render_prompt_template(self, prompt: str, context: Dict[str, Any]) -> Any:
        """Mustache prompt template render function."""

        return pystache.render(template=prompt, context=context)

    async def generate(self, prompt: str, context: Dict[str, Any], response_model: Type) -> Any:
        """Main function to generate response."""

        if self.deployment_name is None:
            raise ValueError("Deployment name is not configured.")

        prompt = self.render_prompt_template(prompt=prompt, context=context)

        try:
            json_schema = response_model.model_json_schema()

            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "Extract the information and return valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
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

        except (json.JSONDecodeError, ValidationError) as e:
            self.logger.error(f"Response parsing failed: {str(e)}.")
            raise ValueError("Cannot perform the query rewriting.") from e
        except Exception as e:
            self.logger.error(f"LLM generation failed: {str(e)}")
            raise ValueError("Cannot perform the query rewriting.") from e
