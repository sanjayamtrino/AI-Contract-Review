import json
from typing import Any, Dict

import pystache
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel, ValidationError

from src.config.logging import Logger
from src.config.settings import get_settings
from src.services.llm.base_model import BaseLLMModel


class AzureOpenAIModel(BaseLLMModel, Logger):
    """Azure Open AI class for generating responses."""

    def __init__(self) -> None:
        """Initialize the agent."""

        super().__init__()
        self.settings = get_settings()
        self.azure_endpoint = self.settings.azure_endpoint_uri
        self.azure_api_key = self.settings.azure_openai_api_key
        self.deployment_name = self.settings.azure_deployment_name

        if self.settings.openai_api_key is not None:
            self.client = OpenAI(
                api_key=self.settings.openai_api_key,
                # base_url=self.azure_endpoint,
                # azure_endpoint=self.azure_endpoint,
                # api_version=self.settings.azure_api_version,
            )
        else:
            raise ValueError("Cannot Intialize the Azure model.")

    def render_prompt_template(self, prompt: str, context: Dict[str, Any]) -> Any:
        """Mustache prompt template render function."""

        return pystache.render(template=prompt, context=context)

    async def generate(self, prompt: str, context: Dict[str, Any], response_model: BaseModel) -> Any:
        """Main function to generate response."""

        if self.deployment_name is None:
            raise ValueError("Deployment name is not configured.")

        prompt = self.render_prompt_template(prompt=prompt, context=context)

        try:

            response = self.client.responses.parse(
                model=self.deployment_name,
                input=[
                    {"role": "system", "content": "Extract the event information."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                text_format=response_model,
            )

            response_data = response.output_parsed
            # validated_response = response_model.model_validate(response_data)
            # return validated_response
            return response_data
        except (json.JSONDecodeError, ValidationError) as e:
            self.logger.error(f"Response parsing failed: {str(e)}.")
            raise ValueError("Cannot perform the query rewriting.") from e
        except Exception as e:
            self.logger.error(f"LLM generation failed: {str(e)}")
            raise ValueError("Cannot perform the query rewriting.") from e
