from typing import Any, Dict

from src.config.logging import Logger
from src.config.settings import get_settings
from src.services.llm.base_model import BaseLLMModel


class GeminiModel(BaseLLMModel, Logger):
    """Gemini LLM model for generating responsess."""

    def __init__(self) -> None:
        """Initialize the gemini model."""

        super().__init__()
        self.settings = get_settings()
        self.client = None

    async def generate(self, prompt: str, context: Dict[str, Any]) -> str:
        return ""
