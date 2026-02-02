from abc import ABC, abstractmethod
from typing import Any, Dict

from pydantic import BaseModel


class BaseLLMModel(ABC):
    """Base interface for all LLM model implementations."""

    @abstractmethod
    async def generate(self, prompt: str, context: Dict[str, Any], response_model: BaseModel) -> str:
        """Generate a response from the model."""
        pass
