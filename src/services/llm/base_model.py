from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseLLMModel(ABC):
    """Base interface for all LLM model implementations."""

    @abstractmethod
    async def generate(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate a response from the model."""
        pass
