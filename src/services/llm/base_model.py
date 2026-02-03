from abc import ABC, abstractmethod
from typing import Any, Dict, Union


class BaseLLMModel(ABC):
    """Base interface for all LLM model implementations."""

    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        context: Dict[str, Any] = None,
        **kwargs
    ) -> Union[Dict[str, Any], str]:
        """Generate a response from the model."""
        pass
