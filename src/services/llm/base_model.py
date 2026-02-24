from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel


class BaseLLMModel(ABC):
    """Base interface for all LLM model implementations."""

    @abstractmethod
    async def generate(self, prompt: str, context: Dict[str, Any], response_model: Type, system_message: str = "Extract the information and return valid JSON.") -> Any:
        """Generate a structured response from the model using a Pydantic schema."""
        pass

    @abstractmethod
    async def chat_completion(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None, tool_choice: str = "auto", temperature: float = 0.3) -> Any:
        """Send a chat completion request to the model."""
        pass
