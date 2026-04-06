"""Base interface for all LLM model implementations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type


class BaseLLMModel(ABC):
    """Abstract base class that all LLM backends must implement."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        context: Dict[str, Any],
        response_model: Type,
        system_message: str = "Extract the information and return valid JSON.",
        temperature: float = 0.2,
    ) -> Any:
        """Generate a structured response validated against a Pydantic schema."""
        pass

    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.3,
    ) -> Any:
        """Send a chat completion request with optional tool calling."""
        pass
