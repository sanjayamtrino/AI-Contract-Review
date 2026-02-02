import os
from functools import lru_cache
from typing import Union

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration settings."""

    # API settings
    api_host: str = Field(default="localhost", description="API host address")
    api_port: int = Field(default=8000, description="API port number")
    debug: bool = Field(default=False, description="Enable or disable debug mode")

    # Chunking Settings
    chunk_size: int = Field(default=1000, description="Size of each text chunk")
    chunk_overlap: int = Field(default=200, description="Overlap size between text chunks")

    # LLM Settings
    gemini_api_key: Union[str, None] = Field(default=None, description="API key for Gemini LLM Model.")
    gemini_embedding_model: str = Field(default="gemini-embedding-001", description="Gemini Embedding model used.")
    gemini_text_generation_model: str = Field(default="gemini-2.5-flash-lite-preview-09-2025", description="Gemini model for Query Rewriting task.")
    openai_api_key: Union[str, None] = Field(default=None, description="API Key  for the OPENAI Models.")
    openai_embedding_model: str = Field(default="text-embedding-3-large", description="OPENAI Embedding model used.")

    # Hugging Face Settings
    hugggingface_minilm_embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Hugging Face Embedding Model.")  # 384
    hugggingface_qwen_embedding_model: str = Field(default="Qwen/Qwen3-Embedding-0.6B", description="Qwen3 0.6B model from Hugging Face.")
    huggingface_jina_embedding_model: str = Field(default="jina-embeddings-v3", description="Jina Embedding model from HuggingFace.")

    # Jina Embeddings
    jina_embedding_model_uri: str = Field(default="https://api.jina.ai/v1/embeddings", description="Jina-Embedding model URI.")
    jina_embedding_API: Union[str, None] = Field(default=None, description="API Token for Jina Embeddings.")

    # Vector Database settings
    # db_dimention: int = Field(default=1536, description="Dimention of the vector to store.")

    # Storage paths
    logs_directory: str = Field(default="./logs", description="Directory for application logs")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings instance."""

    return Settings()


def ensure_directories() -> None:
    """Ensure directories exists."""

    settings = get_settings()
    os.makedirs(settings.logs_directory, exist_ok=True)


ensure_directories()
