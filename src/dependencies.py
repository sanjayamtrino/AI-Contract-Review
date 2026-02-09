"""Service dependency container for managing service lifecycle and initialization."""

from typing import Optional

from src.config.logging import Logger
from src.config.settings import get_settings
from src.services.ingestion.ingestion import IngestionService
from src.services.llm.azure_openai_model import AzureOpenAIModel
from src.services.llm.gemini_model import GeminiModel
from src.services.retrieval.retrieval import RetrievalService
from src.services.vector_store.embeddings.embedding_service import BGEEmbeddingService


class ServiceContainer(Logger):
    """
    Service container for managing application-wide service initialization.

    Services are initialized once at application startup and reused throughout
    the application lifecycle. This ensures services remain stateless and that
    failures in one service do not affect others.
    """

    def __init__(self) -> None:
        """Initialize the service container."""
        super().__init__()
        self.settings = get_settings()

        # Service instances
        self._ingestion_service: Optional[IngestionService] = None
        self._retrieval_service: Optional[RetrievalService] = None
        self._azure_openai_model: Optional[AzureOpenAIModel] = None
        self._gemini_model: Optional[GeminiModel] = None
        self._bge_embedding_service: Optional[BGEEmbeddingService] = None

    def initialize(self) -> None:
        """Initialize all services at application startup."""
        try:
            self.logger.info("Initializing service container...")

            # Initialize embedding service
            self._bge_embedding_service = BGEEmbeddingService()
            self.logger.info("BGEEmbeddingService initialized")

            # Initialize LLM models
            self._azure_openai_model = AzureOpenAIModel()
            self.logger.info("AzureOpenAIModel initialized")

            self._gemini_model = GeminiModel()
            self.logger.info("GeminiModel initialized")

            # Initialize retrieval service
            self._retrieval_service = RetrievalService()
            self.logger.info("RetrievalService initialized")

            # Initialize ingestion service
            self._ingestion_service = IngestionService()
            self.logger.info("IngestionService initialized")

            self.logger.info("Service container initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize service container: {str(e)}")
            raise

    def shutdown(self) -> None:
        """Shutdown all services at application shutdown."""
        try:
            self.logger.info("Shutting down service container...")

            # Services are cleaned up here if needed
            self._ingestion_service = None
            self._retrieval_service = None
            self._azure_openai_model = None
            self._gemini_model = None
            self._bge_embedding_service = None

            self.logger.info("Service container shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during service container shutdown: {str(e)}")

    # Service getters
    @property
    def ingestion_service(self) -> IngestionService:
        """Get the ingestion service instance."""
        if self._ingestion_service is None:
            raise RuntimeError("IngestionService not initialized. Call initialize() first.")
        return self._ingestion_service

    @property
    def retrieval_service(self) -> RetrievalService:
        """Get the retrieval service instance."""
        if self._retrieval_service is None:
            raise RuntimeError("RetrievalService not initialized. Call initialize() first.")
        return self._retrieval_service

    @property
    def azure_openai_model(self) -> AzureOpenAIModel:
        """Get the Azure OpenAI model instance."""
        if self._azure_openai_model is None:
            raise RuntimeError("AzureOpenAIModel not initialized. Call initialize() first.")
        return self._azure_openai_model

    @property
    def gemini_model(self) -> GeminiModel:
        """Get the Gemini model instance."""
        if self._gemini_model is None:
            raise RuntimeError("GeminiModel not initialized. Call initialize() first.")
        return self._gemini_model

    @property
    def bge_embedding_service(self) -> BGEEmbeddingService:
        """Get the BGE embedding service instance."""
        if self._bge_embedding_service is None:
            raise RuntimeError("BGEEmbeddingService not initialized. Call initialize() first.")
        return self._bge_embedding_service


# Global service container instance
_service_container: Optional[ServiceContainer] = None


def get_service_container() -> ServiceContainer:
    """Get the global service container instance."""
    global _service_container
    if _service_container is None:
        _service_container = ServiceContainer()
    return _service_container


def initialize_dependencies() -> ServiceContainer:
    """Initialize all dependencies at application startup."""
    container = get_service_container()
    container.initialize()
    return container


def shutdown_dependencies() -> None:
    """Shutdown all dependencies at application shutdown."""
    container = get_service_container()
    container.shutdown()
