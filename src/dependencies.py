"""
Dependency injection container for application-wide services.

Initializes and manages the lifecycle of all core services:
embedding, LLM, vector store, session management, ingestion, and retrieval.
"""

from typing import Optional

from src.config.logging import Logger
from src.config.settings import Settings
from src.services.ingestion.ingestion import IngestionService
from src.services.llm.azure_openai_model import AzureOpenAIModel
from src.services.retrieval.retrieval import RetrievalService
from src.services.session_manager import SessionManager
from src.services.vector_store.embeddings.embedding_service import (
    HuggingFaceEmbeddingService,
)
from src.services.vector_store.faiss_db import FAISSVectorStore


class ServiceContainer(Logger):
    """Manages initialization and access to all application services.

    Services are created once at startup and reused throughout
    the application lifecycle.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ingestion_service: Optional[IngestionService] = None
        self._retrieval_service: Optional[RetrievalService] = None
        self._azure_openai_model: Optional[AzureOpenAIModel] = None
        self._embedding_service: Optional[HuggingFaceEmbeddingService] = None
        self._session_manager: Optional[SessionManager] = None
        self._faiss_store: Optional[FAISSVectorStore] = None
        self._settings: Optional[Settings] = None

    def initialize(self) -> None:
        """Initialize all services at application startup."""
        try:
            self.logger.info("Initializing service container...")

            # Embedding service first (determines vector dimensions)
            self._embedding_service = HuggingFaceEmbeddingService()
            embedding_dimension = self._embedding_service.get_embedding_dimensions()
            self.logger.info(f"Embedding service initialized (dimension={embedding_dimension})")

            # Session manager
            self._session_manager = SessionManager(embedding_dimension=embedding_dimension)
            self.logger.info("SessionManager initialized")

            # LLM model
            self._azure_openai_model = AzureOpenAIModel()
            self.logger.info("AzureOpenAIModel initialized")

            # Global FAISS index
            self._faiss_store = FAISSVectorStore(embedding_dimension)
            self.logger.info("FAISS store initialized")

            # Retrieval and ingestion
            self._retrieval_service = RetrievalService()
            self.logger.info("RetrievalService initialized")

            self._ingestion_service = IngestionService()
            self.logger.info("IngestionService initialized")

            # Settings
            self._settings = Settings()
            self.logger.info("Service container initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize service container: {str(e)}")
            raise

    async def shutdown(self) -> None:
        """Release all services at application shutdown."""
        try:
            self.logger.info("Shutting down service container...")

            if self._session_manager:
                try:
                    await self._session_manager.stop_cleanup_worker()
                except Exception as e:
                    self.logger.error(f"Error stopping cleanup worker: {str(e)}")

            self._ingestion_service = None
            self._retrieval_service = None
            self._azure_openai_model = None
            self._embedding_service = None
            self._session_manager = None
            self._settings = None

            self.logger.info("Service container shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during service container shutdown: {str(e)}")

    # --- Property accessors ---

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            raise RuntimeError("Settings not initialized. Call initialize() first.")
        return self._settings

    @property
    def session_manager(self) -> SessionManager:
        if self._session_manager is None:
            raise RuntimeError("SessionManager not initialized. Call initialize() first.")
        return self._session_manager

    @property
    def faiss_store(self) -> FAISSVectorStore:
        if self._faiss_store is None:
            raise RuntimeError("FAISSVectorStore not initialized. Call initialize() first.")
        return self._faiss_store

    @property
    def ingestion_service(self) -> IngestionService:
        if self._ingestion_service is None:
            raise RuntimeError("IngestionService not initialized. Call initialize() first.")
        return self._ingestion_service

    @property
    def retrieval_service(self) -> RetrievalService:
        if self._retrieval_service is None:
            raise RuntimeError("RetrievalService not initialized. Call initialize() first.")
        return self._retrieval_service

    @property
    def azure_openai_model(self) -> AzureOpenAIModel:
        if self._azure_openai_model is None:
            raise RuntimeError("AzureOpenAIModel not initialized. Call initialize() first.")
        return self._azure_openai_model

    @property
    def embedding_service(self) -> HuggingFaceEmbeddingService:
        if self._embedding_service is None:
            raise RuntimeError("EmbeddingService not initialized. Call initialize() first.")
        return self._embedding_service


# Singleton instance
_service_container: Optional[ServiceContainer] = None


def get_service_container() -> ServiceContainer:
    """Get or create the global service container."""
    global _service_container
    if _service_container is None:
        _service_container = ServiceContainer()
    return _service_container


async def initialize_dependencies() -> ServiceContainer:
    """Initialize all dependencies at application startup."""
    container = get_service_container()
    container.initialize()

    # Start background session cleanup worker
    if container._session_manager:
        try:
            container._session_manager.start_cleanup_worker_sync()
            container.logger.info("Session cleanup worker started")
        except RuntimeError:
            container.logger.warning("Could not start cleanup worker: no running event loop")

    return container


async def shutdown_dependencies() -> None:
    """Shutdown all dependencies at application shutdown."""
    container = get_service_container()
    await container.shutdown()
