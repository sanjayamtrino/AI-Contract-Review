# Coding Conventions

**Analysis Date:** 2026-02-09

## Naming Patterns

**Files:**
- Snake case for all Python files: `doc_parser.py`, `embedding_service.py`, `base_parser.py`
- Descriptive suffixes for specialized classes:
  - `*_router.py` - FastAPI route handlers (e.g., `ingestion/router.py`)
  - `*_exceptions.py` - Custom exception classes (e.g., `ingestion_exceptions.py`)
  - `*_service.py` - Business logic services (e.g., `ingestion.py`, `embedding_service.py`)
  - `*_model.py` - LLM model implementations (e.g., `gemini_model.py`, `azure_openai_model.py`)
  - `base_*.py` - Abstract base classes (e.g., `base_parser.py`, `base_embedding_service.py`)
- Directory structure follows domain/feature organization: `services/vector_store/embeddings/`, `api/endpoints/ingestion/`

**Functions and Methods:**
- Snake case for all functions and methods: `generate_embeddings()`, `retrieve_data()`, `_clean_text()`
- Private methods prefixed with single underscore: `_parse_data()`, `_clean_text()`
- Async functions use `async def`: `async def generate_embeddings()`, `async def parse()`
- Functions with type hints on all parameters and return values:
  ```python
  async def generate(self, prompt: str, context: Dict[str, Any], response_model: Type) -> str:
  ```
- Public API endpoint functions are descriptive: `ingest_data()`, `retrieve_data()`, `compare_embeddings()`

**Variables:**
- Snake case for all local and instance variables: `chunk_size`, `chunk_overlap`, `total_tokens_processed`, `embedding_service`
- Type hints used consistently:
  ```python
  self.stats: Dict[str, Any] = {
      "embeddings_generated": 0,
      "total_tokens_processed": 0,
      "average_emmbedding_time": 0.0,
  }
  ```
- Settings and configuration variables use descriptive names: `gemini_api_key`, `openai_embedding_model`, `chunk_overlap`
- Private attributes not prefixed by convention (Python culture preferred)

**Classes:**
- PascalCase for all class names: `DocxParser`, `HuggingFaceEmbeddingService`, `GeminiModel`, `IngestionService`
- Abstract base classes prefixed with "Base": `BaseParser`, `BaseEmbeddingService`, `BaseLLMModel`
- Service classes suffixed with "Service": `IngestionService`, `HuggingFaceEmbeddingService`, `RetrievalService`
- Model wrapper classes follow the pattern: `GeminiModel`, `AzureOpenAIModel`
- Exception classes suffixed with "Exception": `AppException`, `DocxCleaningException`, `DocxMetadataExtractionException`

**Types:**
- Pydantic `BaseModel` for all data schemas: `Chunk`, `ParseResult`, `QueryRewriterResponse`
- `Union` type used for optional values: `Union[str, None]` (equivalent to `Optional[str]`)
- Type hints for all function signatures use `Dict[str, Any]`, `List[float]`, `Optional[str]`
- Settings class extends `BaseSettings` from pydantic-settings for configuration

## Code Style

**Formatting:**
- Black formatter configured with:
  - Line length: 200 characters (in `pyproject.toml`)
  - Target version: Python 3.10
  - Trailing commas included, multi-line output style 3
- Files automatically formatted to Black's standards

**Linting:**
- Flake8 configured in `pyproject.toml` with:
  - Max line length: 200
  - Ignored rules: E203 (whitespace), W503 (line break before binary operator)
  - Max complexity: 10
  - Extends B, C, E, F, W, T4, B9 rules
  - Quote style: double quotes for inline, docstrings
  - Import order style: Google
- MyPy configured for strict type checking:
  - `disallow_untyped_defs: true` - All functions must have type hints
  - `check_untyped_defs: true` - Check calls to untyped functions
  - `ignore_missing_imports: true` - Allow untyped third-party imports
  - `strict_optional: true` - Strict None handling

**Indentation:**
- 4 spaces per indentation level (Python standard)
- No tabs; spaces only

## Import Organization

**Order:**
1. Standard library imports: `import json`, `from typing import`, `from io import BytesIO`, `from abc import ABC, abstractmethod`
2. Third-party imports: `from fastapi import`, `from pydantic import`, `from langchain.text_splitter import`, `import torch`, `from sentence_transformers import`
3. Local application imports: `from src.config.logging import`, `from src.services.registry import`, `from src.schemas.registry import`

**Path Aliases:**
- No path aliases configured (use absolute imports from `src/`)
- All imports use absolute paths: `from src.config.settings import get_settings` (not relative `from ..config.settings`)
- Root package is `src` consistently across all files

**Grouped imports within categories:**
- Related imports from same package grouped together:
  ```python
  from src.exceptions.parser_exceptions import (
      DocxCleaningException,
      DocxMetadataExtractionException,
      DocxParagraphExtractionException,
      DocxTableExtractionException,
  )
  ```
- Multi-line imports use trailing commas and proper indentation

**isort configuration:**
- Profile: Black (compatible with Black formatter)
- Line length: 200
- Known first party: configured in `pyproject.toml`
- Skip `__init__.py` files during sorting
- Combine as imports enabled
- Include trailing commas

## Error Handling

**Patterns:**
- Custom exception hierarchy with base class `AppException` in `src/exceptions/base_exception.py`
- Domain-specific exceptions in dedicated files:
  - `ingestion_exceptions.py` for parsing/ingestion errors
  - `parser_exceptions.py` for DOCX parsing errors (e.g., `DocxCleaningException`, `DocxTableExtractionException`)
- Exceptions inherit from `AppException` and store error message:
  ```python
  class AppException(Exception):
      def __init__(self, message: str) -> None:
          super().__init__(message)
          self.message = message
  ```
- Try-except blocks used in service methods with logging:
  ```python
  try:
      # operation
  except Exception as e:
      self.logger.error(f"Operation failed: {e}")
      raise
  ```
- ValueError for invalid inputs: `raise ValueError("Query cannot be empty.")`
- Functions validate inputs and raise descriptive errors early

## Logging

**Framework:** Python standard `logging` module with configuration in `src/config/logging.py`

**Patterns:**
- Logger obtained through `Logger` mixin class (property-based):
  ```python
  class DocxParser(BaseParser, Logger):
      # self.logger property automatically available
  ```
- Logger obtained via module function: `get_logger("ModuleName")`
- Log levels used appropriately:
  - DEBUG: Low-level details, embedding generation times, stats updates
  - INFO: High-level operation flow, completion messages
  - ERROR: Failures and exceptions
- Structured logging with JSON formatter for error logs (in `errors.log`)
- Log format includes: timestamp, logger name, level, filename, line number, function name, message
- Rotating file handlers with 10MB max file size and 5 backup files
- Separate handlers for console (INFO), file (DEBUG), and errors (JSON)

**Log Configuration:**
```python
logging.config.dictConfig(logging_config)
log_filename = f"AI_Contract_Review_{datetime.now().strftime('%Y%m%d')}.log"
```

## Comments

**When to Comment:**
- Docstrings on all public classes, methods, and functions (Google/NumPy style)
- Inline comments for non-obvious logic or complex algorithms
- Comments explain "why", not "what" (code should be self-documenting)
- TODO/FIXME comments when work is incomplete or known issues exist

**Docstring Style:**
- Triple-quoted docstrings (format varies; appears to favor simple single-line summaries)
- Examples observed:
  ```python
  """Cleans and normalize the text content."""

  """Parse the given document."""

  """Get cached application settings instance."""
  ```
- Parameter and return type documentation minimal (rely on type hints instead)

## Function Design

**Size:** No strict line limits observed; methods range from 15-399 lines (e.g., `doc_parser.py` at 399 lines)
- Service methods typically 30-70 lines
- Complex parsing logic concentrated in specialized classes (e.g., `DocxParser`)
- Generally aim for single responsibility but allow domain-specific complexity

**Parameters:**
- Type hints required on all parameters
- Default values used for optional configuration: `top_k: int = 5`, `threshold: Optional[float] = 0.0`
- Self is first parameter for methods
- Related parameters grouped together (settings, then data, then options)

**Return Values:**
- Type hints required on all return values
- Async functions return actual values, not coroutines
- Multiple return values via tuple or Pydantic model:
  ```python
  async def retrieve_data(...) -> Dict[str, Any]:
  ```
- None return explicitly typed when applicable

## Module Design

**Exports:**
- No explicit `__all__` definitions observed
- Modules export classes, functions, and constants directly
- Router objects instantiated at module level and exported: `router = APIRouter()`
- Service instances created at module level for DI: `ingestion_service = IngestionService()`

**Barrel Files:**
- `__init__.py` files exist but generally empty (Python 3.10+ implicit namespace packages)
- No re-exports in `__init__.py` files observed

**File Organization:**
- One main class per file (e.g., `DocxParser` in `doc_parser.py`)
- Base classes separate from implementations
- Related functionality grouped in directories by feature/domain
- Services directory contains business logic; schemas contain data models; config contains configuration

---

*Convention analysis: 2026-02-09*
