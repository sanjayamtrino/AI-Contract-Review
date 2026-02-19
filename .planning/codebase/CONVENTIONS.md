# Coding Conventions

**Analysis Date:** 2026-02-19

## Naming Patterns

**Files:**
- Use `snake_case.py` for all Python modules: `base_model.py`, `session_manager.py`, `clause_extractor.py`
- `__init__.py` files are present in every package but are mostly empty (no re-exports)
- Router files are always named `router.py` inside their endpoint directory: `src/api/endpoints/ingestion/router.py`
- Prompt templates use `snake_case.mustache`: `orchestrator_prompt.mustache`, `clause_comparison_prompt.mustache`
- Schema files map to domain concept: `playbook.py`, `playbook_v2.py`, `registry.py`, `query_rewriter.py`
- Test files use `test_` prefix: `test_embeddings.py`, `test_backend_logic.py`
- Data files use `snake_case.json`: `default_playbook_rules.json`, `playbook_rules_v3.json`

**Classes:**
- Use `PascalCase` for all classes: `AzureOpenAIModel`, `BGEEmbeddingService`, `SessionManager`
- Abstract base classes include `Base` prefix: `BaseLLMModel`, `BaseParser`, `BaseEmbeddingService`, `BaseVectorStore`
- Exception classes use descriptive suffix `Exception`: `DocxCleaningException`, `DocxMetadataExtractionException`
- Pydantic response models use `Response` suffix: `SummaryResponse`, `KeyDetailsResponse`, `ClauseExtractionResponse`, `ReviewReportResponse`
- Service classes use `Service` suffix: `IngestionService`, `RetrievalService`, `BGEEmbeddingService`

**Functions:**
- Use `snake_case` for all functions and methods: `get_summary`, `extract_clauses`, `generate_embeddings`
- Private/internal functions use `_` prefix: `_parse_data`, `_clean_text`, `_extract_metadata`, `_run_full_review`
- Async functions follow same naming; use `async def` consistently
- Tool entry functions are named for their action: `get_summary()`, `get_key_details()`, `extract_clauses()`, `compare_clause_batch()`
- Agent entry points are always `async def run(query: str, session_id: str) -> Dict[str, Any]`

**Variables:**
- Use `snake_case` for all variables: `session_id`, `chunk_store`, `embedding_dimension`
- Module-level private singletons use `_` prefix: `_llm`, `_cached_rules`, `_instance`, `_chunk_store`
- Constants use `UPPER_SNAKE_CASE`: `AGENT_REGISTRY`, `TOOL_DEFINITIONS`, `BATCH_SIZE`, `PROMPTS_DIR`
- Settings fields use `snake_case`: `api_host`, `gemini_api_key`, `chunk_size`

**Types:**
- Pydantic `BaseModel` subclasses for all data schemas
- Pydantic `Field(...)` with `description` kwarg for every field
- `Optional` fields use `Optional[Type] = Field(None, ...)` pattern
- Enums use `str, Enum` inheritance: `class RiskLevel(str, Enum)`

## Code Style

**Formatting:**
- **Black** formatter with `line-length = 200` (configured in `pyproject.toml`)
- VS Code configured for format-on-save via `.vscode/settings.json`
- Target Python version: `py310`

**Linting:**
- **flake8** with `max-line-length = 200`, Google import order style
- **isort** with Black profile, `line_length = 200`, trailing comma enabled, `multi_line_output = 3`
- **mypy** with `disallow_untyped_defs = true`, `strict_optional = true`, `ignore_missing_imports = true`
- Flake8 ignores: `E203, W503`
- Max complexity: 10

**Quotes:**
- Double quotes throughout (enforced by flake8 config: `inline-quotes = "double"`)

**Line Length:**
- 200 characters max (all tools agree: Black, flake8, isort)

## Import Organization

**Order (isort Black profile):**
1. Standard library imports (`import json`, `import asyncio`, `from typing import ...`)
2. Third-party imports (`from fastapi import ...`, `from pydantic import ...`, `import pystache`)
3. Local application imports (`from src.config.logging import Logger`, `from src.dependencies import ...`)

**Path Style:**
- Always use absolute imports from `src` root: `from src.config.settings import get_settings`
- No relative imports observed
- No path aliases configured
- isort skips `__init__.py` files

**Typical Import Pattern (example from `src/tools/clause_extractor.py`):**
```python
from pathlib import Path
from typing import Any, Optional

from src.dependencies import get_service_container
from src.schemas.playbook import ClauseExtractionResponse
from src.services.llm.azure_openai_model import AzureOpenAIModel
from src.services.playbook_loader import get_all_categories, load_default_rules, rules_to_prompt_context
```

**Path Aliases:**
- None configured. All imports are absolute from `src`.

## Common Design Patterns

**Service Container / Dependency Injection:**
- Global singleton `ServiceContainer` in `src/dependencies.py` manages all services
- Access via `get_service_container()` function
- Services are lazily initialized at app startup via `initialize_dependencies()`
- Properties with `RuntimeError` guards enforce initialization order
- Pattern: `container = get_service_container()` then `container.some_service`

**Abstract Base Classes:**
- `BaseLLMModel` (`src/services/llm/base_model.py`) - interface for LLM providers
- `BaseParser` (`src/services/registry/base_parser.py`) - interface for document parsers
- `BaseEmbeddingService` (`src/services/vector_store/embeddings/base_embedding_service.py`) - interface for embeddings
- `BaseVectorStore` (`src/services/vector_store/base_store.py`) - interface for vector databases
- All use `ABC` with `@abstractmethod`

**Logger Mixin:**
- `Logger` class in `src/config/logging.py` is a mixin providing `self.logger` property
- Used via multiple inheritance: `class MyService(SomeBase, Logger)`
- Logger name is auto-derived from class name: `AI_Contract.{ClassName}`

**Registry Pattern:**
- `ParserRegistry` (`src/services/registry/registry.py`) stores parsers by type name
- Agents registered in `AGENT_REGISTRY` dict in `src/orchestrator/main.py`
- Tools registered in `TOOL_REGISTRY` dict per agent module

**Prompt Template Pattern:**
- Mustache templates stored in `src/services/prompts/v1/`
- Loaded via `load_prompt("template_name")` from `src/services/prompts/v1/__init__.py` (uses chevron)
- Or loaded via `Path("src/services/prompts/v1/xxx.mustache").read_text()` directly (uses pystache)
- Two template engines in use: **chevron** (in `load_prompt`) and **pystache** (in LLM model classes)
- Context passed as dict with mustache variable substitution
- pystache used with `escape=lambda u: u` to disable HTML escaping

**LLM Generation Pattern (consistent across all tools):**
```python
_llm = AzureOpenAIModel()  # module-level singleton

async def some_tool(...) -> Any:
    prompt_path = Path("src/services/prompts/v1/some_prompt.mustache")
    prompt = prompt_path.read_text(encoding="utf-8")
    return await _llm.generate(
        prompt=prompt,
        context={"key": "value"},
        response_model=SomePydanticModel,
    )
```

**Session-Based Data Access Pattern:**
```python
container = get_service_container()
session = container.session_manager.get_session(session_id)
if not session:
    raise ValueError(f"Session '{session_id}' not found or expired")
results = session.chunk_store
```

**Agent Architecture Pattern:**
- Each agent module in `src/agents/` exposes an `async def run(query, session_id) -> Dict[str, Any]`
- Agents define `TOOL_REGISTRY` and `TOOL_DEFINITIONS` at module level
- Orchestrator uses OpenAI function-calling to select agents
- Agents use OpenAI function-calling to select tools
- Results returned as `{"agent": name, "tools_called": [...], "tool_results": {...}}`

## Error Handling

**Patterns:**

1. **Custom Exception Hierarchy:**
   - Root: `AppException(Exception)` in `src/exceptions/base_exception.py`
   - Domain exceptions: `DocxCleaningException`, `DocxMetadataExtractionException`, etc.
   - All store a `message` attribute

2. **Try/Except with Logging:**
   ```python
   try:
       result = some_operation()
   except Exception as e:
       self.logger.error(f"Operation failed: {str(e)}")
       raise ValueError("User-facing message") from e
   ```

3. **Graceful Degradation (ParseResult pattern):**
   ```python
   except Exception as e:
       self.logger.error(str(e))
       return ParseResult(success=False, chunks=[], metadata={}, error_message=str(e), processing_time=0.0)
   ```

4. **Agent Error Isolation:**
   ```python
   try:
       result = await TOOL_REGISTRY[func_name](session_id=session_id)
   except Exception as e:
       tool_results[func_name] = {"error": str(e)}
   ```

5. **Guard Clauses:**
   - Settings validation in constructors: `if not self.settings.azure_openai_api_key: raise ValueError(...)`
   - Session existence checks: `if not session: raise ValueError(f"Session not found...")`
   - Empty data checks: `if not results: raise ValueError("No document ingested...")`

**What to follow:**
- Always chain exceptions with `from e`
- Log errors before re-raising
- Use `ValueError` for business logic errors (session not found, no data, etc.)
- Use custom `AppException` subclasses for infrastructure errors
- Return error dicts `{"error": str(e)}` at agent/tool boundaries instead of propagating

## Logging

**Framework:** Python `logging` module with dictConfig setup

**Configuration:** `src/config/logging.py`
- Three handlers: `console` (INFO, simple), `file` (DEBUG, detailed with rotation), `error_file` (ERROR, JSON format with rotation)
- Log files: `./logs/AI_Contract_Review_YYYYMMDD.log` (10MB rotation, 5 backups)
- Error file: `./logs/errors.log` (JSON format, 10MB rotation)
- Named logger: `contract_review`

**Access Patterns:**
1. **Mixin (preferred for classes):** Inherit from `Logger`, use `self.logger`
2. **Standalone function:** `get_logger("module_name")` returns `logging.Logger` prefixed with `AI_Contract.`

**When to Log:**
- `INFO` for service initialization, operations starting/completing, chunk counts
- `DEBUG` for per-chunk processing, embedding timings
- `ERROR` for exceptions, failed operations (always include exception message)
- `WARNING` for dimension mismatches, missing optional config

**Pattern:**
```python
self.logger.info(f"Indexed {len(chunks)} chunks into session {session_data.session_id}")
self.logger.error(f"LLM generation failed: {str(e)}")
self.logger.debug(f"Generated embedding in {generation_time:.4f}s")
```

## Comments

**When to Comment:**
- Docstrings on all public classes and methods (required by mypy `disallow_untyped_defs`)
- Inline comments for non-obvious logic (threshold calculations, flush behavior)
- Section comments for long functions: `# Step 1: ...`, `# Step 2: ...`
- TODO-style comments for known gaps: `# Need to implement this method`

**Docstring Style:**
- Simple one-line docstrings: `"""Parse the given document."""`
- Multi-line for complex functions with parameter descriptions in `Field()` rather than docstrings
- No consistent use of Google/NumPy/reST docstring format

## Function Design

**Size:** Functions are generally moderate (10-50 lines). Larger parsing methods (100+ lines) exist in `src/services/registry/semantic_parser.py` and `src/services/registry/doc_parser.py`.

**Parameters:**
- Type hints on all parameters and return values
- Use `Optional[Type] = None` for optional params
- Use `**kwargs` sparingly (only on tool functions for extensibility)
- `Dict[str, Any]` is the dominant return type for tools and agents

**Return Values:**
- Pydantic models for LLM responses and structured data
- `Dict[str, Any]` for agent/tool results and API responses
- `ParseResult` for ingestion pipeline results
- `List[float]` for embeddings

## Module Design

**Exports:**
- No explicit `__all__` definitions anywhere
- `__init__.py` files are empty (no barrel exports)
- Import directly from the module: `from src.services.llm.azure_openai_model import AzureOpenAIModel`

**Module-Level Singletons:**
- LLM instances: `_llm = AzureOpenAIModel()` at module level in tool files
- Cached data: `_cached_rules: List[PlaybookRule] = []` with guard in `load_default_rules()`
- Thread-safe globals: `_instance`, `_lock`, `_chunk_store` in `src/services/vector_store/manager.py`

**Barrel Files:**
- Not used. Each module is imported individually.

---

*Convention analysis: 2026-02-19*
