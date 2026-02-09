# Codebase Structure

**Analysis Date:** 2026-02-09

## Directory Layout

```
C:\Users\amtri\AI-Contract-Review\
├── src/                           # Main application source code
│   ├── __init__.py               # Package marker
│   ├── api/                      # HTTP API layer
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI app initialization and startup
│   │   └── endpoints/            # Endpoint routers by feature
│   │       ├── __init__.py
│   │       ├── ingestion/        # Document ingestion endpoints
│   │       │   ├── __init__.py
│   │       │   └── router.py     # POST /api/v1/ingest/
│   │       ├── retrieval/        # Document retrieval endpoints
│   │       │   ├── __init__.py
│   │       │   └── router.py     # POST /api/v1/query/, GET /api/v1/summarizer
│   │       └── orchestrator/     # Agent orchestration endpoints (unused)
│   │           ├── __init__.py
│   │           └── router.py
│   │
│   ├── config/                   # Configuration and setup
│   │   ├── __init__.py
│   │   ├── settings.py           # Pydantic Settings for environment config
│   │   └── logging.py            # Logging setup and Logger mixin class
│   │
│   ├── exceptions/               # Custom exception hierarchy
│   │   ├── __init__.py
│   │   ├── base_exception.py     # AppException base class
│   │   ├── ingestion_exceptions.py
│   │   └── parser_exceptions.py  # DocxCleaningException, etc.
│   │
│   ├── schemas/                  # Pydantic data models
│   │   ├── __init__.py
│   │   ├── registry.py           # Chunk, ParseResult schemas
│   │   └── query_rewriter.py     # QueryRewriterResponse schema
│   │
│   ├── services/                 # Core business logic and integrations
│   │   ├── __init__.py
│   │   │
│   │   ├── ingestion/            # Document ingestion pipeline
│   │   │   ├── __init__.py
│   │   │   └── ingestion.py      # IngestionService class
│   │   │
│   │   ├── registry/             # Parser registry and implementations
│   │   │   ├── __init__.py
│   │   │   ├── base_parser.py    # BaseParser abstract class
│   │   │   ├── registry.py       # ParserRegistry (DOCX parser registered)
│   │   │   ├── doc_parser.py     # Basic DOCX parser (superseded)
│   │   │   └── semantic_parser.py # DocxParser with semantic chunking
│   │   │
│   │   ├── llm/                  # Language model services
│   │   │   ├── __init__.py
│   │   │   ├── base_model.py     # BaseLLMModel abstract class
│   │   │   ├── azure_openai_model.py # AzureOpenAIModel implementation
│   │   │   └── gemini_model.py   # GeminiModel (currently unused)
│   │   │
│   │   ├── retrieval/            # Document retrieval pipeline
│   │   │   ├── __init__.py
│   │   │   ├── retrieval.py      # RetrievalService with query rewriting
│   │   │   └── query_rewriter.py # Query rewriting utilities
│   │   │
│   │   ├── vector_store/         # Vector database management
│   │   │   ├── __init__.py
│   │   │   ├── manager.py        # Singleton FAISS store manager with thread-safety
│   │   │   ├── base_store.py     # Base vector store interface
│   │   │   ├── faiss_db.py       # FAISS index wrapper
│   │   │   └── embeddings/       # Embedding generation services
│   │   │       ├── __init__.py
│   │   │       ├── base_embedding_service.py # BaseEmbeddingService abstract
│   │   │       ├── embedding_service.py      # BGEEmbeddingService (active)
│   │   │       ├── openai_embeddings.py      # OpenAI embeddings (commented)
│   │   │       ├── gemini_embeddings.py      # Gemini embeddings (commented)
│   │   │       ├── jina_embeddings.py        # Jina embeddings (commented)
│   │   │       └── qwen_embeddings.py        # Qwen embeddings (commented)
│   │   │
│   │   └── prompts/              # LLM prompt templates
│   │       ├── __init__.py
│   │       └── v1/               # Version 1 prompts
│   │           ├── __init__.py
│   │           ├── query_rewriter.mustache
│   │           ├── llm_response.mustache
│   │           ├── orchestrator_prompt
│   │           └── summary_prompt_template.mustache
│   │
│   ├── orchestrator/             # Agentic orchestration
│   │   ├── __init__.py
│   │   └── main.py               # OpenAIChat client and agent setup
│   │
│   └── tools/                    # Agent tools
│       ├── __init__.py
│       └── summarizer.py         # Dummy tools: get_summary, get_location, get_key_information
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_backend_logic.py
│   └── test_embeddings.py
│
├── docs/                         # Documentation
│   ├── PROMPT_DOCUMENTATION.md
│   └── architecture_overview.png
│
├── logs/                         # Application logs (generated at runtime)
│
├── .planning/                    # GSD planning directory
│   └── codebase/                 # Codebase analysis documents
│
├── .vscode/                      # IDE configuration
├── .git/                         # Git repository
├── .gitignore                    # Git ignore rules
├── .env                          # Environment variables (NEVER COMMIT)
├── .env.example                  # Environment template
├── pyproject.toml                # Poetry project manifest
├── poetry.lock                   # Dependency lock file
├── Makefile                      # Build automation (empty)
└── README.md                     # Project overview
```

## Directory Purposes

**`src/`:**
- Purpose: All application source code
- Contains: Python modules organized by layer/feature
- Key files: Package markers (__init__.py) for Python import

**`src/api/`:**
- Purpose: FastAPI HTTP API layer
- Contains: Main app setup, endpoint routers
- Key files: `main.py` (app initialization)

**`src/api/endpoints/`:**
- Purpose: RESTful endpoint definitions organized by domain
- Contains: FastAPI routers with request/response handlers
- Key files: `ingestion/router.py`, `retrieval/router.py`, `orchestrator/router.py`

**`src/config/`:**
- Purpose: Application configuration and cross-cutting concerns
- Contains: Settings from environment, logging configuration, Logger mixin
- Key files: `settings.py`, `logging.py`

**`src/exceptions/`:**
- Purpose: Custom exception hierarchy for explicit error handling
- Contains: Exception classes with descriptive names
- Key files: `base_exception.py` (root), `parser_exceptions.py` (domain-specific)

**`src/schemas/`:**
- Purpose: Type-safe data validation using Pydantic
- Contains: Input/output data models with field validation
- Key files: `registry.py` (Chunk, ParseResult), `query_rewriter.py` (QueryRewriterResponse)

**`src/services/`:**
- Purpose: Core business logic and external integrations
- Contains: Service classes implementing domain operations
- Key files: Varies by subdirectory

**`src/services/ingestion/`:**
- Purpose: Document upload and parsing coordination
- Contains: IngestionService orchestrating parser registry and vector store
- Key files: `ingestion.py` (main service)

**`src/services/registry/`:**
- Purpose: Pluggable parser management
- Contains: Parser base class, implementations, and registry
- Key files: `base_parser.py`, `semantic_parser.py` (DocxParser), `registry.py`

**`src/services/llm/`:**
- Purpose: Language model integrations
- Contains: LLM client implementations with template rendering
- Key files: `base_model.py`, `azure_openai_model.py` (active), `gemini_model.py` (inactive)

**`src/services/retrieval/`:**
- Purpose: Semantic search and document retrieval
- Contains: RetrievalService with query rewriting and ranking
- Key files: `retrieval.py` (main service), `query_rewriter.py`

**`src/services/vector_store/`:**
- Purpose: Vector database and embedding management
- Contains: FAISS index wrapper, embedding services, chunk storage
- Key files: `manager.py` (singleton pattern), `faiss_db.py`, `embeddings/`

**`src/services/vector_store/embeddings/`:**
- Purpose: Embedding model providers
- Contains: Multiple embedding service implementations (strategy pattern)
- Key files: `base_embedding_service.py`, `embedding_service.py` (BGE active), others inactive

**`src/services/prompts/`:**
- Purpose: LLM prompt templates with versioning
- Contains: Mustache templates for prompt rendering
- Key files: `v1/query_rewriter.mustache`, `v1/llm_response.mustache`, `v1/orchestrator_prompt`

**`src/orchestrator/`:**
- Purpose: Agent framework integration for agentic workflows
- Contains: Custom OpenAIChat client extending BaseChatClient
- Key files: `main.py` (agent initialization and tool binding)

**`src/tools/`:**
- Purpose: Agent-callable tools for orchestrator
- Contains: Tool implementations returning structured data
- Key files: `summarizer.py` (dummy tools: get_summary, get_location, get_key_information)

**`tests/`:**
- Purpose: Test suite for core functionality
- Contains: Unit and integration tests
- Key files: `test_backend_logic.py`, `test_embeddings.py`

**`docs/`:**
- Purpose: Project documentation and architecture diagrams
- Contains: Markdown documentation, images
- Key files: `PROMPT_DOCUMENTATION.md`, `architecture_overview.png`

**`logs/`:**
- Purpose: Application runtime logs
- Contains: Daily rotating log files and error logs
- Key files: `AI_Contract_Review_YYYYMMDD.log`, `errors.log` (generated at runtime)

## Key File Locations

**Entry Points:**
- `src/api/main.py`: FastAPI app and uvicorn startup
- `src/orchestrator/main.py`: Agentic orchestrator with tools
- `tests/test_embeddings.py`: Embedding service testing

**Configuration:**
- `pyproject.toml`: Poetry dependencies, black, isort, mypy, flake8 settings
- `.env`: Environment variables with API keys (DO NOT COMMIT)
- `.env.example`: Template for required environment variables

**Core Logic:**
- `src/services/ingestion/ingestion.py`: Document parsing orchestration
- `src/services/registry/semantic_parser.py`: DOCX semantic chunking implementation
- `src/services/retrieval/retrieval.py`: Query rewriting and vector search
- `src/services/vector_store/manager.py`: Thread-safe chunk and FAISS store management
- `src/services/llm/azure_openai_model.py`: LLM response generation with template rendering

**Schemas/Types:**
- `src/schemas/registry.py`: Chunk, ParseResult (core data contracts)
- `src/schemas/query_rewriter.py`: QueryRewriterResponse

**Prompts:**
- `src/services/prompts/v1/query_rewriter.mustache`: Query expansion template
- `src/services/prompts/v1/llm_response.mustache`: Response generation template
- `src/services/prompts/v1/orchestrator_prompt`: Agent instruction template

## Naming Conventions

**Files:**
- Service files: `{domain}_service.py` (e.g., `ingestion.py`, `retrieval.py`)
- Parser implementations: `{format}_parser.py` (e.g., `doc_parser.py`)
- Parser implementations with features: `semantic_parser.py` (DocxParser with semantic chunking)
- Exception files: `{domain}_exceptions.py` (e.g., `parser_exceptions.py`)
- Embedding implementations: `{provider}_embeddings.py` (e.g., `openai_embeddings.py`, `jina_embeddings.py`)
- Prompts: `{purpose}.mustache` or `{purpose}` (no extension for agent instructions)

**Directories:**
- Feature grouping: `{domain}/` (e.g., `services/ingestion/`, `api/endpoints/ingestion/`)
- Abstraction implementations: Group under feature (e.g., `services/vector_store/embeddings/`)
- API organization: By feature under `endpoints/` (e.g., `endpoints/ingestion/`)

**Python Classes:**
- Service classes: `{Domain}Service` (e.g., `IngestionService`, `RetrievalService`)
- Parser classes: `{Format}Parser` (e.g., `DocxParser`)
- LLM classes: `{Provider}LLMModel` (e.g., `AzureOpenAIModel`, `GeminiModel`)
- Embedding classes: `{Provider}Embeddings` (e.g., `OpenAIEmbeddings`, `JinaEmbeddings`)
- Exception classes: `{Domain}{Action}Exception` (e.g., `DocxCleaningException`)

**Function/Method Names:**
- Service operations: `{verb}_{domain}()` (e.g., `_parse_data()`, `retrieve_data()`)
- Private helpers: Leading underscore (e.g., `_extract_metadata()`, `_clean_text()`)
- Property accessors: No prefix (e.g., `.logger`, `.client`)
- Async operations: `async def` keyword

## Where to Add New Code

**New Feature (e.g., PDF parsing):**
- Implementation file: `src/services/registry/pdf_parser.py` (create new file)
- Register in: `src/services/registry/registry.py` → `_register_default_parsers()` method
- Exception handling: Add to `src/exceptions/parser_exceptions.py` if needed
- Tests: Add to `tests/test_backend_logic.py` or create `tests/test_pdf_parser.py`

**New LLM Provider (e.g., Claude):**
- Implementation: `src/services/llm/claude_model.py` (create new file)
- Inherits from: `src/services/llm/base_model.py` (BaseLLMModel)
- Integration point: `src/config/settings.py` (add config fields)
- Use in: `src/services/retrieval/retrieval.py` or service instantiating it

**New Embedding Model:**
- Implementation: `src/services/vector_store/embeddings/{provider}_embeddings.py`
- Inherits from: `src/services/vector_store/embeddings/base_embedding_service.py`
- Registration: `src/services/retrieval/retrieval.py` (instantiate alternative)
- Config: Add settings to `src/config/settings.py` (API keys, model names)

**New API Endpoint:**
- Router file: `src/api/endpoints/{feature}/router.py` (create directory if needed)
- Create `__init__.py` in feature directory
- Import and register in: `src/api/main.py` → `app.include_router()`
- Route prefix: Should follow `/api/v1/{feature}` pattern

**New Service:**
- File: `src/services/{feature}/{feature}_service.py`
- Inherit from: `Logger` mixin for logging capability
- Use dependency injection: Take dependencies as constructor parameters
- Exception handling: Define domain-specific exceptions in `src/exceptions/{feature}_exceptions.py`

**Utilities (Avoid creating new utility modules):**
- Per README design principle #9: "No utility modules"
- Instead: Add helper methods to relevant service classes
- Example: Text cleaning in `semantic_parser.py._clean_text()`, not in separate utils file

**Configuration:**
- Global settings: Add field to `Settings` class in `src/config/settings.py`
- Logging: Modify handlers in `src/config/logging.py`
- Environment variables: Define in `.env.example` with description

## Special Directories

**`src/__pycache__/`, `src/*/,__pycache__/`:**
- Purpose: Python bytecode cache
- Generated: Yes (automatic)
- Committed: No (in .gitignore)

**`logs/`:**
- Purpose: Application runtime logs
- Generated: Yes (created by logging setup)
- Committed: No (in .gitignore)
- Contents: Daily-rotated log files `AI_Contract_Review_YYYYMMDD.log`, `errors.log`

**`.vscode/`:**
- Purpose: IDE configuration
- Generated: No
- Committed: Yes
- Contains: Editor settings, debug configurations

**`.git/`:**
- Purpose: Version control metadata
- Generated: Yes (git init)
- Committed: Yes (system directory)

**`.pytest_cache/`:**
- Purpose: pytest runtime cache
- Generated: Yes (when running tests)
- Committed: No (in .gitignore)

**`.planning/codebase/`:**
- Purpose: GSD codebase analysis documents
- Generated: Yes (by GSD agent)
- Committed: Yes
- Contents: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, CONCERNS.md

---

*Structure analysis: 2026-02-09*
