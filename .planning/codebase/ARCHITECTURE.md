# Architecture

**Analysis Date:** 2026-02-09

## Pattern Overview

**Overall:** Layered RAG (Retrieval-Augmented Generation) Architecture with Service-Oriented Design

**Key Characteristics:**
- Multi-layer separation: API → Services → Domain Logic
- Async-first architecture throughout
- Plugin-based parser registry pattern
- Singleton vector store management with thread-safe access
- Mustache-templated prompt management for LLM interactions
- Separation of concerns: ingestion, retrieval, and orchestration pipelines

## Layers

**API Layer (Presentation):**
- Purpose: HTTP endpoint exposure and request/response handling
- Location: `src/api/`
- Contains: FastAPI routers, HTTP request/response schemas
- Depends on: Service layer, schemas
- Used by: External clients via HTTP

**Service Layer (Business Logic):**
- Purpose: Core application logic coordinating multiple subsystems
- Location: `src/services/`
- Contains: Ingestion, retrieval, LLM coordination, embedding generation
- Depends on: Schema layer, vector store, config
- Used by: API layer, orchestrator

**Domain/Infrastructure Layer:**
- Purpose: External integrations and concrete implementations
- Location: `src/services/{llm,vector_store,registry}/`
- Contains: LLM clients, embedding models, document parsers, vector databases
- Depends on: Schema layer
- Used by: Service layer

**Configuration Layer:**
- Purpose: Application settings and logging setup
- Location: `src/config/`
- Contains: Settings from environment, logging configuration
- Depends on: None
- Used by: All layers

**Schema Layer:**
- Purpose: Data validation and type contracts
- Location: `src/schemas/`
- Contains: Pydantic models for input/output validation
- Depends on: None
- Used by: All layers

## Data Flow

**Document Ingestion Flow:**

1. User uploads document via `POST /api/v1/ingest/`
2. `IngestionService._parse_data()` receives BytesIO file
3. `ParserRegistry.get_parser()` returns appropriate parser (currently DOCX only)
4. `DocxParser.parse()` processes document:
   - Extracts paragraphs, tables, metadata
   - Generates embeddings for semantic chunking using `BGEEmbeddingService`
   - Computes cosine similarities between consecutive paragraphs
   - Creates chunk boundaries based on similarity threshold
   - Merges orphan chunks (< 5 words)
5. `Chunk` objects created with embeddings metadata
6. `index_chunks()` stores chunks in thread-safe global `_chunk_store` (Dict)
7. Embeddings indexed in FAISS vector store via `FAISSVectorStore.index_embedding()`
8. Returns `ParseResult` with chunks, metadata, processing time

**Document Retrieval Flow:**

1. User queries via `POST /api/v1/query/` with search string
2. `RetrievalService.retrieve_data()` processes query:
   - Rewrites query using `LLM.generate()` with query_rewriter prompt
   - For each rewritten query, generates embedding via `BGEEmbeddingService`
   - Searches FAISS index for top-k similar vectors
   - Retrieves actual chunk content from `_chunk_store` by index
   - Ranks results by cosine similarity score
   - Deduplicates using index-based tracking
3. Returns ranked chunks with metadata, similarity scores, matched queries
4. `AzureOpenAIModel.generate()` fills llm_response template with context
5. Returns LLM-generated response with retrieved chunks

**Orchestrator Agent Flow:**

1. `OpenAIChat` extends `BaseChatClient` from agent_framework
2. Configured with tools: `get_summary()`, `get_location()`, `get_key_information()`
3. Loads orchestrator_prompt from `src/services/prompts/v1/orchestrator_prompt`
4. Receives user query → passes to Azure OpenAI with tool definitions
5. Tool execution loop (max 2 iterations):
   - Check if model requested tool calls
   - Execute matching tools from registry
   - Add tool results to message history
   - Continue conversation
6. Returns final assistant text response

**State Management:**

- **Vector chunks:** Thread-safe global dictionary `_chunk_store` with singleton `Lock`
- **FAISS index:** Singleton instance `_instance` created per embedding dimension
- **LLM models:** Initialized per-service with cached OpenAI/Azure clients
- **Settings:** LRU-cached singleton `Settings()` from environment
- **Logging:** Initialized once on application startup, handlers configured for console and rotating files

## Key Abstractions

**BaseParser (Abstract Base):**
- Purpose: Define parser contract for document format handlers
- Examples: `DocxParser` (semantic chunking DOCX), future PDF/TXT parsers
- Pattern: Registry pattern + Strategy pattern for format selection
- Implements: `parse()`, `clean_document()`, `is_healthy()`
- Custom exception hierarchy: `DocxCleaningException`, `DocxMetadataExtractionException`, etc.

**BaseLLMModel (Abstract Base):**
- Purpose: Unified interface for LLM providers
- Examples: `AzureOpenAIModel`, `GeminiModel` (commented out)
- Pattern: Strategy pattern for model selection
- Implements: `generate(prompt, context, response_model)` → returns Pydantic model instance
- Template rendering: Mustache templates for prompts before passing to LLM

**BaseEmbeddingService (Abstract Base):**
- Purpose: Define embedding generation contract
- Examples: `BGEEmbeddingService` (active), `OpenAIEmbeddings`, `JinaEmbeddings`, `HuggingFaceEmbeddingService` (commented/inactive)
- Pattern: Strategy pattern for embedding model selection
- Implements: `generate_embeddings(text, task)`, `get_embedding_dimensions()`

**ParserRegistry:**
- Purpose: Registry pattern implementation for pluggable parsers
- Stores: `Dict[str, BaseParser]` mapping format names to instances
- Currently registered: DOCX parser only
- Design: Supports dynamic registration via `register_parser(name, parser_class)`

**FAISSVectorStore Manager:**
- Purpose: Singleton pattern for thread-safe vector store management
- Maintains: Global `_instance`, `_instance_dimension`, `_lock`, `_chunk_store`, `_chunk_counter`
- Functions: `get_faiss_vector_store()`, `index_chunks()`, `get_chunks()`, `get_all_chunks()`, `reset_chunks()`
- Thread safety: All operations protected by `threading.Lock()`

**Chunk Schema:**
- Purpose: Type-safe representation of document fragments
- Contains: Content, metadata, embedding model name, created timestamp, source document/chunk indices
- Validation: Pydantic BaseModel with field descriptions
- Methods: `get_content_hash()` for deduplication via SHA256

## Entry Points

**API Entry Point:**
- Location: `src/api/main.py`
- Triggers: `poetry run python -m src.api.main` or `contract-api` command
- Responsibilities:
  - Create FastAPI app with title/version
  - Include ingestion and retrieval routers under `/api/v1` prefix
  - Start uvicorn server on configured host/port
  - Initialize logging and settings

**Orchestrator Agent Entry Point:**
- Location: `src/orchestrator/main.py`
- Triggers: Direct execution or agent framework invocation
- Responsibilities:
  - Create OpenAIChat custom client extending BaseChatClient
  - Configure agent with name, instructions (orchestrator_prompt), and tools
  - Route user queries through agentic loop with tool execution
  - Return final response

**Ingestion Endpoint:**
- Location: `src/api/endpoints/ingestion/router.py`
- Route: `POST /api/v1/ingest/`
- Request: Multipart file upload
- Response: `ParseResult` (success, chunks, metadata, processing_time, error_message)

**Retrieval Endpoint:**
- Location: `src/api/endpoints/retrieval/router.py`
- Route: `POST /api/v1/query/`
- Request: Query string
- Response: LLM result + retrieved chunks with similarity scores

## Error Handling

**Strategy:** Explicit exception hierarchy with contextual error messages

**Patterns:**

1. **Parser Exceptions** (`src/exceptions/parser_exceptions.py`):
   - `DocxCleaningException` - Document preprocessing failures
   - `DocxMetadataExtractionException` - Metadata parsing failures
   - `DocxParagraphExtractionException` - Paragraph extraction failures
   - `DocxTableExtractionException` - Table parsing failures
   - Chained with `from e` to preserve stack traces

2. **Base Exception** (`src/exceptions/base_exception.py`):
   - `AppException` - Root application exception with message preservation

3. **Service-Level Error Handling**:
   - `IngestionService._parse_data()`: Returns `ParseResult` with `error_message` field; logs errors
   - `RetrievalService.retrieve_data()`: Validates query, catches exceptions, re-raises as `ValueError` with context
   - `AzureOpenAIModel.generate()`: Catches `JSONDecodeError`, `ValidationError`, logs, re-raises as `ValueError`
   - `ParserRegistry.register_parser()`: Raises `ValueError` if parser already registered

4. **HTTP-Level Handling**:
   - Ingestion endpoint: Returns HTTP 500 if no parser found or processing fails
   - Retrieval endpoint: Returns HTTP 500 on empty query or retrieval exceptions

## Cross-Cutting Concerns

**Logging:**
- Framework: Python standard `logging` module
- Configuration: `src/config/logging.py` with dictConfig
- Mixin: `Logger` class provides `.logger` property using class name
- Output: Console (INFO), rotating file (DEBUG), rotating error file (JSON format)
- Format: "timestamp - logger_name - level - file:line - function - message"
- Files: Daily rotation in `./logs/` directory

**Validation:**
- Framework: Pydantic BaseModel with Field descriptions
- Input schemas: `UploadFile`, query strings with type hints
- Output schemas: `ParseResult`, `Chunk`, `QueryRewriterResponse`, custom response models
- File/query sanitization in parsers and services (whitespace cleanup, empty check)

**Authentication:**
- Current: None at HTTP level
- LLM auth: API keys from environment via `Settings` (azure_openai_api_key, openai_api_key, etc.)
- Orchestrator: No explicit auth; direct agent invocation

---

*Architecture analysis: 2026-02-09*
