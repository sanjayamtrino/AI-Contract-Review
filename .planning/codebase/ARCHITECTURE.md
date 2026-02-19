# Architecture

**Analysis Date:** 2026-02-19

## Pattern Overview

**Overall:** Layered Monolith with Agent-Based Orchestration (RAG pipeline)

**Key Characteristics:**
- Single FastAPI application serving all functionality through a REST API
- Two-tier LLM-based routing: an **orchestrator** routes user queries to **agents**, which in turn route to **tools**
- Session-scoped in-memory data stores (FAISS vector index + chunk store per session)
- Prompt-driven LLM interactions using versioned Mustache templates
- Pydantic models enforce structured JSON output from every LLM call
- Singleton service container initialised at app startup; accessed globally via `get_service_container()`

## Layers

**API Layer:**
- Purpose: HTTP endpoints, request validation, session header extraction
- Location: `src/api/`
- Contains: FastAPI routers, session utility helpers, the `main.py` application entry point
- Depends on: Orchestrator, Services (via ServiceContainer), Tools (some direct retrieval endpoints)
- Used by: External HTTP clients

**Orchestrator Layer:**
- Purpose: Route a user query to the correct agent using OpenAI function-calling
- Location: `src/orchestrator/main.py`
- Contains: `run_orchestrator()`, agent registry (`AGENT_REGISTRY`), agent tool definitions (`AGENT_DEFINITIONS`)
- Depends on: Agents, LLM service (Azure OpenAI), Prompt loader
- Used by: API layer (`src/api/endpoints/orchestrator/router.py`)

**Agent Layer:**
- Purpose: Domain-specific decision-making; each agent decides which tools to invoke
- Location: `src/agents/`
- Contains: `doc_information.py`, `playbook_review.py`, `playbook_review_v2.py`
- Depends on: Tools, LLM service, Prompt loader, Playbook loader, Schemas
- Used by: Orchestrator

**Tools Layer:**
- Purpose: Execute discrete, atomic operations (LLM calls, embedding searches, report generation)
- Location: `src/tools/`
- Contains: `summarizer.py`, `key_details.py`, `clause_extractor.py`, `clause_comparator.py`, `review_report_generator.py`, `rule_risk_assessor.py`, `embedding_clause_matcher.py`, `master_playbook_assessor.py`
- Depends on: LLM service, SessionManager, Embedding service, Vector store, Playbook loader, Schemas
- Used by: Agents

**Service Layer:**
- Purpose: Core infrastructure services shared across agents and tools
- Location: `src/services/`
- Contains: Ingestion pipeline, Retrieval pipeline, LLM model abstractions, Session management, Embedding services, Vector store management, Prompt loading, Playbook rule loading
- Depends on: Config, Schemas, Exceptions
- Used by: All layers above

**Schema Layer:**
- Purpose: Pydantic models for all structured data (LLM responses, parsing results, playbook rules)
- Location: `src/schemas/`
- Contains: `registry.py` (Chunk, ParseResult), `playbook.py` (rules, clauses, risk assessments, review reports), `playbook_v2.py` (V2/V3 rule models, MasterPlaybookReviewResponse), `query_rewriter.py`
- Depends on: Nothing (pure data models)
- Used by: Every layer

**Config Layer:**
- Purpose: Application settings, logging setup
- Location: `src/config/`
- Contains: `settings.py` (Pydantic Settings from `.env`), `logging.py` (dict-config logging, `Logger` mixin)
- Depends on: `.env` file
- Used by: Every layer

**Exceptions Layer:**
- Purpose: Named, predefined exception hierarchy
- Location: `src/exceptions/`
- Contains: `base_exception.py` (AppException), `parser_exceptions.py`, `ingestion_exceptions.py`
- Depends on: Nothing
- Used by: Service and Tool layers

## Data Flow

**Document Ingestion Flow:**

1. Client sends POST `/api/v1/ingest/` with a DOCX file and `X-Session-ID` header
2. `src/api/endpoints/ingestion/router.py` reads file bytes, resolves session via `SessionManager.get_or_create_session()`
3. `IngestionService._parse_data()` in `src/services/ingestion/ingestion.py` delegates to the parser registry
4. `ParserRegistry.get_parser()` in `src/services/registry/registry.py` returns the active `DocxParser` (currently `src/services/registry/semantic_parser.py`)
5. `DocxParser.parse()` performs: clean document -> extract metadata -> extract paragraphs -> extract tables -> semantic chunking (embedding + cosine similarity breakpoints) -> embed each chunk via `BGEEmbeddingService` -> index into session's FAISS store
6. `index_chunks_in_session()` in `src/services/vector_store/manager.py` stores `Chunk` objects in `session_data.chunk_store` and records document metadata in `session_data.documents`
7. Returns `ParseResult` to client

**Orchestrator Query Flow (primary path):**

1. Client sends POST `/api/v1/orchestrator/query/` with `query` param and `X-Session-ID` header
2. `src/api/endpoints/orchestrator/router.py` calls `run_orchestrator(query, session_id)`
3. `src/orchestrator/main.py` sends the query + system prompt to Azure OpenAI with `tools=AGENT_DEFINITIONS` (function calling)
4. LLM returns a tool_call selecting one of: `doc_information_agent`, `playbook_review_agent`, `playbook_review_v2_agent`
5. Orchestrator dispatches to the selected agent's `run()` function
6. Agent sends query + its own system prompt to Azure OpenAI with `tools=TOOL_DEFINITIONS`
7. LLM returns tool_call(s); agent executes the selected tools
8. Tool results (Pydantic models) are serialized and returned through agent -> orchestrator -> API

**Playbook Review V1 Flow (via `playbook_review` agent):**

1. Agent selects `full_playbook_review`, `fallback_rereview`, or `general_ai_review`
2. `full_playbook_review` calls `extract_clauses()` (LLM) -> groups clauses by category -> `compare_clause_batch()` (LLM, concurrent batches via `asyncio.gather`) -> `generate_review_report()` (LLM)
3. Returns `ReviewReportResponse` with risk assessments, missing clauses, and cross-clause conflicts

**Playbook Review V2 Flow (via `playbook_review_v2` agent):**

1. Agent selects `full_playbook_review_v2` or `single_rule_review`
2. `full_playbook_review_v2` loads V3 rules -> for each rule, calls `match_paragraphs_to_rule_v3()` (embedding similarity search, no LLM) -> `assess_all_rules()` (single batched LLM call)
3. Returns `MasterPlaybookReviewResponse` with per-rule analysis including matched paragraphs and clauses

**RAG Retrieval Flow (direct query endpoint):**

1. Client sends POST `/api/v1/query/` with `query` and `X-Session-ID`
2. `RetrievalService.retrieve_data()` rewrites query via LLM -> generates embedding -> searches session FAISS index -> returns ranked chunks
3. Chunks are passed as context to LLM for final answer generation

**State Management:**
- All session state is in-memory, managed by `SessionManager` in `src/services/session_manager.py`
- Each session holds: `FAISSVectorStore`, `chunk_store: Dict[int, Chunk]`, `documents: Dict[str, Dict]`, TTL metadata
- Sessions are identified by `X-Session-ID` HTTP header (client-provided)
- Background `asyncio.Task` runs cleanup of expired sessions based on configurable TTL
- There is also a legacy global `_chunk_store` and singleton FAISS store in `src/services/vector_store/manager.py` (deprecated but still referenced)

## Key Abstractions

**BaseLLMModel:**
- Purpose: Abstract interface for all LLM model implementations
- Location: `src/services/llm/base_model.py`
- Implementations: `AzureOpenAIModel` (`src/services/llm/azure_openai_model.py`), `GeminiModel` (`src/services/llm/gemini_model.py` -- currently commented out/disabled)
- Pattern: Strategy pattern with `generate()` method accepting prompt template, context dict, and Pydantic response_model; returns validated Pydantic instance

**BaseParser:**
- Purpose: Abstract interface for document parsers
- Location: `src/services/registry/base_parser.py`
- Implementations: `DocxParser` in `src/services/registry/doc_parser.py` (recursive text splitter) and `src/services/registry/semantic_parser.py` (semantic chunking with embeddings)
- Pattern: Strategy pattern; selected via `ParserRegistry`

**BaseEmbeddingService:**
- Purpose: Abstract interface for embedding generation
- Location: `src/services/vector_store/embeddings/base_embedding_service.py`
- Active Implementation: `BGEEmbeddingService` in `src/services/vector_store/embeddings/embedding_service.py`
- Alternative implementations exist but are commented out: `HuggingFaceEmbeddingService`, `OpenAIEmbeddings`, `JinaEmbeddings`, `GeminiEmbeddingService`, `QwenEmbeddings`

**BaseVectorStore:**
- Purpose: Abstract interface for vector databases
- Location: `src/services/vector_store/base_store.py`
- Implementation: `FAISSVectorStore` in `src/services/vector_store/faiss_db.py`
- Pattern: Inner product (cosine similarity via L2 normalization) search

**ServiceContainer:**
- Purpose: Application-wide dependency injection container (singleton)
- Location: `src/dependencies.py`
- Pattern: Lazy-initialized singleton; `get_service_container()` returns the global instance; `initialize_dependencies()` called at FastAPI lifespan startup
- Holds: `IngestionService`, `RetrievalService`, `AzureOpenAIModel`, `BGEEmbeddingService`, `SessionManager`, `Settings`

**SessionData / SessionManager:**
- Purpose: Per-session isolated data stores with TTL-based lifecycle
- Location: `src/services/session_manager.py`
- Each `SessionData` contains its own `FAISSVectorStore` and `chunk_store`
- Thread-safe via `threading.Lock`; async cleanup via `asyncio.Task`

**Prompt System:**
- Purpose: Versioned, template-based prompt management
- Location: `src/services/prompts/v1/` (Mustache templates), `src/services/prompts/v1/__init__.py` (loader)
- Pattern: Load `.mustache` file by name -> render with context dict using `chevron` (or `pystache` in `AzureOpenAIModel`)
- 16 prompt templates covering orchestrator routing, agent instructions, clause extraction, comparison, risk assessment, report generation, and more

**Playbook Rules:**
- Purpose: Static rule definitions the contract is reviewed against
- Location: `src/data/default_playbook_rules.json` (V1 rules), `src/data/playbook_rules_v3.json` (V3 rules)
- Loader: `src/services/playbook_loader.py` (caches rules in module-level lists)
- Schemas: `PlaybookRule` in `src/schemas/playbook.py`, `PlaybookRuleV3` in `src/schemas/playbook_v2.py`

## Entry Points

**FastAPI Application:**
- Location: `src/api/main.py`
- Triggers: `poetry run python -m src.api.main` or `contract-api` (poetry script)
- Responsibilities: Create FastAPI app, attach lifespan (init/shutdown services), register routers, add timing middleware
- Runs on: `uvicorn` at `{api_host}:{api_port}` (default `localhost:8000`)

**API Routers:**

| Prefix | Router Location | Endpoints |
|--------|----------------|-----------|
| `/api/v1` | `src/api/endpoints/ingestion/router.py` | `POST /ingest/` |
| `/api/v1` | `src/api/endpoints/retrieval/router.py` | `POST /query/`, `GET /summarizer`, `GET /key-details` |
| `/api/v1` | `src/api/endpoints/orchestrator/router.py` | `POST /orchestrator/query/` |
| `/api/v1/admin` | `src/api/endpoints/admin/router.py` | `GET /sessions/`, `GET /sessions/{id}`, `DELETE /sessions/{id}`, `POST /sessions/cleanup`, `GET /health/` |

**Test Script (standalone):**
- Location: `test_master_prompt.py` (root)
- Purpose: Ad-hoc test of the master playbook review prompt

## Error Handling

**Strategy:** Named exception hierarchy + per-layer catch-and-wrap

**Patterns:**
- All custom exceptions inherit from `AppException` (`src/exceptions/base_exception.py`)
- Parser exceptions are granular: `DocxCleaningException`, `DocxMetadataExtractionException`, `DocxParagraphExtractionException`, `DocxTableExtractionException` (`src/exceptions/parser_exceptions.py`)
- Ingestion exceptions: `ParserNotFound` (`src/exceptions/ingestion_exceptions.py`)
- LLM layer catches `json.JSONDecodeError` and `pydantic.ValidationError`, wraps them in `ValueError`
- Agent layer catches tool exceptions per-tool-call and returns `{"error": str(e)}` in the results dict
- Orchestrator layer catches agent exceptions per-agent-call similarly
- API layer currently does NOT have global exception handlers or middleware for converting exceptions to HTTP error responses (returns raw dicts with `"error"` keys instead)
- `ServiceContainer` property getters raise `RuntimeError` if accessed before initialization

## Cross-Cutting Concerns

**Logging:**
- Configured in `src/config/logging.py` via `logging.config.dictConfig`
- Three handlers: console (INFO, simple format), file (DEBUG, detailed format, rotating 10MB), error_file (ERROR, JSON format, rotating 10MB)
- Log files: `./logs/AI_Contract_Review_{YYYYMMDD}.log` and `./logs/errors.log`
- `Logger` mixin class provides `self.logger` property to any class that inherits it
- Standalone logger via `get_logger(name)` returning `logging.getLogger(f"AI_Contract.{name}")`

**Validation:**
- All LLM responses validated against Pydantic models via `response_model.model_validate()`
- Uses OpenAI `json_schema` response format with the Pydantic model's JSON schema
- Request validation handled by FastAPI's built-in Pydantic integration
- Session ID validated in `src/api/session_utils.py` (raises HTTP 400 if missing)

**Authentication:**
- No authentication or authorization implemented
- Session identification is via client-provided `X-Session-ID` header (no auth, no validation of ownership)

**Prompt Rendering:**
- Two rendering paths exist: `chevron.render()` in `src/services/prompts/v1/__init__.py` and `pystache.Renderer(escape=lambda u: u)` in `AzureOpenAIModel.render_prompt_template()`
- Both disable HTML escaping to preserve legal text with special characters

---

*Architecture analysis: 2026-02-19*
