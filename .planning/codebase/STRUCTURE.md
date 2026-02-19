# Codebase Structure

**Analysis Date:** 2026-02-19

## Directory Layout

```
AI-Contract-Review/
├── src/                        # All application source code
│   ├── __init__.py
│   ├── dependencies.py         # ServiceContainer (DI singleton) + init/shutdown
│   ├── agents/                 # LLM-driven agent modules
│   │   ├── __init__.py
│   │   ├── doc_information.py  # Document info agent (summary, key details)
│   │   ├── playbook_review.py  # Playbook review agent V1 (clause extraction)
│   │   └── playbook_review_v2.py # Playbook review agent V2 (embedding-based)
│   ├── api/                    # FastAPI HTTP layer
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI app creation and startup
│   │   ├── session_utils.py    # X-Session-ID header extraction helpers
│   │   └── endpoints/          # Router modules grouped by domain
│   │       ├── __init__.py
│   │       ├── admin/
│   │       │   ├── __init__.py
│   │       │   └── router.py   # Session management + health check endpoints
│   │       ├── ingestion/
│   │       │   ├── __init__.py
│   │       │   └── router.py   # Document upload endpoint
│   │       ├── orchestrator/
│   │       │   ├── __init__.py
│   │       │   └── router.py   # Main orchestrator query endpoint
│   │       └── retrieval/
│   │           ├── __init__.py
│   │           └── router.py   # RAG query, summary, key-details endpoints
│   ├── config/                 # Application configuration
│   │   ├── __init__.py
│   │   ├── settings.py         # Pydantic Settings (env-based config)
│   │   └── logging.py          # Logging setup + Logger mixin class
│   ├── data/                   # Static data files
│   │   ├── default_playbook_rules.json  # V1 playbook rules
│   │   └── playbook_rules_v3.json       # V3 playbook rules
│   ├── exceptions/             # Custom exception classes
│   │   ├── __init__.py
│   │   ├── base_exception.py   # AppException base class
│   │   ├── parser_exceptions.py # DOCX parser-specific exceptions
│   │   └── ingestion_exceptions.py # Ingestion service exceptions
│   ├── orchestrator/           # Query routing orchestrator
│   │   ├── __init__.py
│   │   └── main.py             # run_orchestrator(), AGENT_REGISTRY, AGENT_DEFINITIONS
│   ├── schemas/                # Pydantic data models
│   │   ├── __init__.py
│   │   ├── registry.py         # Chunk, ParseResult
│   │   ├── playbook.py         # PlaybookRule, ExtractedClause, ClauseRiskAssessment, ReviewReportResponse
│   │   ├── playbook_v2.py      # PlaybookRuleV3, RuleAnalysis, MasterPlaybookReviewResponse
│   │   └── query_rewriter.py   # QueryRewriterResponse
│   ├── services/               # Core business logic and infrastructure
│   │   ├── __init__.py
│   │   ├── session_manager.py  # SessionData, SessionManager (per-session stores + TTL)
│   │   ├── playbook_loader.py  # Load/cache playbook rules from JSON
│   │   ├── ingestion/
│   │   │   └── ingestion.py    # IngestionService (orchestrates parsing + indexing)
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   ├── base_model.py   # BaseLLMModel abstract class
│   │   │   ├── azure_openai_model.py # AzureOpenAIModel (active LLM)
│   │   │   └── gemini_model.py # GeminiModel (alternative, currently disabled)
│   │   ├── prompts/
│   │   │   ├── __init__.py
│   │   │   └── v1/             # Version 1 prompt templates
│   │   │       ├── __init__.py # load_prompt() helper
│   │   │       └── *.mustache  # 16 Mustache prompt templates
│   │   ├── registry/
│   │   │   ├── base_parser.py  # BaseParser abstract class
│   │   │   ├── doc_parser.py   # DocxParser (recursive text splitter approach)
│   │   │   ├── semantic_parser.py # DocxParser (semantic chunking approach -- active)
│   │   │   └── registry.py     # ParserRegistry (parser factory)
│   │   ├── retrieval/
│   │   │   ├── __init__.py
│   │   │   ├── retrieval.py    # RetrievalService (query rewriting + RAG search)
│   │   │   └── query_rewriter.py # Empty placeholder
│   │   └── vector_store/
│   │       ├── __init__.py
│   │       ├── base_store.py   # BaseVectorStore abstract class
│   │       ├── faiss_db.py     # FAISSVectorStore (in-memory FAISS)
│   │       ├── manager.py      # Singleton store + chunk indexing helpers
│   │       └── embeddings/
│   │           ├── __init__.py
│   │           ├── base_embedding_service.py  # BaseEmbeddingService ABC
│   │           ├── embedding_service.py       # BGEEmbeddingService (active), HuggingFaceEmbeddingService
│   │           ├── gemini_embeddings.py       # GeminiEmbeddingService (alt)
│   │           ├── jina_embeddings.py         # JinaEmbeddings (alt)
│   │           ├── openai_embeddings.py       # OpenAIEmbeddings (alt)
│   │           └── qwen_embeddings.py         # QwenEmbeddings (alt)
│   └── tools/                  # Discrete LLM/search tool implementations
│       ├── __init__.py
│       ├── summarizer.py       # get_summary() — document summarization
│       ├── key_details.py      # get_key_details() — structured data extraction
│       ├── clause_extractor.py # extract_clauses() — LLM clause extraction
│       ├── clause_comparator.py # compare_clause_batch() — clause vs rule comparison
│       ├── review_report_generator.py # generate_review_report() — final report
│       ├── rule_risk_assessor.py # assess_rule_risk() — per-rule risk (V2)
│       ├── embedding_clause_matcher.py # match_paragraphs_to_rule() — embedding search
│       └── master_playbook_assessor.py # assess_all_rules() — batch LLM assessment
├── tests/                      # Test directory
│   ├── __init__.py
│   ├── test_embeddings.py      # Embedding similarity tests
│   └── test_backend_logic.py   # Backend logic tests
├── docs/                       # Documentation assets
│   └── architecture_overview.png
├── logs/                       # Runtime log files (generated, gitignored)
├── .env                        # Environment variables (secrets, config)
├── .env.example                # Template for .env setup
├── pyproject.toml              # Poetry project config, tool settings
├── poetry.lock                 # Locked dependency versions
├── Makefile                    # Build/run shortcuts (empty)
├── README.md                   # Project documentation
├── test_master_prompt.py       # Ad-hoc test script (root level)
└── test_master_prompt_output.json # Ad-hoc test output
```

## Directory Purposes

**`src/agents/`:**
- Purpose: LLM-driven agent modules that receive a query and decide which tools to call
- Contains: One Python file per agent; each defines `TOOL_REGISTRY`, `TOOL_DEFINITIONS`, and an `async run()` function
- Key files: `doc_information.py`, `playbook_review.py`, `playbook_review_v2.py`

**`src/api/`:**
- Purpose: FastAPI HTTP interface; defines all routes, request handling, and ASGI application
- Contains: `main.py` (app factory/entry), `session_utils.py` (header helpers), `endpoints/` subdirectories with `router.py` files
- Key files: `src/api/main.py`, `src/api/endpoints/orchestrator/router.py`

**`src/config/`:**
- Purpose: Centralised configuration and logging setup
- Contains: `settings.py` (environment-based Pydantic Settings), `logging.py` (dict-config + Logger mixin)
- Key files: `src/config/settings.py`

**`src/data/`:**
- Purpose: Static data files loaded at runtime (playbook rules as JSON)
- Contains: `default_playbook_rules.json`, `playbook_rules_v3.json`

**`src/exceptions/`:**
- Purpose: Named exception hierarchy for the entire application
- Contains: Base class + domain-specific exceptions
- Key files: `src/exceptions/base_exception.py`, `src/exceptions/parser_exceptions.py`

**`src/orchestrator/`:**
- Purpose: Top-level query router using LLM function-calling to select agents
- Contains: Single `main.py` with `run_orchestrator()`, agent registry, agent definitions
- Key files: `src/orchestrator/main.py`

**`src/schemas/`:**
- Purpose: All Pydantic models used as LLM response schemas, API contracts, and internal data structures
- Contains: Domain-grouped schema files
- Key files: `src/schemas/registry.py` (Chunk, ParseResult), `src/schemas/playbook.py` (all V1 playbook types), `src/schemas/playbook_v2.py` (V2/V3 types)

**`src/services/`:**
- Purpose: Core infrastructure and business logic services
- Contains: Subdirectories for each service domain

**`src/services/ingestion/`:**
- Purpose: Document parsing orchestration
- Key files: `src/services/ingestion/ingestion.py`

**`src/services/llm/`:**
- Purpose: LLM model abstractions and implementations
- Key files: `src/services/llm/base_model.py`, `src/services/llm/azure_openai_model.py`

**`src/services/prompts/v1/`:**
- Purpose: Versioned Mustache prompt templates and a loader function
- Contains: `__init__.py` (has `load_prompt()`) and 16 `.mustache` template files
- Key files: `src/services/prompts/v1/__init__.py`, `src/services/prompts/v1/orchestrator_prompt.mustache`

**`src/services/registry/`:**
- Purpose: Document parser implementations and a registry to select the correct one
- Key files: `src/services/registry/semantic_parser.py` (active parser), `src/services/registry/registry.py`

**`src/services/retrieval/`:**
- Purpose: RAG retrieval pipeline (query rewriting + vector search + chunk fetching)
- Key files: `src/services/retrieval/retrieval.py`

**`src/services/vector_store/`:**
- Purpose: Vector database abstraction, FAISS implementation, and chunk store management
- Key files: `src/services/vector_store/faiss_db.py`, `src/services/vector_store/manager.py`

**`src/services/vector_store/embeddings/`:**
- Purpose: Embedding model abstractions and implementations
- Key files: `src/services/vector_store/embeddings/embedding_service.py` (BGEEmbeddingService -- active)

**`src/tools/`:**
- Purpose: Atomic tool functions invoked by agents; each performs one LLM call or search operation
- Contains: One file per tool function
- Key files: `src/tools/summarizer.py`, `src/tools/clause_extractor.py`, `src/tools/master_playbook_assessor.py`

**`tests/`:**
- Purpose: Test files
- Contains: `test_embeddings.py`, `test_backend_logic.py`

## Key File Locations

**Entry Points:**
- `src/api/main.py`: FastAPI application creation, lifespan management, router registration, uvicorn startup
- `src/orchestrator/main.py`: Central query dispatcher; receives user query and routes to correct agent

**Configuration:**
- `src/config/settings.py`: All environment-based settings (API host/port, chunk size, LLM keys, model names, session TTL)
- `src/config/logging.py`: Logging configuration and `Logger` mixin
- `pyproject.toml`: Poetry project config, tool settings (black, isort, mypy, flake8)
- `.env`: Runtime secrets and configuration values (existence noted only)
- `.env.example`: Template showing required env vars

**Core Logic:**
- `src/dependencies.py`: `ServiceContainer` class and global init/shutdown functions
- `src/services/session_manager.py`: Per-session state management with TTL cleanup
- `src/services/ingestion/ingestion.py`: Document ingestion orchestration
- `src/services/retrieval/retrieval.py`: RAG retrieval with query rewriting
- `src/services/registry/semantic_parser.py`: Active DOCX parser with semantic chunking
- `src/services/llm/azure_openai_model.py`: Primary LLM model (all structured generation)
- `src/services/playbook_loader.py`: Playbook rule loading and caching

**Agents:**
- `src/agents/doc_information.py`: Handles document understanding queries
- `src/agents/playbook_review.py`: V1 playbook review (clause extraction + comparison)
- `src/agents/playbook_review_v2.py`: V2 playbook review (embedding-based matching)

**Tools:**
- `src/tools/summarizer.py`: Document summarization
- `src/tools/key_details.py`: Structured key detail extraction
- `src/tools/clause_extractor.py`: Clause extraction from contract text
- `src/tools/clause_comparator.py`: Clause vs playbook rule comparison
- `src/tools/review_report_generator.py`: Final review report synthesis
- `src/tools/embedding_clause_matcher.py`: Embedding-based paragraph-to-rule matching
- `src/tools/rule_risk_assessor.py`: Per-rule risk assessment
- `src/tools/master_playbook_assessor.py`: Batch rule assessment (single LLM call)

**Schemas:**
- `src/schemas/registry.py`: `Chunk`, `ParseResult`
- `src/schemas/playbook.py`: `PlaybookRule`, `ExtractedClause`, `ClauseExtractionResponse`, `ClauseRiskAssessment`, `BatchComparisonResponse`, `ReviewReportResponse`, `MissingClause`, `CrossClauseConflict`, `RiskLevel`
- `src/schemas/playbook_v2.py`: `PlaybookRuleV3`, `RuleAnalysis`, `MatchedParagraph`, `MatchedClause`, `MasterPlaybookReviewResponse`, `PlaybookReviewV2Response`
- `src/schemas/query_rewriter.py`: `QueryRewriterResponse`, `Query`

**Prompt Templates (all in `src/services/prompts/v1/`):**
- `orchestrator_prompt.mustache`: Orchestrator routing instructions
- `doc_information_agent_prompt.mustache`: Doc info agent system prompt
- `playbook_review_agent_prompt.mustache`: Playbook review V1 agent prompt
- `playbook_review_v2_agent_prompt.mustache`: Playbook review V2 agent prompt
- `summary_prompt_template.mustache`: Document summarization
- `key_details_prompt_template_v1.mustache`: Key details extraction
- `clause_extraction_prompt.mustache`: Clause extraction from contract
- `clause_comparison_prompt.mustache`: Clause vs rule comparison
- `review_report_prompt.mustache`: Review report generation
- `rule_risk_assessment_prompt.mustache`: Per-rule risk assessment
- `master_playbook_review_prompt.mustache`: Batch master review
- `query_rewriter.mustache`: Query rewriting for RAG
- `llm_response.mustache`: Generic LLM response for RAG queries
- `doc_chat_prompt.mustache`: Document chat prompt

**Static Data:**
- `src/data/default_playbook_rules.json`: V1 playbook rules (category, name, standard/fallback positions, canned responses)
- `src/data/playbook_rules_v3.json`: V3 playbook rules (title, instruction, description)

**Testing:**
- `tests/test_embeddings.py`: Embedding similarity tests
- `tests/test_backend_logic.py`: Backend logic tests
- `test_master_prompt.py`: Root-level ad-hoc test script

## Naming Conventions

**Files:**
- Snake_case for all Python files: `azure_openai_model.py`, `session_manager.py`
- Router files always named `router.py` inside their domain directory
- Schema files named by domain: `playbook.py`, `registry.py`
- Prompt templates: `{descriptive_name}.mustache`
- JSON data files: `{descriptive_name}.json`

**Directories:**
- All lowercase, snake_case: `vector_store/`, `endpoints/`
- Endpoint directories named by domain: `ingestion/`, `retrieval/`, `orchestrator/`, `admin/`
- Version namespacing for prompts: `prompts/v1/`

## Where to Add New Code

**New Agent:**
- Create `src/agents/{agent_name}.py` following the pattern in `src/agents/doc_information.py`:
  - Define `TOOL_REGISTRY`, `TOOL_DEFINITIONS`, and `async def run(query, session_id)`
- Register in `src/orchestrator/main.py`: add to `AGENT_REGISTRY` and `AGENT_DEFINITIONS`
- Create agent prompt template: `src/services/prompts/v1/{agent_name}_prompt.mustache`

**New Tool:**
- Create `src/tools/{tool_name}.py` with an `async` function
- Register in the owning agent's `TOOL_REGISTRY` and `TOOL_DEFINITIONS`
- Create prompt template if LLM-driven: `src/services/prompts/v1/{tool_name}_prompt.mustache`
- Create Pydantic response model in `src/schemas/` if producing structured output

**New API Endpoint:**
- Create `src/api/endpoints/{domain}/router.py` with a `router = APIRouter()`
- Register in `src/api/main.py` via `app.include_router(router, prefix="/api/v1")`

**New Schema:**
- Add Pydantic model to the appropriate file in `src/schemas/` (or create new file for new domain)
- Use `Field(...)` with descriptions for LLM JSON schema generation

**New Exception:**
- Create in `src/exceptions/` inheriting from `AppException`
- Follow pattern in `src/exceptions/parser_exceptions.py`

**New Embedding Service:**
- Create in `src/services/vector_store/embeddings/` inheriting from `BaseEmbeddingService`
- Update `src/dependencies.py` to use new service if replacing the active one

**New Prompt Template:**
- Add `.mustache` file in `src/services/prompts/v1/`
- Load via `load_prompt("template_name")` (omit `.mustache` extension)
- Use `{{variable}}` syntax; rendering disables HTML escaping

**New Playbook Rules:**
- Add to `src/data/default_playbook_rules.json` (V1 format) or `src/data/playbook_rules_v3.json` (V3 format)
- Clear cached rules by restarting the application (module-level cache)

**New Document Parser:**
- Create in `src/services/registry/` inheriting from `BaseParser`
- Register in `src/services/registry/registry.py` `_register_default_parsers()`

## Special Directories

**`logs/`:**
- Purpose: Runtime log files generated by the application
- Generated: Yes
- Committed: No (gitignored)

**`src/data/`:**
- Purpose: Static JSON data files (playbook rules) loaded at runtime
- Generated: No (manually authored)
- Committed: Yes (should be)

**`docs/`:**
- Purpose: Documentation assets (architecture diagram image)
- Generated: No
- Committed: Yes

**`.planning/`:**
- Purpose: GSD project planning files and codebase analysis
- Generated: Yes (by GSD tools)
- Committed: Yes

**`.vscode/`:**
- Purpose: VS Code workspace settings
- Generated: No
- Committed: Yes

---

*Structure analysis: 2026-02-19*
