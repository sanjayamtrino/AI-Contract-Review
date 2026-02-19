# Technology Stack

**Analysis Date:** 2026-02-19

## Languages

**Primary:**
- Python 3.10 - All application code (`src/`, `tests/`)

**Secondary:**
- Mustache (template language) - LLM prompt templates (`src/services/prompts/v1/*.mustache`)
- JSON - Playbook rule definitions (`src/data/*.json`)

## Runtime

**Environment:**
- Python 3.10 (pinned in `pyproject.toml` line 9)
- CPython (standard interpreter)

**Package Manager:**
- Poetry (core dependency manager)
- Build backend: `poetry.core.masonry.api`
- Lockfile: `poetry.lock` present (542 KB, committed to repo)

**Entry Point:**
- CLI script `contract-api` maps to `src.api.main:main_entry` (defined in `pyproject.toml` line 13)
- Direct run: `poetry run python -m src.api.main`

## Frameworks

**Core:**
- FastAPI `*` (unpinned) - REST API framework (`src/api/main.py`)
- Uvicorn `*` (unpinned) - ASGI server, used as FastAPI runner
- Pydantic `*` (unpinned) - Data validation and schema definitions throughout (`src/schemas/`)
- Pydantic Settings `*` (unpinned) - Environment-based configuration (`src/config/settings.py`)

**AI/ML:**
- OpenAI SDK `*` (unpinned) - Azure OpenAI chat completions and embeddings (`src/services/llm/azure_openai_model.py`, `src/services/vector_store/embeddings/openai_embeddings.py`)
- Google GenAI SDK (`google-genai`) `*` (unpinned) - Gemini text generation and embeddings (`src/services/llm/gemini_model.py`, `src/services/vector_store/embeddings/gemini_embeddings.py`)
- LangChain `*` (unpinned) - `RecursiveCharacterTextSplitter` for document chunking (`src/services/registry/doc_parser.py`)
- LangChain Community `*` (unpinned) - Community integrations (imported but usage unclear)
- Sentence Transformers `*` (unpinned) - Local HuggingFace embedding models (`src/services/vector_store/embeddings/embedding_service.py`, `src/services/vector_store/embeddings/qwen_embeddings.py`)
- Transformers (via `AutoModel`/`AutoTokenizer`) - BGE embedding model loading (`src/services/vector_store/embeddings/embedding_service.py`)
- FAISS CPU (`faiss-cpu`) `*` (unpinned) - In-memory vector similarity search (`src/services/vector_store/faiss_db.py`)
- PyTorch (`torch`) - Tensor operations for BGE embeddings (`src/services/vector_store/embeddings/embedding_service.py`)
- Einops `*` (unpinned) - Tensor manipulation (transitive dependency for transformers)

**Templating:**
- Chevron `^0.14.0` - Mustache template rendering for prompt loading (`src/services/prompts/v1/__init__.py`)
- Pystache `*` (unpinned) - Mustache template rendering with custom escape behavior (`src/services/llm/azure_openai_model.py`, `src/services/llm/gemini_model.py`)

**Document Parsing:**
- python-docx `*` (unpinned) - DOCX file reading and parsing (`src/services/registry/doc_parser.py`, `src/services/registry/semantic_parser.py`)
- python-multipart `*` (unpinned) - FastAPI file upload support

**Agent Framework:**
- `agent-framework` `*` (pre-release, unpinned) - Listed in `pyproject.toml` line 35 but not imported anywhere in source code
- `agent-framework-azure-ai` `*` (pre-release, unpinned) - Listed in `pyproject.toml` line 36 but not imported anywhere in source code

**Testing:**
- pytest - Test runner (`.pytest_cache/` present, no explicit config file)

**Dev Tools:**
- Black `*` (unpinned) - Code formatter (line-length 200, target Python 3.10)
- isort `*` (unpinned) - Import sorting (profile "black", line-length 200)
- mypy `*` (unpinned) - Static type checking (Python 3.8 target, strict mode)
- flake8 `*` (unpinned) - Linting (max-line-length 200)

## Key Dependencies

**Critical (required for core functionality):**
- `openai` - Primary LLM client for Azure OpenAI chat completions; used by all agents and tools
- `faiss-cpu` - In-memory vector database for embedding similarity search
- `sentence-transformers` + `transformers` + `torch` - Local BGE embedding model (`BAAI/bge-large-en-v1.5`)
- `fastapi` + `uvicorn` - API server
- `pydantic` + `pydantic-settings` - Schema validation and settings management
- `python-docx` - Contract document parsing (DOCX format only)
- `chevron` + `pystache` - Mustache prompt template rendering

**Infrastructure:**
- `numpy` - Numerical operations for FAISS and embedding normalization
- `requests` - HTTP client for Jina embedding API calls (`src/services/vector_store/embeddings/jina_embeddings.py`)
- `langchain` - Only `RecursiveCharacterTextSplitter` is used from this package

**Unused/Dormant (installed but commented out or not imported):**
- `agent-framework` / `agent-framework-azure-ai` - Declared in `pyproject.toml` but no imports found
- `google-genai` - Gemini model class exists but is commented out in `src/dependencies.py`
- `langchain-community` - Declared but no active imports found

## Configuration

**Environment:**
- Configuration via `.env` file loaded by Pydantic Settings (`src/config/settings.py`)
- `.env.example` present with 4 variables: `GEMINI_API_KEY`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `JINA_EMBEDDING_API`
- `.env` file present (not committed, in `.gitignore`)
- All settings have defaults defined in `src/config/settings.py`

**Key Settings (from `src/config/settings.py`):**
- `api_host` (default: `localhost`), `api_port` (default: `8000`)
- `chunk_size` (default: `1000`), `chunk_overlap` (default: `200`)
- `gemini_api_key`, `gemini_embedding_model`, `gemini_text_generation_model`
- `openai_api_key`, `openai_embedding_model`, `openai_model`
- `azure_openai_api_key`, `base_url`, `azure_openai_responses_deployment_name`, `azure_api_version`
- `jina_embedding_API`, `jina_embedding_model_uri`
- `session_ttl_minutes` (default: `2`), `session_cleanup_interval_minutes` (default: `1.0`)
- `logs_directory` (default: `./logs`)

**Build:**
- `pyproject.toml` - Poetry project definition, tool configuration (Black, isort, mypy, flake8)
- `.vscode/settings.json` - Editor config (format-on-save with Black, organize imports)

## Platform Requirements

**Development:**
- Python 3.10 (exact version pinned)
- Poetry package manager
- Azure OpenAI API access (required for all LLM operations)
- CUDA optional (model runs on CPU by default: `self.model.to("cpu")`)
- ~1GB+ RAM for BGE embedding model (`BAAI/bge-large-en-v1.5` loaded at startup)

**Production:**
- No Dockerfile or deployment configuration found
- No CI/CD pipeline configuration found (no `.github/workflows/`)
- Application runs as a single FastAPI process via Uvicorn
- All data is in-memory (FAISS index, session stores) -- no persistent database

---

*Stack analysis: 2026-02-19*
