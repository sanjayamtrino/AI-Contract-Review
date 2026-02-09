# Technology Stack

**Analysis Date:** 2026-02-09

## Languages

**Primary:**
- Python 3.10 - Backend application, AI/ML services, and orchestration logic
- Mustache - Template language for prompt templating in `src/services/prompts/v1/`

## Runtime

**Environment:**
- Python 3.10 (as specified in `pyproject.toml`)
- Poetry for dependency management and virtual environment handling

**Package Manager:**
- Poetry 1.0.0+ - Manages all Python dependencies and development tools
- Lockfile: Present (`poetry.lock` - 539KB)

## Frameworks

**Core Web Framework:**
- FastAPI - HTTP API framework for contract review endpoints
- Uvicorn - ASGI server for running FastAPI application
  - Configuration: `src/api/main.py` runs on `localhost:8000` by default

**LLM & AI:**
- OpenAI SDK - Integration with OpenAI and Azure OpenAI models
- google-genai - Google Gemini LLM and embedding models
- LangChain - LLM framework for prompt chaining (langchain, langchain-community packages)
- agent-framework - Custom agent orchestration framework for tool-calling models

**Embeddings & Vector Operations:**
- sentence-transformers - HuggingFace embeddings (all-MiniLM-L6-v2 model)
- transformers - HuggingFace model loading (AutoModel, AutoTokenizer)
- torch - Deep learning framework for embedding generation
- faiss-cpu - In-memory vector similarity search and indexing
  - Used in `src/services/vector_store/faiss_db.py` with IndexFlatIP for cosine similarity

**Document Processing:**
- python-docx - DOCX file parsing and extraction
- requests - HTTP client for Jina Embeddings API calls

**Data Validation:**
- Pydantic - Schema validation and serialization
- pydantic-settings - Configuration management with `.env` file support

**Template Rendering:**
- chevron - Mustache template rendering (v0.14.0+)
- pystache - Alternative Mustache rendering (used in `src/services/llm/`)

**Development & Quality:**
- black - Code formatting (line-length: 200)
- isort - Import sorting with black profile
- mypy - Static type checking (strict mode with disallow_untyped_defs)
- flake8 - Linting (max-line-length: 200, max-complexity: 10)

**Utilities:**
- python-multipart - Multipart form data parsing for file uploads

## Key Dependencies

**Critical for Application:**
- openai [Latest] - OpenAI API client for text generation and embeddings
- google-genai [Latest] - Google Gemini client for LLM and embeddings
- fastapi [Latest] - Web API framework
- uvicorn [Latest] - ASGI server
- faiss-cpu [Latest] - Vector search engine
- sentence-transformers [Latest] - Local embeddings (384-dim MiniLM model)
- pydantic [Latest] - Data validation
- pydantic-settings [Latest] - Configuration from .env

**AI/ML Infrastructure:**
- langchain [Latest] - LLM orchestration framework
- langchain-community [Latest] - Community integrations
- torch [Latest] - Deep learning operations for embeddings
- transformers [Latest] - HuggingFace model utilities

**Supporting Libraries:**
- python-docx [Latest] - DOCX document parsing
- requests [Latest] - HTTP requests for API calls
- chevron [0.14.0+] - Mustache template rendering
- agent-framework [Latest] - Custom agent framework for Azure OpenAI tool-calling

## Configuration

**Environment:**
- Configuration loaded from `.env` file via pydantic-settings
- Environment variables (see `.env.example`):
  - `GEMINI_API_KEY` - Google Gemini API authentication
  - `CHUNK_SIZE` - Text chunk size (default: 1000)
  - `CHUNK_OVERLAP` - Chunk overlap for continuity (default: 200)
  - `JINA_EMBEDDING_API` - Jina Embeddings API token

**Build:**
- `pyproject.toml` - Main configuration file with tool settings:
  - Black formatting at 200 char line length
  - isort import sorting with black profile
  - MyPy strict type checking
  - Flake8 linting rules
- Poetry lock file ensures reproducible builds

## Platform Requirements

**Development:**
- Python 3.10 (>3.8 and <=3.10 per README)
- Poetry package manager
- 8+ GB RAM recommended for embedding models (sentence-transformers, BGE models)
- CPU support (no CUDA required - uses torch CPU backend)

**Production:**
- Python 3.10 runtime
- No specific cloud provider lock-in
- Supports multiple LLM backends (OpenAI, Azure OpenAI, Google Gemini)
- Can run locally with HuggingFace models or connect to cloud APIs

## Entry Point

**API Server:**
- Command: `poetry run python -m src.api.main`
- Entry function: `main_entry()` in `src/api/main.py`
- Serves on HTTP at configured host:port (default localhost:8000)
- Swagger docs available at `/docs/`

## Optional Embedding Providers (Pluggable)

The codebase supports multiple embedding backends that can be swapped via configuration:

1. **OpenAI Embeddings** (`src/services/vector_store/embeddings/openai_embeddings.py`)
   - Model: text-embedding-3-large (3072-dim) or text-embedding-3-small (1536-dim)
   - Requires: `OPENAI_API_KEY`

2. **Google Gemini Embeddings** (`src/services/vector_store/embeddings/gemini_embeddings.py`)
   - Model: gemini-embedding-001
   - Requires: `GEMINI_API_KEY`

3. **Jina Embeddings** (`src/services/vector_store/embeddings/jina_embeddings.py`)
   - Model: jina-embeddings-v3
   - Dimension: 1024
   - Requires: `JINA_EMBEDDING_API`

4. **HuggingFace Local** (`src/services/vector_store/embeddings/embedding_service.py`)
   - Models: MiniLM-L6-v2 (384-dim), Qwen3-Embedding-0.6B, BGE models
   - No API key required (runs locally)

---

*Stack analysis: 2026-02-09*
