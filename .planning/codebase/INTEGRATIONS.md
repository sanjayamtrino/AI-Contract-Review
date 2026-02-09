# External Integrations

**Analysis Date:** 2026-02-09

## APIs & External Services

**Large Language Models:**
- OpenAI API - Text generation via `gpt-4o` model (Azure deployment)
  - SDK/Client: `openai` package
  - Auth: `OPENAI_API_KEY` environment variable
  - Implementation: `src/services/llm/azure_openai_model.py`
  - Capabilities: Chat completions with structured output, tool calling

- Google Gemini API - LLM and embedding generation
  - SDK/Client: `google-genai` package
  - Auth: `GEMINI_API_KEY` environment variable
  - Implementation: `src/services/llm/gemini_model.py`
  - Models:
    - Text generation: `gemini-2.5-flash-lite-preview-09-2025`
    - Embeddings: `gemini-embedding-001`

**Embedding Services:**
- Jina Embeddings API - Vector embeddings via HTTP
  - SDK/Client: `requests` HTTP client
  - Auth: `JINA_EMBEDDING_API` bearer token in Authorization header
  - Endpoint: `https://api.jina.ai/v1/embeddings`
  - Model: `jina-embeddings-v3` (1024-dim vectors)
  - Implementation: `src/services/vector_store/embeddings/jina_embeddings.py`

## Data Storage

**Vector Database:**
- FAISS (Facebook AI Similarity Search) - In-memory vector index
  - Type: In-process, CPU-only vector database
  - Client: `faiss-cpu` package
  - Implementation: `src/services/vector_store/faiss_db.py`
  - Index type: IndexFlatIP (inner product/cosine similarity)
  - Persistence: Managed via `src/services/vector_store/manager.py` (chunks stored in memory)
  - Note: Data persists only for the session duration

**Document Storage:**
- Local Filesystem - Document chunks and embeddings
  - Chunks cached in-memory during runtime
  - No persistent database configured (embeddings and chunks lost on restart)

**File Storage:**
- Local Filesystem - Document uploads
  - Upload handling: `src/api/endpoints/ingestion/router.py`
  - File format supported: DOCX (Word documents)
  - Parser implementation: `src/services/registry/doc_parser.py` using python-docx

## Authentication & Identity

**Auth Provider:**
- No centralized authentication service
- API is open (no API key or OAuth required for endpoints)
- Service-to-API authentication uses environment variable API keys:
  - OpenAI: `OPENAI_API_KEY` or `AZURE_OPENAI_API_KEY`
  - Gemini: `GEMINI_API_KEY`
  - Jina: `JINA_EMBEDDING_API`
  - Azure OpenAI: `AZURE_OPENAI_API_KEY`, `AZURE_ENDPOINT_URI`, `AZURE_DEPLOYMENT_NAME`

## Monitoring & Observability

**Error Tracking:**
- Not detected - No external error tracking service (Sentry, DataDog, etc.)

**Logs:**
- File-based logging to `./logs/` directory
- Implementation: `src/config/logging.py`
- Log format: `timestamp - logger_name - level - file:line - function - message`
- Daily log rotation (separate files per day)
- Logger base class: `src/config/logging.py` provides consistent logging interface

**Health Checks:**
- Embedding services provide health status endpoints:
  - `HuggingFaceEmbeddingService.get_health_status()` - Tests embedding generation
  - `GeminiEmbeddingService.get_health_status()` - Tests Gemini API connectivity
  - `OpenAIEmbeddings.get_health_status()` - Tests OpenAI API connectivity
- Ingestion service health check in `src/services/ingestion/ingestion.py`

## CI/CD & Deployment

**Hosting:**
- Not configured - Application is self-hosted Python app
- Can run on any system with Python 3.10 and dependencies installed
- Docker deployment: Not detected (no Dockerfile or docker-compose.yml)

**CI Pipeline:**
- Not detected - No GitHub Actions, Jenkins, or other CI configuration

**Environment Configuration:**
- Poetry scripts defined in `pyproject.toml`:
  - Main entry: `contract-api = "src.api.main:main_entry"`
- Run command: `poetry run contract-api`
- Alternative: `poetry run python -m src.api.main`

## Environment Configuration

**Required Environment Variables:**
- `GEMINI_API_KEY` - Google Gemini API key
- `JINA_EMBEDDING_API` - Jina Embeddings API token
- `OPENAI_API_KEY` - OpenAI API key (required if using OpenAI backend)
- `AZURE_OPENAI_API_KEY` - Azure OpenAI API key (if using Azure)
- `AZURE_ENDPOINT_URI` - Azure OpenAI endpoint URL
- `AZURE_DEPLOYMENT_NAME` - Azure deployment name (default: "gpt-4o")
- `AZURE_API_VERSION` - Azure API version (default: "2024-05-01-preview")

**Optional Environment Variables:**
- `CHUNK_SIZE` - Text chunk size for document splitting (default: 1000)
- `CHUNK_OVERLAP` - Overlap between chunks (default: 200)
- `API_HOST` - API server host (default: "localhost")
- `API_PORT` - API server port (default: 8000)
- `DEBUG` - Enable debug mode (default: false)
- `LOGS_DIRECTORY` - Path to logs directory (default: "./logs")

**Secrets Location:**
- `.env` file in project root (not committed - listed in .gitignore)
- Example template: `.env.example` shows required variables
- Loaded via pydantic-settings `BaseSettings` class

## Webhooks & Callbacks

**Incoming:**
- Not detected - No webhook endpoints for external services

**Outgoing:**
- Not detected - No external service callbacks or webhooks

## Tool Execution

**Agent Framework Integration:**
- Uses `agent-framework` package for tool-calling with Azure OpenAI
- Implementation: `src/orchestrator/main.py`
- Tool execution loop with max 2 iterations
- Tools available to orchestrator agent:
  - `get_summary` - Summarization tool
  - `get_location` - Location extraction tool
  - `get_key_information` - Key information extraction tool
- Tool calling format: OpenAI function calling (tool_calls in response)

## Prompt Management

**Prompt Templates:**
- Format: Mustache template files (`.mustache`)
- Location: `src/services/prompts/v1/`
- Available prompts:
  - `orchestrator_prompt` - Instructions for orchestrator agent
  - `query_rewriter.mustache` - Query rewriting prompt
  - `llm_response.mustache` - Response generation prompt
- Rendering: Via Mustache (chevron or pystache packages)
- Template context injection: Dynamic context dictionaries passed to LLM

## Document Processing Pipeline

**Ingestion Endpoint:**
- `POST /api/v1/ingest/` - Upload and parse documents
- Accepts multipart file upload
- Implementation: `src/api/endpoints/ingestion/router.py`
- Processing:
  1. File received as UploadFile
  2. Parsed via ParserRegistry in `src/services/registry/registry.py`
  3. Document split into chunks with overlap
  4. Chunks indexed into FAISS vector store

**Retrieval Endpoints:**
- `POST /api/v1/query/` - Query documents
  - Query rewriting via LLM
  - Embedding generation
  - Vector similarity search
  - Implementation: `src/services/retrieval/retrieval.py`

- `GET /api/v1/summarizer` - Get document summary
  - Uses summarization tool

---

*Integration audit: 2026-02-09*
