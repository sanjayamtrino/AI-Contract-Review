# External Integrations

**Analysis Date:** 2026-02-19

## APIs & External Services

### Azure OpenAI (Primary -- Active)

The primary LLM integration. All agents and tools use Azure OpenAI for chat completions with structured JSON output.

- **SDK/Client:** `openai` Python SDK (`OpenAI` class, NOT `AzureOpenAI`)
- **Implementation:** `src/services/llm/azure_openai_model.py`
- **Auth:** API key via `AZURE_OPENAI_API_KEY` env var
- **Endpoint:** Custom base URL via `BASE_URL` env var
- **Deployment:** Model name via `AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME` (default: `gpt-4o`)
- **API Version:** `AZURE_API_VERSION` (default: `2024-05-01-preview`)
- **Usage Pattern:**
  ```python
  # From src/services/llm/azure_openai_model.py
  self.client = OpenAI(
      base_url=self.settings.base_url,
      api_key=self.settings.azure_openai_api_key,
  )
  response = self.client.chat.completions.create(
      model=self.deployment_name,
      messages=[...],
      temperature=0.2,
      max_tokens=16384,
      response_format={"type": "json_schema", "json_schema": {...}},
  )
  ```
- **Callers:**
  - `src/orchestrator/main.py` - Agent routing decisions (function calling with `tools`)
  - `src/agents/doc_information.py` - Tool selection (function calling)
  - `src/agents/playbook_review.py` - Tool selection (function calling)
  - `src/agents/playbook_review_v2.py` - Tool selection (function calling)
  - `src/tools/summarizer.py` - Document summarization (structured output)
  - `src/tools/key_details.py` - Key details extraction (structured output)
  - `src/tools/clause_extractor.py` - Clause extraction (structured output)
  - `src/tools/clause_comparator.py` - Clause comparison (structured output)
  - `src/tools/rule_risk_assessor.py` - Risk assessment (structured output)
  - `src/tools/review_report_generator.py` - Report generation (structured output)
  - `src/tools/master_playbook_assessor.py` - Batch rule assessment (structured output)
  - `src/services/retrieval/retrieval.py` - Query rewriting (structured output)

**Important:** The code uses `OpenAI` client (not `AzureOpenAI`), connecting to Azure via a custom `base_url`. This means it uses the OpenAI-compatible endpoint format rather than the Azure-specific SDK.

### Google Gemini (Dormant -- Commented Out)

Gemini integration exists but is disabled in the service container.

- **SDK/Client:** `google-genai` (`genai.Client`)
- **Implementation:** `src/services/llm/gemini_model.py`
- **Auth:** `GEMINI_API_KEY` env var
- **Default Model:** `gemini-2.5-flash-lite-preview-09-2025`
- **Status:** Class exists and is importable, but usage is commented out in `src/dependencies.py` (lines 8, 31, 54-55, 91-92, 137-142)
- **Embedding Service:** `src/services/vector_store/embeddings/gemini_embeddings.py` (also dormant)
- **Default Embedding Model:** `gemini-embedding-001`

### Jina AI Embeddings (Dormant -- Not Active in Default Config)

API-based embedding service, code exists but not used in default configuration.

- **Implementation:** `src/services/vector_store/embeddings/jina_embeddings.py`
- **Endpoint:** `https://api.jina.ai/v1/embeddings` (via `JINA_EMBEDDING_MODEL_URI`)
- **Auth:** `JINA_EMBEDDING_API` env var (sent as `Authorization` header)
- **Model:** `jina-embeddings-v3` (default)
- **Dimensions:** 1024 (hardcoded)
- **Status:** Class exists but is commented out in parser imports. Not used in active code path.

### OpenAI Direct (Dormant -- Not Active in Default Config)

Direct OpenAI (non-Azure) embedding service exists but is not used.

- **Implementation:** `src/services/vector_store/embeddings/openai_embeddings.py`
- **Auth:** `OPENAI_API_KEY` env var
- **Model:** `text-embedding-3-large` (default, 3072 dimensions)
- **Status:** Class exists but is commented out in parser and retrieval imports.

### Hugging Face Models (Local -- Active)

Local model inference, no external API calls.

- **BGE Embeddings (Active):**
  - **Implementation:** `src/services/vector_store/embeddings/embedding_service.py` (class `BGEEmbeddingService`)
  - **Model:** `BAAI/bge-large-en-v1.5` (hardcoded, downloaded from HuggingFace Hub on first run)
  - **Dimensions:** 1024 (from `model.config.hidden_size`)
  - **Device:** CPU (`self.model.to("cpu")`)
  - **Usage:** Primary embedding service for document ingestion and retrieval
  - **Callers:** `src/dependencies.py`, `src/services/registry/doc_parser.py`, `src/services/registry/semantic_parser.py`, `src/services/retrieval/retrieval.py`, `src/tools/embedding_clause_matcher.py`

- **MiniLM Embeddings (Dormant):**
  - **Implementation:** `src/services/vector_store/embeddings/embedding_service.py` (class `HuggingFaceEmbeddingService`)
  - **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
  - **Status:** Commented out, replaced by BGE

- **Qwen3 Embeddings (Dormant):**
  - **Implementation:** `src/services/vector_store/embeddings/qwen_embeddings.py`
  - **Model:** `Qwen/Qwen3-Embedding-0.6B`
  - **Status:** Class exists but not used in any active code path

## Data Storage

### Vector Database

- **Type:** FAISS (Facebook AI Similarity Search) -- In-memory
- **Implementation:** `src/services/vector_store/faiss_db.py`
- **Index Type:** `IndexFlatIP` (Inner Product / cosine similarity with L2 normalization)
- **Persistence:** None -- all data lost on restart
- **Scope:** Per-session FAISS indices managed by `src/services/session_manager.py`
- **Legacy:** Global singleton FAISS store in `src/services/vector_store/manager.py` (deprecated)

### Document Storage

- **Type:** In-memory chunk store
- **Implementation:** `src/services/session_manager.py` (`SessionData.chunk_store`)
- **Format:** `Dict[int, Chunk]` mapping chunk indices to `Chunk` Pydantic models
- **Persistence:** None -- session data expires based on TTL

### Playbook Rules

- **Type:** Static JSON files loaded from filesystem
- **Files:**
  - `src/data/default_playbook_rules.json` - Default playbook rules (v1 format)
  - `src/data/playbook_rules_v3.json` - V3 format rules (title + instruction + description)
- **Loader:** `src/services/playbook_loader.py` (module-level caching with global variables)

### File Storage

- Local filesystem only for logs (`./logs/`)
- No cloud storage integration

### Caching

- In-memory only:
  - Settings cached via `@lru_cache` in `src/config/settings.py`
  - Playbook rules cached in module-level globals in `src/services/playbook_loader.py`
  - No Redis, Memcached, or other external cache

## Authentication & Identity

### Application Auth

- **Provider:** None -- no authentication system implemented
- **Session Tracking:** Header-based session ID (`X-Session-ID` header)
- **Implementation:** `src/api/session_utils.py`
- **Behavior:** Client provides arbitrary session ID; no validation, no auth, no user identity
- **Session Lifecycle:**
  - Created on first use via `session_manager.get_or_create_session()`
  - Auto-expires after `session_ttl_minutes` (default: 2 minutes)
  - Background cleanup task runs every `session_cleanup_interval_minutes` (default: 1 minute)

### LLM Service Auth

- Azure OpenAI: API key in `AZURE_OPENAI_API_KEY` env var, passed directly to OpenAI client
- Gemini: API key in `GEMINI_API_KEY` env var (dormant)
- Jina: API key in `JINA_EMBEDDING_API` env var, sent as `Authorization` header (dormant)
- OpenAI: API key in `OPENAI_API_KEY` env var (dormant)

## Monitoring & Observability

### Logging

- **Framework:** Python `logging` standard library
- **Configuration:** `src/config/logging.py`
- **Log Files:**
  - `./logs/AI_Contract_Review_YYYYMMDD.log` - Daily rotating file (10MB max, 5 backups, detailed format)
  - `./logs/errors.log` - Error-only rotating file (10MB max, 5 backups, JSON format)
- **Console:** Simple format (`timestamp - level - message`)
- **Logger Mixin:** `Logger` class in `src/config/logging.py` provides `self.logger` property to any class inheriting from it
- **Logger Naming:** `AI_Contract.{ClassName}` pattern

### Error Tracking

- No external error tracking service (no Sentry, Datadog, etc.)
- Errors logged to `./logs/errors.log` in JSON format

### Metrics

- Internal stats tracking per service (embedding counts, search requests, timing) via `self.stats` dicts
- No external metrics export (no Prometheus, StatsD, etc.)
- Request timing via middleware: `X-Process-Time` response header (`src/api/main.py` lines 37-43)

### Health Checks

- Admin health endpoint: `GET /api/v1/admin/health/` (`src/api/endpoints/admin/router.py`)
- Per-service health methods: `get_health_status()` on embedding services, parser, ingestion service

## CI/CD & Deployment

### Hosting

- No deployment configuration found
- No Dockerfile, docker-compose, or cloud deployment manifests
- Application designed to run locally as a single process

### CI Pipeline

- No CI/CD pipeline found (no `.github/workflows/`, no `.gitlab-ci.yml`, no `Jenkinsfile`)

## Environment Configuration

### Required Environment Variables (for active functionality)

| Variable | Purpose | Default |
|----------|---------|---------|
| `AZURE_OPENAI_API_KEY` | Azure OpenAI authentication | None (required) |
| `BASE_URL` | Azure OpenAI endpoint URL | None (required) |
| `AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME` | Azure model deployment name | `gpt-4o` |
| `AZURE_API_VERSION` | Azure API version | `2024-05-01-preview` |

### Optional Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `API_HOST` | Server bind address | `localhost` |
| `API_PORT` | Server port | `8000` |
| `DEBUG` | Enable debug mode | `False` |
| `CHUNK_SIZE` | Text chunking size | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap size | `200` |
| `SESSION_TTL_MINUTES` | Session expiry | `2` |
| `SESSION_CLEANUP_INTERVAL_MINUTES` | Cleanup frequency | `1.0` |
| `LOGS_DIRECTORY` | Log file directory | `./logs` |

### Dormant Environment Variables (for inactive features)

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Google Gemini API access |
| `OPENAI_API_KEY` | Direct OpenAI access |
| `JINA_EMBEDDING_API` | Jina embedding service |

### Secrets Location

- `.env` file in project root (gitignored)
- `.env.example` with placeholder values committed to repo
- No secrets manager, vault, or cloud secrets integration

## Webhooks & Callbacks

### Incoming

- None

### Outgoing

- None

## API Endpoints Summary

All endpoints are prefixed with `/api/v1/`:

| Method | Path | Purpose | File |
|--------|------|---------|------|
| POST | `/ingest/` | Upload and process DOCX file | `src/api/endpoints/ingestion/router.py` |
| POST | `/query/` | RAG-based document query | `src/api/endpoints/retrieval/router.py` |
| GET | `/summarizer` | Get document summary | `src/api/endpoints/retrieval/router.py` |
| GET | `/key-details` | Extract key contract details | `src/api/endpoints/retrieval/router.py` |
| POST | `/orchestrator/query/` | AI-routed multi-agent query | `src/api/endpoints/orchestrator/router.py` |
| GET | `/admin/sessions/` | List all sessions | `src/api/endpoints/admin/router.py` |
| GET | `/admin/sessions/{id}` | Get session details | `src/api/endpoints/admin/router.py` |
| DELETE | `/admin/sessions/{id}` | Delete session | `src/api/endpoints/admin/router.py` |
| POST | `/admin/sessions/cleanup` | Manual session cleanup | `src/api/endpoints/admin/router.py` |
| GET | `/admin/health/` | Health check | `src/api/endpoints/admin/router.py` |

All endpoints except admin routes require `X-Session-ID` header.

---

*Integration audit: 2026-02-19*
