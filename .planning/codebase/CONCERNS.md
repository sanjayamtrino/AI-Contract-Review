# Codebase Concerns

**Analysis Date:** 2026-02-09

## Tech Debt

**Hardcoded Embedding Service Selection:**
- Issue: Multiple embedding services are imported and commented out. Only one is active at a time, with the choice hardcoded in each parser/service instantiation.
- Files: `src/services/registry/doc_parser.py` (lines 42-46), `src/services/registry/semantic_parser.py` (lines 48), `src/services/retrieval/retrieval.py` (lines 25-28)
- Impact: Adding or switching embedding providers requires code changes across multiple files. No configuration-driven selection mechanism exists.
- Fix approach: Create an embedding service factory pattern that reads from settings. Use `get_settings().embedding_provider` to instantiate the appropriate embedding service.

**Incomplete Tool Implementations:**
- Issue: Tools in `src/tools/summarizer.py` return hardcoded dummy strings instead of performing actual operations.
- Files: `src/tools/summarizer.py` (lines 32-47)
- Impact: The orchestrator agent has tools that don't actually execute their intended functions. Summarization, location extraction, and key information retrieval are non-functional.
- Fix approach: Implement actual logic for `get_summary()`, `get_location()`, and `get_key_information()`. The commented-out async implementation (lines 18-30) should be completed and activated.

**Unimplemented Health Check Method:**
- Issue: `is_healthy()` in semantic parser is defined as a stub with only `pass`.
- Files: `src/services/registry/semantic_parser.py` (lines 327-329)
- Impact: Health checks cannot validate the semantic parser's status. The method signature differs from the doc parser's async implementation, creating interface inconsistency.
- Fix approach: Implement the method to mirror the health check pattern in `doc_parser.py` (lines 342-399). Make it async and return a comprehensive health status dict.

**Empty Implementation in Retrieval Service:**
- Issue: `retrieve_document()` method returns an empty dict instead of implementing document retrieval.
- Files: `src/services/retrieval/retrieval.py` (lines 43-45)
- Impact: No way to retrieve entire document chunks at once. The function is not called but its existence suggests incomplete API design.
- Fix approach: Either implement full document retrieval by fetching all chunks from the manager, or remove the method if it's not needed.

**Inconsistent Parser Registration:**
- Issue: The parser registry's `get_parser()` method ignores file extensions and always returns the DOCX parser.
- Files: `src/services/registry/registry.py` (lines 31-34)
- Impact: Cannot parse non-DOCX files despite the architecture suggesting multi-format support. The comment "Need to implement this method" (line 30) acknowledges this.
- Fix approach: Modify `get_parser()` to accept a file extension parameter and return the appropriate parser. Update the ingestion router to extract and pass the file extension.

**Unused Vector Store Variable:**
- Issue: `vector_store` is set to `None` in IngestionService.__init__ and never used.
- Files: `src/services/ingestion/ingestion.py` (line 24)
- Impact: Health check references `self.vector_store` but it's never initialized. This makes the health check always report vector store as inaccessible.
- Fix approach: Either initialize the vector store or remove the health check reference.

## Known Bugs

**Untracked Variable in Retrieval:**
- Symptom: `retrieved_chunks` list is created but never populated or used in the retrieval result metadata.
- Files: `src/services/retrieval/retrieval.py` (lines 71, 106)
- Trigger: Any call to `retrieve_data()` method
- Workaround: The `returned_results` field in search metadata will always show 0, even though chunks are returned in the response. The actual chunk count is in `num_results` instead.
- Impact: Metadata reporting is misleading. Consumers checking `returned_results` will see incorrect count.

**Missing Variable Initialization in Retrieval Loop:**
- Symptom: Variable `new_query` is set inside the loop but used outside the loop in return statements.
- Files: `src/services/retrieval/retrieval.py` (lines 58-108)
- Trigger: When a single query is processed (which is the normal case)
- Issue: If `retrieve_data()` is called with an empty rewritten_queries list, `new_query` will be undefined, causing a NameError. This could happen if query rewriting fails silently.
- Impact: Potential runtime crash if query rewriting edge cases occur.

**Dimension Mismatch Warning Not Preventing Issues:**
- Symptom: FAISS vector store manager warns but still returns existing instance with wrong dimensions.
- Files: `src/services/vector_store/manager.py` (lines 32-33)
- Trigger: When two different embedding models with different dimensions try to use the same vector store
- Issue: A logger.warning is issued but execution continues. This can cause silent failures when searching with embeddings of the wrong dimension.
- Impact: Vector search may fail with cryptic FAISS errors rather than clear dimension mismatch errors.

**Embedding Vector Storage Inconsistency:**
- Symptom: `embedding_vector` is set to `None` for table chunks but actual vector_data is generated.
- Files: `src/services/registry/doc_parser.py` (line 296), `src/services/registry/semantic_parser.py` (line 272)
- Trigger: When parsing tables from DOCX documents
- Issue: Vectors are computed and indexed (lines 287-288, 286-287 respectively) but the chunk object stores None instead of the actual vector. Paragraph chunks store the vector (line 257 in doc_parser.py).
- Impact: Inconsistent chunk structure. Retrieval accuracy may suffer because chunk objects don't store their embeddings for later reference.

## Security Considerations

**No Input Validation on File Uploads:**
- Risk: Arbitrary DOCX files are loaded without validation. The file content is not checked before processing.
- Files: `src/api/endpoints/ingestion/router.py` (lines 13-20)
- Current mitigation: FastAPI's UploadFile limits file size by default, but no document-level validation exists.
- Recommendations:
  - Add file size limits in settings
  - Validate DOCX structure before processing
  - Add malware scanning for uploaded documents
  - Implement rate limiting on upload endpoint

**Hardcoded File Paths with String Interpolation:**
- Risk: File paths use raw strings like `r"src\services\prompts\v1\query_rewriter.mustache"` which are Windows-specific and won't work on Linux/Mac.
- Files: `src/services/retrieval/retrieval.py` (line 31), `src/api/endpoints/retrieval/router.py` (line 23)
- Current mitigation: None. Code assumes Windows file separators.
- Recommendations:
  - Use `Path` object from pathlib consistently: `Path(__file__).parent / "prompts" / "v1" / "query_rewriter.mustache"`
  - Store prompt template paths in settings
  - Add unit tests to catch platform-specific issues

**Debug Mode Exposure:**
- Risk: FastAPI app has `debug=settings.debug` which might expose stack traces in production.
- Files: `src/api/main.py` (line 12)
- Current mitigation: Debug is controlled by environment variable
- Recommendations:
  - Never run with debug=True in production
  - Document debug mode requirements
  - Add startup checks to warn if debug is True in production

**API Endpoint Type Hints Missing:**
- Risk: Query endpoint returns `None` type hint but actually returns a dict.
- Files: `src/api/endpoints/retrieval/router.py` (line 27)
- Current mitigation: None - just incorrect documentation
- Recommendations:
  - Fix return type to `Dict[str, Any]`
  - Use Pydantic response models for all endpoints
  - Add response validation

## Performance Bottlenecks

**Synchronous Model Loading in Async Context:**
- Problem: Embedding models are loaded synchronously in service initialization, blocking the event loop.
- Files: `src/services/vector_store/embeddings/embedding_service.py` (lines 120-123 for BGE), `src/services/vector_store/embeddings/gemini_embeddings.py`, `src/services/vector_store/embeddings/openai_embeddings.py`
- Cause: Model initialization happens in `__init__` which is called synchronously. For large models like BGE, this can block for 10+ seconds.
- Improvement path:
  - Implement lazy loading: defer model initialization until first use
  - Load models in background tasks during startup
  - Use asyncio.to_thread() for model loading if needed in async context

**Vector Search Inefficiency with Large Document Sets:**
- Problem: `retrieve_data()` generates multiple embeddings per query (original + rewritten queries) but doesn't deduplicate results efficiently.
- Files: `src/services/retrieval/retrieval.py` (lines 54-108)
- Cause: For each rewritten query, a full search is performed. Results are deduplicated by index, but this requires multiple FAISS searches which can be slow with 100k+ vectors.
- Improvement path:
  - Batch rewritten queries into a single embedding call
  - Use hybrid search combining BM25 + semantic search
  - Implement caching for frequently asked queries
  - Consider approximate nearest neighbor indexes (HNSW instead of IndexFlatIP)

**Chunk Metadata Size Accumulation:**
- Problem: Metadata dict grows with each chunk and is stored redundantly in retrieval results.
- Files: `src/services/registry/doc_parser.py` (lines 258-260), `src/services/registry/semantic_parser.py` (lines 273)
- Cause: Each chunk stores full metadata. In retrieval (lines 81-88 in retrieval.py), metadata is copied to response.
- Improvement path:
  - Store only essential metadata in chunks (ID, type)
  - Retrieve additional metadata on-demand
  - Use separate metadata store with indexed lookups

**No Caching for Embedding Service Calls:**
- Problem: Same text passages may be embedded multiple times.
- Files: `src/services/vector_store/embeddings/embedding_service.py`
- Cause: No memoization or caching layer for embedding generation
- Improvement path:
  - Implement embedding cache using Redis or local LRU cache
  - Hash text to check if embedding already exists
  - Invalidate cache when embedding model changes

## Fragile Areas

**Parser Registry Design:**
- Files: `src/services/registry/registry.py`, `src/services/registry/doc_parser.py`, `src/services/registry/semantic_parser.py`
- Why fragile: Two different parsers with the same name "DocxParser" exist. The registry has hardcoded "DOCX" key. Switching between parsers requires editing the import in registry.py (line 6). No test coverage for parser selection.
- Safe modification:
  - Rename parsers to indicate strategy: `DocxStructuralParser` vs `DocxSemanticParser`
  - Use a config setting to select active parser: `active_parser_type: str = "semantic"`
  - Add unit tests for each parser independently
  - Add integration test to verify selected parser is used

**Orchestrator Tool Execution:**
- Files: `src/orchestrator/main.py` (lines 129-153)
- Why fragile: Tool execution catches all exceptions with generic handler. Missing tools don't raise errors (line 140 just prints). Result variable might be undefined if exception occurs in tool execution.
- Safe modification:
  - Use explicit tool definitions with type hints
  - Create custom exception for missing tools
  - Always initialize result before try block
  - Add logging for tool execution failures
  - Return error objects instead of silently failing

**Vector Store Singleton with Global State:**
- Files: `src/services/vector_store/manager.py` (lines 8-12)
- Why fragile: Global mutable state (_instance, _chunk_store, _chunk_counter) makes testing difficult and can cause issues in multi-process deployments. No way to reset state between requests in tests. Lock only protects dictionary operations, not consistency across methods.
- Safe modification:
  - Create per-request vector store instances or request-scoped singletons
  - Add comprehensive unit tests with state reset
  - Document that this is NOT thread-safe for multi-process deployments
  - Consider using a proper database instead of global dicts

**Embedding Generation without Dimension Validation:**
- Files: `src/services/vector_store/embeddings/embedding_service.py`
- Why fragile: Multiple embedding services (BGE, HuggingFace, Gemini, OpenAI, Qwen, Jina) have different output dimensions. If the wrong service is selected after initialization, vectors of wrong dimension will be generated and silently fail during indexing.
- Safe modification:
  - Add dimension check in FAISS indexing (already done but warning is not fatal)
  - Validate dimension matches during service initialization
  - Store expected dimension in settings
  - Add runtime assertion in generate_embeddings() return

**Hard-Coded Text Processing Constants:**
- Files: `src/services/registry/semantic_parser.py` (lines 38, 42), `src/services/registry/doc_parser.py` (lines 184-206)
- Why fragile: `_HEADING_MAX_WORDS = 8`, `_ORPHAN_MIN_WORDS = 5` are class constants. Chunk size and overlap come from settings. This split is inconsistent and makes tuning chunking behavior difficult.
- Safe modification:
  - Move all chunking parameters to settings
  - Add settings validation to ensure chunk_size > chunk_overlap
  - Document the impact of each parameter
  - Add chunking strategy as pluggable class

## Scaling Limits

**In-Memory Vector Storage with FAISS:**
- Current capacity: Limited by RAM. FAISS IndexFlatIP stores entire index in memory.
- Limit: ~1M vectors × 768 dims × 4 bytes = 3GB RAM for BGE embeddings. Scales poorly beyond this.
- Scaling path:
  - Migrate to persistent vector databases (Weaviate, Qdrant, Milvus, Pinecone)
  - Use FAISS GPU index or approximate indexes (IndexIVFFlat, IndexHNSW)
  - Implement document sharding across multiple vector stores

**Chunk Manager Dictionary Unbounded Growth:**
- Current capacity: All chunks stored in memory in `_chunk_store` dict in manager.py
- Limit: Python dict requires ~240 bytes per entry. 100k chunks = ~24MB. But with metadata, could reach 1GB+ for large document sets.
- Scaling path:
  - Implement TTL-based eviction for chunk cache
  - Use SQLite or PostgreSQL as chunk store backend
  - Implement LRU cache with size limits
  - Stream chunks from persistent storage instead of loading all

**Single Vector Store Instance Bottleneck:**
- Current capacity: Single FAISS index per application instance
- Limit: Cannot load-balance across replicas. Horizontal scaling blocked by state sharing.
- Scaling path:
  - Implement stateless API layer with external vector store
  - Use distributed vector database
  - Add read replicas for vector store
  - Implement request routing to handle index sharding

**Embedding Service GPU Memory:**
- Current capacity: BGE model on CPU uses ~2GB RAM for inference
- Limit: Cannot handle concurrent embedding requests efficiently without GPU acceleration
- Scaling path:
  - Implement GPU acceleration with CUDA/Metal
  - Add batching for embedding generation
  - Implement queue-based embedding service with rate limiting
  - Cache embeddings aggressively

## Dependencies at Risk

**Langchain RecursiveCharacterTextSplitter:**
- Risk: Only used in doc_parser.py (line 9). Langchain is heavy dependency with frequent breaking changes.
- Impact: If langchain upgrades break compatibility, text chunking fails.
- Migration plan: Replace with simple regex-based splitter or use `textwrap` module. The splitting logic is straightforward and doesn't need external dependency.

**Sentence-Transformers Model Loading:**
- Risk: Models are downloaded from Hugging Face at runtime if not cached. Network failures will crash the service.
- Impact: Model not found = ingestion service crashes. No fallback.
- Migration plan:
  - Pre-download and bundle models with docker image
  - Add retry logic with exponential backoff for model loading
  - Implement model cache with version pinning
  - Use local model paths from settings

**FAISS Dimension Mismatch Vulnerability:**
- Risk: FAISS error messages are cryptic. Dimension mismatches cause generic "Unable to search" errors.
- Impact: Hard to debug in production. Silent failures possible.
- Migration plan:
  - Add explicit dimension validation before all FAISS operations
  - Create custom exception type for dimension errors
  - Log embedding dimensions on service startup

## Missing Critical Features

**No Persistent Storage:**
- Problem: All parsed chunks and vector embeddings are lost when the service restarts. No database backend.
- Blocks: Cannot deploy to Kubernetes without losing data on pod restarts. Cannot scale horizontally.

**No Query Rewriting Fallback:**
- Problem: If query rewriting fails, the entire retrieval fails. No graceful degradation.
- Blocks: Service unavailability if LLM service for rewriting is down.

**No Result Ranking by Relevance:**
- Problem: Top-k results are returned but not ranked by semantic relevance to original query vs. rewritten queries.
- Blocks: Users cannot distinguish between high-confidence and low-confidence results.

**No Chunk Deduplication:**
- Problem: Same text can be extracted as both paragraph and table chunk, creating duplicates in results.
- Blocks: Duplicate content in retrieval results confuses users.

## Test Coverage Gaps

**Parser Selection Logic Untested:**
- What's not tested: The registry.get_parser() method and file extension handling
- Files: `src/services/registry/registry.py`
- Risk: Switching to multi-format parsing will likely break without test coverage
- Priority: High - this is a core routing mechanism

**Embedding Vector Quality Untested:**
- What's not tested: Whether generated embeddings are actually meaningful or if dimensions match correctly
- Files: `src/services/vector_store/embeddings/*.py`
- Risk: Silent embedding failures where vectors are generated but wrong dimensions or bad quality
- Priority: High - affects core retrieval quality

**Error Handling in Orchestrator Tool Loop:**
- What's not tested: The orchestrator's exception handling and tool execution logic
- Files: `src/orchestrator/main.py` (lines 129-153)
- Risk: Missing tool exceptions and undefined variables in error cases
- Priority: High - orchestrator is the main user-facing API

**Chunk Metadata Consistency:**
- What's not tested: Whether chunk metadata is preserved correctly through ingestion → indexing → retrieval pipeline
- Files: `src/services/registry/doc_parser.py`, `src/services/vector_store/manager.py`, `src/services/retrieval/retrieval.py`
- Risk: Metadata corruption or loss during processing
- Priority: Medium - affects debugging and tracing

**Vector Search with Empty Index:**
- What's not tested: Behavior when FAISS index is empty
- Files: `src/services/vector_store/faiss_db.py`, `src/services/retrieval/retrieval.py`
- Risk: Potential crash or undefined behavior with edge cases
- Priority: Medium - affects first-time user experience

**API Type Safety:**
- What's not tested: Response schemas match actual return types
- Files: `src/api/endpoints/retrieval/router.py`
- Risk: API clients expecting certain types receive different objects
- Priority: Medium - affects API stability

---

*Concerns audit: 2026-02-09*
