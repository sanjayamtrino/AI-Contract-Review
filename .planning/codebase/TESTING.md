# Testing Patterns

**Analysis Date:** 2026-02-19

## Test Framework

**Runner:**
- **pytest** (inferred from compiled `.pyc` files in `tests/__pycache__/` showing `pytest-9.0.2`)
- No `pytest.ini`, `conftest.py`, or `[tool.pytest.ini_options]` section in `pyproject.toml`
- No test dependencies listed in `pyproject.toml` (pytest is not declared as a dependency)

**Assertion Library:**
- No formal assertion library used
- Tests use `print()` statements and manual visual inspection rather than `assert` statements
- No test runner assertions (no `assert`, no `pytest.raises`, no `assertEqual`)

**Run Commands:**
```bash
pytest tests/                  # Run tests in tests/ directory (if pytest is installed)
python tests/test_embeddings.py          # Run embedding comparison script directly
python tests/test_backend_logic.py       # Run backend integration test directly
python test_master_prompt.py             # Run master prompt test script directly
```

## Test File Organization

**Location:**
- Primary test directory: `tests/` at project root
- One ad-hoc test file at project root: `test_master_prompt.py`
- No nested test directories

**Naming:**
- `test_` prefix: `test_embeddings.py`, `test_backend_logic.py`, `test_master_prompt.py`
- `tests/__init__.py` exists (empty) to make it a package

**Structure:**
```
AI-Contract-Review/
├── tests/
│   ├── __init__.py              # Empty package marker
│   ├── test_embeddings.py       # Embedding service comparison script
│   └── test_backend_logic.py    # Manual integration test for Gemini LLM
├── test_master_prompt.py        # Standalone prompt test against Azure OpenAI
└── test_master_prompt_output.json  # Output from test_master_prompt.py
```

**Compiled Test Cache (historical tests no longer present):**
- `tests/__pycache__/conftest.cpython-310-pytest-9.0.2.pyc` - a conftest.py existed at some point but was deleted
- `tests/__pycache__/test_prompt_engineering.cpython-310-pytest-9.0.2.pyc` - a prompt engineering test existed
- `tests/__pycache__/test_orchestrator.cpython-310-pytest-9.0.2.pyc` - an orchestrator test existed

## Test Structure

**Current tests are NOT structured as pytest test suites.** They are standalone scripts executed via `python filename.py` with `asyncio.run()`. None contain proper `test_` functions discoverable by pytest.

**Pattern in `tests/test_embeddings.py`:**
```python
# No test functions - this is a comparison script
async def generate_embeddings(text: str) -> Dict[str, Any]:
    """Generate embeddings for all services for comparison."""
    embeddings: Dict[str, Any] = {}
    embeddings["MiniLM"] = await sentence_transformers.generate_embeddings(text)
    # ... more services ...
    return embeddings

async def compare_embeddings(texts: Dict[str, str]) -> None:
    """Compare the embeddings of each service used."""
    # prints cosine similarities to console

if __name__ == "__main__":
    asyncio.run(compare_embeddings(texts))
```

**Pattern in `tests/test_backend_logic.py`:**
```python
# Manual integration test with print-based verification
async def start_testing_process():
    print("\n1. INITIALIZING AI ENGINE...")
    try:
        ai_engine = GeminiModel()
    except Exception as e:
        print(f"   Engine Failed to Start: {e}")
        return

    # ... sends test prompt to Gemini API ...

    if result.get('financials', {}).get('total_fee') == "$50,000":
        print("\n TEST STATUS: PASSED")
    else:
        print("\n TEST STATUS: FAILED")

if __name__ == "__main__":
    asyncio.run(start_testing_process())
```

**Pattern in `test_master_prompt.py`:**
```python
# Standalone prompt testing script
def run_test():
    settings = get_settings()

    # Step 1: Load and render the prompt template
    prompt_path = Path("src/services/prompts/v1/master_playbook_review_prompt.mustache")
    template = prompt_path.read_text(encoding="utf-8")
    rendered_prompt = renderer.render(template, context)

    # Step 2: Send to Azure OpenAI
    response = client.chat.completions.create(...)

    # Step 3: Parse and validate with Pydantic
    response_data = json.loads(response_text)
    validated = MasterPlaybookReviewResponse.model_validate(response_data)

    # Step 4: Print results
    for rule_analysis in validated.rules_analysis:
        print(f"  RULE: {rule_analysis.rule_title}")

    # Step 5: Save full response to JSON
    output_path.write_text(json.dumps(response_data, indent=2))

if __name__ == "__main__":
    run_test()
```

## Mocking

**Framework:** None used

**Patterns:** No mocking is employed. All existing tests make real API calls to external services (Gemini, Azure OpenAI, embedding models).

**What to Mock (recommendations for future tests):**
- LLM API calls (`AzureOpenAIModel.generate`, `GeminiModel.generate`)
- Embedding service calls (`BGEEmbeddingService.generate_embeddings`)
- FAISS vector store operations
- Session manager state

**What NOT to Mock:**
- Pydantic schema validation (test real validation)
- Prompt template rendering (test real templates)
- Data transformation functions (`_clauses_to_prompt_context`, `_format_paragraphs_for_prompt`)

## Fixtures and Factories

**Test Data:**
- `tests/test_embeddings.py` contains hardcoded sample texts (news articles) for embedding comparison
- `tests/test_backend_logic.py` contains a hardcoded contract snippet as `test_document_text`
- `test_master_prompt.py` contains `SAMPLE_RULES` and `SAMPLE_MATCH_RESULTS` as inline test data
- `test_master_prompt_output.json` stores LLM output from the master prompt test

**No shared fixtures, factories, or conftest.py exist.**

**Location:**
- All test data is inline within the test files themselves
- `src/data/default_playbook_rules.json` and `src/data/playbook_rules_v3.json` contain production rule data that could serve as test fixtures

## Coverage

**Requirements:** None enforced. No coverage configuration, thresholds, or reporting set up.

**View Coverage:** Not configured. To add:
```bash
pip install pytest-cov
pytest --cov=src tests/
```

## Test Types

**Unit Tests:**
- None exist. No isolated tests of individual functions/methods with mocked dependencies.

**Integration Tests:**
- `tests/test_backend_logic.py` - tests GeminiModel against real API (manual execution)
- `test_master_prompt.py` - tests master prompt template + Azure OpenAI API + Pydantic validation (manual execution)

**E2E Tests:**
- None. No tests exercise the full FastAPI application endpoints.

**Comparison/Benchmarking Scripts:**
- `tests/test_embeddings.py` - compares cosine similarity across 5 embedding providers

## Common Patterns

**Async Testing:**
```python
# Current pattern: asyncio.run() in __main__ block
if __name__ == "__main__":
    asyncio.run(some_async_test_function())
```

**Recommended async pytest pattern (not yet used):**
```python
import pytest

@pytest.mark.asyncio
async def test_something():
    result = await some_async_function()
    assert result is not None
```

**Error Testing:**
No error path testing exists. Future tests should cover:
```python
# Example of what should exist
async def test_session_not_found():
    with pytest.raises(ValueError, match="Session .* not found"):
        await get_summary(session_id="nonexistent")

async def test_empty_document():
    with pytest.raises(ValueError, match="No document ingested"):
        await extract_clauses(session_id="empty_session")
```

## Critical Gaps

1. **No pytest-compatible tests exist** - all current test files are standalone scripts, not discoverable by pytest
2. **No mocking** - every test requires live API keys and network access
3. **No conftest.py** - no shared fixtures or configuration
4. **No CI integration** - no test commands in `pyproject.toml` scripts, no CI pipeline config
5. **pytest not declared as a dependency** in `pyproject.toml`
6. **Zero assertion-based tests** - all verification is print-based and manual
7. **No test coverage** of core logic: parsing, chunking, retrieval, session management, orchestrator routing
8. **Historical tests were deleted** - `.pyc` cache shows `conftest.py`, `test_prompt_engineering.py`, and `test_orchestrator.py` once existed

## Adding New Tests

When adding tests to this codebase:

1. **Add pytest dependencies** to `pyproject.toml`:
   ```toml
   [tool.poetry.group.dev.dependencies]
   pytest = "*"
   pytest-asyncio = "*"
   pytest-cov = "*"
   ```

2. **Create `tests/conftest.py`** with shared fixtures:
   ```python
   import pytest
   from unittest.mock import AsyncMock, MagicMock

   @pytest.fixture
   def mock_llm():
       llm = AsyncMock()
       llm.generate.return_value = SomeModel(...)
       return llm

   @pytest.fixture
   def sample_session_data():
       # Create a SessionData with pre-populated chunks
       ...
   ```

3. **Place test files** in `tests/` directory, mirroring `src/` structure:
   ```
   tests/
   ├── conftest.py
   ├── test_tools/
   │   ├── test_summarizer.py
   │   ├── test_clause_extractor.py
   │   └── test_key_details.py
   ├── test_services/
   │   ├── test_session_manager.py
   │   └── test_ingestion.py
   ├── test_agents/
   │   ├── test_doc_information.py
   │   └── test_playbook_review.py
   └── test_api/
       └── test_endpoints.py
   ```

4. **Follow pytest naming**: `test_` prefix for files and functions, `Test` prefix for classes

5. **Use `@pytest.mark.asyncio`** for all async test functions

---

*Testing analysis: 2026-02-19*
