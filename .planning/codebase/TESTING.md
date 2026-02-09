# Testing Patterns

**Analysis Date:** 2026-02-09

## Test Framework

**Runner:**
- pytest (`.pytest_cache` directory present; no explicit `pytest.ini` found)
- No explicit pytest configuration in `pyproject.toml` (would be detected with `[tool.pytest.ini_options]`)
- Default pytest behavior assumed (test discovery from `test_*.py` files)

**Assertion Library:**
- Standard Python assertions and comparison operators
- No dedicated assertion library configured (pytest's built-in assert rewriting used)

**Run Commands:**
```bash
pytest                          # Run all tests in tests/ directory
pytest tests/test_embeddings.py # Run specific test file
pytest -v                       # Verbose output
pytest --tb=short              # Short traceback format
pytest --cov                   # Coverage (if pytest-cov installed)
```

Note: Actual test execution commands not explicitly documented in codebase; pytest is standard Python testing framework.

## Test File Organization

**Location:**
- Co-located in separate `tests/` directory at project root: `C:\Users\amtri\AI-Contract-Review\tests\`
- Not alongside source code; centralized test directory structure

**Naming:**
- Test files prefixed with `test_`: `test_backend_logic.py`, `test_embeddings.py`
- Descriptive module names matching functionality tested

**Test Count:**
- 3 test files currently in codebase (2026-02-09)
- 49 Python source files in `src/`
- Low test coverage relative to codebase size

**Test Structure:**
```
tests/
├── __init__.py                # Empty, marks directory as package
├── test_backend_logic.py      # Integration/manual testing of AI engine
├── test_embeddings.py         # Comparison testing of embedding services
```

## Test Structure

**Suite Organization:**
- Tests use simple function-based organization (no test classes with TestCase structure)
- Each test file contains standalone async functions that execute tests
- Test functions use `asyncio.run()` to execute async code when run as main

**Example from `test_backend_logic.py`:**
```python
async def start_testing_process():
    print("\n🔵 1. INITIALIZING AI ENGINE...")
    try:
        ai_engine = GeminiModel()
        print(f"   ✅ Engine Online. Using Model: {ai_engine.llm.model}")
    except Exception as e:
        print(f"   ❌ Engine Failed to Start: {e}")
        return

    print("\n🔵 2. PREPARING TEST DOCUMENT...")
    # Test logic continues...

if __name__ == "__main__":
    asyncio.run(start_testing_process())
```

**Patterns:**
- No setup/teardown methods observed (no test fixtures)
- Integration tests that directly instantiate services and test end-to-end
- Manual/exploratory testing approach (print statements, manual verification)
- Not using pytest fixtures or conftest.py
- Tests can be run as standalone scripts directly with Python

## Mocking

**Framework:** No explicit mocking framework detected
- Tests instantiate real services: `GeminiModel()`, `HuggingFaceEmbeddingService()`, `OpenAIEmbeddings()`
- No `unittest.mock`, `pytest-mock`, or similar configured
- Integration tests use actual API connections (requires real credentials in `.env`)

**Patterns:**
- No mock objects observed
- Tests depend on real external services and credentials
- Manual test data construction within test functions:
  ```python
  test_document_text = """
      AGREEM ENT FOR SERVICE
      This Contract is made on March 15th, 2025 betwen Alpha Corp and Beta Ltd.
      Fee is $50,000 payable Net 30.
      """
  ```

**What to Mock (Recommendations):**
- External API calls (Gemini, OpenAI, Azure)
- Vector database operations
- File I/O operations
- LLM responses (for deterministic testing)

**What NOT to Mock:**
- Core business logic (embedding generation, parsing)
- Schema validation (Pydantic models)
- Configuration loading (Settings)

## Fixtures and Factories

**Test Data:**
- No pytest fixtures defined
- Test data created inline in test functions
- Example from `test_embeddings.py`:
  ```python
  texts = {
      "Tech_News_1": """
          Silicon Valley tech giant announces breakthrough in quantum computing...
          """,
      "Tech_News_2": """
          Major smartphone manufacturer unveils latest flagship device...
          """,
  }
  ```
- Multiline strings embedded directly in test functions

**Location:**
- Test data lives within test files (no separate fixtures directory)
- No `conftest.py` file present
- No factory functions for creating test objects

## Coverage

**Requirements:** Not enforced (no coverage thresholds in configuration)

**View Coverage:**
```bash
pytest --cov=src --cov-report=html tests/
pytest --cov=src --cov-report=term-missing tests/
```

**Current Coverage:**
- Estimated at <30% (3 test files for 49 source files)
- No systematic unit test coverage
- Focus on manual integration testing

**Gaps:**
- No unit tests for services, models, or embeddings
- No tests for error handling paths
- No tests for configuration/settings validation
- No tests for exception raising logic
- Schema validation (`Chunk`, `ParseResult`) untested
- Utilities and helper functions untested

## Test Types

**Unit Tests:**
- Not currently implemented
- Would test individual functions/methods in isolation
- Should target: service methods, embedding generation, text cleaning, schema validation
- Scope: Single class or module behavior
- Example candidate: `DocxParser._clean_text()` method

**Integration Tests:**
- Primary testing approach currently used
- Examples:
  - `test_backend_logic.py`: Tests full pipeline from GeminiModel initialization through prompt generation
  - `test_embeddings.py`: Tests embedding generation across multiple services with cosine similarity comparison
- Scope: Multiple components working together with real external services
- Uses real API credentials from `.env`
- Synchronous assertions with print-based verification:
  ```python
  if result.get('financials', {}).get('total_fee') == "$50,000":
      print("\n🟢 TEST STATUS: PASSED (100% Accuracy)")
  else:
      print("\n🔴 TEST STATUS: FAILED (Data Mismatch)")
  ```

**E2E Tests:**
- Not separately defined
- Integration tests serve as de facto E2E tests
- No headless browser testing or external system interaction beyond APIs

## Common Patterns

**Async Testing:**
- Functions wrapped with `async def`:
  ```python
  async def start_testing_process():
      # test implementation
  ```
- Executed with `asyncio.run()`:
  ```python
  if __name__ == "__main__":
      asyncio.run(start_testing_process())
  ```
- Await calls to async service methods:
  ```python
  result = await ai_engine.generate(prompt=test_document_text)
  ```

**Error Testing:**
- Try-except blocks to catch and verify errors:
  ```python
  try:
      ai_engine = GeminiModel()
      print(f"   ✅ Engine Online...")
  except Exception as e:
      print(f"   ❌ Engine Failed: {e}")
      return
  ```
- Error messages printed to stdout
- No assertion of specific exception types
- Early return on error (stops test execution)

**Output Verification:**
- Print statements used for manual verification
- Status indicators with emoji: 🔵 (step), ✅ (success), ❌ (failure), 🟢 (pass), 🔴 (fail)
- Data inspection through dictionary access:
  ```python
  print(f"   Summary:    {result.get('summary_simple')[:50]}...")
  print(f"   Party A:    {result.get('parties', {}).get('party_a')}")
  ```

## Testing Best Practices Opportunities

**Currently Missing:**
1. No unit test framework setup (pytest conventions not fully utilized)
2. No test fixtures or factories for reusable test data
3. No mocking of external services (tests require real credentials)
4. No test organization by test class or marker
5. No CI/CD integration visible (no test output preservation)
6. No parametrized tests for testing multiple scenarios
7. No test documentation or expected outcomes documented

**Recommended Additions:**
1. Create `conftest.py` with pytest fixtures for common test setup
2. Use `pytest.mark` decorators to organize tests (unit, integration, slow)
3. Mock external API calls using `unittest.mock` or `pytest-mock`
4. Convert manual print-based assertions to proper pytest assertions
5. Use `pytest.parametrize` for testing multiple embedding models/scenarios
6. Create factory functions for common test objects (e.g., test chunks, documents)
7. Add test coverage reporting to CI/CD pipeline

---

*Testing analysis: 2026-02-09*
