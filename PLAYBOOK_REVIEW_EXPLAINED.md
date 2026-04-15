# Playbook Review Agent - Complete Technical Explanation

## Table of Contents

1. [What Is This System?](#1-what-is-this-system)
2. [Architecture Overview](#2-architecture-overview)
3. [Libraries and Frameworks Used](#3-libraries-and-frameworks-used)
4. [Complete Request Flow (Step by Step)](#4-complete-request-flow-step-by-step)
5. [File-by-File Breakdown](#5-file-by-file-breakdown)
6. [Prompt Templates Explained](#6-prompt-templates-explained)
7. [Data Schemas (Pydantic Models)](#7-data-schemas-pydantic-models)
8. [How LLM Calls Work](#8-how-llm-calls-work)
9. [Issues Encountered and How They Were Fixed](#9-issues-encountered-and-how-they-were-fixed)
10. [LLM Call Count Breakdown](#10-llm-call-count-breakdown)
11. [Why These Design Decisions](#11-why-these-design-decisions)

---

## 1. What Is This System?

This is an **AI-powered contract review system** called Accorder AI. It takes a legal document (like an NDA), compares it against a set of predefined **playbook rules** (what the organization expects the contract to say), and produces a structured report showing which rules PASS, FAIL, or are NOT FOUND.

**Example:** If your organization's playbook says "NDA duration must be 3 years" but the uploaded contract says "2 years", the system will mark that rule as FAIL with a High risk rating and suggest corrected clause text.

---

## 2. Architecture Overview

The system follows a **multi-layer agent architecture**:

```
User sends query via API
       |
       v
[FastAPI Endpoint]  (src/api/endpoints/orchestrator/router.py)
       |
       v
[Orchestrator]  (src/orchestrator/main.py)
  - Decides WHICH agent to route to
  - Uses LLM with tool/function calling
  - Two agents available: doc_information_agent, playbook_review_agent
       |
       v
[Playbook Review Agent]  (src/agents/playbook_review.py)
  - Decides WHICH tool to use
  - Uses LLM with tool/function calling
  - Two tools: full_playbook_review, single_rule_review
       |
       v
[Playbook Reviewer Tool]  (src/tools/playbook_reviewer.py)
  - Loads rules from JSON
  - Retrieves relevant paragraphs using FAISS vector search
  - Evaluates each rule against paragraphs using LLM (parallel)
  - Builds final report deterministically (no LLM)
       |
       v
[PlaybookReviewReport]  (returned to user as JSON)
```

### Key Design Principle: Minimize LLM Calls

- Paragraph retrieval uses **local embeddings + FAISS** (no LLM needed)
- Report assembly is **pure Python logic** (no LLM needed)
- Only the per-rule evaluation uses the LLM
- Rules are evaluated **in parallel** for speed

---

## 3. Libraries and Frameworks Used

### Backend Framework

| Library | Version | What It Does | Why We Use It |
|---------|---------|-------------|---------------|
| **FastAPI** | - | Web framework for the REST API | Async-native, automatic OpenAPI/Swagger docs at `/docs`, type validation with Pydantic, high performance |
| **Uvicorn** | - | ASGI server that runs FastAPI | Production-grade async server, supports hot-reload during development |
| **Pydantic** | v2 | Data validation and serialization | Defines strict schemas for LLM inputs/outputs, auto-generates JSON schemas for OpenAI structured output, catches invalid data early |

### AI / LLM

| Library | What It Does | Why We Use It |
|---------|-------------|---------------|
| **OpenAI Python SDK** | Communicates with Azure OpenAI API | The LLM (GPT-4) is hosted on Azure OpenAI. The SDK handles API calls, retries, streaming, and structured JSON output |
| **Azure OpenAI Service** | Hosts the GPT-4 model | Enterprise-grade LLM hosting with SLA guarantees, data privacy, and regional deployment |

### Embeddings and Vector Search

| Library | What It Does | Why We Use It |
|---------|-------------|---------------|
| **BGE Embedding Model** (via `BGEEmbeddingService`) | Converts text into numerical vectors (embeddings) | Local model - no API call needed. Fast, free, runs on CPU. Used to find which paragraphs are relevant to each rule |
| **FAISS** (Facebook AI Similarity Search) | Stores and searches vectors efficiently | Blazing fast similarity search. Given a rule's embedding, finds the most similar paragraphs in milliseconds. Each session gets its own FAISS index |

### Document Parsing

| Library | What It Does | Why We Use It |
|---------|-------------|---------------|
| **python-docx** | Reads .docx files | Extracts paragraphs, tables, metadata from Word documents |
| **LangChain** (`RecursiveCharacterTextSplitter`) | Splits long text into chunks | Smart chunking that respects sentence boundaries, paragraph breaks, etc. Configurable chunk size and overlap |

### Prompt Rendering

| Library | What It Does | Why We Use It |
|---------|-------------|---------------|
| **Mustache** (via `chevron` or similar) | Template engine for prompts | Prompts are stored as `.mustache` files with placeholders like `{{rule_title}}`. At runtime, actual values are injected. Keeps prompts separate from code |

### Concurrency

| Library | What It Does | Why We Use It |
|---------|-------------|---------------|
| **asyncio** (Python stdlib) | Async task execution | `asyncio.gather()` runs multiple rule evaluations in parallel. `asyncio.Semaphore` limits concurrency to 5 simultaneous LLM calls |
| **asyncio.to_thread()** | Runs blocking code in thread pool | The OpenAI SDK is synchronous, so we wrap it in `to_thread()` to avoid blocking the async event loop |

### Other Python Standard Libraries

| Library | What It Does |
|---------|-------------|
| `json` | Parse/serialize JSON (LLM responses, playbook rules) |
| `pathlib.Path` | File path handling for prompt templates |
| `uuid` | Generate unique IDs for chunks and documents |
| `re` | Regex for text cleaning in document parser |
| `time` | Measure processing time |
| `threading.Lock` | Thread-safe access to shared vector store |
| `enum.Enum` | Define Verdict (PASS/FAIL/NOT FOUND) and RiskLevel (Low/Medium/High/Critical) |

---

## 4. Complete Request Flow (Step by Step)

Here is exactly what happens when a user sends "Review this contract":

### Step 0: Document Ingestion (happens before review)

```
POST /api/v1/ingest/  (with .docx file + X-Session-ID: 123)
```

1. **Router** (`src/api/endpoints/ingestion/router.py`) receives the file
2. **Session Manager** creates or gets session "123"
3. **DocxParser** (`src/services/registry/doc_parser.py`):
   - Extracts paragraphs and tables from the .docx
   - Splits full text into chunks using `RecursiveCharacterTextSplitter`
   - For each chunk: generates an embedding using BGE model (local, no API call)
   - Adds each embedding vector to the session's FAISS index
4. **Ingestion Service** (`src/services/ingestion/ingestion.py`):
   - Stores chunk objects in `session.chunk_store` (dictionary: index -> Chunk)
   - Records document metadata

**Result:** Session "123" now has a FAISS index with 12 vectors and a chunk_store with 12 chunks.

### Step 1: User Sends Review Query

```
POST /api/v1/orchestrator/query/?query=Review this contract
Headers: X-Session-ID: 123
```

### Step 2: Orchestrator Routes to Agent

**File:** `src/orchestrator/main.py`

1. Loads `orchestrator_prompt.mustache` (system prompt)
2. Sends the user query to the LLM with two "tools" (agents) defined:
   - `doc_information_agent` - for document questions
   - `playbook_review_agent` - for compliance reviews
3. **LLM Call #1**: The LLM sees "Review this contract" and picks `playbook_review_agent`
4. Orchestrator calls `playbook_review.run(query, session_id)`

### Step 3: Agent Picks Tool

**File:** `src/agents/playbook_review.py`

1. Loads `playbook_review_v2_agent_prompt.mustache`
2. Sends query to LLM with two tools:
   - `full_playbook_review` - review ALL rules
   - `single_rule_review` - review ONE specific rule
3. **LLM Call #2**: The LLM sees "Review this contract" and picks `full_playbook_review`
4. Agent calls `full_playbook_review(session_id="123", playbook_name="v3")`

### Step 4: Load Rules (NO LLM)

**File:** `src/tools/playbook_reviewer.py` -> `src/services/playbook_loader.py`

- Reads `src/data/playbook_rules_v3.json`
- Parses 10 rules into `PlaybookRule` Pydantic models
- Each rule has: title, instruction, description

### Step 5: Retrieve Paragraphs per Rule (NO LLM)

**File:** `src/tools/playbook_reviewer.py` -> `_retrieve_paragraphs_for_rule()`

For EACH of the 10 rules (in parallel):
1. Combine rule title + instruction + description into search text
2. Generate embedding using BGE model (local, no API call)
3. Search the session's FAISS index for the top 8 most similar paragraph vectors
4. Filter out results below similarity threshold (0.25)
5. Return list of matching paragraphs with their chunk content

**Result:** Each rule now has 0-8 relevant paragraphs from the contract.

### Step 6: Evaluate Each Rule (LLM, Parallel)

**File:** `src/tools/playbook_reviewer.py` -> `_evaluate_rule()`

For EACH of the 10 rules (max 5 in parallel via Semaphore):

1. If no paragraphs found -> return NOT FOUND immediately (no LLM call)
2. Load `rule_evaluation_v2_prompt.mustache`
3. Render the template with: rule_title, rule_instruction, rule_description, paragraphs_text
4. **LLM Calls #3 through #12**: Send to Azure OpenAI with:
   - System message: "You are an expert Contract Review Analyst..."
   - User message: The rendered prompt
   - `response_format`: JSON Schema matching `RuleEvaluation` model
   - `temperature: 0.0` (deterministic, no randomness)
5. Parse LLM response into `RuleEvaluation` Pydantic model

**What the LLM returns per rule:**
```json
{
  "_reasoning": "Step-by-step analysis...",
  "rule_title": "Term and Survival of Obligations",
  "rule_instruction": "Verify the NDA's duration...",
  "rule_description": "This Agreement shall remain...",
  "para_identifiers": ["2", "10"],
  "status": "FAIL",
  "reason": "The agreement states 'two years' which does not meet the required three years...",
  "suggestion": "Extend the NDA duration to three years...",
  "suggested_fix": "This Agreement shall remain in effect for 3 years...",
  "confidence": 0.95,
  "risk_level": "High"
}
```

### Step 7: Build Report (NO LLM)

**File:** `src/tools/playbook_reviewer.py` -> `_build_report()`

Pure Python logic, no LLM:
1. Convert each `RuleEvaluation` into a `RuleResult` (adds category, paragraphs_retrieved)
2. Compute statistics: total rules, passed, failed, not found, counts per risk level
3. Group rule titles by risk level
4. Compute `missing_clauses` (rules with NOT FOUND status)
5. Compute overall risk level:
   - Any CRITICAL -> CRITICAL
   - Any HIGH -> HIGH
   - 3+ MEDIUM -> HIGH
   - Any MEDIUM -> MEDIUM
   - All LOW -> LOW
6. Sort results by risk (Critical first, Low last)
7. Return `PlaybookReviewReport`

### Step 8: Response Flows Back

```
PlaybookReviewReport
  -> playbook_reviewer.py returns to agent
  -> playbook_review.py wraps in AgentResponse
  -> orchestrator wraps in OrchestratorResponse
  -> FastAPI serializes to JSON
  -> User sees the full report
```

---

## 5. File-by-File Breakdown

### API Layer
| File | Purpose |
|------|---------|
| `src/api/main.py` | FastAPI app setup, includes routers, middleware |
| `src/api/endpoints/orchestrator/router.py` | `POST /api/v1/orchestrator/query/` endpoint |
| `src/api/endpoints/ingestion/router.py` | `POST /api/v1/ingest/` endpoint |
| `src/api/session_utils.py` | Extracts `X-Session-ID` header |

### Orchestrator
| File | Purpose |
|------|---------|
| `src/orchestrator/main.py` | Routes queries to the correct agent using LLM tool calling |

### Agents
| File | Purpose |
|------|---------|
| `src/agents/playbook_review.py` | Decides between full_playbook_review or single_rule_review |
| `src/agents/doc_information.py` | Handles document info queries (summaries, key details) |

### Tools
| File | Purpose |
|------|---------|
| `src/tools/playbook_reviewer.py` | The main pipeline: load rules -> retrieve paragraphs -> evaluate -> build report |
| `src/tools/summarizer.py` | Document summarization (used by doc_information agent) |
| `src/tools/key_details.py` | Key details extraction (used by doc_information agent) |

### Schemas
| File | Purpose |
|------|---------|
| `src/schemas/playbook.py` | Pydantic models: PlaybookRule, RuleEvaluation, RuleResult, PlaybookReviewReport |
| `src/schemas/agents.py` | AgentResponse model (typed return from agents) |
| `src/schemas/errors.py` | OrchestratorResponse, AgentError, ErrorType |
| `src/schemas/registry.py` | Chunk, ParseResult models |

### Services
| File | Purpose |
|------|---------|
| `src/services/playbook_loader.py` | Loads rules from JSON files into PlaybookRule models |
| `src/services/session_manager.py` | Manages per-session state (chunk_store, vector_store, conversation history) |
| `src/services/ingestion/ingestion.py` | Orchestrates document parsing and chunk indexing |
| `src/services/registry/doc_parser.py` | Parses .docx files, creates chunks, generates embeddings |
| `src/services/vector_store/manager.py` | Manages FAISS vector store instances and chunk storage |
| `src/services/vector_store/faiss_db.py` | FAISS index wrapper (add vectors, search vectors) |
| `src/services/vector_store/embeddings/embedding_service.py` | BGE embedding model for text-to-vector conversion |
| `src/services/prompts/v1/__init__.py` | `load_prompt()` function to load .mustache templates |

### Prompt Templates
| File | Purpose | Status |
|------|---------|--------|
| `src/services/prompts/v1/orchestrator_prompt.mustache` | System prompt for orchestrator | Active |
| `src/services/prompts/v1/playbook_review_v2_agent_prompt.mustache` | System prompt for playbook review agent | Active |
| `src/services/prompts/v1/rule_evaluation_v2_prompt.mustache` | Per-rule evaluation prompt | Active |
| `src/services/prompts/v1/*_demo_*.mustache` | 7 legacy prompts (renamed, not used) | Inactive |

### Data
| File | Purpose |
|------|---------|
| `src/data/playbook_rules_v3.json` | 10 NDA-specific rules |
| `src/data/default_playbook_rules.json` | 12 general contract rules |

---

## 6. Prompt Templates Explained

### Why Mustache Templates?

Prompts are stored as `.mustache` files (not hardcoded in Python) because:
- **Separation of concerns**: Prompt engineering is separate from code logic
- **Easy to iterate**: Change a prompt without touching Python code
- **Template variables**: `{{rule_title}}` gets replaced with actual values at runtime

### The Three Active Prompts

#### 1. `orchestrator_prompt.mustache`
- **Used by:** `src/orchestrator/main.py`
- **Purpose:** Tells the LLM to pick the right agent
- **LLM sees:** User query + two agent tool definitions
- **LLM decides:** "This is a review request, route to playbook_review_agent"

#### 2. `playbook_review_v2_agent_prompt.mustache`
- **Used by:** `src/agents/playbook_review.py`
- **Purpose:** Tells the LLM to pick the right tool
- **LLM sees:** User query + two tool definitions (full review vs single rule)
- **LLM decides:** "This is a full review request, call full_playbook_review"

#### 3. `rule_evaluation_v2_prompt.mustache`
- **Used by:** `src/tools/playbook_reviewer.py`
- **Purpose:** The main evaluation prompt - evaluates one rule against contract paragraphs
- **Template variables:** `{{rule_title}}`, `{{rule_instruction}}`, `{{rule_description}}`, `{{paragraphs_text}}`
- **Called:** Once per rule (10 times for v3 playbook)

**Key features of the v2 evaluation prompt:**
- **Chain-of-thought forcing**: `_reasoning` field forces LLM to think step-by-step before answering
- **Echo-back verification**: LLM returns the rule title/instruction/description verbatim (prevents drift)
- **Anti-hallucination guards**: 8 explicit rules like "Existence is not compliance", "No cross-contamination"
- **Flat output**: `para_identifiers` as simple string list (less complex = less hallucination)

---

## 7. Data Schemas (Pydantic Models)

### Why Pydantic?

Pydantic models serve three critical purposes:
1. **Validate LLM output**: If the LLM returns invalid JSON, Pydantic catches it immediately
2. **Generate JSON Schema**: `RuleEvaluation.model_json_schema()` is passed to the OpenAI API's `response_format` parameter, forcing the LLM to output exactly the right structure
3. **Type safety**: Every field has a type, description, and constraints (e.g., `confidence: float, ge=0.0, le=1.0`)

### Model Hierarchy

```
PlaybookRule (INPUT - from JSON file)
  |
  v
RuleEvaluation (LLM OUTPUT - one per rule)
  |
  v
RuleResult (REPORT ENTRY - enriched with metadata)
  |
  v
PlaybookReviewReport (FINAL OUTPUT - all rules + statistics)
```

### PlaybookRule (Input)
```python
class PlaybookRule(BaseModel):
    title: str           # "Term and Survival of Obligations"
    instruction: str     # "Verify the NDA's duration (three years)..."
    description: str     # "This Agreement shall remain in effect..."
    category: str | None # "termination"
    standard_position: str | None
    fallback_position: str | None
```

### RuleEvaluation (LLM Output)
```python
class RuleEvaluation(BaseModel):
    reasoning: str          # Chain-of-thought (alias: _reasoning)
    rule_title: str         # Echo-back
    rule_instruction: str   # Echo-back
    rule_description: str   # Echo-back
    para_identifiers: List[str]  # ["2", "10"] - which paragraphs matched
    status: Verdict         # PASS | FAIL | NOT FOUND
    reason: str             # "The agreement states 'two years'..."
    suggestion: str         # "Extend duration to three years"
    suggested_fix: str      # Full corrected clause text
    confidence: float       # 0.0 to 1.0
    risk_level: RiskLevel   # Low | Medium | High | Critical
```

### PlaybookReviewReport (Final Output)
```python
class PlaybookReviewReport(BaseModel):
    session_id: str
    playbook_source: str           # "v3"
    statistics: ReportStatistics   # Counts of passed/failed/etc.
    overall_risk_level: RiskLevel  # Computed from all rules
    rule_results: List[RuleResult] # Sorted by risk (Critical first)
    rules_by_risk: Dict[str, List[str]]  # Grouped titles
    missing_clauses: List[str]     # Rules with NOT FOUND status
    errors: List[str]              # Any evaluation errors
```

---

## 8. How LLM Calls Work

### OpenAI Structured Output

The system uses OpenAI's **JSON Schema mode** to force the LLM to return valid structured data:

```python
response = llm.client.chat.completions.create(
    model=llm.deployment_name,
    messages=[
        {"role": "system", "content": "You are an expert Contract Review Analyst..."},
        {"role": "user", "content": rendered_prompt},
    ],
    temperature=0.0,          # Deterministic - same input = same output
    max_tokens=16384,          # Max response length
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "RuleEvaluation",
            "schema": RuleEvaluation.model_json_schema(),  # Auto-generated from Pydantic
            "strict": False,
        },
    },
)
```

**What this does:**
1. `temperature=0.0` ensures deterministic output (no randomness)
2. `response_format` with `json_schema` forces the LLM to return JSON matching our Pydantic model exactly
3. The LLM CANNOT return free-text — it MUST return valid JSON with all required fields

### OpenAI Tool/Function Calling

The orchestrator and agent use **tool calling** to let the LLM decide which function to invoke:

```python
response = await client.chat_completion(
    messages=messages,
    tools=TOOL_DEFINITIONS,   # List of available functions
    tool_choice="auto",        # LLM decides which to call
    temperature=0.3,
)
```

The LLM returns a `tool_call` object with the function name and arguments, which the code then executes.

### Parallel Execution

```python
# Semaphore limits to 5 concurrent LLM calls
semaphore = asyncio.Semaphore(5)

# All 10 rules evaluated in parallel (5 at a time)
eval_tasks = [_eval_with_limit(rule, paragraphs) for rule, paragraphs in zip(rules, all_paragraphs)]
eval_results = await asyncio.gather(*eval_tasks)
```

Since the OpenAI SDK is synchronous, each call is wrapped in `asyncio.to_thread()` to run in a thread pool without blocking the event loop.

---

## 9. Issues Encountered and How They Were Fixed

### Issue 1: 500 Internal Server Error - `AttributeError: 'dict' has no attribute 'agent'`

**When:** User sent "Review this contract" via `/api/v1/orchestrator/query/`

**Error:**
```
File "src/orchestrator/main.py", line 155, in run_orchestrator
    agent=agent_data.agent,
AttributeError: 'dict' object has no attribute 'agent'
```

**Root Cause:**
The orchestrator (`main.py:155`) expected `agent_data` to be a Pydantic model with `.agent`, `.tools_called`, `.tool_results` attributes. But the `playbook_review` agent was returning a **plain Python dict**:

```python
# playbook_review.py was returning:
return {
    "agent": "playbook_review",
    "tools_called": tools_called,
    "tool_results": tool_results,
}

# But orchestrator expected:
agent_data.agent          # dict doesn't have .agent attribute
agent_data.tools_called   # dict doesn't have .tools_called attribute
```

The `doc_information` agent worked fine because it returned `AgentResponse(...)` (a Pydantic model).

**Fix:** Changed `playbook_review.py` to return `AgentResponse` instead of a dict:

```python
# Before (broken):
return {"agent": "playbook_review", ...}

# After (fixed):
from src.schemas.agents import AgentResponse
return AgentResponse(agent="playbook_review", tools_called=tools_called, tool_results=tool_results)
```

Also changed the LLM call from raw synchronous to the async retry-wrapped method:
```python
# Before:
response = client.client.chat.completions.create(...)  # Raw, no retries

# After:
response = await client.chat_completion(...)  # Async, with retries
```

**Why It Happened:** The playbook review agent was written before the `AgentResponse` schema was standardized. The doc_information agent was already updated, but the playbook review agent was missed.

---

### Issue 2: Double-Indexing (12 chunks stored as 24)

**Symptom:**
```
Indexed 12 chunks in session 123. Total chunks in session: 12
Indexed 12 chunks in session 123. Total chunks in session: 24
```

The same 12 chunks were stored twice, creating indices 0-11 and 12-23 with identical content.

**Impact:**
- FAISS search could return paragraph ID "3" or "15" for the same text
- Different runs would reference different IDs for the same content
- Made results look inconsistent between runs

**Root Cause:**
`index_chunks_in_session()` was called in TWO places for the same chunks:

1. **Inside the parser** (`semantic_parser.py:399`):
   ```python
   # Parser indexes chunks after creating them
   index_chunks_in_session(session_data, chunks, metadata)  # First: indices 0-11
   ```

2. **In the ingestion service** (`ingestion.py:53`):
   ```python
   # Ingestion service ALSO indexes the same chunks
   index_chunks_in_session(session_data, parsed_data.chunks, parsed_data.metadata)  # Second: indices 12-23
   ```

Similarly, `doc_parser.py:319` called `index_chunks(chunks)` adding to a legacy global store.

**Fix:** Removed indexing calls from both parsers. Now only the ingestion service indexes (single source of truth):

```python
# semantic_parser.py - REMOVED:
# index_chunks_in_session(session_data, chunks, metadata)
# Replaced with comment: "indexing handled by IngestionService._parse_data()"

# doc_parser.py - REMOVED:
# index_chunks(chunks)
# Replaced with comment: "indexing handled by IngestionService._parse_data()"
```

**Why It Happened:** The parser was originally written to index chunks itself (before sessions existed). When session-based indexing was added to the ingestion service, the old indexing call inside the parser was not removed.

---

### Issue 3: Flipped/Inconsistent Results Between Runs

**Symptom:** Same document reviewed twice produced different PASS/FAIL verdicts for 2-3 borderline rules. For example:
- Run 1: "Exclusions" = FAIL, "Remedies" = PASS
- Run 2: "Exclusions" = PASS, "Remedies" = FAIL

**Root Cause:** `temperature=0.2` in the LLM call introduces randomness. For rules where the contract text is ambiguous or borderline, the LLM could land on either side of the PASS/FAIL boundary.

**Fix:** Set `temperature=0.0` for fully deterministic output:

```python
# Before:
temperature=0.2  # Some randomness

# After:
temperature=0.0  # Deterministic: same input always produces same output
```

**Why temperature=0.0 is correct here:**
- Contract review needs precision and consistency, not creativity
- You're comparing specific terms against specific rules - there IS a right answer
- Reproducible results are essential (same document = same report)
- The LLM outputs structured JSON, not creative text

**When would you use temperature > 0?**
- Creative writing, brainstorming, generating diverse options
- NOT for legal analysis, classification, or structured data extraction

---

### Issue 4: Rule 6 Had Placeholder Values

**Symptom:** Rule 6 "Term and Survival of Obligations" in `playbook_rules_v3.json` had `[number]` placeholders:

```json
"description": "...for a period of [number] years...upon [number] days prior written notice..."
```

**Impact:** The LLM couldn't compare the contract's "two years" against `[number]` years - it had no specific benchmark.

**Fix:** Replaced placeholders with specific values matching the instruction:
```json
"description": "...for a period of 3 years...upon 30 days prior written notice...for a period of 3 years from the date of disclosure..."
```

---

### Issue 5: Legacy Prompts Cluttering the Codebase

**Symptom:** 7 old prompt files from the V1 pipeline were unused but still present.

**Fix:** Renamed all 7 with `_demo_` suffix (not deleted, preserved for reference):
- `rule_evaluation_prompt.mustache` -> `rule_evaluation_demo_prompt.mustache`
- `playbook_review_agent_prompt.mustache` -> `playbook_review_agent_demo_prompt.mustache`
- etc.

---

## 10. LLM Call Count Breakdown

### Full Playbook Review ("Review this contract")

| Step | LLM Calls | What Happens |
|------|-----------|--------------|
| Orchestrator picks agent | 1 | Routes to playbook_review_agent |
| Agent picks tool | 1 | Selects full_playbook_review |
| Load 10 rules from JSON | 0 | Pure file read |
| Retrieve paragraphs (10 rules) | 0 | Local BGE embeddings + FAISS search |
| Evaluate 10 rules (parallel) | 10 | One LLM call per rule, max 5 concurrent |
| Build report | 0 | Pure Python logic |
| **TOTAL** | **12** | |

### Single Rule Review ("Check the confidentiality clause")

| Step | LLM Calls | What Happens |
|------|-----------|--------------|
| Orchestrator picks agent | 1 | Routes to playbook_review_agent |
| Agent picks tool | 1 | Selects single_rule_review |
| Retrieve paragraphs (1 rule) | 0 | Local BGE embeddings + FAISS search |
| Evaluate 1 rule | 1 | One LLM call |
| **TOTAL** | **3** | |

---

## 11. Why These Design Decisions

### Why per-rule evaluation instead of one big prompt?

- **Accuracy**: LLMs perform better on focused tasks. Evaluating 1 rule against 8 paragraphs is simpler than evaluating 10 rules against 80 paragraphs
- **Parallelism**: 10 small calls run in parallel are faster than 1 massive call
- **Reliability**: If one rule fails, the other 9 still succeed
- **Token limits**: One massive prompt might exceed the context window

### Why FAISS instead of sending the entire document?

- **Relevance**: Only the most relevant paragraphs are sent to the LLM per rule
- **Token efficiency**: 8 relevant paragraphs << entire contract
- **Cost**: Fewer tokens = lower API costs
- **Accuracy**: Less noise = better LLM focus

### Why echo-back fields (rule_title, rule_instruction, rule_description)?

The LLM returns the exact rule text it was given. This:
- **Anchors** the LLM to the specific rule (prevents drift)
- **Verifies** the LLM processed the right rule
- **Passes through** to the final report without needing to re-join data

### Why chain-of-thought (_reasoning field)?

Forcing the LLM to write its reasoning BEFORE the verdict:
- **Improves accuracy**: The LLM "thinks out loud" before deciding
- **Reduces hallucination**: Step-by-step analysis catches logical errors
- **Debugging**: We can inspect the reasoning if a verdict seems wrong
- **Hidden from user**: The `_reasoning` field is internal, not shown in the report

### Why deterministic report assembly (no LLM)?

The final report (statistics, risk grouping, sorting, missing_clauses) is built in pure Python because:
- **Consistency**: Same inputs always produce same output
- **Speed**: No API call needed
- **Cost**: Zero tokens consumed
- **Reliability**: No chance of LLM hallucination in the report structure
