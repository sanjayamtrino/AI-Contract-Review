---
phase: 05-describe-and-draft-agent
plan: 01
subsystem: api
tags: [pydantic, mustache, fastapi, llm, drafting, cosine-similarity]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: LLM client, prompt loader, session manager, embedding service
provides:
  - POST /api/v1/draft/generate endpoint
  - DraftRequest, DraftResponse, DraftVersion, DraftLLMResponse schemas
  - Drafter tool with document context extraction and similar-clause detection
  - Describe & Draft agent dispatcher
affects: [orchestrator-routing, risk-compliance-agent]

# Tech tracking
tech-stack:
  added: []
  patterns: [document-context-extraction, similar-clause-detection-via-cosine, creative-temperature-0.7]

key-files:
  created:
    - src/schemas/draft.py
    - src/services/prompts/v1/describe_draft_prompt.mustache
    - src/tools/drafter.py
    - src/agents/describe_draft.py
    - src/api/endpoints/draft/__init__.py
    - src/api/endpoints/draft/router.py
  modified:
    - src/api/main.py

key-decisions:
  - "Temperature 0.7 for draft generation (creative task) vs 0.1-0.2 for analysis agents"
  - "Similar clause threshold 0.60 (lower than compare agent's 0.72) to catch loosely related clauses"
  - "Single generic Mustache prompt handles all clause types via document context conditionals"

patterns-established:
  - "Draft agent pattern: schemas -> prompt -> tool -> agent -> router -> main registration"
  - "Optional document context enrichment with best-effort extraction (never blocks)"

requirements-completed: []

# Metrics
duration: 3min
completed: 2026-04-13
---

# Phase 5 Plan 01: Describe & Draft Agent Summary

**POST /api/v1/draft/generate endpoint producing 3 style alternatives (Formal, Plain English, Concise) with optional document context and similar-clause detection**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-13T14:11:29Z
- **Completed:** 2026-04-13T14:14:20Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- Full Describe & Draft agent with Pydantic schemas enforcing exactly 3 drafts via min_length/max_length
- Mustache prompt with conditional document context and similar-clause awareness sections
- Drafter tool with document excerpt extraction, cosine-similarity clause detection, and graceful error handling
- Endpoint registered at /api/v1/draft/generate in the main FastAPI app

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Pydantic schemas, Mustache prompt, and drafter tool** - `34f7f60` (feat)
2. **Task 2: Create agent dispatcher, endpoint router, and register in main app** - `cedfabf` (feat)

## Files Created/Modified
- `src/schemas/draft.py` - DraftRequest, DraftResponse, DraftVersion, DraftLLMResponse models
- `src/services/prompts/v1/describe_draft_prompt.mustache` - Generic prompt with 3 style instructions and document context conditionals
- `src/tools/drafter.py` - Business logic: context extraction, similarity detection, LLM generation
- `src/agents/describe_draft.py` - Thin agent dispatcher delegating to drafter tool
- `src/api/endpoints/draft/__init__.py` - Package init
- `src/api/endpoints/draft/router.py` - POST /generate endpoint
- `src/api/main.py` - Draft router registration at /api/v1/draft

## Decisions Made
- Temperature 0.7 for creative draft generation (higher than 0.1-0.2 used for analysis agents)
- Similar clause threshold at 0.60 to catch loosely related existing clauses without being too restrictive
- Single generic Mustache prompt covers all clause/agreement types via document context conditionals

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Draft agent complete and registered; ready for orchestrator routing integration
- Risk & Compliance agent (Phase 6) can proceed independently

---
*Phase: 05-describe-and-draft-agent*
*Completed: 2026-04-13*
