# Roadmap: AI Contract Review

## Overview

AI-powered contract review backend with standalone agent endpoints. Each agent has its own API endpoint(s). The orchestrator exists but is parked — it can be wired up later when the UI is finalized.

## Architecture Decision (2026-03-11)

**Standalone endpoints per agent** — no orchestrator dependency for now. Each agent is independently callable via its own REST endpoint. Orchestrator routing can be layered on top later.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Foundation Hardening** - Infrastructure, ingestion, FAISS, LLM integration, retry, prompts
- [x] **Phase 2: Custom Playbook Review Agent** - Rule evaluation with statistical + AI review endpoints
- [x] **Phase 3: Doc Information, General Review & DocChat** - Summarizer, key details, general review, RAG Q&A endpoints
- [x] **Phase 4: Version Compare Agent** - Compare contract versions, highlight additions/deletions/modifications
- [ ] **Phase 5: Describe & Draft Agent** - Generate clauses/agreements from user descriptions in 3 styles
- [ ] **Phase 6: Risk & Compliance Agent** - Risk assessment, unfavorable terms, compliance flags
- [ ] **Phase 7: Anti-Hallucination & Quality** - Citation verification, confidence scoring, prompt injection protection

## Phase Details

### Phase 1: Foundation Hardening [COMPLETE]
**Goal**: Solid infrastructure — document ingestion, FAISS vector store, Azure OpenAI integration, retry logic, structured responses.
**Completed**: 2026-02-24
**Endpoints**:
  - `POST /api/v1/ingest/` — DOCX upload
  - `POST /api/v1/ingest-json/` — JSON text ingestion

### Phase 2: Custom Playbook Review Agent [COMPLETE]
**Goal**: Review contracts against custom playbook rules with pass/fail verdicts.
**Completed**: 2026-03 (standalone endpoints)
**Endpoints**:
  - `POST /api/v1/chat/playbook/statistical-review` — similarity-only matching
  - `POST /api/v1/chat/playbook/ai-review` — per-rule LLM evaluation
  - `POST /api/v1/chat/playbook/test-ai` — master playbook review
  - `POST /api/v1/chat/playbook/ai-rule-Review` — single rule review

### Phase 3: Doc Information, General Review & DocChat [COMPLETE]
**Goal**: Document summary, key details extraction, general guideline review, and freeform RAG Q&A.
**Completed**: 2026-03 (standalone endpoints)
**Endpoints**:
  - `GET /api/v1/DocInfo/summarizer` — contract summary
  - `GET /api/v1/DocInfo/key-information` — key details extraction
  - `POST /api/v1/chat/general-review/` — paragraph-level guideline review
  - `POST /api/v1/chat/query/` — freeform document Q&A (RAG)

### Phase 4: Version Compare Agent [COMPLETE]
**Goal**: Compare two versions of a contract — highlight additions, deletions, and modifications with risk assessment per change.
**Completed**: 2026-04-13
**Depends on**: Phase 1 (infrastructure)
**Requirements:** VERS-01, VERS-02
**Success Criteria** (what must be TRUE):
  1. Endpoint accepts session_id with two document IDs and returns structured comparison
  2. Additions, deletions, and modifications identified with specific clause references
  3. Changes grouped by contract section
  4. Risk level assigned per change (high/medium/low)
  5. Response follows existing Pydantic schema patterns

Plans:
- [x] 04-01: Version Compare Agent (schema, prompt, endpoint, service)

### Phase 5: Describe & Draft Agent
**Goal**: Generate new clauses or agreement sections from user descriptions, providing 3 style alternatives (Formal, Plain English, Concise). Optionally enriches drafts with document context when a contract is loaded.
**Depends on**: Phase 1 (infrastructure)
**Success Criteria** (what must be TRUE):
  1. Endpoint accepts session_id + user prompt and returns 3 draft alternatives
  2. Each draft has a distinct style: Formal, Plain English, Concise
  3. Response includes a clean summary restating the user's request
  4. When a document is loaded, drafts use party names, governing law, and contract style from the document
  5. If a similar clause already exists in the document, a note flags it (still drafts)
  6. Single generic Mustache prompt handles all clause/agreement types
  7. Response follows existing Pydantic schema patterns

Plans:
- [ ] 05-01: Describe & Draft Agent (schema, prompt, tool, agent, endpoint)

### Phase 6: Risk & Compliance Agent
**Goal**: Assess contract risk — flag risky, one-sided, or unfavorable terms; provide overall risk level; check compliance concerns.
**Depends on**: Phase 1 (infrastructure)
**Requirements:** RISK-01, RISK-02
**Success Criteria** (what must be TRUE):
  1. Endpoint accepts a session_id and returns structured risk assessment
  2. Risky, one-sided, or unfavorable terms are flagged with specific clause references
  3. Overall risk level (high/medium/low) provided with rationale
  4. Compliance concerns identified where applicable
  5. Response follows existing Pydantic schema patterns

Plans:
- [ ] 06-01: Risk & Compliance agent (schema, prompt, endpoint, service)

### Phase 7: Anti-Hallucination & Quality
**Goal**: Programmatic verification of citation accuracy, confidence scoring, prompt injection protection across all agents.
**Depends on**: Phase 6
**Success Criteria** (what must be TRUE):
  1. All agent responses include citations; responses without citations are flagged/rejected
  2. All inferences include confidence level (high/medium/low)
  3. Python-level citation verification checks cited text exists in source document
  4. Ambiguous clauses surfaced to user for clarification, never guessed
  5. Prompt injection in contract text does not alter agent behavior

Plans:
- [ ] 07-01: Citation enforcement and programmatic verification (rapidfuzz)
- [ ] 07-02: Confidence scoring, ambiguity surfacing, and prompt injection protection

## Progress

**Execution Order:** 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7

| Phase | Status | Completed |
|-------|--------|-----------|
| 1. Foundation Hardening | Complete | 2026-02-24 |
| 2. Custom Playbook Review Agent | Complete | 2026-03 |
| 3. Doc Info, General Review & DocChat | Complete | 2026-03 |
| 4. Version Compare Agent | Complete | 2026-04-13 |
| 5. Describe & Draft Agent | Not started | - |
| 6. Risk & Compliance Agent | Not started | - |
| 7. Anti-Hallucination & Quality | Not started | - |
