# Finalized Prompt Templates

This document covers all production-ready prompt templates, their input/output contracts, and integration details. Use this as the single reference when integrating prompts into any codebase.

**Last updated:** March 6, 2026
**Status:** Tested and confirmed
**Template engine:** Mustache (via `pystache`)
**LLM response format:** JSON Schema (structured output enforced)
**Recommended temperature:** 0.2

---

## 1. General Review

**File:** `src/services/prompts/v1/general_review.mustache`

**Purpose:** Evaluates a single user-selected paragraph against a rule or guideline. Designed for the Word plugin where a user selects a paragraph, provides a rule, and gets a compliance check with surgical fixes.

### Input Variables

| Variable | Type | Description |
|----------|------|-------------|
| `paragraph` | string | The single paragraph selected by the user in Word |
| `rule` | string | The rule or guideline to validate the paragraph against |

### Output Schema

```json
{
  "reason": "2-3 sentence explanation of compliance or non-compliance. References exact text from the paragraph.",
  "suggested_fix": [
    {
      "original_text": "exact verbatim substring from the paragraph that needs fixing",
      "fixed_text": "corrected version of that specific text",
      "fix_summary": "one-line description of what was changed and why"
    }
  ]
}
```

- `suggested_fix` is an **empty array** `[]` when the paragraph is compliant.
- Each fix is shown to the user **individually** with an **accept or ignore** option.
- `original_text` is always an **exact substring** of the paragraph (can be used to highlight in the UI).

### Pydantic Models

```python
from typing import List
from pydantic import BaseModel

class GeneralReviewFix(BaseModel):
    original_text: str
    fixed_text: str
    fix_summary: str

class GeneralReviewLLMResponse(BaseModel):
    reason: str
    suggested_fix: List[GeneralReviewFix]
```

### Request / Response Models (API layer)

```python
class GeneralReviewRequest(BaseModel):
    paragraph: str
    rule: str

class GeneralReviewResponse(BaseModel):
    paragraph: str
    rule: str
    reason: str
    suggested_fix: List[GeneralReviewFix]
```

### API Endpoint

```
POST /api/v1/chat/general-review/
```

### Key Design Decisions

- **Rule-only input** (no free-form questions). This prevents hallucination since every fix is grounded in the user-provided rule.
- **Single paragraph only** sent to the LLM. No document context, no surrounding paragraphs. This eliminates hallucination from external context.
- **Surgical fixes** — only the problematic text is changed, not the entire paragraph.
- **No verdict field** — the `reason` field explains compliance status, and the presence/absence of `suggested_fix` items indicates whether issues were found.

---

## 2. Summarizer

**File:** `src/services/prompts/v1/summary_prompt_template.mustache`

**Purpose:** Generates a structured markdown summary of an ingested legal document. Covers document type, parties, obligations, dates, and risks.

### Input Variables

| Variable | Type | Description |
|----------|------|-------------|
| `text` | string | Full document text (concatenated chunks from the ingested document) |

### Output Schema

The LLM returns a single string in markdown format following this structure:

```
**1. Document Type & Purpose** (1-2 sentences)
**2. Parties Involved** (1-2 sentences)
**3. Key Terms & Obligations** (3-5 bullet points)
**4. Critical Dates & Deadlines** (if any — skipped if no dates exist)
**5. Notable Risks or Concerns** (1-3 points)
```

### Pydantic Models

```python
from pydantic import BaseModel, Field

class SummaryToolResponse(BaseModel):
    summary: str = Field(..., description="Summary of the given document.")
```

### API Endpoint

```
GET /api/v1/DocInfo/summarizer
Headers: X-Session-ID: <session_id>
```

Requires a document to be ingested first via `/api/v1/ingest/` or `/api/v1/ingest-json/`.

### Key Design Decisions

- **Markdown output** — the summary is rendered directly in the UI, not parsed as JSON fields.
- **Target 150-300 words** — concise but complete.
- **Sections are skipped** if no relevant information exists (e.g., no dates section if no dates in the document).
- **One few-shot example** included in the prompt for grounding. This costs tokens but ensures consistent output structure.

---

## 3. Key Information

**File:** `src/services/prompts/v1/key_information_prompt.mustache`

**Purpose:** Extracts structured key contract details (dates, parties, value, duration, governing law, etc.) from an ingested legal document. Returns a strict JSON object.

### Input Variables

| Variable | Type | Description |
|----------|------|-------------|
| `contract_text` | string | Full document text (concatenated chunks from the ingested document) |

### Output Schema

```json
{
  "effective_date": {
    "value": "<ISO date | 'conditional -- see information' | 'blank placeholder' | null>",
    "information": "<source clause or null>"
  },
  "expiration_date": {
    "value": "<ISO date | resolved duration | 'perpetual' | null>",
    "information": "<source clause, calculation, and any early termination notice | null>"
  },
  "contract_value": {
    "value": "<amount with currency | null>",
    "information": "<clause reference, or explanation if null>"
  },
  "duration": {
    "value": "<resolved period e.g. '3 years / 36 months' | null>",
    "information": "<exact contract clause and conversion | null>"
  },
  "net_term": {
    "value": "<e.g. 'Net 30' | null>",
    "information": "<clause reference | null>"
  },
  "parties": [
    {
      "name": "<full legal name>",
      "role": "<role label as used in contract>",
      "address": "<address | null>",
      "email": "<email | null>"
    }
  ],
  "contract_type": {
    "value": "<official title | null>",
    "information": "<source | null>"
  },
  "governing_law": {
    "value": "<jurisdiction and venue | null>",
    "information": "<clause reference | null>"
  },
  "notes": "<observations separated by '; ' | null>"
}
```

### Pydantic Models

```python
from typing import List, Optional
from pydantic import BaseModel, Field

class KeyInformationData(BaseModel):
    value: Optional[str] = Field(..., description="Value of the particular key field.")
    information: Optional[str] = Field(..., description="Detailed description of the key information extracted.")

class KeyInfomationPartiesSchema(BaseModel):
    name: str = Field(..., description="Full legal name exactly as stated.")
    role: str = Field(..., description="Role label exactly as used in the contract.")
    address: Optional[str] = Field(None, description="Address from signature block or notice section.")
    email: Optional[str] = Field(None, description="Email from signature block or notice section.")

class KeyInformationToolResponse(BaseModel):
    effective_date: KeyInformationData
    expiration_date: KeyInformationData
    contract_value: KeyInformationData
    duration: KeyInformationData
    net_term: KeyInformationData
    contract_type: KeyInformationData
    governing_law: KeyInformationData
    notes: Optional[str]
    parties: List[KeyInfomationPartiesSchema]
```

### API Endpoint

```
GET /api/v1/DocInfo/key-information
Headers: X-Session-ID: <session_id>
```

Requires a document to be ingested first via `/api/v1/ingest/` or `/api/v1/ingest-json/`.

### Key Design Decisions

- **Null discipline** — missing fields return `null`, never empty strings or "N/A". This makes frontend conditional rendering simple.
- **Time resolution** — all time-related fields are calculated to concrete values (e.g., "3 years / 36 months") even when the contract uses relative language.
- **One few-shot example** included (NDA with conditional dates and blank placeholders). Covers the most complex edge cases.
- **Notes field** flags ambiguities, blank placeholders, and anything requiring human attention.

---

## Integration Checklist

When integrating any of these prompts into a codebase:

| # | Check | Details |
|---|-------|---------|
| 1 | Template engine | Must use Mustache (`pystache` in Python). If using Jinja2 or string formatting, adapt `{{variable}}` syntax accordingly. |
| 2 | Variable names | The context dict keys must match the mustache variable names exactly. |
| 3 | JSON Schema enforcement | The LLM call must use `response_format` with `json_schema` type to enforce structured output. |
| 4 | Pydantic models | Use the exact models listed above. Field names and types must match the prompt's output schema. |
| 5 | Temperature | Use 0.2 for consistent, deterministic results. |
| 6 | File encoding | Read prompt files with `encoding="utf-8"` to avoid Windows encoding issues. |
| 7 | System message | Default: `"Extract the information and return valid JSON."` |
