# Risk & Compliance Agent — Team Discussion Document

**Prepared for:** Team Discussion
**Date:** 2026-03-11
**Status:** Pre-Development Research & Scoping

---

## 1. What Is a Risk & Compliance Agent?

A Risk & Compliance Agent is an AI-powered module that automatically analyzes a contract to:

1. **Flag high-risk terms** — clauses that are one-sided, aggressive, or expose the company to financial/legal harm (e.g., uncapped liability, broad indemnification, unreasonable termination terms)

2. **Detect deviations from company standards** — compare every clause against the company's approved internal clause library and highlight where the contract differs (e.g., "Liability clause allows unlimited claims — deviates from company standard of 5L limit")

3. **Identify missing protections** — flag clauses that should exist but don't (e.g., "Termination notice period missing", "No force majeure clause found", "No data protection clause")

4. **Catch factual/contextual errors** — detect inconsistencies within the document itself (e.g., jurisdiction says "Florida" in one section but "California" in another, party names used inconsistently, contradicting dates)

5. **Check compliance** — verify the contract meets regulatory requirements (GDPR, HIPAA, etc.) where applicable

### How Is It Different From Playbook Review?

| Aspect | Playbook Review (Existing) | Risk & Compliance (New) |
|--------|---------------------------|------------------------|
| **Input rules** | User provides specific rules to check | Agent uses built-in legal knowledge + company clause library |
| **Question it answers** | "Does this contract follow MY rules?" | "What in this contract could hurt me?" |
| **Scope** | Only checks what the playbook defines | Scans the ENTIRE contract for any risk, deviation, or error |
| **Missing items** | Flags missing rules from the playbook | Flags missing legal protections the company should have |
| **Output** | PASS / FAIL per rule | Risk severity (Critical/High/Medium/Low) per finding |
| **Who defines it** | The user defines rules | The system's legal knowledge + company standards |

**They are complementary, not overlapping.** A playbook checks your specific rules. The risk agent catches everything your playbook might have missed.

---

## 2. How Competitors Use Risk & Compliance

### 2.1 Icertis — RiskAI

**Website:** https://www.icertis.com/
**Risk Feature:** https://www.icertis.com/products/ai-applications/riskai/

**How they do it:**
- Organizations define **risk criteria by contract type** (NDA, MSA, SaaS, etc.)
- Each criterion gets a **risk score** using configurable quantitative/qualitative models
- RiskAI identifies **clause-level risks** with specific mitigation guidance
- Supports **bulk contract analysis** — scan entire contract portfolios at once
- Has a **dedicated GDPR Compliance App** that auto-detects whether a contract falls under GDPR and inserts EU-approved data privacy clauses
- Role-based dashboards show real-time risk across the entire contract repository

**Key takeaway:** Most enterprise-grade approach. Custom risk criteria per contract type + dedicated compliance modules for specific regulations (GDPR, DORA).

---

### 2.2 Luminance

**Website:** https://www.luminance.com/
**Risk Feature:** https://www.luminance.com/analyze/

**How they do it:**
- Uses a **Traffic Light System** — clauses are color-coded Red (high risk), Amber (needs review), Green (compliant)
- AI detects anomalies across **1,000+ legal concepts** automatically — no pre-configured rules needed
- Anomaly-based approach: learns what "normal" looks like, then flags anything unusual
- Can highlight risk down to **individual words** within an otherwise-safe clause
- When a clause is red-flagged, suggests **alternative compliant wording** from a clause library
- Works in **80+ languages** for cross-border contracts

**Key takeaway:** Strongest "find risks I didn't think to look for" capability. The anomaly detection approach catches unusual patterns without needing predefined rules.

---

### 2.3 LinkSquares — Risk Scoring Agent

**Website:** https://linksquares.com/
**Risk Feature:** https://linksquares.com/demo-library/risk-scoring-agent-overview/

**How they do it:**
- Assigns a **0-100 risk score** to each contract with one click
- Organizations create **custom risk profiles** — define what risk factors matter and their thresholds
- Full transparency: shows **why each risk area was scored** and how it contributes to the overall number
- Scans for specific factors: unfavorable termination terms, indemnification gaps, regulatory triggers
- **Compares risk scores across contract versions** to track improvements before renewals
- Risk dashboards with portfolio-wide visibility

**Key takeaway:** Most quantified approach. The 0-100 score with transparent breakdown is very user-friendly and actionable.

---

### 2.4 ThoughtRiver

**Website:** https://www.thoughtriver.com/
**Risk Feature:** https://www.thoughtriver.com/platform/key-features

**How they do it:**
- Uses a proprietary ontology called **Lexible** — 4,150+ lawyer-built legal concepts expressed as questions
- Each contract gets analyzed against these questions (e.g., "Is liability capped?", "Is there a force majeure clause?")
- Creates **issue cards** for every flagged risk — each card shows the risk, relevant clause, and recommended next steps
- Ships with an **out-of-the-box risk policy** that can be customized to match organizational playbooks
- Risk policies can be configured **per contract type** (different rules for NDAs vs service agreements)
- Suggests **remediation language** through a Word plugin

**Key takeaway:** The question-based ontology approach is unique. Instead of pattern matching, it asks structured questions about the contract and flags problematic answers.

---

### 2.5 Ironclad — AI Review Agent

**Website:** https://ironcladapp.com/
**Risk Feature:** https://ironcladapp.com/product/ai-based-contract-management

**How they do it:**
- **Multi-agent architecture** — separate AI agents for Review, Drafting, and Management
- Review Agent identifies missing clauses, risky terms, and compliance gaps
- Drafting Agent generates redlines (tracked changes) aligned to organizational playbooks
- Risk findings feed into **approval routing** — high-risk contracts automatically require senior review
- Produces redlines as **tracked changes in the document** (not just a report)

**Key takeaway:** Tight integration between risk detection and automated remediation (redlining). Risk findings directly trigger workflow actions.

---

### 2.6 LegalSifter — ReviewPro

**Website:** https://www.legalsifter.com/
**Risk Feature:** https://www.legalsifter.com/products/ai-contract-review

**How they do it:**
- Uses **2,200+ pre-built "AI Sifters"** — specialized algorithms each trained on a specific contract concept
- Identifies concepts that are **present, missing, or problematic**
- Auto-applies **redlines in Word** based on company positions
- Attorney-curated playbooks enforce standards and surface deviations
- Claims **95%+ accuracy** in identifying risky clauses
- Identifies the **7 highest-risk clause types**: Limitation of Liability, IP Terms, Liquidated Damages, Indemnification, Warranty Obligations, and more

**Key takeaway:** Largest pre-built knowledge base (2,200+ sifters). Strong emphasis on accuracy metrics and attorney-curated quality.

---

### 2.7 BlackBoiler

**Website:** https://www.blackboiler.com/
**Risk Feature:** https://www.blackboiler.com/resources/contract-review-ai/

**How they do it:**
- **Learns from historical markups** — studies how your lawyers have previously redlined similar clauses and applies those patterns to new contracts
- **ContextAI** explains every redline: what rule triggered it, why the change was made, and examples of how others edited similar language
- Produces changes directly in **Microsoft Word Track Changes**
- **Smart Clause Library** with one-click access to standard clauses that auto-update to fit contract-specific terms
- Gets smarter over time as more markups are processed

**Key takeaway:** Learning-based approach — the system improves by studying your team's actual editing patterns. Unique in the market.

---

### 2.8 Evisort (now Workday Contract Intelligence)

**Website:** https://www.workday.com/en-us/products/contract-management/contract-intelligence.html

**How they do it:**
- **230+ pre-trained data points** covering financial terms, legal clauses, operational details, and compliance requirements
- Automatically identifies and fixes risky language to align with regulations and strategic goals
- Continuous monitoring for compliance issues with **automated alerts**
- Specific support for **GDPR standard contractual clauses** and DPA management
- Portfolio-wide search, reporting, and alerting across all contracts

**Key takeaway:** Strongest compliance monitoring — continuous, automated alerts when contracts fall out of compliance. Acquired by Workday for enterprise integration.

---

### 2.9 SpotDraft — VerifAI

**Website:** https://www.spotdraft.com/
**Risk Feature:** https://www.spotdraft.com/for-legal

**How they do it:**
- **VerifAI** flags risky or non-standard clauses instantly within Microsoft Word
- Compares contracts against organizational guidelines and playbooks, surfacing deviations inline
- Applies edits and redlines directly in the document based on playbook standards
- Runs **entirely on-device** (on Snapdragon processors) — contract data never leaves the local machine
- Important for **data sovereignty and GDPR** compliance — no cloud processing of sensitive contracts

**Key takeaway:** On-device processing is a strong privacy differentiator. The no-cloud approach addresses enterprise data security concerns.

---

### 2.10 Kira Systems (Litera)

**Website:** https://www.litera.com/
**Risk Feature:** https://www.litera.com/products/kira

**How they do it:**
- **1,400+ pre-built smart fields** across 40+ legal areas for clause extraction
- Primarily used for **M&A due diligence** — scanning hundreds of contracts to identify risks across a deal
- **Heat maps** showing provision frequency and risk patterns across document sets
- 90%+ extraction accuracy through hybrid AI
- Users can create **custom smart fields** for organization-specific risk provisions

**Key takeaway:** Best for high-volume contract analysis (due diligence). Extraction-focused rather than risk-scoring focused.

---

## 3. Common Patterns Across All Competitors

### Risk Categories Used by Most Platforms

| # | Risk Category | What It Checks |
|---|--------------|----------------|
| 1 | Indemnification | Scope, caps, mutual vs one-way, carve-outs |
| 2 | Limitation of Liability | Caps, exclusions, consequential damages |
| 3 | Termination | Notice periods, for cause vs convenience, survival |
| 4 | Confidentiality | Scope, duration, exceptions, return/destruction |
| 5 | IP / Ownership | Assignment, licensing, work-for-hire |
| 6 | Force Majeure | Triggering events, notice, suspension rights |
| 7 | Governing Law / Jurisdiction | Venue, choice of law, arbitration |
| 8 | Warranties / Representations | Scope, disclaimers, survival |
| 9 | Assignment / Change of Control | Transferability, consent requirements |
| 10 | Data Privacy | DPA, data handling, cross-border transfers, GDPR |
| 11 | Non-Compete / Non-Solicitation | Scope, duration, geographic limits |
| 12 | Payment Terms | Net days, late fees, financial exposure |
| 13 | Insurance Obligations | Coverage type, minimum amounts |
| 14 | Liquidated Damages | Pre-determined penalty amounts |

### Risk Presentation — 3 Models Used in the Market

| Model | Used By | Description |
|-------|---------|-------------|
| **Traffic Light** (Red/Amber/Green) | Luminance, Icertis | Color-coded severity on each clause |
| **Numeric Score** (0-100) | LinkSquares, Icertis | Quantitative per-contract risk score |
| **Issue Cards / Findings List** | ThoughtRiver, Ironclad | Prioritized list with explanation + next steps |

### What Separates Good From Basic

| Basic | Good | Great |
|-------|------|-------|
| Just flags "this is risky" | Explains WHY it's risky | Suggests alternative language to fix it |
| Per-clause findings only | Per-clause + overall risk level | Category breakdown + overall score |
| Generic for all contracts | Aware of contract type (NDA vs MSA) | Organization-specific thresholds |
| No citations | Cites the clause text | Cites + cross-references against company standard |

---

## 4. How Risk & Compliance Can Work in Our Application

### Our Architecture

Our application uses **standalone API endpoints per agent** with:
- Document ingestion into per-session FAISS vector stores
- Azure OpenAI (GPT-4) for LLM analysis
- Pystache prompt templates
- Structured Pydantic responses

### Proposed Approach: Three Capabilities, One Endpoint

The Risk & Compliance agent would be a **single endpoint** that performs three types of analysis on the ingested contract:

#### Capability 1: Company Standard Deviation Check
- Compare each clause against a **company clause library** (JSON file with approved standards)
- For each clause type, check if the contract's language matches the company's approved position
- Flag deviations with: what the contract says vs. what the company standard is
- Example output: *"Liability clause allows unlimited claims — deviates from company standard (5L limit)"*

#### Capability 2: General Risk Scan
- LLM-driven open-ended analysis using its legal knowledge
- Scan the full contract for aggressive terms, one-sided provisions, missing protections
- No predefined rules needed — the LLM identifies risks based on legal best practices
- Example output: *"Termination notice period missing"*, *"Indemnification is one-sided — only Party A indemnifies"*

#### Capability 3: Factual & Consistency Check
- Detect internal contradictions and errors within the document
- Jurisdiction mismatches, party name inconsistencies, contradicting dates, wrong references
- Example output: *"Governing law states Florida in Section 12 but California in Section 3"*

### Proposed Output Structure

```
{
  "session_id": "...",
  "overall_risk_level": "High | Medium | Low",
  "risk_summary": "Brief overview of the contract's risk posture",
  "total_findings": 12,
  "findings_by_severity": {
    "critical": 2,
    "high": 3,
    "medium": 5,
    "low": 2
  },
  "findings": [
    {
      "finding_type": "deviation | risk | missing_clause | inconsistency",
      "category": "liability | termination | indemnification | ...",
      "severity": "critical | high | medium | low",
      "title": "Uncapped Liability",
      "description": "Liability clause allows unlimited claims",
      "contract_text": "\"...exact quoted text from the contract...\"",
      "company_standard": "Liability capped at 5L (only for deviation type)",
      "suggested_fix": "Alternative language to mitigate this risk",
      "section_reference": "Section 8, Paragraph 3"
    }
  ],
  "category_summary": [
    {
      "category": "liability",
      "risk_level": "high",
      "findings_count": 3
    }
  ]
}
```

### How It Would Work (Flow)

```
1. User uploads contract → /api/v1/ingest/ (already exists)

2. User calls → /api/v1/risk-compliance/analyze
   Input: session_id + (optional) clause_library reference

3. Agent internally:
   a. Retrieves all document chunks from the session
   b. Runs Capability 1: Compares chunks against company clause library
   c. Runs Capability 2: LLM scans for general risks + missing protections
   d. Runs Capability 3: LLM checks for internal inconsistencies/errors
   e. Assembles structured response with all findings

4. Returns: Structured risk report (JSON)
```

---

## 5. Open Questions for Team Discussion

These questions need to be answered before development begins:

### About the Company Clause Library

**Q1:** Do we have (or will we create) a company clause library — a JSON/file with approved clause standards that the agent compares against?
- If yes: What format? Who maintains it? How many clause types initially?
- If no: Should the agent rely purely on the LLM's legal knowledge for risk assessment?
- Or both: LLM for general risk + a clause library for company-specific deviation checks?

**Q2:** Should the clause library be **per contract type** (different standards for NDAs vs MSAs vs SaaS agreements)? Or one universal library?

### About Input & Scope

**Q3:** Does the agent review the **entire contract** at once (given a session_id)? Or does the user select specific sections/paragraphs to check?

**Q4:** Should the agent work with the document's raw chunks from FAISS, or should it get the full document text? (Full text catches cross-section inconsistencies better; chunks are faster and cheaper)

### About the Risk Categories

**Q5:** Which risk categories should we support in v1? All 14 from the industry standard list? Or start with the most critical ones (Liability, Indemnification, Termination, IP, Data Privacy)?

**Q6:** For the factual/consistency check (Capability 3) — how deep should this go?
- Just jurisdiction and party name consistency?
- Also dates, defined terms, cross-references?
- Should it cross-reference against external facts (e.g., the deal was supposed to be California-governed)?

### About Output & Scoring

**Q7:** Should we use a numeric score (0-100 like LinkSquares) or a categorical level (Critical/High/Medium/Low like most platforms)?

**Q8:** Should the agent provide **suggested fix / alternative language** for each finding? Or just flag the issue?

**Q9:** Should findings include a **confidence score** (how sure the AI is about each finding)?

### About Integration

**Q10:** Should the risk report reference findings from other agents? For example:
- "Key Details agent extracted governing law as California, but Section 12 says Florida"
- "Playbook review already flagged this as FAIL — Risk agent adds severity context"

**Q11:** Should the agent be callable **before or after** playbook review? Or independently?

### About Compliance

**Q12:** Do we need specific compliance rule packs (GDPR, HIPAA) in v1? Or is general risk assessment sufficient for now, with compliance packs added later?

---

## 6. Recommended Starting Point

Based on market research and our current architecture, here is a suggested phased approach:

### v1 (MVP)
- General risk scan (LLM-driven, no predefined rules)
- Missing clause detection
- Overall risk level (High/Medium/Low) + per-finding severity
- Suggested fix for each finding
- Single endpoint: `POST /api/v1/risk-compliance/analyze`

### v2 (After Team Feedback)
- Company clause library integration (deviation checking)
- Factual/consistency error detection
- Risk category breakdown
- Per-category risk levels

### v3 (Future)
- Compliance rule packs (GDPR, HIPAA)
- Contract-type-specific risk profiles
- Integration with other agents (cross-reference findings)

---

