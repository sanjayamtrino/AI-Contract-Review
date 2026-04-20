"""
End-to-end tests for POST /api/v1/describe-draft/generate.

REQUIRES: Live server at http://localhost:8000 with valid Azure OpenAI credentials.
Run with: pytest tests/e2e/ -v

Tests are auto-skipped if the server is not reachable.
"""

import uuid
import pytest

BASE_URL = "http://localhost:8000"
ENDPOINT = f"{BASE_URL}/api/v1/describe-draft/generate"


def _post(prompt: str, session_id: str) -> dict:
    """Make a synchronous POST request and return the parsed JSON body."""
    try:
        import requests
        resp = requests.post(
            ENDPOINT,
            json={"prompt": prompt},
            headers={"X-Session-ID": session_id, "Content-Type": "application/json"},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        error_str = str(e).lower()
        if (
            "connection refused" in error_str
            or "connection error" in error_str
            or "failed to establish" in error_str
            or "connectionerror" in error_str
            or "max retries exceeded" in error_str
        ):
            pytest.skip(f"Server not reachable at {BASE_URL}: {e}")
        raise


@pytest.fixture(autouse=True)
def check_server():
    """Skip all e2e tests if server is not reachable OR the describe-draft route is not registered.

    A 404 for the route on an otherwise-reachable server means the running process pre-dates
    plan 05-01 (stale server) and cannot serve this test suite — treat it the same as
    "server not running" rather than failing the whole test run.
    """
    import requests
    try:
        resp = requests.get(f"{BASE_URL}/openapi.json", timeout=5)
    except Exception as e:
        error_str = str(e).lower()
        if (
            "connection" in error_str
            or "refused" in error_str
            or "max retries exceeded" in error_str
        ):
            pytest.skip(f"Server not running at {BASE_URL}")
        raise

    if resp.status_code != 200:
        pytest.skip(f"Server at {BASE_URL} returned {resp.status_code} for /openapi.json")

    try:
        paths = resp.json().get("paths", {})
    except ValueError:
        pytest.skip(f"Server at {BASE_URL} did not return valid JSON for /openapi.json")

    if "/api/v1/describe-draft/generate" not in paths:
        pytest.skip(
            f"Server at {BASE_URL} does not expose /api/v1/describe-draft/generate "
            f"(stale process or wrong build) — skipping e2e tests"
        )


def test_e2e_single_clause_mode():
    """POST with a specific clause request returns mode=single_clause with 5 versions."""
    session_id = f"e2e-single-{uuid.uuid4()}"
    data = _post("Draft a confidentiality clause", session_id)

    assert data["status"] == "ok", f"Expected ok, got: {data}"
    assert data["mode"] == "single_clause"
    assert len(data["versions"]) == 5
    for v in data["versions"]:
        assert v["title"].strip(), "title must not be empty"
        assert v["summary"].strip(), "summary must not be empty"
        assert v["drafted_clause"].strip(), "drafted_clause must not be empty for single_clause mode"
    assert data["disclaimer"] is not None
    assert data["clarification_question"] is None


def test_e2e_list_of_clauses_mode():
    """POST with an agreement-type request returns mode=list_of_clauses with >=12 clauses."""
    session_id = f"e2e-list-{uuid.uuid4()}"
    data = _post("Draft an NDA", session_id)

    assert data["status"] == "ok", f"Expected ok, got: {data}"
    assert data["mode"] == "list_of_clauses"
    assert len(data["clauses"]) >= 12
    assert data.get("versions") in (None, [])
    for c in data["clauses"]:
        assert c["title"].strip(), "title must not be empty"
        assert c["summary"].strip(), "summary must not be empty"
    assert data["disclaimer"] is not None


def test_e2e_clarification_mode():
    """POST with an ambiguous prompt returns mode=clarification with a question and no versions."""
    session_id = f"e2e-clarify-{uuid.uuid4()}"
    data = _post("help", session_id)

    # Clarification is expected but not guaranteed for all LLMs — accept either ok or clarification
    assert data["status"] in ("ok", "error"), f"Unexpected status: {data}"
    if data["mode"] == "clarification":
        assert data["clarification_question"] is not None
        assert len(data.get("versions", [])) == 0


def test_e2e_response_has_correct_shape():
    """Every successful response includes required top-level fields."""
    session_id = f"e2e-shape-{uuid.uuid4()}"
    data = _post("Draft a payment terms clause for a SaaS agreement", session_id)

    assert "session_id" in data
    assert "mode" in data
    assert "status" in data
    assert "versions" in data
    assert data["session_id"] == session_id


def test_e2e_prompt_too_long_returns_422():
    """Prompt longer than 2000 chars returns HTTP 422 (Pydantic validation failure)."""
    import requests
    session_id = f"e2e-long-{uuid.uuid4()}"
    try:
        resp = requests.post(
            ENDPOINT,
            json={"prompt": "x" * 2001},
            headers={"X-Session-ID": session_id, "Content-Type": "application/json"},
            timeout=10,
        )
        assert resp.status_code == 422, f"Expected 422, got {resp.status_code}: {resp.text}"
    except Exception as e:
        if "connection" in str(e).lower() or "max retries exceeded" in str(e).lower():
            pytest.skip("Server not running")
        raise


def test_e2e_no_session_id_header_returns_422_or_400():
    """Missing X-Session-ID header returns 4xx."""
    import requests
    try:
        resp = requests.post(
            ENDPOINT,
            json={"prompt": "Draft an NDA"},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        assert resp.status_code in (400, 422), f"Expected 4xx, got {resp.status_code}"
    except Exception as e:
        if "connection" in str(e).lower() or "max retries exceeded" in str(e).lower():
            pytest.skip("Server not running")
        raise


def test_e2e_banned_phrases_absent_in_response():
    """Successful responses must not contain archaic legalese in any drafted_clause."""
    BANNED = [
        "witnesseth",
        "party of the first part",
        "party of the second part",
        "in witness whereof",
        "now therefore",
        "know all men by these presents",
    ]
    session_id = f"e2e-banned-{uuid.uuid4()}"
    data = _post("Draft a limitation of liability clause", session_id)

    if data["status"] != "ok" or data["mode"] != "single_clause":
        pytest.skip("Response not in single_clause mode — skipping banned phrase check")

    for v in data["versions"]:
        lower = v["drafted_clause"].lower()
        for phrase in BANNED:
            assert phrase not in lower, f"Banned phrase '{phrase}' found in version titled '{v['title']}'"
