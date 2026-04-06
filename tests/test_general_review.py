"""
Test script for the General Review endpoint.
Run: poetry run python tests/test_general_review.py
Requires the server to be running at http://localhost:8000
"""

import requests

BASE_URL = "http://localhost:8000/api/v1/chat/general-review/"

test_cases = [
    {
        "name": "Test 1: Termination Notice Period (non-compliant, 2 fixes expected)",
        "paragraph": (
            "Either party may terminate this Agreement by providing 15 days "
            "written notice to the other party. Upon termination, all confidential "
            "information must be returned within 5 business days."
        ),
        "rule": (
            "The termination notice period must be at least 30 days and "
            "confidential information must be destroyed, not just returned."
        ),
    },
    {
        "name": "Test 2: Liability Cap (compliant, 0 fixes expected)",
        "paragraph": (
            "The total aggregate liability of either party under this Agreement "
            "shall not exceed the total fees paid or payable during the twelve "
            "month period immediately preceding the claim."
        ),
        "rule": "Liability must be capped at no more than 12 months of fees paid.",
    },
    {
        "name": "Test 3: Indemnification Scope (non-compliant, 2 fixes expected)",
        "paragraph": (
            "The Vendor shall indemnify the Client against all third-party claims "
            "arising from intellectual property infringement. The Vendor's total "
            "indemnification obligation shall not exceed USD 50,000."
        ),
        "rule": (
            "Indemnification must cover both IP infringement and data breaches "
            "with no monetary cap on the indemnification obligation."
        ),
    },
    {
        "name": "Test 4: Confidentiality Survival (non-compliant, 1 fix expected)",
        "paragraph": (
            "All confidentiality obligations under this Agreement shall remain "
            "in effect for a period of two years following the termination or "
            "expiration of this Agreement."
        ),
        "rule": "Confidentiality obligations must survive for at least five years after termination.",
    },
    {
        "name": "Test 5: Governing Law (compliant, 0 fixes expected)",
        "paragraph": (
            "This Agreement shall be governed by and construed in accordance "
            "with the laws of the State of Delaware, without regard to its "
            "conflict of laws provisions."
        ),
        "rule": "The governing law must be the State of Delaware.",
    },
]


def run_tests():
    print("=" * 60)
    print("GENERAL REVIEW ENDPOINT TEST")
    print("=" * 60)

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"TEST {i}: {test['name']}")
        print(f"{'=' * 60}")
        print(f"  Paragraph: {test['paragraph'][:80]}...")
        print(f"  Rule:      {test['rule'][:80]}...")
        print()

        payload = {
            "paragraph": test["paragraph"],
            "rule": test["rule"],
        }

        try:
            response = requests.post(BASE_URL, json=payload, timeout=120)

            if response.status_code == 200:
                data = response.json()
                print(f"  Reason: {data['reason']}")
                if data["suggested_fix"]:
                    print(f"  Suggested Fixes ({len(data['suggested_fix'])}):")
                    for j, fix in enumerate(data["suggested_fix"], 1):
                        print(f"    Fix {j}:")
                        print(f"      Original:  {fix['original_text']}")
                        print(f"      Fixed:     {fix['fixed_text']}")
                        print(f"      Summary:   {fix['fix_summary']}")
                else:
                    print("  Suggested Fixes: None (paragraph complies)")
            else:
                print(f"  ERROR [{response.status_code}]: {response.text}")

        except requests.exceptions.ConnectionError:
            print("  ERROR: Cannot connect. Is the server running?")
        except requests.exceptions.Timeout:
            print("  ERROR: Request timed out (120s)")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n{'=' * 60}")
    print("TESTS COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_tests()
