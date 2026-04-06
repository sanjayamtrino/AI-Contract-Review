import json
from pathlib import Path
from typing import List

from src.schemas.playbook import PlaybookRule

PLAYBOOK_DIR = Path("src/data")

PLAYBOOK_FILES = {
    "v3": PLAYBOOK_DIR / "playbook_rules_v3.json",
    "default": PLAYBOOK_DIR / "default_playbook_rules.json",
}


def load_playbook_rules(playbook_name: str = "v3") -> List[PlaybookRule]:
    """Load playbook rules from JSON and normalize to PlaybookRule models.

    Args:
        playbook_name: "v3" or "default"

    Returns:
        List of PlaybookRule, ordered by rule sequence.

    Raises:
        FileNotFoundError: If the playbook file does not exist.
        ValueError: If the JSON is malformed or rules are invalid.
    """
    file_path = PLAYBOOK_FILES.get(playbook_name)
    if not file_path or not file_path.exists():
        raise FileNotFoundError(
            f"Playbook '{playbook_name}' not found. "
            f"Available: {list(PLAYBOOK_FILES.keys())}"
        )

    raw_rules = json.loads(file_path.read_text(encoding="utf-8"))

    if not isinstance(raw_rules, list) or len(raw_rules) == 0:
        raise ValueError(f"Playbook '{playbook_name}' contains no rules.")

    if playbook_name == "v3":
        return _normalize_v3(raw_rules)
    return _normalize_default(raw_rules)


def _normalize_v3(raw_rules: list) -> List[PlaybookRule]:
    """Normalize v3 format: {title, instruction, description}."""
    rules = []
    for i, r in enumerate(raw_rules):
        rules.append(
            PlaybookRule(
                title=r["title"],
                instruction=r["instruction"],
                description=r["description"],
                order=i + 1,
            )
        )
    return rules


def _normalize_default(raw_rules: list) -> List[PlaybookRule]:
    """Normalize default format: {name, category, standard_position, ...}."""
    rules = []
    for r in raw_rules:
        rules.append(
            PlaybookRule(
                title=r["name"],
                instruction=(
                    f"Evaluate whether the contract's {r['name'].lower()} provisions "
                    f"align with the company's standard position. Check specific terms, "
                    f"timeframes, scope, and conditions."
                ),
                description=r["standard_position"],
                category=r.get("category"),
                standard_position=r.get("standard_position"),
                fallback_position=r.get("fallback_position"),
                canned_response=r.get("canned_response"),
                order=r.get("order"),
            )
        )
    return rules
