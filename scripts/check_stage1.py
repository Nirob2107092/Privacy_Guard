"""Run the reproducible Stage 1 acceptance checks."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from privacyguard.dataset.generator import validate_bio_sequence
from privacyguard.preprocessing.cleaning import clean_text
from privacyguard.preprocessing.spelling import correct_spelling


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def main() -> None:
    data_dir = PROJECT_ROOT / "data"
    splits = {
        name: load_jsonl(data_dir / f"{name}.jsonl")
        for name in ("train", "val", "test")
    }
    total = sum(len(rows) for rows in splits.values())
    assert total >= 1000, f"expected at least 1000 examples, found {total}"

    text_sets = {name: {row["text"] for row in rows} for name, rows in splits.items()}
    assert len(text_sets["train"] & text_sets["val"]) == 0
    assert len(text_sets["train"] & text_sets["test"]) == 0
    assert len(text_sets["val"] & text_sets["test"]) == 0
    assert sum(map(len, text_sets.values())) == total

    all_rows = [row for rows in splits.values() for row in rows]
    for row in all_rows:
        validate_bio_sequence(row["tokens"], row["tags"])

    dirty = "<p>Call me 😊 at +880-1712-345678 or Test.User@example.com</p>"
    cleaned = clean_text(dirty)
    assert "<p>" not in cleaned and "😊" not in cleaned
    assert "+880-1712-345678" in cleaned
    assert "Test.User@example.com" in cleaned

    spelling_input = ["nmbr", "01712345678", "fone"]
    spelling_output = correct_spelling(spelling_input)
    assert spelling_output == ["number", "01712345678", "phone"]

    print(f"PASS: {total} unique examples; groups={dict(Counter(row['group'] for row in all_rows))}")
    print("PASS: no exact text overlap across train/val/test")
    print(f"PASS: cleaning -> {cleaned!r}")
    print(f"PASS: spelling {spelling_input} -> {spelling_output}")
    print("\nFive diverse token/tag samples:")
    # Intentionally include multi-token names, formatted phone numbers, email
    # tokenization, addresses, and organizations across all three groups.
    sample_specs = (
        ("clean", "I-PERSON"),
        ("banglish", "I-PHONE_NUMBER"),
        ("noisy", "I-ADDRESS"),
        ("clean", "I-EMAIL"),
        ("noisy", "I-ORGANIZATION"),
    )
    selected: list[dict[str, object]] = []
    for group, required_tag in sample_specs:
        match = next(
            (
                row
                for row in all_rows
                if row["group"] == group and required_tag in row["tags"]
            ),
            None,
        )
        assert match is not None, f"missing sample for {group}/{required_tag}"
        selected.append(match)
    for index, row in enumerate(selected, start=1):
        pairs = list(zip(row["tokens"], row["tags"]))
        print(f"{index}. [{row['group']}] {row['text']}")
        print(f"   {pairs}")


if __name__ == "__main__":
    main()
