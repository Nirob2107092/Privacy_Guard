"""Generate, deduplicate, split, and save the synthetic PrivacyGuard corpus."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from privacyguard.dataset.generator import generate_examples, split_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-examples", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data")
    return parser.parse_args()


def write_jsonl(path: Path, examples: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for example in examples:
            output_file.write(json.dumps(example, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.num_examples < 1000:
        raise SystemExit("--num-examples must be at least 1000 for the Stage 1 corpus")

    examples = generate_examples(args.num_examples, seed=args.seed)
    splits = split_examples(examples, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_rows in splits.items():
        write_jsonl(args.output_dir / f"{split_name}.jsonl", split_rows)

    print(f"Generated {len(examples)} unique examples in {args.output_dir}")
    for split_name, split_rows in splits.items():
        groups = Counter(row["group"] for row in split_rows)
        print(f"  {split_name}: {len(split_rows)} {dict(sorted(groups.items()))}")


if __name__ == "__main__":
    main()
