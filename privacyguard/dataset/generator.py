"""Template filling, character-span tracking, BIO alignment, and splitting."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import TypedDict

from nltk.tokenize import TreebankWordTokenizer

from .pii_values import PIIValueGenerator
from .templates import TEMPLATES, VALID_GROUPS


_PLACEHOLDER_RE = re.compile(r"\{([A-Z_]+)\}")
_SPAN_TOKENIZER = TreebankWordTokenizer()


class Example(TypedDict):
    text: str
    tokens: list[str]
    tags: list[str]
    group: str


@dataclass(frozen=True)
class EntitySpan:
    start: int
    end: int
    label: str
    value: str


def fill_template(
    template: str,
    value_generator: PIIValueGenerator,
    *,
    noisy_format: bool = False,
) -> tuple[str, list[EntitySpan]]:
    """Fill placeholders while recording their exact final character spans."""

    chunks: list[str] = []
    spans: list[EntitySpan] = []
    template_cursor = 0
    output_length = 0

    for match in _PLACEHOLDER_RE.finditer(template):
        literal = template[template_cursor:match.start()]
        chunks.append(literal)
        output_length += len(literal)

        label = match.group(1)
        value = value_generator.generate(label, noisy_format=noisy_format)
        start = output_length
        chunks.append(value)
        output_length += len(value)
        spans.append(EntitySpan(start=start, end=output_length, label=label, value=value))
        template_cursor = match.end()

    tail = template[template_cursor:]
    chunks.append(tail)
    return "".join(chunks), spans


def align_bio_tags(text: str, entity_spans: list[EntitySpan]) -> tuple[list[str], list[str]]:
    """Tokenize with offsets and label token/entity overlap using BIO tags."""

    token_spans = list(_SPAN_TOKENIZER.span_tokenize(text))
    tokens = [text[start:end] for start, end in token_spans]
    tags = ["O"] * len(tokens)

    for entity in entity_spans:
        overlapping = [
            index
            for index, (token_start, token_end) in enumerate(token_spans)
            if token_start < entity.end and token_end > entity.start
        ]
        if not overlapping:
            raise ValueError(f"entity did not overlap a token: {entity}")
        for position, token_index in enumerate(overlapping):
            if tags[token_index] != "O":
                raise ValueError("overlapping entity spans are not supported")
            prefix = "B" if position == 0 else "I"
            tags[token_index] = f"{prefix}-{entity.label}"

    return tokens, tags


def generate_group_example(group: str, rng: random.Random) -> Example:
    """Generate one example for a requested template group."""

    if group not in TEMPLATES:
        raise ValueError(f"group must be one of {VALID_GROUPS}, got {group!r}")
    template = rng.choice(TEMPLATES[group])
    value_generator = PIIValueGenerator(rng)
    # Noisy templates strongly favor formatted values. Other groups include a
    # smaller dose so robustness is not confined to only one split subgroup.
    probability = {"clean": 0.12, "banglish": 0.38, "noisy": 0.78}[group]
    text, spans = fill_template(
        template,
        value_generator,
        noisy_format=rng.random() < probability,
    )
    tokens, tags = align_bio_tags(text, spans)
    return {"text": text, "tokens": tokens, "tags": tags, "group": group}


def generate_examples(total: int = 1200, *, seed: int = 42) -> list[Example]:
    """Generate exactly ``total`` unique examples, balanced across groups."""

    if total < 1:
        raise ValueError("total must be positive")

    rng = random.Random(seed)
    base, remainder = divmod(total, len(VALID_GROUPS))
    targets = {
        group: base + (1 if index < remainder else 0)
        for index, group in enumerate(VALID_GROUPS)
    }
    examples: list[Example] = []
    seen_texts: set[str] = set()

    for group in VALID_GROUPS:
        generated = 0
        attempts = 0
        max_attempts = max(10_000, targets[group] * 100)
        while generated < targets[group] and attempts < max_attempts:
            attempts += 1
            example = generate_group_example(group, rng)
            if example["text"] in seen_texts:
                continue
            seen_texts.add(example["text"])
            examples.append(example)
            generated += 1
        if generated != targets[group]:
            raise RuntimeError(
                f"could only generate {generated}/{targets[group]} unique {group} examples"
            )

    rng.shuffle(examples)
    return examples


def split_examples(
    examples: list[Example],
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, list[Example]]:
    """Deduplicate and make deterministic, group-stratified data splits."""

    if not (0 < train_ratio < 1 and 0 <= val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("ratios must be valid and leave a non-empty test proportion")

    unique_by_text: dict[str, Example] = {}
    for example in examples:
        unique_by_text.setdefault(example["text"], example)

    grouped = {group: [] for group in VALID_GROUPS}
    for example in unique_by_text.values():
        grouped[example["group"]].append(example)

    rng = random.Random(seed)
    splits: dict[str, list[Example]] = {"train": [], "val": [], "test": []}
    for group_examples in grouped.values():
        rng.shuffle(group_examples)
        count = len(group_examples)
        train_end = round(count * train_ratio)
        val_end = train_end + round(count * val_ratio)
        splits["train"].extend(group_examples[:train_end])
        splits["val"].extend(group_examples[train_end:val_end])
        splits["test"].extend(group_examples[val_end:])

    for split_examples_list in splits.values():
        rng.shuffle(split_examples_list)
    return splits


def validate_bio_sequence(tokens: list[str], tags: list[str]) -> None:
    """Raise ``ValueError`` when a token/tag sequence violates BIO structure."""

    if len(tokens) != len(tags):
        raise ValueError("tokens and tags have different lengths")
    previous_label: str | None = None
    for tag in tags:
        if tag == "O":
            previous_label = None
            continue
        if "-" not in tag:
            raise ValueError(f"malformed tag: {tag}")
        prefix, label = tag.split("-", 1)
        if prefix not in {"B", "I"}:
            raise ValueError(f"malformed BIO prefix: {tag}")
        if prefix == "I" and previous_label != label:
            raise ValueError(f"orphaned inside tag: {tag}")
        previous_label = label
