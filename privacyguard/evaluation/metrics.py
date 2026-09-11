"""Reusable token, category, confusion-matrix, and exact-span BIO metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _base_category(tag: str) -> str:
    if tag == "O":
        return "O"
    if "-" not in tag:
        return tag
    return tag.split("-", 1)[1]


def _validate_inputs(
    predicted: Sequence[Sequence[str]], gold: Sequence[Sequence[str]]
) -> None:
    if len(predicted) != len(gold):
        raise ValueError("predicted and gold must contain the same number of sequences")
    for index, (predicted_sequence, gold_sequence) in enumerate(zip(predicted, gold)):
        if len(predicted_sequence) != len(gold_sequence):
            raise ValueError(f"sequence {index} has different predicted/gold lengths")


def _per_label_stats(
    predicted: Sequence[str], gold: Sequence[str], labels: Sequence[str]
) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    for label in labels:
        true_positive = sum(p == label and g == label for p, g in zip(predicted, gold))
        false_positive = sum(p == label and g != label for p, g in zip(predicted, gold))
        false_negative = sum(p != label and g == label for p, g in zip(predicted, gold))
        support = sum(g == label for g in gold)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        stats[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return stats


def _averages(
    predicted: Sequence[str], gold: Sequence[str], labels: Sequence[str]
) -> dict[str, float]:
    stats = _per_label_stats(predicted, gold, labels)
    total_support = sum(int(item["support"]) for item in stats.values())
    if not labels or total_support == 0:
        return {"precision": 0.0, "recall": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "accuracy": 0.0}
    return {
        "precision": sum(float(item["precision"]) for item in stats.values()) / len(labels),
        "recall": sum(float(item["recall"]) for item in stats.values()) / len(labels),
        "macro_f1": sum(float(item["f1"]) for item in stats.values()) / len(labels),
        "weighted_f1": sum(float(item["f1"]) * int(item["support"]) for item in stats.values()) / total_support,
        "accuracy": sum(p == g for p, g in zip(predicted, gold)) / len(gold) if gold else 0.0,
    }


def _extract_spans(sequence: Sequence[str]) -> set[tuple[int, int, str]]:
    """Extract end-exclusive spans, treating malformed I-tags as new spans."""

    spans: set[tuple[int, int, str]] = set()
    active_start: int | None = None
    active_label: str | None = None

    for index, tag in enumerate(list(sequence) + ["O"]):
        if tag == "O":
            prefix, label = "O", None
        elif "-" in tag:
            prefix, label = tag.split("-", 1)
        else:
            prefix, label = "B", tag

        continues = prefix == "I" and label == active_label
        if active_label is not None and not continues:
            spans.add((active_start if active_start is not None else index, index, active_label))
            active_start, active_label = None, None
        if label is not None and not continues:
            active_start, active_label = index, label
    return spans


def evaluate_bio_sequences(
    predicted: Sequence[Sequence[str]],
    gold: Sequence[Sequence[str]],
) -> dict[str, Any]:
    """Evaluate arbitrary BIO tag sequences with no project-specific labels.

    Token-level averages operate over the observed BIO tags, including ``O``.
    Per-category and confusion-matrix values collapse ``B-``/``I-`` prefixes.
    Exact-span metrics require matching sentence, start, end, and category.
    """

    _validate_inputs(predicted, gold)
    flat_predicted = [tag for sequence in predicted for tag in sequence]
    flat_gold = [tag for sequence in gold for tag in sequence]
    bio_labels = sorted(set(flat_predicted) | set(flat_gold), key=lambda value: (value == "O", value))
    token_level = _averages(flat_predicted, flat_gold, bio_labels)

    base_predicted = [_base_category(tag) for tag in flat_predicted]
    base_gold = [_base_category(tag) for tag in flat_gold]
    category_labels = sorted((set(base_predicted) | set(base_gold)) - {"O"})
    category_stats = _per_label_stats(base_predicted, base_gold, category_labels)
    per_category_f1 = {
        label: float(category_stats[label]["f1"]) for label in category_labels
    }

    predicted_spans = {
        (sentence_index, start, end, label)
        for sentence_index, sequence in enumerate(predicted)
        for start, end, label in _extract_spans(sequence)
    }
    gold_spans = {
        (sentence_index, start, end, label)
        for sentence_index, sequence in enumerate(gold)
        for start, end, label in _extract_spans(sequence)
    }
    correct_spans = len(predicted_spans & gold_spans)
    span_precision = correct_spans / len(predicted_spans) if predicted_spans else 0.0
    span_recall = correct_spans / len(gold_spans) if gold_spans else 0.0
    span_f1 = (
        2 * span_precision * span_recall / (span_precision + span_recall)
        if span_precision + span_recall
        else 0.0
    )

    confusion_labels = ["O", *category_labels]
    label_to_index = {label: index for index, label in enumerate(confusion_labels)}
    matrix = [[0 for _ in confusion_labels] for _ in confusion_labels]
    for gold_label, predicted_label in zip(base_gold, base_predicted):
        matrix[label_to_index[gold_label]][label_to_index[predicted_label]] += 1

    return {
        "token_level": token_level,
        "per_category_f1": per_category_f1,
        "exact_span": {
            "precision": span_precision,
            "recall": span_recall,
            "f1": span_f1,
            "correct": correct_spans,
            "predicted": len(predicted_spans),
            "gold": len(gold_spans),
        },
        "confusion_matrix": {"labels": confusion_labels, "matrix": matrix},
    }
