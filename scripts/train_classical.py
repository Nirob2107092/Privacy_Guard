"""Train and evaluate PrivacyGuard's Stage 2 classical model pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from privacyguard.evaluation.metrics import evaluate_bio_sequences
from privacyguard.features.window_features import WindowFeatureBuilder
from privacyguard.models.bio import categories_to_bio
from privacyguard.models.logistic_regression import BinaryLogisticRegression
from privacyguard.models.naive_bayes import MulticlassNaiveBayes


STRUCTURED_CATEGORIES = (
    "PHONE_NUMBER",
    "EMAIL",
    "ACCOUNT_NUMBER",
    "FINANCIAL_INFORMATION",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results" / "stage2_classical_results.json"
    )
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        rows = [json.loads(line) for line in input_file if line.strip()]
    for row in rows:
        if len(row["tokens"]) != len(row["tags"]):
            raise ValueError(f"mismatched tokens/tags in {path}: {row['text']!r}")
    return rows


def base_category(tag: str) -> str:
    return "O" if tag == "O" else tag.split("-", 1)[-1]


def flatten_targets(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    categories = np.asarray(
        [base_category(tag) for row in rows for tag in row["tags"]], dtype=object
    )
    binary = (categories != "O").astype(np.int64)
    return binary, categories


def unflatten(values: np.ndarray, rows: list[dict[str, Any]]) -> list[list[str]]:
    sequences: list[list[str]] = []
    cursor = 0
    for row in rows:
        length = len(row["tokens"])
        sequences.append([str(value) for value in values[cursor:cursor + length]])
        cursor += length
    if cursor != len(values):
        raise ValueError("flat predictions do not match dataset token count")
    return sequences


def pipeline_predictions(
    binary_predictions: np.ndarray,
    category_predictions: np.ndarray,
    rows: list[dict[str, Any]],
) -> list[list[str]]:
    categories = np.where(binary_predictions == 1, category_predictions, "O")
    return [categories_to_bio(sequence) for sequence in unflatten(categories, rows)]


def evaluation_breakdown(
    rows: list[dict[str, Any]], predicted: list[list[str]]
) -> dict[str, Any]:
    gold = [row["tags"] for row in rows]
    result: dict[str, Any] = {"overall": evaluate_bio_sequences(predicted, gold), "by_group": {}}
    for group in sorted({row["group"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["group"] == group]
        result["by_group"][group] = evaluate_bio_sequences(
            [predicted[index] for index in indices],
            [gold[index] for index in indices],
        )
    return result


def classification_scores(
    gold: np.ndarray,
    predicted: np.ndarray,
    *,
    positive_label: object | None = None,
) -> dict[str, float]:
    accuracy = float(np.mean(gold == predicted))
    if positive_label is not None:
        labels = [positive_label]
    else:
        labels = sorted(set(gold.tolist()) | set(predicted.tolist()))
    f1_values: list[float] = []
    for label in labels:
        true_positive = int(np.sum((gold == label) & (predicted == label)))
        false_positive = int(np.sum((gold != label) & (predicted == label)))
        false_negative = int(np.sum((gold == label) & (predicted != label)))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1_values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"accuracy": accuracy, "f1": float(np.mean(f1_values))}


def train_configuration(
    name: str,
    *,
    use_char_ngrams: bool,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    window_size: int,
    learning_rate: float,
    max_iter: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_sentences = [row["tokens"] for row in train_rows]
    val_sentences = [row["tokens"] for row in val_rows]
    test_sentences = [row["tokens"] for row in test_rows]
    train_binary, train_categories = flatten_targets(train_rows)

    started = time.perf_counter()
    builder = WindowFeatureBuilder(
        window_size=window_size,
        use_char_ngrams=use_char_ngrams,
    ).fit(train_sentences)
    train_counts = builder.transform(train_sentences, weighting="count")
    train_tfidf = builder.transform(train_sentences, weighting="tfidf")
    val_counts = builder.transform(val_sentences, weighting="count")
    val_tfidf = builder.transform(val_sentences, weighting="tfidf")
    test_counts = builder.transform(test_sentences, weighting="count")
    test_tfidf = builder.transform(test_sentences, weighting="tfidf")

    logistic = BinaryLogisticRegression(
        learning_rate=learning_rate,
        max_iter=max_iter,
    ).fit(train_tfidf, train_binary)
    pii_train_mask = train_binary == 1
    naive_bayes = MulticlassNaiveBayes(alpha=1.0).fit(
        train_counts[pii_train_mask], train_categories[pii_train_mask]
    )
    training_seconds = time.perf_counter() - started

    predictions: dict[str, list[list[str]]] = {}
    for split_name, rows, count_matrix, tfidf_matrix in (
        ("validation", val_rows, val_counts, val_tfidf),
        ("test", test_rows, test_counts, test_tfidf),
    ):
        binary_predictions = logistic.predict(tfidf_matrix)
        category_predictions = naive_bayes.predict(count_matrix)
        predictions[split_name] = pipeline_predictions(
            binary_predictions, category_predictions, rows
        )

    result = {
        "features": {
            "window_size": window_size,
            "weighting": {"logistic_regression": "tfidf", "naive_bayes": "count"},
            "character_ngrams": list(builder.char_ngram_range) if use_char_ngrams else None,
            "feature_count": len(builder.feature_names_),
        },
        "models": {
            "logistic_regression": {
                "implementation": "from_scratch_numpy_binary",
                "iterations": logistic.n_iter_,
                "final_log_loss": logistic.loss_history_[-1],
                "parameters": int(len(logistic.weights_) + 1),
            },
            "naive_bayes": {
                "implementation": "from_scratch_numpy_multiclass",
                "alpha": naive_bayes.alpha,
                "classes": [str(value) for value in naive_bayes.classes_],
                "parameters": int(np.prod(naive_bayes.feature_log_prob_.shape)),
            },
        },
        "training_seconds": training_seconds,
        "validation": evaluation_breakdown(val_rows, predictions["validation"]),
        "test": evaluation_breakdown(test_rows, predictions["test"]),
    }
    artifacts = {
        "builder": builder,
        "logistic": logistic,
        "naive_bayes": naive_bayes,
        "train_counts": train_counts,
        "train_tfidf": train_tfidf,
        "test_counts": test_counts,
        "test_tfidf": test_tfidf,
        "train_binary": train_binary,
        "train_categories": train_categories,
    }
    print(
        f"Trained {name}: {len(builder.feature_names_)} features, "
        f"{logistic.n_iter_} LR iterations, {training_seconds:.2f}s"
    )
    return result, artifacts


def sklearn_comparison(
    artifacts: dict[str, Any], test_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Sanity-check scratch models on exactly the same custom matrices."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB

    test_binary, test_categories = flatten_targets(test_rows)
    train_binary = artifacts["train_binary"]
    train_categories = artifacts["train_categories"]
    pii_train = train_binary == 1
    pii_test = test_binary == 1

    scratch_lr_prediction = artifacts["logistic"].predict(artifacts["test_tfidf"])
    sklearn_lr = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")
    sklearn_lr.fit(artifacts["train_tfidf"], train_binary)
    sklearn_lr_prediction = sklearn_lr.predict(artifacts["test_tfidf"])

    scratch_nb_prediction = artifacts["naive_bayes"].predict(
        artifacts["test_counts"][pii_test]
    )
    sklearn_nb = MultinomialNB(alpha=1.0)
    sklearn_nb.fit(artifacts["train_counts"][pii_train], train_categories[pii_train])
    sklearn_nb_prediction = sklearn_nb.predict(artifacts["test_counts"][pii_test])

    comparison = {
        "matrix_config": "word_plus_char_ngrams",
        "logistic_regression_binary": {
            "f1_average": "positive_class_PII",
            "scratch": classification_scores(
                test_binary, scratch_lr_prediction, positive_label=1
            ),
            "sklearn": classification_scores(
                test_binary, sklearn_lr_prediction, positive_label=1
            ),
        },
        "naive_bayes_category": {
            "f1_average": "macro_over_PII_categories",
            "scratch": classification_scores(
                test_categories[pii_test], scratch_nb_prediction
            ),
            "sklearn": classification_scores(
                test_categories[pii_test], sklearn_nb_prediction
            ),
        },
    }
    for values in comparison.values():
        if not isinstance(values, dict) or "scratch" not in values:
            continue
        values["absolute_difference"] = {
            metric: abs(values["scratch"][metric] - values["sklearn"][metric])
            for metric in ("accuracy", "f1")
        }
    return comparison


def structured_ablation(configurations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    word_scores = configurations["word_only"]["test"]["overall"]["per_category_f1"]
    char_scores = configurations["word_plus_char_ngrams"]["test"]["overall"]["per_category_f1"]
    per_category = {
        category: {
            "word_only_f1": word_scores.get(category, 0.0),
            "word_plus_char_f1": char_scores.get(category, 0.0),
            "delta": char_scores.get(category, 0.0) - word_scores.get(category, 0.0),
        }
        for category in STRUCTURED_CATEGORIES
    }
    word_macro = float(np.mean([values["word_only_f1"] for values in per_category.values()]))
    char_macro = float(np.mean([values["word_plus_char_f1"] for values in per_category.values()]))
    return {
        "categories": list(STRUCTURED_CATEGORIES),
        "per_category": per_category,
        "macro_f1": {
            "word_only": word_macro,
            "word_plus_char_ngrams": char_macro,
            "delta": char_macro - word_macro,
        },
    }


def print_group_results(configuration: dict[str, Any]) -> None:
    print("\nScratch pipeline test metrics (word + character n-grams)")
    print(f"{'subset':<12} {'macro_f1':>10} {'weighted_f1':>12} {'span_f1':>10}")
    sections = {"overall": configuration["test"]["overall"]}
    sections.update(configuration["test"]["by_group"])
    for name, metrics in sections.items():
        token = metrics["token_level"]
        print(
            f"{name:<12} {token['macro_f1']:>10.4f} "
            f"{token['weighted_f1']:>12.4f} {metrics['exact_span']['f1']:>10.4f}"
        )
    print("Per-category F1 by subset")
    for name, metrics in sections.items():
        scores = ", ".join(
            f"{label}={score:.4f}"
            for label, score in metrics["per_category_f1"].items()
        )
        print(f"  {name}: {scores}")


def print_sklearn_comparison(comparison: dict[str, Any]) -> None:
    print("\nFrom-scratch vs sklearn (same word+char feature matrices)")
    print(f"{'model':<30} {'implementation':<12} {'accuracy':>10} {'f1':>10}")
    for model_name in ("logistic_regression_binary", "naive_bayes_category"):
        for implementation in ("scratch", "sklearn"):
            scores = comparison[model_name][implementation]
            print(
                f"{model_name:<30} {implementation:<12} "
                f"{scores['accuracy']:>10.4f} {scores['f1']:>10.4f}"
            )


def print_ablation(ablation: dict[str, Any]) -> None:
    print("\nStructured-category ablation (scratch pipeline test F1)")
    print(f"{'category':<28} {'word':>10} {'word+char':>12} {'delta':>10}")
    for category, scores in ablation["per_category"].items():
        print(
            f"{category:<28} {scores['word_only_f1']:>10.4f} "
            f"{scores['word_plus_char_f1']:>12.4f} {scores['delta']:>+10.4f}"
        )
    macro = ablation["macro_f1"]
    print(
        f"{'STRUCTURED MACRO':<28} {macro['word_only']:>10.4f} "
        f"{macro['word_plus_char_ngrams']:>12.4f} {macro['delta']:>+10.4f}"
    )


def main() -> None:
    args = parse_args()
    train_rows = load_jsonl(args.data_dir / "train.jsonl")
    val_rows = load_jsonl(args.data_dir / "val.jsonl")
    test_rows = load_jsonl(args.data_dir / "test.jsonl")

    configurations: dict[str, dict[str, Any]] = {}
    runtime_artifacts: dict[str, dict[str, Any]] = {}
    for name, use_char_ngrams in (
        ("word_only", False),
        ("word_plus_char_ngrams", True),
    ):
        result, artifacts = train_configuration(
            name,
            use_char_ngrams=use_char_ngrams,
            train_rows=train_rows,
            val_rows=val_rows,
            test_rows=test_rows,
            window_size=args.window_size,
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
        )
        configurations[name] = result
        runtime_artifacts[name] = artifacts

    comparison = sklearn_comparison(runtime_artifacts["word_plus_char_ngrams"], test_rows)
    ablation = structured_ablation(configurations)
    output = {
        "stage": 2,
        "evaluation_backend": "built_in_exact_BIO_metrics",
        "dataset": {
            "train_sentences": len(train_rows),
            "validation_sentences": len(val_rows),
            "test_sentences": len(test_rows),
            "train_tokens": sum(len(row["tokens"]) for row in train_rows),
            "validation_tokens": sum(len(row["tokens"]) for row in val_rows),
            "test_tokens": sum(len(row["tokens"]) for row in test_rows),
        },
        "configurations": configurations,
        "sklearn_validation": comparison,
        "structured_category_ablation": ablation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(output, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")

    print_group_results(configurations["word_plus_char_ngrams"])
    print_sklearn_comparison(comparison)
    print_ablation(ablation)
    print(f"\nSaved detailed results to {args.output}")


if __name__ == "__main__":
    main()
