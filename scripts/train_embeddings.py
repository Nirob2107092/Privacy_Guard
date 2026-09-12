"""Train and evaluate Stage 3 embedding representations."""

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

from privacyguard.embeddings.document_embedding import build_window_embeddings
from privacyguard.embeddings.pretrained_loader import load_pretrained_embeddings
from privacyguard.embeddings.word2vec_scratch import cosine_similarity, most_similar, train
from privacyguard.features.window_features import WindowFeatureBuilder
from privacyguard.models.bio import categories_to_bio
from privacyguard.models.logistic_regression import BinaryLogisticRegression
from privacyguard.models.naive_bayes import MulticlassNaiveBayes
from privacyguard.evaluation.metrics import evaluate_bio_sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "stage3_embedding_results.json")
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=50)
    parser.add_argument("--embedding-epochs", type=int, default=15)
    parser.add_argument("--embedding-learning-rate", type=float, default=0.03)
    parser.add_argument("--max-iter", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.15)
    parser.add_argument("--pretrained-model", default="glove-wiki-gigaword-100")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def base_category(tag: str) -> str:
    return "O" if tag == "O" else tag.split("-", 1)[-1]


def flatten_targets(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    categories = np.asarray(
        [base_category(tag) for row in rows for tag in row["tags"]], dtype=object
    )
    return (categories != "O").astype(np.int64), categories


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


def evaluation_breakdown(rows: list[dict[str, Any]], predicted: list[list[str]]) -> dict[str, Any]:
    gold = [row["tags"] for row in rows]
    result: dict[str, Any] = {"overall": evaluate_bio_sequences(predicted, gold), "by_group": {}}
    for group in sorted({row["group"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["group"] == group]
        result["by_group"][group] = evaluate_bio_sequences(
            [predicted[index] for index in indices], [gold[index] for index in indices]
        )
    return result


def pipeline_predictions(
    binary_predictions: np.ndarray,
    category_predictions: np.ndarray,
    rows: list[dict[str, Any]],
) -> list[list[str]]:
    categories = np.where(binary_predictions == 1, category_predictions, "O")
    return [categories_to_bio(sequence) for sequence in unflatten(categories, rows)]


def train_and_evaluate(
    name: str,
    train_features: np.ndarray,
    val_features: np.ndarray,
    test_features: np.ndarray,
    train_binary: np.ndarray,
    category_model: MulticlassNaiveBayes,
    val_category_features: Any,
    test_category_features: Any,
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    learning_rate: float,
    max_iter: int,
    feature_details: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    logistic = BinaryLogisticRegression(
        learning_rate=learning_rate, max_iter=max_iter
    ).fit(train_features, train_binary)
    predictions: dict[str, list[list[str]]] = {}
    for split_name, rows, features, category_features in (
        ("validation", val_rows, val_features, val_category_features),
        ("test", test_rows, test_features, test_category_features),
    ):
        predictions[split_name] = pipeline_predictions(
            logistic.predict(features), category_model.predict(category_features), rows
        )
    return {
        "features": feature_details,
        "models": {
            "logistic_regression": {
                "implementation": "from_scratch_numpy_binary",
                "iterations": logistic.n_iter_,
                "final_log_loss": logistic.loss_history_[-1],
                "parameters": int(len(logistic.weights_) + 1),
            },
            "naive_bayes": {
                "implementation": "stage2_from_scratch_numpy_multiclass_unchanged",
                "classes": [str(value) for value in category_model.classes_],
            },
        },
        "training_seconds": time.perf_counter() - started,
        "validation": evaluation_breakdown(val_rows, predictions["validation"]),
        "test": evaluation_breakdown(test_rows, predictions["test"]),
    }


def print_sanity(word2id: dict[str, int], matrix: np.ndarray) -> dict[str, Any]:
    related_pairs = (("call", "phone"), ("email", "contact"), ("account", "number"))
    unrelated_pairs = (("call", "address"), ("email", "location"), ("phone", "organization"))

    def score_pairs(pairs: tuple[tuple[str, str], ...]) -> list[dict[str, Any]]:
        scores = []
        for left, right in pairs:
            if left not in word2id or right not in word2id:
                print(f"  {left}/{right}: unavailable in vocabulary")
                continue
            score = cosine_similarity(matrix[word2id[left]], matrix[word2id[right]])
            print(f"  {left:>10} / {right:<12} {score: .4f}")
            scores.append({"left": left, "right": right, "cosine": score})
        return scores

    print("\nIntrinsic cosine sanity check (scratch embeddings)")
    print("Related pairs:")
    related = score_pairs(related_pairs)
    print("Unrelated pairs:")
    unrelated = score_pairs(unrelated_pairs)
    related_average = float(np.mean([item["cosine"] for item in related])) if related else 0.0
    unrelated_average = float(np.mean([item["cosine"] for item in unrelated])) if unrelated else 0.0
    directionally_sensible = related_average > unrelated_average
    print(
        f"Average related={related_average:.4f}, unrelated={unrelated_average:.4f}; "
        f"related higher: {directionally_sensible}"
    )
    return {
        "related": related,
        "unrelated": unrelated,
        "related_average": related_average,
        "unrelated_average": unrelated_average,
        "related_higher": directionally_sensible,
    }


def main() -> None:
    args = parse_args()
    train_rows = load_jsonl(args.data_dir / "train.jsonl")
    val_rows = load_jsonl(args.data_dir / "val.jsonl")
    test_rows = load_jsonl(args.data_dir / "test.jsonl")
    train_sentences = [row["tokens"] for row in train_rows]
    val_sentences = [row["tokens"] for row in val_rows]
    test_sentences = [row["tokens"] for row in test_rows]
    train_binary, train_categories = flatten_targets(train_rows)

    sparse_builder = WindowFeatureBuilder(window_size=args.window_size, use_char_ngrams=True).fit(train_sentences)
    train_counts = sparse_builder.transform(train_sentences, weighting="count")
    val_counts = sparse_builder.transform(val_sentences, weighting="count")
    test_counts = sparse_builder.transform(test_sentences, weighting="count")
    train_tfidf = sparse_builder.transform(train_sentences, weighting="tfidf")
    val_tfidf = sparse_builder.transform(val_sentences, weighting="tfidf")
    test_tfidf = sparse_builder.transform(test_sentences, weighting="tfidf")
    pii_train = train_binary == 1
    category_model = MulticlassNaiveBayes(alpha=1.0).fit(
        train_counts[pii_train], train_categories[pii_train]
    )

    configurations: dict[str, dict[str, Any]] = {}
    configurations["bow_tfidf"] = train_and_evaluate(
        "bow_tfidf", train_tfidf, val_tfidf, test_tfidf, train_binary, category_model,
        val_counts, test_counts, val_rows, test_rows, args.learning_rate, args.max_iter,
        {"window_size": args.window_size, "representation": "Stage2 word+character TF-IDF"},
    )

    word2id, scratch_matrix = train(
        train_sentences, args.embedding_dim, args.window_size,
        args.embedding_epochs, args.embedding_learning_rate
    )
    print(f"Embedding vector shape: {scratch_matrix.shape}")
    for sample_word in ("call", "email", "account"):
        if sample_word in word2id:
            print(f"Most similar to {sample_word}: {most_similar(sample_word, word2id, scratch_matrix, 3)}")
    sanity = print_sanity(word2id, scratch_matrix)
    mean_train, weighted_train = build_window_embeddings(
        train_sentences, word2id, scratch_matrix, args.window_size, sparse_builder
    )
    mean_val, weighted_val = build_window_embeddings(
        val_sentences, word2id, scratch_matrix, args.window_size, sparse_builder
    )
    mean_test, weighted_test = build_window_embeddings(
        test_sentences, word2id, scratch_matrix, args.window_size, sparse_builder
    )
    for name, train_features, val_features, test_features in (
        ("mean_word2vec", mean_train, mean_val, mean_test),
        ("tfidf_weighted_word2vec", weighted_train, weighted_val, weighted_test),
    ):
        configurations[name] = train_and_evaluate(
            name, train_features, val_features, test_features, train_binary, category_model,
            val_counts, test_counts, val_rows, test_rows, args.learning_rate, args.max_iter,
            {"window_size": args.window_size, "representation": name, "embedding_dim": args.embedding_dim},
        )

    pretrained = load_pretrained_embeddings(word2id, model_name=args.pretrained_model)
    if pretrained is not None:
        pretrained_word2id, pretrained_matrix = pretrained
        pretrained_train, pretrained_weighted_train = build_window_embeddings(
            train_sentences, pretrained_word2id, pretrained_matrix, args.window_size, sparse_builder
        )
        pretrained_val, pretrained_weighted_val = build_window_embeddings(
            val_sentences, pretrained_word2id, pretrained_matrix, args.window_size, sparse_builder
        )
        pretrained_test, pretrained_weighted_test = build_window_embeddings(
            test_sentences, pretrained_word2id, pretrained_matrix, args.window_size, sparse_builder
        )
        configurations["pretrained_tfidf_weighted"] = train_and_evaluate(
            "pretrained_tfidf_weighted", pretrained_weighted_train, pretrained_weighted_val,
            pretrained_weighted_test, train_binary, category_model, val_counts, test_counts,
            val_rows, test_rows, args.learning_rate, args.max_iter,
            {"window_size": args.window_size, "representation": "pretrained TF-IDF-weighted embedding",
             "model": args.pretrained_model, "embedding_dim": int(pretrained_matrix.shape[1])},
        )

    test_scores = {
        name: values["test"]["overall"]["token_level"]["macro_f1"]
        for name, values in configurations.items()
    }
    best_name = max(test_scores, key=test_scores.get)
    hypothesis = (
        "The best representation is expected to reflect the corpus mix: dense embeddings can "
        "share signal across contextual words, while sparse window and character features preserve "
        "the exact digits, punctuation, and spelling patterns that identify structured categories "
        "such as PHONE_NUMBER, EMAIL, and ACCOUNT_NUMBER. If the sparse arm wins, that is evidence "
        "that distributional semantics alone is insufficient for this structured PII task; if an "
        "embedding arm wins, its contextual smoothing likely helps with Banglish and noisy prompts."
    )
    print("\nStage 3 comparison (test set)")
    print(f"{'representation':<30} {'macro_f1':>10} {'weighted_f1':>12} {'span_f1':>10}")
    for name, values in configurations.items():
        metrics = values["test"]["overall"]
        print(f"{name:<30} {metrics['token_level']['macro_f1']:>10.4f} "
              f"{metrics['token_level']['weighted_f1']:>12.4f} {metrics['exact_span']['f1']:>10.4f}")
        for group, group_metrics in values["test"]["by_group"].items():
            print(f"  {name} / {group}: macro_f1={group_metrics['token_level']['macro_f1']:.4f}, "
                  f"weighted_f1={group_metrics['token_level']['weighted_f1']:.4f}, "
                  f"span_f1={group_metrics['exact_span']['f1']:.4f}")
    print(f"Best representation by test macro-F1: {best_name} ({test_scores[best_name]:.4f})")
    print(f"Hypothesis: {hypothesis}")

    output = {
        "stage": 3,
        "evaluation_backend": "built_in_exact_BIO_metrics",
        "dataset": {"train_sentences": len(train_rows), "validation_sentences": len(val_rows),
                     "test_sentences": len(test_rows), "train_tokens": int(len(train_binary)),
                     "validation_tokens": sum(len(row["tokens"]) for row in val_rows),
                     "test_tokens": sum(len(row["tokens"]) for row in test_rows)},
        "embedding_training": {"algorithm": "scratch_numpy_skipgram", "vocabulary_size": len(word2id),
                                "vector_shape": list(scratch_matrix.shape), "epochs": args.embedding_epochs,
                                "window_size": args.window_size, "learning_rate": args.embedding_learning_rate,
                                "intrinsic_sanity": sanity},
        "configurations": configurations,
        "pretrained": {"attempted": True, "model": args.pretrained_model, "available": pretrained is not None},
        "best_representation": {"name": best_name, "test_macro_f1": test_scores[best_name],
                                 "hypothesis": hypothesis},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(output, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")
    print(f"Saved detailed results to {args.output}")


if __name__ == "__main__":
    main()
