"""Mean and TF-IDF-weighted context-window embeddings."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from privacyguard.features.window_features import WindowFeatureBuilder


def _window(tokens: Sequence[str], index: int, window_size: int) -> Sequence[str]:
    left = max(0, index - window_size)
    right = min(len(tokens), index + window_size + 1)
    return tokens[left:right]


def _vectors(tokens: Sequence[str], word2id: dict[str, int], matrix: np.ndarray) -> list[np.ndarray]:
    return [matrix[word2id[token.lower()]] for token in tokens if token.lower() in word2id]


def mean_window_embedding(
    tokens: Sequence[str], index: int, word2id: dict[str, int], matrix: np.ndarray, window_size: int
) -> np.ndarray:
    vectors = _vectors(_window(tokens, index, window_size), word2id, matrix)
    return np.mean(vectors, axis=0) if vectors else np.zeros(matrix.shape[1], dtype=np.float64)


def tfidf_window_embedding(
    tokens: Sequence[str],
    index: int,
    word2id: dict[str, int],
    matrix: np.ndarray,
    window_size: int,
    feature_builder: WindowFeatureBuilder,
) -> np.ndarray:
    window_tokens = _window(tokens, index, window_size)
    weighted: list[np.ndarray] = []
    weights: list[float] = []
    for token in window_tokens:
        normalized = token.lower()
        embedding_id = word2id.get(normalized)
        feature_id = feature_builder.vocabulary_.get(f"word={normalized}")
        if embedding_id is None:
            continue
        weight = float(feature_builder.idf_[feature_id]) if feature_id is not None else 1.0
        weighted.append(matrix[embedding_id])
        weights.append(weight)
    if not weighted:
        return np.zeros(matrix.shape[1], dtype=np.float64)
    return np.average(np.asarray(weighted), axis=0, weights=np.asarray(weights))


def build_window_embeddings(
    sentences: Sequence[Sequence[str]],
    word2id: dict[str, int],
    matrix: np.ndarray,
    window_size: int,
    feature_builder: WindowFeatureBuilder,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one mean and one TF-IDF-weighted vector for every token row."""

    mean_rows: list[np.ndarray] = []
    weighted_rows: list[np.ndarray] = []
    for sentence in sentences:
        for index in range(len(sentence)):
            mean_rows.append(mean_window_embedding(sentence, index, word2id, matrix, window_size))
            weighted_rows.append(
                tfidf_window_embedding(sentence, index, word2id, matrix, window_size, feature_builder)
            )
    return np.asarray(mean_rows), np.asarray(weighted_rows)
