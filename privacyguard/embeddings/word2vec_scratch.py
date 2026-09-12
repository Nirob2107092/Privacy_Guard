"""A small Skip-gram Word2Vec implementation written with NumPy."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    probabilities = np.exp(shifted)
    return probabilities / np.sum(probabilities)


def train(
    tokenized_sentences: Sequence[Sequence[str]],
    embedding_dim: int,
    window_size: int,
    epochs: int,
    lr: float,
) -> tuple[dict[str, int], np.ndarray]:
    """Train Skip-gram embeddings and return the vocabulary and hidden matrix."""

    if embedding_dim < 1 or window_size < 1 or epochs < 1 or lr <= 0:
        raise ValueError("embedding_dim, window_size, epochs, and lr must be positive")
    vocabulary = sorted({token.lower() for sentence in tokenized_sentences for token in sentence})
    if not vocabulary:
        raise ValueError("tokenized_sentences must contain at least one token")
    word2id = {word: index for index, word in enumerate(vocabulary)}
    pairs = [
        (word2id[token.lower()], word2id[context.lower()])
        for sentence in tokenized_sentences
        for index, token in enumerate(sentence)
        for context_index in range(max(0, index - window_size), min(len(sentence), index + window_size + 1))
        if context_index != index
        for context in (sentence[context_index],)
    ]
    if not pairs:
        raise ValueError("the corpus does not contain any context pairs")

    rng = np.random.default_rng(42)
    scale = 0.5 / max(embedding_dim, 1)
    W1 = rng.uniform(-scale, scale, size=(len(vocabulary), embedding_dim))
    W2 = rng.uniform(-scale, scale, size=(embedding_dim, len(vocabulary)))
    one_hot = np.zeros(len(vocabulary), dtype=np.float64)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for input_id, target_id in pairs:
            one_hot.fill(0.0)
            one_hot[input_id] = 1.0
            hidden = one_hot @ W1
            probabilities = _softmax(hidden @ W2)
            total_loss -= float(np.log(max(probabilities[target_id], 1e-12)))

            output_error = probabilities.copy()
            output_error[target_id] -= 1.0
            hidden_error = W2 @ output_error
            W2 -= lr * np.outer(hidden, output_error)
            W1 -= lr * np.outer(one_hot, hidden_error)
        loss = total_loss / len(pairs)
        print(f"Skip-gram epoch {epoch:02d}/{epochs}: loss={loss:.6f}")

    return word2id, W1


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Return cosine similarity, treating a zero vector as having similarity zero."""

    denominator = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    return float(vec_a @ vec_b / denominator) if denominator else 0.0


def most_similar(
    word: str, word2id: dict[str, int], W1: np.ndarray, topn: int = 5
) -> list[tuple[str, float]]:
    """Return the nearest vocabulary words by cosine similarity."""

    normalized = word.lower()
    if normalized not in word2id:
        raise KeyError(f"unknown word: {word}")
    query = W1[word2id[normalized]]
    scores = [
        (candidate, cosine_similarity(query, W1[index]))
        for candidate, index in word2id.items()
        if candidate != normalized
    ]
    return sorted(scores, key=lambda item: item[1], reverse=True)[:topn]
