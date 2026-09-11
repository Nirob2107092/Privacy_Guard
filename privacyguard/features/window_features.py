"""Custom context-window BoW, TF-IDF, and character n-gram features."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from scipy.sparse import csr_matrix


ORTHOGRAPHIC_FEATURES = (
    "orth=is_capitalized",
    "orth=has_digit",
    "orth=has_at_symbol",
    "orth=has_dot",
    "orth=token_length",
    "orth=is_all_digits",
)


class WindowFeatureBuilder:
    """Build sparse per-token features without sklearn vectorizers.

    Word features include both unordered window counts and offset-specific
    indicators. Character n-grams are extracted from the target token only.
    IDF values are learned over token instances in the training sentences.
    """

    def __init__(
        self,
        *,
        window_size: int = 2,
        use_char_ngrams: bool = False,
        char_ngram_range: tuple[int, int] = (3, 4),
        min_frequency: int = 2,
        max_word_features: int = 2500,
        max_char_features: int = 2500,
    ) -> None:
        if window_size < 0:
            raise ValueError("window_size cannot be negative")
        if char_ngram_range[0] < 1 or char_ngram_range[0] > char_ngram_range[1]:
            raise ValueError("invalid character n-gram range")
        self.window_size = window_size
        self.use_char_ngrams = use_char_ngrams
        self.char_ngram_range = char_ngram_range
        self.min_frequency = min_frequency
        self.max_word_features = max_word_features
        self.max_char_features = max_char_features
        self.vocabulary_: dict[str, int] = {}
        self.feature_names_: list[str] = []
        self.idf_: np.ndarray | None = None
        self.n_instances_: int = 0

    @staticmethod
    def _normalize_token(token: str) -> str:
        return token.lower()

    def _lexical_counts(self, tokens: Sequence[str], index: int) -> Counter[str]:
        counts: Counter[str] = Counter()
        left = max(0, index - self.window_size)
        right = min(len(tokens), index + self.window_size + 1)
        for context_index in range(left, right):
            normalized = self._normalize_token(tokens[context_index])
            offset = context_index - index
            counts[f"word={normalized}"] += 1
            counts[f"offset={offset:+d}:word={normalized}"] += 1

        if self.use_char_ngrams:
            target = f"^{self._normalize_token(tokens[index])}$"
            minimum, maximum = self.char_ngram_range
            for width in range(minimum, maximum + 1):
                for start in range(max(0, len(target) - width + 1)):
                    counts[f"char{width}={target[start:start + width]}"] += 1
        return counts

    @staticmethod
    def _orthographic_values(token: str) -> dict[str, float]:
        return {
            "orth=is_capitalized": float(bool(token) and token[0].isupper()),
            "orth=has_digit": float(any(char.isdigit() for char in token)),
            "orth=has_at_symbol": float("@" in token),
            "orth=has_dot": float("." in token),
            # Normalization bounds this continuous feature for stable gradient
            # descent while preserving its ordering and relative magnitude.
            "orth=token_length": min(len(token), 50) / 50.0,
            "orth=is_all_digits": float(token.isdigit()),
        }

    def fit(self, sentences: Sequence[Sequence[str]]) -> "WindowFeatureBuilder":
        """Learn a deterministic vocabulary and smoothed IDF vector."""

        document_frequency: Counter[str] = Counter()
        n_instances = 0
        for tokens in sentences:
            for index in range(len(tokens)):
                document_frequency.update(self._lexical_counts(tokens, index).keys())
                n_instances += 1

        word_features = [
            (feature, frequency)
            for feature, frequency in document_frequency.items()
            if not feature.startswith("char") and frequency >= self.min_frequency
        ]
        char_features = [
            (feature, frequency)
            for feature, frequency in document_frequency.items()
            if feature.startswith("char") and frequency >= self.min_frequency
        ]
        ranking = lambda item: (-item[1], item[0])
        word_features.sort(key=ranking)
        char_features.sort(key=ranking)
        selected = word_features[: self.max_word_features]
        if self.use_char_ngrams:
            selected += char_features[: self.max_char_features]

        self.feature_names_ = [feature for feature, _ in selected]
        self.feature_names_.extend(ORTHOGRAPHIC_FEATURES)
        self.vocabulary_ = {
            feature: index for index, feature in enumerate(self.feature_names_)
        }
        self.n_instances_ = n_instances

        idf = np.ones(len(self.feature_names_), dtype=np.float64)
        for feature, index in self.vocabulary_.items():
            if not feature.startswith("orth="):
                frequency = document_frequency.get(feature, 0)
                idf[index] = np.log((1 + n_instances) / (1 + frequency)) + 1.0
        self.idf_ = idf
        return self

    def transform(
        self,
        sentences: Sequence[Sequence[str]],
        *,
        weighting: str = "count",
    ) -> csr_matrix:
        """Transform sentences to a token-row sparse matrix.

        ``weighting`` is either ``count`` for Naive Bayes or ``tfidf`` for
        Logistic Regression. Orthographic values are never IDF-weighted.
        """

        if not self.vocabulary_ or self.idf_ is None:
            raise RuntimeError("fit must be called before transform")
        if weighting not in {"count", "tfidf"}:
            raise ValueError("weighting must be 'count' or 'tfidf'")

        row_indices: list[int] = []
        column_indices: list[int] = []
        values: list[float] = []
        row = 0
        for tokens in sentences:
            for token_index, token in enumerate(tokens):
                feature_values: dict[str, float] = dict(
                    self._lexical_counts(tokens, token_index)
                )
                feature_values.update(self._orthographic_values(token))
                for feature, value in feature_values.items():
                    column = self.vocabulary_.get(feature)
                    if column is None or value == 0:
                        continue
                    if weighting == "tfidf" and not feature.startswith("orth="):
                        value *= float(self.idf_[column])
                    row_indices.append(row)
                    column_indices.append(column)
                    values.append(value)
                row += 1

        return csr_matrix(
            (np.asarray(values, dtype=np.float64), (row_indices, column_indices)),
            shape=(row, len(self.feature_names_)),
            dtype=np.float64,
        )

    def fit_transform(
        self,
        sentences: Sequence[Sequence[str]],
        *,
        weighting: str = "count",
    ) -> csr_matrix:
        return self.fit(sentences).transform(sentences, weighting=weighting)
