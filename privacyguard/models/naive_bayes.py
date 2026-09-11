"""Multiclass multinomial Naive Bayes implemented from scratch with NumPy."""

from __future__ import annotations

from typing import Any

import numpy as np


class MulticlassNaiveBayes:
    """Multinomial Naive Bayes with Laplace (add-alpha) smoothing."""

    def __init__(self, *, alpha: float = 1.0) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        self.classes_: np.ndarray | None = None
        self.class_log_prior_: np.ndarray | None = None
        self.feature_log_prob_: np.ndarray | None = None

    def fit(self, features: Any, labels: np.ndarray) -> "MulticlassNaiveBayes":
        labels = np.asarray(labels)
        if labels.ndim != 1 or features.shape[0] != labels.shape[0]:
            raise ValueError("features and labels have incompatible shapes")
        stored_values = (
            np.asarray(features.data)
            if hasattr(features, "data")
            else np.asarray(features)
        )
        if np.any(stored_values < 0):
            raise ValueError("multinomial Naive Bayes requires non-negative features")

        classes, encoded = np.unique(labels, return_inverse=True)
        class_count = np.bincount(encoded, minlength=len(classes)).astype(np.float64)
        feature_count = np.zeros((len(classes), features.shape[1]), dtype=np.float64)
        for class_index in range(len(classes)):
            feature_count[class_index] = np.asarray(
                features[encoded == class_index].sum(axis=0)
            ).ravel()

        smoothed = feature_count + self.alpha
        self.classes_ = classes
        self.class_log_prior_ = np.log(class_count / class_count.sum())
        self.feature_log_prob_ = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))
        return self

    def predict_log_proba(self, features: Any) -> np.ndarray:
        if (
            self.classes_ is None
            or self.class_log_prior_ is None
            or self.feature_log_prob_ is None
        ):
            raise RuntimeError("model has not been fitted")
        joint = features @ self.feature_log_prob_.T
        return np.asarray(joint) + self.class_log_prior_

    def predict(self, features: Any) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("model has not been fitted")
        indices = np.argmax(self.predict_log_proba(features), axis=1)
        return self.classes_[indices]
