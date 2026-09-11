"""Binary logistic regression implemented from scratch with NumPy."""

from __future__ import annotations

from typing import Any

import numpy as np


class BinaryLogisticRegression:
    """Full-batch gradient-descent binary logistic regression.

    The input can be a NumPy array or any matrix exposing ``@`` and ``T``
    operations, including SciPy CSR matrices. Model state and optimization are
    implemented with NumPy; sklearn is not used here.
    """

    def __init__(
        self,
        *,
        learning_rate: float = 0.15,
        max_iter: int = 600,
        l2: float = 0.0,
        tolerance: float = 1e-7,
    ) -> None:
        if learning_rate <= 0 or max_iter < 1 or l2 < 0:
            raise ValueError("invalid optimization hyperparameters")
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l2 = l2
        self.tolerance = tolerance
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.loss_history_: list[float] = []
        self.n_iter_: int = 0

    @staticmethod
    def sigmoid(values: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid activation."""

        clipped = np.clip(values, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def fit(self, features: Any, labels: np.ndarray) -> "BinaryLogisticRegression":
        labels = np.asarray(labels, dtype=np.float64)
        if labels.ndim != 1 or features.shape[0] != labels.shape[0]:
            raise ValueError("features and labels have incompatible shapes")
        if not np.all(np.isin(labels, (0.0, 1.0))):
            raise ValueError("binary labels must contain only 0 and 1")

        sample_count, feature_count = features.shape
        self.weights_ = np.zeros(feature_count, dtype=np.float64)
        self.bias_ = 0.0
        self.loss_history_ = []
        previous_loss = float("inf")

        for iteration in range(1, self.max_iter + 1):
            scores = np.asarray(features @ self.weights_).ravel() + self.bias_
            probabilities = self.sigmoid(scores)
            errors = probabilities - labels

            gradient_weights = np.asarray(features.T @ errors).ravel() / sample_count
            if self.l2:
                gradient_weights += self.l2 * self.weights_
            gradient_bias = float(np.mean(errors))
            self.weights_ -= self.learning_rate * gradient_weights
            self.bias_ -= self.learning_rate * gradient_bias

            clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
            loss = float(
                -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
            )
            if self.l2:
                loss += 0.5 * self.l2 * float(self.weights_ @ self.weights_)
            self.loss_history_.append(loss)
            self.n_iter_ = iteration
            if abs(previous_loss - loss) < self.tolerance:
                break
            previous_loss = loss
        return self

    def decision_function(self, features: Any) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("model has not been fitted")
        return np.asarray(features @ self.weights_).ravel() + self.bias_

    def predict_proba(self, features: Any) -> np.ndarray:
        positive = self.sigmoid(self.decision_function(features))
        return np.column_stack((1.0 - positive, positive))

    def predict(self, features: Any, *, threshold: float = 0.5) -> np.ndarray:
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between zero and one")
        return (self.predict_proba(features)[:, 1] >= threshold).astype(np.int64)
