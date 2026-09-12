"""Linear SVM classifier for sparse TF-IDF lexical features."""
from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import Any, Self

from sklearn.svm import LinearSVC


class LinearSVMClassifier:
    """Small wrapper around :class:`sklearn.svm.LinearSVC`.

    The classifier receives already-built feature matrices. It does not
    clean text, create TF-IDF features, or access data splits.
    """

    def __init__(self, C: float = 1.0, random_state: int | None = 42) -> None:
        if isinstance(C, bool) or not isinstance(C, Real) or C <= 0:
            raise ValueError("C must be a positive number.")

        self.C = float(C)
        self._model = LinearSVC(C=self.C, random_state=random_state)
        self._is_fitted = False

    def fit(self, X_train: Any, y_train: Sequence[Any]) -> Self:
        """Fit LinearSVC using training features and matching ESCI labels."""
        if not hasattr(X_train, "shape") or len(X_train.shape) != 2:
            raise ValueError("X_train must be a two-dimensional feature matrix.")
        if not hasattr(y_train, "__len__"):
            raise TypeError("y_train must be a sized sequence of labels.")
        if X_train.shape[0] != len(y_train):
            raise ValueError("X_train and y_train must contain the same number of rows.")
        if len(y_train) == 0:
            raise ValueError("Training data must not be empty.")
        if len(set(y_train)) < 2:
            raise ValueError("LinearSVC requires at least two label classes.")

        self._model.fit(X_train, y_train)
        self._is_fitted = True
        return self

    def predict(self, X: Any):
        """Predict labels from an already-created feature matrix."""
        self._require_fitted()
        return self._model.predict(X)

    def decision_function(self, X: Any):
        """Return LinearSVC decision scores for an already-created feature matrix."""
        self._require_fitted()
        return self._model.decision_function(X)

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("LinearSVMClassifier has not been fitted. Call fit() first.")
