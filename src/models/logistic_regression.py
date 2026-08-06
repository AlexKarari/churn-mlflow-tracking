"""
Logistic Regression from Scratch
================================

Binary classifier using sigmoid activation and gradient descent
with optional L2 (Ridge) regularization.

Mathematical Foundation
-----------------------
Hypothesis:     ŷ = σ(θᵀx) where σ(z) = 1/(1+e⁻ᶻ)
Cost function:  J(θ) = -(1/m)Σ[y·log(ŷ) + (1-y)·log(1-ŷ)] + (λ/2m)Σθ²
Gradient:       ∂J/∂θ = (1/m)Xᵀ(ŷ - y) + (λ/m)θ

Author: ML Learning Journey — Day 3
"""

import numpy as np
from typing import Optional


class LogisticRegressionScratch:
    """
    Logistic Regression binary classifier using gradient descent.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for gradient descent.
    n_iterations : int, default=10000
        Maximum number of gradient descent iterations.
    lambda_reg : float, default=0.0
        L2 regularization strength. 0.0 disables regularization.
    threshold : float, default=0.5
        Classification threshold for converting probabilities to labels.

    Attributes
    ----------
    weights : np.ndarray or None
        Learned feature weights after fitting.
    bias : float or None
        Learned bias (intercept) after fitting.
    cost_history : list of float
        Cross-entropy cost recorded every 100 iterations during training.

    Examples
    --------
    >>> from logistic_regression import LogisticRegressionScratch
    >>> model = LogisticRegressionScratch(learning_rate=0.01, n_iterations=10000)
    >>> model.fit(X_train_scaled, y_train)
    >>> predictions = model.predict(X_test)
    >>> probabilities = model.predict_proba(X_test)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 10000,
        lambda_reg: float = 0.0,
        threshold: float = 0.5,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg
        self.threshold = threshold
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None
        self.cost_history: list = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Numerically stable sigmoid function.

        Uses conditional computation to avoid overflow:
        - For z >= 0: σ(z) = 1 / (1 + e⁻ᶻ)
        - For z < 0:  σ(z) = eᶻ / (1 + eᶻ)

        Parameters
        ----------
        z : np.ndarray
            Linear combination of inputs.

        Returns
        -------
        np.ndarray
            Sigmoid output in range (0, 1).
        """
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z)),
        )

    def _compute_cost(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss with optional L2 regularization.

        J(θ) = -(1/m)Σ[y·log(ŷ) + (1-y)·log(1-ŷ)] + (λ/2m)Σθ²

        Parameters
        ----------
        y : np.ndarray, shape (m,)
            True binary labels.
        y_pred : np.ndarray, shape (m,)
            Predicted probabilities (already clipped).

        Returns
        -------
        float
            Total cost (cross-entropy + regularization).
        """
        m = len(y)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)

        ce_loss = -(1 / m) * np.sum(
            y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
        )
        l2_term = (self.lambda_reg / (2 * m)) * np.sum(self.weights ** 2)

        return ce_loss + l2_term

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        """
        Train the model using gradient descent.

        Parameters
        ----------
        X : np.ndarray, shape (m, n)
            Training feature matrix (should be scaled).
        y : np.ndarray, shape (m,)
            Binary target labels (0 or 1).

        Returns
        -------
        self
            Fitted model instance.
        """
        m, n = X.shape

        self.weights = np.zeros(n)
        self.bias = 0.0
        self.cost_history = []

        for i in range(self.n_iterations):
            # Forward pass
            z = X @ self.weights + self.bias
            y_pred = self._sigmoid(z)

            # Clip predictions to prevent gradient explosion.
            # Without this, saturated predictions (exactly 0.0 or 1.0 in
            # float64) produce runaway gradients -> NaN weights -> all-zero
            # predictions. This is critical for high-dimensional data
            # (e.g., 30 one-hot encoded features on the real Telco dataset).
            eps = 1e-15
            y_pred_safe = np.clip(y_pred, eps, 1 - eps)

            # Gradients (using clipped predictions for stability)
            error = y_pred_safe - y
            dw = (1 / m) * (X.T @ error) + (self.lambda_reg / m) * self.weights
            db = (1 / m) * np.sum(error)

            # Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Record cost
            if i % 100 == 0:
                cost = self._compute_cost(y, y_pred_safe)
                self.cost_history.append(cost)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of class 1.

        Parameters
        ----------
        X : np.ndarray, shape (m, n)
            Feature matrix.

        Returns
        -------
        np.ndarray, shape (m,)
            Predicted probabilities for class 1.
        """
        if self.weights is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        z = X @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels.

        Parameters
        ----------
        X : np.ndarray, shape (m, n)
            Feature matrix.

        Returns
        -------
        np.ndarray, shape (m,)
            Binary predictions (0 or 1).
        """
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def get_params(self) -> dict:
        """Return model hyperparameters."""
        return {
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "lambda_reg": self.lambda_reg,
            "threshold": self.threshold,
        }