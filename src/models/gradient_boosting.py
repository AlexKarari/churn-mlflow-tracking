"""
Gradient Boosting Classifier — built from scratch with NumPy.
Sequential tree fitting on pseudo-residuals (log-loss gradient boosting).
Validated against sklearn.ensemble.GradientBoostingClassifier.
"""

import numpy as np
from dataclasses import dataclass


class DecisionStumpRegressor:
    """
    A shallow CART regression tree used as the weak learner.
    Reuses the same recursive-splitting logic as Day 7, but splits
    on SSE (variance reduction) instead of Gini, since we're fitting
    continuous residuals, not classes.
    """

    def __init__(self, max_depth=3, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _sse(self, y):
        if len(y) == 0:
            return 0.0
        return np.sum((y - np.mean(y)) ** 2)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_gain = -np.inf
        best_feat, best_thresh = None, None
        parent_sse = self._sse(y)

        for feat_idx in range(n_features):
            thresholds = np.unique(X[:, feat_idx])
            # subsample thresholds for speed on continuous features
            if len(thresholds) > 20:
                thresholds = np.percentile(X[:, feat_idx], np.linspace(5, 95, 20))

            for t in thresholds:
                left_mask = X[:, feat_idx] <= t
                right_mask = ~left_mask
                if left_mask.sum() < 1 or right_mask.sum() < 1:
                    continue

                gain = parent_sse - (self._sse(y[left_mask]) + self._sse(y[right_mask]))
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = t

        return best_feat, best_thresh, best_gain

    def _build(self, X, y, depth):
        if (depth >= self.max_depth or len(y) < self.min_samples_split
                or np.all(y == y[0])):
            return {"leaf": True, "value": np.mean(y)}

        feat, thresh, gain = self._best_split(X, y)
        if feat is None or gain <= 1e-12:
            return {"leaf": True, "value": np.mean(y)}

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        return {
            "leaf": False,
            "feature": feat,
            "threshold": thresh,
            "left": self._build(X[left_mask], y[left_mask], depth + 1),
            "right": self._build(X[right_mask], y[right_mask], depth + 1),
        }

    def fit(self, X, y):
        self.tree = self._build(X, y, depth=0)
        return self

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["value"]
        branch = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return self._predict_one(x, branch)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class GBMConfig:
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 3
    min_samples_split: int = 10
    early_stopping_rounds: int = 10
    subsample: float = 1.0
    random_state: int = 42


class GradientBoostingScratch:
    """
    Binary classification gradient boosting via log-loss.

    Core idea:
      1. Start with a constant log-odds prediction (base_score).
      2. At each iteration, compute the negative gradient of log-loss
         w.r.t. current predictions -> this is the "pseudo-residual":
             residual_i = y_i - sigmoid(F_{m-1}(x_i))
      3. Fit a shallow regression tree to those residuals.
      4. Update: F_m(x) = F_{m-1}(x) + learning_rate * tree_m(x)
      5. Repeat, optionally with early stopping on a validation set.
    """

    def __init__(self, config: GBMConfig = None):
        self.config = config or GBMConfig()
        self.trees = []
        self.base_score = 0.0
        self.train_loss_history = []
        self.val_loss_history = []
        self.best_iteration = None

    @staticmethod
    def _log_loss(y_true, y_pred_proba):
        eps = 1e-15
        p = np.clip(y_pred_proba, eps, 1 - eps)
        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def fit(self, X, y, X_val=None, y_val=None, verbose=True):
        rng = np.random.RandomState(self.config.random_state)
        n_samples = X.shape[0]

        # Initialize with log-odds of the base rate
        p0 = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
        self.base_score = np.log(p0 / (1 - p0))

        F_train = np.full(n_samples, self.base_score)
        F_val = None
        if X_val is not None:
            F_val = np.full(X_val.shape[0], self.base_score)

        best_val_loss = np.inf
        rounds_no_improve = 0

        for m in range(self.config.n_estimators):
            proba_train = sigmoid(F_train)
            residuals = y - proba_train  # negative gradient of log-loss

            # Row subsampling (stochastic gradient boosting)
            if self.config.subsample < 1.0:
                idx = rng.choice(n_samples, size=int(n_samples * self.config.subsample), replace=False)
            else:
                idx = np.arange(n_samples)

            tree = DecisionStumpRegressor(
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
            )
            tree.fit(X[idx], residuals[idx])
            self.trees.append(tree)

            update_train = tree.predict(X)
            F_train += self.config.learning_rate * update_train
            train_loss = self._log_loss(y, sigmoid(F_train))
            self.train_loss_history.append(train_loss)

            if X_val is not None:
                update_val = tree.predict(X_val)
                F_val += self.config.learning_rate * update_val
                val_loss = self._log_loss(y_val, sigmoid(F_val))
                self.val_loss_history.append(val_loss)

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    self.best_iteration = m
                    rounds_no_improve = 0
                else:
                    rounds_no_improve += 1

                if rounds_no_improve >= self.config.early_stopping_rounds:
                    if verbose:
                        print(f"Early stopping at iteration {m} "
                              f"(best={self.best_iteration}, val_loss={best_val_loss:.4f})")
                    break

            if verbose and (m % 10 == 0 or m == self.config.n_estimators - 1):
                msg = f"Iter {m:3d} | train_loss={train_loss:.4f}"
                if X_val is not None:
                    msg += f" | val_loss={val_loss:.4f}"
                print(msg)

        return self

    def _decision_function(self, X, n_trees=None):
        n_trees = n_trees or len(self.trees)
        F = np.full(X.shape[0], self.base_score)
        for tree in self.trees[:n_trees]:
            F += self.config.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X):
        return sigmoid(self._decision_function(X))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)