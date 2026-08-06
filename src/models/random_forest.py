"""
random_forest.py
-----------------
From-scratch implementation of a Random Forest Classifier.

Builds directly on the recursive, Gini-impurity-based splitting logic used in
the decision tree algorithm, and wraps it with three ensemble ideas:

  1. Bootstrap Aggregating (Bagging)  -> each tree is trained on a random
     sample of the training rows, drawn WITH replacement.
  2. Feature Randomness               -> each split only considers a random
     subset of features (max_features), which decorrelates the trees.
  3. Out-of-Bag (OOB) Estimation      -> because ~1/3 of rows are never
     selected for a given tree's bootstrap sample, those "left out" rows can
     be used as a free validation set for that tree.
"""

import numpy as np
from collections import Counter

# --------------------------------------------------------------------------
# Internal building block: a single CART-style decision tree
# --------------------------------------------------------------------------
class _Node:
    """A single node in a decision tree. Either a split node or a leaf."""

    __slots__ = ("feature_index", "threshold", "left", "right", "value")

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # only set on leaf nodes

    def is_leaf(self):
        return self.value is not None


class _DecisionTree:
    """
    A single Gini-impurity decision tree used as the base learner inside the
    forest. Mirrors the same splitting logic as the standalone decision tree
    algorithm, with one addition: max_features, which restricts each split to
    a random subset of columns (this is what turns a bag of identical trees
    into a genuinely diverse forest).
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.root = None
        self.n_features_ = None
        self.feature_importances_ = None
        self._rng = np.random.RandomState(random_state)

    # ---- impurity ----
    @staticmethod
    def _gini(y):
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _feature_subset(self):
        """Randomly choose which features are eligible for this split."""
        if self.max_features is None:
            return np.arange(self.n_features_)
        n_sub = self.max_features
        return self._rng.choice(self.n_features_, size=n_sub, replace=False)

    def _best_split(self, X, y):
        best_gain, best_feat, best_thresh = -1.0, None, None
        parent_gini = self._gini(y)
        n = len(y)

        for feat_idx in self._feature_subset():
            values = X[:, feat_idx]
            thresholds = np.unique(values)
            # Using midpoints keeps the split boundary off an actual data point
            if len(thresholds) > 1:
                thresholds = (thresholds[:-1] + thresholds[1:]) / 2

            for t in thresholds:
                left_mask = values <= t
                right_mask = ~left_mask
                n_left, n_right = left_mask.sum(), right_mask.sum()

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])
                weighted_gini = (n_left / n) * gini_left + (n_right / n) * gini_right
                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat_idx, t

        return best_feat, best_thresh, best_gain

    def _build(self, X, y, depth):
        n_samples = len(y)
        n_classes = len(np.unique(y))

        # Stopping conditions -> make a leaf
        if (n_classes == 1
                or n_samples < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)):
            return _Node(value=Counter(y).most_common(1)[0][0])

        feat_idx, threshold, gain = self._best_split(X, y)

        if feat_idx is None or gain <= 0:
            return _Node(value=Counter(y).most_common(1)[0][0])

        # Track impurity decrease for feature importance, weighted by
        # how much of the training data passed through this node
        self._importance_accum[feat_idx] += gain * (n_samples / self._n_total)

        left_mask = X[:, feat_idx] <= threshold
        right_mask = ~left_mask

        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[right_mask], y[right_mask], depth + 1)

        return _Node(feature_index=feat_idx, threshold=threshold, left=left, right=right)

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self._n_total = len(y)
        self._importance_accum = np.zeros(self.n_features_)
        self.root = self._build(X, y, depth=0)

        total = self._importance_accum.sum()
        self.feature_importances_ = (
            self._importance_accum / total if total > 0 else self._importance_accum
        )
        return self

    def _predict_one(self, x, node):
        if node.is_leaf():
            return node.value
        branch = node.left if x[node.feature_index] <= node.threshold else node.right
        return self._predict_one(x, branch)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])


# --------------------------------------------------------------------------
# The ensemble: Random Forest
# --------------------------------------------------------------------------
class RandomForestScratch:
    """
    Random Forest Classifier built from scratch on top of `_DecisionTree`.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    max_depth : int or None
        Max depth passed to every tree.
    min_samples_split : int
        Minimum samples required to split an internal node.
    min_samples_leaf : int
        Minimum samples required at a leaf.
    max_features : {"sqrt", "log2", None, int}
        Number of features considered at each split.
        "sqrt" (default, matches sklearn's classifier default) -> sqrt(n_features)
    bootstrap : bool
        Whether to bootstrap-sample rows per tree. If False, every tree sees
        the full training set (only feature randomness decorrelates them).
    random_state : int
        Seed for reproducibility.
    """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features="sqrt", bootstrap=True,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.trees_ = []
        self.feature_importances_ = None
        self.oob_score_ = None
        self.n_features_ = None

    def _resolve_max_features(self, n_features):
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        if self.max_features is None:
            return n_features
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        raise ValueError(f"Unsupported max_features: {self.max_features}")

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        m_features = self._resolve_max_features(n_features)

        rng = np.random.RandomState(self.random_state)
        self.trees_ = []
        importance_sum = np.zeros(n_features)

        # OOB bookkeeping: for every row, accumulate votes only from trees
        # that did NOT see that row during training
        oob_votes = [Counter() for _ in range(n_samples)]

        for i in range(self.n_estimators):
            tree_seed = rng.randint(0, 1_000_000)

            if self.bootstrap:
                sample_idx = rng.randint(0, n_samples, size=n_samples)
            else:
                sample_idx = np.arange(n_samples)

            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[sample_idx] = False
            oob_idx = np.where(oob_mask)[0]

            tree = _DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=m_features,
                random_state=tree_seed,
            )
            tree.fit(X[sample_idx], y[sample_idx])
            self.trees_.append(tree)
            importance_sum += tree.feature_importances_

            if len(oob_idx) > 0:
                oob_preds = tree.predict(X[oob_idx])
                for idx, pred in zip(oob_idx, oob_preds):
                    oob_votes[idx][pred] += 1

        # Aggregate feature importance across all trees
        self.feature_importances_ = importance_sum / self.n_estimators

        # Compute OOB accuracy for rows that were left out at least once
        correct, total = 0, 0
        for i in range(n_samples):
            if len(oob_votes[i]) > 0:
                predicted = oob_votes[i].most_common(1)[0][0]
                correct += int(predicted == y[i])
                total += 1
        self.oob_score_ = correct / total if total > 0 else None

        return self

    def predict_proba(self, X):
        """Fraction of trees voting for the positive class (class 1)."""
        X = np.asarray(X)
        votes = np.array([tree.predict(X) for tree in self.trees_])  # shape (n_trees, n_samples)
        return votes.mean(axis=0)

    def predict(self, X):
        X = np.asarray(X)
        votes = np.array([tree.predict(X) for tree in self.trees_])  # shape (n_trees, n_samples)
        # Majority vote per sample
        preds = np.apply_along_axis(
            lambda col: Counter(col).most_common(1)[0][0], axis=0, arr=votes
        )
        return preds