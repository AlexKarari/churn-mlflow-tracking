"""
decision_tree.py
================
Decision Tree Classifier built from scratch using NumPy.

Algorithm: CART (Classification and Regression Trees)
Splitting criterion: Gini Impurity
Strategy: Greedy recursive binary splitting

How it works:
-------------------------------
Imagine sorting customers into churn vs. no-churn by asking a series of
yes/no questions:
  "Is their contract month-to-month?" → Yes → likely to churn
                                      → No  → ask another question...

The tree learns WHICH questions to ask and in WHAT ORDER by choosing,
at each step, the split that best separates churners from non-churners.
"Best" is measured by Gini Impurity — a score that tells us how mixed
a group is.

Key Design Decisions:
---------------------
- Binary splits only (left child / right child)
- Gini impurity as the purity metric (faster than entropy, similar results)
- Greedy best-first splitting (no backtracking)
- Stopping criteria: max_depth, min_samples_split, min_samples_leaf

"""

import numpy as np
from collections import Counter


# =============================================================================
# Node — the building block of the tree
# =============================================================================

class Node:
    """
    Represents a single node in the decision tree.

    A node is either:
      - A decision node: stores the split rule (feature + threshold)
                         and has a left and right child
      - A leaf node:     stores the final prediction (majority class)
                         and has NO children

    Parameters
    ----------
    feature_index : int or None
        Index of the feature to split on (None for leaf nodes).
    threshold : float or None
        The value to compare the feature against (None for leaf nodes).
        Split rule: go LEFT if X[feature] <= threshold, else RIGHT.
    left : Node or None
        Left child node (samples where feature <= threshold).
    right : Node or None
        Right child node (samples where feature > threshold).
    value : int or None
        Predicted class label. Only set for leaf nodes.
    gini : float
        Gini impurity of this node's sample set. Useful for debugging.
    n_samples : int
        Number of training samples that reached this node.
    """

    def __init__(
        self,
        feature_index: int = None,
        threshold: float = None,
        left: "Node" = None,
        right: "Node" = None,
        value: int = None,
        gini: float = 0.0,
        n_samples: int = 0,
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value          # Only meaningful for leaf nodes
        self.gini = gini
        self.n_samples = n_samples

    @property
    def is_leaf(self) -> bool:
        """True if this node makes a final prediction (no children)."""
        return self.value is not None


# =============================================================================
# DecisionTreeScratch — the main classifier
# =============================================================================

class DecisionTreeScratch:
    """
    Binary Decision Tree Classifier built from scratch.

    Implements the CART algorithm:
      1. At each node, search every feature and threshold for the best split
      2. "Best" = highest Information Gain (= biggest drop in Gini Impurity)
      3. Split the data and recurse on both halves
      4. Stop when a stopping criterion is met → create a leaf node

    Parameters
    ----------
    max_depth : int, default=10
        Maximum depth of the tree. Deeper trees fit training data better
        but risk overfitting. Controls model complexity.
    min_samples_split : int, default=2
        Minimum number of samples a node must have to attempt a split.
        If a node has fewer samples than this, it becomes a leaf.
    min_samples_leaf : int, default=1
        Minimum number of samples required in each child after a split.
        Prevents splits that create nearly-empty leaves.
    n_thresholds : int or None, default=None
        If set, limits the number of threshold values tried per feature.
        Use this to speed up training on large datasets (trades accuracy
        for speed). None = try all unique values.
    random_state : int or None, default=None
        Seed for reproducible threshold sampling (used with n_thresholds).

    Attributes
    ----------
    root : Node
        The root node of the fitted tree.
    n_features_ : int
        Number of features seen during fit.
    n_classes_ : int
        Number of unique classes in the training labels.
    feature_importances_ : np.ndarray of shape (n_features_,)
        How much each feature contributed to splitting decisions,
        weighted by the number of samples at each split.
        Higher = more important.

    Examples
    --------
    >>> tree = DecisionTreeScratch(max_depth=5)
    >>> tree.fit(X_train, y_train)
    >>> predictions = tree.predict(X_test)
    >>> probabilities = tree.predict_proba(X_test)
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        n_thresholds: int = None,
        random_state: int = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_thresholds = n_thresholds
        self.random_state = random_state

        # Set after fit()
        self.root: Node = None
        self.n_features_: int = None
        self.n_classes_: int = None
        self.feature_importances_: np.ndarray = None

        # Internal: accumulates importance scores during tree building
        self._importance_accumulator: np.ndarray = None

        if random_state is not None:
            np.random.seed(random_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeScratch":
        """
        Build the decision tree from training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray of shape (n_samples,)
            Training labels (integers, e.g. 0 and 1 for binary classification).

        Returns
        -------
        self : DecisionTreeScratch
            Fitted estimator (allows method chaining).
        """
        self.n_features_ = X.shape[1]
        self.n_classes_ = len(np.unique(y))
        self._importance_accumulator = np.zeros(self.n_features_)

        # Recursively build the tree starting from the root
        self.root = self._build_tree(X, y, depth=0)

        # Normalise importances so they sum to 1.0 (same as sklearn)
        total = self._importance_accumulator.sum()
        if total > 0:
            self.feature_importances_ = self._importance_accumulator / total
        else:
            self.feature_importances_ = self._importance_accumulator

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for each sample in X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        return np.array([self._traverse(x, self.root) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for each sample.

        The probability of class 1 at a leaf node is estimated as the
        proportion of training samples at that leaf that belonged to class 1.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        proba : np.ndarray of shape (n_samples, 2)
            proba[:, 0] = probability of class 0
            proba[:, 1] = probability of class 1
        """
        return np.array([self._traverse_proba(x, self.root) for x in X])

    # ------------------------------------------------------------------
    # Tree building (private)
    # ------------------------------------------------------------------

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """
        Recursively build the tree.

        At each call this function either:
          a) Returns a LEAF node (if a stopping criterion is met), or
          b) Finds the best split, divides the data, and recurses into
             two child nodes.

        Stopping criteria (any one triggers a leaf):
          - depth >= max_depth
          - n_samples < min_samples_split
          - Node is already pure (only one class present)

        Parameters
        ----------
        X : np.ndarray — feature matrix for this node's samples
        y : np.ndarray — labels for this node's samples
        depth : int    — current depth in the tree (root = 0)

        Returns
        -------
        Node
        """
        n_samples = len(y)
        gini = self._gini(y)

        # ── Stopping criteria ──────────────────────────────────────────
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or gini == 0.0          # Node is pure — perfect separation
        ):
            return self._make_leaf(y, gini, n_samples)

        # ── Find the best split ────────────────────────────────────────
        best = self._best_split(X, y)

        # If no valid split was found (rare edge case), make a leaf
        if best["gain"] <= 0:
            return self._make_leaf(y, gini, n_samples)

        # ── Partition data and recurse ─────────────────────────────────
        left_mask  = X[:, best["feature"]] <= best["threshold"]
        right_mask = ~left_mask

        left_child  = self._build_tree(X[left_mask],  y[left_mask],  depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        # ── Accumulate feature importance ──────────────────────────────
        # Importance of a split = (n_samples / total_samples) * information_gain
        # We don't have total_samples here, so we accumulate the unnormalised
        # version (n_samples * gain) and normalise in fit().
        self._importance_accumulator[best["feature"]] += n_samples * best["gain"]

        return Node(
            feature_index=best["feature"],
            threshold=best["threshold"],
            left=left_child,
            right=right_child,
            gini=gini,
            n_samples=n_samples,
        )

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Search all features and thresholds for the split with the highest
        Information Gain.

        For each feature:
          1. Collect all unique values as candidate thresholds
          2. For each threshold, compute the weighted Gini of the split
          3. Compute Information Gain = parent_gini - weighted_child_gini
          4. Keep track of the best (feature, threshold) pair

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of shape (n_samples,)

        Returns
        -------
        best : dict with keys:
            "feature"   : int   — index of best feature
            "threshold" : float — best split threshold
            "gain"      : float — information gain achieved
        """
        parent_gini = self._gini(y)
        n_samples, n_features = X.shape

        best = {"feature": None, "threshold": None, "gain": -np.inf}

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)

            # Optionally sub-sample thresholds to speed up training
            if self.n_thresholds is not None and len(thresholds) > self.n_thresholds:
                thresholds = np.random.choice(thresholds, self.n_thresholds, replace=False)

            for threshold in thresholds:
                left_mask  = feature_values <= threshold
                right_mask = ~left_mask

                # Skip if a child would be smaller than min_samples_leaf
                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue

                y_left  = y[left_mask]
                y_right = y[right_mask]

                # Weighted Gini of the two children
                weighted_child_gini = (
                    len(y_left)  / n_samples * self._gini(y_left) +
                    len(y_right) / n_samples * self._gini(y_right)
                )

                # Information Gain = reduction in impurity
                gain = parent_gini - weighted_child_gini

                if gain > best["gain"]:
                    best["gain"]      = gain
                    best["feature"]   = feature_idx
                    best["threshold"] = threshold

        return best

    # ------------------------------------------------------------------
    # Gini Impurity (the core math)
    # ------------------------------------------------------------------

    def _gini(self, y: np.ndarray) -> float:
        """
        Compute Gini Impurity for a set of labels.

        Gini Impurity = 1 - Σ(p_k²)
        where p_k is the proportion of class k in the node.

        Intuition:
          - Gini = 0.0  → perfectly pure  (all samples are the same class)
          - Gini = 0.5  → maximally mixed (50% each class for binary problem)

        Example (binary):
          y = [0, 0, 0, 0]  → p0=1.0, p1=0.0 → Gini = 1 - (1² + 0²) = 0.0
          y = [0, 0, 1, 1]  → p0=0.5, p1=0.5 → Gini = 1 - (0.25 + 0.25) = 0.5
          y = [0, 1, 1, 1]  → p0=0.25, p1=0.75 → Gini = 1 - (0.0625 + 0.5625) = 0.375

        Parameters
        ----------
        y : np.ndarray — class labels

        Returns
        -------
        float — Gini impurity in [0, 0.5] for binary classification
        """
        if len(y) == 0:
            return 0.0

        n = len(y)
        counts = np.bincount(y)       # e.g. [420, 180] for 420 class-0, 180 class-1
        probs = counts / n            # class probabilities
        return 1.0 - np.sum(probs ** 2)

    # ------------------------------------------------------------------
    # Leaf creation
    # ------------------------------------------------------------------

    def _make_leaf(self, y: np.ndarray, gini: float, n_samples: int) -> Node:
        """
        Create a leaf node with a majority-vote prediction.

        The leaf stores the class that appeared most often among the training
        samples that reached this node. In case of a tie, NumPy's argmax
        picks the lower class index.

        Parameters
        ----------
        y         : np.ndarray — labels for samples at this node
        gini      : float      — impurity of this node (informational)
        n_samples : int        — number of samples (informational)

        Returns
        -------
        Node with value set (leaf node)
        """
        # Store class counts for predict_proba
        counts = np.bincount(y, minlength=self.n_classes_)
        majority_class = int(np.argmax(counts))

        leaf = Node(value=majority_class, gini=gini, n_samples=n_samples)
        # Attach counts for probability estimation
        leaf._class_counts = counts
        leaf._n_samples = n_samples
        return leaf

    # ------------------------------------------------------------------
    # Prediction traversal (private)
    # ------------------------------------------------------------------

    def _traverse(self, x: np.ndarray, node: Node) -> int:
        """
        Walk a single sample down the tree until reaching a leaf.

        At each decision node:
          - Go LEFT  if x[feature] <= threshold
          - Go RIGHT if x[feature] >  threshold

        Parameters
        ----------
        x    : np.ndarray of shape (n_features,) — one sample
        node : Node — current position in the tree

        Returns
        -------
        int — predicted class label
        """
        if node.is_leaf:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)

    def _traverse_proba(self, x: np.ndarray, node: Node) -> np.ndarray:
        """
        Walk a single sample down the tree and return class probabilities.

        At a leaf node, probability = (class count) / (total samples at leaf).
        This is a simple frequency estimate.

        Parameters
        ----------
        x    : np.ndarray of shape (n_features,)
        node : Node

        Returns
        -------
        np.ndarray of shape (n_classes,) — class probabilities summing to 1.0
        """
        if node.is_leaf:
            counts = getattr(node, "_class_counts", np.array([0, 0]))
            total  = counts.sum()
            if total == 0:
                return np.ones(self.n_classes_) / self.n_classes_
            return counts / total

        if x[node.feature_index] <= node.threshold:
            return self._traverse_proba(x, node.left)
        else:
            return self._traverse_proba(x, node.right)

    # ------------------------------------------------------------------
    # Tree inspection utilities
    # ------------------------------------------------------------------

    def print_tree(self, node: Node = None, feature_names: list = None,
                   depth: int = 0, max_print_depth: int = 4) -> None:
        """
        Print a human-readable text representation of the tree.

        Useful for understanding what rules the tree learned.

        Parameters
        ----------
        node            : Node — starting node (defaults to root)
        feature_names   : list — feature names for readable output
        depth           : int  — current depth (used for indentation)
        max_print_depth : int  — stop printing beyond this depth (avoids
                                  huge output for deep trees)

        Example Output
        --------------
        [D=0] Contract_Month-to-month <= 0.50  (gini=0.332, n=5634)
        ├── [D=1] tenure <= 9.50  (gini=0.449, n=2891)
        │   ├── [D=2] LEAF → class=1 (churn)  (n=842)
        │   └── [D=2] InternetService_Fiber optic <= 0.50  (n=2049)
        └── [D=1] LEAF → class=0 (stay)  (n=2743)
        """
        if node is None:
            node = self.root

        if depth > max_print_depth:
            return

        indent   = "│   " * depth
        prefix   = "├── " if depth > 0 else ""

        if node.is_leaf:
            class_label = "churn" if node.value == 1 else "stay"
            print(f"{indent}{prefix}[D={depth}] LEAF → class={node.value} ({class_label})  "
                  f"(n={node.n_samples})")
            return

        # Decision node
        fname = (feature_names[node.feature_index]
                 if feature_names else f"feature[{node.feature_index}]")
        print(f"{indent}{prefix}[D={depth}] {fname} <= {node.threshold:.2f}  "
              f"(gini={node.gini:.3f}, n={node.n_samples})")

        self.print_tree(node.left,  feature_names, depth + 1, max_print_depth)
        self.print_tree(node.right, feature_names, depth + 1, max_print_depth)

    def get_depth(self, node: Node = None) -> int:
        """
        Return the actual maximum depth of the fitted tree.

        Parameters
        ----------
        node : Node — starting node (defaults to root)

        Returns
        -------
        int — depth of the deepest leaf
        """
        if node is None:
            node = self.root
        if node is None:
            return 0
        if node.is_leaf:
            return 0
        return 1 + max(self.get_depth(node.left), self.get_depth(node.right))

    def count_nodes(self, node: Node = None) -> int:
        """
        Count total number of nodes (decision + leaf) in the tree.

        Parameters
        ----------
        node : Node — starting node (defaults to root)

        Returns
        -------
        int — total node count
        """
        if node is None:
            node = self.root
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        return 1 + self.count_nodes(node.left) + self.count_nodes(node.right)