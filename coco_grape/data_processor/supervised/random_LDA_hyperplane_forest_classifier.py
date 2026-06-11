import numpy as np
from collections import Counter
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.utils import check_random_state
from numba import njit, prange
from sklearn.preprocessing import LabelEncoder
import itertools

# ---------------------- Node Classes ----------------------

class LeafNode:
    """
    Leaf node that holds the predicted class label, the distribution of classes,
    and a permanent node_id.
    """
    __slots__ = ['predicted_class', 'class_counts', 'node_id']

    def __init__(self, predicted_class, class_counts, node_id=None):
        self.predicted_class = predicted_class
        self.class_counts = class_counts  # 1D numpy array over all classes
        self.node_id = node_id

class InternalNode:
    """
    Internal node that holds a linear discriminant hyperplane (w, b),
    pointers to left/right children, plus a permanent node_id.
    """
    __slots__ = ['w', 'b', 'left_child', 'right_child', 'node_id']

    def __init__(self, w, b, left_child, right_child, node_id=None):
        self.w = w              # weight vector (normal)
        self.b = b              # offset
        self.left_child = left_child
        self.right_child = right_child
        self.node_id = node_id

# ---------------------- Tree Class ----------------------

class RandomLDAHyperplaneTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    A decision tree that recursively partitions the feature space using a one-vs-all
    LDA hyperplane.
    """

    def __init__(self, min_samples=10, min_gini=0.01, max_depth=10, random_state=None):
        self.min_samples = min_samples
        self.min_gini = min_gini
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self._rng = check_random_state(self.random_state)
        self._node_id_counter = 0  # for permanent node IDs
        self.root_ = self._build_tree(X, y, depth=0)
        return self

    def _gini(self, y):
        total = len(y)
        if total == 0:
            return 0.0
        probs = np.array([np.sum(y == cl) / total for cl in self.classes_])
        return 1.0 - np.sum(probs ** 2)

    def _majority_class(self, y):
        counts = [np.sum(y == cl) for cl in self.classes_]
        majority = self.classes_[np.argmax(counts)]
        return majority, np.array(counts, dtype=np.int32)

    def _compute_hyperplane(self, X, y, positive_class):
        # Create binary labels: positive for chosen class, negative for all others.
        pos_mask = (y == positive_class)
        neg_mask = ~pos_mask
        if np.sum(pos_mask) == 0 or np.sum(neg_mask) == 0:
            raise ValueError("Invalid split: one group is empty.")
        X_pos = X[pos_mask]
        X_neg = X[neg_mask]
        mean_pos = X_pos.mean(axis=0)
        mean_neg = X_neg.mean(axis=0)
        # Compute (regularized) covariances
        cov_pos = np.cov(X_pos, rowvar=False, bias=True) if X_pos.shape[0] > 1 else np.eye(self.n_features_in_) * 1e-6
        cov_neg = np.cov(X_neg, rowvar=False, bias=True) if X_neg.shape[0] > 1 else np.eye(self.n_features_in_) * 1e-6
        S_w = cov_pos + cov_neg
        # Compute pseudo-inverse in case S_w is singular
        w = np.linalg.pinv(S_w).dot(mean_pos - mean_neg)
        b = -0.5 * np.dot(w, (mean_pos + mean_neg))
        return w, b

    def _build_tree(self, X, y, depth):
        current_gini = self._gini(y)
        # Terminal conditions
        if (len(y) < self.min_samples or current_gini < self.min_gini or depth >= self.max_depth
                or len(np.unique(y)) == 1):
            return self._make_leaf_node(y)
        # Randomly choose one of the classes present in this node as positive.
        poss = np.unique(y)
        pos_class = self._rng.choice(poss)
        try:
            w, b = self._compute_hyperplane(X, y, pos_class)
        except Exception:
            return self._make_leaf_node(y)
        # Partition data
        decisions = X.dot(w) + b
        left_mask = decisions >= 0  # go to left child
        right_mask = ~left_mask       # go to right child
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return self._make_leaf_node(y)
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return self._make_internal_node(w, b, left_child, right_child)

    def _make_leaf_node(self, y):
        majority, counts = self._majority_class(y)
        node_id = self._node_id_counter
        self._node_id_counter += 1
        return LeafNode(majority, counts, node_id=node_id)

    def _make_internal_node(self, w, b, left_child, right_child):
        node_id = self._node_id_counter
        self._node_id_counter += 1
        return InternalNode(w, b, left_child, right_child, node_id=node_id)

    def _predict_single(self, x, node):
        if isinstance(node, LeafNode):
            return node.predicted_class
        decision = np.dot(node.w, x) + node.b
        if decision >= 0:
            return self._predict_single(x, node.left_child)
        else:
            return self._predict_single(x, node.right_child)

    def predict(self, X):
        check_is_fitted(self, 'root_')
        X = check_array(X)
        preds = np.array([self._predict_single(x, self.root_) for x in X])
        return preds

    def predict_proba(self, X):
        """For each sample, return the probability as stored at the reached leaf."""
        check_is_fitted(self, 'root_')
        X = check_array(X)
        proba = []
        for x in X:
            leaf = self._get_leaf(x, self.root_)
            p = leaf.class_counts / leaf.class_counts.sum()
            proba.append(p)
        return np.array(proba)

    def _get_leaf(self, x, node):
        if isinstance(node, LeafNode):
            return node
        decision = np.dot(node.w, x) + node.b
        if decision >= 0:
            return self._get_leaf(x, node.left_child)
        else:
            return self._get_leaf(x, node.right_child)

    def to_array(self):
        """
        Convert the tree into BFS-ordered arrays for fast (compiled) prediction.
        """
        records = []

        def traverse(node):
            idx = len(records)
            rec = {
                'normal': None,
                'offset': 0.0,
                'left': -1,
                'right': -1,
                'is_leaf': False,
                'leaf_val': -1,
                'node_id': node.node_id
            }
            records.append(rec)
            if isinstance(node, InternalNode):
                rec['normal'] = node.w
                rec['offset'] = node.b
                rec['is_leaf'] = False
                left_idx = traverse(node.left_child)
                right_idx = traverse(node.right_child)
                rec['left'] = left_idx
                rec['right'] = right_idx
            else:
                rec['normal'] = np.zeros(self.n_features_in_)
                rec['offset'] = 0.0
                rec['is_leaf'] = True
                rec['leaf_val'] = node.predicted_class
            return idx

        traverse(self.root_)
        normals = np.array([r['normal'] for r in records], dtype=np.float64)
        offsets = np.array([r['offset'] for r in records], dtype=np.float64)
        left_children = np.array([r['left'] for r in records], dtype=np.int32)
        right_children = np.array([r['right'] for r in records], dtype=np.int32)
        is_leaf = np.array([r['is_leaf'] for r in records], dtype=bool)
        leaf_values = np.array([r['leaf_val'] for r in records], dtype=np.int32)
        node_ids = np.array([r['node_id'] for r in records], dtype=np.int32)
        return normals, offsets, left_children, right_children, leaf_values, is_leaf, node_ids

# ---------------------- Compiled Prediction ----------------------

@njit(parallel=True)
def predict_tree_compiled_parallel(X, normals, offsets, left_children, right_children, leaf_values, is_leaf):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    y_pred = np.empty(n_samples, dtype=np.int32)
    for i in prange(n_samples):
        node = 0  # start at root
        while True:
            if is_leaf[node]:
                y_pred[i] = leaf_values[node]
                break
            decision = 0.0
            for f in range(n_features):
                decision += X[i, f] * normals[node, f]
            decision += offsets[node]
            if decision >= 0:
                node = left_children[node]
            else:
                node = right_children[node]
    return y_pred

# ---------------------- Revised Forest Class ----------------------
class RandomLDAHyperplaneForestClassifier(BaseEstimator, ClassifierMixin):
    """
    A forest classifier that aggregates an ensemble of RandomLDAHyperplaneTreeClassifiers.
    This revised version uses an outer (per‑tree) sequential loop for prediction,
    matching the strategy of the second code base.
    """

    def __init__(self, n_estimators=100, bootstrap=True,
                 min_samples_split=5, min_impurity=0.1, max_depth=10,
                 random_state=None, n_jobs=-1, use_compiled_predict=True):
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.use_compiled_predict = use_compiled_predict

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.le_ = LabelEncoder()
        y_encoded = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        self.n_features_in_ = X.shape[1]
        rng = check_random_state(self.random_state)
        seeds = rng.randint(np.iinfo(np.int32).max, size=self.n_estimators)

        # Train trees in parallel (per-tree parallelism remains here)
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single_tree)(
                X, y_encoded,
                self.min_samples_split, self.min_impurity,
                self.max_depth, self.bootstrap, seed
            ) for seed in seeds
        )
        # Build compiled array representations if needed.
        if self.use_compiled_predict:
            self.compiled_trees_ = [est.to_array() for est in self.estimators_]
        else:
            self.compiled_trees_ = None
        return self

    def predict(self, X):
        """
        Predict class labels for the input samples X.
        This revised version uses a sequential loop over trees (when using compiled arrays).
        """
        check_is_fitted(self, 'estimators_')
        X = check_array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        votes = np.zeros((n_samples, n_classes), dtype=np.int32)

        if self.use_compiled_predict and self.compiled_trees_ is not None:
            # Sequentially loop over trees (instead of using joblib Parallel)
            for arrays in self.compiled_trees_:
                preds = predict_tree_compiled_parallel(
                    X, arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], arrays[5]
                )
                votes[np.arange(n_samples), preds] += 1
        else:
            # Use each tree's standard predict method (can be parallelized)
            for est in self.estimators_:
                preds = est.predict(X)
                for i, pred in enumerate(preds):
                    votes[i, pred] += 1

        y_pred_encoded = np.argmax(votes, axis=1)
        return self.le_.inverse_transform(y_pred_encoded)

    def predict_proba(self, X):
        """
        Predict class probabilities for the input samples X.
        This version also uses a sequential loop over trees when using compiled arrays.
        """
        check_is_fitted(self, 'estimators_')
        X = check_array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes), dtype=float)

        if self.use_compiled_predict and self.compiled_trees_ is not None:
            for arrays in self.compiled_trees_:
                preds = predict_tree_compiled_parallel(
                    X, arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], arrays[5]
                )
                proba[np.arange(n_samples), preds] += 1
        else:
            for est in self.estimators_:
                preds = est.predict(X)
                for i, pred in enumerate(preds):
                    proba[i, pred] += 1

        proba /= self.n_estimators
        return proba

    def predict_mean_std(self, X):
        """
        For each sample, compute the mean and standard deviation (across trees)
        of the one-hot encoded predicted probabilities.
        This method now uses a sequential loop over trees.
        """
        check_is_fitted(self, 'estimators_')
        X = check_array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        all_probas = []  # list of (n_samples x n_classes) arrays

        if self.use_compiled_predict and self.compiled_trees_ is not None:
            for arrays in self.compiled_trees_:
                preds = predict_tree_compiled_parallel(
                    X, arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], arrays[5]
                )
                one_hot = np.zeros((n_samples, n_classes), dtype=float)
                for i in range(n_samples):
                    one_hot[i, preds[i]] = 1.0
                all_probas.append(one_hot)
        else:
            for est in self.estimators_:
                preds = est.predict(X)
                one_hot = np.zeros((n_samples, n_classes), dtype=float)
                for i in range(n_samples):
                    one_hot[i, preds[i]] = 1.0
                all_probas.append(one_hot)

        all_probas = np.array(all_probas)  # shape: (n_estimators, n_samples, n_classes)
        mean_proba = np.mean(all_probas, axis=0)
        std_proba = np.std(all_probas, axis=0)
        return mean_proba, std_proba

# ---------------------- Helper: Fit Single Tree ----------------------

def _fit_single_tree(X, y, min_samples, min_gini, max_depth, bootstrap, seed):
    rng = np.random.default_rng(seed=seed)
    n_samples = X.shape[0]
    if bootstrap:
        # Bootstrap: sample n_samples with replacement.
        indices = rng.choice(n_samples, size=n_samples, replace=True)
    else:
        # 70/30 split: sample 70% without replacement.
        train_size = int(0.7 * n_samples)
        indices = rng.choice(n_samples, size=train_size, replace=False)
    X_subset = X[indices]
    y_subset = y[indices]
    tree = RandomLDAHyperplaneTreeClassifier(min_samples=min_samples, min_gini=min_gini,
                                              max_depth=max_depth, random_state=seed)
    tree.fit(X_subset, y_subset)
    return tree
