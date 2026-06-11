import numpy as np
from collections import Counter
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import (
    check_is_fitted, check_X_y, check_array
)
from sklearn.utils import check_random_state
import itertools
from numba import njit, prange
from sklearn.preprocessing import LabelEncoder
import os
import copy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

os.environ["KMP_WARNINGS"] = "0"

# ----------------- Custom Node Classes Definition -----------------

class LeafNode:
    """
    Leaf node that holds the predicted class label, plus a permanent node_id.
    """
    __slots__ = ['predicted_class', 'node_id']

    def __init__(self, predicted_class, node_id=None):
        self.predicted_class = predicted_class
        self.node_id = node_id  # permanent ID assigned by the tree builder


class InternalNode:
    """
    Internal node that holds a hyperplane (normal, offset),
    left/right subtrees, plus a permanent node_id.
    """
    __slots__ = ['normal', 'offset', 'left_child', 'right_child', 'node_id']

    def __init__(self, normal, offset, left_child, right_child, node_id=None):
        self.normal = normal
        self.offset = offset
        self.left_child = left_child
        self.right_child = right_child
        self.node_id = node_id  # permanent ID assigned by the tree builder

# ----------------- RandomHyperplaneTreeClassifier Definition -----------------
class RandomHyperplaneTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    A random hyperplane tree classifier that recursively splits data
    by picking candidate hyperplanes (constructed from two random points)
    and then selecting one at random with probability proportional
    to the average distance (margin) of the points to that hyperplane.
    """

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        random_state=None,
        max_attempts_random_pair=10,
        min_impurity=0.10,
        num_candidates=3,
        gamma=10.0
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.max_attempts_random_pair = max_attempts_random_pair
        self.min_impurity = min_impurity
        self.num_candidates = num_candidates  
        self.gamma = gamma

        # Counter for assigning permanent IDs to nodes
        self._node_id_counter = 0

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self._rng = check_random_state(self.random_state)
        self.y_ = y  # Store encoded class labels
        # If bootstrapping was not used, _global_indices will not be set yet; then use the natural order.
        if not hasattr(self, '_global_indices'):
            self._global_indices = np.arange(len(X))
        indices = np.arange(len(X))
        self._pair_counts = {}  # Count of each pair of instances used for splitting
        self.root_ = self._build_tree(X, y, indices, depth=0)
        return self

    def _build_tree(self, X, y, indices, depth):
        y_node = y[indices]
        gini = self._gini_impurity(y_node)

        # Check terminal conditions.
        if (
            len(np.unique(y_node)) == 1
            or depth >= self.max_depth
            or len(indices) < self.min_samples_split
            or gini <= self.min_impurity
        ):
            return self._make_leaf_node(y_node)

        # The _pair_counts dictionary is already created in fit.
        # ---- Begin Candidate Hyperplane Selection for Multiclass ----
        candidates = []       # Each element: (normal, offset, majority_neg, majority_pos, candidate_pair)
        candidate_values = [] # Candidate's score based on the worst-case margin

        for candidate_idx in range(self.num_candidates):
            found_pair = False
            # Try random attempts to pick two points with different classes.
            for attempt in range(self.max_attempts_random_pair):
                i, j = self._rng.choice(len(indices), size=2, replace=False)
                idx_i, idx_j = indices[i], indices[j]
                if y[idx_i] != y[idx_j]:
                    found_pair = True
                    break

            if not found_pair:
                classes = np.unique(y_node)
                if len(classes) < 2:
                    continue
                self._rng.shuffle(classes)
                for class_a, class_b in itertools.combinations(classes, 2):
                    indices_a = indices[y[indices] == class_a]
                    indices_b = indices[y[indices] == class_b]
                    if len(indices_a) > 0 and len(indices_b) > 0:
                        idx_i = self._rng.choice(indices_a)
                        idx_j = self._rng.choice(indices_b)
                        found_pair = True
                        break
                if not found_pair:
                    continue  # move to next candidate

            # Build candidate hyperplane from points a and b.
            a = X[idx_i]
            b = X[idx_j]
            alpha = self._rng.uniform()  # random weight in [0, 1]
            q = a + alpha * (b - a)
            normal = b - a

            norm_sq = np.dot(normal, normal)
            if norm_sq < 1e-12:
                continue  # skip degenerate candidate
            offset = -np.dot(normal, q)
            norm_val = np.sqrt(norm_sq)

            # Compute decision values and distances for all samples in the node.
            decisions = X[indices].dot(normal) + offset
            distances = np.abs(decisions) / norm_val

            # Split the node.
            neg_mask = decisions < 0
            pos_mask = ~neg_mask  # decisions >= 0
            if not np.any(neg_mask) or not np.any(pos_mask):
                continue

            # Determine the majority label in each half.
            labels_neg = y[indices][neg_mask]
            labels_pos = y[indices][pos_mask]
            majority_neg = Counter(labels_neg).most_common(1)[0][0]
            majority_pos = Counter(labels_pos).most_common(1)[0][0]

            gamma = self.gamma if hasattr(self, "gamma") else 10.0  # default penalty factor

            margins_neg = distances[neg_mask] * np.where(labels_neg == majority_neg, 1, -gamma)
            margins_pos = distances[pos_mask] * np.where(labels_pos == majority_pos, 1, -gamma)
            candidate_margins = np.concatenate([margins_neg, margins_pos])
            candidate_value = np.min(candidate_margins)

            # Map the candidate pair indices (idx_i, idx_j) from the bootstrap sample to global indices.
            global_idx_i = self._global_indices[idx_i]
            global_idx_j = self._global_indices[idx_j]
            candidate_pair = (min(global_idx_i, global_idx_j), max(global_idx_i, global_idx_j))

            candidates.append((normal, offset, majority_neg, majority_pos, candidate_pair))
            candidate_values.append(candidate_value)

        if len(candidates) == 0:
            return self._make_leaf_node(y_node)

        chosen_idx = np.argmax(candidate_values)
        normal, offset, majority_neg, majority_pos, chosen_pair = candidates[chosen_idx]

        if chosen_pair in self._pair_counts:
            self._pair_counts[chosen_pair] += 1
        else:
            self._pair_counts[chosen_pair] = 1

        decisions = X[indices].dot(normal) + offset
        left_mask = decisions < 0
        right_mask = ~left_mask

        if not np.any(left_mask) or not np.any(right_mask):
            return self._make_leaf_node(y_node)

        left_indices = indices[left_mask]
        right_indices = indices[right_mask]

        left_child = self._build_tree(X, y, left_indices, depth + 1)
        right_child = self._build_tree(X, y, right_indices, depth + 1)

        return self._make_internal_node(normal, offset, left_child, right_child)



    def _make_leaf_node(self, y_node):
        """Create a LeafNode with a new permanent node_id."""
        majority_class = self._majority_class(y_node)
        node_id = self._node_id_counter
        self._node_id_counter += 1
        return LeafNode(majority_class, node_id=node_id)

    def _make_internal_node(self, normal, offset, left_child, right_child):
        """Create an InternalNode with a new permanent node_id."""
        node_id = self._node_id_counter
        self._node_id_counter += 1
        return InternalNode(normal, offset, left_child, right_child, node_id=node_id)

    def _gini_impurity(self, y):
        counter = Counter(y)
        total = len(y)
        if total == 0:
            return 0.0
        gini = 1.0 - sum((count / total) ** 2 for count in counter.values())
        return gini

    def predict(self, X):
        check_is_fitted(self, 'root_')
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Number of features of the model must match the input. "
                f"Model n_features_in_ = {self.n_features_in_}, but got {X.shape[1]}."
            )

        return np.array([self._predict_single(x, self.root_) for x in X])

    def _predict_single(self, x, node):
        if isinstance(node, LeafNode):
            return node.predicted_class
        decision = np.dot(node.normal, x) + node.offset
        if decision < 0:
            return self._predict_single(x, node.left_child)
        else:
            return self._predict_single(x, node.right_child)

    def _majority_class(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def to_array(self):
        """
        Convert the entire tree into BFS arrays for normals, offsets,
        left/right children, etc., including each node's permanent ID.
        """
        node_records = []

        def traverse(node):
            node_id_local = len(node_records)  # BFS index
            node_records.append({
                'normal': None,
                'offset': 0.0,
                'left': -1,
                'right': -1,
                'is_leaf': False,
                'leaf_val': -1,
                'node_id': node.node_id  # store the permanent ID
            })

            if isinstance(node, InternalNode):
                # fill in normal, offset, children BFS
                node_records[node_id_local]['normal'] = node.normal
                node_records[node_id_local]['offset'] = node.offset
                node_records[node_id_local]['is_leaf'] = False

                left_id = traverse(node.left_child)
                right_id = traverse(node.right_child)
                node_records[node_id_local]['left'] = left_id
                node_records[node_id_local]['right'] = right_id
            else:
                # LeafNode
                node_records[node_id_local]['is_leaf'] = True
                node_records[node_id_local]['leaf_val'] = node.predicted_class
                node_records[node_id_local]['normal'] = np.zeros(self.n_features_in_)
                node_records[node_id_local]['offset'] = 0.0

            return node_id_local

        traverse(self.root_)

        # Build arrays
        normals = np.array([rec['normal'] for rec in node_records], dtype=np.float64)
        offsets = np.array([rec['offset'] for rec in node_records], dtype=np.float64)
        left_children = np.array([rec['left'] for rec in node_records], dtype=np.int32)
        right_children = np.array([rec['right'] for rec in node_records], dtype=np.int32)
        is_leaf = np.array([rec['is_leaf'] for rec in node_records], dtype=bool)
        leaf_values = np.array([rec['leaf_val'] for rec in node_records], dtype=np.int32)
        node_ids = np.array([rec['node_id'] for rec in node_records], dtype=np.int32)

        return normals, offsets, left_children, right_children, leaf_values, is_leaf, node_ids

    # ----------------- New Methods for Pruning -----------------
    def prune(self, X, y):
        """
        Prune the decision tree using a simple post‑order strategy.
        For each internal node, if replacing it with a leaf (using the majority class)
        does not increase the misclassification error on (X, y), then prune it.
        
        This method uses a single bottom‑up traversal of the tree.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data that reached the tree.
        
        y : array-like of shape (n_samples,)
            The corresponding labels.
        """
        # Recursive helper: returns a tuple (node, error, n)
        # where:
        #   - node is the (possibly pruned) subtree
        #   - error is the number of misclassified examples in that subtree
        #   - n is the total number of examples that reached the subtree
        def prune_node(node, X_node, y_node):
            # If the node is a leaf, compute its error.
            if isinstance(node, LeafNode):
                error = np.sum(y_node != node.predicted_class)
                return node, error, len(y_node)
            
            # Otherwise, node is an internal node.
            # Compute the decision values for X_node.
            decisions = X_node.dot(node.normal) + node.offset
            left_mask = decisions < 0
            right_mask = ~left_mask
            
            X_left, y_left = X_node[left_mask], y_node[left_mask]
            X_right, y_right = X_node[right_mask], y_node[right_mask]
            
            # Recursively prune left and right children.
            pruned_left, error_left, count_left = prune_node(node.left_child, X_left, y_left)
            pruned_right, error_right, count_right = prune_node(node.right_child, X_right, y_right)
            
            subtree_error = error_left + error_right
            total_count = count_left + count_right
            
            # Compute the majority class at this node.
            majority_class = self._majority_class(y_node)
            # Error if this node were replaced by a leaf.
            pruned_error = np.sum(y_node != majority_class)
            
            # If pruning does not worsen error, replace the node with a leaf.
            if pruned_error <= subtree_error:
                new_leaf = LeafNode(majority_class, node_id=node.node_id)
                return new_leaf, pruned_error, total_count
            else:
                # Otherwise, update the node's children to the pruned versions.
                node.left_child = pruned_left
                node.right_child = pruned_right
                return node, subtree_error, total_count

        # Start the pruning process from the root using the full training data.
        self.root_, total_error, total_count = prune_node(self.root_, X, y)


# ----------------- Numba-Compiled Prediction Function -----------------

@njit(parallel=True)
def predict_tree_compiled_parallel(
    X, normals, offsets, left_children, right_children, leaf_values, is_leaf
):
    """
    Predict class labels for samples X using a compiled tree in parallel.
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    y_pred = np.empty(n_samples, dtype=leaf_values.dtype)

    for i in prange(n_samples):
        node = 0  # Start at root index in BFS arrays
        while True:
            if is_leaf[node]:
                y_pred[i] = leaf_values[node]
                break
            decision = 0.0
            for f in range(n_features):
                decision += X[i, f] * normals[node, f]
            decision += offsets[node]
            if decision < 0:
                node = left_children[node]
            else:
                node = right_children[node]

    return y_pred

# ----------------- Helper Function Definition -----------------

def _fit_single_tree_classifier(
    X, y, max_depth, min_samples_split,
    bootstrap, max_attempts_random_pair,
    seed, min_impurity, num_candidates, gamma
):
    rng = np.random.default_rng(seed=seed)
    n_samples = X.shape[0]
    if bootstrap:
        # Sample rows with replacement and record the original (global) indices.
        global_indices = rng.integers(0, n_samples, size=n_samples)
        X_bootstrap = X[global_indices]
        y_bootstrap = y[global_indices]
    else:
        X_bootstrap = X
        y_bootstrap = y
        global_indices = np.arange(n_samples)

    tree = RandomHyperplaneTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=seed,
        max_attempts_random_pair=max_attempts_random_pair,
        min_impurity=min_impurity,
        num_candidates=num_candidates,
        gamma=gamma
    )
    # **Set the global indices BEFORE fitting the tree**
    tree._global_indices = global_indices
    tree.fit(X_bootstrap, y_bootstrap)
    return tree

# ----------------- RandomHyperplaneForestClassifier Definition -----------------
class RandomHyperplaneForestClassifier(BaseEstimator, ClassifierMixin):
    """
    A Random Hyperplane Forest classifier that trains multiple
    Random Hyperplane Trees and aggregates via majority voting.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        min_impurity=0.1,
        num_candidates=3,
        gamma=10,
        bootstrap=True,
        max_attempts_random_pair=100,
        n_jobs=-1,
        random_state=None,
        use_compiled_predict=True,  # also used for selecting transform method
        use_prune=True
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.num_candidates = num_candidates
        self.gamma = gamma  # penalty factor for misclassified points
        self.bootstrap = bootstrap
        self.max_attempts_random_pair = max_attempts_random_pair
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.min_impurity = min_impurity
        self.use_compiled_predict = use_compiled_predict
        self.use_prune = use_prune

    def fit(self, X, y):
        self.le = LabelEncoder()
        y_encoded = self.le.fit_transform(y)
        self.classes_ = self.le.classes_

        X, y_encoded = check_X_y(X, y_encoded)
        self.n_features_in_ = X.shape[1]
        self._rng = check_random_state(self.random_state)
        self.y_ = y_encoded

        # Random seeds for reproducibility
        if self.random_state is not None:
            seeds = self._rng.randint(np.iinfo(np.int32).max, size=self.n_estimators)
        else:
            seeds = [None] * self.n_estimators

        # Build trees in parallel
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single_tree_classifier)(
                X, y_encoded,
                self.max_depth,
                self.min_samples_split,
                self.bootstrap,
                self.max_attempts_random_pair,
                seed,
                self.min_impurity,
                self.num_candidates,
                self.gamma
            )
            for seed in seeds
        )

        if self.use_prune:
            for tree in self.estimators_:
                tree.prune(X, y)

        # If compiled predict is enabled, build the array representation for each tree
        if self.use_compiled_predict:
            self.compiled_trees_ = []
            for tree in self.estimators_:
                arrays = tree.to_array()  # (normals, offsets, left, right, leaf_vals, is_leaf, node_ids)
                self.compiled_trees_.append(arrays)
        else:
            self.compiled_trees_ = None

        # ---- Aggregate the _pair_counts from all trees into a global dictionary ----
        self._global_pair_counts = {}
        for tree in self.estimators_:
            if hasattr(tree, '_pair_counts'):
                for candidate_pair, count in tree._pair_counts.items():
                    self._global_pair_counts[candidate_pair] = self._global_pair_counts.get(candidate_pair, 0) + count

        return self

    def predict(self, X):
        check_is_fitted(self, 'estimators_')
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Number of features must match. Model n_features_in_ = {self.n_features_in_}, got {X.shape[1]}."
            )

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        votes = np.zeros((n_samples, n_classes), dtype=np.int32)

        if self.use_compiled_predict and self.compiled_trees_ is not None:
            # Use compiled predictions for each tree
            for (normals, offsets, left_children, right_children,
                 leaf_values, is_leaf, node_ids) in self.compiled_trees_:
                preds = predict_tree_compiled_parallel(
                    X, normals, offsets, left_children, right_children, leaf_values, is_leaf
                )
                votes[np.arange(n_samples), preds] += 1
        else:
            # Standard Python predict
            def tree_vote(tree):
                return tree.predict(X)

            all_pred_indices = Parallel(n_jobs=self.n_jobs)(
                delayed(tree_vote)(tree) for tree in self.estimators_
            )
            all_pred_indices = np.array(all_pred_indices)  # shape: (n_estimators, n_samples)

            for tree_preds in all_pred_indices:
                votes[np.arange(n_samples), tree_preds] += 1

        y_pred_indices = np.argmax(votes, axis=1)
        return self.le.inverse_transform(y_pred_indices)

    def predict_proba(self, X):
        check_is_fitted(self, 'estimators_')
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Number of features must match. Model n_features_in_ = {self.n_features_in_}, got {X.shape[1]}."
            )

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes), dtype=float)

        if self.use_compiled_predict and self.compiled_trees_ is not None:
            for (normals, offsets, left_children, right_children,
                 leaf_values, is_leaf, node_ids) in self.compiled_trees_:
                preds = predict_tree_compiled_parallel(
                    X, normals, offsets, left_children, right_children, leaf_values, is_leaf
                )
                proba[np.arange(n_samples), preds] += 1
        else:
            # Standard Python predict
            def tree_vote(tree):
                return tree.predict(X)

            all_pred_indices = Parallel(n_jobs=self.n_jobs)(
                delayed(tree_vote)(tree) for tree in self.estimators_
            )
            all_pred_indices = np.array(all_pred_indices)

            for tree_preds in all_pred_indices:
                proba[np.arange(n_samples), tree_preds] += 1

        proba /= self.n_estimators
        return proba

    @property
    def global_pair_counts(self):
        """
        Return the aggregated pair counts from all trees in the forest.
        This dictionary maps candidate pairs (sorted tuples of instance indices)
        to the total number of times they were used across the entire forest.
        """
        return self._global_pair_counts if hasattr(self, '_global_pair_counts') else {}

    def transform(self, X, y=None):
        """
        Two versions of transform:
          - If use_compiled_predict == True, use array-based BFS transform
          - Else, use an object-based transform with node_id references
        """
        check_is_fitted(self, 'estimators_')
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Number of features must match. Model n_features_in_ = {self.n_features_in_}, got {X.shape[1]}."
            )

        if self.use_compiled_predict and self.compiled_trees_ is not None:
            return self._transform_array_based(X)
        else:
            return self._transform_object_based(X)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X,y)
    
    # ----------------------------------------------------------------
    # 1) Array-based transform (BFS indices)
    # ----------------------------------------------------------------
    def _transform_array_based(self, X):
        """
        Uses each tree's BFS arrays to track which nodes each sample visits.
        """
        n_samples = X.shape[0]
        # Count total nodes across all trees
        total_nodes = 0
        for arrays in self.compiled_trees_:
            # arrays = (normals, offsets, left_children, right_children, leaf_values, is_leaf, node_ids)
            total_nodes += len(arrays[0])  # length = # of nodes

        X_trans = np.zeros((n_samples, total_nodes), dtype=int)

        offset = 0
        for (normals, offsets, left_children, right_children, leaf_values, is_leaf, node_ids) in self.compiled_trees_:
            n_nodes = len(is_leaf)
            # Build partial indicator for this tree
            partial = np.zeros((n_samples, n_nodes), dtype=int)

            for i in range(n_samples):
                idx = 0  # BFS index of the root
                while True:
                    partial[i, idx] = 1
                    if is_leaf[idx]:
                        break
                    decision = np.dot(X[i], normals[idx]) + offsets[idx]
                    if decision < 0:
                        idx = left_children[idx]
                    else:
                        idx = right_children[idx]

            # Place partial into X_trans
            X_trans[:, offset:offset+n_nodes] = partial
            offset += n_nodes

        return X_trans

    # ----------------------------------------------------------------
    # 2) Object-based transform (permanent node_id references)
    # ----------------------------------------------------------------
    def _transform_object_based(self, X):
        """
        Traverses each tree via its root_ node references. We rely on
        node.node_id (assigned permanently in the tree) to place 1's in
        the correct columns.

        We'll build a global ID map:
            global_node_id = consecutive integer
        so that transform produces a shape of (n_samples, total_nodes).
        """
        # Build a global map from (estimator_index, local node_id) to "global column index"
        if not hasattr(self, '_global_offset_map'):
            self._build_object_based_map()

        n_samples = X.shape[0]
        total_nodes = self._total_nodes_  # from _build_object_based_map
        X_trans = np.zeros((n_samples, total_nodes), dtype=int)

        # For each tree, we traverse each sample:
        for tree_idx, tree in enumerate(self.estimators_):
            # We'll record visited columns in a partial (n_samples, #tree_nodes).
            # But simpler to fill directly into X_trans if we know each local node_id -> global column index.
            for i, x in enumerate(X):
                node = tree.root_
                while True:
                    # Look up the global column index
                    global_col = self._global_offset_map_[(tree_idx, node.node_id)]
                    X_trans[i, global_col] = 1

                    if isinstance(node, LeafNode):
                        break
                    decision = np.dot(node.normal, x) + node.offset
                    if decision < 0:
                        node = node.left_child
                    else:
                        node = node.right_child
        return X_trans

    def _build_object_based_map(self):
        """
        BFS each tree to figure out how many total nodes across the forest,
        and map (tree_idx, node_id) -> global column index.
        """
        self._global_offset_map_ = {}
        current_global_id = 0

        for tree_idx, tree in enumerate(self.estimators_):
            # BFS the tree
            stack = [tree.root_]
            while stack:
                node = stack.pop()
                # Assign a global ID if not already assigned
                if (tree_idx, node.node_id) not in self._global_offset_map_:
                    self._global_offset_map_[(tree_idx, node.node_id)] = current_global_id
                    current_global_id += 1

                if isinstance(node, InternalNode):
                    stack.append(node.left_child)
                    stack.append(node.right_child)

        self._total_nodes_ = current_global_id

    def prune(self, X, y):
        """
        Prune every tree in the forest using the provided dataset (X,y). For each tree,
        the method prunes a fraction alpha of its internal nodes using the new bottom-up
        strategy.
        
        Parameters:
            X : array-like of shape (n_samples, n_features)
            y : array-like of shape (n_samples,)
        """
        X, y = check_X_y(X, y)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Number of features must match. Model n_features_in_ = {self.n_features_in_}, got {X.shape[1]}."
            )
        for tree in self.estimators_:
            tree.prune(X, y)
        if self.use_compiled_predict:
            self.compiled_trees_ = [tree.to_array() for tree in self.estimators_]

# ----------------- Hyperparameter Tuning -----------------
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

def tune_random_hyperplane_forest(
    X,
    y,
    cv=5,
    n_iter=50,
    scoring='accuracy',
    random_state=None,
    n_jobs=-1,
    verbose=1
):
    """
    Perform hyperparameter optimization for RandomHyperplaneForestClassifier using RandomizedSearchCV.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training input samples.
    y : array-like of shape (n_samples,)
        Target labels.
    cv : int, default=5
        Number of cross-validation folds.
    n_iter : int, default=50
        Number of parameter settings sampled in the random search.
    scoring : str or callable, default='accuracy'
        A string or a scorer callable object/function with signature scorer(estimator, X, y).
    random_state : int, default=None
        Seed for reproducibility.
    n_jobs : int, default=-1
        Number of jobs to run in parallel. -1 means using all processors.
    verbose : int, default=1
        Controls the verbosity: the higher, the more messages.

    Returns
    -------
    best_estimator : RandomHyperplaneForestClassifier
        The estimator that was chosen by the search (i.e. the estimator with the best found parameters).
    best_params : dict
        Parameter setting that gave the best results on the hold out data.
    best_score : float
        Mean cross-validated score of the best_estimator.
    search_results : RandomizedSearchCV
        The fitted RandomizedSearchCV object containing all results.
    """

    # Define the hyperparameter distributions to sample from.
    # You may adjust or remove parameters (like 'num_candidates' and 'gamma') if desired.
    param_distributions = {
        'n_estimators': randint(100, 501),               # number of trees in the forest (100 to 500)
        'max_depth': randint(3, 51),                     # maximum depth of each tree (3 to 50)
        'min_samples_split': randint(2, 21),             # minimum samples required to split an internal node (2 to 20)
        'bootstrap': [True, False],                      # whether bootstrap samples are used when building trees
        'max_attempts_random_pair': randint(100, 201),   # attempts to find different label pairs (100 to 200)
        'min_impurity': uniform(0.01, 0.19),             # minimum impurity required to perform a split (0.01 to 0.20)
        'num_candidates': randint(2, 10),                # number of candidate hyperplanes to consider (e.g., 2 to 10)
        'gamma': uniform(5, 15)                          # penalty factor for misclassified points (e.g., from 5 to 20)
    }

    # Initialize the classifier with the provided random state and number of jobs.
    clf = RandomHyperplaneForestClassifier(
        random_state=random_state,
        n_jobs=n_jobs
    )

    # Initialize RandomizedSearchCV using the classifier and the hyperparameter distributions.
    random_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )

    # Execute the random search on the provided data.
    random_search.fit(X, y)

    # Extract the best estimator, its parameters, and the cross-validated score.
    best_estimator = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    return best_estimator, best_params, best_score, random_search

# ----------------- DeepRandomHyperplaneForestClassifier -----------------
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD, KernelPCA

def DeepRandomHyperplaneForestClassifier(
    n_layers=1,
    n_estimators=100,
    max_depth=5,
    min_samples_split=3,
    min_impurity=0.1,
    bootstrap=True,
    max_attempts_random_pair=100,
    n_jobs=-1,
    random_state=None,
    use_compiled_predict=True,
    n_components=20,
    embedding_method='svd',  # New parameter to choose embedding method
    kpca_gamma=1e-2          # Optional parameter for KernelPCA's gamma
):
    """
    Construct a "deep" ensemble of Random Hyperplane Forests in a pipeline.

    This function creates a pipeline consisting of:
      - Repeated blocks (layers) of:
         (a) A RandomHyperplaneForestClassifier (as a feature transformer),
         (b) A TruncatedSVD or KernelPCA (as a dimensionality-reduction step).
      - A final RandomHyperplaneForestClassifier serving as the classifier.

    Parameters
    ----------
    n_layers : int, default=1
        Number of (Forest -> Embedding) layers to apply before the final classification.
    
    n_estimators : int, default=100
        Number of trees in each RandomHyperplaneForestClassifier.

    max_depth : int, default=5
        Maximum depth of each tree in the forest.

    min_samples_split : int, default=3
        Minimum number of samples required to split an internal node in the forest.

    min_impurity : float, default=0.1
        Minimum Gini impurity required to perform a split in each tree.

    bootstrap : bool, default=True
        Whether to use bootstrap samples (sampling with replacement) when building trees.

    max_attempts_random_pair : int, default=100
        Maximum number of attempts to find a pair of samples with distinct labels to form a split hyperplane.

    n_jobs : int, default=-1
        Number of jobs to run in parallel for both fitting and prediction.
        If -1, use all available CPU cores.

    random_state : int or None, default=None
        Controls the randomness of the algorithm for reproducible splits. 
        If None, randomness is not fixed.

    use_compiled_predict : bool, default=False
        Whether to use a Numba-compiled predict function in the forest 
        (and, as implemented, also affects which transform path is used).

    n_components : int, default=100
        Number of components for the embedding step (TruncatedSVD or KernelPCA) in each layer.

    embedding_method : {'svd', 'kpca'}, default='svd'
        The dimensionality reduction technique to use in each layer.
        - 'svd': TruncatedSVD
        - 'kpca': KernelPCA with RBF (Gaussian) kernel

    kpca_gamma : float, default=1.0
        Kernel coefficient for KernelPCA when `embedding_method` is 'kpca'.
        Ignored if `embedding_method` is not 'kpca'.

    Returns
    -------
    estimator : Pipeline
        A scikit-learn Pipeline object consisting of:
          [ (transformer_1), (embedder_1),
            (transformer_2), (embedder_2),
            ...,
            (transformer_n_layers), (embedder_n_layers),
            (classifier) ]

    Examples
    --------
    >>> # Example usage with TruncatedSVD
    >>> model = DeepRandomHyperplaneForestClassifier(
    ...     n_layers=2,
    ...     n_estimators=50,
    ...     max_depth=7,
    ...     n_components=20,
    ...     embedding_method='svd'
    ... )
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)

    >>> # Example usage with KernelPCA
    >>> model = DeepRandomHyperplaneForestClassifier(
    ...     n_layers=3,
    ...     n_estimators=100,
    ...     max_depth=10,
    ...     n_components=30,
    ...     embedding_method='kpca',
    ...     kpca_gamma=0.5
    ... )
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    """
    layers = []
    rng = np.random.RandomState(random_state) if random_state is not None else None

    # Validate embedding_method
    if embedding_method not in {'svd', 'kpca'}:
        raise ValueError("embedding_method must be either 'svd' or 'kpca'.")

    for i in range(n_layers):
        # Create a NEW forest instance for this layer:
        forest_transformer = RandomHyperplaneForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_impurity=min_impurity,
            bootstrap=bootstrap,
            max_attempts_random_pair=max_attempts_random_pair,
            n_jobs=n_jobs,
            random_state=(rng.randint(0, 2**31-1) if rng else None),
            use_compiled_predict=use_compiled_predict
        )
        
        # Choose the embedding method
        if embedding_method == 'svd':
            embedder = TruncatedSVD(n_components=n_components, random_state=random_state)
            embedder_name = f"svd_{i+1}"
        else:  # embedding_method == 'kpca'
            embedder = KernelPCA(
                n_components=n_components,
                kernel='rbf',
                gamma=kpca_gamma,
                random_state=random_state
            )
            embedder_name = f"kpca_{i+1}"
        
        layers.append((f"transformer_{i+1}", forest_transformer))
        layers.append((embedder_name, embedder))

    # Finally, a separate forest as the classifier
    forest_classifier = RandomHyperplaneForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_impurity=min_impurity,
        bootstrap=bootstrap,
        max_attempts_random_pair=max_attempts_random_pair,
        n_jobs=n_jobs,
        random_state=(rng.randint(0, 2**31-1) if rng else None),
        use_compiled_predict=use_compiled_predict
    )

    layers.append(("classifier", forest_classifier))

    pipeline = Pipeline(layers)
    return pipeline
