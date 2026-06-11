import numpy as np
import itertools
from collections import Counter
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.preprocessing import KBinsDiscretizer
from numba import njit, prange
import os
os.environ["KMP_WARNINGS"] = "0"

# ================= Helper Function: Train a Linear Model =================
def _train_linear_model(X, y):
    """
    Train a simple linear model y = X.dot(coef) + intercept via least squares.
    If the design matrix is singular, fall back to a constant predictor.
    """
    n, d = X.shape
    # Add a column for the intercept.
    X_aug = np.hstack([X, np.ones((n, 1))])
    try:
        params, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        coef = params[:-1]
        intercept = params[-1]
    except np.linalg.LinAlgError:
        coef = np.zeros(d)
        intercept = np.mean(y)
    return coef, intercept

# ==================== Numba-Compiled Prediction Function ====================
@njit(parallel=True)
def predict_tree_compiled_parallel_regressor(X, normals, offsets, left_children, right_children, 
                                             leaf_coefs, leaf_intercepts, is_leaf, node_ids):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    y_pred = np.empty(n_samples, dtype=np.float64)
    for i in prange(n_samples):
        node = 0  # start at root (BFS index 0)
        while True:
            if is_leaf[node]:
                pred = 0.0
                for f in range(n_features):
                    pred += X[i, f] * leaf_coefs[node, f]
                pred += leaf_intercepts[node]
                y_pred[i] = pred
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

# =========================== Node Classes ===========================
class LeafNodeRegressor:
    """
    A leaf node that stores a linear predictor (its coefficients and intercept)
    along with a permanent node id.
    """
    __slots__ = ['coef', 'intercept', 'node_id']
    
    def __init__(self, coef, intercept, node_id=None):
        self.coef = coef
        self.intercept = intercept
        self.node_id = node_id

class InternalNode:
    """
    An internal node that stores a hyperplane (normal, offset) and pointers to left/right children,
    as well as a permanent node id.
    """
    __slots__ = ['normal', 'offset', 'left_child', 'right_child', 'node_id']
    
    def __init__(self, normal, offset, left_child, right_child, node_id=None):
        self.normal = normal
        self.offset = offset
        self.left_child = left_child
        self.right_child = right_child
        self.node_id = node_id

# =================== RandomHyperplaneTreeRegressor ===================
class RandomHyperplaneTreeRegressor(BaseEstimator, RegressorMixin):
    """
    A random hyperplane tree regressor that builds its splits using a discretized version
    of the continuous target (i.e. as if it were a multi‑class classification problem).
    At each leaf, a linear predictor is trained (via least squares) on the original continuous
    targets that fall into that region.
    
    Parameters
    ----------
    max_depth : int, default=5
        Maximum tree depth.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    random_state : int, default=None
        Controls the randomness of the splits.
    max_attempts_random_pair : int, default=10
        Maximum attempts to find two points with different discretized targets.
    num_candidates : int, default=3
        Number of candidate hyperplanes to consider at each split.
    gamma : float, default=10.0
        Penalty factor used in scoring candidate splits.
    k_bins : int, default=5
        Number of bins used to discretize the continuous target.
    """
    def __init__(self, max_depth=5, min_samples_split=2, random_state=None,
                 max_attempts_random_pair=10, num_candidates=3, gamma=10.0, k_bins=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.max_attempts_random_pair = max_attempts_random_pair
        self.num_candidates = num_candidates
        self.gamma = gamma
        self.k_bins = k_bins

    def fit(self, X, y):
        """
        Fit the tree by first discretizing the continuous target and then building
        the tree using the multi‑class splitting strategy. In each leaf, a linear model
        is trained on the original continuous targets.
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        # Discretize the continuous target.
        self.discretizer_ = KBinsDiscretizer(n_bins=self.k_bins, encode='ordinal', strategy='uniform')
        y_disc = self.discretizer_.fit_transform(y.reshape(-1, 1)).astype(int).ravel()
        self.y_cont_ = y  # original continuous targets
        self.y_disc_ = y_disc
        self._node_id_counter = 0
        self._rng = np.random.default_rng(seed=self.random_state)
        indices = np.arange(len(X))
        self.root_ = self._build_tree(X, y, y_disc, indices, depth=0)
        return self

    def _build_tree(self, X, y, y_disc, indices, depth):
        y_node = y[indices]
        y_disc_node = y_disc[indices]
        # Terminal condition: if all discretized targets are the same,
        # or maximum depth is reached, or too few samples remain.
        if (len(np.unique(y_disc_node)) == 1 or
            depth >= self.max_depth or
            len(indices) < self.min_samples_split):
            return self._make_leaf_node(X[indices], y_node)
        
        # Candidate hyperplane selection.
        candidates = []       # each candidate: (normal, offset)
        candidate_values = [] # score for each candidate
        
        for candidate_idx in range(self.num_candidates):
            found_pair = False
            for attempt in range(self.max_attempts_random_pair):
                i, j = self._rng.choice(len(indices), size=2, replace=False)
                idx_i, idx_j = indices[i], indices[j]
                if y_disc[idx_i] != y_disc[idx_j]:
                    found_pair = True
                    break
            if not found_pair:
                # Try iterating over class combinations.
                classes = np.unique(y_disc_node)
                if len(classes) < 2:
                    continue
                for class_a, class_b in itertools.combinations(classes, 2):
                    indices_a = indices[y_disc[indices] == class_a]
                    indices_b = indices[y_disc[indices] == class_b]
                    if len(indices_a) > 0 and len(indices_b) > 0:
                        idx_i = self._rng.choice(indices_a)
                        idx_j = self._rng.choice(indices_b)
                        found_pair = True
                        break
                if not found_pair:
                    continue  # skip this candidate
            
            # Build candidate hyperplane from points a and b.
            a = X[idx_i]
            b = X[idx_j]
            alpha = self._rng.uniform()
            q = a + alpha * (b - a)
            normal = b - a
            norm_sq = np.dot(normal, normal)
            if norm_sq < 1e-12:
                continue
            offset = -np.dot(normal, q)
            norm_val = np.sqrt(norm_sq)
            decisions = X[indices].dot(normal) + offset
            distances = np.abs(decisions) / norm_val
            neg_mask = decisions < 0
            pos_mask = ~neg_mask
            if not np.any(neg_mask) or not np.any(pos_mask):
                continue
            labels_neg = y_disc[indices][neg_mask]
            labels_pos = y_disc[indices][pos_mask]
            majority_neg = Counter(labels_neg).most_common(1)[0][0]
            majority_pos = Counter(labels_pos).most_common(1)[0][0]
            margins_neg = distances[neg_mask] * np.where(labels_neg == majority_neg, 1, -self.gamma)
            margins_pos = distances[pos_mask] * np.where(labels_pos == majority_pos, 1, -self.gamma)
            candidate_margins = np.concatenate([margins_neg, margins_pos])
            candidate_value = np.min(candidate_margins)
            candidates.append((normal, offset))
            candidate_values.append(candidate_value)
        
        if len(candidates) == 0:
            return self._make_leaf_node(X[indices], y_node)
        
        chosen_idx = np.argmax(candidate_values)
        normal, offset = candidates[chosen_idx]
        decisions = X[indices].dot(normal) + offset
        left_mask = decisions < 0
        right_mask = ~left_mask
        if not np.any(left_mask) or not np.any(right_mask):
            return self._make_leaf_node(X[indices], y_node)
        left_indices = indices[left_mask]
        right_indices = indices[right_mask]
        left_child = self._build_tree(X, y, y_disc, left_indices, depth + 1)
        right_child = self._build_tree(X, y, y_disc, right_indices, depth + 1)
        return self._make_internal_node(normal, offset, left_child, right_child)
    
    def _make_leaf_node(self, X_node, y_node):
        """Create a leaf node by training a linear model on X_node and y_node."""
        coef, intercept = _train_linear_model(X_node, y_node)
        node_id = self._node_id_counter
        self._node_id_counter += 1
        return LeafNodeRegressor(coef, intercept, node_id=node_id)
    
    def _make_internal_node(self, normal, offset, left_child, right_child):
        """Create an internal node with a unique node id."""
        node_id = self._node_id_counter
        self._node_id_counter += 1
        return InternalNode(normal, offset, left_child, right_child, node_id=node_id)
    
    def to_array(self):
        """
        Convert the tree into a set of arrays (in BFS order) for use with the compiled predict function.
        The arrays returned are:
          - normals:      (n_nodes, n_features) – hyperplane normals (0 for leaves)
          - offsets:      (n_nodes,) – hyperplane offsets (0 for leaves)
          - left_children:(n_nodes,) – index of left child (-1 for leaves)
          - right_children:(n_nodes,) – index of right child (-1 for leaves)
          - leaf_coefs:   (n_nodes, n_features) – linear model coefficients (0 for internal nodes)
          - leaf_intercepts: (n_nodes,) – linear model intercepts (0 for internal nodes)
          - is_leaf:      (n_nodes,) – boolean flag (True for leaves)
          - node_ids:     (n_nodes,) – permanent node ids
        """
        node_records = []
        def traverse(node):
            pos = len(node_records)
            record = {}
            record['node_id'] = node.node_id
            if isinstance(node, InternalNode):
                record['is_leaf'] = False
                record['normal'] = node.normal
                record['offset'] = node.offset
                record['left'] = -1
                record['right'] = -1
                record['leaf_coef'] = np.zeros(self.n_features_in_)
                record['leaf_intercept'] = 0.0
                node_records.append(record)
                left_id = traverse(node.left_child)
                right_id = traverse(node.right_child)
                node_records[pos]['left'] = left_id
                node_records[pos]['right'] = right_id
                return pos
            else:
                record['is_leaf'] = True
                record['normal'] = np.zeros(self.n_features_in_)
                record['offset'] = 0.0
                record['left'] = -1
                record['right'] = -1
                record['leaf_coef'] = node.coef
                record['leaf_intercept'] = node.intercept
                node_records.append(record)
                return pos
        traverse(self.root_)
        n_nodes = len(node_records)
        normals = np.zeros((n_nodes, self.n_features_in_), dtype=np.float64)
        offsets = np.zeros(n_nodes, dtype=np.float64)
        left_children = -np.ones(n_nodes, dtype=np.int32)
        right_children = -np.ones(n_nodes, dtype=np.int32)
        is_leaf = np.empty(n_nodes, dtype=bool)
        leaf_coefs = np.zeros((n_nodes, self.n_features_in_), dtype=np.float64)
        leaf_intercepts = np.zeros(n_nodes, dtype=np.float64)
        node_ids = np.zeros(n_nodes, dtype=np.int32)
        for i, rec in enumerate(node_records):
            normals[i, :] = rec['normal']
            offsets[i] = rec['offset']
            left_children[i] = rec['left']
            right_children[i] = rec['right']
            is_leaf[i] = rec['is_leaf']
            leaf_coefs[i, :] = rec['leaf_coef']
            leaf_intercepts[i] = rec['leaf_intercept']
            node_ids[i] = rec['node_id']
        return normals, offsets, left_children, right_children, leaf_coefs, leaf_intercepts, is_leaf, node_ids

    def predict(self, X):
        """
        Predict continuous targets for X by traversing the tree and using the
        linear predictor in each leaf.
        """
        check_is_fitted(self, 'root_')
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")
        return np.array([self._predict_single(x, self.root_) for x in X])
    
    def _predict_single(self, x, node):
        if isinstance(node, LeafNodeRegressor):
            return np.dot(x, node.coef) + node.intercept
        decision = np.dot(node.normal, x) + node.offset
        if decision < 0:
            return self._predict_single(x, node.left_child)
        else:
            return self._predict_single(x, node.right_child)

# ================ Helper Function for Building a Single Tree ================
def _fit_single_tree_regressor(X, y, max_depth, min_samples_split, bootstrap,
                               max_attempts_random_pair, num_candidates, gamma, k_bins, seed):
    """
    Build a single RandomHyperplaneTreeRegressor on (possibly bootstrapped) data.
    """
    rng = np.random.default_rng(seed=seed)
    n_samples = X.shape[0]
    if bootstrap:
        indices = rng.integers(0, n_samples, size=n_samples)
        X_bootstrap = X[indices]
        y_bootstrap = y[indices]
    else:
        X_bootstrap = X
        y_bootstrap = y
    tree = RandomHyperplaneTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=seed,
        max_attempts_random_pair=max_attempts_random_pair,
        num_candidates=num_candidates,
        gamma=gamma,
        k_bins=k_bins
    )
    tree.fit(X_bootstrap, y_bootstrap)
    return tree

# ================= RandomHyperplaneForestRegressor =================
class RandomHyperplaneForestRegressor(BaseEstimator, RegressorMixin):
    """
    A forest regressor that builds multiple RandomHyperplaneTreeRegressor trees
    (each built by discretizing the continuous target and training linear predictors in the leaves)
    and aggregates their predictions by averaging.
    
    Additionally, if `use_compiled_predict` is True (default), each tree is converted to an
    array-based representation and predictions are computed via a Numba-compiled function.
    
    Parameters
    ----------
    n_estimators : int, default=10
        Number of trees in the forest.
    max_depth : int, default=5
        Maximum depth of each tree.
    min_samples_split : int, default=2
        Minimum samples required to split an internal node.
    bootstrap : bool, default=True
        Whether to use bootstrap samples when building each tree.
    max_attempts_random_pair : int, default=10
        Maximum attempts to find a candidate pair for splitting.
    num_candidates : int, default=3
        Number of candidate hyperplanes to consider per split.
    gamma : float, default=10.0
        Penalty factor used in scoring candidate splits.
    k_bins : int, default=5
        Number of bins used to discretize the continuous target.
    n_jobs : int, default=-1
        Number of parallel jobs.
    random_state : int, default=None
        Seed for reproducibility.
    use_compiled_predict : bool, default=True
        If True, use the array-based (compiled) prediction function.
    """
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, bootstrap=True,
                 max_attempts_random_pair=10, num_candidates=3, gamma=10.0, k_bins=5,
                 n_jobs=-1, random_state=None, use_compiled_predict=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.max_attempts_random_pair = max_attempts_random_pair
        self.num_candidates = num_candidates
        self.gamma = gamma
        self.k_bins = k_bins
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.use_compiled_predict = use_compiled_predict

    def fit(self, X, y):
        """
        Fit the forest by building n_estimators trees on (possibly bootstrapped) data.
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(seed=self.random_state)
        seeds = rng.integers(np.iinfo(np.int32).max, size=self.n_estimators)
        
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single_tree_regressor)(
                X, y,
                self.max_depth,
                self.min_samples_split,
                self.bootstrap,
                self.max_attempts_random_pair,
                self.num_candidates,
                self.gamma,
                self.k_bins,
                seeds[i]
            ) for i in range(self.n_estimators)
        )
        
        if self.use_compiled_predict:
            # Compile each tree into an array representation.
            self.compiled_trees_ = [est.to_array() for est in self.estimators_]
        else:
            self.compiled_trees_ = None
        return self

    def predict(self, X):
        """
        Predict continuous targets for X by averaging the predictions from all trees.
        If use_compiled_predict is True, the compiled (array-based) prediction function is used.
        """
        check_is_fitted(self, 'estimators_')
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")
        if self.use_compiled_predict and self.compiled_trees_ is not None:
            n_samples = X.shape[0]
            all_preds = np.empty((len(self.compiled_trees_), n_samples), dtype=np.float64)
            for i, arrays in enumerate(self.compiled_trees_):
                preds = predict_tree_compiled_parallel_regressor(X, *arrays)
                all_preds[i, :] = preds
            return np.mean(all_preds, axis=0)
        else:
            all_preds = Parallel(n_jobs=self.n_jobs)(
                delayed(est.predict)(X) for est in self.estimators_
            )
            all_preds = np.array(all_preds)
            return np.mean(all_preds, axis=0)
    
    def predict_mean_std(self, X):
        """
        Predict continuous targets for X and return both the mean and standard deviation of
        predictions across the ensemble.
        
        Returns:
            mean_pred : ndarray of shape (n_samples,)
                Mean predictions across trees.
            std_pred : ndarray of shape (n_samples,)
                Standard deviation of predictions across trees.
        """
        check_is_fitted(self, 'estimators_')
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")
        if self.use_compiled_predict and self.compiled_trees_ is not None:
            n_samples = X.shape[0]
            all_preds = np.empty((len(self.compiled_trees_), n_samples), dtype=np.float64)
            for i, arrays in enumerate(self.compiled_trees_):
                preds = predict_tree_compiled_parallel_regressor(X, *arrays)
                all_preds[i, :] = preds
        else:
            all_preds = Parallel(n_jobs=self.n_jobs)(
                delayed(est.predict)(X) for est in self.estimators_
            )
            all_preds = np.array(all_preds)
        mean_pred = np.mean(all_preds, axis=0)
        std_pred = np.std(all_preds, axis=0)
        return mean_pred, std_pred


import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

def tune_random_hyperplane_forest_regressor(
    X,
    y,
    cv=5,
    n_iter=50,
    scoring='neg_mean_squared_error',
    random_state=None,
    n_jobs=-1,
    verbose=1
):
    """
    Perform hyperparameter optimization for RandomHyperplaneForestRegressor using RandomizedSearchCV.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training input samples.
    y : array-like of shape (n_samples,)
        Continuous target values.
    cv : int, default=5
        Number of cross-validation folds.
    n_iter : int, default=50
        Number of parameter settings sampled in the random search.
    scoring : str or callable, default='neg_mean_squared_error'
        Scoring metric to optimize.
    random_state : int, default=None
        Seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs; -1 means use all available cores.
    verbose : int, default=1
        Verbosity level.
    
    Returns
    -------
    best_estimator : RandomHyperplaneForestRegressor
        The best estimator found by the random search.
    best_params : dict
        Parameter setting that gave the best results.
    best_score : float
        Mean cross‑validated score of the best_estimator.
    search_results : RandomizedSearchCV
        The fitted RandomizedSearchCV object containing all results.
    """
    # Define the hyperparameter distributions to sample from.
    param_distributions = {
        'n_estimators': randint(100, 101),         # e.g., 50 to 200 trees
        'max_depth': randint(2, 10),              # e.g., depth between 3 and 50
        'min_samples_split': randint(2, 5),      # e.g., minimum samples per split between 2 and 20
        'bootstrap': [True, False],              # use or not use bootstrap samples
        'max_attempts_random_pair': randint(3, 5),  # e.g., 5 to 20 attempts to find a valid split pair
        'num_candidates': randint(3, 10),         # e.g., try between 2 and 10 candidate splits per node
        'gamma': uniform(2, 100),                  # e.g., penalty factor between 5 and 20
        'k_bins': randint(2, 11)                  # e.g., discretize continuous target into 3 to 10 bins
    }
    
    # Initialize the regressor with the provided random state and compiled prediction flag.
    regressor = RandomHyperplaneForestRegressor(
        random_state=random_state,
        use_compiled_predict=True
    )
    
    # Initialize the RandomizedSearchCV object.
    random_search = RandomizedSearchCV(
        estimator=regressor,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )
    
    # Execute the random search.
    random_search.fit(X, y)
    
    best_estimator = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    return best_estimator, best_params, best_score, random_search
