import numpy as np
import networkx as nx
from sklearn.base import BaseEstimator

from coco_grape.module.quotientgraph.vectorize import QuotientGraphTransformer
from coco_grape.data_processor.generative.decompositional_graph_rewriter import DecompositionalGraphRewriter

from coco_grape.data_processor.generative.feasibility_estimator import (
    FeasibilityEstimatorFeatureCannotExist, FeasibilityEstimator,
)

from coco_grape.module.quotientgraph.operator import (
    neighborhood, compose, cycle, unlabel,
    filter_by_number_of_connected_components, combination, add, tree,
)
# ------------------------------------------------------------------------


class FeasibleGraphInterpolator(BaseEstimator):
    """
    Scikit-learn-style estimator that

    1.  builds & fits a FeasibilityEstimator in ``fit()``;
    2.  generates a feasible “morph” from `graph_src` to `graph_dest`
        in ``interpolate()``;
    3.  returns the graphs sorted by projection onto the straight line
        between *src* and *dest* in QuotientGraph feature space.
    """

    # ----------------------------- init ---------------------------------
    def __init__(
        self,
        *,
        decomposition_function=None,
        nbits: int = 11,
        parallel: bool = True,
        k_samples: int = 16,
        n_iter: int = 20,
        tol: float = 1e-9,
        metric_tol: int = 2, 
    ):
        self.decomposition_function = (
            decomposition_function if decomposition_function is not None
            else add(cycle(), tree())
        )
        self.nbits     = nbits
        self.parallel  = parallel
        self.k_samples = k_samples
        self.n_iter    = n_iter
        self.tol       = tol
        self.metric_tol = metric_tol

        # --- runtime slots
        self._feasibility_df        = None
        self._feasibility_estimator = None
        self._is_fitted             = False

    # --------------------------- public API -----------------------------
    def fit(self, graphs, y=None):
        """Construct and train the feasibility estimator."""
        self._feasibility_df        = self._build_feasibility_df()
        self._feasibility_estimator = self._build_feasibility_estimator(
            graphs, self._feasibility_df
        )
        self._is_fitted = True
        return self

    def interpolate(
        self,
        graph_src: nx.Graph,
        graph_dest: nx.Graph,
        *,
        metric=lambda G: G.number_of_nodes(),
    ):
        """
        Generate feasible graphs starting at *graph_src*, biased toward
        *graph_dest*, and return them ordered along the A→B projection.
        """
        if not self._is_fitted:
            raise RuntimeError("Call .fit(...) before .interpolate().")

        path      = self._generate_path(graph_src, graph_dest, metric)
        X, vA, vB = self._vectorize_graphs(path, graph_src, graph_dest)
        ordered   = self._project_and_sort(path, X, vA, vB, self.tol)
        ordered.append(graph_dest)  # ensure dest is always included
        return ordered

    # ------------------------ helper builders ---------------------------
    def _build_feasibility_df(self):
        """Compose the three decomposition DSLs exactly as in the spec."""
        df1 = neighborhood(radius=1)
        df2 = compose(cycle(), unlabel())
        df3 = compose(
            filter_by_number_of_connected_components(number_of_components=1),
            combination(distance=0),
            compose(cycle(), unlabel()),
        )
        return add(df1, df2, df3)

    def _build_feasibility_estimator(self, graphs, df):
        """Return a fitted FeasibilityEstimator."""
        estimators = [
            FeasibilityEstimatorFeatureCannotExist(
                decomposition_function=df,
                nbits=self.nbits,
            )
        ]
        fe = FeasibilityEstimator(estimators, parallel=self.parallel)
        fe.fit(graphs)
        return fe

    def _build_rewriter(self, graph_dest):
        """Return a DecompositionalGraphRewriter trained on dest."""
        return (
            DecompositionalGraphRewriter(
                decomposition_function=self.decomposition_function,
                max_permutations=1,
            ).fit([graph_dest])
        )

    # ----------------------- generation helpers ------------------------
    def _generate_path(self, graph_src, graph_dest, metric):
        """
        Iterative generate-and-select loop:

        1. keep candidates whose |metric − target| ≤ self.metric_tol
           (fallback to the global minimum if none qualify);
        2. among those, pick the graph whose **vector distance to dest**
           is smallest.
        """
        rewriter = self._build_rewriter(graph_dest)
        target   = metric(graph_dest)

        # transformer for quick dense vectors (once per interpolation run)
        qt = QuotientGraphTransformer(
            nbits=self.nbits,
            decomposition_function=self._feasibility_df,
            return_dense=True,
            n_jobs=1,
        )
        v_dest = qt.transform([graph_dest])[0]     # shape (d,)

        path = [graph_src]

        for _ in range(self.n_iter):
            # ❶ generate and filter for feasibility
            candidates = rewriter.generate(path[-1], n_samples=self.k_samples)
            candidates = self._feasibility_estimator.filter(candidates)
            if not candidates:
                break

            # ❷ keep only those within the metric tolerance
            diffs = np.array([abs(metric(G) - target) for G in candidates])
            mask  = diffs <= self.metric_tol
            if not mask.any():                     # nothing within tol → fallback
                mask = diffs == diffs.min()

            keep = [g for g, m in zip(candidates, mask) if m]

            # ❸ pick the one closest (Euclidean) to graph_dest in feature space
            X      = qt.transform(keep)            # shape (k, d)
            dists  = np.linalg.norm(X - v_dest, axis=1)
            best   = keep[int(dists.argmin())]

            path.append(best)

        return path

    # ------------------------ vectorisation ----------------------------
    def _vectorize_graphs(self, graphs, graph_src, graph_dest):
        """
        Vectorise *graphs* with the same decomposition/hash as feasibility.
        Returns (X, vA, vB) where:
            X  : dense matrix (len(graphs), d)
            vA : vector for graph_src
            vB : vector for graph_dest
        """
        qt = QuotientGraphTransformer(
            nbits=self.nbits,
            decomposition_function=self._feasibility_df,
            return_dense=True,
            n_jobs=1,
        ).fit([graph_src, graph_dest])

        X  = qt.transform(graphs)
        vA = X[0]
        vB = qt.transform([graph_dest])[0]
        return X, vA, vB

    # -------------------- projection & sorting -------------------------
    @staticmethod
    def _project_and_sort(graphs, X, vA, vB, tol):
        """
        Project rows of X onto line vA→vB, merge duplicates (tolerance *tol*),
        and return graphs ordered by projection coordinate.
        """
        direction = vB - vA
        norm = np.linalg.norm(direction)

        if norm == 0:
            return graphs                              # degenerate: keep order

        unit    = direction / norm
        scalars = (X - vA) @ unit                      # length len(graphs)

        buckets = {}
        for s, g in zip(scalars, graphs):
            key = round(s / tol)                       # bucket by tolerance
            if key not in buckets:
                buckets[key] = (s, g)

        ordered_graphs = [
            g for s, g in sorted(buckets.values(), key=lambda p: p[0])
        ]
        return ordered_graphs
