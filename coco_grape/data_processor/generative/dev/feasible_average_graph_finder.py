# ─────────── average_graph_finder.py  (updated) ───────────
from __future__ import annotations
import numpy as np
import networkx as nx
from typing import List, Optional, Callable

from sklearn.base import BaseEstimator

# ── CoCoGraPE plumbing ─────────────────────────────────────────────────
from coco_grape.module.quotientgraph.vectorize import QuotientGraphTransformer
from coco_grape.data_processor.generative.decompositional_graph_rewriter import (
    DecompositionalGraphRewriter,
)
from coco_grape.data_processor.generative.feasibility_estimator import (
    FeasibilityEstimator,
    FeasibilityEstimatorFeatureCannotExist,
)
from coco_grape.module.quotientgraph.operator import (
    neighborhood, compose, cycle, unlabel,
    filter_by_number_of_connected_components, combination, add, tree,
)
from coco_grape.data_processor.generative.feasible_graph_interpolator import FeasibleGraphInterpolator


# ──────────────────────────  helper math  ──────────────────────────────
def _pairwise_distances(V: np.ndarray) -> np.ndarray:
    """Return full Euclidean distance matrix (float)."""
    gram   = V @ V.T                       # counts → still int
    gram   = gram.astype(float, copy=False)  # ← promote once

    sqnorm = np.diag(gram)
    D2 = sqnorm[:, None] + sqnorm[None, :] - 2.0 * gram
    np.maximum(D2, 0, out=D2)
    np.fill_diagonal(D2, 0.0)
    return np.sqrt(D2, out=D2)             # now dtype=float so OK
# ───────────────────────────────────────────────────────────────────────


class AverageGraphFinder(BaseEstimator):
    """
    Search for a graph closest to the centroid of a sample set.
    Optionally seeds the search by *farthest-pair halving*.

    Parameters
    ----------
    use_farthest_pair_halving_seed_graph : bool
        If True, the initial beam is seeded with the result of the
        farthest-pair halving heuristic (uses the same feasibility stack).
    All other arguments are unchanged from the previous revision.
    """

    # ------------------------------------------------------------------
    def __init__(self,
                 *,
                 nbits: int = 11,
                 decomposition_function: Optional[Callable] = None,
                 beam_size: int = 4,
                 max_iters: int = 25,
                 k_neigh: int = 16,
                 lambda_size: float = 0.0,
                 use_farthest_pair_halving_seed_graph: bool = True):

        self.nbits  = nbits
        self.decomposition_function = (
            decomposition_function if decomposition_function is not None
            else add(cycle(), tree())
        )
        self.beam_size = beam_size
        self.max_iters = max_iters
        self.k_neigh   = k_neigh
        self.lambda_size    = lambda_size
        self.use_farthest_pair_halving_seed_graph = use_farthest_pair_halving_seed_graph

        # fitted-time slots
        self._vectorizer:           Optional[QuotientGraphTransformer] = None
        self._centroid:             Optional[np.ndarray] = None
        self._avg_nodes:            Optional[float]      = None
        self._feas:                 Optional[FeasibilityEstimator] = None
        self._train_graphs:         Optional[List[nx.Graph]] = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    #                                FIT
    # ------------------------------------------------------------------
    def fit(self, graphs: List[nx.Graph], y=None):
        """vectorise, compute centroid, build feasibility estimator"""
        self._train_graphs = graphs

        df1 = neighborhood(radius=1)
        df2 = compose(cycle(), unlabel())
        df3 = compose(
            filter_by_number_of_connected_components(number_of_components=1),
            combination(distance=0),
            compose(cycle(), unlabel()),
        )
        feasibility_df = add(df1, df2, df3)

        self._vectorizer = QuotientGraphTransformer(
            nbits=self.nbits,
            decomposition_function=feasibility_df,
            return_dense=True,
            n_jobs=1,
        ).fit(graphs)

        X = self._vectorizer.transform(graphs)
        self._centroid  = X.mean(axis=0)
        self._avg_nodes = float(np.mean([g.number_of_nodes() for g in graphs]))

        feas_estimators = [
            FeasibilityEstimatorFeatureCannotExist(
                decomposition_function=feasibility_df,
                nbits=self.nbits,
            )
        ]
        self._feas = FeasibilityEstimator(feas_estimators, parallel=True).fit(graphs)

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    #                         PUBLIC PREDICT
    # ------------------------------------------------------------------
    def predict(self, init_graphs: List[nx.Graph]) -> nx.Graph:
        """Return the graph found closest to the centroid."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        # ―― optional farthest-pair halving seed ――――――――――――――――――――――
        if self.use_farthest_pair_halving_seed_graph and len(init_graphs) >= 2:
            seed = self._farthest_pair_halving(init_graphs)
            beam = [seed] + init_graphs
        else:
            beam = init_graphs

        # keep only beam_size distinct graphs
        beam = beam[: self.beam_size]
        beam_scores = [self._score(g) for g in beam]

        rewriter = DecompositionalGraphRewriter(
            decomposition_function=self.decomposition_function,
            max_permutations=1,
        ).fit(self._train_graphs)

        for _ in range(self.max_iters):
            neigh: List[nx.Graph] = []
            for g in beam:
                neigh += rewriter.generate(g, n_samples=self.k_neigh)

            neigh = self._feas.filter(neigh)
            if not neigh:
                break

            scores = np.array([self._score(g) for g in neigh])
            idx    = np.argsort(scores)[: self.beam_size]
            beam   = [neigh[i] for i in idx]
            beam_scores = scores[idx]

        return beam[int(np.argmin(beam_scores))]

    # ------------------------------------------------------------------
    #                        FARTHEST-PAIR HALVING
    # ------------------------------------------------------------------
    def _farthest_pair_halving(self, graphs: List[nx.Graph]) -> nx.Graph:
        """
        Internal implementation that *shares* the already-built vectoriser
        and feasibility estimator.  Uses a lightweight FeasibleGraphInterpolator
        (same nbits & decomposition) to produce mid-points.
        """
        interpolator = FeasibleGraphInterpolator(
            nbits=self.nbits,
            decomposition_function=self.decomposition_function,
            n_iter=10,
        ).fit(self._train_graphs)

        current = graphs
        while len(current) > 1:
            V = self._vectorizer.transform(current)
            D = _pairwise_distances(V)

            idx_remaining = list(range(len(current)))
            next_round: List[nx.Graph] = []

            while len(idx_remaining) >= 2:
                sub_D = D[np.ix_(idx_remaining, idx_remaining)]
                i_rel, j_rel = divmod(sub_D.argmax(), sub_D.shape[1])
                i, j = idx_remaining[i_rel], idx_remaining[j_rel]
                idx_remaining.remove(i)
                idx_remaining.remove(j)

                path  = interpolator.interpolate(current[i], current[j])
                g_mid = path[len(path) // 2]   # midpoint graph
                g_mid = self._feas.filter([g_mid] + [current[i], current[j]])[0]  # ensure feasibility
                next_round.append(g_mid)

            #if idx_remaining: next_round.append(current[idx_remaining[0]])

            current = next_round
        return current[0]

    # ------------------------------------------------------------------
    #                              SCORE
    # ------------------------------------------------------------------
    def _score(self, g: nx.Graph) -> float:
        v = self._vectorizer.transform([g])[0]
        dist_vec  = np.linalg.norm(v - self._centroid)
        dist_size = abs(g.number_of_nodes() - self._avg_nodes)
        viol      = self._feas.number_of_violations([g])[0]
        return dist_vec + self.lambda_size * dist_size + viol
