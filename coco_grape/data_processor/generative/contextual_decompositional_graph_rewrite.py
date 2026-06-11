"""
ContextualDecompositionalGraphRewriter
=====================================

A context‑aware extension of :class:`DecompositionalGraphRewriter` that selects
replacement parts based on both *cut signatures* **and** similarity of their
surrounding *context* subgraphs.

Key changes vs. the previous draft
----------------------------------
1. **`context_decomposition_function`** now has **the same signature** as
   `decomposition_function` – i.e. it transforms a :class:`QuotientGraph` and
   returns it.
2. A single built‑in **relation** mechanism with two modes:
   * ``overlap`` – contexts that share ≥1 node with the part.
   * ``proximity`` – contexts whose minimum shortest‑path distance to the part
     lies within a user‑supplied ``distance_range = (min_dist, max_dist)``.
3. Public constructor parameters reflect these changes – no external relation
   function is required.

Example
~~~~~~~
```python
rewriter = ContextualDecompositionalGraphRewriter(
    decomposition_function           = primary_pipeline,
    context_decomposition_function   = context_pipeline,
    relation_mode                    = "proximity",
    distance_range                   = (0, 2),
    max_permutations                 = 3,
    temperature                      = 0.5,
)
rewriter.fit(training_graphs)
augmented = rewriter.generate(test_graph, n_samples=50)
```
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np  # only for optional type hints

# ---------------------------------------------------------------------------
# External dependencies (adjust paths to match your project)
# ---------------------------------------------------------------------------
from coco_grape.module.quotientgraph.definitions import graph_hash_label_function_factory
from coco_grape.module.quotientgraph.type import QuotientGraph
from coco_grape.module.graph_duplicate_detection_estimator import (
    GraphDuplicateDetectionEstimator,
)

try:
    from .decompositional_graph_rewriter import DecompositionalGraphRewriter
except ImportError:  # flat script fallback
    from decompositional_graph_rewriter import DecompositionalGraphRewriter  # type: ignore

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_context_subgraph_extractor(
    context_decomposition_function: Callable[[QuotientGraph], QuotientGraph],
    *,
    label_function: Callable[[nx.Graph], int] | None = None,
    attribute_function: Callable[[nx.Graph], np.ndarray] | None = None,
    edge_function: Callable[[QuotientGraph], QuotientGraph] | None = None,
) -> Callable[[nx.Graph], List[nx.Graph]]:
    """Return a function that extracts context subgraphs from a raw graph."""

    def extractor(G: nx.Graph) -> List[nx.Graph]:
        qg = QuotientGraph(
            graph=G,
            label_function=label_function or graph_hash_label_function_factory(),
            attribute_function=attribute_function,
            edge_function=edge_function,
        ).create_default_image_node().update()

        qg = context_decomposition_function(qg)
        qg.update()
        return qg.get_image_nodes_associations()

    return extractor

# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ContextualDecompositionalGraphRewriter(DecompositionalGraphRewriter):
    """Context‑aware variant of :class:`DecompositionalGraphRewriter`."""

    # ---------------------------------------------------------------------
    # INITIALISATION
    # ---------------------------------------------------------------------

    def __init__(
        self,
        decomposition_function: Callable[[QuotientGraph], QuotientGraph],
        context_decomposition_function: Callable[[QuotientGraph], QuotientGraph],
        *,
        relation_mode: str = "overlap",  # "overlap" | "proximity"
        distance_range: Tuple[int, int] = (0, 2),
        label_function: Callable[[nx.Graph], int] | None = None,
        attribute_function: Callable[[nx.Graph], np.ndarray] | None = None,
        edge_function: Callable[[QuotientGraph], QuotientGraph] | None = None,
        context_label_function: Callable[[nx.Graph], int] | None = None,
        max_permutations: int = 1,
        temperature: float = 1.0,
    ) -> None:
        """Create a context‑aware decompositional rewriter.

        Parameters
        ----------
        decomposition_function
            Primary part decomposition pipeline (operates on ``QuotientGraph``).
        context_decomposition_function
            Secondary decomposition that yields *context* subgraphs, sharing the
            same signature as ``decomposition_function``.
        relation_mode
            ``"overlap"`` or ``"proximity"``.
        distance_range
            *(min_dist, max_dist)* tuple used when ``relation_mode="proximity"``.
        temperature
            Soft‑max temperature converting similarity → sampling weight.
        All other parameters are forwarded to the parent class.
        """

        super().__init__(
            decomposition_function,
            label_function=label_function,
            attribute_function=attribute_function,
            edge_function=edge_function,
            max_permutations=max_permutations,
        )
        self.label_function      = label_function  or graph_hash_label_function_factory()
        self.attribute_function  = attribute_function
        self.edge_function       = edge_function
        
        # ---------------------------------------------------------------
        # Context extraction helpers
        # ---------------------------------------------------------------
        self._context_subgraph_extractor = _make_context_subgraph_extractor(
            context_decomposition_function,
            label_function=label_function,
            attribute_function=attribute_function,
            edge_function=edge_function,
        )

        # ---------------------------------------------------------------
        # Configuration
        # ---------------------------------------------------------------
        self.relation_mode: str = relation_mode.lower()
        if self.relation_mode not in {"overlap", "proximity"}:
            raise ValueError("relation_mode must be 'overlap' or 'proximity'")

        self.distance_range: Tuple[int, int] = distance_range
        self.context_label_function: Callable[[nx.Graph], int] = (
            context_label_function or graph_hash_label_function_factory()
        )
        self.temperature: float = max(temperature, 1e-6)

        # Override parent`s part_db to also store context bags
        # cut_signature → List[(part_subgraph, cut_edges, context_bag)]
        self.part_db: defaultdict[int, list] = defaultdict(list)

    # ---------------------------------------------------------------------
    # PRIVATE UTILITY METHODS
    # ---------------------------------------------------------------------

    @staticmethod
    def _cosine(a: Counter[int], b: Counter[int]) -> float:
        if not a or not b:
            return 0.0
        inter = a.keys() & b.keys()
        dot = sum(min(a[k], b[k]) for k in inter)
        norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(
            sum(v * v for v in b.values())
        )
        return 0.0 if norm == 0 else dot / norm

    # ---------------------------------------------------------------------
    # RELATION CHECK
    # ---------------------------------------------------------------------

    def _relation(self, G: nx.Graph, part: nx.Graph, ctx: nx.Graph) -> bool:
        """Return *True* if *ctx* qualifies as context for *part*."""
        if self.relation_mode == "overlap":
            return bool(set(part) & set(ctx))

        # proximity mode
        min_dist, max_dist = self.distance_range
        best = max_dist + 1  # sentinel
        for u in part.nodes():
            for v in ctx.nodes():
                try:
                    d = nx.shortest_path_length(G, u, v)
                    if d < best:
                        best = d
                        if best <= min_dist:
                            return True  # early exit (inside range)
                except nx.NetworkXNoPath:
                    continue
        return min_dist <= best <= max_dist

    # ---------------------------------------------------------------------
    # TRAINING PHASE
    # ---------------------------------------------------------------------

    def fit(self, graphs: Iterable[nx.Graph]):  # type: ignore[override]
        for G in graphs:
            parts = self.cut_signature_partitioner(G)
            contexts = self._context_subgraph_extractor(G)
            ctx_labels = {id(c): self.context_label_function({"association": c}) for c in contexts}

            for subgraph, cut_edges in parts:
                sig = self._cut_signature(cut_edges)
                bag: Counter[int] = Counter()
                for ctx in contexts:
                    if self._relation(G, subgraph, ctx):
                        bag[ctx_labels[id(ctx)]] += 1
                self.part_db[sig].append((subgraph.copy(), cut_edges, bag))
        return self

    # ---------------------------------------------------------------------
    # GENERATION PHASE
    # ---------------------------------------------------------------------

    def generate(
        self,
        graph: nx.Graph,
        n_samples: Optional[int] = None,
        *,
        decomposition_function: Optional[Callable[[QuotientGraph], QuotientGraph]] = None,
    ):
        # 1. Pick partitioner
        if decomposition_function is None:
            cut_partitioner = self.cut_signature_partitioner
        else:
            cut_partitioner = self.quotient_pipeline_to_cut_signature_partitioner(
                decomposition_function,
                self.label_function,
                self.attribute_function,
                self.edge_function,
            )

        # 2. Decompose target graph & contexts
        parts = cut_partitioner(graph)
        random.shuffle(parts)
        contexts = self._context_subgraph_extractor(graph)
        ctx_labels = {id(c): self.context_label_function({"association": c}) for c in contexts}

        # 3. Build context bags for target parts
        target_bags: List[Counter[int]] = []
        for part_subgraph, _ in parts:
            bag = Counter()
            for ctx in contexts:
                if self._relation(graph, part_subgraph, ctx):
                    bag[ctx_labels[id(ctx)]] += 1
            target_bags.append(bag)

        # 4. Select candidates and create variants
        new_graphs: List[nx.Graph] = []
        for (part_subgraph, cut_edges), tgt_bag in zip(parts, target_bags):
            sig = self._cut_signature(cut_edges)
            candidates = self.part_db.get(sig, [])
            if not candidates:
                continue
            sims = [self._cosine(tgt_bag, c_bag) for *_ , c_bag in candidates]
            weights = [math.exp(s / self.temperature) for s in sims]
            if not any(weights):
                weights = [1.0] * len(candidates)
            cand_subgraph, cand_cut_edges, _ = random.choices(candidates, weights=weights)[0]
            variants = self._replace(graph, part_subgraph, cut_edges, cand_subgraph, cand_cut_edges)
            random.shuffle(variants)
            new_graphs.extend(variants)
            if n_samples is not None and len(new_graphs) >= n_samples:
                break
        if new_graphs:
            new_graphs = GraphDuplicateDetectionEstimator().fit_filter(new_graphs)
        return new_graphs[: n_samples] if n_samples else new_graphs


# ---------------------------------------------------------------------------
# Export control
# ---------------------------------------------------------------------------

__all__: Sequence[str] = ["ContextualDecompositionalGraphRewriter"]
