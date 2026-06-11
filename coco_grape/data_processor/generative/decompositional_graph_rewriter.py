import networkx as nx
from collections import defaultdict
from itertools import permutations
import random
from coco_grape.module.graph_duplicate_detection_estimator import GraphDuplicateDetectionEstimator
from coco_grape.module.quotientgraph.definitions import graph_hash_label_function_factory
from coco_grape.module.quotientgraph.type        import QuotientGraph
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union
import hashlib
import json          # easiest way to make a deterministic byte string


class DecompositionalGraphRewriter:
    def __init__(self,         
                 decomposition_function,
                 label_function     = None,
                 attribute_function = None,
                 edge_function      = None,  
                 max_permutations   = 1):
        self.label_function      = label_function  or graph_hash_label_function_factory()
        self.attribute_function  = attribute_function
        self.edge_function       = edge_function
        self.cut_signature_partitioner = self.quotient_pipeline_to_cut_signature_partitioner(
            decomposition_function,
            self.label_function,
            self.attribute_function,
            self.edge_function
        )
        self.max_permutations = max_permutations
        self.part_db = defaultdict(list)

    def quotient_pipeline_to_cut_signature_partitioner(
            self,
            decomposition_function,
            label_function     = None,
            attribute_function = None,
            edge_function      = None
        ):
        """
        Returns a function with signature  cut_signature_partitioner(graph: nx.Graph) → List[(sub, cut_edges)]

        - `decomposition_function` is the operator you composed with `compose`, `add`, etc.
        - Optional *label/attribute/edge* functions are forwarded to the internal QuotientGraph.
        """

        def cut_signature_partitioner(graph: nx.Graph):
            # 1. Build a QuotientGraph from the raw graph
            qg = QuotientGraph(
                graph               = graph,
                label_function      = label_function  or graph_hash_label_function_factory(),
                attribute_function  = attribute_function,
                edge_function       = edge_function,
            ).create_default_image_node().update()

            # 2. Apply the user-defined decomposition pipeline
            qg = decomposition_function(qg)
            qg.update()                     # ensure image nodes / edges are up-to-date

            # 3. Convert image-node associations into (subgraph, cut_edges) pairs
            parts = []
            for sub in qg.get_image_nodes_associations():
                inner = set(sub.nodes())
                cut   = [ ((u, v), graph.edges[u, v])
                        for u in inner
                        for v in graph.neighbors(u)
                        if v not in inner ]
                parts.append((sub.copy(), cut))   # copy to detach from qg

            return parts

        return cut_signature_partitioner
    
    def _cut_signature(self, cut_edges):
        """
        Return a deterministic SHA-256 hex digest of the **multiset of edge labels**
        that form the cut.  Two cuts with exactly the same label multiset (order
        independent) produce the same 64-char string; different multisets collide
        only with astronomically low probability (≈2⁻¹²⁸).
        """
        # 1. Multiset canonicalisation  →  sorted list
        labels = sorted(attr["label"] for (_, _), attr in cut_edges)   # list[str|int]

        # 2. Serialise to a stable byte string
        #    * json.dumps with separators=(',', ':') avoids whitespace variance
        #    * ensure_ascii=False lets non-ASCII labels through unchanged
        payload = json.dumps(labels, separators=(",", ":"), ensure_ascii=False).encode()

        # 3. Hash
        return hashlib.sha256(payload).hexdigest()   # 64-char hex string
    
    def fit(self, graphs):
        """Extract and store parts indexed by cut signature."""
        for graph in graphs:
            parts = self.cut_signature_partitioner(graph)
            for subgraph, cut_edges in parts:
                sig = self._cut_signature(cut_edges)
                self.part_db[sig].append((subgraph.copy(), cut_edges))
        return self


    def generate(
        self,
        graph: nx.Graph,
        n_samples: Optional[int] = None,
        *,
        decomposition_function: Optional[Callable[[QuotientGraph], QuotientGraph]] = None,
    ):
        # 1. Select partitioner
        if decomposition_function is None:
            cut_partitioner = self.cut_signature_partitioner
        else:
            cut_partitioner = self.quotient_pipeline_to_cut_signature_partitioner(
                decomposition_function,
                self.label_function,
                self.attribute_function,
                self.edge_function,
            )

        # 2. Decompose target graph
        parts = cut_partitioner(graph)
        random.shuffle(parts)

        # 3. Build variants
        new_graphs: List[nx.Graph] = []
        for subgraph, cut_edges in parts:
            sig = self._cut_signature(cut_edges)
            for cand_subgraph, cand_cut_edges in self.part_db.get(sig, []):
                variants = self._replace(
                    graph, subgraph, cut_edges, cand_subgraph, cand_cut_edges
                )
                random.shuffle(variants)
                new_graphs.extend(variants)
                if n_samples is not None and len(new_graphs) >= n_samples:
                    break
            if n_samples is not None and len(new_graphs) >= n_samples:
                break

        if new_graphs:
            new_graphs = GraphDuplicateDetectionEstimator().fit_filter(new_graphs)
        return new_graphs[: n_samples] if n_samples else new_graphs


    def _replace(self, graph, subgraph, cut_edges, cand_subgraph, cand_cut_edges):
        if len(cut_edges) != len(cand_cut_edges):
            return []

        inner_nodes = set(subgraph.nodes())
        complement_nodes = set(graph.nodes()) - inner_nodes
        complement = graph.subgraph(complement_nodes).copy()

        # Remap candidate subgraph nodes to new IDs
        next_id = max(complement.nodes(), default=-1) + 1
        id_map = {n: next_id + i for i, n in enumerate(cand_subgraph.nodes())}

        # Add candidate subgraph
        new_g = complement.copy()
        for n, attrs in cand_subgraph.nodes(data=True):
            new_g.add_node(id_map[n], **attrs)
        for u, v, attrs in cand_subgraph.edges(data=True):
            new_g.add_edge(id_map[u], id_map[v], **attrs)

        # Try wiring candidate cut to graph cut
        rewired_graphs = []
        for perm in permutations(cand_cut_edges, r=len(cut_edges)):
            trial = new_g.copy()
            valid = True
            for ((cu, _), cattr), ((gu, gv), gattr) in zip(perm, cut_edges):
                if cattr['label'] != gattr['label']:
                    valid = False
                    break
                mapped_node = id_map[cu]
                external_node = gv if gu in inner_nodes else gu
                trial.add_edge(mapped_node, external_node, **gattr)
            if valid:
                rewired_graphs.append(trial)
            if len(rewired_graphs) >= self.max_permutations:
                break
        return rewired_graphs
