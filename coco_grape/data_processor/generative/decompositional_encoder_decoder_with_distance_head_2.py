#!/usr/bin/env python
"""Provides interface."""

import copy
import numpy as np
import networkx as nx
import random
import pulp
import dill as pickle
from coco_grape.utils.timeit import timeit
import torch
from typing import List, Tuple, Optional, Any, Sequence, Dict, Union
from coco_grape.data_processor.supervised.low_rank_mlp import LowRankMLP
from coco_grape.data_processor.generative.conditional_node_generator_base import ConditionalNodeGeneratorBase
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict

def compute_hop_supervision(
    graphs: List[nx.Graph],
    *,
    force_bi_directional_edges: bool = True,
    treat_unreachable_as_plus: bool = True
) -> Tuple[np.ndarray, List[Tuple[int,int,int]], Dict[int, str]]:
    """
    Returns:
      y      : np.ndarray[int] with class ids {0:'1-hop',1:'2-hop',2:'3-hop',3:'4+-hop'}
      pairs  : list of (g_idx, i, j)
      names  : id->name dict
    """
    class_names = {0: "1-hop", 1: "2-hop", 2: "3-hop", 3: "4+-hop"}
    labels, pairs = [], []

    for g_idx, G in enumerate(graphs):
        nodes = list(G.nodes())
        idx = {u: i for i, u in enumerate(nodes)}
        for u in nodes:
            sp = nx.single_source_shortest_path_length(G, u)  # dict: node->hops
            n = len(nodes)
            hop = np.full(n, np.inf, dtype=float)
            for v, d in sp.items():
                hop[idx[v]] = float(d)
            iu = idx[u]
            for v in nodes:
                iv = idx[v]
                if iu == iv:
                    continue
                d = hop[iv]
                if np.isinf(d):
                    if not treat_unreachable_as_plus:
                        continue
                    cls = 3
                else:
                    d = int(d)
                    cls = 0 if d == 1 else 1 if d == 2 else 2 if d == 3 else 3
                pairs.append((g_idx, iu, iv))
                labels.append(cls)
                if force_bi_directional_edges and not G.is_directed():
                    pairs.append((g_idx, iv, iu))
                    labels.append(cls)
    return np.asarray(labels, dtype=int), pairs, class_names
def inject_labels_from_graphs(
    node_encs_list,
    graphs,
    *,
    label_index=2,
    label_to_idx=None,
    unlabeled="__UNLABELED__",
):
    """
    Overwrite column `label_index` with integer IDs from G.nodes[u]['label'].
    Pads columns if missing. Returns (new_list, label_to_idx).
    """
    out = []

    # build/normalize mapping
    if label_to_idx is None:
        all_labels = []
        for G in graphs:
            for u in G.nodes():
                v = G.nodes[u].get("label", unlabeled)
                if v is None or (isinstance(v, str) and v.strip() == ""):
                    v = unlabeled
                all_labels.append(str(v))
        uniq = list(dict.fromkeys(all_labels))
        label_to_idx = {lab: i for i, lab in enumerate(uniq)}
    else:
        label_to_idx = {str(k): int(v) for k, v in label_to_idx.items()}

    default_id = label_to_idx.get(unlabeled, next(iter(label_to_idx.values()), 0))

    for enc, G in zip(node_encs_list, graphs):
        E = np.asarray(enc, dtype=float).copy()
        N, D = E.shape
        if D <= label_index:
            E = np.hstack([E, np.zeros((N, label_index + 1 - D), dtype=E.dtype)])

        nodes = list(G.nodes())
        ids = np.zeros(N, dtype=float)
        for i, u in enumerate(nodes[:N]):
            v = G.nodes[u].get("label", unlabeled)
            if v is None or (isinstance(v, str) and v.strip() == ""):
                v = unlabeled
            ids[i] = float(label_to_idx.get(str(v), default_id))
        E[:, label_index] = ids
        out.append(E)

    return out, label_to_idx

def scaled_slerp(v0: np.ndarray, v1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation (slerp) between vectors v0 and v1,
    with linear interpolation of their magnitudes.

    Parameters
    ----------
    v0, v1 : np.ndarray
        Input vectors to interpolate between.
    t : float
        Interpolation parameter (<0: extrapolate towards v0 beyond; >1: beyond v1).

    Returns
    -------
    np.ndarray
        The interpolated vector at fraction t of the way from v0 to v1.
    """
    # Compute magnitudes
    mag0 = np.linalg.norm(v0)
    mag1 = np.linalg.norm(v1)

    # Normalize directions (guard against zero)
    v0_unit = v0 / mag0 if mag0 != 0 else v0
    v1_unit = v1 / mag1 if mag1 != 0 else v1

    # Compute angle between
    dot = np.clip(np.dot(v0_unit, v1_unit), -1.0, 1.0)
    theta = np.arccos(dot)

    # Slerp the direction
    if theta < 1e-6:
        # Nearly colinear: fall back to linear interpolation + renormalization
        direction = (1 - t) * v0_unit + t * v1_unit
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm != 0 else direction
    else:
        sin_theta = np.sin(theta)
        direction = (
            np.sin((1 - t) * theta) * v0_unit +
            np.sin(t * theta) * v1_unit
        ) / sin_theta

    # Linearly interpolate magnitudes
    mag = (1 - t) * mag0 + t * mag1
    return direction * mag


def scaled_slerp_average(vectors: np.ndarray) -> np.ndarray:
    """
    Compute a spherical-style centroid of multiple vectors, preserving
    average direction on the unit sphere while linearly averaging magnitudes.
    """
    vs = np.asarray(vectors, dtype=float)             # (B, D)
    mags = np.linalg.norm(vs, axis=1)                # (B,)
    unit_vs = np.zeros_like(vs)                      # (B, D)
    nonzero = mags > 0
    unit_vs[nonzero] = vs[nonzero] / mags[nonzero, None]

    avg_dir = unit_vs.sum(axis=0)                    # (D,)
    norm = np.linalg.norm(avg_dir)
    if norm > 0:
        avg_dir /= norm

    avg_mag = mags.mean()
    return avg_dir * avg_mag                         # (D,)


# Suppress numpy warnings for invalid operations and divisions
np.seterr(invalid='ignore', divide='ignore')

# =============================================================================
# DecompositionalNodeEncoderDecoder Class
# =============================================================================

class DecompositionalNodeEncoderDecoder(object):
    """
    DecompositionalNodeEncoderDecoder

    Implements an encoder-decoder framework for processing graph nodes.
    This class trains classifiers to predict node labels, edge labels, and the adjacency matrix.
    It supports transforming graphs into training data, applying augmentation, and reconstructing graphs 
    from predicted node embeddings.
    """
    
    def __init__(
        self,
        adjacency_matrix_classifier: LowRankMLP,
        node_label_classifier: LowRankMLP,
        edge_label_classifier: LowRankMLP,       
        verbose: bool = True,
        non_edges_factor: int = 1,
        existence_threshold: float = 0.5,
        num_augmentation_iterations: int = 0,
        augmentation_noise: float = 1e-2,
        enforce_connectivity: bool = True,
        degree_slack_penalty: float = 1e6,
        warm_start_mst: bool = True
    ) -> None:
        """
        Initializes the encoder-decoder with classifiers and configuration options.
        
        Parameters:
            adjacency_matrix_classifier: Classifier for predicting edge existence probabilities.
            node_label_classifier      : Classifier for node labels.
            edge_label_classifier      : Classifier for edge labels.
            verbose                    : Verbosity flag.
            non_edges_factor           : Ratio for sampling negative edges per positive.
            existence_threshold        : Threshold to consider a node existent.
            num_augmentation_iterations: Number of augmentation noise iterations.
            augmentation_noise         : Maximum noise amplitude for augmentation.
            enforce_connectivity       : Whether to enforce a single connected component.
            degree_slack_penalty       : Weight applied to slack variables for degree deviations.
            warm_start_mst            : Whether to warm start solver using maximum spanning tree.
        """
        self.adjacency_matrix_classifier = copy.deepcopy(adjacency_matrix_classifier)
        self.node_label_classifier      = copy.deepcopy(node_label_classifier)
        self.edge_label_classifier      = copy.deepcopy(edge_label_classifier)
        self.verbose                    = verbose
        self.non_edges_factor           = non_edges_factor
        self.existence_threshold        = existence_threshold
        self.num_augmentation_iterations= num_augmentation_iterations
        self.augmentation_noise         = augmentation_noise
        self.enforce_connectivity       = enforce_connectivity
        self.degree_slack_penalty       = degree_slack_penalty
        self.warm_start_mst             = warm_start_mst

    def optimize_adjacency_matrix(
        self,
        prob_matrix: np.ndarray,
        target_degrees: List[int],
        timeLimit: int = 60,
        verbose: bool = False,
        alpha: float = 0.7,
        connectivity: Optional[bool] = None
    ) -> np.ndarray:
        """
        Uses PuLP+CBC to optimize edge selection under degree and connectivity constraints.
        Can warm-start with an MST based on probabilities.
        """
        n = prob_matrix.shape[0]
        # Smooth probabilities
        if alpha != 1.0:
            prob_matrix = np.power(prob_matrix, alpha)

        # Connectivity setting
        if connectivity is None:
            connectivity = self.enforce_connectivity

        # Build LP
        prob = pulp.LpProblem("AdjacencyMatrixOptimization", pulp.LpMaximize)

        # Decision vars
        x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
             for i in range(n) for j in range(i+1, n)}
        u = {i: pulp.LpVariable(f"u_{i}", lowBound=0, cat="Integer") for i in range(n)}
        v = {i: pulp.LpVariable(f"v_{i}", lowBound=0, cat="Integer") for i in range(n)}

        # Objective
        prob += (
            pulp.lpSum(prob_matrix[i, j] * x[(i, j)] for i in range(n) for j in range(i+1, n))
            - self.degree_slack_penalty * pulp.lpSum(u[i] + v[i] for i in range(n))
        )

        # Degree constraints
        for i in range(n):
            incident = [x[(i,j)] for j in range(i+1, n)] + [x[(j,i)] for j in range(i) if (j,i) in x]
            prob += (pulp.lpSum(incident) + u[i] - v[i] == target_degrees[i]), f"Degree_{i}"

        # Connectivity via flow
        if connectivity:
            directed_edges = [(i,j) for (i,j) in x] + [(j,i) for (i,j) in x]
            f_vars = {(u_, v_): pulp.LpVariable(f"f_{u_}_{v_}", lowBound=0, cat="Continuous")
                      for u_,v_ in directed_edges}
            M = n-1
            root = 0
            for v_idx in range(n):
                inflow  = pulp.lpSum(f_vars[(u_,v2)] for (u_,v2) in directed_edges if v2==v_idx)
                outflow = pulp.lpSum(f_vars[(v2,w)] for (v2,w) in directed_edges if v2==v_idx)
                prob += ((outflow-inflow)==M if v_idx==root else (inflow-outflow)==1), f"Flow_{v_idx}"
            for u_,v_ in directed_edges:
                i,j = min(u_,v_), max(u_,v_)
                prob += (f_vars[(u_,v_)] <= M * x[(i,j)]), f"FlowCouple_{u_}_{v_}"

        # Warm-start with MST
        if self.warm_start_mst:
            G = nx.Graph()
            G.add_nodes_from(range(n))
            for i in range(n):
                for j in range(i+1, n):
                    G.add_edge(i, j, weight=prob_matrix[i,j])
            T = nx.maximum_spanning_tree(G)
            # Initialize x
            for (i,j), var in x.items():
                var.start = 1 if T.has_edge(i,j) else 0

        # Solve
        solver = pulp.PULP_CBC_CMD(timeLimit=timeLimit, msg=verbose)
        prob.solve(solver)

        # Build adjacency
        adj = np.zeros((n,n), dtype=int)
        for (i,j), var in x.items():
            adj[i,j] = adj[j,i] = int(pulp.value(var))
        return adj

    def graphs_to_adjacency_matrices(self, graphs: List[nx.Graph]) -> List[np.ndarray]:
        """
        Converts a list of NetworkX graphs into a list of corresponding adjacency matrices.
        
        Parameters:
            graphs: List of NetworkX graph objects.
        
        Returns:
            List of numpy arrays representing the adjacency matrices of the graphs.
        """
        adj_mtx_list = []
        for graph in graphs:
            # Convert graph to a numpy array with integer type.
            adj_mtx = nx.to_numpy_array(graph, dtype=int)
            adj_mtx_list.append(adj_mtx)
        return adj_mtx_list

    def adj_mtx_to_targets(
        self,
        adj_mtx_list: List[np.ndarray],
        node_encodings_list: List[np.ndarray],
        use_edge_fraction: float, # <-- New parameter
        force_bi_directional_edges: bool = True,
        is_training: bool = False
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        """
        For each graph in adj_mtx_list, this function processes each node i.
        Edge sampling is only performed during training.
        
        Parameters:
            adj_mtx_list: List of adjacency matrices (numpy arrays) for graphs.
            node_encodings_list: List of corresponding node encodings.
            force_bi_directional_edges: When True, adds both directions of each edge.
            is_training: Whether this is being called during training (controls edge sampling).
        """
        # Collect all targets and pairs first
        all_targets = []
        all_pairs = []
        
        for g_idx, (adj_mtx, encodings) in enumerate(zip(adj_mtx_list, node_encodings_list)):
            n_nodes = adj_mtx.shape[0]
            for i in range(n_nodes):
                # Collect positive neighbors for node i
                pos_neighbors = [j for j in range(n_nodes) if j != i and adj_mtx[i, j] == 1]
                
                # Add positive examples (both directions if force_bi_directional_edges)
                for j in pos_neighbors:
                    all_targets.append(1)
                    all_pairs.append((g_idx, i, j))
                    if force_bi_directional_edges:
                        all_targets.append(1)  
                        all_pairs.append((g_idx, j, i))
                
                # Determine number of negative samples
                num_pos = len(pos_neighbors) * (2 if force_bi_directional_edges else 1)
                num_neg_samples = int(round(self.non_edges_factor * num_pos))
                if num_neg_samples <= 0:
                    continue
                
                # Build candidate list for negative examples
                candidate_indices = [k for k in range(n_nodes) if k != i and adj_mtx[i, k] == 0]
                if not candidate_indices:
                    continue
                
                # Get distances and sort candidates
                distances = np.array([np.linalg.norm(encodings[i] - encodings[k]) for k in candidate_indices])
                sorted_candidate_indices = np.argsort(distances)
                selected_negatives = [candidate_indices[idx] for idx in sorted_candidate_indices[:num_neg_samples]]
                
                # Add negative examples (both directions if force_bi_directional_edges)
                for k in selected_negatives:
                    all_targets.append(0)
                    all_pairs.append((g_idx, i, k))
                    if force_bi_directional_edges:
                        all_targets.append(0)
                        all_pairs.append((g_idx, k, i))
        
        # Apply edge sampling only if is_training is True and use_edge_fraction < 1.0
        if is_training and use_edge_fraction < 1.0: # Check use_edge_fraction value
            num_edges = len(all_pairs)
            num_edges_to_use = int(round(num_edges * use_edge_fraction)) # round to nearest int
            
            if self.verbose and num_edges > 0 : # Add check for num_edges > 0
                print(f"adj_mtx_to_targets: Sampling {num_edges_to_use} edges ({use_edge_fraction:.2%}) from {num_edges} total pairs.")
            
            if num_edges_to_use < num_edges and num_edges_to_use > 0 : # Ensure sampling is meaningful
                indices = np.random.choice(num_edges, num_edges_to_use, replace=False)
                all_targets = [all_targets[i] for i in indices]
                all_pairs = [all_pairs[i] for i in indices]
            elif num_edges_to_use == 0 and num_edges > 0:
                 if self.verbose:
                    print(f"adj_mtx_to_targets: Warning - num_edges_to_use is 0 with use_edge_fraction={use_edge_fraction} and num_edges={num_edges}. No edges will be used.")
                 return np.array([]), []
            elif num_edges_to_use == 0 and num_edges == 0: # No pairs to sample from
                return np.array([]), []

        return np.array(all_targets), all_pairs

    def compute_edge_supervision(
        self, 
        graphs: List[nx.Graph], 
        node_encodings_list: List[np.ndarray],
        use_edge_fraction: float  # <-- New parameter
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        """Compute edge supervision for training."""
        adj = self.graphs_to_adjacency_matrices(graphs)
        return self.adj_mtx_to_targets(adj, node_encodings_list, use_edge_fraction=use_edge_fraction, is_training=True)
    
    def compute_edge_label_supervision(
        self,
        graphs: List[nx.Graph],
        force_bi_directional_edges: bool = True
    ) -> Tuple[np.ndarray, List[Tuple[int,int,int]], Dict[str,int]]:
        """
        Collect per-edge label classes for supervision.
        Returns (y_edge_labels, pair_indices, label_to_idx).

        Only real edges are included; if force_bi_directional_edges, both (i,j) and (j,i) are added.
        """
        # discover label universe
        all_labels = []
        for G in graphs:
            for u, v in G.edges():
                all_labels.append(str(G.edges[u, v].get("label", "")))
        uniq = list(dict.fromkeys(all_labels))
        label_to_idx = {lab: i for i, lab in enumerate(uniq)} if uniq else {}

        # if <=1 label, return empty to disable training
        if len(label_to_idx) <= 1:
            return np.array([]), [], label_to_idx

        y = []
        pairs = []
        for g_idx, G in enumerate(graphs):
            nodes = list(G.nodes())
            idx = {u: i for i, u in enumerate(nodes)}
            for u, v in G.edges():
                lab = str(G.edges[u, v].get("label", uniq[0]))
                c = label_to_idx.get(lab, 0)
                i, j = idx[u], idx[v]
                y.append(c)
                pairs.append((g_idx, i, j))
                if force_bi_directional_edges:
                    y.append(c)
                    pairs.append((g_idx, j, i))
        return np.asarray(y, dtype=int), pairs, label_to_idx


    def encodings_and_adj_mtx_to_dataset(
        self,
        node_encodings_list: List[np.ndarray],
        adj_mtx_list: List[np.ndarray],
        use_edge_fraction: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Creates training dataset; returns empty arrays if no pairs."""
        y, pair_indices = self.adj_mtx_to_targets(
            adj_mtx_list, node_encodings_list,
            use_edge_fraction=use_edge_fraction,
            is_training=True
        )
        if len(pair_indices) == 0 or y.size == 0:
            # 2*D features when not using graph-level features
            feat_dim = node_encodings_list[0].shape[1] * 2
            return np.empty((0, feat_dim), dtype=float), np.empty((0,), dtype=float)
        X = self.encodings_to_instances(node_encodings_list, pair_indices)
        return X, y

    def encodings_to_instances(
        self,
        node_encodings_list: List[np.ndarray],
        pair_indices: Optional[List[Tuple[int, int, int]]] = None,
        use_graph_encoding: bool = False
    ) -> np.ndarray:
        """
        Creates feature instances from a list of node encodings.
        
        If pair_indices is provided, then for each tuple (graph_index, i, j) in pair_indices, 
        a graph-level encoding (sum of all node encodings) is computed and concatenated with 
        the source node encoding (i) and target node encoding (j).
        
        If pair_indices is None, then for each graph all pairs of distinct nodes (i, j) are used.
        Both directions (i, j) and (j, i) are evaluated.
        
        Parameters:
            node_encodings_list: List of numpy arrays where each array contains node encodings for a graph.
            pair_indices: (Optional) List of tuples (graph_index, i, j) specifying the node pairs for which 
                        to create instances. Default is None, meaning that all distinct pairs are used.
        
        Returns:
            A numpy array where each row is a feature instance for a given node pair.
        """
        instances = []
        if pair_indices is not None:
            # Use provided pair indices.
            for g_idx, i, j in pair_indices:
                encodings = node_encodings_list[g_idx]
                if use_graph_encoding: 
                    graph_encoding = np.sum(encodings, axis=0)
                    instance = np.hstack([graph_encoding, encodings[i], encodings[j]])
                else:
                    instance = np.hstack([encodings[i], encodings[j]])
                instances.append(instance)
        else:
            # Evaluate all pairs (i, j) with i != j for every graph.
            for g_idx, encodings in enumerate(node_encodings_list):
                if use_graph_encoding: 
                    graph_encoding = np.sum(encodings, axis=0)
                n_nodes = encodings.shape[0]
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        if i != j:
                            if use_graph_encoding:
                                instance = np.hstack([graph_encoding, encodings[i], encodings[j]])
                            else:
                                instance = np.hstack([encodings[i], encodings[j]])
                            instances.append(instance)
        return np.vstack(instances)
        
    @timeit
    def encodings_and_graphs_to_node_label_dataset(
        self,
        node_encodings_list: List[np.ndarray],
        graphs: List[nx.Graph]
    ) -> Tuple[np.ndarray, List[Any]]:
        X, y = [], []
        for graph, enc in zip(graphs, node_encodings_list):
            enc_no_lbl = np.delete(enc, 2, axis=1) if enc.shape[1] > 2 else enc
            for i, u in enumerate(list(graph.nodes())):
                X.append(enc_no_lbl[i])
                y.append(graph.nodes[u]['label'])
        return np.vstack(X), y

    def encodings_and_graphs_to_edge_label_dataset(
        self,
        node_encodings_list: List[np.ndarray],
        graphs: List[nx.Graph]
    ) -> Tuple[np.ndarray, List[Any]]:
        """
        Creates a dataset for training the edge label classifier.
        Extracts concatenated node encodings for each edge and the corresponding edge labels.
        
        Parameters:
            node_encodings_list: List of numpy arrays containing node encodings.
            graphs         : List of NetworkX graph objects.
        
        Returns:
            Tuple (X, edge_labels) where X is the feature matrix for edges and edge_labels is the list of labels.
        """
        instances = []
        edge_labels = []
        for graph, encodings in zip(graphs, node_encodings_list):
            # Iterate over nodes to consider each potential edge.
            for i, u in enumerate(list(graph.nodes())):
                for j, v in enumerate(list(graph.nodes())):
                    # If an edge exists between the nodes, create an instance.
                    if graph.has_edge(u, v):
                        instance = np.hstack([encodings[i], encodings[j]])
                        instances.append(instance)
                        edge_labels.append(graph.edges[u, v]['label'])
        if instances:
            instances = np.vstack(instances)
        return instances, edge_labels

    def encodings_and_adj_mtx_to_edge_dataset(
        self,
        node_encodings_list: List[np.ndarray],
        adj_mtx_list: List[np.ndarray]
    ) -> np.ndarray:
        """
        Creates a dataset of edge instances based on node encodings and the corresponding adjacency matrices.
        For each graph, every non-zero entry in the adjacency matrix (indicating an edge) is used to form an instance.
        
        Parameters:
            node_encodings_list: List of numpy arrays containing node encodings.
            adj_mtx_list  : List of adjacency matrices.
        
        Returns:
            A numpy array of concatenated node encoding pairs for all detected edges.
        """
        instances = []
        for encodings, adj_mtx in zip(node_encodings_list, adj_mtx_list):
            n_nodes = encodings.shape[0]
            # Iterate over all pairs of nodes.
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if adj_mtx[i, j] != 0:
                        instance = np.hstack([encodings[i], encodings[j]])
                        instances.append(instance)
        if instances:
            instances = np.vstack(instances)
        return instances
    
    @timeit
    def node_label_classifier_fit(
        self,
        node_encodings_list: List[np.ndarray],
        graphs: List[nx.Graph]
    ) -> None:
        # >>> add this guard <<<
        if self.node_label_classifier is None:
            if self.verbose:
                print("No node label classifier provided — skipping training and using label IDs in enc[:,2] with the stored mapping.")
            self.single_node_label = None
            return
        # --- existing code below ---
        X, y = self.encodings_and_graphs_to_node_label_dataset(node_encodings_list, graphs)
        unique_labels = np.unique(y)
        if len(unique_labels) == 1:
            if self.verbose:
                print(f"Only one node label found: {unique_labels[0]}. Skipping training and storing the label.")
            self.single_node_label = unique_labels[0]
        else:
            if self.verbose:
                print(f"Training node label predictor on {X.shape[0]} instances with {X.shape[1]} features")
            self.node_label_classifier.fit(X, y)
            self.single_node_label = None

    @timeit
    def edge_label_classifier_fit(
        self,
        node_encodings_list: List[np.ndarray],
        graphs: List[nx.Graph]
    ) -> None:
        """
        Trains the edge label classifier using the provided node encodings and graphs.
        
        Parameters:
            node_encodings_list: List of numpy arrays containing node encodings.
            graphs         : List of NetworkX graph objects.
        
        Side Effects:
            Sets self.single_edge_label if only one unique label exists.
            Trains self.edge_label_classifier if multiple unique labels are present.
        """
        X, y = self.encodings_and_graphs_to_edge_label_dataset(node_encodings_list, graphs)
        unique_labels = np.unique(y)
        # If only one edge label is found, store it and skip training.
        if len(unique_labels) == 1:
            if self.verbose:
                print("Only one edge label found: {}. Skipping training and storing the label.".format(unique_labels[0]))
            self.single_edge_label = unique_labels[0]
        else:
            if self.verbose:
                print('Training edge label predictor on {} instances with {} features'.format(X.shape[0], X.shape[1]))
            self.edge_label_classifier.fit(X, y)
            self.single_edge_label = None

    @timeit
    def adjacency_matrix_classifier_fit(
        self,
        node_encodings_list: List[np.ndarray],
        graphs: List[nx.Graph],
        use_edge_fraction: float
    ) -> None:
        """
        Train the adjacency matrix classifier. Skips cleanly if no pairs.
        """
        adj_mtx_list = self.graphs_to_adjacency_matrices(graphs)
        # Build dataset (handles sampling)
        X, y = self.encodings_and_adj_mtx_to_dataset(
        node_encodings_list=node_encodings_list,
        adj_mtx_list=adj_mtx_list,
             use_edge_fraction=use_edge_fraction
    )
        if X.size == 0 or y.size == 0:
            if self.verbose:
                print("Adjacency predictor: no edge pairs after sampling — skipping training.")
            return

        if self.verbose:
            print(f"Training adjacency matrix predictor on {X.shape[0]} instances with {X.shape[1]} features")
        self.adjacency_matrix_classifier.fit(X, y)

    @timeit
    def fit(
        self,
        graphs: List[nx.Graph],
        node_encodings_list: List[np.ndarray],
        use_edge_fraction_for_adj_mtx: float  # <-- New parameter (unchanged)
    ) -> 'DecompositionalNodeEncoderDecoder':
        """
        Fits the node-, edge- and adjacency-matrix classifiers.

        Instead of repeatedly refitting with warm starts, we build one
        big augmented dataset (original + noisy variants) and train once.

        Parameters
        ----------
        graphs : List[nx.Graph]
            Original graphs.
        node_encodings_list : List[np.ndarray]
            Original node-level embeddings (one array per graph).
        use_edge_fraction_for_adj_mtx : float
            Fraction of edges to sample when training the adjacency-matrix predictor.

        Returns
        -------
        DecompositionalNodeEncoderDecoder
            Self (trained).
        """

        # ------------------------------------------------------------------
        # 1. Build augmented dataset
        # ------------------------------------------------------------------
        combined_graphs: List[nx.Graph] = list(graphs)                 # shallow copy is fine
        combined_encodings: List[np.ndarray] = list(node_encodings_list)

        if self.num_augmentation_iterations > 0:
            noise_list = np.linspace(0, self.augmentation_noise, self.num_augmentation_iterations + 1).tolist()

            for noise in noise_list:
                if self.verbose:
                    print(f'<<Generating noisy encodings (noise={noise})>>')

                for enc in node_encodings_list:
                    perturbed = enc + np.random.rand(*enc.shape) * noise
                    combined_encodings.append(perturbed)

                # Point to the same graphs for every new encoding set
                combined_graphs.extend(graphs)

        # ------------------------------------------------------------------
        # 2. Train once on the enlarged dataset
        # ------------------------------------------------------------------
        self.node_label_classifier_fit(combined_encodings, combined_graphs)
        self.edge_label_classifier_fit(combined_encodings, combined_graphs)
        self.adjacency_matrix_classifier_fit(
            combined_encodings, combined_graphs,
            use_edge_fraction=use_edge_fraction_for_adj_mtx
        )

        return self

    def constrained_node_encodings_list(
        self,
        original_node_encodings_list: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Constrains the node encodings to be non-negative.
        
        Parameters:
            original_node_encodings_list: List of numpy arrays with raw node encodings.
        
        Returns:
            A new list of numpy arrays where all negative values have been set to zero.
        """
        constrained = []
        for encoding in original_node_encodings_list:
            new_enc = encoding.copy()
            # Replace negative values with 0.
            new_enc[new_enc < 0] = 0
            constrained.append(new_enc)
        return constrained

    def get_degrees(
        self,
        encodings: np.ndarray,
        n_nodes: int,
        threshold: float = 0.5
    ) -> List[int]:
        """
        Extracts target degrees from node encodings.
        If a node's existence flag is below the threshold, its target degree is set to 0;
        otherwise, the degree is rounded and enforced to be at least 1.
        
        Parameters:
            encodings: Numpy array of node encodings.
            n_nodes  : Number of nodes in the graph.
            threshold: Float, nodes with an existence value < threshold are considered non-existent.
        
        Returns:
            A list of integer degrees for each node.
        """
        degs = np.rint(encodings[:n_nodes, 1])
        existent = encodings[:n_nodes, 0] >= threshold
        # For existent nodes enforce at least 1, for non-existent nodes set to 0.
        degs = np.where(existent, np.maximum(degs, 1), 0)
        return degs.astype(int).tolist()

    def decode_adjacency_matrix(
        self,
        original_node_encodings_list: List[np.ndarray],
        existence_threshold: float = 0.5
    ) -> List[np.ndarray]:
        """
        Predicts adjacency matrices for a list of node encoding matrices while accounting for node existence.
        Nodes with an existence flag below the threshold will have their incident edge probabilities zeroed out
        and their target degrees set to 0.
        
        Parameters:
            original_node_encodings_list: List of numpy arrays with raw node encodings.
            existence_threshold: Float threshold for determining if a node exists (default 0.5).
        
        Returns:
            List of binary adjacency matrices (numpy arrays) after optimization.
        """
        # Constrain encodings to be non-negative.
        node_encodings_list = self.constrained_node_encodings_list(original_node_encodings_list)
        
        # Calculate the number of instances per graph (all ordered pairs except self-pairs).
        sizes = [enc.shape[0]**2 - enc.shape[0] for enc in node_encodings_list]
        # Generate instances for adjacency matrix prediction.
        X = self.encodings_to_instances(node_encodings_list)
        predicted_probs = self.adjacency_matrix_classifier.predict_proba(X)[:, -1]
        # Split predicted probabilities for each graph based on the sizes computed.
        predicted_probs_list = np.split(predicted_probs, np.cumsum(sizes)[:-1])
        
        adj_mtx_list = []
        # Process each graph's predictions.
        for prob_list, encodings in zip(predicted_probs_list, node_encodings_list):
            n_nodes = encodings.shape[0]
            idx = 0
            prob_matrix = np.zeros((n_nodes, n_nodes))
            # Reconstruct the probability matrix from the flat list.
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        prob_matrix[i, j] = prob_list[idx]
                        idx += 1
            # Zero out probabilities for edges where either node is non-existent.
            existent = encodings[:, 0] >= existence_threshold
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if not (existent[i] and existent[j]):
                        prob_matrix[i, j] = 0
            # Ensure the matrix is symmetric.
            prob_matrix = (prob_matrix + prob_matrix.T) / 2
            # Extract target degrees from encodings (modified to set degree=0 for non-existent nodes).
            target_degrees = self.get_degrees(encodings, n_nodes, threshold=existence_threshold)
            # Optimize the probability matrix into a binary adjacency matrix.
            adj = self.optimize_adjacency_matrix(prob_matrix, target_degrees)
            adj_mtx_list.append(adj)
        return adj_mtx_list

    def decode_node_labels(
        self,
        node_encodings_list: List[np.ndarray]
    ) -> List[np.ndarray]:
        # single-label short-circuit remains
        if hasattr(self, 'single_node_label') and self.single_node_label is not None:
            return [np.array([self.single_node_label] * enc.shape[0]) for enc in node_encodings_list]

        # prefer using feature column 2 when available + mapping provided
        if hasattr(self, "_idx_to_label") and all(enc.shape[1] > 2 for enc in node_encodings_list):
            out = []
            for enc in node_encodings_list:
                ids = enc[:, 2].astype(int)
                out.append(np.array([self._idx_to_label.get(int(i), i) for i in ids], dtype=object))
            return out

        # fallback to classifier (drop potential label-id leakage column)
        encs_drop = [np.delete(enc, 2, axis=1) if enc.shape[1] > 2 else enc for enc in node_encodings_list]
        if len(encs_drop) == 0:
            return []
        X = np.vstack(encs_drop)
        predicted_node_labels = self.node_label_classifier.predict(X)
        sizes = [enc.shape[0] for enc in encs_drop]
        return np.split(predicted_node_labels, np.cumsum(sizes)[:-1])


    def decode_edge_labels(
        self,
        node_encodings_list: List[np.ndarray],
        adj_mtx_list: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Decodes edge labels for each graph using the edge label classifier.
        
        Parameters:
            node_encodings_list: List of numpy arrays containing node encodings.
            adj_mtx_list  : List of binary adjacency matrices.
        
        Returns:
            List of numpy arrays where each array contains predicted edge labels for a graph.
        """
        # If a single edge label exists, return it for all edges.
        if hasattr(self, 'single_edge_label') and self.single_edge_label is not None:
            predicted_edge_labels_list = []
            for adj in adj_mtx_list:
                n_edges = int(np.sum(adj))
                predicted_edge_labels_list.append(np.array([self.single_edge_label] * n_edges))
            return predicted_edge_labels_list

        # Create instances based on encodings and adjacency matrices.
        X = self.encodings_and_adj_mtx_to_edge_dataset(node_encodings_list, adj_mtx_list)
        if len(X) < 1:
            return [[] for _ in node_encodings_list]
        predicted_edge_labels = self.edge_label_classifier.predict(X)
        sizes = [np.sum(adj) for adj in adj_mtx_list]
        predicted_edge_labels_list = np.split(predicted_edge_labels, np.cumsum(sizes)[:-1])
        return predicted_edge_labels_list

    @timeit
    def decode(
        self,
        original_node_encodings_list: List[np.ndarray]
    ) -> List[nx.Graph]:
        """
        Decodes node encodings into complete graphs with predicted node and edge labels, while considering node existence.
        Nodes with an existence flag (first feature) below the threshold are considered non-existent and are removed.
        
        Parameters:
            original_node_encodings_list: List of numpy arrays with raw node encodings.
            
        Returns:
            List of reconstructed NetworkX graph objects with predicted labels and filtered non-existent nodes.
        """
        # Step 1: Decode the adjacency matrices using the modified method that accounts for node existence.
        adj_mtx_list = self.decode_adjacency_matrix(original_node_encodings_list, existence_threshold=self.existence_threshold)
        
        # Step 2: Decode node labels (this method remains unchanged).
        predicted_node_labels_list = self.decode_node_labels(original_node_encodings_list)
        
        # Step 3: Decode edge labels based on the updated adjacency matrices.
        predicted_edge_labels_list = self.decode_edge_labels(original_node_encodings_list, adj_mtx_list)
        
        graphs = []
        # Step 4: Reconstruct each graph and filter out non-existent nodes.
        for encodings, node_labels, edge_labels, adj_mtx in zip(
                original_node_encodings_list, predicted_node_labels_list, predicted_edge_labels_list, adj_mtx_list):
            # Create the initial graph from the predicted adjacency matrix.
            graph = nx.from_numpy_array(adj_mtx)
            
            # Assign node labels. (Assumes ordering in node_labels matches node indices.)
            node_label_map = {i: label for i, label in enumerate(node_labels)}
            nx.set_node_attributes(graph, node_label_map, 'label')
            
            # If edges exist, assign edge labels.
            if np.sum(adj_mtx) > 0:
                n_nodes = graph.number_of_nodes()
                edge_idx = 0
                edge_attr = {}
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        if adj_mtx[i, j] != 0:
                            edge_attr[(i, j)] = edge_labels[edge_idx]
                            edge_idx += 1
                nx.set_edge_attributes(graph, edge_attr, 'label')
            
            # Filter out nodes that do not meet the existence threshold.
            existent_indices = np.where(encodings[:, 0] >= self.existence_threshold)[0]
            # Create a subgraph that includes only existent nodes.
            filtered_graph = graph.subgraph(existent_indices).copy()
            graphs.append(filtered_graph)
        
        return graphs
    
    def save(self, filename: str = 'generative_model.obj') -> None:
        """
        Saves the current instance of the model to a file using pickle.
        
        Parameters:
            filename: The file name or path where the model object will be saved.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def load(self, filename: str = 'generative_model.obj') -> 'DecompositionalNodeEncoderDecoder':
        """
        Loads a model instance from a file.
        
        Parameters:
            filename: The file name or path from which the model object will be loaded.
        
        Returns:
            The loaded model object.
        """
        with open(filename, 'rb') as f:
            self = pickle.load(f)
        return self

# =============================================================================
# ConditionalNodeGeneratorModel Class
# =============================================================================
class ConditionalNodeGeneratorModel(object):
    """
    ConditionalNodeGeneratorModel

    Provides a wrapper around a transformer-based conditional diffusion generator.
    It models node encodings conditioned on graph-level features and offers methods for training, 
    prediction, and sampling of node encoding matrices.
    """
    def __init__(
            self, 
            conditional_node_generator: Optional[ConditionalNodeGeneratorBase] = None,
            verbose: bool = True
            ) -> None:
        """
        Initializes the DecompositionalNodeTransformerConditionalDiffusionModel instance.

        Parameters:
            conditional_node_generator: An instance of ConditionalNodeGeneratorBase
                used for generating node encodings based on conditioning inputs.
            verbose: Boolean flag to enable or disable verbose logging.
        """
        self.conditional_node_generator = copy.deepcopy(conditional_node_generator)
        self.verbose = verbose

    @timeit
    def fit(
        self,
        node_encodings_list: List[np.ndarray],
        conditional_graph_encodings: Any,
        edge_pairs: Optional[List[Tuple[int, int, int]]] = None,
        edge_targets: Optional[np.ndarray] = None,
        node_mask: Optional[np.ndarray] = None,
        edge_label_pairs: Optional[List[Tuple[int,int,int]]] = None,   # NEW
        edge_label_targets: Optional[np.ndarray] = None,
        distance_pairs: Optional[List[Tuple[int,int,int]]] = None,
        distance_targets: Optional[np.ndarray] = None
            # NEW
    ) -> 'ConditionalNodeGeneratorModel':
        if self.verbose:
            print(f"Training conditional model on {len(node_encodings_list)} graphs with {node_encodings_list[0].shape[0]} nodes each.")

        if edge_pairs is not None and edge_targets is not None:
            if self.verbose:
                print(f"Using edge supervision with {len(edge_pairs)} edge pairs.")
            self.conditional_node_generator.setup(
                node_encodings_list=node_encodings_list,
                conditional_graph_encodings=conditional_graph_encodings,
                edge_pairs=edge_pairs,
                edge_targets=edge_targets,
                node_mask=node_mask,
                edge_label_pairs=edge_label_pairs,
                edge_label_targets=edge_label_targets,
                # NEW ↓
                distance_pairs=distance_pairs,
                distance_targets=distance_targets
            )
            self.conditional_node_generator.fit(
                node_encodings_list=node_encodings_list,
                conditional_graph_encodings=conditional_graph_encodings,
                edge_pairs=edge_pairs,
                edge_targets=edge_targets,
                node_mask=node_mask
            )
        else:
            self.conditional_node_generator.setup(
                node_encodings_list=node_encodings_list,
                conditional_graph_encodings=conditional_graph_encodings,
                node_mask=node_mask,
                edge_label_pairs=edge_label_pairs,        # optional
                edge_label_targets=edge_label_targets,    # optional
                distance_pairs=distance_pairs,            # keep
                distance_targets=distance_targets         # keep
            )

            self.conditional_node_generator.fit(
                node_encodings_list=node_encodings_list,
                conditional_graph_encodings=conditional_graph_encodings,
                node_mask=node_mask
            )

        return self

    @timeit
    def predict(
        self,
        conditional_graph_encodings: Any,
        desired_class: Optional[Union[int, Sequence[int]]] = None
    ) -> List[np.ndarray]:
        if self.verbose:
            print(f"Predicting node matrices for {len(conditional_graph_encodings)} graphs...")
        predicted_node_encodings_list = self.conditional_node_generator.predict(conditional_graph_encodings, desired_class=desired_class)
        return predicted_node_encodings_list
    
# =============================================================================
# DecompositionalEncoderDecoder Class 
# =============================================================================

class DecompositionalEncoderDecoder(object):
    """
    DecompositionalEncoderDecoder

    Integrates a full encoder-decoder pipeline that maps graphs to conditioned node embeddings
    and reconstructs graphs from these embeddings. It combines graph vectorization, node encoding,
    and conditional generation to support tasks such as training, unconditional sampling, conditional
    sampling, interpolation, and mean graph computation.
    """
    def __init__(
            self,
            graph_vectorizer: Any = None,
            node_graph_vectorizer: Any = None,
            conditioning_to_node_embeddings_generator: Optional[ConditionalNodeGeneratorModel] = None,
            node_embeddings_to_graph_generator: Optional[DecompositionalNodeEncoderDecoder] = None,
            verbose: bool = True,
            use_edge_supervision: bool = False,
            use_edge_fraction: float = 1.0,
            use_distance_supervision: bool = True
            ) -> None:
        """
        Initializes the DecompositionalEncoderDecoder instance.

        Parameters:
            ...
            graph_vectorizer: Object responsible for converting graphs into global conditioning vectors.
            node_graph_vectorizer: Object responsible for encoding individual graph nodes.
            conditioning_to_node_embeddings_generator: Generator that maps conditioning vectors to node embeddings.
            node_embeddings_to_graph_generator: Generator that reconstructs graphs from node embeddings.
            verbose: Boolean flag to enable or disable verbose logging.
            use_edge_supervision: Whether to use edge supervision during training.
            use_edge_fraction: Fraction of edges to use for supervision (default=1.0).
        """
        self.graph_vectorizer = graph_vectorizer
        self.node_graph_vectorizer = node_graph_vectorizer
        self.conditioning_to_node_embeddings_generator = conditioning_to_node_embeddings_generator
        self.node_embeddings_to_graph_generator = node_embeddings_to_graph_generator
        self.verbose = verbose
        self.use_edge_supervision = use_edge_supervision
        if not 0.0 < use_edge_fraction <= 1.0:
            raise ValueError("use_edge_fraction must be between 0.0 (exclusive) and 1.0 (inclusive)")
        self.use_edge_fraction = use_edge_fraction
        self.use_distance_supervision = use_distance_supervision

    def toggle_verbose(self) -> None:
        """
        Toggles verbosity on all sub-components.
        """
        self.verbose = not self.verbose
        if self.conditioning_to_node_embeddings_generator is not None:
            self.conditioning_to_node_embeddings_generator.verbose = self.verbose
        if self.node_embeddings_to_graph_generator is not None:
            self.node_embeddings_to_graph_generator.verbose = self.verbose

    @timeit
    def fit(
        self,
        graphs: List[nx.Graph],
        train_conditioning_to_node_embeddings_generator: bool = True,
        train_node_embeddings_to_graph_generator: bool = True
    ) -> 'DecompositionalEncoderDecoder':
        if self.verbose:
            print(f"Fitting model on {len(graphs)} graphs")

        # Fit vectorizers
        self.graph_vectorizer.fit(graphs)
        self.node_graph_vectorizer.fit(graphs)

        # Generate encodings
        node_encodings_list, conditional_graph_encodings = self.encode(graphs)
        self._train_cond_encs = list(conditional_graph_encodings)  # add this
        if train_conditioning_to_node_embeddings_generator:


            edge_pairs_for_cond_gen = None
            edge_targets_for_cond_gen = None
            edge_label_pairs_for_cond_gen = None
            edge_label_targets_for_cond_gen = None
            node_mask_for_cond_gen = None


            if self.use_edge_supervision:
                if self.verbose:
                    print("Using edge supervision for training the conditioning to node embeddings generator.")
                edge_targets_for_cond_gen, edge_pairs_for_cond_gen = \
                    self.node_embeddings_to_graph_generator.compute_edge_supervision(
                        graphs, node_encodings_list, use_edge_fraction=self.use_edge_fraction
                    )
                edge_label_targets_for_cond_gen, edge_label_pairs_for_cond_gen, _ = \
                    self.node_embeddings_to_graph_generator.compute_edge_label_supervision(graphs)
                    
            if self.use_distance_supervision:
                   dist_targets_for_cond_gen, dist_pairs_for_cond_gen, _ = compute_hop_supervision(graphs)
            else:
                   dist_targets_for_cond_gen, dist_pairs_for_cond_gen = None, None

            self.conditioning_to_node_embeddings_generator.fit(
                node_encodings_list=node_encodings_list,
                conditional_graph_encodings=conditional_graph_encodings,
                edge_pairs=edge_pairs_for_cond_gen,
                edge_targets=edge_targets_for_cond_gen,
                node_mask=node_mask_for_cond_gen,
                edge_label_pairs=edge_label_pairs_for_cond_gen,          # NEW
                edge_label_targets=edge_label_targets_for_cond_gen,       # NEW
                distance_pairs=dist_pairs_for_cond_gen,
                distance_targets=dist_targets_for_cond_gen
            )


        if train_node_embeddings_to_graph_generator:
            self.node_embeddings_to_graph_generator.fit(graphs, node_encodings_list, use_edge_fraction_for_adj_mtx=self.use_edge_fraction)

        return self

    @timeit
    def node_encode(self, graphs: List[nx.Graph]) -> List[np.ndarray]:
        """
        Node-level embeddings for each graph.
        """
        if self.verbose:
            print(f"Node encoding {len(graphs)} graphs")
        return self.node_graph_vectorizer.transform(graphs)

    @timeit
    def graph_encode(self, graphs: List[nx.Graph]) -> List[np.ndarray]:
        """
        Global conditioning vectors for each graph.
        """
        if self.verbose:
            print(f"Encoding {len(graphs)} graphs")
        return self.graph_vectorizer.transform(graphs)

    def encode(self, graphs: List[nx.Graph]) -> Tuple[List[np.ndarray], Any]:
        node_encs = self.node_encode(graphs)

        # write NX node labels → feature column 2 (as ints in float dtype)
        node_encs, label_to_idx = inject_labels_from_graphs(
            node_encs, graphs,
            label_index=2,
            label_to_idx=getattr(self, "_label_to_idx", None),
        )
        # keep mapping for decode
        self._label_to_idx = label_to_idx
        self._idx_to_label = {v: k for k, v in label_to_idx.items()}

        # also expose mapping to the graph decoder so it can decode from col 2
        if self.node_embeddings_to_graph_generator is not None:
            self.node_embeddings_to_graph_generator._idx_to_label = self._idx_to_label

        cond_encs = self.graph_encode(graphs)
        return node_encs, cond_encs
    
    def _sample_conditions(self, n_samples: int):
        """
        Sample conditioning vectors from the empirical pool collected during fit().
        """
        if not hasattr(self, "_train_cond_encs") or len(self._train_cond_encs) == 0:
            raise RuntimeError("No conditioning vectors available. Call fit() before sample().")
        idx = np.random.choice(len(self._train_cond_encs), size=n_samples, replace=True)
        return [self._train_cond_encs[i] for i in idx]
    

    def decode(self, conditioning_vectors: Any, desired_class: Optional[Union[int, Sequence[int]]] = None) -> List[nx.Graph]:
        """
        Decode conditioning vectors to graphs via node embeddings.

        Parameters
        ----------
        conditioning_vectors : Any
            The conditioning vectors to decode
        desired_class : Optional[Union[int, Sequence[int]]], default=None
            If provided, guides the generation toward the specified class(es)
            using classifier guidance.
        
        Returns
        -------
        List[nx.Graph]
            The decoded graphs
        """
        if self.verbose:
            print(f"Decoding {len(conditioning_vectors)} conditioning vectors")
            if desired_class is not None:
                print(f"Using classifier guidance toward class(es): {desired_class}")
        
        node_feats = self.conditioning_to_node_embeddings_generator.predict(
            conditioning_vectors, desired_class=desired_class
        )
        return self.node_embeddings_to_graph_generator.decode(node_feats)

    @timeit
    def sample(self, n_samples: int = 1, desired_class: Optional[Union[int, Sequence[int]]] = None) -> List[nx.Graph]:
        """
        Unconditional sampling: cond->node_embeddings->graphs.

        Parameters
        ----------
        n_samples : int, default=1
            Number of samples to generate
        desired_class : Optional[Union[int, Sequence[int]]], default=None
            If provided, guides the generation toward the specified class(es)
            using classifier guidance.
        """
        if self.verbose:
            print(f"Sampling {n_samples} graphs")
            if desired_class is not None:
                print(f"Using classifier guidance toward class(es): {desired_class}")
        
        node_feats = self.conditioning_to_node_embeddings_generator.predict(
            self._sample_conditions(n_samples), desired_class=desired_class
        )
        return self.node_embeddings_to_graph_generator.decode(node_feats)

    @timeit
    def conditional_sample(
        self,
        graphs: List[nx.Graph],
        n_samples: int = 1,
        desired_class: Optional[Union[int, Sequence[int]]] = None
    ) -> List[List[nx.Graph]]:
        """
        Conditional sampling: graphs->cond_encs->y_samples->node_feats->graphs.

        Parameters
        ----------
        graphs : List[nx.Graph]
            Input graphs to condition on
        n_samples : int, default=1
            Number of samples per input graph
        desired_class : Optional[Union[int, Sequence[int]]], default=None
            If provided, guides the generation toward the specified class(es)
            using classifier guidance.
        """
        _, cond_encs = self.encode(graphs)
        cond_encs = [[cond_enc]*n_samples for cond_enc in cond_encs]
        
        results = []
        for i in range(len(graphs)):
            y_i = cond_encs[i]
            node_feats = self.conditioning_to_node_embeddings_generator.predict(
                y_i, desired_class=desired_class
            )
            decoded = self.node_embeddings_to_graph_generator.decode(node_feats)
            results.append(decoded)
        
        return results

    def sample_from(self, graphs, n_samples=1):
        sampled_seed_graphs = random.choices(graphs, k=n_samples)
        reconstructed_graphs_list = self.conditional_sample(sampled_seed_graphs, n_samples=1)
        sampled_graphs = [reconstructed_graphs[0] for reconstructed_graphs in reconstructed_graphs_list]
        return sampled_graphs

    def interpolate(
        self,
        G1: nx.Graph,
        G2: nx.Graph,
        n_steps: int = 10,
        t_start: float = 0.0,
        t_end: float = 1.0
    ) -> List[nx.Graph]:
        """
        Interpolate between G1 and G2 in specified mode.
        """
        ts = np.linspace(t_start, t_end, n_steps)
        results = []
        emb1 = self.graph_vectorizer.transform([G1])[0]
        emb2 = self.graph_vectorizer.transform([G2])[0]
        for t in ts:
            emb_t = scaled_slerp(emb1, emb2, t)
            results.append(self.decode([emb_t])[0])
        
        return results

    def mean(
        self,
        graphs: List[nx.Graph]
    ) -> nx.Graph:
        """
        Compute a centroid graph from a list of input graphs.
        """
        Y = np.vstack(self.graph_vectorizer.transform(graphs))
        centroid = scaled_slerp_average(Y)
        return self.decode([centroid])[0]
    
    @timeit
    def fit_classifier(self, graphs, targets, epochs=20, lr=1e-3):
        """
        Fits a classifier for conditional guidance based on provided graphs and targets.
        Automatically sets up the guidance classifier and plots training/validation loss.
        """
        # --- Step 1: Encode inputs ---
        node_encs = self.node_graph_vectorizer.transform(graphs)  # List of node arrays
        cond_vecs = self.graph_vectorizer.transform(graphs)       # 2D array

        # --- Step 2: Infer number of classes ---
        targets_np = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else np.array(targets)
        num_classes = int(np.max(targets_np)) + 1

        # --- Step 3: Access underlying model ---
        model = self.conditioning_to_node_embeddings_generator.conditional_node_generator.model

        # --- Step 4: Ensure guidance classifier is initialized correctly ---
        if not hasattr(model, "guidance_classifier") or model.guidance_classifier is None:
            model.set_guidance_classifier(num_classes)
        else:
            try:
                current_dim = model.guidance_classifier.net[-1].out_features
            except AttributeError:
                current_dim = None
            if current_dim != num_classes:
                print(f"Resetting guidance classifier (was {current_dim}, now {num_classes})")
                model.set_guidance_classifier(num_classes)

        # --- Step 5: Train the classifier with internal validation and plot ---
        model.train_guidance_classifier(
            node_feats=node_encs,
            cond_vecs=cond_vecs,
            labels=targets_np,
            epochs=epochs,
            lr=lr,
            verbose=self.verbose
        )