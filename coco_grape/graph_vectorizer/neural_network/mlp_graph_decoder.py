from coco_grape.graph_vectorizer.neural_network.io_xact_gnn import GraphEncoder
from itertools import combinations
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Union, Dict, Any
import cvxpy as cp  # Ensure cvxpy is installed
import datetime
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


def extract_node_degrees(graphs: List[nx.Graph]) -> List[np.ndarray]:
    return [np.array([G.degree(n) for n in sorted(G.nodes())], dtype=np.int32) for G in graphs]


class NetworkxGraphDecoder(BaseEstimator, TransformerMixin):
    """
    NetworkxGraphDecoder is a scikit-learn compatible transformer that decodes node embeddings
    into NetworkX graphs with predicted adjacency matrices, node attributes, and edge attributes.

    The class processes the output from the MLPGraphDecoder's transform method along with node
    degree information and reconstructs the corresponding NetworkX graphs by mapping label
    indices back to string labels and incorporating 'vec' attributes if present. It utilizes
    the node degree information to determine the most probable connections using exact b-matching.

    Parameters
    ----------
    graph_encoder : Optional[GraphEncoder], default=None
        An optional instance of the fitted GraphEncoder containing label encoders and vector dimensions.
        If provided, the NetworkxGraphDecoder will initialize its label decoders and vector dimensions
        based on the fitted GraphEncoder. If not provided, these will need to be initialized during the fit phase.
    verbose : int, default=1
        Verbosity level:
            - 0: No output.
            - 1: Basic information.
            - 2: Detailed information for debugging.
            - 3: Extensive debugging information.
    enforce_connectedness : bool, default=True
        If True, ensures that the reconstructed graph is connected by first computing a
        Degree-Constrained Spanning Tree (DCST) based on the predicted adjacency probabilities and
        including these edges in the final graph.
    """
    def __init__(
        self,
        graph_encoder: Optional[GraphEncoder] = None,
        verbose: int = 1,
        enforce_connectedness: bool = True
    ):
        self.verbose = verbose
        self.enforce_connectedness = enforce_connectedness

        if graph_encoder is not None:
            if not hasattr(graph_encoder, 'is_fitted_') or not graph_encoder.is_fitted_:
                raise ValueError("The provided GraphEncoder instance must be fitted before initializing NetworkxGraphDecoder.")

            # Initialize label decoders from the fitted GraphEncoder's label encoders
            self.node_label_decoder = list(graph_encoder.node_label_encoder.classes_)
            self.edge_label_decoder = list(graph_encoder.edge_label_encoder.classes_)

            # Set label sizes based on the decoders
            self.node_label_size = len(self.node_label_decoder)
            self.edge_label_size = len(self.edge_label_decoder)

            # Set vector dimensions based on the GraphEncoder
            self.node_vec_dim = graph_encoder.node_vec_dim
            self.edge_vec_dim = graph_encoder.edge_vec_dim

            if self.verbose >= 1:
                print("Initialized NetworkxGraphDecoder with label decoders from GraphEncoder.")
                print(f"Node Label Decoder: {self.node_label_decoder}")
                print(f"Edge Label Decoder: {self.edge_label_decoder}")
                print(f"Node 'vec' Dimension: {self.node_vec_dim}")
                print(f"Edge 'vec' Dimension: {self.edge_vec_dim}")
                print(f"Enforce Connectedness: {self.enforce_connectedness}")

            self.is_fitted_ = True
        else:
            # Decoders and vector dimensions will be initialized during the fit phase
            self.node_label_decoder = None
            self.edge_label_decoder = None
            self.node_label_size = 0
            self.edge_label_size = 0
            self.node_vec_dim = 0
            self.edge_vec_dim = 0

            if self.verbose >= 1:
                print("Initialized NetworkxGraphDecoder without a pre-fitted GraphEncoder.")
                print("Label decoders and vector dimensions will be set during the fit phase.")

            self.is_fitted_ = False


    def fit(self, graphs: List[nx.Graph], y=None):
        """
        Fit the NetworkxGraphDecoder by extracting label alphabets and vector dimensions
        from the provided NetworkX graphs.

        Parameters
        ----------
        graphs : List[networkx.Graph]
            List of NetworkX graphs to fit the decoder.

        y : Ignored
            Included for compatibility with scikit-learn's TransformerMixin.

        Returns
        -------
        self : NetworkxGraphDecoder
            Fitted transformer.
        """
        if self.verbose >= 1:
            print("Fitting NetworkxGraphDecoder...")

        node_labels = set()
        edge_labels = set()
        node_vec_dims = []
        edge_vec_dims = []

        for idx, G in enumerate(graphs):
            # Collect node labels and vec dimensions
            for node, attrs in G.nodes(data=True):
                label = attrs.get('label')
                if label is not None:
                    node_labels.add(label)
                vec = attrs.get('vec')
                if vec is not None:
                    node_vec_dims.append(len(vec))
            # Collect edge labels and vec dimensions
            for u, v, attrs in G.edges(data=True):
                label = attrs.get('label')
                if label is not None:
                    edge_labels.add(label)
                vec = attrs.get('vec')
                if vec is not None:
                    edge_vec_dims.append(len(vec))

        # Determine node and edge vector dimensions
        self.node_vec_dim = max(node_vec_dims) if node_vec_dims else 0
        self.edge_vec_dim = max(edge_vec_dims) if edge_vec_dims else 0

        # Sort labels to have consistent ordering
        self.node_label_decoder = sorted(list(node_labels)) if self.node_label_decoder is None else self.node_label_decoder
        self.edge_label_decoder = sorted(list(edge_labels)) if self.edge_label_decoder is None else self.edge_label_decoder

        self.node_label_size = len(self.node_label_decoder)
        self.edge_label_size = len(self.edge_label_decoder)

        if self.verbose >= 1:
            print(f"Found {self.node_label_size} unique node labels.")
            print(f"Found {self.edge_label_size} unique edge labels.")
            print(f"Node 'vec' dimension: {self.node_vec_dim}")
            print(f"Edge 'vec' dimension: {self.edge_vec_dim}")
            print(f"Enforce connectedness: {self.enforce_connectedness}")

        self.is_fitted_ = True
        return self

    def _degree_constrained_spanning_tree(
        self, 
        adj_probs: np.ndarray, 
        edge_indices: np.ndarray, 
        degrees: np.ndarray,
        active_nodes: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Compute a Degree-Constrained Spanning Tree (DCST) using a greedy heuristic, excluding nodes with degree 0.

        Parameters
        ----------
        adj_probs : np.ndarray
            Predicted adjacency matrix with probabilities.
        edge_indices : np.ndarray
            Array of edge endpoint indices.
        degrees : np.ndarray
            Desired degrees for each node.
        active_nodes : List[int]
            List of node indices that can have edges (desired degree >=1).

        Returns
        -------
        dcst_edges : List[Tuple[int, int]]
            List of edges constituting the DCST.
        """
        n_nodes = adj_probs.shape[0]
        # Create a list of edges with weights, excluding edges involving inactive nodes
        edges_with_weights = [
            (i, j, adj_probs[i, j]) 
            for i, j in edge_indices 
            if i < j and i in active_nodes and j in active_nodes  # Exclude edges with inactive nodes
        ]
        # Sort edges by descending weight
        edges_sorted = sorted(edges_with_weights, key=lambda x: x[2], reverse=True)

        dcst = nx.Graph()
        dcst.add_nodes_from(active_nodes)
        dcst_edges = []

        # Initialize remaining degrees
        remaining_degrees = degrees[active_nodes].copy()

        # Map node index to its position in remaining_degrees
        node_to_degree = {node: remaining_degrees[idx] for idx, node in enumerate(active_nodes)}

        for u, v, weight in edges_sorted:
            if not dcst.has_edge(u, v):
                # Check if adding this edge violates degree constraints
                if node_to_degree[u] > 0 and node_to_degree[v] > 0:
                    # Check if adding the edge would create a cycle
                    if not nx.has_path(dcst, u, v) or len(dcst.edges()) == len(active_nodes) - 1:
                        dcst.add_edge(u, v, weight=weight)
                        dcst_edges.append((u, v))
                        node_to_degree[u] -= 1
                        node_to_degree[v] -= 1
                        if self.verbose >= 3:
                            print(f"DCST: Added edge ({u}, {v}) with weight {weight:.4f}. Remaining degrees: {u}: {node_to_degree[u]}, {v}: {node_to_degree[v]}")
                        # Check if DCST is connected and spans all active nodes
                        if nx.is_connected(dcst) and len(dcst.edges()) >= len(active_nodes) - 1:
                            break

        if not nx.is_connected(dcst):
            warnings.warn("Degree-Constrained Spanning Tree could not connect all active nodes with the given degree constraints.")

        return dcst_edges

    def transform(
        self,
        X: Tuple[
            List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
            List[np.ndarray]
        ],
        y=None
    ) -> List[nx.Graph]:
        """
        Transform the decoded outputs from MLPGraphDecoder and node degrees into NetworkX graphs.

        Parameters
        ----------
        X : Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], List[np.ndarray]]
            A tuple containing:
                1. decoded_outputs: List of tuples where each tuple contains:
                    - Predicted adjacency matrix (n x n) with probabilities.
                    - Node label predictions (n x node_attribute_size).
                    - Edge label predictions (num_edges x edge_attribute_size).
                    - Edge endpoint indices (num_edges x 2) where each row is [src_idx, dest_idx].
                2. degrees_per_graph: List of one-dimensional NumPy arrays where each array
                   contains the number of edges for each node in the corresponding graph.

        y : Ignored
            Included for compatibility with scikit-learn's TransformerMixin.

        Returns
        -------
        graphs : List[networkx.Graph]
            List of reconstructed NetworkX graphs with predicted attributes.
        """
        if not self.is_fitted_:
            raise RuntimeError("NetworkxGraphDecoder must be fitted before calling transform.")

        decoded_outputs, degrees_per_graph = X

        if self.verbose >= 1:
            print("Transforming decoded outputs into NetworkX graphs using b-matching...")

        if len(decoded_outputs) != len(degrees_per_graph):
            raise ValueError("The number of decoded_outputs must match the number of degrees_per_graph.")

        reconstructed_graphs = []

        for graph_idx, ((adj_probs, node_preds, edge_preds, edge_indices), degrees) in enumerate(zip(decoded_outputs, degrees_per_graph)):
            if self.verbose >= 2:
                print(f"Processing graph {graph_idx + 1}/{len(decoded_outputs)}...")

            n_nodes = adj_probs.shape[0]
            G = nx.Graph()

            # Identify active and inactive nodes
            active_nodes = [node for node in range(n_nodes) if degrees[node] > 0]
            inactive_nodes = [node for node in range(n_nodes) if degrees[node] == 0]

            if self.verbose >= 3:
                print(f"Active nodes (degree >=1): {active_nodes}")
                print(f"Inactive nodes (degree=0): {inactive_nodes}")

            # Initialize remaining degrees as a copy of degrees
            remaining_degrees = degrees.copy()

            # Process node predictions
            for node_idx in range(n_nodes):
                node_attr = {}
                # Extract label one-hot and vec
                label_one_hot = node_preds[node_idx, :self.node_label_size]
                vec = node_preds[node_idx, self.node_label_size:] if self.node_vec_dim > 0 else None

                # Decode label based on argmax without strict one-hot check
                label_index = np.argmax(label_one_hot)
                # Directly assign the label based on the highest score
                if label_index < self.node_label_size:
                    label_str = self.node_label_decoder[label_index]
                else:
                    label_str = 'unknown'

                node_attr['label'] = label_str
                if self.node_vec_dim > 0:
                    node_attr['vec'] = vec.tolist()

                G.add_node(node_idx, **node_attr)
                if self.verbose >= 3:
                    print(f"Added node {node_idx} with attributes: {node_attr}")

            # If enforce_connectedness is True, compute and add the Degree-Constrained Spanning Tree
            dcst_edges = []
            if self.enforce_connectedness and len(active_nodes) > 1:
                if self.verbose >= 2:
                    print("Enforcing connectedness by computing Degree-Constrained Spanning Tree (DCST)...")
                try:
                    # Compute DCST using the helper function
                    dcst_edges = self._degree_constrained_spanning_tree(adj_probs, edge_indices, remaining_degrees, active_nodes)

                    if self.verbose >= 3:
                        print(f"DCST has {len(dcst_edges)} edges.")

                    # Add DCST edges to the graph
                    for u, v in dcst_edges:
                        edge_attr = {}
                        # Assuming that edge_preds contains all possible edges, find the corresponding edge prediction
                        # Find the index of the edge in edge_indices
                        mask = ((edge_indices[:, 0] == u) & (edge_indices[:, 1] == v)) | ((edge_indices[:, 0] == v) & (edge_indices[:, 1] == u))
                        edge_idx_candidates = np.where(mask)[0]
                        if len(edge_idx_candidates) == 0:
                            warnings.warn(f"DCST edge ({u}, {v}) not found in edge_preds. Assigning default attributes.")
                            label_str = 'unknown'
                            edge_attr['label'] = label_str
                            if self.edge_vec_dim > 0:
                                edge_attr['vec'] = [0.0] * self.edge_vec_dim
                        else:
                            edge_idx = edge_idx_candidates[0]
                            edge_pred = edge_preds[edge_idx]
                            # Extract label one-hot and vec
                            label_one_hot = edge_pred[:self.edge_label_size]
                            vec = edge_pred[self.edge_label_size:] if self.edge_vec_dim > 0 else None

                            # Decode label based on argmax without strict one-hot check
                            label_index = np.argmax(label_one_hot)
                            if label_index < self.edge_label_size:
                                label_str = self.edge_label_decoder[label_index]
                            else:
                                label_str = 'unknown'

                            edge_attr['label'] = label_str
                            if self.edge_vec_dim > 0:
                                edge_attr['vec'] = vec.tolist()

                        G.add_edge(u, v, **edge_attr)
                        if self.verbose >= 3:
                            print(f"Added DCST edge ({u}, {v}) with attributes: {edge_attr}")

                        # Update remaining degrees
                        remaining_degrees[u] -= 1
                        remaining_degrees[v] -= 1

                except Exception as e:
                    warnings.warn(f"Failed to compute DCST for graph {graph_idx +1}: {e}")
                    # Proceed without enforcing connectedness

            # Handle inactive nodes: Ensure they have no edges
            for node in inactive_nodes:
                if G.degree(node) != 0:
                    warnings.warn(f"Node {node} is supposed to have degree 0 but has degree {G.degree(node)}.")
                    # Remove any edges connected to this node
                    connected_edges = list(G.edges(node))
                    G.remove_edges_from(connected_edges)
                    if self.verbose >= 3:
                        print(f"Removed edges connected to node {node} to enforce degree 0.")

            # Validate remaining degrees
            if np.any(remaining_degrees < 0):
                warnings.warn(f"Negative remaining degrees after adding DCST for graph {graph_idx +1}. Check degree constraints.")
                # Reset remaining degrees to zero where negative
                remaining_degrees = np.maximum(remaining_degrees, 0)

            # Extract edge probabilities and corresponding indices excluding DCST edges and any edges involving inactive nodes
            if self.enforce_connectedness and dcst_edges:
                # Create a set of DCST edge tuples for exclusion
                dcst_edge_set = set()
                for u, v in dcst_edges:
                    dcst_edge_set.add((u, v))
                    dcst_edge_set.add((v, u))  # Since the graph is undirected

                # Filter out DCST edges and any edges involving inactive nodes from edge_indices and edge_preds
                mask = np.array([
                    not ((u, v) in dcst_edge_set or (v, u) in dcst_edge_set) and
                    u in active_nodes and v in active_nodes
                    for u, v in edge_indices
                ])
                filtered_edge_indices = edge_indices[mask]
                filtered_edge_preds = edge_preds[mask]
            else:
                # Exclude edges involving inactive nodes
                mask = np.array([
                    u in active_nodes and v in active_nodes
                    for u, v in edge_indices
                ])
                filtered_edge_indices = edge_indices[mask]
                filtered_edge_preds = edge_preds[mask]

            if self.verbose >= 3:
                print(f"Number of possible edges after DCST and inactive nodes exclusion: {filtered_edge_indices.shape[0]}")

            # If no remaining degrees, skip b-matching
            if np.sum(remaining_degrees) == 0:
                if self.verbose >= 3:
                    print("No remaining degrees to satisfy after adding DCST.")
                reconstructed_graphs.append(G)
                continue

            # Define CVXPY variables
            num_possible_edges = filtered_edge_indices.shape[0]
            if num_possible_edges == 0:
                if self.verbose >= 2:
                    print("No possible edges remaining for b-matching.")
                reconstructed_graphs.append(G)
                continue

            x = cp.Variable(num_possible_edges, boolean=True)

            # Define the objective: maximize sum of probabilities * x
            probabilities = adj_probs[filtered_edge_indices[:, 0], filtered_edge_indices[:, 1]]
            objective = cp.Maximize(cp.sum(cp.multiply(probabilities, x)))

            # Define the constraints:
            # For each node, the sum of edges incident to it equals the remaining degree
            constraints = []
            for node in active_nodes:
                desired_degree = remaining_degrees[node]
                if desired_degree < 0:
                    # This should not happen; degrees have been adjusted after DCST
                    desired_degree = 0
                # Find indices of edges incident to the current node
                incident_edges = np.where(
                    (filtered_edge_indices[:, 0] == node) | (filtered_edge_indices[:, 1] == node)
                )[0]
                constraints.append(cp.sum(x[incident_edges]) == desired_degree)

            # Formulate the problem
            problem = cp.Problem(objective, constraints)

            # Solve the problem
            try:
                problem.solve(solver=cp.GLPK_MI, verbose=(self.verbose >= 3))
            except cp.error.SolverError as e:
                warnings.warn(f"Solver failed for graph {graph_idx +1}: {e}")
                reconstructed_graphs.append(G)
                continue

            if x.value is None:
                warnings.warn(f"No solution found for graph {graph_idx +1}.")
                reconstructed_graphs.append(G)
                continue

            # Retrieve selected edges
            selected_edge_indices = np.where(x.value >= 0.99)[0]
            selected_edges = filtered_edge_indices[selected_edge_indices]  # [num_selected_edges, 2]

            # Reconstruct (i,j) ordering based on edge_indices
            # Create a mapping from (i,j) to edge_pred
            edge_map = {}
            for idx in range(filtered_edge_indices.shape[0]):
                i, j = filtered_edge_indices[idx]
                edge_map[(i, j)] = filtered_edge_preds[idx]

            # Iterate over selected edges and add to graph
            for edge in selected_edges:
                i, j = edge
                edge_pred = edge_map.get((i, j))
                if edge_pred is None:
                    # Try the reverse direction
                    edge_pred = edge_map.get((j, i))
                if edge_pred is None:
                    warnings.warn(f"Edge ({i}, {j}) not found in edge_preds. Skipping.")
                    continue
                edge_attr = {}
                # Extract label one-hot and vec
                label_one_hot = edge_pred[:self.edge_label_size]
                vec = edge_pred[self.edge_label_size:] if self.edge_vec_dim > 0 else None

                # Decode label based on argmax without strict one-hot check
                label_index = np.argmax(label_one_hot)
                if label_index < self.edge_label_size:
                    label_str = self.edge_label_decoder[label_index]
                else:
                    label_str = 'unknown'

                edge_attr['label'] = label_str
                if self.edge_vec_dim > 0:
                    edge_attr['vec'] = vec.tolist()

                G.add_edge(i, j, **edge_attr)
                if self.verbose >= 3:
                    print(f"Added edge ({i}, {j}) with attributes: {edge_attr}")

            # Final validation: Check if all node degrees are satisfied
            actual_degrees = np.array([G.degree(node) for node in range(n_nodes)])
            if not np.array_equal(actual_degrees, degrees):
                warnings.warn(
                    f"Node degrees mismatch for graph {graph_idx +1}: "
                    f"expected {degrees}, got {actual_degrees}"
                )
                if self.verbose >= 3:
                    print(f"Expected degrees: {degrees}")
                    print(f"Actual degrees: {actual_degrees}")

            reconstructed_graphs.append(G)
            if self.verbose >= 3:
                print(f"Graph {graph_idx +1} reconstructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

        if self.verbose >= 1:
            print("All decoded graphs have been reconstructed using b-matching with connectedness enforced.")

        return reconstructed_graphs


def create_residual_mlp(
    num_layers: int,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    dropout_rate: float
) -> nn.Module:
    """
    Creates an MLP with residual connections, layer normalization, dropout, and Leaky ReLU activations.

    Parameters:
    ----------
    num_layers : int
        Number of linear layers in the MLP.
    input_dim : int
        Dimension of the input features.
    output_dim : int
        Dimension of the output features.
    hidden_dim : int
        Dimension of the hidden layers.
    dropout_rate : float
        Dropout probability.

    Returns:
    -------
    nn.Module
        An MLP model with the specified architecture.
    """
    
    class ResidualMLP(nn.Module):
        """
        ResidualMLP is a custom MLP module with residual connections, layer normalization,
        dropout, and Leaky ReLU activations.

        Attributes:
        ----------
        layers : nn.ModuleList
            List of layers comprising the MLP.
        use_residual : bool
            Flag indicating whether to use residual connections.
        """

        def __init__(self, num_layers: int, input_dim: int, output_dim: int, hidden_dim: int, dropout_rate: float):
            """
            Initializes the ResidualMLP module.

            Parameters:
            ----------
            num_layers : int
                Number of linear layers in the MLP.
            input_dim : int
                Dimension of the input features.
            output_dim : int
                Dimension of the output features.
            hidden_dim : int
                Dimension of the hidden layers.
            dropout_rate : float
                Dropout probability.
            """
            super(ResidualMLP, self).__init__()
            self.num_layers = num_layers
            self.layers = nn.ModuleList()
            
            for layer_idx in range(num_layers):
                # Determine input and output dimensions
                if layer_idx == 0:
                    layer_input_dim = input_dim
                elif layer_idx == num_layers - 1:
                    layer_input_dim = hidden_dim
                else:
                    layer_input_dim = hidden_dim
                
                if layer_idx == num_layers - 1:
                    layer_output_dim = output_dim
                else:
                    layer_output_dim = hidden_dim
                
                # Define the layer components
                linear = nn.Linear(layer_input_dim, layer_output_dim)
                layer_norm = nn.LayerNorm(layer_output_dim)
                activation = nn.LeakyReLU()
                dropout = nn.Dropout(dropout_rate)
                
                # Append to layers list
                self.layers.append(nn.ModuleList([linear, layer_norm, activation, dropout]))
            
            # Use residual connections only if input_dim equals hidden_dim and there are multiple layers
            self.use_residual = (input_dim == hidden_dim) and (num_layers > 1)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through the ResidualMLP.

            Parameters:
            ----------
            x : torch.Tensor
                Input tensor of shape [batch_size, input_dim].

            Returns:
            -------
            torch.Tensor
                Output tensor of shape [batch_size, output_dim].
            """
            for layer_idx, layer in enumerate(self.layers):
                linear, layer_norm, activation, dropout = layer
                residual = x if self.use_residual else 0
                out = linear(x)
                out = layer_norm(out)
                out = activation(out)
                out = dropout(out)
                if self.use_residual and layer_idx < self.num_layers - 1:
                    out = out + residual
                x = out
            return x
    
    return ResidualMLP(num_layers, input_dim, output_dim, hidden_dim, dropout_rate)




# Assuming the ResidualMLP and create_residual_mlp have been defined as above.

# -----------------------------
# Custom Dataset and Collate Function
# -----------------------------
class GraphDataset(Dataset):
    """
    Custom Dataset for handling graph data in a multi-task learning setup.
    Each sample corresponds to one graph and contains adjacency data, node data, and edge data.

    Attributes:
    ----------
    node_embeddings : List[np.ndarray]
        List of node embeddings for each graph.
    graphs : List[nx.Graph]
        List of NetworkX graphs corresponding to the node embeddings.
    graph_encoder : GraphEncoder
        Instance of GraphEncoder used for encoding labels.
    verbose : int
        Verbosity level.
    """

    def __init__(self, node_embeddings: List[np.ndarray], graphs: List[nx.Graph], graph_encoder: 'GraphEncoder', verbose: int =1):
        """
        Initializes the GraphDataset.

        Parameters:
        ----------
        node_embeddings : List[np.ndarray]
            List of numpy arrays, each of shape [num_nodes, embedding_dim].
        graphs : List[nx.Graph]
            List of NetworkX graphs corresponding to the node embeddings.
        graph_encoder : GraphEncoder
            An instance of GraphEncoder used for encoding labels.
        verbose : int, optional (default=1)
            Verbosity level.
        """
        self.node_embeddings = node_embeddings
        self.graphs = graphs
        self.graph_encoder = graph_encoder
        self.verbose = verbose

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
        -------
        int
            Number of samples.
        """
        return len(self.node_embeddings)

    def __getitem__(self, idx):
        """
        Retrieves the data for a single graph.

        Parameters:
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns:
        -------
        dict
            Dictionary containing adjacency inputs & labels, node inputs & labels, edge inputs & labels.
        """
        emb = self.node_embeddings[idx]
        G = self.graphs[idx]
        num_nodes = emb.shape[0]

        # Prepare node labels
        node_label_features = []
        for node, attrs in G.nodes(data=True):
            label = attrs.get('label', 'unknown')
            vec = attrs.get('vec') if self.graph_encoder.node_vec_dim > 0 else []
            if label not in self.graph_encoder.node_label_encoder.classes_:
                label = 'unknown'
            label_encoded = self.graph_encoder.node_label_encoder.transform([label])[0]
            label_one_hot = np.zeros(len(self.graph_encoder.node_label_encoder.classes_), dtype=np.float32)
            label_one_hot[label_encoded] = 1.0
            if self.graph_encoder.node_vec_dim > 0:
                if vec is not None and len(vec) == self.graph_encoder.node_vec_dim:
                    vec = np.array(vec, dtype=np.float32)
                else:
                    if self.verbose >=2:
                        print(f"Graph {idx}: Node {node} has invalid or missing 'vec' attribute.")
                    vec = np.zeros(self.graph_encoder.node_vec_dim, dtype=np.float32)
                node_feature = np.concatenate([label_one_hot, vec]).astype(np.float32)
            else:
                node_feature = label_one_hot.astype(np.float32)
            node_label_features.append(node_feature)
        node_labels = torch.tensor(node_label_features, dtype=torch.float)  # [num_nodes, node_attribute_size]

        # Prepare adjacency data
        adj_matrix = nx.adjacency_matrix(G).todense().astype(np.float32)
        adjacency_inputs = []
        adjacency_labels = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue  # Skip self-loops
                adjacency_inputs.append(np.concatenate([emb[i], emb[j]]))  # [2 * embedding_dim]
                adjacency_labels.append(adj_matrix[i, j])
        adjacency_inputs = np.array(adjacency_inputs, dtype=np.float32)  # [num_adj_samples, 2 * embedding_dim]
        adjacency_labels = np.array(adjacency_labels, dtype=np.float32)  # [num_adj_samples]

        # Prepare edge attributes
        edge_inputs = []
        edge_labels = []
        for u, v, attrs in G.edges(data=True):
            # Ensure consistent ordering
            if u > v:
                u, v = v, u
            edge_input = np.concatenate([emb[u], emb[v]])  # [2 * embedding_dim]
            edge_inputs.append(edge_input)
            label = attrs.get('label', 'unknown')
            vec = attrs.get('vec') if self.graph_encoder.edge_vec_dim > 0 else []
            if label not in self.graph_encoder.edge_label_encoder.classes_:
                label = 'unknown'
            label_encoded = self.graph_encoder.edge_label_encoder.transform([label])[0]
            label_one_hot = np.zeros(len(self.graph_encoder.edge_label_encoder.classes_), dtype=np.float32)
            label_one_hot[label_encoded] = 1.0
            if self.graph_encoder.edge_vec_dim > 0:
                if vec is not None and len(vec) == self.graph_encoder.edge_vec_dim:
                    vec = np.array(vec, dtype=np.float32)
                else:
                    if self.verbose >=2:
                        print(f"Graph {idx}: Edge ({u}, {v}) has invalid or missing 'vec' attribute.")
                    vec = np.zeros(self.graph_encoder.edge_vec_dim, dtype=np.float32)
                edge_feature = np.concatenate([label_one_hot, vec]).astype(np.float32)
            else:
                edge_feature = label_one_hot.astype(np.float32)
            edge_labels.append(edge_feature)
        if len(edge_inputs) > 0:
            edge_inputs = np.array(edge_inputs, dtype=np.float32)  # [num_edges, 2 * embedding_dim]
            edge_labels = np.array(edge_labels, dtype=np.float32)  # [num_edges, edge_attribute_size]
        else:
            edge_inputs = np.empty((0, 2 * emb.shape[1]), dtype=np.float32)
            edge_labels = np.empty((0, len(self.graph_encoder.edge_label_encoder.classes_) + self.graph_encoder.edge_vec_dim), dtype=np.float32)

        return {
            'adjacency_input': torch.tensor(adjacency_inputs, dtype=torch.float),  # [num_adj_samples, 2 * embedding_dim]
            'adjacency_label': torch.tensor(adjacency_labels, dtype=torch.float),  # [num_adj_samples]
            'node_input': torch.tensor(emb, dtype=torch.float),  # [num_nodes, embedding_dim]
            'node_label': node_labels,  # [num_nodes, node_attribute_size]
            'edge_input': torch.tensor(edge_inputs, dtype=torch.float),  # [num_edges, 2 * embedding_dim]
            'edge_label': torch.tensor(edge_labels, dtype=torch.float)   # [num_edges, edge_attribute_size]
        }

def graph_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to batch graph data for multi-task learning.

    Parameters:
    ----------
    batch : List[Dict[str, Any]]
        List of dictionaries from GraphDataset.__getitem__()

    Returns:
    -------
    Dict[str, Any]
        Batched dictionary containing all necessary tensors.
    """
    # Aggregate adjacency data
    adjacency_inputs = torch.cat([item['adjacency_input'] for item in batch], dim=0)  # [total_adj_samples, 2 * embedding_dim]
    adjacency_labels = torch.cat([item['adjacency_label'] for item in batch], dim=0)  # [total_adj_samples]

    # Aggregate node data
    node_inputs = torch.cat([item['node_input'] for item in batch], dim=0)  # [total_nodes, embedding_dim]
    node_labels = torch.cat([item['node_label'] for item in batch], dim=0)  # [total_nodes, node_attribute_size]

    # Aggregate edge data
    edge_inputs = torch.cat([item['edge_input'] for item in batch], dim=0)  # [total_edges, 2 * embedding_dim]
    edge_labels = torch.cat([item['edge_label'] for item in batch], dim=0)  # [total_edges, edge_attribute_size]

    return {
        'adjacency_input': adjacency_inputs,
        'adjacency_label': adjacency_labels,
        'node_input': node_inputs,
        'node_label': node_labels,
        'edge_input': edge_inputs,
        'edge_label': edge_labels
    }


class MLPGraphDecoderModel(pl.LightningModule):
    """
    MLPGraphDecoderModel is a PyTorch Lightning module for decoding graph structures
    from node embeddings using a multi-task learning approach. It predicts the
    adjacency matrix, node attributes, and edge attributes.

    Attributes:
    ----------
    adjacency_mlp : nn.Module
        MLP for predicting adjacency probabilities.
    node_mlp : nn.Module
        MLP for predicting node attributes.
    edge_mlp : nn.Module
        MLP for predicting edge attributes.
    adj_loss_fn : nn.Module
        Loss function for adjacency prediction.
    node_loss_fn : nn.Module
        Loss function for node attribute prediction.
    edge_loss_fn : nn.Module
        Loss function for edge attribute prediction.
    learning_rate : float
        Learning rate for the optimizer.
    alpha : float
        Weighting factor to determine the importance of adjacency matrix loss
        relative to node and edge attribute losses.
    verbose : int
        Verbosity level.
    """

    def __init__(
        self,
        embedding_dim: int,
        adjacency_hidden_dim: int,
        adjacency_num_layers: int,
        node_hidden_dim: int,
        node_num_layers: int,
        edge_hidden_dim: int,
        edge_num_layers: int,
        node_output_dim: int,
        edge_output_dim: int,
        dropout_rate: float = 0.5,
        learning_rate: float = 1e-3,
        alpha: float = 0.5,
        verbose: int =1
    ):
        """
        Initializes the MLPGraphDecoderModel.

        Parameters:
        ----------
        embedding_dim : int
            Dimension of the node embeddings.
        adjacency_hidden_dim : int
            Hidden layer size for the adjacency MLP.
        adjacency_num_layers : int
            Number of layers in the adjacency MLP.
        node_hidden_dim : int
            Hidden layer size for the node attribute MLP.
        node_num_layers : int
            Number of layers in the node attribute MLP.
        edge_hidden_dim : int
            Hidden layer size for the edge attribute MLP.
        edge_num_layers : int
            Number of layers in the edge attribute MLP.
        node_output_dim : int
            Dimension of the node attribute outputs.
        edge_output_dim : int
            Dimension of the edge attribute outputs.
        dropout_rate : float, optional (default=0.5)
            Dropout probability for all MLPs.
        learning_rate : float, optional (default=1e-3)
            Learning rate for the optimizer.
        alpha : float, optional (default=0.5)
            Weighting factor to balance the importance of adjacency matrix loss
            versus node and edge attribute losses. A higher alpha places more
            emphasis on the adjacency loss.
        verbose : int, optional (default=1)
            Verbosity level (0: silent, 1: progress, 2: detailed).
        """
        super(MLPGraphDecoderModel, self).__init__()
        self.save_hyperparameters()
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.alpha = alpha

        # Adjacency MLP with Residual Connections
        self.adjacency_mlp = create_residual_mlp(
            num_layers=adjacency_num_layers,
            input_dim=2 * embedding_dim,
            output_dim=1,  # Output logit
            hidden_dim=adjacency_hidden_dim,
            dropout_rate=dropout_rate
        )

        # Node MLP with Residual Connections
        self.node_mlp = create_residual_mlp(
            num_layers=node_num_layers,
            input_dim=embedding_dim,
            output_dim=node_output_dim,
            hidden_dim=node_hidden_dim,
            dropout_rate=dropout_rate
        )

        # Edge MLP with Residual Connections
        self.edge_mlp = create_residual_mlp(
            num_layers=edge_num_layers,
            input_dim=2 * embedding_dim,
            output_dim=edge_output_dim,
            hidden_dim=edge_hidden_dim,
            dropout_rate=dropout_rate
        )

        # Loss functions
        self.adj_loss_fn = nn.BCEWithLogitsLoss()
        self.node_loss_fn = nn.MSELoss()
        self.edge_loss_fn = nn.MSELoss()

    def forward(
        self,
        adjacency_input: torch.Tensor,
        node_input: torch.Tensor,
        edge_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the MLPGraphDecoderModel.

        Parameters:
        ----------
        adjacency_input : torch.Tensor
            Tensor of shape [batch_adj_samples, 2 * embedding_dim].
        node_input : torch.Tensor
            Tensor of shape [batch_nodes, embedding_dim].
        edge_input : torch.Tensor
            Tensor of shape [batch_edges, 2 * embedding_dim].

        Returns:
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - adj_logits: Tensor of shape [batch_adj_samples].
            - node_preds: Tensor of shape [batch_nodes, node_output_dim].
            - edge_preds: Tensor of shape [batch_edges, edge_output_dim].
        """
        adj_logits = self.adjacency_mlp(adjacency_input).squeeze(1)  # [batch_adj_samples]
        node_preds = self.node_mlp(node_input)  # [batch_nodes, node_output_dim]
        edge_preds = self.edge_mlp(edge_input)  # [batch_edges, edge_output_dim]
        return adj_logits, node_preds, edge_preds

    def training_step(self, batch, batch_idx):
        adjacency_input = batch['adjacency_input']
        adjacency_label = batch['adjacency_label']
        node_input = batch['node_input']
        node_label = batch['node_label']
        edge_input = batch['edge_input']
        edge_label = batch['edge_label']

        adj_logits, node_preds, edge_preds = self.forward(adjacency_input, node_input, edge_input)

        # Compute individual losses
        adj_loss = self.adj_loss_fn(adj_logits, adjacency_label)
        node_loss = self.node_loss_fn(node_preds, node_label)
        edge_loss = self.edge_loss_fn(edge_preds, edge_label)

        # Aggregate losses with alpha determining the importance of adjacency loss
        total_loss = self.alpha * adj_loss + (1 - self.alpha) * (node_loss + edge_loss)

        # Determine batch size (number of adjacency samples)
        batch_size = adjacency_input.size(0)

        # Logging
        self.log('train_total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('train_adj_loss', adj_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('train_node_loss', node_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('train_edge_loss', edge_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        # Add detailed logging for verbose=2
        if self.hparams.verbose >= 2:
            print(f"Training Batch {batch_idx}: "
                  f"adj_loss={adj_loss.item():.4f}, "
                  f"node_loss={node_loss.item():.4f}, "
                  f"edge_loss={edge_loss.item():.4f}, "
                  f"total_loss={total_loss.item():.4f}")

        return total_loss

    def validation_step(self, batch, batch_idx):
        adjacency_input = batch['adjacency_input']
        adjacency_label = batch['adjacency_label']
        node_input = batch['node_input']
        node_label = batch['node_label']
        edge_input = batch['edge_input']
        edge_label = batch['edge_label']

        adj_logits, node_preds, edge_preds = self.forward(adjacency_input, node_input, edge_input)

        # Compute individual losses
        adj_loss = self.adj_loss_fn(adj_logits, adjacency_label)
        node_loss = self.node_loss_fn(node_preds, node_label)
        edge_loss = self.edge_loss_fn(edge_preds, edge_label)

        # Aggregate losses with alpha determining the importance of adjacency loss
        total_loss = self.alpha * adj_loss + (1 - self.alpha) * (node_loss + edge_loss)

        # Determine batch size (number of adjacency samples)
        batch_size = adjacency_input.size(0)

        # Logging
        self.log('val_total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('val_adj_loss', adj_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('val_node_loss', node_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('val_edge_loss', edge_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        # Add detailed logging for verbose=2
        if self.hparams.verbose >= 2:
            print(f"Validation Batch {batch_idx}: "
                  f"adj_loss={adj_loss.item():.4f}, "
                  f"node_loss={node_loss.item():.4f}, "
                  f"edge_loss={edge_loss.item():.4f}, "
                  f"total_loss={total_loss.item():.4f}")

        return total_loss

    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns:
        -------
        torch.optim.Optimizer
            Configured optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class MLPGraphDecoder(BaseEstimator, TransformerMixin):
    """
    MLPGraphDecoder is a scikit-learn compatible transformer that decodes node embeddings into graph structures.
    It predicts the adjacency matrix, node attributes, and edge attributes using a multi-task learning approach.

    Parameters:
    ----------
    num_layers_adjacency_mlp : int, default=3
        Number of layers in the adjacency MLP.
    hidden_size_adjacency_mlp : int, default=128
        Hidden layer size for the adjacency MLP.
    num_layers_node_mlp : int, default=2
        Number of layers in the node attribute MLP.
    hidden_size_node_mlp : int, default=64
        Hidden layer size for the node attribute MLP.
    num_layers_edge_mlp : int, default=2
        Number of layers in the edge attribute MLP.
    hidden_size_edge_mlp : int, default=64
        Hidden layer size for the edge attribute MLP.
    alpha : float, default=0.5
        Weighting factor to balance the importance of adjacency matrix loss
        versus node and edge attribute losses. A higher alpha places more emphasis
        on the adjacency loss during training.
    dropout_rate : float, default=0.5
        Dropout probability for all MLPs.
    batch_size : int, default=32
        Batch size for training.
    epochs : int, default=50
        Number of training epochs.
    patience : int, default=50
        Number of epochs with no improvement after which training will be stopped.
    learning_rate : float, default=1e-3
        Learning rate for the optimizer.
    validation_split : float, default=0.2
        Fraction of data to use for validation.
    random_state : int, default=42
        Random state for reproducibility.
    verbose : int, default=1
        Verbosity level (0: silent, 1: progress, 2: detailed).
    device : Optional[str], default=None
        Device to run the model on ('cuda' or 'cpu'). If None, automatically determined.
    plot_training : bool, default=False
        If True, plots training and validation metrics after training.
    """

    def __init__(
        self,
        num_layers_adjacency_mlp: int = 3,
        hidden_size_adjacency_mlp: int = 128,
        num_layers_node_mlp: int = 2,
        hidden_size_node_mlp: int = 64,
        num_layers_edge_mlp: int = 2,
        hidden_size_edge_mlp: int = 64,
        alpha: float = 0.5,
        dropout_rate: float = 0.5,
        batch_size: int = 32,
        epochs: int = 50,
        patience: int = 50,
        learning_rate: float = 1e-3,
        validation_split: float = 0.2,
        random_state: int = 42,
        verbose: int =1,
        device: Optional[str] = None,
        plot_training: bool = False
    ):
        """
        Initializes the MLPGraphDecoder with user-defined parameters.

        Parameters:
        ----------
        num_layers_adjacency_mlp : int, default=3
            Number of layers in the adjacency MLP.
        hidden_size_adjacency_mlp : int, default=128
            Hidden layer size for the adjacency MLP.
        num_layers_node_mlp : int, default=2
            Number of layers in the node attribute MLP.
        hidden_size_node_mlp : int, default=64
            Hidden layer size for the node attribute MLP.
        num_layers_edge_mlp : int, default=2
            Number of layers in the edge attribute MLP.
        hidden_size_edge_mlp : int, default=64
            Hidden layer size for the edge attribute MLP.
        alpha : float, default=0.5
            Weighting factor to balance the importance of adjacency matrix loss
            versus node and edge attribute losses. A higher alpha places more emphasis
            on the adjacency loss during training.
        dropout_rate : float, default=0.5
            Dropout probability for all MLPs.
        batch_size : int, default=32
            Batch size for training.
        epochs : int, default=50
            Number of training epochs.
        patience : int, default=50
            Number of epochs with no improvement after which training will be stopped.
        learning_rate : float, default=1e-3
            Learning rate for the optimizer.
        validation_split : float, default=0.2
            Fraction of data to use for validation.
        random_state : int, default=42
            Random state for reproducibility.
        verbose : int, default=1
            Verbosity level (0: silent, 1: progress, 2: detailed).
        device : Optional[str], default=None
            Device to run the model on ('cuda' or 'cpu'). If None, automatically determined.
        plot_training : bool, default=False
            If True, plots training and validation metrics after training.
        """
        self.num_layers_adjacency_mlp = num_layers_adjacency_mlp
        self.hidden_size_adjacency_mlp = hidden_size_adjacency_mlp
        self.num_layers_node_mlp = num_layers_node_mlp
        self.hidden_size_node_mlp = hidden_size_node_mlp
        self.num_layers_edge_mlp = num_layers_edge_mlp
        self.hidden_size_edge_mlp = hidden_size_edge_mlp
        self.alpha = alpha
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.random_state = random_state
        self.verbose = verbose
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.plot_training = plot_training

        # Initialize placeholders
        self.model = None
        self.is_fitted_ = False
        self.graph_encoder = GraphEncoder(verbose=self.verbose >=2)
        self.node_attribute_size = None
        self.edge_attribute_size = None
        self.output_dim = None  # Assuming GraphEncoder has an output_dim attribute

        # Initialize LossRecorder
        self.loss_recorder = self.LossRecorder()

    class LossRecorder:
        """
        Helper class to record training and validation metrics.
        """
        def __init__(self):
            self.train_total_losses = []
            self.train_adj_losses = []
            self.train_node_losses = []
            self.train_edge_losses = []
            self.val_total_losses = []
            self.val_adj_losses = []
            self.val_node_losses = []
            self.val_edge_losses = []

    class MetricsRecorder(pl.Callback):
        """
        PyTorch Lightning Callback to record metrics at each epoch.
        """
        def __init__(self, loss_recorder):
            super().__init__()
            self.loss_recorder = loss_recorder

        def on_train_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            train_total_loss = metrics.get('train_total_loss')
            train_adj_loss = metrics.get('train_adj_loss')
            train_node_loss = metrics.get('train_node_loss')
            train_edge_loss = metrics.get('train_edge_loss')
            
            if train_total_loss is not None:
                self.loss_recorder.train_total_losses.append(train_total_loss.item())
            if train_adj_loss is not None:
                self.loss_recorder.train_adj_losses.append(train_adj_loss.item())
            if train_node_loss is not None:
                self.loss_recorder.train_node_losses.append(train_node_loss.item())
            if train_edge_loss is not None:
                self.loss_recorder.train_edge_losses.append(train_edge_loss.item())

        def on_validation_epoch_end(self, trainer, pl_module):
            metrics = trainer.callback_metrics
            val_total_loss = metrics.get('val_total_loss')
            val_adj_loss = metrics.get('val_adj_loss')
            val_node_loss = metrics.get('val_node_loss')
            val_edge_loss = metrics.get('val_edge_loss')
            
            if val_total_loss is not None:
                self.loss_recorder.val_total_losses.append(val_total_loss.item())
            if val_adj_loss is not None:
                self.loss_recorder.val_adj_losses.append(val_adj_loss.item())
            if val_node_loss is not None:
                self.loss_recorder.val_node_losses.append(val_node_loss.item())
            if val_edge_loss is not None:
                self.loss_recorder.val_edge_losses.append(val_edge_loss.item())

    def fit(
        self,
        node_embeddings: List[np.ndarray],
        graphs: List[nx.Graph]
    ):
        """
        Fits the MLPGraphDecoder model on the provided node embeddings and graphs.

        Parameters:
        ----------
        node_embeddings : List[np.ndarray]
            List of numpy arrays, each of shape [num_nodes, embedding_dim].
        graphs : List[nx.Graph]
            List of NetworkX graphs corresponding to the node embeddings.

        Returns:
        -------
        self : MLPGraphDecoder
            Fitted transformer.
        """
        if self.verbose >=1:
            print("Starting MLPGraphDecoder fit process.")

        if len(node_embeddings) != len(graphs):
            raise ValueError("Number of node embeddings must match number of graphs.")

        # Fit the GraphEncoder to extract adjacency and feature information
        if self.verbose >=1:
            print("Fitting GraphEncoder...")
        self.graph_encoder.fit(graphs)

        # Ensure that 'output_dim' is set
        if self.graph_encoder.output_dim == 0:
            if self.verbose >=1:
                print("No 'output' vectors found in any nodes. Cannot proceed with decoding.")
            raise ValueError("No 'output' vectors found in any nodes. Each node must have an 'output' attribute for decoding.")

        self.output_dim = self.graph_encoder.output_dim

        # Determine attribute sizes
        self.node_attribute_size = len(self.graph_encoder.node_label_encoder.classes_) + self.graph_encoder.node_vec_dim
        self.edge_attribute_size = len(self.graph_encoder.edge_label_encoder.classes_) + self.graph_encoder.edge_vec_dim

        if self.verbose >=1:
            print(f"Node Attribute Size: {self.node_attribute_size}")
            print(f"Edge Attribute Size: {self.edge_attribute_size}")

        # Create GraphDataset
        dataset = GraphDataset(node_embeddings, graphs, self.graph_encoder, verbose=self.verbose)

        # Split into training and validation
        total_size = len(dataset)
        val_size = int(self.validation_split * total_size)
        train_size = total_size - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.random_state)
        )

        # Create DataLoaders with custom collate_fn
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=graph_collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=graph_collate_fn
        )

        # Initialize MLPGraphDecoderModel
        adjacency_hidden_dim = self.hidden_size_adjacency_mlp
        adjacency_num_layers = self.num_layers_adjacency_mlp
        node_hidden_dim = self.hidden_size_node_mlp
        node_num_layers = self.num_layers_node_mlp
        edge_hidden_dim = self.hidden_size_edge_mlp
        edge_num_layers = self.num_layers_edge_mlp
        node_output_dim = self.node_attribute_size
        edge_output_dim = self.edge_attribute_size
        embedding_dim = node_embeddings[0].shape[1]

        self.model = MLPGraphDecoderModel(
            embedding_dim=embedding_dim,
            adjacency_hidden_dim=adjacency_hidden_dim,
            adjacency_num_layers=adjacency_num_layers,
            node_hidden_dim=node_hidden_dim,
            node_num_layers=node_num_layers,
            edge_hidden_dim=edge_hidden_dim,
            edge_num_layers=edge_num_layers,
            node_output_dim=node_output_dim,
            edge_output_dim=edge_output_dim,
            dropout_rate=self.dropout_rate,
            learning_rate=self.learning_rate,
            alpha=self.alpha,  # Pass alpha to the model
            verbose=self.verbose
        )
        self.model.to(self.device)

        # Define logger
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logger = TensorBoardLogger(
            save_dir=os.path.join("graph_decoder_checkpoints", f"run_{timestamp}"),
            name="lightning_logs"
        )

        # Define callbacks
        checkpoint_dir = os.path.join("graph_decoder_checkpoints", f"run_{timestamp}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            monitor='val_total_loss',
            dirpath=checkpoint_dir,
            filename='best_model',
            save_top_k=1,
            mode='min',
            verbose=self.verbose >=1
        )
        early_stop_callback = EarlyStopping(
            monitor='val_total_loss',
            patience=self.patience,
            verbose=self.verbose >=1,
            mode='min'
        )

        # Initialize MetricsRecorder callback
        metrics_recorder = self.MetricsRecorder(self.loss_recorder)

        # Initialize Trainer
        trainer = pl.Trainer(
            max_epochs=self.epochs,
            accelerator='gpu' if self.device == 'cuda' else 'cpu',
            devices=1 if self.device == 'cuda' else None,  # Specify the number of GPUs if using GPU
            callbacks=[checkpoint_callback, early_stop_callback, metrics_recorder],
            deterministic=True,
            logger=logger,
            enable_progress_bar=self.verbose >=1
        )

        if self.verbose >=1:
            print("Starting training...")

        # Train the model
        trainer.fit(self.model, train_loader, val_loader)

        # Load the best model
        if checkpoint_callback.best_model_path:
            if self.verbose >=1:
                print(f"Loading best model from {checkpoint_callback.best_model_path}")
            self.model = MLPGraphDecoderModel.load_from_checkpoint(
                checkpoint_callback.best_model_path,
                embedding_dim=embedding_dim,
                adjacency_hidden_dim=adjacency_hidden_dim,
                adjacency_num_layers=adjacency_num_layers,
                node_hidden_dim=node_hidden_dim,
                node_num_layers=node_num_layers,
                edge_hidden_dim=edge_hidden_dim,
                edge_num_layers=edge_num_layers,
                node_output_dim=node_output_dim,
                edge_output_dim=edge_output_dim,
                dropout_rate=self.dropout_rate,
                learning_rate=self.learning_rate,
                alpha=self.alpha,  # Ensure alpha is loaded correctly
                verbose=self.verbose
            )
            self.model.to(self.device)
            self.model.eval()

        self.is_fitted_ = True

        # Plot training metrics if requested
        if self.plot_training:
            self._plot_metrics()

        if self.verbose >=1:
            print("MLPGraphDecoder training complete.")
        return self


    def transform(self, node_embeddings: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Decodes node embeddings into predicted adjacency matrices, node attributes, edge attributes,
        and edge endpoint indices.

        Parameters:
        ----------
        node_embeddings : List[np.ndarray]
            List of numpy arrays, each of shape [num_nodes, embedding_dim].

        Returns:
        -------
        List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
            List of tuples where each tuple contains:
                1. Predicted adjacency matrix: [num_nodes, num_nodes] with probabilities.
                2. Node label predictions: [num_nodes, node_attribute_size].
                3. Edge label predictions: [num_edges, edge_attribute_size].
                4. Edge endpoint indices: [num_edges, 2] where each row is [src_idx, dest_idx].
        """
        if not self.is_fitted_:
            raise RuntimeError("MLPGraphDecoder must be fitted before calling transform.")

        if self.verbose >=1:
            print("Starting decoding process.")

        decoded_outputs = []

        with torch.no_grad():
            self.model.eval()
            for idx, emb in enumerate(node_embeddings):
                num_nodes = emb.shape[0]
                embedding_tensor = torch.tensor(emb, dtype=torch.float).to(self.device)  # [num_nodes, embedding_dim]

                # Prepare adjacency inputs: all possible node pairs (excluding self-loops)
                node_indices = torch.arange(num_nodes)
                src, dst = torch.meshgrid(node_indices, node_indices, indexing='ij')
                src = src.flatten()
                dst = dst.flatten()
                mask = src != dst
                src = src[mask]
                dst = dst[mask]
                adjacency_inputs = torch.cat([embedding_tensor[src], embedding_tensor[dst]], dim=1)  # [num_pairs, 2 * embedding_dim]

                # Predict adjacency probabilities
                adj_logits = self.model.adjacency_mlp(adjacency_inputs).squeeze(1)  # [num_pairs]
                adj_probs_flat = torch.sigmoid(adj_logits).cpu().numpy()  # [num_pairs]

                # Initialize full adjacency matrix
                adj_probs = np.zeros((num_nodes, num_nodes), dtype=np.float32)

                # Assign predicted probabilities to non-diagonal entries
                adj_probs[src.cpu().numpy(), dst.cpu().numpy()] = adj_probs_flat

                # Optionally, handle self-loops (e.g., set to zero or a specific value)
                # adj_probs[np.arange(num_nodes), np.arange(num_nodes)] = 0.0  # Example: No self-loops

                # Predict node attributes
                node_preds = self.model.node_mlp(embedding_tensor).cpu().numpy()  # [num_nodes, node_attribute_size]

                # Predict edge attributes
                edge_preds = self.model.edge_mlp(adjacency_inputs).cpu().numpy()  # [num_pairs, edge_attribute_size]

                # Prepare edge endpoint indices
                edge_indices = np.stack([src.cpu().numpy(), dst.cpu().numpy()], axis=1)  # [num_pairs, 2]

                # Append to decoded_outputs
                decoded_outputs.append((adj_probs, node_preds, edge_preds, edge_indices))

                if self.verbose >1:
                    print(f"Decoded graph {idx + 1}/{len(node_embeddings)}.")

        if self.verbose >=1:
            print("Decoding process complete.")
        return decoded_outputs


    def fit_transform(
        self,
        node_embeddings: List[np.ndarray],
        graphs: List[nx.Graph]
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Fits the MLPGraphDecoder model and then decodes the node embeddings.

        Parameters:
        ----------
        node_embeddings : List[np.ndarray]
            List of numpy arrays, each of shape [num_nodes, embedding_dim].
        graphs : List[nx.Graph]
            List of NetworkX graphs corresponding to the node embeddings.

        Returns:
        -------
        List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
            List of tuples containing predicted adjacency matrices, node attributes, and edge attributes.
        """
        self.fit(node_embeddings, graphs)
        return self.transform(node_embeddings)


    def _plot_metrics(self, window: int = 10, use_pandas: bool = True):
        """
        Plots the training and validation loss curves using matplotlib.
        Additionally, overlays a symmetric running mean to smooth out the curves.

        Parameters:
        - window (int): The window size on each side for the running mean. Total window size is 2*window + 1.
        - use_pandas (bool): If True, uses Pandas for running mean calculation. Otherwise, uses NumPy.
        """
        
        def running_mean_symmetric(data, window):
            """
            Computes the symmetric running mean of a 1D array using NumPy.

            Parameters:
            - data (list or np.ndarray): Input data.
            - window (int): The window size on each side. Total window size is 2*window + 1.

            Returns:
            - np.ndarray: Running mean of the data.
            """
            if window < 0:
                raise ValueError("Window size must be non-negative.")
            if window == 0:
                return np.array(data)
            
            kernel = np.ones(2 * window + 1) / (2 * window + 1)
            padded_data = np.pad(data, (window, window), mode='edge')
            running_mean = np.convolve(padded_data, kernel, mode='valid')
            return running_mean

        def running_mean_symmetric_pandas(data, window):
            """
            Computes the symmetric running mean of a 1D array using Pandas.

            Parameters:
            - data (list or np.ndarray): Input data.
            - window (int): The window size on each side. Total window size is 2*window + 1.

            Returns:
            - np.ndarray: Running mean of the data.
            """
            if window < 0:
                raise ValueError("Window size must be non-negative.")
            if window == 0:
                return np.array(data)
            
            df = pd.Series(data)
            rolling_mean = df.rolling(window=2*window + 1, center=True, min_periods=1).mean()
            return rolling_mean.to_numpy()

        # Determine which running mean function to use
        if use_pandas:
            running_mean_func = running_mean_symmetric_pandas
        else:
            running_mean_func = running_mean_symmetric

        # Check if any loss metrics are available
        available_train_losses = [
            self.loss_recorder.train_total_losses,
            self.loss_recorder.train_adj_losses,
            self.loss_recorder.train_node_losses,
            self.loss_recorder.train_edge_losses
        ]
        available_val_losses = [
            self.loss_recorder.val_total_losses,
            self.loss_recorder.val_adj_losses,
            self.loss_recorder.val_node_losses,
            self.loss_recorder.val_edge_losses
        ]
        
        if not any(available_train_losses) and not any(available_val_losses):
            if self.verbose >=1:
                print("No loss metrics to plot.")
            return
        
        if self.verbose >=1:
            print(f"Number of Training Total Losses: {len(self.loss_recorder.train_total_losses)}")
            print(f"Number of Validation Total Losses: {len(self.loss_recorder.val_total_losses)}")
            print(f"Number of Training Adjacency Losses: {len(self.loss_recorder.train_adj_losses)}")
            print(f"Number of Validation Adjacency Losses: {len(self.loss_recorder.val_adj_losses)}")
            print(f"Number of Training Node Losses: {len(self.loss_recorder.train_node_losses)}")
            print(f"Number of Validation Node Losses: {len(self.loss_recorder.val_node_losses)}")
            print(f"Number of Training Edge Losses: {len(self.loss_recorder.train_edge_losses)}")
            print(f"Number of Validation Edge Losses: {len(self.loss_recorder.val_edge_losses)}")

        # Determine the number of epochs based on the minimum length among all loss lists
        all_train_lengths = [len(lst) for lst in available_train_losses]
        all_val_lengths = [len(lst) for lst in available_val_losses]
        
        min_train_len = min(all_train_lengths) if all_train_lengths else 0
        min_val_len = min(all_val_lengths) if all_val_lengths else 0
        min_len = min(min_train_len, min_val_len) if min_train_len and min_val_len else max(min_train_len, min_val_len)
        
        if min_len == 0:
            min_len = max(all_train_lengths + all_val_lengths)  # Take the maximum available length
        
        epochs = range(1, min_len + 1)
        
        # Prepare data for plotting
        train_total_losses = np.array(self.loss_recorder.train_total_losses[:min_len])
        val_total_losses = np.array(self.loss_recorder.val_total_losses[:min_len])
        
        train_adj_losses = np.array(self.loss_recorder.train_adj_losses[:min_len])
        val_adj_losses = np.array(self.loss_recorder.val_adj_losses[:min_len])
        
        train_node_losses = np.array(self.loss_recorder.train_node_losses[:min_len])
        val_node_losses = np.array(self.loss_recorder.val_node_losses[:min_len])
        
        train_edge_losses = np.array(self.loss_recorder.train_edge_losses[:min_len])
        val_edge_losses = np.array(self.loss_recorder.val_edge_losses[:min_len])
        
        # Calculate running means if window > 0
        if window > 0:
            train_total_rm = running_mean_func(train_total_losses, window)
            val_total_rm = running_mean_func(val_total_losses, window)
            
            train_adj_rm = running_mean_func(train_adj_losses, window)
            val_adj_rm = running_mean_func(val_adj_losses, window)
            
            train_node_rm = running_mean_func(train_node_losses, window)
            val_node_rm = running_mean_func(val_node_losses, window)
            
            train_edge_rm = running_mean_func(train_edge_losses, window)
            val_edge_rm = running_mean_func(val_edge_losses, window)
        else:
            train_total_rm = train_total_losses
            val_total_rm = val_total_losses
            train_adj_rm = train_adj_losses
            val_adj_rm = val_adj_losses
            train_node_rm = train_node_losses
            val_node_rm = val_node_losses
            train_edge_rm = train_edge_losses
            val_edge_rm = val_edge_losses
        
        plt.figure(figsize=(20, 12))
        
        # Subplot 1: Total Loss
        plt.subplot(2, 2, 1)
        plt.plot(epochs, train_total_losses, markersize=6, color='navy', marker='o', linestyle='-', 
                 label='Training Total Loss', alpha=0.4, markerfacecolor='white', markeredgecolor='navy', markeredgewidth=1)
        plt.plot(epochs, val_total_losses, markersize=6, color='orange', marker='o', linestyle='-', 
                 label='Validation Total Loss', alpha=0.4, markerfacecolor='white', markeredgecolor='orange', markeredgewidth=1)
        
        if window > 0:
            plt.plot(epochs, train_total_rm, color='navy', linewidth=3, 
                     label=f'Training Total Loss (Running Mean, window={window})')
            plt.plot(epochs, val_total_rm, color='orange', linewidth=3, 
                     label=f'Validation Total Loss (Running Mean, window={window})')
        
        plt.xlabel('Epochs')
        plt.yscale('log')  # Set the y-axis to a logarithmic scale
        plt.ylabel('Total Loss')
        plt.title('Training and Validation Total Loss')
        plt.legend()
        plt.grid(True)
        
        # Subplot 2: Adjacency Loss
        plt.subplot(2, 2, 2)
        plt.plot(epochs, train_adj_losses, markersize=6, color='green', marker='o', linestyle='-', 
                 label='Training Adjacency Loss', alpha=0.4, markerfacecolor='white', markeredgecolor='green', markeredgewidth=1)
        plt.plot(epochs, val_adj_losses, markersize=6, color='red', marker='o', linestyle='-', 
                 label='Validation Adjacency Loss', alpha=0.4, markerfacecolor='white', markeredgecolor='red', markeredgewidth=1)
        
        if window > 0:
            plt.plot(epochs, train_adj_rm, color='green', linewidth=3, 
                     label=f'Training Adjacency Loss (Running Mean, window={window})')
            plt.plot(epochs, val_adj_rm, color='red', linewidth=3, 
                     label=f'Validation Adjacency Loss (Running Mean, window={window})')
        
        plt.xlabel('Epochs')
        plt.yscale('log')
        plt.ylabel('Adjacency Loss')
        plt.title('Training and Validation Adjacency Loss')
        plt.legend()
        plt.grid(True)
        
        # Subplot 3: Node Loss
        plt.subplot(2, 2, 3)
        plt.plot(epochs, train_node_losses, markersize=6, color='purple', marker='o', linestyle='-', 
                 label='Training Node Loss', alpha=0.4, markerfacecolor='white', markeredgecolor='purple', markeredgewidth=1)
        plt.plot(epochs, val_node_losses, markersize=6, color='brown', marker='o', linestyle='-', 
                 label='Validation Node Loss', alpha=0.4, markerfacecolor='white', markeredgecolor='brown', markeredgewidth=1)
        
        if window > 0:
            plt.plot(epochs, train_node_rm, color='purple', linewidth=3, 
                     label=f'Training Node Loss (Running Mean, window={window})')
            plt.plot(epochs, val_node_rm, color='brown', linewidth=3, 
                     label=f'Validation Node Loss (Running Mean, window={window})')
        
        plt.xlabel('Epochs')
        plt.yscale('log')
        plt.ylabel('Node Loss')
        plt.title('Training and Validation Node Loss')
        plt.legend()
        plt.grid(True)
        
        # Subplot 4: Edge Loss
        plt.subplot(2, 2, 4)
        plt.plot(epochs, train_edge_losses, markersize=6, color='teal', marker='o', linestyle='-', 
                 label='Training Edge Loss', alpha=0.4, markerfacecolor='white', markeredgecolor='teal', markeredgewidth=1)
        plt.plot(epochs, val_edge_losses, markersize=6, color='magenta', marker='o', linestyle='-', 
                 label='Validation Edge Loss', alpha=0.4, markerfacecolor='white', markeredgecolor='magenta', markeredgewidth=1)
        
        if window > 0:
            plt.plot(epochs, train_edge_rm, color='teal', linewidth=3, 
                     label=f'Training Edge Loss (Running Mean, window={window})')
            plt.plot(epochs, val_edge_rm, color='magenta', linewidth=3, 
                     label=f'Validation Edge Loss (Running Mean, window={window})')
        
        plt.xlabel('Epochs')
        plt.yscale('log')
        plt.ylabel('Edge Loss')
        plt.title('Training and Validation Edge Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()



#===================================================================================================================================================================================================================


# Define a LossRecorder to store metrics
class LossRecorder:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_mses = []
        self.val_mses = []

# Custom Dataset for AutoRegressor.
class AutoRegressorDataset(Dataset):
    """
    Custom Dataset for AutoRegressor.
    """
    def __init__(self, inputs, targets):
        """
        Initializes the dataset with inputs and targets.
        
        Args:
            inputs (List[np.ndarray]): List of input vectors of size f.
            targets (List[np.ndarray]): List of target matrices of size n x f.
        """
        self.inputs = inputs  # List of vectors (size f)
        self.targets = targets  # List of matrices (n_i x f)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class CollateFn:
    """
    Custom collate function to pad target matrices to a fixed max_n.
    """
    def __init__(self, max_n, f):
        """
        Initializes the collate function.
        
        Args:
            max_n (int): The fixed maximum number of rows for padding.
            f (int): Number of features.
        """
        self.max_n = max_n
        self.f = f

    def __call__(self, batch):
        """
        Pads the target matrices in the batch to max_n and creates masks.
        
        Args:
            batch (List[Tuple[np.ndarray, np.ndarray]]): Batch of (input, target) pairs.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Padded inputs, targets, and masks.
        """
        inputs, targets = zip(*batch)
        inputs = torch.tensor(inputs, dtype=torch.float)
        
        # Initialize padded targets and masks
        padded_targets = torch.zeros(len(targets), self.max_n, self.f)
        masks = torch.zeros(len(targets), self.max_n)
        
        for i, target in enumerate(targets):
            n = target.shape[0]
            if n > self.max_n:
                raise ValueError(f"Target with n={n} exceeds the global max_n={self.max_n}.")
            padded_targets[i, :n, :] = torch.tensor(target, dtype=torch.float)
            masks[i, :n] = 1  # Mask indicating valid rows
        
        return inputs, padded_targets, masks

class AutoRegressorTransformerModel(pl.LightningModule):
    """
    PyTorch Lightning Module for AutoRegressor using Transformer Architecture.
    """
    def __init__(self, input_dim, max_n, f, embed_dim=128, num_heads=8, num_layers=4, learning_rate=1e-3, loss_recorder=None):
        """
        Initializes the transformer-based model.
        
        Args:
            input_dim (int): Dimension of the input vector (f).
            max_n (int): Fixed maximum number of rows (n) in the target matrices.
            f (int): Number of features.
            embed_dim (int, optional): Embedding dimension. Defaults to 128.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            num_layers (int, optional): Number of transformer layers. Defaults to 4.
            learning_rate (float, optional): Learning rate. Defaults to 1e-3.
            loss_recorder (LossRecorder, optional): Recorder to store losses and MSEs.
        """
        super(AutoRegressorTransformerModel, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.max_n = max_n
        self.f = f
        self.embed_dim = embed_dim
        self.loss_recorder = loss_recorder  # Initialize the loss recorder
        
        # Encoder: Project input vector to embedding
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Learnable query embeddings for decoder
        self.query_embeddings = nn.Parameter(torch.randn(max_n, embed_dim))
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, f)
        
        self.loss_fn = nn.MSELoss()
        self.mse_fn = nn.MSELoss(reduction='none')  # For per-sample MSE

    def forward(self, x):
        """
        Forward pass of the transformer-based model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, f).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, max_n, f).
        """
        batch_size = x.size(0)
        
        # Encode input
        encoder_output = self.input_proj(x)  # (batch_size, embed_dim)
        encoder_output = encoder_output.unsqueeze(0)  # (1, batch_size, embed_dim)
        
        # Prepare decoder inputs (query embeddings)
        queries = self.query_embeddings.unsqueeze(1).repeat(1, batch_size, 1)  # (max_n, batch_size, embed_dim)
        
        # Decode
        decoder_output = self.transformer_decoder(queries, encoder_output)  # (max_n, batch_size, embed_dim)
        decoder_output = decoder_output.permute(1, 0, 2)  # (batch_size, max_n, embed_dim)
        
        # Project to output features
        output = self.output_proj(decoder_output)  # (batch_size, max_n, f)
        
        return output

    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Batch data.
            batch_idx (int): Batch index.
        
        Returns:
            torch.Tensor: Loss value.
        """
        inputs, targets, masks = batch  # inputs: (batch_size, f), targets: (batch_size, max_n, f), masks: (batch_size, max_n)
        outputs = self.forward(inputs)  # (batch_size, max_n, f)
        
        # Apply masks to consider only valid rows
        masks = masks.unsqueeze(-1)  # (batch_size, max_n, 1)
        loss = self.loss_fn(outputs * masks, targets * masks)
        
        # Calculate MSE for monitoring
        mse = self.mse_fn(outputs * masks, targets * masks)
        mse = mse.sum(dim=(1,2)) / masks.sum(dim=(1,2))  # Mean MSE per sample
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        self.log('train_mse', mse.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=False)
        
        # Record metrics if recorder is provided
        if self.loss_recorder is not None:
            self.loss_recorder.train_losses.append(loss.item())
            self.loss_recorder.train_mses.append(mse.mean().item())
        
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Batch data.
            batch_idx (int): Batch index.
        """
        inputs, targets, masks = batch
        outputs = self.forward(inputs)
        
        # Apply masks
        masks = masks.unsqueeze(-1)
        loss = self.loss_fn(outputs * masks, targets * masks)
        
        # Calculate MSE
        mse = self.mse_fn(outputs * masks, targets * masks)
        mse = mse.sum(dim=(1,2)) / masks.sum(dim=(1,2))
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.log('val_mse', mse.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=False)
        
        # Record metrics if recorder is provided
        if self.loss_recorder is not None:
            self.loss_recorder.val_losses.append(loss.item())
            self.loss_recorder.val_mses.append(mse.mean().item())

    def configure_optimizers(self):
        """
        Configures the optimizer.
        
        Returns:
            torch.optim.Optimizer: Optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class AutoRegressor(TransformerMixin, BaseEstimator):
    """
    AutoRegressor Transformer using a Transformer Architecture and PyTorch Lightning.
    """
    def __init__(
        self, 
        embed_dim: int = 128, 
        num_heads: int = 8, 
        num_layers: int = 4, 
        learning_rate: float = 1e-3, 
        epochs: int = 50, 
        batch_size: int = 32,
        verbose: int = 1,
        validation_split: float = 0.2,
        random_state: int = 42,
        plot_training: bool = True
    ):
        """
        Initializes the AutoRegressor.
        
        Args:
            embed_dim (int, optional): Embedding dimension for the transformer. Defaults to 128.
            num_heads (int, optional): Number of attention heads in the transformer. Defaults to 8.
            num_layers (int, optional): Number of transformer layers. Defaults to 4.
            learning_rate (float, optional): Learning rate. Defaults to 1e-3.
            epochs (int, optional): Maximum number of training epochs. Defaults to 50.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            verbose (int, optional): Verbosity level for plotting. Defaults to 1.
            validation_split (float, optional): Fraction of data to use for validation. Defaults to 0.2.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.
            plot_training (bool, optional): If True, plots training and validation metrics at the end of fitting. Defaults to False.
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.validation_split = validation_split
        self.random_state = random_state
        self.plot_training = plot_training
        self.pytorch_lightning_module = None
        self.input_dim = None
        self.f = None
        self.max_n = None
        self.loss_recorder = LossRecorder()

    def fit(self, X, y=None):
        """
        Fits the AutoRegressor model.
        
        Args:
            X (List[Tuple[np.ndarray, np.ndarray]] or List[np.ndarray]): 
                - If y is None, X can be a list of matrices or a list of (vector, matrix) pairs.
                - If y is provided, X should be a list of vectors and y a list of matrices.
            y (List[np.ndarray], optional): List of target matrices. Defaults to None.
        
        Returns:
            self: Fitted estimator.
        """
        # Determine input vectors and target matrices
        if y is None:
            if isinstance(X, list) and len(X) > 0 and isinstance(X[0], (tuple, list)) and len(X[0]) == 2:
                # X is a list of pairs (vector, matrix)
                inputs, targets = zip(*X)
            else:
                # X is a list of matrices; compute input vectors by summing rows
                inputs = [matrix.sum(axis=0) for matrix in X]
                targets = [matrix for matrix in X]
        else:
            # X is a list of input vectors, y is a list of matrices
            inputs = X
            targets = y

        if len(inputs) == 0:
            raise ValueError("Empty input list.")

        # Determine feature size (f) and global maximum number of rows (max_n)
        self.f = len(inputs[0])
        self.input_dim = self.f
        self.max_n = max(target.shape[0] for target in targets)

        # Split the data into training and validation sets
        if self.validation_split > 0.0:
            inputs_train, inputs_val, targets_train, targets_val = train_test_split(
                inputs, 
                targets, 
                test_size=self.validation_split, 
                random_state=self.random_state
            )
            if self.verbose >=1:
                print(f"Training samples: {len(inputs_train)}, Validation samples: {len(inputs_val)}")
        else:
            inputs_train, targets_train = inputs, targets
            inputs_val, targets_val = [], []
            if self.verbose >=1:
                print(f"Training samples: {len(inputs_train)}, No validation set provided.")

        # Prepare training dataset and dataloader
        train_dataset = AutoRegressorDataset(inputs_train, targets_train)
        train_collate_fn = CollateFn(max_n=self.max_n, f=self.f)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=train_collate_fn
        )

        # Prepare validation dataloader if validation_split > 0
        if self.validation_split > 0.0:
            val_dataset = AutoRegressorDataset(inputs_val, targets_val)
            val_collate_fn = CollateFn(max_n=self.max_n, f=self.f)
            val_dataloader = DataLoader(
                val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                collate_fn=val_collate_fn
            )
        else:
            val_dataloader = None

        # Initialize the PyTorch Lightning model with the loss recorder
        self.pytorch_lightning_module = AutoRegressorTransformerModel(
            input_dim=self.input_dim, 
            max_n=self.max_n, 
            f=self.f, 
            embed_dim=self.embed_dim, 
            num_heads=self.num_heads, 
            num_layers=self.num_layers,
            learning_rate=self.learning_rate,
            loss_recorder=self.loss_recorder
        )

        # Initialize the trainer with updated parameters
        trainer = pl.Trainer(
            max_epochs=self.epochs, 
            logger=False, 
            enable_checkpointing=False,  # Disable checkpointing
            enable_progress_bar=True,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1 if torch.cuda.is_available() else None,  # Use GPU if available
            log_every_n_steps=1,
            callbacks=[]
        )

        # Train the model
        if self.validation_split > 0.0 and val_dataloader is not None:
            trainer.fit(self.pytorch_lightning_module, train_dataloader, val_dataloaders=val_dataloader)
        else:
            trainer.fit(self.pytorch_lightning_module, train_dataloader)

        # Plot training metrics if requested
        if self.plot_training:
            self._plot_metrics(window=10, use_pandas=True)

        return self

    def transform(self, X):
        """
        Transforms input vectors to output matrices using the trained model.
        
        Args:
            X (List[np.ndarray] or np.ndarray): 
                - List of vectors of size f.
                - Data matrix of shape (m, f).
        
        Returns:
            List[np.ndarray]: List of output matrices of size n x f.
        """
        if isinstance(X, list):
            inputs = X
        elif isinstance(X, np.ndarray):
            if X.ndim == 2:
                inputs = [x for x in X]
            else:
                raise ValueError("Input data matrix must be 2D.")
        else:
            raise ValueError("Input must be a list of vectors or a 2D numpy array.")

        if self.pytorch_lightning_module is None:
            raise ValueError("The model has not been fitted yet. Please call fit before transform.")

        # Convert inputs to tensor
        inputs_tensor = torch.tensor(inputs, dtype=torch.float)
        device = self.pytorch_lightning_module.device
        inputs_tensor = inputs_tensor.to(device)
        
        # Set model to evaluation mode
        self.pytorch_lightning_module.eval()
        
        # Disable gradient computation
        with torch.no_grad():
            outputs = self.pytorch_lightning_module(inputs_tensor)  # (batch_size, max_n, f)
        
        # Move outputs to CPU and convert to numpy
        outputs = outputs.cpu().numpy()
        
        transformed = []
        for i, input_vec in enumerate(inputs):
            # Here, you need to determine 'n' based on your specific use case.
            # For demonstration, we'll assume 'n' is the maximum number of rows.
            # If 'n' is determined differently, adjust this logic accordingly.
            n = self.max_n  # Alternatively, extract n from input_vec or another source
            if n > self.max_n:
                raise ValueError(f"n={n} in input exceeds the maximum n={self.max_n} learned during training.")
            matrix = outputs[i][:n, :]
            # Enforce the first feature to be 1
            matrix[:,0] = 1.0
            transformed.append(matrix)
        
        return transformed

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fits the model and transforms the input data.
        
        Args:
            X (List[Tuple[np.ndarray, np.ndarray]] or List[np.ndarray]): 
                - If y is None, X can be a list of matrices or a list of (vector, matrix) pairs.
                - If y is provided, X should be a list of vectors and y a list of matrices.
            y (List[np.ndarray], optional): List of target matrices. Defaults to None.
            **fit_params: Additional fit parameters.
        
        Returns:
            List[np.ndarray]: Transformed output matrices.
        """
        self.fit(X, y)
        return self.transform(X if y is None else X)

    def _plot_metrics(self, window: int = 10, use_pandas: bool = True):
        """
        Plots the training and validation loss curves and MSE curves using matplotlib.
        Additionally, overlays a symmetric running mean to smooth out the curves.

        Parameters:
        - window (int): The window size on each side for the running mean. Total window size is 2*window + 1.
        - use_pandas (bool): If True, uses Pandas for running mean calculation. Otherwise, uses NumPy.
        """
        
        def running_mean_symmetric(data, window):
            """
            Computes the symmetric running mean of a 1D array using NumPy.

            Parameters:
            - data (list or np.ndarray): Input data.
            - window (int): The window size on each side. Total window size is 2*window + 1.

            Returns:
            - np.ndarray: Running mean of the data.
            """
            if window < 0:
                raise ValueError("Window size must be non-negative.")
            if window == 0:
                return np.array(data)
            
            kernel = np.ones(2 * window + 1) / (2 * window + 1)
            padded_data = np.pad(data, (window, window), mode='edge')
            running_mean = np.convolve(padded_data, kernel, mode='valid')
            return running_mean

        def running_mean_symmetric_pandas(data, window):
            """
            Computes the symmetric running mean of a 1D array using Pandas.

            Parameters:
            - data (list or np.ndarray): Input data.
            - window (int): The window size on each side. Total window size is 2*window + 1.

            Returns:
            - np.ndarray: Running mean of the data.
            """
            if window < 0:
                raise ValueError("Window size must be non-negative.")
            if window == 0:
                return np.array(data)
            
            df = pd.Series(data)
            rolling_mean = df.rolling(window=2*window + 1, center=True, min_periods=1).mean()
            return rolling_mean.to_numpy()

        # Determine which running mean function to use
        if use_pandas:
            running_mean_func = running_mean_symmetric_pandas
        else:
            running_mean_func = running_mean_symmetric

        train_len = len(self.loss_recorder.train_losses)
        val_len = len(self.loss_recorder.val_losses)
        
        if not self.loss_recorder.train_losses and not self.loss_recorder.val_losses:
            if self.verbose >=1:
                print("No loss metrics to plot.")
            return
        
        if self.verbose >=1:
            print(f"Number of Training Losses: {train_len}")
            print(f"Number of Validation Losses: {val_len}")
        
        # Determine the number of epochs based on the minimum of the two
        min_len = min(train_len, val_len) if self.validation_split > 0.0 else train_len
        epochs = range(1, min_len + 1)
        
        # Prepare data for plotting
        train_losses = np.array(self.loss_recorder.train_losses[:min_len])
        val_losses = np.array(self.loss_recorder.val_losses[:min_len]) if self.validation_split > 0.0 else None
        
        # Calculate running means
        if window > 0:
            if use_pandas and val_losses is not None:
                train_losses_rm = running_mean_func(train_losses, window)
                val_losses_rm = running_mean_func(val_losses, window)
            elif not use_pandas and val_losses is not None:
                train_losses_rm = running_mean_func(train_losses, window)
                val_losses_rm = running_mean_func(val_losses, window)
            else:
                train_losses_rm = running_mean_func(train_losses, window)
                val_losses_rm = None
            epochs_rm = range(1, min_len + 1)  # Align epochs with running mean data
        else:
            train_losses_rm = train_losses
            val_losses_rm = val_losses
            epochs_rm = epochs

        plt.figure(figsize=(20, 6))
        
        # Subplot 1: Loss Curves
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses,  markersize=6, color='navy', marker='o', linestyle='-', label='Training Loss', alpha=0.4, markerfacecolor='white', markeredgecolor='navy', markeredgewidth=1)
        if val_losses is not None:
            plt.plot(epochs, val_losses, markersize=6, color='orange', marker='o', linestyle='-', label='Validation Loss', alpha=0.4, markerfacecolor='white', markeredgecolor='orange', markeredgewidth=1)
        
        if window > 0:
            plt.plot(epochs_rm, train_losses_rm, color='navy', linewidth=3, label=f'Training Loss (Running Mean, window={window})')
            if val_losses_rm is not None:
                plt.plot(epochs_rm, val_losses_rm, color='orange', linewidth=3, label=f'Validation Loss (Running Mean, window={window})')
        
        plt.xlabel('Epochs')
        plt.yscale('log')  # Set the y-axis to a logarithmic scale
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Subplot 2: MSE Curves
        plt.subplot(1, 2, 2)
        if self.loss_recorder.train_mses and (self.loss_recorder.val_mses or self.validation_split == 0.0):
            train_mses = np.array(self.loss_recorder.train_mses[:min_len])
            val_mses = np.array(self.loss_recorder.val_mses[:min_len]) if self.validation_split > 0.0 else None
            
            if window > 0:
                if use_pandas and val_mses is not None:
                    train_mse_rm = running_mean_func(train_mses, window)
                    val_mse_rm = running_mean_func(val_mses, window)
                elif not use_pandas and val_mses is not None:
                    train_mse_rm = running_mean_func(train_mses, window)
                    val_mse_rm = running_mean_func(val_mses, window)
                else:
                    train_mse_rm = running_mean_func(train_mses, window)
                    val_mse_rm = None

            plt.plot(epochs, train_mses, markersize=6, color='navy', marker='o', linestyle='-', label='Training MSE', alpha=0.4, markerfacecolor='white', markeredgecolor='navy', markeredgewidth=1)
            if val_mses is not None:
                plt.plot(epochs, val_mses, markersize=6, color='orange', marker='o', linestyle='-', label='Validation MSE', alpha=0.4, markerfacecolor='white', markeredgecolor='orange', markeredgewidth=1)
            
            if window > 0:
                plt.plot(epochs_rm, train_mse_rm, color='navy', linewidth=3, label=f'Training MSE (Running Mean, window={window})')
                if val_mse_rm is not None:
                    plt.plot(epochs_rm, val_mse_rm, color='orange', linewidth=3, label=f'Validation MSE (Running Mean, window={window})')
        
            plt.ylabel('Mean Squared Error (MSE)')
            plt.title('Training and Validation MSE')
        else:
            plt.text(0.5, 0.5, 'No MSE data available.', horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
        plt.xlabel('Epochs')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
