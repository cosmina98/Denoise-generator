# Core dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any, Optional, Tuple, Union
import pytorch_lightning as pl
from torch_geometric.nn import NNConv, Linear, LayerNorm, global_mean_pool
from torch_geometric.data import HeteroData, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import os

# Import QuotientGraph type (required)
try:
    from coco_grape.module.quotientgraph.type import QuotientGraph
except ImportError:
    raise ImportError(
        "Could not import QuotientGraph from coco_grape. "
        "This module requires coco_grape to be installed."
    )

# Special label for association edges
ASSOCIATION_EDGE_LABEL = "contains"
MISSING_LABEL = "<missing>"  # Add constant for missing labels
# Edge types as tuples (src_type, edge_type, dst_type)
ASSOCIATION_EDGE_TYPE = ('image', ASSOCIATION_EDGE_LABEL, 'preimage')
REVERSE_ASSOCIATION_TYPE = ('preimage', 'in', 'image')
PREIMAGE_INTERNAL_TYPE = ('preimage', 'to', 'preimage')
IMAGE_INTERNAL_TYPE = ('image', 'to', 'image')

# Store node types for metadata
NODE_TYPES = ['preimage', 'image']
EDGE_TYPES = [
    PREIMAGE_INTERNAL_TYPE,
    IMAGE_INTERNAL_TYPE,
    ASSOCIATION_EDGE_TYPE,
    REVERSE_ASSOCIATION_TYPE
]

def worker_init_fn(worker_id: int):
    """Initialize each worker with deterministic seed based on base seed."""
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)

def _lazy_import_plotting():
    """Lazily import plotting dependencies only when needed."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        return plt, pd
    except ImportError:
        raise ImportError(
            "Plotting requires matplotlib and pandas. "
            "Install them with: pip install matplotlib pandas"
        )

class LossRecorder(pl.Callback):
    """
    PyTorch Lightning callback to record loss and accuracy metrics during training.
    """
    def __init__(self, verbose: int = 0):
        super().__init__()
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_accuracies: List[float] = []
        self.verbose = verbose

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        metrics = trainer.callback_metrics
        train_loss = metrics.get('train_loss', 0.0)
        train_acc = metrics.get('train_acc', 0.0)
        
        if isinstance(train_loss, torch.Tensor):
            train_loss = train_loss.item()
        if isinstance(train_acc, torch.Tensor):
            train_acc = train_acc.item()
            
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        metrics = trainer.callback_metrics
        val_loss = metrics.get('val_loss', 0.0)
        val_acc = metrics.get('val_acc', 0.0)
        
        if isinstance(val_loss, torch.Tensor):
            val_loss = val_loss.item()
        if isinstance(val_acc, torch.Tensor):
            val_acc = val_acc.item()
            
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)

        if self.verbose >= 1:
            print(f"Epoch {trainer.current_epoch}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

# ----------------------------------------------------------------------------
# 1. QuotientGraphEncoder 
# ----------------------------------------------------------------------------
class QuotientGraphEncoder(BaseEstimator, TransformerMixin):
    """
    Transforms QuotientGraph objects into dictionaries of tensors suitable for HMPN models.

    Fits OneHotEncoders for node and edge labels based on training data and handles
    concatenation with numerical node attributes.

    Parameters
    ----------
    node_attribute_dim : Optional[int], default=None
        The expected dimension of the numerical attribute vectors for nodes.
        If None, it will be inferred during `fit`. If attributes are missing
        or inconsistent, errors might occur or zero padding might be used.
    handle_unknown_node_labels : {'error', 'ignore', 'infrequent_if_exist'}, default='ignore'
        How to handle unknown node labels encountered during `transform`. Passed to OneHotEncoder.
        'ignore' outputs all zeros for unknown labels.
    handle_unknown_edge_labels : {'error', 'ignore', 'infrequent_if_exist'}, default='ignore'
        How to handle unknown edge labels encountered during `transform`. Passed to OneHotEncoder.
        'ignore' outputs all zeros for unknown labels.
    zero_if_missing_attribute : bool, default=True
        If True, replaces missing node attributes (None) with a zero vector of
        `node_attribute_dim_`. If False, raises an error.
    verbose : int, default=1
         Controls the verbosity: 0 = silent, 1 = progress messages.
    """

    def __init__(self,
                 node_attribute_dim: Optional[int] = None,
                 handle_unknown_node_labels: str = 'ignore',
                 handle_unknown_edge_labels: str = 'ignore',
                 zero_if_missing_attribute: bool = True,
                 verbose: int = 1): # Added verbose

        self.node_attribute_dim = node_attribute_dim
        self.handle_unknown_node_labels = handle_unknown_node_labels
        self.handle_unknown_edge_labels = handle_unknown_edge_labels
        self.zero_if_missing_attribute = zero_if_missing_attribute
        self.verbose = verbose # Store verbose

        # Attributes learned during fit
        self.node_label_encoder_: Optional[OneHotEncoder] = None
        self.edge_label_encoder_: Optional[OneHotEncoder] = None
        self.node_attribute_dim_: Optional[int] = node_attribute_dim
        self.node_feature_dim_: Optional[int] = None
        self.edge_feature_dim_: Optional[int] = None
        self.node_label_dim_: Optional[int] = None
        self.edge_label_dim_: Optional[int] = None
        self.has_node_labels_: bool = False
        self.has_edge_labels_: bool = False
        self._fitted = False

    def fit(self, X: List[QuotientGraph], y=None):
        """
        Fits the encoders based on the node/edge labels and attributes found in the training data.

        Parameters
        ----------
        X : List[QuotientGraph]
            A list of QuotientGraph objects to learn from.
        y : None
            Ignored. This parameter exists only for compatibility with
            sklearn pipelines.

        Returns
        -------
        self : QuotientGraphEncoder
            The fitted encoder instance.
        """
        if self.verbose >= 1: print("Fitting QuotientGraphEncoder...") # Added verbose
        all_node_labels = set()
        all_edge_labels = set([ASSOCIATION_EDGE_LABEL, MISSING_LABEL])  # Include missing label
        first_valid_attribute = None
        found_attribute_dim = None

        for i, qg in enumerate(X):
            if not isinstance(qg, QuotientGraph):
                 raise TypeError(f"Input element at index {i} is not a QuotientGraph object, but {type(qg)}")
            if not hasattr(qg, 'preimage_graph') or not hasattr(qg, 'image_graph'):
                 raise AttributeError(f"QuotientGraph at index {i} is missing 'preimage_graph' or 'image_graph'.")

            # Collect node labels and check attributes
            for graph_type, graph in [("preimage", qg.preimage_graph), ("image", qg.image_graph)]:
                if graph is None: continue
                for node, data in graph.nodes(data=True):
                    # Node Labels
                    label = data.get("label", None) # Assumes label function was applied
                    if label is not None:
                        # Convert label to string to ensure consistent sorting
                        label = str(label)
                        all_node_labels.add(label)

                    # Node Attributes (only needed for dim check if not provided)
                    if self.node_attribute_dim is None:
                        attr = data.get("attribute", None) # Assumes attr function was applied
                        if attr is not None:
                            # Convert scalar numeric types to numpy arrays
                            if np.isscalar(attr) or isinstance(attr, (int, float, np.number)):
                                attr = np.array([float(attr)])
                            
                            if not isinstance(attr, (np.ndarray, torch.Tensor)):
                                raise TypeError(f"Node attributes must be NumPy arrays, PyTorch Tensors, or numeric scalars, found {type(attr)}")
                            # Ensure attr is numpy for consistent shape checking
                            if isinstance(attr, torch.Tensor):
                                attr_np = attr.detach().cpu().numpy()
                            else:
                                attr_np = attr
                            current_dim = np.array(attr_np).flatten().shape[0] # Flatten to handle various shapes
                            if found_attribute_dim is None:
                                found_attribute_dim = current_dim
                            elif found_attribute_dim != current_dim:
                                raise ValueError(
                                    f"Inconsistent node attribute dimensions found: "
                                    f"{found_attribute_dim} vs {current_dim}. "
                                    f"Ensure attribute_function returns consistent shapes or set node_attribute_dim."
                                )

            # Collect edge labels
            for graph_type, graph in [("preimage", qg.preimage_graph), ("image", qg.image_graph)]:
                 if graph is None: continue
                 for _, _, data in graph.edges(data=True):
                    label = data.get("label", None) # Use 'label' as the default edge label key
                    if label is not None:
                        # Convert edge labels to strings as well
                        label = str(label)
                        all_edge_labels.add(label)

        # --- Finalize attribute dimension ---
        if self.node_attribute_dim is None:
            if found_attribute_dim is None:
                 if self.verbose >= 1: print("Warning: No node attributes found during fit. Setting node_attribute_dim_ to 0.") # Added verbose
                 self.node_attribute_dim_ = 0
            else:
                 self.node_attribute_dim_ = found_attribute_dim
                 if self.verbose >= 1: print(f"Inferred node_attribute_dim_: {self.node_attribute_dim_}") # Added verbose
        else:
            self.node_attribute_dim_ = self.node_attribute_dim # Use provided value
            if self.verbose >= 1: print(f"Using provided node_attribute_dim: {self.node_attribute_dim_}") # Added verbose

        # --- Fit Node Label Encoder ---
        node_labels_list = [[str(label)] for label in sorted(all_node_labels)]
        self.has_node_labels_ = bool(node_labels_list)
        
        if self.has_node_labels_:
            self.node_label_encoder_ = OneHotEncoder(
                handle_unknown=self.handle_unknown_node_labels,
                sparse_output=False,
                dtype=np.float32
            )
            self.node_label_encoder_.fit(node_labels_list)
            self.node_label_dim_ = self.node_label_encoder_.categories_[0].shape[0]
            if (self.handle_unknown_node_labels == 'infrequent_if_exist' and 
                hasattr(self.node_label_encoder_, 'infrequent_categories_') and 
                self.node_label_encoder_.infrequent_categories_ is not None):
                self.node_label_dim_ += 1
        else:
            self.node_label_encoder_ = None
            self.node_label_dim_ = 0
            if self.verbose >= 1:
                print("No node labels found. Skipping node label encoding.")

        # --- Fit Edge Label Encoder ---
        edge_labels_to_fit = sorted(str(label) for label in all_edge_labels)
        if ASSOCIATION_EDGE_LABEL not in edge_labels_to_fit:
            edge_labels_to_fit.append(ASSOCIATION_EDGE_LABEL)
            edge_labels_to_fit.sort()
        
        edge_labels_list = [[label] for label in edge_labels_to_fit]
        self.has_edge_labels_ = bool(edge_labels_list)
        
        if self.has_edge_labels_:
            self.edge_label_encoder_ = OneHotEncoder(
                handle_unknown=self.handle_unknown_edge_labels,
                sparse_output=False,
                dtype=np.float32
            )
            self.edge_label_encoder_.fit(edge_labels_list)
            self.edge_label_dim_ = self.edge_label_encoder_.categories_[0].shape[0]
            if (self.handle_unknown_edge_labels == 'infrequent_if_exist' and 
                hasattr(self.edge_label_encoder_, 'infrequent_categories_') and 
                self.edge_label_encoder_.infrequent_categories_ is not None):
                self.edge_label_dim_ += 1
        else:
            self.edge_label_encoder_ = None
            self.edge_label_dim_ = 0
            if self.verbose >= 1:
                print("No edge labels found. Skipping edge label encoding.")

        # --- Set final dimensions ---
        self.node_feature_dim_ = self.node_label_dim_ + self.node_attribute_dim_
        self.edge_feature_dim_ = self.edge_label_dim_

        if self.verbose >= 1:
            print(f"Node feature dimension: {self.node_feature_dim_} (Label: {self.node_label_dim_}, Attr: {self.node_attribute_dim_})")
            print(f"Edge feature dimension: {self.edge_feature_dim_}")
            print("Encoder fitting complete.")

        self._fitted = True
        return self

    def _get_node_features(self, graph: Any, node_map: Dict[Any, int]) -> torch.Tensor:
        """Extract and combine node features for a graph.
        
        Args:
            graph: networkx graph to process
            node_map: mapping from original node IDs to consecutive indices
        
        Returns:
            torch.Tensor of shape [num_nodes, feature_dim]
        """
        num_nodes = len(node_map)
        features = torch.zeros((num_nodes, self.node_feature_dim_), dtype=torch.float32)
        if num_nodes == 0:
            return features

        # Process labels if we have them
        if self.has_node_labels_:
            node_labels = []
            node_indices = []
            for node_id, data in graph.nodes(data=True):
                if node_id not in node_map:
                    continue
                label = data.get("label")
                if label is not None:
                    node_labels.append([str(label)])
                    node_indices.append(node_map[node_id])

            if node_labels:
                encoded_labels = self.node_label_encoder_.transform(node_labels)
                features[node_indices, :self.node_label_dim_] = torch.from_numpy(encoded_labels)

        # Process attributes if we have them
        if self.node_attribute_dim_ > 0:
            attr_start = self.node_label_dim_
            for node_id, data in graph.nodes(data=True):
                if node_id not in node_map:
                    continue
                    
                idx = node_map[node_id]
                attr = data.get("attribute")
                
                if attr is not None:
                    # Convert to tensor efficiently
                    if isinstance(attr, torch.Tensor):
                        attr_tensor = attr.float().view(-1)
                    else:
                        # numpy array or scalar
                        attr_tensor = torch.tensor(attr, dtype=torch.float32).view(-1)
                    
                    features[idx, attr_start:] = attr_tensor
                elif self.zero_if_missing_attribute:
                    # Already initialized to zero
                    pass
                else:
                    raise ValueError(f"Missing attribute for node {node_id}")

        return features

    def _get_edge_indices_and_attrs(self, graph: Any, node_map: Dict[Any, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract edge indices and features.
        
        Args:
            graph: networkx graph to process
            node_map: mapping from original node IDs to consecutive indices
            
        Returns:
            Tuple of (edge_index, edge_attr) tensors
        """
        num_edges = graph.number_of_edges()
        if num_edges == 0:
            return (torch.empty((2, 0), dtype=torch.long),
                    torch.empty((0, self.edge_feature_dim_), dtype=torch.float32))

        edge_indices = []
        edge_labels = []

        for u, v, data in graph.edges(data=True):
            if u not in node_map or v not in node_map:
                continue

            edge_indices.append((node_map[u], node_map[v]))
            label = data.get("label")
            # Convert None to MISSING_LABEL for consistency
            edge_labels.append([str(label) if label is not None else MISSING_LABEL])

        if not edge_indices:
            return (torch.empty((2, 0), dtype=torch.long),
                    torch.empty((0, self.edge_feature_dim_), dtype=torch.float32))

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        
        if self.has_edge_labels_:
            edge_attr = torch.from_numpy(self.edge_label_encoder_.transform(edge_labels))
        else:
            edge_attr = torch.empty((len(edge_indices), 0), dtype=torch.float32)

        return edge_index, edge_attr

    def _empty_tensors(self, prefix: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create empty node and edge tensors with proper device placement.
        
        Args:
            prefix: The graph type prefix ('preimage' or 'image')
            
        Returns:
            Tuple of (node features, edge indices, edge attributes)
        """
        x = torch.empty((0, self.node_feature_dim_), dtype=torch.float32)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, self.edge_feature_dim_), dtype=torch.float32)
        return x, edge_index, edge_attr

    def _add_graph(
        self, 
        data: HeteroData, 
        graph: Any, 
        node_type: str, 
        edge_type: Tuple[str, str, str],
        cross_edges: Optional[List[Tuple[str, str, str, dict]]] = None
    ) -> Dict[str, int]:
        """Add a graph's tensors to HeteroData with proper fallback for empty graphs.
        
        Args:
            data: HeteroData object to update
            graph: Graph to process (or None)
            node_type: Type of nodes ('preimage' or 'image')
            edge_type: Edge type tuple (src_type, rel_type, dst_type)
            cross_edges: Optional list of (src_type, rel_type, dst_type, data) tuples
                        for cross-graph relations
            
        Returns:
            Dictionary mapping original node IDs to consecutive indices, or empty dict if no graph
        """
        if not graph:
            x, edge_index, edge_attr = self._empty_tensors(node_type)
            data[node_type].x = x
            data[edge_type[0], edge_type[1], edge_type[2]].edge_index = edge_index
            data[edge_type[0], edge_type[1], edge_type[2]].edge_attr = edge_attr
            data[node_type].num_nodes = 0
            return {}
            
        # Process existing graph
        nodes = sorted(list(graph.nodes()))
        node_map = {node_id: i for i, node_id in enumerate(nodes)}
        
        x = self._get_node_features(graph, node_map)
        edge_index, edge_attr = self._get_edge_indices_and_attrs(graph, node_map)
        
        data[node_type].x = x
        data[edge_type[0], edge_type[1], edge_type[2]].edge_index = edge_index
        data[edge_type[0], edge_type[1], edge_type[2]].edge_attr = edge_attr
        data[node_type].num_nodes = x.size(0)

        # Handle cross-graph relations if provided
        if cross_edges:
            for src_type, rel_type, dst_type, rel_data in cross_edges:
                edge_data = data[src_type, rel_type, dst_type]
                cross_edges_list = []
                for src_node, node_data in graph.nodes(data=True):
                    if src_node not in node_map:
                        continue
                        
                    src_idx = node_map[src_node]
                    target_nodes = rel_data.get(src_node, [])
                    if hasattr(target_nodes, 'nodes'):  # Handle subgraph case
                        target_nodes = target_nodes.nodes()
                        
                    for dst_node in target_nodes:
                        if dst_node in rel_data.get('dst_map', {}):
                            dst_idx = rel_data['dst_map'][dst_node]
                            cross_edges_list.append((src_idx, dst_idx))

                if cross_edges_list:
                    edge_index = torch.tensor(cross_edges_list, dtype=torch.long).t()
                    edge_attr = self.edge_label_encoder_.transform([[rel_type]] * len(cross_edges_list))
                    edge_attr = torch.from_numpy(edge_attr).float()
                else:
                    edge_index, edge_attr = self._empty_tensors('cross')[1:]

                edge_data.edge_index = edge_index
                edge_data.edge_attr = edge_attr
        
        return node_map

    def _extract_tensors(self, qg: QuotientGraph) -> HeteroData:
        """Extract and pack tensors from a QuotientGraph into HeteroData format."""
        data = HeteroData()
        
        # Initialize metadata
        data.node_types = NODE_TYPES
        data.edge_types = EDGE_TYPES
        
        # Initialize empty node stores
        for node_type in NODE_TYPES:
            data[node_type].x = torch.empty((0, self.node_feature_dim_))
            data[node_type].num_nodes = 0
        
        # Add image graph first to get its node mapping
        img_map = self._add_graph(data, qg.image_graph, 'image', IMAGE_INTERNAL_TYPE)
        
        # Add preimage graph with cross-edges to image graph
        cross_edges = []
        if img_map:  # Only add cross edges if we have an image graph
            cross_edges = [
                ('preimage', 'in', 'image', {'dst_map': img_map}),
                ('image', ASSOCIATION_EDGE_LABEL, 'preimage', {
                    node: data.get('association', []) 
                    for node, data in qg.image_graph.nodes(data=True)
                })
            ]
            
        pre_map = self._add_graph(
            data, 
            qg.preimage_graph, 
            'preimage', 
            PREIMAGE_INTERNAL_TYPE,
            cross_edges
        )
        
        # Ensure all edge types have edge_index and edge_attr, even if empty
        for edge_type in EDGE_TYPES:
            if not hasattr(data[edge_type], 'edge_index'):
                data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
            if not hasattr(data[edge_type], 'edge_attr'):
                data[edge_type].edge_attr = torch.empty((0, self.edge_feature_dim_), dtype=torch.float32)
                
        # Validate all required attributes are present
        for edge_type in EDGE_TYPES:
            assert hasattr(data[edge_type], 'edge_index'), f"{edge_type} missing edge_index"
            assert hasattr(data[edge_type], 'edge_attr'), f"{edge_type} missing edge_attr"
                
        return data

    def encode_graph(self, qg: QuotientGraph) -> HeteroData:
        """Transform a single QuotientGraph into HeteroData format.
        
        Args:
            qg: QuotientGraph to encode
            
        Returns:
            HeteroData object containing graph tensors
        """
        if not self._fitted:
            raise RuntimeError("Call 'fit' first.")
            
        if not isinstance(qg, QuotientGraph):
            raise TypeError(f"Input must be a QuotientGraph, got {type(qg)}")
            
        return self._extract_tensors(qg)
    
    def encode_graphs(self, X: List[QuotientGraph]) -> List[HeteroData]:
        """Transform multiple QuotientGraphs into HeteroData format.
        
        Args:
            X: List of QuotientGraph objects to encode
            
        Returns:
            List of HeteroData objects
        """
        if not self._fitted:
            raise RuntimeError("Call 'fit' first.")

        results = [self.encode_graph(qg) for qg in X]
        
        if self.verbose >= 1:
            print(f"Encoded {len(results)} QuotientGraphs to HeteroData")
        return results
    
    def transform(self, X: List[QuotientGraph]) -> List[HeteroData]:
        """Legacy method - use encode_graphs() instead."""
        return self.encode_graphs(X)

# ----------------------------------------------------------------------------
# 2. Subgraph GNN 
# ----------------------------------------------------------------------------
class SubgraphQuotientGNN(pl.LightningModule):
    """
    Subgraph GNN for QuotientGraphs, manually implementing HMPN logic.
    Processes batched dictionaries from the custom collate function.
    """
    def __init__(self,
                 node_feat_dim: int,
                 edge_feat_dim: int,
                 hidden_dim: int,
                 output_dim: int, # Embedding dimension before classification
                 num_layers: int,
                 num_classes: int,
                 dropout: float = 0.2,
                 lr: float = 1e-3,
                 warmup_epochs: int = 10,
                 lr_start_factor: float = 0.1):
        super().__init__()
        self.save_hyperparameters()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_val = dropout # Renamed to avoid conflict
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.lr_start_factor = lr_start_factor

        # --- Initial Projections ---
        self.lin_p = Linear(node_feat_dim, hidden_dim)
        self.lin_q = Linear(node_feat_dim, hidden_dim)
        self.norm_p_init = LayerNorm(hidden_dim)
        self.norm_q_init = LayerNorm(hidden_dim)

        # --- Message Passing Layers (Separate for each type) ---
        self.convs_pp = nn.ModuleList() # Preimage -> Preimage
        self.convs_qq = nn.ModuleList() # Image -> Image
        self.convs_qp = nn.ModuleList() # Image -> Preimage (Downward)
        self.convs_pq = nn.ModuleList() # Preimage -> Image (Upward)

        self.norms_p = nn.ModuleList() # Norms for preimage updates
        self.norms_q = nn.ModuleList() # Norms for image updates
        self.norms_skip_p = nn.ModuleList() # Norms for preimage skip connections
        self.norms_skip_q = nn.ModuleList() # Norms for image skip connections
        
        # Learnable skip connection weights
        self.skip_weight_p = nn.Parameter(torch.ones(num_layers))
        self.skip_weight_q = nn.Parameter(torch.ones(num_layers))

        for _ in range(num_layers):
            # Edge network MLPs (can be shared or separate)
            nn_pp = nn.Sequential(nn.Linear(edge_feat_dim, hidden_dim * hidden_dim), nn.ReLU())
            nn_qq = nn.Sequential(nn.Linear(edge_feat_dim, hidden_dim * hidden_dim), nn.ReLU())
            nn_qp = nn.Sequential(nn.Linear(edge_feat_dim, hidden_dim * hidden_dim), nn.ReLU())
            nn_pq = nn.Sequential(nn.Linear(edge_feat_dim, hidden_dim * hidden_dim), nn.ReLU()) # For p->q

            self.convs_pp.append(NNConv(hidden_dim, hidden_dim, nn=nn_pp, aggr='sum'))
            self.convs_qq.append(NNConv(hidden_dim, hidden_dim, nn=nn_qq, aggr='sum'))
            self.convs_qp.append(NNConv(hidden_dim, hidden_dim, nn=nn_qp, aggr='sum'))
            self.convs_pq.append(NNConv(hidden_dim, hidden_dim, nn=nn_pq, aggr='sum')) # For p->q

            self.norms_p.append(LayerNorm(hidden_dim))
            self.norms_q.append(LayerNorm(hidden_dim))
            self.norms_skip_p.append(LayerNorm(hidden_dim))
            self.norms_skip_q.append(LayerNorm(hidden_dim))

        # --- Readout ---
        self.fc_embed = Linear(hidden_dim * 2, output_dim) # Concatenated pooled features
        self.classifier = Linear(output_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def _safe_pool(self, x: torch.Tensor, batch: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Helper function to safely pool node features, handling empty graphs.
        
        Args:
            x: Node features of shape [num_nodes, channels]
            batch: Batch assignment tensor of shape [num_nodes]
            batch_size: Total number of graphs in the batch
            
        Returns:
            Pooled features of shape [batch_size, channels]
        """
        if x.shape[0] == 0:  # No nodes
            return x.new_zeros((batch_size, self.hidden_dim))  # Device placement handled by Lightning
        
        # Use PyG's global_mean_pool with size hint to handle empty graphs
        return global_mean_pool(x, batch, size=batch_size)

    def _empty_features(self, prototype: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """Create empty feature tensor matching prototype's device and dtype."""
        shape = (batch_size or 0, self.hidden_dim)
        return torch.zeros(shape, dtype=prototype.dtype, device=self.device)

    def _apply_message_passing(
        self,
        conv: nn.Module,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Apply message passing if edges exist, return zeros if no edges.
        
        Args:
            conv: The convolutional layer to apply
            x: Node features (single tensor) or (source, target) tensors for bipartite
            edge_index: Edge connectivity tensor
            edge_attr: Edge feature tensor
        
        Returns:
            Result of message passing, or zero tensor matching input shape
        """
        if edge_index.numel() == 0:
            return x[1].new_zeros(x[1].shape) if isinstance(x, tuple) else x.new_zeros(x.shape)
        return conv(x, edge_index=edge_index, edge_attr=edge_attr)

    def forward(self, batch: Union[HeteroData, Batch]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass using PyG's heterogeneous batching."""
        assert isinstance(batch, (HeteroData, Batch)), f"Expected HeteroData or Batch, got {type(batch)}"
        
        # Move batch to correct device and access node features
        batch = batch.to(self.device)
        x_p = batch['preimage'].x
        x_q = batch['image'].x
        batch_size = batch.num_graphs

        # Validate feature dimensions
        if x_p.shape[1] != self.hparams.node_feat_dim:
            raise ValueError(f"Preimage node features have wrong dimension: {x_p.shape[1]}, expected {self.hparams.node_feat_dim}")
        if x_q.shape[1] != self.hparams.node_feat_dim:
            raise ValueError(f"Image node features have wrong dimension: {x_q.shape[1]}, expected {self.hparams.node_feat_dim}")

        # Initial projections with empty handling
        x_p = x_p if x_p.shape[0] > 0 else self._empty_features(x_p)
        x_q = x_q if x_q.shape[0] > 0 else self._empty_features(x_q)
        
        x_p = self.norm_p_init(self.lin_p(x_p)).relu()
        x_q = self.norm_q_init(self.lin_q(x_q)).relu()
        
        x_p_prev = x_p
        x_q_prev = x_q

        # Message passing layers
        for i in range(self.num_layers):
            # Initialize message aggregation
            msg_agg_p = torch.zeros_like(x_p)
            msg_agg_q = torch.zeros_like(x_q)

            # Within-graph messages
            msg_pp = self._apply_message_passing(
                self.convs_pp[i], x_p,
                batch[PREIMAGE_INTERNAL_TYPE].edge_index,
                batch[PREIMAGE_INTERNAL_TYPE].edge_attr
            )
            msg_agg_p = msg_agg_p + msg_pp

            msg_qq = self._apply_message_passing(
                self.convs_qq[i], x_q,
                batch[IMAGE_INTERNAL_TYPE].edge_index,
                batch[IMAGE_INTERNAL_TYPE].edge_attr
            )
            msg_agg_q = msg_agg_q + msg_qq

            # Cross-graph messages
            msg_qp = self._apply_message_passing(
                self.convs_qp[i], (x_q, x_p),
                batch[ASSOCIATION_EDGE_TYPE].edge_index,
                batch[ASSOCIATION_EDGE_TYPE].edge_attr
            )
            msg_agg_p = msg_agg_p + msg_qp

            msg_pq = self._apply_message_passing(
                self.convs_pq[i], (x_p, x_q),
                batch[REVERSE_ASSOCIATION_TYPE].edge_index,
                batch[REVERSE_ASSOCIATION_TYPE].edge_attr
            )
            msg_agg_q = msg_agg_q + msg_pq

            # Update with normalized residuals
            msg_agg_p = self.norms_p[i](msg_agg_p)
            msg_agg_q = self.norms_q[i](msg_agg_q)
            
            skip_p = self.norms_skip_p[i](x_p_prev)
            skip_q = self.norms_skip_q[i](x_q_prev)
            
            x_p_new = F.relu(msg_agg_p)
            x_q_new = F.relu(msg_agg_q)
            
            x_p_new = F.dropout(x_p_new, p=self.dropout_val, training=self.training)
            x_q_new = F.dropout(x_q_new, p=self.dropout_val, training=self.training)
            
            x_p = x_p_new + self.skip_weight_p[i] * skip_p
            x_q = x_q_new + self.skip_weight_q[i] * skip_q
            
            x_p_prev = x_p
            x_q_prev = x_q

        # Simplified pooling using helper function
        pooled_p = self._safe_pool(x_p, batch['preimage'].batch, batch_size)
        pooled_q = self._safe_pool(x_q, batch['image'].batch, batch_size)

        # Final prediction
        graph_embedding = self.fc_embed(torch.cat([pooled_p, pooled_q], dim=-1)).relu()
        logits = self.classifier(graph_embedding)

        return logits, graph_embedding, {'x_p': x_p, 'x_q': x_q}

    def training_step(self, batch: Batch, batch_idx: int):
        logits, _, _ = self(batch)
        loss = self.loss_fn(logits, batch['y'])
        acc = accuracy(logits.argmax(dim=-1), batch['y'], 
                      task="multiclass", num_classes=self.hparams.num_classes)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        logits, _, _ = self(batch)
        loss = self.loss_fn(logits, batch['y'])
        acc = accuracy(logits.argmax(dim=-1), batch['y'], 
                      task="multiclass", num_classes=self.hparams.num_classes)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def test_step(self, batch: Batch, batch_idx: int):
        logits, _, _ = self(batch)
        loss = self.loss_fn(logits, batch['y'])
        acc = accuracy(logits.argmax(dim=-1), batch['y'], 
                      task="multiclass", num_classes=self.hparams.num_classes)
        self.log('test_loss', loss, batch_size=batch.num_graphs)
        self.log('test_acc', acc, batch_size=batch.num_graphs)
        return loss

    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0):
        _, graph_embedding, _ = self(batch)
        return graph_embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.warmup_epochs > 0:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.lr_start_factor,
                total_iters=self.warmup_epochs
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return optimizer

# ----------------------------------------------------------------------------
# 3. Scikit-learn Wrapper 
# ----------------------------------------------------------------------------
class SubgraphQuotientGraphNeuralNetworkTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer using the manual HMPN GNN.
    Uses QuotientGraphEncoder and a custom collate function.
    """
    def __init__(
        self,
        hidden_dim: int = 64,
        output_dim: int = 128, # Embedding dimension before classification
        num_layers: int = 3,
        node_attribute_dim: Optional[int] = None, # Infer if None
        batch_size: int = 32,
        epochs: int = 100,
        validation_split: float = 0.2,
        random_state: int = 42,
        num_workers: Optional[int] = None,
        plot_training: bool = False,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 0.001,
        warmup_epochs: int = 10,
        lr_start_factor: float = 0.1,
        lr: float = 1e-3,
        dropout: float = 0.2,
        device: Optional[str] = None,
        verbose: int = 0,
        checkpoint_dir: str = "hmpn_dict_checkpoints" # Changed default dir
    ):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.node_attribute_dim = node_attribute_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.random_state = random_state
        self.num_workers = num_workers if num_workers is not None else max(1, os.cpu_count() // 2)
        self.plot_training = plot_training
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.warmup_epochs = warmup_epochs
        self.lr_start_factor = lr_start_factor
        self.lr = lr
        self.dropout = dropout
        self.device = "gpu" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            self.device = "cpu"
        self.verbose = verbose
        self.checkpoint_dir = checkpoint_dir

        # Use the provided QuotientGraphEncoder
        self.encoder = QuotientGraphEncoder(
            node_attribute_dim=self.node_attribute_dim,
            verbose=max(0, self.verbose - 1)
        )
        self.model: Optional[SubgraphQuotientGNN] = None
        self.is_fitted_ = False
        self.num_classes_ = None
        # Initialize loss recorder
        self.loss_recorder = LossRecorder(verbose=max(0, self.verbose - 1))
        self.trainer: Optional[pl.Trainer] = None

    def _setup_trainer(self) -> pl.Trainer:
        """Create a new trainer instance."""
        callbacks = [
            self.loss_recorder,
            EarlyStopping(
                monitor='val_loss',
                min_delta=self.early_stopping_min_delta,
                patience=self.early_stopping_patience,
                verbose=self.verbose >= 1,
                mode='min'
            ),
            ModelCheckpoint(
                monitor='val_loss',
                dirpath=self.checkpoint_dir,
                filename='best_model',
                save_top_k=1,
                mode='min',
                verbose=self.verbose >= 1
            )
        ]
        
        return pl.Trainer(
            max_epochs=self.epochs,
            logger=False,
            enable_checkpointing=True,
            accelerator="auto",  # Let Lightning handle device placement
            devices=1,
            deterministic=True,
            log_every_n_steps=10,
            enable_progress_bar=self.verbose >= 1,
            callbacks=callbacks,
            check_val_every_n_epoch=1
        )

    def _batch_forward(self, data_list: List[HeteroData]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Run forward pass on a batch of graphs."""
        batch = Batch.from_data_list(data_list, follow_batch=['preimage', 'image'])
        return self.model(batch)
        
    def _batch_predict(self, data_list: List[HeteroData]) -> torch.Tensor:
        """Get embeddings for a batch of graphs."""
        batch = Batch.from_data_list(data_list, follow_batch=['preimage', 'image'])
        return self.model.predict_step(batch, 0)

    def fit(self, X: List[QuotientGraph], y: Union[np.ndarray, List]):
        """ Fits the encoder and the GNN model using custom collate. """
        if self.verbose >= 1: print("Fitting SubgraphQuotientGraphNeuralNetworkTransformer (Manual HMPN)...")
        pl.seed_everything(self.random_state, workers=True)

        y = np.array(y)
        if y.dtype.kind not in {'i', 'u', 'b'}:
            raise ValueError("Target 'y' must be integer labels for classification.")
        self.num_classes_ = len(np.unique(y))
        if self.num_classes_ < 2:
             raise ValueError("Classification task requires at least 2 classes.")
        if self.verbose >= 1: print(f"Detected {self.num_classes_} classes.")

        # 1. Fit Encoder
        self.encoder.fit(X)
        if self.encoder.node_feature_dim_ is None or self.encoder.edge_feature_dim_ is None:
             raise RuntimeError("Encoder fitting failed to set feature dimensions.")

        # 2. Transform Data (List of HeteroData)
        hetero_data_list = self.encoder.encode_graphs(X)

        # Assign graph-level labels 'y' to each HeteroData
        for data, label in zip(hetero_data_list, y):
            data.y = torch.tensor(label, dtype=torch.long)

        # 3. Split Data (List of HeteroData)
        train_data, val_data = train_test_split(
            hetero_data_list,
            test_size=self.validation_split,
            random_state=self.random_state,
            stratify=y
        )
        if self.verbose >= 1:
            print(f"Training graphs: {len(train_data)}, Validation graphs: {len(val_data)}")

        # 4. Create DataLoaders with proper heterogeneous batching
        train_loader = PyGDataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=worker_init_fn,
            follow_batch=['preimage', 'image']
        )
        val_loader = PyGDataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,  
            worker_init_fn=worker_init_fn,
            follow_batch=['preimage', 'image']
        )

        # 5. Initialize Model
        self.model = SubgraphQuotientGNN(
            node_feat_dim=self.encoder.node_feature_dim_,
            edge_feat_dim=self.encoder.edge_feature_dim_,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=self.num_layers,
            num_classes=self.num_classes_,
            dropout=self.dropout,
            lr=self.lr,
            warmup_epochs=self.warmup_epochs,
            lr_start_factor=self.lr_start_factor
        )

        # 6. Setup Training (Callbacks, Trainer)
        self.trainer = self._setup_trainer()

        # 7. Train
        if self.verbose >= 1: print(f"Starting model training on device: {self.device}...")
        self.trainer.fit(self.model, train_loader, val_loader)

        # 8. Load Best Model and move to correct device
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        if best_model_path and os.path.exists(best_model_path):
            if self.verbose >= 1: print(f"Loading best model from {best_model_path}")
            self.model = SubgraphQuotientGNN.load_from_checkpoint(
                best_model_path,
                node_feat_dim=self.encoder.node_feature_dim_,
                edge_feat_dim=self.encoder.edge_feature_dim_,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                num_layers=self.num_layers,
                num_classes=self.num_classes_,
                dropout=self.dropout,
                lr=self.lr,
                warmup_epochs=self.warmup_epochs,
                lr_start_factor=self.lr_start_factor
            )
        else:
            if self.verbose >= 1: print("Warning: No best model checkpoint found. Using the model from the last training epoch.")
        self.model.eval()
        self.is_fitted_ = True

        if self.plot_training:
            self.plot_metrics()

        if self.verbose >= 1: print("Fitting complete.")
        return self

    def embed_graphs(self, X: List[QuotientGraph]) -> np.ndarray:
        """Get graph embeddings using the trained model."""
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted")
            
        # Use Lightning's predict for device handling
        data_loader = PyGDataLoader(
            self.encoder.encode_graphs(X),
            batch_size=self.batch_size,
            shuffle=False
        )
        embeddings = self.trainer.predict(self.model, data_loader)
        return torch.cat(embeddings).cpu().numpy()
    
    def embed_preimage_nodes(self, X: List[QuotientGraph]) -> List[np.ndarray]:
        """Get embeddings for preimage nodes."""
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted")
            
        results = []
        for qg in X:
            logits, _, node_embs = self._batch_forward([self.encoder.encode_graph(qg)])
            results.append(node_embs['x_p'].cpu().numpy())
        return results

    def embed_image_nodes(self, X: List[QuotientGraph]) -> List[np.ndarray]:
        """Get embeddings for image nodes."""
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted")
            
        results = []
        for qg in X:
            logits, _, node_embs = self._batch_forward([self.encoder.encode_graph(qg)])
            results.append(node_embs['x_q'].cpu().numpy())
        return results
    
    def predict(self, X: List[QuotientGraph]) -> np.ndarray:
        """Predict class labels for input graphs."""
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted")
            
        data_loader = PyGDataLoader(
            self.encoder.encode_graphs(X),
            batch_size=self.batch_size,
            shuffle=False,
            follow_batch=['preimage', 'image']  # Add follow_batch
        )
        
        all_logits = []
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                logits, _, _ = self.model(batch)  # Pass batch directly
                all_logits.append(logits)
            
        logits = torch.cat(all_logits, dim=0)
        return logits.argmax(dim=-1).cpu().numpy()
    
    def predict_proba(self, X: List[QuotientGraph]) -> np.ndarray:
        """Predict class probabilities for input graphs."""
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted")
            
        data_loader = PyGDataLoader(
            self.encoder.encode_graphs(X),
            batch_size=self.batch_size,
            shuffle=False,
            follow_batch=['preimage', 'image']  # Add follow_batch
        )
        
        all_logits = []
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                logits, _, _ = self.model(batch)  # Pass batch directly
                all_logits.append(logits)
            
        logits = torch.cat(all_logits, dim=0)
        return F.softmax(logits, dim=-1).cpu().numpy()
    
    # Legacy method names with deprecation warnings
    def transform(self, X):
        """Legacy method - use embed_graphs() instead."""
        return self.embed_graphs(X)
        
    def transform_preimage_nodes(self, X):
        """Legacy method - use embed_preimage_nodes() instead."""
        return self.embed_preimage_nodes(X)
        
    def transform_image_nodes(self, X):
        """Legacy method - use embed_image_nodes() instead."""
        return self.embed_image_nodes(X)

    def score(self, X, y):
        preds = self.predict(X)
        return (preds == np.array(y)).mean()


    def plot_metrics(self, window: int = 10):
        """Plots training and validation metrics."""
        plt, pd = _lazy_import_plotting()
        
        # Check if we have metrics to plot
        if not hasattr(self, 'loss_recorder') or not self.loss_recorder.train_losses:
            if self.verbose >= 1: print("No metrics available to plot. Model must be fitted first.")
            return

        def running_mean_symmetric(data, window):
            """Simple centered moving average using pandas."""
            if window <= 0: return np.array(data)
            s = pd.Series(data)
            # Use pandas' built-in centered rolling - no manual padding needed
            return s.rolling(window=2*window+1, center=True, min_periods=1).mean()

        min_len = min(len(self.loss_recorder.train_losses), len(self.loss_recorder.val_losses))
        if min_len == 0:
            if self.verbose >= 1: print("Not enough data points to plot metrics.")
            return
        epochs = range(1, min_len + 1)

        # Get raw metrics
        train_losses = np.array(self.loss_recorder.train_losses[:min_len])
        val_losses = np.array(self.loss_recorder.val_losses[:min_len])
        train_acc = np.array(self.loss_recorder.train_accuracies[:min_len])
        val_acc = np.array(self.loss_recorder.val_accuracies[:min_len])

        # Calculate smoothed metrics
        train_losses_smooth = running_mean_symmetric(train_losses, window)
        val_losses_smooth = running_mean_symmetric(val_losses, window)
        train_acc_smooth = running_mean_symmetric(train_acc, window)
        val_acc_smooth = running_mean_symmetric(val_acc, window)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot losses
        ax1.plot(epochs, train_losses, 'b-', alpha=0.3, label='Training')
        ax1.plot(epochs, val_losses, 'r-', alpha=0.3, label='Validation')
        ax1.plot(epochs, train_losses_smooth, 'b-', linewidth=2, label=f'Training (MA {window})')
        ax1.plot(epochs, val_losses_smooth, 'r-', linewidth=2, label=f'Validation (MA {window})')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_yscale('log')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Plot accuracies
        ax2.plot(epochs, train_acc, 'b-', alpha=0.3, label='Training')
        ax2.plot(epochs, val_acc, 'r-', alpha=0.3, label='Validation')
        ax2.plot(epochs, train_acc_smooth, 'b-', linewidth=2, label=f'Training (MA {window})')
        ax2.plot(epochs, val_acc_smooth, 'r-', linewidth=2, label=f'Validation (MA {window})')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training & Validation Accuracy')
        ax2.legend()
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Set accuracy y-limits with padding
        if val_acc.size > 0:
            ymin = max(0, np.min(val_acc) - 0.1)
            ymax = min(1.05, np.max(val_acc) + 0.1)
            ax2.set_ylim(bottom=ymin, top=ymax)

        plt.tight_layout()
        plt.show()

