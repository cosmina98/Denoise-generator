import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks import StochasticWeightAveraging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, global_mean_pool
from typing import List, Optional, Union
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytorch_lightning as pl
import pandas as pd

"""
The CrossAttentionConvolutionTransformerGraphNeuralNetwork is a sophisticated graph neural network 
architecture that seamlessly combines the strengths of NNConv layers with Transformer-based attention 
mechanisms to effectively model complex graph-structured data. At its core, the model employs 
multiple NNConv layers from PyTorch Geometric to capture local node and edge interactions, 
transforming raw graph data into rich feature representations. These convolutional layers are 
strategically integrated with custom Transformer Encoder layers that utilize cross-attention, 
allowing the model to dynamically align and integrate information across different layers of the 
network. This cross-attention mechanism ensures that the Transformer layers can focus on relevant 
features extracted by the NNConv layers, enhancing the model's ability to understand both local and 
global graph structures. Additionally, the architecture is designed to adapt the attention strategy 
based on the relative number of NNConv and Transformer layers, ensuring optimal information flow 
whether the Transformer component is deeper or shallower than the convolutional part. 
Built with PyTorch Lightning for efficient training and scalability, and compatible with 
scikit-learn for seamless integration into machine learning pipelines, 
CrossAttentionConvolutionTransformerGraphNeuralNetwork is well-suited for a variety of tasks 
including graph classification and regression, offering enhanced flexibility and performance by 
leveraging the complementary capabilities of convolutional operations and Transformer-based attention.
"""


# -----------------------------
# GraphEncoder Class
# -----------------------------
class GraphEncoder(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer for encoding NetworkX graphs into
    PyTorch Geometric Data objects. It handles node and edge label encoding,
    including an 'unknown' category for unseen labels, and concatenates one-hot
    encoded labels with continuous vector attributes if present.

    Parameters:
    - verbose: Verbosity level (0, 1, 2).
    """
    def __init__(self, verbose: int = 1):
        self.verbose = verbose
        self.node_label_encoder = LabelEncoder()
        self.edge_label_encoder = LabelEncoder()
        self.node_vec_dim = 0
        self.edge_vec_dim = 0
        self.is_fitted_ = False

    def fit(self, graphs: List[nx.Graph], y=None):
        """
        Fits the label encoders for node and edge labels based on the provided graphs.
        Adds an 'unknown' category to handle unseen labels during transformation.

        Parameters:
        - graphs: List of NetworkX graphs.
        - y: Ignored. Included for compatibility.

        Returns:
        - self
        """
        node_labels = []
        edge_labels = []
        for idx, G in enumerate(graphs):
            node_labels.extend([node[1]['label'] for node in G.nodes(data=True) if 'label' in node[1]])
            edge_labels.extend([edge[2]['label'] for edge in G.edges(data=True) if 'label' in edge[2]])

        # Add 'unknown' to handle unseen labels
        node_labels_with_unknown = node_labels + ['unknown']
        edge_labels_with_unknown = edge_labels + ['unknown']

        self.node_label_encoder.fit(node_labels_with_unknown)
        if edge_labels:
            self.edge_label_encoder.fit(edge_labels_with_unknown)
        else:
            # If no edges have labels, create a dummy encoder with 'unknown'
            self.edge_label_encoder.fit(['unknown'])

        # Determine 'vec' dimensions for nodes and edges
        node_vec_dims = [len(node[1]['vec']) for G in graphs for node in G.nodes(data=True) if 'vec' in node[1]]
        if node_vec_dims:
            if len(set(node_vec_dims)) != 1:
                if self.verbose >= 2:
                    print("All node 'vec' attributes must have the same dimension.")
                raise ValueError("All node 'vec' attributes must have the same dimension.")
            self.node_vec_dim = node_vec_dims[0]
        else:
            self.node_vec_dim = 0  # No 'vec' attributes for nodes

        edge_vec_dims = [len(edge[2]['vec']) for G in graphs for edge in G.edges(data=True) if 'vec' in edge[2]]
        if edge_vec_dims:
            if len(set(edge_vec_dims)) != 1:
                if self.verbose >= 2:
                    print("All edge 'vec' attributes must have the same dimension.")
                raise ValueError("All edge 'vec' attributes must have the same dimension.")
            self.edge_vec_dim = edge_vec_dims[0]
        else:
            self.edge_vec_dim = 0  # No 'vec' attributes for edges

        if self.verbose >= 1:
            print(f"Node Label Encoder Classes: {self.node_label_encoder.classes_}")
            print(f"Edge Label Encoder Classes: {self.edge_label_encoder.classes_}")
            print(f"Node 'vec' Dimension: {self.node_vec_dim}")
            print(f"Edge 'vec' Dimension: {self.edge_vec_dim}")

        self.is_fitted_ = True
        return self

    def transform(self, graphs: List[nx.Graph]) -> List[Data]:
        """
        Transforms the provided graphs into PyTorch Geometric Data objects with encoded features.

        Parameters:
        - graphs: List of NetworkX graphs.

        Returns:
        - List of PyTorch Geometric Data objects.
        """
        if not self.is_fitted_:
            raise RuntimeError("GraphEncoder must be fitted before calling transform.")

        data_list = []
        for idx, G in enumerate(graphs):
            # Encode nodes
            node_features = []
            for node in G.nodes(data=True):
                label = node[1].get('label')
                vec = node[1].get('vec') if self.node_vec_dim > 0 else None
                if label is None:
                    if self.verbose >= 2:
                        print(f"Graph {idx}: Each node must have a 'label' attribute.")
                    raise ValueError(f"Graph {idx}: Each node must have a 'label' attribute.")

                # Handle unseen labels by mapping to 'unknown'
                if label not in self.node_label_encoder.classes_:
                    label = 'unknown'

                label_encoded = self.node_label_encoder.transform([label])[0]
                label_one_hot = np.zeros(len(self.node_label_encoder.classes_), dtype=np.float32)
                label_one_hot[label_encoded] = 1.0

                if self.node_vec_dim > 0:
                    if vec is not None:
                        if len(vec) != self.node_vec_dim:
                            if self.verbose >= 2:
                                print(f"Graph {idx}: All node 'vec' attributes must have dimension {self.node_vec_dim}.")
                            raise ValueError(f"Graph {idx}: All node 'vec' attributes must have dimension {self.node_vec_dim}.")
                        vec = np.array(vec, dtype=np.float32)
                    else:
                        vec = np.zeros(self.node_vec_dim, dtype=np.float32)
                    feature = np.concatenate([label_one_hot, vec]).astype(np.float32)
                else:
                    feature = label_one_hot.astype(np.float32)
                node_features.append(feature)

            # Efficient Conversion: Stack and convert to tensor
            node_features = np.stack(node_features)  # Combine list into single NumPy array
            x = torch.from_numpy(node_features).float()

            # Encode edges
            edge_indices = []
            edge_features = []
            for edge in G.edges(data=True):
                u, v, attr = edge
                edge_indices.append([u, v])
                label = attr.get('label')
                vec = attr.get('vec') if self.edge_vec_dim > 0 else None
                if label is None:
                    if self.verbose >= 2:
                        print(f"Graph {idx}: Each edge must have a 'label' attribute.")
                    raise ValueError(f"Graph {idx}: Each edge must have a 'label' attribute.")

                # Handle unseen labels by mapping to 'unknown'
                if label not in self.edge_label_encoder.classes_:
                    label = 'unknown'

                label_encoded = self.edge_label_encoder.transform([label])[0]
                label_one_hot = np.zeros(len(self.edge_label_encoder.classes_), dtype=np.float32)
                label_one_hot[label_encoded] = 1.0

                if self.edge_vec_dim > 0:
                    if vec is not None:
                        if len(vec) != self.edge_vec_dim:
                            if self.verbose >= 2:
                                print(f"Graph {idx}: All edge 'vec' attributes must have dimension {self.edge_vec_dim}.")
                            raise ValueError(f"Graph {idx}: All edge 'vec' attributes must have dimension {self.edge_vec_dim}.")
                        vec = np.array(vec, dtype=np.float32)
                    else:
                        vec = np.zeros(self.edge_vec_dim, dtype=np.float32)
                    edge_attr_enc = np.concatenate([label_one_hot, vec]).astype(np.float32)
                else:
                    edge_attr_enc = label_one_hot.astype(np.float32)
                edge_features.append(edge_attr_enc)

            if edge_indices:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_features = np.stack(edge_features)  # Combine list into single NumPy array
                edge_attr = torch.from_numpy(edge_features).float()
                if self.verbose >= 2:
                    print(f"Graph {idx}: Encoded Edge Attributes Shape: {edge_attr.shape}")
            else:
                # Handle graphs with no edges
                edge_index = torch.empty((2, 0), dtype=torch.long)
                if self.edge_vec_dim > 0:
                    edge_attr = torch.empty((0, len(self.edge_label_encoder.classes_) + self.edge_vec_dim), dtype=torch.float)
                else:
                    edge_attr = torch.empty((0, len(self.edge_label_encoder.classes_)), dtype=torch.float)
                if self.verbose >= 2:
                    print(f"Graph {idx}: No edges present.")

            # Create Data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

            data_list.append(data)

        return data_list

    def fit_transform(self, graphs: List[nx.Graph], y=None) -> List[Data]:
        """
        Fits the encoder and transforms the graphs in one step.

        Parameters:
        - graphs: List of NetworkX graphs.
        - y: Ignored. Included for compatibility.

        Returns:
        - List of PyTorch Geometric Data objects.
        """
        return self.fit(graphs, y).transform(graphs)

# -----------------------------
# LossRecorder Class
# -----------------------------
class LossRecorder(pl.Callback):
    """
    PyTorch Lightning Callback to record training and validation losses,
    accuracies (for classification), and MSEs (for regression).
    Respects the verbosity levels:
        - verbose=0: No output.
        - verbose=1: Only training measurements info.
        - verbose=2: Additionally, graph construction checks.
    """
    def __init__(self, verbose: int = 1):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_mses = []
        self.val_mses = []
        self.epoch_count = 0  # To keep track of epochs
        self.verbose = verbose

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_count += 1
        if self.verbose >=2:
            print(f"[LossRecorder] on_train_epoch_end called for epoch {self.epoch_count}")
        # Retrieve the logged metrics
        train_loss = trainer.callback_metrics.get('train_loss')
        train_acc = trainer.callback_metrics.get('train_acc')
        train_mse = trainer.callback_metrics.get('train_mse')
        
        # Conditional Debugging statements based on verbosity
        if self.verbose >= 2:
            if train_loss is None:
                print(f"Warning: 'train_loss' not found for epoch {self.epoch_count}")
            if train_acc is None and train_mse is None:
                print(f"Warning: Neither 'train_acc' nor 'train_mse' found for epoch {self.epoch_count}")

        # Append losses if present
        if train_loss is not None:
            self.train_losses.append(train_loss.cpu().detach().item())
            if self.verbose >=2:
                print(f"Recorded train_loss: {train_loss.cpu().detach().item()}")
        else:
            if self.verbose >= 1:
                print(f"Skipping training loss recording for epoch {self.epoch_count} due to missing loss metrics.")

        # Append accuracies or MSEs if present
        if train_acc is not None:
            self.train_accuracies.append(train_acc.cpu().detach().item())
            if self.verbose >=2:
                print(f"Recorded train_acc: {train_acc.cpu().detach().item()}")
        elif train_mse is not None:
            self.train_mses.append(train_mse.cpu().detach().item())
            if self.verbose >=2:
                print(f"Recorded train_mse: {train_mse.cpu().detach().item()}")
        else:
            if self.verbose >= 1:
                print(f"Warning: Incomplete training metrics for epoch {self.epoch_count}. Skipping metric recording.")

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.verbose >=2:
            print(f"[LossRecorder] on_validation_epoch_end called for epoch {self.epoch_count}")
        # Retrieve the logged metrics
        val_loss = trainer.callback_metrics.get('val_loss')
        val_acc = trainer.callback_metrics.get('val_acc')
        val_mse = trainer.callback_metrics.get('val_mse')
        
        # Conditional Debugging statements based on verbosity
        if self.verbose >= 2:
            if val_loss is None:
                print(f"Warning: 'val_loss' not found for epoch {self.epoch_count}")
            if val_acc is None and val_mse is None:
                print(f"Warning: Neither 'val_acc' nor 'val_mse' found for epoch {self.epoch_count}")

        # Append losses if present
        if val_loss is not None:
            self.val_losses.append(val_loss.cpu().detach().item())
            if self.verbose >=2:
                print(f"Recorded val_loss: {val_loss.cpu().detach().item()}")
        else:
            if self.verbose >= 1:
                print(f"Skipping validation loss recording for epoch {self.epoch_count} due to missing loss metrics.")

        # Append accuracies or MSEs if present
        if val_acc is not None:
            self.val_accuracies.append(val_acc.cpu().detach().item())
            if self.verbose >=2:
                print(f"Recorded val_acc: {val_acc.cpu().detach().item()}")
        elif val_mse is not None:
            self.val_mses.append(val_mse.cpu().detach().item())
            if self.verbose >=2:
                print(f"Recorded val_mse: {val_mse.cpu().detach().item()}")
        else:
            if self.verbose >= 1:
                print(f"Warning: Incomplete validation metrics for epoch {self.epoch_count}. Skipping metric recording.")

# -----------------------------
# CrossTransformerEncoderLayer Class
# -----------------------------
class CrossTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super(CrossTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # Activation function
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        cross_src: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
        cross_key_padding_mask: Optional[torch.Tensor] = None,
        perform_cross_attention: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for the custom Transformer Encoder layer.

        Parameters:
        - src: Tensor of shape [seq_len, batch_size, d_model]
        - src_mask: Optional tensor for self-attention masking
        - src_key_padding_mask: Optional tensor for self-attention key padding
        - cross_src: Tensor for cross-attention [seq_len_cross, batch_size, d_model]
        - cross_mask: Optional tensor for cross-attention masking
        - cross_key_padding_mask: Optional tensor for cross-attention key padding
        - perform_cross_attention: Bool flag to perform cross-attention

        Returns:
        - Tensor of shape [seq_len, batch_size, d_model]
        """
        # Self-Attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        if perform_cross_attention and cross_src is not None:
            # Cross-Attention: Query from src, Key and Value from cross_src
            src2 = self.cross_attn(src, cross_src, cross_src, attn_mask=cross_mask, key_padding_mask=cross_key_padding_mask)[0]
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        
        # Feedforward Network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm3(src)
        
        return src

# -----------------------------
# CrossAttentionConvolutionTransformerGraphNeuralNetworkEncoder Class
# -----------------------------
class CrossAttentionConvolutionTransformerGraphNeuralNetworkEncoder(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        edge_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        task_type: str = 'classification',
        num_classes: Optional[int] = None,
        dropout_prob: float = 0.2,
        aggr: str = 'sum',
        warmup_start_lr: float = 1e-4,
        warmup_end_lr: float = 1e-3,
        warmup_epochs: int = 50,
        verbose: int =1,
        transformer_hidden_dim: int = 128,
        transformer_num_layers: int = 2,
        transformer_num_heads: int = 8
    ):
        """
        CrossAttentionConvolutionTransformerGraphNeuralNetworkEncoder: A PyTorch Lightning module that integrates NNConv layers with a custom Transformer Encoder
        supporting cross-attention between NNConv layers and Transformer layers.
        """
        super(CrossAttentionConvolutionTransformerGraphNeuralNetworkEncoder, self).__init__()
        self.save_hyperparameters()
        self.task_type = task_type
        self.dropout_prob = dropout_prob
        self.aggr = aggr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_end_lr = warmup_end_lr
        self.warmup_epochs = warmup_epochs
        self.verbose = verbose

        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.leaky_relus = nn.ModuleList()
        self.residuals = nn.ModuleList()

        # Define NNConv layers
        # Input layer
        nn_input = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim * input_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim * input_dim, hidden_dim * input_dim)
        )
        self.convs.append(NNConv(in_channels=input_dim, out_channels=hidden_dim, nn=nn_input, aggr=self.aggr))
        self.layer_norms.append(nn.LayerNorm(hidden_dim))
        self.dropouts.append(nn.Dropout(p=self.dropout_prob))
        self.leaky_relus.append(nn.LeakyReLU())
        
        # Initialize residual connections for the input layer
        if input_dim != hidden_dim:
            self.residuals.append(nn.Linear(input_dim, hidden_dim))
        else:
            self.residuals.append(nn.Identity())

        # Hidden layers
        for i in range(num_layers - 1):
            nn_hidden = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim * hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim * hidden_dim, hidden_dim * hidden_dim)
            )
            self.convs.append(NNConv(in_channels=hidden_dim, out_channels=hidden_dim, nn=nn_hidden, aggr=self.aggr))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(p=self.dropout_prob))
            self.leaky_relus.append(nn.LeakyReLU())
            
            # Initialize residual connections for hidden layers
            self.residuals.append(nn.Identity())

        # Define the Transformer Encoder with Cross-Attention
        self.transformer_layers = nn.ModuleList([
            CrossTransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=transformer_num_heads,
                dim_feedforward=transformer_hidden_dim,
                dropout=dropout_prob,
                activation='relu'
            )
            for _ in range(transformer_num_layers)
        ])
        
        # Final embedding layer
        self.fc_embed = nn.Linear(hidden_dim, output_dim)

        # Task-specific head
        if self.task_type == 'classification':
            if num_classes is None:
                raise ValueError("num_classes must be provided for classification tasks.")
            self.fc_out = nn.Linear(output_dim, num_classes)
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.task_type == 'regression':
            self.fc_out = nn.Linear(output_dim, 1)
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError("task_type must be either 'classification' or 'regression'.")

    def forward(self, x, edge_index, edge_attr, batch):
        nnconv_embeddings = []  # To store embeddings after each NNConv layer

        # Pass through NNConv layers
        for conv, layer_norm, dropout, leaky_relu, residual in zip(
            self.convs, self.layer_norms, self.dropouts, self.leaky_relus, self.residuals
        ):
            x_in = x  # Store input for residual connection
            x = conv(x, edge_index, edge_attr)
            if x.ndim > 1 and x.shape[0] > 1:
                x = layer_norm(x)
            x = leaky_relu(x)
            x = dropout(x)
            x = x + residual(x_in)  # Add residual connection
            nnconv_embeddings.append(x)  # Save the embedding after this layer

        # Transformer Encoder Integration with Cross-Attention
        num_nnconv = len(self.convs)
        num_transformer = len(self.transformer_layers)

        transformer_output = x  # Initialize with the last NNConv layer's output

        for layer_idx, transformer_layer in enumerate(self.transformer_layers):
            if num_transformer > num_nnconv:
                # More Transformer layers than NNConv layers
                if layer_idx < num_nnconv:
                    # Initial Transformer layers use cross-attention with corresponding NNConv layers
                    corresponding_nnconv_idx = layer_idx
                    cross_src = nnconv_embeddings[corresponding_nnconv_idx]
                    perform_cross_attention = True
                    if self.verbose >= 2:
                        print(f"Transformer Layer {layer_idx + 1}: Using cross-attention with NNConv Layer {corresponding_nnconv_idx + 1}")
                else:
                    # Remaining Transformer layers use self-attention
                    cross_src = None
                    perform_cross_attention = False
                    if self.verbose >= 2:
                        print(f"Transformer Layer {layer_idx + 1}: Using self-attention")
            else:
                # Transformer layers ≤ NNConv layers
                corresponding_nnconv_idx = num_nnconv - num_transformer + layer_idx
                cross_src = nnconv_embeddings[corresponding_nnconv_idx]
                perform_cross_attention = True
                if self.verbose >= 2:
                    print(f"Transformer Layer {layer_idx + 1}: Using cross-attention with NNConv Layer {corresponding_nnconv_idx + 1}")

            # Prepare inputs for Transformer layer
            # Transformer expects [seq_len, batch_size, d_model]
            # PyG provides [num_nodes, d_model], need to add batch dimension
            src = transformer_output.unsqueeze(1)  # [num_nodes, 1, d_model]
            if cross_src is not None:
                cross_src_t = cross_src.unsqueeze(1)  # [num_nodes_cross, 1, d_model]
            else:
                cross_src_t = None

            # Pass through the Transformer layer
            transformer_output = transformer_layer(
                src,
                perform_cross_attention=perform_cross_attention,
                cross_src=cross_src_t
            ).squeeze(1)  # [num_nodes, d_model]

        x = transformer_output

        # Global pooling
        x = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        # Embedding
        embed = self.fc_embed(x)  # [batch_size, output_dim]
        return embed

    def get_node_embeddings(self, x, edge_index, edge_attr, batch):
        """
        Returns the node embeddings after the last transformer layer but before pooling.
        """
        nnconv_embeddings = []

        # Pass through NNConv layers
        for conv, layer_norm, dropout, leaky_relu, residual in zip(
            self.convs, self.layer_norms, self.dropouts, self.leaky_relus, self.residuals
        ):
            x_in = x  # Store input for residual connection
            x = conv(x, edge_index, edge_attr)
            if x.ndim > 1 and x.shape[0] > 1:
                x = layer_norm(x)
            x = leaky_relu(x)
            x = dropout(x)
            x = x + residual(x_in)  # Add residual connection
            nnconv_embeddings.append(x)  # Save the embedding after this layer

        # Transformer Encoder Integration with Cross-Attention
        num_nnconv = len(self.convs)
        num_transformer = len(self.transformer_layers)

        transformer_output = x  # Initialize with the last NNConv layer's output
        node_embeddings = transformer_output  # To keep track before pooling

        for layer_idx, transformer_layer in enumerate(self.transformer_layers):
            if num_transformer > num_nnconv:
                if layer_idx < num_nnconv:
                    corresponding_nnconv_idx = layer_idx
                    cross_src = nnconv_embeddings[corresponding_nnconv_idx]
                    perform_cross_attention = True
                    if self.verbose >= 2:
                        print(f"Transformer Layer {layer_idx + 1}: Using cross-attention with NNConv Layer {corresponding_nnconv_idx + 1}")
                else:
                    cross_src = None
                    perform_cross_attention = False
                    if self.verbose >= 2:
                        print(f"Transformer Layer {layer_idx + 1}: Using self-attention")
            else:
                corresponding_nnconv_idx = num_nnconv - num_transformer + layer_idx
                cross_src = nnconv_embeddings[corresponding_nnconv_idx]
                perform_cross_attention = True
                if self.verbose >= 2:
                    print(f"Transformer Layer {layer_idx + 1}: Using cross-attention with NNConv Layer {corresponding_nnconv_idx + 1}")

            # Prepare inputs for Transformer layer
            src = transformer_output.unsqueeze(1)  # [num_nodes, 1, d_model]
            if cross_src is not None:
                cross_src_t = cross_src.unsqueeze(1)  # [num_nodes_cross, 1, d_model]
            else:
                cross_src_t = None

            # Pass through the Transformer layer
            transformer_output = transformer_layer(
                src,
                perform_cross_attention=perform_cross_attention,
                cross_src=cross_src_t
            ).squeeze(1)  # [num_nodes, d_model]

            node_embeddings = transformer_output  # Update node embeddings

        return node_embeddings  # Return node embeddings before pooling

    def training_step(self, batch, batch_idx):
        embed = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        if self.task_type == 'classification':
            logits = self.fc_out(embed)
            loss = self.loss_fn(logits, batch.y)
            preds = logits.argmax(dim=1)
            acc = accuracy_score(batch.y.cpu(), preds.cpu())
            self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
            self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        elif self.task_type == 'regression':
            preds = self.fc_out(embed).squeeze()
            loss = self.loss_fn(preds, batch.y.float())
            mse = mean_squared_error(batch.y.cpu(), preds.cpu())
            self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
            self.log('train_mse', mse, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        embed = self.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        if self.task_type == 'classification':
            logits = self.fc_out(embed)
            loss = self.loss_fn(logits, batch.y)
            preds = logits.argmax(dim=1)
            acc = accuracy_score(batch.y.cpu(), preds.cpu())
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
            self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        elif self.task_type == 'regression':
            preds = self.fc_out(embed).squeeze()
            loss = self.loss_fn(preds, batch.y.float())
            mse = mean_squared_error(batch.y.cpu(), preds.cpu())
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
            self.log('val_mse', mse, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.warmup_end_lr)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=self.warmup_start_lr / self.warmup_end_lr,
            total_iters=self.warmup_epochs,
            verbose=self.verbose >=1  # Verbose output based on level
        )
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler_dict]

# -----------------------------
# CrossAttentionConvolutionTransformerGraphNeuralNetworkTransformer Class
# -----------------------------
class CrossAttentionConvolutionTransformerGraphNeuralNetworkTransformer(BaseEstimator, TransformerMixin):
    """
    CrossAttentionConvolutionTransformerGraphNeuralNetworkTransformer is a 
    Scikit-Learn compatible transformer designed for graph-structured data. 
    It integrates Neural Network Convolution (NNConv) layers from PyTorch Geometric 
    with custom Transformer Encoder layers that incorporate cross-attention mechanisms. 
    This architecture leverages the strengths of both convolutional operations and 
    Transformer-based attention to capture complex local and global interactions 
    within graphs.

    Additionally, the transformer supports Stochastic Weight Averaging (SWA) 
    to enhance model generalization by averaging multiple points along the 
    optimization trajectory. The class is built using PyTorch Lightning for 
    efficient training and scalability and is compatible with scikit-learn's 
    pipeline, allowing seamless integration into machine learning workflows.

    Parameters
    ----------
    hidden_dim : int, default=64
        The dimensionality of the hidden layers in the NNConv modules.
    
    output_dim : int, default=128
        The dimensionality of the output embeddings produced by the model.
    
    edge_hidden_dim : int, default=64
        The dimensionality of the hidden layers in the edge feature networks.
    
    num_layers : int, default=2
        The number of NNConv layers in the convolutional part of the network.
    
    transformer_hidden_dim : int, default=128
        The dimensionality of the hidden layers in the Transformer Encoder.
    
    transformer_num_layers : int, default=2
        The number of Transformer Encoder layers to be stacked.
    
    transformer_num_heads : int, default=8
        The number of attention heads in each Transformer Encoder layer.
    
    batch_size : int, default=32
        The number of graphs per batch during training and inference.
    
    epochs : int, default=10
        The maximum number of training epochs.
    
    validation_split : float, default=0.2
        The proportion of the dataset to include in the validation split.
    
    random_state : int, default=42
        The seed used by the random number generator for reproducibility.
    
    num_workers : Optional[int], default=None
        The number of worker processes to use for data loading. If None, 
        it defaults to the number of CPU cores available.
    
    plot_training : bool, default=False
        If set to True, plots training and validation metrics after training.
    
    early_stopping_patience : int, default=5
        The number of epochs with no improvement after which training will be 
        stopped early.
    
    early_stopping_min_delta : float, default=0.00
        The minimum change in the monitored metric to qualify as an improvement 
        for early stopping.
    
    warmup_start_lr : float, default=1e-4
        The initial learning rate at the start of the warmup phase.
    
    warmup_end_lr : float, default=1e-3
        The target learning rate at the end of the warmup phase.
    
    warmup_epochs : int, default=50
        The number of epochs over which to linearly increase the learning rate 
        from `warmup_start_lr` to `warmup_end_lr`.
    
    device : Optional[str], default=None
        The device to run the model on ('cuda' or 'cpu'). If None, 
        automatically determined based on availability.
    
    aggr : str, default='sum'
        The aggregation method for NNConv layers. Common options include 'add', 
        'mean', 'max', and 'sum'.
    
    dropout_prob : float, default=0.2
        The probability of an element to be zeroed in dropout layers.
    
    verbose : int, default=1
        The verbosity level:
            - 0: No output.
            - 1: Training and validation information.
            - 2: Additional graph construction checks and detailed logs.
    
    swa_enable : bool, default=False
        Whether to enable Stochastic Weight Averaging (SWA). 
        SWA can improve model generalization by averaging weights over multiple training epochs.
    
    swa_start : int, default=75
        The epoch number to start applying SWA. SWA will begin after this epoch.
    
    swa_freq : int, default=5
        The frequency (in epochs) at which SWA updates are applied.
        For example, with `swa_freq=5`, SWA will update every 5 epochs after `swa_start`.
    
    swa_lr : Optional[float], default=None
        The learning rate for SWA. If set to None, the optimizer's current learning rate is used.

    Attributes
    ----------
    graph_encoder : GraphEncoder
        A scikit-learn compatible transformer for encoding NetworkX graphs into 
        PyTorch Geometric Data objects.
    
    model : CrossAttentionConvolutionTransformerGraphNeuralNetworkEncoder
        The PyTorch Lightning model integrating NNConv and Transformer layers.
    
    loss_recorder : LossRecorder
        A PyTorch Lightning callback to record training and validation losses, 
        accuracies, and MSEs.
    
    is_fitted_ : bool
        Indicates whether the transformer has been fitted. Set to True after calling `fit`.
    
    task_type : str
        The type of task ('classification' or 'regression') inferred from the target variable.
    
    num_classes : Optional[int]
        The number of classes in the classification task. Set only if `task_type` is 'classification'.
    
    Notes
    -----
    - This transformer requires that each node and edge in the input NetworkX graphs 
      has a 'label' attribute. Optionally, nodes and edges can have a 'vec' attribute 
      containing continuous feature vectors.
    
    - The transformer integrates seamlessly with scikit-learn pipelines, allowing 
      for combined preprocessing and modeling workflows.
    
    - SWA can significantly enhance model performance, especially in scenarios where 
      the loss landscape benefits from weight averaging. It is recommended to enable 
      SWA for large-scale training regimes.
    
    Examples
    --------
    >>> import networkx as nx
    >>> from sklearn.datasets import make_classification
    >>> # Create sample graphs
    >>> G1 = nx.Graph()
    >>> G1.add_node(0, label='A', vec=[0.1, 0.2])
    >>> G1.add_node(1, label='B', vec=[0.3, 0.4])
    >>> G1.add_edge(0, 1, label='connects', vec=[0.5])
    >>> G2 = nx.Graph()
    >>> G2.add_node(0, label='A', vec=[0.2, 0.1])
    >>> G2.add_node(1, label='C', vec=[0.4, 0.3])
    >>> G2.add_edge(0, 1, label='connects', vec=[0.6])
    >>> graphs = [G1, G2]
    >>> y = [0, 1]  # Classification labels
    >>> # Initialize the transformer
    >>> transformer = CrossAttentionConvolutionTransformerGraphNeuralNetworkTransformer(
    ...     hidden_dim=128,
    ...     output_dim=256,
    ...     num_layers=3,
    ...     transformer_hidden_dim=256,
    ...     transformer_num_layers=4,
    ...     transformer_num_heads=16,
    ...     batch_size=2,
    ...     epochs=100,
    ...     swa_enable=True,
    ...     swa_start=75,
    ...     swa_freq=5,
    ...     swa_lr=1e-3,
    ...     verbose=2,
    ...     plot_training=True
    ... )
    >>> # Fit the transformer
    >>> transformer.fit(graphs=graphs, y=y)
    >>> # Transform new graphs into embeddings
    >>> embeddings = transformer.transform(new_graphs)
    >>> # Get node embeddings
    >>> node_embeddings = transformer.node_transform(new_graphs)
    >>> # Integrate into scikit-learn pipeline
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.pipeline import Pipeline
    >>> pipeline = Pipeline([
    ...     ('graph_transformer', transformer),
    ...     ('classifier', LogisticRegression())
    ... ])
    >>> pipeline.fit(graphs, y)
    >>> predictions = pipeline.predict(new_graphs)

    References
    ----------
    - [SWA Paper](https://arxiv.org/abs/1803.05407): Stochastic Weight Averaging for Deep Learning.
    - [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
    - [PyTorch Lightning Documentation](https://pytorch-lightning.readthedocs.io/)
    """
    def __init__(
        self,
        hidden_dim: int = 64,
        output_dim: int = 128,
        edge_hidden_dim: int = 64,
        num_layers: int = 2,
        transformer_hidden_dim: int = 128,  # Hidden dimension for Transformer
        transformer_num_layers: int = 2,    # Number of Transformer layers
        transformer_num_heads: int = 8,     # Number of attention heads in Transformer
        batch_size: int = 32,
        epochs: int = 10,
        validation_split: float = 0.2,
        random_state: int = 42,
        num_workers: Optional[int] = None,  # Number of worker processes for data loading
        plot_training: bool = False,        # If True, plots training and validation metrics
        early_stopping_patience: int = 5,   # Number of epochs with no improvement for early stopping
        early_stopping_min_delta: float = 0.00,  # Minimum change to qualify as improvement
        warmup_start_lr: float = 1e-4,      # Initial learning rate for warmup
        warmup_end_lr: float = 1e-3,        # Target learning rate after warmup
        warmup_epochs: int = 50,            # Number of epochs over which to perform warmup
        device: Optional[str] = None,       # Device to run the model on ('cuda' or 'cpu'). If None, automatically determined.
        aggr: str = 'sum',                   # Aggregation method for NNConv layers (default 'sum')
        dropout_prob: float = 0.2,          # Dropout probability (default 0.2)
        verbose: int =1,                     # Verbosity level: 0=no output, 1=training info, 2=additional graph checks
        swa_enable: bool = True,           # Enable Stochastic Weight Averaging
        swa_start: int = 75,                 # Epoch to start SWA
        swa_freq: int = 5,                   # Frequency of SWA updates
        swa_lr: Optional[float] = None       # Learning rate for SWA. If None, uses the optimizer's lr
    ):
        """
        CrossAttentionConvolutionTransformerGraphNeuralNetworkTransformer: A Scikit-Learn Compatible Transformer for Graph Neural Networks with Transformer Integration.

        [Comprehensive docstring as provided earlier]
        """
        super(CrossAttentionConvolutionTransformerGraphNeuralNetworkTransformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.random_state = random_state
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.plot_training = plot_training
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.warmup_start_lr = warmup_start_lr
        self.warmup_end_lr = warmup_end_lr
        self.warmup_epochs = warmup_epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.aggr = aggr
        self.dropout_prob = dropout_prob
        self.transformer_hidden_dim = transformer_hidden_dim
        self.transformer_num_layers = transformer_num_layers
        self.transformer_num_heads = transformer_num_heads
        self.verbose = verbose

        # SWA Parameters
        self.swa_enable = swa_enable
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_lr = swa_lr

        # Initialize GraphEncoder
        self.graph_encoder = GraphEncoder(verbose=self.verbose >=2)
        
        self.model = None  # Will be initialized in fit
        self.is_fitted_ = False
        self.task_type = None
        self.num_classes = None
        self.loss_recorder = LossRecorder(verbose=self.verbose >=2)  # Initialize the LossRecorder callback

    def fit(
        self,
        graphs: List[nx.Graph],
        y: Optional[Union[np.ndarray, List]] = None,
        checkpoint_dir: str = "checkpoints"  # Set default checkpoint directory
    ):
        """
        Fits the GCN model on the provided graphs, integrating SWA if enabled.

        Parameters:
        - graphs: List of NetworkX graphs.
        - y: List or array of labels.
        - checkpoint_dir: Directory to save model checkpoints.
        
        Returns:
        - self
        """
        if y is None:
            if self.verbose >=1:
                print("y cannot be None. It must be provided to determine the task type.")
            raise ValueError("y cannot be None. It must be provided to determine the task type.")

        # Determine task type
        y = np.array(y)
        if y.dtype.kind in {'i', 'u', 'b'}:
            self.task_type = 'classification'
            self.num_classes = len(np.unique(y))
        elif y.dtype.kind in {'f'}:
            self.task_type = 'regression'
        else:
            if self.verbose >=1:
                print("y must be either integer (for classification) or float (for regression).")
            raise ValueError("y must be either integer (for classification) or float (for regression).")

        # Fit the GraphEncoder
        self.graph_encoder.fit(graphs, y)

        # Encode graphs
        data_list = self.graph_encoder.transform(graphs)

        # Assign labels to data
        for data, label in zip(data_list, y):
            if self.task_type == 'classification':
                data.y = torch.tensor(label, dtype=torch.long)
            elif self.task_type == 'regression':
                data.y = torch.tensor(label, dtype=torch.float)
            # Add assertions
            assert hasattr(data, 'x'), "Data object missing 'x' attribute."
            assert hasattr(data, 'edge_index'), "Data object missing 'edge_index' attribute."
            assert hasattr(data, 'edge_attr'), "Data object missing 'edge_attr' attribute."
            assert hasattr(data, 'y'), "Data object missing 'y' attribute."

        # Split into training and validation sets
        if self.task_type == 'classification':
            stratify = y
        else:
            stratify = None

        train_data, val_data = train_test_split(
            data_list,
            test_size=self.validation_split,
            random_state=self.random_state,
            stratify=stratify
        )

        if self.verbose >=1:
            print(f"Number of Training Graphs: {len(train_data)}")
            print(f"Number of Validation Graphs: {len(val_data)}")

        # Create DataLoaders with specified num_workers
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        # Determine input and edge dimensions from GraphEncoder
        input_dim = len(self.graph_encoder.node_label_encoder.classes_) + self.graph_encoder.node_vec_dim
        edge_dim = len(self.graph_encoder.edge_label_encoder.classes_) + self.graph_encoder.edge_vec_dim

        if self.verbose >=1:
            print(f"Training Input Dimension: {input_dim}")
            print(f"Training Edge Dimension: {edge_dim}")

        if edge_dim == 0:
            if self.verbose >=1:
                print("Edge attributes are required for NNConv. Ensure that edges have 'label' and optionally 'vec' attributes.")
            raise ValueError("Edge attributes are required for NNConv. Ensure that edges have 'label' and optionally 'vec' attributes.")

        # Initialize the model with warmup parameters and Transformer parameters
        self.model = CrossAttentionConvolutionTransformerGraphNeuralNetworkEncoder(
            input_dim=input_dim,
            edge_dim=edge_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=self.num_layers,
            task_type=self.task_type,
            num_classes=self.num_classes if self.task_type == 'classification' else None,
            dropout_prob=self.dropout_prob,            # Pass the dropout probability
            aggr=self.aggr,                            # Pass the aggregation method
            warmup_start_lr=self.warmup_start_lr,
            warmup_end_lr=self.warmup_end_lr,
            warmup_epochs=self.warmup_epochs,
            verbose=self.verbose,
            transformer_hidden_dim=self.transformer_hidden_dim,  # Pass Transformer hidden dim
            transformer_num_layers=self.transformer_num_layers,  # Pass Transformer num layers
            transformer_num_heads=self.transformer_num_heads     # Pass Transformer num heads
        )
        self.model.to(self.device)

        # Define unique checkpoint directory by appending timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_checkpoint_dir = os.path.join(checkpoint_dir, f"run_{timestamp}")
        os.makedirs(unique_checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(unique_checkpoint_dir, "best_model.ckpt")

        if self.verbose >=1:
            print(f"Using checkpoint directory: {unique_checkpoint_dir}")

        # Initialize PyTorch Lightning callbacks
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=self.early_stopping_min_delta,
            patience=self.early_stopping_patience,
            verbose=self.verbose >=1,
            mode='min'
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=unique_checkpoint_dir,  # Use the unique checkpoint directory
            filename='best_model',
            save_top_k=1,
            mode='min',
            verbose=self.verbose >=1
        )

        callbacks = [self.loss_recorder, early_stop_callback, checkpoint_callback]

        # Integrate Stochastic Weight Averaging if enabled
        if self.swa_enable:
            try:
                # Set swa_lrs to swa_lr if provided, else default to warmup_end_lr
                if self.swa_lr is not None:
                    swa_lrs = self.swa_lr
                else:
                    swa_lrs = self.warmup_end_lr  # Default to warmup_end_lr or another appropriate value

                # Ensure swa_lrs is a positive float or list of positive floats
                if isinstance(swa_lrs, float) and swa_lrs > 0:
                    pass  # Valid
                elif isinstance(swa_lrs, list) and all(isinstance(lr, float) and lr > 0 for lr in swa_lrs):
                    pass  # Valid
                else:
                    if self.verbose >=1:
                        print("swa_lr must be a positive float or a list of positive floats.")
                    raise MisconfigurationException("The `swa_lrs` should be a positive float, or a list of positive floats")

                swa_callback = StochasticWeightAveraging(
                    swa_epoch_start=self.swa_start,
                    swa_lrs=swa_lrs,
                    annealing_epochs=10,
                    annealing_strategy='cos'
                )
                callbacks.append(swa_callback)
                if self.verbose >=1:
                    print("Stochastic Weight Averaging (SWA) enabled.")
            except MisconfigurationException as e:
                if self.verbose >=1:
                    print(f"Error initializing StochasticWeightAveraging: {e}")
                    print("Please ensure that your swa_lr is a positive float or a list of positive floats.")
                raise
        else:
            if self.verbose >=2:
                print("Stochastic Weight Averaging (SWA) not enabled.")

        # Initialize PyTorch Lightning trainer with the callbacks
        trainer = pl.Trainer(
            max_epochs=self.epochs,
            logger=False,
            enable_checkpointing=True,  # Enable checkpointing to save the best model
            accelerator='gpu' if self.device == 'cuda' else 'cpu',
            devices=1,
            deterministic=True,
            log_every_n_steps=10,
            enable_progress_bar=self.verbose >=1,  # Enable progress bar based on verbosity
            callbacks=callbacks,
            check_val_every_n_epoch=1  # Ensure validation every epoch
        )

        if self.verbose >=1:
            print("Starting model training...")

        # Train the model
        trainer.fit(self.model, train_loader, val_loader)

        # Load the best model from checkpoint
        if os.path.exists(checkpoint_path):
            if self.verbose >=1:
                print(f"Loading best model from {checkpoint_path}")
            self.model = CrossAttentionConvolutionTransformerGraphNeuralNetworkEncoder.load_from_checkpoint(
                checkpoint_path, 
                strict=False,
                transformer_hidden_dim=self.transformer_hidden_dim,  # Ensure Transformer params are loaded
                transformer_num_layers=self.transformer_num_layers,
                transformer_num_heads=self.transformer_num_heads
            )
            self.model.to(self.device)
            self.model.eval()
        else:
            if self.verbose >=1:
                print("No checkpoint found. Using the last model.")

        # If SWA was enabled, SWA weights have already been applied during training.
        if self.swa_enable:
            if self.verbose >=1:
                print("SWA has been applied during training.")

        self.is_fitted_ = True

        # Plot training and validation loss and accuracy/mse if requested
        if self.plot_training:
            self._plot_metrics()

        return self




    def _plot_metrics(self, window: int = 10, use_pandas: bool = True):
        """
        Plots the training and validation loss curves and accuracy/mse curves using matplotlib.
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
        min_len = min(train_len, val_len)
        epochs = range(1, min_len + 1)
        
        # Prepare data for plotting
        train_losses = np.array(self.loss_recorder.train_losses[:min_len])
        val_losses = np.array(self.loss_recorder.val_losses[:min_len])
        
        # Calculate running means
        if window > 0:
            if use_pandas:
                train_losses_rm = running_mean_func(train_losses, window)
                val_losses_rm = running_mean_func(val_losses, window)
            else:
                train_losses_rm = running_mean_func(train_losses, window)
                val_losses_rm = running_mean_func(val_losses, window)
            epochs_rm = range(1, min_len + 1)  # Align epochs with running mean data
        else:
            train_losses_rm = train_losses
            val_losses_rm = val_losses
            epochs_rm = epochs
        
        plt.figure(figsize=(20, 6))
        
        # Subplot 1: Loss Curves
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses,  markersize=6, color='navy', marker='o', linestyle='-', label='Training Loss', alpha=0.4, markerfacecolor='white', markeredgecolor='navy', markeredgewidth=1)
        plt.plot(epochs, val_losses, markersize=6, color='orange', marker='o', linestyle='-', label='Validation Loss', alpha=0.4, markerfacecolor='white', markeredgecolor='orange', markeredgewidth=1)
        
        if window > 0:
            plt.plot(epochs_rm, train_losses_rm, color='navy', linewidth=3, label=f'Training Loss (Running Mean, window={window})')
            plt.plot(epochs_rm, val_losses_rm, color='orange', linewidth=3, label=f'Validation Loss (Running Mean, window={window})')
        
        plt.xlabel('Epochs')
        plt.yscale('log')  # Set the y-axis to a logarithmic scale
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Subplot 2: Accuracy or MSE Curves
        plt.subplot(1, 2, 2)
        if self.task_type == 'classification':
            if self.loss_recorder.train_accuracies and self.loss_recorder.val_accuracies:
                train_accuracies = np.array(self.loss_recorder.train_accuracies[:min_len])
                val_accuracies = np.array(self.loss_recorder.val_accuracies[:min_len])
                
                if window > 0:
                    if use_pandas:
                        train_acc_rm = running_mean_func(train_accuracies, window)
                        val_acc_rm = running_mean_func(val_accuracies, window)
                    else:
                        train_acc_rm = running_mean_func(train_accuracies, window)
                        val_acc_rm = running_mean_func(val_accuracies, window)
                
                plt.plot(epochs, train_accuracies, markersize=6, color='navy', marker='o', linestyle='-', label='Training Accuracy', alpha=0.4, markerfacecolor='white', markeredgecolor='navy', markeredgewidth=1)
                plt.plot(epochs, val_accuracies, markersize=6, color='orange', marker='o', linestyle='-', label='Validation Accuracy', alpha=0.4, markerfacecolor='white', markeredgecolor='orange', markeredgewidth=1)
                
                if window > 0:
                    plt.plot(epochs_rm, train_acc_rm, color='navy', linewidth=3, label=f'Training Acc (Running Mean, window={window})')
                    plt.plot(epochs_rm, val_acc_rm, color='orange', linewidth=3, label=f'Validation Acc (Running Mean, window={window})')
                
                plt.ylabel('Accuracy')
                plt.title('Training and Validation Accuracy')
            else:
                plt.text(0.5, 0.5, 'No accuracy data available.', horizontalalignment='center', verticalalignment='center')
                plt.axis('off')
        elif self.task_type == 'regression':
            if self.loss_recorder.train_mses and self.loss_recorder.val_mses:
                train_mses = np.array(self.loss_recorder.train_mses[:min_len])
                val_mses = np.array(self.loss_recorder.val_mses[:min_len])
                
                if window > 0:
                    if use_pandas:
                        train_mse_rm = running_mean_func(train_mses, window)
                        val_mse_rm = running_mean_func(val_mses, window)
                    else:
                        train_mse_rm = running_mean_func(train_mses, window)
                        val_mse_rm = running_mean_func(val_mses, window)
                
                plt.plot(epochs, train_mses, markersize=6, color='navy', marker='o', linestyle='-', label='Training MSE', alpha=0.4, markerfacecolor='white', markeredgecolor='navy', markeredgewidth=1)
                plt.plot(epochs, val_mses, markersize=6, color='orange', marker='o', linestyle='-', label='Validation MSE', alpha=0.4, markerfacecolor='white', markeredgecolor='orange', markeredgewidth=1)
                
                if window > 0:
                    plt.plot(epochs_rm, train_mse_rm, color='navy', linewidth=3, label=f'Training MSE (Running Mean, window={window})')
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

    def transform(self, graphs: List[nx.Graph]) -> np.ndarray:
        """
        Transforms input NetworkX graphs into graph-level embeddings using the trained GNN model.

        [Docstring as provided earlier]
        """
        if self.verbose >=1:
            print("Starting transformation of graphs into embeddings...")
        check_is_fitted(self, 'is_fitted_')

        # Encode graphs
        data_list = self.graph_encoder.transform(graphs)
        loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        embeddings = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                batch = batch.to(self.device)
                embed = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                embeddings.append(embed.cpu().numpy())
                if self.verbose >1:
                    print(f"Processed batch {batch_idx + 1}/{len(loader)}: Embedding Shape {embed.shape}")  # Debug
        embeddings = np.vstack(embeddings)
        if self.verbose >=1:
            print("Transformation complete. Embeddings generated.")
        return embeddings

    def node_transform(self, graphs: List[nx.Graph]) -> List[np.ndarray]:
        """
        Transforms the input graphs into node embeddings using the trained GNN model.

        Parameters:
        - graphs: List of NetworkX graphs.

        Returns:
        - List of NumPy arrays, where each array corresponds to a graph and has shape
          [number_of_nodes_in_graph, node_embedding_dimension].
        """
        if self.verbose >=1:
            print("Starting node transformation of graphs into node embeddings...")
        check_is_fitted(self, 'is_fitted_')

        # Encode graphs
        data_list = self.graph_encoder.transform(graphs)
        loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        node_embeddings = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                batch = batch.to(self.device)
                # Get node embeddings before pooling
                node_embed = self.model.get_node_embeddings(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                node_embed = node_embed.cpu().numpy()
                
                # Split node embeddings by graph in the batch
                batch_graph_indices = batch.batch.cpu().numpy()
                unique_graphs = np.unique(batch_graph_indices)
                for graph_idx in unique_graphs:
                    node_indices = np.where(batch_graph_indices == graph_idx)[0]
                    graph_node_embed = node_embed[node_indices]
                    node_embeddings.append(graph_node_embed)
                    
                if self.verbose >1:
                    print(f"Processed batch {batch_idx + 1}/{len(loader)}: Node Embeddings Shape {node_embed.shape}")  # Debug

        if self.verbose >=1:
            print("Node transformation complete. Node embeddings generated.")
        return node_embeddings

    def fit_node_transform(self, graphs: List[nx.Graph], y: Optional[Union[np.ndarray, List]] = None) -> List[np.ndarray]:
        """
        Fits the model and transforms the graphs into node embeddings.

        Parameters:
        - graphs: List of NetworkX graphs.
        - y: Optional list or array of labels.

        Returns:
        - List of NumPy arrays containing node embeddings.
        """
        self.fit(graphs, y)
        return self.node_transform(graphs)

    def fit_transform(self, graphs: List[nx.Graph], y: Optional[Union[np.ndarray, List]] = None) -> np.ndarray:
        """
        Fits the model and transforms the graphs into embeddings.

        Parameters:
        - graphs: List of NetworkX graphs.
        - y: Optional list or array of labels.

        Returns:
        - NumPy array of graph-level embeddings.
        """
        self.fit(graphs, y)
        return self.transform(graphs)
