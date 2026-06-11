import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
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


class GCNEncoder(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        edge_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        task_type: str = 'classification',
        num_classes: Optional[int] = None,
        dropout_prob: float = 0.2,  # Dropout probability (default 0.2)
        aggr: str = 'sum',          # Aggregation method (default 'sum')
        warmup_start_lr: float = 1e-4,
        warmup_end_lr: float = 1e-3,
        warmup_epochs: int = 50,
        verbose: int =1
    ):
        """
        Graph Neural Network Encoder with dynamic hidden layers, Dropout, LeakyReLU activations,
        Batch Normalization, Residual Connections, and supervised learning capabilities.

        [Docstring as before, updated to include residual connections]
        """
        super(GCNEncoder, self).__init__()
        self.save_hyperparameters()
        self.task_type = task_type
        self.dropout_prob = dropout_prob
        self.aggr = aggr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_end_lr = warmup_end_lr
        self.warmup_epochs = warmup_epochs
        self.verbose = verbose

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.leaky_relus = nn.ModuleList()
        self.residuals = nn.ModuleList()  # ModuleList for residual connections

        # Define NNConv layers with dynamic number of hidden layers
        # Input layer
        nn_input = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim * input_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim * input_dim, hidden_dim * input_dim)
        )
        self.convs.append(NNConv(in_channels=input_dim, out_channels=hidden_dim, nn=nn_input, aggr=self.aggr))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
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
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(p=self.dropout_prob))
            self.leaky_relus.append(nn.LeakyReLU())
            
            # Initialize residual connections for hidden layers (hidden_dim -> hidden_dim)
            self.residuals.append(nn.Identity())  # Identity since dimensions match

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
        for conv, batch_norm, dropout, leaky_relu, residual in zip(
            self.convs, self.batch_norms, self.dropouts, self.leaky_relus, self.residuals
        ):
            x_in = x  # Store input for residual connection
            x = conv(x, edge_index, edge_attr)
            if x.ndim > 1 and x.shape[0] > 1:  # Ensure batch normalization is only applied when batch size > 1
                x = batch_norm(x)
            x = leaky_relu(x)
            x = dropout(x)
            x = x + residual(x_in)  # Add residual connection
        # Global pooling
        x = global_mean_pool(x, batch)
        # Embedding
        embed = self.fc_embed(x)
        return embed

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


class GCNTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        hidden_dim: int = 64,
        output_dim: int = 128,
        edge_hidden_dim: int = 64,
        num_layers: int = 2,
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
        verbose: int =1                      # Verbosity level: 0=no output, 1=training info, 2=additional graph checks
    ):
        """
        Initializes the GCNTransformer.

        [Docstring as before]
        """
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
        self.verbose = verbose

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
        Fits the GCN model on the provided graphs.

        [Docstring as before]
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

        # Initialize the model with warmup parameters
        self.model = GCNEncoder(
            input_dim=input_dim,
            edge_dim=edge_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=self.num_layers,
            task_type=self.task_type,
            num_classes=self.num_classes if self.task_type == 'classification' else None,
            dropout_prob=self.dropout_prob,  # Pass the dropout probability
            aggr=self.aggr,                  # Pass the aggregation method
            warmup_start_lr=self.warmup_start_lr,
            warmup_end_lr=self.warmup_end_lr,
            warmup_epochs=self.warmup_epochs,
            verbose=self.verbose
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

        # Combine all callbacks
        callbacks = [self.loss_recorder, early_stop_callback, checkpoint_callback]

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
            self.model = GCNEncoder.load_from_checkpoint(checkpoint_path, strict=False)
            self.model.to(self.device)
            self.model.eval()
        else:
            if self.verbose >=1:
                print("No checkpoint found. Using the last model.")

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
        plt.plot(epochs, train_losses,  markersize=6, color='navy', marker='o', linestyle='-', label='Training Accuracy', alpha=0.4, markerfacecolor='white', markeredgecolor='navy', markeredgewidth=1)
        plt.plot(epochs, val_losses, markersize=6, color='orange', marker='o', linestyle='-', label='Validation Accuracy', alpha=0.4, markerfacecolor='white', markeredgecolor='orange', markeredgewidth=1)
        
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
                
                plt.plot(epochs, train_mses, markersize=6, color='navy', marker='o', linestyle='-', label='Training Accuracy', alpha=0.4, markerfacecolor='white', markeredgecolor='navy', markeredgewidth=1)
                plt.plot(epochs, val_mses, markersize=6, color='orange', marker='o', linestyle='-', label='Validation Accuracy', alpha=0.4, markerfacecolor='white', markeredgecolor='orange', markeredgewidth=1)
                
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
        Transforms the input graphs into vector embeddings using the trained GCN.

        [Docstring as before]
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

    def fit_transform(self, graphs: List[nx.Graph], y: Optional[Union[np.ndarray, List]] = None) -> np.ndarray:
        """
        Fits the model and transforms the graphs into embeddings.

        [Docstring as before]
        """
        self.fit(graphs, y)
        return self.transform(graphs)
