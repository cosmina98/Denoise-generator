import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import networkx as nx
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Dict, Sequence, List, Tuple, Optional, Any, Union
from pathlib import Path


def _build_projector_mlp(
    input_dim: int,
    projection_hidden_dims: Sequence[int],
    projection_dim: int,
    use_batchnorm: bool,
    dropout: float,
) -> nn.Sequential:
    """Builds the MLP for projecting node attributes."""
    projector_layers: List[nn.Module] = []
    dims = [input_dim] + list(projection_hidden_dims) + [projection_dim]
    for i in range(len(dims) - 1):
        projector_layers.append(nn.Linear(dims[i], dims[i+1]))
        if use_batchnorm:
            # LayerNorm normalises over the LAST dim (works with (B, N, C))
            projector_layers.append(nn.LayerNorm(dims[i+1]))
        if i < len(dims) - 2: # No activation/dropout after the last linear layer
            projector_layers.append(nn.ReLU())
            if dropout > 0:
                projector_layers.append(nn.Dropout(dropout))
    return nn.Sequential(*projector_layers)


def _build_multihead_transforms(
    num_heads: int,
    transform_input_dim: int,
    transform_output_dim: int,
    low_rank_dim: Optional[int],
) -> nn.ModuleList:
    """Builds the multi-head transformation parameters."""
    transforms = nn.ModuleList()
    for _ in range(num_heads):
        if low_rank_dim is not None:
            U = nn.Parameter(torch.empty(transform_input_dim, low_rank_dim))
            V = nn.Parameter(torch.empty(low_rank_dim, transform_output_dim))
            nn.init.xavier_uniform_(U)
            nn.init.xavier_uniform_(V)
            transforms.append(nn.ParameterList([U, V]))
        else:
            W = nn.Parameter(torch.empty(transform_input_dim, transform_output_dim))
            nn.init.xavier_uniform_(W)
            transforms.append(nn.ParameterList([W]))
    return transforms


def _build_classifier_head(
    num_heads: int,
    projection_dim: int,
    feature_dim: int,
    classifier_hidden_dims: Sequence[int],
    num_classes: int,
    use_batchnorm: bool,
    dropout: float
) -> nn.Sequential: # type: ignore
    """Builds the classification head MLP."""
    # Input dimension for the classifier head
    # Concatenation of:
    # - sum_A: (num_heads * projection_dim)
    # - sum_F: (feature_dim)
    # - sum_P: (num_heads * feature_dim)
    in_dim = (num_heads * projection_dim) + feature_dim + (num_heads * feature_dim)
    
    classifier_layers: List[nn.Module] = []
    dims = [in_dim] + list(classifier_hidden_dims) + [num_classes]
    for i in range(len(dims) - 1):
        classifier_layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2: # No activation/dropout/norm before the final output layer
            if use_batchnorm:
                classifier_layers.append(nn.BatchNorm1d(dims[i+1]))
            classifier_layers.append(nn.ReLU())
            if dropout > 0:
                classifier_layers.append(nn.Dropout(dropout))
    return nn.Sequential(*classifier_layers)


class MultiHeadLinearTransformProjector(nn.Module):
    """
    Projects node attributes using multiple independent linear transformation heads 
    (optionally low-rank) and concatenates their outputs.
    Applies orthogonality penalty to these transformations.
    """
    def __init__(self, node_input_dim: int, num_projection_heads: int,
                 transform_output_dim_per_head: int, 
                 low_rank_dim_per_head: Optional[int]):
        super().__init__()
        self.heads = nn.ModuleList()
        self.num_projection_heads = num_projection_heads
        self.use_low_rank = low_rank_dim_per_head is not None

        for _ in range(num_projection_heads):
            if self.use_low_rank:
                # Each head: input_dim -> low_rank_dim -> transform_output_dim_per_head
                U = nn.Parameter(torch.empty(node_input_dim, low_rank_dim_per_head)) # type: ignore
                V = nn.Parameter(torch.empty(low_rank_dim_per_head, transform_output_dim_per_head)) # type: ignore
                nn.init.xavier_uniform_(U)
                nn.init.xavier_uniform_(V)
                self.heads.append(nn.ParameterList([U, V]))
            else:
                # Each head: input_dim -> transform_output_dim_per_head (full rank)
                W = nn.Parameter(torch.empty(node_input_dim, transform_output_dim_per_head))
                nn.init.xavier_uniform_(W)
                self.heads.append(nn.ParameterList([W]))

    def forward(self, x_nodes: torch.Tensor) -> torch.Tensor:
        head_outputs = []
        for params in self.heads:
            W_head = torch.matmul(params[0], params[1]) if self.use_low_rank else params[0]
            transformed_x = torch.matmul(x_nodes, W_head)
            head_outputs.append(transformed_x)
        return torch.cat(head_outputs, dim=-1)

    def get_ortho_penalty(self) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=self.heads[0][0].device if len(self.heads) > 0 and len(self.heads[0]) > 0 else torch.device('cpu'))
        # Assuming transform_output_dim_per_head is node_input_dim for square W_head
        # and for V in low-rank case.
        for params in self.heads:
            if self.use_low_rank:
                U, V = params # U: (node_input_dim, r), V: (r, node_input_dim)
                # Penalty for U: columns of U should be orthonormal (U.T @ U = I_r)
                if U.size(0) >= U.size(1): # Only if U is tall or square
                    I_r_U = torch.eye(U.size(1), device=U.device)
                    penalty = penalty + torch.norm(U.T @ U - I_r_U, p='fro')
                
                # Penalty for V: rows of V should be orthonormal (V @ V.T = I_r)
                if V.size(1) >= V.size(0): # Only if V is wide or square
                    I_r_V = torch.eye(V.size(0), device=V.device)
                    penalty = penalty + torch.norm(V @ V.T - I_r_V, p='fro')
            else: # Full rank W_head
                W_head, = params # W_head: (node_input_dim, node_input_dim)
                # This assumes W_head is square, which it is in this setup.
                I_d = torch.eye(W_head.size(0), device=W_head.device) 
                penalty = penalty + torch.norm(W_head.T @ W_head - I_d, p='fro')
        return penalty

class MultiHeadBilinearGraphClassifier(pl.LightningModule):
    """
    A PyTorch Lightning module for graph classification.

    This class implements a neural network for graph classification that combines node attributes
    and graph structure. The model first projects node attributes using a configurable MLP,
    then applies a learnable transformation (optionally low-rank) to the projected attributes.
    It performs a message passing step by combining projected node attributes and node features,
    aggregates node-level information to produce a graph-level representation, and finally
    classifies the graph using a configurable MLP head. The architecture supports batch/layer
    normalization, dropout, and flexible optimizer/scheduler choices.

    The learned transformation can be regularized to be orthonormal via an orthogonality penalty.
    This constraint is beneficial because:
      - It prevents redundant or highly correlated features in the projected space.
      - It stabilizes training by preserving vector norms and avoiding exploding/vanishing gradients.
      - It acts as a regularizer, reducing overfitting and improving generalization.
      - It preserves information by ensuring the transformation is invertible and does not amplify or shrink input vectors.
      - It makes the transformation interpretable as a rotation/reflection, aiding analysis and visualization.
      - It is a common technique in deep learning for stability and expressiveness.

    Parameters
    ----------
    input_dim : int
        Dimension of input node attributes
    feature_dim : int
        Dimension of graph structural features
    projection_dim : int
        Dimension of the projected node attributes
    projection_hidden_dims : list of int
        Hidden layer dimensions for the projection MLP
    num_classes : int
        Number of target classes
    classifier_hidden_dims : list of int, default=[128]
        Hidden layer dimensions for the classification head
    lr : float, default=1e-3
        Learning rate for optimization
    dropout : float, default=0.0
        Dropout rate for regularization
    use_batchnorm : bool, default=False
        Whether to use batch normalization
    optimizer_class : type, default=torch.optim.Adam
        PyTorch optimizer to use
    optimizer_kwargs : dict or None
        Additional arguments for optimizer
    scheduler_class : type, default=torch.optim.lr_scheduler.ReduceLROnPlateau
        Additional arguments for scheduler
    low_rank_dim : int or None, default=None
        If set, use a low-rank factorization for the learned transform in the classifier.
    ortho_lambda : float, default=0.0
        Strength of orthogonality penalty for the learned transform.
    num_heads : int, default=1
        Number of heads for multi-head transformations.
    """
    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        projection_dim: int,
        projection_hidden_dims: Sequence[int],
        num_classes: int,
        classifier_hidden_dims: Sequence[int] = (128,),
        lr: float = 1e-3,
        dropout: float = 0.0, # type: ignore
        use_batchnorm: bool = False,
        optimizer_class: type = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler_class: type = torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        low_rank_dim: Optional[int] = None,
        ortho_lambda: float = 0.0, # This ortho_lambda applies to bilinear transforms
        num_heads: int = 1,
        rank_factor: float = 1.0, # Replaces low_rank_dim and initial_head_low_rank_dim
    ):
        super().__init__()
        self.save_hyperparameters()
        # store optimizer/scheduler
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs or {"mode": "min", "patience": 5, "verbose": True}

        # Calculate actual low_rank_dim for initial attribute projector heads
        max_rank_initial_heads = input_dim
        calculated_initial_head_low_rank_dim = None
        if rank_factor < 1.0:
            calculated_initial_head_low_rank_dim = max(1, int(rank_factor * max_rank_initial_heads))

        # 1. Multi-head linear transform projector for initial node attributes.
        #    Each head is a linear (possibly low-rank) transform: input_dim -> input_dim.
        #    The concatenated output will be `num_heads * input_dim`.
        self.attribute_projector_heads = MultiHeadLinearTransformProjector(
            node_input_dim=input_dim,
            num_projection_heads=num_heads, # Use hparams.num_heads for the number of projector heads
            transform_output_dim_per_head=input_dim, # Each head outputs input_dim
            low_rank_dim_per_head=calculated_initial_head_low_rank_dim
        )
        
        # Dimension after initial multi-head projection
        dim_after_multihead_concat = num_heads * input_dim

        # 2. MLP to map from concatenated (num_heads * input_dim) to `projection_dim`
        #    This `projection_dim` is the one used for bilinear interactions. 
        #    This MLP uses `projection_hidden_dims`.
        self.final_attribute_projector_mlp = _build_projector_mlp(
            input_dim=dim_after_multihead_concat,
            projection_hidden_dims=projection_hidden_dims, # Use hparams.projection_hidden_dims here
            projection_dim=projection_dim,                 # Target final projection_dim
            use_batchnorm=use_batchnorm,                   # hparams.use_batchnorm controls LayerNorm
            dropout=dropout                                # hparams.dropout
        )

        # Calculate actual low_rank_dim for bilinear transformation bank
        max_rank_bilinear_transforms = projection_dim
        calculated_bilinear_low_rank_dim = None
        if rank_factor < 1.0:
            calculated_bilinear_low_rank_dim = max(1, int(rank_factor * max_rank_bilinear_transforms))

        self.num_heads = num_heads
        self.use_low_rank = calculated_bilinear_low_rank_dim is not None # For bilinear transforms
        # 3. Bilinear transformation bank (operates on the final projection_dim)
        self.transforms = _build_multihead_transforms(
            num_heads=num_heads,
            transform_input_dim=projection_dim, # Operates on the output of final_attribute_projector
            transform_output_dim=projection_dim,
            low_rank_dim=calculated_bilinear_low_rank_dim
        )

        # 4. Final classifier head
        self.classifier = _build_classifier_head(
            num_heads=num_heads,
            projection_dim=projection_dim, # Classifier input based on `projection_dim` from bilinear heads
            feature_dim=feature_dim,
            classifier_hidden_dims=classifier_hidden_dims,
            num_classes=num_classes,
            use_batchnorm=use_batchnorm,
            dropout=dropout
        )

        self.lr = lr
        self.ortho_lambda = ortho_lambda

    def forward(self, A_batch: torch.Tensor, F_batch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Batched forward pass:
          - A_batch: (B, N_max, input_dim)
          - F_batch: (B, N_max, feature_dim)
          - mask:    (B, N_max) boolean indicating valid nodes
        """
        # 1) Multi-head linear transform node attributes (each head: input_dim -> input_dim)
        # A_multi_head_proj_concat: (B, N_max, num_heads * input_dim)
        A_multi_head_proj_concat: torch.Tensor = self.attribute_projector_heads(A_batch)

        # 1.5) MLP projection to final projection_dim (the p' for bilinear interaction)
        # A_proj: (B, N_max, projection_dim)
        A_proj: torch.Tensor = self.final_attribute_projector_mlp(A_multi_head_proj_concat)

        # 2) Zero out padded nodes based on the original mask
        mask_exp = mask.unsqueeze(-1)           # (B, N_max, 1)
        A_proj = A_proj * mask_exp              # (B, N_max, p')
        F_batch = F_batch * mask_exp.float()    # (B, N_max, f)

        # 3) Compute M via batched bilinear: M[b] = A_proj[b].T @ F_batch[b]
        #    → M: (B, p', f)
        M: torch.Tensor = torch.bmm(A_proj.transpose(1, 2), F_batch)

        # 4-5) Apply each head, collect A_proj_h and P_h
        proj_heads: List[torch.Tensor] = []
        P_heads: List[torch.Tensor] = []
        for params in self.transforms:
            if self.use_low_rank:
                U, V = params
                W = torch.matmul(U, V)            # (p', p')
            else:
                W, = params                       # (p', p')
            A_trans = torch.matmul(A_proj, W)     # (B, N_max, p')
            P_h     = torch.bmm(A_trans, M)       # (B, N_max, f)
            proj_heads.append(A_trans)
            P_heads.append(P_h)

        # Concatenate along feature dim
        A_trans_cat: torch.Tensor = torch.cat(proj_heads, dim=2)  # (B, N_max, num_heads·p')
        P_cat: torch.Tensor = torch.cat(P_heads,  dim=2)    # (B, N_max, num_heads·f)

        # 6) Pool to graph-level: sum over node dimension
        sum_A: torch.Tensor = A_trans_cat.sum(dim=1)  # (B, num_heads·p')
        sum_F: torch.Tensor = F_batch.sum(dim=1)     # (B, f)
        sum_P: torch.Tensor = P_cat.sum(dim=1)       # (B, num_heads·f)

        # 7) Concatenate and classify in one batch: (B, num_heads·p'+2f) → (B, num_classes)
        graph_repr: torch.Tensor = torch.cat([sum_A, sum_F, sum_P], dim=1)
        return self.classifier(graph_repr)

    def embed(self, A_batch: torch.Tensor, F_batch: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes node-level embeddings/representations.

        Parameters
        ----------
        A_batch : torch.Tensor
            Batched node attributes (B, N_max, input_dim).
        F_batch : torch.Tensor
            Batched graph features (B, N_max, feature_dim).
        mask : torch.Tensor
            Boolean mask indicating valid nodes (B, N_max).

        Returns
        -------
        A_trans_cat : torch.Tensor
            Concatenated transformed node attributes (B, N_max, num_heads * projection_dim).
        F_batch_masked : torch.Tensor
            Masked original graph features (B, N_max, feature_dim).
        P_cat : torch.Tensor
            Concatenated message-passed features (B, N_max, num_heads * feature_dim).
        """
        # 1) Multi-head linear transform node attributes (input_dim -> input_dim per head)
        A_multi_head_proj_concat: torch.Tensor = self.attribute_projector_heads(A_batch)
        
        # 1.5) MLP projection to final projection_dim
        A_proj: torch.Tensor = self.final_attribute_projector_mlp(A_multi_head_proj_concat)

        mask_exp: torch.Tensor = mask.unsqueeze(-1)
        A_proj_masked: torch.Tensor = A_proj * mask_exp # Apply mask after reduction
        F_batch_masked: torch.Tensor = F_batch * mask_exp.float()
        M: torch.Tensor = torch.bmm(A_proj_masked.transpose(1, 2), F_batch_masked)

        proj_heads: List[torch.Tensor] = []
        P_heads: List[torch.Tensor] = []
        for params in self.transforms:
            W = torch.matmul(params[0], params[1]) if self.use_low_rank else params[0]
            A_trans = torch.matmul(A_proj_masked, W)
            P_h = torch.bmm(A_trans, M)
            proj_heads.append(A_trans)
            P_heads.append(P_h)

        return torch.cat(proj_heads, dim=2), F_batch_masked, torch.cat(P_heads, dim=2)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        A, features, targets, mask = batch
        logits = self.forward(A, features, mask)
        # explicitly use the functional API so we don't clash with our variable names
        loss: torch.Tensor = nn.functional.cross_entropy(logits, targets)
        
        # Orthogonality penalty for initial attribute_projector_heads
        ortho_penalty_initial_heads = self.attribute_projector_heads.get_ortho_penalty()
        
        # Orthogonality penalty for bilinear transformation heads
        ortho_penalty_bilinear_heads = torch.tensor(0.0, device=loss.device)
        for params in self.transforms:
            if self.use_low_rank: # This refers to low_rank_dim for the bilinear transforms
                U, V = params
                # U: (projection_dim, r_bilinear), V: (r_bilinear, projection_dim)
                # Penalty for U: columns of U should be orthonormal (U.T @ U = I_r)
                if U.size(0) >= U.size(1): # if projection_dim >= r_bilinear
                    I_r_U_bilinear = torch.eye(U.size(1), device=U.device)
                    ortho_penalty_bilinear_heads = ortho_penalty_bilinear_heads + torch.norm(U.T @ U - I_r_U_bilinear, p='fro')

                # Penalty for V: rows of V should be orthonormal (V @ V.T = I_r)
                if V.size(1) >= V.size(0): # if projection_dim >= r_bilinear
                    I_r_V_bilinear = torch.eye(V.size(0), device=V.device)
                    ortho_penalty_bilinear_heads = ortho_penalty_bilinear_heads + torch.norm(V @ V.T - I_r_V_bilinear, p='fro')
            else: # Full rank W for bilinear transform
                W, = params
                I_p_prime = torch.eye(W.size(1), device=W.device) # W.size(1) is projection_dim
                ortho_penalty_bilinear_heads = ortho_penalty_bilinear_heads + torch.norm(W.T @ W - I_p_prime, p='fro')
        
        ortho_penalty = ortho_penalty_initial_heads + ortho_penalty_bilinear_heads
        loss = loss + self.ortho_lambda * ortho_penalty
        self.log("train_loss", loss)
        self.log("ortho_penalty", ortho_penalty)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        A, features, targets, mask = batch
        logits = self.forward(A, features, mask)
        loss: torch.Tensor = nn.functional.cross_entropy(logits, targets)
        acc: torch.Tensor = (logits.argmax(dim=1) == targets).float().mean()
        self.log("val_loss", loss, prog_bar=False)
        self.log("val_acc", acc, prog_bar=False)
        return loss

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Tuple[List[torch.optim.Optimizer], List[Dict[str, Any]]]]:
        # be sure to pass through the lr you saved in __init__
        opt = self.optimizer_class(self.parameters(), lr=self.lr, **self.optimizer_kwargs)
        if self.scheduler_class:
            sch = self.scheduler_class(opt, **self.scheduler_kwargs)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}} # type: ignore
        return opt

class MultiHeadBilinearMLPGraphClassifier(pl.LightningModule):
    """
    A PyTorch Lightning module for graph classification.

    This class implements a neural network for graph classification that combines node attributes
    and graph structure. The model first projects node attributes using a configurable MLP,
    then applies a learnable transformation (optionally low-rank) to the projected attributes.
    It performs a message passing step by combining projected node attributes and node features,
    aggregates node-level information to produce a graph-level representation, and finally
    classifies the graph using a configurable MLP head. The architecture supports batch/layer
    normalization, dropout, and flexible optimizer/scheduler choices.

    The learned transformation can be regularized to be orthonormal via an orthogonality penalty.
    This constraint is beneficial because:
      - It prevents redundant or highly correlated features in the projected space.
      - It stabilizes training by preserving vector norms and avoiding exploding/vanishing gradients.
      - It acts as a regularizer, reducing overfitting and improving generalization.
      - It preserves information by ensuring the transformation is invertible and does not amplify or shrink input vectors.
      - It makes the transformation interpretable as a rotation/reflection, aiding analysis and visualization.
      - It is a common technique in deep learning for stability and expressiveness.

    Parameters
    ----------
    input_dim : int
        Dimension of input node attributes
    feature_dim : int
        Dimension of graph structural features
    projection_dim : int
        Dimension of the projected node attributes
    projection_hidden_dims : list of int
        Hidden layer dimensions for the projection MLP
    num_classes : int
        Number of target classes
    classifier_hidden_dims : list of int, default=[128]
        Hidden layer dimensions for the classification head
    lr : float, default=1e-3
        Learning rate for optimization
    dropout : float, default=0.0
        Dropout rate for regularization
    use_batchnorm : bool, default=False
        Whether to use batch normalization
    optimizer_class : type, default=torch.optim.Adam
        PyTorch optimizer to use
    optimizer_kwargs : dict or None
        Additional arguments for optimizer
    scheduler_class : type, default=torch.optim.lr_scheduler.ReduceLROnPlateau
        Additional arguments for scheduler
    low_rank_dim : int or None, default=None
        If set, use a low-rank factorization for the learned transform in the classifier.
    ortho_lambda : float, default=0.0
        Strength of orthogonality penalty for the learned transform.
    num_heads : int, default=1
        Number of heads for multi-head transformations.
    """
    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        projection_dim: int,
        projection_hidden_dims: Sequence[int],
        num_classes: int,
        classifier_hidden_dims: Sequence[int] = (128,),
        lr: float = 1e-3,
        dropout: float = 0.0, # type: ignore
        use_batchnorm: bool = False,
        optimizer_class: type = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler_class: type = torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        low_rank_dim: Optional[int] = None,
        ortho_lambda: float = 0.0,
        num_heads: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        # store optimizer/scheduler
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs or {"mode": "min", "patience": 5, "verbose": True}

        self.projector = _build_projector_mlp(
            input_dim=input_dim,
            projection_hidden_dims=projection_hidden_dims,
            projection_dim=projection_dim,
            use_batchnorm=use_batchnorm,
            dropout=dropout
        )

        self.num_heads = num_heads
        self.use_low_rank = low_rank_dim is not None
        self.transforms = _build_multihead_transforms(
            num_heads=num_heads,
            transform_input_dim=projection_dim, # Input to W is A_proj (dim p')
            transform_output_dim=projection_dim, # Output of W must be p' for P_h = A_trans @ M
            low_rank_dim=low_rank_dim 
        )

        self.classifier = _build_classifier_head(
            num_heads=num_heads,
            projection_dim=projection_dim,
            feature_dim=feature_dim,
            classifier_hidden_dims=classifier_hidden_dims,
            num_classes=num_classes,
            use_batchnorm=use_batchnorm,
            dropout=dropout
        )

        self.lr = lr
        self.ortho_lambda = ortho_lambda

    def forward(self, A_batch: torch.Tensor, F_batch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Batched forward pass:
          - A_batch: (B, N_max, input_dim)
          - F_batch: (B, N_max, feature_dim)
          - mask:    (B, N_max) boolean indicating valid nodes
        """
        # 1) Project node attributes across the batch: (B, N_max, p')
        A_proj: torch.Tensor = self.projector(A_batch)

        # 2) Zero out padded nodes
        mask_exp = mask.unsqueeze(-1)           # (B, N_max, 1)
        A_proj = A_proj * mask_exp              # (B, N_max, p')
        F_batch = F_batch * mask_exp.float()    # (B, N_max, f)

        # 3) Compute M via batched bilinear: M[b] = A_proj[b].T @ F_batch[b]
        #    → M: (B, p', f)
        M: torch.Tensor = torch.bmm(A_proj.transpose(1, 2), F_batch)

        # 4-5) Apply each head, collect A_proj_h and P_h
        proj_heads: List[torch.Tensor] = []
        P_heads: List[torch.Tensor] = []
        for params in self.transforms:
            if self.use_low_rank:
                U, V = params
                W = torch.matmul(U, V)            # (p', p')
            else:
                W, = params                       # (p', p')
            A_trans = torch.matmul(A_proj, W)     # (B, N_max, p')
            P_h     = torch.bmm(A_trans, M)       # (B, N_max, f)
            proj_heads.append(A_trans)
            P_heads.append(P_h)

        # Concatenate along feature dim
        A_trans_cat: torch.Tensor = torch.cat(proj_heads, dim=2)  # (B, N_max, num_heads·p')
        P_cat: torch.Tensor = torch.cat(P_heads,  dim=2)    # (B, N_max, num_heads·f)

        # 6) Pool to graph-level: sum over node dimension
        sum_A: torch.Tensor = A_trans_cat.sum(dim=1)  # (B, num_heads·p')
        sum_F: torch.Tensor = F_batch.sum(dim=1)     # (B, f)
        sum_P: torch.Tensor = P_cat.sum(dim=1)       # (B, num_heads·f)

        # 7) Concatenate and classify in one batch: (B, num_heads·p'+2f) → (B, num_classes)
        graph_repr: torch.Tensor = torch.cat([sum_A, sum_F, sum_P], dim=1)
        return self.classifier(graph_repr)

    def embed(self, A_batch: torch.Tensor, F_batch: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes node-level embeddings/representations.

        Parameters
        ----------
        A_batch : torch.Tensor
            Batched node attributes (B, N_max, input_dim).
        F_batch : torch.Tensor
            Batched graph features (B, N_max, feature_dim).
        mask : torch.Tensor
            Boolean mask indicating valid nodes (B, N_max).

        Returns
        -------
        A_trans_cat : torch.Tensor
            Concatenated transformed node attributes (B, N_max, num_heads * projection_dim).
        F_batch_masked : torch.Tensor
            Masked original graph features (B, N_max, feature_dim).
        P_cat : torch.Tensor
            Concatenated message-passed features (B, N_max, num_heads * feature_dim).
        """
        A_proj: torch.Tensor = self.projector(A_batch)
        mask_exp: torch.Tensor = mask.unsqueeze(-1)
        A_proj_masked: torch.Tensor = A_proj * mask_exp
        F_batch_masked: torch.Tensor = F_batch * mask_exp.float()
        M: torch.Tensor = torch.bmm(A_proj_masked.transpose(1, 2), F_batch_masked)

        proj_heads: List[torch.Tensor] = []
        P_heads: List[torch.Tensor] = []
        for params in self.transforms:
            W = torch.matmul(params[0], params[1]) if self.use_low_rank else params[0]
            A_trans = torch.matmul(A_proj_masked, W)
            P_h = torch.bmm(A_trans, M)
            proj_heads.append(A_trans)
            P_heads.append(P_h)

        return torch.cat(proj_heads, dim=2), F_batch_masked, torch.cat(P_heads, dim=2)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        A, features, targets, mask = batch
        logits = self.forward(A, features, mask)
        # explicitly use the functional API so we don't clash with our variable names
        loss: torch.Tensor = nn.functional.cross_entropy(logits, targets)
        # accumulate ortho penalty across heads
        ortho_penalty = 0.0
        for params in self.transforms:
            if self.use_low_rank:
                U, V = params
                W = torch.matmul(U, V)
            else:
                W, = params
            I = torch.eye(W.size(1), device=W.device)
            ortho_penalty = ortho_penalty + torch.norm(W.T @ W - I, p='fro')
        loss = loss + self.ortho_lambda * ortho_penalty
        self.log("train_loss", loss)
        self.log("ortho_penalty", ortho_penalty)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        A, features, targets, mask = batch
        logits = self.forward(A, features, mask)
        loss: torch.Tensor = nn.functional.cross_entropy(logits, targets)
        acc: torch.Tensor = (logits.argmax(dim=1) == targets).float().mean()
        self.log("val_loss", loss, prog_bar=False)
        self.log("val_acc", acc, prog_bar=False)
        return loss

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Tuple[List[torch.optim.Optimizer], List[Dict[str, Any]]]]:
        # be sure to pass through the lr you saved in __init__
        opt = self.optimizer_class(self.parameters(), lr=self.lr, **self.optimizer_kwargs)
        if self.scheduler_class:
            sch = self.scheduler_class(opt, **self.scheduler_kwargs)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}} # type: ignore
        return opt


class GraphDataset(Dataset):
    """
    A PyTorch Dataset for handling graph data.

    This class prepares graph data for neural network training and inference. It precomputes and stores
    node attributes (optionally concatenated with node-level features), graph-level features, and
    targets for each graph. It supports efficient batching and padding for variable-size graphs,
    enabling mini-batch training of graph neural networks.

    Parameters
    ----------
    graphs : list of networkx.Graph
        List of input graphs
    targets : array-like or None
        Target labels for the graphs
    vectorizer : object
        Graph vectorizer object with transform method (for graph-level features)
    node_vectorizer : object
        Node vectorizer object with transform method (for node-level features)
    attribute_key : str, default="attr"
        Key for accessing node attributes in the graphs
    """
    def __init__(self, graphs, targets, vectorizer, node_vectorizer, attribute_key="attr"):
        self.graphs: List[nx.Graph] = graphs
        self.targets: Optional[Union[np.ndarray, List[Any]]] = targets
        self.attribute_key: str = attribute_key
        
        # Pre-compute all node attributes
        node_attrs_list: List[torch.Tensor] = [
            torch.stack([ # type: ignore
                torch.tensor(g.nodes[node][attribute_key], dtype=torch.float32)
                if attribute_key in g.nodes[node]
                else torch.tensor([1.0], dtype=torch.float32)
                for node in g.nodes
            ])
            for g in graphs
        ]
        node_features: Union[np.ndarray, List[np.ndarray]] = node_vectorizer.transform(graphs)  # (n_graphs, n_features) or list of features
        if isinstance(node_features, list):
            node_features_tensors: List[torch.Tensor] = [torch.as_tensor(f, dtype=torch.float32) for f in node_features]
        else:
            node_features_tensors: List[torch.Tensor] = [torch.as_tensor(node_features[i], dtype=torch.float32) for i in range(len(graphs))]

        # Concatenate node_attrs and node_features along the last dimension
        self.node_attrs: List[torch.Tensor] = [
            torch.cat([attr, feat], dim=-1)
            for attr, feat in zip(node_attrs_list, node_features_tensors)
        ]

        # Pre-compute all graph features at once
        features: Union[np.ndarray, List[np.ndarray]] = vectorizer.transform(graphs)  # (n_graphs, n_features) or list of features
        if isinstance(features, list):
            self.features: List[torch.Tensor] = [torch.as_tensor(f, dtype=torch.float32) for f in features]
        else:
            self.features: List[torch.Tensor] = [torch.as_tensor(features[i], dtype=torch.float32) for i in range(len(graphs))]

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx):
        # Just return pre-computed tensors
        target = self.targets[idx] if self.targets is not None else torch.tensor(-1)
        return self.node_attrs[idx], self.features[idx], target


class MultiHeadBilinearGraphModel(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for the MultiHeadBilinearGraphClassifier.

    This class provides a scikit-learn style API (fit, predict, predict_proba, transform)
    for the `MultiHeadBilinearGraphClassifier` PyTorch Lightning module. It seamlessly
    integrates graph vectorization, neural network training, and prediction into
    standard machine learning pipelines.

    The underlying neural network architecture features:
    - Multi-head transformations on projected node attributes.
    - Bilinear interactions for message passing between node attributes and graph features.
    - Optional low-rank factorization for transformations and orthogonality regularization.

    The wrapper handles:
    - Data vectorization using provided graph and node vectorizers.
    - Splitting data for training and validation.
    - Managing the PyTorch Lightning training loop, including optimizer and scheduler configuration.
    - Performing predictions and probability estimates.
    - Extracting node-level representations via the `transform` method.

    Parameters
    ----------
    vectorizer : object
        Graph vectorizer with fit and transform methods (for graph-level features).
    node_vectorizer : object
        Node vectorizer with fit and transform methods (for node-level features).
    projection_dim : int, default=128
        Dimension for attribute projection.
    projection_hidden_dims : tuple of int, default=(128,)
        Hidden dimensions for the MLP that projects node attributes.
    classifier_hidden_dims : list of int, default=[128]
        Hidden dimensions for the final classification MLP head.
    lr : float, default=1e-3
        Learning rate for the optimizer.
    dropout : float, default=0.2
        Dropout rate applied in the projection and classification MLPs.
    use_batchnorm : bool, default=True
        Whether to use LayerNorm in the projector and BatchNorm1d in the classifier head.
    batch_size : int, default=16
        Mini-batch size for training.
    max_epochs : int, default=100
        Maximum number of training epochs.
    validation_split : float, default=0.2
        Fraction of data to use for validation during training.
    attribute_key : str, default="attr"
        Key used to access node attributes in the input NetworkX graphs.
    optimizer_class : type, default=torch.optim.Adam
        PyTorch optimizer class to use for training.
    optimizer_kwargs : dict or None
        Additional keyword arguments for the optimizer.
    scheduler_class : type, default=torch.optim.lr_scheduler.ReduceLROnPlateau
        PyTorch learning rate scheduler class.
    scheduler_kwargs : dict or None, default=None
        Additional keyword arguments for the scheduler. If None, defaults to
        `{"mode": "min", "patience": 5, "verbose": True}` for ReduceLROnPlateau.
    low_rank_dim : int or None, default=None
        If set, use a low-rank factorization for the learned transform in the classifier.
    ortho_lambda : float, default=1e-3
        Strength of orthogonality penalty for the learned transform.
    num_heads : int, default=4
        Number of heads for multi-head transformations.
    """
    def __init__( # type: ignore
        self,
        vectorizer: Any,
        node_vectorizer: Any,
        projection_dim=128,  
        projection_hidden_dims=(128,),  
        classifier_hidden_dims: List[int] = [128],
        lr: float =1e-3,
        dropout: float =0.2,
        use_batchnorm: bool =True,
        batch_size: int =16,
        max_epochs: int =100,
        validation_split: float =0.2,
        attribute_key: str ="attr",
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] =None,
        scheduler_class=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_kwargs: Optional[Dict[str, Any]] =None, # Removed low_rank_dim
        rank_factor: float = 1.0, # Added rank_factor
        ortho_lambda: float =1e-3,
        num_heads: int =4,
    ):
        self.vectorizer = vectorizer
        self.node_vectorizer = node_vectorizer
        self.projection_dim = projection_dim
        self.projection_hidden_dims = projection_hidden_dims
        self.classifier_hidden_dims = classifier_hidden_dims
        self.lr = lr
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.validation_split = validation_split
        self.attribute_key = attribute_key
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs or {"mode": "min", "patience": 10, "verbose": True}
        
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        self.rank_factor = rank_factor # Store rank_factor
        self.ortho_lambda = ortho_lambda
        self.num_heads = num_heads
        
        self.classifier: Optional[MultiHeadBilinearGraphClassifier] = None
        self.input_dim: Optional[int] = None
        self.feature_dim: Optional[int] = None
        self.num_classes: Optional[int] = None
        self.train_logs_path: Optional[Path] = None

    def info(self) -> Optional[pl.LightningModule]:
        """
        Returns the underlying PyTorch Lightning model for inspection or visualization.

        Returns
        -------
        Optional[pl.LightningModule]
            The `MultiHeadBilinearGraphClassifier` model instance if `fit` has been called,
            otherwise None.
        """
        if self.classifier is None:
            print("Model has not been fitted yet. Call fit() to initialize the model.")
        return self.classifier

    def fit(self, graphs: List[nx.Graph], targets: Union[np.ndarray, List[Any]]) -> "MultiHeadBilinearGraphModel":
        self.vectorizer.fit(graphs, targets)

        # Split data into train and validation sets
        if self.validation_split > 0:
            graphs_train, graphs_val, targets_train, targets_val = train_test_split(
                graphs, targets, test_size=self.validation_split, random_state=42, stratify=targets
            )
        else:
            graphs_train, graphs_val = graphs, graphs
            targets_train, targets_val = targets, targets

        # Create datasets first
        train_dataset = GraphDataset(graphs_train, targets_train, self.vectorizer, self.node_vectorizer, attribute_key=self.attribute_key)
        val_dataset = GraphDataset(graphs_val, targets_val, self.vectorizer, self.node_vectorizer, attribute_key=self.attribute_key)

        # Extract dimensions from first item in dataset
        A, F, _ = train_dataset[0]
        self.input_dim = A.size(-1)
        self.feature_dim = F.size(-1)
        self.num_classes = len(np.unique(targets))

        # Initialize classifier with extracted dimensions
        self.classifier = MultiHeadBilinearGraphClassifier(
            input_dim=self.input_dim,
            feature_dim=self.feature_dim,
            projection_dim=self.projection_dim,
            projection_hidden_dims=self.projection_hidden_dims,
            num_classes=self.num_classes,
            classifier_hidden_dims=self.classifier_hidden_dims,
            lr=self.lr,
            dropout=self.dropout,
            use_batchnorm=self.use_batchnorm,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            scheduler_class=self.scheduler_class,
            scheduler_kwargs=self.scheduler_kwargs,
            rank_factor=self.rank_factor, # Pass rank_factor
            ortho_lambda=self.ortho_lambda,
            num_heads=self.num_heads,
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            drop_last=True,
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)
        
        # Configure trainer with CSV logging and GPU support
        csv_logger = pl.loggers.CSVLogger(save_dir="logs", name="graph_model")
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            logger=csv_logger,
            accelerator=self.device, # "gpu" or "cpu"
            devices=1, # Use 1 device (either 1 GPU or 1 CPU core)
            enable_checkpointing=False,
            enable_progress_bar=False,  # <--- Add this line to disable progress bar
            enable_model_summary=False, # <--- (Optional) disables model summary printout
            log_every_n_steps=1,       # <--- Log every step
            )
        # record path for plot_metrics
        self.train_logs_path = Path(csv_logger.log_dir) / "metrics.csv"

        trainer.fit(self.classifier, train_loader, val_loader)
        return self

    def plot_metrics(self, window: int = 10, alpha: float = 0.3, log_scale_loss: bool = True) -> None:
        import pandas as pd
        import matplotlib.pyplot as plt
        df = pd.read_csv(self.train_logs_path)
        metric_names: List[str] = [c for c in df.columns if any(m in c for m in ["train_loss", "val_loss", "val_acc", "ortho_penalty"])]
        train_metrics = {k: df[k].dropna().tolist() for k in metric_names if "train" in k}
        val_metrics = {k: df[k].dropna().tolist() for k in metric_names if "val" in k}

        def _moving_average(data: Sequence[float], window_size: int) -> np.ndarray:
            arr = np.asarray(data, dtype=float)
            if len(arr) < window_size:
                return np.array([])
            # Clamp values for log scale if needed, but only if log_scale_loss is True for that metric
            arr_clamped = np.where(arr <= 0, np.finfo(float).tiny, arr) if log_scale_loss and "loss" in name else arr
            log_arr = np.log(arr_clamped)
            kernel = np.ones(window_size, dtype=float) / window_size
            smoothed_log = np.convolve(log_arr, kernel, mode='valid')
            return np.exp(smoothed_log)

        fig, ax0 = plt.subplots(figsize=(15, 8))
        metrics = list(train_metrics.keys())
        axes: List[plt.Axes] = [ax0] + [ax0.twinx() for _ in range(len(metrics) - 1)] # type: ignore
        for i, ax in enumerate(axes[1:], start=1):
            ax.spines['right'].set_position(('outward', 60 * i))
        colors = ["blue", "red", "green", "purple", "orange"]
        lines, labels = [], []
        for name, ax, color in zip(metrics, axes, colors):
            train_vals = train_metrics.get(name, [])
            val_name = name.replace("train_", "val_")
            val_vals = val_metrics.get(val_name, [])
            if len(train_vals) < 1 or len(val_vals) < 1:
                continue
            N = min(len(train_vals), len(val_vals))
            train = train_vals[:N]
            val = val_vals[:N]
            epochs = np.arange(1, N+1)
            ax.plot(epochs, train, color=color, alpha=alpha)
            ax.plot(epochs, val, color=color, linestyle='--', alpha=alpha)
            sm_train = _moving_average(train, window)
            sm_val = _moving_average(val, window)
            if sm_train.size:
                sm_epochs = np.arange(window, window + len(sm_train))
                l1, = ax.plot(sm_epochs, sm_train, color=color, linewidth=2,
                           label=f"Train {name} (MA{window})")
                l2, = ax.plot(sm_epochs, sm_val, color=color, linewidth=2, linestyle='--',
                           label=f"Val {val_name} (MA{window})")
                lines += [l1, l2]
                labels += [f"Train {name} (MA{window})", f"Val {val_name} (MA{window})"]
            ax.set_ylabel(name, color=color)
            ax.tick_params(axis='y', labelcolor=color)
            if log_scale_loss and ("loss" in name or "penalty" in name):
                ax.set_yscale('log')

        fig.legend(lines, labels, loc='upper center', ncol=max(1, len(lines)//2), fontsize='small')
        ax0.set_xlabel("Epoch")
        ax0.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.show()
    
    def predict(self, graphs: List[nx.Graph]) -> List[int]:
        self.classifier.eval()
        dataset = GraphDataset(graphs, targets=None, vectorizer=self.vectorizer, node_vectorizer=self.node_vectorizer, attribute_key=self.attribute_key)
        dataloader: DataLoader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)
        predictions = []
        # figure out where the model lives (CPU or GPU)
        dev = next(self.classifier.parameters()).device
        with torch.no_grad():
            for A, F, _, mask in dataloader:
                A, F, mask = A.to(dev), F.to(dev), mask.to(dev)
                logits = self.classifier(A, F, mask)
                predictions.extend(logits.argmax(dim=1).tolist())
        return predictions

    def predict_proba(self, graphs: List[nx.Graph]) -> np.ndarray:
        """
        Return class probabilities (shape: [n_samples, n_classes]).
        Mirrors scikit‑learn's API so you can plug this model into
        CalibratedClassifierCV, VotingClassifier, etc.
        """
        self.classifier.eval()
        dataset = GraphDataset(
            graphs, targets=None, vectorizer=self.vectorizer,
            node_vectorizer=self.node_vectorizer,
            attribute_key=self.attribute_key
        )
        loader: DataLoader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            collate_fn=self.collate_fn
        )
        dev = next(self.classifier.parameters()).device
        probs = []
        with torch.no_grad():
            for A, F, _, mask in loader:
                A, F, mask = A.to(dev), F.to(dev), mask.to(dev)
                logits = self.classifier(A, F, mask)
                probs.append(torch.softmax(logits, dim=1).cpu())
        return torch.cat(probs).numpy()

    def transform(self, graphs: List[nx.Graph]) -> List[np.ndarray]:
        """
        For each graph, return a numpy array of shape (num_nodes, num_heads·p' + (num_heads+1)·f),
        where each row is the concatenation of the transformed node attributes (A_trans_cat),
        node features (F_batch), and P (message-passed features).
        Padded nodes are removed.
        Returns: list of np.ndarray, one per graph.
        """
        self.classifier.eval()
        dataset = GraphDataset(
            graphs, targets=None, vectorizer=self.vectorizer,
            node_vectorizer=self.node_vectorizer,
            attribute_key=self.attribute_key
        )
        loader: DataLoader = DataLoader(
            dataset, batch_size=1, shuffle=False, collate_fn=self.collate_fn
        )
        dev = next(self.classifier.parameters()).device
        results: List[np.ndarray] = []
        with torch.no_grad():
            for A_b, F_b, _, mask_b in loader: # A_b, F_b, mask_b have batch_size=1
                A_b, F_b, mask_b = A_b.to(dev), F_b.to(dev), mask_b.to(dev)
                
                # Get node-level representations from the classifier's embed method
                # Shapes: (1, N_max, num_heads*p'), (1, N_max, f), (1, N_max, num_heads*f)
                A_trans_cat_b, F_masked_b, P_cat_b = self.classifier.embed(A_b, F_b, mask_b) # type: ignore
                
                # Squeeze the batch dimension
                A_trans_cat_single = A_trans_cat_b.squeeze(0) # (N_max, num_heads*p')
                F_single = F_masked_b.squeeze(0)             # (N_max, f)
                P_cat_single = P_cat_b.squeeze(0)             # (N_max, num_heads*f)
                
                # Get the 1D mask for valid nodes
                valid_idx = mask_b.squeeze(0).bool()      # (N_max,)

                # Concatenate features for valid nodes
                node_repr = torch.cat([
                    A_trans_cat_single[valid_idx],
                    F_single[valid_idx], # F_single is already masked F_batch
                    P_cat_single[valid_idx]
                ], dim=1)
                results.append(node_repr.cpu().numpy())
        return results

    def collate_fn(self, batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        A_batch, F_batch, T_batch = zip(*batch)
        A_padded: torch.Tensor = pad_sequence(A_batch, batch_first=True)
        F_padded: torch.Tensor = pad_sequence(F_batch, batch_first=True)
        mask: torch.Tensor = torch.tensor([[1]*len(x)+[0]*(A_padded.size(1)-len(x)) for x in A_batch], dtype=torch.bool)
        T_tensor: torch.Tensor = torch.tensor(T_batch, dtype=torch.long)
        return A_padded, F_padded, T_tensor, mask
