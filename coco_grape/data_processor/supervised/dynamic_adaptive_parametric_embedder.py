import math
import os
import sys
import contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split

from scipy.stats import norm, t

############################################
# Utility Context Manager
############################################

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

############################################
# Callback to Interrupt Training if LR is Too Low
############################################

class StopWhenLRBelow(pl.Callback):
    def __init__(self, min_lr=1e-8, verbose=False):
        super().__init__()
        self.min_lr = min_lr
        self.verbose = verbose

    def on_validation_epoch_end(self, trainer, pl_module):
        optimizer = trainer.optimizers[0]
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < self.min_lr:
            if self.verbose:
                print(f"Learning rate reached {current_lr:.2e}, which is below {self.min_lr:.2e}. Stopping training.")
            trainer.should_stop = True

############################################
# Custom Low-Rank Linear Layer
############################################

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, thin_size, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.thin_size = thin_size
        self.A = nn.Parameter(torch.Tensor(in_features, thin_size))
        self.B = nn.Parameter(torch.Tensor(thin_size, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        out = x @ self.A @ self.B
        if self.bias is not None:
            out = out + self.bias
        return out

############################################
# Residual Block with Dropout and LeakyReLU
############################################

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, thin_size, dropout_prob, negative_slope):
        super().__init__()
        self.linear = LowRankLinear(in_features, out_features, thin_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.LeakyReLU(negative_slope)
        if in_features != out_features:
            self.skip = LowRankLinear(in_features, out_features, thin_size)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        out = self.linear(x)
        out = self.dropout(out)
        out = self.activation(out)
        return out + self.skip(x)

############################################
# LowRankMLP Network using Residual Blocks
############################################

class LowRankMLPNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, hidden_dim, thin_size, dropout, negative_slope):
        super().__init__()
        self.hidden_layers = hidden_layers
        layers = []
        if hidden_layers > 0:
            layers.append(ResidualBlock(input_dim, hidden_dim, thin_size, dropout, negative_slope))
            for _ in range(hidden_layers - 1):
                layers.append(ResidualBlock(hidden_dim, hidden_dim, thin_size, dropout, negative_slope))
            self.blocks = nn.ModuleList(layers)
            self.out_layer = LowRankLinear(hidden_dim, output_dim, thin_size)
        else:
            self.out_layer = LowRankLinear(input_dim, output_dim, thin_size)
        
    def forward(self, x):
        if self.hidden_layers > 0:
            for block in self.blocks:
                x = block(x)
        return self.out_layer(x)

############################################
# Helper: 1D Piecewise Linear Interpolation for Spline
############################################

def linear_interp_1d(x, xp, fp):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    x = x.to(xp.device)
    x_clamped = torch.clamp(x, min=xp[0], max=xp[-1])
    indices = torch.searchsorted(xp, x_clamped, right=True) - 1
    indices = torch.clamp(indices, 0, len(xp) - 2)
    x0 = xp[indices]
    x1 = xp[indices + 1]
    y0 = fp[indices]
    y1 = fp[indices + 1]
    fraction = (x_clamped - x0) / (x1 - x0 + 1e-12)
    return y0 + fraction * (y1 - y0)

############################################
# Merging Strategy for Knot Positions
############################################

def merge_close_knots(knots, min_spacing, min_knots=3):
    """
    Merges knots that are closer than `min_spacing` and enforces a minimum number of knots.
    """
    # Ensure knots are sorted
    knots = np.sort(knots)
    merged = [knots[0]]
    for k in knots[1:]:
        if k - merged[-1] < min_spacing:
            # Merge: update the last knot to be the average
            merged[-1] = (merged[-1] + k) / 2.0
        else:
            merged.append(k)
    merged = np.array(merged)
    # Enforce minimum number of knots if merging was too aggressive.
    if len(merged) < min_knots:
        merged = np.linspace(knots[0], knots[-1], min_knots)
    return merged

############################################
# Piecewise Linear Spline for Low-Dim Similarities
############################################

class PiecewiseLinearSpline(nn.Module):
    def __init__(self, knot_positions):
        super().__init__()
        # Enforce that we have at least 3 unique knot positions for stability.
        knot_positions = np.array(knot_positions)
        if len(np.unique(knot_positions)) < 3:
            raise ValueError("At least 3 unique knot positions are required.")
        self.register_buffer("knot_positions", torch.tensor(knot_positions, dtype=torch.float32))
        self.theta = nn.Parameter(torch.zeros(len(knot_positions), dtype=torch.float32))

    def forward_spline(self, r):
        return linear_interp_1d(r, self.knot_positions, self.theta)

    def normalization_constant(self, integration_steps=100):
        # Create a grid from 0 to the maximum knot value.
        grid = torch.linspace(0, self.knot_positions[-1], steps=integration_steps, device=self.theta.device)
        spline_vals = linear_interp_1d(grid, self.knot_positions, self.theta)
        integrand = torch.exp(spline_vals)
        Z_val = torch.trapz(integrand, grid)
        return Z_val

    def pdf(self, r):
        spline_vals = self.forward_spline(r)
        Z = self.normalization_constant()
        return torch.exp(spline_vals) / (Z + 1e-12)

    def smoothness_reg(self):
        theta_diff2 = torch.diff(self.theta, n=2)
        return torch.sum(theta_diff2 ** 2)

    def monotonic_reg(self):
        first_diffs = torch.diff(self.theta)
        return torch.sum(torch.clamp(first_diffs, min=0.0) ** 2)

    def tail_decay_reg(self, alpha=1.0):
        idx_start = int(0.8 * len(self.knot_positions))
        if idx_start >= len(self.knot_positions):
            return torch.tensor(0.0, device=self.theta.device)
        r_c = self.knot_positions[idx_start]
        tail_knots = self.knot_positions[idx_start:]
        tail_theta = self.theta[idx_start:]
        start_val = tail_theta[0]
        target = start_val - alpha * (tail_knots - r_c)
        return torch.sum((tail_theta - target) ** 2)

    def total_reg(self, w_smooth=1.0, w_mono=1.0, w_tail=1.0):
        return (w_smooth * self.smoothness_reg() +
                w_mono * self.monotonic_reg() +
                w_tail * self.tail_decay_reg())

############################################
# PyTorch Lightning Module: DynamicAdaptiveParametricEmbedderModel
############################################

class DynamicAdaptiveParametricEmbedderModel(pl.LightningModule):
    def __init__(self, X_tensor, y_tensor, P_tensor, knot_positions,
                 input_dim, embedding_dim, output_dim, task,
                 encoder_params, task_net_params,
                 lambda_estimator=1.0, lambda_smooth=1.0, lambda_mono=1.0, lambda_tail=1.0,
                 learning_rate=1e-3, verbose=False, alpha_final=1.0, disable_dual_kl=False):
        super().__init__()
        self.save_hyperparameters(ignore=["X_tensor", "y_tensor", "P_tensor"])
        # Register the input tensors as buffers so they move with the model.
        self.register_buffer("X", X_tensor)
        self.register_buffer("y", y_tensor)
        self.register_buffer("P", P_tensor)
        self.n = X_tensor.shape[0]
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.task = task
        
        self.encoder = LowRankMLPNet(
            input_dim=input_dim,
            output_dim=embedding_dim,
            hidden_layers=encoder_params.get("hidden_layers", 3),
            hidden_dim=encoder_params.get("hidden_dim", 64),
            thin_size=encoder_params.get("thin_size", 32),
            dropout=encoder_params.get("dropout", 0.2),
            negative_slope=encoder_params.get("negative_slope", 0.01)
        )
        
        self.estimator = LowRankMLPNet(
            input_dim=embedding_dim,
            output_dim=task_net_params.get("output_dim", output_dim),
            hidden_layers=task_net_params.get("hidden_layers", 2),
            hidden_dim=task_net_params.get("hidden_dim", 32),
            thin_size=task_net_params.get("thin_size", 16),
            dropout=task_net_params.get("dropout", 0.1),
            negative_slope=task_net_params.get("negative_slope", 0.01)
        )
        
        self.spline = PiecewiseLinearSpline(knot_positions)
        
        self.lambda_estimator = lambda_estimator
        self.lambda_smooth = lambda_smooth
        self.lambda_mono = lambda_mono
        self.lambda_tail = lambda_tail
        self.lr = learning_rate
        self.verbose = verbose
        self.alpha_final = alpha_final  # Target value at final epoch.
        self.disable_dual_kl = disable_dual_kl  # If True, only use KL(P||Q).
        
        self.train_losses = []
        self.val_losses = []
        self.train_task_losses = []
        self.val_task_losses = []
        self.train_imp_losses = []
        self.val_imp_losses = []
        self._train_total = []
        self._train_task = []
        self._train_imp = []
        self._val_total = []
        self._val_task = []
        self._val_imp = []
        
        if self.task == "classification":
            self.task_loss_fn = nn.CrossEntropyLoss()
        elif self.task == "regression":
            self.task_loss_fn = nn.MSELoss()
        else:
            raise ValueError("Task must be 'classification' or 'regression'.")
    
    def forward(self, x):
        embedding = self.encoder(x)
        preds = self.estimator(embedding)
        return preds, embedding
    
    def compute_kl_divergence(self, indices):
        indices = indices.to(self.X.device)  # Ensure indices are on the same device as X
        X_batch = self.X[indices]
        with torch.no_grad():
            P_batch = self.P[indices][:, indices] + 1e-12
        embedding = self.encoder(X_batch)
        diff = embedding.unsqueeze(1) - embedding.unsqueeze(0)
        dist = torch.sqrt(torch.clamp(torch.sum(diff**2, dim=-1), min=1e-12))
        spline_vals = self.spline.forward_spline(dist)
        Q = torch.exp(spline_vals)
        Q = Q / (torch.sum(Q) + 1e-12) + 1e-12
        
        kl_pq = torch.sum(P_batch * torch.log(P_batch / Q))
        if self.disable_dual_kl:
            return kl_pq
        
        kl_qp = torch.sum(Q * torch.log(Q / P_batch))
        if self.trainer is not None:
            T = self.trainer.max_epochs
            t = self.current_epoch
            alpha = 1 - (t / T) * (1 - self.alpha_final)
        else:
            alpha = 1.0
        return alpha * kl_pq + (1 - alpha) * kl_qp

    def compute_task_loss(self, indices):
        indices = indices.to(self.X.device)  # Ensure indices are on the same device as X
        X_batch = self.X[indices]
        y_batch = self.y[indices]
        preds, _ = self(X_batch)
        return self.task_loss_fn(preds, y_batch)
    
    def training_step(self, batch, batch_idx):
        indices = batch  # Tensor of global indices.
        kl_loss = self.compute_kl_divergence(indices)
        task_loss = self.compute_task_loss(indices)
        reg_loss = self.spline.total_reg(w_smooth=self.lambda_smooth,
                                         w_mono=self.lambda_mono,
                                         w_tail=self.lambda_tail)
        total_loss = kl_loss + self.lambda_estimator * task_loss + reg_loss
        self._train_total.append(total_loss.detach())
        self._train_task.append(task_loss.detach())
        self._train_imp.append((kl_loss + reg_loss).detach())
        self.log("train_total_loss", total_loss, prog_bar=False)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        indices = batch
        kl_loss = self.compute_kl_divergence(indices)
        task_loss = self.compute_task_loss(indices)
        reg_loss = self.spline.total_reg(w_smooth=self.lambda_smooth,
                                         w_mono=self.lambda_mono,
                                         w_tail=self.lambda_tail)
        total_loss = kl_loss + self.lambda_estimator * task_loss + reg_loss
        self._val_total.append(total_loss.detach())
        self._val_task.append(task_loss.detach())
        self._val_imp.append((kl_loss + reg_loss).detach())
        self.log("val_total_loss", total_loss, prog_bar=False)
        return total_loss
    
    def on_train_epoch_end(self):
        if self._train_total:
            avg_total = torch.stack(self._train_total).mean().item()
            avg_task = torch.stack(self._train_task).mean().item()
            avg_imp = torch.stack(self._train_imp).mean().item()
            self.train_losses.append(avg_total)
            self.train_task_losses.append(avg_task)
            self.train_imp_losses.append(avg_imp)
            self._train_total = []
            self._train_task = []
            self._train_imp = []
    
    def on_validation_epoch_end(self):
        if self._val_total:
            avg_total = torch.stack(self._val_total).mean().item()
            avg_task = torch.stack(self._val_task).mean().item()
            avg_imp = torch.stack(self._val_imp).mean().item()
            self.val_losses.append(avg_total)
            self.val_task_losses.append(avg_task)
            self.val_imp_losses.append(avg_imp)
            self._val_total = []
            self._val_task = []
            self._val_imp = []
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=3,
            factor=0.1,
            verbose=self.verbose
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_total_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def on_train_end(self):
        if self.verbose:
            min_length = min(len(self.train_losses), len(self.val_losses),
                             len(self.train_task_losses), len(self.val_task_losses),
                             len(self.train_imp_losses), len(self.val_imp_losses))
            if min_length == 0:
                print("No training or validation losses recorded.")
                return
            self.train_losses = self.train_losses[:min_length]
            self.val_losses = self.val_losses[:min_length]
            self.train_task_losses = self.train_task_losses[:min_length]
            self.val_task_losses = self.val_task_losses[:min_length]
            self.train_imp_losses = self.train_imp_losses[:min_length]
            self.val_imp_losses = self.val_imp_losses[:min_length]
            skip_first = 5 if min_length > 5 else 0
            epochs = range(skip_first + 1, min_length + 1)
            
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, self.train_losses[skip_first:], label='Overall Train Loss')
            plt.plot(epochs, self.val_losses[skip_first:], label='Overall Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Overall Training and Validation Losses')
            plt.yscale('log')
            plt.legend()
            plt.grid(True)
            plt.show()
            
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, self.train_task_losses[skip_first:], label=f'Task Train Loss ({self.task})')
            plt.plot(epochs, self.val_task_losses[skip_first:], label=f'Task Val Loss ({self.task})')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Task Training and Validation Losses ({self.task})')
            plt.yscale('log')
            plt.legend()
            plt.grid(True)
            plt.show()
            
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, self.train_imp_losses[skip_first:], label='Encoder Train Loss')
            plt.plot(epochs, self.val_imp_losses[skip_first:], label='Encoder Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Encoder (High-dim to Low-dim) Training and Validation Losses')
            plt.yscale('log')
            plt.legend()
            plt.grid(True)
            plt.show()

############################################
# Dataset yielding indices for mini-batch training
############################################

class IndexDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx]

############################################
# Cached Mixed Mini-Batch Dataset for Improved Sampling
############################################

class CachedMixedIndexDataset(Dataset):
    """
    Precomputes and caches mixed mini-batches.
    
    For each anchor in the provided indices, the mini-batch is built by:
      - Including the anchor.
      - Including a fixed number of its near neighbors (from a precomputed neighbor dictionary).
      - Filling the rest of the batch with randomly chosen points from the global indices.
    
    This caching avoids repeated Python-level overhead during training.
    """
    def __init__(self, indices, neighbor_dict, batch_size, near_frac=0.5):
        self.indices = np.array(indices)
        self.neighbor_dict = neighbor_dict
        self.batch_size = batch_size
        self.near_frac = near_frac
        self.cached_batches = []
        self.precompute_batches()
       
    def precompute_batches(self):
        self.cached_batches = []
        for anchor in self.indices:
            near_count = int(self.batch_size * self.near_frac)
            random_count = self.batch_size - near_count - 1
            neighbors = self.neighbor_dict.get(anchor, [])
            near_neighbors = neighbors[:near_count] if len(neighbors) >= near_count else list(neighbors)
            available = set(self.indices) - {anchor} - set(near_neighbors)
            if len(available) < random_count:
                random_neighbors = list(available)
            else:
                random_neighbors = np.random.choice(list(available), size=random_count, replace=False).tolist()
            mini_batch = [anchor] + near_neighbors + random_neighbors
            self.cached_batches.append(torch.tensor(mini_batch, dtype=torch.long))
       
    def __len__(self):
        return len(self.indices)
       
    def __getitem__(self, idx):
        return self.cached_batches[idx]

############################################
# scikit-learn–Style Wrapper: DynamicAdaptiveParametricEmbedder
############################################

class DynamicAdaptiveParametricEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self,
                 task="classification",
                 embedding_dim=2,
                 n_neighbors=30,
                 batch_size=32,
                 lambda_estimator=1.0,
                 lambda_smooth=1.0,
                 lambda_mono=1.0,
                 lambda_tail=1.0,
                 learning_rate=1e-3,
                 max_epochs=200,
                 random_state=42,
                 encoder_hidden_layers=3,
                 encoder_hidden_dim=64,
                 encoder_thin_size=32,
                 encoder_dropout=0.2,
                 encoder_negative_slope=0.01,
                 task_hidden_layers=2,
                 task_hidden_dim=32,
                 task_thin_size=16,
                 task_dropout=0.1,
                 task_negative_slope=0.01,
                 verbose=False,
                 near_frac=0.5,
                 alpha_final=0.5,
                 disable_dual_kl=False,
                 disable_mixed_sampling=False,
                 n_knots=4):
        self.task = task
        self.embedding_dim = embedding_dim
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size
        self.lambda_estimator = lambda_estimator
        self.lambda_smooth = lambda_smooth
        self.lambda_mono = lambda_mono
        self.lambda_tail = lambda_tail
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.random_state = random_state
        self.encoder_hidden_layers = encoder_hidden_layers
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_thin_size = encoder_thin_size
        self.encoder_dropout = encoder_dropout
        self.encoder_negative_slope = encoder_negative_slope
        self.task_hidden_layers = task_hidden_layers
        self.task_hidden_dim = task_hidden_dim
        self.task_thin_size = task_thin_size
        self.task_dropout = task_dropout
        self.task_negative_slope = task_negative_slope
        self.verbose = verbose
        self.near_frac = near_frac
        self.alpha_final = alpha_final
        self.disable_dual_kl = disable_dual_kl
        self.disable_mixed_sampling = disable_mixed_sampling
        self.n_knots = n_knots  # initial number of knots (before merging)
        self.module_ = None
        self.trainer_ = None

    def _compute_P(self, X_np, n_neighbors):
        dist_mat = pairwise_distances(X_np, metric='euclidean')
        n = dist_mat.shape[0]
        P = np.zeros_like(dist_mat)
        for i in range(n):
            indices = np.argsort(dist_mat[i])
            indices = indices[indices != i]
            neighbors = indices[:n_neighbors]
            d = dist_mat[i, neighbors]
            sigma = np.median(d)
            P[i, neighbors] = np.exp(-d**2 / (2 * sigma**2))
        P = (P + P.T) / 2
        P /= P.sum()
        return P

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        X_np = np.asarray(X, dtype=np.float32)
        if self.task == "classification":
            y_np = np.asarray(y, dtype=np.int64)
            output_dim = len(np.unique(y_np))
        elif self.task == "regression":
            y_np = np.asarray(y, dtype=np.float32)
            output_dim = y_np.shape[1] if y_np.ndim > 1 else 1
        else:
            raise ValueError("Task must be 'classification' or 'regression'")
        n, input_dim = X_np.shape
        P_np = self._compute_P(X_np, self.n_neighbors)
        dist_vals = pairwise_distances(X_np, metric='euclidean').ravel()
        dist_vals = dist_vals[dist_vals > 0]
        # Compute initial quantile-based knot positions
        initial_knots = np.quantile(dist_vals, np.linspace(0, 1, self.n_knots))
        # Define a minimum spacing threshold as 1% of the range
        min_spacing = 0.01 * (initial_knots[-1] - initial_knots[0])
        # Merge knots that are too close, enforcing at least 3 knots.
        final_knots = merge_close_knots(initial_knots, min_spacing, min_knots=3)
        knot_positions = final_knots

        X_tensor = torch.tensor(X_np)
        y_tensor = torch.tensor(y_np)
        P_tensor = torch.tensor(P_np, dtype=torch.float32)
        indices = np.arange(n)
        train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=self.random_state)
        train_idx = torch.tensor(train_idx, dtype=torch.long)
        val_idx = torch.tensor(val_idx, dtype=torch.long)
        
        # Build neighbor dictionary for training indices using high-dimensional space.
        train_indices = train_idx.numpy()
        train_X = X_np[train_indices]
        dist_matrix = pairwise_distances(train_X, metric='euclidean')
        neighbor_dict = {}
        for i, global_idx in enumerate(train_indices):
            sorted_indices = np.argsort(dist_matrix[i])
            sorted_indices = sorted_indices[sorted_indices != i]
            neighbor_global = train_indices[sorted_indices]
            neighbor_dict[global_idx] = neighbor_global.tolist()
        
        encoder_params = {
            "hidden_layers": self.encoder_hidden_layers,
            "hidden_dim": self.encoder_hidden_dim,
            "thin_size": self.encoder_thin_size,
            "dropout": self.encoder_dropout,
            "negative_slope": self.encoder_negative_slope
        }
        task_params = {
            "hidden_layers": self.task_hidden_layers,
            "hidden_dim": self.task_hidden_dim,
            "thin_size": self.task_thin_size,
            "dropout": self.task_dropout,
            "negative_slope": self.task_negative_slope,
            "output_dim": output_dim
        }
        self.module_ = DynamicAdaptiveParametricEmbedderModel(
            X_tensor=X_tensor,
            y_tensor=y_tensor,
            P_tensor=P_tensor,
            knot_positions=knot_positions,
            input_dim=input_dim,
            embedding_dim=self.embedding_dim,
            output_dim=output_dim,
            encoder_params=encoder_params,
            task_net_params=task_params,  
            lambda_estimator=self.lambda_estimator,
            lambda_smooth=self.lambda_smooth,
            lambda_mono=self.lambda_mono,
            lambda_tail=self.lambda_tail,
            learning_rate=self.learning_rate,
            verbose=self.verbose,
            task=self.task,
            alpha_final=self.alpha_final,
            disable_dual_kl=self.disable_dual_kl
        )
        
        # Choose dataset type based on mixed sampling flag.
        if self.disable_mixed_sampling:
            train_dataset = IndexDataset(train_indices)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        else:
            train_dataset = CachedMixedIndexDataset(train_indices, neighbor_dict, self.batch_size, near_frac=self.near_frac)
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
            
        val_dataset = IndexDataset(val_idx.numpy())
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        if not self.verbose:
            import logging
            logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
            from pytorch_lightning.utilities import rank_zero
            rank_zero.info = lambda *args, **kwargs: None
            rank_zero.warn = lambda *args, **kwargs: None

        self.trainer_ = pl.Trainer(
            max_epochs=self.max_epochs,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            callbacks=[StopWhenLRBelow(min_lr=1e-8, verbose=self.verbose)]
        )

        if not self.verbose:
            with suppress_output():
                self.trainer_.fit(self.module_, train_loader, val_loader)
        else:
            self.trainer_.fit(self.module_, train_loader, val_loader)
        return self

    def transform(self, X):
        self.module_.eval()
        X_tensor = torch.tensor(np.asarray(X, dtype=np.float32))
        # Ensure the input tensor is on the same device as the model's buffers
        X_tensor = X_tensor.to(self.module_.X.device)
        with torch.no_grad():
            embedding = self.module_.encoder(X_tensor)
        return embedding.cpu().numpy()

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def display_pdf(self, degrees_of_freedom=[1, 2, 5], r_min=0, r_max=None, num_points=200):
        """
        Plots the spline PDF learned by a fitted DynamicAdaptiveParametricEmbedder along with
        a Gaussian PDF (unit variance) and t-Student PDFs for a list of degrees of freedom.
        The knot positions (nodes) of the embedder are marked with white circles with black edges.
        """
        spline = self.module_.spline
        knot_positions = spline.knot_positions.cpu().numpy()
        if r_max is None:
            r_max = knot_positions[-1] * 1.2  # extend a bit beyond the last knot

        r_values = np.linspace(r_min, r_max, num_points)
        r_tensor = torch.tensor(r_values, dtype=torch.float32, device=spline.theta.device)

        spline.eval()
        with torch.no_grad():
            spline_pdf = spline.pdf(r_tensor).cpu().numpy()
        area_spline = np.trapz(spline_pdf, r_values)

        gaussian_pdf = 2 * norm.pdf(r_values, loc=0, scale=1)
        area_gaussian = np.trapz(gaussian_pdf, r_values)

        t_pdfs = {}
        area_t = {}
        for df in degrees_of_freedom:
            pdf_values = 2 * t.pdf(r_values, df=df)
            t_pdfs[df] = pdf_values
            area_t[df] = np.trapz(pdf_values, r_values)

        knot_r_tensor = torch.tensor(knot_positions, dtype=torch.float32, device=spline.theta.device)
        with torch.no_grad():
            knot_pdf = spline.pdf(knot_r_tensor).cpu().numpy()

        plt.figure(figsize=(10, 6))
        plt.plot(r_values, spline_pdf, label=f"Spline PDF (area={area_spline:.2f})", linewidth=1.5, color='black')
        plt.plot(r_values, gaussian_pdf, label=f"Gaussian PDF (σ=1, area={area_gaussian:.2f})", linestyle="--")
        for df, pdf_values in t_pdfs.items():
            plt.plot(r_values, pdf_values, label=f"t-Student PDF (df={df}, area={area_t[df]:.2f})", linestyle=":")

        plt.scatter(knot_positions, knot_pdf, s=40, linewidth=1.5, facecolors='white', edgecolors='black', zorder=5, label="Spline Nodes")
        plt.xlabel("Distance r")
        plt.ylabel("PDF")
        plt.title("Spline PDF vs. Gaussian and t-Student PDFs")
        plt.legend()
        plt.grid(True)
        plt.show()
