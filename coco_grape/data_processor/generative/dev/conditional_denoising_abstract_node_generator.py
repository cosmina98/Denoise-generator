import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import contextlib, os, sys
from torch.utils.data import random_split, DataLoader, TensorDataset, Dataset
from typing import Dict, Sequence, Optional, Union, Tuple, List, Any
import math

class CustomRobustScaler:
    """
    Robust‐scale each numeric feature using (median, IQR).  
    *Constant* columns are copied through unchanged; a designated existence-flag
    column is always left raw.  O(1) feature-index lookup is provided.

    Parameters
    ----------
    quantile_range : tuple(float, float), default (5, 95)
        Lower / upper percentiles used to compute IQR.
    epsilon : float, default 1e-8
        Threshold below which an IQR is treated as zero.
    special_features : Sequence[int] | None, default None
        Features that must **never** be dropped even if their IQR ≈ 0.
    fallback_scale : float, default 1.0
        Substitute scale applied when a special-feature IQR is below *epsilon*.
    exist_col : Optional[int], default None
        If given, that feature index will never be scaled or dropped.

    Notes
    -----
    `transform_aggregated` / `inverse_transform_aggregated`
    additionally divide / multiply the **non-existence features** by a
    per-sample aggregation factor (shape `(B,)`).
    """

    # --------------------------------------------------------------------- #
    #  CONSTRUCTOR
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        quantile_range: Tuple[float, float] = (5.0, 95.0),
        epsilon: float = 1e-8,
        special_features: Optional[Sequence[int]] = None,
        fallback_scale: float = 1.0,
        exist_col: Optional[int] = None,
    ):
        self.quantile_range = quantile_range
        self.epsilon = epsilon
        self.special_features = set(special_features) if special_features else set()
        self.fallback_scale = fallback_scale
        self.exist_col = exist_col  # may be None

        # populated by fit()
        self.original_dim: Optional[int] = None
        self.constant_mask: Optional[np.ndarray] = None        # bool (d,)
        self.nonconstant_mask: Optional[np.ndarray] = None     # bool (d,)
        self.scale_mask: Optional[np.ndarray] = None           # bool (d,) — scaled cols
        self.median_: Optional[np.ndarray] = None              # float64 (k,)
        self.iqr_: Optional[np.ndarray] = None                 # float64 (k,)
        self.constant_means: Optional[np.ndarray] = None       # float64 (c,)
        self.constant_index_map: dict[int, int] = {}           # orig idx → const row
        self.orig2scaled_idx: Optional[np.ndarray] = None      # int (d,)  -1 for const

    # --------------------------------------------------------------------- #
    #  FIT
    # --------------------------------------------------------------------- #
    def fit(self, X: np.ndarray):
        if X.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_features).")
        self.original_dim = X.shape[1]

        # if an existence‐flag column is specified, validate it
        if self.exist_col is not None:
            if not (0 <= self.exist_col < self.original_dim):
                raise IndexError(f"exist_col {self.exist_col} out of bounds for "
                                 f"{self.original_dim}-feature input.")

        # basic stats
        med = np.median(X, axis=0)
        q_low = np.percentile(X, self.quantile_range[0], axis=0)
        q_high = np.percentile(X, self.quantile_range[1], axis=0)
        iqr = q_high - q_low

        # identify constant columns
        constant_mask = iqr < self.epsilon
        # but never drop special features
        for sf in self.special_features:
            if 0 <= sf < self.original_dim:
                constant_mask[sf] = False

        # existence column is *never scaled*, if given
        if self.exist_col is not None:
            constant_mask[self.exist_col] = True

        self.constant_mask = constant_mask
        self.nonconstant_mask = ~constant_mask
        self.scale_mask = self.nonconstant_mask.copy()      # to be reused quickly
        if self.exist_col is not None:
            self.scale_mask[self.exist_col] = False         # make sure exist_col False

        # med / iqr only for columns we will scale
        self.median_ = med[self.scale_mask].astype(np.float64)
        self.iqr_ = iqr[self.scale_mask].astype(np.float64)

        # guarantee non-zero scale on special features (fallback)
        zero_iqr = self.iqr_ < self.epsilon
        self.iqr_[zero_iqr] = self.fallback_scale

        # constant means
        all_means = np.mean(X, axis=0)
        self.constant_means = all_means[self.constant_mask]
        const_idx = np.where(self.constant_mask)[0]
        self.constant_index_map = dict(zip(const_idx, range(len(const_idx))))

        # O(1) mapping original → scaled index
        self.orig2scaled_idx = np.full(self.original_dim, -1, dtype=int)
        self.orig2scaled_idx[self.scale_mask] = np.arange(self.median_.size)

        return self

    # --------------------------------------------------------------------- #
    #  INTERNAL VECTORISED HELPERS
    # --------------------------------------------------------------------- #
    def _vectorised_scale(self, X: np.ndarray) -> np.ndarray:
        Xs = X.astype(np.float64, copy=True)
        if self.scale_mask.any():                                 # guard empty mask
            Xs[:, self.scale_mask] = (Xs[:, self.scale_mask] - self.median_) / self.iqr_
        if self.constant_means.size:
            fill_mask = self.constant_mask.copy()
            # don't overwrite the existence column if set
            if self.exist_col is not None:
                fill_mask[self.exist_col] = False
            if fill_mask.any():
                const_idx = np.where(fill_mask[self.constant_mask])[0]
                Xs[:, fill_mask] = self.constant_means[const_idx]
        return Xs

    def _vectorised_inverse(self, Xs: np.ndarray) -> np.ndarray:
        Xi = Xs.astype(np.float64, copy=True)
        if self.scale_mask.any():
            Xi[:, self.scale_mask] = Xi[:, self.scale_mask] * self.iqr_ + self.median_
        if self.constant_means.size:
            fill_mask = self.constant_mask.copy()
            if self.exist_col is not None:
                fill_mask[self.exist_col] = False
            if fill_mask.any():
                const_idx = np.where(fill_mask[self.constant_mask])[0]
                Xi[:, fill_mask] = self.constant_means[const_idx]
        return Xi

    # --------------------------------------------------------------------- #
    #  PUBLIC TRANSFORMERS
    # --------------------------------------------------------------------- #
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._vectorised_scale(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        return self._vectorised_inverse(X_scaled)

    # ------------ Aggregated versions (per-sample scaling factors) -------- #
    def transform_aggregated(self, Y: np.ndarray, agg: np.ndarray) -> np.ndarray:
        if Y.shape[0] != agg.shape[0]:
            raise ValueError("agg must have same batch size as Y.")
        Y_scaled = Y.astype(np.float64, copy=True)
        median = self.median_[None, :] * agg[:, None]
        iqr = self.iqr_[None, :] * agg[:, None]
        if self.scale_mask.any():                              # no scalable cols → skip
            Y_scaled[:, self.scale_mask] = (
                Y_scaled[:, self.scale_mask] - median
            ) / iqr
        if self.constant_means.size:
            fill_mask = self.constant_mask.copy()
            if self.exist_col is not None:
                fill_mask[self.exist_col] = False
            if fill_mask.any():
                const_idx = np.where(fill_mask[self.constant_mask])[0]
                Y_scaled[:, fill_mask] = self.constant_means[const_idx]
        return Y_scaled

    def inverse_transform_aggregated(self, Ys: np.ndarray, agg: np.ndarray) -> np.ndarray:
        if Ys.shape[0] != agg.shape[0]:
            raise ValueError("agg must have same batch size as Ys.")
        Ys_inv = Ys.astype(np.float64, copy=True)
        median = self.median_[None, :] * agg[:, None]
        iqr = self.iqr_[None, :] * agg[:, None]
        if self.scale_mask.any():
            Ys_inv[:, self.scale_mask] = Ys_inv[:, self.scale_mask] * iqr + median
        if self.constant_means.size:
            fill_mask = self.constant_mask.copy()
            if self.exist_col is not None:
                fill_mask[self.exist_col] = False
            if fill_mask.any():
                const_idx = np.where(fill_mask[self.constant_mask])[0]
                Ys_inv[:, fill_mask] = self.constant_means[const_idx]
        return Ys_inv

    # --------------------------------------------------------------------- #
    #  UTILITY
    # --------------------------------------------------------------------- #
    def map_feature_index(self, original_index: int) -> int:
        if original_index < 0 or original_index >= self.original_dim:
            raise ValueError("Feature index out of range.")
        if not self.nonconstant_mask[original_index]:
            raise ValueError(f"Feature {original_index} is constant and was removed.")
        return int(self.orig2scaled_idx[original_index])


# --- Utility Context Manager ---
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

# --- Sinusoidal Time Embedding ---
def get_sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Convert diffusion timesteps t into sinusoidal embeddings.
    
    Args:
        t: Tensor of shape (B,1) with values in [0,1]
        dim: Desired embedding dimension (must be even)
    Returns:
        Tensor of shape (B,dim) containing time embeddings
    """
    half_dim = dim // 2
    inv_freq = torch.exp(
        torch.arange(0, half_dim, device=t.device).float() * (-math.log(10000) / (half_dim - 1))
    )
    # Shape: (B,1) * (D/2,) -> (B,D/2)
    angles = t * inv_freq.view(1, -1)
    # Shape: (B,D)
    return torch.cat([angles.sin(), angles.cos()], dim=-1)

# --- Cross-Attention Transformer Layer ---
class CrossTransformerEncoderLayer(nn.Module):
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1):
        super().__init__()
        # Self attention block
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross attention block with memory tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed-forward block
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Self attention
        x = x + self.dropout1(self.self_attn(x, x, x)[0])
        x = self.norm1(x)
        
        # Cross attention with memory tokens 
        x = x + self.dropout2(self.cross_attn(x, k, v)[0])
        x = self.norm2(x)
        
        # Feed-forward
        x = x + self.dropout3(self.ff(x))
        x = self.norm3(x)
        return x

# --- Plotting Metrics ---
def plot_metrics(
    train_metrics: Dict[str, Sequence[float]],
    val_metrics: Dict[str, Sequence[float]],
    window: int = 10,
    alpha: float = 0.3
) -> None:
    """Plot training metrics with geometric moving averages."""
    def _moving_average(data: Sequence[float], window_size: int) -> np.ndarray:
        arr = np.asarray(data, dtype=float)
        if len(arr) < window_size:
            return np.array([])
        arr_clamped = np.where(arr <= 0, np.finfo(float).tiny, arr)
        log_arr = np.log(arr_clamped)
        kernel = np.ones(window_size, dtype=float) / window_size
        smoothed_log = np.convolve(log_arr, kernel, mode='valid')
        return np.exp(smoothed_log)

    fig, ax0 = plt.subplots(figsize=(15, 8))
    metrics = list(train_metrics.keys())
    axes = [ax0] + [ax0.twinx() for _ in range(len(metrics) - 1)]
    for i, ax in enumerate(axes[1:], start=1):
        ax.spines['right'].set_position(('outward', 60 * i))
    colors = ["blue", "red", "green", "purple", "orange"]
    lines, labels = [], []
    for name, ax, color in zip(metrics, axes, colors):
        train_vals = train_metrics[name]
        val_vals = val_metrics[name]
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
                       label=f"Val {name} (MA{window})")
            lines += [l1, l2]
            labels += [f"Train {name} (MA{window})", f"Val {name} (MA{window})"]
        ax.set_ylabel(name, color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_yscale('log')
    
    fig.legend(lines, labels, loc='upper center', ncol=len(lines)//2, fontsize='small')
    ax0.set_xlabel("Epoch")
    ax0.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()

# =============================================================================
# Revised IterativeDenoisingAutoencoderTransformerModel with Cross-Attention
# =============================================================================

class IterativeDenoisingAutoencoderTransformerModel(pl.LightningModule):
    """
    PyTorch Lightning module implementing a transformer-based diffusion model for graph generation.
    
    This model combines a transformer architecture with noise scheduling and degree-specific 
    handling for generating graph-like structures.

    Parameters
    ----------
    number_of_rows_per_example : int
        Maximum number of rows per input example.
    input_feature_dimension : int
        Number of features per row in the input.
    condition_feature_dimension : int
        Dimension of the conditioning vector.
    latent_embedding_dimension : int
        Dimension of the latent space embeddings.
    number_of_transformer_layers : int
        Number of transformer encoder layers.
    transformer_attention_head_count : int
        Number of attention heads in transformer layers.
    transformer_dropout : float, default=0.1
        Dropout rate in transformer layers.
    learning_rate : float, default=1e-3
        Learning rate for optimization.
    verbose : bool, default=False
        Whether to print additional information.
    important_feature_index : int, default=1
        Index of the feature to be treated specially (typically degree).
    max_degree : int, default=None
        Maximum degree value for classification.
    lambda_degree_importance : float, default=1.0
        Weight factor for degree classification loss.
    noise_degree_factor : float, default=2.0
        Factor by which to reduce noise on the degree feature.
    degree_temperature : float | None, default=None
        Temperature parameter for controlling the degree prediction distribution.
    """
    def __init__(self,
                 number_of_rows_per_example: int,
                 input_feature_dimension: int,
                 condition_feature_dimension: int,
                 latent_embedding_dimension: int,
                 number_of_transformer_layers: int,
                 transformer_attention_head_count: int,
                 transformer_dropout: float = 0.1,
                 learning_rate: float = 1e-3,
                 verbose: bool = False,
                 important_feature_index: int = 1,
                 max_degree: int = None,
                 lambda_degree_importance: float = 1.0,
                 noise_degree_factor: float = 2.0,
                 degree_temperature: Optional[float] = None,
                 degree_median: float = 0.0,
                 degree_iqr: float = 1.0,
                 lambda_node_exist_importance: float = 1.0,
                 use_edge_supervision: bool = False,
                 lambda_edge_importance: float = 1.0,
                 exist_pos_weight: Union[torch.Tensor, float] = 1.0):
        super().__init__()
        self.save_hyperparameters(ignore=['verbose'])
        # Must set use_edge_supervision _before_ we refer to it below:
        self.use_edge_supervision = use_edge_supervision

        self.number_of_rows_per_example = number_of_rows_per_example
        self.input_feature_dimension = input_feature_dimension
        self.condition_feature_dimension = condition_feature_dimension
        self.latent_embedding_dimension = latent_embedding_dimension
        self.number_of_transformer_layers = number_of_transformer_layers
        self.transformer_attention_head_count = transformer_attention_head_count
        self.transformer_dropout = transformer_dropout
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.important_feature_index = important_feature_index
        self.max_degree = max_degree
        self.lambda_degree_importance = lambda_degree_importance
        self.noise_degree_factor = noise_degree_factor
        self.degree_temperature = degree_temperature
        self.lambda_node_exist_importance = lambda_node_exist_importance
        self.register_buffer(
            "exist_pos_weight",
            torch.as_tensor(exist_pos_weight, dtype=torch.float32)
        )

        if degree_iqr == 0.0:
            degree_iqr = 1.0
        self.register_buffer('deg_median', torch.tensor(degree_median, dtype=torch.float32))
        self.register_buffer('deg_iqr', torch.tensor(degree_iqr, dtype=torch.float32))

        # Initialize metric lists
        self.train_losses = []
        self.val_losses   = []
        self.train_deg_ce = []
        self.val_deg_ce   = []
        self.train_loss_all = []
        self.val_loss_all   = []
        self.train_exist    = []
        self.val_exist      = []
        if self.use_edge_supervision:
            self.train_edge_loss = []
            self.val_edge_loss   = []
            self.train_edge_acc = []
            self.val_edge_acc = []

        # Model layers
        self.layernorm_in = nn.LayerNorm(input_feature_dimension, elementwise_affine=True)
        self.linear_encoder_input_to_latent = nn.Linear(input_feature_dimension, latent_embedding_dimension)
        self.linear_encoder_condition_to_latent = nn.Linear(condition_feature_dimension, latent_embedding_dimension)
        
        # Replace transformer with cross-attention version
        self.shared_transformer = nn.ModuleList([
            CrossTransformerEncoderLayer(
                embed_dim=latent_embedding_dimension,
                num_heads=transformer_attention_head_count,
                dropout=transformer_dropout,
            ) for _ in range(number_of_transformer_layers)
        ])
        
        self.linear_decoder_latent_to_output = nn.Linear(latent_embedding_dimension, input_feature_dimension)
        self.degree_head = nn.Linear(latent_embedding_dimension, max_degree + 1)
        self.exist_head = nn.Linear(latent_embedding_dimension, 1)
        self.use_edge_supervision = use_edge_supervision
        self.lambda_edge_importance = lambda_edge_importance
        if self.use_edge_supervision:
            self.edge_head = nn.Bilinear(latent_embedding_dimension,
                                          latent_embedding_dimension,
                                          1)

    def forward(self, input_rows, global_condition_vector, diffusion_time_step, return_latents: bool = False) -> tuple:
        # Process the full input through the network
        noisy_input = self.apply_noise_schedule(input_rows, diffusion_time_step)
        x_norm = self.layernorm_in(noisy_input)
        latent_tokens = self.linear_encoder_input_to_latent(x_norm)
        
        # Create separate time and condition tokens
        time_token = get_sinusoidal_time_embedding(diffusion_time_step, self.latent_embedding_dimension)
        cond_token = self.linear_encoder_condition_to_latent(global_condition_vector)
        
        # Stack into memory sequence (B,2,D)
        mem = torch.stack([time_token, cond_token], dim=1)
        
        # Pass through transformer layers
        for layer in self.shared_transformer:
            latent_tokens = layer(latent_tokens, k=mem, v=mem)
        
        # Generate predictions from all heads
        logits_deg = self.degree_head(latent_tokens)
        logits_exist = self.exist_head(latent_tokens).squeeze(-1)  # shape (B,N)
        pred_cont = self.linear_decoder_latent_to_output(latent_tokens)
        if return_latents:
            return pred_cont, logits_deg, logits_exist, latent_tokens
        return pred_cont, logits_deg, logits_exist

    def apply_noise_schedule(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # base Gaussian noise
        noise = torch.randn_like(x)
        
        # time-dependent level for all features
        sigma_min, sigma_max = 0.1, 1.0
        sigma_all = sigma_min + t * (sigma_max - sigma_min)          # (B,1)
        sigma_all = sigma_all.unsqueeze(-1)                          # (B,1,1)
        noise_scale = torch.ones_like(x) * sigma_all                 # broadcasts to (B,N,D)

        # down-weight the important feature (degree) by noise_degree_factor
        noise_scale[..., self.important_feature_index] /= self.noise_degree_factor

        # add noise
        return x + noise * noise_scale
  
    # ---------------------------------------------------------------------------
    # single-source loss computation – returns all partials
    # ---------------------------------------------------------------------------
    def compute_weighted_loss(self, prediction: tuple, target: torch.Tensor) -> dict:
        pred_cont, logits_deg, logits_exist = prediction
        
        # Binary existence prediction
        target_exist = (target[...,0] >= 0.5).float()
        loss_exist = F.binary_cross_entropy_with_logits(
            logits_exist,
            target_exist,
            pos_weight=self.exist_pos_weight            # already on correct device
        )

        # consistency: non-existent rows should stay 0
        non_existent_mask = (target[..., 0] < 0.5)
        if non_existent_mask.any():
            loss_consistency = torch.mean(
                (pred_cont[non_existent_mask] - target[non_existent_mask]) ** 2)
        else:
            loss_consistency = torch.tensor(0.0, device=target.device)

        # degree classification loss
        deg_orig = target[..., 1] * self.deg_iqr + self.deg_median
        true_deg_class = torch.clamp(torch.round(deg_orig), 0, self.max_degree).long()
        loss_deg_ce = F.cross_entropy(logits_deg.reshape(-1, self.max_degree+1),
                                    true_deg_class.reshape(-1))

        # whole-tensor reconstruction for other features (excluding exist and degree)
        mask = torch.ones_like(pred_cont)
        mask[..., 0] = 0  # exclude existence
        mask[..., self.important_feature_index] = 0  # exclude degree
        loss_all = F.mse_loss(pred_cont * mask, target * mask, reduction='mean')

        # total loss with weighted components
        total_loss = (loss_consistency + 
                     loss_all +
                     self.lambda_node_exist_importance * loss_exist +
                     self.lambda_degree_importance * loss_deg_ce)

        return {
            "total": total_loss,
            "exist": loss_exist,
            "deg_ce": loss_deg_ce,
            "all_features": loss_all,
            "consistency": loss_consistency
        }
    # ---------------------------------------------------------------------------


    # ---------------------------------------------------------------------------
    # TRAINING STEP – uses the dict
    # ---------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        if self.use_edge_supervision:
            input_examples, global_condition, edge_idx, edge_labels, node_mask = batch
        else:
            input_examples, global_condition = batch

        diffusion_time_step = torch.rand(input_examples.size(0), 1, device=self.device)
        if self.use_edge_supervision:
            result = self.forward(input_examples, global_condition, diffusion_time_step, return_latents=True)
            pred_cont, logits_deg, logits_exist, latent_tokens = result
        else:
            pred_cont, logits_deg, logits_exist = self.forward(input_examples, global_condition, diffusion_time_step)

        losses = self.compute_weighted_loss((pred_cont, logits_deg, logits_exist), input_examples)
        loss = losses["total"]

        if self.use_edge_supervision:
            # Filter out padded nodes using node_mask (shape: B x N).
            b, i, j = edge_idx.unbind(dim=1)
            valid = node_mask[b, i] & node_mask[b, j]  # only keep valid pairs
            if valid.any():
                edge_idx = edge_idx[valid]
                edge_labels = edge_labels[valid]
                b, i, j = edge_idx.unbind(dim=1)
                zi = latent_tokens[b, i]
                zj = latent_tokens[b, j]
                logits_e = self.edge_head(zi, zj).squeeze(-1)
                loss_e = F.binary_cross_entropy_with_logits(logits_e, edge_labels)

                preds = (torch.sigmoid(logits_e) > 0.5).float()
                acc = (preds == edge_labels).float().mean()
                self.log("train_edge_loss", loss_e, on_step=False, on_epoch=True, prog_bar=True)
                self.log("train_edge_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

                loss = loss + self.lambda_edge_importance * loss_e
            # else: no valid edges in this batch, skip edge supervision loss

        # Log metrics
        self.log("train_total",     losses["total"],        on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_deg_ce",    losses["deg_ce"],       on_step=False, on_epoch=True)
        self.log("train_all",       losses["all_features"], on_step=False, on_epoch=True)
        self.log("train_exist",     losses["exist"],        on_step=False, on_epoch=True)

        return loss
    # ---------------------------------------------------------------------------


    # ---------------------------------------------------------------------------
    # VALIDATION STEP – same pattern
    # ---------------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        if self.use_edge_supervision:
            input_examples, global_condition, edge_idx, edge_labels, node_mask = batch
        else:
            input_examples, global_condition = batch

        diffusion_time_step = torch.rand(input_examples.size(0), 1, device=self.device)
        if self.use_edge_supervision:
            pred_cont, logits_deg, logits_exist, latent_tokens = self.forward(input_examples, global_condition, diffusion_time_step, return_latents=True)
        else:
            pred_cont, logits_deg, logits_exist = self.forward(input_examples, global_condition, diffusion_time_step)

        losses = self.compute_weighted_loss((pred_cont, logits_deg, logits_exist), input_examples)
        loss = losses["total"]

        if self.use_edge_supervision:
            b, i, j = edge_idx.unbind(dim=1)
            valid = node_mask[b, i] & node_mask[b, j]
            if valid.any():
                edge_idx = edge_idx[valid]
                edge_labels = edge_labels[valid]
                b, i, j = edge_idx.unbind(dim=1)
                zi = latent_tokens[b, i]
                zj = latent_tokens[b, j]
                logits_e = self.edge_head(zi, zj).squeeze(-1)
                loss_e = F.binary_cross_entropy_with_logits(logits_e, edge_labels)
                preds = (torch.sigmoid(logits_e) > 0.5).float()
                acc = (preds == edge_labels).float().mean()
                self.log("val_edge_loss", loss_e, on_step=False, on_epoch=True, prog_bar=True)
                self.log("val_edge_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
                loss = loss + self.lambda_edge_importance * loss_e
            # else: no valid edges, skip edge loss

        # Log metrics
        self.log("val_total",     losses["total"],        on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_deg_ce",    losses["deg_ce"],       on_step=False, on_epoch=True)
        self.log("val_all",       losses["all_features"], on_step=False, on_epoch=True)
        self.log("val_exist",     losses["exist"],        on_step=False, on_epoch=True)

        return loss

    def on_train_end(self):
        if not self.verbose:
            return
        plot_metrics(
            train_metrics={
                "total": self.train_losses,
                "deg_ce": self.train_deg_ce, 
                "all": self.train_loss_all,
                "exist": self.train_exist,
                **({"edge": self.train_edge_loss} if self.use_edge_supervision else {})
            },
            val_metrics={
                "total": self.val_losses,
                "deg_ce": self.val_deg_ce,
                "all": self.val_loss_all, 
                "exist": self.val_exist,
                **({"edge": self.val_edge_loss} if self.use_edge_supervision else {})
            },
            window=10,
            alpha=0.1
        )
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def generate(self, global_condition: torch.Tensor, total_diffusion_steps: int = 1000) -> torch.Tensor:
        batch_size = global_condition.size(0)
        generated = torch.randn(
            batch_size,
            self.number_of_rows_per_example,
            self.input_feature_dimension,
            device=global_condition.device
        )
        for step in range(total_diffusion_steps):
            # use (step + 1) so the very first step is 1 / total_diffusion_steps
            diffusion_time_step = torch.full(
                (batch_size, 1),
                (step + 1) / total_diffusion_steps,
                device=global_condition.device
            )
            pred_cont, logits_deg, logits_exist = self.forward(generated, global_condition, diffusion_time_step)
            
            # -----------------------------------------------------------
            # Existence column (index 0)
            # -----------------------------------------------------------
            # ▸ Keep *soft* probabilities during the iterative denoising
            #   so the model can revise its decision at later steps.
            # ▸ Make a hard 0/1 choice only in the final step
            #   (you may keep stochastic or switch to threshold 0.5).
            probs_exist = torch.sigmoid(logits_exist)            # (B,N)
            probs_exist = probs_exist.clamp(1e-4, 1-1e-4)


            if step == total_diffusion_steps - 1:
                # final step – emit a hard mask (stochastic)
                pred_exist = torch.bernoulli(probs_exist)
            else:
                # intermediate step – feed back soft probabilities
                pred_exist = probs_exist

            pred_cont[..., 0] = pred_exist

            # Handle degree predictions
            if self.degree_temperature is None:
                pred_deg = torch.argmax(logits_deg, -1).float()
            else:
                probs = torch.softmax(logits_deg / self.degree_temperature, -1)
                pred_deg = torch.distributions.Categorical(probs).sample().float()
            
            deg_scaled = (pred_deg - self.deg_median) / self.deg_iqr
            pred_cont[..., 1] = deg_scaled
            
            generated = pred_cont.detach()
        return generated

# =============================================================================
# Revised TransformerConditionalDiffusionGenerator with Piecewise Scheduling Parameters
# =============================================================================
class GraphWithEdgesDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,                    # (B, N, D)
        Y: np.ndarray,                    # (B, C)
        edge_pairs: List[Tuple[int, int, int]],
        edge_targets: np.ndarray,
        node_mask: Optional[np.ndarray] = None   # (B, N) boolean
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        B, N, _ = X.shape
        if node_mask is None:
            node_mask = np.ones((B, N), dtype=bool)
        self.node_mask = torch.tensor(node_mask, dtype=torch.bool)
        self.edge_idx_by_graph = {b: [] for b in range(len(X))}
        self.edge_lbl_by_graph = {b: [] for b in range(len(X))}
        for (b, i, j), lbl in zip(edge_pairs, edge_targets):
            self.edge_idx_by_graph[b].append((i, j))
            self.edge_lbl_by_graph[b].append(lbl)
    def __len__(self):
        return self.X.size(0)
    def __getitem__(self, idx):
        x = self.X[idx]         # (N, D)
        y = self.Y[idx]         # (C,)
        mask = self.node_mask[idx]  # (N,)
        ei = torch.tensor(self.edge_idx_by_graph[idx], dtype=torch.long) if self.edge_idx_by_graph[idx] else torch.empty((0,2), dtype=torch.long)
        el = torch.tensor(self.edge_lbl_by_graph[idx], dtype=torch.float32) if self.edge_lbl_by_graph[idx] else torch.empty((0,), dtype=torch.float32)
        return x, y, ei, el, mask

def collate_graph_with_edges(batch):
    """
    Expects each sample as (X, Y, local_edge_idx, edge_lbl, mask).
    Returns:
      X: (B, N, D)
      Y: (B, C)
      edge_idx: (E, 3) with batch-index prefixed
      edge_lbl: (E,)
      mask: (B, N)
    """
    xs, ys, masks = [], [], []
    local_edge_idxs, local_edge_lbls = [], []
    for x, y, ei, el, mask in batch:
        xs.append(x)
        ys.append(y)
        masks.append(mask)
        local_edge_idxs.append(ei)
        local_edge_lbls.append(el)
    X = torch.stack(xs)         # (B, N, D)
    Y = torch.stack(ys)         # (B, C)
    M = torch.stack(masks)      # (B, N)
    all_edge_idxs = []
    all_edge_lbls = []
    for b, (ei, el) in enumerate(zip(local_edge_idxs, local_edge_lbls)):
        if ei.numel() == 0:
            continue
        b_col = torch.full((ei.size(0), 1), b, dtype=torch.long)
        global_idx = torch.cat([b_col, ei], dim=1)  # (E_b, 3)
        all_edge_idxs.append(global_idx)
        all_edge_lbls.append(el)
    if all_edge_idxs:
        edge_idx = torch.cat(all_edge_idxs, dim=0)
        edge_lbl = torch.cat(all_edge_lbls, dim=0)
    else:
        edge_idx = torch.empty((0, 3), dtype=torch.long)
        edge_lbl = torch.empty((0,), dtype=torch.float32)
    return X, Y, edge_idx, edge_lbl, M

class ConditionalNodeGenerator:
    """
    A scikit-learn compatible diffusion generator that wraps a Transformer-based 
    diffusion model for generating structured data. This model combines a transformer 
    architecture with a diffusion process and conditional generation capabilities.

    The model is particularly suited for generating graph-like structures where each
    example consists of multiple rows (nodes/edges) and features, with special handling
    for degree features and existence flags.

    Parameters
    ----------
    latent_embedding_dimension : int, default=128
        Dimension of the latent space embeddings used throughout the transformer.
        Higher values allow for more complex node representations.
    
    number_of_transformer_layers : int, default=4
        Number of stacked transformer encoder layers. Deeper networks can model
        more complex dependencies but are harder to train.
    
    transformer_attention_head_count : int, default=4
        Number of parallel attention heads in each transformer layer. Multiple 
        heads allow the model to attend to different aspects simultaneously.
    
    transformer_dropout : float, default=0.1
        Dropout probability in transformer layers to prevent overfitting.
        Values between 0.1 and 0.3 typically work well.
    
    learning_rate : float, default=1e-3
        Learning rate for the Adam optimizer. Critical for stable training.
    
    maximum_epochs : int, default=10
        Maximum number of full passes through the training data.
    
    batch_size : int, default=32
        Number of samples per training batch. Larger batches give more stable
        gradients but require more memory.
    
    total_diffusion_steps : int, default=1000
        Number of steps in the diffusion process during generation.
        More steps give finer control but slower generation.
    
    verbose : bool, default=False
        Whether to print training progress and display metric plots.
    
    important_feature_index : int, default=1
        Index of the feature to be treated with special importance (typically degree).
        This feature receives less noise during diffusion.
    
    lambda_degree_importance : float, default=1.0
        Weight multiplier for the degree prediction loss term.
        Higher values prioritize accurate degree predictions.
    
    noise_degree_factor : float, default=2.0
        Factor by which to reduce noise on the degree feature.
        Higher values preserve degree information better during diffusion.
    
    degree_temperature : Optional[float], default=None
        Temperature for degree sampling. None means deterministic (argmax),
        while positive values enable exploration via softmax.
    
    lambda_node_exist_importance : float, default=1.0
        Weight multiplier for the node existence prediction loss term.
    
    default_exist_pos_weight : float, default=1.0
        Class weight for positive examples in node existence prediction.
        Useful for handling class imbalance.
    
    lambda_edge_importance : float, default=1.0
        Weight multiplier for the edge prediction loss term when using
        edge supervision.
    
    Methods
    -------
    fit(node_encodings_list, conditional_graph_encodings, edge_pairs=None, ...)
        Fit the model to training data, optionally with edge supervision.
    
    predict(y)
        Generate samples conditioned on the given conditional encodings.
    
    plot_metrics(window, alpha)
        Plot training metrics with geometric moving averages.
    """
    def __init__(self,
                 latent_embedding_dimension: int = 128,
                 number_of_transformer_layers: int = 4,
                 transformer_attention_head_count: int = 4,
                 transformer_dropout: float = 0.1,
                 learning_rate: float = 1e-3,
                 maximum_epochs: int = 10,
                 batch_size: int = 32,
                 total_diffusion_steps: int = 1000,
                 verbose: bool = False,
                 important_feature_index: int = 1,
                 lambda_degree_importance: float = 1.0,
                 noise_degree_factor: float = 2.0,
                 degree_temperature: Optional[float] = None,
                 lambda_node_exist_importance: float = 1.0,
                 default_exist_pos_weight: float = 1.0,
                 lambda_edge_importance: float = 1.0):
        self.latent_embedding_dimension = latent_embedding_dimension
        self.number_of_transformer_layers = number_of_transformer_layers
        self.transformer_attention_head_count = transformer_attention_head_count
        self.transformer_dropout = transformer_dropout
        self.learning_rate = learning_rate
        self.maximum_epochs = maximum_epochs
        self.batch_size = batch_size
        self.total_diffusion_steps = total_diffusion_steps
        self.verbose = verbose
        self.important_feature_index = important_feature_index
        self.lambda_degree_importance = lambda_degree_importance
        self.noise_degree_factor = noise_degree_factor
        self.degree_temperature = degree_temperature
        self.lambda_node_exist_importance = lambda_node_exist_importance
        self.default_exist_pos_weight = default_exist_pos_weight
        self.lambda_edge_importance = lambda_edge_importance

        self.number_of_rows_per_example = None
        self.input_feature_dimension = None
        self.model = None
        self.conditional_generator_estimator = None
        self.input_scaler = None
        self.D_max = None

    def _fit_scalers(self, X_array, y_array):
        B, n, d = X_array.shape
        X_reshaped = X_array.reshape(-1, d)
        self.input_scaler = CustomRobustScaler(
            special_features=[0, 1],
            exist_col=0,
            fallback_scale=1.0).fit(X_reshaped)
        # save both the original and the mapped index
        self.raw_degree_index = self.important_feature_index       # original position in raw tensor
        self.important_feature_index = self.input_scaler.map_feature_index(
            self.important_feature_index
        )

    def _transform_data(self, X_array, y_array):
        B, n, d = X_array.shape
        X_reshaped = X_array.reshape(-1, d)
        X_scaled_temp = self.input_scaler.transform(X_reshaped)
        new_d = X_scaled_temp.shape[1]
        X_scaled = X_scaled_temp.reshape(B, n, new_d)
        aggregation_factors = y_array[:, 0]
        y_scaled = self.input_scaler.transform_aggregated(y_array, aggregation_factors)
        return X_scaled, y_scaled

    def _inverse_transform_input(self, X_array):
        B, n, new_d = X_array.shape
        X_reshaped = X_array.reshape(-1, new_d)
        X_orig = self.input_scaler.inverse_transform(X_reshaped).reshape(B, n, self.input_scaler.original_dim)
        X_orig[..., self.raw_degree_index] = np.clip(
            X_orig[..., self.raw_degree_index], 0, self.D_max
        )
        return X_orig

    def fit(
        self,
        node_encodings_list: List[np.ndarray],
        conditional_graph_encodings: Any,
        edge_pairs: Optional[List[Tuple[int, int, int]]] = None,
        edge_targets: Optional[np.ndarray] = None,
        node_mask: Optional[np.ndarray] = None
    ):
        max_num_rows = max(x.shape[0] for x in node_encodings_list)
        self.number_of_rows_per_example = max_num_rows
        X_padded = []
        for x in node_encodings_list:
            n_rows = x.shape[0]
            if n_rows < max_num_rows:
                pad_width = ((0, max_num_rows - n_rows), (0, 0))
                x = np.pad(x, pad_width=pad_width, mode='constant', constant_values=0)
            X_padded.append(x)
        X_array = np.stack(X_padded, axis=0)
        y_array = np.array(conditional_graph_encodings)
        
        self._fit_scalers(X_array, y_array)

        # ------------------------------------------------------------
        # Compute class-imbalance weight for BCEWithLogitsLoss
        # ------------------------------------------------------------
        exist_mask = (X_array[..., 0] >= 0.5)
        ones  = int(exist_mask.sum())                 # rows where exist == 1
        zeros = int(exist_mask.size) - ones           # rows where exist == 0

        # BCEWithLogitsLoss multiplies the *positive* (1-class) loss by
        # pos_weight.  It should be >1 **only when 1's are the minority**.
        if ones == 0:
            exist_pos_weight = 1.0                    # avoid div-by-zero
        elif zeros > ones:                            # positives rarer
            exist_pos_weight = float(zeros) / float(ones)
        else:                                         # positives majority or equal
            exist_pos_weight = 1.0

        # Get degree scaling parameters
        deg_idx = self.important_feature_index  # mapped index
        deg_median = self.input_scaler.median_[deg_idx]
        deg_iqr = max(self.input_scaler.iqr_[deg_idx], 1e-8)
        
        X_scaled, y_scaled = self._transform_data(X_array, y_array)
        self.input_feature_dimension = X_scaled.shape[2]
        cond_feature_dim = y_scaled.shape[1]
        
        # Detect maximum degree from raw data
        raw_degrees = X_array[..., self.raw_degree_index]  # shape (B, N)
        self.D_max = int(raw_degrees.max())  # global max
        
        # Initialize the model with updated flags for edge supervision
        self.model = IterativeDenoisingAutoencoderTransformerModel(
            number_of_rows_per_example=self.number_of_rows_per_example,
            input_feature_dimension=self.input_feature_dimension,
            condition_feature_dimension=cond_feature_dim,
            latent_embedding_dimension=self.latent_embedding_dimension,
            number_of_transformer_layers=self.number_of_transformer_layers,
            transformer_attention_head_count=self.transformer_attention_head_count,
            transformer_dropout=self.transformer_dropout,
            learning_rate=self.learning_rate,
            verbose=self.verbose,
            important_feature_index=self.important_feature_index,
            max_degree=self.D_max,
            lambda_degree_importance=self.lambda_degree_importance,
            noise_degree_factor=self.noise_degree_factor,
            degree_temperature=self.degree_temperature,
            degree_median=deg_median,
            degree_iqr=deg_iqr,
            lambda_node_exist_importance=self.lambda_node_exist_importance,
            use_edge_supervision=(edge_pairs is not None),
            lambda_edge_importance=self.lambda_edge_importance,
            exist_pos_weight=exist_pos_weight,
        )

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
        if edge_pairs is not None:
            if node_mask is None:
                B, N, _ = X_scaled.shape
                node_mask_arr = np.ones((B, N), dtype=bool)
            else:
                node_mask_arr = node_mask
            dataset = GraphWithEdgesDataset(
                X_scaled,
                y_scaled,
                edge_pairs,
                edge_targets,
                node_mask_arr
            )
            
            # Split into train/val
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True,
                collate_fn=collate_graph_with_edges
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,  # No need to shuffle validation
                collate_fn=collate_graph_with_edges
            )
        else:
            dataset = TensorDataset(X_tensor, y_tensor)
            
            # Split into train/val
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        trainer = pl.Trainer(
            max_epochs=self.maximum_epochs,
            callbacks=[MetricsLogger()],
            logger=True,
            enable_checkpointing=False,
            enable_progress_bar=False
        )
        if not self.verbose:
            with suppress_output():
                trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        else:
            trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    def plot_metrics(self, window: int = 10, alpha: float = 0.3):
        if self.model is None:
            print("Model is not fitted yet.")
            return
        plot_metrics(
            train_metrics = {
                "total": self.model.train_losses,
                "deg_ce": self.model.train_deg_ce,
                "all": self.model.train_loss_all,
                "exist": self.model.train_exist,
                **({"edge": self.model.train_edge_loss} if self.model.use_edge_supervision else {})
            },
            val_metrics = {
                "total": self.model.val_losses,
                "deg_ce": self.model.val_deg_ce,
                "all": self.model.val_loss_all,
                "exist": self.model.val_exist,
                **({"edge": self.model.val_edge_loss} if self.model.use_edge_supervision else {})
            },
            window=window,
            alpha=alpha
        )
    
    def predict(self, y):
        self.model.eval()
        with torch.no_grad():
            y = np.array(y)
            aggregation_factors = y[:, 0]
            y_scaled = self.input_scaler.transform_aggregated(y, aggregation_factors)
            y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
            generated = self.model.generate(y_tensor, total_diffusion_steps=self.total_diffusion_steps)
            generated_np = generated.cpu().numpy()
            generated_orig = self._inverse_transform_input(generated_np)
            return [generated_orig[i] for i in range(generated_orig.shape[0])]

class MetricsLogger(pl.callbacks.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        pl_module.train_losses.append(m.get("train_total", torch.tensor(0.0)).item())
        pl_module.train_deg_ce.append(m.get("train_deg_ce", torch.tensor(0.0)).item())
        pl_module.train_loss_all.append(m.get("train_all", torch.tensor(0.0)).item())
        pl_module.train_exist.append(m.get("train_exist", torch.tensor(0.0)).item())
        if pl_module.use_edge_supervision:
            pl_module.train_edge_loss.append(m.get("train_edge_loss", torch.tensor(0.0)).item())
            pl_module.train_edge_acc.append(m.get("train_edge_acc", torch.tensor(0.0)).item())

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        pl_module.val_losses.append(m.get("val_total", torch.tensor(0.0)).item())
        pl_module.val_deg_ce.append(m.get("val_deg_ce", torch.tensor(0.0)).item())
        pl_module.val_loss_all.append(m.get("val_all", torch.tensor(0.0)).item())
        pl_module.val_exist.append(m.get("val_exist", torch.tensor(0.0)).item())
        if pl_module.use_edge_supervision:
            pl_module.val_edge_loss.append(m.get("val_edge_loss", torch.tensor(0.0)).item())
            pl_module.val_edge_acc.append(m.get("val_edge_acc", torch.tensor(0.0)).item())
