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

# --- Cosine Beta Schedule ---
def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(T + 1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(torch.tensor(betas, dtype=torch.float32), 1e-8, 0.999)

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
        # Self attention; key_padding_mask removed
        x = x + self.dropout1(
            self.self_attn(x, x, x, key_padding_mask=None)[0]
        )
        x = self.norm1(x)
        
        # Cross attention with key_padding_mask removed
        x = x + self.dropout2(
            self.cross_attn(x, k, v)[0]
        )
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
# Revised SharedDiffusionTransformerModel with Cross-Attention
# =============================================================================

class SharedDiffusionTransformerModel(pl.LightningModule):
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
                 lambda_edge_importance: float = 1.0,   # parameter exists in signature
                 exist_pos_weight: Union[torch.Tensor, float] = 1.0,
                 total_diffusion_steps: int = 1000,
                 exist_noise_sigma: float = 0.05,
                 lambda_x0_importance: float = 0.1):
        super().__init__()
        self.save_hyperparameters(ignore=['verbose'])
        self.lambda_x0_importance = lambda_x0_importance           # NEW
        self.lambda_edge_importance = lambda_edge_importance       # ADDED to fix AttributeError
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
        self.exist_noise_sigma = exist_noise_sigma       # ← NEW
        self.use_edge_supervision = use_edge_supervision # Fix #1: Set missing attribute

        if degree_iqr == 0.0:
            degree_iqr = 1.0
        self.register_buffer('deg_median', torch.tensor(degree_median, dtype=torch.float32))
        self.register_buffer('deg_iqr', torch.tensor(degree_iqr, dtype=torch.float32))

        # Initialize metric lists
        self.train_losses = []
        self.val_losses   = []
        self.train_deg_ce = []
        self.val_deg_ce   = []
        self.train_exist    = []
        self.val_exist      = []
        self.train_x0     = []        # NEW
        self.val_x0       = []        # NEW
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
        
        # Replace old decoder with ε̂ branch and add new x̂₀ branch:
        self.eps_head = nn.Linear(latent_embedding_dimension, input_feature_dimension)     # RENAMED ε̂ branch
        self.x0_head  = nn.Linear(latent_embedding_dimension, input_feature_dimension)     # NEW x̂₀ branch
        self.degree_head = nn.Linear(latent_embedding_dimension, max_degree + 1)
        self.exist_head = nn.Linear(latent_embedding_dimension, 1)
        if self.use_edge_supervision:
            self.edge_head = nn.Bilinear(latent_embedding_dimension,
                                          latent_embedding_dimension,
                                          1)

        # Register diffusion schedule buffers
        self.T = total_diffusion_steps
        betas = cosine_beta_schedule(self.T)                      # cosine schedule
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def q_sample(self, x0: torch.Tensor, t_idx: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        x0   : (B,N,D) clean input
        t_idx: (B,) integers in [0,T‑1]
        noise: (B,N,D) Gaussian ε
        """
        a_bar = self.alpha_bar[t_idx].view(-1, 1, 1)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise

    def forward(self, input_rows, global_condition_vector, t_norm, return_latents: bool = False):
        """
        Convert timesteps to tokens and runs through the transformer.
        Note: padding_mask is no longer used.
        """
        t_norm = t_norm.view(-1, 1)  # Ensure t_norm has shape (B,1)
        x_norm = self.layernorm_in(input_rows)
        latent_tokens = self.linear_encoder_input_to_latent(x_norm)
        
        # Create time and condition tokens using t_norm in [0,1]
        time_token = get_sinusoidal_time_embedding(t_norm, self.latent_embedding_dimension)
        cond_token = self.linear_encoder_condition_to_latent(global_condition_vector)
        mem = torch.stack([time_token, cond_token], dim=1)
        for layer in self.shared_transformer:
            latent_tokens = layer(latent_tokens, k=mem, v=mem)
        
        eps_pred   = self.eps_head(latent_tokens)            # ε̂ branch
        x0_pred    = self.x0_head(latent_tokens)             # NEW x̂₀ branch
        logits_deg   = self.degree_head(latent_tokens)
        logits_exist = self.exist_head(latent_tokens).squeeze(-1)
        if return_latents:
            return eps_pred, x0_pred, logits_deg, logits_exist, latent_tokens  # NEW return signature
        return eps_pred, x0_pred, logits_deg, logits_exist

    def training_step(self, batch, batch_idx):
        if self.use_edge_supervision:
            input_examples, global_condition, edge_idx, edge_labels, node_mask = batch
        else:
            input_examples, global_condition = batch

        B = input_examples.size(0)
        t_idx = torch.randint(1, self.T, (B,), device=self.device)  # Sample timesteps from 1…T-1
        # Replace the eps block with:
        eps = torch.randn_like(input_examples)
        # ---- existence flag: add mild Gaussian noise instead of freezing ----
        eps[..., 0] = self.exist_noise_sigma * torch.randn_like(eps[..., 0])
        # --------------------------------------------------------------------
        eps[..., self.important_feature_index] /= self.noise_degree_factor
        x_t   = self.q_sample(input_examples, t_idx, eps)
        t_norm = t_idx.float() / (self.T - 1)

        if self.use_edge_supervision:
            eps_pred, x0_pred, logits_deg, logits_exist, latent_tokens = self.forward(
                x_t, global_condition, t_norm, return_latents=True
            )
        else:
            eps_pred, x0_pred, logits_deg, logits_exist = self.forward(
                x_t, global_condition, t_norm
            )

        loss_eps = F.mse_loss(eps_pred, eps)
        loss_x0  = F.mse_loss(x0_pred, input_examples)          # NEW reconstruction loss
        target_exist = (input_examples[..., 0] >= 0.5).float()
        real_mask = (input_examples.abs().sum(dim=-1) > 0)  # a row is “real” if at least one feature is non‑zero
        loss_exist = F.binary_cross_entropy_with_logits(
            logits_exist[real_mask],        # predictions on real rows only
            target_exist[real_mask],        # both 0 and 1 labels
            pos_weight=self.exist_pos_weight
        )
        deg_orig = input_examples[..., 1] * self.deg_iqr + self.deg_median
        true_deg_class = torch.clamp(torch.round(deg_orig), 0, self.max_degree).long()
        loss_deg_ce = F.cross_entropy(
            logits_deg.reshape(-1, self.max_degree+1),
            true_deg_class.reshape(-1)
        )
        total_loss = ( loss_eps
                     + self.lambda_x0_importance * loss_x0      # updated loss aggregation
                     + self.lambda_node_exist_importance * loss_exist
                     + self.lambda_degree_importance * loss_deg_ce )

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
                self.log("train_edge_loss", loss_e, on_step=False, on_epoch=True, prog_bar=True)
                self.log("train_edge_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
                total_loss = total_loss + self.lambda_edge_importance * loss_e

        self.log("train_exist", loss_exist, on_step=False, on_epoch=True)
        self.log("train_total", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_deg_ce", loss_deg_ce, on_step=False, on_epoch=True)
        self.log("train_x0", loss_x0, on_step=False, on_epoch=True)         # NEW log
        return total_loss

    def validation_step(self, batch, batch_idx):
        if self.use_edge_supervision:
            input_examples, global_condition, edge_idx, edge_labels, node_mask = batch
        else:
            input_examples, global_condition = batch

        B = input_examples.size(0)
        t_idx = torch.randint(1, self.T, (B,), device=self.device)  # Sample timesteps from 1…T-1
        # Replace the eps block with:
        eps = torch.randn_like(input_examples)
        # ---- existence flag: add mild Gaussian noise instead of freezing ----
        eps[..., 0] = self.exist_noise_sigma * torch.randn_like(eps[..., 0])
        # --------------------------------------------------------------------
        eps[..., self.important_feature_index] /= self.noise_degree_factor
        x_t   = self.q_sample(input_examples, t_idx, eps)
        t_norm = t_idx.float() / (self.T - 1)

        if self.use_edge_supervision:
            eps_pred, x0_pred, logits_deg, logits_exist, latent_tokens = self.forward(
                x_t, global_condition, t_norm, return_latents=True
            )
        else:
            eps_pred, x0_pred, logits_deg, logits_exist = self.forward(
                x_t, global_condition, t_norm
            )

        loss_eps = F.mse_loss(eps_pred, eps)
        loss_x0  = F.mse_loss(x0_pred, input_examples)          # NEW reconstruction loss
        target_exist = (input_examples[..., 0] >= 0.5).float()
        real_mask = (input_examples.abs().sum(dim=-1) > 0)  # a row is “real” if at least one feature is non‑zero
        loss_exist = F.binary_cross_entropy_with_logits(
            logits_exist[real_mask],
            target_exist[real_mask],
            pos_weight=self.exist_pos_weight
        )
        deg_orig = input_examples[..., 1] * self.deg_iqr + self.deg_median
        true_deg_class = torch.clamp(torch.round(deg_orig), 0, self.max_degree).long()
        loss_deg_ce = F.cross_entropy(
            logits_deg.reshape(-1, self.max_degree+1),
            true_deg_class.reshape(-1)
        )
        total_loss = ( loss_eps
                     + self.lambda_x0_importance * loss_x0      # updated loss aggregation
                     + self.lambda_node_exist_importance * loss_exist
                     + self.lambda_degree_importance * loss_deg_ce )

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
                total_loss = total_loss + self.lambda_edge_importance * loss_e

        self.log("val_total", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_deg_ce", loss_deg_ce, on_step=False, on_epoch=True)
        self.log("val_exist", loss_exist, on_step=False, on_epoch=True)
        self.log("val_x0", loss_x0, on_step=False, on_epoch=True)           # NEW log
        return total_loss

    def on_train_end(self):
        if not self.verbose:
            return
        plot_metrics(
            train_metrics={
                "total": self.train_losses,
                "deg_ce": self.train_deg_ce,
                "exist": self.train_exist,      # ← add this line
                "x0": self.train_x0,           # NEW
                **({"edge": self.train_edge_loss} if self.use_edge_supervision else {})
            },
            val_metrics={
                "total": self.val_losses,
                "deg_ce": self.val_deg_ce,
                "exist": self.val_exist,        # ← add this line
                "x0": self.val_x0,             # NEW
                **({"edge": self.val_edge_loss} if self.use_edge_supervision else {})
            },
            window=10,
            alpha=0.1
        )
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def generate(self, global_condition: torch.Tensor, total_steps: Union[int, Sequence[int]] = 50) -> torch.Tensor:
        device = global_condition.device
        B = global_condition.size(0)
        x_t = torch.randn(B, self.number_of_rows_per_example, self.input_feature_dimension, device=device)
        if isinstance(total_steps, int):
            t_schedule = np.linspace(self.T-1, 0, total_steps, dtype=int)
        else:
            t_schedule = np.asarray(total_steps, dtype=int)
        for t in t_schedule:
            t_idx = torch.full((B,), t, device=device, dtype=torch.long)
            t_norm = t_idx.float() / (self.T - 1)
            # ------------------------------------------------------------------
            # use ε̂ (eps_pred) for the DDPM mean‑variance formula, *not* logits
            # ------------------------------------------------------------------
            if self.use_edge_supervision:
                eps_pred, _, logits_deg, logits_exist, _ = self.forward(
                    x_t, global_condition, t_norm, return_latents=True
                )
            else:
                eps_pred, _, logits_deg, logits_exist = self.forward(
                    x_t, global_condition, t_norm
                )
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bar[t]
            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)
            mu = coef1 * (x_t - coef2 * eps_pred)
            if t > 0:
                sigma = torch.sqrt(beta_t)
                z = torch.randn_like(x_t)
                x_t = mu + sigma * z
            else:
                x_t = mu
        t_idx = torch.zeros((B,), device=device, dtype=torch.long)
        t_norm = t_idx.float() / (self.T - 1)
        if self.use_edge_supervision:
            _, _, logits_deg, logits_exist, _ = self.forward(x_t, global_condition, t_norm, return_latents=True)
        else:
            _, _, logits_deg, logits_exist = self.forward(x_t, global_condition, t_norm)
        prob_exist = torch.sigmoid(logits_exist)
        x_t[..., 0] = torch.bernoulli(prob_exist)
        deg_class = torch.softmax(logits_deg, dim=-1).argmax(-1).float()
        x_t[..., 1] = (deg_class - self.deg_median) / self.deg_iqr
        return x_t

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
    A generative model combining Transformers and diffusion for structured data,
    particularly suited for graph-like structures. It supports conditional
    generation and edge supervision.

    Parameters
    ----------
    latent_embedding_dimension : int, default=128
        Dimension of latent embeddings.
    number_of_transformer_layers : int, default=4
        Number of Transformer layers.
    transformer_attention_head_count : int, default=4
        Number of attention heads in Transformers.
    transformer_dropout : float, default=0.1
        Dropout rate in Transformer layers.
    learning_rate : float, default=1e-3
        Learning rate for Adam optimizer.
    maximum_epochs : int, default=10
        Maximum training epochs.
    batch_size : int, default=32
        Batch size for training.
    total_diffusion_steps : int, default=1000
        Number of diffusion steps.
    verbose : bool, default=False
        Enable verbose logging.
    important_feature_index : int, default=1
        Index of the important feature (e.g., degree).
    lambda_degree_importance : float, default=1.0
        Weight for degree prediction loss.
    noise_degree_factor : float, default=2.0
        Factor to reduce noise on the degree feature.
    degree_temperature : Optional[float], default=None
        Temperature for degree sampling.
    lambda_node_exist_importance : float, default=1.0
        Weight for node existence loss.
    default_exist_pos_weight : float, default=1.0
        Positive class weight for node existence.
    lambda_edge_importance : float, default=1.0
        Weight for edge prediction loss (if edge supervision is used).
    lambda_x0_importance : float, default=0.1
        Weight for the x0 reconstruction loss.

    Methods
    -------
    fit(node_encodings_list, conditional_graph_encodings, edge_pairs=None, edge_targets=None, node_mask=None)
        Fit the model to training data.
    predict(y)
        Generate samples conditioned on y.
    plot_metrics(window=10, alpha=0.3)
        Plot training metrics.
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
                 lambda_edge_importance: float = 1.0,
                 lambda_x0_importance: float = 0.1):  # NEW: expose lambda_x0_importance
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
        self.lambda_x0_importance = lambda_x0_importance  # NEW

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
        # Compute pos_weight for BCEWithLogitsLoss ignoring structural pads
        flat   = X_array.reshape(-1, X_array.shape[-1])        # (B·N, D)
        real_rows_flat = (flat.any(axis=1))                    # True for non‑pad rows
        exist_flat     = (flat[:, 0] >= 0.5) & real_rows_flat  # real & exist==1
        nonexist_flat  = (~exist_flat) & real_rows_flat        # real & exist==0

        ones  = int(exist_flat.sum())
        zeros = int(nonexist_flat.sum())
        exist_pos_weight = 1.0 if zeros <= ones else zeros / (ones + 1e-6)
        # ------------------------------------------------------------
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
        self.model = SharedDiffusionTransformerModel(
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
            total_diffusion_steps=self.total_diffusion_steps,
            lambda_x0_importance=self.lambda_x0_importance   # Fix #2: forward hyper-parameter
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
                "exist":  self.model.train_exist,  # ← add back
                "x0": self.model.train_x0,        # NEW
                **({"edge": self.model.train_edge_loss} if self.model.use_edge_supervision else {})
            },
            val_metrics = {
                "total": self.model.val_losses,
                "deg_ce": self.model.val_deg_ce,
                "exist":  self.model.val_exist,    # ← add back
                "x0": self.model.val_x0,          # NEW
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
            y_tensor = torch.tensor(y_scaled, dtype=torch.float32, device=self.model.device)
            generated = self.model.generate(y_tensor, total_steps=list(range(self.model.T-1, -1, -1)))
            generated_np = generated.cpu().numpy()
            generated_orig = self._inverse_transform_input(generated_np)
            return [generated_orig[i] for i in range(generated_orig.shape[0])]

class MetricsLogger(pl.callbacks.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        pl_module.train_losses.append(m.get("train_total", torch.tensor(0.0)).item())
        pl_module.train_deg_ce.append(m.get("train_deg_ce", torch.tensor(0.0)).item())
        pl_module.train_exist.append(m.get("train_exist", torch.tensor(0.0)).item())
        pl_module.train_x0.append(m.get("train_x0", torch.tensor(0.0)).item())  # NEW
        if pl_module.use_edge_supervision:
            pl_module.train_edge_loss.append(m.get("train_edge_loss", torch.tensor(0.0)).item())
            pl_module.train_edge_acc.append(m.get("train_edge_acc", torch.tensor(0.0)).item())

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        pl_module.val_losses.append(m.get("val_total", torch.tensor(0.0)).item())
        pl_module.val_deg_ce.append(m.get("val_deg_ce", torch.tensor(0.0)).item())
        pl_module.val_exist.append(m.get("val_exist", torch.tensor(0.0)).item())
        pl_module.val_x0.append(m.get("val_x0", torch.tensor(0.0)).item())  # NEW
        if pl_module.use_edge_supervision:
            pl_module.val_edge_loss.append(m.get("val_edge_loss", torch.tensor(0.0)).item())
            pl_module.val_edge_acc.append(m.get("val_edge_acc", torch.tensor(0.0)).item())
