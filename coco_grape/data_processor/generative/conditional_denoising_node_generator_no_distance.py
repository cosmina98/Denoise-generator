import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import contextlib, os, sys
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import random_split, DataLoader, TensorDataset, Dataset
from typing import Dict, Sequence, Optional, Union, Tuple, List, Any
import math
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
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
    """
    A transformer encoder layer with self-attention and cross-attention mechanisms.
    
    This layer implements a pre-norm architecture with three main components:
    1. Self-attention: allows nodes to attend to each other
    2. Cross-attention: allows nodes to attend to condition tokens
    3. Feed-forward network: processes each node's features independently
    
    Each component is followed by residual connection, dropout, and layer normalization.
    
    Args:
        embed_dim: Dimension of input and output embeddings
        num_heads: Number of attention heads for both self and cross attention
        dropout: Dropout probability for attention and feed-forward layers (default: 0.1)
    
    Notes:
        - Uses pre-normalization for better training stability
        - Feed-forward network expands features by 4x then projects back
        - All attention modules use scaled dot-product attention
    """
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1):
        super().__init__()
        # Self attention block
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross attention block with memory tokens
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed-forward block with increased width
        self.norm3 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Process input sequence with self-attention and cross-attention.

        Args:
            x: Input tensor of shape (B, N, D) where:
               B = batch size
               N = sequence length (number of nodes)
               D = embedding dimension
            k: Key tensor for cross-attention (B, M, D) where:
               M = number of memory/condition tokens
            v: Value tensor for cross-attention (B, M, D)

        Returns:
            Output tensor of shape (B, N, D) with processed node features
        """
        # Pre-norm architecture (more stable training)
        # Self attention
        x1 = self.norm1(x)
        x = x + self.dropout1(self.self_attn(x1, x1, x1)[0])
        
        # Cross attention with memory tokens
        x2 = self.norm2(x) 
        x = x + self.dropout2(self.cross_attn(x2, k, v)[0])
        
        # Feed-forward
        x3 = self.norm3(x)
        x = x + self.dropout3(self.ff(x3))
        return x

# --- Plotting Metrics ---

def plot_metrics(
    train_metrics: Dict[str, Sequence[float]],
    val_metrics: Dict[str, Sequence[float]],
    window: int = 10,
    alpha: float = 0.35
) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle

    # Okabe–Ito palette (color-blind friendly)
    OKABE_ITO = [
        "#0072B2",  # blue
        "#D55E00",  # vermillion (orange-red)
        "#009E73",  # bluish green
        "#CC79A7",  # reddish purple
        "#56B4E9",  # sky blue
        "#E69F00",  # orange
        "#000000",  # black
        "#F0E442",  # yellow (bright; still distinguishable with thicker lines)
    ]

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

    # one y-axis per metric
    axes = [ax0] + [ax0.twinx() for _ in range(len(metrics) - 1)]
    for i, ax in enumerate(axes[1:], start=1):
        ax.spines['right'].set_position(('outward', 60 * i))

    # cycle colors if there are more metrics than palette colors
    color_cycler = cycle(OKABE_ITO)
    colors = [next(color_cycler) for _ in metrics]

    lines, labels = [], []
    for name, ax, color in zip(metrics, axes, colors):
        train_vals = train_metrics[name]
        val_vals   = val_metrics[name]
        if len(train_vals) < 1 or len(val_vals) < 1:
            continue

        N = min(len(train_vals), len(val_vals))
        train = train_vals[:N]
        val   = val_vals[:N]
        epochs = np.arange(1, N + 1)

        # raw
        ax.plot(epochs, train, color=color, alpha=alpha, linewidth=1.5)
        ax.plot(epochs, val,   color=color, linestyle='--', alpha=alpha, linewidth=1.5)

        # smoothed (thicker, easier to see)
        sm_train = _moving_average(train, window)
        sm_val   = _moving_average(val, window)
        if sm_train.size:
            sm_epochs = np.arange(window, window + len(sm_train))
            l1, = ax.plot(sm_epochs, sm_train, color=color, linewidth=2.5,
                          label=f"Train {name} (MA{window})", zorder=3)
            l2, = ax.plot(sm_epochs, sm_val,   color=color, linewidth=2.5, linestyle='--',
                          label=f"Val {name} (MA{window})", zorder=3)
            lines += [l1, l2]
            labels += [f"Train {name} (MA{window})", f"Val {name} (MA{window})"]

        ax.set_ylabel(name, color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_yscale('log')

    fig.legend(lines, labels, loc='upper center', ncol=max(2, len(lines)//2), fontsize='small')
    ax0.set_xlabel("Epoch")
    ax0.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()



# =============================================================================
# Revised IterativeDenoisingAutoencoderTransformerModel with Cross-Attention
# =============================================================================
class GuidanceMLP(nn.Module):
    """
    Two-hidden-layer MLP for classifier guidance.

    Args
    ----
    input_dim  : int  – dimension of pooled transformer latents
    hidden_dim : int  – width of *both* hidden layers
    output_dim : int  – number of classes
    dropout    : float, default 0.2 – drop probability after each activation
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.2
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,  hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class EdgeMLP(nn.Module):
    """
    Simple one-hidden-layer MLP edge predictor.
    Combines pairwise node features and learns a nonlinear mapping to edge logits.
    """
    def __init__(self, latent_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 2 * latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(4 * latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        """
        Compute edge logits given node embeddings.
        h_i, h_j: (E, D)
        Returns logits (E,) for BCEWithLogitsLoss.
        """
        diff = torch.abs(h_i - h_j)
        prod = h_i * h_j
        x = torch.cat([h_i, h_j, diff, prod], dim=-1)
        return self.mlp(x).squeeze(-1)
    


    


class EdgeLabelMLP(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 2 * latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(4 * latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(h_i - h_j)
        prod = h_i * h_j
        x = torch.cat([h_i, h_j, diff, prod], dim=-1)
        return self.mlp(x)  # (E, C)
    

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
    sigma_min : float, default=0.1
        Lower bound of the diffusion noise schedule encountered during training.
    sigma_max : float, default=1.0
        Upper bound of the diffusion noise schedule encountered during training.
    sampling_final_sigma : float, default=0.0
        Final noise level used during deterministic sampling; must not exceed sigma_max.
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
                 max_degree: Optional[int] = None,
                 lambda_degree_importance: float = 1.0,
                 noise_degree_factor: float = 2.0,
                 degree_temperature: Optional[float] = None,
                 degree_min_val: float = 0.0, # Changed from degree_median
                 degree_range_val: float = 1.0, # Changed from degree_iqr
                 lambda_node_exist_importance: float = 1.0,
                 use_edge_supervision: bool = False,
                 lambda_edge_importance: float = 1.0,
                 exist_pos_weight: Union[torch.Tensor, float] = 1.0,
                 use_guidance: bool = False,
                 guidance_weight: float = 1.0,
                 sigma_min: float = 0.1,
                 sigma_max: float = 1.0,
                 sampling_final_sigma: float = 0.0,
                 lambda_consistency: float = 0.3,
                 label_feature_index: Optional[int] = 2,
                 max_label: Optional[int] = None,
                 lambda_label_importance: float = 1.0,
                 noise_label_factor: float = 2.0,
                 label_min_val: float = 0.0,
                 label_range_val: float = 1.0,
                 use_edge_label_supervision: bool = False,
                max_edge_label: Optional[int] = None,
                lambda_edge_label_importance: float = 1.0
                 
                 
                 
                 
                 ):
        

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
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.lambda_consistency = lambda_consistency

        if not self.sigma_min < self.sigma_max:
            raise ValueError(f"sigma_min must be < sigma_max (got {self.sigma_min} >= {self.sigma_max})")
        self.sampling_final_sigma = float(sampling_final_sigma)
        if self.sampling_final_sigma < 0:
            raise ValueError(f"sampling_final_sigma must be non-negative (got {self.sampling_final_sigma})")
        if self.sampling_final_sigma > self.sigma_max:
            raise ValueError(
                f"sampling_final_sigma must be <= sigma_max "
                f"(got {self.sampling_final_sigma} > {self.sigma_max})"
            )
        self.verbose = verbose
        self.important_feature_index = important_feature_index
        self.max_degree = max_degree
        if self.max_degree is None:
            raise ValueError("max_degree must be provided when initializing the diffusion model.")
        self.lambda_degree_importance = lambda_degree_importance
        self.noise_degree_factor = noise_degree_factor
        self.degree_temperature = degree_temperature
        self.lambda_node_exist_importance = lambda_node_exist_importance
        self.register_buffer(
            "exist_pos_weight",
            torch.as_tensor(exist_pos_weight, dtype=torch.float32)
        )

        # ----------  guidance flags ----------
        self.use_guidance = use_guidance
        self.guidance_weight = guidance_weight
        self.guidance_classifier: Optional[GuidanceMLP] = None

        # Store degree scaling parameters (MinMaxScaler based)
        self.register_buffer('deg_min_val', torch.tensor(degree_min_val, dtype=torch.float32))
        # Ensure range is not zero to avoid division by zero in scaling
        self.register_buffer('deg_range_val', torch.tensor(max(degree_range_val, 1e-8), dtype=torch.float32))
                # store label scaling (like degree)
        self.register_buffer('lab_min_val',   torch.tensor(label_min_val, dtype=torch.float32))
        self.register_buffer('lab_range_val', torch.tensor(max(label_range_val, 1e-8), dtype=torch.float32))


        # Backward compatibility for checkpoints created prior to schedule metadata
        self._ensure_schedule_metadata()

        # Initialize metric lists
        self.train_losses = []
        self.val_losses   = []
        self.train_deg_ce = []
        self.val_deg_ce   = []
        self.train_loss_all = []
        self.val_loss_all   = []
        self.train_exist    = []
        self.val_exist      = []
        self.train_recon = []
        self.val_recon = []
        self.train_label_ce = []
        self.val_label_ce = []
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
        self.max_label = max_label
        self.lambda_label_importance = lambda_label_importance
        self.noise_label_factor = noise_label_factor
        self.label_feature_index = label_feature_index
        self.use_edge_label_supervision = use_edge_label_supervision
        self.max_edge_label = max_edge_label
        self.lambda_edge_label_importance = lambda_edge_label_importance
        
        if self.use_edge_supervision:
            self.edge_head = EdgeMLP(
                latent_dim=latent_embedding_dimension,
                hidden_dim=2 * latent_embedding_dimension,
                dropout=transformer_dropout
            )

        if self.use_edge_label_supervision and (self.max_edge_label is not None) and (self.max_edge_label >= 1):
            self.edge_label_head = EdgeLabelMLP(
                latent_dim=latent_embedding_dimension,
                num_classes=self.max_edge_label + 1,
                hidden_dim=2 * latent_embedding_dimension,
                dropout=transformer_dropout
            )
        else:
            self.edge_label_head = None




        if self.use_edge_supervision:
            self.train_edge_loss = []; self.val_edge_loss = []; self.train_edge_acc = []; self.val_edge_acc = []

        if self.use_edge_label_supervision:
            self.train_edge_label_loss = []; self.val_edge_label_loss = []
            self.train_edge_label_acc  = []; self.val_edge_label_acc  = []


           


        # heads

        if self.max_label is not None:
            self.label_head = nn.Linear(latent_embedding_dimension, self.max_label + 1)  # <-- NEW
        else:
            self.label_head = None

    def forward(
        self,
        input_rows: torch.Tensor,
        global_condition_vector: torch.Tensor,
        diffusion_time_step: torch.Tensor,
        return_latents: bool = False,
        add_noise: bool = True,
    ):
        """
        Forward pass: predict ε (noise) from a noisy input x_t.

        Args:
            input_rows: Clean inputs when `add_noise=True`, otherwise the already-noisy x_t.
            global_condition_vector: Conditioning tokens (B, C).
            diffusion_time_step: Normalized time steps t ∈ [0, 1].
            return_latents: When True, also return intermediate latent tokens.
            add_noise: If True, sample fresh noise via the training schedule; if False,
                assume `input_rows` already contains x_t and skip additional perturbation.
        """
        self._ensure_schedule_metadata()
        if add_noise:
            noisy_input, eps, sigma_t = self.apply_noise_schedule(input_rows, diffusion_time_step)
        else:
            noisy_input = input_rows
            eps = None
            sigma_t = None

        x_norm = self.layernorm_in(noisy_input)
        latent_tokens = self.linear_encoder_input_to_latent(x_norm)

        # Build memory (time + condition)
        time_token = get_sinusoidal_time_embedding(diffusion_time_step, self.latent_embedding_dimension)
        cond_token = self.linear_encoder_condition_to_latent(global_condition_vector)
        mem = torch.stack([time_token, cond_token], dim=1)  # (B,2,D)

        # Transformer with cross-attention
        for layer in self.shared_transformer:
            latent_tokens = layer(latent_tokens, k=mem, v=mem)

        # Predict ε for continuous features
        pred_eps = self.linear_decoder_latent_to_output(latent_tokens)

        # Other prediction heads remain the same
        # Build per-feature noise scale from *time* (works for train + sampling)
        sigma_from_t = self._sigma_from_t(diffusion_time_step).unsqueeze(-1)  # (B,1,1)
        noise_scale = torch.ones_like(noisy_input) * sigma_from_t
        noise_scale[..., self.important_feature_index] /= self.noise_degree_factor

        # x0 estimate; detach pred_eps in this path to prevent trivial shortcuts
        x0_hat = noisy_input - noise_scale * pred_eps.detach()

        # Re-encode x0_hat for cleaner head inputs (cheap: same encoder; no extra transformer pass)
        x0_latents = self.linear_encoder_input_to_latent(self.layernorm_in(x0_hat))

        logits_deg   = self.degree_head(x0_latents)
        logits_exist = self.exist_head(x0_latents).squeeze(-1)
        logits_lab   = self.label_head(x0_latents) if self.label_head is not None else None  # <-- NEW


        if return_latents:
            return pred_eps, logits_deg, logits_exist, logits_lab, latent_tokens, eps, sigma_t
        return pred_eps, logits_deg, logits_exist, logits_lab, eps, sigma_t



    def _sigma_from_t(self, t: torch.Tensor) -> torch.Tensor:
        """Convert normalized time steps to per-feature noise scales."""
        return self.sigma_min + t * (self.sigma_max - self.sigma_min)

    def _t_from_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        """Map noise scales back to the normalized diffusion time domain."""
        return ((sigma - self.sigma_min) / (self.sigma_max - self.sigma_min)).clamp(0.0, 1.0)

    def _ensure_schedule_metadata(self) -> None:
        """Ensure schedule attributes exist for backward compatibility."""
        if not hasattr(self, "sigma_min"):
            self.sigma_min = 0.1
        if not hasattr(self, "sigma_max"):
            self.sigma_max = 1.0
        if not hasattr(self, "sampling_final_sigma"):
            self.sampling_final_sigma = 0.0

    def apply_noise_schedule(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply a time-dependent Gaussian noise schedule.

        Returns:
            x_t      : Noisy version of x_0
            eps      : The actual Gaussian noise added
            sigma_t  : The scalar noise level used at each step (B,1,1)
        """
        self._ensure_schedule_metadata()
        # base Gaussian noise
        eps = torch.randn_like(x)

        sigma_t = self._sigma_from_t(t)    # (B,1)
        sigma_t = sigma_t.unsqueeze(-1)                      # (B,1,1)

        # featurewise scaling (reduce noise on degree column)
        noise_scale = torch.ones_like(x) * sigma_t
        noise_scale[..., self.important_feature_index] /= self.noise_degree_factor
        if self.label_feature_index is not None:
            noise_scale[..., self.label_feature_index] /= self.noise_label_factor  # <-- NEW


        x_t = x + eps * noise_scale
        return x_t, eps, sigma_t

  
    # ---------------------------------------------------------------------------
    # single-source loss computation – returns all partials
        # ---------------------------------------------------------------------------
    def compute_weighted_loss(self, prediction, target) -> dict:
        # unpack
        pred_eps, logits_deg, logits_exist, logits_lab, eps, sigma_t = prediction

        B, N, D = pred_eps.shape
        mask = torch.ones((1, 1, D), device=pred_eps.device, dtype=pred_eps.dtype)
        mask[..., 0] = 0  # existence
        mask[..., self.important_feature_index] = 0  # degree
        if self.label_feature_index is not None:
            mask[..., self.label_feature_index] = 0  # <-- exclude label from ε
        loss_eps = F.mse_loss(pred_eps * mask, eps * mask, reduction='mean')

        # consistency loss (also exclude label channel in cont_mask)
        lam_cons = getattr(self, "lambda_consistency", 0.0)
        if lam_cons > 0:
            cont_mask = torch.ones_like(target)
            cont_mask[..., 0] = 0
            cont_mask[..., self.important_feature_index] = 0
            if self.label_feature_index is not None:
                cont_mask[..., self.label_feature_index] = 0
            absent = (target[..., 0] < 0.5).float().unsqueeze(-1)
            noise_scale = torch.ones_like(target) * sigma_t
            noise_scale[..., self.important_feature_index] /= self.noise_degree_factor
            if self.label_feature_index is not None:
                noise_scale[..., self.label_feature_index] /= self.noise_label_factor
            x_t    = target + noise_scale * eps
            x0_hat = x_t - noise_scale * pred_eps
            loss_cons = ((x0_hat * cont_mask)**2 * absent).mean()
        else:
            loss_cons = pred_eps.new_zeros(())

        # targets
        target_exist = (target[..., 0] >= 0.5).float()

        # degree classes
        deg_unscaled = target[..., self.important_feature_index] * self.deg_range_val + self.deg_min_val
        true_deg_class = torch.clamp(torch.round(deg_unscaled), 0, self.max_degree).long()

        # label classes (like degree)
        if (self.label_head is not None) and (self.label_feature_index is not None):
            lab_unscaled = target[..., self.label_feature_index] * self.lab_range_val + self.lab_min_val
            true_lab_class = torch.clamp(torch.round(lab_unscaled), 0, self.max_label).long()
        else:
            true_lab_class = None

        # t-weights
        w = 1.0 / (sigma_t.squeeze(-1).squeeze(-1) ** 2 + 1e-4)

        # existence loss
        exist_elem = F.binary_cross_entropy_with_logits(
            logits_exist, target_exist, pos_weight=self.exist_pos_weight, reduction='none'
        )
        loss_exist = (exist_elem * w.view(-1, 1)).mean()

        # degree CE
        Bn, Nn, C = logits_deg.shape
        deg_ce_elem = F.cross_entropy(
            logits_deg.view(Bn * Nn, C),
            true_deg_class.view(Bn * Nn),
            reduction='none'
        ).view(Bn, Nn)
        loss_deg_ce = (deg_ce_elem * w.view(-1, 1)).mean()

        # label CE (NEW)
        if (self.label_head is not None) and (true_lab_class is not None):
            Bl, Nl, Cl = logits_lab.shape
            lab_ce_elem = F.cross_entropy(
                logits_lab.view(Bl * Nl, Cl),
                true_lab_class.view(Bl * Nl),
                reduction='none'
            ).view(Bl, Nl)
            loss_lab_ce = (lab_ce_elem * w.view(-1, 1)).mean()
        else:
            loss_lab_ce = pred_eps.new_zeros(())

        total_loss = (
            loss_eps
            + self.lambda_node_exist_importance * loss_exist
            + self.lambda_degree_importance     * loss_deg_ce
            + self.lambda_label_importance      * loss_lab_ce   # <-- NEW
            + lam_cons * loss_cons
        )
        return {"total": total_loss, "eps": loss_eps, "exist": loss_exist, "deg_ce": loss_deg_ce, "lab_ce": loss_lab_ce}

    # ---------------------------------------------------------------------------


    # ---------------------------------------------------------------------------
    # TRAINING STEP – uses the dict
        # ---------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        """
        Perform a single training step with optional edge and edge-label supervision.
        """
        if self.use_edge_supervision:
            # collate_graph_with_edges returns 7 items; keep compatibility if a 5-item batch sneaks in
            if len(batch) == 7:
                input_examples, global_condition, edge_idx, edge_labels, node_mask, edge_cls_idx, edge_cls_lbl = batch
            else:
                input_examples, global_condition, edge_idx, edge_labels, node_mask = batch
                edge_cls_idx = torch.empty((0, 3), dtype=torch.long, device=input_examples.device)
                edge_cls_lbl = torch.empty((0,),   dtype=torch.long, device=input_examples.device)
        else:
            input_examples, global_condition = batch
            edge_idx    = torch.empty((0, 3), dtype=torch.long, device=input_examples.device)
            edge_labels = torch.empty((0,),   dtype=torch.float32, device=input_examples.device)
            node_mask   = torch.ones(input_examples.shape[:2], dtype=torch.bool, device=input_examples.device)
            edge_cls_idx = torch.empty((0, 3), dtype=torch.long, device=input_examples.device)
            edge_cls_lbl = torch.empty((0,),   dtype=torch.long,  device=input_examples.device)

        diffusion_time_step = torch.rand(input_examples.size(0), 1, device=self.device)

        # forward
        if self.use_edge_supervision:
            pred_eps, logits_deg, logits_exist, logits_lab, latent_tokens, eps, sigma_t = \
                self.forward(input_examples, global_condition, diffusion_time_step, return_latents=True)
        else:
            pred_eps, logits_deg, logits_exist, logits_lab, eps, sigma_t = \
                self.forward(input_examples, global_condition, diffusion_time_step)

        # core diffusion + node heads
        losses = self.compute_weighted_loss((pred_eps, logits_deg, logits_exist, logits_lab, eps, sigma_t), input_examples)
        total_loss = losses["total"]

        # ------- edge existence (binary) -------
        if self.use_edge_supervision and edge_idx.numel() > 0:
            b, i, j = edge_idx.unbind(1)
            valid = node_mask[b, i] & node_mask[b, j]
            if valid.any():
                b, i, j = b[valid], i[valid], j[valid]
                h_i = latent_tokens[b, i]
                h_j = latent_tokens[b, j]
                edge_logits = self.edge_head(h_i, h_j)
                edge_loss = F.binary_cross_entropy_with_logits(edge_logits, edge_labels[valid])
                total_loss = total_loss + self.lambda_edge_importance * edge_loss

                with torch.no_grad():
                    edge_pred = (torch.sigmoid(edge_logits) > 0.5).float()
                    edge_acc = (edge_pred == edge_labels[valid]).float().mean()

                self.log("train_edge_loss", edge_loss, on_step=False, on_epoch=True)
                self.log("train_edge_acc",  edge_acc,  on_step=False, on_epoch=True)

        # ------- edge label (multi-class) -------
        if (getattr(self, "edge_label_head", None) is not None) and (edge_cls_idx.numel() > 0):
            b2, i2, j2 = edge_cls_idx.unbind(1)
            valid2 = node_mask[b2, i2] & node_mask[b2, j2]
            if valid2.any():
                b2, i2, j2 = b2[valid2], i2[valid2], j2[valid2]
                h_i2 = latent_tokens[b2, i2]
                h_j2 = latent_tokens[b2, j2]
                logits_edge_lab = self.edge_label_head(h_i2, h_j2)         # (E, C)
                loss_edge_lab = F.cross_entropy(logits_edge_lab, edge_cls_lbl[valid2].long())
                total_loss = total_loss + self.lambda_edge_label_importance * loss_edge_lab

                with torch.no_grad():
                    acc_edge_lab = (logits_edge_lab.argmax(-1) == edge_cls_lbl[valid2].long()).float().mean()

                self.log("train_edge_label_loss", loss_edge_lab, on_step=False, on_epoch=True)
                self.log("train_edge_label_acc",  acc_edge_lab,  on_step=False, on_epoch=True)

        # logging
        self.log("train_total", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_recon", losses["eps"],    on_step=False, on_epoch=True)
        self.log("train_deg_ce", losses["deg_ce"],on_step=False, on_epoch=True)
        self.log("train_exist",  losses["exist"], on_step=False, on_epoch=True)
        self.log("train_label_ce", losses["lab_ce"], on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation with optional edge and edge-label supervision.
        """
        if self.use_edge_supervision:
            if len(batch) == 7:
                input_examples, global_condition, edge_idx, edge_labels, node_mask, edge_cls_idx, edge_cls_lbl = batch
            else:
                input_examples, global_condition, edge_idx, edge_labels, node_mask = batch
                edge_cls_idx = torch.empty((0, 3), dtype=torch.long, device=input_examples.device)
                edge_cls_lbl = torch.empty((0,),   dtype=torch.long, device=input_examples.device)
        else:
            input_examples, global_condition = batch
            edge_idx    = torch.empty((0, 3), dtype=torch.long, device=input_examples.device)
            edge_labels = torch.empty((0,),   dtype=torch.float32, device=input_examples.device)
            node_mask   = torch.ones(input_examples.shape[:2], dtype=torch.bool, device=input_examples.device)
            edge_cls_idx = torch.empty((0, 3), dtype=torch.long, device=input_examples.device)
            edge_cls_lbl = torch.empty((0,),   dtype=torch.long,  device=input_examples.device)

        diffusion_time_step = torch.rand(input_examples.size(0), 1, device=self.device)

        if self.use_edge_supervision:
            pred_eps, logits_deg, logits_exist, logits_lab, latent_tokens, eps, sigma_t = \
                self.forward(input_examples, global_condition, diffusion_time_step, return_latents=True)
        else:
            pred_eps, logits_deg, logits_exist, logits_lab, eps, sigma_t = \
                self.forward(input_examples, global_condition, diffusion_time_step)

        losses = self.compute_weighted_loss((pred_eps, logits_deg, logits_exist, logits_lab, eps, sigma_t), input_examples)
        total_loss = losses["total"]

        # edge existence
        if self.use_edge_supervision and edge_idx.numel() > 0:
            b, i, j = edge_idx.unbind(1)
            valid = node_mask[b, i] & node_mask[b, j]
            if valid.any():
                b, i, j = b[valid], i[valid], j[valid]
                h_i = latent_tokens[b, i]
                h_j = latent_tokens[b, j]
                edge_logits = self.edge_head(h_i, h_j)
                edge_loss = F.binary_cross_entropy_with_logits(edge_logits, edge_labels[valid])
                total_loss = total_loss + self.lambda_edge_importance * edge_loss

                with torch.no_grad():
                    edge_pred = (torch.sigmoid(edge_logits) > 0.5).float()
                    edge_acc = (edge_pred == edge_labels[valid]).float().mean()

                self.log("val_edge_loss", edge_loss, on_step=False, on_epoch=True)
                self.log("val_edge_acc",  edge_acc,  on_step=False, on_epoch=True)

        # edge label (multi-class)
        if (getattr(self, "edge_label_head", None) is not None) and (edge_cls_idx.numel() > 0):
            b2, i2, j2 = edge_cls_idx.unbind(1)
            valid2 = node_mask[b2, i2] & node_mask[b2, j2]
            if valid2.any():
                b2, i2, j2 = b2[valid2], i2[valid2], j2[valid2]
                h_i2 = latent_tokens[b2, i2]
                h_j2 = latent_tokens[b2, j2]
                logits_edge_lab = self.edge_label_head(h_i2, h_j2)
                loss_edge_lab = F.cross_entropy(logits_edge_lab, edge_cls_lbl[valid2].long())
                total_loss = total_loss + self.lambda_edge_label_importance * loss_edge_lab

                with torch.no_grad():
                    acc_edge_lab = (logits_edge_lab.argmax(-1) == edge_cls_lbl[valid2].long()).float().mean()

                self.log("val_edge_label_loss", loss_edge_lab, on_step=False, on_epoch=True)
                self.log("val_edge_label_acc",  acc_edge_lab,  on_step=False, on_epoch=True)

        self.log("val_total", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_recon", losses["eps"],    on_step=False, on_epoch=True)
        self.log("val_deg_ce", losses["deg_ce"],on_step=False, on_epoch=True)
        self.log("val_exist",  losses["exist"], on_step=False, on_epoch=True)
        self.log("val_label_ce", losses["lab_ce"], on_step=False, on_epoch=True)

        return total_loss

    def on_train_end(self):
        if not self.verbose: return
        train_extra = {}
        val_extra = {}
        if getattr(self, "use_edge_supervision", False):
            train_extra["edge"] = self.train_edge_loss
            val_extra["edge"]   = self.val_edge_loss
        if getattr(self, "edge_label_head", None) is not None:
            train_extra["edge_label"] = getattr(self, "train_edge_label_loss", [])
            val_extra["edge_label"]   = getattr(self, "val_edge_label_loss", [])

        plot_metrics(
            train_metrics={"total": self.train_losses, "deg_ce": self.train_deg_ce,
                        "recon": self.train_recon, "exist": self.train_exist,
                        "label_ce": self.train_label_ce, **train_extra},
            val_metrics={"total": self.val_losses, "deg_ce": self.val_deg_ce,
                        "recon": self.val_recon, "exist": self.val_exist,
                        "label_ce": self.val_label_ce, **val_extra},
            window=10, alpha=0.1
        )
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5, min_lr=1e-6, threshold=1e-4
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "monitor": "val_total"}
        }
    
    # ───────────────────────────────────────────────────────────────────
    #  Classifier-guidance utilities  
    # ───────────────────────────────────────────────────────────────────
    def set_guidance_classifier(self, num_classes: int) -> None:
        """Create a small MLP that maps pooled transformer latents → class-logits."""
        self.guidance_classifier = GuidanceMLP(
            input_dim=self.latent_embedding_dimension,
            hidden_dim=2 * self.latent_embedding_dimension,
            output_dim=num_classes
        ).to(self.device)

    
    def train_guidance_classifier(
        self,
        node_feats: List[np.ndarray],
        cond_vecs: np.ndarray,
        labels: np.ndarray,
        epochs: int = 20,
        lr: float = 1e-3,
        verbose: bool = True
    ):
        """Train guidance classifier with internal validation and loss plotting."""
        if self.guidance_classifier is None:
            raise RuntimeError("call set_guidance_classifier() first")

        self.eval()
        self.guidance_classifier.train()
        opt = torch.optim.Adam(self.guidance_classifier.parameters(), lr=lr)

        max_rows = self.number_of_rows_per_example
        padded_feats = []
        for f in node_feats:
            if f.shape[0] < max_rows:
                f = np.pad(f, ((0, max_rows - f.shape[0]), (0, 0)))
            padded_feats.append(f[:max_rows])

        X = torch.tensor(np.stack(padded_feats), dtype=torch.float32)
        Y = torch.tensor(cond_vecs, dtype=torch.float32)
        L = torch.tensor(labels, dtype=torch.long)

        # --- Split into train/val ---
        X_tr, X_val, Y_tr, Y_val, L_tr, L_val = train_test_split(
            X.numpy(), Y.numpy(), L.numpy(), test_size=0.2, random_state=42
        )
        X_tr = torch.tensor(X_tr, device=self.device)
        Y_tr = torch.tensor(Y_tr, device=self.device)
        L_tr = torch.tensor(L_tr, device=self.device)
        X_val = torch.tensor(X_val, device=self.device)
        Y_val = torch.tensor(Y_val, device=self.device)
        L_val = torch.tensor(L_val, device=self.device)

        T_tr = torch.zeros(X_tr.size(0), 1, device=self.device)
        T_val = torch.zeros(X_val.size(0), 1, device=self.device)

        with torch.no_grad():
            pass  # ensure no lingering grads on X/Y


        train_losses = []
        val_losses = []

        for _ in range(epochs):
            _, _, _, _, lat, _, _ = self.forward(x, global_condition, t_norm, return_latents=True, add_noise=False)
            logits = self.guidance_classifier(lat.mean(dim=1))
            loss = F.cross_entropy(logits, L_tr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

            with torch.no_grad():
                _, _, _, _, lat_val, _, _ = self.forward(X_val, Y_val, T_val, return_latents=True, add_noise=False)

                logits_val = self.guidance_classifier(lat_val.mean(dim=1))
                loss_val = F.cross_entropy(logits_val, L_val)
                val_losses.append(loss_val.item())

        if verbose:
            print(f"Trained guidance classifier for {epochs} epochs with learning rate {lr}.")
            print(f"Final train loss: {train_losses[-1]:.4f}, val loss: {val_losses[-1]:.4f}")
            # --- Plot losses ---
            plt.figure(figsize=(8, 5))
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Val Loss")
            plt.yscale('log')
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title("Guidance Classifier Loss")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()



    def _karras_sigmas(self, n: int, rho: float = 7.0, eps: float = 1e-4) -> torch.Tensor:
        # Avoid zero for numerical stability
        s_min = max(self.sampling_final_sigma, eps)
        s_max = self.sigma_max
        ramp = torch.linspace(0, 1, n, device=self.device if self.device else None)
        min_inv = s_min ** (1 / rho)
        max_inv = s_max ** (1 / rho)
        return (max_inv + ramp * (min_inv - max_inv)) ** rho


    def generate(
        self,
        global_condition: torch.Tensor,
        total_steps: int = 200,
        desired_class: Optional[Union[int, Sequence[int]]] = None,
        use_heads_projection: bool = True,   # NEW: use exist/deg heads to “snap” outputs
        exist_threshold: float = 0.5,        # threshold in prob space
    ) -> torch.Tensor:
        """
        Deterministic sampler consistent with training: x_t = x0 + sigma(t)*eps.
        Optionally projects the existence/degree channels using the auxiliary heads.
        """
        self.eval()
        B = global_condition.size(0)
        device = global_condition.device

        # --- sigma schedule consistent with training metadata ---
        self._ensure_schedule_metadata()
        # sigmas = torch.linspace(
        #     self.sigma_max,
        #     self.sampling_final_sigma,
        #     total_steps,
        #     device=device
        # )  # (T,)
        sigmas = self._karras_sigmas(total_steps).to(device)

        # Start from pure noise at the largest sigma
        x = torch.randn(
            B, self.number_of_rows_per_example, self.input_feature_dimension, device=device
        )

        # autocast for speed on GPU (skip on unsupported backends)
        if device.type == "cuda":
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
        elif device.type == "mps":
            autocast_ctx = torch.autocast(device_type="mps", dtype=torch.float16)
        else:
            autocast_ctx = contextlib.nullcontext()

        with (torch.inference_mode() if not self.use_guidance else contextlib.nullcontext()):
            with autocast_ctx:
                for i in range(total_steps - 1):
                    sigma_t   = sigmas[i]
                    sigma_next = sigmas[i + 1]

                    # time embeddings for current and next sigmas
                    t_norm  = self._t_from_sigma(sigma_t).expand(B, 1)
                    t_next  = self._t_from_sigma(sigma_next).expand(B, 1)

                    # predictor (Euler)
                    pred_eps, _, _, _, _, _ = self.forward(x, global_condition, t_norm, add_noise=False)
                    step = (sigma_t - sigma_next) * pred_eps
                    x_pred = x - step

                    # corrector (Heun / RK2)
                    pred_eps_next, _, _, _, _, _ = self.forward(x_pred, global_condition, t_next, add_noise=False)
                    step_corr = (sigma_t - sigma_next) * 0.5 * (pred_eps + pred_eps_next)
                    x_corr = x - step_corr

                    # adopt corrected state
                    x = x_corr

                    # optional: classifier guidance here (rebuild graph as needed)
                    if self.use_guidance and self.guidance_classifier is not None:
                        x.requires_grad_(True)
                        _, _, _, _, lat, _, _ = self.forward(x, global_condition, t_next, return_latents=True, add_noise=False)
                        pooled = lat.mean(dim=1)
                        logits_cls = self.guidance_classifier(pooled)
                        if desired_class is not None:
                            if isinstance(desired_class, int):
                                tgt = torch.full_like(logits_cls[:, 0], desired_class, dtype=torch.long)
                            else:
                                tgt = torch.as_tensor(desired_class, device=x.device)
                            sel = logits_cls[torch.arange(tgt.numel()), tgt]
                        else:
                            sel = logits_cls.softmax(-1).max(dim=-1).values
                        grad = torch.autograd.grad(sel.sum(), x, retain_graph=False)[0]
                        x = (x - self.guidance_weight * grad).detach()

                    # early exit: stop when the actual update is tiny in scaled space
                    if step_corr.abs().mean().item() < 1e-5:   # tune threshold
                        break


        # Optional: project existence/degree using the auxiliary heads once at (approx) t=0
        if use_heads_projection:
            with torch.inference_mode():
                sigma_proj = torch.tensor(self.sampling_final_sigma, device=device)
                t0 = self._t_from_sigma(sigma_proj).expand(B, 1)
                _, logits_deg, logits_exist, _, _, _, _ = self.forward(
                    x, global_condition, t0, return_latents=True, add_noise=False
                )
                exist_bin = (torch.sigmoid(logits_exist) >= exist_threshold).to(x.dtype)
                x[..., 0] = exist_bin  # now legal
                self._last_deg_classes = torch.argmax(logits_deg, dim=-1).cpu()
                if self.label_head is not None:
                    _, _, _, logits_lab, _, _, _ = self.forward(
                        x, global_condition, t0, return_latents=True, add_noise=False
                    )
                    self._last_lab_classes = torch.argmax(logits_lab, dim=-1).cpu()


        return x

# =============================================================================
# Revised TransformerConditionalDiffusionGenerator with Piecewise Scheduling Parameters
# =============================================================================
class GraphWithEdgesDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        edge_pairs: List[Tuple[int, int, int]],
        edge_targets: np.ndarray,
        node_mask: Optional[np.ndarray] = None,
        edge_label_pairs: Optional[List[Tuple[int,int,int]]] = None,  # NEW
        edge_label_targets: Optional[np.ndarray] = None               # NEW
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


        self.edge_cls_idx_by_graph = {b: [] for b in range(len(X))}
        self.edge_cls_lbl_by_graph = {b: [] for b in range(len(X))}
        if edge_label_pairs is not None and edge_label_targets is not None:
            for (b, i, j), c in zip(edge_label_pairs, edge_label_targets):
                self.edge_cls_idx_by_graph[b].append((i, j))
                self.edge_cls_lbl_by_graph[b].append(int(c))

    def __len__(self) -> int:
        """Return the number of graphs in the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single graph with its associated data.
        
        Returns:
            Tuple containing:
            - x: Node features (N, D)
            - y: Graph condition (C,)
            - edge_idx: Edge indices (E, 2)
            - edge_labels: Edge labels (E,)
            - mask: Node mask (N,)
        """
        x = self.X[idx]  # (N, D)
        y = self.Y[idx]  # (C,)
        mask = self.node_mask[idx]  # (N,)
        
        # Get edges and labels for this graph
        edge_idxs = torch.tensor(self.edge_idx_by_graph[idx], dtype=torch.long) if self.edge_idx_by_graph[idx] else torch.empty((0, 2), dtype=torch.long)
        edge_lbls = torch.tensor(self.edge_lbl_by_graph[idx], dtype=torch.float32) if self.edge_lbl_by_graph[idx] else torch.empty((0,), dtype=torch.float32)

        edge_cls_idxs = torch.tensor(self.edge_cls_idx_by_graph[idx], dtype=torch.long) if self.edge_cls_idx_by_graph[idx] else torch.empty((0, 2), dtype=torch.long)
        edge_cls_lbls = torch.tensor(self.edge_cls_lbl_by_graph[idx], dtype=torch.long) if self.edge_cls_lbl_by_graph[idx] else torch.empty((0,), dtype=torch.long)

        return x, y, edge_idxs, edge_lbls, mask, edge_cls_idxs, edge_cls_lbls  # ← NEW trailing items

def collate_graph_with_edges(batch):
    xs, ys, masks = [], [], []
    exist_ei, exist_el = [], []
    cls_ei, cls_el = [], []

    for x, y, ei, el, mask, eci, ecl in batch:
        xs.append(x); ys.append(y); masks.append(mask)
        exist_ei.append(ei); exist_el.append(el)
        cls_ei.append(eci); cls_el.append(ecl)

    X = torch.stack(xs); Y = torch.stack(ys); M = torch.stack(masks)

    def _stack_edges(edge_idxs_list, edge_lbls_list):
        all_idx, all_lbl = [], []
        for b, (ei, el) in enumerate(zip(edge_idxs_list, edge_lbls_list)):
            if ei.numel() == 0: continue
            bcol = torch.full((ei.size(0), 1), b, dtype=torch.long)
            all_idx.append(torch.cat([bcol, ei], dim=1))
            all_lbl.append(el)
        if all_idx:
            return torch.cat(all_idx, dim=0), torch.cat(all_lbl, dim=0)
        return torch.empty((0,3), dtype=torch.long), torch.empty((0,), dtype=edge_lbls_list[0].dtype)

    edge_idx, edge_lbl = _stack_edges(exist_ei, exist_el)       # (E_exist,3), (E_exist,)
    edge_cls_idx, edge_cls_lbl = _stack_edges(cls_ei, cls_el)   # (E_cls,3), (E_cls,)

    return X, Y, edge_idx, edge_lbl, M, edge_cls_idx, edge_cls_lbl


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
    
    total_steps : int, default=1000
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
                 total_steps: int = 1000,
                 verbose: bool = False,
                 important_feature_index: int = 1,
                 lambda_degree_importance: float = 1.0,
                 noise_degree_factor: float = 2.0,
                 degree_temperature: Optional[float] = None,
                 lambda_node_exist_importance: float = 1.0,
                 default_exist_pos_weight: float = 1.0,
                 lambda_edge_importance: float = 1.0,
                 use_guidance: bool = False,
                early_stopping: bool = True,
                early_stop_metric: str = "val_total",
                early_stop_mode: str = "min",
                early_stop_patience: int = 10,
                early_stop_min_delta: float = 1e-4,
                restore_best_weights: bool = True,
                checkpoint_dir: Optional[str] = None,
                checkpoint_policy: str = "best",  # "best" | "all" | "none"
                lambda_consistency: float = 0.3,
                lambda_label_importance: float = 1.0,
                noise_label_factor: float = 2.0,
                label_feature_index: int = 2,
                lambda_edge_label_importance: float = 1.0,
    ):
        self.latent_embedding_dimension = latent_embedding_dimension
        self.number_of_transformer_layers = number_of_transformer_layers
        self.transformer_attention_head_count = transformer_attention_head_count
        self.transformer_dropout = transformer_dropout
        self.learning_rate = learning_rate
        self.maximum_epochs = maximum_epochs
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.verbose = verbose
        self.important_feature_index = important_feature_index
        self.lambda_degree_importance = lambda_degree_importance
        self.noise_degree_factor = noise_degree_factor
        self.degree_temperature = degree_temperature
        self.lambda_node_exist_importance = lambda_node_exist_importance
        self.default_exist_pos_weight = default_exist_pos_weight
        self.lambda_edge_importance = lambda_edge_importance
        self.use_guidance = use_guidance

        self.number_of_rows_per_example = None
        self.input_feature_dimension = None
        self.model = None
        # self.conditional_generator_estimator = None # This was not used
        self.x_scaler = None # Scaler for node features
        self.y_scaler = None # Scaler for conditional features
        self.D_max = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # <-- FIXED
        self.early_stopping       = early_stopping
        self.early_stop_metric    = early_stop_metric
        self.early_stop_mode      = early_stop_mode
        self.early_stop_patience  = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.restore_best_weights = restore_best_weights
        self.checkpoint_dir       = checkpoint_dir
        self.checkpoint_policy    = checkpoint_policy
        self.lambda_consistency = lambda_consistency

        self.label_feature_index = label_feature_index              # <-- label lives at column 2
        self.lambda_label_importance = lambda_label_importance
        self.noise_label_factor = noise_label_factor
        self.L_max = None
        self.lambda_edge_label_importance = lambda_edge_label_importance


    def _fit_scalers(self, X_array, y_array):
        B, n, d = X_array.shape
        # X_reshaped = X_array.reshape(-1, d) # No longer needed for CustomRobustScaler

        # For MinMaxScaler, we'll fit X and Y separately
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()

        # Fit scaler for X (node features)
        # X_array has shape (B, N, D_x), reshape to (B*N, D_x) for scaler
        num_features_x = X_array.shape[2]
        self.x_scaler.fit(X_array.reshape(-1, num_features_x))

        # Fit scaler for Y (conditional features)
        self.y_scaler.fit(y_array)

        # self.raw_degree_index is already set by __init__ (defaults to 1)
        # With MinMaxScaler, the important_feature_index (for scaled data)
        # will be the same as raw_degree_index, as MinMaxScaler doesn't change column order.

    def _transform_data(self, X_array, y_array):
        B, N, D_x = X_array.shape
        X_scaled_flat = self.x_scaler.transform(X_array.reshape(-1, D_x))
        X_scaled = X_scaled_flat.reshape(B, N, D_x)
        
        y_scaled = self.y_scaler.transform(y_array)
        return X_scaled, y_scaled

    def _inverse_transform_input(self, X_array):
        B, N, D_x = X_array.shape # D_x is dimension of scaled features, same as original with MinMaxScaler
        X_orig_flat = self.x_scaler.inverse_transform(X_array.reshape(-1, D_x))
        X_orig = X_orig_flat.reshape(B, N, D_x) # Reshape back to (B, N, D_x_original)
        X_orig[..., self.important_feature_index] = np.clip(
            X_orig[..., self.important_feature_index], 0, self.D_max
        )
        return X_orig
    


    def setup(
        self,
        node_encodings_list: List[np.ndarray],
        conditional_graph_encodings: Any,
        edge_pairs: Optional[List[Tuple[int, int, int]]] = None,
        edge_targets: Optional[np.ndarray] = None,
        node_mask: Optional[np.ndarray] = None,
        edge_label_pairs: Optional[List[Tuple[int,int,int]]] = None,   # NEW
        edge_label_targets: Optional[np.ndarray] = None                # NEW
    ):

        """
        Setup the model for training.

        This method prepares the data, initializes scalers, and sets up the
        IterativeDenoisingAutoencoderTransformerModel. It computes scaling
        parameters, determines class imbalance weights, and initializes the
        model architecture.

        Parameters
        ----------
        node_encodings_list : List[np.ndarray]
            List of node encoding arrays, where each array represents a graph.
            Each array should have shape (num_nodes, feature_dimension).
        conditional_graph_encodings : Any
            Array of conditional graph encodings, where each encoding
            represents a graph-level condition.
        edge_pairs : Optional[List[Tuple[int, int, int]]], default=None
            Optional list of edge pairs for edge supervision. Each tuple
            represents an edge (graph_index, node_i, node_j).
        edge_targets : Optional[np.ndarray], default=None
            Optional array of edge targets for edge supervision. Each value
            represents the target for the corresponding edge pair.
        node_mask : Optional[np.ndarray], default=None
            Optional boolean mask indicating valid nodes in each graph.
            Used to exclude padded nodes from edge supervision.
        """
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

                # ---- label stats (like degree) ----
        lab_idx = self.label_feature_index
        raw_labels = X_array[..., lab_idx]                       # (B, N)
        self.L_max = int(raw_labels.max())                       # global max class id

        lab_col = raw_labels.reshape(-1, 1)
        lab_scaler = MinMaxScaler().fit(lab_col)                 # MinMax only on label col
        lab_min_val   = float(lab_scaler.data_min_[0])
        lab_range_val =float(lab_scaler.data_range_[0]) or 1e-8
        
        # 
       

                
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

        # Get degree scaling parameters (min and range) from x_scaler for the degree column
        # self.important_feature_index is the column index for degree in the raw X_array
        deg_column_data_for_minmax = X_array[..., self.important_feature_index].reshape(-1, 1)
        temp_degree_scaler = MinMaxScaler().fit(deg_column_data_for_minmax) # Fit only on the degree column
        
        deg_min_val = temp_degree_scaler.data_min_[0]
        deg_range_val = temp_degree_scaler.data_range_[0]
        if deg_range_val == 0: # Handle case where all degrees are the same
            deg_range_val = 1e-8 # Avoid division by zero, effectively makes scaled value 0 if val == min
        
        X_scaled, y_scaled = self._transform_data(X_array, y_array)
        self.input_feature_dimension = X_scaled.shape[2]
        cond_feature_dim = y_scaled.shape[1]
        
        # Detect maximum degree from raw data
        raw_degrees = X_array[..., self.important_feature_index]  # shape (B, N)
        self.D_max = int(raw_degrees.max())  # global max

        E_max_label = None
        use_edge_label_supervision = False
        if edge_label_targets is not None and len(edge_label_targets) > 0:
            uniq_edge_lbls = np.unique(edge_label_targets)
            if len(uniq_edge_lbls) > 1:
                E_max_label = int(uniq_edge_lbls.max())
                use_edge_label_supervision = True
            elif self.verbose:
                print("Only one edge label class found — disabling edge-label loss.")

        
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
            degree_min_val=deg_min_val,
            degree_range_val=deg_range_val,
            lambda_node_exist_importance=self.lambda_node_exist_importance,
            use_edge_supervision=(edge_pairs is not None),
            lambda_edge_importance=self.lambda_edge_importance,
            exist_pos_weight=exist_pos_weight,
            use_guidance=self.use_guidance,      # NEW
            guidance_weight=1.0,      
            lambda_consistency=self.lambda_consistency,          # tweak as needed
            label_feature_index=self.label_feature_index,   # 2
            max_label=self.L_max,
            lambda_label_importance=self.lambda_label_importance,
            noise_label_factor=self.noise_label_factor,
            label_min_val=lab_min_val,
            label_range_val=lab_range_val,
            use_edge_label_supervision=use_edge_label_supervision,
            max_edge_label=E_max_label,
            lambda_edge_label_importance=getattr(self, "lambda_edge_label_importance", 1.0)
        )
        self.model.use_guidance = self.use_guidance
        self._edge_pairs = edge_pairs
        self._edge_targets = edge_targets
        self._edge_label_pairs = edge_label_pairs
        self._edge_label_targets = edge_label_targets
        # self._node_mask = node_mask_arr  # created earlier

    def fit(
        self,
        node_encodings_list: List[np.ndarray],
        conditional_graph_encodings: Any,
        edge_pairs: Optional[List[Tuple[int, int, int]]] = None,
        edge_targets: Optional[np.ndarray] = None,
        node_mask: Optional[np.ndarray] = None
    ):
        """
        Fit the model to training data, optionally with edge supervision.

        This method prepares the data loaders and trains the initialized model
        using PyTorch Lightning. It assumes that the setup method has already
        been called to initialize scalers and the model architecture.

        Parameters
        ----------
        node_encodings_list : List[np.ndarray]
            List of node encoding arrays, where each array represents a graph.
            Each array should have shape (num_nodes, feature_dimension).
        conditional_graph_encodings : Any
            Array of conditional graph encodings, where each encoding
            represents a graph-level condition.
        edge_pairs : Optional[List[Tuple[int, int, int]]], default=None
            Optional list of edge pairs for edge supervision. Each tuple
            represents an edge (graph_index, node_i, node_j).
        edge_targets : Optional[np.ndarray], default=None
            Optional array of edge targets for edge supervision. Each value
            represents the target for the corresponding edge pair.
        node_mask : Optional[np.ndarray], default=None
            Optional boolean mask indicating valid nodes in each graph.
            Used to exclude padded nodes from edge supervision.
        """
        X_array = np.stack([np.pad(x, ((0, self.number_of_rows_per_example - x.shape[0]), (0, 0)), mode='constant', constant_values=0)
                        for x in node_encodings_list], axis=0)
        y_array = np.array(conditional_graph_encodings)
        X_scaled, y_scaled = self._transform_data(X_array, y_array)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
        use_graph_dataset = (edge_pairs is not None) or (getattr(self, "_edge_label_pairs", None) is not None)
        if edge_pairs is not None:
            if node_mask is None:
                B, N, _ = X_scaled.shape
                node_mask_arr = np.ones((B, N), dtype=bool)
            else:
                node_mask_arr = node_mask

            if use_graph_dataset:
                dataset = GraphWithEdgesDataset(
                    X_scaled, y_scaled,
                    edge_pairs=edge_pairs if edge_pairs is not None else [],
                    edge_targets=edge_targets if edge_targets is not None else np.array([], dtype=np.float32),
                    node_mask=node_mask_arr,
                    edge_label_pairs=getattr(self, "_edge_label_pairs", None),
                    edge_label_targets=getattr(self, "_edge_label_targets", None)
                )
            # Split into train/val
            dataset_size = len(node_encodings_list)  # <--- Use the length of the original data
            train_size = int(0.9 * dataset_size)
            val_size = dataset_size - train_size
            # Create indices for the split
            indices = torch.randperm(len(node_encodings_list)).tolist()
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            val_dataset = torch.utils.data.Subset(dataset, val_indices)
            
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
        
        # --- callbacks ---
        callbacks = [MetricsLogger()]

        # Early stopping
        if self.early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor=self.early_stop_metric,
                    mode=self.early_stop_mode,     # "min" for losses, "max" for accuracies
                    patience=self.early_stop_patience,
                    min_delta=self.early_stop_min_delta,
                    check_finite=True,
                    verbose=False,
                )
            )

        # Checkpointing
        ckpt_callback = None
        if (self.checkpoint_policy or "best") != "none":
            save_top_k = 1 if self.checkpoint_policy == "best" else -1  # -1 = save all
            ckpt_callback = ModelCheckpoint(
                dirpath=self.checkpoint_dir or "lightning_ckpts",
                filename="{epoch:03d}-{" + self.early_stop_metric + ":.5f}",
                monitor=self.early_stop_metric,
                mode=self.early_stop_mode,
                save_top_k=save_top_k,
                save_last=(self.checkpoint_policy != "best"),
                auto_insert_metric_name=False,
            )
            callbacks.append(ckpt_callback)

        trainer = pl.Trainer(
            max_epochs=self.maximum_epochs,
            callbacks=callbacks,
            logger=True,
            enable_checkpointing=(ckpt_callback is not None),
            enable_progress_bar=False,
        )
        if not self.verbose:
            with suppress_output():
                trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        else:
            trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        if self.restore_best_weights and ckpt_callback is not None:
            best_path = ckpt_callback.best_model_path
            if best_path:
                state = torch.load(best_path, map_location=self.device)
                self.model.load_state_dict(state["state_dict"])
                self.model.eval()

    def predict(
        self,
        conditional_graph_encodings: Any,
        desired_class: Optional[Union[int, Sequence[int]]] = None
    ) -> List[np.ndarray]:
        """
        Generate node-level latent matrices conditioned on global graph encodings.

        Steps:
        1. Calls the diffusion model's generate() to produce latent node embeddings.
        2. Converts back to original (pre-scaled) feature space.
        3. Overwrites existence and degree channels using the trained heads
        for better stability and interpretability at inference.
        """
        if self.verbose:
            print(f"Predicting node matrices for {len(conditional_graph_encodings)} graphs...")
            
        self.device = next(self.model.parameters()).device

        # ------------------------------------------------------------------
        # 1. Generate denoised node embeddings (scaled feature space)
        # ------------------------------------------------------------------
        with torch.no_grad():
            y_scaled = self.y_scaler.transform(np.asarray(conditional_graph_encodings))
            cond_tensor = torch.tensor(y_scaled, dtype=torch.float32, device=self.device)

            generated = self.model.generate(
                cond_tensor,
                total_steps=self.total_steps,
                desired_class=desired_class
            )

        # ------------------------------------------------------------------
        # 2. Convert to numpy and inverse-transform to original scale
        # ------------------------------------------------------------------
        gen_np = generated.detach().cpu().numpy()
        gen_orig = self._inverse_transform_input(gen_np)
        lab_classes = getattr(self.model, "_last_lab_classes", None)
        if lab_classes is not None:
            lab_classes = lab_classes.cpu().numpy()
            for i in range(len(gen_orig)):
                gen_orig[i][..., self.label_feature_index] = np.clip(lab_classes[i], 0, self.L_max)


        # ------------------------------------------------------------------
        # 3. Optional overwrite of existence / degree channels using heads
        # ------------------------------------------------------------------
        try:
            # Fetch stored head outputs (set inside generate)
            deg_classes = getattr(self.model, "_last_deg_classes", None)

            # Existence logits were already used inside generate (projected before inverse)
            # But if you prefer to re-compute existence probabilities here, uncomment below:
            # with torch.no_grad():
            #     t0 = torch.zeros(cond_tensor.size(0), 1, device=self.device)
            #     _, logits_deg, logits_exist, _, _, _ = self.model.forward(generated, cond_tensor, t0, return_latents=True)
            #     exist_probs = torch.sigmoid(logits_exist).cpu().numpy()
            #     exist_bin = (exist_probs >= 0.5).astype(float)
            #     for i in range(len(gen_orig)):
            #         gen_orig[i][..., 0] = exist_bin[i]

            # If degree logits were stored during generation, overwrite degree channel
            if deg_classes is not None:
                deg_classes = deg_classes.cpu().numpy()
                for i in range(len(gen_orig)):
                    # Overwrite the degree channel (assumed channel index 1)
                    deg_idx = self.important_feature_index
                    gen_orig[i][..., deg_idx] = np.clip(deg_classes[i], 0, self.D_max)


            if self.verbose:
                print("Applied head-based projection for existence/degree channels.")

        except Exception as e:
            if self.verbose:
                print(f"[Warning] Head projection skipped due to: {e}")

        return gen_orig

    def plot_metrics(self, window: int = 10, alpha: float = 0.3):
        if self.model is None:
            print("Model is not fitted yet.")
            return
        plot_metrics(
            train_metrics={
                "total": self.model.train_losses,
                "deg_ce": self.model.train_deg_ce,
                "recon": self.model.train_recon,
                "exist": self.model.train_exist,
                "label_ce": self.model.train_label_ce,
                **({"edge": self.model.train_edge_loss} if self.model.use_edge_supervision else {})
            },
            val_metrics={
                "total": self.model.val_losses,
                "deg_ce": self.model.val_deg_ce,
                "recon": self.model.val_recon,
                "exist": self.model.val_exist,
                "label_ce": self.model.val_label_ce,
                **({"edge": self.model.val_edge_loss} if self.model.use_edge_supervision else {})
            },
            window=window,
            alpha=alpha
        )

    
    # -------------- Guidance helpers --------------
    def set_guidance_classifier(self, num_classes: int):
        if self.model is None:
            raise RuntimeError("call setup() first")
        self.model.set_guidance_classifier(num_classes)

    def train_guidance_classifier(
        self, node_feats, cond_vecs, labels, *, epochs: int = 20, lr: float = 1e-3
    ):
        if self.model is None:
            raise RuntimeError("call setup() first")
        self.model.train_guidance_classifier(node_feats, cond_vecs, labels,
                                             epochs=epochs, lr=lr)

class MetricsLogger(pl.callbacks.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        pl_module.train_losses.append(m.get("train_total", torch.tensor(0.0)).item())
        pl_module.train_deg_ce.append(m.get("train_deg_ce", torch.tensor(0.0)).item())
        pl_module.train_recon.append(m.get("train_recon", torch.tensor(0.0)).item())
        pl_module.train_exist.append(m.get("train_exist", torch.tensor(0.0)).item())
        pl_module.train_label_ce.append(m.get("train_label_ce", torch.tensor(0.0)).item())
        if getattr(pl_module, "use_edge_supervision", False):
            pl_module.train_edge_loss.append(m.get("train_edge_loss", torch.tensor(0.0)).item())
            pl_module.train_edge_acc.append(m.get("train_edge_acc", torch.tensor(0.0)).item())
        if getattr(pl_module, "edge_label_head", None) is not None:
            # only when edge-label head is active (i.e., >1 unique label)
            pl_module.train_edge_label_loss.append(m.get("train_edge_label_loss", torch.tensor(0.0)).item())
            pl_module.train_edge_label_acc.append(m.get("train_edge_label_acc", torch.tensor(0.0)).item())

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        pl_module.val_losses.append(m.get("val_total", torch.tensor(0.0)).item())
        pl_module.val_deg_ce.append(m.get("val_deg_ce", torch.tensor(0.0)).item())
        pl_module.val_recon.append(m.get("val_recon", torch.tensor(0.0)).item())
        pl_module.val_exist.append(m.get("val_exist", torch.tensor(0.0)).item())
        pl_module.val_label_ce.append(m.get("val_label_ce", torch.tensor(0.0)).item())
        if getattr(pl_module, "use_edge_supervision", False):
            pl_module.val_edge_loss.append(m.get("val_edge_loss", torch.tensor(0.0)).item())
            pl_module.val_edge_acc.append(m.get("val_edge_acc", torch.tensor(0.0)).item())
        if getattr(pl_module, "edge_label_head", None) is not None:
            pl_module.val_edge_label_loss.append(m.get("val_edge_label_loss", torch.tensor(0.0)).item())
            pl_module.val_edge_label_acc.append(m.get("val_edge_label_acc", torch.tensor(0.0)).item())

