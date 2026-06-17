import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import contextlib, os, sys
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Sequence, Optional, Union, Tuple, List, Any
import math
from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import warnings
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression, Ridge
from scipy.optimize import linear_sum_assignment

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
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(embed_dim, eps=1e-5)
        hidden = embed_dim * 4
        # SwiGLU: Wi -> 2*hidden, split, SiLU on gate, elementwise product -> Wo
        self.wi = nn.Linear(embed_dim, 2 * hidden)
        self.wo = nn.Linear(hidden, embed_dim)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        x1 = self.norm1(x)
        x = x + self.dropout1(self.self_attn(x1, x1, x1)[0])

        x2 = self.norm2(x)
        x = x + self.dropout2(self.cross_attn(x2, k, v)[0])

        x3 = self.norm3(x)
        u, v_ = self.wi(x3).chunk(2, dim=-1)      # (B,N,hidden), (B,N,hidden)
        ff = F.silu(u) * v_                        # SwiGLU gate
        x = x + self.dropout3(self.wo(ff))
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
class BiaffineEdge(nn.Module):
    """
    Binary edge scorer: s(i,j) = h_i^T U h_j + w^T [h_i; h_j] + b
    Returns logits (E,) for BCEWithLogitsLoss.
    """
    def __init__(self, d: int, use_affine: bool = True):
        super().__init__()
        self.U = nn.Parameter(torch.empty(d, d))
        nn.init.xavier_uniform_(self.U)
        self.use_affine = use_affine
        if use_affine:
            self.w = nn.Linear(2*d, 1)  # includes bias

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        # h_i, h_j: (E, d)
        bilinear = torch.einsum('ed,dm,em->e', h_i, self.U, h_j)  # (E,)
        if self.use_affine:
            affine = self.w(torch.cat([h_i, h_j], dim=-1)).squeeze(-1)  # (E,)
            return bilinear + affine
        return bilinear


class BiaffineEdgeLabel(nn.Module):
    """
    Multiclass edge label scorer:
      s_c(i,j) = h_i^T U_c h_j + W_c [h_i;h_j] + b_c  for c=1..C
    Returns logits (E, C) for CrossEntropyLoss.
    """
    def __init__(self, d: int, num_classes: int, use_affine: bool = True):
        super().__init__()
        self.U = nn.Parameter(torch.empty(num_classes, d, d))
        nn.init.xavier_uniform_(self.U)
        self.use_affine = use_affine
        if use_affine:
            self.W = nn.Linear(2*d, num_classes)
            self.b = nn.Parameter(torch.zeros(num_classes))
        else:
            self.register_parameter('W', None)
            self.register_parameter('b', None)

    def forward(self, h_i: torch.Tensor, h_j: torch.Tensor) -> torch.Tensor:
        # h_i, h_j: (E, d)
        bilinear = torch.einsum('ed,cdf,ef->ec', h_i, self.U, h_j)  # (E, C)
        if self.use_affine:
            return bilinear + self.W(torch.cat([h_i, h_j], dim=-1)) + self.b
        return bilinear

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


class EdgeStateMessagePassing(nn.Module):
    """
    Lightweight graph-state refinement for joint node/edge denoising.

    The transformer still provides global row interactions. This block adds a
    local structural update from the current noisy/candidate adjacency A_t:
    h_i <- h_i + MLP([h_i, mean_j A_t[i,j] h_j]).
    """
    def __init__(self, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.update = nn.Sequential(
            nn.Linear(2 * latent_dim, 2 * latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * latent_dim, latent_dim),
        )

    def forward(self, h: torch.Tensor, edge_state: torch.Tensor) -> torch.Tensor:
        edge_state = edge_state.to(dtype=h.dtype, device=h.device)
        edge_state = edge_state.clamp(0.0, 1.0)
        denom = edge_state.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neigh = torch.bmm(edge_state, h) / denom
        delta = self.update(torch.cat([self.norm(h), neigh], dim=-1))
        return h + delta

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
                 degree_class_weight: Optional[Union[torch.Tensor, Sequence[float]]] = None,
                 lambda_node_exist_importance: float = 1.0,
                 use_edge_supervision: bool = False,
                 lambda_edge_importance: float = 1.0,
                 lambda_clean_edge_importance: float = 0.0,
                 balance_edge_loss: bool = False,
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
                 label_class_weight: Optional[Union[torch.Tensor, Sequence[float]]] = None,
                 label_min_val: float = 0.0,
                 label_range_val: float = 1.0,
                 use_edge_label_supervision: bool = False,
                max_edge_label: Optional[int] = None,
                lambda_edge_label_importance: float = 1.0,

                use_distance_supervision: bool = True,
                max_distance_class: int = 3,          # 0..3 == 1,2,3,4+
                lambda_distance_importance: float = 1.0,
                lambda_recon_importance: float = 1.0,
                lambda_x0_importance: float = 0.0,
                lambda_condition_x0_importance: float = 0.0,
                condition_x0_sampling_blend: float = 0.0,
                denoise_discrete_channels: bool = False,
                discrete_diffusion_mode: str = "none",
                discrete_projection_sigma_threshold: float = 0.25,
                row_embedding_scale: float = 0.1,
                use_joint_edge_denoising: bool = False,
                joint_edge_delete_prob: float = 0.25,
                joint_edge_add_prob: float = 0.05,
                joint_edge_update_momentum: float = 0.5,
                joint_edge_threshold: float = 0.5,
                joint_edge_message_passing_steps: int = 2,
                project_existence_during_sampling: bool = True,
                project_degree_during_sampling: bool = True,
                project_label_during_sampling: bool = True,
                use_matched_row_loss: bool = False
                 
                 
                 
                 
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
        self.lambda_recon_importance = lambda_recon_importance
        self.lambda_x0_importance = lambda_x0_importance
        self.lambda_condition_x0_importance = lambda_condition_x0_importance
        self.condition_x0_sampling_blend = float(condition_x0_sampling_blend)
        self.denoise_discrete_channels = bool(denoise_discrete_channels)
        self.discrete_diffusion_mode = str(discrete_diffusion_mode).lower()
        self.discrete_projection_sigma_threshold = float(discrete_projection_sigma_threshold)
        self.row_embedding_scale = float(row_embedding_scale)
        self.use_joint_edge_denoising = bool(use_joint_edge_denoising)
        self.joint_edge_delete_prob = float(joint_edge_delete_prob)
        self.joint_edge_add_prob = float(joint_edge_add_prob)
        self.joint_edge_update_momentum = float(joint_edge_update_momentum)
        self.joint_edge_threshold = float(joint_edge_threshold)
        self.joint_edge_message_passing_steps = max(1, int(joint_edge_message_passing_steps))
        self.project_existence_during_sampling = bool(project_existence_during_sampling)
        self.project_degree_during_sampling = bool(project_degree_during_sampling)
        self.project_label_during_sampling = bool(project_label_during_sampling)
        self.use_matched_row_loss = bool(use_matched_row_loss)
        if self.discrete_diffusion_mode not in {"none", "random_replace", "absorbing"}:
            raise ValueError(
                "discrete_diffusion_mode must be one of "
                "'none', 'random_replace', or 'absorbing'"
            )


        
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
        if degree_class_weight is None:
            degree_class_weight = torch.ones(max_degree + 1, dtype=torch.float32)
        self.register_buffer(
            'degree_class_weight',
            torch.as_tensor(degree_class_weight, dtype=torch.float32)
        )
        if label_class_weight is None:
            if max_label is None:
                label_class_weight = torch.ones(1, dtype=torch.float32)
            else:
                label_class_weight = torch.ones(max_label + 1, dtype=torch.float32)
        self.register_buffer(
            'label_class_weight',
            torch.as_tensor(label_class_weight, dtype=torch.float32)
        )
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
        self.train_x0 = []
        self.val_x0 = []
        self.train_condition_x0 = []
        self.val_condition_x0 = []
        self.train_label_ce = []
        self.val_label_ce = []
        self.train_joint_edge_loss = []
        self.val_joint_edge_loss = []
        self.train_joint_edge_acc = []
        self.val_joint_edge_acc = []
        if self.use_edge_supervision:
            self.train_edge_loss = []
            self.val_edge_loss   = []
            self.train_edge_acc = []
            self.val_edge_acc = []
            self.train_clean_edge_loss = []
            self.val_clean_edge_loss = []
            self.train_clean_edge_acc = []
            self.val_clean_edge_acc = []

        # Model layers
        self.layernorm_in = nn.LayerNorm(input_feature_dimension, elementwise_affine=True)
        self.linear_encoder_input_to_latent = nn.Linear(input_feature_dimension, latent_embedding_dimension)
        self.linear_encoder_condition_to_latent = nn.Linear(condition_feature_dimension, latent_embedding_dimension)

        self.row_embedding = nn.Embedding(
            number_of_rows_per_example,
            latent_embedding_dimension,
        )
        self.condition_x0_trunk = nn.Sequential(
            nn.LayerNorm(condition_feature_dimension),
            nn.Linear(condition_feature_dimension, latent_embedding_dimension),
            nn.SiLU(),
            nn.Linear(latent_embedding_dimension, latent_embedding_dimension),
            nn.SiLU(),
        )
        self.condition_x0_table_head = nn.Linear(
            latent_embedding_dimension,
            number_of_rows_per_example * input_feature_dimension,
        )
        self.condition_x0_exist_head = nn.Linear(
            latent_embedding_dimension,
            number_of_rows_per_example,
        )
        self.condition_x0_degree_head = nn.Linear(
            latent_embedding_dimension,
            number_of_rows_per_example * (max_degree + 1),
        )
        self.condition_x0_label_head = (
            nn.Linear(latent_embedding_dimension, number_of_rows_per_example * (max_label + 1))
            if max_label is not None
            else None
        )
        
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
        self.lambda_clean_edge_importance = lambda_clean_edge_importance
        self.balance_edge_loss = bool(balance_edge_loss)
        self.max_label = max_label
        self.lambda_label_importance = lambda_label_importance
        self.noise_label_factor = noise_label_factor
        self.label_feature_index = label_feature_index
        self.use_edge_label_supervision = use_edge_label_supervision
        self.max_edge_label = max_edge_label
        self.lambda_edge_label_importance = lambda_edge_label_importance

        self.use_distance_supervision = use_distance_supervision
        self.max_distance_class = max_distance_class
        self.lambda_distance_importance = lambda_distance_importance

        
        if self.use_edge_supervision:
            self.edge_head = BiaffineEdge(latent_embedding_dimension, use_affine=True)
        elif self.use_joint_edge_denoising:
            self.use_joint_edge_denoising = False
            warnings.warn(
                "use_joint_edge_denoising=True requires edge supervision; disabling it."
            )

        self.edge_state_message_passing = (
            nn.ModuleList([
                EdgeStateMessagePassing(latent_embedding_dimension, dropout=transformer_dropout)
                for _ in range(self.joint_edge_message_passing_steps)
            ])
            if self.use_joint_edge_denoising
            else None
        )


        if self.use_edge_label_supervision and (self.max_edge_label is not None) and (self.max_edge_label >= 1):
            self.edge_label_head = BiaffineEdgeLabel(
                d=latent_embedding_dimension,
                num_classes=self.max_edge_label + 1,
                use_affine=True
            )
        else:
            self.edge_label_head = None

        self.train_dist_loss = []
        self.val_dist_loss = []

        # distance head toggled by flag
        if self.use_distance_supervision:
            self.distance_head = EdgeLabelMLP(
                latent_dim=latent_embedding_dimension,
                num_classes=self.max_distance_class + 1,
                hidden_dim=2 * latent_embedding_dimension,
                dropout=transformer_dropout
            )
        else:
            self.distance_head = None




        if self.use_edge_supervision:
            self.train_edge_loss = []; self.val_edge_loss = []; self.train_edge_acc = []; self.val_edge_acc = []
            self.train_clean_edge_loss = []; self.val_clean_edge_loss = []
            self.train_clean_edge_acc = []; self.val_clean_edge_acc = []

        self.train_joint_edge_loss = []; self.val_joint_edge_loss = []
        self.train_joint_edge_acc = []; self.val_joint_edge_acc = []

        if self.use_edge_label_supervision:
            self.train_edge_label_loss = []; self.val_edge_label_loss = []
            self.train_edge_label_acc  = []; self.val_edge_label_acc  = []


           


        # heads

        if self.max_label is not None:
            self.label_head = nn.Linear(latent_embedding_dimension, self.max_label + 1)  # <-- NEW
        else:
            self.label_head = None

    def predict_condition_x0(
        self,
        global_condition_vector: torch.Tensor,
        *,
        project_discrete: bool = False,
        return_logits: bool = False,
        exist_threshold: float = 0.5,
    ):
        """Predict a clean node table directly from the conditioning vector."""
        B = global_condition_vector.shape[0]
        h = self.condition_x0_trunk(global_condition_vector)
        x0 = self.condition_x0_table_head(h).view(
            B, self.number_of_rows_per_example, self.input_feature_dimension
        )
        logits_exist = self.condition_x0_exist_head(h)
        logits_deg = self.condition_x0_degree_head(h).view(
            B, self.number_of_rows_per_example, self.max_degree + 1
        )
        logits_lab = None
        if self.condition_x0_label_head is not None:
            logits_lab = self.condition_x0_label_head(h).view(
                B, self.number_of_rows_per_example, self.max_label + 1
            )

        if project_discrete:
            x0 = x0.clone()
            x0[..., 0] = (torch.sigmoid(logits_exist) >= exist_threshold).to(x0.dtype)
            deg_cls = torch.argmax(logits_deg, dim=-1).to(x0.dtype)
            x0[..., self.important_feature_index] = (
                deg_cls - self.deg_min_val
            ) / self.deg_range_val
            if logits_lab is not None and self.label_feature_index is not None:
                lab_cls = torch.argmax(logits_lab, dim=-1).to(x0.dtype)
                x0[..., self.label_feature_index] = (
                    lab_cls - self.lab_min_val
                ) / self.lab_range_val

        if return_logits:
            return x0, logits_exist, logits_deg, logits_lab
        return x0

    def _unpack_batch(self, batch):
            # 9-tuple: (x,y,edge_idx,edge_lbls,mask,edge_cls_idx,edge_cls_lbl,dist_cls_idx,dist_cls_lbl)
            if len(batch) == 9:
                return batch

            # 7-tuple (legacy: existence + edge-label)
            if len(batch) == 7:
                x, y, edge_idx, edge_labels, mask, edge_cls_idx, edge_cls_lbl = batch
                device = x.device
                empty3 = torch.empty((0,3), dtype=torch.long, device=device)
                emptyL = torch.empty((0,),   dtype=torch.long, device=device)
                return x, y, edge_idx, edge_labels, mask, edge_cls_idx, edge_cls_lbl, empty3, emptyL

            # 5-tuple (legacy: existence only)
            if len(batch) == 5:
                x, y, edge_idx, edge_labels, mask = batch
                device = x.device
                empty3 = torch.empty((0,3), dtype=torch.long, device=device)
                emptyL = torch.empty((0,),   dtype=torch.long, device=device)
                return x, y, edge_idx, edge_labels, mask, empty3, emptyL, empty3, emptyL

            # 2-tuple (no supervision)
            if len(batch) == 2:
                x, y = batch
                B, N, _ = x.shape
                device = x.device
                mask = torch.ones((B, N), dtype=torch.bool, device=device)
                empty3 = torch.empty((0,3), dtype=torch.long, device=device)
                emptyF = torch.empty((0,),   dtype=torch.float32, device=device)
                emptyL = torch.empty((0,),   dtype=torch.long, device=device)
                return x, y, empty3, emptyF, mask, empty3, emptyL, empty3, emptyL

            raise ValueError(f"Unexpected batch format with {len(batch)} items")


    def forward(
        self,
        input_rows: torch.Tensor,
        global_condition_vector: torch.Tensor,
        diffusion_time_step: torch.Tensor,
        return_latents: bool = False,
        add_noise: bool = True,
        edge_state: Optional[torch.Tensor] = None,
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

        row_ids = torch.arange(
            self.number_of_rows_per_example,
            device=input_rows.device,
        ).unsqueeze(0).expand(input_rows.shape[0], -1)

        if self.row_embedding_scale != 0:
            latent_tokens = latent_tokens + self.row_embedding_scale * self.row_embedding(row_ids)

        # Build memory (time + condition)
        time_token = get_sinusoidal_time_embedding(diffusion_time_step, self.latent_embedding_dimension)
        cond_token = self.linear_encoder_condition_to_latent(global_condition_vector)
        mem = torch.stack([time_token, cond_token], dim=1)  # (B,2,D)

        # Transformer with cross-attention
        for layer in self.shared_transformer:
            latent_tokens = layer(latent_tokens, k=mem, v=mem)

        if self.edge_state_message_passing is not None and edge_state is not None:
            for message_layer in self.edge_state_message_passing:
                latent_tokens = message_layer(latent_tokens, edge_state)

        # Predict ε for continuous features
        pred_eps = self.linear_decoder_latent_to_output(latent_tokens)

        # Other prediction heads remain the same
        # Build per-feature noise scale from *time* (works for train + sampling)
        sigma_from_t = self._sigma_from_t(diffusion_time_step).unsqueeze(-1)  # (B,1,1)
        noise_scale = torch.ones_like(noisy_input) * sigma_from_t
        noise_scale[..., self.important_feature_index] /= self.noise_degree_factor

        # Use the final transformer latents for all discrete heads.  These
        # latents have seen self-attention plus cross-attention to the graph
        # condition, whereas a row-wise re-encoding of x0_hat is mostly local
        # and makes degree/label prediction much less stable during sampling.
        logits_deg   = self.degree_head(latent_tokens)
        logits_exist = self.exist_head(latent_tokens).squeeze(-1)
        logits_lab   = self.label_head(latent_tokens) if self.label_head is not None else None


        if return_latents:
            return pred_eps, logits_deg, logits_exist, logits_lab, latent_tokens, eps, sigma_t
        return pred_eps, logits_deg, logits_exist, logits_lab, eps, sigma_t

    def _dense_adjacency_from_pairs(
        self,
        edge_idx: torch.Tensor,
        edge_labels: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, N = node_mask.shape
        adj = torch.zeros((B, N, N), dtype=torch.float32, device=node_mask.device)
        if edge_idx.numel() > 0:
            b, i, j = edge_idx.unbind(1)
            valid = node_mask[b, i] & node_mask[b, j]
            if valid.any():
                b, i, j = b[valid], i[valid], j[valid]
                values = edge_labels[valid].float()
                adj[b, i, j] = values
        adj = torch.maximum(adj, adj.transpose(1, 2))
        eye = torch.eye(N, dtype=torch.bool, device=node_mask.device).unsqueeze(0)
        adj = adj.masked_fill(eye, 0.0)
        valid_nodes = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        return adj * valid_nodes.float()

    def _corrupt_adjacency_state(
        self,
        clean_adj: torch.Tensor,
        node_mask: torch.Tensor,
        diffusion_time_step: torch.Tensor,
    ) -> torch.Tensor:
        if clean_adj.numel() == 0:
            return clean_adj
        B, N, _ = clean_adj.shape
        t = diffusion_time_step.clamp(0.0, 1.0).view(B, 1, 1)
        valid = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        eye = torch.eye(N, dtype=torch.bool, device=clean_adj.device).unsqueeze(0)
        valid = valid & ~eye

        delete_p = (self.joint_edge_delete_prob * t).clamp(0.0, 1.0)
        add_p = (self.joint_edge_add_prob * t).clamp(0.0, 1.0)

        keep_true = torch.rand_like(clean_adj) >= delete_p
        add_false = torch.rand_like(clean_adj) < add_p

        noisy = clean_adj.clone()
        noisy = torch.where(clean_adj >= 0.5, keep_true.float(), noisy)
        noisy = torch.where((clean_adj < 0.5) & add_false, torch.ones_like(noisy), noisy)
        noisy = noisy * valid.float()
        noisy = torch.maximum(noisy, noisy.transpose(1, 2))
        return noisy.masked_fill(eye, 0.0)

    def _dense_edge_probabilities_from_latents(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        B, N, _ = latent_tokens.shape
        probs = torch.zeros((B, N, N), dtype=latent_tokens.dtype, device=latent_tokens.device)
        if getattr(self, "edge_head", None) is None:
            return probs
        for i in range(N):
            h_i = latent_tokens[:, i].unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
            h_j = latent_tokens.reshape(B * N, -1)
            logits = self.edge_head(h_i, h_j).view(B, N)
            probs[:, i, :] = torch.sigmoid(logits)
        eye = torch.eye(N, dtype=torch.bool, device=latent_tokens.device).unsqueeze(0)
        probs = probs.masked_fill(eye, 0.0)
        return 0.5 * (probs + probs.transpose(1, 2))

    def _dense_edge_logits_from_latents(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        B, N, _ = latent_tokens.shape
        logits_dense = torch.zeros((B, N, N), dtype=latent_tokens.dtype, device=latent_tokens.device)
        if getattr(self, "edge_head", None) is None:
            return logits_dense
        flat_h = latent_tokens.reshape(B * N, -1)
        for i in range(N):
            h_i = latent_tokens[:, i].unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
            logits_dense[:, i, :] = self.edge_head(h_i, flat_h).view(B, N)
        logits_dense = 0.5 * (logits_dense + logits_dense.transpose(1, 2))
        eye = torch.eye(N, dtype=torch.bool, device=latent_tokens.device).unsqueeze(0)
        return logits_dense.masked_fill(eye, 0.0)

    def _dense_edge_reconstruction_loss(
        self,
        latent_tokens: torch.Tensor,
        clean_adj: torch.Tensor,
        node_mask: torch.Tensor,
        sigma_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_dense = self._dense_edge_logits_from_latents(latent_tokens)
        B, N, _ = logits_dense.shape
        upper = torch.triu(torch.ones((N, N), dtype=torch.bool, device=logits_dense.device), diagonal=1)
        valid = node_mask.unsqueeze(1) & node_mask.unsqueeze(2) & upper.unsqueeze(0)
        if not valid.any():
            zero = logits_dense.sum() * 0.0
            return zero, zero

        logits = logits_dense[valid]
        targets = clean_adj.to(dtype=logits.dtype, device=logits.device)[valid]
        graph_ids = torch.arange(B, device=logits.device).view(B, 1, 1).expand(B, N, N)[valid]
        sigma_weight = 1.0 / (
            sigma_t.squeeze(-1).squeeze(-1)[graph_ids].to(dtype=logits.dtype) ** 2 + 1e-4
        )

        n_pos = targets.sum()
        n_neg = targets.numel() - n_pos
        if self.balance_edge_loss and n_pos > 0 and n_neg > 0:
            pos_weight = torch.clamp(n_neg / (n_pos + 1e-8), min=1.0, max=20.0)
            loss_elem = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=pos_weight,
                reduction="none",
            )
        else:
            loss_elem = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = (loss_elem * sigma_weight).sum() / (sigma_weight.sum() + 1e-8)
        with torch.no_grad():
            acc = ((torch.sigmoid(logits) >= 0.5).to(targets.dtype) == targets).float().mean()
        return loss, acc

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
        if not hasattr(self, "lambda_condition_x0_importance"):
            self.lambda_condition_x0_importance = 0.0
        if not hasattr(self, "condition_x0_sampling_blend"):
            self.condition_x0_sampling_blend = 0.0
        if not hasattr(self, "discrete_diffusion_mode"):
            self.discrete_diffusion_mode = "none"

    def sample_diffusion_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample uniform training timesteps in normalized diffusion time."""
        return torch.rand(batch_size, 1, device=device)

    def _scaled_degree_from_class(self, degree_class: torch.Tensor) -> torch.Tensor:
        return (degree_class.to(torch.float32) - self.deg_min_val) / self.deg_range_val

    def _scaled_label_from_class(self, label_class: torch.Tensor) -> torch.Tensor:
        return (label_class.to(torch.float32) - self.lab_min_val) / self.lab_range_val

    def _degree_mask_class(self) -> int:
        return int(self.max_degree) + 1

    def _label_mask_class(self) -> int:
        return int(self.max_label) + 1

    def _corrupt_discrete_columns(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Corrupt degree/label columns as categorical variables during training."""
        mode = getattr(self, "discrete_diffusion_mode", "none")
        if mode == "none":
            return x_t

        out = x_t.clone()
        B, N, _ = out.shape
        replace_prob = t.clamp(0.0, 1.0).view(B, 1).expand(B, N)

        deg_replace = torch.rand(B, N, device=out.device) < replace_prob
        if mode == "absorbing":
            deg_cls = torch.full(
                (B, N),
                self._degree_mask_class(),
                dtype=torch.long,
                device=out.device,
            )
        else:
            deg_cls = torch.randint(
                low=0,
                high=self.max_degree + 1,
                size=(B, N),
                device=out.device,
            )
        out[..., self.important_feature_index] = torch.where(
            deg_replace,
            self._scaled_degree_from_class(deg_cls),
            out[..., self.important_feature_index],
        )

        if self.label_feature_index is not None and self.max_label is not None:
            lab_replace = torch.rand(B, N, device=out.device) < replace_prob
            if mode == "absorbing":
                lab_cls = torch.full(
                    (B, N),
                    self._label_mask_class(),
                    dtype=torch.long,
                    device=out.device,
                )
            else:
                lab_cls = torch.randint(
                    low=0,
                    high=self.max_label + 1,
                    size=(B, N),
                    device=out.device,
                )
            out[..., self.label_feature_index] = torch.where(
                lab_replace,
                self._scaled_label_from_class(lab_cls),
                out[..., self.label_feature_index],
            )

        return out

    def _project_discrete_columns_from_logits(
        self,
        x: torch.Tensor,
        logits_deg: torch.Tensor,
        logits_lab: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Update generated degree/label columns from categorical heads."""
        out = x.clone()
        if self.project_degree_during_sampling:
            deg_cls = torch.argmax(logits_deg, dim=-1)
            out[..., self.important_feature_index] = self._scaled_degree_from_class(deg_cls).to(out.dtype)

        if (
            self.project_label_during_sampling
            and
            logits_lab is not None
            and self.label_feature_index is not None
            and self.label_feature_index < self.input_feature_dimension
        ):
            lab_cls = torch.argmax(logits_lab, dim=-1)
            out[..., self.label_feature_index] = self._scaled_label_from_class(lab_cls).to(out.dtype)

        return out

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
        x_t = self._corrupt_discrete_columns(x_t, t)
        return x_t, eps, sigma_t

    def _hungarian_match_targets(self, x0_hat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Reorder target rows using only continuous structural features.

        The assignment itself is non-differentiable and computed on detached
        tensors. Existence, degree, and label are deliberately excluded from
        the matching cost so their supervised losses cannot choose an
        artificially favorable alignment. Gradients still flow through losses
        involving x0_hat/logits after the assignment has been selected.
        """
        if x0_hat.shape != target.shape:
            raise ValueError("x0_hat and target must have the same shape for row matching")

        B, N, D = x0_hat.shape
        excluded_indices = {
            0,
            int(self.important_feature_index),
        }
        if self.label_feature_index is not None:
            excluded_indices.add(int(self.label_feature_index))
        match_indices = [
            index for index in range(D)
            if index not in excluded_indices
        ]

        # Without independent structural features there is no honest basis for
        # permutation matching. Preserve the existing row order in that case.
        if not match_indices:
            return target

        matched = torch.empty_like(target)
        pred_detached = x0_hat.detach()[..., match_indices].float()
        target_detached = target.detach()[..., match_indices].float()

        for b in range(B):
            cost = torch.cdist(
                pred_detached[b],
                target_detached[b],
                p=2,
            ).cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            matched[b, row_ind] = target[b, col_ind]

        return matched

  
    # ---------------------------------------------------------------------------
    # single-source loss computation – returns all partials
        # ---------------------------------------------------------------------------
    def compute_weighted_loss(self, prediction, target, condition_x0_prediction=None) -> dict:
        # unpack
        pred_eps, logits_deg, logits_exist, logits_lab, eps, sigma_t = prediction

        B, N, D = pred_eps.shape
        mask = torch.ones((1, 1, D), device=pred_eps.device, dtype=pred_eps.dtype)
        mask[..., 0] = 0  # existence
        discrete_mode = getattr(self, "discrete_diffusion_mode", "none")
        gaussian_discrete = self.denoise_discrete_channels and discrete_mode == "none"
        if not gaussian_discrete:
            mask[..., self.important_feature_index] = 0  # degree
        if (not gaussian_discrete) and self.label_feature_index is not None:
            mask[..., self.label_feature_index] = 0  # <-- exclude label from ε

        # Continuous epsilon prediction is only meaningful on real/existent
        # rows. Padded rows are handled by the existence and consistency
        # terms; including them here can make the continuous objective mostly
        # about reconstructing padding for variable-size graphs.
        original_target_exist = (target[..., 0] >= 0.5).float()
        real_row_mask = original_target_exist.unsqueeze(-1)
        cont_sq = ((pred_eps - eps) * mask) ** 2
        cont_weight = real_row_mask * mask
        loss_eps = (cont_sq * cont_weight).sum() / (cont_weight.sum() + 1e-8)

        noise_scale = torch.ones_like(target) * sigma_t
        noise_scale[..., self.important_feature_index] /= self.noise_degree_factor
        if self.label_feature_index is not None:
            noise_scale[..., self.label_feature_index] /= self.noise_label_factor
        x_t = target + noise_scale * eps
        x0_hat = x_t - noise_scale * pred_eps

        target_for_loss = target
        if getattr(self, "use_matched_row_loss", False):
            target_for_loss = self._hungarian_match_targets(x0_hat, target)
            loss_eps = pred_eps.new_zeros(())

        target_exist = (target_for_loss[..., 0] >= 0.5).float()
        real_row_mask = target_exist.unsqueeze(-1)

        # Direct clean-table reconstruction keeps the reverse process anchored
        # to X0, instead of only teaching the network to predict epsilon.
        x0_weight = real_row_mask.expand_as(target) * mask.expand_as(target)
        loss_x0 = ((x0_hat - target_for_loss) ** 2 * x0_weight).sum() / (x0_weight.sum() + 1e-8)

        if condition_x0_prediction is not None:
            if isinstance(condition_x0_prediction, tuple):
                condition_x0_hat, cond_logits_exist, cond_logits_deg, cond_logits_lab = condition_x0_prediction
            else:
                condition_x0_hat = condition_x0_prediction
                cond_logits_exist = cond_logits_deg = cond_logits_lab = None

            condition_cont_mask = mask.expand_as(target)
            condition_x0_weight = real_row_mask.expand_as(target) * condition_cont_mask
            loss_condition_x0_mse = (
                ((condition_x0_hat - target_for_loss) ** 2 * condition_x0_weight).sum()
                / (condition_x0_weight.sum() + 1e-8)
            )
        else:
            condition_x0_hat = None
            cond_logits_exist = cond_logits_deg = cond_logits_lab = None
            loss_condition_x0_mse = pred_eps.new_zeros(())

        # consistency loss (also exclude label channel in cont_mask)
        lam_cons = getattr(self, "lambda_consistency", 0.0)
        if lam_cons > 0:
            cont_mask = torch.ones_like(target)
            cont_mask[..., 0] = 0
            cont_mask[..., self.important_feature_index] = 0
            if self.label_feature_index is not None:
                cont_mask[..., self.label_feature_index] = 0
            absent = (target_for_loss[..., 0] < 0.5).float().unsqueeze(-1)
            loss_cons = ((x0_hat * cont_mask)**2 * absent).mean()
        else:
            loss_cons = pred_eps.new_zeros(())

        # degree classes
        deg_unscaled = target_for_loss[..., self.important_feature_index] * self.deg_range_val + self.deg_min_val
        true_deg_class = torch.clamp(torch.round(deg_unscaled), 0, self.max_degree).long()

        # label classes (like degree)
        if (self.label_head is not None) and (self.label_feature_index is not None):
            lab_unscaled = target_for_loss[..., self.label_feature_index] * self.lab_range_val + self.lab_min_val
            true_lab_class = torch.clamp(torch.round(lab_unscaled), 0, self.max_label).long()
        else:
            true_lab_class = None

        # t-weights
        w = 1.0 / (sigma_t.squeeze(-1).squeeze(-1) ** 2 + 1e-4)

        # existence loss
        exist_elem = F.binary_cross_entropy_with_logits(
            logits_exist, target_exist, pos_weight=self.exist_pos_weight, reduction='none'
        )
        exist_weight = w.view(-1, 1).expand_as(exist_elem)
        loss_exist = (exist_elem * exist_weight).sum() / (exist_weight.sum() + 1e-8)

        # degree CE
        Bn, Nn, C = logits_deg.shape
        deg_ce_elem = F.cross_entropy(
            logits_deg.view(Bn * Nn, C),
            true_deg_class.view(Bn * Nn),
            weight=self.degree_class_weight,
            reduction='none'
        ).view(Bn, Nn)
        # Train degree classes only where a node exists. Padded/non-existent
        # rows have degree 0 by construction and otherwise dominate the CE.
        deg_weight = target_exist * w.view(-1, 1)
        loss_deg_ce = (deg_ce_elem * deg_weight).sum() / (deg_weight.sum() + 1e-8)

        # label CE (NEW)
        if (self.label_head is not None) and (true_lab_class is not None):
            Bl, Nl, Cl = logits_lab.shape
            lab_ce_elem = F.cross_entropy(
                logits_lab.view(Bl * Nl, Cl),
                true_lab_class.view(Bl * Nl),
                weight=self.label_class_weight,
                reduction='none'
            ).view(Bl, Nl)
            label_weight = target_exist * w.view(-1, 1)
            loss_lab_ce = (lab_ce_elem * label_weight).sum() / (label_weight.sum() + 1e-8)
        else:
            loss_lab_ce = pred_eps.new_zeros(())

        if cond_logits_exist is not None:
            cond_exist_elem = F.binary_cross_entropy_with_logits(
                cond_logits_exist,
                target_exist,
                pos_weight=self.exist_pos_weight,
                reduction='none',
            )
            loss_condition_exist = cond_exist_elem.mean()
        else:
            loss_condition_exist = pred_eps.new_zeros(())

        if cond_logits_deg is not None:
            Bp, Np, Cp = cond_logits_deg.shape
            cond_deg_ce_elem = F.cross_entropy(
                cond_logits_deg.view(Bp * Np, Cp),
                true_deg_class.view(Bp * Np),
                weight=self.degree_class_weight,
                reduction='none',
            ).view(Bp, Np)
            loss_condition_deg_ce = (
                (cond_deg_ce_elem * target_exist).sum()
                / (target_exist.sum() + 1e-8)
            )
        else:
            loss_condition_deg_ce = pred_eps.new_zeros(())

        if (cond_logits_lab is not None) and (true_lab_class is not None):
            Bp, Np, Cp = cond_logits_lab.shape
            cond_lab_ce_elem = F.cross_entropy(
                cond_logits_lab.view(Bp * Np, Cp),
                true_lab_class.view(Bp * Np),
                weight=self.label_class_weight,
                reduction='none',
            ).view(Bp, Np)
            loss_condition_lab_ce = (
                (cond_lab_ce_elem * target_exist).sum()
                / (target_exist.sum() + 1e-8)
            )
        else:
            loss_condition_lab_ce = pred_eps.new_zeros(())

        loss_condition_x0 = (
            loss_condition_x0_mse
            + loss_condition_exist
            + loss_condition_deg_ce
            + loss_condition_lab_ce
        )

        weighted_eps = self.lambda_recon_importance * loss_eps
        weighted_x0 = self.lambda_x0_importance * loss_x0
        weighted_condition_x0 = self.lambda_condition_x0_importance * loss_condition_x0
        weighted_exist = self.lambda_node_exist_importance * loss_exist
        weighted_deg_ce = self.lambda_degree_importance * loss_deg_ce
        weighted_lab_ce = self.lambda_label_importance * loss_lab_ce
        weighted_cons = lam_cons * loss_cons

        total_loss = (
            weighted_eps
            + weighted_x0
            + weighted_condition_x0
            + weighted_exist
            + weighted_deg_ce
            + weighted_lab_ce
            + weighted_cons
        )
        return {
            "total": total_loss,
            "eps": loss_eps,
            "x0": loss_x0,
            "condition_x0": loss_condition_x0,
            "exist": loss_exist,
            "deg_ce": loss_deg_ce,
            "lab_ce": loss_lab_ce,
            "cons": loss_cons,
            "weighted_eps": weighted_eps,
            "weighted_x0": weighted_x0,
            "weighted_condition_x0": weighted_condition_x0,
            "weighted_exist": weighted_exist,
            "weighted_deg_ce": weighted_deg_ce,
            "weighted_lab_ce": weighted_lab_ce,
            "weighted_cons": weighted_cons,
        }

    # ---------------------------------------------------------------------------


    # ---------------------------------------------------------------------------
    # TRAINING STEP – uses the dict
        # ---------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        """
        Perform a single training step with optional edge and edge-label supervision.
        """
        input_examples, global_condition, edge_idx, edge_labels, node_mask, edge_cls_idx, edge_cls_lbl, dist_cls_idx, dist_cls_lbl = self._unpack_batch(batch)

        diffusion_time_step = self.sample_diffusion_time(input_examples.size(0), self.device)
        clean_adj = None
        edge_state = None
        if self.use_joint_edge_denoising and edge_idx.numel() > 0:
            clean_adj = self._dense_adjacency_from_pairs(edge_idx, edge_labels, node_mask)
            edge_state = self._corrupt_adjacency_state(
                clean_adj,
                node_mask,
                diffusion_time_step,
            )

        # replace the current forward() calls in both training_step and validation_step:
        need_latents = (
            self.use_edge_supervision
            or (getattr(self, "edge_label_head", None) is not None)
            or (getattr(self, "distance_head", None) is not None)
        )

        if need_latents:
            pred_eps, logits_deg, logits_exist, logits_lab, latent_tokens, eps, sigma_t = \
                self.forward(
                    input_examples,
                    global_condition,
                    diffusion_time_step,
                    return_latents=True,
                    edge_state=edge_state,
                )
        else:
            pred_eps, logits_deg, logits_exist, logits_lab, eps, sigma_t = \
                self.forward(
                    input_examples,
                    global_condition,
                    diffusion_time_step,
                    edge_state=edge_state,
                )

        condition_x0_prediction = None
        if self.lambda_condition_x0_importance > 0:
            condition_x0_prediction = self.predict_condition_x0(
                global_condition,
                return_logits=True,
            )

        # core diffusion + node heads
        losses = self.compute_weighted_loss(
            (pred_eps, logits_deg, logits_exist, logits_lab, eps, sigma_t),
            input_examples,
            condition_x0_prediction=condition_x0_prediction,
        )
        total_loss = losses["total"]

        if self.use_joint_edge_denoising and clean_adj is not None and need_latents:
            dense_edge_loss, dense_edge_acc = self._dense_edge_reconstruction_loss(
                latent_tokens,
                clean_adj,
                node_mask,
                sigma_t,
            )
            total_loss = total_loss + self.lambda_edge_importance * dense_edge_loss
            self.log("train_joint_edge_loss", dense_edge_loss, on_step=False, on_epoch=True)
            self.log("train_joint_edge_acc", dense_edge_acc, on_step=False, on_epoch=True)

        # ------- edge existence (binary) -------
        if self.use_edge_supervision and edge_idx.numel() > 0:
            b, i, j = edge_idx.unbind(1)
            valid = node_mask[b, i] & node_mask[b, j]
            if valid.any():
                b, i, j = b[valid], i[valid], j[valid]
                h_i = latent_tokens[b, i]
                h_j = latent_tokens[b, j]
                edge_logits = self.edge_head(h_i, h_j)
                edge_targets = edge_labels[valid].float()
                n_pos = edge_targets.sum()
                n_neg = edge_targets.numel() - n_pos
                w_pair = 1.0 / (sigma_t.squeeze(-1).squeeze(-1)[b]**2 + 1e-4)
                if self.balance_edge_loss and n_pos > 0 and n_neg > 0:
                    pos_weight = torch.clamp(n_neg / (n_pos + 1e-8), min=1.0, max=20.0)
                    edge_loss_elem = F.binary_cross_entropy_with_logits(
                        edge_logits,
                        edge_targets,
                        pos_weight=pos_weight,
                        reduction='none',
                    )
                else:
                    edge_loss_elem = F.binary_cross_entropy_with_logits(
                        edge_logits,
                        edge_targets,
                        reduction='none',
                    )
                edge_loss = (edge_loss_elem * w_pair).sum() / (w_pair.sum() + 1e-8)
                total_loss = total_loss + self.lambda_edge_importance * edge_loss

                with torch.no_grad():
                    edge_pred = (torch.sigmoid(edge_logits) > 0.5).float()
                    edge_acc = (edge_pred == edge_targets).float().mean()

                self.log("train_edge_loss", edge_loss, on_step=False, on_epoch=True)
                self.log("train_edge_acc",  edge_acc,  on_step=False, on_epoch=True)

                if self.use_joint_edge_denoising:
                    sigma_clean = torch.tensor(self.sampling_final_sigma, device=self.device)
                    t_clean = self._t_from_sigma(sigma_clean).expand(input_examples.size(0), 1)
                    _, _, _, _, base_latent_tokens, _, _ = self.forward(
                        input_examples,
                        global_condition,
                        t_clean,
                        return_latents=True,
                        add_noise=False,
                        edge_state=None,
                    )
                    base_edge_logits = self.edge_head(base_latent_tokens[b, i], base_latent_tokens[b, j])
                    if self.balance_edge_loss and n_pos > 0 and n_neg > 0:
                        pos_weight = torch.clamp(n_neg / (n_pos + 1e-8), min=1.0, max=20.0)
                        base_edge_loss = F.binary_cross_entropy_with_logits(
                            base_edge_logits,
                            edge_targets,
                            pos_weight=pos_weight,
                        )
                    else:
                        base_edge_loss = F.binary_cross_entropy_with_logits(
                            base_edge_logits,
                            edge_targets,
                        )
                    total_loss = total_loss + 0.5 * self.lambda_edge_importance * base_edge_loss
                    with torch.no_grad():
                        base_edge_pred = (torch.sigmoid(base_edge_logits) > 0.5).float()
                        base_edge_acc = (base_edge_pred == edge_targets).float().mean()
                    self.log("train_base_edge_loss", base_edge_loss, on_step=False, on_epoch=True)
                    self.log("train_base_edge_acc", base_edge_acc, on_step=False, on_epoch=True)

                if self.lambda_clean_edge_importance > 0:
                    sigma_clean = torch.tensor(self.sampling_final_sigma, device=self.device)
                    t_clean = self._t_from_sigma(sigma_clean).expand(input_examples.size(0), 1)
                    _, _, _, _, clean_latent_tokens, _, _ = self.forward(
                        input_examples,
                        global_condition,
                        t_clean,
                        return_latents=True,
                        add_noise=False,
                        edge_state=None,
                    )
                    clean_h_i = clean_latent_tokens[b, i]
                    clean_h_j = clean_latent_tokens[b, j]
                    clean_edge_logits = self.edge_head(clean_h_i, clean_h_j)
                    if self.balance_edge_loss and n_pos > 0 and n_neg > 0:
                        pos_weight = torch.clamp(n_neg / (n_pos + 1e-8), min=1.0, max=20.0)
                        clean_edge_loss = F.binary_cross_entropy_with_logits(
                            clean_edge_logits,
                            edge_targets,
                            pos_weight=pos_weight,
                        )
                    else:
                        clean_edge_loss = F.binary_cross_entropy_with_logits(
                            clean_edge_logits,
                            edge_targets,
                        )
                    total_loss = total_loss + self.lambda_clean_edge_importance * clean_edge_loss
                    with torch.no_grad():
                        clean_edge_pred = (torch.sigmoid(clean_edge_logits) > 0.5).float()
                        clean_edge_acc = (clean_edge_pred == edge_targets).float().mean()
                    self.log("train_clean_edge_loss", clean_edge_loss, on_step=False, on_epoch=True)
                    self.log("train_clean_edge_acc", clean_edge_acc, on_step=False, on_epoch=True)

        # ------- edge label (multi-class) -------
        if (getattr(self, "edge_label_head", None) is not None) and (edge_cls_idx.numel() > 0):
            b2, i2, j2 = edge_cls_idx.unbind(1)
            valid2 = node_mask[b2, i2] & node_mask[b2, j2]
            if valid2.any():
                b2, i2, j2 = b2[valid2], i2[valid2], j2[valid2]
                h_i2 = latent_tokens[b2, i2]
                h_j2 = latent_tokens[b2, j2]
                logits_edge_lab = self.edge_label_head(h_i2, h_j2)         # (E, C)
                w_pair2 = 1.0 / (sigma_t.squeeze(-1).squeeze(-1)[b2]**2 + 1e-4)
                loss_edge_lab_elem = F.cross_entropy(
                    logits_edge_lab,
                    edge_cls_lbl[valid2].long(),
                    reduction='none',
                )
                loss_edge_lab = (loss_edge_lab_elem * w_pair2).sum() / (w_pair2.sum() + 1e-8)
                total_loss = total_loss + self.lambda_edge_label_importance * loss_edge_lab

                with torch.no_grad():
                    acc_edge_lab = (logits_edge_lab.argmax(-1) == edge_cls_lbl[valid2].long()).float().mean()

                self.log("train_edge_label_loss", loss_edge_lab, on_step=False, on_epoch=True)
                self.log("train_edge_label_acc",  acc_edge_lab,  on_step=False, on_epoch=True)

        # ------- hop distance (multi-class 0..3) -------
        if (getattr(self, "distance_head", None) is not None) and (dist_cls_idx.numel() > 0):
            b3, i3, j3 = dist_cls_idx.unbind(1)
            valid3 = node_mask[b3, i3] & node_mask[b3, j3]
            if valid3.any():
                b3, i3, j3 = b3[valid3], i3[valid3], j3[valid3]
                h_i3 = latent_tokens[b3, i3]
                h_j3 = latent_tokens[b3, j3]
                logits_dist = self.distance_head(h_i3, h_j3)   # (E, 4)
                w_pair = 1.0 / (sigma_t.squeeze(-1).squeeze(-1)[b3]**2 + 1e-4)

                loss_dist_elem = F.cross_entropy(logits_dist, dist_cls_lbl[valid3].long(), reduction='none')
                loss_dist = (loss_dist_elem * w_pair).sum() / (w_pair.sum() + 1e-8)
                total_loss = total_loss + self.lambda_distance_importance * loss_dist


                self.log("train_dist_loss", loss_dist, on_step=False, on_epoch=True)


        # logging
        self.log("train_total", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_recon", losses["eps"],    on_step=False, on_epoch=True)
        self.log("train_x0", losses["x0"], on_step=False, on_epoch=True)
        self.log("train_condition_x0", losses["condition_x0"], on_step=False, on_epoch=True)
        self.log("train_deg_ce", losses["deg_ce"],on_step=False, on_epoch=True)
        self.log("train_exist",  losses["exist"], on_step=False, on_epoch=True)
        self.log("train_label_ce", losses["lab_ce"], on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Validation with optional edge and edge-label supervision.
        """
        input_examples, global_condition, edge_idx, edge_labels, node_mask, edge_cls_idx, edge_cls_lbl, dist_cls_idx, dist_cls_lbl = self._unpack_batch(batch)

        diffusion_time_step = self.sample_diffusion_time(input_examples.size(0), self.device)
        clean_adj = None
        edge_state = None
        if self.use_joint_edge_denoising and edge_idx.numel() > 0:
            clean_adj = self._dense_adjacency_from_pairs(edge_idx, edge_labels, node_mask)
            edge_state = self._corrupt_adjacency_state(
                clean_adj,
                node_mask,
                diffusion_time_step,
            )

# replace the current forward() calls in both training_step and validation_step:
        need_latents = (
            self.use_edge_supervision
            or (getattr(self, "edge_label_head", None) is not None)
            or (getattr(self, "distance_head", None) is not None)
        )

        if need_latents:
            pred_eps, logits_deg, logits_exist, logits_lab, latent_tokens, eps, sigma_t = \
                self.forward(
                    input_examples,
                    global_condition,
                    diffusion_time_step,
                    return_latents=True,
                    edge_state=edge_state,
                )
        else:
            pred_eps, logits_deg, logits_exist, logits_lab, eps, sigma_t = \
                self.forward(
                    input_examples,
                    global_condition,
                    diffusion_time_step,
                    edge_state=edge_state,
                )


        condition_x0_prediction = None
        if self.lambda_condition_x0_importance > 0:
            condition_x0_prediction = self.predict_condition_x0(
                global_condition,
                return_logits=True,
            )

        losses = self.compute_weighted_loss(
            (pred_eps, logits_deg, logits_exist, logits_lab, eps, sigma_t),
            input_examples,
            condition_x0_prediction=condition_x0_prediction,
        )
        total_loss = losses["total"]

        if self.use_joint_edge_denoising and clean_adj is not None and need_latents:
            dense_edge_loss, dense_edge_acc = self._dense_edge_reconstruction_loss(
                latent_tokens,
                clean_adj,
                node_mask,
                sigma_t,
            )
            total_loss = total_loss + self.lambda_edge_importance * dense_edge_loss
            self.log("val_joint_edge_loss", dense_edge_loss, on_step=False, on_epoch=True)
            self.log("val_joint_edge_acc", dense_edge_acc, on_step=False, on_epoch=True)

        # edge existence
        if self.use_edge_supervision and edge_idx.numel() > 0:
            b, i, j = edge_idx.unbind(1)
            valid = node_mask[b, i] & node_mask[b, j]
            if valid.any():
                b, i, j = b[valid], i[valid], j[valid]
                h_i = latent_tokens[b, i]
                h_j = latent_tokens[b, j]
                edge_logits = self.edge_head(h_i, h_j)
                edge_targets = edge_labels[valid].float()
                n_pos = edge_targets.sum()
                n_neg = edge_targets.numel() - n_pos
                w_pair = 1.0 / (sigma_t.squeeze(-1).squeeze(-1)[b]**2 + 1e-4)
                if self.balance_edge_loss and n_pos > 0 and n_neg > 0:
                    pos_weight = torch.clamp(n_neg / (n_pos + 1e-8), min=1.0, max=20.0)
                    edge_loss_elem = F.binary_cross_entropy_with_logits(
                        edge_logits,
                        edge_targets,
                        pos_weight=pos_weight,
                        reduction='none',
                    )
                else:
                    edge_loss_elem = F.binary_cross_entropy_with_logits(
                        edge_logits,
                        edge_targets,
                        reduction='none',
                    )
                edge_loss = (edge_loss_elem * w_pair).sum() / (w_pair.sum() + 1e-8)
                total_loss = total_loss + self.lambda_edge_importance * edge_loss

                with torch.no_grad():
                    edge_pred = (torch.sigmoid(edge_logits) > 0.5).float()
                    edge_acc = (edge_pred == edge_targets).float().mean()

                self.log("val_edge_loss", edge_loss, on_step=False, on_epoch=True)
                self.log("val_edge_acc",  edge_acc,  on_step=False, on_epoch=True)

                if self.use_joint_edge_denoising:
                    sigma_clean = torch.tensor(self.sampling_final_sigma, device=self.device)
                    t_clean = self._t_from_sigma(sigma_clean).expand(input_examples.size(0), 1)
                    _, _, _, _, base_latent_tokens, _, _ = self.forward(
                        input_examples,
                        global_condition,
                        t_clean,
                        return_latents=True,
                        add_noise=False,
                        edge_state=None,
                    )
                    base_edge_logits = self.edge_head(base_latent_tokens[b, i], base_latent_tokens[b, j])
                    if self.balance_edge_loss and n_pos > 0 and n_neg > 0:
                        pos_weight = torch.clamp(n_neg / (n_pos + 1e-8), min=1.0, max=20.0)
                        base_edge_loss = F.binary_cross_entropy_with_logits(
                            base_edge_logits,
                            edge_targets,
                            pos_weight=pos_weight,
                        )
                    else:
                        base_edge_loss = F.binary_cross_entropy_with_logits(
                            base_edge_logits,
                            edge_targets,
                        )
                    total_loss = total_loss + 0.5 * self.lambda_edge_importance * base_edge_loss
                    with torch.no_grad():
                        base_edge_pred = (torch.sigmoid(base_edge_logits) > 0.5).float()
                        base_edge_acc = (base_edge_pred == edge_targets).float().mean()
                    self.log("val_base_edge_loss", base_edge_loss, on_step=False, on_epoch=True)
                    self.log("val_base_edge_acc", base_edge_acc, on_step=False, on_epoch=True)

                if self.lambda_clean_edge_importance > 0:
                    sigma_clean = torch.tensor(self.sampling_final_sigma, device=self.device)
                    t_clean = self._t_from_sigma(sigma_clean).expand(input_examples.size(0), 1)
                    _, _, _, _, clean_latent_tokens, _, _ = self.forward(
                        input_examples,
                        global_condition,
                        t_clean,
                        return_latents=True,
                        add_noise=False,
                        edge_state=None,
                    )
                    clean_h_i = clean_latent_tokens[b, i]
                    clean_h_j = clean_latent_tokens[b, j]
                    clean_edge_logits = self.edge_head(clean_h_i, clean_h_j)
                    if self.balance_edge_loss and n_pos > 0 and n_neg > 0:
                        pos_weight = torch.clamp(n_neg / (n_pos + 1e-8), min=1.0, max=20.0)
                        clean_edge_loss = F.binary_cross_entropy_with_logits(
                            clean_edge_logits,
                            edge_targets,
                            pos_weight=pos_weight,
                        )
                    else:
                        clean_edge_loss = F.binary_cross_entropy_with_logits(
                            clean_edge_logits,
                            edge_targets,
                        )
                    total_loss = total_loss + self.lambda_clean_edge_importance * clean_edge_loss
                    with torch.no_grad():
                        clean_edge_pred = (torch.sigmoid(clean_edge_logits) > 0.5).float()
                        clean_edge_acc = (clean_edge_pred == edge_targets).float().mean()
                    self.log("val_clean_edge_loss", clean_edge_loss, on_step=False, on_epoch=True)
                    self.log("val_clean_edge_acc", clean_edge_acc, on_step=False, on_epoch=True)

        # edge label (multi-class)
        if (getattr(self, "edge_label_head", None) is not None) and (edge_cls_idx.numel() > 0):
            b2, i2, j2 = edge_cls_idx.unbind(1)
            valid2 = node_mask[b2, i2] & node_mask[b2, j2]
            if valid2.any():
                b2, i2, j2 = b2[valid2], i2[valid2], j2[valid2]
                h_i2 = latent_tokens[b2, i2]
                h_j2 = latent_tokens[b2, j2]
                logits_edge_lab = self.edge_label_head(h_i2, h_j2)
                w_pair2 = 1.0 / (sigma_t.squeeze(-1).squeeze(-1)[b2]**2 + 1e-4)
                loss_edge_lab_elem = F.cross_entropy(
                    logits_edge_lab,
                    edge_cls_lbl[valid2].long(),
                    reduction='none',
                )
                loss_edge_lab = (loss_edge_lab_elem * w_pair2).sum() / (w_pair2.sum() + 1e-8)
                total_loss = total_loss + self.lambda_edge_label_importance * loss_edge_lab

                with torch.no_grad():
                    acc_edge_lab = (logits_edge_lab.argmax(-1) == edge_cls_lbl[valid2].long()).float().mean()

                self.log("val_edge_label_loss", loss_edge_lab, on_step=False, on_epoch=True)
                self.log("val_edge_label_acc",  acc_edge_lab,  on_step=False, on_epoch=True)


        # hop distance (multi-class)
        if (getattr(self, "distance_head", None) is not None) and (dist_cls_idx.numel() > 0):
            b3, i3, j3 = dist_cls_idx.unbind(1)
            valid3 = node_mask[b3, i3] & node_mask[b3, j3]
            if valid3.any():
                b3, i3, j3 = b3[valid3], i3[valid3], j3[valid3]
                h_i3 = latent_tokens[b3, i3]
                h_j3 = latent_tokens[b3, j3]
                logits_dist = self.distance_head(h_i3, h_j3)
                w_pair = 1.0 / (sigma_t.squeeze(-1).squeeze(-1)[b3]**2 + 1e-4)
                loss_dist_elem = F.cross_entropy(logits_dist, dist_cls_lbl[valid3].long(), reduction='none')
                loss_dist = (loss_dist_elem * w_pair).sum() / (w_pair.sum() + 1e-8)
                total_loss = total_loss + self.lambda_distance_importance * loss_dist
                self.log("val_dist_loss", loss_dist, on_step=False, on_epoch=True)


        self.log("val_total", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_recon", losses["eps"],    on_step=False, on_epoch=True)
        self.log("val_x0", losses["x0"], on_step=False, on_epoch=True)
        self.log("val_condition_x0", losses["condition_x0"], on_step=False, on_epoch=True)
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
            if getattr(self, "lambda_clean_edge_importance", 0.0) > 0:
                train_extra["clean_edge"] = getattr(self, "train_clean_edge_loss", [])
                val_extra["clean_edge"] = getattr(self, "val_clean_edge_loss", [])
        if getattr(self, "use_joint_edge_denoising", False):
            train_extra["joint_edge"] = getattr(self, "train_joint_edge_loss", [])
            val_extra["joint_edge"] = getattr(self, "val_joint_edge_loss", [])
        if getattr(self, "edge_label_head", None) is not None:
            train_extra["edge_label"] = getattr(self, "train_edge_label_loss", [])
            val_extra["edge_label"]   = getattr(self, "val_edge_label_loss", [])

        if (
            getattr(self, "distance_head", None) is not None
            and getattr(self, "lambda_distance_importance", 0.0) > 0
        ):
            train_extra["dist_loss"] = getattr(self, "train_dist_loss", [])
            # train_extra["dist_acc"]  = getattr(self, "train_dist_acc",  [])
            val_extra["dist_loss"]   = getattr(self, "val_dist_loss",   [])
            # val_extra["dist_acc"]    = getattr(self, "val_dist_acc",    [])

        plot_metrics(
            train_metrics={
                "total": self.train_losses,
                "deg_ce": self.train_deg_ce,
                **({"recon": self.train_recon} if getattr(self, "lambda_recon_importance", 0.0) > 0 else {}),
                **({"x0": self.train_x0} if getattr(self, "lambda_x0_importance", 0.0) > 0 else {}),
                **({"condition_x0": self.train_condition_x0} if getattr(self, "lambda_condition_x0_importance", 0.0) > 0 else {}),
                "exist": self.train_exist,
                **({"label_ce": self.train_label_ce} if getattr(self, "label_head", None) is not None else {}),
                **train_extra,
            },
            val_metrics={
                "total": self.val_losses,
                "deg_ce": self.val_deg_ce,
                **({"recon": self.val_recon} if getattr(self, "lambda_recon_importance", 0.0) > 0 else {}),
                **({"x0": self.val_x0} if getattr(self, "lambda_x0_importance", 0.0) > 0 else {}),
                **({"condition_x0": self.val_condition_x0} if getattr(self, "lambda_condition_x0_importance", 0.0) > 0 else {}),
                "exist": self.val_exist,
                **({"label_ce": self.val_label_ce} if getattr(self, "label_head", None) is not None else {}),
                **val_extra,
            },
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


        train_losses = []
        val_losses = []

        # Backbone latents are treated as fixed features for guidance training.
        with torch.no_grad():
            _, _, _, _, lat_tr, _, _ = self.forward(
                X_tr, Y_tr, T_tr, return_latents=True, add_noise=False
            )
            _, _, _, _, lat_val, _, _ = self.forward(
                X_val, Y_val, T_val, return_latents=True, add_noise=False
            )
            lat_tr = lat_tr.mean(dim=1).detach()
            lat_val = lat_val.mean(dim=1).detach()

        for _ in range(epochs):
            logits = self.guidance_classifier(lat_tr)
            loss = F.cross_entropy(logits, L_tr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

            with torch.no_grad():
                logits_val = self.guidance_classifier(lat_val)
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
        self._last_deg_classes = None
        self._last_lab_classes = None
        self._last_edge_probabilities = None

        # --- sigma schedule consistent with training metadata ---
        self._ensure_schedule_metadata()
        # sigmas = torch.linspace(
        #     self.sigma_max,
        #     self.sampling_final_sigma,
        #     total_steps,
        #     device=device
        # )  # (T,)
        sigmas = self._karras_sigmas(total_steps).to(device)

        # Start from a neutral table plus noise. Optionally blend in a clean
        # condition-only prior, then let diffusion refine it.
        sigma_start = sigmas[0]
        noise_scale = torch.ones(
            B,
            self.number_of_rows_per_example,
            self.input_feature_dimension,
            device=device,
        ) * sigma_start

        noise_scale[..., self.important_feature_index] /= self.noise_degree_factor

        if self.label_feature_index is not None and self.label_feature_index < self.input_feature_dimension:
            noise_scale[..., self.label_feature_index] /= self.noise_label_factor

        blend = max(0.0, min(1.0, self.condition_x0_sampling_blend))
        prior_logits_exist = prior_logits_deg = prior_logits_lab = None

        base = torch.zeros(
            B,
            self.number_of_rows_per_example,
            self.input_feature_dimension,
            device=device,
        )

        base[..., 0] = 0.5
        base[..., self.important_feature_index] = 0.5

        if self.label_feature_index is not None and self.label_feature_index < self.input_feature_dimension:
            base[..., self.label_feature_index] = 0.5

        noise = torch.randn_like(base) * noise_scale

        if blend > 0:
            condition_x0, prior_logits_exist, prior_logits_deg, prior_logits_lab = self.predict_condition_x0(
                global_condition,
                project_discrete=True,
                return_logits=True,
            )
            x = blend * (condition_x0 + noise) + (1.0 - blend) * (base + noise)
        else:
            x = base + noise

        edge_state = None
        if self.use_joint_edge_denoising and getattr(self, "edge_head", None) is not None:
            initial_density = min(0.25, max(0.02, 2.0 / max(1, self.number_of_rows_per_example)))
            edge_state = (torch.rand(
                B,
                self.number_of_rows_per_example,
                self.number_of_rows_per_example,
                device=device,
            ) < initial_density).to(x.dtype)
            edge_state = torch.maximum(edge_state, edge_state.transpose(1, 2))
            eye = torch.eye(self.number_of_rows_per_example, dtype=torch.bool, device=device).unsqueeze(0)
            edge_state = edge_state.masked_fill(eye, 0.0)

        discrete_mode = getattr(self, "discrete_diffusion_mode", "none")
        if discrete_mode != "none" and blend <= 0:
            if discrete_mode == "absorbing":
                deg_cls = torch.full(
                    (B, self.number_of_rows_per_example),
                    self._degree_mask_class(),
                    dtype=torch.long,
                    device=device,
                )
            else:
                deg_cls = torch.randint(
                    low=0,
                    high=self.max_degree + 1,
                    size=(B, self.number_of_rows_per_example),
                    device=device,
                )
            x[..., self.important_feature_index] = self._scaled_degree_from_class(deg_cls).to(x.dtype)
            if self.label_feature_index is not None and self.max_label is not None:
                if discrete_mode == "absorbing":
                    lab_cls = torch.full(
                        (B, self.number_of_rows_per_example),
                        self._label_mask_class(),
                        dtype=torch.long,
                        device=device,
                    )
                else:
                    lab_cls = torch.randint(
                        low=0,
                        high=self.max_label + 1,
                        size=(B, self.number_of_rows_per_example),
                        device=device,
                    )
                x[..., self.label_feature_index] = self._scaled_label_from_class(lab_cls).to(x.dtype)

        # autocast for speed on GPU (skip on unsupported backends)
        if device.type == "cuda":
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
        elif device.type == "mps":
            autocast_ctx = torch.autocast(device_type="mps", dtype=torch.float16)
        else:
            autocast_ctx = contextlib.nullcontext()

        with (torch.inference_mode() if not self.use_guidance else contextlib.nullcontext()):
            with autocast_ctx:
                # The epsilon head is trained only on continuous feature columns.
                # Existence, degree and node-label columns are handled by their
                # dedicated heads, so their epsilon channels are intentionally
                # unsupervised and must not drive the ODE sampler.
                denoise_mask = torch.ones(
                    (1, 1, self.input_feature_dimension),
                    dtype=x.dtype,
                    device=device,
                )
                denoise_mask[..., 0] = 0.0
                discrete_mode = getattr(self, "discrete_diffusion_mode", "none")
                gaussian_discrete = self.denoise_discrete_channels and discrete_mode == "none"
                if not gaussian_discrete:
                    denoise_mask[..., self.important_feature_index] = 0.0
                if (
                    (not gaussian_discrete)
                    and self.label_feature_index is not None
                    and self.label_feature_index < self.input_feature_dimension
                ):
                    denoise_mask[..., self.label_feature_index] = 0.0

                for i in range(total_steps - 1):
                    sigma_t   = sigmas[i]
                    sigma_next = sigmas[i + 1]

                    # time embeddings for current and next sigmas
                    t_norm  = self._t_from_sigma(sigma_t).expand(B, 1)
                    t_next  = self._t_from_sigma(sigma_next).expand(B, 1)

                    # predictor (Euler)
                    pred_eps, _, _, _, _, _ = self.forward(
                        x,
                        global_condition,
                        t_norm,
                        add_noise=False,
                        edge_state=edge_state,
                    )
                    pred_eps = pred_eps * denoise_mask
                    step = (sigma_t - sigma_next) * pred_eps
                    x_pred = x - step

                    # corrector (Heun / RK2)
                    pred_eps_next, logits_deg_next, _, logits_lab_next, latent_next, _, _ = self.forward(
                        x_pred,
                        global_condition,
                        t_next,
                        add_noise=False,
                        return_latents=True,
                        edge_state=edge_state,
                    )
                    pred_eps_next = pred_eps_next * denoise_mask
                    step_corr = (sigma_t - sigma_next) * 0.5 * (pred_eps + pred_eps_next)
                    x_corr = x - step_corr

                    # adopt corrected state
                    x = x_corr
                    project_sigma_threshold = float(
                        getattr(self, "discrete_projection_sigma_threshold", 0.25)
                    )
                    should_project_discrete = (
                        getattr(self, "discrete_diffusion_mode", "none") != "none"
                        and float(sigma_next.detach().cpu()) <= project_sigma_threshold
                    )
                    if should_project_discrete:
                        x = self._project_discrete_columns_from_logits(
                            x,
                            logits_deg_next,
                            logits_lab_next,
                        )

                    if edge_state is not None:
                        edge_probs = self._dense_edge_probabilities_from_latents(latent_next)
                        momentum = min(max(self.joint_edge_update_momentum, 0.0), 1.0)
                        edge_state = momentum * edge_state + (1.0 - momentum) * edge_probs
                        if float(sigma_next.detach().cpu()) <= project_sigma_threshold:
                            edge_state = (edge_state >= self.joint_edge_threshold).to(edge_state.dtype)

                    # optional: classifier guidance here (rebuild graph as needed)
                    if self.use_guidance and self.guidance_classifier is not None:
                        x.requires_grad_(True)
                        _, _, _, _, lat, _, _ = self.forward(
                            x,
                            global_condition,
                            t_next,
                            return_latents=True,
                            add_noise=False,
                            edge_state=edge_state,
                        )
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


        # Optional: project discrete channels using the auxiliary heads once at (approx) t=0
        if use_heads_projection:
            with torch.inference_mode():
                if blend > 0 and prior_logits_exist is not None and prior_logits_deg is not None:
                    logits_exist = prior_logits_exist
                    logits_deg = prior_logits_deg
                    logits_lab = prior_logits_lab
                else:
                    sigma_proj = torch.tensor(self.sampling_final_sigma, device=device)
                    t0 = self._t_from_sigma(sigma_proj).expand(B, 1)

                    _, logits_deg, logits_exist, logits_lab, final_latents, _, _ = self.forward(
                        x,
                        global_condition,
                        t0,
                        return_latents=True,
                        add_noise=False,
                        edge_state=edge_state,
                    )
                    if edge_state is not None:
                        edge_state = self._dense_edge_probabilities_from_latents(final_latents)

                if self.project_existence_during_sampling:
                    exist_bin = (torch.sigmoid(logits_exist) >= exist_threshold).to(x.dtype)
                    x[..., 0] = exist_bin

                if self.project_degree_during_sampling:
                    deg_cls = torch.argmax(logits_deg, dim=-1)
                    x[..., self.important_feature_index] = self._scaled_degree_from_class(deg_cls).to(x.dtype)
                    self._last_deg_classes = deg_cls.detach().cpu()

                if (
                    self.project_label_during_sampling
                    and logits_lab is not None
                    and self.label_feature_index is not None
                ):
                    lab_cls = torch.argmax(logits_lab, dim=-1)
                    x[..., self.label_feature_index] = self._scaled_label_from_class(lab_cls).to(x.dtype)
                    self._last_lab_classes = lab_cls.detach().cpu()
        if edge_state is not None:
            self._last_edge_probabilities = edge_state.detach().cpu()
        else:
            self._last_edge_probabilities = None
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
        edge_label_targets: Optional[np.ndarray] = None,               # NEW
        distance_pairs: Optional[List[Tuple[int,int,int]]] = None,
        distance_targets: Optional[np.ndarray] = None,
        random_row_permutation: bool = False,
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.random_row_permutation = bool(random_row_permutation)
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

        self.dist_cls_idx_by_graph = {b: [] for b in range(len(X))}
        self.dist_cls_lbl_by_graph = {b: [] for b in range(len(X))}
        if distance_pairs is not None and distance_targets is not None:
            for (b, i, j), c in zip(distance_pairs, distance_targets):
                self.dist_cls_idx_by_graph[b].append((i, j))
                self.dist_cls_lbl_by_graph[b].append(int(c))

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

        dist_cls_idxs = torch.tensor(self.dist_cls_idx_by_graph[idx], dtype=torch.long) \
                        if self.dist_cls_idx_by_graph[idx] else torch.empty((0, 2), dtype=torch.long)
        dist_cls_lbls = torch.tensor(self.dist_cls_lbl_by_graph[idx], dtype=torch.long) \
                        if self.dist_cls_lbl_by_graph[idx] else torch.empty((0,), dtype=torch.long)

        if self.random_row_permutation:
            perm = torch.randperm(x.shape[0])
            old_to_new = torch.empty_like(perm)
            old_to_new[perm] = torch.arange(perm.numel(), dtype=torch.long)

            x = x[perm]
            mask = mask[perm]

            def _remap_pairs(pairs: torch.Tensor) -> torch.Tensor:
                if pairs.numel() == 0:
                    return pairs
                return old_to_new[pairs]

            edge_idxs = _remap_pairs(edge_idxs)
            edge_cls_idxs = _remap_pairs(edge_cls_idxs)
            dist_cls_idxs = _remap_pairs(dist_cls_idxs)

        return x, y, edge_idxs, edge_lbls, mask, edge_cls_idxs, edge_cls_lbls, dist_cls_idxs, dist_cls_lbls
  # ← NEW trailing items


class RowPermutationTensorDataset(Dataset):
    """TensorDataset variant that randomly permutes node rows during training."""
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, random_row_permutation: bool = False):
        self.X = X
        self.Y = Y
        self.random_row_permutation = bool(random_row_permutation)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.random_row_permutation:
            x = x[torch.randperm(x.shape[0])]
        return x, self.Y[idx]

def collate_graph_with_edges(batch):
    xs, ys, masks = [], [], []
    exist_ei, exist_el = [], []
    cls_ei, cls_el = [], []
    # NEW
    dist_ei, dist_el = [], []

    for x, y, ei, el, mask, eci, ecl, dci, dcl in batch:
        xs.append(x); ys.append(y); masks.append(mask)
        exist_ei.append(ei); exist_el.append(el)
        cls_ei.append(eci); cls_el.append(ecl)
        dist_ei.append(dci); dist_el.append(dcl)

    X = torch.stack(xs); Y = torch.stack(ys); M = torch.stack(masks)

    def _stack_edges(edge_idxs_list, edge_lbls_list, force_dtype=None):
        all_idx, all_lbl = [], []
        for b, (ei, el) in enumerate(zip(edge_idxs_list, edge_lbls_list)):
            if ei.numel() == 0:
                continue
            bcol = torch.full((ei.size(0), 1), b, dtype=torch.long)
            all_idx.append(torch.cat([bcol, ei], dim=1))
            all_lbl.append(el)
        if all_idx:
            idx = torch.cat(all_idx, dim=0)
            lbl = torch.cat(all_lbl, dim=0)
            if force_dtype is not None and lbl.dtype != force_dtype:
                lbl = lbl.to(force_dtype)
            return idx, lbl
        # Empty case
        if force_dtype is None:
            force_dtype = torch.long
        return torch.empty((0,3), dtype=torch.long), torch.empty((0,), dtype=force_dtype)

    edge_idx, edge_lbl = _stack_edges(exist_ei, exist_el)
    edge_cls_idx, edge_cls_lbl = _stack_edges(cls_ei, cls_el)
    # NEW:
    dist_cls_idx, dist_cls_lbl = _stack_edges(dist_ei, dist_el)

    return X, Y, edge_idx, edge_lbl, M, edge_cls_idx, edge_cls_lbl, dist_cls_idx, dist_cls_lbl



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
                 balance_degree_loss: bool = False,
                 degree_class_weight_cap: float = 5.0,
                 lambda_node_exist_importance: float = 1.0,
                 default_exist_pos_weight: float = 1.0,
                 lambda_edge_importance: float = 1.0,
                 lambda_clean_edge_importance: float = 0.0,
                 balance_edge_loss: bool = False,
                 use_guidance: bool = False,
                early_stopping: bool = True,
                early_stop_metric: str = "val_total",
                early_stop_mode: str = "min",
                early_stop_patience: int = 10,
                early_stop_min_delta: float = 1e-4,
                restore_best_weights: bool = True,
                checkpoint_dir: Optional[str] = None,
                checkpoint_policy: str = "none",  # "best" | "all" | "none"
                lambda_consistency: float = 0.3,
                lambda_label_importance: float = 1.0,
                noise_label_factor: float = 2.0,
                balance_label_loss: bool = False,
                label_class_weight_cap: float = 5.0,
                label_feature_index: int = 2,
                lambda_edge_label_importance: float = 1.0,
                lambda_distance_importance: float = 1.0,
                max_distance_class=3,
                use_distance_supervision: bool = False,
                lambda_recon_importance: float = 1.0,
                lambda_x0_importance: float = 0.0,
                lambda_condition_x0_importance: float = 0.0,
                condition_x0_sampling_blend: float = 0.0,
                denoise_discrete_channels: bool = False,
                discrete_diffusion_mode: str = "none",
                discrete_projection_sigma_threshold: float = 0.25,
                sigma_min: float = 0.1,
                sigma_max: float = 1.0,
                sampling_final_sigma: float = 0.0,
                row_embedding_scale: float = 0.1,
                use_joint_edge_denoising: bool = False,
                joint_edge_delete_prob: float = 0.25,
                joint_edge_add_prob: float = 0.05,
                joint_edge_update_momentum: float = 0.5,
                joint_edge_threshold: float = 0.5,
                joint_edge_message_passing_steps: int = 2,
                random_row_permutation: bool = False,
                use_matched_row_loss: bool = False,
                project_existence_during_sampling: bool = True,
                project_degree_during_sampling: bool = True,
                project_label_during_sampling: bool = True,
                use_condition_node_count: bool = False,
                use_condition_degree_histogram: bool = False,
                use_condition_label_histogram: bool = False,
                use_dim_reduction: bool = False,
                dim_reduction_method: str = "pca",
                dim_reduction_components: int = 125,
                dim_reduction_keep_prefix: int = 3,
                pca_whiten: bool = False,
                use_condition_dim_reduction: bool = False,
                condition_dim_reduction_method: str = "svd",
                condition_dim_reduction_components: int = 128,
                condition_dim_reduction_keep_prefix: int = 0,
                random_state: int = 42
                

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
        self.balance_degree_loss = bool(balance_degree_loss)
        self.degree_class_weight_cap = float(degree_class_weight_cap)
        self.lambda_node_exist_importance = lambda_node_exist_importance
        self.default_exist_pos_weight = default_exist_pos_weight
        self.lambda_edge_importance = lambda_edge_importance
        self.lambda_clean_edge_importance = lambda_clean_edge_importance
        self.balance_edge_loss = bool(balance_edge_loss)
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
        self.balance_label_loss = bool(balance_label_loss)
        self.label_class_weight_cap = float(label_class_weight_cap)
        self.L_max = None
        self.lambda_edge_label_importance = lambda_edge_label_importance
        self.lambda_distance_importance = lambda_distance_importance
        self.max_distance_class=max_distance_class 

        self.use_distance_supervision = use_distance_supervision
        self.lambda_recon_importance = lambda_recon_importance
        self.lambda_x0_importance = lambda_x0_importance
        self.lambda_condition_x0_importance = lambda_condition_x0_importance
        self.condition_x0_sampling_blend = float(condition_x0_sampling_blend)
        self.denoise_discrete_channels = bool(denoise_discrete_channels)
        self.discrete_diffusion_mode = str(discrete_diffusion_mode).lower()
        self.discrete_projection_sigma_threshold = float(discrete_projection_sigma_threshold)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sampling_final_sigma = float(sampling_final_sigma)
        self.row_embedding_scale = float(row_embedding_scale)
        self.use_joint_edge_denoising = bool(use_joint_edge_denoising)
        self.joint_edge_delete_prob = float(joint_edge_delete_prob)
        self.joint_edge_add_prob = float(joint_edge_add_prob)
        self.joint_edge_update_momentum = float(joint_edge_update_momentum)
        self.joint_edge_threshold = float(joint_edge_threshold)
        self.joint_edge_message_passing_steps = max(1, int(joint_edge_message_passing_steps))
        self.random_row_permutation = bool(random_row_permutation)
        self.use_matched_row_loss = bool(use_matched_row_loss)
        self.project_existence_during_sampling = bool(project_existence_during_sampling)
        self.project_degree_during_sampling = bool(project_degree_during_sampling)
        self.project_label_during_sampling = bool(project_label_during_sampling)
        self.use_condition_node_count = bool(use_condition_node_count)
        self.node_count_model = None
        self.valid_node_counts = []
        self.fixed_node_count = None
        self._last_predicted_node_counts = None
        self._last_existence_probabilities = None
        self.use_condition_degree_histogram = bool(
            use_condition_degree_histogram
        )
        self.degree_histogram_model = None
        self.fixed_degree_histogram = None
        self._last_predicted_degree_histograms = None
        self.use_condition_label_histogram = bool(
            use_condition_label_histogram
        )
        self.label_histogram_model = None
        self.fixed_label_histogram = None
        self._last_predicted_label_histograms = None
        self.use_dim_reduction = bool(use_dim_reduction)
        self.dim_reduction_method = str(dim_reduction_method).lower()
        self.dim_reduction_components = int(dim_reduction_components)
        self.dim_reduction_keep_prefix = int(dim_reduction_keep_prefix)
        self.pca_whiten = bool(pca_whiten)
        self.use_condition_dim_reduction = bool(use_condition_dim_reduction)
        self.condition_dim_reduction_method = str(condition_dim_reduction_method).lower()
        self.condition_dim_reduction_components = int(condition_dim_reduction_components)
        self.condition_dim_reduction_keep_prefix = int(condition_dim_reduction_keep_prefix)
        self.random_state = random_state
        self.reducer = None
        self._orig_tail_dim = None
        self.condition_reducer = None
        self._orig_condition_tail_dim = None


    def _ensure_model_device(self) -> torch.device:
        """Keep the wrapped PyTorch model and newly-created tensors on one device."""
        if self.model is None:
            return self.device

        target_device = torch.device(self.device)
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = target_device

        if model_device != target_device:
            self.model.to(target_device)

        return target_device


    def _fit_scalers(self, X_array, y_array):
        self._fit_reducer(X_array)
        X_red = (
            self._apply_reducer(X_array)
            if self.use_dim_reduction and self.reducer is not None
            else X_array
        )
        y_red = self._fit_condition_reducer(y_array)

        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()

        B, N, Dred = X_red.shape
        Xf = X_red.reshape(-1, Dred)
        mask = self._flat_mask_real_rows(X_array)    # mask from original (exist col 0)

        Xf_real = Xf[mask]
        if Xf_real.shape[0] == 0:
            # fallback safety
            Xf_real = Xf

        self.x_scaler.fit(Xf_real)
        self.y_scaler.fit(y_red)

        self._x_reduced_min = Xf_real.min(axis=0)
        self._x_reduced_max = Xf_real.max(axis=0)


    def _transform_data(self, X_array, y_array):
        raw_existence = X_array[..., 0].copy()
        if self.use_dim_reduction and self.reducer is not None:
            X_array = self._apply_reducer(X_array)
        X_scaled_flat = self.x_scaler.transform(X_array.reshape(-1, X_array.shape[2]))
        X_scaled = X_scaled_flat.reshape(X_array.shape[0], X_array.shape[1], -1)
        # MinMaxScaler maps a constant existence column of all real rows
        # (usually 1) to 0. That would make every real node look absent to
        # the loss. Keep existence as the binary mask used by the model.
        X_scaled[..., 0] = raw_existence
        y_scaled = self._transform_condition_data(y_array)
        return X_scaled, y_scaled

    def _as_condition_2d(self, y_array: Any) -> np.ndarray:
        y = np.asarray(y_array)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        return y

    def _fit_condition_reducer(self, y_array: Any) -> np.ndarray:
        """Fit a separate reducer for graph-level conditioning vectors."""
        y = self._as_condition_2d(y_array)
        if not self.use_condition_dim_reduction:
            self.condition_reducer = None
            return y

        _, D = y.shape
        kp = min(max(0, self.condition_dim_reduction_keep_prefix), D)
        prefix = y[:, :kp]
        tail = y[:, kp:]

        self._orig_condition_tail_dim = tail.shape[1]
        if self._orig_condition_tail_dim == 0:
            self.condition_reducer = None
            return y

        method = self.condition_dim_reduction_method
        if method == "pca":
            max_components = min(
                int(self.condition_dim_reduction_components),
                int(tail.shape[0]),
                int(tail.shape[1]),
            )
        else:
            max_components = min(
                int(self.condition_dim_reduction_components),
                max(1, int(tail.shape[0]) - 1),
                int(tail.shape[1]),
            )
        max_components = max(1, max_components)

        if method == "pca":
            reducer = PCA(
                n_components=max_components,
                whiten=False,
                random_state=self.random_state,
            )
        elif method == "svd":
            reducer = TruncatedSVD(
                n_components=max_components,
                random_state=self.random_state,
            )
        else:
            raise ValueError("condition_dim_reduction_method must be 'pca' or 'svd'")

        reducer.fit(tail)
        self.condition_reducer = reducer
        red_tail = reducer.transform(tail)
        return np.hstack([prefix, red_tail])

    def _apply_condition_reducer(self, y_array: Any) -> np.ndarray:
        """Apply the fitted condition reducer and keep semantic prefix columns raw."""
        y = self._as_condition_2d(y_array)
        if not self.use_condition_dim_reduction or self.condition_reducer is None:
            return y

        D = y.shape[1]
        kp = min(max(0, self.condition_dim_reduction_keep_prefix), D)
        prefix = y[:, :kp]
        tail = y[:, kp:]
        red_tail = self.condition_reducer.transform(tail)
        return np.hstack([prefix, red_tail])

    def _transform_condition_data(self, y_array: Any) -> np.ndarray:
        y_reduced = self._apply_condition_reducer(y_array)
        return self.y_scaler.transform(y_reduced)

    def _inverse_transform_input(self, X_array):
        X_reduced_flat = self.x_scaler.inverse_transform(X_array.reshape(-1, X_array.shape[2]))

        if hasattr(self, "_x_reduced_min") and hasattr(self, "_x_reduced_max"):
            X_reduced_flat = np.clip(
                X_reduced_flat,
                self._x_reduced_min,
                self._x_reduced_max,
            )

        X_reduced = X_reduced_flat.reshape(X_array.shape[0], X_array.shape[1], -1)
        if self.use_dim_reduction and self.reducer is not None:
            X_orig = self._inverse_reducer(X_reduced)
        else:
            X_orig = X_reduced

        X_orig[..., 0] = np.clip(X_orig[..., 0], 0.0, 1.0)
        X_orig[..., self.important_feature_index] = np.clip(
            X_orig[..., self.important_feature_index], 0, self.D_max
        )
        return X_orig

    def _snap_degrees_to_existing_support(self, degrees: np.ndarray) -> np.ndarray:
        """Map predicted degree classes to classes observed on real nodes."""
        valid = np.asarray(
            getattr(self, "valid_existing_degree_classes", []),
            dtype=int,
        )
        if valid.size == 0:
            return degrees

        deg = np.rint(degrees).astype(int)
        deg = np.clip(
            deg,
            int(getattr(self, "min_existing_degree", valid.min())),
            int(getattr(self, "max_existing_degree", valid.max())),
        )

        # If the support is not contiguous, snap each value to the nearest
        # observed class instead of allowing unseen degree classes through.
        if valid.size != (valid.max() - valid.min() + 1):
            flat = deg.reshape(-1)
            nearest = valid[np.abs(flat[:, None] - valid[None, :]).argmin(axis=1)]
            deg = nearest.reshape(deg.shape)

        return deg

    
    def _flat_mask_real_rows(self, X_array: np.ndarray) -> np.ndarray:
        """
        Returns a flat mask (B*N,) selecting rows whose existence flag (col 0) >= 0.5.
        Falls back to all-True if no row passes (edge-case safety).
        """
        exist = X_array[..., 0]                 # (B, N)
        mask = (exist >= 0.5).reshape(-1)       # (B*N,)
        if not np.any(mask):
            mask = np.ones(exist.size, dtype=bool)
        return mask

    def _fit_reducer(self, X_array: np.ndarray):
        """Fit a reducer on tail columns, keeping semantic prefix columns raw."""
        if not self.use_dim_reduction:
            self.reducer = None
            return

        B, N, D = X_array.shape
        kp = min(max(0, self.dim_reduction_keep_prefix), D)
        Xf = X_array.reshape(-1, D)
        mask = self._flat_mask_real_rows(X_array)
        tail = Xf[:, kp:]
        tail_real = tail[mask]

        self._orig_tail_dim = tail.shape[1]
        if self._orig_tail_dim == 0:
            self.reducer = None
            return
        if tail_real.shape[0] == 0:
            tail_real = tail

        if self.dim_reduction_method == "pca":
            max_components = min(
                int(self.dim_reduction_components),
                int(tail_real.shape[0]),
                int(tail_real.shape[1]),
            )
        else:
            max_components = min(
                int(self.dim_reduction_components),
                max(1, int(tail_real.shape[0]) - 1),
                int(tail_real.shape[1]),
            )
        max_components = max(1, max_components)

        if self.dim_reduction_method == "pca":
            if self.pca_whiten:
                warnings.warn(
                    "PCA whitening + MinMax can destabilize training. "
                    "Consider pca_whiten=False or switch to StandardScaler on Z."
                )
            reducer = PCA(
                n_components=max_components,
                whiten=self.pca_whiten,
                random_state=self.random_state,
            )
        elif self.dim_reduction_method == "svd":
            reducer = TruncatedSVD(
                n_components=max_components,
                random_state=self.random_state,
            )
        else:
            raise ValueError("dim_reduction_method must be 'pca' or 'svd'")

        reducer.fit(tail_real)
        self.reducer = reducer

    def _apply_reducer(self, X_array: np.ndarray) -> np.ndarray:
        """Apply the fitted reducer to tail columns and return [prefix || reduced_tail]."""
        if not self.use_dim_reduction or self.reducer is None:
            return X_array
        B, N, D = X_array.shape
        kp = min(max(0, self.dim_reduction_keep_prefix), D)
        Xf = X_array.reshape(-1, D)
        prefix = Xf[:, :kp]
        tail = Xf[:, kp:]
        red_tail = self.reducer.transform(tail)
        Xr = np.hstack([prefix, red_tail])
        return Xr.reshape(B, N, -1)

    def _inverse_reducer(self, X_reduced: np.ndarray) -> np.ndarray:
        """Inverse reduced tail back to original width for the graph decoder."""
        if not self.use_dim_reduction or self.reducer is None:
            return X_reduced
        B, N, Dred = X_reduced.shape
        kp = min(max(0, self.dim_reduction_keep_prefix), Dred)
        Xf = X_reduced.reshape(-1, Dred)
        prefix = Xf[:, :kp]
        red_tail = Xf[:, kp:]
        tail = self.reducer.inverse_transform(red_tail)
        Xfull = np.hstack([prefix, tail])
        return Xfull.reshape(B, N, -1)

    def setup(
        self,
        node_encodings_list: List[np.ndarray],
        conditional_graph_encodings: Any,
        edge_pairs: Optional[List[Tuple[int, int, int]]] = None,
        edge_targets: Optional[np.ndarray] = None,
        node_mask: Optional[np.ndarray] = None,
        edge_label_pairs: Optional[List[Tuple[int,int,int]]] = None,   # NEW
        edge_label_targets: Optional[np.ndarray] = None ,               # NEW
        edge_label_idx_to_label: Optional[Dict[int, Any]] = None,
        distance_pairs: Optional[List[Tuple[int,int,int]]] = None,
        distance_targets: Optional[np.ndarray] = None
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

        exist_mask = (X_array[..., 0] >= 0.5)

        # ---- label stats (like degree). Use only existent rows so the
        # class mapping matches the scaler/loss rows and padded zeros do not
        # become artificial labels.
        lab_idx = self.label_feature_index
        raw_labels = X_array[..., lab_idx]                       # (B, N)
        existing_raw_labels = raw_labels[exist_mask]
        if existing_raw_labels.size == 0:
            existing_raw_labels = raw_labels.reshape(-1)
        self.L_max = int(existing_raw_labels.max())              # global max class id

        lab_col = existing_raw_labels.reshape(-1, 1)
        lab_scaler = MinMaxScaler().fit(lab_col)                 # MinMax only on label col
        lab_min_val   = float(lab_scaler.data_min_[0])
        lab_range_val =float(lab_scaler.data_range_[0]) or 1e-8
        
        # 
       

                
        self._fit_scalers(X_array, y_array)

        # ------------------------------------------------------------
        # Compute class-imbalance weight for BCEWithLogitsLoss
        # ------------------------------------------------------------
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

        # Get degree scaling parameters from existent rows only. This must
        # match _fit_scalers(), which fits x_scaler on real rows only. If
        # padded zero rows are included here, CE targets can be shifted toward
        # class 0 even when real nodes have minimum degree 1.
        deg_column_data_for_minmax = X_array[..., self.important_feature_index][exist_mask].reshape(-1, 1)
        if deg_column_data_for_minmax.shape[0] == 0:
            deg_column_data_for_minmax = X_array[..., self.important_feature_index].reshape(-1, 1)
        temp_degree_scaler = MinMaxScaler().fit(deg_column_data_for_minmax) # Fit only on degree col
        
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
        existing_raw_degrees = raw_degrees[exist_mask]
        if existing_raw_degrees.size > 0:
            valid_degree_classes = np.unique(np.rint(existing_raw_degrees).astype(int))
            valid_degree_classes = np.clip(valid_degree_classes, 0, self.D_max)
            self.valid_existing_degree_classes = valid_degree_classes.astype(int).tolist()
            self.min_existing_degree = int(valid_degree_classes.min())
            self.max_existing_degree = int(valid_degree_classes.max())
        else:
            self.valid_existing_degree_classes = list(range(self.D_max + 1))
            self.min_existing_degree = 0
            self.max_existing_degree = self.D_max
        if self.verbose:
            print(
                "Existing-node degree support:",
                self.valid_existing_degree_classes,
            )
        degree_class_weight = np.ones(self.D_max + 1, dtype=np.float32)
        if self.balance_degree_loss and existing_raw_degrees.size > 0:
            degree_classes = np.clip(
                np.rint(existing_raw_degrees).astype(int),
                0,
                self.D_max,
            )
            counts = np.bincount(degree_classes, minlength=self.D_max + 1).astype(np.float32)
            present = counts > 0
            if present.any():
                weights = counts[present].sum() / (present.sum() * counts[present])
                weights = weights / max(float(weights.mean()), 1e-8)
                weights = np.clip(weights, 1.0 / self.degree_class_weight_cap, self.degree_class_weight_cap)
                degree_class_weight[present] = weights.astype(np.float32)
            if self.verbose:
                print("Degree class weights:", degree_class_weight.tolist())
        label_class_weight = np.ones(self.L_max + 1, dtype=np.float32)
        if self.balance_label_loss and existing_raw_labels.size > 0:
            label_classes = np.clip(
                np.rint(existing_raw_labels).astype(int),
                0,
                self.L_max,
            )
            counts = np.bincount(label_classes, minlength=self.L_max + 1).astype(np.float32)
            present = counts > 0
            if present.any():
                weights = counts[present].sum() / (present.sum() * counts[present])
                weights = weights / max(float(weights.mean()), 1e-8)
                weights = np.clip(weights, 1.0 / self.label_class_weight_cap, self.label_class_weight_cap)
                label_class_weight[present] = weights.astype(np.float32)
            if self.verbose:
                print("Label class weights:", label_class_weight.tolist())
        E_max_label = None
        use_edge_label_supervision = False
        if edge_label_targets is not None and len(edge_label_targets) > 0:
            uniq_edge_lbls = np.unique(edge_label_targets)
            if len(uniq_edge_lbls) > 1 and self.lambda_edge_label_importance > 0:
                E_max_label = int(uniq_edge_lbls.max())
                use_edge_label_supervision = True
            elif len(uniq_edge_lbls) > 1 and self.verbose:
                print("Edge-label loss weight is 0 — disabling edge-label head.")
            elif self.verbose:
                print("Only one edge label class found — disabling edge-label loss.")

        data_has_distance = (
                (distance_pairs is not None)
                and (distance_targets is not None)
                and (len(distance_targets) > 0)
            )
        # Respect the constructor flag *and* require data to exist
        use_distance_supervision = self.use_distance_supervision and data_has_distance

        # If OFF, drop the data so downstream heads/dataset stay disabled
        if not use_distance_supervision:
            distance_pairs = []
            distance_targets = np.array([], dtype=np.int64)

        
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
            degree_class_weight=degree_class_weight,
            lambda_node_exist_importance=self.lambda_node_exist_importance,
            use_edge_supervision=(edge_pairs is not None and len(edge_pairs) > 0),
            lambda_edge_importance=self.lambda_edge_importance,
            lambda_clean_edge_importance=self.lambda_clean_edge_importance,
            balance_edge_loss=self.balance_edge_loss,
            exist_pos_weight=exist_pos_weight,
            use_guidance=self.use_guidance,      # NEW
            guidance_weight=1.0,      
            lambda_consistency=self.lambda_consistency,          # tweak as needed
            label_feature_index=self.label_feature_index,   # 2
            max_label=self.L_max,
            lambda_label_importance=self.lambda_label_importance,
            noise_label_factor=self.noise_label_factor,
            label_class_weight=label_class_weight,
            label_min_val=lab_min_val,
            label_range_val=lab_range_val,
            use_edge_label_supervision=use_edge_label_supervision,
            max_edge_label=E_max_label,
            lambda_edge_label_importance=getattr(self, "lambda_edge_label_importance", 1.0),
            use_distance_supervision=use_distance_supervision,
            lambda_distance_importance=self.lambda_distance_importance,
            max_distance_class=self.max_distance_class,
            lambda_recon_importance=self.lambda_recon_importance,
            lambda_x0_importance=self.lambda_x0_importance,
            lambda_condition_x0_importance=self.lambda_condition_x0_importance,
            condition_x0_sampling_blend=self.condition_x0_sampling_blend,
            denoise_discrete_channels=self.denoise_discrete_channels,
            discrete_diffusion_mode=self.discrete_diffusion_mode,
            discrete_projection_sigma_threshold=self.discrete_projection_sigma_threshold,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            sampling_final_sigma=self.sampling_final_sigma,
            row_embedding_scale=self.row_embedding_scale,
            use_joint_edge_denoising=self.use_joint_edge_denoising,
            joint_edge_delete_prob=self.joint_edge_delete_prob,
            joint_edge_add_prob=self.joint_edge_add_prob,
            joint_edge_update_momentum=self.joint_edge_update_momentum,
            joint_edge_threshold=self.joint_edge_threshold,
            joint_edge_message_passing_steps=self.joint_edge_message_passing_steps,
            project_existence_during_sampling=self.project_existence_during_sampling,
            project_degree_during_sampling=self.project_degree_during_sampling,
            project_label_during_sampling=self.project_label_during_sampling,
	            use_matched_row_loss=self.use_matched_row_loss

	        )
        self.model.use_guidance = self.use_guidance
        self._edge_pairs = edge_pairs
        self._edge_targets = edge_targets
        self._edge_label_pairs = edge_label_pairs
        self._edge_label_targets = edge_label_targets
        self._edge_label_idx_to_label = dict(edge_label_idx_to_label or {})
        self._distance_pairs     = distance_pairs
        self._distance_targets   = distance_targets
        # self._node_mask = node_mask_arr  # created earlier

    def fit(
        self,
        node_encodings_list: List[np.ndarray],
        conditional_graph_encodings: Any,
        edge_pairs: Optional[List[Tuple[int, int, int]]] = None,
        edge_targets: Optional[np.ndarray] = None,
        node_mask: Optional[np.ndarray] = None,
        distance_pairs: Optional[List[Tuple[int,int,int]]] = None,
        distance_targets: Optional[np.ndarray] = None    
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

        node_counts = np.asarray(
            [
                int(np.asarray(node_encoding).shape[0])
                for node_encoding in node_encodings_list
            ],
            dtype=np.int64,
        )
        self.valid_node_counts = sorted(np.unique(node_counts).astype(int).tolist())
        self.node_count_model = None
        self.fixed_node_count = None
        if self.use_condition_node_count:
            if len(self.valid_node_counts) == 1:
                self.fixed_node_count = int(self.valid_node_counts[0])
                if self.verbose:
                    print(
                        "Condition node-count control: fixed count "
                        f"{self.fixed_node_count}."
                    )
            elif len(self.valid_node_counts) > 1:
                self.node_count_model = LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=int(self.random_state),
                )
                self.node_count_model.fit(y_scaled, node_counts)
                train_accuracy = float(
                    self.node_count_model.score(y_scaled, node_counts)
                )
                if self.verbose:
                    print(
                        "Condition node-count control: learned counts "
                        f"{self.valid_node_counts}; training accuracy "
                        f"{train_accuracy:.3f}."
                    )

        degree_histograms = np.zeros(
            (len(node_encodings_list), self.D_max + 1),
            dtype=np.float32,
        )
        for graph_index, node_encoding in enumerate(node_encodings_list):
            raw_degrees = np.rint(
                np.asarray(node_encoding)[:, self.important_feature_index]
            ).astype(np.int64)
            raw_degrees = np.clip(raw_degrees, 0, self.D_max)
            degree_histograms[graph_index] = np.bincount(
                raw_degrees,
                minlength=self.D_max + 1,
            )
        self.degree_histogram_model = None
        self.fixed_degree_histogram = None
        if self.use_condition_degree_histogram:
            unique_histograms = np.unique(degree_histograms, axis=0)
            if len(unique_histograms) == 1:
                self.fixed_degree_histogram = unique_histograms[0].copy()
                if self.verbose:
                    print(
                        "Condition degree-histogram control: fixed histogram "
                        f"{self.fixed_degree_histogram.astype(int).tolist()}."
                    )
            else:
                self.degree_histogram_model = Ridge(alpha=1.0)
                self.degree_histogram_model.fit(y_scaled, degree_histograms)
                train_prediction = self.degree_histogram_model.predict(y_scaled)
                train_mae = float(
                    np.mean(np.abs(train_prediction - degree_histograms))
                )
                if self.verbose:
                    print(
                        "Condition degree-histogram control: learned "
                        f"{self.D_max + 1} bins; training MAE "
                        f"{train_mae:.3f} nodes/bin."
                    )

        label_histograms = np.zeros(
            (len(node_encodings_list), self.L_max + 1),
            dtype=np.float32,
        )
        for graph_index, node_encoding in enumerate(node_encodings_list):
            raw_labels = np.rint(
                np.asarray(node_encoding)[:, self.label_feature_index]
            ).astype(np.int64)
            raw_labels = np.clip(raw_labels, 0, self.L_max)
            label_histograms[graph_index] = np.bincount(
                raw_labels,
                minlength=self.L_max + 1,
            )
        self.label_histogram_model = None
        self.fixed_label_histogram = None
        if self.use_condition_label_histogram:
            unique_histograms = np.unique(label_histograms, axis=0)
            if len(unique_histograms) == 1:
                self.fixed_label_histogram = unique_histograms[0].copy()
                if self.verbose:
                    print(
                        "Condition label-histogram control: fixed histogram "
                        f"{self.fixed_label_histogram.astype(int).tolist()}."
                    )
            else:
                self.label_histogram_model = Ridge(alpha=1.0)
                self.label_histogram_model.fit(y_scaled, label_histograms)
                train_prediction = self.label_histogram_model.predict(y_scaled)
                train_mae = float(
                    np.mean(np.abs(train_prediction - label_histograms))
                )
                if self.verbose:
                    print(
                        "Condition label-histogram control: learned "
                        f"{self.L_max + 1} bins; training MAE "
                        f"{train_mae:.3f} nodes/bin."
                    )

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

                # Pull optional supervision stored during setup()
        edge_label_pairs = getattr(self, "_edge_label_pairs", []) or []
        edge_label_targets = getattr(self, "_edge_label_targets", None)
        if edge_label_targets is None:
            edge_label_targets = np.array([], dtype=np.int64)

        # Distance supervision can be passed in, or fall back to setup() values
        if distance_pairs is None:
            distance_pairs = getattr(self, "_distance_pairs", []) or []
        if distance_targets is None:
            _dt = getattr(self, "_distance_targets", None)
            distance_targets = _dt if _dt is not None else np.array([], dtype=np.int64)

        if not getattr(self.model, "use_distance_supervision", False):
            distance_pairs = []
            distance_targets = np.array([], dtype=np.int64)

        use_graph_dataset = any([
            edge_pairs is not None,
            len(edge_label_pairs) > 0,
            len(distance_pairs) > 0
        ])
 
        if use_graph_dataset:
            if node_mask is None:
                B, N, _ = X_scaled.shape
                node_mask_arr = np.ones((B, N), dtype=bool)
            else:
                node_mask_arr = node_mask

            train_base_dataset = GraphWithEdgesDataset(
                X_scaled, y_scaled,
                edge_pairs=edge_pairs or [],
                edge_targets=edge_targets if edge_targets is not None else np.array([], dtype=np.float32),
                node_mask=node_mask_arr,
                edge_label_pairs=edge_label_pairs,
                edge_label_targets=edge_label_targets,
                distance_pairs=distance_pairs,
                distance_targets=distance_targets,
                random_row_permutation=self.random_row_permutation,
            )
            val_base_dataset = GraphWithEdgesDataset(
                X_scaled, y_scaled,
                edge_pairs=edge_pairs or [],
                edge_targets=edge_targets if edge_targets is not None else np.array([], dtype=np.float32),
                node_mask=node_mask_arr,
                edge_label_pairs=edge_label_pairs,
                edge_label_targets=edge_label_targets,
                distance_pairs=distance_pairs,
                distance_targets=distance_targets,
                random_row_permutation=False,
            )
            # Split into train/val
            dataset_size = len(node_encodings_list)  # <--- Use the length of the original data
            train_size = int(0.9 * dataset_size)
            val_size = dataset_size - train_size
            # Create indices for the split
            indices = torch.randperm(len(node_encodings_list)).tolist()
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            train_dataset = torch.utils.data.Subset(train_base_dataset, train_indices)
            val_dataset = torch.utils.data.Subset(val_base_dataset, val_indices)
            
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
            train_base_dataset = RowPermutationTensorDataset(
                X_tensor,
                y_tensor,
                random_row_permutation=self.random_row_permutation,
            )
            val_base_dataset = RowPermutationTensorDataset(
                X_tensor,
                y_tensor,
                random_row_permutation=False,
            )
            
            # Split into train/val
            train_size = int(0.9 * len(train_base_dataset))
            val_size = len(train_base_dataset) - train_size
            indices = torch.randperm(len(train_base_dataset)).tolist()
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            train_dataset = torch.utils.data.Subset(train_base_dataset, train_indices)
            val_dataset = torch.utils.data.Subset(val_base_dataset, val_indices)
            
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
                # Lightning checkpoints are created locally during this run.
                # PyTorch 2.6 defaults torch.load(..., weights_only=True), which
                # rejects some Lightning checkpoint metadata.
                state = torch.load(best_path, map_location=self.device, weights_only=False)
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
            
        self._ensure_model_device()

        # ------------------------------------------------------------------
        # 1. Generate denoised node embeddings (scaled feature space)
        # ------------------------------------------------------------------
        with torch.no_grad():
            y_scaled = self._transform_condition_data(conditional_graph_encodings)
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
        if self.project_label_during_sampling and lab_classes is not None:
            lab_classes = lab_classes.cpu().numpy()
            for i in range(len(gen_orig)):
                gen_orig[i][..., self.label_feature_index] = np.clip(lab_classes[i], 0, self.L_max)

        final_degree_logits = None
        final_label_logits = None
        needs_final_logits = (
            self.use_condition_node_count
            or self.use_condition_degree_histogram
            or self.use_condition_label_histogram
        )
        if needs_final_logits:
            predicted_counts = None
            if self.use_condition_node_count and self.fixed_node_count is not None:
                predicted_counts = np.full(
                    len(gen_orig),
                    int(self.fixed_node_count),
                    dtype=np.int64,
                )
            elif self.use_condition_node_count and self.node_count_model is not None:
                predicted_counts = np.asarray(
                    self.node_count_model.predict(y_scaled),
                    dtype=np.int64,
                )
            elif self.use_condition_node_count:
                raise RuntimeError(
                    "Condition node-count control is enabled but has not been "
                    "fitted. Call fit() before predict()."
                )

            with torch.no_grad():
                t0 = torch.zeros(
                    cond_tensor.size(0),
                    1,
                    dtype=cond_tensor.dtype,
                    device=self.device,
                )
                final_prediction = self.model.forward(
                    generated,
                    cond_tensor,
                    t0,
                    return_latents=True,
                    add_noise=False,
                )
                existence_probabilities = torch.sigmoid(
                    final_prediction[2]
                ).detach().cpu().numpy()
                if self.use_condition_degree_histogram:
                    final_degree_logits = final_prediction[1].detach().cpu().numpy()
                if self.use_condition_label_histogram and final_prediction[3] is not None:
                    final_label_logits = final_prediction[3].detach().cpu().numpy()

        if self.use_condition_node_count:
            for graph_index, (encoding, count) in enumerate(
                zip(gen_orig, predicted_counts)
            ):
                count = int(np.clip(count, 0, encoding.shape[0]))
                ranking = np.argsort(
                    existence_probabilities[graph_index],
                    kind="stable",
                )
                selected = ranking[-count:] if count > 0 else np.array([], dtype=int)
                encoding[:, 0] = 0.0
                encoding[selected, 0] = 1.0
                inactive = np.ones(encoding.shape[0], dtype=bool)
                inactive[selected] = False
                encoding[inactive, self.important_feature_index] = 0.0

            self._last_predicted_node_counts = predicted_counts.copy()
            self._last_existence_probabilities = existence_probabilities.copy()
            if self.verbose:
                values, counts = np.unique(predicted_counts, return_counts=True)
                summary = {
                    int(value): int(count)
                    for value, count in zip(values, counts)
                }
                print("Condition-predicted node counts:", summary)

        if self.use_condition_degree_histogram:
            if self.fixed_degree_histogram is not None:
                raw_histograms = np.repeat(
                    self.fixed_degree_histogram[None, :],
                    len(gen_orig),
                    axis=0,
                )
            elif self.degree_histogram_model is not None:
                raw_histograms = self.degree_histogram_model.predict(y_scaled)
            else:
                raise RuntimeError(
                    "Condition degree-histogram control is enabled but has not "
                    "been fitted. Call fit() before predict()."
                )

            predicted_histograms = []
            for graph_index, encoding in enumerate(gen_orig):
                active_indices = np.flatnonzero(encoding[:, 0] >= 0.5)
                node_count = len(active_indices)
                raw_histogram = np.maximum(
                    np.asarray(raw_histograms[graph_index], dtype=float),
                    0.0,
                )
                if raw_histogram.sum() <= 0:
                    raw_histogram = np.ones_like(raw_histogram)
                scaled_histogram = raw_histogram * (
                    node_count / raw_histogram.sum()
                )
                histogram = np.floor(scaled_histogram).astype(np.int64)
                remainder = node_count - int(histogram.sum())
                if remainder > 0:
                    fractions = scaled_histogram - histogram
                    add_bins = np.argsort(-fractions, kind="stable")[:remainder]
                    histogram[add_bins] += 1
                elif remainder < 0:
                    removable = np.argsort(
                        scaled_histogram - histogram,
                        kind="stable",
                    )
                    for degree_class in removable:
                        take = min(histogram[degree_class], -remainder)
                        histogram[degree_class] -= take
                        remainder += take
                        if remainder == 0:
                            break

                degree_slots = np.repeat(
                    np.arange(len(histogram), dtype=np.int64),
                    histogram,
                )
                if len(degree_slots) != node_count:
                    raise RuntimeError(
                        "Projected degree histogram does not match node count."
                    )

                if node_count > 0:
                    logits = final_degree_logits[
                        graph_index,
                        active_indices,
                        : len(histogram),
                    ]
                    logits = logits - logits.max(axis=1, keepdims=True)
                    probabilities = np.exp(logits)
                    probabilities /= np.maximum(
                        probabilities.sum(axis=1, keepdims=True),
                        1e-12,
                    )
                    assignment_cost = -np.log(
                        np.maximum(probabilities[:, degree_slots], 1e-12)
                    )
                    row_indices, slot_indices = linear_sum_assignment(
                        assignment_cost
                    )
                    assigned_degrees = np.zeros(node_count, dtype=np.int64)
                    assigned_degrees[row_indices] = degree_slots[slot_indices]
                    encoding[
                        active_indices,
                        self.important_feature_index,
                    ] = assigned_degrees
                encoding[
                    encoding[:, 0] < 0.5,
                    self.important_feature_index,
                ] = 0.0
                predicted_histograms.append(histogram)

            self._last_predicted_degree_histograms = np.asarray(
                predicted_histograms,
                dtype=np.int64,
            )
            if self.verbose:
                histogram_summary = {
                    tuple(histogram.tolist()): count
                    for histogram, count in zip(
                        *np.unique(
                            self._last_predicted_degree_histograms,
                            axis=0,
                            return_counts=True,
                        )
                    )
                }
                print(
                    "Condition-predicted degree histograms:",
                    histogram_summary,
                )

        if self.use_condition_label_histogram:
            if final_label_logits is None:
                raise RuntimeError(
                    "Condition label-histogram control requires the label head "
                    "to be available during prediction."
                )
            if self.fixed_label_histogram is not None:
                raw_histograms = np.repeat(
                    self.fixed_label_histogram[None, :],
                    len(gen_orig),
                    axis=0,
                )
            elif self.label_histogram_model is not None:
                raw_histograms = self.label_histogram_model.predict(y_scaled)
            else:
                raise RuntimeError(
                    "Condition label-histogram control is enabled but has not "
                    "been fitted. Call fit() before predict()."
                )

            predicted_histograms = []
            for graph_index, encoding in enumerate(gen_orig):
                active_indices = np.flatnonzero(encoding[:, 0] >= 0.5)
                node_count = len(active_indices)
                raw_histogram = np.maximum(
                    np.asarray(raw_histograms[graph_index], dtype=float),
                    0.0,
                )
                if raw_histogram.sum() <= 0:
                    raw_histogram = np.ones_like(raw_histogram)
                scaled_histogram = raw_histogram * (
                    node_count / raw_histogram.sum()
                )
                histogram = np.floor(scaled_histogram).astype(np.int64)
                remainder = node_count - int(histogram.sum())
                if remainder > 0:
                    fractions = scaled_histogram - histogram
                    add_bins = np.argsort(-fractions, kind="stable")[:remainder]
                    histogram[add_bins] += 1
                elif remainder < 0:
                    removable = np.argsort(
                        scaled_histogram - histogram,
                        kind="stable",
                    )
                    for label_class in removable:
                        take = min(histogram[label_class], -remainder)
                        histogram[label_class] -= take
                        remainder += take
                        if remainder == 0:
                            break

                label_slots = np.repeat(
                    np.arange(len(histogram), dtype=np.int64),
                    histogram,
                )
                if len(label_slots) != node_count:
                    raise RuntimeError(
                        "Projected label histogram does not match node count."
                    )

                if node_count > 0:
                    logits = final_label_logits[
                        graph_index,
                        active_indices,
                        : len(histogram),
                    ]
                    logits = logits - logits.max(axis=1, keepdims=True)
                    probabilities = np.exp(logits)
                    probabilities /= np.maximum(
                        probabilities.sum(axis=1, keepdims=True),
                        1e-12,
                    )
                    assignment_cost = -np.log(
                        np.maximum(probabilities[:, label_slots], 1e-12)
                    )
                    row_indices, slot_indices = linear_sum_assignment(
                        assignment_cost
                    )
                    assigned_labels = np.zeros(node_count, dtype=np.int64)
                    assigned_labels[row_indices] = label_slots[slot_indices]
                    encoding[
                        active_indices,
                        self.label_feature_index,
                    ] = assigned_labels
                encoding[
                    encoding[:, 0] < 0.5,
                    self.label_feature_index,
                ] = 0.0
                predicted_histograms.append(histogram)

            self._last_predicted_label_histograms = np.asarray(
                predicted_histograms,
                dtype=np.int64,
            )
            if self.verbose:
                histogram_summary = {
                    tuple(histogram.tolist()): count
                    for histogram, count in zip(
                        *np.unique(
                            self._last_predicted_label_histograms,
                            axis=0,
                            return_counts=True,
                        )
                    )
                }
                print(
                    "Condition-predicted label histograms:",
                    histogram_summary,
                )


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
            if (
                self.project_degree_during_sampling
                and not self.use_condition_degree_histogram
                and deg_classes is not None
            ):
                deg_classes = deg_classes.cpu().numpy()
                for i in range(len(gen_orig)):
                    # Overwrite the degree channel (assumed channel index 1)
                    deg_idx = self.important_feature_index
                    snapped_degrees = self._snap_degrees_to_existing_support(deg_classes[i])
                    gen_orig[i][..., deg_idx] = np.clip(snapped_degrees, 0, self.D_max)


            if self.verbose:
                projected = []
                if self.project_existence_during_sampling:
                    projected.append("existence")
                if self.project_degree_during_sampling:
                    projected.append("degree")
                if (
                    self.project_label_during_sampling
                    and not self.use_condition_label_histogram
                ):
                    projected.append("label")
                if projected:
                    print(f"Applied head-based projection for {', '.join(projected)} channels.")
                else:
                    print("Head-based projection disabled.")

        except (AttributeError, RuntimeError, ValueError, IndexError, TypeError) as e:
            if self.verbose:
                print(f"[Warning] Head projection skipped due to: {e}")

        self._last_generated_node_encodings = [
            np.asarray(enc, dtype=float).copy()
            for enc in gen_orig
        ]
        return gen_orig

    def predict_edge_labels(
        self,
        conditional_graph_encodings: Any,
        node_encodings_list: List[np.ndarray],
        adj_mtx_list: List[np.ndarray],
    ) -> Optional[List[np.ndarray]]:
        """
        Predict edge labels directly from the generator's edge-label head.

        Parameters
        ----------
        conditional_graph_encodings : Any
            Conditioning vectors corresponding to the generated node matrices.
        node_encodings_list : List[np.ndarray]
            Generated node-feature matrices in original feature space.
        adj_mtx_list : List[np.ndarray]
            Final decoded adjacency matrices.

        Returns
        -------
        Optional[List[np.ndarray]]
            Per-graph edge-label arrays ordered exactly like the decoder's
            directed edge traversal, or None when the edge-label head is inactive.
        """
        if self.model is None or getattr(self.model, "edge_label_head", None) is None:
            return None

        if len(node_encodings_list) != len(adj_mtx_list):
            raise ValueError("node_encodings_list and adj_mtx_list must have the same length")

        if len(node_encodings_list) == 0:
            return []

        self._ensure_model_device()

        # Pad generated node matrices exactly like fit()/predict().
        X_array = np.stack([
            np.pad(
                x,
                ((0, self.number_of_rows_per_example - x.shape[0]), (0, 0)),
                mode='constant',
                constant_values=0,
            )
            for x in node_encodings_list
        ], axis=0)
        y_array = np.asarray(conditional_graph_encodings)
        X_scaled, y_scaled = self._transform_data(X_array, y_array)

        x_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        cond_tensor = torch.tensor(y_scaled, dtype=torch.float32, device=self.device)

        sigma_proj = torch.tensor(getattr(self.model, "sampling_final_sigma", 0.0), device=self.device)
        t0 = self.model._t_from_sigma(sigma_proj).expand(len(node_encodings_list), 1)

        with torch.no_grad():
            _, _, _, _, latent_tokens, _, _ = self.model.forward(
                x_tensor,
                cond_tensor,
                t0,
                return_latents=True,
                add_noise=False,
            )

        idx_to_label = getattr(self, "_edge_label_idx_to_label", {}) or {}
        predicted_edge_labels_list: List[np.ndarray] = []

        for b, adj in enumerate(adj_mtx_list):
            labels = []
            n_nodes = adj.shape[0]
            symmetric_cache = {}
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if adj[i, j] == 0:
                        continue
                    key = (min(i, j), max(i, j))
                    if adj[j, i] != 0 and key in symmetric_cache:
                        labels.append(symmetric_cache[key])
                        continue

                    logits_ij = self.model.edge_label_head(
                        latent_tokens[b, i].unsqueeze(0),
                        latent_tokens[b, j].unsqueeze(0),
                    )
                    if adj[j, i] != 0 and i != j:
                        logits_ji = self.model.edge_label_head(
                            latent_tokens[b, j].unsqueeze(0),
                            latent_tokens[b, i].unsqueeze(0),
                        )
                        logits = 0.5 * (logits_ij + logits_ji)
                    else:
                        logits = logits_ij
                    pred_idx = int(logits.argmax(dim=-1).item())
                    pred_label = idx_to_label.get(pred_idx, pred_idx)
                    if adj[j, i] != 0:
                        symmetric_cache[key] = pred_label
                    labels.append(pred_label)
            predicted_edge_labels_list.append(np.asarray(labels, dtype=object))

        return predicted_edge_labels_list

    def predict_edge_probabilities(
        self,
        conditional_graph_encodings: Any,
        node_encodings_list: List[np.ndarray],
    ) -> Optional[List[np.ndarray]]:
        """
        Predict dense pairwise edge probabilities directly from the generator's edge head.

        Returns one (N, N) probability matrix per graph, with zeros on the diagonal.
        """
        if self.model is None or getattr(self.model, "edge_head", None) is None:
            return None

        if len(node_encodings_list) == 0:
            return []

        cached_edge_probabilities = getattr(self.model, "_last_edge_probabilities", None)
        cached_node_encodings = getattr(self, "_last_generated_node_encodings", None)
        cache_matches_nodes = (
            cached_node_encodings is not None
            and len(cached_node_encodings) == len(node_encodings_list)
        )
        if cache_matches_nodes:
            for cached_enc, enc in zip(cached_node_encodings, node_encodings_list):
                cached_arr = np.asarray(cached_enc, dtype=float)
                enc_arr = np.asarray(enc, dtype=float)
                if cached_arr.shape != enc_arr.shape or not np.allclose(cached_arr, enc_arr, atol=1e-6):
                    cache_matches_nodes = False
                    break

        if (
            cache_matches_nodes
            and cached_edge_probabilities is not None
            and len(cached_edge_probabilities) == len(node_encodings_list)
        ):
            cached_np = cached_edge_probabilities.detach().cpu().numpy()
            prob_matrices = []
            cache_usable = True
            for b, enc in enumerate(node_encodings_list):
                n_nodes = enc.shape[0]
                if cached_np.shape[1] < n_nodes or cached_np.shape[2] < n_nodes:
                    cache_usable = False
                    break
                pm = np.asarray(cached_np[b, :n_nodes, :n_nodes], dtype=float)
                pm = 0.5 * (pm + pm.T)
                np.fill_diagonal(pm, 0.0)
                prob_matrices.append(pm)
            if cache_usable:
                return prob_matrices

        self._ensure_model_device()

        X_array = np.stack([
            np.pad(
                x,
                ((0, self.number_of_rows_per_example - x.shape[0]), (0, 0)),
                mode='constant',
                constant_values=0,
            )
            for x in node_encodings_list
        ], axis=0)
        y_array = np.asarray(conditional_graph_encodings)
        X_scaled, y_scaled = self._transform_data(X_array, y_array)

        x_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        cond_tensor = torch.tensor(y_scaled, dtype=torch.float32, device=self.device)

        sigma_proj = torch.tensor(getattr(self.model, "sampling_final_sigma", 0.0), device=self.device)
        t0 = self.model._t_from_sigma(sigma_proj).expand(len(node_encodings_list), 1)

        with torch.no_grad():
            _, _, _, _, latent_tokens, _, _ = self.model.forward(
                x_tensor,
                cond_tensor,
                t0,
                return_latents=True,
                add_noise=False,
            )

        prob_matrices: List[np.ndarray] = []
        for b, enc in enumerate(node_encodings_list):
            n_nodes = enc.shape[0]
            prob_matrix = np.zeros((n_nodes, n_nodes), dtype=float)
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i == j:
                        continue
                    logits = self.model.edge_head(
                        latent_tokens[b, i].unsqueeze(0),
                        latent_tokens[b, j].unsqueeze(0),
                    )
                    prob_matrix[i, j] = float(torch.sigmoid(logits).item())
            prob_matrices.append(prob_matrix)

        return prob_matrices

    def predict_distance_probabilities(
        self,
        conditional_graph_encodings: Any,
        node_encodings_list: List[np.ndarray],
        distance_class: int = 0,
    ) -> Optional[List[np.ndarray]]:
        """
        Predict dense pairwise probabilities for a hop-distance class.

        By convention class 0 is one-hop, so this can be blended with the edge
        head before constrained decoding.
        """
        if self.model is None or getattr(self.model, "distance_head", None) is None:
            return None

        if len(node_encodings_list) == 0:
            return []

        self._ensure_model_device()

        X_array = np.stack([
            np.pad(
                x,
                ((0, self.number_of_rows_per_example - x.shape[0]), (0, 0)),
                mode='constant',
                constant_values=0,
            )
            for x in node_encodings_list
        ], axis=0)
        y_array = np.asarray(conditional_graph_encodings)
        X_scaled, y_scaled = self._transform_data(X_array, y_array)

        x_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
        cond_tensor = torch.tensor(y_scaled, dtype=torch.float32, device=self.device)

        sigma_proj = torch.tensor(getattr(self.model, "sampling_final_sigma", 0.0), device=self.device)
        t0 = self.model._t_from_sigma(sigma_proj).expand(len(node_encodings_list), 1)

        with torch.no_grad():
            _, _, _, _, latent_tokens, _, _ = self.model.forward(
                x_tensor,
                cond_tensor,
                t0,
                return_latents=True,
                add_noise=False,
            )

        prob_matrices: List[np.ndarray] = []
        for b, enc in enumerate(node_encodings_list):
            n_nodes = enc.shape[0]
            prob_matrix = np.zeros((n_nodes, n_nodes), dtype=float)
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i == j:
                        continue
                    logits = self.model.distance_head(
                        latent_tokens[b, i].unsqueeze(0),
                        latent_tokens[b, j].unsqueeze(0),
                    )
                    probs = torch.softmax(logits, dim=-1)
                    cls = min(max(int(distance_class), 0), probs.shape[-1] - 1)
                    prob_matrix[i, j] = float(probs[..., cls].item())
            prob_matrices.append(prob_matrix)

        return prob_matrices

    def plot_metrics(self, window: int = 10, alpha: float = 0.3):
        if self.model is None:
            print("Model is not fitted yet."); return

        base_train = {
            "total": self.model.train_losses,
            "deg_ce": self.model.train_deg_ce,
            "exist": self.model.train_exist,
        }
        base_val = {
            "total": self.model.val_losses,
            "deg_ce": self.model.val_deg_ce,
            "exist": self.model.val_exist,
        }
        if getattr(self.model, "lambda_recon_importance", 0.0) > 0:
            base_train["recon"] = self.model.train_recon
            base_val["recon"] = self.model.val_recon
        if getattr(self.model, "lambda_x0_importance", 0.0) > 0:
            base_train["x0"] = self.model.train_x0
            base_val["x0"] = self.model.val_x0
        if getattr(self.model, "lambda_condition_x0_importance", 0.0) > 0:
            base_train["condition_x0"] = self.model.train_condition_x0
            base_val["condition_x0"] = self.model.val_condition_x0
        if getattr(self.model, "label_head", None) is not None:
            base_train["label_ce"] = self.model.train_label_ce
            base_val["label_ce"] = self.model.val_label_ce
        if getattr(self.model, "use_edge_supervision", False):
            base_train["edge"] = self.model.train_edge_loss
            base_val["edge"]   = self.model.val_edge_loss
            if getattr(self.model, "lambda_clean_edge_importance", 0.0) > 0:
                base_train["clean_edge"] = getattr(self.model, "train_clean_edge_loss", [])
                base_val["clean_edge"] = getattr(self.model, "val_clean_edge_loss", [])
        if getattr(self.model, "use_joint_edge_denoising", False):
            base_train["joint_edge"] = getattr(self.model, "train_joint_edge_loss", [])
            base_val["joint_edge"] = getattr(self.model, "val_joint_edge_loss", [])
        if getattr(self.model, "edge_label_head", None) is not None:
            base_train["edge_label"] = getattr(self.model, "train_edge_label_loss", [])
            base_val["edge_label"]   = getattr(self.model, "val_edge_label_loss", [])
        if (
            getattr(self.model, "distance_head", None) is not None
            and getattr(self.model, "lambda_distance_importance", 0.0) > 0
        ):
            base_train["dist_loss"] = getattr(self.model, "train_dist_loss", [])
            # base_train["dist_acc"]  = getattr(self.model, "train_dist_acc",  [])
            base_val["dist_loss"]   = getattr(self.model, "val_dist_loss",   [])
            # base_val["dist_acc"]    = getattr(self.model, "val_dist_acc",    [])

        plot_metrics(base_train, base_val, window=window, alpha=alpha)

    
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
        pl_module.train_x0.append(m.get("train_x0", torch.tensor(0.0)).item())
        pl_module.train_condition_x0.append(m.get("train_condition_x0", torch.tensor(0.0)).item())
        pl_module.train_exist.append(m.get("train_exist", torch.tensor(0.0)).item())
        pl_module.train_label_ce.append(m.get("train_label_ce", torch.tensor(0.0)).item())
        pl_module.train_dist_loss.append(m.get("train_dist_loss", torch.tensor(0.0)).item())
 
        if getattr(pl_module, "use_edge_supervision", False):
            pl_module.train_edge_loss.append(m.get("train_edge_loss", torch.tensor(0.0)).item())
            pl_module.train_edge_acc.append(m.get("train_edge_acc", torch.tensor(0.0)).item())
            pl_module.train_clean_edge_loss.append(m.get("train_clean_edge_loss", torch.tensor(0.0)).item())
            pl_module.train_clean_edge_acc.append(m.get("train_clean_edge_acc", torch.tensor(0.0)).item())
        if getattr(pl_module, "use_joint_edge_denoising", False):
            pl_module.train_joint_edge_loss.append(m.get("train_joint_edge_loss", torch.tensor(0.0)).item())
            pl_module.train_joint_edge_acc.append(m.get("train_joint_edge_acc", torch.tensor(0.0)).item())
        if getattr(pl_module, "edge_label_head", None) is not None:
            # only when edge-label head is active (i.e., >1 unique label)
            pl_module.train_edge_label_loss.append(m.get("train_edge_label_loss", torch.tensor(0.0)).item())
            pl_module.train_edge_label_acc.append(m.get("train_edge_label_acc", torch.tensor(0.0)).item())
        

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        pl_module.val_losses.append(m.get("val_total", torch.tensor(0.0)).item())
        pl_module.val_deg_ce.append(m.get("val_deg_ce", torch.tensor(0.0)).item())
        pl_module.val_recon.append(m.get("val_recon", torch.tensor(0.0)).item())
        pl_module.val_x0.append(m.get("val_x0", torch.tensor(0.0)).item())
        pl_module.val_condition_x0.append(m.get("val_condition_x0", torch.tensor(0.0)).item())
        pl_module.val_exist.append(m.get("val_exist", torch.tensor(0.0)).item())
        pl_module.val_label_ce.append(m.get("val_label_ce", torch.tensor(0.0)).item())
        pl_module.val_dist_loss.append(m.get("val_dist_loss", torch.tensor(0.0)).item())

        if getattr(pl_module, "use_edge_supervision", False):
            pl_module.val_edge_loss.append(m.get("val_edge_loss", torch.tensor(0.0)).item())
            pl_module.val_edge_acc.append(m.get("val_edge_acc", torch.tensor(0.0)).item())
            pl_module.val_clean_edge_loss.append(m.get("val_clean_edge_loss", torch.tensor(0.0)).item())
            pl_module.val_clean_edge_acc.append(m.get("val_clean_edge_acc", torch.tensor(0.0)).item())
        if getattr(pl_module, "use_joint_edge_denoising", False):
            pl_module.val_joint_edge_loss.append(m.get("val_joint_edge_loss", torch.tensor(0.0)).item())
            pl_module.val_joint_edge_acc.append(m.get("val_joint_edge_acc", torch.tensor(0.0)).item())
        if getattr(pl_module, "edge_label_head", None) is not None:
            pl_module.val_edge_label_loss.append(m.get("val_edge_label_loss", torch.tensor(0.0)).item())
            pl_module.val_edge_label_acc.append(m.get("val_edge_label_acc", torch.tensor(0.0)).item())
