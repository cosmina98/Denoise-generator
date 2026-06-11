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
from itertools import cycle  # Add this import


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

# ───────────────────────────────────────────────────────────────────
# Tiny, self-contained Set-Transformer primitives (no external deps)
# ───────────────────────────────────────────────────────────────────
class MiniMHA(nn.Module):
    """
    Lightweight Multi-Head Attention used by SAB / ISAB.
    A `dropout` argument is now forwarded to `nn.MultiheadAttention`
    so `transformer_dropout` really affects attention weights.
    """
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim   = dim,
            num_heads   = heads,
            dropout     = dropout,      # <--- NEW
            batch_first = True,
        )

    def forward(self, q, k, v):
        out, _ = self.attn(q, k, v, need_weights=False)
        return out

class SAB(nn.Module):
    """Self-Attention Block – permutation-equivariant."""
    def __init__(self, dim, heads=4, ff_mult=4, dropout=0.0):
        super().__init__()
        self.mha   = MiniMHA(dim, heads, dropout)  # Pass dropout
        self.norm1 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim*ff_mult), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim*ff_mult, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, X):
        X = X + self.mha(X, X, X); X = self.norm1(X)
        X = X + self.ff(X);        X = self.norm2(X)
        return X

class ISAB(nn.Module):
    """
    Induced Set-Attention Block (Lee et al., 2019).
    Preserves input→output alignment and is permutation-equivariant.
    """
    def __init__(self, dim, heads=4, m=32, ff_mult=4, dropout=0.0):
        super().__init__()
        self.inducing_pts = nn.Parameter(torch.randn(1, m, dim))
        self.h1 = MiniMHA(dim, heads, dropout)   # set → inducing 
        self.h2 = MiniMHA(dim, heads, dropout)   # set ↤ inducing
        self.norm = nn.LayerNorm(dim)
        self.ff   = nn.Sequential(
            nn.Linear(dim, dim*ff_mult), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim*ff_mult, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, X):
        B = X.size(0)
        I = self.inducing_pts.expand(B, -1, -1)           # (B,m,D)
        H = self.h1(I, X, X)                              # aggregate into m vectors
        X = X + self.h2(X, H, H)                          # transmit back
        X = self.norm(X)
        X = X + self.ff(X); X = self.norm2(X)
        return X

# =============================================================================
# Flow-Matching Graph Generator with Cross-Attention
# ────────────────────────────────────────────────────────────────────────────
#  Transformer-based implementation of continuous-time flow matching
# =============================================================================
class FMNodeTransformer(pl.LightningModule):
    def __init__(self,
                 number_of_rows_per_example: int,
                 input_feature_dimension: int,
                 condition_feature_dimension: int,
                 latent_embedding_dimension: int,
                 number_of_transformer_layers: int,
                 transformer_attention_head_count: int,
                 transformer_dropout: float = 0.1,
                 num_inducing_points: int = 32,
                 learning_rate: float = 1e-3,
                 verbose: bool = False,
                 important_feature_index: int = 1,
                 max_degree: Optional[int] = None,   # <-- Optional added
                 lambda_degree_importance: float = 1.0,
                 noise_degree_factor: float = 2.0,
                 degree_temperature: Optional[float] = None,
                 degree_min_val: float = 0.0, # Changed from degree_median
                 degree_range_val: float = 1.0, # Changed from degree_iqr
                 lambda_node_exist_importance: float = 1.0,
                 use_edge_supervision: bool = False,
                 lambda_edge_importance: float = 1.0,
                 exist_pos_weight: Union[torch.Tensor, float] = 1.0):
        super().__init__()
        self.save_hyperparameters(ignore=['verbose'])
        
        # Ensure even dimension for sinusoidal embedding
        if latent_embedding_dimension % 2 != 0:
            raise ValueError("latent_embedding_dimension must be even")
            
        # ------------------------------------------------------------------
        # Sanity-check `max_degree`
        # ------------------------------------------------------------------
        if max_degree is None or max_degree < 0:
            raise ValueError(
                "`max_degree` must be a non-negative int; "
                "call ConditionalNodeGenerator.setup() first so it can "
                "compute the value from data."
            )

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


        # Store degree scaling parameters
        self.register_buffer('deg_min_val', torch.tensor(degree_min_val, dtype=torch.float32))
        # Integer offset that aligns class 0 → actual degree = deg_min_val
        self.register_buffer('deg_offset', torch.tensor(int(round(degree_min_val)), dtype=torch.long))
        self.register_buffer('deg_range_val', torch.tensor(max(degree_range_val, 1e-8), dtype=torch.float32))

        # Initialize metric lists (unchanged names – keeps plotting intact)
        self.train_losses = []
        self.val_losses   = []
        self.train_deg_ce = []
        self.val_deg_ce   = []
        self.train_exist    = []
        self.val_exist      = []
        if self.use_edge_supervision:
            self.train_edge_loss = []
            self.val_edge_loss   = []
            self.train_edge_acc = []
            self.val_edge_acc = []

        # Model layers
        self.layernorm_in = nn.LayerNorm(input_feature_dimension)
        self.linear_encoder_input_to_latent = nn.Linear(input_feature_dimension, latent_embedding_dimension)
        self.linear_encoder_condition_to_latent = nn.Linear(condition_feature_dimension, latent_embedding_dimension)
        
        # Stack of permutation-equivariant ISABs for node processing
        self.shared_transformer = nn.ModuleList([
            ISAB(
                dim   = self.latent_embedding_dimension,
                heads = self.transformer_attention_head_count,
                m     = num_inducing_points,
                dropout = self.transformer_dropout
            )
            for _ in range(self.number_of_transformer_layers)
        ])
        
        # conditioning-set processing and cross-attention
        self.isab_cond = ISAB(
            dim   = self.latent_embedding_dimension,
            heads = self.transformer_attention_head_count,
            m     = num_inducing_points,
            dropout = self.transformer_dropout
        )
        self.cross_attn_nodes_to_cond = MiniMHA(
            dim     = self.latent_embedding_dimension,
            heads   = self.transformer_attention_head_count,
            dropout = self.transformer_dropout,   # propagate
        )
        
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
        x_norm = self.layernorm_in(input_rows)
        latent_tokens = self.linear_encoder_input_to_latent(x_norm)
        
        # ---------- build conditioning token set ----------
        if global_condition_vector.ndim == 2:      # (B,C) → (B,1,C)
            global_condition_vector = global_condition_vector.unsqueeze(1)
        B, M, Cdim = global_condition_vector.shape

        cond_latent = self.linear_encoder_condition_to_latent(
            global_condition_vector.view(-1, Cdim)
        ).view(B, M, -1)                            # (B,M,D)

        # add SAME time embedding to every cond token
        time_emb = get_sinusoidal_time_embedding(
            diffusion_time_step.squeeze(-1) if diffusion_time_step.ndim == 3 else diffusion_time_step,
            self.latent_embedding_dimension
        ).to(latent_tokens.dtype)                  # (B,D)
        cond_latent = cond_latent + time_emb.unsqueeze(1)     # broadcast over M

        # permutation-invariant processing of cond set
        cond_latent = self.isab_cond(cond_latent)             # (B,M,D)

        # cross-attention: nodes query cond set
        latent_tokens = latent_tokens + self.cross_attn_nodes_to_cond(
            latent_tokens, cond_latent, cond_latent
        )
        
        # Run node ISAB stack
        for block in self.shared_transformer:
            latent_tokens = block(latent_tokens)
        
        # Generate predictions from all heads
        v_field = self.linear_decoder_latent_to_output(latent_tokens)
        logits_deg = self.degree_head(latent_tokens)
        logits_exist = self.exist_head(latent_tokens).squeeze(-1)  # shape (B,N)
        if return_latents:
            return v_field, logits_deg, logits_exist, latent_tokens
        return v_field, logits_deg, logits_exist

    # ───────────────────────────────────────────────────────────────────
    #  Auxiliary losses (existence & degree)
    # ───────────────────────────────────────────────────────────────────
    def compute_aux_losses(
        self,
        x_t: torch.Tensor,
        logits_deg: torch.Tensor,
        logits_exist: torch.Tensor,
        x0: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        target_exist = (x0[..., 0] >= 0.5).float()
        loss_exist = F.binary_cross_entropy_with_logits(
            logits_exist, target_exist, pos_weight=self.exist_pos_weight
        )

        deg_unscaled = x0[..., self.important_feature_index] * self.deg_range_val + self.deg_min_val
        true_deg_cls = torch.clamp(
            torch.round(deg_unscaled) - self.deg_offset,  # shift so min → 0
            0, self.max_degree
        ).long()
        loss_deg = F.cross_entropy(
            logits_deg.reshape(-1, self.max_degree + 1),
            true_deg_cls.reshape(-1)
        )

        aux_total = (
            self.lambda_node_exist_importance * loss_exist +
            self.lambda_degree_importance * loss_deg
        )
        return {
            "aux_total": aux_total,
            "deg_ce":    loss_deg,
            "exist":     loss_exist,
        }

    # ───────────────────────────────────────────────────────────────────
    #  Flow-matching coupling: x_t  &   target velocity  v*
    # ───────────────────────────────────────────────────────────────────
    def sample_xt_and_target_v(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Linear coupling  x_t = (1-t)·x0 + t·z   with   z ~ N(0,1)
        Target velocity  v*  =  d/dt x_t = z − x0
        
        Args:
            x0: Input tensor of shape (B, N, D)
            t: Time tensor of shape (B, 1) or (B, 1, 1)
        """
        if t.ndim == 2:  # (B,1) → (B,1,1)
            t = t.unsqueeze(-1)
        z = torch.randn_like(x0)
        x_t = (1.0 - t) * x0 + t * z
        v_star = z - x0
        return x_t, v_star

    # ---------------------------------------------------------------------------
    # TRAINING STEP – uses the dict
    # ---------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        if self.use_edge_supervision:
            input_examples, global_condition, edge_idx, edge_labels, node_mask = batch
        else:
            input_examples, global_condition = batch

        # ❶ draw random time t ~ U(0,1) with correct broadcasting shape
        t = torch.rand(input_examples.size(0), 1, 1, device=self.device)
        # ❷ build (x_t , v*)
        x_t, v_star = self.sample_xt_and_target_v(input_examples, t)

        # ❸ predict velocity field
        if self.use_edge_supervision:
            v_pred, logits_deg, logits_exist, latent_tokens = self.forward(
                x_t, global_condition, t, return_latents=True
            )
        else:
            v_pred, logits_deg, logits_exist = self.forward(x_t, global_condition, t)

        # if we want to remove the existence and degree columns from the loss uncomment the following:
        # ❹ main FM loss (ignore existence & degree columns)
        # mask = torch.ones_like(v_pred)
        # mask[..., 0] = 0
        # mask[..., self.important_feature_index] = 0
        # loss_fm = F.mse_loss(v_pred * mask, v_star * mask)

        loss_fm = F.mse_loss(v_pred, v_star)

        # ❺ auxiliary losses
        aux = self.compute_aux_losses(x_t, logits_deg, logits_exist, input_examples)
        loss = loss_fm + aux["aux_total"]

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
            else:
                # No edges this batch – log NaN so graphs show the gap
                self.log("train_edge_loss", torch.nan,
                         on_step=False, on_epoch=True, prog_bar=False)
                self.log("train_edge_acc",  torch.nan,
                         on_step=False, on_epoch=True, prog_bar=False)

        # Log metrics
        self.log("train_total",  loss,              on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_deg_ce", aux["deg_ce"],     on_step=False, on_epoch=True)
        self.log("train_exist",  aux["exist"],      on_step=False, on_epoch=True)

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

        # ❶ draw random time t ~ U(0,1) with correct broadcasting shape
        t = torch.rand(input_examples.size(0), 1, 1, device=self.device)
        # ❷ build (x_t , v*)
        x_t, v_star = self.sample_xt_and_target_v(input_examples, t)

        # ❸ predict velocity field
        if self.use_edge_supervision:
            v_pred, logits_deg, logits_exist, latent_tokens = self.forward(
                x_t, global_condition, t, return_latents=True
            )
        else:
            v_pred, logits_deg, logits_exist = self.forward(x_t, global_condition, t)

        # if we want to remove the existence and degree columns from the loss uncomment the following:
        # ❹ main FM loss (ignore existence & degree columns)
        # mask = torch.ones_like(v_pred)
        # mask[..., 0] = 0
        # mask[..., self.important_feature_index] = 0
        # loss_fm = F.mse_loss(v_pred * mask, v_star * mask)
        
        loss_fm = F.mse_loss(v_pred, v_star)

        # ❺ auxiliary losses
        aux = self.compute_aux_losses(x_t, logits_deg, logits_exist, input_examples)
        loss = loss_fm + aux["aux_total"]


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
            else:
                # No edges this batch – log NaN so graphs show the gap
                self.log("val_edge_loss", torch.nan,
                         on_step=False, on_epoch=True, prog_bar=False)
                self.log("val_edge_acc",  torch.nan,
                         on_step=False, on_epoch=True, prog_bar=False)

        # Log metrics
        self.log("val_total",  loss,              on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_deg_ce", aux["deg_ce"],     on_step=False, on_epoch=True)
        self.log("val_exist",  aux["exist"],      on_step=False, on_epoch=True)

        return loss

    def on_train_end(self):
        if not self.verbose:
            return
        plot_metrics(
            train_metrics = {
                "total": self.train_losses,
                "deg_ce": self.train_deg_ce,
                "exist": self.train_exist,
                **({"edge": self.train_edge_loss} if self.use_edge_supervision else {})
            },
            val_metrics = {
                "total": self.val_losses,
                "deg_ce": self.val_deg_ce,
                "exist": self.val_exist,
                **({"edge": self.val_edge_loss} if self.use_edge_supervision else {})
            },
            window=10,
            alpha=0.1
        )
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    @torch.no_grad()
    def generate(self, global_condition: torch.Tensor, total_steps: int = 100) -> torch.Tensor:
        """
        Generate samples by integrating the learned probability flow ODE.
        
        Starting from Gaussian noise, follows the learned vector field backwards in time
        using **Heun's (RK-2) predictor-corrector** steps.

        Args:
            global_condition: Tensor of shape (B, C) containing graph-level conditions
            total_steps: Number of integration steps (default: 100)
        Returns:
            Tensor of shape (B, N, D) containing the generated samples
        """
        B = global_condition.size(0)
        x = torch.randn(
            B,
            self.number_of_rows_per_example,
            self.input_feature_dimension,
            device=global_condition.device,
        )
        dt = -1.0 / total_steps
        t  = torch.full((B, 1), 1.0, device=global_condition.device)
        for _ in range(total_steps):
            # --- Heun's method (predictor-corrector) ---
            # 1) slope at the beginning
            v1, logits_deg1, logits_exist1 = self.forward(x, global_condition, t)

            # 2) provisional step
            x_temp = x + v1 * dt
            t_next = t + dt

            # 3) slope at the end
            v2, logits_deg2, logits_exist2 = self.forward(x_temp, global_condition, t_next)

            # 4) corrector
            x = x + 0.5 * (v1 + v2) * dt

            # 5) keep auxiliary channels coherent (use slope at t_next)
            x[..., 0] = torch.sigmoid(logits_exist2)
            deg_cls  = torch.softmax(logits_deg2, -1).argmax(-1)

            # Cast buffers to runtime dtype for AMP compatibility
            deg_min   = self.deg_min_val.to(x.dtype)
            deg_range = self.deg_range_val.to(x.dtype)
            deg_off   = self.deg_offset.to(x.dtype)

            deg_scaled = (deg_cls.to(x.dtype) + deg_off - deg_min) / deg_range
            x[..., self.important_feature_index] = deg_scaled.clamp(0.0, 1.0)

            # advance time
            t = t_next

        return x

# =============================================================================
# Helper classes for data handling (Dataset, DataLoader collate_fn)
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
        
        return x, y, edge_idxs, edge_lbls, mask

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

    # Check if there's any data to plot
    if not train_metrics or not val_metrics:
        print("No metrics data available for plotting.")
        return

    fig, ax0 = plt.subplots(figsize=(15, 8))
    metrics = list(train_metrics.keys())
    axes = [ax0] + [ax0.twinx() for _ in range(len(metrics) - 1)]
    for i, ax in enumerate(axes[1:], start=1):
        ax.spines['right'].set_position(('outward', 60 * i))
    
    # Initialize colors
    default_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ["C{}".format(i) for i in range(10)])
    colors = cycle(default_cycle)
    
    # Initialize empty lists for lines and labels that will be populated
    plot_lines = []
    plot_labels = []
    
    for name, ax in zip(metrics, axes):
        color = next(colors)
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
            plot_lines.extend([l1, l2])
            plot_labels.extend([f"Train {name} (MA{window})", f"Val {name} (MA{window})"])
        
        ax.set_ylabel(name, color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_yscale('log')
    
    # Only create legend if we have lines to show
    if plot_lines:
        fig.legend(plot_lines, plot_labels, loc='upper center', 
                  ncol=len(plot_lines)//2, fontsize='small')
    
    ax0.set_xlabel("Epoch")
    ax0.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()

class ConditionalNodeGenerator:
    """
    A scikit-learn compatible flow-matching generator that wraps a transformer-based model
    for structured graph generation. This model learns a continuous vector field that 
    transports noise to data through an ODE flow, conditioned on global graph properties.

    The generator is particularly suited for graph-like structures where each example
    consists of multiple rows (nodes) and features, with special handling for structural
    constraints like node degrees and existence.

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
    
    total_steps : int, default=100
        Number of Euler steps for ODE integration during generation.
        Flow matching typically needs fewer steps (10-100) than diffusion.
    
    verbose : bool, default=False
        Whether to print training progress and display metric plots.
    
    important_feature_index : int, default=1
        Index of the feature to be treated with special importance (typically degree).
        This feature's prediction is handled by an auxiliary head.
    
    lambda_degree_importance : float, default=1.0
        Weight multiplier for the degree prediction loss term.
        Higher values prioritize accurate degree predictions.
    
    noise_degree_factor : float, default=2.0
        Factor by which to reduce noise on the degree feature (relevant for DAEs, less so for FM's direct velocity prediction but kept for consistency if model params are shared/reused).
    
    degree_temperature : Optional[float], default=None
        Temperature for degree sampling during generation (if stochastic sampling from logits is desired).
        None means deterministic (argmax).
    
    lambda_node_exist_importance : float, default=1.0
        Weight multiplier for the node existence prediction loss term.
    
    default_exist_pos_weight : float, default=1.0
        Class weight for positive examples in node existence prediction.
        Useful for handling class imbalance.
    
    lambda_edge_importance : float, default=1.0
        Weight multiplier for the edge prediction loss term when using
        edge supervision.
    
    gradient_clip_val : float, default=1.0
        Maximum gradient norm (or absolute value, depending on algorithm)
        per optimization step. Set to ``None`` to disable.
    
    Methods
    -------
    fit(node_encodings_list, conditional_graph_encodings, edge_pairs=None, ...)
        Fit the model to training data, optionally with edge supervision.
    
    predict(y)
        Generate samples conditioned on the given conditional encodings using ODE integration.
    
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
                 total_steps: int = 100, # Default changed from 1000 to 100
                 verbose: bool = False,
                 important_feature_index: int = 1,
                 lambda_degree_importance: float = 1.0,
                 noise_degree_factor: float = 2.0,
                 degree_temperature: Optional[float] = None,
                 lambda_node_exist_importance: float = 1.0,
                 default_exist_pos_weight: float = 1.0,
                 lambda_edge_importance: float = 1.0,
                 gradient_clip_val: float = 1.0):   # NEW
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
        self.gradient_clip_val = gradient_clip_val

        self.number_of_rows_per_example = None
        self.input_feature_dimension = None
        self.model = None
        self.x_scaler = None # Scaler for node features
        self.y_scaler = None # Scaler for conditional features
        self.deg_max_val = None
        self.deg_min_val = None           # NEW – cache lower bound

    def _fit_scalers(self, X_array, y_array):
        B, n, d = X_array.shape

        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()

        # Fit scaler for X (node features)
        # X_array has shape (B, N, D_x), reshape to (B*N, D_x) for scaler
        num_features_x = X_array.shape[2]
        self.x_scaler.fit(X_array.reshape(-1, num_features_x))

        # Fit scaler for Y (conditional features)
        self.y_scaler.fit(y_array)

    def _transform_data(self, X_array, y_array):
        B, N, D_x = X_array.shape
        X_scaled_flat = self.x_scaler.transform(X_array.reshape(-1, D_x))
        X_scaled = X_scaled_flat.reshape(B, N, D_x)
        
        y_scaled = self.y_scaler.transform(y_array)
        return X_scaled, y_scaled

    def _inverse_transform_input(self, X_array):
        """
        Transform array back to original scale.
        
        Clips the degree feature to [deg_min_val, deg_max_val] so rounding/offset
        never produces out-of-range values.
        """
        B, N, D_x = X_array.shape 
        X_orig_flat = self.x_scaler.inverse_transform(X_array.reshape(-1, D_x))
        X_orig = X_orig_flat.reshape(B, N, D_x) 
        lower = 0.0 if self.deg_min_val is None else self.deg_min_val
        X_orig[..., self.important_feature_index] = np.clip(
            X_orig[..., self.important_feature_index],
            lower,
            self.deg_max_val
        )
        return X_orig

    def setup(
        self,
        node_encodings_list: List[np.ndarray],
        conditional_graph_encodings: Any,
        edge_pairs: Optional[List[Tuple[int, int, int]]] = None,
        edge_targets: Optional[np.ndarray] = None,
        node_mask: Optional[np.ndarray] = None
    ):
        """
        Setup the model for training.

        This method prepares the data, initializes scalers, and sets up the
        FMNodeTransformer model. It computes scaling
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
        
        self._fit_scalers(X_array, y_array)

        # ------------------------------------------------------------
        # Compute class-imbalance weight for BCEWithLogitsLoss
        # ------------------------------------------------------------
        exist_mask = (X_array[..., 0] >= 0.5)
        ones  = int(exist_mask.sum())                 # rows where exist == 1
        zeros = int(exist_mask.size) - ones           # rows where exist == 0

        if ones == 0:
            exist_pos_weight = 1.0                    # avoid div-by-zero
        elif zeros > ones:                            # positives rarer
            exist_pos_weight = float(zeros) / float(ones)
        else:                                         # positives majority or equal
            exist_pos_weight = 1.0

        deg_column_data_for_minmax = X_array[..., self.important_feature_index].reshape(-1, 1)
        temp_degree_scaler = MinMaxScaler().fit(deg_column_data_for_minmax) 
        
        deg_min_val = temp_degree_scaler.data_min_[0]
        self.deg_min_val = deg_min_val           # remember for inverse-xform
        deg_range_val = temp_degree_scaler.data_range_[0]
        if deg_range_val == 0: 
            deg_range_val = 1e-8 
        
        X_scaled, y_scaled = self._transform_data(X_array, y_array)
        self.input_feature_dimension = X_scaled.shape[2]
        cond_feature_dim = y_scaled.shape[1]
        
        raw_degrees = X_array[..., self.important_feature_index]  # shape (B, N)
        self.deg_max_val = int(raw_degrees.max())  # global max
        
        # Initialize the model with FM backbone
        self.model = FMNodeTransformer(
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
            max_degree=self.deg_max_val,
            lambda_degree_importance=self.lambda_degree_importance,
            noise_degree_factor=self.noise_degree_factor,
            degree_temperature=self.degree_temperature,
            degree_min_val=deg_min_val,
            degree_range_val=deg_range_val,
            lambda_node_exist_importance=self.lambda_node_exist_importance,
            use_edge_supervision=(edge_pairs is not None),
            lambda_edge_importance=self.lambda_edge_importance,
            exist_pos_weight=exist_pos_weight,
        )

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
            
            dataset_size = len(node_encodings_list)  
            train_size = int(0.9 * dataset_size)
            val_size = dataset_size - train_size
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
                shuffle=False,  
                collate_fn=collate_graph_with_edges
            )
        else:
            dataset = TensorDataset(X_tensor, y_tensor)
            
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        trainer = pl.Trainer(
            max_epochs=self.maximum_epochs,
            callbacks=[MetricsLogger()],  # Required for plotting
            logger=True,
            enable_checkpointing=False,
            enable_progress_bar=False,
            # ↓↓↓ gradient clipping ↓↓↓
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm="norm",    # or "value"
        )
        
        if not any(isinstance(cb, MetricsLogger) for cb in trainer.callbacks):
            import warnings
            warnings.warn(
                "MetricsLogger callback is required for plot_metrics() to work. "
                "Training will proceed but plotting will be disabled.",
                RuntimeWarning
            )

        if not self.verbose:
            try:
                with suppress_output():
                    trainer.fit(self.model,
                                train_dataloaders=train_loader,
                                val_dataloaders=val_loader)
            except Exception as e:
                # Restore output streams before re-raising so the traceback is visible
                print("Training failed – re-raising exception.")
                raise
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
                "exist": self.model.train_exist,
                **({"edge": self.model.train_edge_loss} if self.model.use_edge_supervision else {})
            },
            val_metrics = {
                "total": self.model.val_losses,
                "deg_ce": self.model.val_deg_ce,
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
            y_scaled = self.y_scaler.transform(y) # Use y_scaler for conditional input
            y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
            generated = self.model.generate(y_tensor, total_steps=self.total_steps)
            generated_np = generated.cpu().numpy()
            generated_orig = self._inverse_transform_input(generated_np)
            return [generated_orig[i] for i in range(generated_orig.shape[0])]

class MetricsLogger(pl.callbacks.Callback):
    """Collects epoch-level metrics so they can be plotted after training."""

    @staticmethod
    def _get_metric(metrics: Dict[str, torch.Tensor], name: str) -> float:
        """
        Grab a scalar from Lightning's `callback_metrics`, falling back to the
        `<name>_epoch` key used by older PL versions.
        """
        return metrics.get(name,
                           metrics.get(f"{name}_epoch",
                                       torch.tensor(0.0))).item()

    # ------------- training -----------------
    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        pl_module.train_losses.append(self._get_metric(m, "train_total"))
        pl_module.train_deg_ce.append(self._get_metric(m, "train_deg_ce"))
        pl_module.train_exist.append(self._get_metric(m, "train_exist"))
        if pl_module.use_edge_supervision:
            pl_module.train_edge_loss.append(self._get_metric(m, "train_edge_loss"))
            pl_module.train_edge_acc.append(self._get_metric(m, "train_edge_acc"))

    # ------------- validation ---------------
    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        pl_module.val_losses.append(self._get_metric(m, "val_total"))
        pl_module.val_deg_ce.append(self._get_metric(m, "val_deg_ce"))
        pl_module.val_exist.append(self._get_metric(m, "val_exist"))
        if pl_module.use_edge_supervision:
            pl_module.val_edge_loss.append(self._get_metric(m, "val_edge_loss"))
            pl_module.val_edge_acc.append(self._get_metric(m, "val_edge_acc"))


if __name__ == "__main__":
    B,N,D = 2, 10, 64
    x  = torch.randn(B,N,D)
    y  = torch.randn(B,16)
    t  = torch.rand(B,1,1)

    model = FMNodeTransformer(
        number_of_rows_per_example=N,
        input_feature_dimension=D,
        condition_feature_dimension=y.size(1),
        latent_embedding_dimension=D,
        number_of_transformer_layers=2,
        transformer_attention_head_count=4,
        max_degree=5
    )

    v0, *_ = model(x, y, t)
    perm = torch.randperm(N)
    v1, *_ = model(x[:, perm], y, t)
    assert torch.allclose(v0[:, perm], v1, atol=1e-5)
    print("Permutation-equivariance test passed ✅")
