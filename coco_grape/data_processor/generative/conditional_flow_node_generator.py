import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from scipy.stats import pearsonr  # Add this import
import matplotlib.pyplot as plt
import contextlib, os, sys, math  # Added math to imports
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import random_split, DataLoader, TensorDataset, Dataset
from typing import Dict, Sequence, Optional, Union, Tuple, List, Any
from itertools import cycle  # Add this import
from sklearn.model_selection import train_test_split

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

# =============================================================================
# Flow-Matching Graph Generator with Cross-Attention
# ────────────────────────────────────────────────────────────────────────────
class GuidanceMLP(nn.Module):
    """
    MLP classifier for providing gradient-based guidance during sampling.
    
    This network pools the transformer's latent representations into a single
    vector and maps it to class logits, allowing the flow to be guided toward
    specific target classes during generation.

    Architecture:
        input → Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear → output
        with residual connections between the hidden layers.

    Args:
        input_dim: Dimension of pooled transformer latents
        hidden_dim: Width of both hidden layers (usually 2×input_dim)
        output_dim: Number of target classes
        dropout: Drop probability after each activation (default: 0.2)
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


class FMNodeTransformer(pl.LightningModule):
    """A transformer-based continuous-time generative model using flow matching.
    
    This model learns a time-dependent vector field that describes the probability
    flow from noise to data. It combines:
    - Cross-attention between node features and global conditions
    - Self-attention between nodes for capturing graph structure
    - Auxiliary prediction heads for degrees and node existence
    - Optional classifier guidance for controlled generation
    - SDE support for stochastic sampling (β·t noise schedule)
    
    The flow field is parameterized by a stack of transformer layers that process
    node features while attending to timestep and condition embeddings.

    Args:
        number_of_rows_per_example: Maximum nodes per graph (for padding)
        input_feature_dimension: Node feature dimension
        condition_feature_dimension: Global condition dimension
        latent_embedding_dimension: Width of transformer layers (must be even)
        number_of_transformer_layers: Depth of the transformer stack
        transformer_attention_head_count: Number of attention heads per layer
        transformer_dropout: Dropout rate in transformer (default: 0.1)
        learning_rate: Adam learning rate (default: 1e-3)
        important_feature_index: Index of degree feature for special handling
        max_degree: Maximum node degree for classification
        lambda_degree_importance: Weight for degree loss (default: 1.0)
        noise_degree_factor: Factor by which to reduce noise on degrees
        degree_temperature: Temperature for degree sampling (None = deterministic)
        sde_beta: Strength of SDE noise during sampling (default: 0.1)
        use_edge_supervision: Whether to use edge-level supervision
        lambda_edge_importance: Weight for edge loss when supervised
    """
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
                 exist_pos_weight: Union[torch.Tensor, float] = 1.0,
                 # ── NEW: soft-mask schedule ───────────────────────────
                 mask_warmup_epochs: int = 30,
                 mask_temp_start: float = 0.25,
                 mask_temp_end:   float = 0.05,
                 mask_temp_decay_epochs: int = 20,
                 sde_beta: float = 0.1,            # SDE noise strength
                 lambda_cfm: float = 0.8,           # CFM strength
                 lambda_consistency: float = 0.1,    # Consistency loss weight
                 class_weights: Optional[np.ndarray] = None, # FIX 3b: Add class weights parameter
                 ):
        # Change hyperparameter saving to exclude class_weights
        super().__init__()
        self.save_hyperparameters(ignore=['verbose', 'class_weights'])
        
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


        # ── store soft-mask schedule ──────────────────────────────────
        self.mask_warmup_epochs = mask_warmup_epochs
        self.mask_temp_start  = mask_temp_start
        self.mask_temp_end    = mask_temp_end
        self.mask_temp_decay_epochs = max(1, mask_temp_decay_epochs)

        # Store SDE noise strength & consistency loss weight
        self.sde_beta = sde_beta
        self.lambda_cfm = lambda_cfm              # Store CFM strength
        if lambda_consistency < 0:
            raise ValueError("lambda_consistency must be ≥ 0")
        self.lambda_consistency = lambda_consistency
        if verbose:
            print(f"λ_consistency = {self.lambda_consistency}")
            print(f"λ_cfm = {self.lambda_cfm}")

        # Store degree scaling parameters
        self.register_buffer('deg_min_val', torch.tensor(degree_min_val, dtype=torch.float32))
        # Integer offset that aligns class 0 → actual degree = deg_min_val
        self.register_buffer('deg_offset', torch.tensor(int(round(degree_min_val)), dtype=torch.long))
        self.register_buffer('deg_range_val', torch.tensor(max(degree_range_val, 1e-8), dtype=torch.float32))

        # FIX 3b: Store class weights buffer
        if class_weights is None:
            class_weights = np.ones(max_degree + 1, dtype=np.float32)
        self.register_buffer(
            "deg_class_weights",
            torch.as_tensor(class_weights, dtype=torch.float32)
        )

        # Initialize metric lists (unchanged names – keeps plotting intact)
        self.train_losses = []
        self.val_losses   = []
        self.train_deg_loss = []
        self.val_deg_loss   = []
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
        
        # Replace ISAB stack with standard TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_embedding_dimension,
            nhead=self.transformer_attention_head_count,
            dim_feedforward=4 * self.latent_embedding_dimension,
            dropout=self.transformer_dropout,
            activation=F.gelu,  # Changed from ReLU to GELU
            batch_first=True,
            norm_first=True,    # Apply normalization before attention (better stability)
            layer_norm_eps=1e-5 # Explicit epsilon for layer norm
        )
        self.shared_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.number_of_transformer_layers,
            norm=nn.LayerNorm(self.latent_embedding_dimension) # Add final layer norm
        )
        
        # Replace ISAB + MiniMHA with single MultiheadAttention for cross-attention
        self.cross_attn_nodes_to_cond = nn.MultiheadAttention(
            embed_dim=self.latent_embedding_dimension,
            num_heads=self.transformer_attention_head_count,
            dropout=self.transformer_dropout,
            batch_first=True
        )
        
        # NEW: Add residual MLP to boost condition pathway
        self.cond_mlp = nn.Sequential(
            nn.LayerNorm(latent_embedding_dimension),
            nn.Linear(latent_embedding_dimension, 4 * latent_embedding_dimension),
            nn.GELU(),
            nn.Linear(4 * latent_embedding_dimension, latent_embedding_dimension)
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

        # Guidance components
        self.use_guidance = False
        self.guidance_weight = 1.0
        self.guidance_classifier = None  # Initialized later

        # ────────────────────────────────────────────────────────────────
        #  (2) FiLM conditioning after every encoder block
        #      γ, β ← MLP(cond agg)   ;   nodes ← γ·nodes + β
        # ────────────────────────────────────────────────────────────────
        self.film_mlp = nn.Sequential(
            nn.LayerNorm(latent_embedding_dimension),
            nn.Linear(latent_embedding_dimension, 4 * latent_embedding_dimension),
            nn.GELU(),
            nn.Linear(4 * latent_embedding_dimension, 2 * latent_embedding_dimension)
        )

        # ────────────────────────────────────────────────────────────────
        #  (3) learnable time-embedding MLP    e_t = MLP(sinusoid(t))
        # ────────────────────────────────────────────────────────────────
        self.time_mlp = nn.Sequential(
            nn.Linear(latent_embedding_dimension, 4 * latent_embedding_dimension),
            nn.GELU(),
            nn.Linear(4 * latent_embedding_dimension, latent_embedding_dimension)
        )

    def set_guidance_classifier(self, num_classes: int) -> None:
        """
        Initialize the classifier for guidance (must be called before training it).
        
        Args:
            num_classes: Number of target classes to distinguish
        """
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
    ) -> None:
        """
        Train the guidance classifier on a labeled dataset, with internal validation split
        and a loss plot at the end.
        """
        
        self.eval()
        self.guidance_classifier.train()
        optimizer = torch.optim.Adam(self.guidance_classifier.parameters(), lr=lr)

        # Pad variable-length node feature arrays to shape (B, N, D)
        max_rows = self.number_of_rows_per_example
        padded_node_feats = []
        for f in node_feats:
            n_rows, dim = f.shape
            if n_rows < max_rows:
                pad_width = ((0, max_rows - n_rows), (0, 0))
                f_padded = np.pad(f, pad_width, mode='constant', constant_values=0)
            else:
                f_padded = f[:max_rows]
            padded_node_feats.append(f_padded)

        X = torch.tensor(np.stack(padded_node_feats), dtype=torch.float32, device=self.device)
        Y = torch.tensor(cond_vecs, dtype=torch.float32, device=self.device)
        labels = torch.tensor(labels, dtype=torch.long, device=self.device)
        T = torch.zeros(X.size(0), 1, 1, device=self.device)

        # --- Internal train/val split ---
        X_train, X_val, Y_train, Y_val, labels_train, labels_val = train_test_split(
            X.cpu().numpy(), Y.cpu().numpy(), labels.cpu().numpy(), test_size=0.2, random_state=42
        )
        X_train = torch.tensor(X_train, device=self.device)
        X_val = torch.tensor(X_val, device=self.device)
        Y_train = torch.tensor(Y_train, device=self.device)
        Y_val = torch.tensor(Y_val, device=self.device)
        labels_train = torch.tensor(labels_train, device=self.device)
        labels_val = torch.tensor(labels_val, device=self.device)
        T_train = T[:len(X_train)]
        T_val = T[:len(X_val)]

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # --- Train pass ---
            _, _, _, latents_train = self.forward(X_train, Y_train, T_train, return_latents=True)
            logits_train = self.guidance_classifier(latents_train.mean(dim=1))
            loss_train = F.cross_entropy(logits_train, labels_train)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            train_losses.append(loss_train.item())

            # --- Validation pass ---
            with torch.no_grad():
                _, _, _, latents_val = self.forward(X_val, Y_val, T_val, return_latents=True)
                logits_val = self.guidance_classifier(latents_val.mean(dim=1))
                loss_val = F.cross_entropy(logits_val, labels_val)
                val_losses.append(loss_val.item())

        if verbose:
            print(f"Guidance classifier trained for {epochs} epochs with learning rate {lr}.")
            print(f"Final train loss: {train_losses[-1]:.4f}, val loss: {val_losses[-1]:.4f}")
            # --- Plot losses ---
            plt.figure(figsize=(15, 8))
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Val Loss")
            plt.yscale('log')
            plt.xlabel("Epoch")
            plt.ylabel("Cross-Entropy Loss")
            plt.title("Guidance Classifier Training")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()


    def forward(self, input_rows, global_condition_vector, diffusion_time_step, return_latents: bool = False) -> tuple:
        x_norm = self.layernorm_in(input_rows)
        latent_tokens = self.linear_encoder_input_to_latent(x_norm)
        
        # ---------- build conditioning & time token set ----------
        if global_condition_vector.ndim == 2:      # (B,C) → (B,1,C)
            global_condition_vector = global_condition_vector.unsqueeze(1)
        B, M, Cdim = global_condition_vector.shape

        cond_latent = self.linear_encoder_condition_to_latent(
            global_condition_vector.view(-1, Cdim)
        ).view(B, M, -1)                            # (B,M,D)

        # NEW: Apply residual MLP boost to condition pathway
        cond_latent = cond_latent + self.cond_mlp(cond_latent)   # (B,M,D) residual boost

        # ---- create explicit [time] token with learned MLP ----
        raw_t = get_sinusoidal_time_embedding(
            diffusion_time_step.squeeze(-1) if diffusion_time_step.ndim == 3 else diffusion_time_step,
            self.latent_embedding_dimension
        ).to(latent_tokens.dtype)                  # (B,D)
        time_emb = self.time_mlp(raw_t)            # (B,D)
        time_token = time_emb.unsqueeze(1)         # (B,1,D)

        # Concatenate: [time] + cond₁…M + nodes
        seq = torch.cat([time_token, cond_latent, latent_tokens], dim=1)  # (B,1+M+N,D)
        start_nodes = 1 + M                                               # index where node block starts

        # Cross-attention: nodes attend to (time+cond)
        latent_tokens, _ = self.cross_attn_nodes_to_cond(
            query=latent_tokens,
            key=seq[:, :start_nodes],      # (time+cond)
            value=seq[:, :start_nodes],
            need_weights=False
        )

        # Re-insert updated node tokens into the sequence so every
        # transformer layer sees the conditioning information.
        seq = torch.cat(
            [seq[:, :start_nodes],   # [time] + cond  (unchanged)
             latent_tokens],         # NEW node tokens  (contain cond info)
            dim=1
        )

        # Push the *full* sequence through the encoder stack, injecting FiLM
        for enc_layer in self.shared_transformer.layers:
            seq = enc_layer(seq)           # standard transformer layer

            # --- FiLM on the *node* slice -----------------------------
            cond_context = cond_latent.mean(1)              # (B,D)
            gamma, beta = self.film_mlp(cond_context).chunk(2, dim=-1)  # two (B,D)

            nodes = seq[:, start_nodes:]                    # (B,N,D)
            nodes = gamma.unsqueeze(1) * nodes + beta.unsqueeze(1)
            seq   = torch.cat([seq[:, :start_nodes], nodes], dim=1)

        # final norm
        if self.shared_transformer.norm is not None:
            seq = self.shared_transformer.norm(seq)

        # Extract UPDATED latent tokens post-FiLM
        latent_tokens = seq[:, start_nodes:]        # <-- updated nodes
        v_field      = self.linear_decoder_latent_to_output(latent_tokens)
        logits_deg   = self.degree_head(latent_tokens)
        logits_exist = self.exist_head(latent_tokens).squeeze(-1)

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
        # FIX 3: Use class weights in degree loss
        loss_deg = F.cross_entropy(
            logits_deg.view(-1, self.max_degree + 1),
            true_deg_cls.view(-1),
            weight=self.deg_class_weights      # now non-uniform
        )

        aux_total = (
            self.lambda_node_exist_importance * loss_exist +
            self.lambda_degree_importance * loss_deg
        )
        return {
            "aux_total": aux_total,
            "deg_loss": loss_deg,     # Changed from "deg_ce"
            "exist":    loss_exist,
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
        
        # FIX 1: Damped noise for degree column
        noise_scale = torch.ones_like(x0)
        noise_scale[..., self.important_feature_index] /= self.noise_degree_factor
        z = torch.randn_like(x0) * noise_scale
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

        # ---------------------------------------------------------------
        # (A)  SOFT GATE instead of hard mask (delay hard masking)
        #      warm-up: first N epochs every row has weight = 1
        # ---------------------------------------------------------------
        with torch.no_grad():                     # keep grads out of gate
            p_exist = torch.sigmoid(logits_exist).unsqueeze(-1)  # (B,N,1)
            if self.current_epoch < self.mask_warmup_epochs:
                gate = torch.ones_like(p_exist)                  # all rows
            else:
                T = self._current_temperature()              # <-- NEW
                gate = torch.sigmoid((p_exist - 0.5) / T)
                self.log("gate_temp", T, on_epoch=True, prog_bar=False)

        mask_on  = gate
        mask_off = 1.0 - gate
        # optional: track how sharp the gate is becoming
        self.log("mean_gate", gate.mean(),
                 on_step=False, on_epoch=True, prog_bar=True)

        # ---------------------------------------------------------------
        # (B)  MAIN FM LOSS — average **only over active rows**
        # ---------------------------------------------------------------
        sq_err   = (v_pred - v_star).pow(2) * mask_on        # (B,N,D)
        num_on   = mask_on.sum().clamp(min=1.0)              # avoid /0
        loss_fm  = sq_err.sum() / num_on                     # scalar

        # ---------------------------------------------------------------
        # (C)  CONSISTENCY LOSS — unchanged, but keep float mask
        # ---------------------------------------------------------------
        loss_consistency = F.mse_loss(
            v_pred * mask_off,
            torch.zeros_like(v_pred) * mask_off
        )

        # ---------------------------------------------------------------
        # (D)  CFM "push" term — mask it too so junk rows stay silent
        # ---------------------------------------------------------------
        y_shuf   = global_condition[torch.randperm(global_condition.size(0))]
        v_wrong  = self.forward(x_t.detach(), y_shuf, t)[0]
        loss_push = (((v_wrong - v_star).pow(2) * mask_on).sum()
                    / num_on)

        loss_cfm  = loss_fm + self.lambda_cfm * loss_push

        # ❺ auxiliary losses
        aux  = self.compute_aux_losses(x_t, logits_deg, logits_exist, input_examples)
        loss = (loss_cfm + 
                aux["aux_total"] +
                self.lambda_consistency * loss_consistency)

        self.log("train_consist",
                 loss_consistency,
                 on_step=False, on_epoch=True)

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

                loss = loss + self.lambda_edge_importance * (loss_e / num_on)
            else:
                # No edges this batch – log NaN so graphs show the gap
                self.log("train_edge_loss", torch.nan,
                         on_step=False, on_epoch=True, prog_bar=False)
                self.log("train_edge_acc",  torch.nan,
                         on_step=False, on_epoch=True, prog_bar=False)

        # Log metrics (add CFM push term)
        self.log("train_total",     loss,              on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_cfm_push",  loss_push,         on_step=False, on_epoch=True)
        self.log("train_cfm_loss",  loss_cfm,          on_step=False, on_epoch=True)
        self.log("train_deg_loss",  aux["deg_loss"],   on_step=False, on_epoch=True)
        self.log("train_exist",     aux["exist"],      on_step=False, on_epoch=True)

        # ── Exact inverse of linear coupling ───────────────────────────
        # x₀ ≈ x_t - t · v_pred   (broadcast t to (B,1,1) if needed)
        t_broadcast = t if t.ndim == 3 else t.unsqueeze(-1)
        x_pred = x_t - t_broadcast * v_pred.detach()
        return {"loss": loss, "node_pred": x_pred}
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

        # ---------------------------------------------------------------
        # (A)  same soft-gate used in training
        # ---------------------------------------------------------------
        with torch.no_grad():
            p_exist = torch.sigmoid(logits_exist).unsqueeze(-1)
            if self.current_epoch < self.mask_warmup_epochs:
                gate = torch.ones_like(p_exist)
            else:
                T = self._current_temperature()              # <-- NEW
                gate = torch.sigmoid((p_exist - 0.5) / T)
                self.log("gate_temp", T, on_epoch=True, prog_bar=False)

        mask_on  = gate
        mask_off = 1.0 - gate
        self.log("mean_gate", gate.mean(),
                 on_step=False, on_epoch=True, prog_bar=False)

        # ---------------------------------------------------------------
        # (B)  MAIN FM LOSS — average **only over active rows**
        # ---------------------------------------------------------------
        sq_err   = (v_pred - v_star).pow(2) * mask_on        # (B,N,D)
        num_on   = mask_on.sum().clamp(min=1.0)              # avoid /0
        loss_fm  = sq_err.sum() / num_on                     # scalar

        # ---------------------------------------------------------------
        # (C)  CONSISTENCY LOSS — unchanged, but keep float mask
        # ---------------------------------------------------------------
        loss_consistency = F.mse_loss(
            v_pred * mask_off,
            torch.zeros_like(v_pred) * mask_off
        )

        # ---------------------------------------------------------------
        # (D)  CFM "push" term — mask it too so junk rows stay silent
        # ---------------------------------------------------------------
        y_shuf   = global_condition[torch.randperm(global_condition.size(0))]
        v_wrong  = self.forward(x_t.detach(), y_shuf, t)[0]
        loss_push = (((v_wrong - v_star).pow(2) * mask_on).sum()
                    / num_on)

        loss_cfm  = loss_fm + self.lambda_cfm * loss_push

        # ❺ auxiliary losses
        aux  = self.compute_aux_losses(x_t, logits_deg, logits_exist, input_examples)
        # use the fully-masked CFM loss that matches training
        loss = (loss_cfm +
                aux["aux_total"] +
                self.lambda_consistency * loss_consistency)

        self.log("val_consist",
                 loss_consistency,
                 on_step=False, on_epoch=True)

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
                loss = loss + self.lambda_edge_importance * (loss_e / num_on)
            else:
                # No edges this batch – log NaN so graphs show the gap
                self.log("val_edge_loss", torch.nan,
                         on_step=False, on_epoch=True, prog_bar=False)
                self.log("val_edge_acc",  torch.nan,
                         on_step=False, on_epoch=True, prog_bar=False)

        # Log metrics (add CFM push term)
        self.log("val_total",     loss,              on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_cfm_push",  loss_push,         on_step=False, on_epoch=True)
        self.log("val_cfm_loss",  loss_cfm,          on_step=False, on_epoch=True)
        self.log("val_deg_loss",  aux["deg_loss"],   on_step=False, on_epoch=True)
        self.log("val_exist",     aux["exist"],      on_step=False, on_epoch=True)

        # Analytic inversion as in training_step
        t_broadcast = t if t.ndim == 3 else t.unsqueeze(-1)
        x_pred = x_t - t_broadcast * v_pred.detach()
        return {"loss": loss, "node_pred": x_pred}

    def on_train_end(self):
        if not self.verbose:
            return
        # Get the metrics logger callback
        metrics_logger = next(
            cb for cb in self.trainer.callbacks 
            if isinstance(cb, MetricsLogger)
        )
        
        # Plot both loss curves and correlations
        plot_metrics(
            train_metrics = {
                "total": self.train_losses,
                "deg_loss": self.train_deg_loss,
                "exist": self.train_exist,
                **({"edge": self.train_edge_loss} if self.use_edge_supervision else {})
            },
            val_metrics = {
                "total": self.val_losses,
                "deg_loss": self.val_deg_loss,
                "exist": self.val_exist,
                **({"edge": self.val_edge_loss} if self.use_edge_supervision else {})
            },
            window=10,
            alpha=0.1
        )
        metrics_logger.plot_correlations()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Linear noise schedule σ(t) = β·t for stochastic sampling.
        
        Args:
            t: Time tensor of shape (B,1) or (B,1,1)
        Returns: 
            Noise scale σ(t) of same shape as input
        """
        if t.ndim == 2:  # (B,1) → (B,1,1)
            t = t.unsqueeze(-1)
        return self.sde_beta * t

    def generate(
        self,
        global_condition: torch.Tensor,
        total_steps: int = 100,
        desired_class: Optional[Union[int, Sequence[int]]] = None
    ) -> torch.Tensor:
        """
        Generate samples by integrating the learned probability flow ODE.
        
        Args:
            global_condition: Tensor of shape (B, C) containing graph-level conditions
            total_steps: Number of integration steps (default: 100) 
            desired_class: int or array, class index(es) for classifier guidance.
                         If None, uses model's own prediction.
        Returns:
            Tensor of shape (B, N, D) containing the generated samples
        """
        B = global_condition.size(0)
        x = torch.randn(
            B, self.number_of_rows_per_example, self.input_feature_dimension,
            device=global_condition.device, requires_grad=True
        )
        dt = -1.0 / total_steps
        t = torch.full((B,1,1), 1.0, device=global_condition.device)

        for step in range(total_steps):
            # ---------- drift ----------
            v, logits_deg, logits_exist = self.forward(x, global_condition, t)

            # ---- optional classifier guidance ----
            if self.use_guidance and self.guidance_classifier is not None:
                x.requires_grad_(True)
                _, _, _, lat = self.forward(x, global_condition, t, return_latents=True)
                pooled = lat.mean(1)
                logits_c = self.guidance_classifier(pooled)
                if desired_class is None:
                    desired_class = logits_c.argmax(-1)
                selected = logits_c[torch.arange(B), desired_class]
                grad = torch.autograd.grad(selected.sum(), x, retain_graph=True)[0]
                w = (step + 1) / total_steps * self.guidance_weight
                v = v + w * grad

            # ---------- stochastic step (Euler–Maruyama) ----------
            sigma = self.sigma_t(t)                       # (B,1,1)
            # FIX 1: Same damped noise scale in sampling
            noise_scale = torch.ones_like(x)
            noise_scale[..., self.important_feature_index] /= self.noise_degree_factor
            noise = torch.randn_like(x) * noise_scale * sigma * math.sqrt(-dt)
            x = x + v * dt + noise
            t = t + dt

            # ---------- keep auxiliary channels coherent ----------
            _, logits_deg, logits_exist = self.forward(x, global_condition, t)
            x[..., 0] = torch.sigmoid(logits_exist)
            deg_cls   = logits_deg.softmax(-1).argmax(-1)           # (B,N)
            deg_min   = self.deg_min_val.to(x.dtype)
            deg_range = self.deg_range_val.to(x.dtype)
            real_deg  = deg_cls.to(x.dtype) + self.deg_offset        # un-normalised
            x[..., self.important_feature_index] = (
                (real_deg - deg_min) / deg_range
            ).clamp(0., 1.)

            x = x.detach().requires_grad_()

        # ------------------------------------------------------------
        # OPTIONAL CORRECTOR  (Heun-style)  – one last drift step at t≈0
        # helps every non-auxiliary feature snap into place.
        # ------------------------------------------------------------
        with torch.no_grad():
            v0 = self.forward(x, global_condition,
                              torch.zeros_like(t))[0]  # t = 0
        x = x - 0.5 * v0          # ½-step towards data manifold

        return x.detach()

    # ────────────────────────────────────────────────────────────────
    # Temperature schedule for the soft-mask gate
    # ────────────────────────────────────────────────────────────────
    def _current_temperature(self) -> float:
        """
        Piece-wise-linear decay of the gate temperature **T**.

        Epoch ranges
        -------------
        • 0 … mask_warmup_epochs - 1
              → constant `mask_temp_start`

        • mask_warmup_epochs … mask_warmup_epochs + mask_temp_decay_epochs - 1
              → linear decay *start → end*

        • afterwards
              → constant `mask_temp_end`
        """
        # Warm-up: keep a very smooth gate
        if self.current_epoch < self.mask_warmup_epochs:
            return self.mask_temp_start

        # Progress inside the decay window ∈ [0, 1]
        progress = min(
            1.0,
            (self.current_epoch - self.mask_warmup_epochs)
            / self.mask_temp_decay_epochs,
        )

        # Linear interpolation: T = (1-p)·T_start + p·T_end
        return (1.0 - progress) * self.mask_temp_start + progress * self.mask_temp_end

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
        
        if len(train) >= window:  # Only compute MA if we have enough points
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
    
    gradient_clip_val : float, default=5.0
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
                 total_steps: int = 100, 
                 verbose: bool = False,
                 important_feature_index: int = 1,
                 lambda_degree_importance: float = 1.0,
                 noise_degree_factor: float = 2.0,
                 degree_temperature: Optional[float] = None,
                 lambda_node_exist_importance: float = 1.0,
                 default_exist_pos_weight: float = 1.0,
                 lambda_edge_importance: float = 1.0,
                 gradient_clip_val: float = 5.0,
                 use_guidance: bool = False,
                 sde_beta: float = 0.1,            # SDE noise strength
                 lambda_cfm: float = 0.8,           # CFM strength
                 lambda_consistency: float = 0.1):    # Consistency loss weight
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
        self.use_guidance = use_guidance
        self.sde_beta = sde_beta
        self.lambda_cfm = lambda_cfm            # Store CFM strength
        self.lambda_consistency = lambda_consistency

        self.number_of_rows_per_example = None
        self.input_feature_dimension = None
        self.model = None
        self.x_scaler = None # Scaler for node features
        self.y_scaler = None # Scaler for conditional features
        self.deg_max_val = None
        self.deg_min_val = None           

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _inverse_transform_input(self, X_array: np.ndarray) -> np.ndarray:
        """
        Undo the Min-Max scaling applied to *node* features **and**
        clip the degree column so rounding/offset can never exceed the
        valid domain.

        Parameters
        ----------
        X_array : np.ndarray
            Normalised node-feature tensor of shape (B, N, D) **in NumPy
            space** (already detached / moved to CPU).

        Returns
        -------
        np.ndarray
            Rescaled tensor back in the original data domain.
        """
        if self.x_scaler is None:
            raise RuntimeError("x_scaler has not been fitted yet.")

        # reshape → inverse-transform → reshape back
        B, N, D = X_array.shape
        X_orig = self.x_scaler.inverse_transform(
            X_array.reshape(-1, D)
        ).reshape(B, N, D)

        # keep the degree feature numerically sane
        lower = 0.0 if self.deg_min_val is None else self.deg_min_val
        X_orig[..., self.important_feature_index] = np.clip(
            X_orig[..., self.important_feature_index],
            lower,
            self.deg_max_val,
        )
        return X_orig

    def _fit_scalers(self, y_array):
        """
        Fit ONLY the y-scaler. The x-scaler is already fitted on 
        the genuine rows before padding.
        """
        self.y_scaler = MinMaxScaler().fit(y_array)

    def _transform_data(self, X_array, y_array):
        """Transform both X and Y using their respective fitted scalers."""
        B, N, D_x = X_array.shape
        X_scaled_flat = self.x_scaler.transform(X_array.reshape(-1, D_x))
        X_scaled = X_scaled_flat.reshape(B, N, D_x)
        
        y_scaled = self.y_scaler.transform(y_array)
        return X_scaled, y_scaled

    def setup(
        self,
        node_encodings_list: List[np.ndarray],
        conditional_graph_encodings: Any,
        edge_pairs: Optional[List[Tuple[int, int, int]]] = None,
        edge_targets: Optional[np.ndarray] = None,
        node_mask: Optional[np.ndarray] = None
    ):
        # ------------------------------------------------------------
        # 1. Work out paddings *but* fit the scaler BEFORE padding so
        #    zeros do not skew min/max statistics.
        # ------------------------------------------------------------
        max_num_rows = max(x.shape[0] for x in node_encodings_list)
        self.number_of_rows_per_example = max_num_rows

        # ---- Fit on genuine rows only --------------------------------
        real_rows_concat = np.concatenate(node_encodings_list, axis=0)  # (ΣN, D)
        self.x_scaler = MinMaxScaler().fit(real_rows_concat)

        # ---- Now pad each graph --------------------------------------
        X_padded = []
        for x in node_encodings_list:
            if x.shape[0] < max_num_rows:
                pad = ((0, max_num_rows - x.shape[0]), (0, 0))
                x = np.pad(x, pad_width=pad, mode="constant", constant_values=0)
            X_padded.append(x)
        X_array = np.stack(X_padded, axis=0)  # (B, N, D)
        y_array = np.array(conditional_graph_encodings)
        
        # Fit y-scaler only (x-scaler already fitted on real rows)
        self._fit_scalers(y_array)

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
        # FIX 3a: Compute inverse-frequency class weights
        counts = np.bincount(raw_degrees.astype(int).ravel(),
                           minlength=int(raw_degrees.max()) + 1)
        class_weights = 1.0 / (counts + 1e-6)          # inverse-frequency
        class_weights = class_weights / class_weights.mean()

        self.deg_max_val = int(raw_degrees.max())
        
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
            sde_beta=self.sde_beta,               # Pass through to FM model
            lambda_cfm=self.lambda_cfm,            # Pass through CFM strength
            lambda_consistency=self.lambda_consistency,  # ← pass through
            class_weights=class_weights,               # NEW ARG
        )
        self.model.use_guidance = self.use_guidance  # <-- Set guidance flag

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
            callbacks=[MetricsLogger(
                x_scaler=self.x_scaler,
                y_scaler=self.y_scaler
            )],
            logger=True,
            enable_checkpointing=False,
            enable_progress_bar=False,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm="norm",
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
                "deg_loss": self.model.train_deg_loss,  # Changed display label only
                "exist": self.model.train_exist,
                **({"edge": self.model.train_edge_loss} if self.model.use_edge_supervision else {})
            },
            val_metrics = {
                "total": self.model.val_losses,
                "deg_loss": self.model.val_deg_loss,    # Changed display label only
                "exist": self.model.val_exist,
                **({"edge": self.model.val_edge_loss} if self.model.use_edge_supervision else {})
            },
            window=window,
            alpha=alpha
        )
    
    def predict(
        self,
        y,
        desired_class: Optional[Union[int, Sequence[int]]] = None
    ):
        """Generate samples conditioned on the given encodings.
        
        Args:
            y: Conditional input encodings 
            desired_class: Optional class index(es) for classifier guidance.
                         If None, uses model's own prediction.
        """
        self.model.eval()
        need_grad = self.model.use_guidance and self.model.guidance_classifier is not None

        def _run_generate():
            y_arr = np.array(y)
            y_scaled = self.y_scaler.transform(y_arr)
            # --- NEW: put the conditioning on the same device as the model ---
            device  = next(self.model.parameters()).device          # cpu / cuda:0 / …
            y_tensor = torch.tensor(y_scaled, dtype=torch.float32, device=device)
            return self.model.generate(
                y_tensor,
                total_steps=self.total_steps,
                desired_class=desired_class
            )

        with torch.set_grad_enabled(need_grad):
            generated = _run_generate()

        generated_np = generated.cpu().numpy()
        generated_orig = self._inverse_transform_input(generated_np)
        return [generated_orig[i] for i in range(generated_orig.shape[0])]

class MetricsLogger(pl.callbacks.Callback):
    """
    Tracks the Pearson correlation between
      Σ generated_node_features  ↔  conditioning vector
    in *original* feature space.
    """

    def __init__(self, *, x_scaler, y_scaler):
        super().__init__()
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        # ── running trajectories ──────────────────────────────────────────
        self.train_corr_over_epochs = []   # Σ(node_feats) ↔ cond   (train)
        self.val_corr_over_epochs   = []   # Σ(node_feats) ↔ cond   (val)
        self.current_epoch_correlations = None
    
    def compute_batch_correlations(
        self, node_sums_scaled: torch.Tensor, cond_scaled: torch.Tensor
    ) -> float:
        """Correlate masked-sum(node_features) with the conditioning vector."""
        node_sums_orig = self.x_scaler.inverse_transform(
            node_sums_scaled.detach().cpu().numpy()
        )
        cond_orig = self.y_scaler.inverse_transform(
            cond_scaled.detach().cpu().numpy()
        )

        correlations = []
        for i in range(node_sums_orig.shape[0]):
            x = node_sums_orig[i]
            y = cond_orig[i]
            
            # Handle potential feature dimension mismatch
            D = min(len(x), len(y))
            try:
                corr, _ = pearsonr(x[:D], y[:D])
                if not np.isnan(corr):
                    correlations.append(corr)
            except ValueError as e:
                print(f"Warning: pearsonr failed on batch {i}: {e}")
                continue
        
        return np.mean(correlations) if correlations else float("nan")

    # ---------------------------------------------------------------------
    #                      ─── TRAIN SPLIT ────────────────────────────────
    # ---------------------------------------------------------------------
    def on_train_epoch_start(self, *a, **k):
        self._train_correlations = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch,
                           batch_idx, dataloader_idx=0):
        """Compute correlation on every training batch using predicted features."""
        # Use model-predicted node features, not ground-truth
        node_pred = outputs["node_pred"]                          # (B,N,D)
        cond = batch[1]                                          # always 2nd field
        
        node_sums = node_pred.sum(dim=1)                         # Σ predicted feats
        c = self.compute_batch_correlations(node_sums, cond)
        self._train_correlations.append(c)

    def on_train_epoch_end(self, trainer, pl_module):
        # Get aggregated epoch metrics
        m = trainer.callback_metrics
        self._safe_append(pl_module.train_losses, self._get_metric(m, "train_total"))
        self._safe_append(pl_module.train_deg_loss, self._get_metric(m, "train_deg_loss"))
        self._safe_append(pl_module.train_exist, self._get_metric(m, "train_exist"))
        if pl_module.use_edge_supervision:
            self._safe_append(pl_module.train_edge_loss, self._get_metric(m, "train_edge_loss"))
            self._safe_append(pl_module.train_edge_acc, self._get_metric(m, "train_edge_acc"))

        # Keep existing correlation logging
        if self._train_correlations:
            mean_c = float(np.mean(self._train_correlations))
            self.train_corr_over_epochs.append(mean_c)
            pl_module.log("train_latent_correlation", mean_c,
                         on_epoch=True, prog_bar=True)

    # ---------------------------------------------------------------------
    #                      ─── VALIDATION SPLIT ───────────────────────────
    # ---------------------------------------------------------------------
    def on_validation_batch_end(
        self, 
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.current_epoch_correlations is None:
            self.current_epoch_correlations = []
            
        # Predicted node features from validation_step
        node_pred = outputs["node_pred"]             # (B, N, D)
        cond = batch[1]                             # second element regardless of supervision
        
        node_sums = node_pred.sum(dim=1)
        corr = self.compute_batch_correlations(node_sums, cond)
        self.current_epoch_correlations.append(corr)

    def on_validation_epoch_start(self, *args, **kwargs):
        """Reset correlation tracking for new epoch."""
        self.current_epoch_correlations = None

    def _get_metric(self, metrics: Dict[str, Any], name: str) -> float:
        """Safely extract a metric value from the callback_metrics dict."""
        for k in (name, f"{name}_epoch"):
            if k in metrics:
                val = metrics[k]
                if val is None:  # Guard against missing metrics
                    continue
                if torch.is_tensor(val):
                    val = val.detach().cpu()
                if not math.isnan(float(val)):
                    return float(val)
        return float("nan")

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log mean correlation for the epoch."""
        m = trainer.callback_metrics
        self._safe_append(pl_module.val_losses, self._get_metric(m, "val_total"))
        self._safe_append(pl_module.val_deg_loss, self._get_metric(m, "val_deg_loss"))
        self._safe_append(pl_module.val_exist, self._get_metric(m, "val_exist"))
        if pl_module.use_edge_supervision:
            self._safe_append(pl_module.val_edge_loss, self._get_metric(m, "val_edge_loss"))
            self._safe_append(pl_module.val_edge_acc, self._get_metric(m, "val_edge_acc"))
        
        if self.current_epoch_correlations:
            mean_corr = float(np.mean(self.current_epoch_correlations))
            self.val_corr_over_epochs.append(mean_corr)
            pl_module.log("val_latent_correlation", mean_corr,
                          on_epoch=True, prog_bar=True)

    def plot_correlations(self, save_path: Optional[str] = None):
        """Plot correlation trajectory after training."""
        if not (self.train_corr_over_epochs or self.val_corr_over_epochs):
            print("No correlation data available.")
            return
            
        plt.figure(figsize=(15, 8))
        if self.train_corr_over_epochs:
            plt.plot(self.train_corr_over_epochs,  label="Train")
        if self.val_corr_over_epochs:
            plt.plot(self.val_corr_over_epochs, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Correlation Σ(nodes) ↔ cond")
        plt.title("Latent-Condition Alignment")
        plt.grid(True)
        if plt.gca().has_data():
            plt.legend()
            if save_path:
                plt.savefig(save_path)
        plt.show()

    def _safe_append(self, lst: List[float], val: float) -> None:
        """Safely append non-None, non-NaN values to a list."""
        if val is not None and not math.isnan(val):
            lst.append(val)