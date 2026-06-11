import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from tqdm.auto import tqdm          # chooses the right flavour (widget in Jupyter, tty elsewhere)
from pytorch_lightning.callbacks import TQDMProgressBar
TQDMProgressBar.tqdm = staticmethod(tqdm)   # monkey-patch Lightning’s progress bar
import math

'''
### **Goal**

To build an end-to-end pipeline for **classifying dynamic 3D brain volumes over time**, optionally **pretraining the model to predict future volumes** as a form of self-supervision, and analyzing predictions using **saliency maps** and **embedding visualizations**.

---

### **Core Components**

#### 1. **3D Encoder (U-Net style)**

* Operates on 4D brain volumes (`T × Z × Y × X`) as sequences of 3D images.
* Uses convolutional layers with downsampling (via `encoder_stride`) and residual blocks.
* Extracts spatial features independently at each time point.

#### 2. **Temporal Reasoning Head (1D CNN)**

* After flattening each 3D volume, the model applies 1D convolutions over time.
* Learns to reason about **temporal evolution** in the brain signal.
* Number of layers, kernel sizes, stride, and dropout are all configurable.

#### 3. **Classification Head**

* A final MLP receives the aggregated (pooled) output from the temporal CNN.
* Predicts a discrete label (e.g. a brain state or condition class).
* Can **fine-tune the encoder** or leave it frozen depending on `fine_tune_encoder`.

---

### **Optional: Pretraining via Future Prediction**

* If `fit_with_pretrain=True`, the model is first trained to predict the 3D volume at `t + future_horizon` from the volume at `t`.
* This teaches the encoder to capture **temporal dynamics** even without labels.
* After this, the model is fine-tuned using labelled data.

---

### **Training and Evaluation**

* The orchestrator handles internal train/val splitting (`val_split`) and training logic.
* A classification report is generated after testing on a held-out test set.
* Embeddings are extracted and visualized using **t-SNE** to explore cluster structure in latent space.

---

### **Saliency Analysis**

* Uses **Integrated Gradients** to compute voxel-wise saliency.
* Helps interpret **which regions in the volume** contributed most to a given class prediction.
* Saliency is visualized per-class using a grid of slices with optional overlays.

---

### **What This Enables**

* Training a classifier with or without prior pretraining.
* Understanding model attention with voxel-level explanation.
* Visualizing and validating latent structure of learned representations.
* Supporting both scientific insight (via saliency) and performance validation (via metrics).

'''
class ResidualBlock3D(nn.Module):
    def __init__(self, channels, dropout=0.0):
        """
        3D residual block with two convolutional layers and a skip connection.

        Args:
            channels (int): Number of input/output channels.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass for the residual block.
        Args:
            x (Tensor): Input tensor of shape (B, C, D, H, W).
        Returns:
            Tensor: Output tensor of same shape as input.
        """
        return self.relu(x + self.block(x))


class Encoder3DWithDelta(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        base_channels:   int = 32,
        num_layers:      int = 2,
        stride:          int = 2,
        dropout:         float = 0.0,
    ):
        """
        3D encoder-decoder (U-Net style) for volumetric data.

        Args:
            input_channels (int): Number of input channels.
            base_channels (int): Number of base channels for convolutions.
            num_layers (int): Number of residual layers in encoder/decoder.
            stride (int): Stride for MaxPool3d/Upsample.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.dropout = dropout
        self.base_channels = base_channels
        encoder_layers = [
            nn.Conv3d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        for _ in range(num_layers):
            encoder_layers.append(ResidualBlock3D(base_channels, dropout=dropout))
        encoder_layers.append(nn.MaxPool3d(stride))
        for _ in range(num_layers):
            encoder_layers.append(ResidualBlock3D(base_channels, dropout=dropout))
        encoder_layers.append(nn.Identity())
        self.encoder = nn.Sequential(*encoder_layers)

        # ─── temporal embedding removed ───
        self.latent_dim = None        # still needed by the Orchestrator

        self.up_blocks      = nn.ModuleList()
        self._decoder_built = False
        self.input_channels = input_channels

    def forward(self, x, delta_t=None):     # delta_t kept for API compatibility
        """
        Forward pass for the encoder-decoder network.

        Args:
            x (Tensor): Input tensor of shape (B, C, D, H, W).
            delta_t: Unused, kept for API compatibility.

        Returns:
            Tensor: Output tensor of same shape as input.
        """
        # ---------- encode & collect ALL ResidualBlock outputs ----------
        skips, z = [], x
        for layer in self.encoder:
            z = layer(z)
            if isinstance(layer, ResidualBlock3D):
                skips.append(z)          # keep every ResBlock feature map

        # Dynamic pooling → preserves aspect ratio while shrinking ~4×
        D,H,W = z.shape[-3:]
        kD,kH,kW = max(1, math.ceil(D/4)), max(1, math.ceil(H/4)), max(1, math.ceil(W/4))
        z = F.avg_pool3d(z, kernel_size=(kD,kH,kW), ceil_mode=True)

        # Keep track of the flattened size so downstream Temporal-CNN can be built
        if self.latent_dim is None:
            self.latent_dim = z.view(z.size(0), -1).shape[1]

        # ---------- lazily build decoder modules once ----------
        if not self._decoder_built:
            device = z.device
            in_ch = z.shape[1]
            self.up_blocks = nn.ModuleList()
            for skip in reversed(skips):
                blk = nn.Sequential(
                    nn.Conv3d(in_ch + skip.shape[1],
                              self.base_channels, kernel_size=1),
                    ResidualBlock3D(self.base_channels, dropout=self.dropout),
                ).to(device)
                self.up_blocks.append(blk)
                in_ch = self.base_channels

            self.out_conv = nn.Conv3d(
                in_ch, self.input_channels,
                kernel_size=3, padding=1, device=device
            )
            self.add_module("out_conv", self.out_conv)
            self._decoder_built = True

        # ---------- decode: resize to each skip tensor, then refine ----------
        for blk, skip in zip(self.up_blocks, reversed(skips)):
            z = F.interpolate(
                z, size=skip.shape[-3:], mode="trilinear", align_corners=False
            )
            z = torch.cat([z, skip], dim=1)
            z = blk(z)

        # final resize (if needed) to exactly the input shape
        if z.shape[-3:] != x.shape[-3:]:
            z = F.interpolate(z, size=x.shape[-3:],
                              mode="trilinear", align_corners=False)
        return self.out_conv(z)


class Temporal1DCNN(nn.Module):
    def __init__(
        self,
        latent_dim,
        num_classes,
        num_layers=2,
        num_kernels=None,
        stride=1,
        dropout=0.0
    ):
        """
        1D temporal CNN for sequence classification.

        Args:
            latent_dim (int): Input feature dimension.
            num_classes (int): Number of output classes.
            num_layers (int): Number of Conv1d layers.
            num_kernels (int, list, or None): Number of kernels per Conv1d layer.
            stride (int): Stride for Conv1d layers.
            dropout (float): Dropout probability.
        """
        super().__init__()
        if num_kernels is None:
            num_kernels = [latent_dim] + [latent_dim // 2] * (num_layers - 1)
        elif isinstance(num_kernels, int):
            num_kernels = [num_kernels] * num_layers
        elif isinstance(num_kernels, (list, tuple)):
            assert len(num_kernels) == num_layers

        conv_layers = []
        in_channels = latent_dim
        for i in range(num_layers):
            conv_layers.append(nn.Conv1d(in_channels, num_kernels[i], kernel_size=3, padding=1, stride=stride))
            conv_layers.append(nn.ReLU())
            if dropout > 0:
                conv_layers.append(nn.Dropout(dropout))
            in_channels = num_kernels[i]
        conv_layers.append(nn.AdaptiveAvgPool1d(1))
        self.net = nn.Sequential(*conv_layers)

        self.classifier = nn.Sequential(
            nn.Linear(num_kernels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Forward pass for the temporal CNN.

        Args:
            x (Tensor): Input tensor of shape (B, latent_dim, T).

        Returns:
            Tensor: Output logits of shape (B, num_classes).
        """
        x = self.net(x).squeeze(-1)
        return self.classifier(x)


class AutoEncoderLightning(pl.LightningModule):
    def __init__(self, encoder3d, lr=1e-3):
        """
        PyTorch Lightning module for auto-encoding 3D volumes.

        Args:
            encoder3d (nn.Module): 3D encoder-decoder model.
            lr (float): Learning rate for optimizer.
        """ 
        super().__init__()
        self.encoder3d = encoder3d
        self.lr = lr
    def forward(self, x, delta_t=None):
        """
        Forward pass for the autoencoder.

        Args:
            x (Tensor): Input tensor of shape (B, C, D, H, W).
            delta_t: Unused, kept for API compatibility.

        Returns:
            Tensor: Reconstructed tensor of same shape as input.
        """
        return self.encoder3d(x)
    def training_step(self, batch, _):
        """
        Training step for autoencoder.

        Args:
            batch (tuple): Tuple containing input tensor.

        Returns:
            Tensor: MSE loss value.
        """
        x, = batch
        B, C, D, H, W = x.shape
        recon = self.encoder3d(x)
        loss = F.mse_loss(recon, x)
        self.log("recon_loss", loss, prog_bar=True)
        return loss
    def configure_optimizers(self):
        """
        Configure optimizer for autoencoder.

        Returns:
            torch.optim.Optimizer: Adam optimizer.
        """
        return torch.optim.Adam(self.encoder3d.parameters(), lr=self.lr)


class BrainScanLightningModule(pl.LightningModule):
    def __init__(self, encoder3d, temporal_model, num_classes,
                 fine_tune_encoder=False, lr=1e-4, weight_decay=0.0,
                 verbose=False):
        """
        PyTorch Lightning module for brain scan sequence classification.

        Args:
            encoder3d (nn.Module): 3D encoder model.
            temporal_model (nn.Module): Temporal model for sequence.
            num_classes (int): Number of classes.
            fine_tune_encoder (bool): Whether to fine-tune encoder.
            lr (float): Learning rate.
            weight_decay (float): L2 regularization coefficient for Adam.
            verbose (bool): Print epoch metrics if True.
        """
        super().__init__()
        self.encoder3d = encoder3d
        self.temporal_model = temporal_model
        self.num_classes = num_classes
        self.fine_tune_encoder = fine_tune_encoder
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_losses = []
        self.val_losses = []
        self._verbose_print = verbose

    def forward(self, x_seq):
        """
        Forward pass for the full model.

        Args:
            x_seq (Tensor): Input sequence tensor (B, T, C, D, H, W).

        Returns:
            Tensor: Logits of shape (B, num_classes).
        """
        B, T = x_seq.shape[:2]
        z_seq = []
        for t in range(T):
            x_t     = x_seq[:, t]                       # (B, C, D, H, W)
            z_enc   = self.encoder3d.encoder(x_t)       # (B, C_enc, D,H,W)
            D, H, W = z_enc.shape[-3:]
            kD, kH, kW = max(1, math.ceil(D/4)), \
                         max(1, math.ceil(H/4)), \
                         max(1, math.ceil(W/4))
            z_pool  = F.avg_pool3d(z_enc, kernel_size=(kD, kH, kW), ceil_mode=True)
            latent  = z_pool.view(B, -1)
            z_seq.append(latent)
        z_seq = torch.stack(z_seq, dim=1).transpose(1, 2)
        return self.temporal_model(z_seq)

    def training_step(self, batch, batch_idx):
        """
        Training step for one batch.

        Args:
            batch (tuple): Tuple of (inputs, targets).
            batch_idx (int): Batch index.

        Returns:
            Tensor: Loss value.
        """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.train_losses.append(loss.item())
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for one batch.

        Args:
            batch (tuple): Tuple of (inputs, targets).
            batch_idx (int): Batch index.

        Returns:
            Tensor: Loss value.
        """
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.val_losses.append(loss.item())
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        """
        Optionally print epoch metrics if verbose.
        """
        if not self._verbose_print:
            return
        # aggregated metrics are in callback_metrics
        cm = self.trainer.callback_metrics
        tl = cm.get("train_loss_epoch") or cm.get("train_loss")
        vl = cm.get("val_loss")
        if tl is not None and vl is not None:
            self.print(f"Epoch {self.current_epoch:03d} "
                       f"| train_loss={tl:.5f} | val_loss={vl:.5f}")

    def configure_optimizers(self):
        """
        Configure optimizers for training.

        Returns:
            torch.optim.Optimizer: Adam optimizer.
        """
        params = list(self.temporal_model.parameters())
        if self.fine_tune_encoder:
            params += list(self.encoder3d.parameters())
        return torch.optim.Adam(params,
                                lr=self.lr,
                                weight_decay=self.weight_decay)


class BrainScanOrchestrator:
    def __init__(
        self,
        base_channels=32,
        encoder_num_layers=2,
        encoder_stride=2,
        encoder_dropout=0.0,
        temporal_num_layers=2,
        temporal_num_kernels=None,
        temporal_stride=1,
        temporal_dropout=0.0,
        fine_tune_encoder=False,
        classify_epochs=10,
        batch_size=8,
        val_split=0.2,
        verbose=False,
        future_horizon: int = 1,
        fit_with_pretrain=False,     
        pretrain_epochs=10,
        pretrain_lr=1e-3,
        finetune_lr=1e-4,
        weight_decay=1e-4,
    ):
        """
        Orchestrates training, prediction, and interpretation for brain scan models.

        Args:
            encoder_num_layers (int): Number of residual layers in 3D CNN encoder/decoder.
            encoder_stride (int): Stride for MaxPool3d/Upsample in 3D CNN.
            encoder_dropout (float): Dropout for 3D CNN.
            temporal_num_layers (int): Number of Conv1d layers in 1D CNN.
            temporal_num_kernels (int, list, or None): Number of kernels per Conv1d layer.
            temporal_stride (int): Stride for Conv1d layers in 1D CNN.
            temporal_dropout (float): Dropout for 1D CNN.
            fine_tune_encoder (bool): Whether to fine-tune encoder.
            classify_epochs (int): Number of training epochs.
            batch_size (int): Batch size.
            val_split (float): Validation split ratio.
            verbose (bool): Whether to show progress bar during training.
            future_horizon (int): How far ahead to predict during pre-training.
            finetune_lr (float): Learning rate for fine-tune phase.
            weight_decay (float): L2 weight-decay coefficient used by Adam.
        """
        self.num_classes = None
        self.input_channels = None
        self.encoder3d = None
        self.temporal_model = None

        self.base_channels = base_channels
        self.encoder_num_layers = encoder_num_layers
        self.encoder_stride = encoder_stride
        self.encoder_dropout = encoder_dropout
        self.future_horizon = future_horizon
        self.fine_tune_encoder = fine_tune_encoder

        self._tmp_num_layers   = temporal_num_layers
        self._tmp_num_kernels  = temporal_num_kernels
        self._tmp_stride       = temporal_stride
        self._tmp_dropout      = temporal_dropout

        self.classify_epochs = classify_epochs
        self.batch_size = batch_size
        self.val_split = val_split
        self.verbose = verbose
        self.fit_with_pretrain = fit_with_pretrain
        self.pretrain_epochs   = pretrain_epochs
        self.pretrain_lr       = pretrain_lr
        self.finetune_lr       = finetune_lr
        self.weight_decay      = weight_decay
        # no per-frame δ-embedding – horizon only used during pre-train
        self.is_fitted_ = False

    # ────────────────────────────────────────────────────────────────
    #  Utility: move data to the same device as the trained model
    # ────────────────────────────────────────────────────────────────
    def _to_device(self, array, dtype=torch.float32):
        """
        Move a NumPy array or Torch tensor to the same device as the trained model.

        Args:
            array (np.ndarray or torch.Tensor): Input data.
            dtype (torch.dtype): Desired tensor dtype.

        Returns:
            torch.Tensor: Tensor on the correct device.
        """
        device = torch.device("cpu")
        if hasattr(self, "model") and isinstance(self.model, pl.LightningModule):
            device = next(self.model.parameters()).device
        return torch.as_tensor(array, dtype=dtype, device=device)

    # ────────────────────────────────────────────────────────────────────────────────
    #  NEW 1️⃣ :  self-supervised module that predicts slice t+δ from slice t
    # ────────────────────────────────────────────────────────────────────────────────

    class NextSliceLightning(pl.LightningModule):
        """
        Self-supervised module to train Encoder3DWithDelta to predict a fixed future volume (future_horizon steps ahead).
        """
        def __init__(self, encoder3d: nn.Module, lr: float = 1e-3,
                     weight_decay: float = 0.0, verbose=False):
            super().__init__()
            self.encoder3d = encoder3d
            self.lr = lr
            self.weight_decay = weight_decay
            self.loss_fn = nn.MSELoss()
            self.train_losses, self.val_losses = [], []
            self._verbose_print = verbose

        def forward(self, x):
            return self.encoder3d(x)

        def training_step(self, batch, _):
            x_now, x_future = batch
            pred = self(x_now)
            loss = self.loss_fn(pred, x_future)
            self.train_losses.append(loss.item())
            self.log("next_loss", loss, prog_bar=True, on_epoch=True)
            return loss

        def validation_step(self, batch, _):
            x_now, x_future = batch
            pred = self(x_now)
            loss = self.loss_fn(pred, x_future)
            self.val_losses.append(loss.item())
            self.log("val_next_loss", loss, prog_bar=True, on_epoch=True)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.encoder3d.parameters(),
                                    lr=self.lr,
                                    weight_decay=self.weight_decay)

        def on_validation_epoch_end(self):
            if not self._verbose_print:
                return
            cm = self.trainer.callback_metrics
            tl = cm.get("next_loss_epoch") or cm.get("next_loss")
            vl = cm.get("val_next_loss")
            if tl is not None and vl is not None:
                self.print(f"Epoch {self.current_epoch:03d} "
                           f"| pretrain_train={tl:.5f} | pretrain_val={vl:.5f}")

    # ────────────────────────────────────────────────────────────────
    #  NEW 2️⃣ :  pre-train on predict-next-slice for all delta_offsets
    # ────────────────────────────────────────────────────────────────
    def _pretrain(self, X: np.ndarray):
        """
        Pre-train the encoder/decoder to predict a fixed future volume (future_horizon steps ahead).
        Uses 80/20 train/validation split and plots loss curves if verbose.

        Args:
            X (np.ndarray): Input data of shape (N, T, C, D, H, W).
        """
        N, T = X.shape[:2]
        d = int(self.future_horizon)
        if d <= 0 or d >= T:
            return
        cur = X[:, :T-d].reshape(-1, *X.shape[2:])
        fut = X[:, d:].reshape(-1, *X.shape[2:])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_now    = torch.tensor(cur, dtype=torch.float32, device=device)
        x_future = torch.tensor(fut, dtype=torch.float32, device=device)

        dataset = TensorDataset(x_now, x_future)
        val_len = int(0.2 * len(dataset))
        train_len = len(dataset) - val_len
        train_ds, val_ds = random_split(dataset, [train_len, val_len])
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)
        # Ensure encoder3d is built before pretraining
        if self.encoder3d is None:
            self.encoder3d = Encoder3DWithDelta(
                input_channels=self.input_channels,
                base_channels=self.base_channels,
                num_layers=self.encoder_num_layers,
                stride=self.encoder_stride,
                dropout=self.encoder_dropout,
            )
        module = self.NextSliceLightning(
            encoder3d=self.encoder3d,
            lr=self.pretrain_lr,
            weight_decay=self.weight_decay,
            verbose=self.verbose
        )

        # ─── FIX: ensure encoder3d matches trainer device ───
        self.encoder3d.to(device)

        # same real-time bar you use in fine-tuning
        pb = TQDMProgressBar(refresh_rate=20) if self.verbose else None

        pl.Trainer(
            max_epochs=self.pretrain_epochs,
            enable_progress_bar=self.verbose,
            callbacks=[pb] if pb else None,
            log_every_n_steps=1,
        ).fit(module, train_loader, val_loader)
        if self.verbose:
            self._plot_pretrain_loss(module.train_losses, module.val_losses)
        if not self.fine_tune_encoder:
            for p in self.encoder3d.parameters():
                p.requires_grad = False

    @staticmethod
    def _plot_pretrain_loss(train_l, val_l):
        """
        Plot pre-training loss curves for train and validation sets.

        Args:
            train_l (list): Training loss values.
            val_l (list): Validation loss values.
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        x_train = np.arange(len(train_l))
        x_val = np.linspace(0, len(train_l) - 1, len(val_l))
        plt.plot(x_train, train_l, label="Pre-train Train")
        plt.plot(x_val, val_l, label="Pre-train Val")
        plt.yscale("log")
        plt.xlabel("Batch")
        plt.ylabel("MSE")
        plt.title("Predict-Next-Slice Pre-training Loss (log-scale)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def fit(self, X, y):
        """
        Train the model on provided data.

        Args:
            X (np.ndarray or Tensor): Input data of shape (N, T, C, D, H, W).
            y (np.ndarray or Tensor): Target labels of shape (N,).
        """
        # -------- infer num_classes and input_channels --------
        if self.num_classes is None:
            self.num_classes = int(np.max(y)) + 1
        if self.input_channels is None:
            self.input_channels = X.shape[2]  # (N, T, C, D, H, W) → C

        # -------- optional pre-training --------
        if self.fit_with_pretrain:
            self._pretrain(X)      # uses same encoder3d instance

        X_torch = torch.tensor(X, dtype=torch.float32)
        y_torch = torch.tensor(y, dtype=torch.long)

        dataset = TensorDataset(X_torch, y_torch)
        val_len = int(len(dataset) * self.val_split)
        train_len = len(dataset) - val_len
        train_data, val_data = random_split(dataset, [train_len, val_len])
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size)

        # ── build encoder and temporal model if not already built ──
        if self.encoder3d is None:
            self.encoder3d = Encoder3DWithDelta(
                input_channels=self.input_channels,
                base_channels=self.base_channels,
                num_layers=self.encoder_num_layers,
                stride=self.encoder_stride,
                dropout=self.encoder_dropout,
            )
        if self.temporal_model is None:
            sample_x = X_torch[0, 0].unsqueeze(0)                # (1, C, D, H, W)
            _ = self.encoder3d(sample_x)
            latent_dim = self.encoder3d.latent_dim
            self.temporal_model = Temporal1DCNN(
                latent_dim      = latent_dim,
                num_classes     = self.num_classes,
                num_layers      = self._tmp_num_layers,
                num_kernels     = self._tmp_num_kernels,
                stride          = self._tmp_stride,
                dropout         = self._tmp_dropout,
            )

        model = BrainScanLightningModule(
            self.encoder3d,
            self.temporal_model,
            self.num_classes,
            self.fine_tune_encoder,
            lr=self.finetune_lr,
            weight_decay=self.weight_decay,
            verbose=self.verbose,
        )
        # show a real-time progress bar inside notebooks
        pb = TQDMProgressBar(refresh_rate=20) if self.verbose else None
        trainer = pl.Trainer(
            max_epochs=self.classify_epochs,
            enable_progress_bar=self.verbose,
            callbacks=[pb] if pb else None,
            log_every_n_steps=1  # so the bar updates every batch
        )
        trainer.fit(model, train_loader, val_loader)
        # Grab the instance that actually ran inside the Trainer
        self.model = trainer.lightning_module
        self.is_fitted_ = True
        if self.verbose:
            self.plot_loss()

    def predict(self, X):
        """
        Predict class labels for input data.

        Args:
            X (np.ndarray or Tensor): Input data of shape (N, T, C, D, H, W).

        Returns:
            np.ndarray: Predicted class indices.
        """
        assert self.is_fitted_, "Model not trained yet."
        self.model.eval()
        X_torch = self._to_device(X)
        with torch.no_grad():
            logits = self.model(X_torch)
            return torch.argmax(logits, dim=1).cpu().numpy()

    def predict_proba(self, X):
        """
        Predict class probabilities for input data.

        Args:
            X (np.ndarray or Tensor): Input data of shape (N, T, C, D, H, W).

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        assert self.is_fitted_, "Model not trained yet."
        self.model.eval()
        X_torch = self._to_device(X)
        with torch.no_grad():
            logits = self.model(X_torch)
            probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()

    # ------------------------------------------------------------------
    # NEW: feature-extraction helper
    # ------------------------------------------------------------------
    def transform(self, X, return_tensor=False):
        """
        Extract penultimate embeddings for each 5-D video sample.

        Args:
            X (np.ndarray or Tensor): Input data of shape (N, T, C, D, H, W).
            return_tensor (bool): If True, return a torch.Tensor on CPU; else return a NumPy array.

        Returns:
            embeddings: (N, feature_dim) torch.Tensor or np.ndarray
        """
        assert self.is_fitted_, "Model not trained yet."
        self.model.eval()

        X_torch = self._to_device(X)

        with torch.no_grad():
            B, T = X_torch.shape[:2]
            z_seq = []
            for t in range(T):
                x_t     = X_torch[:, t]                       # (B, C, D, H, W)
                z_enc   = self.encoder3d.encoder(x_t)       # (B, C, D,H,W)
                D, H, W = z_enc.shape[-3:]
                kD, kH, kW = max(1, math.ceil(D/4)), \
                             max(1, math.ceil(H/4)), \
                             max(1, math.ceil(W/4))
                z_pool  = F.avg_pool3d(z_enc, kernel_size=(kD, kH, kW), ceil_mode=True)
                latent  = z_pool.view(B, -1)
                z_seq.append(latent)

            z_seq = torch.stack(z_seq, dim=1).transpose(1, 2)  # (B, latent_dim, T)

            penultimate = self.temporal_model.net(z_seq).squeeze(-1)  # (B, feat)

        if return_tensor:
            return penultimate.cpu()
        else:
            return penultimate.cpu().numpy()

    def saliency(self, input_seq, target_class=None, normalize=True):
        """
        Compute saliency map using Integrated Gradients.

        Args:
            input_seq (Tensor): Input sequence of shape (1, T, C, D, H, W).
            target_class (int, optional): Target class index. If None, uses predicted class.
            normalize (bool): Whether to normalize the saliency map.

        Returns:
            Tensor: Saliency map of shape (T, C, D, H, W).
        """
        self.model.eval()
        input_seq = input_seq.clone().detach().requires_grad_(True)
        B, T, C, Z, Y, X = input_seq.shape

        def forward_fn(x):
            # x: (B_ig, T, C, D, H, W)
            # Iterate over time steps and extract features
            B_ig, T_, C_, D_, H_, W_ = x.shape
            z_seq = []
            for t in range(T_):
                x_t   = x[:, t]                                # (B_ig, C, D, H, W)
                z_enc = self.model.encoder3d.encoder(x_t)      # (B_ig,C,D,H,W)
                D, H, W = z_enc.shape[-3:]
                kD, kH, kW = max(1, math.ceil(D/4)), \
                             max(1, math.ceil(H/4)), \
                             max(1, math.ceil(W/4))
                z_pool = F.avg_pool3d(
                    z_enc, kernel_size=(kD, kH, kW), ceil_mode=True
                )
                latent = z_pool.view(B_ig, -1)
                z_seq.append(latent)
            z_stack = torch.stack(z_seq, dim=1).transpose(1, 2)
            return self.model.temporal_model(z_stack)

        with torch.no_grad():
            preds = forward_fn(input_seq)
            if target_class is None:
                target_class = preds.argmax(dim=1).item()

        ig = IntegratedGradients(forward_fn)
        baseline = torch.zeros_like(input_seq)
        attributions, _ = ig.attribute(inputs=input_seq,
                                       baselines=baseline,
                                       target=target_class,
                                       return_convergence_delta=True)

        saliency = (attributions * input_seq).squeeze(0)
        if normalize:
            saliency -= saliency.min()
            saliency /= saliency.max() + 1e-8
        return saliency.cpu()

    def visualize_saliency_grid(
        self,
        saliency,
        input_seq=None,
        max_rows=8,
        max_cols=10,
        vmax=None,
        overlay=False,
    ):
        """
        Visualize a grid of saliency maps with Z-slices as rows and time points as columns.

        Args:
            saliency (np.ndarray or torch.Tensor): Saliency map of shape (T, C, D, H, W).
            input_seq (np.ndarray or torch.Tensor or None): Input sequence for overlay, same shape as saliency except for batch.
            max_rows (int): Maximum number of Z slices (rows) to display.
            max_cols (int): Maximum number of time points (columns) to display.
            vmax (float or None): Maximum value for color scaling. If None, determined automatically.
            overlay (bool): If True, overlay saliency on input image.
        """
        # --- ensure NumPy ---
        if isinstance(saliency, torch.Tensor):
            saliency = saliency.detach().cpu().numpy()
        if input_seq is not None and isinstance(input_seq, torch.Tensor):
            input_seq = input_seq.detach().cpu().numpy()

        T, C, Z, Y, X = saliency.shape
        if input_seq is not None:
            input_seq = input_seq[0]                       # drop batch dim

        # choose slices & time points
        z_idx = np.linspace(0, Z - 1, num=min(Z, max_rows), dtype=int)
        t_idx = np.linspace(0, T - 1, num=min(T, max_cols), dtype=int)

        # --- Compute global value range for consistent color scale ---
        saliency_min = np.min(saliency)
        if vmax is not None:
            saliency_max = vmax
        else:
            # Optionally use percentile for robustness:
            # saliency_max = np.percentile(saliency, 99.5)
            saliency_max = np.max(saliency)

        fig, axes = plt.subplots(
            len(z_idx), len(t_idx),
            figsize=(3 * len(t_idx), 3 * len(z_idx))
        )
        axes_flat = np.array(axes).reshape(-1)   # do this once

        for i, zi in enumerate(z_idx):
            for j, tj in enumerate(t_idx):
                ax = axes_flat[i * len(t_idx) + j]
                s_slice = saliency[tj, 0, zi]              # (H, W)

                if overlay and input_seq is not None:
                    img_slice = input_seq[tj, 0, zi]
                    ax.imshow(img_slice, cmap="gray", interpolation="none")
                    ax.imshow(s_slice, cmap="hot", alpha=0.5,
                              interpolation="none", vmin=saliency_min, vmax=saliency_max)
                else:
                    ax.imshow(s_slice, cmap="hot", interpolation="none",
                              vmin=saliency_min, vmax=saliency_max)

                if i == 0:
                    ax.set_title(f"t={tj}")
                if j == 0:
                    ax.set_ylabel(f"z={zi}")
                ax.axis("off")

        plt.tight_layout()
        plt.suptitle("Saliency (rows: Z, cols: time)", y=1.02, fontsize=16)
        plt.show()

    def plot_loss(self):
        """
        Plot training and validation loss curves using matplotlib.

        Raises:
            RuntimeError: If model is not trained.
        """
        if not (hasattr(self, "model") and hasattr(self.model, "train_losses")
                and hasattr(self.model, "val_losses")):
            raise RuntimeError("Model must be trained before plotting loss.")
        plt.figure(figsize=(8, 5))
        train_x = np.arange(len(self.model.train_losses))
        val_x   = np.linspace(0, len(train_x) - 1, len(self.model.val_losses))
        plt.plot(train_x, self.model.train_losses, label="Train Loss")
        plt.plot(val_x,   self.model.val_losses,   label="Validation Loss")
        plt.yscale("log")
        plt.xlabel("Batch")
        plt.ylabel("Loss (log)")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.show()