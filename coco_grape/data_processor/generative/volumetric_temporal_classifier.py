# volumetric_temporal_model.py  ── UPDATED (configurable conv schedule)
from __future__ import annotations
from typing import Optional, List, Union, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import torch.nn.functional as F


# ───────────────────────────── Encoders ──────────────────────────────
KernelStride = Tuple[Tuple[int, int, int], Tuple[int, int, int]]  # (kernel, stride)


class Spatial3DEncoder(nn.Module):
    """
    3D convolutional encoder applied independently to each time slice.

    Args:
        in_channels (int): Number of input channels.
        base_channels (int): Number of filters in the first block; doubles with each block.
        depth (int): Number of convolutional blocks.
        conv_schedule (dict[int, ((kz, ky, kx), (sz, sy, sx), (dz, dy, dx))] or None): Optional per-block kernel/stride/dilation schedule.
            Each entry: (kernel, stride, dilation). If stride/dilation omitted, defaults to (1,1,1).
        drop_prob_spatial (float): Dropout probability after each block.
    """
    def __init__(self,
                 in_channels: int = 1,
                 base_channels: int = 8,
                 depth: int = 4,
                 conv_schedule: Dict[int, KernelStride] | None = None,
                 drop_prob_spatial: float = 0.0):
        super().__init__()
        conv_schedule = conv_schedule or {}

        layers: list[nn.Module] = []
        cin = in_channels
        for d in range(depth):
            cout = base_channels * 2 ** d

            # Defaults
            k = (3, 3, 3)
            s = (1, 1, 1)
            dil = (1, 1, 1)

            # Unpack schedule with backward compatibility
            if d in conv_schedule:
                values = conv_schedule[d]
                if len(values) == 3:
                    k, s, dil = values
                elif len(values) == 2:
                    k, s = values
                else:
                    k = values[0]

            padding = tuple(((k_i - 1) * dil_i) // 2 for k_i, dil_i in zip(k, dil))

            layers += [
                nn.Conv3d(cin, cout, kernel_size=k, stride=s,
                          dilation=dil, padding=padding),
                nn.GroupNorm(num_groups=4, num_channels=cout),
                nn.ReLU(inplace=True),
            ]
            if drop_prob_spatial > 0:
                layers.append(nn.Dropout3d(p=drop_prob_spatial))
            cin = cout
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, z, y, xdim = x.shape
        x = self.encoder(x.view(b * t, c, z, y, xdim))
        _, cout, z_out, y_out, x_out = x.shape
        return x.view(b, t, cout, z_out, y_out, x_out)


class Temporal1DEncoder(nn.Module):
    """
    1D convolutional encoder for temporal feature extraction.

    Args:
        input_dim (int): Input feature dimension per time step.
        hidden_dim (int): Output channels for each temporal conv layer.
        n_layers (int): Number of temporal conv layers.
        kernel_size (int): Temporal kernel size.
        stride_temporal (int): Stride for all temporal layers.
        dilation_temporal (int): Dilation for all temporal layers.
        pool (str or None): Pooling mode ("mean", "cls", or None).
        drop_prob_temporal (float): Dropout probability after each conv block.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 n_layers: int,
                 kernel_size: int = 3,
                 stride_temporal: int = 1,
                 dilation_temporal: int = 1,
                 pool: Optional[str] = "mean",
                 drop_prob_temporal: float = 0.0):
        super().__init__()
        layers = []
        cin = input_dim
        k_size  = kernel_size
        d_t     = dilation_temporal
        padding = (k_size * d_t) // 2 if stride_temporal == 1 else 0
        for _ in range(n_layers):
            layers += [
                nn.Conv1d(cin, hidden_dim,
                          kernel_size=k_size,
                          padding=padding,
                          stride=stride_temporal,
                          dilation=d_t),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
            if drop_prob_temporal > 0:
                layers.append(nn.Dropout(p=drop_prob_temporal))
            cin = hidden_dim
        self.temporal = nn.Sequential(*layers)
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, f = x.shape
        x = self.temporal(x.permute(0, 2, 1))        # (B, H, T)
        if self.pool == "mean":
            return x.mean(dim=-1)
        if self.pool == "cls":
            return x[..., 0]
        return x.permute(0, 2, 1)


class ClassificationHead(nn.Module):
    """
    Linear classification head.

    Args:
        in_dim (int): Input feature dimension.
        num_classes (int): Number of output classes.
    """
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):                            # (B, H) or (B, T, H)
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.fc(x)


# Input-level dropout module
class InputDropout(nn.Module):
    """
    Dropout applied to the input volume.

    Args:
        p (float): Dropout probability.
    """
    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        return F.dropout(x, p=self.p, training=True)


class VolumetricTemporalBackbone(nn.Module):
    """
    Combines input dropout, spatial 3D encoder, and temporal 1D encoder.

    Args:
        spatial (Spatial3DEncoder): Spatial encoder module.
        temporal (Temporal1DEncoder): Temporal encoder module.
        input_dropout (float): Dropout probability for input.
    """
    def __init__(self, spatial: Spatial3DEncoder, temporal: Temporal1DEncoder,
                 input_dropout: float = 0.0):  # ← ADD this
        super().__init__()
        self.input_dropout = InputDropout(p=input_dropout)  # ← ADD this
        self.spatial, self.temporal = spatial, temporal

    def forward(self, x):
        x = self.input_dropout(x)           # ← ADD this line
        s = self.spatial(x)                 # (B, T, C, H, W, D)
        b, t, c, h, w, d = s.shape
        s = s.reshape(b, t, c * h * w * d)  # (B, T, F)
        return self.temporal(s)


# ───────────────────── LightningModule ────────────────────────────────
class VolumetricLitModule(pl.LightningModule):
    """
    PyTorch Lightning module for volumetric-temporal classification.

    Args:
        backbone (VolumetricTemporalBackbone): Feature extractor.
        head (ClassificationHead): Classification head.
        lr (float): Learning rate.
        train_hist (list[float] or None): Optional training loss history.
        val_hist (list[float] or None): Optional validation loss history.
        verbose (bool): Verbosity flag.
    """
    def __init__(self, backbone: VolumetricTemporalBackbone,
                 head: ClassificationHead,
                 lr: float = 1e-3,
                 train_hist: Optional[List[float]] = None,    # NEW
                 val_hist:   Optional[List[float]] = None,    # NEW
                 verbose: bool = False):                      # NEW
        super().__init__()
        self.backbone, self.head, self.lr = backbone, head, lr
        self._train_hist = train_hist                         # NEW
        self._val_hist   = val_hist                           # NEW
        self._verbose    = verbose                            # NEW
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x): return self.head(self.backbone(x))

    def training_step(self, batch, _):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_end(self):                             # NEW
        if self._train_hist is not None:
            tl = self.trainer.callback_metrics.get("train_loss")
            if tl is not None:
                self._train_hist.append(tl.item())
        if self._verbose and (tl := self.trainer.callback_metrics.get("train_loss")):
            print(f"Epoch {self.current_epoch:3d}  train_loss={tl:.4f}")

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        self.log_dict(
            {"val_loss": self.criterion(logits, y),
             "val_acc": (logits.argmax(1) == y).float().mean()},
            prog_bar=True)

    def on_validation_epoch_end(self):                       # NEW
        if self._val_hist is not None:
            vl = self.trainer.callback_metrics.get("val_loss")
            if vl is not None:
                self._val_hist.append(vl.item())
        if self._verbose and (vl := self.trainer.callback_metrics.get("val_loss")):
            print(f"Epoch {self.current_epoch:3d}    val_loss={vl:.4f}")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# ───────────────────── Dataset & DataModule ───────────────────────────
class VolumetricDataset(Dataset):
    """
    Dataset for volumetric time series data.

    Args:
        volumes (array-like): Input volumes (N, T, Z, Y, X).
        labels (array-like): Class labels.
        transform (callable or None): Optional transform applied to each sample.
    """
    def __init__(self, volumes, labels, transform=None):
        self.volumes = torch.as_tensor(volumes, dtype=torch.float32)
        self.labels  = torch.as_tensor(labels,  dtype=torch.long)
        self.transform = transform

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        x = self.volumes[idx]                        # (T, Z, Y, X)
        if self.transform: x = self.transform(x)
        return x.unsqueeze(1), self.labels[idx]      # add channel dim


class VolumetricDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for volumetric datasets.

    Args:
        volumes (array-like): Input volumes.
        labels (array-like): Class labels.
        batch_size (int): Batch size.
        val_split (float): Fraction of data for validation.
        num_workers (int): Number of DataLoader workers.
    """
    def __init__(self, volumes, labels, batch_size=4,
                 val_split=0.2, num_workers=4):
        super().__init__()
        self.volumes, self.labels = volumes, labels
        self.batch_size, self.val_split, self.num_workers = \
            batch_size, val_split, num_workers

    def setup(self, stage=None):
        N = len(self.labels)
        idx = np.random.permutation(N)
        split = int(N * (1 - self.val_split))
        self.train_ds = VolumetricDataset(self.volumes[idx[:split]],
                                          self.labels[idx[:split]])
        self.val_ds   = VolumetricDataset(self.volumes[idx[split:]],
                                          self.labels[idx[split:]])

    def train_dataloader(self):
        return DataLoader(self.train_ds, self.batch_size, True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, self.batch_size, False,
                          num_workers=self.num_workers)


# ───────────────────── scikit-learn wrapper ───────────────────────────
class VolumetricTemporalClassifier(BaseEstimator,
                                   ClassifierMixin,
                                   TransformerMixin):
    """
    scikit-learn compatible volumetric-temporal classifier using PyTorch Lightning.

    Supports 3D spatial and 1D temporal convolutions with configurable schedules.
    Provides fit, predict, transform, and gradient-based saliency mapping.

    Args:
        max_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate.
        temporal_pool (str): Temporal pooling strategy ("mean", "cls", or None).
        spatial_conv_schedule (dict[int, KernelStride] or None): Spatial encoder schedule.
        drop_prob_spatial (float): Dropout probability in spatial encoder.
        drop_prob_temporal (float): Dropout probability in temporal encoder.
        gpus (int or None): Number of GPUs to use.
        random_state (int or None): Random seed.
        verbose (bool): Verbosity flag.
        temporal_conv_schedule (tuple[int, int, int]): Temporal encoder schedule (n_layers, kernel_size, stride).
        base_channels (int): Number of filters in the first spatial block.
        input_dropout (float): Dropout probability for input volume.
    """
    def __init__(self,
                 max_epochs: int = 20,
                 batch_size: int = 4,
                 lr: float = 1e-3,
                 temporal_pool: str = "mean",
                 spatial_conv_schedule: Dict[int, KernelStride] | None = None,
                 drop_prob_spatial: float = 0.0,
                 drop_prob_temporal: float = 0.0,
                 gpus: Optional[int] = None,
                 random_state: Optional[int] = None,
                 verbose: bool = False,
                 temporal_conv_schedule: tuple[int, int, int, int] = (2, 3, 1, 1),
                 base_channels: int = 8,
                 input_dropout: float = 0.2):
        # (n_layers, kernel_size, stride)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.temporal_pool = temporal_pool
        self.spatial_conv_schedule = spatial_conv_schedule
        self.gpus = gpus
        self.random_state = random_state
        self.verbose = verbose
        self.drop_prob_spatial = drop_prob_spatial
        self.drop_prob_temporal = drop_prob_temporal
        # Back-compat: accept legacy 2-tuple or 3-tuple
        if len(temporal_conv_schedule) == 2:
            temporal_conv_schedule = (temporal_conv_schedule[0], 3, temporal_conv_schedule[1], 1)
        elif len(temporal_conv_schedule) == 3:
            temporal_conv_schedule = (*temporal_conv_schedule, 1)
        self.temporal_conv_schedule = temporal_conv_schedule
        self.base_channels = base_channels
        self.input_dropout = input_dropout  # ← ADD this

    # --------------------------- fit ---------------------------
    def fit(self, X, y):
        """
        Train the model on the provided data.

        Args:
            X (np.ndarray): Input volumes of shape (N, T, Z, Y, X).
            y (np.ndarray): Class labels.

        Returns:
            self
        """
        torch.manual_seed(self.random_state or 0)
        np.random.seed(self.random_state or 0)

        N, T, Z, Y, Xdim = X.shape
        n_classes = int(np.max(y)) + 1

        spatial = Spatial3DEncoder(
            in_channels=1,
            base_channels=self.base_channels,
            depth=4,
            conv_schedule=self.spatial_conv_schedule or {},
            drop_prob_spatial=self.drop_prob_spatial
        )

        # compute flattened F
        with torch.no_grad():
            dummy_out = spatial(torch.randn(1, T, 1, Z, Y, Xdim))
        _, _, C, H, W, D = dummy_out.shape
        F = C * H * W * D

        n_layers_t, kernel_t, stride_t, dil_t = self.temporal_conv_schedule
        temporal = Temporal1DEncoder(
            input_dim=F,
            hidden_dim=256,
            n_layers=n_layers_t,
            kernel_size=kernel_t,
            stride_temporal=stride_t,
            dilation_temporal=dil_t,
            pool=self.temporal_pool,
            drop_prob_temporal=self.drop_prob_temporal
        )
        emb_dim = temporal(torch.randn(1, T, F)).shape[-1]

        backbone = VolumetricTemporalBackbone(
            spatial,
            temporal,
            input_dropout=self.input_dropout  # ← pass argument
        )
        head = ClassificationHead(emb_dim, n_classes)

        # — loss history containers (filled by LitModule callbacks) —
        train_hist: List[float] = []
        val_hist:   List[float] = []

        lit_model = VolumetricLitModule(
            backbone,
            head,
            lr=self.lr,
            train_hist=train_hist if self.verbose else None,
            val_hist=val_hist   if self.verbose else None,
            verbose=self.verbose,                             # NEW
        )

        # 90 / 10 internal split
        dm = VolumetricDataModule(X, y, self.batch_size, val_split=0.1)

        # — choose accelerator safely —
        use_gpu = self.gpus not in (None, 0, [], False)
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator="gpu" if use_gpu else "cpu",
            devices=self.gpus if use_gpu else 1,
            log_every_n_steps=1 if self.verbose else 50  # NEW: force frequent logs
        )
        trainer.fit(lit_model, dm)

        # — optional matplotlib plot —
        if self.verbose and train_hist and val_hist:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(train_hist, label="train")
            plt.plot(val_hist,   label="val")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.yscale("log")
            plt.title("Training progress")
            plt.legend()
            plt.grid()
            plt.show()

        self.model_ = lit_model.eval()
        return self

    # ----------------------- inference ------------------------
    def _logits(self, X):
        """
        Compute raw logits for input X.

        Args:
            X (np.ndarray): Input volumes.

        Returns:
            torch.Tensor: Logits.
        """
        check_is_fitted(self, "model_")
        with torch.no_grad():
            return self.model_(
                torch.as_tensor(X, dtype=torch.float32).unsqueeze(2))

    def predict(self, X):
        """
        Predict class labels for input X.

        Args:
            X (np.ndarray): Input volumes.

        Returns:
            np.ndarray: Predicted class labels.
        """
        return self._logits(X).argmax(1).cpu().numpy()

    def predict_proba(self, X):
        """
        Predict class probabilities for input X.

        Args:
            X (np.ndarray): Input volumes.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        return torch.softmax(self._logits(X), 1).cpu().numpy()

    def transform(self, X):
        """
        Extract feature embeddings from the backbone.

        Args:
            X (np.ndarray): Input volumes.

        Returns:
            np.ndarray: Feature embeddings.
        """
        check_is_fitted(self, "model_")
        with torch.no_grad():
            x = torch.as_tensor(X, dtype=torch.float32).unsqueeze(2)
            return self.model_.backbone(x).cpu().numpy()
        
    def saliency(self, X: np.ndarray, num_samples: int = 25, noise_std: float = 0.1) -> np.ndarray:
        """
        Compute SmoothGrad with Gradient × Input attribution.

        For each input, adds Gaussian noise multiple times, computes the gradient × input,
        and averages the squared results to yield a smooth attribution map.

        Args:
            X (np.ndarray): Input volumes of shape (N, T, Z, Y, X).
            num_samples (int): Number of noisy samples to average.
            noise_std (float): Standard deviation of noise (relative to input range).

        Returns:
            np.ndarray: Smooth attribution map of shape (N, T, Z, Y, X).
        """
        check_is_fitted(self, "model_")
        self.model_.eval()
        device = next(self.model_.parameters()).device

        x_orig = torch.tensor(X, dtype=torch.float32, device=device)  # (N, T, Z, Y, X)
        x_orig = x_orig.unsqueeze(2)  # (N, T, 1, Z, Y, X) to match your model

        N = x_orig.shape[0]
        all_grads = []

        for _ in range(num_samples):
            noise = torch.randn_like(x_orig) * noise_std
            x_noisy = (x_orig + noise).clone().detach().requires_grad_(True)

            logits = self.model_(x_noisy)               # (N, C)
            preds = logits.argmax(dim=1)                # (N,)
            selected = logits[torch.arange(N), preds]   # (N,)

            grads, = torch.autograd.grad(
                outputs=selected,
                inputs=x_noisy,
                grad_outputs=torch.ones_like(selected),
                create_graph=False,
                retain_graph=False,
                only_inputs=True
            )

            # Gradient × Input (with noise)
            grad_input = grads * x_noisy                # shape: (N, T, 1, Z, Y, X)
            all_grads.append(grad_input.detach())

        grads_stack = torch.stack(all_grads, dim=0)     # (S, N, T, 1, Z, Y, X)
        smooth = grads_stack.pow(2).mean(dim=0)         # average over S
        saliency = smooth.sqrt().squeeze(2)             # (N, T, Z, Y, X)

        return saliency.cpu().numpy()


        # ────────────────────── Conv schedule utils ──────────────────────────
        # Utility functions for computing output shapes in a conv schedule.
        # This is useful for debugging and understanding the architecture.
        # It computes the output shape after each convolutional block
        # based on the input shape and the specified kernel/stride/dilation schedule.
        # The output is a list of tuples representing the shape after each block.
        # It can also print the schedule in a human-readable format.
        # Usage:
        #   from coco_grape.data_processor.generative.volumetric_temporal_classifier import conv_shape_progression, schedule_info
        #   schedule = {0: ((3, 3, 1), (1, 2, 2), (1, 1, 1)),
        #               1: ((3, 3, 1), (1, 2, 2), (1, 1, 1)),
        #               2: ((3, 3, 1), (1, 2, 2), (1, 1, 1)),
        #               3: ((3, 3, 3), (2, 2, 2), (1, 1, 1))}  # Example schedule
        #   input_shape = (10, 64, 64)  # Example input shape (Z, Y, X)
        #   shapes = conv_shape_progression(
        #       input_shape=input_shape,
        #       schedule=schedule,
        #       depth=len(schedule)
        #   )
        #   schedule_info(
        #       vols=np.zeros((1, 8, 10, 64, 64)),  # (N, T, Z, Y, X)
        #       spatial_conv_schedule=schedule,
        #       temporal_conv_schedule=(2, 3, 1, 1)
        #   )
        #   # This will print the output shape after each block in a formatted way.
        #   # The output will look like:
        #   # after block input → (10, 64, 64)
        #   # after block     0 → (10, 32, 32)
        #   # after block     1 → (10, 16, 16)
        #   # after block     2 → (10,  8,  8)
        #   # after block     3 → ( 5,  4,  4)
        #   # This shows how the spatial dimensions change after each convolutional block.
        # ──────────────────────────────────────────────────────────────────────
from typing import Dict, Tuple, List, Optional, Union
import numpy as np

# Types
Kernel3D  = Tuple[int, int, int]
Stride3D  = Tuple[int, int, int]
Dilation3D = Tuple[int, int, int]
SpatialSchedule = Dict[int, Tuple[Kernel3D, Stride3D, Dilation3D]]
TemporalSchedule = Tuple[int, int, int, int]  # (n_layers, kernel_size, stride, dilation)

def _out_dim(size: int, k: int, s: int, d: int, padding: Optional[int] = None) -> int:
    """
    Compute output size along one axis after convolution.

    Args:
        size (int): Input size.
        k (int): Kernel size.
        s (int): Stride.
        d (int): Dilation.
        padding (int or None): If None, defaults to 'same' for odd kernels.

    Returns:
        int: Output size.
    """
    p = ((k - 1) * d) // 2 if padding is None else padding
    return (size + 2 * p - d * (k - 1) - 1) // s + 1

def conv_shape_progression(
    input_shape: Tuple[int, int, int],
    schedule: SpatialSchedule,
    depth: int,
    default_kernel: Kernel3D = (3, 3, 3),
    default_stride: Stride3D = (1, 1, 1),
    default_dilation: Dilation3D = (1, 1, 1)
) -> List[Tuple[int, int, int]]:
    """
    Compute spatial shape progression through 3D conv blocks.

    Args:
        input_shape (tuple): Initial (Z, Y, X) shape.
        schedule (dict): Per-block (kernel, stride, dilation).
        depth (int): Number of spatial blocks.

    Returns:
        list[tuple]: List of (Z, Y, X) shapes after each block.
    """
    z, y, x = input_shape
    shapes = [(z, y, x)]

    for d in range(depth):
        k, s, dil = schedule.get(d, (default_kernel, default_stride, default_dilation))
        kz, ky, kx = k
        sz, sy, sx = s
        dz, dy, dx = dil

        z = _out_dim(z, kz, sz, dz)
        y = _out_dim(y, ky, sy, dy)
        x = _out_dim(x, kx, sx, dx)

        shapes.append((z, y, x))

    return shapes

def schedule_info(
    vols: Union[np.ndarray, 'torch.Tensor'],
    spatial_conv_schedule: SpatialSchedule,
    temporal_conv_schedule: TemporalSchedule,
    base_channels: int = 8,
):
    """
    Print spatial + temporal shape progression for debugging.

    Args:
        vols: Input volume (N, T, Z, Y, X).
        spatial_conv_schedule: Dict[int, (kernel, stride, dilation)].
        temporal_conv_schedule: Tuple (n_layers, kernel, stride, dilation).
        base_channels: First channel count.
    """
    _, T, Z, Y, X = vols.shape

    depth_spatial = max(spatial_conv_schedule.keys()) + 1
    shapes = conv_shape_progression(
        input_shape=(Z, Y, X),
        schedule=spatial_conv_schedule,
        depth=depth_spatial
    )

    # Pretty formatting
    w_z = max(len(str(s[0])) for s in shapes)
    w_y = max(len(str(s[1])) for s in shapes)
    w_x = max(len(str(s[2])) for s in shapes)

    print("Spatial encoder:")
    for i, (z, y, x) in enumerate(shapes):
        label = "input" if i == 0 else str(i - 1)
        print(f"  after block {label:>5} → ({z:>{w_z}}, {y:>{w_y}}, {x:>{w_x}})")

    # Final spatial dimensions
    z_f, y_f, x_f = shapes[-1]
    c_f = base_channels * 2 ** (depth_spatial - 1)
    F_in = c_f * z_f * y_f * x_f

    # Temporal schedule unpacking
    n_layers_t, k_t, s_t, d_t = temporal_conv_schedule
    pad_t = ((k_t - 1) * d_t) // 2 if s_t == 1 else 0

    # Temporal progression
    lengths = [T]
    for _ in range(n_layers_t):
        prev = lengths[-1]
        out = _out_dim(prev, k_t, s_t, d_t, padding=pad_t)
        lengths.append(out if out > 0 else 0)

    w_t = max(len(str(l)) for l in lengths)

    print("\nTemporal encoder:")
    print(f"  input length     : {T}")
    print(f"  input feature dim: {F_in}")
    print(f"  kernel size      : {k_t}")
    print(f"  padding          : {pad_t}")
    print(f"  stride           : {s_t}")
    print(f"  dilation         : {d_t}")
    for i, l in enumerate(lengths):
        label = "input" if i == 0 else str(i - 1)
        note = " (truncated)" if l == 0 else ""
        print(f"  after layer {label:>5} → {l:>{w_t}}{note}")
        if l == 0:
            break
