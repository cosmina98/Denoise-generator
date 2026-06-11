import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from typing import List
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar

def get_trainer(max_epochs=10, show_progress_bar: bool = True, **kwargs):
    # We control enable_progress_bar explicitly based on show_progress_bar,
    # so remove it from kwargs to prevent override.
    kwargs.pop("enable_progress_bar", None)

    # Extract user-defined callbacks (if any)
    user_callbacks = kwargs.pop("callbacks", [])
    # Remove any TQDMProgressBar from user_callbacks, as we'll manage it or disable it.
    final_callbacks = [cb for cb in user_callbacks if not isinstance(cb, TQDMProgressBar)]

    if show_progress_bar:
        final_callbacks.insert(0, TQDMProgressBar(refresh_rate=1)) # Add TQDMProgressBar

    return Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=show_progress_bar, # Directly determined by show_progress_bar
        callbacks=final_callbacks,
        log_every_n_steps=1,
        gradient_clip_val=1.0,
        **kwargs
    )


# -----------------------------
# Network components
# -----------------------------
class EncoderMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.resblock1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.resblock2 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.resblock1(x)
        x = x + self.resblock2(x)
        x = self.output_layer(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.downsample = downsample or (in_channels != out_channels)

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        self.skip_proj = nn.Identity()
        if self.downsample:
            self.skip_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip_proj(x)
        out = self.conv_block(x)
        return self.activation(out + identity)


class EncoderConv2D(nn.Module):
    def __init__(self,
                 in_hw: int,
                 out_dim: int,
                 dropout: float = 0.1,
                 num_channels: int = 1,
                 channels: List[int] = [32, 64, 128],
                 res_blocks_per_stage: int = 2,
                 use_gap: bool = True):   
        super().__init__()
        self.in_hw = in_hw
        self.in_channels = num_channels
        self.channels = channels
        self.use_gap = use_gap

        blocks = []
        C_in = num_channels
        for i, C_out in enumerate(channels):
            blocks.append(ResidualBlock(C_in, C_out, downsample=(i > 0)))
            C_in = C_out
            for _ in range(res_blocks_per_stage - 1):
                blocks.append(ResidualBlock(C_out, C_out, downsample=False))

        self.features = nn.Sequential(*blocks)

        # Compute output size after conv stack
        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, in_hw, in_hw)
            dummy_out = self.features(dummy)
            if use_gap:
                flat_dim = dummy_out.shape[1]  # just the number of channels
            else:
                flat_dim = dummy_out.view(1, -1).shape[1]

        if use_gap:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))  # GAP
        else:
            self.pool = nn.Identity()

        self.flatten = nn.Flatten()


        self.proj = nn.Sequential(
            self.pool,
            self.flatten,
            nn.Dropout(dropout),
            nn.Linear(flat_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def _to_image(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            # Assumes (Batch, Features) where Features = C * H * W
            B = x.size(0)
            expected_flat_dim = self.in_channels * self.in_hw * self.in_hw
            if x.shape[1] != expected_flat_dim:
                raise ValueError(
                    f"Input tensor has {x.shape[1]} features, but expected {expected_flat_dim} "
                    f"({self.in_channels} channels * {self.in_hw} H * {self.in_hw} W) for 2D input."
                )
            return x.view(B, self.in_channels, self.in_hw, self.in_hw)
        elif x.ndim == 3: # Assumes (Batch, H, W)
            if self.in_channels == 1:
                if x.shape[1] != self.in_hw or x.shape[2] != self.in_hw:
                    raise ValueError(
                        f"Input tensor has H,W dimensions {x.shape[1:]}, but model expects ({self.in_hw}, {self.in_hw})."
                    )
                return x.unsqueeze(1)  # Add channel dimension: (B, 1, H, W)
            else:
                raise ValueError(
                    f"Input tensor has 3 dimensions (shape {x.shape}), but model's in_channels is {self.in_channels} (not 1). "
                    "Provide 2D (B, C*H*W) or 4D (B, C, H, W) input."
                )
        elif x.ndim == 4: # Assumes (Batch, C, H, W)
            if x.shape[1] != self.in_channels:
                raise ValueError(f"Input tensor has {x.shape[1]} channels, but model expects {self.in_channels}.")
            if x.shape[2] != self.in_hw or x.shape[3] != self.in_hw:
                raise ValueError(f"Input tensor has H,W dimensions {x.shape[2:]}, but model expects ({self.in_hw}, {self.in_hw}).")
            return x
        else:
            raise ValueError(f"Unsupported input tensor ndim: {x.ndim}. Expected 2, 3, or 4 dimensions.")

    def forward(self, x):
        x = self._to_image(x)
        x = self.features(x)
        return self.proj(x)

class ClassifierHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes)
    def forward(self, z):
        return self.classifier(z)

def cosine_kernel(z1, z2, eps=1e-8):
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    return torch.matmul(z1, z2.T)

def safe_logdet(K, lambda_reg=1e-3):
    K_reg = K + lambda_reg * torch.eye(K.size(0), device=K.device)
    sign, logdet = torch.linalg.slogdet(K_reg)
    if sign <= 0:
        return torch.tensor(10.0, device=K.device)  # penalise badly behaved K
    return logdet


def volume_trace_loss(K, lambda_reg=1e-3, alpha=0.1):
    logdet = safe_logdet(K, lambda_reg)
    trace = torch.trace(K)
    return -logdet + alpha * trace

def inter_class_orthogonality_loss(Z, y):
    loss = 0.0
    unique_labels = torch.unique(y)
    for i, c1 in enumerate(unique_labels):
        for c2 in unique_labels[i+1:]:
            Z1 = Z[y == c1]
            Z2 = Z[y == c2]
            if Z1.size(0) == 0 or Z2.size(0) == 0:
                continue
            # Normalize row-wise for cosine similarity
            Z1 = F.normalize(Z1, p=2, dim=1)
            Z2 = F.normalize(Z2, p=2, dim=1)
            inner = torch.matmul(Z1, Z2.T)  # shape [n1, n2]
            loss += torch.sum(inner ** 2)
    return loss

# -----------------------------
# Lightning Module
# -----------------------------
class VolumeModel(pl.LightningModule):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 embedding_dim,
                 num_classes,
                 lambda_d=0.1,
                 lambda_t=1.0,
                 lambda_o=0.1,
                 alpha=0.1,
                 lr=1e-3,
                 weight_decay=1e-4,
                 dropout=0.1,
                 warmup_epochs=5,
                 encoder_type: str = "mlp",
                 image_side: int = 28,
                 num_channels: int = 1,
                 channels: List[int] = [32, 64],
                 res_blocks_per_stage=2,
                 use_gap=True):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.encoder_type == "mlp":
            self.encoder = EncoderMLP(
                in_dim=self.hparams.in_dim,
                hidden_dim=self.hparams.hidden_dim,
                out_dim=self.hparams.embedding_dim,
                dropout=self.hparams.dropout
            )
        elif self.hparams.encoder_type == "conv2d":
            self.encoder = EncoderConv2D(
                in_hw=self.hparams.image_side,
                out_dim=self.hparams.embedding_dim,
                dropout=self.hparams.dropout,
                num_channels=self.hparams.num_channels,
                channels=self.hparams.channels,
                res_blocks_per_stage=self.hparams.res_blocks_per_stage,
                use_gap=self.hparams.use_gap
            )
        else:
            raise ValueError(f"Unknown encoder_type '{self.hparams.encoder_type}'.")
        self.classifier = ClassifierHead(
            in_dim=self.hparams.embedding_dim,
            num_classes=self.hparams.num_classes
        )

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z_all = self.encoder(x)
        logits = self.classifier(z_all)
        loss_task = F.cross_entropy(logits, y)

        # Compute shared kernel
        K_all = cosine_kernel(z_all, z_all)

        # Volume loss (per class)
        loss_volume = 0.0
        classes_in_batch = torch.unique(y)
        class_count = 0

        for c in classes_in_batch:
            mask = (y == c)
            if mask.sum() < 2:
                continue
            z_c = z_all[mask]
            K_c = cosine_kernel(z_c, z_c)
            loss_dim = volume_trace_loss(K_c, lambda_reg=1e-3, alpha=self.hparams.alpha)
            loss_volume += self.hparams.lambda_d * loss_dim
            class_count += 1

        if class_count > 0:
            loss_volume /= class_count
        else:
            loss_volume = torch.tensor(0.0, device=x.device)

        # Orthogonality loss
        loss_orth = inter_class_orthogonality_loss(z_all, y)

        # Smooth ramp-up
        ramp = min(1.0, self.current_epoch / self.hparams.warmup_epochs)
        total_loss = ramp * (loss_volume + self.hparams.lambda_o * loss_orth) + self.hparams.lambda_t * loss_task

        self.log("loss_total", total_loss)
        self.log("loss_task", loss_task)
        self.log("loss_volume", loss_volume)
        self.log("loss_orth", loss_orth)
        self.log("train_acc", (logits.argmax(dim=1) == y).float().mean(), prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=5
        )
        return {"optimizer": opt, "lr_scheduler": sched, "monitor": "val_loss"}

    def embed(self, x):
        self.eval()
        with torch.no_grad():
            return self.encoder(x)

    def classify(self, x):
        self.eval()
        with torch.no_grad():
            z = self.encoder(x)
            return torch.argmax(self.classifier(z), dim=1)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        logits = self.classifier(z)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss
                                         


# -----------------------------
# Scikit-Learn Compatible Wrapper
class VolumeRegularisedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 hidden_dim=128,
                 embedding_dim=32,
                 epochs=10,
                 batch_size=64,
                 lr=1e-4,
                 lambda_d=0.1,
                 lambda_t=1.0,
                 lambda_o=0.1,
                 alpha=0.1,
                 dropout=0.1,
                 warmup_epochs=5,
                 weight_decay=1e-4,
                 encoder_type: str = "mlp",
                 image_side: int = 28,
                 num_channels: int = 1,
                 channels: List[int] = [32, 64],  # 🆕 NEW
                 trainer_kwargs=None):
                                         
        self.encoder_type = encoder_type
        self.image_side   = image_side
        self.num_channels = num_channels
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lambda_d = lambda_d
        self.lambda_t = lambda_t
        self.lambda_o = lambda_o
        self.alpha = alpha
        self.dropout = dropout
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay 
        self.trainer_kwargs = trainer_kwargs or {}

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        in_dim = X.shape[1]
        num_classes = len(np.unique(y))

        self.model = VolumeModel(
            in_dim,
            self.hidden_dim,
            self.embedding_dim,
            num_classes,
            lambda_d=self.lambda_d,
            lambda_t=self.lambda_t,
            lambda_o=self.lambda_o,
            weight_decay=self.weight_decay,
            alpha=self.alpha,
            lr=self.lr,
            dropout=self.dropout,
            warmup_epochs=self.warmup_epochs,
            encoder_type=self.encoder_type,
            image_side=self.image_side,
            num_channels=self.num_channels,
            channels=self.channels,  # 🆕
        )
        # Split indices
        total_indices = np.arange(len(X))
        val_size = max(1, int(0.1 * len(X)))
        val_indices = np.random.choice(total_indices, size=val_size, replace=False)
        train_indices = np.setdiff1d(total_indices, val_indices)

        # Subsets for train and val
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        # Datasets
        train_dataset = ClassAwareDataset(X_train, y_train)
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

        try:
            batch_sampler = BalancedMulticlassSampler(train_dataset, batch_size=self.batch_size, samples_per_class=4)
            train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
        except ValueError as e:
            print("⚠️  Warning:", e)
            print("Falling back to standard shuffled batches.")
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Track loss
        self.loss_callback = LossHistoryCallback(warmup_epochs=self.warmup_epochs)


        trainer = get_trainer(max_epochs=self.epochs, callbacks=[self.loss_callback], **self.trainer_kwargs)
        trainer.fit(self.model, train_loader, val_loader)

        return self



    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        return self.model.classify(X_tensor).cpu().numpy()

    def transform(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        return self.model.embed(X_tensor).cpu().numpy()
    
    def plot_loss_curves(self):
        if hasattr(self, 'loss_callback'):
            self.loss_callback.plot()
        else:
            raise RuntimeError("Loss callback not found. Train the model first.")

class LossHistoryCallback(pl.callbacks.Callback):
    def __init__(self, warmup_epochs=5):
        self.train_losses = []
        self.task_losses = []
        self.volume_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.warmup_epochs = warmup_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if (loss := metrics.get("loss_total")) is not None:
            self.train_losses.append(loss.item())
        if (task := metrics.get("loss_task")) is not None:
            self.task_losses.append(task.item())
        if (vol := metrics.get("loss_volume")) is not None:
            self.volume_losses.append(vol.item())
        if (acc := metrics.get("train_acc")) is not None:
            self.train_accs.append(acc.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if (loss := metrics.get("val_loss")) is not None:
            self.val_losses.append(loss.item())
        if (acc := metrics.get("val_acc")) is not None:
            self.val_accs.append(acc.item())

    def plot(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # --- Losses
        ax[0].plot(self.train_losses, label="Total Loss", color='blue')
        ax[0].plot(self.task_losses, label="Task Loss", color='green')
        ax[0].plot(self.volume_losses, label="Volume Loss", color='orange')
        if self.val_losses:
            ax[0].plot(self.val_losses, label="Val Loss", color='red', linestyle='--')
        ax[0].set_yscale("log")
        ax[0].set_ylabel("Loss")
        ax[0].set_title("Loss Components")
        ax[0].axvline(self.warmup_epochs, color='black', linestyle=':', label='Warm-Up End / Volume Fully On')
        ax[0].legend()
        ax[0].grid(True)

        # --- Accuracies
        if self.train_accs:
            ax[1].plot(self.train_accs, label="Train Accuracy", color='blue', marker='o')
        if self.val_accs:
            ax[1].plot(self.val_accs, label="Val Accuracy", color='red', marker='x')
        ax[1].set_ylabel("Accuracy")
        ax[1].set_xlabel("Epoch")
        ax[1].set_title("Accuracy")
        ax[1].legend()
        ax[1].grid(True)

        plt.tight_layout()
        plt.show()



import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
import random

class ClassAwareDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.class_to_indices = defaultdict(list)
        for idx, label in enumerate(y):
            self.class_to_indices[int(label)].append(idx)
        self.num_classes = len(self.class_to_indices)
        self.all_indices = np.arange(len(y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BalancedMulticlassSampler:
    def __init__(self, dataset, batch_size, samples_per_class=4, max_attempts=1000):
        self.dataset = dataset
        self.batch_size = batch_size
        self.class_to_indices = dataset.class_to_indices
        self.classes = list(self.class_to_indices.keys())
        self.max_attempts = max_attempts

        # Compute feasible configuration
        self.num_classes = len(self.classes)
        max_samples = max(len(v) for v in self.class_to_indices.values())
        
        # Adjust samples_per_class downward if needed
        spc = samples_per_class
        while spc > 0:
            num_classes_needed = batch_size // spc
            if num_classes_needed <= self.num_classes:
                self.samples_per_class = spc
                self.num_classes_per_batch = num_classes_needed
                break
            spc -= 1
        else:
            raise ValueError("Cannot construct a valid batch: not enough classes or samples.")

    def __iter__(self):
        attempts = 0
        while attempts < self.max_attempts:
            selected_classes = random.sample(self.classes, self.num_classes_per_batch)
            batch = []
            for c in selected_classes:
                candidates = self.class_to_indices[c]
                if len(candidates) < self.samples_per_class:
                    break  # not enough in this class → retry
                batch += random.sample(candidates, self.samples_per_class)

            if len(batch) == self.batch_size:
                random.shuffle(batch)
                yield batch
                attempts = 0  # reset attempts after success
            else:
                attempts += 1

        raise RuntimeError("BalancedMulticlassSampler failed to generate a valid batch.")

    def __len__(self):
        return len(self.dataset) // self.batch_size
