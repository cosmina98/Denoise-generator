import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import MinMaxScaler
import contextlib
import logging

############################################
# Custom low‐rank linear layer (LowRankLinear)
############################################

class LowRankLinear(nn.Module):
    """
    A linear layer whose weight matrix is factorized as A @ B,
    where A is (in_features x thin_size) and B is (thin_size x out_features).
    """
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
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        out = x @ self.A @ self.B
        if self.bias is not None:
            out = out + self.bias
        return out

############################################
# Residual block with dropout and LeakyReLU
############################################

class ResidualBlock(nn.Module):
    """
    A residual block with low-rank linear -> dropout -> LeakyReLU, plus skip.
    """
    def __init__(self, in_features, out_features, thin_size, dropout_prob, negative_slope, use_layernorm: bool = False):
        super().__init__()
        self.linear = LowRankLinear(in_features, out_features, thin_size)
        self.norm = nn.LayerNorm(out_features) if use_layernorm else nn.Identity()
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.LeakyReLU(negative_slope)
        self.skip = LowRankLinear(in_features, out_features, thin_size) if in_features != out_features else nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)  # Calculate skip connection first
        out = self.linear(x)
        out = self.norm(out)     # Apply optional LayerNorm
        out = self.dropout(out)  # Apply dropout
        out = self.activation(out)  # Apply activation
        return out + identity    # Add skip connection

############################################
# LowRankMLP Network
############################################

class LowRankMLPNet(nn.Module):
    """
    MLP built from ResidualBlocks and a final LowRankLinear.
    """
    def __init__(self, input_dim, output_dim,
                 hidden_layers, hidden_dim, thin_size, dropout,
                 negative_slope, use_layernorm_in_residual: bool = False):
        super().__init__()
        self.hidden_layers = hidden_layers
        if hidden_layers > 0:
            blocks = [ResidualBlock(input_dim, hidden_dim, thin_size, dropout, negative_slope, use_layernorm_in_residual)]
            blocks += [
                ResidualBlock(hidden_dim, hidden_dim, thin_size, dropout, negative_slope, use_layernorm_in_residual)
                for _ in range(hidden_layers - 1)
            ]
            self.blocks = nn.ModuleList(blocks)
            self.out_layer = LowRankLinear(hidden_dim, output_dim, thin_size)
        else:
            self.out_layer = LowRankLinear(input_dim, output_dim, thin_size)

    def forward(self, x):
        for block in getattr(self, 'blocks', []):
            x = block(x)
        return self.out_layer(x)

############################################
# StopWhenLRBelow callback with min_epochs
############################################

class StopWhenLRBelow(pl.Callback):
    """
    Stops training when LR falls below a threshold, but only after a minimum number
    of epochs.
    """
    def __init__(self, min_lr=1e-8, min_epochs=0):
        super().__init__()
        self.min_lr = min_lr
        self.min_epochs = min_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch < self.min_epochs:
            return
        lr = trainer.optimizers[0].param_groups[0]['lr']
        if lr < self.min_lr:
            print(f"LR {lr:.2e} below {self.min_lr:.2e} at epoch {epoch}; stopping.")
            trainer.should_stop = True

############################################
# Lightning Module
############################################

class LowRankMLPModule(pl.LightningModule):
    """
    Lightning wrapper around LowRankMLPNet, tracks losses.
    """
    def __init__(self, net, lr, task='classification', lr_patience=3, class_weights=None, verbose=False):
        super().__init__()
        self.net = net
        self.lr = lr
        self.task = task
        self.lr_patience = lr_patience
        self.verbose = verbose

        self.mse_loss = nn.MSELoss()

        if task == 'classification' and class_weights is not None:
            cw = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", cw)  # moves with the module
        else:
            self.register_buffer("class_weights", torch.tensor([], dtype=torch.float32))

        self.train_losses, self.val_losses = [], []
        self._train_batch, self._val_batch = [], []

    def forward(self, x):
        return self.net(x)

    def _ce(self, logits, targets):
        w = self.class_weights if self.class_weights.numel() else None
        return F.cross_entropy(logits, targets, weight=w)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if self.task == 'classification':
            loss = self._ce(logits, y)
        else:
            loss = self.mse_loss(logits, y)
        self.log('train_loss', loss)
        self._train_batch.append(loss.detach())
        return loss

    def on_train_epoch_end(self):
        if self._train_batch:
            avg = torch.stack(self._train_batch).mean().item()
            self.train_losses.append(avg)
            self._train_batch = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._ce(logits, y) if self.task == 'classification' else self.mse_loss(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self._val_batch.append(loss.detach())

    def on_validation_epoch_end(self):
        if self._val_batch:
            avg = torch.stack(self._val_batch).mean().item()
            self.val_losses.append(avg)
            self._val_batch = []

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        try:
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, patience=self.lr_patience, factor=0.1, verbose=True
            )
        except TypeError:
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, patience=self.lr_patience, factor=0.1
            )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }

    def on_train_end(self):
        if not self.verbose or not (self.train_losses and self.val_losses):
            return
        min_len = min(len(self.train_losses), len(self.val_losses))
        skip = 5 if min_len > 5 else 0
        t = self.train_losses[skip:min_len]
        v = self.val_losses[skip:min_len]
        epochs = range(skip + 1, skip + 1 + len(t))
        plt.figure(figsize=(10,5))
        plt.plot(epochs, t, label='Train Loss')
        plt.plot(epochs, v, label='Validation Loss')
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

############################################
# sklearn-compatible estimator with checkpoint policy
############################################

class LowRankMLP(BaseEstimator, ClassifierMixin):
    """
    sklearn-like estimator wrapping our Lightning MLP.
    - Warm start: preserves model weights but reinitializes optimizer / scheduler.
    - Checkpoint policy: 'last' | 'best' | 'all' | 'off'
    """
    def __init__(
        self,
        hidden_layers=2,
        hidden_dim=100,
        thin_size=10,
        dropout=0.5,
        negative_slope=0.01,
        lr=1e-3,
        max_epochs=30,
        batch_size=32,
        task='classification',         # 'classification' or 'regression'
        lr_patience=5,
        verbose=False,
        enable_progress_bar=False,
        balance=True,
        min_lr=1e-6,
        warm_start=True,
        min_epochs=0,
        use_layernorm_in_residual: bool = True,
        use_minmax_scaler: bool = True,

        # --- checkpoint controls ---
        dataset_name: str | None = None,        # e.g. "QM9" (optional subfolder)
        checkpoint_dir: str = "checkpoints",    # root for all runs
        checkpoint_name: str | None = None,     # filename stem; if None we auto-name
        component_name: str | None = None,      # e.g. "node"/"edge"/"adj" (optional)
        checkpoint_policy: str = "off",         # "last" | "best" | "all" | "off"
        checkpoint_every_n_epochs: int = 1,     # for "all" (and val cadence)
        save_weights_only: bool = True,         # big size win

        # --- early stopping ---
        use_early_stopping: bool = True,
        early_stop_patience: int = 10,
        early_stop_min_delta: float = 0.0,
        early_stop_monitor: str = "val_loss",   # make sure you log this metric
        early_stop_mode: str = "min",
    ):
        # core hyperparams
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.thin_size = thin_size
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.task = task
        self.lr_patience = lr_patience
        self.verbose = verbose
        self.enable_progress_bar = enable_progress_bar
        self.balance = balance
        self.min_lr = min_lr
        self.warm_start = warm_start
        self.min_epochs = min_epochs
        self.use_layernorm_in_residual = use_layernorm_in_residual
        self.use_minmax_scaler = use_minmax_scaler
        self.scaler_ = None

        # checkpoint settings (assign BEFORE using)
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_name = checkpoint_name  # may be None
        self.component_name = component_name or task  # fallback to task

        self.checkpoint_policy = checkpoint_policy
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.save_weights_only = save_weights_only

        # early stopping
        self.use_early_stopping = use_early_stopping
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.early_stop_monitor = early_stop_monitor
        self.early_stop_mode = early_stop_mode

        # paths will be filled after fit()
        self._ckpt_last = None
        self._ckpt_best = None
        self._checkpoint_path = None

    def fit(self, X, y):
        # 1) Data
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        if self.use_minmax_scaler:
            if not (self.warm_start and hasattr(self, 'scaler_') and self.scaler_ is not None):
                self.scaler_ = MinMaxScaler()
                X_arr = self.scaler_.fit_transform(X_arr)
            elif self.scaler_ is not None:
                X_arr = self.scaler_.transform(X_arr)

        # 2) Warm-start checks
        if self.warm_start and hasattr(self, 'net_'):
            if X_arr.shape[1] != self.input_dim_:
                raise ValueError(f"Warm start input_dim={self.input_dim_}, got {X_arr.shape[1]}")
            if self.task == 'classification':
                new_cls = np.unique(y_arr.ravel())
                if not np.array_equal(new_cls, self.classes_):
                    raise ValueError(f"Warm start classes changed from {self.classes_} to {new_cls}")
            else:
                out_dim = y_arr.shape[1] if y_arr.ndim > 1 else 1
                if out_dim != self.output_dim_:
                    raise ValueError(f"Warm start output_dim={self.output_dim_}, got {out_dim}")

        # 3) Targets
        class_weights = None
        if self.task == 'classification':
            flat = y_arr.ravel()
            if not (self.warm_start and hasattr(self, 'classes_')):
                self.classes_ = np.unique(flat)
            mapping = {c: i for i, c in enumerate(self.classes_)}
            mapped = np.vectorize(mapping.get)(flat)
            y_tensor = torch.from_numpy(mapped).long()

            if self.balance:
                counts = np.bincount(mapped, minlength=len(self.classes_)).astype(np.float64)
                eps = 1e-8
                weights = 1.0 / (counts + eps)
                weights *= (len(weights) / weights.sum())  # normalize to mean 1
                class_weights = torch.tensor(weights, dtype=torch.float32)

            out_dim = len(self.classes_)
        else:  # regression
            y_arr = y_arr.astype(np.float32)
            if y_arr.ndim == 1:
                y_arr = y_arr[:, None]
            y_tensor = torch.from_numpy(y_arr)
            out_dim = y_arr.shape[1]

        # 4) Build or reuse model weights
        if not (self.warm_start and hasattr(self, 'net_')):
            self.input_dim_ = X_arr.shape[1]
            self.output_dim_ = out_dim
            self.net_ = LowRankMLPNet(
                input_dim=self.input_dim_,
                output_dim=self.output_dim_,
                hidden_layers=self.hidden_layers,
                hidden_dim=self.hidden_dim,
                thin_size=self.thin_size,
                dropout=self.dropout,
                negative_slope=self.negative_slope,
                use_layernorm_in_residual=self.use_layernorm_in_residual
            )
        net = self.net_

        # 5) LightningModule (fresh optimizer & scheduler each fit)
        self.module_ = LowRankMLPModule(
            net,
            lr=self.lr,
            task=self.task,
            lr_patience=self.lr_patience,
            class_weights=class_weights,
            verbose=self.verbose
        )

        # 6) DataLoaders
        X_tensor = torch.from_numpy(X_arr.astype(np.float32))
        ds = TensorDataset(X_tensor, y_tensor)
        val_size = max(1, int(0.1 * len(ds))) if len(ds) > 1 else 1
        train_size = max(0, len(ds) - val_size)
        if (
            self.task == 'classification'
            and len(ds) > 1
            and train_size > 0
            and len(np.unique(mapped)) > 1
            and val_size >= len(np.unique(mapped))
            and np.min(np.bincount(mapped, minlength=len(np.unique(mapped)))) >= 2
        ):
            indices = np.arange(len(ds))
            tr_idx, vl_idx = train_test_split(
                indices,
                test_size=val_size,
                random_state=42,
                stratify=mapped,
            )
            tr_ds = torch.utils.data.Subset(ds, tr_idx.tolist())
            vl_ds = torch.utils.data.Subset(ds, vl_idx.tolist())
        else:
            tr_ds, vl_ds = random_split(ds, [train_size, val_size])
        tr_ld = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True) if train_size > 0 else None
        vl_ld = DataLoader(vl_ds, batch_size=self.batch_size)

        # 7) Callbacks
        callbacks = [StopWhenLRBelow(min_lr=self.min_lr, min_epochs=self.min_epochs)]
        if self.enable_progress_bar:
            callbacks.append(RichProgressBar(refresh_rate=30))

        if self.use_early_stopping:
            callbacks.append(EarlyStopping(
                monitor=self.early_stop_monitor,
                mode=self.early_stop_mode,
                patience=self.early_stop_patience,
                min_delta=self.early_stop_min_delta,
                check_finite=True,
            ))

        # 8) Checkpoint policy (keep both BEST and LAST where applicable)
        ckpt = None
        if self.checkpoint_policy != "off":
            ckpt_dir = self.checkpoint_dir
            if self.dataset_name:
                ckpt_dir = os.path.join(ckpt_dir, self.dataset_name)
            if self.component_name:
                ckpt_dir = os.path.join(ckpt_dir, self.component_name)
            os.makedirs(ckpt_dir, exist_ok=True)

            fname = self.checkpoint_name or f"lowrankmlp-{self.component_name}-h{self.hidden_dim}"

            common = dict(
                dirpath=ckpt_dir,
                filename=fname,
                save_weights_only=self.save_weights_only,
            )

            if self.checkpoint_policy == "last":
                ckpt = ModelCheckpoint(
                    **common,
                    save_last=True,        # rolling last.ckpt
                    save_top_k=0,
                    monitor=None
                )
            elif self.checkpoint_policy == "best":
                ckpt = ModelCheckpoint(
                    **common,
                    save_last=True,        # <-- keep last as well
                    save_top_k=1,          # keep best by monitor
                    monitor=self.early_stop_monitor,
                    mode=self.early_stop_mode
                )
            elif self.checkpoint_policy == "all":
                ckpt = ModelCheckpoint(
                    **common,
                    save_last=True,        # <-- keep last as well
                    save_top_k=-1,         # keep ALL checkpoints
                    monitor=self.early_stop_monitor,
                    mode=self.early_stop_mode,
                    every_n_epochs=self.checkpoint_every_n_epochs
                )
            else:
                raise ValueError(f"Unknown checkpoint_policy: {self.checkpoint_policy}")

            callbacks.append(ckpt)

        # 9) Trainer
        trainer_kwargs = dict(
            max_epochs=self.max_epochs,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            callbacks=callbacks,
            enable_checkpointing=(self.checkpoint_policy != "off"),
            enable_progress_bar=self.enable_progress_bar,
            check_val_every_n_epoch=self.checkpoint_every_n_epochs,
        )

        if not self.verbose:
            log = logging.getLogger('pytorch_lightning'); prev = log.level; log.setLevel(logging.ERROR)
            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                trainer = pl.Trainer(logger=False, **trainer_kwargs)
                trainer.fit(self.module_, tr_ld, vl_ld)
            log.setLevel(prev)
        else:
            trainer = pl.Trainer(**trainer_kwargs)
            trainer.fit(self.module_, tr_ld, vl_ld)

        # 10) Record state & return
        self._epochs_trained_ = trainer.current_epoch
        if ckpt is None:
            self._ckpt_last = None
            self._ckpt_best = None
            self._checkpoint_path = None
        else:
            self._ckpt_last = ckpt.last_model_path or None
            self._ckpt_best = ckpt.best_model_path or None
            # prefer best for inference; fall back to last
            self._checkpoint_path = self._ckpt_best or self._ckpt_last

        self.net_ = self.module_.net.eval()
        return self

    # ---------------- Inference API ----------------

    def _forward_in_batches(self, X):
        """
        Device-safe batched forward pass for inference.
        """
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        batch_size = max(1, int(self.batch_size))
        net_device = next(self.net_.parameters()).device
        outputs = []

        self.net_.eval()
        with torch.no_grad():
            for start in range(0, X_arr.shape[0], batch_size):
                xb = torch.from_numpy(X_arr[start:start + batch_size]).to(net_device)
                outputs.append(self.net_(xb).detach().cpu())

        if not outputs:
            out_dim = getattr(self, "output_dim_", 1)
            return torch.empty((0, out_dim), dtype=torch.float32)
        return torch.cat(outputs, dim=0)

    def predict(self, X):
        """
        Predict class labels or regression outputs for X.
        """
        if self.use_minmax_scaler and self.scaler_ is not None:
            X = self.scaler_.transform(X)
        out = self._forward_in_batches(X)
        if self.task == 'classification':
            idx = out.argmax(dim=1).cpu().numpy()
            return self.classes_[idx]
        return out.cpu().numpy()

    def predict_proba(self, X):
        """
        Predict class probabilities for classification task.
        """
        if self.task != 'classification':
            raise AttributeError("predict_proba only available for classification tasks.")
        if self.use_minmax_scaler and self.scaler_ is not None:
            X = self.scaler_.transform(X)
        logits = self._forward_in_batches(X)
        return F.softmax(logits, dim=1).cpu().numpy()
    
    def score(self, X, y):
        """
        Returns accuracy for classification, R^2 for regression.
        """
        from sklearn.metrics import accuracy_score, r2_score
        y_pred = self.predict(X)
        if self.task == 'classification':
            return accuracy_score(y, y_pred)
        else:
            return r2_score(y, y_pred)

    # ---------------- Checkpoint utilities ----------------

    def load_weights(self, which: str = "best", map_location: str | torch.device = "cpu"):
        """
        Load weights from a saved checkpoint into self.net_.

        Parameters
        ----------
        which : {"best", "last"}
            Which checkpoint to load.
        map_location : str or torch.device
            Where to map tensors when loading.

        Notes
        -----
        - We saved the LightningModule `state_dict`. Keys are usually prefixed with "net.".
          This method strips that prefix before loading into the bare `self.net_`.
        """
        path = {"best": self._ckpt_best, "last": self._ckpt_last}.get(which)
        if not path:
            raise ValueError(f"No {which} checkpoint available to load.")
        obj = torch.load(path, map_location=map_location)
        if "state_dict" not in obj:
            raise RuntimeError(f"Checkpoint at {path} has no 'state_dict' key.")
        sd = obj["state_dict"]

        # Extract only the sub-keys for the network, stripping "net." if present
        if any(k.startswith("net.") for k in sd.keys()):
            clean = {k[len("net."):]: v for k, v in sd.items() if k.startswith("net.")}
        else:
            # fallback: try to load as-is (strict=False to ignore non-matching keys)
            clean = sd

        missing, unexpected = self.net_.load_state_dict(clean, strict=False)
        if self.verbose:
            print(f"Loaded '{which}' weights from {path}")
            if missing:
                print("Missing keys:", missing)
            if unexpected:
                print("Unexpected keys:", unexpected)
