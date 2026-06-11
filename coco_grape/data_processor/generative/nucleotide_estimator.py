import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import matplotlib.pyplot as plt

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Callback
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
import numpy as np
from sklearn.cluster import KMeans
from scipy.signal import savgol_filter
import ruptures as rpt

def cut_points_kmeans(saliency: np.ndarray, k=2):
    """
    Segment a saliency array by K-means clustering on raw saliency values.

    Args:
        saliency (np.ndarray): 1D array of saliency values.
        k (int): Number of clusters.

    Returns:
        np.ndarray: Indices where cluster assignments change (cut points).
    """
    values = saliency
    labels = KMeans(n_clusters=k, n_init="auto").fit_predict(values[:, None])
    return np.where(np.diff(labels) != 0)[0] + 1

def cut_points_cusum(saliency: np.ndarray, penalty=10):
    """
    Segment a saliency array using binary segmentation (CUSUM) via ruptures.

    Args:
        saliency (np.ndarray): 1D array of saliency values.
        penalty (float): Penalty value for segmentation.

    Returns:
        list: Indices of cut points (excluding last segment end).
    """
    values = saliency
    model = rpt.Binseg(model="l2").fit(values)
    cuts = model.predict(pen=penalty)
    return cuts[:-1]

def cut_points_gradient(saliency: np.ndarray, window=20, threshold=0.05):
    """
    Detect transitions in saliency via smoothed gradient magnitude.

    Args:
        saliency (np.ndarray): 1D array of saliency values.
        window (int): Window size for smoothing.
        threshold (float): Threshold for gradient magnitude.

    Returns:
        list: Indices where gradient exceeds threshold.
    """
    values = saliency
    grad = np.convolve(values, [-1] * window + [1] * window, mode="same")
    change_points = np.where(np.abs(grad) > threshold)[0]
    return change_points.tolist()

def cut_points_savgol(saliency: np.ndarray, window=31, poly=3):
    """
    Detect inflection points via 2nd derivative of a Savitzky-Golay smoothed signal.

    Args:
        saliency (np.ndarray): 1D array of saliency values.
        window (int): Window length for smoothing.
        poly (int): Polynomial order for smoothing.

    Returns:
        list: Indices of zero-crossings in the 2nd derivative.
    """
    values = saliency
    d2 = savgol_filter(values, window_length=window, polyorder=poly, deriv=2)
    zero_crossings = np.where(np.diff(np.sign(d2)))[0]
    return zero_crossings.tolist()

def cut_points_threshold(saliency: np.ndarray, percentile=60, min_gap=30):
    """
    Segment a saliency array by thresholding with minimum spacing between cut points.

    Args:
        saliency (np.ndarray): 1D array of saliency values.
        percentile (float): Percentile for thresholding.
        min_gap (int): Minimum gap between consecutive cut points.

    Returns:
        list: Indices of cut points.
    """
    values = saliency
    threshold = np.percentile(values, percentile)
    binary = (values > threshold).astype(int)
    cut_points = []
    last = binary[0]
    for i in range(1, len(binary)):
        if binary[i] != last:
            if len(cut_points) == 0 or (i - cut_points[-1]) > min_gap:
                cut_points.append(i)
            last = binary[i]
    return cut_points
from typing import List, Tuple
import numpy as np

def cut_points_threshold_auto(
    saliency: np.ndarray,
    target_segments: int = 10,
    min_gap: int = 30,
    percentile_range: Tuple[int, int] = (95, 50),
    max_iter: int = 20
) -> List[int]:
    """
    Automatically tune the threshold percentile to obtain approximately the target number of segments.

    Args:
        saliency (np.ndarray): 1D saliency array.
        target_segments (int): Desired number of segments.
        min_gap (int): Minimum spacing between cut points.
        percentile_range (Tuple[int, int]): (high, low) percentiles for search.
        max_iter (int): Maximum number of binary search steps.

    Returns:
        List[int]: Indices of cut points.
    """
    low_p, high_p = percentile_range
    best_cuts = []

    for _ in range(max_iter):
        mid_p = (low_p + high_p) / 2
        threshold = np.percentile(saliency, mid_p)
        binary = (saliency > threshold).astype(int)

        cut_points = []
        last = binary[0]
        for i in range(1, len(binary)):
            if binary[i] != last:
                if len(cut_points) == 0 or (i - cut_points[-1]) > min_gap:
                    cut_points.append(i)
                last = binary[i]

        n_segments = len(cut_points) + 1
        best_cuts = cut_points

        if n_segments < target_segments:
            high_p = mid_p
        else:
            low_p = mid_p

    return best_cuts
def cut_points_gradient_auto(
    saliency: np.ndarray,
    target_segments: int = 10,
    window: int = 20,
    threshold_range: Tuple[float, float] = (0.001, 0.1),
    max_iter: int = 20
) -> List[int]:
    """
    Automatically tune the gradient threshold to obtain approximately the target number of segments.

    Args:
        saliency (np.ndarray): 1D saliency array.
        target_segments (int): Desired number of segments.
        window (int): Smoothing window size.
        threshold_range (Tuple[float, float]): (low, high) search bounds for threshold.
        max_iter (int): Maximum number of binary search steps.

    Returns:
        List[int]: Indices of cut points.
    """
    low_t, high_t = threshold_range
    best_cuts = []

    grad = np.convolve(saliency, [-1] * window + [1] * window, mode="same")

    for _ in range(max_iter):
        mid_t = (low_t + high_t) / 2
        cut_points = np.where(np.abs(grad) > mid_t)[0].tolist()

        # Post-process to ensure spacing
        filtered = []
        for i in cut_points:
            if len(filtered) == 0 or (i - filtered[-1]) > window:
                filtered.append(i)

        n_segments = len(filtered) + 1
        best_cuts = filtered

        if n_segments < target_segments:
            high_t = mid_t
        else:
            low_t = mid_t

    return best_cuts

class MLPHead(nn.Module):
    """
    Multi-layer perceptron head for sequence classification/regression.

    Args:
        in_dim (int): Input feature dimension.
        hidden (Tuple[int, ...]): Hidden layer sizes.
        n_out (int): Output dimension.
        dropout (float): Dropout rate.
        activation (str): Activation function ("relu" or "gelu").
    """
    def __init__(self, in_dim: int, hidden: Tuple[int, ...], n_out: int,
                 dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        act = {"relu": nn.ReLU(), "gelu": nn.GELU()}[activation]
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), act, nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ConvHead(nn.Module):
    """
    1D convolutional feature extractor for nucleotide sequences.

    Args:
        in_channels (int): Number of input channels.
        filters (Tuple[int, ...]): Output channels for each conv layer.
        kernel_size (int): Default kernel size.
        stride (int): Default stride.
        dropout (float): Dropout rate.
        activation (str): Activation function ("relu" or "gelu").
        global_pool (str): Pooling type ("max" or "mean").
        conv_params_list (Optional[List[dict]]): Per-layer parameter overrides.
    """
    def __init__(
        self,
        in_channels: int,
        filters: Tuple[int, ...],
        *,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
        activation: str = "relu",
        global_pool: str = "max",
        conv_params_list: 'Optional[List[dict]]' = None,   # NEW
    ):
        super().__init__()
        act = {"relu": nn.ReLU(), "gelu": nn.GELU()}[activation]
        layers, prev = [], in_channels
        for idx, f in enumerate(filters):
            # layer-specific overrides if provided ------------------
            layer_cfg = (conv_params_list[idx]           # type: ignore[index]
                         if conv_params_list and idx < len(conv_params_list)
                         else {})
            k  = layer_cfg.get("kernel_size", kernel_size)
            st = layer_cfg.get("stride",      stride)
            dr = layer_cfg.get("dropout",     dropout)
            layers += [
                nn.Conv1d(prev, f, k, stride=st, padding=k // 2),
                act,
                nn.Dropout(dr),
            ]
            prev = f
        self.net = nn.Sequential(*layers)
        self.pool = global_pool
        self.output_dim = filters[-1]

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)
        return x.max(-1).values if self.pool == "max" else x.mean(-1)

class NucleotideClassifier(LightningModule):
    """
    PyTorch Lightning module for nucleotide sequence classification/regression.

    Args:
        task (str): "classification" or "regression".
        embedding_dim (int): Embedding dimension for nucleotides.
        conv_layers (tuple): Output channels for each conv layer.
        conv_params (dict): Parameters for convolutional layers.
        conv_params_list (Optional[List[dict]]): Per-layer conv parameter overrides.
        mlp_layers (tuple): Hidden sizes for MLP head.
        mlp_params (dict): Parameters for MLP head.
        lr (float): Learning rate.
        num_classes (int): Number of output classes.
    """
    def __init__(self, task="classification", embedding_dim=16,
                 conv_layers=(32, 64),
                 conv_params=None,
                 conv_params_list: 'Optional[List[dict]]' = None,   # NEW
                 mlp_layers=(128, 64), mlp_params=None,
                 lr=1e-3, num_classes=2):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        self.lr = lr
        self.nt_vocab = {"A": 0, "C": 1, "G": 2, "T": 3}

        # if per-layer list is given it overrides the single dict
        conv_params = {} if conv_params_list is not None else (conv_params or {})
        mlp_params = mlp_params or {}

        self.nt_embedding = nn.Embedding(4, embedding_dim)
        self.conv = ConvHead(
            embedding_dim,
            conv_layers,
            conv_params_list=conv_params_list,   # NEW
            **conv_params,
        )

        n_out = 1 if task == "regression" else num_classes
        self.mlp = MLPHead(self.conv.output_dim, mlp_layers, n_out=n_out, **mlp_params)

        self.criterion = nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()

    def forward(self, x):
        emb = self.nt_embedding(x)
        feats = self.conv(emb)
        return self.mlp(feats)

    def _step(self, batch):
        x, y = batch
        logits = self(x)
        if self.task == "regression":
            logits = logits.squeeze()
        loss = self.criterion(logits, y)
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)

    def encode_sequences(self, seqs):
        """
        Encode a list of nucleotide sequences as integer arrays.

        Args:
            seqs (List[str]): List of nucleotide strings.

        Returns:
            torch.Tensor: Encoded sequences (batch, seq_len).
        """
        max_len = max(len(s) for s in seqs)
        arr = np.full((len(seqs), max_len), 0)
        for i, s in enumerate(seqs):
            for j, base in enumerate(s):
                arr[i, j] = self.nt_vocab.get(base.upper(), 0)
        return torch.tensor(arr, dtype=torch.long)

class LossTrackerCallback(Callback):
    """
    PyTorch Lightning callback to track train and validation losses per epoch.
    """
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        if "train_loss" in trainer.callback_metrics:
            tr_loss = trainer.callback_metrics["train_loss"].item()
            self.train_losses.append(tr_loss)

    def on_validation_epoch_end(self, trainer, pl_module):
        if "val_loss" in trainer.callback_metrics:
            tr_loss = trainer.callback_metrics.get("train_loss", -1)
            val_loss = trainer.callback_metrics["val_loss"].item()
            self.val_losses.append(val_loss)
            print(f"Epoch  | train loss: {tr_loss:.4f} |  val loss: {val_loss:.4f}")

class NucleotideEstimator:
    """
    Wrapper for training, inference, and saliency analysis of a NucleotideClassifier.

    Args:
        classifier_model (NucleotideClassifier): The model to use.
        task (str): "classification" or "regression".
        verbose (bool): Whether to plot training curves.
    """
    def __init__(self, classifier_model: NucleotideClassifier, task="classification", verbose=True):
        self.model = classifier_model
        self.task = task
        self.verbose = verbose

    def fit(self, X, y, max_epochs=10, batch_size=32):
        """
        Fit the model to data.

        Args:
            X (List[str]): List of nucleotide sequences.
            y (array-like): Labels or regression targets.
            max_epochs (int): Number of training epochs.
            batch_size (int): Batch size.
        """
        X_tensor = self.model.encode_sequences(X)
        y_dtype = torch.float32 if self.task == "regression" else torch.long
        y_tensor = torch.tensor(y, dtype=y_dtype)

        train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.1, stratify=y, random_state=42)
        train_ds = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
        val_ds = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        loss_callback = LossTrackerCallback()
        progress_bar = TQDMProgressBar(refresh_rate=1)

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=False,
            callbacks=[loss_callback, progress_bar],
            enable_model_summary=False,
            enable_checkpointing=False
        )
        trainer.fit(self.model, train_loader, val_loader)

        if self.verbose:
            plt.figure(figsize=(8, 4))
            if loss_callback.train_losses:
                plt.plot(loss_callback.train_losses, label="Train Loss")
            if loss_callback.val_losses:
                plt.plot(loss_callback.val_losses, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.yscale("log")
            plt.title("Training Curve")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def predict(self, X, batch_size=64):
        """
        Predict class labels or regression outputs for input sequences.

        Args:
            X (List[str]): List of nucleotide sequences.
            batch_size (int): Batch size.

        Returns:
            np.ndarray: Predicted labels or values.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        X_tensor = self.model.encode_sequences(X).to(device)
        loader = DataLoader(X_tensor, batch_size=batch_size)
        preds = []
        with torch.no_grad():
            for batch in loader:
                out = self.model(batch)
                if self.task == "classification":
                    preds.append(out.argmax(dim=-1).cpu())
                else:
                    preds.append(out.squeeze().cpu())
        return torch.cat(preds).numpy()

    def predict_proba(self, X, batch_size=64):
        """
        Predict class probabilities for input sequences (classification only).

        Args:
            X (List[str]): List of nucleotide sequences.
            batch_size (int): Batch size.

        Returns:
            np.ndarray: Predicted class probabilities.

        Raises:
            ValueError: If called for regression task.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        X_tensor = self.model.encode_sequences(X).to(device)
        loader = DataLoader(X_tensor, batch_size=batch_size)
        probs = []
        with torch.no_grad():
            for batch in loader:
                out = self.model(batch)
                if self.task == "classification":
                    prob = F.softmax(out, dim=-1).cpu()
                    probs.append(prob)
                else:
                    raise ValueError("predict_proba only defined for classification.")
        return torch.cat(probs).numpy()

    def transform(self, X):
        """
        Compute saliency maps for input sequences using input gradients.

        Args:
            X (List[str]): List of nucleotide sequences.

        Returns:
            List[np.ndarray]: List of saliency arrays (per sequence).
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        saliencies = []
        for seq in X:
            inp = self.model.encode_sequences([seq]).to(device)
            emb = self.model.nt_embedding(inp)
            emb.requires_grad_()
            emb.retain_grad()
            feats = self.model.conv(emb)
            out = self.model.mlp(feats)
            target = out[0, out.argmax()] if self.task == "classification" else out.squeeze()
            target.backward()
            saliency = emb.grad.norm(dim=-1).squeeze().detach().cpu().numpy()
            saliencies.append(saliency)
        return saliencies

    def likelihood(self, X):
        """
        Compute a likelihood-like score from saliency maps for each sequence.

        Args:
            X (List[str]): List of nucleotide sequences.

        Returns:
            np.ndarray: Likelihood scores.
        """
        scores = []
        for sal in self.transform(X):
            score = -np.mean(np.log1p(1.0 / (sal + 1e-6)))
            scores.append(score)
        return np.array(scores)
    
    def cut_points(self, seq, method="kmeans", **kwargs):
        """
        Segment a sequence into regions using a chosen cut-point detection method.

        Args:
            seq (str): Nucleotide sequence.
            method (str): Cut-point method ("kmeans", "cusum", "gradient", etc).
            **kwargs: Additional method-specific parameters.

        Returns:
            list: Indices of cut points.
        """
        sal = self.transform([seq])[0]

        if method == "kmeans":
            kwargs.setdefault("k", 2)
            return cut_points_kmeans(sal, **kwargs)
        elif method == "cusum":
            kwargs.setdefault("penalty", 10)
            return cut_points_cusum(sal, **kwargs)
        elif method == "gradient":
            kwargs.setdefault("window", 20)
            kwargs.setdefault("threshold", 0.03)
            return cut_points_gradient(sal, **kwargs)
        elif method == "gradient_auto":
            kwargs.setdefault("target_segments", 12)
            kwargs.setdefault("window", 20)
            return cut_points_gradient_auto(sal, **kwargs)
        elif method == "threshold":
            kwargs.setdefault("percentile", 85)
            kwargs.setdefault("min_gap", 30)
            return cut_points_threshold(sal, **kwargs)
        elif method == "threshold_auto":
            kwargs.setdefault("target_segments", 12)
            kwargs.setdefault("min_gap", 30)
            return cut_points_threshold_auto(sal, **kwargs)
        else:
            raise ValueError(f"Unknown cut-point method: {method}")

    def get_sequence_embeddings(self, seqs: List[str], grad: bool = False) -> torch.Tensor:
        """
        Return feature embeddings before the final classification head.

        Useful for downstream analyses such as clustering or optimization.

        Args:
            seqs (List[str]): List of nucleotide sequences.
            grad (bool): If True, enables gradient computation.

        Returns:
            torch.Tensor: Feature embeddings.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        X_tensor = self.model.encode_sequences(seqs).to(device)
        with torch.set_grad_enabled(grad):
            emb = self.model.nt_embedding(X_tensor)
            feats = self.model.conv(emb)
        return feats.detach() if not grad else feats
