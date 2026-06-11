"""
### Summary of the **Bi-Rank Low-Rank Graph-Kernel-MLP** Method

*(two independent ranks $r_A$ and $r_F$; orthogonality regularised; sklearn-friendly)*

1. **Input Representation**

   * **Node Embeddings (F)** `node_vectorizer` → $n × f$ matrix.
   * **Node Attributes (A)** `attrib_extractor` → $n × p$ matrix.

2. **Low-Rank Adaptive Kernel (two factors)**

   Instead of one full $f × p$ kernel we learn **two** skinny matrices

   $$
     V\in\mathbb R^{p\times r_A},\qquad U\in\mathbb R^{f\times r_F},
   $$

   where $r_A\ll p$ and $r_F\ll f$.
   Parameter count drops from $fp$ to $r_A p + r_F f$.

3. **Graph-Level Descriptor**

   ```
   Z = A · V          #  n × r_A    (attribute projection)
   G = Zᵀ · F · U     #  r_A × r_F  (cross-covariance in compressed spaces)
   ```

   *No big matrices are ever materialised.*
   **G** is then flattened to length $r_A · r_F$.

4. **Orthogonality Regulariser**

   $$
     \mathcal L_\text{ortho}=λ\bigl(\|V^{\top}V-I\|_F^{2}
                               +\|U^{\top}U-I\|_F^{2}\bigr)
   $$

   encourages the columns of **V** and **U** to form near-orthonormal bases.

5. **MLP Classification Head**

   * Flattened $G$ → **LayerNorm** → MLP (`hidden_dims`, `dropout`) → logits.

6. **Training Framework (PyTorch Lightning)**

   * **Data** - custom `LightningDataModule`; fresh **90 / 10 train-val split**, batch = 1.
   * **Loss** - cross-entropy + orthogonality penalty.
   * **Verbose** - prints one line per epoch: `Epoch k | train_loss=… | val_loss=…`.
   * **EarlyStopping** - pass Lightning’s callback via `trainer_kwargs` if desired.

7. **Embeddings API**

   `transform()` returns the **raw flattened $r_A×r_F$ matrix** (before LayerNorm/MLP) as a task-adapted graph embedding.

8. **Hyper-parameters**

   | name                 | role                            |
   | -------------------- | ------------------------------- |
   | `rank_A`, `rank_F`   | compression ranks for A and F   |
   | `hidden_dims`        | MLP layer sizes (int or list)   |
   | `dropout`            | dropout probability inside MLP  |
   | `lr`, `weight_decay` | Adam optimiser settings         |
   | `ortho_lambda`       | weight of orthogonality penalty |
   | `verbose`            | epoch-level console logging     |

9. **Practical Notes**

   * Keep `num_workers=0` on macOS to avoid multiprocessing issues.
   * All tensors cast to `float32` for safe `torch.matmul`.

---

The result is a **memory-efficient, end-to-end-trainable** graph classifier that jointly learns two low-rank, near-orthogonal projections (**V**, **U**) 
and a downstream MLP head—exposed through the familiar scikit-learn interface (`fit`, `predict`, `predict_proba`, `transform`).

"""

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
import numpy as np
import networkx as nx
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from typing import Sequence, Callable, List, Optional, Union
from collections.abc import Iterable

# ---------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------
class GraphPairDataset(Dataset):
    """Stores (A, F, y) for each graph in torch.float32."""
    def __init__(self,
                 graphs: Sequence[nx.Graph],
                 labels: Sequence[int],
                 node_vectorizer: Callable[[Sequence[nx.Graph]], List[np.ndarray]],
                 attrib_extractor:  Callable[[nx.Graph], np.ndarray]):
        self.F = [f.astype(np.float32) for f in node_vectorizer(graphs)]
        self.A = [attrib_extractor(g).astype(np.float32) for g in graphs]
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.A)

    def __getitem__(self, idx):
        return {"A": torch.from_numpy(self.A[idx]),
                "F": torch.from_numpy(self.F[idx]),
                "y": self.y[idx]}

class SplitGraphDataModule(LightningDataModule):
    """90 / 10 split performed in .setup()."""
    def __init__(self,
                 graphs, labels,
                 node_vec, attr_extr,
                 batch_size: int = 1,
                 num_workers: int = 0,
                 val_frac: float = 0.1):
        super().__init__()
        self.graphs, self.labels = graphs, labels
        self.node_vec, self.attr_extr = node_vec, attr_extr
        self.batch_size, self.num_workers, self.val_frac = batch_size, num_workers, val_frac

    def setup(self, stage=None):
        full = GraphPairDataset(self.graphs, self.labels,
                                self.node_vec, self.attr_extr)
        n_val = max(1, int(len(full) * self.val_frac))
        n_train = len(full) - n_val
        self.train_ds, self.val_ds = random_split(
            full, [n_train, n_val],
            generator=torch.Generator().manual_seed(0))

    def _loader(self, ds, shuffle):
        return DataLoader(ds,
                          batch_size=self.batch_size,
                          shuffle=shuffle,
                          num_workers=self.num_workers,
                          collate_fn=lambda b: b[0])

    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds, False)

# ---------------------------------------------------------------------
# 2. MODEL (two-rank version)
# ---------------------------------------------------------------------
class LowRankKernelMLP(LightningModule):
    """
    Builds a graph descriptor G ∈ ℝ^{r_A × r_F} and feeds it to an MLP.

        Z = A  V             (n × r_A)
        G = Zᵀ F U           (r_A × r_F)

    Loss = CrossEntropy + λ (‖VᵀV - I‖² + ‖UᵀU - I‖²)
    """
    def __init__(self,
                 p: int, f: int, n_classes: int,
                 rank_A: int = 16,
                 rank_F: int = 16,
                 hidden_dims: Union[int, Sequence[int]] = (128, 64),
                 dropout: float = 0.5,
                 lr: float = 1e-3,
                 wd: float = 0.0,
                 ortho_lambda: float = 1e-2,
                 verbose: bool = False):
        super().__init__()

        # ── ensure hidden_dims iterable ───────────────────────────────
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        elif not isinstance(hidden_dims, Iterable):
            raise TypeError("hidden_dims must be int or iterable of ints")

        self.save_hyperparameters()

        # ── learnable low-rank factors ────────────────────────────────
        self.V = nn.Parameter(torch.empty(p, rank_A))
        self.U = nn.Parameter(torch.empty(f, rank_F))
        nn.init.xavier_uniform_(self.V)
        nn.init.xavier_uniform_(self.U)

        # ── LayerNorm + MLP head ──────────────────────────────────────
        flat_dim = rank_A * rank_F
        self.ln_input = nn.LayerNorm(flat_dim)

        layers: List[nn.Module] = []
        in_dim = flat_dim
        for h in hidden_dims:
            layers += [nn.LayerNorm(in_dim),
                       nn.Linear(in_dim, h),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.mlp = nn.Sequential(*layers)

        # ── misc ──────────────────────────────────────────────────────
        self.loss_fn      = nn.CrossEntropyLoss()
        self.ortho_lambda = ortho_lambda
        self.lr, self.wd  = lr, wd
        self.verbose      = verbose

    # ------------------------- helpers -----------------------------------
    @staticmethod
    def _column_ortho_penalty(M: torch.Tensor) -> torch.Tensor:
        r = M.shape[1]
        return ((M.T @ M - torch.eye(r, dtype=M.dtype, device=M.device))**2).sum()

    def _G(self, A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        """Compute r_A × r_F graph matrix."""
        if F.ndim == 1:
            F = F.unsqueeze(0)
        Z = A @ self.V           # (n, r_A)
        return (Z.T @ F) @ self.U  # (r_A, r_F)

    # --------------------- Lightning interface ---------------------------
    def forward(self, A: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        G_flat = self._G(A, F).flatten().unsqueeze(0)  # (1, r_A·r_F)
        x = self.ln_input(G_flat)
        return self.mlp(x).squeeze(0)                  # (n_classes,)

    def _shared_step(self, batch):
        logits = self(batch["A"], batch["F"]).unsqueeze(0)
        ce_loss = self.loss_fn(logits, batch["y"].unsqueeze(0))
        ortho = (self._column_ortho_penalty(self.U) + self._column_ortho_penalty(self.V)) * self.ortho_lambda
        return ce_loss + ortho, ce_loss, ortho

    def training_step(self, batch, _):
        total, ce, ortho = self._shared_step(batch)
        self.log_dict({"train_loss": total,
                       "train_ce": ce,
                       "train_ortho": ortho},
                      on_epoch=True, batch_size=1)
        return total

    def validation_step(self, batch, _):
        total, ce, ortho = self._shared_step(batch)
        self.log_dict({"val_loss": total,
                       "val_ce": ce,
                       "val_ortho": ortho},
                      on_epoch=True, batch_size=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

    def on_validation_epoch_end(self):
        if not self.verbose or self.trainer.sanity_checking:
            return

        tr = self.trainer.callback_metrics.get("train_loss")
        va = self.trainer.callback_metrics.get("val_loss")

        # Save to parent (GraphKernelMLPClassifier) for plotting
        if hasattr(self, "classifier_ref"):
            if tr is not None:
                self.classifier_ref._train_losses.append(float(tr))
            if va is not None:
                self.classifier_ref._val_losses.append(float(va))

        if tr is not None and va is not None:
            print(f"Epoch {self.current_epoch:03d} | "
                f"train_loss={float(tr):.4f} | val_loss={float(va):.4f}")

# ---------------------------------------------------------------------
# 3. SKLEARN-COMPATIBLE WRAPPER
# ---------------------------------------------------------------------
class GraphKernelMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    sklearn-style wrapper around LowRankKernelMLP.
    """
    def __init__(self,
                 node_vectorizer: Callable[[Sequence[nx.Graph]], List[np.ndarray]],
                 attribute_extractor: Callable[[nx.Graph], np.ndarray],
                 rank_A: int = 16,
                 rank_F: int = 16,
                 hidden_dims: Union[int, Sequence[int]] = (128, ),
                 dropout: float = 0.5,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 ortho_lambda: float = 1e-2,
                 trainer_kwargs: Optional[dict] = None,
                 accelerator: Optional[str] = None,
                 devices: Optional[int] = None,
                 verbose: bool = False):
        self.node_vectorizer = node_vectorizer
        self.attribute_extractor = attribute_extractor
        self.rank_A, self.rank_F = rank_A, rank_F
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.ortho_lambda = ortho_lambda
        self.trainer_kwargs = trainer_kwargs or {}
        self.accelerator, self.devices = accelerator, devices
        self.verbose = verbose
        self._train_losses = []
        self._val_losses = []


    # ------------------------------ API ----------------------------------
    def fit(self, X: Sequence[nx.Graph], y: Sequence[int]):
        self.n_classes_ = len(set(y))
        # dims from the first graph
        f = (1 if self.node_vectorizer([X[0]])[0].ndim == 1
             else self.node_vectorizer([X[0]])[0].shape[1])
        p = self.attribute_extractor(X[0]).shape[1]

        dm = SplitGraphDataModule(
            X, y,
            node_vec=self.node_vectorizer,
            attr_extr=self.attribute_extractor,
            batch_size=1, num_workers=0, val_frac=0.1)

        self.model_ = LowRankKernelMLP(
            p=p, f=f, n_classes=self.n_classes_,
            rank_A=self.rank_A, rank_F=self.rank_F,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            lr=self.lr, wd=self.weight_decay,
            ortho_lambda=self.ortho_lambda,
            verbose=self.verbose)

        # handle EarlyStopping resets (optional)
        tkw = dict(self.trainer_kwargs)
        callbacks = tkw.pop("callbacks", None)
        new_cbs = []
        if callbacks:
            for cb in callbacks:
                if isinstance(cb, EarlyStopping):
                    new_cbs.append(EarlyStopping(monitor=cb.monitor,
                                                 min_delta=cb.min_delta,
                                                 patience=cb.patience,
                                                 mode=cb.mode,
                                                 strict=cb.strict,
                                                 verbose=cb.verbose))
                else:
                    new_cbs.append(cb)

        trainer_args = dict(enable_checkpointing=False, **tkw)
        if new_cbs:
            trainer_args["callbacks"] = new_cbs
        if self.accelerator: trainer_args["accelerator"] = self.accelerator
        if self.devices:     trainer_args["devices"] = self.devices

        self.model_.classifier_ref = self  # for logging back into classifier
        trainer = Trainer(**trainer_args)
        trainer.fit(self.model_, dm)

        # ── Plot train and val loss curves if verbose ───────────────────
        
        if self.verbose and self._train_losses and self._val_losses:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 4))
            plt.plot(self._train_losses, label="Train Loss")
            plt.plot(self._val_losses, label="Val Loss")
            plt.yscale("log")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return self

    def _logits(self, graphs):
        check_is_fitted(self, "model_")
        self.model_.eval()
        outs = []
        for g in graphs:
            A = torch.from_numpy(self.attribute_extractor(g).astype(np.float32))
            F = torch.from_numpy(self.node_vectorizer([g])[0].astype(np.float32))
            with torch.no_grad():
                outs.append(self.model_(A, F).unsqueeze(0))
        return torch.vstack(outs)

    def predict(self, X):       return self._logits(X).argmax(dim=1).numpy()
    def predict_proba(self, X): return torch.softmax(self._logits(X), dim=1).numpy()

    # ---------- graph embeddings: raw (r_A × r_F) vector -----------------
    def transform(self, X: Sequence[nx.Graph]) -> np.ndarray:
        """Return vectorised G (size r_A·r_F) **before** LayerNorm & MLP."""
        check_is_fitted(self, "model_")
        vecs = []
        for g in X:
            A = torch.from_numpy(self.attribute_extractor(g).astype(np.float32))
            F = torch.from_numpy(self.node_vectorizer([g])[0].astype(np.float32))
            with torch.no_grad():
                vecs.append(self.model_._G(A, F).flatten())
        return torch.stack(vecs).numpy()

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    


    '''
    USAGE:
    %%time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import StochasticWeightAveraging

# 1) Early stopping
early_stop_cb = pl.callbacks.early_stopping.EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="min",
    verbose=True
)

# 2) Stochastic Weight Averaging
swa_cb = StochasticWeightAveraging(
    swa_lrs=1e-2,
    swa_epoch_start=0.5
)

# ---- build & train classifier ------------------------------
clf = GraphKernelMLPClassifier(
    node_vectorizer=node_vectorizer,
    attribute_extractor=attribute_extractor,
    r_A=10,
    r_F=10,
    ortho_lambda=1e-2,
    hidden_dims=(128,),
    dropout=0.5,
    lr=1e-4,
    weight_decay=1e-5,
    trainer_kwargs={
        "max_epochs": 100,
        "callbacks": [early_stop_cb, swa_cb],
        "enable_progress_bar": False,
        "logger": False,
    },
    verbose=True
)
clf.fit(g_train, y_train)

# ---- evaluation --------------------------------------------
y_pred       = clf.predict(g_test)
y_pred_proba = clf.predict_proba(g_test)[:, 1]    

acc  = accuracy_score(y_test, y_pred)
auc  = roc_auc_score(y_test, y_pred_proba)
err  = (y_pred != y_test).sum()                    
total= len(y_test)

print(f"Test accuracy  : {acc:.3f}")
print(f"ROC-AUC        : {auc:.3f}")
print(f"Errors         : {err} / {total}")         
    
    '''