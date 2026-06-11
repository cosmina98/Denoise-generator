"""multi_scale_path_tree_convolution.py
=======================================
Complete implementation of MultiScalePathTreeConvolutionClassifier.

Dependencies:
  - torch
  - pytorch_lightning
  - networkx
  - numpy
  - scikit-learn
"""

from __future__ import annotations
from typing import List, Sequence, Optional, Dict, Tuple
from collections import defaultdict
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# optional progress bar
try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm not installed – we’ll fall back to prints
    tqdm = None

from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelEncoder

# ───────────────────────────── Utilities ──────────────────────────────
def extract_all_paths_bfs(G: nx.Graph) -> List[List[int]]:
    """
    Extract all shortest paths from each node to every other node using BFS.

    Parameters
    ----------
    G : nx.Graph
        Input graph.

    Returns
    -------
    List[List[int]]
        List of node index paths (each path is a list of node indices).
    """
    paths: List[List[int]] = []
    for root in G.nodes():
        paths.extend(nx.single_source_shortest_path(G, root).values())
    return paths

# ──────────────────── Path Feature Composer ───────────────────────────
class PathFeatureComposer:
    """
    Compose per-node features for a path: one-hot(label) | attribute | depth.

    Parameters
    ----------
    label_encoder : LabelEncoder
        Mapping from node label to integer index.
    attr_dim : int
        Dimension of the node attribute vector.

    Methods
    -------
    tensor_from_path(G, path)
        Returns a tensor of shape (L, F) for a path of length L.
    """
    def __init__(self, label_encoder: LabelEncoder, attr_dim: int):
        self.le = label_encoder
        self.attr_dim = attr_dim
        self._eye = np.eye(len(self.le.classes_), dtype=np.float32)

    @property
    def out_dim(self) -> int:
        return len(self.le.classes_) + self.attr_dim + 1

    def tensor_from_path(self, G: nx.Graph, path: List[int]) -> torch.Tensor:
        max_d = max(1, len(path) - 1)
        feats = []
        for depth, nid in enumerate(path):
            # -------------- label --------------
            label = G.nodes[nid].get("label", "UNK")
            if label not in self.le.classes_:
                label = "UNK"
            idx = self.le.transform([label])[0]
            label_vec = self._eye[idx]

            # -------------- attribute ----------
            raw_attr = G.nodes[nid].get("attribute", None)
            if raw_attr is None:
                attr_vec = np.ones(self.attr_dim, dtype=np.float32)
            else:
                attr_vec = np.asarray(raw_attr, dtype=np.float32)
                # pad / truncate to self.attr_dim
                if attr_vec.size < self.attr_dim:
                    attr_vec = np.pad(attr_vec, (0, self.attr_dim - attr_vec.size), constant_values=1.)
                elif attr_vec.size > self.attr_dim:
                    attr_vec = attr_vec[: self.attr_dim]

            # -------------- depth --------------
            depth_vec = np.array([depth / max_d], dtype=np.float32)
            feats.append(np.concatenate([label_vec, attr_vec, depth_vec]))
        return torch.tensor(np.stack(feats), dtype=torch.float32)

# ───────────────────── Multi-Scale CNN Encoder ───────────────────────-
class MultiScaleCNNPathEncoder(nn.Module):
    """
    Applies distinct 1-D CNNs to paths bucketed by length.

    Parameters
    ----------
    in_dim : int
        Input feature dimension for each node in the path.
    distance_cutoffs : List[int]
        List of path length cutoffs to define buckets.
    cnn_channels : Sequence[int]
        Number of output channels for each CNN layer.
    cnn_kernels : Sequence[int] or int
        Kernel size(s) for each CNN layer.
    cnn_strides : Sequence[int] or int
        Stride(s) for each CNN layer.
    residual_every : Optional[int]
        Insert identity (residual marker) every N layers.
    dropout : float
        Dropout probability after each CNN layer.
    batch_norm : bool
        If True, apply **GroupNorm(1, C)** (safe for tiny batches) after each
        Conv1d layer. Set to False for no normalisation.

    Methods
    -------
    forward(path_tensors)
        Apply the appropriate CNN to each path tensor.
    output_dim()
        Returns the output feature dimension after CNN.
    """
    def __init__(
        self,
        in_dim: int,
        distance_cutoffs: List[int] = (),
        cnn_channels: Sequence[int] = (64, 64),
        cnn_kernels: Sequence[int] | int = 3,
        cnn_strides: Sequence[int] | int = 1,
        residual_every: Optional[int] = None,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.cutoffs = sorted(distance_cutoffs)
        self.in_dim = in_dim
        self.residual_every = residual_every
        self.dropout = dropout
        self.batch_norm = batch_norm

        self.buckets = self._build_buckets()
        self.cnns = nn.ModuleList(
            [self._build_single_cnn(in_dim, cnn_channels, cnn_kernels, cnn_strides)
             for _ in self.buckets]
        )

    # bucket helpers ----------------------------------------------------
    def _build_buckets(self) -> List[Tuple[int, Optional[int]]]:
        if not self.cutoffs:
            return [(0, None)]
        starts = [0] + [c + 1 for c in self.cutoffs]
        ends = self.cutoffs + [None]
        return list(zip(starts, ends))

    def _bucket_index(self, length: int) -> int:
        for i, (lo, hi) in enumerate(self.buckets):
            if hi is None or (lo <= length <= hi):
                return i
        raise RuntimeError("length outside bucket ranges")

    # CNN builder -------------------------------------------------------
    def _build_single_cnn(
        self,
        in_dim: int,
        channels: Sequence[int],
        kernels: Sequence[int] | int,
        strides: Sequence[int] | int,
    ) -> nn.Module:
        kernels = [kernels] * len(channels) if isinstance(kernels, int) else list(kernels)
        strides = [strides] * len(channels) if isinstance(strides, int) else list(strides)
        layers: List[nn.Module] = []
        c_in = in_dim
        conv_idx = 0
        for c_out, k, s in zip(channels, kernels, strides):
            layers.append(nn.Conv1d(c_in, c_out, k, stride=s, padding=(k - 1) // 2))
            if self.batch_norm:
                # GroupNorm(1, C) is BN-free but works for batch/length = 1
                layers.append(nn.GroupNorm(1, c_out))
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            conv_idx += 1
            if self.residual_every and conv_idx % self.residual_every == 0:
                layers.append(nn.Identity())  # marker
            c_in = c_out
        return nn.Sequential(*layers)

    # forward -----------------------------------------------------------
    def forward(self, path_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        outs: List[torch.Tensor] = []
        for seq in path_tensors:
            idx = self._bucket_index(seq.size(0))
            cnn = self.cnns[idx]
            x = seq.unsqueeze(0).transpose(1, 2)        # (1, F, L)
            x = cnn(x).transpose(1, 2).squeeze(0)       # (L, E)
            outs.append(x)
        return outs

    def output_dim(self) -> int:
        dummy = torch.zeros(1, self.in_dim, 4)
        return self.cnns[0](dummy).shape[1]

# ───────────────────────── Graph Encoder ─────────────────────────────-
class GraphEncoder(nn.Module):
    """
    Encodes a NetworkX graph into a fixed-size vector via path-CNNs.

    Parameters
    ----------
    label_encoder : LabelEncoder
        Mapping from node label to integer index.
    attr_dim : int
        Dimension of the node attribute vector.
    distance_cutoffs : List[int]
        List of path length cutoffs for bucketing.
    cnn_channels, cnn_kernels, cnn_strides, residual_every, cnn_dropout, cnn_batch_norm
        Passed to MultiScaleCNNPathEncoder.

    Methods
    -------
    forward(G)
        Returns a fixed-size embedding vector for the input graph.
    embed_dim
        Output embedding dimension.
    """
    def __init__(
        self,
        label_encoder: LabelEncoder,
        attr_dim: int,
        distance_cutoffs: List[int],
        cnn_channels=(64, 64),
        cnn_kernels=3,
        cnn_strides=1,
        residual_every=None,
        cnn_dropout: float = 0.0,
        cnn_batch_norm: bool = False,
    ):
        super().__init__()
        self.feat = PathFeatureComposer(label_encoder, attr_dim)
        self.path_encoder = MultiScaleCNNPathEncoder(
            in_dim=self.feat.out_dim,
            distance_cutoffs=distance_cutoffs,
            cnn_channels=cnn_channels,
            cnn_kernels=cnn_kernels,
            cnn_strides=cnn_strides,
            residual_every=residual_every,
            dropout=cnn_dropout,
            batch_norm=cnn_batch_norm,
        )
        self.register_buffer('_dev', torch.tensor(0))

    @property
    def embed_dim(self) -> int:
        return self.path_encoder.output_dim()

    def forward(self, G: nx.Graph) -> torch.Tensor:
        device = self._dev.device
        paths = extract_all_paths_bfs(G)
        ptensors = [self.feat.tensor_from_path(G, p).to(device) for p in paths]
        p_embs = self.path_encoder(ptensors)             # list of (L, E)

        node_embs: Dict[int, List[torch.Tensor]] = defaultdict(list)
        for p, emb in zip(paths, p_embs):
            for depth, nid in enumerate(p):
                # keep gradient flow
                node_embs[nid].append(emb[depth])

        per_node = torch.stack([torch.stack(v).mean(0) for v in node_embs.values()])
        return per_node.mean(0)

    # --------------------------------------------------
    # NEW: per-node embeddings (no gradient required)
    # --------------------------------------------------
    @torch.no_grad()
    def node_embeddings(self, G: nx.Graph, as_numpy: bool = True) -> Dict[int, torch.Tensor]:
        """
        Return an embedding vector for every node in `G`.

        Parameters
        ----------
        G : nx.Graph
        as_numpy : bool (default True)
            If True, returns `np.ndarray`s.  Otherwise leaves tensors
            on *this module's* device.

        Returns
        -------
        Dict[int, np.ndarray | torch.Tensor]
            Mapping: node-id → embedding (shape = [E])
        """
        device = self._dev.device
        paths   = extract_all_paths_bfs(G)
        pt      = [self.feat.tensor_from_path(G, p).to(device) for p in paths]
        p_embs  = self.path_encoder(pt)                       # list[(L, E)]

        tmp: Dict[int, List[torch.Tensor]] = defaultdict(list)
        for path, emb in zip(paths, p_embs):
            for d, nid in enumerate(path):
                tmp[nid].append(emb[d])

        result = {nid: torch.stack(vecs).mean(0) for nid, vecs in tmp.items()}
        if as_numpy:
            result = {nid: v.cpu().numpy() for nid, v in result.items()}
        return result

# ───────────────────── Lightning Wrapper Module ─────────────────────--
class _LightningGraphClassifier(pl.LightningModule):
    """
    Internal Lightning module used by the sklearn wrapper.

    Parameters
    ----------
    encoder : GraphEncoder
        Graph encoder module.
    n_classes : int
        Number of output classes.
    mlp_hidden : Sequence[int]
        Hidden layer sizes for the MLP classifier.
    mlp_dropout : float
        Dropout probability for the MLP.
    mlp_batch_norm : bool
        Whether to use batch normalization in the MLP.
    lr : float
        Learning rate.

    Methods
    -------
    forward(graphs)
        Returns logits for a batch of graphs.
    training_step, validation_step, configure_optimizers
        Lightning training hooks.
    """
    def __init__(
        self,
        encoder: GraphEncoder,
        n_classes: int,
        mlp_hidden: Sequence[int],
        mlp_dropout: float,
        mlp_batch_norm: bool,
        lr: float,
        weight_decay: float = 0.0,   # NEW
    ):
        super().__init__()
        self.encoder = encoder
        dims = [encoder.embed_dim] + list(mlp_hidden) + [n_classes]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if mlp_batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())
                if mlp_dropout > 0:
                    layers.append(nn.Dropout(mlp_dropout))
        self.mlp = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay   # NEW
        # running loss buffers for plotting
        self.train_losses, self.val_losses = [], []

    def training_step(self, batch, batch_idx):
        bucket_batch, labels = batch
        logits = self(bucket_batch)
        loss = self.loss_fn(logits, labels.to(self.device))
        self.log('train_loss', loss,
                 prog_bar=False,
                 on_step=False, on_epoch=True,
                 batch_size=labels.size(0))
        return loss

    def on_train_epoch_end(self):
        # grab last logged train loss
        tl = self.trainer.callback_metrics.get("train_loss")
        if tl is not None:
            self.train_losses.append(tl.item())

    def validation_step(self, batch, batch_idx):
        bucket_batch, labels = batch
        logits = self(bucket_batch)
        val_loss = self.loss_fn(logits, labels)
        val_acc = (logits.argmax(1) == labels).float().mean()
        self.log('val_loss', val_loss,
                 prog_bar=False,
                 on_step=False, on_epoch=True,
                 batch_size=labels.size(0))
        self.log('val_acc', val_acc,
                 prog_bar=False,
                 on_step=False, on_epoch=True,
                 batch_size=labels.size(0))
        return val_loss

    def on_validation_epoch_end(self):
        vl = self.trainer.callback_metrics.get("val_loss")
        if vl is not None:
            self.val_losses.append(vl.item())

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay   # NEW
        )

    # ------------------------------------------------------------
    # forward: run each bucket once, pool per-graph, pass to MLP
    # ------------------------------------------------------------
    def forward(self, bucket_batch: Dict[int, Tuple[torch.Tensor, torch.Tensor]]
                ) -> torch.Tensor:
        """
        bucket_batch : { bucket_idx : (paths, graph_ids) }
           paths      : (N_paths, F, L_max)
           graph_ids  : (N_paths,)  ints in [0, B-1]
        returns logits: (B, n_classes)
        """
        device = self.device
        B = max(gid.max().item() for _, (_, gid) in bucket_batch.items()) + 1
        per_graph = defaultdict(list)      # gid → list[C]

        with torch.set_grad_enabled(self.training):
            for b, (x, gid) in bucket_batch.items():
                x   = x.to(device)                         # (N, F, L)
                cnn = self.encoder.path_encoder.cnns[b].to(device)
                out = cnn(x).mean(-1)                      # (N, C)
                for v, g in zip(out, gid.to(device)):
                    per_graph[int(g)].append(v)

        graph_embs = torch.stack(
            [torch.stack(per_graph[i]).mean(0) for i in range(B)]
        ).to(device)                                       # (B, C)
        return self.mlp(graph_embs)
# ───────────────────── Dataset Helper ─────────────────────────────--
class _GraphDataset(Dataset):
    """
    Pre-compute path feature tensors once per graph and bucket them
    according to `distance_cutoffs`. Each item is
        (bucket_dict, graph_label)
    where `bucket_dict[bucket_idx] = list[Tensor(L, F)]`.
    """
    def __init__(
        self,
        graphs: List[nx.Graph],
        labels: List[int],
        label_encoder: LabelEncoder,
        attr_dim: int,
        distance_cutoffs: List[int],
        verbose: bool = False,
    ):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.distance_cutoffs = sorted(distance_cutoffs)
        self.buckets = (
            [(0, None)]
            if not self.distance_cutoffs
            else list(
                zip([0] + [c + 1 for c in self.distance_cutoffs],
                    self.distance_cutoffs + [None])
            )
        )

        composer = PathFeatureComposer(label_encoder, attr_dim)

        # storage *before* the loop  (was missing)
        self.bucketed_paths: List[Dict[int, List[torch.Tensor]]] = []

        # choose iterator with optional progress bar
        itr = graphs
        if verbose and tqdm is not None:
            itr = tqdm(graphs, desc="Pre-computing path tensors",
                       unit="graph")
        elif verbose:
            print("Pre-computing path tensors ...")

        for G in itr:
            paths_dict: Dict[int, List[torch.Tensor]] = defaultdict(list)
            for p in extract_all_paths_bfs(G):
                b_idx = self._bucket_index(len(p))
                paths_dict[b_idx].append(composer.tensor_from_path(G, p))
            self.bucketed_paths.append(paths_dict)

        # final message if tqdm unavailable
        if verbose and tqdm is None:
            print(f"✓  Finished tensors for {len(self.bucketed_paths)} graphs.")

    def _bucket_index(self, length: int) -> int:
        for i, (lo, hi) in enumerate(self.buckets):
            if hi is None or (lo <= length <= hi):
                return i
        raise RuntimeError("length outside bucket ranges")

    def __len__(self):
        return len(self.bucketed_paths)

    def __getitem__(self, idx):
        return self.bucketed_paths[idx], self.labels[idx]

# ---------------------------------------------------------------
#  Batched bucket collate: pads per bucket and tracks graph IDs
# ---------------------------------------------------------------
def _bucket_collate(batch):
    """
    batch : List[Tuple[Dict[int, List[Tensor]],  label]]
    Returns
    -------
    batched : Dict[int, Tuple[ Tensor[N, F, Lmax], Tensor[N] ]]
        key  = bucket idx
        value= (padded_path_tensor, graph_idx_of_each_path)
    labels : Tensor[B]   (graph-level labels)
    """
    path_dicts, labels = zip(*batch)          # len=B graphs
    labels = torch.stack(labels)              # (B,)

    # 1. collect all (bucket, tensor, g_idx) triples
    from collections import defaultdict
    bucket_map = defaultdict(list)
    for g_idx, pd in enumerate(path_dicts):
        for b, plist in pd.items():
            for t in plist:
                bucket_map[b].append((t, g_idx))

    # 2. pad per bucket
    batched = {}
    for b, tpl_list in bucket_map.items():
        paths, g_ids = zip(*tpl_list)
        device = paths[0].device                # keep original device (cpu/cuda)
        F = paths[0].size(1)
        L_max = max(p.size(0) for p in paths)
        padded = torch.zeros(len(paths), F, L_max, device=device)
        for i, p in enumerate(paths):
            L = p.size(0)
            padded[i, :, :L] = p.T            # (F, L)
        batched[b] = (padded, torch.tensor(g_ids, dtype=torch.long, device=device))

    return batched, labels

# ─────────────── MultiScalePathTreeConvolutionClassifier ─────────────
class MultiScalePathTreeConvolutionClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    Scikit-learn style wrapper for the multi-scale path tree convolution graph classifier.

    Parameters
    ----------
    cnn_channels : Sequence[int]
        Output channels for each CNN layer.
    cnn_kernels : Sequence[int] or int
        Kernel size(s) for each CNN layer.
    cnn_strides : Sequence[int] or int
        Stride(s) for each CNN layer.
    residual_every : Optional[int]
        Insert identity (residual marker) every N layers.
    cnn_dropout : float
        Dropout probability after each CNN layer.
    cnn_batch_norm : bool
        Whether to use batch normalization after each CNN layer.
    distance_cutoffs : List[int] or Tuple[int, ...]
        Path length cutoffs for bucketing.
    mlp_hidden : Sequence[int]
        Hidden layer sizes for the MLP classifier.
    mlp_dropout : float
        Dropout probability for the MLP.
    mlp_batch_norm : bool
        Whether to use batch normalization in the MLP.
    batch_size : int
        Training batch size.
    max_epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    accelerator : Optional[str]
        Accelerator for PyTorch Lightning Trainer ("cpu", "cuda", etc).

    Methods
    -------
    fit(X, y)
        Train the classifier on a list of graphs X and labels y.
    predict(X)
        Predict class labels for a list of graphs X.
    predict_proba(X)
        Predict class probabilities for a list of graphs X.
    transform(X)
        Return graph embeddings for a list of graphs X.
    """
    def __init__(
        self,
        # CNN params
        cnn_channels=(64, 64),
        cnn_kernels=3,
        cnn_strides=1,
        residual_every=None,
        cnn_dropout: float = 0.0,
        cnn_batch_norm: bool = False,
        distance_cutoffs: List[int] | Tuple[int, ...] = (),
        # MLP params
        mlp_hidden=(128,),
        mlp_dropout: float = 0.0,
        mlp_batch_norm: bool = False,
        # Training params
        batch_size: int = 16,
        max_epochs: int = 20,
        lr: float = 1e-3,
        weight_decay: float = 0.0,   # NEW
        accelerator: Optional[str] = None,
        verbose: bool = False,
    ):
        self.cnn_channels = cnn_channels
        self.cnn_kernels = cnn_kernels
        self.cnn_strides = cnn_strides
        self.residual_every = residual_every
        self.cnn_dropout = cnn_dropout
        self.cnn_batch_norm = cnn_batch_norm
        self.distance_cutoffs = list(distance_cutoffs)
        self.mlp_hidden = mlp_hidden
        self.mlp_dropout = mlp_dropout
        self.mlp_batch_norm = mlp_batch_norm
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = weight_decay   # NEW
        self.accelerator = accelerator or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        # helper for DataLoaders
        self._pin_memory = (self.accelerator == "cuda")

    # ---------------------------- fit ---------------------------------
    def fit(self, X: List[nx.Graph], y: Sequence[int]):
        # ───── LabelEncoder with fallback "UNK" ─────
        node_labels = [G.nodes[n].get("label", "UNK") for G in X for n in G.nodes]
        # include UNK before fitting so classes_ stays sorted
        if "UNK" not in node_labels:
            node_labels.append("UNK")
        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(node_labels)

        # ───── attribute dimension (robust) ─────
        try:
            sample_attr = next(attr for _, attr in
                               nx.get_node_attributes(X[0], "attribute").items())
            self.attr_dim_ = int(np.asarray(sample_attr).size)
        except StopIteration:
            self.attr_dim_ = 1
        self.n_classes_ = len(set(y))

        encoder = GraphEncoder(
            label_encoder=self.label_encoder_,
            attr_dim=self.attr_dim_,
            distance_cutoffs=self.distance_cutoffs,
            cnn_channels=self.cnn_channels,
            cnn_kernels=self.cnn_kernels,
            cnn_strides=self.cnn_strides,
            residual_every=self.residual_every,
            cnn_dropout=self.cnn_dropout,
            cnn_batch_norm=self.cnn_batch_norm,
        )

        lit_model = _LightningGraphClassifier(
            encoder=encoder,
            n_classes=self.n_classes_,
            mlp_hidden=self.mlp_hidden,
            mlp_dropout=self.mlp_dropout,
            mlp_batch_norm=self.mlp_batch_norm,
            lr=self.lr,
            weight_decay=self.weight_decay,   # NEW
        )

        dataset = _GraphDataset(
            graphs=X,
            labels=list(y),
            label_encoder=self.label_encoder_,
            attr_dim=self.attr_dim_,
            distance_cutoffs=self.distance_cutoffs,
            verbose=self.verbose,
        )
        # 90/10 split
        val_size = max(1, int(0.1 * len(dataset)))
        train_size = len(dataset) - val_size
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=_bucket_collate,
            pin_memory=self._pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_bucket_collate,
            pin_memory=self._pin_memory,
        )

        if self.verbose:
            print(f"Dataset built: {len(dataset)} graphs "
                  f"(train {train_size} | val {val_size})")

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=self.verbose,
        )
        trainer.fit(lit_model, train_loader, val_loader)
        lit_model.eval()
        self._model_ = lit_model

        # ─── Plot losses ───────────────────────────────────────────
        if self.verbose and lit_model.train_losses:
            plt.figure(figsize=(6, 4))
            plt.plot(lit_model.train_losses, label="train")
            if lit_model.val_losses:
                plt.plot(lit_model.val_losses, label="val")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title("Training curve")
            plt.legend()
            plt.tight_layout()
            plt.show()
        return self

    # ------------------------- predict/proba --------------------------
    def _check_fitted(self):
        check_is_fitted(self, "_model_")

    def _make_loader(self, X, batch_size=32):
        ds = _GraphDataset(
            graphs=X,
            labels=[0]*len(X),          # dummy labels
            label_encoder=self.label_encoder_,
            attr_dim=self.attr_dim_,
            distance_cutoffs=self.distance_cutoffs,
            verbose=False,              # no progress bars during inference
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            collate_fn=_bucket_collate,
            pin_memory=self._pin_memory,
        )

    @torch.no_grad()
    def predict(self, X: List[nx.Graph]) -> np.ndarray:
        self._check_fitted()
        loader = self._make_loader(X)
        preds = []
        for batch in loader:
            bucket_batch, _ = batch
            preds.append(self._model_(bucket_batch))            # stay on model device
        logits = torch.cat(preds, 0)
        return logits.argmax(1).cpu().numpy()                   # single CPU hop

    @torch.no_grad()
    def predict_proba(self, X: List[nx.Graph]) -> np.ndarray:
        self._check_fitted()
        loader = self._make_loader(X)
        preds = []
        for batch in loader:
            bucket_batch, _ = batch
            preds.append(self._model_(bucket_batch))
        logits = torch.cat(preds, 0)
        return torch.softmax(logits, dim=1).cpu().numpy()

    @torch.no_grad()
    def transform(self, X: List[nx.Graph]) -> np.ndarray:
        self._check_fitted()
        loader = self._make_loader(X)
        embs = []
        for batch in loader:
            bucket_batch, _ = batch
            # replicate forward logic, but return graph embeddings before MLP
            device = self._model_.device
            B = max(gid.max().item() for _, (_, gid) in bucket_batch.items()) + 1
            per_graph_accum = defaultdict(list)
            for b, (x, g_ids) in bucket_batch.items():
                x = x.to(device)
                cnn = self._model_.encoder.path_encoder.cnns[b]
                out = cnn(x).mean(-1)
                for vec, gid in zip(out, g_ids.to(device)):
                    per_graph_accum[int(gid)].append(vec)
            graph_embs = torch.stack([
                torch.stack(per_graph_accum[i]).mean(0) for i in range(B)
            ])                                               # (B, C) – on model device
            embs.append(graph_embs.cpu().numpy())            # single CPU hop
        return np.concatenate(embs, axis=0)

    # --------------------------------------------------
    # NEW: attach node embeddings back to NetworkX graphs
    # --------------------------------------------------
    def annotate(self, graphs: List[nx.Graph], in_place: bool = False
                 ) -> List[nx.Graph]:
        """
        Compute a vector for every node and write it to the node
        attribute ``'attribute'``.  Returns the (possibly-copied) list.

        Parameters
        ----------
        graphs : List[nx.Graph]
        in_place : bool (default False)
            • True  – modify the original graph objects.  
            • False – work on ``G.copy()`` so originals stay unchanged.

        Returns
        -------
        List[nx.Graph]  (same length/order as input)
        """
        self._check_fitted()
        enc   = self._model_.encoder        # already on correct device
        out   = []

        for G in graphs:
            G2            = G if in_place else G.copy()
            node2vec      = enc.node_embeddings(G2, as_numpy=True)
            for nid, vec in node2vec.items():
                G2.nodes[nid]["attribute"] = vec          # overwrite / create
            out.append(G2)

        return out