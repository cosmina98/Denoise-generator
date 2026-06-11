from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, GINEConv, global_add_pool
from torch_geometric.utils.convert import from_networkx


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def _to_float_feature(x, n: int) -> torch.Tensor:
    arr = np.asarray(x, dtype=float)
    arr = _ensure_2d(arr)
    if arr.shape[0] != n:
        arr = np.resize(arr, (n, arr.shape[1]))
    return torch.tensor(arr, dtype=torch.float32)


def _normalize_graph_attributes_for_pyg(graph):
    """
    Ensure homogeneous node/edge attribute keys for torch_geometric.from_networkx.
    Some generators attach extra hash/debug attrs only to a subset of nodes.
    """
    g = graph.copy()
    # Graph-level metadata (e.g., provenance "history") may appear only on some
    # graphs and breaks PyG DataLoader collation with KeyError on missing keys.
    # Keep conversion strictly node/edge-feature based.
    g.graph.clear()

    for _, node_data in g.nodes(data=True):
        node_feature = None
        for key in ("attr", "label", "display_label", "true_label"):
            if key in node_data:
                node_feature = node_data[key]
                break
        if node_feature is None:
            node_feature = 1.0
        node_data.clear()
        node_data["attr"] = node_feature

    for _, _, edge_data in g.edges(data=True):
        edge_feature = edge_data.get("edge_attr", edge_data.get("edge_label", 1.0))
        edge_data.clear()
        edge_data["edge_attr"] = edge_feature

    return g


def nx_graph_to_pyg(graph, y=None) -> Data:
    g = _normalize_graph_attributes_for_pyg(graph)
    d = from_networkx(g)

    # Node features: prefer attr -> label -> ones
    x = None
    if hasattr(d, "attr"):
        x = d.attr
    elif hasattr(d, "label"):
        x = d.label
    elif hasattr(d, "display_label"):
        x = d.display_label
    if x is None:
        x = torch.ones((g.number_of_nodes(), 1), dtype=torch.float32)
    else:
        x = _to_float_feature(x, g.number_of_nodes())
    d.x = x

    # Edge features: prefer edge_attr -> edge_label -> ones
    e = None
    if hasattr(d, "edge_attr") and d.edge_attr is not None:
        e = d.edge_attr
    elif hasattr(d, "edge_label"):
        e = d.edge_label
    if e is None:
        e = torch.ones((d.edge_index.shape[1], 1), dtype=torch.float32)
    else:
        e = _to_float_feature(e, d.edge_index.shape[1])
    d.edge_attr = e

    if y is None:
        d.y = None
    else:
        d.y = torch.tensor(int(y), dtype=torch.long)
    return d


def get_pyg_graphs_from_nx(graphs: Sequence, targets: Optional[Sequence] = None) -> List[Data]:
    if targets is None:
        return [nx_graph_to_pyg(g) for g in graphs]
    return [nx_graph_to_pyg(g, y) for g, y in zip(graphs, targets)]


class MLP(nn.Module):
    def __init__(self, num_layers: int, in_channels: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.linears = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        if num_layers == 1:
            self.linears.append(nn.Linear(in_channels, output_dim))
        else:
            self.linears.append(nn.Linear(in_channels, hidden_dim))
            for _ in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            for _ in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
        h = x
        for i in range(self.num_layers - 1):
            h = F.relu(self.batch_norms[i](self.linears[i](h)))
        return self.linears[-1](h)

    # Keep compatibility with PyG's in-channel inference helper.
    def __len__(self):
        return self.num_layers

    def __getitem__(self, item):
        return self.linears[item]


def make_gin_conv(input_dim: int, out_dim: int, edge_dim: Optional[int] = None):
    # Use nn.Sequential to keep PyG channel-inference robust across versions.
    mlp = nn.Sequential(
        nn.Linear(input_dim, out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim),
    )
    if edge_dim is None:
        return GINConv(mlp)
    # Some PyG versions still fail channel inference in GINEConv; fallback keeps flow alive.
    try:
        return GINEConv(mlp, edge_dim=edge_dim)
    except Exception:
        return GINConv(mlp)


class GConv(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 3, edge_dim: Optional[int] = None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(make_gin_conv(in_dim, hidden_dim, edge_dim=edge_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        proj_dim = hidden_dim * num_layers
        self.project = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            if edge_attr is None:
                z = conv(z, edge_index)
            else:
                z = conv(z, edge_index, edge_attr)
            z = bn(F.relu(z))
            zs.append(z)
        gs = [global_add_pool(zi, batch) for zi in zs]
        z_cat, g_cat = [torch.cat(xx, dim=1) for xx in [zs, gs]]
        return z_cat, g_cat

    def get_graph_embed(self, x, edge_index, batch, edge_attr=None):
        self.eval()
        with torch.no_grad():
            _, g = self.forward(x, edge_index, batch, edge_attr)
            return g


@dataclass
class NnVectorizerConfig:
    hidden_dim: int = 32
    num_layers: int = 3
    batch_size: int = 64
    lr: float = 1e-2
    epochs: int = 200
    device: str = "cpu"
    model_dir: str = "artifacts/nn_embedding_models"
    allow_random_fallback: bool = True
    graphcl_temperature: float = 0.2
    graphcl_edge_drop: float = 0.2
    graphcl_feature_mask: float = 0.2


class NnGraphVectorizer:
    """
    Unified vectorizer for graph embeddings:
    - feature_extractor='infograph'
    - feature_extractor='graphcl'
    - feature_extractor='gin-random'

    Notes:
    - This implementation is intentionally lightweight and notebook-safe.
    - If training dependencies for contrastive methods are missing, it can fallback to random GIN embeddings.
    """

    def __init__(
        self,
        feature_extractor: str = "infograph",
        dataset_name: str = "dataset",
        mode: str = "test",
        config: Optional[NnVectorizerConfig] = None,
        parallel: bool = False,
    ):
        self.feature_extractor = feature_extractor
        self.dataset_name = dataset_name
        self.mode = mode
        self.parallel = parallel
        self.config = config or NnVectorizerConfig()
        self.device = torch.device(self.config.device)
        self.model = None

    def _model_path(self) -> Path:
        d = Path(self.config.model_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{self.feature_extractor}_{self.dataset_name}.pt"

    def _init_model_from_data(self, pyg_graphs: Sequence[Data]) -> GConv:
        input_dim = int(pyg_graphs[0].x.shape[1]) if pyg_graphs and pyg_graphs[0].x is not None else 1
        edge_dim = int(pyg_graphs[0].edge_attr.shape[1]) if pyg_graphs and pyg_graphs[0].edge_attr is not None else None
        model = GConv(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            edge_dim=edge_dim,
        )
        return model.to(self.device)

    def _train_gin_random(self, pyg_graphs: Sequence[Data], targets: Optional[Sequence] = None) -> GConv:
        # Lightweight supervised training if labels exist; otherwise returns random initialized model.
        model = self._init_model_from_data(pyg_graphs)
        if targets is None or len(targets) == 0:
            return model

        loader = DataLoader(pyg_graphs, batch_size=self.config.batch_size, shuffle=True)
        opt = Adam(model.parameters(), lr=self.config.lr)
        clf = nn.Linear(model.hidden_dim * model.num_layers, int(len(set(targets)))).to(self.device)
        loss_fn = nn.CrossEntropyLoss()
        params = list(model.parameters()) + list(clf.parameters())
        opt = Adam(params, lr=self.config.lr)

        model.train()
        for _ in range(self.config.epochs):
            for batch in loader:
                batch = batch.to(self.device)
                opt.zero_grad()
                _, g = model(batch.x.float(), batch.edge_index, batch.batch, batch.edge_attr.float())
                y = batch.y.long()
                logits = clf(g)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()
        return model

    def _augment_batch(self, batch):
        x = batch.x.float().clone()
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr.float() if batch.edge_attr is not None else None

        if self.config.graphcl_feature_mask > 0:
            feat_mask = torch.rand_like(x) < float(self.config.graphcl_feature_mask)
            x = x.masked_fill(feat_mask, 0.0)

        if self.config.graphcl_edge_drop > 0 and edge_index.size(1) > 1:
            keep = torch.rand(edge_index.size(1), device=edge_index.device) >= float(self.config.graphcl_edge_drop)
            if not bool(keep.any()):
                keep[torch.randint(edge_index.size(1), (1,), device=edge_index.device)] = True
            edge_index = edge_index[:, keep]
            if edge_attr is not None:
                edge_attr = edge_attr[keep]

        return x, edge_index, edge_attr

    def _train_graphcl(self, pyg_graphs: Sequence[Data]) -> GConv:
        model = self._init_model_from_data(pyg_graphs)
        loader = DataLoader(pyg_graphs, batch_size=self.config.batch_size, shuffle=True)
        graph_dim = model.hidden_dim * model.num_layers
        projector = nn.Sequential(
            nn.Linear(graph_dim, graph_dim),
            nn.ReLU(inplace=True),
            nn.Linear(graph_dim, graph_dim),
        ).to(self.device)

        opt = Adam(list(model.parameters()) + list(projector.parameters()), lr=self.config.lr)
        tau = float(self.config.graphcl_temperature)

        model.train()
        projector.train()
        for _ in range(self.config.epochs):
            for batch in loader:
                batch = batch.to(self.device)
                if batch.num_graphs <= 1:
                    continue

                x1, ei1, ea1 = self._augment_batch(batch)
                x2, ei2, ea2 = self._augment_batch(batch)

                _, g1 = model(x1, ei1, batch.batch, ea1)
                _, g2 = model(x2, ei2, batch.batch, ea2)

                z1 = F.normalize(projector(g1), dim=-1)
                z2 = F.normalize(projector(g2), dim=-1)
                logits = z1 @ z2.t() / tau
                labels = torch.arange(logits.size(0), device=logits.device)
                loss = 0.5 * (
                    F.cross_entropy(logits, labels)
                    + F.cross_entropy(logits.t(), labels)
                )

                opt.zero_grad()
                loss.backward()
                opt.step()

        return model

    def _train_infograph(self, pyg_graphs: Sequence[Data]) -> GConv:
        model = self._init_model_from_data(pyg_graphs)
        loader = DataLoader(pyg_graphs, batch_size=self.config.batch_size, shuffle=True)
        graph_dim = model.hidden_dim * model.num_layers
        local_proj = nn.Linear(graph_dim, graph_dim).to(self.device)
        global_proj = nn.Linear(graph_dim, graph_dim).to(self.device)

        opt = Adam(
            list(model.parameters()) + list(local_proj.parameters()) + list(global_proj.parameters()),
            lr=self.config.lr,
        )

        model.train()
        local_proj.train()
        global_proj.train()
        for _ in range(self.config.epochs):
            for batch in loader:
                batch = batch.to(self.device)
                if batch.num_graphs <= 1:
                    continue

                node_z, graph_z = model(
                    batch.x.float(),
                    batch.edge_index,
                    batch.batch,
                    batch.edge_attr.float(),
                )
                node_h = local_proj(node_z)
                graph_h = global_proj(graph_z)

                pos_graph = graph_h[batch.batch]
                perm = torch.randperm(graph_h.size(0), device=graph_h.device)
                neg_graph = graph_h[perm][batch.batch]

                pos_score = (node_h * pos_graph).sum(dim=-1)
                neg_score = (node_h * neg_graph).sum(dim=-1)

                loss = (
                    F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score))
                    + F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
                )

                opt.zero_grad()
                loss.backward()
                opt.step()

        return model

    def _train_or_fallback(self, pyg_graphs: Sequence[Data], targets: Optional[Sequence] = None) -> GConv:
        if self.feature_extractor == "graphcl":
            return self._train_graphcl(pyg_graphs)
        if self.feature_extractor == "infograph":
            return self._train_infograph(pyg_graphs)
        if self.feature_extractor == "gin-random":
            return self._train_gin_random(pyg_graphs, targets=targets)
        raise ValueError(
            "feature_extractor must be 'infograph', 'graphcl', or 'gin-random'"
        )

    def fit(self, graphs: Sequence, targets: Optional[Sequence] = None):
        pyg_graphs = get_pyg_graphs_from_nx(graphs, targets)
        model_path = self._model_path()

        if self.mode == "test" and model_path.exists():
            try:
                payload = torch.load(model_path, map_location=self.device)
                model = self._init_model_from_data(pyg_graphs)
                model.load_state_dict(payload["state_dict"])
                self.model = model
                return self
            except Exception:
                # Fall through to fallback behavior below.
                pass

        if self.mode == "train":
            self.model = self._train_or_fallback(pyg_graphs, targets=targets)
            torch.save({"state_dict": self.model.state_dict()}, model_path)
            return self

        if self.config.allow_random_fallback:
            self.model = self._init_model_from_data(pyg_graphs)
            return self

        raise RuntimeError(f"Model not found at {model_path} and fallback disabled.")

    def transform(self, graphs: Sequence, targets: Optional[Sequence] = None):
        pyg_graphs = get_pyg_graphs_from_nx(graphs, targets)
        if self.model is None:
            self.fit(graphs, targets=targets)

        loader = DataLoader(pyg_graphs, batch_size=self.config.batch_size, shuffle=False)
        embeds = []
        ys = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                g = self.model.get_graph_embed(batch.x.float(), batch.edge_index, batch.batch, batch.edge_attr.float())
                embeds.append(g.cpu().numpy())
                if getattr(batch, "y", None) is not None:
                    ys.append(batch.y.cpu().numpy())

        x = np.vstack(embeds) if embeds else np.zeros((0, self.config.hidden_dim * self.config.num_layers))
        x = sparse.csr_matrix(x)
        if targets is not None and len(targets) > 0:
            y = np.concatenate(ys).ravel() if ys else np.asarray(targets)
            return x, y
        return x

    def fit_transform(self, graphs: Sequence, targets: Optional[Sequence] = None):
        return self.fit(graphs, targets=targets).transform(graphs, targets=targets)
