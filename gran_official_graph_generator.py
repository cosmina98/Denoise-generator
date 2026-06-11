"""Official GRAN wrapper for NetworkX graphs with class-conditional fit/generate API."""

from __future__ import annotations

import importlib
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Hashable, List, Optional, Sequence

import networkx as nx
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader


def _ensure_gran_import_path(gran_repo_path: str) -> None:
    repo = str(Path(gran_repo_path).resolve())
    if repo in sys.path:
        sys.path.remove(repo)
    sys.path.insert(0, repo)
    importlib.invalidate_caches()

    existing_utils = sys.modules.get("utils")
    existing_path = getattr(existing_utils, "__file__", None)
    existing_pkg_path = getattr(existing_utils, "__path__", None)
    if existing_utils is not None:
        path_text = " ".join(str(p) for p in (existing_pkg_path or []))
        if (
            existing_path is not None
            and repo not in str(existing_path)
        ) or (
            existing_pkg_path is not None
            and repo not in path_text
        ):
            sys.modules.pop("utils", None)
            for name in list(sys.modules):
                if name.startswith("utils."):
                    sys.modules.pop(name, None)

    importlib.import_module("utils.data_helper")


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_integer_labeled_graph(graph: nx.Graph) -> nx.Graph:
    H = nx.convert_node_labels_to_integers(graph.copy(), ordering="default")
    for u in H.nodes():
        H.nodes[u]["label"] = int(H.nodes[u].get("label", H.nodes[u].get("attr", 0)))
    for u, v in H.edges():
        H.edges[u, v]["label"] = int(H.edges[u, v].get("label", H.edges[u, v].get("edge_attr", 0)))
    return H


@dataclass
class _TypeState:
    model: torch.nn.Module
    num_nodes_pmf: np.ndarray
    label_values: np.ndarray
    label_probs: np.ndarray
    max_num_nodes: int


class OfficialGRANGraphGenerator:
    """Use official GRAN model/dataset code with a simple sklearn-like interface."""

    def __init__(
        self,
        *,
        gran_repo_path: str = "/Users/cosmina/Documents/CoCoGraPE/GRAN",
        hidden_dim: int = 128,
        embedding_dim: int = 128,
        num_gnn_layers: int = 7,
        num_gnn_prop: int = 1,
        num_mix_component: int = 20,
        block_size: int = 1,
        sample_stride: int = 1,
        num_canonical_order: int = 1,
        has_attention: bool = True,
        dimension_reduce: bool = True,
        edge_weight: float = 1.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        batch_size: int = 1,
        max_epoch: int = 200,
        num_workers: int = 0,
        num_fwd_pass: int = 1,
        num_subgraph_batch: int = 50,
        node_order: str = "DFS",
        enforce_connected: bool = True,
        random_state: int = 42,
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.gran_repo_path = str(gran_repo_path)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.num_gnn_layers = int(num_gnn_layers)
        self.num_gnn_prop = int(num_gnn_prop)
        self.num_mix_component = int(num_mix_component)
        self.block_size = int(block_size)
        self.sample_stride = int(sample_stride)
        self.num_canonical_order = int(num_canonical_order)
        self.has_attention = bool(has_attention)
        self.dimension_reduce = bool(dimension_reduce)
        self.edge_weight = float(edge_weight)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epoch = int(max_epoch)
        self.num_workers = int(num_workers)
        self.num_fwd_pass = int(num_fwd_pass)
        self.num_subgraph_batch = int(num_subgraph_batch)
        self.node_order = str(node_order)
        self.enforce_connected = bool(enforce_connected)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda:0"
            else:
                self.device = "cpu"
        else:
            self.device = str(device)

        self._states: Dict[Hashable, _TypeState] = {}
        self._is_fitted = False

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[OFFICIAL-GRAN] {message}")

    def _build_config(self, *, seed: int, max_num_nodes: int) -> Any:
        data_path = str(Path(self.gran_repo_path) / "data")
        save_dir = str(Path(self.gran_repo_path) / "exp" / "cocogrape_tmp")
        os.makedirs(save_dir, exist_ok=True)
        use_gpu = self.device.startswith("cuda") and torch.cuda.is_available()
        return SimpleNamespace(
            seed=seed,
            device=self.device if use_gpu else "cpu",
            use_gpu=use_gpu,
            gpus=[0] if use_gpu else [],
            save_dir=save_dir,
            dataset=SimpleNamespace(
                data_path=data_path,
                name="cocogrape_inline",
                node_order=self.node_order,
                num_fwd_pass=self.num_fwd_pass,
                is_sample_subgraph=True,
                num_subgraph_batch=self.num_subgraph_batch,
                is_overwrite_precompute=True,
            ),
            model=SimpleNamespace(
                name="GRANMixtureBernoulli",
                max_num_nodes=max_num_nodes,
                hidden_dim=self.hidden_dim,
                embedding_dim=self.embedding_dim,
                is_sym=True,
                block_size=self.block_size,
                sample_stride=self.sample_stride,
                num_GNN_prop=self.num_gnn_prop,
                num_GNN_layers=self.num_gnn_layers,
                num_canonical_order=self.num_canonical_order,
                num_mix_component=self.num_mix_component,
                dimension_reduce=self.dimension_reduce,
                has_attention=self.has_attention,
                edge_weight=self.edge_weight,
            ),
            train=SimpleNamespace(
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                max_epoch=self.max_epoch,
                lr=self.learning_rate,
                wd=self.weight_decay,
                display_iter=50,
            ),
            test=SimpleNamespace(batch_size=20),
        )

    @staticmethod
    def _label_distribution(graphs: Sequence[nx.Graph]) -> tuple[np.ndarray, np.ndarray]:
        label_counter: Counter[int] = Counter()
        for g in graphs:
            for u in g.nodes():
                label_counter[int(g.nodes[u].get("label", 0))] += 1
        if not label_counter:
            return np.array([0], dtype=np.int64), np.array([1.0], dtype=np.float64)
        values = np.array(sorted(label_counter.keys()), dtype=np.int64)
        probs = np.array([label_counter[v] for v in values], dtype=np.float64)
        probs = probs / probs.sum()
        return values, probs

    @staticmethod
    def _num_nodes_pmf(graphs: Sequence[nx.Graph], max_num_nodes: int) -> np.ndarray:
        sizes = [g.number_of_nodes() for g in graphs if g.number_of_nodes() > 0]
        if not sizes:
            pmf = np.zeros(max_num_nodes, dtype=np.float32)
            pmf[0] = 1.0
            return pmf
        bins = np.bincount(sizes, minlength=max_num_nodes + 1)[1 : max_num_nodes + 1].astype(np.float64)
        if bins.sum() <= 0:
            bins[:] = 1.0
        bins = bins / bins.sum()
        return bins.astype(np.float32)

    def fit(self, graphs: Sequence[nx.Graph], targets: Optional[Sequence[Hashable]] = None) -> "OfficialGRANGraphGenerator":
        if len(graphs) == 0:
            raise ValueError("graphs must not be empty.")
        if targets is not None and len(targets) != len(graphs):
            raise ValueError("targets must have the same length as graphs.")

        _ensure_gran_import_path(self.gran_repo_path)
        from dataset.gran_data import GRANData  # type: ignore
        from model.gran_mixture_bernoulli import GRANMixtureBernoulli  # type: ignore

        if targets is None:
            buckets: Dict[Hashable, List[nx.Graph]] = {"default": [_to_integer_labeled_graph(g) for g in graphs]}
        else:
            buckets = defaultdict(list)
            for g, t in zip(graphs, targets):
                buckets[t].append(_to_integer_labeled_graph(g))

        self._states = {}
        for type_idx, (type_key, bucket_graphs) in enumerate(buckets.items()):
            if len(bucket_graphs) == 0:
                continue

            seed = self.random_state + type_idx
            _set_global_seeds(seed)
            max_num_nodes = max(g.number_of_nodes() for g in bucket_graphs)
            cfg = self._build_config(seed=seed, max_num_nodes=max_num_nodes)

            self._log(f"[type={type_key}] building GRANData with {len(bucket_graphs)} graphs")
            train_dataset = GRANData(cfg, bucket_graphs, tag=f"train_{type_key}")
            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.train.batch_size,
                shuffle=cfg.train.shuffle,
                num_workers=cfg.train.num_workers,
                collate_fn=train_dataset.collate_fn,
                drop_last=False,
            )

            model = GRANMixtureBernoulli(cfg).to(cfg.device)
            optimizer = Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)

            iter_count = 0
            for epoch in range(cfg.train.max_epoch):
                model.train()
                epoch_losses: List[float] = []
                for batch_data in train_loader:
                    optimizer.zero_grad()
                    avg_loss = 0.0
                    for ff in range(cfg.dataset.num_fwd_pass):
                        data = {}
                        for k, v in batch_data[ff].items():
                            if torch.is_tensor(v):
                                data[k] = v.to(cfg.device)
                            else:
                                data[k] = v
                        loss = model(data)
                        avg_loss = avg_loss + loss
                        loss.backward()
                    avg_loss = avg_loss / float(cfg.dataset.num_fwd_pass)
                    optimizer.step()
                    iter_count += 1
                    epoch_losses.append(float(avg_loss.detach().cpu().item()))

                if self.verbose and ((epoch + 1) % 20 == 0 or epoch == 0 or epoch + 1 == cfg.train.max_epoch):
                    mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
                    self._log(f"[type={type_key}] epoch={epoch+1:03d}/{cfg.train.max_epoch:03d} nll={mean_loss:.6f}")

            label_values, label_probs = self._label_distribution(bucket_graphs)
            num_nodes_pmf = self._num_nodes_pmf(bucket_graphs, max_num_nodes)
            self._states[type_key] = _TypeState(
                model=model.eval(),
                num_nodes_pmf=num_nodes_pmf,
                label_values=label_values,
                label_probs=label_probs,
                max_num_nodes=max_num_nodes,
            )

        if len(self._states) == 0:
            raise RuntimeError("No fitted GRAN states were created.")
        self._is_fitted = True
        return self

    def _resolve_state(self, graph_type: Optional[Hashable]) -> tuple[Hashable, _TypeState]:
        if len(self._states) == 1 and graph_type is None:
            k = next(iter(self._states.keys()))
            return k, self._states[k]
        if graph_type is None:
            raise ValueError(f"graph_type required. Known keys: {list(self._states.keys())}")
        if graph_type not in self._states:
            raise KeyError(f"Unknown graph_type {graph_type!r}. Known keys: {list(self._states.keys())}")
        return graph_type, self._states[graph_type]

    def _assign_node_labels(self, graph: nx.Graph, label_values: np.ndarray, label_probs: np.ndarray, rng: np.random.Generator) -> None:
        if graph.number_of_nodes() == 0:
            return
        sampled = rng.choice(label_values, size=graph.number_of_nodes(), p=label_probs)
        for u in graph.nodes():
            graph.nodes[u]["label"] = int(sampled[int(u)])

    def _connect_components(self, graph: nx.Graph, rng: np.random.Generator) -> None:
        if graph.number_of_nodes() <= 1 or nx.is_connected(graph):
            return
        components = [list(c) for c in nx.connected_components(graph)]
        while len(components) > 1:
            u = int(rng.choice(components[0]))
            v = int(rng.choice(components[1]))
            graph.add_edge(u, v)
            components = [list(c) for c in nx.connected_components(graph)]

    def generate(
        self,
        *,
        num_graphs: int,
        graph_type: Optional[Hashable] = None,
        num_nodes: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[nx.Graph]:
        if not self._is_fitted:
            raise RuntimeError("Call fit(...) before generate(...).")
        if num_graphs <= 0:
            raise ValueError("num_graphs must be positive.")

        _, state = self._resolve_state(graph_type)
        if seed is None:
            seed = self.random_state
        _set_global_seeds(int(seed))
        rng = np.random.default_rng(int(seed))

        model = state.model
        model.eval()

        if num_nodes is None:
            pmf = state.num_nodes_pmf
        else:
            n = int(num_nodes)
            if n <= 0:
                raise ValueError("num_nodes must be positive.")
            max_nodes = max(state.max_num_nodes, n)
            pmf = np.zeros(max_nodes, dtype=np.float32)
            pmf[n - 1] = 1.0

        outputs: List[nx.Graph] = []
        batch_size = 20
        remaining = int(num_graphs)
        while remaining > 0:
            b = min(batch_size, remaining)
            with torch.no_grad():
                sampled = model({"is_sampling": True, "batch_size": b, "num_nodes_pmf": pmf})
            for adj_t in sampled:
                adj = adj_t.detach().cpu().numpy().astype(np.int64)
                graph = nx.from_numpy_array(adj)
                self._assign_node_labels(graph, state.label_values, state.label_probs, rng)
                if self.enforce_connected:
                    self._connect_components(graph, rng)
                outputs.append(graph)
            remaining -= b
        return outputs
