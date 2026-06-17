"""Official GDSS wrapper for class-conditional NetworkX graph generation."""

from __future__ import annotations

import importlib
import os
import pickle
import random
import re
import sys
import types
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Hashable, List, Optional, Sequence

import networkx as nx
import numpy as np
import torch
import yaml


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def _pushd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _safe_token(value: Hashable) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value))
    return s[:48] if s else "cls"


def _to_integer_labeled_graph(graph: nx.Graph) -> nx.Graph:
    g = nx.Graph(graph)
    g.remove_edges_from(nx.selfloop_edges(g))
    if g.number_of_nodes() == 0:
        g.add_node(0)
    g = nx.convert_node_labels_to_integers(g, ordering="default")
    for u in g.nodes():
        g.nodes[u]["label"] = int(g.nodes[u].get("label", g.nodes[u].get("attr", 0)))
    for u, v in g.edges():
        g.edges[u, v]["label"] = int(g.edges[u, v].get("label", g.edges[u, v].get("edge_attr", 0)))
    return g


@dataclass
class _ClassState:
    config_name: str
    ckpt_name: str
    label_values: np.ndarray
    label_probs: np.ndarray
    label_to_idx: Dict[int, int]
    idx_to_label: Dict[int, int]
    train_graphs: List[nx.Graph]


class OfficialGDSSGraphGenerator:
    """Thin wrapper around official GDSS Trainer/Sampler internals."""

    def __init__(
        self,
        *,
        gdss_repo_path: str = "/Users/cosmina/Documents/CoCoGraPE/GDSS",
        config_template: str = "community_small",
        random_state: int = 42,
        internal_test_split: float = 0.05,
        num_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        print_interval: Optional[int] = None,
        save_interval: Optional[int] = None,
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
        sampler_predictor: Optional[str] = None,
        sampler_corrector: Optional[str] = None,
        sampler_snr: Optional[float] = None,
        sampler_scale_eps: Optional[float] = None,
        sampler_n_steps: Optional[int] = None,
        enforce_connected: bool = True,
        use_ema_for_sampling: bool = False,
        use_semantic_node_labels: bool = True,
        verbose: bool = False,
    ) -> None:
        self.gdss_repo_path = Path(gdss_repo_path).resolve()
        self.config_template = str(config_template)
        self.random_state = int(random_state)
        self.internal_test_split = float(max(0.0, min(0.49, internal_test_split)))
        self.num_epochs = int(num_epochs) if num_epochs is not None else None
        self.batch_size = int(batch_size) if batch_size is not None else None
        self.print_interval = int(print_interval) if print_interval is not None else None
        self.save_interval = int(save_interval) if save_interval is not None else None
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.sampler_predictor = sampler_predictor
        self.sampler_corrector = sampler_corrector
        self.sampler_snr = sampler_snr
        self.sampler_scale_eps = sampler_scale_eps
        self.sampler_n_steps = int(sampler_n_steps) if sampler_n_steps is not None else None
        self.enforce_connected = bool(enforce_connected)
        self.use_ema_for_sampling = bool(use_ema_for_sampling)
        self.use_semantic_node_labels = bool(use_semantic_node_labels)
        self.verbose = bool(verbose)

        self._states: Dict[Hashable, _ClassState] = {}
        self._is_fitted = False

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[OFFICIAL-GDSS] {msg}")

    def _ensure_imports(self) -> None:
        repo = str(self.gdss_repo_path.resolve())
        if repo in sys.path:
            sys.path.remove(repo)
        sys.path.insert(0, repo)
        importlib.invalidate_caches()

        # GDSS uses bare imports like ``from utils.loader import ...`` and
        # ``from data.data_generators import ...``. Notebook kernels often
        # already contain unrelated modules with those short names, so clear
        # any non-GDSS namespace before importing the official code.
        for root_name in (
            "data",
            "evaluation",
            "losses",
            "models",
            "parsers",
            "sde",
            "solver",
            "utils",
        ):
            existing = sys.modules.get(root_name)
            if existing is None:
                continue
            existing_path = getattr(existing, "__file__", None)
            existing_pkg_path = getattr(existing, "__path__", None)
            path_text = " ".join(str(p) for p in (existing_pkg_path or []))
            if (
                existing_path is not None
                and repo not in str(existing_path)
            ) or (
                existing_pkg_path is not None
                and repo not in path_text
            ):
                sys.modules.pop(root_name, None)
                for name in list(sys.modules):
                    if name.startswith(f"{root_name}."):
                        sys.modules.pop(name, None)

        # This local GDSS copy keeps ``data_generators.py`` at the repo root,
        # while ``utils.data_loader`` imports ``data.data_generators``. Provide
        # that package-style alias without modifying the vendored GDSS files.
        data_generators = importlib.import_module("data_generators")
        data_package = types.ModuleType("data")
        data_package.__path__ = [str(self.gdss_repo_path / "data")]
        data_package.data_generators = data_generators
        sys.modules["data"] = data_package
        sys.modules["data.data_generators"] = data_generators

        # Import eagerly so notebook kernels fail here, not deep inside GDSS.
        importlib.import_module("utils.loader")

    @staticmethod
    def _label_distribution(graphs: Sequence[nx.Graph]) -> tuple[np.ndarray, np.ndarray]:
        counter: Counter[int] = Counter()
        for g in graphs:
            for u in g.nodes():
                counter[int(g.nodes[u].get("label", 0))] += 1
        if not counter:
            return np.array([0], dtype=np.int64), np.array([1.0], dtype=np.float64)
        values = np.array(sorted(counter.keys()), dtype=np.int64)
        probs = np.array([counter[v] for v in values], dtype=np.float64)
        probs /= probs.sum()
        return values, probs

    def _write_dataset_and_config(
        self,
        *,
        class_key: Hashable,
        graphs: Sequence[nx.Graph],
        seed: int,
        num_node_labels: int,
    ) -> str:
        token = _safe_token(class_key)
        feature_tag = "labels" if self.use_semantic_node_labels else "degree"
        config_name = f"cocogrape_gdss_{feature_tag}_{token}_{seed}"

        data_dir = self.gdss_repo_path / "data"
        config_dir = self.gdss_repo_path / "config"
        data_dir.mkdir(parents=True, exist_ok=True)
        config_dir.mkdir(parents=True, exist_ok=True)

        data_path = data_dir / f"{config_name}.pkl"
        with data_path.open("wb") as f:
            pickle.dump(list(graphs), f, protocol=pickle.HIGHEST_PROTOCOL)

        template_path = config_dir / f"{self.config_template}.yaml"
        if not template_path.exists():
            raise FileNotFoundError(f"GDSS config template not found: {template_path}")
        with template_path.open("r") as f:
            cfg = yaml.safe_load(f)

        max_nodes = max(g.number_of_nodes() for g in graphs)
        max_degree = max((max(dict(g.degree()).values()) if g.number_of_nodes() > 0 else 0) for g in graphs)
        if self.use_semantic_node_labels:
            max_feat_num = max(1, int(num_node_labels))
            cfg["data"]["init"] = "label"
        else:
            max_feat_num = max(4, int(max_degree + 2))
            cfg["data"]["init"] = "deg"
        num_graphs = len(graphs)

        # GDSS's internal loader computes ``int(test_split * len(graphs))``.
        # On small datasets this can become 0, which later crashes inside
        # feature masking when the test graph list is empty.
        safe_test_count = max(1, int(self.internal_test_split * num_graphs))
        safe_test_count = min(safe_test_count, max(1, num_graphs - 1))
        safe_test_split = float(safe_test_count / float(num_graphs))

        cfg["data"]["data"] = config_name
        cfg["data"]["dir"] = "./data"
        cfg["data"]["test_split"] = safe_test_split
        if self.batch_size is not None:
            cfg["data"]["batch_size"] = int(self.batch_size)
        cfg["data"]["max_node_num"] = int(max_nodes)
        cfg["data"]["max_feat_num"] = int(max_feat_num)
        cfg["train"]["name"] = "cocogrape_compare"
        if self.num_epochs is not None:
            cfg["train"]["num_epochs"] = int(self.num_epochs)
        train_epochs = int(cfg["train"]["num_epochs"])
        if self.print_interval is not None:
            cfg["train"]["print_interval"] = int(max(1, min(self.print_interval, train_epochs)))
        if self.save_interval is not None:
            cfg["train"]["save_interval"] = int(max(1, min(self.save_interval, train_epochs)))
        if self.learning_rate is not None:
            cfg["train"]["lr"] = float(self.learning_rate)
        if self.weight_decay is not None:
            cfg["train"]["weight_decay"] = float(self.weight_decay)
        if self.sampler_predictor is not None:
            cfg["sampler"]["predictor"] = str(self.sampler_predictor)
        if self.sampler_corrector is not None:
            cfg["sampler"]["corrector"] = str(self.sampler_corrector)
        if self.sampler_snr is not None:
            cfg["sampler"]["snr"] = float(self.sampler_snr)
        if self.sampler_scale_eps is not None:
            cfg["sampler"]["scale_eps"] = float(self.sampler_scale_eps)
        if self.sampler_n_steps is not None:
            cfg["sampler"]["n_steps"] = int(self.sampler_n_steps)
        cfg["sample"]["use_ema"] = bool(self.use_ema_for_sampling)
        cfg["sample"]["seed"] = int(seed)

        self._log(
            f"[class={class_key}] internal split adjusted to "
            f"{safe_test_count}/{num_graphs} (test_split={safe_test_split:.4f})"
        )

        with (config_dir / f"{config_name}.yaml").open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return config_name

    def fit(self, graphs: Sequence[nx.Graph], targets: Optional[Sequence[Hashable]] = None) -> "OfficialGDSSGraphGenerator":
        if len(graphs) == 0:
            raise ValueError("graphs must not be empty")
        if targets is not None and len(targets) != len(graphs):
            raise ValueError("targets length must match graphs")
        if not self.gdss_repo_path.exists():
            raise FileNotFoundError(f"GDSS repo path not found: {self.gdss_repo_path}")

        self._ensure_imports()
        with _pushd(self.gdss_repo_path):
            from parsers.config import get_config  # type: ignore
            from trainer import Trainer  # type: ignore

            if targets is None:
                buckets: Dict[Hashable, List[nx.Graph]] = {"default": [_to_integer_labeled_graph(g) for g in graphs]}
            else:
                buckets = defaultdict(list)
                for g, t in zip(graphs, targets):
                    buckets[t].append(_to_integer_labeled_graph(g))

            self._states = {}
            for i, (class_key, cls_graphs) in enumerate(buckets.items()):
                if len(cls_graphs) < 2:
                    continue
                seed = self.random_state + i
                _set_global_seeds(seed)

                label_values, label_probs = self._label_distribution(cls_graphs)
                label_to_idx = {
                    int(label): idx
                    for idx, label in enumerate(label_values.tolist())
                }
                idx_to_label = {
                    idx: label
                    for label, idx in label_to_idx.items()
                }

                if self.use_semantic_node_labels:
                    model_graphs = []
                    for graph in cls_graphs:
                        model_graph = graph.copy()
                        for node in model_graph.nodes():
                            raw_label = int(model_graph.nodes[node].get("label", 0))
                            model_graph.nodes[node]["label"] = label_to_idx[raw_label]
                        model_graphs.append(model_graph)
                else:
                    model_graphs = cls_graphs

                config_name = self._write_dataset_and_config(
                    class_key=class_key,
                    graphs=model_graphs,
                    seed=seed,
                    num_node_labels=len(label_values),
                )
                self._log(f"[class={class_key}] training with config={config_name}")
                config = get_config(config_name, seed)
                trainer = Trainer(config)
                ckpt_name = trainer.train(ts=f"{config_name}_seed{seed}")

                self._states[class_key] = _ClassState(
                    config_name=config_name,
                    ckpt_name=ckpt_name,
                    label_values=label_values,
                    label_probs=label_probs,
                    label_to_idx=label_to_idx,
                    idx_to_label=idx_to_label,
                    train_graphs=list(cls_graphs),
                )

        if not self._states:
            raise RuntimeError("No GDSS class models were trained.")
        self._is_fitted = True
        return self

    def _resolve_state(self, graph_type: Optional[Hashable]) -> _ClassState:
        if len(self._states) == 1 and graph_type is None:
            return next(iter(self._states.values()))
        if graph_type is None:
            raise ValueError(f"graph_type required. Known keys: {list(self._states.keys())}")
        if graph_type not in self._states:
            raise KeyError(f"Unknown graph_type={graph_type!r}. Known keys: {list(self._states.keys())}")
        return self._states[graph_type]

    @staticmethod
    def _connect_components(g: nx.Graph, rng: np.random.Generator) -> None:
        if g.number_of_nodes() <= 1 or nx.is_connected(g):
            return
        comps = [list(c) for c in nx.connected_components(g)]
        while len(comps) > 1:
            u = int(rng.choice(comps[0]))
            v = int(rng.choice(comps[1]))
            g.add_edge(u, v)
            comps = [list(c) for c in nx.connected_components(g)]

    def _assign_labels(self, g: nx.Graph, state: _ClassState, rng: np.random.Generator) -> None:
        sampled = rng.choice(state.label_values, size=g.number_of_nodes(), p=state.label_probs)
        for u in g.nodes():
            g.nodes[u]["label"] = int(sampled[int(u)])

    @staticmethod
    def _decode_semantic_labels(
        g: nx.Graph,
        sampled_x: np.ndarray,
        state: _ClassState,
    ) -> None:
        if sampled_x.ndim != 2:
            raise ValueError(f"Expected sampled X with shape [N, F], got {sampled_x.shape}")
        label_indices = np.asarray(sampled_x).argmax(axis=-1)
        idx_to_label = getattr(state, "idx_to_label", {})
        for row, node in enumerate(g.nodes()):
            label_idx = int(label_indices[row])
            g.nodes[node]["label"] = int(idx_to_label.get(label_idx, label_idx))

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
            raise ValueError("num_graphs must be positive")

        state = self._resolve_state(graph_type)
        gen_seed = self.random_state if seed is None else int(seed)
        _set_global_seeds(gen_seed)
        rng = np.random.default_rng(gen_seed)

        self._ensure_imports()
        outputs: List[nx.Graph] = []
        with _pushd(self.gdss_repo_path):
            from parsers.config import get_config  # type: ignore
            from utils.loader import (  # type: ignore
                load_ckpt,
                load_data,
                load_device,
                load_ema_from_ckpt,
                load_model_from_ckpt,
                load_sampling_fn,
            )
            from utils.graph_utils import init_flags, quantize  # type: ignore

            config = get_config(state.config_name, gen_seed)
            config.ckpt = state.ckpt_name
            config.sample.use_ema = bool(self.use_ema_for_sampling)
            config.sample.seed = int(gen_seed)
            device = load_device()

            ckpt = load_ckpt(config, device)
            configt = ckpt["config"]
            train_graphs, _ = load_data(configt, get_graph_list=True)

            model_x = load_model_from_ckpt(ckpt["params_x"], ckpt["x_state_dict"], device)
            model_adj = load_model_from_ckpt(ckpt["params_adj"], ckpt["adj_state_dict"], device)
            if config.sample.use_ema:
                ema_x = load_ema_from_ckpt(model_x, ckpt["ema_x"], configt.train.ema)
                ema_adj = load_ema_from_ckpt(model_adj, ckpt["ema_adj"], configt.train.ema)
                ema_x.copy_to(model_x.parameters())
                ema_adj.copy_to(model_adj.parameters())

            sampling_fn = load_sampling_fn(configt, config.sampler, config.sample, device)
            device_id = f"cuda:{device[0]}" if isinstance(device, list) else device

            max_rounds = max(20, int(np.ceil((num_graphs * 10) / max(1, configt.data.batch_size))))
            rounds = 0
            while len(outputs) < num_graphs and rounds < max_rounds:
                rounds += 1
                flags = init_flags(train_graphs, configt).to(device_id)
                sampled_x, adj, _ = sampling_fn(model_x, model_adj, flags)
                adjs_bin = quantize(adj).detach().cpu().numpy()
                sampled_x_np = sampled_x.detach().cpu().numpy()
                flags_np = flags.detach().cpu().numpy()

                for arr, x_arr, flag_arr in zip(adjs_bin, sampled_x_np, flags_np):
                    active = np.flatnonzero(np.asarray(flag_arr) > 0.5)
                    if active.size == 0:
                        active = np.array([0], dtype=int)
                    active_adj = np.asarray(arr, dtype=np.int64)[np.ix_(active, active)]
                    active_x = np.asarray(x_arr)[active]

                    g = nx.from_numpy_array(active_adj)
                    g.remove_edges_from(nx.selfloop_edges(g))
                    if g.number_of_nodes() < 1:
                        g.add_node(0)
                    g = nx.convert_node_labels_to_integers(g, ordering="default")
                    if num_nodes is not None and g.number_of_nodes() != int(num_nodes):
                        continue
                    if getattr(self, "use_semantic_node_labels", False):
                        self._decode_semantic_labels(g, active_x, state)
                    else:
                        self._assign_labels(g, state, rng)
                    if self.enforce_connected:
                        self._connect_components(g, rng)
                    outputs.append(g)
                    if len(outputs) >= num_graphs:
                        break

        if len(outputs) < num_graphs:
            raise RuntimeError(f"GDSS generated only {len(outputs)} / {num_graphs} graphs.")
        return outputs[:num_graphs]
