from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np


def stratified_split_graphs(
    graphs: Sequence[nx.Graph],
    targets: Sequence[Hashable],
    *,
    test_size: float | int = 0.2,
    seed: int = 42,
) -> Tuple[List[nx.Graph], List[nx.Graph], List[Hashable], List[Hashable]]:
    """
    Deterministic stratified split without sklearn dependency.
    """
    if len(graphs) != len(targets):
        raise ValueError("graphs and targets must have the same length")
    if len(graphs) == 0:
        raise ValueError("graphs cannot be empty")

    rng = random.Random(seed)
    by_class: Dict[Hashable, List[int]] = defaultdict(list)
    for i, t in enumerate(targets):
        by_class[t].append(i)

    test_idx: List[int] = []
    train_idx: List[int] = []
    n_total = len(graphs)
    abs_test = int(round(test_size * n_total)) if isinstance(test_size, float) else int(test_size)
    abs_test = max(1, min(n_total - 1, abs_test))

    # Initial per-class allocation
    for cls, idxs in by_class.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        if isinstance(test_size, float):
            n_cls_test = int(round(test_size * len(idxs)))
        else:
            n_cls_test = int(round(abs_test * (len(idxs) / n_total)))
        n_cls_test = max(1, min(len(idxs) - 1, n_cls_test)) if len(idxs) > 1 else 0
        test_idx.extend(idxs[:n_cls_test])
        train_idx.extend(idxs[n_cls_test:])

    # Adjust exact test size if needed
    if len(test_idx) > abs_test:
        rng.shuffle(test_idx)
        moved = test_idx[abs_test:]
        test_idx = test_idx[:abs_test]
        train_idx.extend(moved)
    elif len(test_idx) < abs_test:
        rng.shuffle(train_idx)
        need = abs_test - len(test_idx)
        moved = train_idx[:need]
        train_idx = train_idx[need:]
        test_idx.extend(moved)

    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]
    train_targets = [targets[i] for i in train_idx]
    test_targets = [targets[i] for i in test_idx]
    return train_graphs, test_graphs, train_targets, test_targets


def generate_graphs_same_way(
    model: Any,
    *,
    model_kind: str,
    train_graphs: Sequence[nx.Graph],
    train_targets: Optional[Sequence[Hashable]],
    n_per_class: int = 50,
    seed: int = 42,
    class_values: Optional[Sequence[Hashable]] = None,
    denoise_generation_mode: str = "prior",
    max_attempts: int = 5,
) -> Tuple[List[nx.Graph], List[Hashable]]:
    """
    Unified class-conditional generation wrapper for:
      - model_kind='gran'  : model.generate(num_graphs, graph_type=class)
      - model_kind='vgae'  : model.generate(num_graphs, graph_type=class)
      - model_kind='digress': model.generate(num_graphs, graph_type=class)
      - model_kind='gdss'  : model.generate(num_graphs, graph_type=class)
      - model_kind='denoise':
          - prior mode (default): DecompositionalEncoderDecoder.sample(...)
          - seeded mode: DecompositionalEncoderDecoder.conditional_sample(...)
    """
    if n_per_class <= 0:
        raise ValueError("n_per_class must be positive")
    has_targets = (
        train_targets is not None
        and len(train_targets) == len(train_graphs)
        and all(t is not None for t in train_targets)
    )
    if train_targets is not None and len(train_graphs) != len(train_targets):
        raise ValueError("train_graphs and train_targets must have same length")
    if len(train_graphs) == 0:
        raise ValueError("train_graphs is empty")

    if has_targets and class_values is None:
        class_values = sorted(set(train_targets), key=lambda x: repr(x))
    elif not has_targets:
        class_values = [None]

    out_graphs: List[nx.Graph] = []
    out_targets: List[Hashable] = []
    rng = random.Random(seed)

    kind = model_kind.lower().strip()
    denoise_mode = denoise_generation_mode.lower().strip()
    if denoise_mode not in {"prior", "seeded"}:
        raise ValueError("denoise_generation_mode must be 'prior' or 'seeded'")

    def _to_list(x: Any) -> List[Any]:
        if x is None:
            return []
        if isinstance(x, list):
            return x
        if isinstance(x, tuple):
            return list(x)
        return [x]

    for class_idx, cls in enumerate(class_values):
        if has_targets:
            cls_graphs = [g for g, t in zip(train_graphs, train_targets) if t == cls]
        else:
            cls_graphs = list(train_graphs)
        if len(cls_graphs) == 0:
            continue

        g: List[nx.Graph] = []
        for attempt in range(max_attempts):
            remaining = n_per_class - len(g)
            if remaining <= 0:
                break

            if kind in {"gran", "vgae", "digress", "gdss"}:
                kwargs = dict(num_graphs=remaining, seed=seed + class_idx + attempt)
                if cls is not None:
                    kwargs["graph_type"] = cls
                chunk = model.generate(**kwargs)
                g.extend(_to_list(chunk))

            elif kind == "denoise":
                if denoise_mode == "prior":
                    chunk = model.sample(n_samples=remaining, desired_class=cls)
                    g.extend(_to_list(chunk))
                else:
                    seeds = [rng.choice(cls_graphs) for _ in range(remaining)]
                    decoded = model.conditional_sample(seeds, n_samples=1, desired_class=cls)
                    flat = [x[0] if isinstance(x, (list, tuple)) and len(x) > 0 else x for x in _to_list(decoded)]
                    g.extend(flat)
            else:
                raise ValueError(f"Unknown model_kind='{model_kind}'")

        # Enforce exact per-class counts for fair comparison.
        if len(g) == 0:
            raise RuntimeError(f"{model_kind} failed to generate any graphs for class={cls!r}")
        if len(g) < n_per_class:
            g.extend(rng.choices(g, k=n_per_class - len(g)))
        elif len(g) > n_per_class:
            g = g[:n_per_class]

        out_graphs.extend(g)
        out_targets.extend([cls] * len(g))

    return out_graphs, out_targets


def _node_label(graph: nx.Graph, node: Hashable, label_key: str) -> Hashable:
    attrs = graph.nodes[node]
    if label_key in attrs:
        return attrs[label_key]
    return attrs.get("label", 0)


def _pair_key(a: Hashable, b: Hashable) -> Tuple[Hashable, Hashable]:
    return (a, b) if repr(a) <= repr(b) else (b, a)


def fit_label_pair_edge_model(
    graphs: Sequence[nx.Graph],
    *,
    label_key: str = "display_label",
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Dict[str, Any]:
    """
    Common model-agnostic Bernoulli edge model by unordered node-label pair.
    """
    if len(graphs) == 0:
        return {"pair_prob": {}, "global_prob": 0.5, "alpha": alpha, "beta": beta}

    edge_count: Dict[Tuple[Hashable, Hashable], int] = defaultdict(int)
    total_count: Dict[Tuple[Hashable, Hashable], int] = defaultdict(int)
    global_edges = 0
    global_total = 0

    for G in graphs:
        nodes = list(G.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                lu = _node_label(G, u, label_key)
                lv = _node_label(G, v, label_key)
                k = _pair_key(lu, lv)
                total_count[k] += 1
                global_total += 1
                if G.has_edge(u, v):
                    edge_count[k] += 1
                    global_edges += 1

    pair_prob: Dict[Tuple[Hashable, Hashable], float] = {}
    for k, t in total_count.items():
        e = edge_count.get(k, 0)
        pair_prob[k] = float((e + alpha) / (t + alpha + beta))

    global_prob = float((global_edges + alpha) / (global_total + alpha + beta)) if global_total > 0 else 0.5
    return {
        "pair_prob": pair_prob,
        "global_prob": global_prob,
        "alpha": alpha,
        "beta": beta,
    }


def graph_cpe(
    graph: nx.Graph,
    stats: Dict[str, Any],
    *,
    label_key: str = "display_label",
    eps: float = 1e-9,
) -> float:
    """
    Cross-entropy-per-edge over all upper-triangle node pairs.
    """
    nodes = list(graph.nodes())
    if len(nodes) <= 1:
        return float("nan")

    pair_prob = stats.get("pair_prob", {})
    p_global = float(stats.get("global_prob", 0.5))

    ce_sum = 0.0
    n_pairs = 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            u, v = nodes[i], nodes[j]
            lu = _node_label(graph, u, label_key)
            lv = _node_label(graph, v, label_key)
            p = float(pair_prob.get(_pair_key(lu, lv), p_global))
            p = min(max(p, eps), 1.0 - eps)
            y = 1.0 if graph.has_edge(u, v) else 0.0
            ce_sum += -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
            n_pairs += 1

    return float(ce_sum / max(1, n_pairs))


def compute_classwise_cpe(
    *,
    test_graphs: Sequence[nx.Graph],
    test_targets: Sequence[Hashable],
    generated_graphs: Sequence[nx.Graph],
    generated_targets: Sequence[Hashable],
    label_key: str = "display_label",
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Dict[str, Any]:
    """
    Fair, model-agnostic CPE surrogate:
      1) fit edge-prob model from generated graphs per class
      2) evaluate CE on held-out test graphs of same class
    """
    if len(test_graphs) != len(test_targets):
        raise ValueError("test_graphs and test_targets length mismatch")
    if len(generated_graphs) != len(generated_targets):
        raise ValueError("generated_graphs and generated_targets length mismatch")

    classes = sorted(set(list(test_targets) + list(generated_targets)), key=lambda x: repr(x))
    cpe_per_class: Dict[Hashable, float] = {}
    n_per_class: Dict[Hashable, int] = {}

    for cls in classes:
        g_cls = [g for g, t in zip(generated_graphs, generated_targets) if t == cls]
        t_cls = [g for g, t in zip(test_graphs, test_targets) if t == cls]
        n_per_class[cls] = len(t_cls)
        if len(g_cls) == 0 or len(t_cls) == 0:
            cpe_per_class[cls] = float("nan")
            continue
        stats = fit_label_pair_edge_model(g_cls, label_key=label_key, alpha=alpha, beta=beta)
        vals = [graph_cpe(g, stats, label_key=label_key) for g in t_cls]
        vals = [v for v in vals if np.isfinite(v)]
        cpe_per_class[cls] = float(np.mean(vals)) if vals else float("nan")

    finite_vals = [v for v in cpe_per_class.values() if np.isfinite(v)]
    macro_cpe = float(np.mean(finite_vals)) if finite_vals else float("nan")
    return {
        "cpe_per_class": cpe_per_class,
        "macro_cpe": macro_cpe,
        "n_test_per_class": n_per_class,
    }


def evaluate_generation_fair(
    *,
    test_graphs: Sequence[nx.Graph],
    test_targets: Sequence[Hashable],
    generated_graphs: Sequence[nx.Graph],
    generated_targets: Sequence[Hashable],
    label_key: str = "display_label",
) -> Dict[str, Any]:
    """
    Unified lightweight evaluator (same for all generators).
    """
    cpe = compute_classwise_cpe(
        test_graphs=test_graphs,
        test_targets=test_targets,
        generated_graphs=generated_graphs,
        generated_targets=generated_targets,
        label_key=label_key,
    )

    def density(g: nx.Graph) -> float:
        n = g.number_of_nodes()
        return 0.0 if n <= 1 else (2.0 * g.number_of_edges()) / (n * (n - 1))

    classes = sorted(set(list(test_targets) + list(generated_targets)), key=lambda x: repr(x))
    density_abs_err: Dict[Hashable, float] = {}
    n_nodes_abs_err: Dict[Hashable, float] = {}
    label_l1: Dict[Hashable, float] = {}

    for cls in classes:
        gt = [g for g, t in zip(generated_graphs, generated_targets) if t == cls]
        tt = [g for g, t in zip(test_graphs, test_targets) if t == cls]
        if len(gt) == 0 or len(tt) == 0:
            density_abs_err[cls] = float("nan")
            n_nodes_abs_err[cls] = float("nan")
            label_l1[cls] = float("nan")
            continue

        density_abs_err[cls] = abs(float(np.mean([density(g) for g in gt])) - float(np.mean([density(g) for g in tt])))
        n_nodes_abs_err[cls] = abs(float(np.mean([g.number_of_nodes() for g in gt])) - float(np.mean([g.number_of_nodes() for g in tt])))

        c_gt: Counter = Counter()
        c_tt: Counter = Counter()
        for g in gt:
            for u in g.nodes():
                c_gt[_node_label(g, u, label_key)] += 1
        for g in tt:
            for u in g.nodes():
                c_tt[_node_label(g, u, label_key)] += 1
        keys = sorted(set(c_gt) | set(c_tt), key=lambda x: repr(x))
        s_gt = sum(c_gt.values()) or 1
        s_tt = sum(c_tt.values()) or 1
        l1 = 0.0
        for k in keys:
            l1 += abs((c_gt.get(k, 0) / s_gt) - (c_tt.get(k, 0) / s_tt))
        label_l1[cls] = float(l1)

    return {
        "cpe": cpe,
        "density_abs_err_per_class": density_abs_err,
        "n_nodes_abs_err_per_class": n_nodes_abs_err,
        "label_l1_per_class": label_l1,
        "n_generated": len(generated_graphs),
        "n_test": len(test_graphs),
    }
