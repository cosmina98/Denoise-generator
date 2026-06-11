import hashlib
import os
import subprocess as sp
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_kernels

from coco_grape.data_processor.generative.embedding_distribution_metrics import evaluate_embedding_distributions


def graph_to_structural_vector(graph: nx.Graph) -> np.ndarray:
    """
    Lightweight graph-level feature vector for metric comparisons.
    """
    g = nx.Graph(graph)
    n = float(g.number_of_nodes())
    m = float(g.number_of_edges())
    density = float(nx.density(g)) if n > 1 else 0.0

    if n > 0:
        degrees = np.array([d for _, d in g.degree()], dtype=float)
        deg_mean = float(np.mean(degrees))
        deg_std = float(np.std(degrees))
    else:
        deg_mean = 0.0
        deg_std = 0.0

    n_components = float(nx.number_connected_components(g)) if n > 0 else 0.0
    clustering = float(nx.average_clustering(g)) if n > 2 else 0.0
    triangles = float(sum(nx.triangles(g).values()) / 3.0) if n > 2 else 0.0

    if n > 0:
        largest_cc = max(nx.connected_components(g), key=len)
        lcc = g.subgraph(largest_cc).copy()
        lcc_n = float(lcc.number_of_nodes())
        if lcc_n > 1:
            try:
                diameter = float(nx.diameter(lcc))
            except Exception:
                diameter = 0.0
        else:
            diameter = 0.0
    else:
        lcc_n = 0.0
        diameter = 0.0

    return np.array(
        [n, m, density, deg_mean, deg_std, n_components, clustering, triangles, lcc_n, diameter],
        dtype=np.float64,
    )


def graphs_to_structural_vectors(graphs: Sequence[nx.Graph]) -> np.ndarray:
    if len(graphs) == 0:
        return np.zeros((0, 10), dtype=np.float64)
    return np.vstack([graph_to_structural_vector(g) for g in graphs])


def evaluate_graphs_structural(
    reference_graphs: Sequence[nx.Graph],
    generated_graphs: Sequence[nx.Graph],
    nearest_k: int = 5,
) -> Dict[str, float]:
    """
    Compare two graph sets by first mapping each graph to handcrafted structural vectors.
    """
    ref_x = graphs_to_structural_vectors(reference_graphs)
    gen_x = graphs_to_structural_vectors(generated_graphs)
    return evaluate_embedding_distributions(ref_x, gen_x, nearest_k=nearest_k, include_timing=True)


def _to_undirected_nonempty(graphs: Sequence[nx.Graph]) -> Sequence[nx.Graph]:
    return [nx.Graph(g) for g in graphs if nx.Graph(g).number_of_nodes() > 0]


def _normalize_histograms(samples: Sequence[np.ndarray], width: Optional[int] = None) -> np.ndarray:
    if len(samples) == 0:
        return np.zeros((0, 1), dtype=np.float64)
    max_len = int(width) if width is not None else max(len(np.asarray(s).ravel()) for s in samples)
    out = np.zeros((len(samples), max_len), dtype=np.float64)
    for i, s in enumerate(samples):
        arr = np.asarray(s, dtype=np.float64).ravel()
        if arr.size > 0:
            out[i, : min(arr.size, max_len)] = arr[:max_len]
            denom = out[i].sum()
            if denom > 0:
                out[i] /= denom
    return out


def _rbf_mmd_from_distance(
    ref_samples: Sequence[np.ndarray],
    gen_samples: Sequence[np.ndarray],
    sigma: float = 1.0,
    distance: str = "l1tv",
) -> float:
    if len(ref_samples) == 0 or len(gen_samples) == 0:
        return float("nan")
    max_len = max(
        max(len(np.asarray(s).ravel()) for s in ref_samples),
        max(len(np.asarray(s).ravel()) for s in gen_samples),
    )
    ref_x = _normalize_histograms(ref_samples, width=max_len)
    gen_x = _normalize_histograms(gen_samples, width=max_len)
    if ref_x.shape[0] == 0 or gen_x.shape[0] == 0:
        return float("nan")
    d_ref_ref = _pairwise_distance(ref_x, ref_x, distance=distance)
    d_gen_gen = _pairwise_distance(gen_x, gen_x, distance=distance)
    d_ref_gen = _pairwise_distance(ref_x, gen_x, distance=distance)

    gamma = 1.0 / (2.0 * (float(sigma) ** 2 + 1e-12))
    k_ref_ref = np.exp(-gamma * (d_ref_ref ** 2))
    k_gen_gen = np.exp(-gamma * (d_gen_gen ** 2))
    k_ref_gen = np.exp(-gamma * (d_ref_gen ** 2))
    return float(k_ref_ref.mean() + k_gen_gen.mean() - 2.0 * k_ref_gen.mean())


def _pairwise_distance(x: np.ndarray, y: np.ndarray, distance: str = "l1tv") -> np.ndarray:
    if distance == "l1tv":
        # total variation distance for histogram-like rows
        return 0.5 * np.abs(x[:, None, :] - y[None, :, :]).sum(axis=2)
    # default: l2
    return np.sqrt(((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2))


def _degree_histograms(graphs: Sequence[nx.Graph]) -> Sequence[np.ndarray]:
    return [np.asarray(nx.degree_histogram(g), dtype=np.float64) for g in graphs]


def _cluster_histograms(graphs: Sequence[nx.Graph], bins: int = 100) -> Sequence[np.ndarray]:
    out = []
    for g in graphs:
        vals = list(nx.clustering(g).values())
        hist, _ = np.histogram(vals, bins=bins, range=(0.0, 1.0), density=False)
        out.append(hist.astype(np.float64))
    return out


def _spectral_histograms(graphs: Sequence[nx.Graph], bins: int = 200) -> Sequence[np.ndarray]:
    out = []
    for g in graphs:
        try:
            eigs = np.linalg.eigvalsh(nx.normalized_laplacian_matrix(g).todense())
        except Exception:
            eigs = np.zeros(max(1, g.number_of_nodes()), dtype=np.float64)
        hist, _ = np.histogram(eigs, bins=bins, range=(-1e-5, 2.0), density=False)
        hist = hist.astype(np.float64)
        s = hist.sum()
        if s > 0:
            hist /= s
        out.append(hist)
    return out


def _find_orca_binary(orca_bin: Optional[str]) -> Optional[str]:
    if orca_bin:
        p = Path(orca_bin)
        return str(p) if p.exists() else None

    candidates = [
        Path.cwd() / "DiGress" / "src" / "analysis" / "orca" / "orca",
        Path.cwd() / "GDSS" / "evaluation" / "orca" / "orca",
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    from shutil import which

    return which("orca")


def _edge_list_reindexed(graph: nx.Graph) -> Sequence[Tuple[int, int]]:
    id2idx = {str(u): i for i, u in enumerate(graph.nodes())}
    return [(id2idx[str(u)], id2idx[str(v)]) for (u, v) in graph.edges()]


def _orbit_vectors(graphs: Sequence[nx.Graph], orca_bin: Optional[str]) -> Sequence[np.ndarray]:
    if not orca_bin:
        return []
    start_str = "orbit counts:"
    out = []
    for g in graphs:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmpf:
            tmpf.write(f"{g.number_of_nodes()} {g.number_of_edges()}\n")
            for (u, v) in _edge_list_reindexed(g):
                tmpf.write(f"{u} {v}\n")
            tmp_name = tmpf.name
        try:
            raw = sp.check_output([orca_bin, "node", "4", tmp_name, "std"]).decode("utf8", errors="ignore")
            idx = raw.find(start_str)
            if idx < 0:
                continue
            raw = raw[idx + len(start_str) :].strip()
            rows = [r.strip() for r in raw.splitlines() if r.strip()]
            if not rows:
                continue
            node_orbits = np.array([list(map(int, r.split())) for r in rows], dtype=np.float64)
            out.append(node_orbits.sum(axis=0) / max(g.number_of_nodes(), 1))
        except Exception:
            continue
        finally:
            try:
                os.remove(tmp_name)
            except OSError:
                pass
    return out


def _wl_mmd(
    reference_graphs: Sequence[nx.Graph],
    generated_graphs: Sequence[nx.Graph],
    n_iter: int = 4,
) -> float:
    try:
        import grakel
        from grakel.kernels import VertexHistogram, WeisfeilerLehman

        ref_nx = [g.copy() for g in reference_graphs]
        gen_nx = [g.copy() for g in generated_graphs]
        for g in ref_nx:
            nx.set_node_attributes(g, dict(g.degree()), "degree")
        for g in gen_nx:
            nx.set_node_attributes(g, dict(g.degree()), "degree")

        wl = WeisfeilerLehman(n_iter=n_iter, base_graph_kernel=VertexHistogram, normalize=True)
        k_ref_ref = wl.fit_transform(grakel.graph_from_networkx(ref_nx, node_labels_tag="degree"))
        k_ref_gen = wl.transform(grakel.graph_from_networkx(gen_nx, node_labels_tag="degree"))
        k_gen_gen = wl.fit_transform(grakel.graph_from_networkx(gen_nx, node_labels_tag="degree"))
        return float(k_gen_gen.mean() + k_ref_ref.mean() - 2.0 * k_ref_gen.mean())
    except Exception:
        # Fallback: hash-based WL histogram MMD.
        def graph_hash(g: nx.Graph) -> str:
            try:
                return nx.weisfeiler_lehman_graph_hash(g, iterations=n_iter)
            except Exception:
                basis = f"{g.number_of_nodes()}|{g.number_of_edges()}"
                return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]

        ref_hashes = [graph_hash(g) for g in reference_graphs]
        gen_hashes = [graph_hash(g) for g in generated_graphs]
        vocab = {h: i for i, h in enumerate(sorted(set(ref_hashes).union(set(gen_hashes))))}
        ref_vec = np.zeros((len(ref_hashes), len(vocab)), dtype=np.float64)
        gen_vec = np.zeros((len(gen_hashes), len(vocab)), dtype=np.float64)
        for i, h in enumerate(ref_hashes):
            ref_vec[i, vocab[h]] = 1.0
        for i, h in enumerate(gen_hashes):
            gen_vec[i, vocab[h]] = 1.0
        k_ref_ref = ref_vec @ ref_vec.T
        k_ref_gen = ref_vec @ gen_vec.T
        k_gen_gen = gen_vec @ gen_vec.T
        return float(k_gen_gen.mean() + k_ref_ref.mean() - 2.0 * k_ref_gen.mean())


def _nspdk_mmd(
    reference_graphs: Sequence[nx.Graph],
    generated_graphs: Sequence[nx.Graph],
    radius: int = 1,
    distance: int = 4,
    nbits: int = 11,
    n_jobs: Optional[int] = None,
) -> float:
    def _ensure_discrete_labels(graphs: Sequence[nx.Graph]) -> Sequence[nx.Graph]:
        fixed = []
        for g in graphs:
            h = nx.Graph(g).copy()
            for u in h.nodes():
                raw = h.nodes[u].get("label", h.nodes[u].get("display_label", h.nodes[u].get("true_label", 0)))
                try:
                    lab = int(raw)
                except Exception:
                    lab = 0
                # Eden/NSPDK path expects string-like discrete labels.
                h.nodes[u]["label"] = str(lab)
            for u, v in h.edges():
                eraw = h.edges[u, v].get("label", h.edges[u, v].get("true_label", 1))
                try:
                    elab = int(eraw)
                except Exception:
                    elab = 1
                h.edges[u, v]["label"] = str(elab)
            fixed.append(h)
        return fixed

    reference_graphs = _ensure_discrete_labels(reference_graphs)
    generated_graphs = _ensure_discrete_labels(generated_graphs)

    # Try Eden NSPDK first.
    try:
        from eden.graph import vectorize

        ref_x = vectorize(reference_graphs, complexity=max(radius, distance), discrete=True)
        gen_x = vectorize(generated_graphs, complexity=max(radius, distance), discrete=True)
        k_ref_ref = pairwise_kernels(ref_x, ref_x, metric="linear", n_jobs=n_jobs)
        k_gen_gen = pairwise_kernels(gen_x, gen_x, metric="linear", n_jobs=n_jobs)
        k_ref_gen = pairwise_kernels(ref_x, gen_x, metric="linear", n_jobs=n_jobs)
        return float(k_gen_gen.mean() + k_ref_ref.mean() - 2.0 * k_ref_gen.mean())
    except Exception:
        # Fallback to local NSPPK vectorizer.
        from coco_grape.graph_vectorizer.nsppk import NSPPK

        vec = NSPPK(
            radius=int(radius),
            distance=int(distance),
            connector=0,
            nbits=int(nbits),
            dense=False,
            parallel=False,
        )
        ref_x = _as_embedding_matrix(vec.fit_transform(reference_graphs))
        gen_x = _as_embedding_matrix(vec.transform(generated_graphs))
        if sparse.issparse(ref_x):
            ref_x = ref_x.tocsr()
        if sparse.issparse(gen_x):
            gen_x = gen_x.tocsr()
        k_ref_ref = pairwise_kernels(ref_x, ref_x, metric="linear")
        k_gen_gen = pairwise_kernels(gen_x, gen_x, metric="linear")
        k_ref_gen = pairwise_kernels(ref_x, gen_x, metric="linear")
        return float(k_gen_gen.mean() + k_ref_ref.mean() - 2.0 * k_ref_gen.mean())


def evaluate_graphs_classical_mmd(
    reference_graphs: Sequence[nx.Graph],
    generated_graphs: Sequence[nx.Graph],
    methods: Optional[Sequence[str]] = None,
    wl_n_iter: int = 4,
    nspdk_radius: int = 1,
    nspdk_distance: int = 4,
    nspdk_nbits: int = 11,
    cluster_bins: int = 100,
    spectral_bins: int = 200,
    orca_bin: Optional[str] = None,
    include_timing: bool = True,
) -> Dict[str, float]:
    """
    Classical graph-distribution MMD metrics with selectable methods.
    Supported methods:
    - 'wl'
    - 'nspdk'
    - 'degree'
    - 'cluster'
    - 'spectral'
    - 'orca' (orbit-based, optional binary)
    """
    if methods is None:
        methods = ["wl", "nspdk", "degree", "cluster"]

    name_map = {
        "clustering": "cluster",
        "orbits": "orca",
    }
    selected = []
    for m in methods:
        mm = name_map.get(str(m).strip().lower(), str(m).strip().lower())
        if mm in {"wl", "nspdk", "degree", "cluster", "spectral", "orca"}:
            selected.append(mm)

    ref_graphs = _to_undirected_nonempty(reference_graphs)
    gen_graphs = _to_undirected_nonempty(generated_graphs)
    out: Dict[str, float] = {}
    if len(ref_graphs) == 0 or len(gen_graphs) == 0:
        for mm in selected:
            out[f"{mm}_mmd"] = float("nan")
            if include_timing:
                out[f"{mm}_mmd_time"] = 0.0
        return out

    for mm in selected:
        t0 = time.time()
        key = f"{mm}_mmd"
        try:
            if mm == "wl":
                out[key] = float(_wl_mmd(ref_graphs, gen_graphs, n_iter=int(wl_n_iter)))
            elif mm == "nspdk":
                out[key] = float(
                    _nspdk_mmd(
                        ref_graphs,
                        gen_graphs,
                        radius=int(nspdk_radius),
                        distance=int(nspdk_distance),
                        nbits=int(nspdk_nbits),
                    )
                )
            elif mm == "degree":
                out[key] = float(
                    _rbf_mmd_from_distance(
                        _degree_histograms(ref_graphs),
                        _degree_histograms(gen_graphs),
                        sigma=1.0,
                        distance="l1tv",
                    )
                )
            elif mm == "cluster":
                out[key] = float(
                    _rbf_mmd_from_distance(
                        _cluster_histograms(ref_graphs, bins=int(cluster_bins)),
                        _cluster_histograms(gen_graphs, bins=int(cluster_bins)),
                        sigma=0.1,
                        distance="l1tv",
                    )
                )
            elif mm == "spectral":
                out[key] = float(
                    _rbf_mmd_from_distance(
                        _spectral_histograms(ref_graphs, bins=int(spectral_bins)),
                        _spectral_histograms(gen_graphs, bins=int(spectral_bins)),
                        sigma=1.0,
                        distance="l1tv",
                    )
                )
            elif mm == "orca":
                orbit_ref = _orbit_vectors(ref_graphs, _find_orca_binary(orca_bin))
                orbit_gen = _orbit_vectors(gen_graphs, _find_orca_binary(orca_bin))
                out[key] = float(_rbf_mmd_from_distance(orbit_ref, orbit_gen, sigma=30.0, distance="l2"))
        except Exception:
            out[key] = float("nan")
        if include_timing:
            out[f"{mm}_mmd_time"] = time.time() - t0
    return out


def _vectorizer_fit_transform(vectorizer: Any, graphs: Sequence[nx.Graph], targets: Optional[Sequence[Any]] = None):
    if hasattr(vectorizer, "fit_transform"):
        if targets is None:
            return vectorizer.fit_transform(graphs)
        try:
            return vectorizer.fit_transform(graphs, targets)
        except TypeError:
            return vectorizer.fit_transform(graphs)
    raise ValueError("Vectorizer must implement fit_transform().")


def _vectorizer_transform(vectorizer: Any, graphs: Sequence[nx.Graph], targets: Optional[Sequence[Any]] = None):
    if hasattr(vectorizer, "transform"):
        if targets is None:
            return vectorizer.transform(graphs)
        try:
            return vectorizer.transform(graphs, targets)
        except TypeError:
            return vectorizer.transform(graphs)
    raise ValueError("Vectorizer must implement transform().")


def _as_embedding_matrix(out: Any) -> np.ndarray:
    """
    Normalize vectorizer outputs:
    - X
    - (X, y)
    """
    x = out[0] if isinstance(out, (tuple, list)) and len(out) > 0 else out

    if sparse.issparse(x):
        return x

    try:
        import torch

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass

    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def evaluate_graphs_with_vectorizer(
    reference_graphs: Sequence[nx.Graph],
    generated_graphs: Sequence[nx.Graph],
    vectorizer: Any,
    reference_targets: Optional[Sequence[Any]] = None,
    generated_targets: Optional[Sequence[Any]] = None,
    fit_on_reference: bool = True,
    nearest_k: int = 5,
) -> Dict[str, float]:
    """
    Compare two graph sets with user-provided vectorizer embeddings then evaluate
    FID/KID/PRDC/MMD on those embeddings.
    """
    if fit_on_reference:
        ref_out = _vectorizer_fit_transform(vectorizer, reference_graphs, reference_targets)
        gen_out = _vectorizer_transform(vectorizer, generated_graphs, generated_targets)
    else:
        all_graphs = list(reference_graphs) + list(generated_graphs)
        all_targets = None
        if reference_targets is not None and generated_targets is not None:
            all_targets = list(reference_targets) + list(generated_targets)
        all_out = _vectorizer_fit_transform(vectorizer, all_graphs, all_targets)
        all_x = _as_embedding_matrix(all_out)
        ref_x = all_x[: len(reference_graphs)]
        gen_x = all_x[len(reference_graphs) :]
        return evaluate_embedding_distributions(ref_x, gen_x, nearest_k=nearest_k, include_timing=True)

    ref_x = _as_embedding_matrix(ref_out)
    gen_x = _as_embedding_matrix(gen_out)
    return evaluate_embedding_distributions(ref_x, gen_x, nearest_k=nearest_k, include_timing=True)


def evaluate_generated_by_source(
    generated_by_source: Dict[str, Dict[str, Tuple[Sequence[nx.Graph], Sequence[Any]]]],
    reference_splits: Dict[str, Tuple[Sequence[nx.Graph], Sequence[Any]]],
    use_structural: bool = True,
    vectorizer: Optional[Any] = None,
    nearest_k: int = 5,
    use_classical_graph_mmd: bool = False,
    classical_methods: Optional[Sequence[str]] = None,
    wl_n_iter: int = 4,
    nspdk_radius: int = 1,
    nspdk_distance: int = 4,
    nspdk_nbits: int = 11,
    cluster_bins: int = 100,
    spectral_bins: int = 200,
    orca_bin: Optional[str] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Paired evaluation by source.
    Example pairing:
    - source 'train1'      -> reference_splits['train1']
    - source 'train_whole' -> reference_splits['train_whole']
    """
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for source, model_map in generated_by_source.items():
        if source not in reference_splits:
            continue
        ref_graphs, ref_targets = reference_splits[source]
        out[source] = {}
        for model_name, (gen_graphs, gen_targets) in model_map.items():
            if use_classical_graph_mmd:
                metrics = evaluate_graphs_classical_mmd(
                    reference_graphs=ref_graphs,
                    generated_graphs=gen_graphs,
                    methods=classical_methods,
                    wl_n_iter=wl_n_iter,
                    nspdk_radius=nspdk_radius,
                    nspdk_distance=nspdk_distance,
                    nspdk_nbits=nspdk_nbits,
                    cluster_bins=cluster_bins,
                    spectral_bins=spectral_bins,
                    orca_bin=orca_bin,
                    include_timing=True,
                )
            elif use_structural:
                metrics = evaluate_graphs_structural(ref_graphs, gen_graphs, nearest_k=nearest_k)
            elif vectorizer is not None:
                metrics = evaluate_graphs_with_vectorizer(
                    reference_graphs=ref_graphs,
                    generated_graphs=gen_graphs,
                    vectorizer=vectorizer,
                    reference_targets=ref_targets,
                    generated_targets=gen_targets,
                    nearest_k=nearest_k,
                )
            else:
                raise ValueError("Set use_structural=True or provide vectorizer.")
            out[source][model_name] = metrics
    return out
