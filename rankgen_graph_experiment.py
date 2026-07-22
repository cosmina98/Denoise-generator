"""Utilities for RankGen-style graph-generator experiments.

The intended split is:
  - train1: real graphs used to train the generator
  - train2: held-out real reference set, not used by the generator
  - test: held-out labelled set for the downstream classifier
  - generated: graphs produced by a generator trained on train1

The helper also checks that train2 adds measurable downstream value over
train1 before using it as the RankGen reference set.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from coco_grape.data_processor.generative.performance_measure import (
    adjusted_balanced_accuracy_score,
    concrete_discriminative_generative_quality_score,
    discriminative_generative_quality_score_rank,
    exploitable_exchangeability_and_exploitable_creativity_rank,
)
from coco_grape.data_processor.generative.unified_generator_eval import generate_graphs_same_way
from coco_grape.graph_vectorizer.nsppk import NSPPK


@dataclass
class RankGenSplit:
    train1_graphs: List[nx.Graph]
    train1_targets: List[Hashable]
    train2_graphs: List[nx.Graph]
    train2_targets: List[Hashable]
    test_graphs: List[nx.Graph]
    test_targets: List[Hashable]
    train_whole_graphs: List[nx.Graph]
    train_whole_targets: List[Hashable]
    selected_train1_fraction: float
    reference_gain: float
    train1_score: float
    train1_plus_train2_score: float
    gap_metric: str


def default_nsppk_factory(
    *,
    radius: int = 2,
    distance: int = 4,
    connector: int = 0,
    nbits: int = 12,
    dense: bool = False,
    parallel: bool = False,
) -> Callable[[], NSPPK]:
    """Return a factory so every evaluation gets a fresh vectorizer."""

    def _make() -> NSPPK:
        return NSPPK(
            radius=radius,
            distance=distance,
            connector=connector,
            nbits=nbits,
            dense=dense,
            parallel=parallel,
        )

    return _make


def default_classifier_factory(
    *,
    seed: int = 42,
    n_estimators: int = 300,
) -> Callable[[int], ExtraTreesClassifier]:
    """Return a deterministic ExtraTrees classifier factory."""

    def _make(offset: int = 0) -> ExtraTreesClassifier:
        return ExtraTreesClassifier(
            n_estimators=n_estimators,
            random_state=int(seed) + int(offset),
            n_jobs=-1,
        )

    return _make


def macro_f1_score(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    """Macro-F1 scorer for binary or multiclass graph-label tasks."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def score_predictions(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    metric: str = "macro_f1",
) -> float:
    metric = str(metric).strip().lower()
    if metric in {"macro_f1", "f1", "f1_macro"}:
        return macro_f1_score(y_true, y_pred)
    if metric in {"balanced_accuracy", "bal_acc", "balanced_acc"}:
        return float(balanced_accuracy_score(y_true, y_pred))
    if metric in {"adjusted_balanced_accuracy", "adjusted_bal_acc"}:
        return float(adjusted_balanced_accuracy_score(y_true, y_pred))
    if metric in {"accuracy", "acc"}:
        return float(accuracy_score(y_true, y_pred))
    raise ValueError(
        "metric must be one of: macro_f1, balanced_accuracy, "
        "adjusted_balanced_accuracy, accuracy"
    )


def load_graphs_targets_pickle(
    graphs_path: str | Path,
    targets_path: Optional[str | Path] = None,
) -> Tuple[List[nx.Graph], List[Hashable]]:
    """Load generated or real graph lists from pickle files."""
    import pickle

    graphs_path = Path(graphs_path)
    with graphs_path.open("rb") as handle:
        payload = pickle.load(handle)

    if targets_path is not None:
        with Path(targets_path).open("rb") as handle:
            targets = pickle.load(handle)
        return list(payload), list(targets)

    if isinstance(payload, Mapping):
        graphs = payload.get("graphs", payload.get("generated_graphs"))
        targets = payload.get("targets", payload.get("generated_targets"))
        if graphs is None:
            raise ValueError("Pickle mapping must contain graphs/generated_graphs.")
        if targets is None:
            targets = [None] * len(graphs)
        return list(graphs), list(targets)

    if (
        isinstance(payload, tuple)
        and len(payload) >= 2
    ):
        return list(payload[0]), list(payload[1])

    return list(payload), [None] * len(payload)


def save_graphs_targets_pickle(
    graphs: Sequence[nx.Graph],
    targets: Sequence[Hashable],
    path: str | Path,
) -> Path:
    """Save a compact graph/target payload for another RankGen notebook."""
    import pickle

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({"graphs": list(graphs), "targets": list(targets)}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def make_artificial_binary_graph_dataset(
    *,
    n_per_class: int = 500,
    alphabet_size: int = 30,
    target_size: int = 5,
    context_size: int = 5,
    n_link_edges: int = 1,
    positive_target_type: str = "cycle",
    positive_context_type: str = "cycle",
    negative_target_type: str = "tree",
    negative_context_type: str = "cycle",
    canonicalise: bool = True,
) -> Tuple[List[nx.Graph], List[int]]:
    """Build the binary artificial graph task used by the generator notebooks."""
    from coco_grape.utils.artificial_graph_constructor import ArtificialGraphDatasetConstructor

    graphs, targets = ArtificialGraphDatasetConstructor(
        graph_generator_target_type_pos=positive_target_type,
        graph_generator_context_type_pos=positive_context_type,
        graph_generator_target_type_neg=negative_target_type,
        graph_generator_context_type_neg=negative_context_type,
        target_size_pos=int(target_size),
        context_size_pos=int(context_size),
        alphabet_size_pos=int(alphabet_size),
        n_link_edges_pos=int(n_link_edges),
        target_size_neg=int(target_size),
        context_size_neg=int(context_size),
        alphabet_size_neg=int(alphabet_size),
        n_link_edges_neg=int(n_link_edges),
    ).sample(int(n_per_class))

    graphs = normalise_graph_labels(graphs)
    if canonicalise:
        try:
            from coco_grape.utils.canonical_order import canonicalise

            graphs = [canonicalise(graph) for graph in graphs]
        except Exception:
            pass
    return graphs, [int(target) for target in targets]


def resolve_rankgen_dataset(
    *,
    dataset_choice: str = "artificial_binary",
    dataset_pickle: Optional[str | Path] = None,
    targets_pickle: Optional[str | Path] = None,
    graphs: Optional[Sequence[nx.Graph]] = None,
    targets: Optional[Sequence[Hashable]] = None,
    artificial_n_per_class: int = 500,
    artificial_alphabet_size: int = 30,
    artificial_target_size: int = 5,
    artificial_context_size: int = 5,
    artificial_link_edges: int = 1,
    artificial_positive_target_type: str = "cycle",
    artificial_positive_context_type: str = "cycle",
    artificial_negative_target_type: str = "tree",
    artificial_negative_context_type: str = "cycle",
    artificial_canonicalise: bool = True,
) -> Tuple[List[nx.Graph], List[Hashable]]:
    """Resolve the labelled graph dataset for a fresh notebook kernel."""
    if dataset_pickle is not None:
        return load_graphs_targets_pickle(dataset_pickle, targets_pickle)

    if graphs is not None and targets is not None:
        return normalise_graph_labels(graphs), list(targets)

    choice = str(dataset_choice).strip().lower()
    if choice in {"artificial_binary", "binary", "synthetic_binary"}:
        return make_artificial_binary_graph_dataset(
            n_per_class=artificial_n_per_class,
            alphabet_size=artificial_alphabet_size,
            target_size=artificial_target_size,
            context_size=artificial_context_size,
            n_link_edges=artificial_link_edges,
            positive_target_type=artificial_positive_target_type,
            positive_context_type=artificial_positive_context_type,
            negative_target_type=artificial_negative_target_type,
            negative_context_type=artificial_negative_context_type,
            canonicalise=artificial_canonicalise,
        )

    raise ValueError(
        "No dataset available. Set dataset_pickle/targets_pickle, pass graphs/targets, "
        "or use dataset_choice='artificial_binary'."
    )


def save_rankgen_split(
    split: RankGenSplit,
    path: str | Path,
) -> Path:
    """Persist the exact train1/train2/test split for all generator notebooks."""
    import pickle

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(split, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_rankgen_split(path: str | Path) -> RankGenSplit:
    """Load a previously saved RankGenSplit."""
    import pickle

    path = Path(path)
    with path.open("rb") as handle:
        split = pickle.load(handle)
    if not isinstance(split, RankGenSplit):
        raise TypeError(f"Expected RankGenSplit in {path}, found {type(split)!r}.")
    return split


def normalise_graph_labels(
    graphs: Sequence[nx.Graph],
    *,
    default_node_label: int = 0,
    default_edge_label: int = 0,
) -> List[nx.Graph]:
    """Copy graphs and put node/edge labels under the common keys used here."""
    out: List[nx.Graph] = []
    for graph in graphs:
        h = nx.Graph(graph).copy()
        for _, attrs in h.nodes(data=True):
            raw = attrs.get(
                "label",
                attrs.get("display_label", attrs.get("true_label", attrs.get("attr", default_node_label))),
            )
            attrs["label"] = _coerce_label(raw, default_node_label)
            attrs["display_label"] = attrs["label"]
            attrs["true_label"] = attrs["label"]
            attrs["attr"] = attrs["label"]
        for _, _, attrs in h.edges(data=True):
            raw = attrs.get(
                "label",
                attrs.get("display_label", attrs.get("true_label", attrs.get("edge_attr", default_edge_label))),
            )
            attrs["label"] = _coerce_label(raw, default_edge_label)
            attrs["display_label"] = attrs["label"]
            attrs["true_label"] = attrs["label"]
            attrs["edge_attr"] = attrs["label"]
        out.append(h)
    return out


def make_rankgen_split(
    graphs: Sequence[nx.Graph],
    targets: Sequence[Hashable],
    *,
    test_size: float = 0.2,
    train1_fraction: float = 0.5,
    seed: int = 42,
    normalise_labels: bool = True,
) -> Tuple[List[nx.Graph], List[Hashable], List[nx.Graph], List[Hashable], List[nx.Graph], List[Hashable]]:
    """Create train1/train2/test with stratification when possible."""
    _validate_targets(targets)
    if len(graphs) != len(targets):
        raise ValueError("graphs and targets must have the same length.")
    if not (0.0 < float(test_size) < 1.0):
        raise ValueError("test_size must be between 0 and 1.")
    if not (0.0 < float(train1_fraction) < 1.0):
        raise ValueError("train1_fraction must be between 0 and 1.")

    graph_list = normalise_graph_labels(graphs) if normalise_labels else [nx.Graph(g).copy() for g in graphs]
    target_list = list(targets)
    indices = np.arange(len(graph_list))

    train_idx, test_idx = _safe_train_test_split_indices(
        indices,
        target_list,
        test_size=float(test_size),
        seed=int(seed),
    )
    train_targets = [target_list[i] for i in train_idx]
    split1_idx, split2_idx = _safe_train_test_split_indices(
        train_idx,
        train_targets,
        test_size=1.0 - float(train1_fraction),
        seed=int(seed) + 17,
    )

    train1_graphs = [graph_list[i] for i in split1_idx]
    train1_targets = [target_list[i] for i in split1_idx]
    train2_graphs = [graph_list[i] for i in split2_idx]
    train2_targets = [target_list[i] for i in split2_idx]
    test_graphs = [graph_list[i] for i in test_idx]
    test_targets = [target_list[i] for i in test_idx]
    return train1_graphs, train1_targets, train2_graphs, train2_targets, test_graphs, test_targets


def evaluate_reference_gap(
    *,
    train1_graphs: Sequence[nx.Graph],
    train1_targets: Sequence[Hashable],
    train2_graphs: Sequence[nx.Graph],
    train2_targets: Sequence[Hashable],
    test_graphs: Sequence[nx.Graph],
    test_targets: Sequence[Hashable],
    vectorizer_factory: Optional[Callable[[], Any]] = None,
    classifier_factory: Optional[Callable[[int], Any]] = None,
    metric: str = "macro_f1",
    repeats: int = 3,
) -> Dict[str, float]:
    """Evaluate train1 vs train1+train2 on the held-out test set."""
    if vectorizer_factory is None:
        vectorizer_factory = default_nsppk_factory()
    if classifier_factory is None:
        classifier_factory = default_classifier_factory()

    rows = []
    for repeat in range(int(repeats)):
        train1_score = _fit_predictive_score(
            train1_graphs,
            train1_targets,
            test_graphs,
            test_targets,
            vectorizer_factory=vectorizer_factory,
            classifier=classifier_factory(repeat),
            metric=metric,
        )
        combined_score = _fit_predictive_score(
            list(train1_graphs) + list(train2_graphs),
            list(train1_targets) + list(train2_targets),
            test_graphs,
            test_targets,
            vectorizer_factory=vectorizer_factory,
            classifier=classifier_factory(10_000 + repeat),
            metric=metric,
        )
        rows.append((train1_score, combined_score, combined_score - train1_score))

    arr = np.asarray(rows, dtype=float)
    return {
        "train1_score": float(arr[:, 0].mean()),
        "train1_plus_train2_score": float(arr[:, 1].mean()),
        "reference_gain": float(arr[:, 2].mean()),
        "train1_score_std": float(arr[:, 0].std()),
        "train1_plus_train2_score_std": float(arr[:, 1].std()),
        "reference_gain_std": float(arr[:, 2].std()),
    }


def make_rankgen_split_with_gap(
    graphs: Sequence[nx.Graph],
    targets: Sequence[Hashable],
    *,
    test_size: float = 0.2,
    candidate_train1_fractions: Sequence[float] = (0.50, 0.40, 0.33, 0.25, 0.20, 0.15, 0.10),
    min_reference_gain: float = 0.05,
    seed: int = 42,
    vectorizer_factory: Optional[Callable[[], Any]] = None,
    classifier_factory: Optional[Callable[[int], Any]] = None,
    metric: str = "macro_f1",
    repeats: int = 3,
    raise_on_fail: bool = True,
) -> Tuple[RankGenSplit, pd.DataFrame]:
    """
    Search split sizes and keep the largest train1 fraction with enough gain.

    The gain is:
        score(train1 + train2 -> test) - score(train1 -> test)
    A 0.05 gain corresponds to a five percentage-point improvement.
    """
    if vectorizer_factory is None:
        vectorizer_factory = default_nsppk_factory()
    if classifier_factory is None:
        classifier_factory = default_classifier_factory(seed=seed)

    rows = []
    split_payloads = {}
    for fraction in candidate_train1_fractions:
        try:
            split = make_rankgen_split(
                graphs,
                targets,
                test_size=test_size,
                train1_fraction=float(fraction),
                seed=seed,
            )
            gap = evaluate_reference_gap(
                train1_graphs=split[0],
                train1_targets=split[1],
                train2_graphs=split[2],
                train2_targets=split[3],
                test_graphs=split[4],
                test_targets=split[5],
                vectorizer_factory=vectorizer_factory,
                classifier_factory=classifier_factory,
                metric=metric,
                repeats=repeats,
            )
            row = {
                "train1_fraction": float(fraction),
                "train1_n": len(split[0]),
                "train2_n": len(split[2]),
                "test_n": len(split[4]),
                "passes_min_gain": bool(gap["reference_gain"] >= float(min_reference_gain)),
            }
            row.update(gap)
            rows.append(row)
            split_payloads[float(fraction)] = split
        except Exception as exc:
            rows.append({
                "train1_fraction": float(fraction),
                "error": repr(exc),
                "passes_min_gain": False,
            })

    gap_table = pd.DataFrame(rows)
    ok = gap_table[gap_table["passes_min_gain"] == True]  # noqa: E712
    if len(ok) == 0:
        best_row = gap_table.sort_values("reference_gain", ascending=False, na_position="last").iloc[0]
        message = (
            f"No candidate split reached the required {min_reference_gain:.3f} "
            f"reference gain for {metric}. Best gain was "
            f"{float(best_row.get('reference_gain', np.nan)):.3f} at "
            f"train1_fraction={float(best_row['train1_fraction']):.2f}."
        )
        if raise_on_fail:
            raise RuntimeError(message)
        selected_row = best_row
    else:
        selected_row = ok.sort_values("train1_fraction", ascending=False).iloc[0]

    selected_fraction = float(selected_row["train1_fraction"])
    split = split_payloads[selected_fraction]
    train1_g, train1_y, train2_g, train2_y, test_g, test_y = split
    train_whole_g = list(train1_g) + list(train2_g)
    train_whole_y = list(train1_y) + list(train2_y)
    selected = RankGenSplit(
        train1_graphs=train1_g,
        train1_targets=train1_y,
        train2_graphs=train2_g,
        train2_targets=train2_y,
        test_graphs=test_g,
        test_targets=test_y,
        train_whole_graphs=train_whole_g,
        train_whole_targets=train_whole_y,
        selected_train1_fraction=selected_fraction,
        reference_gain=float(selected_row.get("reference_gain", np.nan)),
        train1_score=float(selected_row.get("train1_score", np.nan)),
        train1_plus_train2_score=float(selected_row.get("train1_plus_train2_score", np.nan)),
        gap_metric=str(metric),
    )
    return selected, gap_table


def generation_counts_from_targets(targets: Sequence[Hashable]) -> Dict[Hashable, int]:
    """Return per-class generated counts matching train1."""
    _validate_targets(targets)
    return dict(Counter(targets))


def generate_from_model(
    model: Any,
    *,
    model_kind: str,
    train1_graphs: Sequence[nx.Graph],
    train1_targets: Sequence[Hashable],
    seed: int = 42,
    denoise_generation_mode: str = "seeded",
) -> Tuple[List[nx.Graph], List[Hashable]]:
    """Generate class counts matching train1 from a fitted generator."""
    kind = _normalise_model_kind(model_kind)
    class_counts = generation_counts_from_targets(train1_targets)
    generated_graphs: List[nx.Graph] = []
    generated_targets: List[Hashable] = []

    for offset, (cls, count) in enumerate(sorted(class_counts.items(), key=lambda item: repr(item[0]))):
        graphs, targets = generate_graphs_same_way(
            model,
            model_kind=kind,
            train_graphs=train1_graphs,
            train_targets=train1_targets,
            n_per_class=int(count),
            seed=int(seed) + offset,
            class_values=[cls],
            denoise_generation_mode=denoise_generation_mode,
        )
        generated_graphs.extend(graphs[:count])
        generated_targets.extend(targets[:count])

    return normalise_graph_labels(generated_graphs), list(generated_targets)


def transform_rankgen_data(
    *,
    generated_graphs: Sequence[nx.Graph],
    generated_targets: Sequence[Hashable],
    split: RankGenSplit,
    vectorizer_factory: Optional[Callable[[], Any]] = None,
    fit_scope: str = "train1_train2",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorize generated/train1/train2/test in one compatible feature space."""
    if vectorizer_factory is None:
        vectorizer_factory = default_nsppk_factory()

    generated_graphs = normalise_graph_labels(generated_graphs)
    train1_graphs = normalise_graph_labels(split.train1_graphs)
    train2_graphs = normalise_graph_labels(split.train2_graphs)
    test_graphs = normalise_graph_labels(split.test_graphs)

    generated_targets = list(generated_targets)
    train1_targets = list(split.train1_targets)
    train2_targets = list(split.train2_targets)
    test_targets = list(split.test_targets)

    vec = vectorizer_factory()
    fit_scope = str(fit_scope).strip().lower()
    if fit_scope not in {"train1_train2", "generated_train1_train2", "all"}:
        raise ValueError("fit_scope must be 'train1_train2', 'generated_train1_train2', or 'all'.")

    if fit_scope == "all" or not hasattr(vec, "transform"):
        all_graphs = generated_graphs + train1_graphs + train2_graphs + test_graphs
        all_targets = generated_targets + train1_targets + train2_targets + test_targets
        all_x = _fit_transform_vectorizer(vec, all_graphs, all_targets)
        n_gen = len(generated_graphs)
        n_t1 = len(train1_graphs)
        n_t2 = len(train2_graphs)
        generated_x = all_x[:n_gen]
        train1_x = all_x[n_gen:n_gen + n_t1]
        train2_x = all_x[n_gen + n_t1:n_gen + n_t1 + n_t2]
        test_x = all_x[n_gen + n_t1 + n_t2:]
    elif fit_scope == "train1_train2":
        fit_graphs = train1_graphs + train2_graphs
        fit_targets = train1_targets + train2_targets
        fit_x = _fit_transform_vectorizer(vec, fit_graphs, fit_targets)
        generated_x = _transform_vectorizer(vec, generated_graphs, generated_targets)
        test_x = _transform_vectorizer(vec, test_graphs, test_targets)
        n_t1 = len(train1_graphs)
        train1_x = fit_x[:n_t1]
        train2_x = fit_x[n_t1:]
    else:
        fit_graphs = generated_graphs + train1_graphs + train2_graphs
        fit_targets = generated_targets + train1_targets + train2_targets
        fit_x = _fit_transform_vectorizer(vec, fit_graphs, fit_targets)
        test_x = _transform_vectorizer(vec, test_graphs, test_targets)
        n_gen = len(generated_graphs)
        n_t1 = len(train1_graphs)
        generated_x = fit_x[:n_gen]
        train1_x = fit_x[n_gen:n_gen + n_t1]
        train2_x = fit_x[n_gen + n_t1:]

    return (
        _as_dense_2d(generated_x),
        np.asarray(generated_targets),
        _as_dense_2d(train1_x),
        np.asarray(train1_targets),
        _as_dense_2d(train2_x),
        np.asarray(train2_targets),
        _as_dense_2d(test_x),
        np.asarray(test_targets),
    )


def run_rankgen_for_generated(
    *,
    model_name: str,
    generated_graphs: Sequence[nx.Graph],
    generated_targets: Sequence[Hashable],
    split: RankGenSplit,
    vectorizer_factory: Optional[Callable[[], Any]] = None,
    n_iterations: int = 100,
    fraction: float = 0.8,
    use_resampling: bool = True,
    use_replacement: bool = False,
    estimator_n: int = 300,
    score_metric: str = "macro_f1",
    parallel: bool = False,
    verbose: int = 1,
    fit_scope: str = "train1_train2",
) -> Dict[str, Any]:
    """Run RankGen/DGQS on one generated graph set."""
    scorer = _rankgen_score_function(score_metric)
    data = transform_rankgen_data(
        generated_graphs=generated_graphs,
        generated_targets=generated_targets,
        split=split,
        vectorizer_factory=vectorizer_factory,
        fit_scope=fit_scope,
    )
    (
        exchangeability,
        creativity,
        scores,
        predictive_performances,
        exchangeability_std,
        creativity_std,
        scores_std,
        predictive_performances_std,
    ) = concrete_discriminative_generative_quality_score(
        *data,
        n_iterations=int(n_iterations),
        use_resampling=bool(use_resampling),
        use_replacement=bool(use_replacement),
        fraction=float(fraction),
        data_estimator=ExtraTreesClassifier(n_estimators=int(estimator_n), n_jobs=-1),
        discriminative_performance_func=scorer,
        verbose=int(verbose),
        parallel=bool(parallel),
    )

    quality, utility, indistinguishability, similarity = [float(x) for x in scores]
    quality_std, utility_std, indistinguishability_std, similarity_std = [float(x) for x in scores_std]
    (
        perf_real_train1,
        perf_generated,
        perf_real_train1_plus_train2,
        perf_real_train1_plus_generated,
        perf_real_vs_generated,
    ) = [float(x) for x in predictive_performances]
    (
        perf_real_train1_std,
        perf_generated_std,
        perf_real_train1_plus_train2_std,
        perf_real_train1_plus_generated_std,
        perf_real_vs_generated_std,
    ) = [float(x) for x in predictive_performances_std]

    return {
        "model": str(model_name),
        "n_generated": int(len(generated_graphs)),
        "n_train1": int(len(split.train1_graphs)),
        "n_train2": int(len(split.train2_graphs)),
        "n_test": int(len(split.test_graphs)),
        "selected_train1_fraction": float(split.selected_train1_fraction),
        "split_reference_gain": float(split.reference_gain),
        "exchangeability": float(exchangeability),
        "creativity": float(creativity),
        "quality": quality,
        "utility": utility,
        "indistinguishability": indistinguishability,
        "similarity": similarity,
        "exchangeability_std": float(exchangeability_std),
        "creativity_std": float(creativity_std),
        "quality_std": quality_std,
        "utility_std": utility_std,
        "indistinguishability_std": indistinguishability_std,
        "similarity_std": similarity_std,
        "perf_real_train1": perf_real_train1,
        "perf_generated": perf_generated,
        "perf_real_train1_plus_train2": perf_real_train1_plus_train2,
        "perf_real_train1_plus_generated": perf_real_train1_plus_generated,
        "perf_real_vs_generated": perf_real_vs_generated,
        "perf_real_train1_std": perf_real_train1_std,
        "perf_generated_std": perf_generated_std,
        "perf_real_train1_plus_train2_std": perf_real_train1_plus_train2_std,
        "perf_real_train1_plus_generated_std": perf_real_train1_plus_generated_std,
        "perf_real_vs_generated_std": perf_real_vs_generated_std,
        "rankgen_score_metric": str(score_metric),
    }


def run_rankgen_many(
    generated_sets: Mapping[str, Tuple[Sequence[nx.Graph], Sequence[Hashable]]],
    *,
    split: RankGenSplit,
    vectorizer_factory: Optional[Callable[[], Any]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Run RankGen for several generators and add stochastic rank columns."""
    rows = []
    for model_name, (graphs, targets) in generated_sets.items():
        rows.append(
            run_rankgen_for_generated(
                model_name=model_name,
                generated_graphs=graphs,
                generated_targets=targets,
                split=split,
                vectorizer_factory=vectorizer_factory,
                **kwargs,
            )
        )
    return add_rank_columns(pd.DataFrame(rows))


def add_rank_columns(
    results: pd.DataFrame,
    *,
    n_iter: int = 5000,
    std_correction_factor: float = 0.1,
) -> pd.DataFrame:
    """Add RankGen stochastic dominance ranks to a result table."""
    if results is None or len(results) == 0:
        return pd.DataFrame()
    df = results.copy()
    base_cols = ["quality", "utility", "indistinguishability", "similarity"]
    base_std_cols = [f"{col}_std" for col in base_cols]
    exchange_cols = ["exchangeability", "creativity"]
    exchange_std_cols = [f"{col}_std" for col in exchange_cols]

    valid = df[base_cols + base_std_cols].notna().all(axis=1)
    if valid.any():
        scores_list = [
            (
                df.loc[idx, base_cols].to_numpy(dtype=float),
                df.loc[idx, base_std_cols].to_numpy(dtype=float),
            )
            for idx in df.index[valid]
        ]
        df.loc[df.index[valid], "rank_quality_utility_indist_similarity"] = (
            discriminative_generative_quality_score_rank(
                scores_list,
                n_iter=int(n_iter),
                std_correction_factor=float(std_correction_factor),
            )
        )

    valid_ec = df[exchange_cols + exchange_std_cols].notna().all(axis=1)
    if valid_ec.any():
        scores_list = [
            (
                df.loc[idx, exchange_cols].to_numpy(dtype=float),
                df.loc[idx, exchange_std_cols].to_numpy(dtype=float),
            )
            for idx in df.index[valid_ec]
        ]
        df.loc[df.index[valid_ec], "rank_exchangeability_creativity"] = (
            exploitable_exchangeability_and_exploitable_creativity_rank(
                scores_list,
                n_iter=int(n_iter),
                std_correction_factor=float(std_correction_factor),
            )
        )

    df["rank_exchangeability_only"] = df["exchangeability"].rank(ascending=False, method="min")
    return df.sort_values(["rank_quality_utility_indist_similarity", "rank_exchangeability_only", "model"])


def write_results(
    results: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Write a RankGen result table as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False)
    return path


def _coerce_label(raw: Any, default_value: int) -> Any:
    if raw is None:
        return int(default_value)
    if isinstance(raw, np.generic):
        raw = raw.item()
    if isinstance(raw, str):
        text = raw.strip()
        if text == "":
            return int(default_value)
        try:
            value = float(text)
            if abs(value - round(value)) < 1e-9:
                return int(round(value))
            return value
        except Exception:
            return text
    if isinstance(raw, (int, bool)):
        return int(raw)
    if isinstance(raw, float):
        if abs(raw - round(raw)) < 1e-9:
            return int(round(raw))
        return float(raw)
    try:
        return int(raw)
    except Exception:
        return int(default_value)


def _validate_targets(targets: Sequence[Hashable]) -> None:
    if targets is None:
        raise ValueError("RankGen needs supervised graph labels; targets is None.")
    target_list = list(targets)
    if len(target_list) == 0:
        raise ValueError("targets is empty.")
    if any(target is None for target in target_list):
        raise ValueError("RankGen needs supervised graph labels; found None targets.")
    if len(set(target_list)) < 2:
        raise ValueError("RankGen quality/utility needs at least two graph-label classes.")


def _safe_train_test_split_indices(
    indices: np.ndarray,
    targets: Sequence[Hashable],
    *,
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    stratify = list(targets) if _can_stratify(targets) else None
    try:
        return train_test_split(
            np.asarray(indices),
            test_size=float(test_size),
            random_state=int(seed),
            stratify=stratify,
        )
    except Exception:
        return train_test_split(
            np.asarray(indices),
            test_size=float(test_size),
            random_state=int(seed),
            stratify=None,
        )


def _can_stratify(targets: Sequence[Hashable]) -> bool:
    counts = Counter(targets)
    return len(counts) >= 2 and min(counts.values()) >= 2


def _fit_predictive_score(
    train_graphs: Sequence[nx.Graph],
    train_targets: Sequence[Hashable],
    test_graphs: Sequence[nx.Graph],
    test_targets: Sequence[Hashable],
    *,
    vectorizer_factory: Callable[[], Any],
    classifier: Any,
    metric: str,
) -> float:
    vec = vectorizer_factory()
    x_train = _fit_transform_vectorizer(vec, train_graphs, train_targets)
    x_test = _transform_vectorizer(vec, test_graphs, test_targets)
    classifier.fit(x_train, train_targets)
    predictions = classifier.predict(x_test)
    return score_predictions(test_targets, predictions, metric=metric)


def _fit_transform_vectorizer(
    vectorizer: Any,
    graphs: Sequence[nx.Graph],
    targets: Optional[Sequence[Hashable]] = None,
) -> Any:
    if targets is None:
        out = vectorizer.fit_transform(graphs)
    else:
        try:
            out = vectorizer.fit_transform(graphs, targets)
        except TypeError:
            out = vectorizer.fit_transform(graphs)
    return _matrix_from_vectorizer_output(out)


def _transform_vectorizer(
    vectorizer: Any,
    graphs: Sequence[nx.Graph],
    targets: Optional[Sequence[Hashable]] = None,
) -> Any:
    if targets is None:
        out = vectorizer.transform(graphs)
    else:
        try:
            out = vectorizer.transform(graphs, targets)
        except TypeError:
            out = vectorizer.transform(graphs)
    return _matrix_from_vectorizer_output(out)


def _matrix_from_vectorizer_output(out: Any) -> Any:
    if isinstance(out, (tuple, list)) and len(out) > 0:
        out = out[0]
    return out


def _as_dense_2d(x: Any) -> np.ndarray:
    if sparse.issparse(x):
        arr = x.toarray()
    else:
        arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return np.asarray(arr, dtype=float)


def _rankgen_score_function(metric: str) -> Callable[[Sequence[Any], Sequence[Any]], float]:
    metric = str(metric).strip().lower()
    if metric in {"macro_f1", "f1", "f1_macro"}:
        return macro_f1_score
    if metric in {"balanced_accuracy", "bal_acc", "balanced_acc"}:
        return balanced_accuracy_score
    if metric in {"adjusted_balanced_accuracy", "adjusted_bal_acc"}:
        return adjusted_balanced_accuracy_score
    if metric in {"accuracy", "acc"}:
        return accuracy_score
    raise ValueError(
        "score_metric must be one of: macro_f1, balanced_accuracy, "
        "adjusted_balanced_accuracy, accuracy"
    )


def _normalise_model_kind(model_kind: str) -> str:
    kind = str(model_kind).strip().lower()
    aliases = {
        "vae": "vgae",
        "gcdg": "denoise",
        "gcddg": "denoise",
        "my_generator": "denoise",
        "my-generator": "denoise",
    }
    return aliases.get(kind, kind)
