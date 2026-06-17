import copy
import hashlib
import json
import os
from collections import Counter

import dill
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from coco_grape.data_processor.generative.decompositional_encoder_decoder import (
    ConditionalNodeGeneratorModel,
)


def dense_condition_matrix(values):
    if sparse.issparse(values):
        return values.toarray().astype(np.float32)

    rows = []
    for value in values:
        row = value.toarray() if sparse.issparse(value) else np.asarray(value)
        rows.append(np.asarray(row, dtype=np.float32).reshape(-1))
    return np.stack(rows, axis=0)


def usable_targets(targets, n_graphs):
    if targets is None:
        return None
    values = list(targets)
    if len(values) != n_graphs or any(value is None for value in values):
        return None
    return values


def array_fingerprint(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def graph_dataset_fingerprint(graphs):
    digest = hashlib.sha256()
    node_match_key = "label"
    edge_match_key = "label"
    for graph in graphs:
        nodes = list(graph.nodes())
        index = {node: i for i, node in enumerate(nodes)}
        node_labels = [graph.nodes[node].get(node_match_key) for node in nodes]
        edges = sorted(
            (
                min(index[u], index[v]),
                max(index[u], index[v]),
                data.get(edge_match_key),
            )
            for u, v, data in graph.edges(data=True)
        )
        payload = {
            "directed": bool(graph.is_directed()),
            "nodes": node_labels,
            "edges": edges,
        }
        digest.update(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        )
    return digest.hexdigest()


def generator_config_fingerprint(generator):
    keys = (
        "latent_embedding_dimension",
        "number_of_transformer_layers",
        "transformer_attention_head_count",
        "transformer_dropout",
        "learning_rate",
        "maximum_epochs",
        "batch_size",
        "total_steps",
        "early_stop_metric",
        "early_stop_mode",
        "early_stop_patience",
        "early_stop_min_delta",
        "lambda_recon_importance",
        "lambda_degree_importance",
        "lambda_node_exist_importance",
        "lambda_edge_importance",
        "lambda_clean_edge_importance",
        "lambda_label_importance",
        "balance_degree_loss",
        "degree_class_weight_cap",
        "balance_label_loss",
        "label_class_weight_cap",
        "balance_edge_loss",
        "lambda_x0_importance",
        "lambda_condition_x0_importance",
        "condition_x0_sampling_blend",
        "noise_degree_factor",
        "noise_label_factor",
        "sigma_min",
        "sigma_max",
        "sampling_final_sigma",
        "denoise_discrete_channels",
        "discrete_diffusion_mode",
        "discrete_projection_sigma_threshold",
        "row_embedding_scale",
        "project_existence_during_sampling",
        "project_degree_during_sampling",
        "project_label_during_sampling",
        "use_condition_node_count",
        "use_condition_degree_histogram",
        "use_condition_label_histogram",
        "use_guidance",
        "random_row_permutation",
        "use_matched_row_loss",
        "use_joint_edge_denoising",
        "joint_edge_message_passing_steps",
        "joint_edge_delete_prob",
        "joint_edge_add_prob",
        "joint_edge_update_momentum",
        "joint_edge_threshold",
        "use_dim_reduction",
        "dim_reduction_method",
        "dim_reduction_components",
        "dim_reduction_keep_prefix",
    )
    payload = {key: getattr(generator, key, None) for key in keys}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def fit_ridge_adapter(z_train, c_train, alpha=10.0):
    z_scaler = StandardScaler().fit(z_train)
    c_scaler = StandardScaler().fit(c_train)
    model = Ridge(alpha=float(alpha))
    model.fit(z_scaler.transform(z_train), c_scaler.transform(c_train))
    return {
        "model": model,
        "z_scaler": z_scaler,
        "c_scaler": c_scaler,
    }


def predict_ridge_adapter(adapter, z):
    predicted = adapter["model"].predict(adapter["z_scaler"].transform(z))
    return adapter["c_scaler"].inverse_transform(predicted).astype(np.float32)


def raw_latent_condition_control(z_train, z_test, c_train):
    """Place standardized z directly in standardized C coordinates without learning."""
    z_scaler = StandardScaler().fit(z_train)
    c_scaler = StandardScaler().fit(c_train)
    z_scaled = z_scaler.transform(z_test)
    c_scaled = np.zeros((len(z_test), c_train.shape[1]), dtype=np.float32)
    width = min(z_scaled.shape[1], c_scaled.shape[1])
    c_scaled[:, :width] = z_scaled[:, :width]
    return c_scaler.inverse_transform(c_scaled).astype(np.float32)


def condition_regression_metrics(c_true, c_pred, prefix):
    return {
        "method": prefix,
        "condition_mae": float(mean_absolute_error(c_true, c_pred)),
        "condition_rmse": float(mean_squared_error(c_true, c_pred) ** 0.5),
        "condition_r2": float(
            r2_score(c_true, c_pred, multioutput="variance_weighted")
        ),
    }


def _validate_training_inputs(
    train_graphs,
    train_node_encodings,
    train_conditions,
):
    conditions = np.asarray(train_conditions, dtype=np.float32)
    n_graphs = len(train_graphs)
    if conditions.ndim != 2:
        raise ValueError(
            f"Conditions must be 2D [graphs, features], got {conditions.shape}."
        )
    if len(train_node_encodings) != n_graphs or conditions.shape[0] != n_graphs:
        raise ValueError(
            "Training count mismatch: "
            f"graphs={n_graphs}, node_encodings={len(train_node_encodings)}, "
            f"conditions={conditions.shape[0]}."
        )
    if not np.isfinite(conditions).all():
        raise ValueError("Training conditions contain NaN or infinite values.")

    feature_dims = set()
    for index, value in enumerate(train_node_encodings):
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] < 3:
            raise ValueError(
                f"Node encoding {index} must be 2D with at least 3 columns; "
                f"got {array.shape}."
            )
        if not np.isfinite(array).all():
            raise ValueError(
                f"Node encoding {index} contains NaN or infinite values."
            )
        feature_dims.add(int(array.shape[1]))
    if len(feature_dims) != 1:
        raise ValueError(
            f"Node feature-width mismatch across graphs: {sorted(feature_dims)}."
        )
    return conditions


def decode_with_condition_model(condition_model, node_decoder, conditions):
    conditions = np.asarray(conditions, dtype=np.float32)
    if conditions.ndim != 2:
        raise ValueError(
            f"Decode conditions must be 2D [graphs, features], got {conditions.shape}."
        )
    if not np.isfinite(conditions).all():
        raise ValueError("Decode conditions contain NaN or infinite values.")
    expected_dim = getattr(condition_model, "_expected_raw_condition_dim", None)
    if expected_dim is not None and conditions.shape[1] != int(expected_dim):
        raise ValueError(
            "Condition-width mismatch for this trained decoder: "
            f"expected {expected_dim}, got {conditions.shape[1]}."
        )
    node_encodings = condition_model.predict(conditions)

    old_provider = getattr(node_decoder, "_conditional_edge_provider", None)
    old_conditions = getattr(node_decoder, "_last_conditioning_vectors", None)
    node_decoder._conditional_edge_provider = condition_model
    node_decoder._last_conditioning_vectors = list(conditions)
    try:
        graphs = node_decoder.decode(
            node_encodings,
            conditional_graph_encodings=conditions,
        )
    finally:
        node_decoder._conditional_edge_provider = old_provider
        node_decoder._last_conditioning_vectors = old_conditions
    return graphs


def _checkpoint_path(root, name):
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, f"{name}.dill")


def fit_or_load_condition_model(
    *,
    name,
    template_generator,
    node_decoder,
    train_graphs,
    train_node_encodings,
    train_conditions,
    checkpoint_root,
    retrain=False,
    maximum_epochs=None,
):
    path = _checkpoint_path(checkpoint_root, name)
    train_conditions = _validate_training_inputs(
        train_graphs,
        train_node_encodings,
        train_conditions,
    )
    condition_component_cap = max(
        1,
        int(getattr(template_generator, "condition_dim_reduction_components", 128)),
    )
    reduce_conditions = train_conditions.shape[1] > condition_component_cap
    expected_metadata = {
        "condition_dim": int(train_conditions.shape[1]),
        "condition_reduction": bool(reduce_conditions),
        "condition_components": int(condition_component_cap),
        "node_rows": int(max(x.shape[0] for x in train_node_encodings)),
        "node_feature_dim": int(train_node_encodings[0].shape[1]),
        "label_max": int(
            max(np.asarray(x)[:, 2].max() for x in train_node_encodings)
        ),
        "dataset_fingerprint": graph_dataset_fingerprint(train_graphs),
        "condition_fingerprint": array_fingerprint(train_conditions),
        "generator_config_fingerprint": generator_config_fingerprint(
            template_generator
        ),
    }
    if not retrain and os.path.exists(path):
        with open(path, "rb") as handle:
            saved = dill.load(handle)
        if isinstance(saved, dict) and "model" in saved:
            saved_metadata = saved.get("metadata", {})
            if saved_metadata == expected_metadata:
                loaded_model = saved["model"]
                loaded_model._expected_raw_condition_dim = int(
                    expected_metadata["condition_dim"]
                )
                return loaded_model, "loaded"
            print(
                f"Ignoring incompatible {name!r} checkpoint. "
                f"Saved metadata={saved_metadata}, expected={expected_metadata}."
            )
        else:
            print(
                f"Ignoring legacy {name!r} checkpoint without shape metadata."
            )

    generator = copy.deepcopy(template_generator)
    generator.model = None
    generator.x_scaler = None
    generator.y_scaler = None
    generator.reducer = None
    generator.condition_reducer = None
    generator.use_condition_dim_reduction = bool(reduce_conditions)
    generator.condition_dim_reduction_method = "svd"
    generator.condition_dim_reduction_components = int(condition_component_cap)
    # External/combined coordinates do not share the quotient semantic prefix.
    generator.condition_dim_reduction_keep_prefix = 0
    generator.checkpoint_dir = os.path.join(checkpoint_root, f"{name}_lightning")
    if maximum_epochs is not None:
        generator.maximum_epochs = int(maximum_epochs)

    model = ConditionalNodeGeneratorModel(
        conditional_node_generator=generator,
        verbose=getattr(generator, "verbose", True),
    )

    edge_targets, edge_pairs = node_decoder.compute_edge_supervision(
        train_graphs,
        train_node_encodings,
        use_edge_fraction=1,
    )
    edge_label_targets, edge_label_pairs, edge_label_to_idx = (
        node_decoder.compute_edge_label_supervision(train_graphs)
    )
    edge_label_idx_to_label = {v: k for k, v in edge_label_to_idx.items()}

    model.fit(
        node_encodings_list=train_node_encodings,
        conditional_graph_encodings=train_conditions,
        edge_pairs=edge_pairs,
        edge_targets=edge_targets,
        edge_label_pairs=edge_label_pairs,
        edge_label_targets=edge_label_targets,
        edge_label_idx_to_label=edge_label_idx_to_label,
    )
    model._expected_raw_condition_dim = int(expected_metadata["condition_dim"])
    with open(path, "wb") as handle:
        dill.dump(
            {
                "model": model,
                "metadata": expected_metadata,
            },
            handle,
        )
    return model, "trained"


def graph_pair_metrics(reference_graphs, generated_graphs, method):
    n = min(len(reference_graphs), len(generated_graphs))
    if n == 0:
        return {"method": method, "n": 0}

    node_match = nx.algorithms.isomorphism.categorical_node_match("label", None)
    node_count = edge_count = degree_hist = label_hist = 0
    unlabeled_iso = labeled_iso = 0

    for true_graph, pred_graph in zip(reference_graphs[:n], generated_graphs[:n]):
        node_count += true_graph.number_of_nodes() == pred_graph.number_of_nodes()
        edge_count += true_graph.number_of_edges() == pred_graph.number_of_edges()
        degree_hist += Counter(dict(true_graph.degree()).values()) == Counter(
            dict(pred_graph.degree()).values()
        )
        label_hist += Counter(
            data.get("label") for _, data in true_graph.nodes(data=True)
        ) == Counter(data.get("label") for _, data in pred_graph.nodes(data=True))
        unlabeled_iso += nx.is_isomorphic(true_graph, pred_graph)
        labeled_iso += nx.is_isomorphic(
            true_graph,
            pred_graph,
            node_match=node_match,
        )

    return {
        "method": method,
        "n": n,
        "node_count_match": node_count / n,
        "edge_count_match": edge_count / n,
        "degree_hist_match": degree_hist / n,
        "label_hist_match": label_hist / n,
        "unlabeled_iso": unlabeled_iso / n,
        "labeled_iso": labeled_iso / n,
        "mean_nodes": float(
            np.mean([graph.number_of_nodes() for graph in generated_graphs[:n]])
        ),
        "mean_edges": float(
            np.mean([graph.number_of_edges() for graph in generated_graphs[:n]])
        ),
    }


def comparison_table(reference_graphs, graph_sets, condition_rows=None):
    rows = [
        graph_pair_metrics(reference_graphs, graphs, name)
        for name, graphs in graph_sets.items()
    ]
    frame = pd.DataFrame(rows)
    if condition_rows:
        condition_frame = pd.DataFrame(condition_rows)
        frame = frame.merge(condition_frame, on="method", how="left")
    return frame
