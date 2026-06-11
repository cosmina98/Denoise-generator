#!/usr/bin/env python
"""Provides interface."""

import numpy as np
import networkx as nx
from toolz import curry
from coco_grape.module.construct import make_decomposition_function
from coco_grape.module.construct import make_multi_decomposition_function

@curry
@make_decomposition_function
def filter_by_number_of_connected_components_decomposition_function(subgraph, basegraph=None, **args):
    size = args.get('size',None)
    min_size = args.get('min_size',size)
    max_size = args.get('max_size',size)
    if min_size is None and max_size is None: min_size = max_size = 3

    components = []
    cc = list(nx.connected_components(subgraph))
    if min_size<=len(cc)<=max_size:
        components = [list(subgraph.nodes())]
    return components

filter_by_number_of_connected_components = filter_by_number_of_connected_components_decomposition_function(use_edges=False)


@curry
@make_decomposition_function
def filter_by_size_decomposition_function(subgraph, basegraph=None, **args):
    size = args.get('size',None)
    min_size = args.get('min_size',size)
    max_size = args.get('max_size',size)
    if min_size is None and max_size is None: min_size = max_size = 3

    use_edges = args.get('use_edges',False)
    components = []
    if use_edges:
        if min_size<=nx.number_of_edges(subgraph)<=max_size:
            components = [list(subgraph.nodes())]
    else:
        if min_size<=nx.number_of_nodes(subgraph)<=max_size:
            components = [list(subgraph.nodes())]
    return components

filter_by_node_size = filter_by_size_decomposition_function(use_edges=False)
filter_by_edge_size = filter_by_size_decomposition_function(use_edges=True)


@curry
@make_multi_decomposition_function
def filter_by_max_size_decomposition_function(subgraphs, basegraph=None, **args):
    size = args.get('size',1)
    use_edges = args.get('use_edges',False)
    components = []
    if use_edges:
        number_of_edges = [nx.number_of_edges(subgraph) for subgraph in subgraphs]
        number_of_edges = np.array(number_of_edges)
        sorted_number_of_edges_idxs = np.argsort(-number_of_edges)
        sorted_number_of_edges_idxs = sorted_number_of_edges_idxs[:size]
        components = [list(subgraphs[idx].nodes()) for idx in sorted_number_of_edges_idxs]
        component_idxs = [sorted_number_of_edges_idxs]
    else:
        number_of_nodes = [nx.number_of_nodes(subgraph) for subgraph in subgraphs]
        number_of_nodes = np.array(number_of_nodes)
        sorted_number_of_nodes_idxs = np.argsort(-number_of_nodes)
        sorted_number_of_nodes_idxs = sorted_number_of_nodes_idxs[:size]
        components = [list(subgraphs[idx].nodes()) for idx in sorted_number_of_nodes_idxs]
        component_idxs = [[idx] for idx in sorted_number_of_nodes_idxs]
    return components, component_idxs

filter_by_max_node_size = filter_by_max_size_decomposition_function(use_edges=False)
filter_by_max_edge_size = filter_by_max_size_decomposition_function(use_edges=True)


@curry
@make_multi_decomposition_function
def filter_by_min_size_decomposition_function(subgraphs, basegraph=None, **args):
    size = args.get('size',1)
    use_edges = args.get('use_edges',False)
    components = []
    if use_edges:
        number_of_edges = [nx.number_of_edges(subgraph) for subgraph in subgraphs]
        number_of_edges = np.array(number_of_edges)
        sorted_number_of_edges_idxs = np.argsort(number_of_edges)
        sorted_number_of_edges_idxs = sorted_number_of_edges_idxs[:size]
        components = [list(subgraphs[idx].nodes()) for idx in sorted_number_of_edges_idxs]
        component_idxs = [sorted_number_of_edges_idxs]
    else:
        number_of_nodes = [nx.number_of_nodes(subgraph) for subgraph in subgraphs]
        number_of_nodes = np.array(number_of_nodes)
        sorted_number_of_nodes_idxs = np.argsort(number_of_nodes)
        sorted_number_of_nodes_idxs = sorted_number_of_nodes_idxs[:size]
        components = [list(subgraphs[idx].nodes()) for idx in sorted_number_of_nodes_idxs]
        component_idxs = [[idx] for idx in sorted_number_of_nodes_idxs]
    return components, component_idxs

filter_by_min_node_size = filter_by_min_size_decomposition_function(use_edges=False)
filter_by_min_edge_size = filter_by_min_size_decomposition_function(use_edges=True)

@curry
@make_decomposition_function
def filter_by_node_label_decomposition_function(subgraph, basegraph=None, **args):
    key = args.get('key', 'label')
    must_have_one_of_in_list = args.get('must_have_one_of_in_list', [])
    cannot_have_any_in_list = args.get('cannot_have_any_in_list', [])
    if len(must_have_one_of_in_list) > 0:
        must_conditions_are_met = False
        for u in subgraph.nodes(): 
            if subgraph.nodes[u].get(key,0) in must_have_one_of_in_list:
                must_conditions_are_met = True
                break
    else: must_conditions_are_met = True

    if len(cannot_have_any_in_list) > 0:
        cannot_conditions_are_met = True
        for u in subgraph.nodes(): 
            if subgraph.nodes[u].get(key,0) in cannot_have_any_in_list:
                cannot_conditions_are_met = False
                break
    else:
        cannot_conditions_are_met = True
    components = []
    if must_conditions_are_met and cannot_conditions_are_met:
        component = list(subgraph.nodes())
        components.append(component)
    return components

filter_by_node_label = filter_by_node_label_decomposition_function()

@curry
@make_decomposition_function
def filter_by_edge_label_decomposition_function(subgraph, basegraph=None, **args):
    key = args.get('key', 'label')
    must_have_one_of_in_list = args.get('must_have_one_of_in_list', [])
    cannot_have_any_in_list = args.get('cannot_have_any_in_list', [])
    if len(must_have_one_of_in_list) > 0:
        must_conditions_are_met = False
        for e in subgraph.edges(): 
            if subgraph.edges[e].get(key,0) in must_have_one_of_in_list:
                must_conditions_are_met = True
                break
    else: must_conditions_are_met = True

    if len(cannot_have_any_in_list) > 0:
        cannot_conditions_are_met = True
        for e in subgraph.edges(): 
            if subgraph.edges[e].get(key,0) in cannot_have_any_in_list:
                cannot_conditions_are_met = False
                break
    else:
        cannot_conditions_are_met = True
    components = []
    if must_conditions_are_met and cannot_conditions_are_met:
        component = list(subgraph.edges())
        components.append(component)
    return components

filter_by_edge_label = filter_by_edge_label_decomposition_function()


@curry
@make_multi_decomposition_function
def filter_by_feature_importance_decomposition_function(subgraphs, basegraph=None, **args):
    feature_ids = np.array(args.get('feature_ids', []))
    unique_feature_ids = np.unique(feature_ids)
    feature_importances = np.array(args.get('feature_importances', []))
    size = args.get('size', .5)
    if size > 1: n = size 
    else: n = int(len(unique_feature_ids)*size)

    local_unique_feature_importances = feature_importances[unique_feature_ids]
    important_feature_local_idxs = np.argsort(-local_unique_feature_importances)[:n]
    important_feature_idxs = unique_feature_ids[important_feature_local_idxs]
    
    selected_component_idxs = [idx for idx, feature_id in enumerate(feature_ids) if feature_id in important_feature_idxs]
    components = [[u for u in subgraphs[selected_component_idx].nodes()] for selected_component_idx in selected_component_idxs]
    component_idxs = [[selected_component_idx] for selected_component_idx in selected_component_idxs]
    return components, component_idxs

filter_by_feature_importance = filter_by_feature_importance_decomposition_function()

@curry
@make_multi_decomposition_function
def filter_by_feature_id_function(subgraphs, basegraph=None, **args):
    feature_ids = np.array(args.get('feature_ids', []))
    feature_mask = args.get('feature_mask', [True]*len(feature_ids))
    selected_component_idxs = [idx for idx, feature_id in enumerate(feature_ids) if feature_mask[feature_id]]
    components = [[u for u in subgraphs[selected_component_idx].nodes()] for selected_component_idx in selected_component_idxs]
    component_idxs = [[selected_component_idx] for selected_component_idx in selected_component_idxs]
    return components, component_idxs

filter_by_feature_id = filter_by_feature_id_function()
