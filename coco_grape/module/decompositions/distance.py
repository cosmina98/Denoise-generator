#!/usr/bin/env python
"""Provides interface."""

import networkx as nx
from toolz import curry
from coco_grape.module.construct import make_multi_decomposition_function

def get_most_central_node(graph):
    node_centrality_dict = nx.betweenness_centrality(graph)
    return max(node_centrality_dict, key=lambda u:node_centrality_dict[u])

def get_distance(graph1, graph2, basegraph):
    return min(nx.shortest_path_length(basegraph, source=u, target=v) for u in graph1.nodes() for v in graph2.nodes())

@curry
@make_multi_decomposition_function
def distance_decomposition_function(subgraphs, basegraph=None, **args):
    size = args.get('size',None)
    min_size = args.get('min_size',size)
    max_size = args.get('max_size',size)
    if min_size is None and max_size is None: min_size = max_size = 1
    use_centrality = args.get('use_centrality',False)
    components = []
    component_idxs = []
    if use_centrality:
        for i, subgraph_i in enumerate(subgraphs):
            c_i = get_most_central_node(subgraph_i)
            for j, subgraph_j in enumerate(subgraphs):
                if j > i:
                    component_idxs.append([i,j])
                    c_j = get_most_central_node(subgraph_j)
                    try:
                        dist = nx.shortest_path_length(basegraph, source=c_i, target=c_j)
                        if min_size <= dist <= max_size:
                            union_set = set(subgraph_i.nodes()) | set(subgraph_j.nodes())
                            components.append(union_set)
                    except Exception:
                        pass
    else:
        for i, subgraph_i in enumerate(subgraphs):
            for j, subgraph_j in enumerate(subgraphs):
                if j > i:
                    component_idxs.append([i,j])
                    try:
                        dist = get_distance(subgraph_i, subgraph_j, basegraph)
                        if min_size <= dist <= max_size:
                            union_set = set(subgraph_i.nodes()) | set(subgraph_j.nodes())
                            components.append(union_set)
                    except Exception:
                        pass
    return components, component_idxs

distance = distance_decomposition_function()