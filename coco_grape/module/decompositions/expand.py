#!/usr/bin/env python
"""Provides interface."""

import networkx as nx
from toolz import curry
from coco_grape.module.construct import make_decomposition_function


def expand_decomposition(graph, base_graph, size):
    node_components = []
    for u in graph.nodes():
        ego_graph = nx.ego_graph(base_graph, u, radius=size)
        node_components.extend(list(ego_graph.nodes()))
    return list(set(node_components))


@curry
@make_decomposition_function
def expand_decomposition_function(subgraph, basegraph=None, **args):
    size = args.get('size',None)
    min_size = args.get('min_size',size)
    max_size = args.get('max_size',size)
    if min_size is None and max_size is None: min_size = max_size = 1
    components = []
    for s in range(min_size, max_size+1):
        component = expand_decomposition(subgraph, basegraph, size=s)
        components.append(component)
    return components

expand = expand_decomposition_function()