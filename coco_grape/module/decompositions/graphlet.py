#!/usr/bin/env python
"""Provides interface."""

import networkx as nx
from toolz import curry
import itertools
from coco_grape.module.construct import make_decomposition_function

@curry
@make_decomposition_function
def graphlet_decomposition_function(subgraph, basegraph=None, **args):
    size = args.get('size',None)
    min_size = args.get('min_size',size)
    max_size = args.get('max_size',size)
    if min_size is None and max_size is None: min_size = max_size = 2
    radius = args.get('radius',1)
    components = []
    for size in range(min_size, max_size + 1):
        for u in subgraph.nodes():
            ego_graph = nx.ego_graph(subgraph, u, radius=radius)
            for sub_nodes in itertools.combinations(ego_graph.nodes(), size):
                sub_subgraph = ego_graph.subgraph(sub_nodes)
                if nx.is_connected(sub_subgraph):
                    components.append(tuple(sorted(set(sub_nodes))))
    components = list(set(components))
    return components

graphlet = graphlet_decomposition_function()