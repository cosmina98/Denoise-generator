#!/usr/bin/env python
"""Provides interface."""

import networkx as nx
from toolz import curry
from coco_grape.module.construct import make_decomposition_function

@curry
@make_decomposition_function
def path_decomposition_function(subgraph, basegraph=None, **args):
    size = args.get('size',None)
    min_size = args.get('min_size',size)
    max_size = args.get('max_size',size)
    if min_size is None and max_size is None: min_size = max_size = 3
    
    edge_components = []
    for n in subgraph.nodes():
        ego_graph = nx.ego_graph(subgraph, n, radius=max_size+1)
        for v in ego_graph.nodes():
            try:
                for path in nx.all_shortest_paths(ego_graph, source=n, target=v):
                    edge_component = set()
                    if len(path) >= min_size + 1 and len(path) <= max_size + 1:
                        for i, u in enumerate(path[:-1]):
                            w = path[i + 1]
                            edge_component.add(u)
                            edge_component.add(w)
                    if edge_component:
                        edge_component = tuple(sorted(edge_component))
                        edge_components.append(edge_component)
            except Exception:
                pass
    components = list(set(edge_components))
    return components

path = path_decomposition_function()