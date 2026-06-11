#!/usr/bin/env python
"""Provides interface."""

import networkx as nx
from toolz import curry
from coco_grape.module.construct import make_decomposition_function

@curry
@make_decomposition_function
def degree_decomposition_function(subgraph, basegraph=None, **args):
    size = args.get('size',None)
    min_size = args.get('min_size',size)
    max_size = args.get('max_size',size)
    if min_size is None and max_size is None: min_size = max_size = 3
    
    deg = dict(nx.degree(subgraph))
    if max_size is None:
        max_size = nx.number_of_nodes(subgraph)
    component = set([u for u in deg if max_size >= deg[u] and  deg[u] >= min_size])
    components = [component]
    return components

degree = degree_decomposition_function()