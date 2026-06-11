#!/usr/bin/env python
"""Provides interface."""

import networkx as nx
from toolz import curry
from coco_grape.module.construct import make_decomposition_function

@curry
@make_decomposition_function
def clique_decomposition_function(subgraph, basegraph=None, **args):
    size = args.get('size',None)
    min_size = args.get('min_size',size)
    max_size = args.get('max_size',size)
    if min_size is None and max_size is None: min_size = max_size = 3
    cliques = nx.enumerate_all_cliques(subgraph)
    components = list(filter(lambda x: min_size <= len(x) <= max_size, cliques))
    return components

clique = clique_decomposition_function()
triangle = clique_decomposition_function(size=3)
