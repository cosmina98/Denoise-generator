#!/usr/bin/env python
"""Provides interface."""

import networkx as nx
from toolz import curry
from coco_grape.module.construct import make_decomposition_function

@curry
@make_decomposition_function
def betweenness_centrality_decomposition_function(subgraph, basegraph=None, **args):
    size = args.get('size',1)
    use_perifery = args.get('use_perifery',False) 
    n_dict = nx.betweenness_centrality(subgraph)
    if use_perifery: reverse = False
    else: reverse = True
    selected_ids = sorted(n_dict, key=lambda x: n_dict[x], reverse=reverse)[:size]
    components = [selected_ids] 
    return components

betweenness_centrality = betweenness_centrality_decomposition_function(use_perifery=False)
betweenness_perifery = betweenness_centrality_decomposition_function(use_perifery=True)