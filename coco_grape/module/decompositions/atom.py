#!/usr/bin/env python
"""Provides interface."""

import networkx as nx
from toolz import curry
from coco_grape.module.construct import make_decomposition_function

@curry
@make_decomposition_function
def atom_decomposition_function(subgraph, basegraph=None, **args):
    use_nodes = args.get('use_nodes',True)
    use_edges = args.get('use_edges',True)
    components = []
    if use_nodes is True:
        for n in subgraph.nodes():
            component = [n]
            components.append(component)
    if use_edges is True:
        for i,j in subgraph.edges():
            component = [i,j]
            components.append(component)
    return components

atom = atom_decomposition_function()

node = atom_decomposition_function(use_nodes=True, use_edges=False)
edge = atom_decomposition_function(use_nodes=False, use_edges=True)