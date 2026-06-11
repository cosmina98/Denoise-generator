#!/usr/bin/env python
"""Provides interface."""

import networkx as nx
from toolz import curry
from coco_grape.module.construct import make_decomposition_function

@curry
@make_decomposition_function
def connected_component_decomposition_function(subgraph, basegraph=None, **args):
    components = list(nx.connected_components(subgraph))
    return components

connected_component = connected_component_decomposition_function()
