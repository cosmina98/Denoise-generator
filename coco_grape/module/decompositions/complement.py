#!/usr/bin/env python
"""Provides interface."""

import networkx as nx
from toolz import curry
from coco_grape.module.construct import make_decomposition_function

@curry
@make_decomposition_function
def complement_decomposition_function(subgraph, basegraph=None, **args):
    component = list(subgraph.nodes())
    negative_component = set(basegraph.nodes()).difference(set(component))
    negative_components = [negative_component]
    return negative_components

complement = complement_decomposition_function()