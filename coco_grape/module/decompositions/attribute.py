#!/usr/bin/env python
"""Provides interface."""

import networkx as nx
from toolz import curry
from coco_grape.module.construct import make_decomposition_function

@curry
@make_decomposition_function
def node_attribute_decomposition_function(subgraph, basegraph=None, **args):
    key = args.get('key','label')
    components = []
    attributes = list(set(subgraph.nodes[u].get(key,0) for u in subgraph.nodes()))
    for attribute in attributes:
        component = []
        for u in subgraph.nodes():
            if subgraph.nodes[u].get(key,0) == attribute:
                component.append(u)
        components.append(component)
    return components

node_attribute = node_attribute_decomposition_function()

@curry
@make_decomposition_function
def edge_attribute_decomposition_function(subgraph, basegraph=None, **args):
    key = args.get('key','label')
    components = []
    attributes = list(set(subgraph.edges[u,v].get(key,0) for u,v in subgraph.edges()))
    for attribute in attributes:
        component = []
        for u,v in subgraph.edges():
            if subgraph.edges[u,v].get(key,0) == attribute:
                component.append((u,v))
        components.append(component)

    return components

edge_attribute = edge_attribute_decomposition_function()