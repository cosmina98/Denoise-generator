#!/usr/bin/env python
"""Provides interface."""

import networkx as nx
from toolz import curry
from coco_grape.module.construct import make_decomposition_function

def get_neighborhood(graph, component, radius=1):
    nbunch = set()
    for u in component:
        nbunch.update(set(nx.ego_graph(graph, u, radius=radius).nodes()))
    return nbunch


def get_edges_bunch(graph, nbunch):
    g2 = nx.subgraph(graph, nbunch)
    ebunch = set()
    for u, v in g2.edges():
        if u < v:
            ebunch.add((u, v))
        else:
            ebunch.add((v, u))
    return ebunch


def get_neighborhood_edges(graph, component, radius=1):
    nbunch = get_neighborhood(graph, component, radius)
    ebunch = get_edges_bunch(graph, nbunch)
    return ebunch

@curry
@make_decomposition_function
def context_decomposition_function(subgraph, basegraph=None, **args):
    size = args.get('size',None)
    min_size = args.get('min_size',size)
    max_size = args.get('max_size',size)
    if min_size is None and max_size is None: min_size = max_size = 1
    components = []
    component = set(subgraph.nodes())
    component_edges = get_edges_bunch(basegraph, component)
    neighbor_edges = get_neighborhood_edges(basegraph, component, size)
    context_edges = neighbor_edges - component_edges
    context_edges = list(context_edges)
    components = [context_edges]
    return components

context = context_decomposition_function()