#!/usr/bin/env python
"""Provides interface."""

import networkx as nx
from toolz import curry
from coco_grape.module.construct import make_decomposition_function

def get_edges_from_cycle(cycle):
    for i, c in enumerate(cycle):
        j = (i + 1) % len(cycle)
        u, v = cycle[i], cycle[j]
        if u < v:
            yield u, v
        else:
            yield v, u


def get_cycle_basis_edges(g):
    ebunch = []
    cs = nx.cycle_basis(g)
    for c in cs:
        ebunch += list(get_edges_from_cycle(c))
    return ebunch


def edge_complement(g, ebunch):
    edge_set = set(ebunch)
    other_ebunch = [e for e in g.edges() if e not in edge_set]
    return other_ebunch


def edge_subgraph(g, ebunch):
    if nx.is_directed(g):
        g2 = nx.DiGraph()
    else:
        g2 = nx.Graph()
    g2.add_nodes_from(g.nodes())
    for u, v in ebunch:
        g2.add_edge(u, v)
        g2.edges[u, v].update(g.edges[u, v])
    return g2


def edge_complement_subgraph(g, ebunch):
    """Induce graph from edges that are not in ebunch."""
    if nx.is_directed(g):
        g2 = nx.DiGraph()
    else:
        g2 = nx.Graph()
    g2.add_nodes_from(g.nodes())
    for e in g.edges():
        if e not in ebunch:
            u, v = e
            g2.add_edge(u, v)
            g2.edges[u, v].update(g.edges[u, v])
    return g2

@curry
@make_decomposition_function
def cycle_tree_decomposition_function(subgraph, basegraph=None, **args):
    size = args.get('size',None)
    min_size = args.get('min_size',size)
    max_size = args.get('max_size',size)
    use_cycle = args.get('use_cycle',True)
    use_tree = args.get('use_tree',True)
    
    cs = nx.cycle_basis(subgraph)
    cycle_components = list(map(set, cs))
    if min_size is not None and max_size is not None:
        cycle_components = [cyc for cyc in cycle_components if len(cyc)>= min_size and len(cyc)<= max_size ]
    cycle_ebunch = get_cycle_basis_edges(subgraph)
    g2 = edge_complement_subgraph(subgraph, cycle_ebunch)
    non_cycle_components = nx.connected_components(g2)
    non_cycle_components = [c for c in non_cycle_components if len(c) >= 2]
    non_cycle_components = list(map(set, non_cycle_components))
    if min_size is not None and max_size is not None:
        non_cycle_components = [cyc for cyc in non_cycle_components if len(cyc)>= min_size and len(cyc)<= max_size ]
    components = []
    if use_cycle: components.extend(cycle_components)
    if use_tree: components.extend(non_cycle_components)
    return components

cycle_tree = cycle_tree_decomposition_function()
cycle = cycle_tree_decomposition_function(use_tree=False)
tree = cycle_tree_decomposition_function(use_cycle=False)
