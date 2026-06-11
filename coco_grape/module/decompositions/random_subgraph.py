#!/usr/bin/env python
"""Provides interface."""

import numpy as np
import networkx as nx
from toolz import curry
from coco_grape.module.construct import make_decomposition_function

def edge_samples(graph, size, edge_size, weight_key='weight'):
    edges = np.array(list(graph.edges()))
    weights = [graph.edges[u,v].get(weight_key, 1) for u,v in graph.edges()]
    weights = np.array(weights)
    weights = weights/np.sum(weights)
    n_edges = len(edges)
    if edge_size<1: effective_size = int(n_edges * edge_size)
    else: effective_size = edge_size
    subgraphs = []
    for i in range(size):
        edge_idxs = np.random.choice(n_edges, size=effective_size, replace=True, p=weights)
        edge_idxs = np.unique(edge_idxs)
        subgraph_edges = edges[edge_idxs]
        subgraph_edges = [(e[0],e[1]) for e in subgraph_edges]
        subgraph = nx.edge_subgraph(graph, subgraph_edges)
        subgraph = nx.Graph(subgraph)
        subgraphs.append(subgraph)
    return subgraphs

@curry
@make_decomposition_function
def random_subgraph_decomposition_function(subgraph, basegraph=None, **args):
    size = args.get('size',1) #number of components to output
    edge_size = args.get('edge_size',1) #number or fraction of edges to sample
    weight_key = args.get('weight_key','weight') #edge attribute that is used to compute the probabilities with which to sample edges
    graph_fragments = edge_samples(subgraph, size, edge_size, weight_key)
    components = [list(graph_fragment.nodes()) for graph_fragment in graph_fragments]
    return components

random_subgraph = random_subgraph_decomposition_function()