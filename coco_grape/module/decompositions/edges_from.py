#!/usr/bin/env python
"""Provides interface."""

import networkx as nx
from toolz import curry
from coco_grape.module.construct import make_multi_edge_decomposition_function

def _get_edges(graph, nbunch):
    edges = []
    for u in nbunch:
        neighs = graph.neighbors(u)
        for v in neighs:
            if u < v:
                edges.append((u, v))
            else:
                edges.append((v, u))
    return edges


def valid_edge_node_intersection(subgraph_i, subgraph_j, min_size, max_size):
    intersect = set(subgraph_i.nodes()) & set(subgraph_j.nodes())
    intersection_size = len(intersect)
    is_valid = min_size <= intersection_size <= max_size
    return is_valid


def valid_edge_edge_intersection(subgraph_i, subgraph_j, min_size, max_size):
    subgraph_i_all_edges = _get_edges(subgraph_i, subgraph_i.nodes())
    subgraph_j_all_edges = _get_edges(subgraph_j, subgraph_j.nodes())
    intersect = set(subgraph_i_all_edges) & set(subgraph_j_all_edges)
    intersection_size = len(intersect)
    is_valid = min_size <= intersection_size <= max_size
    #is_valid = intersection_size > 0
    return is_valid


def distance(subgraph_i, subgraph_j, base_graph):
    try:
        d = min(nx.shortest_path_length(base_graph, source=u, target=v) for u in subgraph_i.nodes() for v in subgraph_j.nodes())
    except:
        d = 1e6
        pass 
    return d


def valid_distance(subgraph_i, subgraph_j, basegraph, min_size, max_size):
    dist = distance(subgraph_i, subgraph_j, basegraph)
    is_valid = (min_size <= dist <= max_size)
    return is_valid

@curry
@make_multi_edge_decomposition_function
def edges_from_decomposition_function(subgraphs, basegraph=None, **args):
    use_node_intersection = args.get('use_node_intersection', True)
    use_edge_intersection = args.get('use_edge_intersection', False)
    use_distance = args.get('use_distance',False)
    size = args.get('size',None)
    min_size = args.get('min_size',size)
    max_size = args.get('max_size',size)
    if min_size is None and max_size is None: min_size = max_size = 1
    
    edges = []
    for i, subgraph_i in enumerate(subgraphs):
        for j, subgraph_j in enumerate(subgraphs):
            if j > i:
                if use_node_intersection is True and valid_edge_node_intersection(subgraph_i, subgraph_j, min_size, max_size): edges.append((i,j))
                if use_edge_intersection is True and valid_edge_edge_intersection(subgraph_i, subgraph_j, min_size, max_size): edges.append((i,j))
                if use_distance is True and valid_distance(subgraph_i, subgraph_j, basegraph, min_size, max_size): edges.append((i,j))
                    
    return edges

edges_from_node_intersection = edges_from_decomposition_function(use_node_intersection=True, use_edge_intersection=False, use_distance=False)
edges_from_edge_intersection = edges_from_decomposition_function(use_node_intersection=False, use_edge_intersection=True, use_distance=False)
edges_from_distance = edges_from_decomposition_function(use_node_intersection=False, use_edge_intersection=False, use_distance=True)

