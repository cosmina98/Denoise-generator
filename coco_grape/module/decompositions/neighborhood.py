#!/usr/bin/env python
"""Provides interface."""

import networkx as nx
from toolz import curry
from collections import defaultdict
from coco_grape.module.construct import make_decomposition_function


@curry
@make_decomposition_function
def neighborhood_decomposition_function(subgraph, basegraph=None, **args):
    size = args.get('size',None)
    min_size = args.get('min_size',size)
    max_size = args.get('max_size',size)
    if min_size is None and max_size is None: min_size = max_size = 1
    components = []
    for cutoff in range(min_size, max_size+1):
        for u in subgraph.nodes():
            ego_graph = nx.ego_graph(subgraph, u, radius=cutoff)
            component = list(ego_graph.nodes())
            components.append(component)
    return components

neighborhood =  neighborhood_decomposition_function()


def invert_dict(mydict):
    reversed_dict = defaultdict(list)
    for key, value in mydict.items():
        reversed_dict[value].append(key)
    return reversed_dict


def get_distances(graph, cutoff=None):
    return {node_id:invert_dict(nx.single_source_shortest_path_length(graph, node_id, cutoff=cutoff)) for node_id in graph.nodes()}


def get_neighborhood(node_id, radius, distances_dict):
    nbunch = []
    for dist in range(radius+1):
        nbunch.extend(distances_dict[node_id][dist])
    return nbunch

@curry
@make_decomposition_function
def pairwise_neighborhood_function(subgraph, basegraph=None, **args):
    size = args.get('size',None)
    min_size = args.get('min_size',size)
    max_size = args.get('max_size',size)
    if min_size is None and max_size is None: min_size = max_size = 1
    
    distance = args.get('distance',None)
    min_distance = args.get('min_distance',distance)
    max_distance = args.get('max_distance',distance)
    if min_distance is None and max_distance is None: min_distance = max_distance = 1
    
    cutoff = max(max_distance, max_size)
    components = []
    distances_dict = get_distances(subgraph, cutoff)
    for radius in range(min_size, max_size+1):  
        for i in subgraph.nodes():
            neighborhood_i = get_neighborhood(i, radius, distances_dict)
            for dist in range(min_distance, max_distance+1):
                js = distances_dict[i][dist]
                for j in js:
                    neighborhood_j = get_neighborhood(j, radius, distances_dict)
                    component = set(neighborhood_i+neighborhood_j)
                    components.append(component)
    return components

pairwise_neighborhood = pairwise_neighborhood_function()
