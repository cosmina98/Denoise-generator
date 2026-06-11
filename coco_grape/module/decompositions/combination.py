#!/usr/bin/env python
"""Provides interface."""

import networkx as nx
import numpy as np
from toolz import curry
from itertools import combinations, product
from coco_grape.module.construct import make_multi_decomposition_function
from coco_grape.module.construct import make_binary_multi_decomposition_function

def get_distance(graph1, graph2, basegraph):
    return min(nx.shortest_path_length(basegraph, source=u, target=v) for u in graph1.nodes() for v in graph2.nodes())

def get_distance_matrix(subgraphs1, subgraphs2, basegraph, max_distance, min_distance):
    distance_matrix = np.zeros((len(subgraphs1), len(subgraphs2)))
    for i, subgraph_i in enumerate(subgraphs1):
        for j, subgraph_j in enumerate(subgraphs2):
            try:
                dist = get_distance(subgraph_i, subgraph_j, basegraph)
                if min_distance <= dist <= max_distance:
                    distance_matrix[i,j] = dist
                    #distance_matrix[j,i] = dist
                else:
                    distance_matrix[i,j] = np.nan
                    #distance_matrix[j,i] = np.nan
            except Exception:
                distance_matrix[i,j] = np.nan
                #distance_matrix[j,i] = np.nan
                pass
    return distance_matrix

def all_distances_are_feasible(combination_idxs, distance_matrix):
    pairs = combinations(combination_idxs, 2)
    for i,j in pairs:
        distance = distance_matrix[i,j]
        if np.isnan(distance): 
            return False
    return True  

@curry
@make_multi_decomposition_function
def combination_decomposition_function(subgraphs, basegraph=None, **args):
    size = args.get('size',None)
    min_size = args.get('min_size',size)
    max_size = args.get('max_size',size)
    if min_size is None and max_size is None: min_size = max_size = 2
    max_distance = args.get('max_distance', None)
    min_distance = args.get('min_distance', 0)
    distance = args.get('distance', None)
    if distance is not None: max_distance = min_distance = distance
    if max_distance is not None: distance_matrix = get_distance_matrix(subgraphs, subgraphs, basegraph, max_distance, min_distance)
    else: distance_matrix = None
    components = []
    component_combination_idxs = []
    component_combinations = [list(subgraph.nodes()) for subgraph in subgraphs]
    for order in range(min_size, max_size+1):
        combination_idxs_list = combinations(range(len(component_combinations)), order)
        for combination_idxs in combination_idxs_list:
            if distance_matrix is not None and all_distances_are_feasible(combination_idxs, distance_matrix) is False: continue #i.e. skip to next combination_idxs
            component_combination = [component_combinations[combination_idx] for combination_idx in combination_idxs]
            component = set([node for combination_nodes in component_combination for node in combination_nodes])
            components.append(component)
            component_combination_idxs.append(combination_idxs)
    return components, component_combination_idxs

combination = combination_decomposition_function()


@curry
@make_binary_multi_decomposition_function
def binary_combination_decomposition_function(subgraphs1, subgraphs2, basegraph=None, **args):
    size = args.get('size',None)
    min_size = args.get('min_size',size)
    max_size = args.get('max_size',size)
    if min_size is None and max_size is None: min_size = max_size = 2
    max_distance = args.get('max_distance', None)
    min_distance = args.get('min_distance', 0)
    distance = args.get('distance', None)
    if distance is not None: max_distance = min_distance = distance
    if max_distance is not None: distance_matrix = get_distance_matrix(subgraphs1, subgraphs2, basegraph, max_distance, min_distance)
    else: distance_matrix = None
    components = []
    component_combination_idxs = []
    component_combinations1 = [list(subgraph.nodes()) for subgraph in subgraphs1]
    component_combinations2 = [list(subgraph.nodes()) for subgraph in subgraphs2]
    combination_idxs_list = product(range(len(component_combinations1)), range(len(component_combinations2)))
    for combination_idxs in combination_idxs_list:
        if distance_matrix is not None and all_distances_are_feasible(combination_idxs, distance_matrix) is False: continue #i.e. skip to next combination_idxs
        nodes1_list = [node for node in component_combinations1[combination_idxs[0]]]
        nodes2_list = [node for node in component_combinations2[combination_idxs[1]]]
        component = set(nodes1_list+nodes2_list)
        components.append(component)
        component_combination_idxs.append(([combination_idxs[0]],[combination_idxs[1]]))
    return components, component_combination_idxs

binary_combination = binary_combination_decomposition_function()