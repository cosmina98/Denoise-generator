#!/usr/bin/env python
"""Provides interface."""

import networkx as nx
from toolz import curry
from collections import defaultdict
from itertools import combinations
from coco_grape.module.construct import make_decomposition_function

@curry
@make_decomposition_function
def fragment_by_node_removal_decomposition_function(subgraph, basegraph=None, **args):
	size = args.get('size',1)
    
	components = []
	for nodes in combinations(subgraph.nodes(),size):
		gp = subgraph.copy()
		for u in nodes:
			gp.remove_node(u)
		connected_components = list(nx.connected_components(gp))
		if len(connected_components) >= 2: 
			components.extend(connected_components)
	return components

fragment_by_node_removal = fragment_by_node_removal_decomposition_function()

@curry
@make_decomposition_function
def fragment_by_edge_removal_decomposition_function(subgraph, basegraph=None, **args):
	size = args.get('size',1)
    
	components = []
	for edges in combinations(subgraph.edges(),size):
		gp = subgraph.copy()
		for i,j in edges:
			gp.remove_edge(i, j)
		connected_components = list(nx.connected_components(gp))
		if len(connected_components) >= 2: 
			components.extend(connected_components)
	return components

fragment_by_edge_removal = fragment_by_edge_removal_decomposition_function()