#!/usr/bin/env python
"""Provides interface."""

import networkx as nx
from toolz import curry
from coco_grape.module.construct import make_binary_multi_decomposition_function

@curry
@make_binary_multi_decomposition_function
def if_not_empty_decomposition_function(subgraphs1, subgraphs2, basegraph=None, **args):
	components = []
	component_idxs = []

	if len(subgraphs1) > 0:
		for i, subgraph in enumerate(subgraphs1):
			components.append([u for u in subgraph.nodes()])
			component_idxs.append( [[i],[None]] )
	else:
		for i, subgraph in enumerate(subgraphs2):
			components.append([u for u in subgraph.nodes()])
			component_idxs.append( [[None],[i]] )
	return components, component_idxs

if_not_empty = if_not_empty_decomposition_function()