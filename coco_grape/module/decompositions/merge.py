#!/usr/bin/env python
"""Provides interface."""

import networkx as nx
from toolz import curry
from coco_grape.module.construct import make_multi_decomposition_function

@curry
@make_multi_decomposition_function
def merge_neighborhood_function(subgraphs, basegraph=None, **args):
	if len(subgraphs) == 0: return [],[[]]
	g = subgraphs[0]
	for subgraph in subgraphs[1:]:
		g = nx.compose(g, subgraph)
	component = list(g.nodes())
	components = [component]
	component_idxs = [list(range(len(subgraphs)))]
	return components, component_idxs

merge = merge_neighborhood_function()