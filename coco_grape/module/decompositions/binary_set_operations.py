#!/usr/bin/env python
"""Provides interface."""

import networkx as nx
from toolz import curry
from coco_grape.module.construct import make_binary_multi_decomposition_function
from copy import copy

@curry
@make_binary_multi_decomposition_function
def binary_union_decomposition_function(subgraphs1, subgraphs2, basegraph=None, **args):
	components = []
	component_idxs = []
	for i, subgraph1 in enumerate(subgraphs1):
		component1 = set([u for u in subgraph1.nodes()])
		for j, subgraph2 in enumerate(subgraphs2):
			component = copy(component1)
			component.update([u for u in subgraph2.nodes()])
			components.append(list(component))
			component_idxs.append( [[i],[j]] )
	return components, component_idxs

binary_union = binary_union_decomposition_function()

@curry
@make_binary_multi_decomposition_function
def binary_intersection_decomposition_function(subgraphs1, subgraphs2, basegraph=None, **args):
	components = []
	component_idxs = []
	for i, subgraph1 in enumerate(subgraphs1):
		component1 = set([u for u in subgraph1.nodes()])
		for j, subgraph2 in enumerate(subgraphs2):
			component2 = set([u for u in subgraph2.nodes()])
			component = component1.intersection(component2)
			if len(component) > 0:
				components.append(list(component))
				component_idxs.append( [[i],[j]] )
	return components, component_idxs

binary_intersection = binary_intersection_decomposition_function()


@curry
@make_binary_multi_decomposition_function
def binary_difference_decomposition_function(subgraphs1, subgraphs2, basegraph=None, **args):
	components = []
	component_idxs = []
	for i, subgraph1 in enumerate(subgraphs1):
		component1 = set([u for u in subgraph1.nodes()])
		for j, subgraph2 in enumerate(subgraphs2):
			component2 = set([u for u in subgraph2.nodes()])
			component = component1.difference(component2)
			if len(component) > 0:
				components.append(list(component))
				component_idxs.append( [[i],[j]] )
	return components, component_idxs

binary_difference = binary_difference_decomposition_function()
