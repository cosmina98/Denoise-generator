#!/usr/bin/env python
"""Provides interface."""

import networkx as nx
from toolz import curry
from coco_grape.module.construct import make_decomposition_function

@curry
@make_decomposition_function
def node_label_decomposition_function(subgraph, basegraph=None, **args):
	key = args.get('key','label')
	components = []
	label_set = set([subgraph.nodes[u].get(key,0) for u in subgraph.nodes()])
	for label in label_set:
		component = set([u for u in subgraph.nodes() if subgraph.nodes[u].get(key,0)==label])
		components.append(component)
	return components

node_label = node_label_decomposition_function()


@curry
@make_decomposition_function
def edge_label_decomposition_function(subgraph, basegraph=None, **args):
	key = args.get('key','label')
	components = []
	label_set = set([subgraph.edges[e].get(key,0) for e in subgraph.edges()])
	for label in label_set:
		component = set([e for e in subgraph.edges() if subgraph.edges[e].get(key,0)==label])
		components.append(component)
	return components

edge_label = edge_label_decomposition_function()