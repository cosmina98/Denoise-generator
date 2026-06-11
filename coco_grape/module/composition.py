#!/usr/bin/env python
"""Provides interface."""

import networkx as nx
import toolz as tz


def add(*decomposition_functions):
	def composed_decomposition_function(graphofsubgraphs):
		combined_out_graphofsubgraphs = None
		for decomposition_function in decomposition_functions:
			out_graphofsubgraphs = decomposition_function(graphofsubgraphs)
			if combined_out_graphofsubgraphs is None:
				combined_out_graphofsubgraphs = out_graphofsubgraphs
			else:
				combined_out_graphofsubgraphs = nx.disjoint_union(combined_out_graphofsubgraphs, out_graphofsubgraphs)
		return combined_out_graphofsubgraphs
	return composed_decomposition_function


def compose(*decomposition_functions):
	return tz.compose(*decomposition_functions)


def binary_combine(decomposition_function_combination, decomposition_function_1, decomposition_function_2):
	def combined_decomposition_function(graphofsubgraphs):
		return decomposition_function_combination(decomposition_function_1(graphofsubgraphs), decomposition_function_2(graphofsubgraphs))
	return combined_decomposition_function


def ternary_combine(decomposition_function_combination, decomposition_function_1, decomposition_function_2, decomposition_function_3):
	def combined_decomposition_function(graphofsubgraphs):
		return decomposition_function_combination(decomposition_function_1(graphofsubgraphs), decomposition_function_2(graphofsubgraphs), decomposition_function_3(graphofsubgraphs))
	return combined_decomposition_function