#!/usr/bin/env python
"""Provides interface."""

from collections import defaultdict
import networkx as nx
from toolz import curry
from toolz import partition_all
import multiprocessing_on_dill as mp

def hash_list(seq):
    """
    Hashes a list by converting it to a tuple and then hashing.

    Args:
        seq (list): The list to hash.

    Returns:
        int: The hash value of the tuple.
    """
    return hash(tuple(seq))


def masked_hash_value(value, bitmask=4294967295):
    """
    Applies a bitmask to the hash of a value to limit its size.

    Args:
        value: The value to hash and mask.
        bitmask (int, optional): The bitmask to apply. Defaults to 4294967295.

    Returns:
        int: The masked hash value.
    """
    return hash(value) & bitmask


def hash_value(value, context=1, nbits=10):
    """
    Hashes a value and its context, limiting the result to a specified number of bits.

    Args:
        value: The value to hash.
        context (int, optional): Additional context value to be hashed together with the main value.
            Can be used to create different hash spaces for the same value. Defaults to 1.
        nbits (int, optional): Number of bits to limit the hash. Output will be in range [2, 2**nbits - 1].
            Defaults to 10.

    Returns:
        int: The hashed value limited to nbits, guaranteed to be at least 2.
    """
    max_index = 2 ** nbits
    h = masked_hash_value((value, context), max_index - 3)
    h += 2
    return h


@curry
def node_neighborhood_hash(u, graph=None):
    uh = hash(graph.nodes[u]['label'])
    edges_h = [hash((hash(graph.nodes[v]['label']), hash(graph.edges[u, v]['label']))) for v in graph.neighbors(u)]
    nh = hash_list(sorted(edges_h))
    ext_node_h = hash((uh, nh))
    return ext_node_h


def rooted_breadth_first_hash(graph, root):
    def invert_dict(mydict):
        reversed_dict = defaultdict(list)
        for key, value in mydict.items(): reversed_dict[value].append(key)
        return reversed_dict

    node_neighborhood_hash_func = node_neighborhood_hash(graph=graph)
    gid_dist_dict = nx.single_source_shortest_path_length(graph, root)
    dist_gids_dict = invert_dict(gid_dist_dict)
    distance_based_hashes = [sorted(list(map(node_neighborhood_hash_func, dist_gids_dict[d]))) for d in sorted(dist_gids_dict)]
    hash_bfs = [hash_list(seq) for seq in distance_based_hashes]
    return hash_list(hash_bfs)


def nocontext_nodes_hashes(graph):
    nocontext_nodes_hashes_list = [rooted_breadth_first_hash(graph, u) for u in graph.nodes()]
    return nocontext_nodes_hashes_list


def nocontext_edges_hashes(graph):
    nocontext_nodes_hashes_list = nocontext_nodes_hashes(graph)
    nocontext_nodes_hashes_dict = {u:nocontext_node_hash for u, nocontext_node_hash in zip(graph.nodes(), nocontext_nodes_hashes_list)} 
    nocontext_edges_hashes_list = [(*sorted([nocontext_nodes_hashes_dict[u], nocontext_nodes_hashes_dict[v]]), hash(graph.edges[u, v]['label'])) for u,v in graph.edges()]
    return nocontext_edges_hashes_list, nocontext_nodes_hashes_list


def nodes_hash(orig_graph, context=1, nbits=19, use_node_unlabelled_graph=False, use_edge_unlabelled_graph=False):
    if use_node_unlabelled_graph or use_edge_unlabelled_graph: graph = orig_graph.copy()
    else: graph = orig_graph
    if use_node_unlabelled_graph: 
        for u in graph.nodes(): graph.nodes[u]['label'] = '-'
    if use_edge_unlabelled_graph: 
        for e in graph.edges(): graph.edges[e]['label'] = '-'
    nocontext_edges_hashes_list, nocontext_nodes_hashes_list = nocontext_edges_hashes(graph)
    g_hash = hash_list(sorted(nocontext_edges_hashes_list))
    nodes_hashes_list = [hash_value((g_hash,nocontext_node_hash), context, nbits) for nocontext_node_hash in nocontext_nodes_hashes_list]
    return nodes_hashes_list


def graph_hash(orig_graph, context=1, nbits=19, use_node_unlabelled_graph=False, use_edge_unlabelled_graph=False):
    if use_node_unlabelled_graph or use_edge_unlabelled_graph: graph = orig_graph.copy()
    else: graph = orig_graph
    if use_node_unlabelled_graph: 
        for u in graph.nodes(): graph.nodes[u]['label'] = '-'
    if use_edge_unlabelled_graph: 
        for e in graph.edges(): graph.edges[e]['label'] = '-'
    nocontext_edges_hashes_list, nocontext_nodes_hashes_list = nocontext_edges_hashes(graph)
    g_hash = hash_list(sorted(nocontext_nodes_hashes_list)+sorted(nocontext_edges_hashes_list))
    g_hash = hash_value(g_hash, context, nbits)
    return g_hash