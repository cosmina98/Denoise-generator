import networkx as nx
import numpy as np
from typing import Optional, Callable, Any, List, Iterable, Tuple, Dict, Set 
from coco_grape.module.hash_graph import hash_graph, hash_bounded


DEFAULT_NBITS = 14

#==========================================================================================
# Label functions for QuotientGraph
#==========================================================================================
def graph_hash_label_function_factory(nbits: int = DEFAULT_NBITS) -> Callable[[dict], int]:
    """
    Returns a label function that computes a hash of the 'subgraph' using hash_graph
    with the specified number of bits.

    Args:
        nbits: The number of bits for the hash output (default: 14).

    Returns:
        A label function that takes node attributes and returns an integer hash.
    """
    def label_fn(node_attrs: dict) -> int:
        subgraph = node_attrs.get("association", None)
        if subgraph is None:
            raise ValueError("Node attributes must contain an 'association' key.")
        return hash_graph(subgraph, nbits=nbits)
    label_fn.nbits = nbits # Attach nbits as an attribute
    return label_fn

def graph_structure_hash_label_function_factory(nbits: int = DEFAULT_NBITS) -> Callable[[dict], int]:
    """
    Returns a label function that hashes only the structure of the 'subgraph',
    ignoring all node and edge labels. The hash is computed with the given bit size.

    Args:
        nbits: The number of bits for the hash output (default: 14).

    Returns:
        A label function that takes node attributes and returns a structure-based hash.
    """
    def label_fn(node_attrs: dict) -> int:
        subgraph = node_attrs.get("association", None)
        if subgraph is None:
            raise ValueError("Node attributes must contain an 'association' key.")

        # Copy and sanitize node and edge labels
        structure_graph = subgraph.copy()
        for node in structure_graph.nodes:
            structure_graph.nodes[node]["label"] = "-"
        for u, v in structure_graph.edges:
            structure_graph.edges[u, v]["label"] = "-"

        return hash_graph(structure_graph, nbits=nbits)
    label_fn.nbits = nbits # Attach nbits as an attribute
    return label_fn



def source_function_hash_label_function_factory(nbits: int = DEFAULT_NBITS) -> Callable[[dict], int]:
    """
    Returns a label function that hashes the 'source_function' stored in the 'meta' dictionary
    of node attributes into an integer in the range [0, 2**nbits - 1].

    Args:
        nbits: The number of bits to use for the hash output (e.g. 8 → 0-255).

    Returns:
        A function that takes node attributes and returns an integer label.
    """
    def label_fn(node_attrs: dict) -> int:
        # Extract the source function identifier from metadata; default to 'unknown'
        source = node_attrs.get("meta", {}).get("source_function", "unknown")
        # Use stable, bounded hashing for reproducibility and column consistency
        return hash_bounded(source, nbits=nbits)
    label_fn.nbits = nbits # Attach nbits as an attribute
    return label_fn

#==================================================================================================
# Attribute functions for QuotientGraph
#==================================================================================================


def sum_attribute_function(subgraph: nx.Graph) -> np.ndarray:
    """
    Attribute function that sums all 'attribute' numpy arrays from nodes in the given subgraph.
    
    Assumes that each node's data has an 'attribute' key associated with a NumPy array.
    
    Args:
        subgraph: The NetworkX subgraph from which to aggregate attributes.
    
    Returns:
        A NumPy array representing the sum of all attributes in the subgraph.
    """
    attr_list = [data.get('attribute', np.array(0)) for _, data in subgraph.nodes(data=True)]
    if attr_list:
        return np.sum(attr_list, axis=0)
    else:
        return np.array(0)

#==================================================================================================
# Edge functions for QuotientGraph
#==================================================================================================

def intersection_edge_function(quotient_graph: "QuotientGraph") -> "QuotientGraph":
    """
    External function that generates edges in the image graph based on intersections between the subgraphs.
    
    An edge is created if two image nodes share at least one common node in their subgraphs.
    
    Args:
        quotient_graph: The QuotientGraph instance to update with new edges.
    """
    nodes = list(quotient_graph.image_graph.nodes)
    for i, node1 in enumerate(nodes):
        for node2 in nodes[i+1:]:
            subgraph1 = quotient_graph.image_graph.nodes[node1]["association"]
            subgraph2 = quotient_graph.image_graph.nodes[node2]["association"]
            if set(subgraph1.nodes) & set(subgraph2.nodes):
                quotient_graph.image_graph.add_edge(node1, node2)
    return quotient_graph

def null_edge_function(quotient_graph: "QuotientGraph") -> "QuotientGraph":
    """
    Edge function that does nothing. It simply returns the quotient_graph unchanged.
    
    Args:
        quotient_graph: The QuotientGraph instance.
    
    Returns:
        The unchanged QuotientGraph instance.
    """
    return quotient_graph
