"""
Deterministic hashing utilities for graphs and general Python values.

Goals
- Deterministic across runs and platforms (no use of Python's built-in hash).
- Order handling by intent: provide both order-aware (sequences) and
  order-independent (multisets) combinators.
- Compact, bounded indices for downstream feature maps (via nbits).

Core building blocks
- canonicalize / canonical_bytes: Convert arbitrary values into a stable,
  type-tagged, JSON-serializable form. Handles scalars, containers, numpy arrays,
  bytes, and dicts (sorted by canonicalized key). This prevents collisions like 1 vs "1" and
  avoids platform-dependent stringification.
- hash_value: SHA-256 over canonical_bytes(value) → big integer (base 16).
- hash_sequence: Order-aware combiner for lists/tuples.
- hash_set: Order-independent combiner for multisets by hashing elements first
  and sorting the resulting integers before combining.
- hash_bounded: Map a large hash to a fixed bit-width with a reserved low range
  (returns in [2, 2**nbits - 1]), leaving 0/1 available for special features.

Graph hashing (label-aware)
- hash_node: Hash of a node’s own label combined with an order-independent
  multiset of (neighbor label, edge label) pairs. Uses degree==0 to detect
  isolated nodes.
- hash_rooted_graph: For each root, perform BFS to group nodes by distance and
  combine their node hashes per layer; then combine layers in order to form a
  rooted-subgraph hash.
- hash_graph: Aggregate information across the whole graph by combining rooted
  subgraph hashes along edges with edge labels, plus a multiset hash of all
  node labels. Finally bound to nbits using hash_bounded.

Notes and limitations
- Only the 'label' attribute is used for nodes/edges; extend if other
  attributes matter (e.g., weights). For MultiGraph/DiGraph, adapt edge
  iteration to handle keys/direction explicitly.
- canonicalize does not special-case NetworkX graphs; use `hash_graph` directly
  when you intend to hash a graph object.
- Complexity: hashing all rooted subgraphs is roughly O(V·(V+E)). For large
  graphs, consider caching, limiting radius, or alternative summaries.
- Collisions: Cryptographic hashing minimizes but does not eliminate collisions;
  reducing to nbits increases collision risk. Choose nbits to balance size and
  collision tolerance.
"""

import networkx as nx
import hashlib
import json
import math
from base64 import b64encode
from collections import defaultdict
from typing import Any, Optional, List, Tuple, Dict, Union
import numpy as np

def canonicalize(value: Any) -> Any:
    """
    Produce a deterministic, JSON-serializable, type-tagged representation of `value`.

    Ensures stability across Python versions and platforms, preserves ordering for
    ordered containers, and sorts unordered containers. Prevents collisions across
    differing types (e.g., 1 vs "1").
    """
    # None and booleans
    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", int(value))

    # Numeric types (integers, floats)
    if isinstance(value, (int, np.integer)):
        return ("int", int(value))
    if isinstance(value, (float, np.floating)):
        x = float(value)
        if math.isnan(x):
            return ("float", "NaN")
        if math.isinf(x):
            return ("float", "Inf" if x > 0 else "-Inf")
        # repr maintains precision for typical floats
        return ("float", repr(x))

    # Strings and bytes-like
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return ("bytes", b64encode(bytes(value)).decode("ascii"))

    # NumPy arrays
    if isinstance(value, np.ndarray):
        # Encode dtype, shape, and raw bytes to avoid large JSON lists
        return ("ndarray", str(value.dtype), tuple(value.shape), b64encode(value.tobytes()).decode("ascii"))

    # Containers
    if isinstance(value, (list, tuple)):
        tag = "list" if isinstance(value, list) else "tuple"
        return (tag, [canonicalize(v) for v in value])
    if isinstance(value, (set, frozenset)):
        elems = [canonicalize(v) for v in value]
        elems.sort()
        tag = "set" if isinstance(value, set) else "frozenset"
        return (tag, elems)
    if isinstance(value, dict):
        items = []
        for k, v in value.items():
            k_canon = canonicalize(k)
            # Use JSON string of canonical key to ensure a stable, comparable key
            k_str = json.dumps(k_canon, separators=(",", ":"), ensure_ascii=False)
            items.append((k_str, canonicalize(v)))
        items.sort(key=lambda kv: kv[0])
        return ("dict", items)

    # Fallback: type name + repr
    return ("object", type(value).__name__, repr(value))


def canonical_bytes(value: Any) -> bytes:
    """Serialize canonicalized value to compact UTF-8 JSON bytes."""
    return json.dumps(canonicalize(value), separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def hash_value(value: Any) -> int:
    """
    Generates a consistent hash for a given value using SHA-256 over a
    canonical, type-tagged JSON serialization.

    Args:
        value (Any): The value to hash.

    Returns:
        int: The integer representation of the SHA-256 hash.
    """
    sha256_hash = hashlib.sha256(canonical_bytes(value)).hexdigest()
    return int(sha256_hash, 16)


def hash_sequence(iterable: List[Any]) -> int:
    """
    Hashes a list by converting it to a tuple and then applying a consistent hash.

    Args:
        iterable (List[Any]): The list to hash.

    Returns:
        int: The consistent hash value of the tuple.
    """
    # Convert the list to a tuple to make it immutable and hashable
    tuple_representation = tuple(iterable)
    # Generate a consistent hash for the tuple
    return hash_value(tuple_representation)


def hash_set(iterable: List[Any]) -> int:
    """
    Hash a multiset order-independently by hashing elements first and sorting the hashes.

    Args:
        iterable (List[Any]): Elements to hash order-independently.

    Returns:
        int: Consistent hash of the multiset contents.
    """
    hashed = sorted(hash_value(e) for e in iterable)
    return hash_value(tuple(hashed))


def hash_bounded(value: Any, nbits: int = 10) -> int:
    """
    Hashes a value, applies a bitmask to limit its size, and ensures the result is within a specified range.

    This function combines the functionality of masking and range limitation into a single step.
    It first generates a consistent hash of the input value, applies a bitmask based on `nbits` to limit
    the hash size, and then adjusts the result to ensure it falls within the range [2, 2**nbits - 1].

    Args:
        value (Any): The value to hash.
        nbits (int, optional): Number of bits to limit the hash. Defaults to 10.

    Returns:
        int: The hashed value limited to `nbits`, and in the range [2, 2**nbits - 1]

    Raises:
        ValueError: If `nbits` is less than 2.
    """
    if nbits < 2:
        raise ValueError("nbits must be at least 2 to ensure a valid hash range.")

    # Calculate the maximum value based on the number of bits
    max_index = 2 ** nbits
    # Create a bitmask with the specified number of bits (e.g., nbits=10 -> mask=0b1111111111)
    mask = (1 << nbits) - 1
    # Apply the bitmask to the consistent hash using bitwise AND to limit the hash size
    masked_hash = hash_value(value) & mask
    # Use modulo to ensure the masked hash is within [0, max_index - 3]
    limited_hash = masked_hash % (max_index - 2)
    # Adjust the hash to ensure it is at least 2
    final_hash = limited_hash + 2
    return final_hash

def hash_node(node_idx: int, graph: nx.Graph) -> int:
    """
    Computes a hash for a node based on its label and the labels of its neighbors.

    Args:
        node_idx (int): The index of the node in the graph.
        graph (nx.Graph): The graph containing the node.

    Returns:
        int: The computed hash for the node.
    """
    # Retrieve the label of the current node.
    # If the node doesn't have a 'label' attribute, default to an empty string.
    node_label = graph.nodes[node_idx].get('label', '')
    
    # Generate a consistent hash for the node's label using the hash_value function.
    node_label_hash = hash_value(node_label)
    
    # Fast check for isolated node: degree 0
    if graph.degree[node_idx] == 0:
        return node_label_hash
    
    # Initialize a list to store hashes related to the node's neighborhood.
    neighborhood_hashes = []
    
    # Iterate over all neighbors of the current node.
    for neighbor_idx in graph.neighbors(node_idx):
        # Retrieve the neighbor's label; default to an empty string if not present.
        neighbor_label = graph.nodes[neighbor_idx].get('label', '')
        # Generate a consistent hash for the neighbor's label.
        neighbor_label_hash = hash_value(neighbor_label)
        
        # Retrieve the label of the edge connecting the current node and its neighbor.
        # Default to an empty string if the edge doesn't have a 'label' attribute.
        edge_label = graph.edges[node_idx, neighbor_idx].get('label', '')
        # Generate a consistent hash for the edge's label.
        edge_label_hash = hash_value(edge_label)
        
        # Combine the neighbor's label hash and the edge's label hash into a list.
        combined_hashes = [neighbor_label_hash, edge_label_hash]
        # Generate a consistent hash for the combined list using the hash_set function.
        combined_hash = hash_set(combined_hashes)
        
        # Append the combined hash to the neighborhood_hashes list.
        neighborhood_hashes.append(combined_hash)
    
    # After processing all neighbors, combine all neighborhood-related hashes.
    # The hash_set function ensures that the combination is order-independent.
    neighborhood_hash = hash_set(neighborhood_hashes)
    
    # Create a sequence containing the neighborhood hash and the node's own label hash.
    combined_sequence = [neighborhood_hash, node_label_hash]
    # Generate the final hash for the node by hashing the combined sequence.
    node_hash = hash_sequence(combined_sequence)
    
    # Return the computed hash for the node.
    return node_hash


def compute_node_labels_set_hash(graph: nx.Graph) -> int:
    """
    Computes a hash value based on the set of node labels in the graph.
    
    The function retrieves the 'label' attribute from each node (using '-' as a default),
    sorts them to ensure order-independence, and then computes the hash using hash_set.
    
    Args:
        graph (nx.Graph): A NetworkX graph with nodes that may have a 'label' attribute.
    
    Returns:
        int: The hash value representing the set of node labels in the graph.
    """
    labels = [graph.nodes[node].get('label', '-') for node in graph.nodes()]
    return hash_set(labels)


def hash_rooted_graph(node_idx: int, graph: nx.Graph, node_hash_dict: Dict[int, int]) -> int:
    """
    Computes a hash for the subgraph rooted at a given node, considering all reachable nodes.

    Args:
        node_idx (int): The index of the root node.
        graph (nx.Graph): The graph containing the node.
        node_hash_dict (Dict[int, int]): Auxiliary dictionary containing precomputed hashes for each node.

    Returns:
        int: The hash representing the rooted subgraph.
    """
    def invert_dict(d: Dict[Any, Any]) -> Dict[Any, List[Any]]:
        """
        Inverts a dictionary mapping keys to single values into a dictionary mapping values to lists of keys.

        Args:
            d (Dict[Any, Any]): The dictionary to invert.

        Returns:
            Dict[Any, List[Any]]: The inverted dictionary where each key maps to a list of original keys.
        """
        # Initialize a defaultdict of lists to collect keys for each value
        inverted = defaultdict(list)
        # Iterate over each key-value pair in the original dictionary
        for key, value in d.items():
            # Append the key to the list corresponding to the value in the inverted dictionary
            inverted[value].append(key)
        # Convert defaultdict back to a regular dict before returning
        return dict(inverted)
    # Find all nodes reachable from the root node along with their shortest path lengths
    node_distance_dict = nx.single_source_shortest_path_length(graph, node_idx)

    # Invert the dictionary to group nodes by their distance from the root
    distance_to_nodes = invert_dict(node_distance_dict)

    # Initialize a list to store hash codes for each distance level
    distance_hashes = []
    # Iterate over each distance level in sorted order
    for distance, nodes_at_distance in sorted(distance_to_nodes.items()):
        # Collect the precomputed hashes of all nodes at the current distance
        node_hashes = [node_hash_dict[node] for node in nodes_at_distance]
        # Sort the node hashes to ensure consistent ordering
        sorted_node_hashes = sorted(node_hashes)
        # Hash the sorted list of node hashes to get a combined hash for this distance level
        distance_combined_hash = hash_sequence(sorted_node_hashes)
        # Append the combined hash to the list
        distance_hashes.append(distance_combined_hash)

    # Hash the list of distance-based hashes to obtain the final subgraph hash
    subgraph_hash = hash_sequence(distance_hashes)
    return subgraph_hash


def hash_graph(graph: nx.Graph, nbits: int = 19) -> int:
    """
    Computes a hash for the entire graph by hashing all rooted subgraphs.

    Args:
        graph (nx.Graph): The graph to hash.
        nbits (int, optional): Number of bits to limit the final hash. Defaults to 19.

    Returns:
        int: The hash representing the entire graph.

    Raises:
        ValueError: If `nbits` is not a positive integer.
    """
    # Validate that nbits is a positive integer
    if not isinstance(nbits, int) or nbits <= 0:
        raise ValueError("nbits must be a positive integer")

    # Create a dictionary mapping each node to its computed hash
    node_hashes: Dict[int, int] = {node: hash_node(node, graph) for node in graph.nodes()}

    # Create a dictionary mapping each node to the hash of its rooted subgraph
    rooted_subgraph_hashes: Dict[int, int] = {
        node: hash_rooted_graph(node, graph, node_hashes) for node in graph.nodes()
    }

    # Initialize a list to store hashes of all edges in the graph
    hashes_list = []
    node_labels_set_hash = compute_node_labels_set_hash(graph)
    hashes_list.append(node_labels_set_hash)
    
    # Iterate over each edge in the graph
    for u, v in graph.edges():
        # Retrieve and hash the edge's label
        edge_label = graph.edges[u, v].get('label', '')
        edge_label_hash = hash_value(edge_label)
        # Combine the rooted subgraph hashes of both nodes connected by the edge
        combined_node_hashes = hash_set([rooted_subgraph_hashes[u], rooted_subgraph_hashes[v]])
        # Hash the combined node hashes along with the edge label hash
        combined_edge_hash = hash_sequence([combined_node_hashes, edge_label_hash])
        # Append the combined edge hash to the list
        hashes_list.append(combined_edge_hash)

    # Hash the set of all edge hashes to obtain the final graph hash, bounded by nbits
    final_graph_hash = hash_bounded(hash_set(hashes_list), nbits=nbits)
    return final_graph_hash
