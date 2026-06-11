import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, vstack
from pathos.multiprocessing import ProcessingPool as Pool  
from coco_grape.module.quotient_graph import QuotientGraph
import networkx as nx
from scipy.sparse import lil_matrix, csr_matrix, vstack
from typing import Callable, Any, Dict, List
from collections import defaultdict
from coco_grape.module.quotient_graph import QuotientGraph


def graph_vectorize(
    graph: nx.Graph,
    decomposition_function: Callable[[Any], Any],
    nbits: int
) -> csr_matrix:
    """
    Converts a graph into a sparse CSR matrix based on node and edge label counts.

    Parameters:
    - graph (nx.Graph): The input NetworkX graph.
    - decomposition_function (Callable[[Any], Any]): Function to process the QuotientGraph.
    - nbits (int): Number of bits for hashing labels, determining feature space size.

    Returns:
    - csr_matrix: Sparse matrix representing the vectorized graph.
    """
    # Apply decomposition to obtain the quotient graph
    quotient_graph = decomposition_function(QuotientGraph(graph, nbits=nbits))
    
    # Extract labels from nodes
    node_hashes = [
        quotient_graph.quotient_graph.nodes[node]['label']
        for node in quotient_graph.quotient_graph.nodes()
    ]
    
    # Extract labels from edges
    edge_hashes = [
        quotient_graph.quotient_graph.edges[i, j]['label']
        for i, j in quotient_graph.quotient_graph.edges()
    ]
    
    # Combine node and edge labels
    combined_hashes = node_hashes + edge_hashes
    
    # Find unique labels and their counts
    idxs, counts = np.unique(combined_hashes, return_counts=True)
    
    # Prepend node and edge counts with reserved indices 0 and 1
    idxs = np.hstack([[0, 1], idxs])
    counts = np.hstack([
        [nx.number_of_nodes(graph), 2 * nx.number_of_edges(graph)],
        counts
    ])
    
    # Define feature vector size based on nbits
    max_idx = 2 ** nbits
    
    # Create a single-row sparse CSR matrix with counts at corresponding indices
    sparse_vector = csr_matrix(
        (counts, (np.zeros_like(idxs), idxs)),
        shape=(1, max_idx)
    )
    
    return sparse_vector

def quotient_graph_vectorize(
    graphs: List[nx.Graph],
    decomposition_function: Callable[[Any], Any],
    nbits: int,
    parallel=True
) -> List[csr_matrix]:
    """
    Vectorizes a list of graphs in parallel using Dill for serialization.

    Parameters:
    - graphs (List[nx.Graph]): List of NetworkX graphs to graph_vectorize.
    - decomposition_function (Callable[[Any], Any]): Function to process each graph.
    - nbits (int): Number of bits for hashing labels, determining feature space size.
    - n_jobs (int, optional): Number of parallel processes. Defaults to number of CPU cores.

    Returns:
    - List[csr_matrix]: List of sparse CSR matrices representing the vectorized graphs.
    """
    if parallel == False:
        return vstack([graph_vectorize(graph, decomposition_function, nbits) for graph in graphs])
    
    # Worker function to graph_vectorize a single graph
    def worker(graph: nx.Graph) -> csr_matrix:
        return graph_vectorize(graph, decomposition_function, nbits)
    
    # Initialize the processing pool
    with Pool(nodes=None) as pool:
        # Map the worker function to the list of graphs in parallel
        sparse_vectors = pool.map(worker, graphs)
    
    data_mtx = vstack(sparse_vectors)
    return data_mtx


def occurrences_list_to_csr(
    occurrences_list: List[Any], 
    degree: int, 
    nbits: int
) -> csr_matrix:
    """
    Converts a list of label occurrences into a sparse CSR matrix, reserving specific indices for graph statistics.

    Parameters:
    - occurrences_list (List[Any]): List of label occurrences associated with a subgraph node.
    - degree (int): Degree of the original node in the quotient graph.
    - nbits (int): Number of bits for hashing labels, determining feature space size.

    Returns:
    - csr_matrix: A 1x(2^nbits) sparse matrix where:
        - Index 0 contains a fixed count (e.g., 1) representing a reserved feature.
        - Index 1 contains the node's degree.
        - Subsequent indices correspond to hashed label counts.
    """
    # Find unique labels and their counts
    unique_labels, label_counts = np.unique(occurrences_list, return_counts=True)
    
    # Define reserved indices and their corresponding counts
    RESERVED_INDICES = [0, 1]
    RESERVED_COUNTS = [1, degree]
    
    # Combine reserved indices with label indices and counts
    all_indices = np.hstack([RESERVED_INDICES, unique_labels])
    all_counts = np.hstack([RESERVED_COUNTS, label_counts])
    
    # Define feature vector size based on nbits
    max_feature_index = 2 ** nbits
    
    # Create a single-row sparse CSR matrix with counts at corresponding indices
    sparse_vector = csr_matrix(
        (all_counts, (np.zeros_like(all_indices), all_indices)),
        shape=(1, max_feature_index)
    )
    
    return sparse_vector


def node_vectorize(
    graph: nx.Graph,
    decomposition_function: Callable[[Any], Any],
    nbits: int
) -> csr_matrix:
    """
    Converts the label associations of subgraph nodes into a sparse CSR matrix.

    Each row in the matrix corresponds to a node in the original graph, and each column
    represents a feature based on label occurrences. The first column (index 0) is
    reserved with a fixed value of 1 for all nodes.

    Parameters:
    - graph (nx.Graph): The input NetworkX graph to be vectorized.
    - decomposition_function (Callable[[Any], Any]): Function to process the QuotientGraph.
    - nbits (int): Number of bits for hashing labels, determining feature space size.

    Returns:
    - csr_matrix: Sparse matrix where each row corresponds to a graph node and columns represent label counts.
    """
    # Apply decomposition to obtain the quotient graph
    quotient_graph = decomposition_function(QuotientGraph(graph, nbits=nbits))
    
    # Initialize a defaultdict to map each subgraph node to its list of labels
    graph_dict: Dict[Any, List[Any]] = defaultdict(list)
    
    # Iterate over each node in the quotient graph
    for node_id in quotient_graph.quotient_graph.nodes():
        # Retrieve the label of the current node; handle missing labels
        label = quotient_graph.quotient_graph.nodes[node_id].get('label')
        if label is None:
            # Assign a default label or skip the node
            label = 0  # Assuming 0 is a valid default label; adjust as needed
            # Optionally, log a warning
            import logging
            logging.warning(f"Node {node_id} missing 'label' attribute. Assigned default label {label}.")
        
        # Obtain the subgraph associated with the current node
        subgraph = quotient_graph.subgraph(node_id)
        
        # Iterate over each node in the subgraph and append the label
        for subgraph_node_id in subgraph.nodes():
            graph_dict[subgraph_node_id].append(label)
    
    # Initialize a list to hold sparse vectors for each node
    node_feature_vector_list: List[csr_matrix] = []
    
    # Get a sorted list of all node IDs to maintain consistent ordering
    all_node_ids_sorted = sorted(graph.nodes())
    
    # Iterate over all nodes in the original graph
    for node_id in all_node_ids_sorted:
        if node_id in graph_dict:
            # Node has associated labels
            degree = graph.degree(node_id)
            occurrences = graph_dict[node_id]
            sparse_vector = occurrences_list_to_csr(occurrences, degree, nbits)
        else:
            # Node is missing from graph_dict; set reserved index 0 to 1
            # and other indices to 0
            reserved_indices = [0, 1]
            reserved_counts = [1, 0]  # Degree is 0 since node is missing
            all_indices = np.array(reserved_indices)
            all_counts = np.array(reserved_counts)
            max_feature_index = 2 ** nbits
            
            sparse_vector = csr_matrix(
                (all_counts, (np.zeros_like(all_indices), all_indices)),
                shape=(1, max_feature_index)
            )
        
        node_feature_vector_list.append(sparse_vector)
    
    # Vertically stack all node feature vectors into a single sparse matrix
    node_data_mtx = vstack(node_feature_vector_list)
    
    return node_data_mtx.tocsr()


def quotient_graph_node_vectorize(
    graphs: List[nx.Graph],
    decomposition_function: Callable[[Any], Any],
    nbits: int,
    parallel: bool = True
) -> List[csr_matrix]:
    """
    Vectorizes a list of graphs in parallel using Dill for serialization.

    Parameters:
    - graphs (List[nx.Graph]): List of NetworkX graphs to graph_vectorize.
    - decomposition_function (Callable[[Any], Any]): Function to process each graph.
    - nbits (int): Number of bits for hashing labels, determining feature space size.
    - n_jobs (int, optional): Number of parallel processes. Defaults to number of CPU cores.

    Returns:
    - List[csr_matrix]: List of sparse CSR matrices representing the vectorized graphs.
    """
    if parallel == False:
        return [node_vectorize(graph, decomposition_function, nbits) for graph in graphs]
    
    # Define a worker function that vectorizes a single graph using quotient_graph_node_vectorizer
    def worker(graph: nx.Graph) -> csr_matrix:
        return node_vectorize(graph, decomposition_function, nbits)
    
    # Initialize the processing pool with the specified number of jobs
    # If n_jobs is None, it defaults to the number of available CPU cores
    with Pool(nodes=None) as pool:
        # Map the worker function to the list of graphs in parallel
        sparse_data_mtx_list = pool.map(worker, graphs)
    
    # Return the list of sparse matrices
    return sparse_data_mtx_list


#------------------------------------------------------------------------------------------------------------------
# compute the average number of collisions as the average difference in the number of non-zero elements between two vectorized matrices of graphs

def avg_number_of_collisions(graphs, decomposition_function, nbits, parallel=True):
    """
    Computes the average difference in the number of non-zero elements between two
    vectorized matrices of graphs using a fixed `nbits1` and a variable `nbits2`.

    This function vectorizes a list of graphs twice:
    1. Using a fixed `nbits1` value (set to 20).
    2. Using a user-specified `nbits2` value.

    It then calculates the difference in the number of non-zero elements between
    the two resulting matrices and returns the average difference per graph.

    Parameters:
    - graphs (list): 
        A list of graph objects to be vectorized. Each graph should be compatible 
        with the `decomposition_function` provided.
        
    - decomposition_function (callable): 
        A function that takes a graph as input and returns its decomposition. This 
        function is applied to each graph in the `graphs` list before vectorization.
        
    - nbits (int): 
        The second `nbits` parameter (`nbits2`) used during the vectorization process. 
        This typically determines the dimensionality or granularity of the feature vectors.
        The first `nbits1` is fixed at 20 within the function.
        
    - parallel (bool, optional): 
        Determines whether the vectorization should be performed in parallel.
        - `True`: Utilizes `parallel_vectorize` for concurrent processing, which 
          can speed up computation on large datasets.
        - `False`: Uses the standard `vectorize` function for sequential processing.
        Default is `True`.

    Returns:
    - float: 
        The average difference in the number of non-zero elements per graph between 
        the two vectorized matrices. This is calculated as 
        `(nnz1 - nnz2) / len(graphs)`, where `nnz1` and `nnz2` are the number of 
        non-zero elements in the first (fixed `nbits1`) and second (`nbits2`) matrices, respectively.
    """
    # Define the first nbits value (fixed)
    nbits1 = 32
    # The second nbits value is provided as an argument
    nbits2 = nbits
    
    if parallel:
        # Vectorize all graphs using the first `nbits1` parameter with parallel processing
        mtx1 = quotient_graph_vectorize(graphs, decomposition_function, nbits=nbits1)
        # Vectorize all graphs using the second `nbits2` parameter with parallel processing
        mtx2 = quotient_graph_vectorize(graphs, decomposition_function, nbits=nbits2)
    else:
        # Vectorize all graphs using the first `nbits1` parameter without parallel processing
        mtx1 = quotient_graph_vectorize(graphs, decomposition_function, nbits=nbits1, n_jobs=1)
        # Vectorize all graphs using the second `nbits2` parameter without parallel processing
        mtx2 = quotient_graph_vectorize(graphs, decomposition_function, nbits=nbits2, n_jobs=1)
    
    # Compute the number of non-zero elements in the first matrix
    nnz1 = mtx1.nnz
    # Compute the number of non-zero elements in the second matrix
    nnz2 = mtx2.nnz
    
    # Calculate the difference in non-zero elements
    # This represents how the change in `nbits` affects the sparsity of the vectors
    diff = nnz1 - nnz2
    
    # Calculate the average difference per graph
    average_diff = diff / len(graphs)
    
    # Return the average difference
    return average_diff