#!/usr/bin/env python
"""Provides interface."""

import networkx as nx
import numpy as np
import scipy as sp
from toolz import curry
from collections import Counter
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
import random
from toolz import partition_all
import multiprocessing_on_dill as mp
from coco_grape.module.construct import construct

#------------------------------------------------------------------------------------------------------------------
# vectorization of a graph: features are the hashed node subgraphs and the hashed pairs of nodes at the endopoints of edges
# values are the size of the subgraphs.
# when identical multiple subgraphs exist, then we sum up their values (i.e. equyivalently, multiply the num of occurrences of identical subgraphs by their size)


def counts_and_size_to_vec(vec, feature_counts, feature_size_dict):
    features_counter = Counter(feature_counts)
    for feature_id in features_counter:
        vec[0,feature_id] += features_counter[feature_id] * feature_size_dict[feature_id]
    return vec
    

def vectorize_vec(graphofgraphs):
    feature_size = graphofgraphs.graph['base'].graph['feature_size']
    vec = lil_matrix((1, feature_size), dtype=np.int8)

    node_feature_counts = [graphofgraphs.nodes[u]['label'] for u in graphofgraphs.nodes()]
    node_feature_size_dict = {graphofgraphs.nodes[u]['label']:graphofgraphs.nodes[u]['subgraph'].number_of_nodes() for u in graphofgraphs.nodes()}
    vec = counts_and_size_to_vec(vec, node_feature_counts, node_feature_size_dict)
    
    edge_feature_counts = [graphofgraphs.edges[e]['label'] for e in graphofgraphs.edges()]
    edge_feature_size_dict = {graphofgraphs.edges[e]['label']:(graphofgraphs.nodes[e[0]]['subgraph'].number_of_nodes()+graphofgraphs.nodes[e[1]]['subgraph'].number_of_nodes()) for e in graphofgraphs.edges()}
    vec = counts_and_size_to_vec(vec, edge_feature_counts, edge_feature_size_dict)

    vec[0,0] = graphofgraphs.graph['base'].number_of_nodes() #feature 0 encodes the existence of each node
    return vec


def vectorize(graphs, decomposition_function=None, nbits=16, dense=False):
    mtx = vstack([vectorize_vec(decomposition_function(construct(graph, nbits=nbits))) for graph in graphs])
    mtx = csr_matrix(mtx)
    if dense: mtx = mtx.todense().A 
    return mtx


# NOTE: when there is a collision with the hashed edge label, then the encodings in vectorize and node_vectorize can differ, 
# this is because edge_feature_size_dict can store only one value per edge label hash, while in the node_vectorize
# the values are summed up for each colliding edge label hash. In a sufficiently large hash domain, this discrepancy vanishes.


#------------------------------------------------------------------------------------------------------------------
# vectorization of each base node in a graph: the features are the hashed subgraphs which include the given base node and all edge features involving that base node    

def node_vectorize_vec(graphofgraphs, dense=False):
    base_graph = graphofgraphs.graph['base']
    feature_size = base_graph.graph['feature_size']
    nbits = base_graph.graph['nbits']
    number_of_nodes = base_graph.number_of_nodes()
    mtx = lil_matrix((number_of_nodes, feature_size), dtype=np.int8)
    for node_id in base_graph.nodes():
        for feature_id in base_graph.nodes[node_id]['location_graph_process_hash']: mtx[node_id,feature_id] += 1
        for feature_id in base_graph.nodes[node_id]['location_node_unlabelled_graph_process_hash']: mtx[node_id,feature_id] += 1
        for feature_id in base_graph.nodes[node_id]['location_edge_unlabelled_graph_process_hash']: mtx[node_id,feature_id] += 1
        for feature_id in base_graph.nodes[node_id]['location_unlabelled_graph_process_hash']: mtx[node_id,feature_id] += 1
        for feature_id in base_graph.nodes[node_id]['graph_process_hash']: mtx[node_id,feature_id] += 1
        for feature_id in base_graph.nodes[node_id]['node_unlabelled_graph_process_hash']: mtx[node_id,feature_id] += 1
        for feature_id in base_graph.nodes[node_id]['edge_unlabelled_graph_process_hash']: mtx[node_id,feature_id] += 1
        for feature_id in base_graph.nodes[node_id]['unlabelled_graph_process_hash']: mtx[node_id,feature_id] += 1
        for feature_id in base_graph.nodes[node_id]['process_hash']: mtx[node_id,feature_id] += 1
    for edge_id in graphofgraphs.edges():
        feature_id = graphofgraphs.edges[edge_id]['label']
        for node_id in graphofgraphs.nodes[edge_id[0]]['subgraph'].nodes(): mtx[node_id,feature_id] += 1
        for node_id in graphofgraphs.nodes[edge_id[1]]['subgraph'].nodes(): mtx[node_id,feature_id] += 1
    #overwrite reserved featrues 
    mtx[:,0] = 1 #feature 0 encodes the existence of a node
    for node_id in base_graph.nodes(): mtx[node_id,1] = base_graph.degree[node_id] #feature 1 encodes the degree of a node
    if dense: mtx = mtx.todense().A 
    return mtx

def node_vectorize(graphs, decomposition_function=None, nbits=16, dense=False):
    list_of_mtx = [node_vectorize_vec(decomposition_function(construct(graph, nbits=nbits)), dense=dense) for graph in graphs]
    return list_of_mtx

def graph_node_vectorize(graphs, decomposition_function=None, nbits=16, dense=False):
    mtx = vstack([csr_matrix(node_vectorize_vec(decomposition_function(construct(graph, nbits=nbits))).sum(axis=0)) for graph in graphs])
    mtx = csr_matrix(mtx)
    if dense: mtx = mtx.todense().A 
    return mtx


#------------------------------------------------------------------------------------------------------------------
# vectorization of each base edge in a graph: features are hashed subgraphs that involve that base edge and all edge features involving that base edge

def edge_vectorize_vec(graphofgraphs, dense=False):
    feature_size = graphofgraphs.graph['base'].graph['feature_size']
    graph = graphofgraphs.graph['base']
    edge_to_idx_map = {(min(edge_u, edge_v),max(edge_u, edge_v)):idx for idx, (edge_u, edge_v) in enumerate(graph.edges())}
    number_of_edges = graph.number_of_edges()
    mtx = lil_matrix((number_of_edges, feature_size), dtype=np.int8)
    mtx[:,0] = 1 #feature 0 encodes the existence of an edge
    for u in graphofgraphs.nodes():
        feature_id = graphofgraphs.nodes[u]['label']
        for edge_u, edge_v in graphofgraphs.nodes[u]['subgraph'].edges():
            edge_id = edge_to_idx_map[(min(edge_u, edge_v),max(edge_u, edge_v))]
            mtx[edge_id,feature_id] += 1
    for e in graphofgraphs.edges():
        feature_id = graphofgraphs.edges[e]['label']
        for edge_u, edge_v in graphofgraphs.nodes[e[0]]['subgraph'].edges():
            edge_id = edge_to_idx_map[(min(edge_u, edge_v),max(edge_u, edge_v))]
            mtx[edge_id,feature_id] += 1
        for edge_u, edge_v in graphofgraphs.nodes[e[1]]['subgraph'].edges():
            edge_id = edge_to_idx_map[(min(edge_u, edge_v),max(edge_u, edge_v))]
            mtx[edge_id,feature_id] += 1
    if dense: mtx = mtx.todense().A 
    return mtx



def edge_vectorize(graphs, decomposition_function=None, nbits=16, dense=False):
    list_of_mtx = [edge_vectorize_vec(decomposition_function(construct(graph, nbits=nbits)), dense=dense) for graph in graphs]
    return list_of_mtx


#------------------------------------------------------------------------------------------------------------------
# vectorization of an attributed graph: each feature is associated to the sum of the node attributes in the subgraph. 
# All node features are then summed up to obtain the graph vectorization. 


def get_node_attributes_matrix(graph, attribute_label='vec', dense=False):
    """Return a n_nodes x n_attribute_features matrix."""
    vec_list = [graph.nodes[u][attribute_label] for u in graph.nodes() if attribute_label in graph.nodes[u]]
    if len(vec_list) == 0:
        return csr_matrix(np.ones((graph.number_of_nodes(),1)))
    if sp.sparse.issparse(vec_list[0]):
        attribute_data_matrix = sp.sparse.vstack(vec_list)
    else:
        attribute_data_matrix = np.vstack(vec_list)
    #determine the dimensionality of the attribute
    n = attribute_data_matrix.shape[1]
    #add a zero vec = [1,0,0,...,0] when attribute is missing
    zero_vec = np.zeros((1,n))
    zero_vec[0,0] = 1
    if sp.sparse.issparse(vec_list[0]):
        vec_list = [graph.nodes[u].get(attribute_label,csr_matrix(zero_vec)) for u in graph.nodes()]
        attribute_data_matrix = sp.sparse.vstack(vec_list)
    else:
        vec_list = [graph.nodes[u].get(attribute_label,zero_vec) for u in graph.nodes()]
        attribute_data_matrix = np.vstack(vec_list)

    attribute_data_matrix = csr_matrix(attribute_data_matrix)
    if dense: attribute_data_matrix = attribute_data_matrix.todense().A 
    return attribute_data_matrix


def get_edge_attributes_matrix(graph, attribute_label='vec', dense=False):
    """Return a n_edges x n_attribute_features matrix."""
    vec_list = [graph.edges[e][attribute_label] for e in graph.edges() if attribute_label in graph.edges[e]]
    if len(vec_list) == 0:
        return csr_matrix(np.ones((graph.number_of_edges(),1)))
    if sp.sparse.issparse(vec_list[0]):
        attribute_data_matrix = sp.sparse.vstack(vec_list)
    else:
        attribute_data_matrix = np.vstack(vec_list)
    #determine the dimensionality of the attribute
    n = attribute_data_matrix.shape[1]
    #add a zero vec = [1,0,0,...,0] when attribute is missing
    zero_vec = np.zeros((1,n))
    zero_vec[0,0] = 1
    if sp.sparse.issparse(vec_list[0]):
        vec_list = [graph.edges[e].get(attribute_label,csr_matrix(zero_vec)) for e in graph.edges()]
        attribute_data_matrix = sp.sparse.vstack(vec_list)
    else:
        vec_list = [graph.edges[e].get(attribute_label,zero_vec) for e in graph.edges()]
        attribute_data_matrix = np.vstack(vec_list)

    attribute_data_matrix = csr_matrix(attribute_data_matrix)
    if dense: attribute_data_matrix = attribute_data_matrix.todense().A 
    return attribute_data_matrix


def structures(graphofgraphs, dense=False):
    graph = graphofgraphs.graph['base']
    attribute_label = graphofgraphs.graph['base'].graph['attribute_label']
    
    #for nodes
    node_structure_data_matrix = node_vectorize_vec(graphofgraphs, dense=dense) # n_nodes x n_features_sparse
    node_attribute_data_matrix = get_node_attributes_matrix(graph, attribute_label, dense=dense) # n_nodes x n_attribute_features 
    
    #for edges
    edge_structure_data_matrix = edge_vectorize_vec(graphofgraphs, dense=dense) # n_edges x n_features_sparse
    edge_attribute_data_matrix = get_edge_attributes_matrix(graph, attribute_label, dense=dense) # n_edges x n_attribute_features 

    return node_structure_data_matrix, node_attribute_data_matrix, edge_structure_data_matrix, edge_attribute_data_matrix


def structures_vectorize(graphs, decomposition_function=None, nbits=16, dense=False):
    list_of_mtx = [structures(decomposition_function(construct(graph, nbits=nbits)), dense=dense) for graph in graphs]
    return list_of_mtx
    

def attributed_vectorize_vec(graphofgraphs, dense=False):
    node_structure_data_matrix, node_attribute_data_matrix, edge_structure_data_matrix, edge_attribute_data_matrix = structures(graphofgraphs, dense=dense)

    #for nodes
    # sum all attributes vectors for each node in each fragment and obtain a n_attribute_features x n_features_sparse matrix
    node_attributed_data_matrix = node_structure_data_matrix.T.dot(node_attribute_data_matrix) # n_features_sparse x n_attribute_features
    # concatenate all vectors to flatten the representation
    vec_node_attributed_data_matrix = csr_matrix(node_attributed_data_matrix.reshape(1,-1)) # 1 x (n_attribute_features * n_features_sparse)
    
    #for edges
    # sum all attributes vectors for each edge in each fragment and obtain a n_attribute_features x n_features_sparse matrix
    edge_attributed_data_matrix = edge_structure_data_matrix.T.dot(edge_attribute_data_matrix) # n_features_sparse x n_attribute_features
    # concatenate all vectors to flatten the representation
    vec_edge_attributed_data_matrix = csr_matrix(edge_attributed_data_matrix.reshape(1,-1)) # 1 x (n_attribute_features * n_features_sparse)
    
    vec_attributed_data_matrix = sp.sparse.hstack([vec_node_attributed_data_matrix, vec_edge_attributed_data_matrix])
    
    if dense: vec_attributed_data_matrix = vec_attributed_data_matrix.todense().A 
    return vec_attributed_data_matrix


def attributed_vectorize(graphs, decomposition_function=None, nbits=16, dense=False):
    mtx = vstack([attributed_vectorize_vec(decomposition_function(construct(graph, nbits=nbits))) for graph in graphs])
    if dense: mtx = mtx.todense().A 
    return mtx


def attributed_nodes_edges_vectorize_vec(graphofgraphs, dense=False):
    node_structure_data_matrix, node_attribute_data_matrix, edge_structure_data_matrix, edge_attribute_data_matrix = structures(graphofgraphs, dense=dense)

    #for nodes
    # sum all attributes vectors for each node in each fragment and obtain a n_attribute_features x n_features_sparse matrix
    node_attributed_data_matrix = node_structure_data_matrix.T.dot(node_attribute_data_matrix) # n_features_sparse x n_attribute_features
    
    #for edges
    # sum all attributes vectors for each edge in each fragment and obtain a n_attribute_features x n_features_sparse matrix
    edge_attributed_data_matrix = edge_structure_data_matrix.T.dot(edge_attribute_data_matrix) # n_features_sparse x n_attribute_features
    
    return node_attributed_data_matrix, edge_attributed_data_matrix

def attributed_nodes_edges_vectorize(graphs, decomposition_function=None, nbits=16, dense=False):
    list_of_mtx = [attributed_nodes_edges_vectorize_vec(decomposition_function(construct(graph, nbits=nbits)), dense=dense) for graph in graphs]
    return list_of_mtx

#------------------------------------------------------------------------------------------------------------------
# annotate adds a sparse vector to each node (and edge) corresponding to the graph kernel feature description for that node


def annotate(graphs, decomposition_function=None, nbits=16, attribute_label='vec', concatenate_attributes=True):
    node_mtx_list = node_vectorize(graphs, decomposition_function, nbits)
    edge_mtx_list = edge_vectorize(graphs, decomposition_function, nbits)
    
    
    out_graphs = []
    for graph, node_mtx, edge_mtx in zip(graphs, node_mtx_list, edge_mtx_list):
        if concatenate_attributes:
            node_attribute_data_matrix = get_node_attributes_matrix(graph, attribute_label) # n_nodes x n_attribute_features 
            node_mtx = sp.sparse.hstack([csr_matrix(node_attribute_data_matrix), csr_matrix(node_mtx)]) # n_nodes x (n_attribute_features + n_features_sparse)
            node_mtx = node_mtx.todense().A
            edge_attribute_data_matrix = get_edge_attributes_matrix(graph, attribute_label) # n_edges x n_attribute_features 
            edge_mtx = sp.sparse.hstack([csr_matrix(edge_attribute_data_matrix), csr_matrix(edge_mtx)]) # n_edges x (n_attribute_features + n_features_sparse)
            edge_mtx = edge_mtx.todense().A

        out_graph = nx.Graph(graph)
        for vec,u in zip(node_mtx, graph.nodes()):
            out_graph.nodes[u][attribute_label] = vec.flatten()
        for vec,e in zip(edge_mtx, graph.edges()):
            out_graph.edges[e][attribute_label] = vec.flatten()
        out_graphs.append(out_graph)
    return out_graphs


#------------------------------------------------------------------------------------------------------------------
# parallel versions

def parallel_vectorize(graphs, decomposition_function=None, nbits=None, dense=False):
    def _make_vectorize_func(decomposition_function, nbits):
        def vectorize_func(graphs):
            mtx = vectorize(graphs, decomposition_function, nbits=nbits)
            return mtx
        return vectorize_func

    n_cpus = mp.cpu_count()
    batch_size = len(graphs)//n_cpus
    if len(graphs)<n_cpus: graphs_list = [graphs]
    else: graphs_list = list(partition_all(batch_size, graphs))
    vectorize_func = _make_vectorize_func(decomposition_function, nbits)
    pool = mp.Pool(n_cpus)
    results = pool.map(vectorize_func, graphs_list)
    pool.close()
    mtx = sp.sparse.vstack(results)
    if dense: mtx = mtx.todense().A 
    return mtx


def parallel_graph_node_vectorize(graphs, decomposition_function=None, nbits=None, dense=False):
    def _make_vectorize_func(decomposition_function, nbits):
        def vectorize_func(graphs):
            mtx = graph_node_vectorize(graphs, decomposition_function, nbits=nbits)
            return mtx
        return vectorize_func

    n_cpus = mp.cpu_count()
    batch_size = len(graphs)//n_cpus
    if len(graphs) < n_cpus: graphs_list = [graphs]
    else: graphs_list = list(partition_all(batch_size, graphs))
    vectorize_func = _make_vectorize_func(decomposition_function, nbits)
    pool = mp.Pool(n_cpus)
    results = pool.map(vectorize_func, graphs_list)
    pool.close()
    mtx = sp.sparse.vstack(results)
    if dense: mtx = mtx.todense().A 
    return mtx


def parallel_node_vectorize(graphs, decomposition_function=None, nbits=None, dense=False):
    def _make_node_vectorize_func(decomposition_function, nbits):
        def node_vectorize_func(graphs):
            mtx = node_vectorize(graphs, decomposition_function, nbits=nbits, dense=dense)
            return mtx
        return node_vectorize_func

    n_cpus = mp.cpu_count()
    batch_size = len(graphs)//n_cpus
    if batch_size < 2:
        graphs_list = [graphs]
    else:    
        graphs_list = list(partition_all(batch_size, graphs))
    node_vectorize_func = _make_node_vectorize_func(decomposition_function, nbits)
    pool = mp.Pool(n_cpus)
    results = pool.map(node_vectorize_func, graphs_list)
    pool.close()
    all_list_of_mtx = []
    for list_of_mtx in results:
        all_list_of_mtx.extend(list_of_mtx)
    return all_list_of_mtx


def parallel_edge_vectorize(graphs, decomposition_function=None, nbits=None, dense=False):
    def _make_edge_vectorize_func(decomposition_function, nbits):
        def edge_vectorize_func(graphs):
            mtx = edge_vectorize(graphs, decomposition_function, nbits=nbits, dense=dense)
            return mtx
        return edge_vectorize_func

    n_cpus = mp.cpu_count()
    batch_size = len(graphs)//n_cpus
    if batch_size < 2:
        graphs_list = [graphs]
    else:    
        graphs_list = list(partition_all(batch_size, graphs))
    edge_vectorize_func = _make_edge_vectorize_func(decomposition_function, nbits)
    pool = mp.Pool(n_cpus)
    results = pool.map(edge_vectorize_func, graphs_list)
    pool.close()
    all_list_of_mtx = []
    for list_of_mtx in results:
        all_list_of_mtx.extend(list_of_mtx)
    return all_list_of_mtx


def parallel_structures_vectorize(graphs, decomposition_function=None, nbits=16, dense=False):
    def _make_structures_vectorize_func(decomposition_function, nbits):
        def structures_vectorize_func(graphs):
            mtx = structures_vectorize(graphs, decomposition_function, nbits=nbits, dense=dense)
            return mtx
        return structures_vectorize_func

    n_cpus = mp.cpu_count()
    batch_size = len(graphs)//n_cpus
    if batch_size < 2:
        graphs_list = [graphs]
    else:    
        graphs_list = list(partition_all(batch_size, graphs))
    structures_vectorize_func = _make_structures_vectorize_func(decomposition_function, nbits)
    pool = mp.Pool(n_cpus)
    results = pool.map(structures_vectorize_func, graphs_list)
    pool.close()
    all_list_of_mtx = []
    for list_of_mtx in results:
        all_list_of_mtx.extend(list_of_mtx)
    return all_list_of_mtx 


def parallel_attributed_vectorize(graphs, decomposition_function=None, nbits=16, dense=False):
    def _make_attributed_vectorize_func(decomposition_function, nbits):
        def attributed_vectorize_func(graphs):
            mtx = attributed_vectorize(graphs, decomposition_function, nbits=nbits, dense=dense)
            return mtx
        return attributed_vectorize_func

    n_cpus = mp.cpu_count()
    batch_size = len(graphs)//n_cpus
    if batch_size < 2:
        graphs_list = [graphs]
    else:    
        graphs_list = list(partition_all(batch_size, graphs))
    attributed_vectorize_func = _make_attributed_vectorize_func(decomposition_function, nbits)
    pool = mp.Pool(n_cpus)
    results = pool.map(attributed_vectorize_func, graphs_list)
    pool.close()
    if dense: data_mtx = np.vstack(results)
    else: data_mtx = sp.sparse.vstack(results)
    return data_mtx


def parallel_attributed_nodes_edges_vectorize(graphs, decomposition_function=None, nbits=None, dense=False):
    def _make_attributed_nodes_edges_vectorize_func(decomposition_function, nbits):
        def attributed_nodes_edges_vectorize_func(graphs):
            list_of_mtx = attributed_nodes_edges_vectorize(graphs, decomposition_function, nbits=nbits, dense=dense)
            return list_of_mtx
        return attributed_nodes_edges_vectorize_func

    n_cpus = mp.cpu_count()
    batch_size = len(graphs)//n_cpus
    if batch_size < 2:
        graphs_list = [graphs]
    else:    
        graphs_list = list(partition_all(batch_size, graphs))
    attributed_nodes_edges_vectorize_func = _make_attributed_nodes_edges_vectorize_func(decomposition_function, nbits)
    pool = mp.Pool(n_cpus)
    results = pool.map(attributed_nodes_edges_vectorize_func, graphs_list)
    pool.close()
    all_list_of_mtx = []
    for list_of_mtx in results:
        all_list_of_mtx.extend(list_of_mtx)
    return all_list_of_mtx

#------------------------------------------------------------------------------------------------------------------
# annotate adds a dense vector to each node (and edge) corresponding to the graph kernel feature description for that node

def parallel_annotate(graphs, decomposition_function=None, nbits=16, attribute_label='vec', concatenate_attributes=True):
    node_mtx_list = parallel_node_vectorize(graphs, decomposition_function, nbits)
    edge_mtx_list = parallel_edge_vectorize(graphs, decomposition_function, nbits)
    
    
    out_graphs = []
    for graph, node_mtx, edge_mtx in zip(graphs, node_mtx_list, edge_mtx_list):
        if concatenate_attributes:
            node_attribute_data_matrix = get_node_attributes_matrix(graph, attribute_label) # n_nodes x n_attribute_features 
            node_mtx = sp.sparse.hstack([csr_matrix(node_attribute_data_matrix), csr_matrix(node_mtx)]) # n_nodes x (n_attribute_features + n_features_sparse)
            node_mtx = node_mtx.todense().A
            edge_attribute_data_matrix = get_edge_attributes_matrix(graph, attribute_label) # n_edges x n_attribute_features 
            edge_mtx = sp.sparse.hstack([csr_matrix(edge_attribute_data_matrix), csr_matrix(edge_mtx)]) # n_edges x (n_attribute_features + n_features_sparse)
            edge_mtx = edge_mtx.todense().A

        out_graph = nx.Graph(graph)
        for vec,u in zip(node_mtx, graph.nodes()):
            out_graph.nodes[u][attribute_label] = vec.flatten()
        for vec,e in zip(edge_mtx, graph.edges()):
            out_graph.edges[e][attribute_label] = vec.flatten()
        out_graphs.append(out_graph)
    return out_graphs


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
    nbits1 = 20
    # The second nbits value is provided as an argument
    nbits2 = nbits
    
    if parallel:
        # Vectorize all graphs using the first `nbits1` parameter with parallel processing
        mtx1 = parallel_vectorize(graphs, decomposition_function, nbits=nbits1)
        # Vectorize all graphs using the second `nbits2` parameter with parallel processing
        mtx2 = parallel_vectorize(graphs, decomposition_function, nbits=nbits2)
    else:
        # Vectorize all graphs using the first `nbits1` parameter without parallel processing
        mtx1 = vectorize(graphs, decomposition_function, nbits=nbits1)
        # Vectorize all graphs using the second `nbits2` parameter without parallel processing
        mtx2 = vectorize(graphs, decomposition_function, nbits=nbits2)
    
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