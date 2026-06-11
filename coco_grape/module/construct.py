#!/usr/bin/env python
"""Provides interface."""

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
from toolz import partition_all
from functools import reduce
import multiprocessing_on_dill as mp
import inspect
from coco_grape.module.graph_hash import graph_hash, hash_value, nodes_hash

def function_signature(function, function_arguments_dict):
    if 'graphofsubgraphs' in function_arguments_dict: function_arguments_dict.pop('graphofsubgraphs','')
    if 'graphofsubgraphs1' in function_arguments_dict: function_arguments_dict.pop('graphofsubgraphs1','')
    if 'graphofsubgraphs2' in function_arguments_dict: function_arguments_dict.pop('graphofsubgraphs2','')
    if 'decomposition_subroutine' in function_arguments_dict: function_arguments_dict.pop('decomposition_subroutine','')
    if 'edge_decomposition_subroutine' in function_arguments_dict: function_arguments_dict.pop('edge_decomposition_subroutine','')
    if 'binary_decomposition_subroutine' in function_arguments_dict: function_arguments_dict.pop('binary_decomposition_subroutine','')
    #function_name = inspect.stack()[1][3]
    function_name = function.__name__
    signature = function_name + str(function_arguments_dict)
    return signature

def serial_decomposition(graphs, decomposition_function, nbits):
    graphofsubgraphss = [decomposition_function(construct(graph, nbits=nbits)) for graph in graphs]
    return graphofsubgraphss

def parallel_decomposition(graphs, decomposition_function, nbits):
    def _make_decomposition_func(decomposition_function, nbits):
        def decomposition_func(graphs):
            return serial_decomposition(graphs, decomposition_function, nbits=nbits)
        return decomposition_func

    n_cpus = mp.cpu_count()
    batch_size = len(graphs)//n_cpus
    if batch_size < 2:
        graphs_list = [graphs]
    else:    
        graphs_list = list(partition_all(batch_size, graphs))
    decomposition_func = _make_decomposition_func(decomposition_function, nbits)
    pool = mp.Pool(n_cpus)
    results = pool.map(decomposition_func, graphs_list)
    pool.close()
    all_list_of_mtx = []
    for list_of_mtx in results:
        all_list_of_mtx.extend(list_of_mtx)
    return all_list_of_mtx

def decomposition(graphs, decomposition_function, nbits, parallel=True):
    if parallel == True:
        graphofsubgraphss = parallel_decomposition(graphs, decomposition_function, nbits)
    else:
        graphofsubgraphss = serial_decomposition(graphs, decomposition_function, nbits)
    return graphofsubgraphss


def make_signature(underlying_signature='', added_signature='', min_size=None, max_size=None):
    signature = added_signature
    if min_size is not None and max_size is not None: signature += '%d:%d'%(min_size,max_size)
    if underlying_signature != 'base': signature += '(%s)'%(underlying_signature)
    return signature

def make_edges_of_graph_of_subgraphs(graphofsubgraphs):
    #TODO: add abstraction_level
    nbits = graphofsubgraphs.graph['base'].graph['nbits']
    for edge_id in graphofsubgraphs.edges():
        u,v = edge_id
        edge_signature = graphofsubgraphs.edges[edge_id]['signature']
        signature_hash = hash(edge_signature)
        edge_label = hash(tuple(sorted([graphofsubgraphs.nodes[u]['label'],graphofsubgraphs.nodes[v]['label']])))
        edge_label = hash_value(edge_label, context=signature_hash, nbits=nbits)
        graphofsubgraphs.edges[edge_id]['label'] = edge_label
        process_hash = hash_value(edge_signature, context=signature_hash, nbits=nbits)
        graphofsubgraphs.edges[edge_id]['process_hash'] = process_hash
    return graphofsubgraphs

def make_subgraphs_of_graph_of_subgraphs(base_graph, bunches=None, signatures=[], subgraph_mode='node', abstraction_level='graph_process'):
    graphofsubgraphs = nx.Graph()
    nbits = base_graph.graph['nbits']
    for u in base_graph.nodes(): base_graph.nodes[u]['location_graph_process_hash'] = []
    for u in base_graph.nodes(): base_graph.nodes[u]['location_node_unlabelled_graph_process_hash'] = []
    for u in base_graph.nodes(): base_graph.nodes[u]['location_edge_unlabelled_graph_process_hash'] = []
    for u in base_graph.nodes(): base_graph.nodes[u]['location_unlabelled_graph_process_hash'] = []
    for u in base_graph.nodes(): base_graph.nodes[u]['graph_process_hash'] = []
    for u in base_graph.nodes(): base_graph.nodes[u]['node_unlabelled_graph_process_hash'] = []
    for u in base_graph.nodes(): base_graph.nodes[u]['edge_unlabelled_graph_process_hash'] = []
    for u in base_graph.nodes(): base_graph.nodes[u]['unlabelled_graph_process_hash'] = []
    for u in base_graph.nodes(): base_graph.nodes[u]['process_hash'] = []
    
    #add graph_process_hash and process_hash to graphofsubgraphs nodes
    for u, (bunch, signature) in enumerate(zip(bunches, signatures)):
        if subgraph_mode == 'node':
            subgraph = nx.subgraph(base_graph, bunch)
        elif subgraph_mode == 'edge':
            subgraph = nx.edge_subgraph(base_graph, bunch)
        signature_hash = hash(signature)
        process_hash = hash_value(signature, context=signature_hash, nbits=nbits)
        graph_process_hash = graph_hash(subgraph, context=signature_hash, nbits=nbits, use_node_unlabelled_graph=False, use_edge_unlabelled_graph=False)
        node_unlabelled_graph_process_hash = graph_hash(subgraph, context=signature_hash, nbits=nbits, use_node_unlabelled_graph=True, use_edge_unlabelled_graph=False)     
        edge_unlabelled_graph_process_hash = graph_hash(subgraph, context=signature_hash, nbits=nbits, use_node_unlabelled_graph=False, use_edge_unlabelled_graph=True)
        unlabelled_graph_process_hash = graph_hash(subgraph, context=signature_hash, nbits=nbits, use_node_unlabelled_graph=True, use_edge_unlabelled_graph=True)
        if abstraction_level=='graph_process': label = graph_process_hash
        if abstraction_level=='node_unlabelled_graph_process': label = node_unlabelled_graph_process_hash
        if abstraction_level=='edge_unlabelled_graph_process': label = edge_unlabelled_graph_process_hash
        if abstraction_level=='unlabelled_graph_process': label = unlabelled_graph_process_hash
        if abstraction_level=='process': label = process_hash
        graphofsubgraphs.add_node(
            u, 
            label=label, 
            graph_process_hash=graph_process_hash, 
            node_unlabelled_graph_process_hash=node_unlabelled_graph_process_hash,
            edge_unlabelled_graph_process_hash=edge_unlabelled_graph_process_hash,
            unlabelled_graph_process_hash=unlabelled_graph_process_hash,
            process_hash=process_hash, 
            subgraph=nx.Graph(subgraph), 
            signature=signature)
    
    #add location_graph_process_hash to base_graph nodes
    #append graph_process_hash and process_hash for each graphofsubgraphs nodes they are in to base_graph nodes
    for u in graphofsubgraphs.nodes():
        subgraph = graphofsubgraphs.nodes[u]['subgraph']
        graph_process_hash = graphofsubgraphs.nodes[u]['graph_process_hash']
        node_unlabelled_graph_process_hash = graphofsubgraphs.nodes[u]['node_unlabelled_graph_process_hash']
        edge_unlabelled_graph_process_hash = graphofsubgraphs.nodes[u]['edge_unlabelled_graph_process_hash']
        unlabelled_graph_process_hash = graphofsubgraphs.nodes[u]['unlabelled_graph_process_hash']
        process_hash = graphofsubgraphs.nodes[u]['process_hash']
        location_graph_process_hash_list = nodes_hash(subgraph, context=graph_process_hash, nbits=nbits, use_node_unlabelled_graph=False, use_edge_unlabelled_graph=False)
        location_node_unlabelled_graph_process_hash_list = nodes_hash(subgraph, context=node_unlabelled_graph_process_hash, nbits=nbits, use_node_unlabelled_graph=True, use_edge_unlabelled_graph=False)
        location_edge_unlabelled_graph_process_hash_list = nodes_hash(subgraph, context=edge_unlabelled_graph_process_hash, nbits=nbits, use_node_unlabelled_graph=False, use_edge_unlabelled_graph=True)
        location_unlabelled_graph_process_hash_list = nodes_hash(subgraph, context=unlabelled_graph_process_hash, nbits=nbits, use_node_unlabelled_graph=True, use_edge_unlabelled_graph=True)
        for node_id, location_graph_process_hash, location_node_unlabelled_graph_process_hash, location_edge_unlabelled_graph_process_hash, location_unlabelled_graph_process_hash in zip(subgraph.nodes(),location_graph_process_hash_list, location_node_unlabelled_graph_process_hash_list, location_edge_unlabelled_graph_process_hash_list, location_unlabelled_graph_process_hash_list):
            base_graph.nodes[node_id]['location_graph_process_hash'].append(location_graph_process_hash)
            base_graph.nodes[node_id]['location_node_unlabelled_graph_process_hash'].append(location_node_unlabelled_graph_process_hash)
            base_graph.nodes[node_id]['location_edge_unlabelled_graph_process_hash'].append(location_edge_unlabelled_graph_process_hash)
            base_graph.nodes[node_id]['location_unlabelled_graph_process_hash'].append(location_unlabelled_graph_process_hash)
        for node_id in graphofsubgraphs.nodes[u]['subgraph'].nodes():
            base_graph.nodes[node_id]['graph_process_hash'].append(graph_process_hash)
            base_graph.nodes[node_id]['node_unlabelled_graph_process_hash'].append(node_unlabelled_graph_process_hash)
            base_graph.nodes[node_id]['edge_unlabelled_graph_process_hash'].append(edge_unlabelled_graph_process_hash)
            base_graph.nodes[node_id]['unlabelled_graph_process_hash'].append(unlabelled_graph_process_hash)
            base_graph.nodes[node_id]['process_hash'].append(process_hash)
    graphofsubgraphs.graph['base'] = base_graph
    return graphofsubgraphs


def make_graph_of_subgraphs(base_graph, node_bunches=None, edge_bunches=None, edges=None, node_signatures=[], edge_signatures=[], abstraction_level='graph_process'):
    """
    Make a graph of graph starting from the base graph in 'base_graph' and a list of lists of node ids or list of lists of edges ids. 

    node_bunches: Each list of nodes is used to induce a subgraph to be associated to a node of the graph of graph.
    edge_bunches: Each subgraph can be identified by a list of nodes, or a list of edges (in this case it is a edge induced subgraph).
    edges: Edges can be provided explicitly between nodes of the graph of graph. 
    The label of a node of the graph of graph is computed as a specific permutation invariant hash of the subgraph.
    A 'signature' string is used to seed the hash function so that two isomophic subgraphs that are produced by different procedures get a distinct encoding. 
    """
    if node_bunches is not None:
        return make_subgraphs_of_graph_of_subgraphs(base_graph, bunches=node_bunches, signatures=node_signatures, subgraph_mode='node', abstraction_level=abstraction_level)
    if edge_bunches is not None:
        return make_subgraphs_of_graph_of_subgraphs(base_graph, bunches=edge_bunches, signatures=edge_signatures, subgraph_mode='edge', abstraction_level=abstraction_level)
    return graphofsubgraphs


def extend_bunch_data(bunches, bunch_signatures, components, underlying_signature, signature):
    bunches.extend(components)
    node_signature = make_signature(underlying_signature=underlying_signature, added_signature=signature)
    bunch_signatures.extend([node_signature]*len(components))

def make_decomposition_function(decomposition_subroutine):
    def decomposition_function(graphofsubgraphs, **args):
        subgraph_mode = args.get('subgraph_mode','node')
        abstraction_level = args.get('abstraction_level', 'graph_process')
        signature = function_signature(decomposition_subroutine, locals())
        bunches = []
        bunch_signatures = []
        subgraphs = [graphofsubgraphs.nodes[u]['subgraph'] for u in graphofsubgraphs.nodes()]
        feature_ids = [graphofsubgraphs.nodes[u]['label'] for u in graphofsubgraphs.nodes()]
        underlying_signatures = [graphofsubgraphs.nodes[u]['signature'] for u in graphofsubgraphs.nodes()]
        for subgraph, feature_id, underlying_signature in zip(subgraphs, feature_ids, underlying_signatures):
            components = decomposition_subroutine(subgraph, basegraph=graphofsubgraphs.graph['base'], feature_id=feature_id, **args)
            extend_bunch_data(bunches, bunch_signatures, components, underlying_signature, signature)
        if subgraph_mode == 'node':
            out_graphofsubgraphs = make_graph_of_subgraphs(base_graph=graphofsubgraphs.graph['base'], node_bunches=bunches, node_signatures=bunch_signatures, abstraction_level=abstraction_level)
        elif subgraph_mode == 'edge':
            out_graphofsubgraphs = make_graph_of_subgraphs(base_graph=graphofsubgraphs.graph['base'], edge_bunches=bunches, edge_signatures=bunch_signatures, abstraction_level=abstraction_level)
        return out_graphofsubgraphs
    return decomposition_function

def make_multi_decomposition_function(decomposition_subroutine):
    def decomposition_function(graphofsubgraphs, **args):
        subgraph_mode = args.get('subgraph_mode','node')
        abstraction_level = args.get('abstraction_level', 'graph_process')
        signature = function_signature(decomposition_subroutine, locals())
        bunches = []
        bunch_signatures = []
        subgraphs = [graphofsubgraphs.nodes[u]['subgraph'] for u in graphofsubgraphs.nodes()]
        feature_ids = [graphofsubgraphs.nodes[u]['label'] for u in graphofsubgraphs.nodes()]
        underlying_signatures = [graphofsubgraphs.nodes[u]['signature'] for u in graphofsubgraphs.nodes()]
        components, component_idxs_list = decomposition_subroutine(subgraphs, basegraph=graphofsubgraphs.graph['base'], feature_ids=feature_ids, **args)
        for component, component_idxs in zip(components, component_idxs_list):
            signature_combination = reduce(lambda s,t:s+'+'+t,[underlying_signatures[component_idx] for component_idx in component_idxs])
            extend_bunch_data(bunches, bunch_signatures, [component], signature_combination, signature)
        if subgraph_mode == 'node':
            out_graphofsubgraphs = make_graph_of_subgraphs(base_graph=graphofsubgraphs.graph['base'], node_bunches=bunches, node_signatures=bunch_signatures, abstraction_level=abstraction_level)
        elif subgraph_mode == 'edge':
            out_graphofsubgraphs = make_graph_of_subgraphs(base_graph=graphofsubgraphs.graph['base'], edge_bunches=bunches, edge_signatures=bunch_signatures, abstraction_level=abstraction_level)
        return out_graphofsubgraphs
    return decomposition_function

def make_multi_edge_decomposition_function(edge_decomposition_subroutine):
    def edge_decomposition_function(graphofsubgraphs, **args):
        abstraction_level = args.get('abstraction_level', 'graph_process')
        signature = function_signature(edge_decomposition_subroutine, locals())
        bunches = []
        bunch_signatures = []
        subgraphs = [graphofsubgraphs.nodes[u]['subgraph'] for u in graphofsubgraphs.nodes()]
        feature_ids = [graphofsubgraphs.nodes[u]['label'] for u in graphofsubgraphs.nodes()]
        #underlying_signatures = [graphofsubgraphs.nodes[u]['signature'] for u in graphofsubgraphs.nodes()]
        out_graphofsubgraphs = graphofsubgraphs.copy()
        edges = edge_decomposition_subroutine(subgraphs, basegraph=graphofsubgraphs.graph['base'], feature_ids=feature_ids, **args)
        for edge in edges:
            out_graphofsubgraphs.add_edge(edge[0],edge[1], signature=signature)
        out_graphofsubgraphs = make_edges_of_graph_of_subgraphs(out_graphofsubgraphs)
        return out_graphofsubgraphs
    return edge_decomposition_function

def make_binary_multi_decomposition_function(binary_decomposition_subroutine):
    def binary_decomposition_function(graphofsubgraphs1, graphofsubgraphs2, **args):
        subgraph_mode = args.get('subgraph_mode','node')
        abstraction_level = args.get('abstraction_level', 'graph_process')
        signature = function_signature(binary_decomposition_subroutine, locals())
        bunches = []
        bunch_signatures = []
        subgraphs1 = [graphofsubgraphs1.nodes[u]['subgraph'] for u in graphofsubgraphs1.nodes()]
        feature_ids1 = [graphofsubgraphs1.nodes[u]['label'] for u in graphofsubgraphs1.nodes()]
        underlying_signatures1 = [graphofsubgraphs1.nodes[u]['signature'] for u in graphofsubgraphs1.nodes()]
        
        subgraphs2 = [graphofsubgraphs2.nodes[u]['subgraph'] for u in graphofsubgraphs2.nodes()]
        feature_ids2 = [graphofsubgraphs2.nodes[u]['label'] for u in graphofsubgraphs2.nodes()]
        underlying_signatures2 = [graphofsubgraphs2.nodes[u]['signature'] for u in graphofsubgraphs2.nodes()]
        
        components, component_idxs_list = binary_decomposition_subroutine(subgraphs1, subgraphs2, basegraph=graphofsubgraphs1.graph['base'], feature_ids1=feature_ids1, feature_ids2=feature_ids2, **args)
        for component, (component_idxs1, component_idxs2) in zip(components, component_idxs_list):
            signature_combination = reduce(lambda s,t:s+'+'+t,[(underlying_signatures1[component_idx1] if component_idx1 is not None else '') +(underlying_signatures2[component_idx2] if component_idx2 is not None else '') for component_idx1 in component_idxs1 for component_idx2 in component_idxs2])
            extend_bunch_data(bunches, bunch_signatures, [component], signature_combination, signature)
        if subgraph_mode == 'node':
            out_graphofsubgraphs = make_graph_of_subgraphs(base_graph=graphofsubgraphs1.graph['base'], node_bunches=bunches, node_signatures=bunch_signatures, abstraction_level=abstraction_level)
        elif subgraph_mode == 'edge':
            out_graphofsubgraphs = make_graph_of_subgraphs(base_graph=graphofsubgraphs1.graph['base'], edge_bunches=bunches, edge_signatures=bunch_signatures, abstraction_level=abstraction_level)
        return out_graphofsubgraphs
    return binary_decomposition_function


def construct(graph, attribute_label='vec', nbits=16):
    """
    Construct a graph of subgraphs from a base graph.

    A graph of subgraphs is a graph that has as nodes subgraphs of a base graph and as edges relations between these subgraphs.
    The default constructor builds a graph of subgraphs made of a single node which has a subgraph the whole base graph.
    The attribute_label is the dictionary key that allow access to real valued arrays for each node in the base graph.
    """
    base_graph = nx.Graph(graph)
    base_graph = nx.convert_node_labels_to_integers(base_graph)
    base_graph.graph['nbits'] = nbits
    base_graph.graph['bitmask'] = pow(2, nbits) - 1
    base_graph.graph['feature_size'] = base_graph.graph['bitmask'] + 1
    base_graph.graph['attribute_label'] = attribute_label
    node_bunches = [list(base_graph.nodes())]
    graphofsubgraphs = make_graph_of_subgraphs(base_graph, node_bunches=node_bunches, node_signatures=['base'])
    return graphofsubgraphs