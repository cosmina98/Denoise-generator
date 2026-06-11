import numpy as np
import scipy as sp
import networkx as nx
from scipy.sparse import csr_matrix, vstack
from copy import copy
import multiprocessing_on_dill as mp
from collections import defaultdict
from coco_grape.module.construct import decomposition

def hypervector(seed=1, sparsity=0.1, nbits=19):
    hypervec = np.random.default_rng(seed=seed).choice(2, 2**nbits, p=[1-sparsity,sparsity])
    hypervec = hypervec / np.sum(hypervec)
    hypervec = csr_matrix(hypervec)
    return hypervec

def hypervector_set(hypervectors_list):
    assert len(hypervectors_list) > 0
    if len(hypervectors_list) == 1: set_mtx = hypervectors_list[0]
    else: set_mtx = vstack(hypervectors_list)
    if set_mtx.ndim == 1: return csr_matrix(set_mtx)
    hypervec = csr_matrix(set_mtx.mean(axis=0))
    return hypervec

def permutation_matrix(size, seed=1):
    P = np.eye(size)  
    np.random.default_rng(seed=seed).shuffle(P)
    P = csr_matrix(P)
    return P

def hypervector_sequence(hypervectors_list, seed=42):
    size = hypervectors_list[0].shape[1]
    hypervectors_sequence_list = [hypervec.dot(permutation_matrix(size, seed=seed+i)) for i, hypervec in enumerate(hypervectors_list)]
    hypervec = hypervector_set(hypervectors_sequence_list)
    return hypervec

def hypervector_range(size, sparsity, nbits, seed=42):
    #n_idxs
    range_hypervecs = [hypervector(seed=seed+i, sparsity=sparsity, nbits=nbits) for i in range(size)]
    current_hypervec_list = [range_hypervecs[0]]
    current_hypervecs_list = [copy(current_hypervec_list)]
    for range_hypervec in range_hypervecs[1:]:
        current_hypervec_list.extend(range_hypervec)
        current_hypervecs_list.append(copy(current_hypervec_list))
    range_hypervecs = [hypervector_set(hypervectors_list) for hypervectors_list in current_hypervecs_list]
    return range_hypervecs

def label_hypervector_encoder(label, sparsity, nbits):
    return hypervector(seed=abs(hash(label)), sparsity=sparsity, nbits=nbits)

def annotate_node_and_edge_hyper_label(graph, sparsity, nbits, label_key):
    for node_idx in graph.nodes():
        graph.nodes[node_idx]['hyper_label'] = label_hypervector_encoder(graph.nodes[node_idx][label_key], sparsity, nbits)
    for node_src_idx, node_dst_idx in graph.edges():
        graph.edges[node_src_idx, node_dst_idx]['hyper_label'] = label_hypervector_encoder(graph.edges[node_src_idx, node_dst_idx][label_key], sparsity, nbits)
    return graph

def simple_edge_hypervector_encoder(node_src_idx, node_dst_idx, graph, sparsity, nbits, label_key='label'):
    node_src_label_code = graph.nodes[node_src_idx]['hyper_label']
    node_dst_label_code = graph.nodes[node_dst_idx]['hyper_label']
    edge_label_code = graph.edges[node_src_idx, node_dst_idx]['hyper_label']
    code = hypervector_set([node_src_label_code, node_dst_label_code])
    code = hypervector_sequence([code, edge_label_code])
    return code
    
def annotate_simple_edge_hypervector(graph, sparsity, nbits, label_key):
    for node_src_idx, node_dst_idx in graph.edges():
        simple_edge_hypervector = simple_edge_hypervector_encoder(node_src_idx, node_dst_idx, graph, sparsity, nbits, label_key)
        graph.edges[node_src_idx, node_dst_idx]['simple_edge_hypervector'] = simple_edge_hypervector
    return graph

def neighborhood_hypervector_encoder(node_idx, graph, sparsity, nbits, label_key='label'):
    neihborhood_list = [graph.nodes[node_idx]['hyper_label']] #add node label hypervector to address corner case of isolated node with no neighbors
    neighbor_idxs = [u for u in graph.neighbors(node_idx)]
    neihborhood_list += [graph.edges[node_idx, neighbor_idx]['simple_edge_hypervector'] for neighbor_idx in neighbor_idxs]
    neihborhood_code = hypervector_set(neihborhood_list)
    return neihborhood_code

def annotate_neighborhood_hypervector(graph, sparsity, nbits, label_key):
    for node_idx in graph.nodes():
        neighborhood_hypervector = neighborhood_hypervector_encoder(node_idx, graph, sparsity, nbits, label_key)
        graph.nodes[node_idx]['neighborhood_hypervector'] = neighborhood_hypervector
    return graph
        

def invert_dict(mydict):
    reversed_dict = defaultdict(list)
    for key, value in mydict.items(): reversed_dict[value].append(key)
    return reversed_dict

def node_hypervector_encoder(node_idx, graph, radius, distance_codes, sparsity, nbits, label_key='label'):
    node_idxs_to_dist_dict = nx.single_source_shortest_path_length(graph, node_idx, cutoff=radius)
    dist_to_node_idxs_dict = invert_dict(node_idxs_to_dist_dict)
    iso_distance_hypervectors_list = []
    for dist in sorted(dist_to_node_idxs_dict):
        node_idxs = dist_to_node_idxs_dict[dist]
        hypervectors_list = [graph.nodes[curr_node_idx]['neighborhood_hypervector'] for curr_node_idx in node_idxs]
        code = hypervector_set(hypervectors_list)
        code = hypervector_sequence([code, distance_codes[dist]])
        iso_distance_hypervectors_list.append(code)
    code = hypervector_sequence(iso_distance_hypervectors_list)
    return code

def annotate_node_hypervector(graph, radius, sparsity, nbits, label_key):
    distance_codes = hypervector_range(size=radius+1, sparsity=sparsity, nbits=nbits)    
    for node_idx in graph.nodes():
        node_hypervector = node_hypervector_encoder(node_idx, graph, radius, distance_codes, sparsity, nbits, label_key)
        graph.nodes[node_idx]['node_hypervector'] = node_hypervector
    return graph
    
def edge_hypervector_encoder(node_src_idx, node_dst_idx, graph, sparsity, nbits, label_key='label'):
    node_src_code = graph.nodes[node_src_idx]['neighborhood_hypervector']
    node_dst_code = graph.nodes[node_dst_idx]['neighborhood_hypervector']
    edge_label_code = graph.edges[node_src_idx, node_dst_idx]['hyper_label']
    code = hypervector_set([node_src_code, node_dst_code])
    code = hypervector_sequence([code, edge_label_code])
    return code
    
def annotate_edge_hypervector(graph, sparsity, nbits, label_key):
    for node_src_idx, node_dst_idx in graph.edges():
        edge_hypervector = edge_hypervector_encoder(node_src_idx, node_dst_idx, graph, sparsity, nbits, label_key)
        graph.edges[node_src_idx, node_dst_idx]['edge_hypervector'] = edge_hypervector
    return graph

def node_graph_hypervector_encoder(graph, radius, sparsity, nbits, label_key='label'):
    hypervector_graph = graph.copy()
    hypervector_graph = annotate_node_and_edge_hyper_label(hypervector_graph, sparsity, nbits, label_key)
    hypervector_graph = annotate_simple_edge_hypervector(hypervector_graph, sparsity, nbits, label_key)
    hypervector_graph = annotate_neighborhood_hypervector(hypervector_graph, sparsity, nbits, label_key)
    hypervector_graph = annotate_node_hypervector(hypervector_graph, radius, sparsity, nbits, label_key)
    hypervectors_list = [hypervector_graph.nodes[node_idx]['neighborhood_hypervector'] for node_idx in hypervector_graph.nodes()]
    return hypervectors_list
    
def node_graphs_hypervector_encoder(graphs, radius, sparsity, nbits, label_key='label', parallel=True):
    if parallel:
        def func(graph): return node_graph_hypervector_encoder(graph, radius, sparsity, nbits, label_key)
        n_cpus = mp.cpu_count()
        pool = mp.Pool(n_cpus)
        node_graph_hypervector_list = pool.map(func, graphs)
        pool.close()
    else:
        node_graph_hypervector_list = [node_graph_hypervector_encoder(graph, radius, sparsity, nbits, label_key) for graph in graphs]
    return node_graph_hypervector_list


def graph_hypervector_encoder(graph, radius, sparsity, nbits, label_key='label'):
    hypervector_graph = graph.copy()
    hypervector_graph = annotate_node_and_edge_hyper_label(hypervector_graph, sparsity, nbits, label_key)
    hypervector_graph = annotate_simple_edge_hypervector(hypervector_graph, sparsity, nbits, label_key)
    hypervector_graph = annotate_neighborhood_hypervector(hypervector_graph, sparsity, nbits, label_key)
    hypervector_graph = annotate_node_hypervector(hypervector_graph, radius, sparsity, nbits, label_key)
    hypervector_graph = annotate_edge_hypervector(hypervector_graph, sparsity, nbits, label_key)
    hypervectors_list = [hypervector_graph.nodes[node_idx]['neighborhood_hypervector'] for node_idx in hypervector_graph.nodes()] #add nodes label hypervectors to address corner case of graph with no edges
    hypervectors_list += [hypervector_graph.edges[node_src_idx, node_dst_idx]['edge_hypervector'] for node_src_idx, node_dst_idx in hypervector_graph.edges()]
    hypervectors_mtx = sp.sparse.vstack(hypervectors_list)
    output_hypervector = hypervectors_mtx.sum(axis=0)
    output_hypervector = csr_matrix(output_hypervector)
    #output_hypervector = hypervector_set(hypervectors_list)
    return output_hypervector


def graphs_hypervector_encoder(graphs, radius, sparsity, nbits, label_key='label', parallel=True):
    if parallel:
        def func(graph): return graph_hypervector_encoder(graph, radius, sparsity, nbits, label_key)
        n_cpus = mp.cpu_count()
        pool = mp.Pool(n_cpus)
        graph_hypervectors = pool.map(func, graphs)
        pool.close()
    else:
        graph_hypervectors = [graph_hypervector_encoder(graph, radius, sparsity, nbits, label_key) for graph in graphs]
    graphs_hypervector_mtx = vstack(graph_hypervectors)
    return graphs_hypervector_mtx


class HypervectorGraphVectorizer(object):
    def __init__(self, radius, sparsity, nbits, label_key='label', parallel=True):
        self.radius = radius
        self.sparsity =sparsity
        self.nbits = nbits
        self.label_key = label_key
        self.parallel = parallel

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
    
    def fit(self, graphs, targets=None):
        return self

    def transform(self, graphs):
        return graphs_hypervector_encoder(graphs, self.radius, self.sparsity, self.nbits, self.label_key, self.parallel)

    def fit_transform(self, graphs, targets=None):
        return self.fit(graphs, targets).transform(graphs)


class HypervectorNodeGraphVectorizer(object):
    def __init__(self, radius, sparsity, nbits, label_key='label', parallel=True):
        self.radius = radius
        self.sparsity =sparsity
        self.nbits = nbits
        self.label_key = label_key
        self.parallel = parallel

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
    
    def fit(self, graphs, targets=None):
        return self

    def transform(self, graphs):
        return node_graphs_hypervector_encoder(graphs, self.radius, self.sparsity, self.nbits, self.label_key, self.parallel)

    def fit_transform(self, graphs, targets=None):
        return self.fit(graphs, targets).transform(graphs)



class DecompositionHypervectorGraphVectorizer(object):
    def __init__(self, decomposition_function, nbits, radius=5, sparsity=.1, parallel=True):
        self.hypervector_graph_vectorizer = HypervectorGraphVectorizer(radius=radius, sparsity=sparsity, nbits=nbits, parallel=parallel)
        self.decomposition_function = decomposition_function
        self.nbits = nbits
        self.parallel = parallel

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
    
    def fit(self, graphs, targets=None):
        self.hypervector_graph_vectorizer.fit(graphs, targets)
        return self
    
    def decompose(self, graphs):
        graph_of_subgraphs_list = decomposition(graphs, self.decomposition_function, self.nbits, parallel=self.parallel)
        return graph_of_subgraphs_list
    
    def transform_single(self, graph_of_subgraphs):
        subgraphs = [graph_of_subgraphs.nodes[node_idx]['subgraph'] for node_idx in graph_of_subgraphs.nodes()]
        subgraph_embeddings_mtx = self.hypervector_graph_vectorizer.transform(subgraphs)
        embedding = subgraph_embeddings_mtx.sum(axis=0)
        embedding = csr_matrix(embedding)
        return embedding
            
    def transform(self, graphs):
        graph_of_subgraphs_list = self.decompose(graphs)
        embeddings = sp.sparse.vstack([self.transform_single(graph_of_subgraphs) for graph_of_subgraphs in graph_of_subgraphs_list])
        return embeddings

    def fit_transform(self, graphs, targets=None):
        return self.fit(graphs, targets).transform(graphs)

