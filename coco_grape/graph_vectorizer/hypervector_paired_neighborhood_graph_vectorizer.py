import numpy as np
import scipy as sp
import networkx as nx

from collections import defaultdict
from scipy.sparse import csr_matrix, vstack
from copy import copy
import multiprocessing_on_dill as mp

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

#----------------------------------------------------------------------------------------------------------------------------------

def hash_list(seq):
    return hash(tuple(seq))

def masked_hash_value(value, bitmask=4294967295):
    return hash(value) & bitmask

def hash_value(value, nbits=10):
    bitmask = pow(2, nbits) - 1
    return masked_hash_value(value, bitmask)

def node_hash(node_idx, graph):
    uh = hash(graph.nodes[node_idx]['label'])
    edges_h = [hash((hash(graph.nodes[v]['label']), hash(graph.edges[node_idx, v]['label']))) for v in graph.neighbors(node_idx)]
    nh = hash_list(sorted(edges_h))
    ext_node_h = hash((uh, nh))
    return ext_node_h

def invert_dict(mydict):
    reversed_dict = defaultdict(list)
    for key, value in mydict.items(): reversed_dict[value].append(key)
    return reversed_dict

def rooted_graph_hash(node_idx, graph, radius=1):
    node_idxs_to_dist_dict = nx.single_source_shortest_path_length(graph, node_idx, cutoff=radius)
    dist_to_node_idxs_dict = invert_dict(node_idxs_to_dist_dict)
    iso_distance_codes_list = []
    for dist in sorted(dist_to_node_idxs_dict):
        node_idxs = dist_to_node_idxs_dict[dist]
        codes_list = [graph.nodes[curr_node_idx]['node_hash'] for curr_node_idx in node_idxs]
        code = hash_list(sorted(codes_list))
        iso_distance_codes_list.append(code)
    code = hash_list(iso_distance_codes_list)
    return code
    

def graph_hypervector(original_graph, radius, distance, sparsity, nbits):
    graph = original_graph.copy()
    
    distance_hypervectors = hypervector_range(size=distance+1, sparsity=sparsity, nbits=nbits, seed=42)

    for node_idx in graph.nodes():
        graph.nodes[node_idx]['node_hash'] = node_hash(node_idx, graph)
    for node_idx in graph.nodes():
    	for r in range(radius):
	        label = rooted_graph_hash(node_idx, graph, radius=r)
	        if 'rooted_graph_hash' not in graph.nodes[node_idx]:
                graph.nodes[node_idx]['rooted_graph_hash'] = np.zeros(radius)
	        graph.nodes[node_idx]['rooted_graph_hash'] = label
	        graph.nodes[node_idx]['rooted_graph_hypervector'] = label_hypervector_encoder(label, sparsity, nbits)

    hypervectors_list = []
    for node_idx in graph.nodes():
        hypervector_i = graph.nodes[node_idx]['rooted_graph_hypervector']
        node_idxs_to_dist_dict = nx.single_source_shortest_path_length(graph, node_idx, cutoff=distance)
        dist_to_node_idxs_dict = invert_dict(node_idxs_to_dist_dict)
        for dist in sorted(dist_to_node_idxs_dict):
            node_idxs = dist_to_node_idxs_dict[dist]
            for curr_node_idx in node_idxs:
                hypervector_j = graph.nodes[curr_node_idx]['rooted_graph_hypervector']
                pair_hypervector = hypervector_set([hypervector_i, hypervector_j])
                pair_hypervector = hypervector_sequence([pair_hypervector, distance_hypervectors[dist]])
                hypervectors_list.append(pair_hypervector)
    hypervector = hypervector_set(hypervectors_list)
    return hypervector


def paired_graphs_hypervector_encoder(graphs, radius, distance, sparsity, nbits, parallel=True):
    if parallel:
        def func(graph): return graph_hypervector(graph, radius, distance, sparsity, nbits)
        n_cpus = mp.cpu_count()
        pool = mp.Pool(n_cpus)
        graph_hypervectors = pool.map(func, graphs)
        pool.close()
    else:
        graph_hypervectors = [graph_hypervector(graph, radius, distance, sparsity, nbits) for graph in graphs]
    graphs_hypervector_mtx = vstack(graph_hypervectors)
    return graphs_hypervector_mtx


class HypervectorPairedNeighborhoodGraphVectorizer(object):
    def __init__(self, radius, distance, sparsity, nbits, parallel=True):
        self.radius = radius
        self.distance = distance
        self.sparsity =sparsity
        self.nbits = nbits
        self.parallel = parallel

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
    
    def fit(self, graphs, targets=None):
        return self

    def transform(self, graphs):
        return paired_graphs_hypervector_encoder(graphs, self.radius, self.distance, self.sparsity, self.nbits, self.parallel)

    def fit_transform(self, graphs, targets=None):
        return self.fit(graphs, targets).transform(graphs)