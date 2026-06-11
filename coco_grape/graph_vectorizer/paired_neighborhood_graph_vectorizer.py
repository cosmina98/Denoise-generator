from collections import Counter
from collections import defaultdict
from copy import copy
from scipy.sparse import csr_matrix, vstack
import multiprocessing_on_dill as mp
import networkx as nx
import numpy as np
import scipy as sp
import hashlib


def _stable_hash(value):
    """Deterministic hash across processes/runs."""
    value_bytes = repr(value).encode("utf-8")
    return int(hashlib.sha256(value_bytes).hexdigest(), 16)

def hash_list(seq):
    return _stable_hash(tuple(seq))

def masked_hash_value(value, bitmask=4294967295):
    return _stable_hash(value) & bitmask

def hash_value(value, nbits=10):
    bitmask = pow(2, nbits) - 1
    return masked_hash_value(value, bitmask)

def node_hash(node_idx, graph):
    uh = _stable_hash(graph.nodes[node_idx]['label'])
    edges_h = [
        _stable_hash(
            (
                _stable_hash(graph.nodes[v]['label']),
                _stable_hash(graph.edges[node_idx, v]['label'])
            )
        )
        for v in graph.neighbors(node_idx)
    ]
    nh = hash_list(sorted(edges_h))
    ext_node_h = _stable_hash((uh, nh))
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

def items_to_sparse_histogram(items, nbits):
    histogram_dict = Counter(items)
    rows = np.zeros(len(histogram_dict)).astype(int)
    cols = np.array(list(histogram_dict.keys())).astype(int)
    data = np.array(list(histogram_dict.values())).astype(int)
    vector = sp.sparse.csr_matrix((data, (rows, cols)), shape=(1,2**nbits))
    return vector 

def graph_vector(original_graph, radius, distance, nbits):
    graph = original_graph.copy()
    
    for node_idx in graph.nodes():
        graph.nodes[node_idx]['node_hash'] = node_hash(node_idx, graph)
    for node_idx in graph.nodes():
        for r in range(radius):
            label = rooted_graph_hash(node_idx, graph, radius=r)
            if 'rooted_graph_hash' not in graph.nodes[node_idx]:
                graph.nodes[node_idx]['rooted_graph_hash'] = np.zeros(radius)
            graph.nodes[node_idx]['rooted_graph_hash'][r] = label
    
    codes_list = []
    for node_idx in graph.nodes():
        for code_i in graph.nodes[node_idx]['rooted_graph_hash']:
            node_idxs_to_dist_dict = nx.single_source_shortest_path_length(graph, node_idx, cutoff=distance)
            dist_to_node_idxs_dict = invert_dict(node_idxs_to_dist_dict)
            for dist in sorted(dist_to_node_idxs_dict):
                node_idxs = dist_to_node_idxs_dict[dist]
                for curr_node_idx in node_idxs:
                    for code_j in graph.nodes[curr_node_idx]['rooted_graph_hash']:
                        paired_code = hash_list([code_i, dist, code_j])
                        paired_code = hash_value(paired_code, nbits=nbits)
                        codes_list.append(paired_code)
    vector = items_to_sparse_histogram(codes_list, nbits)
    return vector

def paired_graphs_vector_encoder(graphs, radius, distance, nbits, parallel=True):
    if parallel:
        def func(graph): return graph_vector(graph, radius, distance, nbits)
        n_cpus = mp.cpu_count()
        pool = mp.Pool(n_cpus)
        graph_hypervectors = pool.map(func, graphs)
        pool.close()
    else:
        graph_hypervectors = [graph_vector(graph, radius, distance, nbits) for graph in graphs]
    graphs_vector_mtx = sp.sparse.vstack(graph_hypervectors)
    return graphs_vector_mtx


class PairedNeighborhoodGraphVectorizer(object):
    def __init__(self, radius=1, distance=3, nbits=10, dense=True, parallel=True):
        self.radius = radius
        self.distance = distance
        self.nbits = nbits
        self.dense = dense
        self.parallel = parallel

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
    
    def fit(self, graphs, targets=None):
        return self

    def transform(self, graphs):
        data_mtx = paired_graphs_vector_encoder(graphs, self.radius, self.distance, self.nbits, self.parallel)
        if self.dense:
            data_mtx = data_mtx.todense().A
        return data_mtx

    def fit_transform(self, graphs, targets=None):
        return self.fit(graphs, targets).transform(graphs)
