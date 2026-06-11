#!/usr/bin/env python
"""Provides scikit interface."""

import networkx as nx
import numpy as np

from collections import Counter
from collections import defaultdict
from scipy.sparse import dok_matrix, csr_matrix
from scipy.sparse import vstack
from toolz import partition_all
import multiprocessing_on_dill as mp

from coco_grape.module.graph_hash import graph_hash


def circular_convolution(signal, kernel):
    dim = max(signal.shape[0],kernel.shape[0])
    padded_signal = np.zeros(dim)
    padded_signal[:signal.shape[0]] = signal
    padded_kernel = np.zeros(dim)
    padded_kernel[:kernel.shape[0]] = kernel
    conv = np.real(np.fft.ifft( np.fft.fft(padded_signal)*np.fft.fft(padded_kernel) ))
    return conv

def compute_node_id_distance_node_ids_list(graph, max_dist):
    node_id_distances_dict = dict(nx.all_pairs_shortest_path_length(graph, cutoff=max_dist))

    node_id_distance_node_ids_list = []
    for node_id, node_id_distances in node_id_distances_dict.items():
        distance_list = [[] for i in range(max_dist+1)]
        for node_id, distance in node_id_distances.items():
            distance_list[distance].append(node_id)
        node_id_distance_node_ids_list.append(distance_list)
    return node_id_distance_node_ids_list

def compute_radius_node_id_hash_list(graph, max_radius, nbits, use_node_unlabelled_graph=False, use_edge_unlabelled_graph=False):
    radius_node_id_hash_list = [[graph_hash(nx.ego_graph(graph, node_id, radius=r), nbits=nbits, use_node_unlabelled_graph=use_node_unlabelled_graph, use_edge_unlabelled_graph=use_edge_unlabelled_graph) for node_id in graph.nodes()] for r in range(max_radius+1)]
    return radius_node_id_hash_list

def make_feature(src_node_id_hash, dist, dest_node_id_hash, nbits=None):
    if src_node_id_hash > dest_node_id_hash: src_node_id_hash, dest_node_id_hash = dest_node_id_hash, src_node_id_hash
    if nbits is None: bitmask = 4294967295
    else: bitmask = pow(2, nbits) - 1
    code = hash((src_node_id_hash, dist, dest_node_id_hash)) & bitmask
    return code 

def make_node_features(graph, node_id_distance_node_ids_list, radius_node_id_hash_list, max_radius, max_dist, nbits):
    n_nodes = len(node_id_distance_node_ids_list)
    node_features = []
    for src_node_id in range(n_nodes):
        features = []
        for src_radius in range(max_radius+1):
            src_node_id_hash = radius_node_id_hash_list[src_radius][src_node_id]
            for dist in range(1,max_dist+1):
                dest_node_ids = node_id_distance_node_ids_list[src_node_id][dist]
                for dest_node_id in dest_node_ids:
                    for dest_radius in range(max_radius+1):
                        dest_node_id_hash = radius_node_id_hash_list[dest_radius][dest_node_id]
                        feature_src_dist_dest = make_feature(src_node_id_hash, dist, dest_node_id_hash, nbits=nbits)
                        features.append(feature_src_dist_dest)
        node_features.append(features)
    return node_features

def make_higher_order_node_features(graph, node_features, node_id_distance_node_ids_list, radius_node_id_hash_list, max_radius, max_dist, nbits):
    n_nodes = len(node_id_distance_node_ids_list)
    out_node_features = [[] for i in range(n_nodes)]
    for src_node_id in range(n_nodes):
        features = []
        for radius in range(max_radius+1):
            for src_node_id_hash in node_features[src_node_id]:
                for dist in range(1, max_dist+1):
                    dest_node_ids = node_id_distance_node_ids_list[src_node_id][dist]
                    for dest_node_id in dest_node_ids:
                        dest_node_id_hash = radius_node_id_hash_list[radius][dest_node_id]
                        feature_src_dist_dest = make_feature(src_node_id_hash, dist, dest_node_id_hash, nbits=nbits)
                        out_node_features[src_node_id].append(feature_src_dist_dest)
                        out_node_features[dest_node_id].append(feature_src_dist_dest)
    return out_node_features

def make_order_node_features(graph, order, max_radius, max_dist, nbits):
    def merge_list_of_nodes_features(node_features_list):
        curr_node_features = node_features_list[0]
        for i in range(1,len(node_features_list)):
            curr_node_features = [curr_node_features_i+node_features_list_i for curr_node_features_i,node_features_list_i in zip(curr_node_features, node_features_list[i])]
        return curr_node_features

    node_features_list = []
    radius_node_id_hash_list = compute_radius_node_id_hash_list(graph, max_radius=max_radius, nbits=nbits, use_node_unlabelled_graph=False, use_edge_unlabelled_graph=False)
    order0_node_features = [list(features) for features in zip(*radius_node_id_hash_list)]
    node_features_list.append(order0_node_features)
    unlabelled_radius_node_id_hash_list = compute_radius_node_id_hash_list(graph, max_radius=max_radius, nbits=nbits, use_node_unlabelled_graph=True, use_edge_unlabelled_graph=True)
    unlabelled_order0_node_features = [list(features) for features in zip(*unlabelled_radius_node_id_hash_list)]
    node_features_list.append(unlabelled_order0_node_features)
    unlabelled_radius_node_id_hash_list = compute_radius_node_id_hash_list(graph, max_radius=max_radius, nbits=nbits, use_node_unlabelled_graph=True, use_edge_unlabelled_graph=False)
    unlabelled_order0_node_features = [list(features) for features in zip(*unlabelled_radius_node_id_hash_list)]
    node_features_list.append(unlabelled_order0_node_features)
    unlabelled_radius_node_id_hash_list = compute_radius_node_id_hash_list(graph, max_radius=max_radius, nbits=nbits, use_node_unlabelled_graph=False, use_edge_unlabelled_graph=True)
    unlabelled_order0_node_features = [list(features) for features in zip(*unlabelled_radius_node_id_hash_list)]
    node_features_list.append(unlabelled_order0_node_features)
    
    if order > 0:
        node_id_distance_node_ids_list = compute_node_id_distance_node_ids_list(graph, max_dist=max_dist)
        node_features = make_node_features(graph, node_id_distance_node_ids_list, radius_node_id_hash_list, max_radius, max_dist, nbits)
        node_features_list.append(node_features)
        for i in range(order-1):
            node_features = make_higher_order_node_features(graph, node_features, node_id_distance_node_ids_list, radius_node_id_hash_list, max_radius, max_dist, nbits)
            node_features_list.append(node_features)
    node_features = merge_list_of_nodes_features(node_features_list)  
    return node_features

def transform(graph, order, max_radius, max_dist, nbits, use_attributes, attribute_key):
    graph = nx.convert_node_labels_to_integers(graph)
    node_features = make_order_node_features(graph, order, max_radius, max_dist, nbits)
    if use_attributes is False: 
        features = sum(node_features, [])
        feature_histogram = Counter(features)
        x = dok_matrix((1, 2**nbits), dtype=int)
        for feature in feature_histogram: x[0,feature] = feature_histogram[feature]
        x = csr_matrix(x)
        return x
    else:
        #extract node attributes
        attributes = [graph.nodes[node_id][attribute_key] for node_id in graph.nodes()]
        node_embeddings = np.array([circular_convolution(node_features[node_id].todense(), attributes[node_id]) for node_id in graph.nodes()])
        x = csr_matrix(node_embeddings.sum(axis=0).reshape(1,-1))
        return x


class NeighborhoodCombinationGraphVectorizer(object):
    def __init__(self,
                 order=1,
                 radius=1,
                 distance=6,
                 nbits=16,
                 use_attributes=False,
                 attribute_key='vec',
                 parallel=True):
        self.order = order
        self.radius = radius
        self.distance = distance
        self.nbits = nbits
        self.use_attributes = use_attributes
        self.attribute_key = attribute_key
        self.parallel = parallel
        
    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
    
    def fit(self, graphs, targets=None):
        return self
    
    def transform(self, graphs):
        if self.parallel: return self.transform_parallel(graphs)
        else: return self.transform_sequential(graphs)

    def transform_parallel(self, graphs):
        n_cpus = mp.cpu_count()
        batch_size = len(graphs)//n_cpus
        if len(graphs) < n_cpus: graphs_list = [graphs]
        else: graphs_list = list(partition_all(batch_size, graphs))
        pool = mp.Pool(n_cpus)
        results = pool.map(self.transform_sequential, graphs_list)
        pool.close()
        data_mtx = vstack(results)
        return data_mtx

    def transform_sequential(self, graphs):
        embeddings = vstack([transform(graph, order=self.order, max_radius=self.radius, max_dist=self.distance, nbits=self.nbits, use_attributes=self.use_attributes, attribute_key=self.attribute_key) for graph in graphs])
        return embeddings

    def fit_transform(self, graphs, targets=None):
        return self.fit(graphs, targets).transform(graphs)