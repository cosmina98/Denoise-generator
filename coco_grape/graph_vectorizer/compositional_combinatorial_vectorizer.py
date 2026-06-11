#!/usr/bin/env python
"""Provides scikit interface."""

import multiprocessing_on_dill as mp
import networkx as nx
import numpy as np
import scipy as sp

from collections import Counter
from collections import defaultdict
from itertools import combinations
from scipy.sparse import lil_matrix
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from toolz import partition_all
from sklearn.ensemble import ExtraTreesClassifier
from coco_grape.data_processor.processor import DataEstimator
from scipy.sparse.linalg import norm
from coco_grape.module.construct import construct


def circular_convolution(signal, kernel):
    dim = max(signal.shape[0],kernel.shape[0])
    padded_signal = np.zeros(dim)
    padded_signal[:signal.shape[0]] = signal
    padded_kernel = np.zeros(dim)
    padded_kernel[:kernel.shape[0]] = kernel
    conv = np.real(np.fft.ifft( np.fft.fft(padded_signal)*np.fft.fft(padded_kernel) ))
    return conv

def distance_relation_function(graph1, graph2, basegraph, min_size, max_size):
    try:
        dist = min(nx.shortest_path_length(basegraph, source=u, target=v) for u in graph1.nodes() for v in graph2.nodes())
        if min_size <= dist <= max_size: relation_value = dist
        else: relation_value = np.nan
    except Exception:
        relation_value = np.nan
        pass
    return relation_value 

def intersection_relation_function(graph1, graph2, basegraph, min_size, max_size, scale=10):
    try:
        nodes1 = set(u for u in graph1.nodes())
        nodes2 = set(u for u in graph2.nodes())
        score = nodes1.intersection(nodes2)
        score = score / sp.stats.gmean([len(nodes1), len(nodes2)])
        score = int(score * scale)
        if min_size <= score <= max_size: relation_value = score
        else: relation_value = np.nan
    except Exception:
        relation_value = np.nan
        pass
    return relation_value 

def get_relation_matrix(basegraph, subgraphs, min_size, max_size, relation_function):    
    n = len(subgraphs)
    relation_matrix = np.zeros((n,n))
    for i, subgraph_i in enumerate(subgraphs):
        for j, subgraph_j in enumerate(subgraphs):
            if j > i: #we always assume relation is symmetric
                relation_value = relation_function(subgraph_i, subgraph_j, basegraph, min_size, max_size)
                relation_matrix[i,j] = relation_matrix[j,i] = relation_value
    return relation_matrix

def make_feature(src_node_id_hash, relation_value, relation_type, dest_node_id_hash, nbits=None):
    if src_node_id_hash > dest_node_id_hash: src_node_id_hash, dest_node_id_hash = dest_node_id_hash, src_node_id_hash
    if nbits is None: bitmask = 4294967295
    else: bitmask = pow(2, nbits) - 1
    code = hash((src_node_id_hash, relation_value, relation_type, dest_node_id_hash)) & bitmask
    return code 

def make_combination_feature(hash_list, nbits=None):
    hash_list = sorted(hash_list)
    if nbits is None: bitmask = 4294967295
    else: bitmask = pow(2, nbits) - 1
    code = hash(tuple(hash_list)) & bitmask
    return code 

def all_relations_are_feasible(pairs, relation_matrix):
    for i,j in pairs:
        if np.isnan(relation_matrix[i,j]): return False
    return True     

def get_combination_idxs_list(order, relation_matrix):
    #identify j indices that are not associated to nan values (i.e. for which the relation i,j is a valid number) for each fragment i
    idxs_list = [np.where(~np.isnan(relation_matrix[i]))[0] for i in range(relation_matrix.shape[0])]
    combination_idxs_list = []
    for idxs in idxs_list:
        #add all order=k combinations of indices for each fragment i 
        combination_idxs_list.extend(combinations(idxs, order))
    #remove redundant combinations 
    combination_idxs_list = set(combination_idxs_list)
    return combination_idxs_list

def make_fragment_features(fragment_id_to_fragment_hash, relation_matrix, relation_type, fragment_id_to_fragment_weight, order, nbits, iterate_over_order_values, use_dont_care):
    fragment_feature_weight_dict = dict()
    num_fragments = len(fragment_id_to_fragment_hash)
    #initialize fragment_features with each single feature
    fragment_features = [[fragment_id_to_fragment_hash[i]] for i in range(num_fragments)]
    for fragment_hash, fragment_weight in zip(fragment_id_to_fragment_hash, fragment_id_to_fragment_weight): 
        fragment_feature_weight_dict[fragment_hash] = fragment_weight
    if order <= 1:
        return fragment_features, fragment_feature_weight_dict

    if iterate_over_order_values: order_range = list(range(2,order+1))
    else: order_range = [order]
    for curr_order in order_range:
        combination_idxs_list = get_combination_idxs_list(order, relation_matrix)
        for combination_idxs in combination_idxs_list:
            #all frgments are explicitely used in the combination
            pairs = list(combinations(combination_idxs, 2))
            if all_relations_are_feasible(pairs, relation_matrix):
                features = []
                combination_fragment_weight = []
                for i,j in pairs:
                    relation_value = relation_matrix[i,j]
                    feature = make_feature(fragment_id_to_fragment_hash[i], relation_value, relation_type, fragment_id_to_fragment_hash[j], nbits)
                    features.append(feature)
                    combination_fragment_weight.append(fragment_feature_weight_dict[fragment_id_to_fragment_hash[i]])
                    combination_fragment_weight.append(fragment_feature_weight_dict[fragment_id_to_fragment_hash[j]])
                combination_feature_hash = make_combination_feature(features, nbits)
                fragment_feature_weight_dict[combination_feature_hash] = np.mean(combination_fragment_weight)
                #add the clique feature to each fragment in the combination
                for idx in combination_idxs:
                    fragment_features[idx].append(combination_feature_hash)
            
            #replace one fragment at a time with a don't care
            for reference_idx in combination_idxs:
                pairs = list(combinations(combination_idxs, 2))
                if all_relations_are_feasible(pairs, relation_matrix):
                    features = []
                    combination_fragment_weight = []
                    for i,j in pairs:
                        relation_value = relation_matrix[i,j]
                        if i != reference_idx:
                            feature = make_feature(fragment_id_to_fragment_hash[i], relation_value, relation_type, fragment_id_to_fragment_hash[j], nbits)
                        else:
                            #when we consider the feature with reference_idx, then we substitute it with a dummy 0. In this way that feature acts as a don't care.
                            feature = make_feature(0, relation_value, relation_type, fragment_id_to_fragment_hash[j], nbits) 
                        features.append(feature)
                        if i != reference_idx:
                            combination_fragment_weight.append(fragment_feature_weight_dict[fragment_id_to_fragment_hash[i]])
                        else:
                            combination_fragment_weight.append(1)
                        combination_fragment_weight.append(fragment_feature_weight_dict[fragment_id_to_fragment_hash[j]])
                    combination_feature_hash = make_combination_feature(features, nbits)
                    fragment_feature_weight_dict[combination_feature_hash] = np.mean(combination_fragment_weight)
                    #add the clique feature to each fragment in the combination
                    for idx in combination_idxs:
                        fragment_features[idx].append(combination_feature_hash)
    return fragment_features, fragment_feature_weight_dict

def fragment_features_to_matrix(fragment_features, feature_size):
    n = len(fragment_features)
    mtx = lil_matrix((n, feature_size), dtype=np.int8)
    for i, feature_instances in enumerate(fragment_features):
        features_counter = Counter(feature_instances)
        for feature_id in features_counter:
            mtx[i,feature_id] += features_counter[feature_id]
    mtx = csr_matrix(mtx)
    return mtx

def dict_to_vector(values_dict, feature_size):
    vec = lil_matrix((1, feature_size))
    for key in sorted(values_dict):
        vec[0,key] = values_dict[key]
    vec = csr_matrix(vec)
    return vec

def compute_fragment_attributes(graphofsubgraphs, attribute_key):
    #extract fragment attributes as sum of all node attributes that compose the fragment 
    fragment_attributes = []
    for node_idx in graphofsubgraphs.nodes():
        subgraph = graphofsubgraphs.nodes[node_idx]['subgraph']
        attributes = [subgraph.nodes[node_id][attribute_key] for node_id in subgraph.nodes()]
        fragment_attribute = np.sum(attributes, axis=0).flatten()
        fragment_attributes.append(fragment_attribute)
    fragment_attributes = np.array(fragment_attributes)
    return fragment_attributes

def combinatorial_matrix(graph, order, decomposition_function, relation_functions, aggregate_function, min_size, max_size, nbits, iterate_over_order_values, use_attributes, attribute_key, use_dont_care):
    feature_size = 2**nbits
    graph = nx.convert_node_labels_to_integers(graph)
    graphofsubgraphs = decomposition_function(construct(graph, nbits=nbits))
    basegraph = graphofsubgraphs.graph['base']
    subgraphs = [graphofsubgraphs.nodes[node_idx]['subgraph'] for node_idx in graphofsubgraphs.nodes()]
    fragment_id_to_fragment_weight = [aggregate_function(subgraph) for subgraph in subgraphs] 
    for relation_type, relation_function in enumerate(relation_functions):
        if order == 1:
            relation_matrix = None
        else:
            relation_matrix = get_relation_matrix(basegraph, subgraphs, min_size, max_size, relation_function)
        fragment_id_to_fragment_hash = [graphofsubgraphs.nodes[node_idx]['label'] for node_idx in graphofsubgraphs.nodes()]
        fragment_features, fragment_feature_weight_dict = make_fragment_features(fragment_id_to_fragment_hash, relation_matrix, relation_type, fragment_id_to_fragment_weight, order, nbits, iterate_over_order_values, use_dont_care)
        fragment_mtx_ = fragment_features_to_matrix(fragment_features, feature_size) #each row is a fragment, each col is the num occurences of the combinatorial feature in which the feature occurs
        fragment_weights_vec_ = dict_to_vector(fragment_feature_weight_dict, feature_size)
        fragment_mtx_ = fragment_mtx_.multiply(fragment_weights_vec_) #use bradcasting to multiply each feature by the corresponding weight
        if relation_type == 0: fragment_mtx = fragment_mtx_
        else: fragment_mtx += fragment_mtx_
    return fragment_mtx, graphofsubgraphs

def combinatorial_vectorize(graph, order, decomposition_function, relation_functions, aggregate_function, min_size, max_size, nbits, iterate_over_order_values, use_attributes, attribute_key, use_dont_care):
    fragment_mtx, graphofsubgraphs = combinatorial_matrix(graph, order, decomposition_function, relation_functions, aggregate_function, min_size, max_size, nbits, iterate_over_order_values, use_attributes, attribute_key, use_dont_care)
    if use_attributes is False: 
        graph_embedding = fragment_mtx.sum(axis=0)
        graph_embedding = csr_matrix(graph_embedding)
        return graph_embedding
    else:
        #extract fragment attributes as sum of all node attributes that compose the fragment 
        fragment_attributes = compute_fragment_attributes(graphofsubgraphs, attribute_key)
        graph_embedding = csr_matrix(fragment_mtx.T.dot(fragment_attributes).reshape(1,-1)) # 1 x (n_attribute_features * n_features_sparse)
        return graph_embedding

def compute_node_feature_matrix(graphofsubgraphs):
    m = nx.number_of_nodes(graphofsubgraphs)
    subgraphs = [graphofsubgraphs.nodes[node_idx]['subgraph'] for node_idx in graphofsubgraphs.nodes()]
    n = nx.number_of_nodes(graphofsubgraphs.graph['base'])
    subgraphs_nodes_idxs = [[node_idx for node_idx in subgraph.nodes()] for subgraph in subgraphs] 
    nodes_x_feature_ids = np.zeros((n,m))
    for feature_id,subgraph in enumerate(subgraphs):
        for node_idx in subgraph.nodes():
            nodes_x_feature_ids[node_idx, feature_id] = 1
    return nodes_x_feature_ids

def combinatorial_node_vectorize(graph, order, decomposition_function, relation_functions, aggregate_function, min_size, max_size, nbits, iterate_over_order_values, use_attributes, attribute_key, use_dont_care):
    fragment_mtx, graphofsubgraphs = combinatorial_matrix(graph, order, decomposition_function, relation_functions, aggregate_function, min_size, max_size, nbits, iterate_over_order_values, use_attributes, attribute_key, use_dont_care)
    node_feature_matrix = compute_node_feature_matrix(graphofsubgraphs)
    node_matrix = fragment_mtx.T.dot(node_feature_matrix.T).T
    node_matrix = csr_matrix(node_matrix)
    return node_matrix

def constant_aggregate_function(subgraph):
    return 1

def node_size_aggregate_function(subgraph):
    return nx.number_of_nodes(subgraph)

def edge_weight_geometric_mean_aggregate_function(subgraph, attribute_key='weight', epsilon=0):
    weights = [subgraph.edges[u,v].get(attribute_key, 1) for u,v in subgraph.edges()] 
    if len(weights)>0:
        score = sp.stats.gmean(weights)
        score = np.nan_to_num(score)
    else:
        score = 0
    score += epsilon
    return score
    
class CompositionalCombinatorialGraphVectorizer(object):
    def __init__(self,
                 order=1,
                 relation_functions=[distance_relation_function],
                 aggregate_function=constant_aggregate_function,
                 size=1,
                 min_size=1,
                 decomposition_function=None,
                 nbits=16,
                 iterate_over_order_values=True,
                 use_attributes=False,
                 attribute_key='vec',
                 use_dont_care=True,
                 parallel=True):
        self.order = order
        self.relation_functions = relation_functions
        self.aggregate_function = aggregate_function
        self.size = size
        self.min_size = min_size
        self.decomposition_function = decomposition_function
        self.size = size
        self.nbits = nbits
        self.iterate_over_order_values = iterate_over_order_values
        self.use_attributes = use_attributes
        self.attribute_key = attribute_key
        self.use_dont_care = use_dont_care
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
        if len(graphs) < n_cpus: n_cpus = len(graphs)
        batch_size = len(graphs)//n_cpus
        graphs_list = list(partition_all(batch_size, graphs))
        pool = mp.Pool(n_cpus)
        results = pool.map(self.transform_sequential, graphs_list)
        pool.close()
        data_mtx = vstack(results)
        return data_mtx

    def transform_sequential(self, graphs):
        mtx = [combinatorial_vectorize(graph, order=self.order, decomposition_function=self.decomposition_function, relation_functions=self.relation_functions, aggregate_function=self.aggregate_function, min_size=self.min_size, max_size=self.size, nbits=self.nbits, iterate_over_order_values=self.iterate_over_order_values, use_attributes=self.use_attributes, attribute_key=self.attribute_key, use_dont_care=self.use_dont_care) for graph in graphs]
        embeddings = sp.sparse.vstack(mtx)
        return embeddings

    def fit_transform(self, graphs, targets=None):
        return self.fit(graphs, targets).transform(graphs)


class CompositionalCombinatorialNodeGraphVectorizer(object):
    def __init__(self,
                 order=1,
                 relation_functions=[distance_relation_function, intersection_relation_function],
                 aggregate_function=constant_aggregate_function,
                 size=1,
                 min_size=1,
                 decomposition_function=None,
                 nbits=16,
                 iterate_over_order_values=True,
                 use_attributes=False,
                 attribute_key='vec',
                 use_dont_care=True,
                 parallel=True):
        self.order = order
        self.relation_functions = relation_functions
        self.aggregate_function = aggregate_function
        self.size = size
        self.min_size = min_size
        self.decomposition_function = decomposition_function
        self.size = size
        self.nbits = nbits
        self.iterate_over_order_values = iterate_over_order_values
        self.use_attributes = use_attributes
        self.attribute_key = attribute_key
        self.use_dont_care = use_dont_care
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
        out = sum(results, [])
        return out

    def transform_sequential(self, graphs):
        node_embeddings = [combinatorial_node_vectorize(graph, order=self.order, decomposition_function=self.decomposition_function, relation_functions=self.relation_functions, aggregate_function=self.aggregate_function, min_size=self.min_size, max_size=self.size, nbits=self.nbits, iterate_over_order_values=self.iterate_over_order_values, use_attributes=self.use_attributes, attribute_key=self.attribute_key, use_dont_care=self.use_dont_care) for graph in graphs]
        return node_embeddings

    def fit_transform(self, graphs, targets=None):
        return self.fit(graphs, targets).transform(graphs)



#----------------------------------------------------------------------------------------------------------------------------------------------
class NodeImportanceCompositionalCombinatorialEstimator(object):
    def __init__(self, decomposition_function, nbits=14, order=1, size=1, min_size=1, n_estimators=300, use_attributes=False, parallel=True):
        self.graph_vectorizer = CompositionalCombinatorialGraphVectorizer(decomposition_function=decomposition_function, nbits=nbits, order=order, size=size, min_size=min_size, use_attributes=use_attributes, parallel=parallel)
        self.node_vectorizer = CompositionalCombinatorialNodeGraphVectorizer(decomposition_function=decomposition_function, nbits=nbits, order=order, size=size, min_size=min_size, use_attributes=use_attributes, parallel=parallel)
        self.classifier = ExtraTreesClassifier(n_estimators=n_estimators)
        self.estimator = DataEstimator(data_transformer=self.graph_vectorizer, estimator=self.classifier)
        self.parallel = parallel
        
    def fit(self, graphs, targets):
        self.estimator.fit(graphs, targets)
        return self
    
    def predict(self, graphs):
        return self.estimator.predict(graphs)
    
    def predict_proba(self, graphs):
        return self.estimator.predict_proba(graphs)

    def predict_and_predict_proba(self, graphs):
        return self.estimator.predict_and_predict_proba(graphs)
    
    def node_importance_sequential(self, latent_list):
        nodes_importances = [instance_mtx.dot(self.estimator.estimator.feature_importances_)/norm(instance_mtx, 1) for instance_mtx in latent_list]
        return nodes_importances

    def node_importance_parallel(self, latent_list):
        n_cpus = mp.cpu_count()
        batch_size = len(latent_list)//n_cpus
        if len(latent_list) < n_cpus: latent_list_list = [latent_list]
        else: latent_list_list = list(partition_all(batch_size, latent_list))
        pool = mp.Pool(n_cpus)
        results = pool.map(self.node_importance_sequential, latent_list_list)
        pool.close()
        all_list_of_mtx = []
        for list_of_mtx in results:
            all_list_of_mtx.extend(list_of_mtx)
        return all_list_of_mtx

    def node_importance(self, graphs):
        latent_list = self.node_vectorizer.transform(graphs)
        if self.parallel: nodes_importances = self.node_importance_parallel(latent_list)
        else: nodes_importances = self.node_importance_sequential(latent_list)
        return nodes_importances
