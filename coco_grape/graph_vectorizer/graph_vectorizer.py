#!/usr/bin/env python
"""Provides scikit interface."""

from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from sklearn import svm
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.ensemble import ExtraTreesClassifier
from toolz import partition_all
import multiprocessing_on_dill as mp
from coco_grape.module.construct import decomposition
from coco_grape.module.vectorize import vectorize, node_vectorize, graph_node_vectorize, attributed_vectorize, parallel_vectorize, parallel_node_vectorize, parallel_graph_node_vectorize, parallel_attributed_vectorize
from coco_grape.data_processor.processor import DataEstimator

import networkx as nx
import numpy as np
import operator
import scipy as sp
from scipy.sparse.linalg import norm


class GraphVectorizer(object):
    def __init__(self, 
                 decomposition_function=None,
                 nbits=16,
                 feature_type='feature', #feature, node, node_list
                 use_attributes=False,
                 parallel=True,
                 dense=False):
        self.decomposition_function = decomposition_function
        self.nbits = nbits
        self.use_attributes = use_attributes
        self.feature_type = feature_type
        self.parallel = parallel
        self.dense = dense
        self.bitmask = pow(2, nbits) - 1
        
    def __repr__(self):
        infos=[]
        infos.append('nbits=%d'%self.nbits)
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
    
    def fit(self, graphs, targets=None):
        return self
    
    def transform(self, graphs):
        if self.parallel:
            if self.use_attributes: encodings = parallel_attributed_vectorize(graphs, decomposition_function=self.decomposition_function, nbits=self.nbits, dense=self.dense)
            else:
                if self.feature_type == 'feature': encodings = parallel_vectorize(graphs, decomposition_function=self.decomposition_function, nbits=self.nbits, dense=self.dense)
                elif self.feature_type == 'node': encodings = parallel_graph_node_vectorize(graphs, decomposition_function=self.decomposition_function, nbits=self.nbits, dense=self.dense)
                elif self.feature_type == 'node_list': encodings = parallel_node_vectorize(graphs, decomposition_function=self.decomposition_function, nbits=self.nbits, dense=self.dense)
                else: raise Error('Unknown feature_type:%s'%self.feature_type)
        else:
            if self.use_attributes: encodings = attributed_vectorize(graphs, decomposition_function=self.decomposition_function, nbits=self.nbits, dense=self.dense)
            else:
                if self.feature_type == 'feature': encodings = vectorize(graphs, decomposition_function=self.decomposition_function, nbits=self.nbits, dense=self.dense)
                elif self.feature_type == 'node': encodings = graph_node_vectorize(graphs, decomposition_function=self.decomposition_function, nbits=self.nbits, dense=self.dense)
                elif self.feature_type == 'node_list': encodings = node_vectorize(graphs, decomposition_function=self.decomposition_function, nbits=self.nbits, dense=self.dense)
                else: raise Error('Unknown feature_type:%s'%self.feature_type)
        return encodings


    def fit_transform(self, graphs, targets=None):
        return self.fit(graphs, targets).transform(graphs)

def NodeGraphVectorizer(decomposition_function=None, nbits=16, use_attributes=False, parallel=True, dense=False):
    return GraphVectorizer(decomposition_function=decomposition_function, nbits=nbits, feature_type='node_list', use_attributes=use_attributes, parallel=parallel, dense=dense)



#----------------------------------------------------------------------------------------------------------------------------------------------
class NodeImportanceDecompositionalEstimator(object):
    def __init__(self, decomposition_function, nbits=14, n_estimators=300, use_attributes=False, parallel=True):
        self.graph_vectorizer = GraphVectorizer(decomposition_function, nbits, feature_type='node', use_attributes=use_attributes, parallel=parallel)
        self.node_vectorizer = GraphVectorizer(decomposition_function, nbits, feature_type='node_list', use_attributes=use_attributes, parallel=parallel)
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
    
#----------------------------------------------------------------------------------------------------------------------------------------------

def extract_node_attributes_single(graph, attribute_key='vec'):
    attribute_mtx = [graph.nodes[node_idx][attribute_key] for node_idx in graph.nodes()]
    attribute_mtx = np.vstack(attribute_mtx)
    return attribute_mtx

def extract_node_attributes(graphs, attribute_key='vec'):
    node_mtx_list = [extract_node_attributes_single(graph, attribute_key=attribute_key) for graph in graphs]
    return node_mtx_list

class NodeAttributesNodeDecompositionGraphVectorizer(object):
    def __init__(self, node_graph_vectorizer, attribute_key='vec'):
        self.node_graph_vectorizer = node_graph_vectorizer
        self.attribute_key = attribute_key
        
    def fit(self, graphs, targets=None):
        self.node_graph_vectorizer.fit(graphs, targets)
        return self
    
    def transform_single(self, node_attributes_mtx, node_mtx):
        A = node_attributes_mtx
        F = node_mtx
        X = F.T.dot(A).T.reshape(1,-1)
        X = csr_matrix(X)
        return X
    
    def transform(self, graphs):
        node_attributes_mtx_list = extract_node_attributes(graphs, attribute_key=self.attribute_key)
        node_mtx_list = self.node_graph_vectorizer.transform(graphs)
        mtx = [self.transform_single(node_attributes_mtx, node_mtx) for node_attributes_mtx, node_mtx in zip(node_attributes_mtx_list, node_mtx_list)]
        mtx = vstack(mtx)
        return mtx
    
    def fit_transform(self, graphs, targets=None):
        return self.fit(graphs, targets).transform(graphs)


#----------------------------------------------------------------------------------------------------------------------------------------------

def accumulate(iterable, func=operator.add, *, initial=None):
    'Return running totals'
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], initial=100) --> 100 101 103 106 110 115
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    total = initial
    if initial is None:
        try:
            total = next(it)
        except StopIteration:
            return
    yield total
    for element in it:
        total = func(total, element)
        yield total
        
def sparse_vsplit(mtx, indices):
    mtx_list = []
    prev_i = 0
    for i in range(len(indices)):
        mtx_list.append(mtx[prev_i:indices[i]])
        prev_i = indices[i]
    return mtx_list

class DecompositionalGraphSetKernelEstimator(object):
    def __init__(self, metric='linear', base_decomposition_function=None, decomposition_function=None, nbits=None, parallel=True, probability=False):
        self.metric = metric
        self.base_decomposition_function = base_decomposition_function
        self.decomposition_function = decomposition_function
        self.nbits = nbits
        self.parallel = parallel
        self.estimator = svm.SVC(kernel='precomputed', probability=probability)
        
    def fit(self, graphs, targets=None):
        self.list_of_encodings = self.transform(graphs)
        if targets is not None:
            train_gram = self.train_gram_matrix()
            self.estimator.fit(train_gram, targets)
        return self
    
    def predict(self, graphs):
        test_gram = self.test_gram_matrix(graphs)
        preds = self.estimator.predict(test_gram)
        return preds
    
    def decision_func(self, graphs):
        test_gram = self.test_gram_matrix(graphs)
        preds = self.estimator.decision_func(test_gram)
        return preds

    def predict_proba(self, graphs):
        test_gram = self.test_gram_matrix(graphs)
        preds = self.estimator.predict_proba(test_gram)
        return preds

    def predict_and_predict_proba(self, graphs):
        test_gram = self.test_gram_matrix(graphs)
        preds = self.estimator.predict(test_gram)
        probs = self.estimator.predict_proba(test_gram)
        return preds, probs

    def vectorize(self, graphs):
        if self.parallel: encodings = parallel_vectorize(graphs, self.decomposition_function, self.nbits)
        else: encodings = vectorize(graphs, self.decomposition_function, self.nbits)
        return encodings
    
    def decompose(self, graphs):
        graph_of_subgraphs_list = decomposition(graphs, self.decomposition_function, self.nbits, parallel=self.parallel)
        return graph_of_subgraphs_list
    
    def transform(self, graphs):
        graph_of_subgraphs_list = self.decompose(graphs)
        #extract subgraphs
        list_of_list_of_subgraphs = [[graph_of_subgraphs.nodes[node_idx]['subgraph'] for node_idx in graph_of_subgraphs.nodes()] for graph_of_subgraphs in graph_of_subgraphs_list]
        #find size of list of subgraphs for each decomposed graph
        list_of_sizes = [len(list_of_subgraphs) for list_of_subgraphs in list_of_list_of_subgraphs]
        list_of_indices = list(accumulate(list_of_sizes))
        #vectorize all subgraphs
        all_encodings = self.vectorize(sum(list_of_list_of_subgraphs,[]))
        #split into lists of encodings each of size=size of list of original subgraphs
        list_of_encodings = sparse_vsplit(all_encodings, list_of_indices)
        return list_of_encodings
    
    def set_kernel(self, src_encodings, dst_encodings):
        mtx = pairwise_kernels(src_encodings, dst_encodings, metric=self.metric)
        score = np.mean(mtx)
        return score

    def train_gram_matrix(self):
        gram_mtx = np.array(self.gram_matrix(self.list_of_encodings))
        return gram_mtx
    
    def test_gram_matrix(self, graphs):
        test_list_of_encodings = self.transform(graphs)
        gram_mtx = np.array(self.gram_matrix(test_list_of_encodings))
        return gram_mtx
    
    def gram_row(self, test_encoding):
        return [self.set_kernel(test_encoding, train_encoding) for train_encoding in self.list_of_encodings]
    
    def gram_matrix_sequential(self, test_list_of_encodings):
        return [self.gram_row(test_encoding) for test_encoding in test_list_of_encodings]

    def gram_matrix_parallel(self, test_list_of_encodings):
        n_cpus = mp.cpu_count()
        batch_size = len(test_list_of_encodings)//n_cpus
        if batch_size < 2:
            test_list_of_encodings_list = [test_list_of_encodings]
        else:    
            test_list_of_encodings_list = list(partition_all(batch_size, test_list_of_encodings))
        pool = mp.Pool(n_cpus)
        results = pool.map(self.gram_matrix_sequential, test_list_of_encodings_list)
        pool.close()
        all_list_of_mtx = []
        for list_of_mtx in results:
            all_list_of_mtx.extend(list_of_mtx)
        return all_list_of_mtx

    def gram_matrix(self, test_list_of_encodings):
        if self.parallel: return self.gram_matrix_parallel(test_list_of_encodings)
        else: return self.gram_matrix_sequential(test_list_of_encodings)
        
    
