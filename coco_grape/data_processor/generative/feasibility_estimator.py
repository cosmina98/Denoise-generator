#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
import networkx as nx
from toolz import partition_all
import multiprocessing_on_dill as mp
from coco_grape.module.quotientgraph.vectorize import QuotientGraphTransformer
from coco_grape.module.vectorize import vectorize as legacy_vectorize


def label_attribute_is_present(graph):
    for u in graph.nodes():
        if 'label' not in graph.nodes[u]:
            return False
    for u,v in graph.edges():
        if 'label' not in graph.edges[u,v]:
            return False
    return True

def filter_graphs_without_node_and_edge_label_attribute(graphs):
    return [graph for graph in graphs if label_attribute_is_present(graph)]

class FeasibilityEstimatorFromBooleanFunction(object):
    def __init__(self, boolean_function):
        self.boolean_function = boolean_function

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def fit(self, graphs):
        return self

    def number_of_violations(self, graphs):
        preds = np.array([self.boolean_function(graph) for graph in graphs]).astype(int)
        return preds

    def predict(self, graphs):
        preds = np.array([self.boolean_function(graph) for graph in graphs])
        return preds


def FeasibilityEstimatorHasNodeAndEdgeLabelAttribute():
    return FeasibilityEstimatorFromBooleanFunction(boolean_function=label_attribute_is_present)


def FeasibilityEstimatorHasNoSelfLoops():
    return FeasibilityEstimatorFromBooleanFunction(boolean_function=lambda graph:nx.number_of_selfloops(graph)==0)


def FeasibilityEstimatorNumberOfNodesInRange(min_size=1, max_size=None):
    def is_n_nodes_in_range(graph):
        number_of_nodes = nx.number_of_nodes(graph)
        if max_size is None: return number_of_nodes >= min_size
        return  min_size <= number_of_nodes <= max_size

    return FeasibilityEstimatorFromBooleanFunction(boolean_function=is_n_nodes_in_range)


def FeasibilityEstimatorNumberOfEdgesInRange(min_size=1, max_size=None):
    def is_n_edges_in_range(graph):
        number_of_edges = nx.number_of_edges(graph)
        if max_size is None: return number_of_edges >= min_size
        return  min_size <= number_of_edges <= max_size

    return FeasibilityEstimatorFromBooleanFunction(boolean_function=is_n_edges_in_range)

class WithinRangeFeasibilityEstimatorFromNumericalFunction(object):
    def __init__(self, numerical_function, quantile=None):
        self.numerical_function = numerical_function
        self.quantile = quantile

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def fit(self, graphs):
        vals = np.array([self.numerical_function(graph) for graph in graphs])
        if self.quantile is None:
            self.min = np.min(vals)
            self.max = np.max(vals)
        else:
            self.min = np.quantile(vals, self.quantile)
            self.max = np.quantile(vals, 1-self.quantile)
        return self

    def is_feasible(self, graph):
        val = self.numerical_function(graph)
        test = val >= self.min and val <= self.max 
        return test

    def size_of_violation(self, graph):
        val = self.numerical_function(graph)
        if val <= self.min: 
            return self.min - val 
        elif val >= self.max: 
            return val - self.max
        else:
            return 0

    def number_of_violations(self, graphs):
        preds = np.array([self.size_of_violation(graph) for graph in graphs]).astype(int)
        return preds

    def predict(self, graphs):
        preds = np.array([self.is_feasible(graph) for graph in graphs])
        return preds


class FeasibilityEstimatorIsConnected(object):
    def __init__(self, number_connected_components=1):
        self.number_connected_components = number_connected_components

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def fit(self, graphs):
        return self

    def number_of_violations(self, graphs):
        preds = np.array([nx.number_connected_components(graph)-1 for graph in graphs])
        return preds

    def predict(self, graphs):
        preds = np.array([nx.number_connected_components(graph)==self.number_connected_components for graph in graphs])
        return preds


class FeasibilityEstimatorFeatureMustExist(object):
    def __init__(self, decomposition_function=None, nbits=14, parallel=True):
        self.decomposition_function = decomposition_function
        self.nbits = nbits
        self.must_exist_features_vec = None
        self.parallel = parallel

    def transform(self, graphs):
        """
        Convert a list of NetworkX graphs to a CSR feature matrix using the
        new QuotientGraph pipeline.  Remains sparse so downstream .dot() and
        .A calls behave exactly like before.
        """
        # Honour the old parallel flag
        n_jobs = -1 if self.parallel else 1

        # Default to identity decomposition if none provided
        decomp = self.decomposition_function or (lambda qg: qg)

        try:
            transformer = QuotientGraphTransformer(
                nbits=self.nbits,
                decomposition_function=decomp,
                return_dense=False,   # keep CSR (original code expects it)
                n_jobs=n_jobs,
            )
            return transformer.transform(graphs)
        except AttributeError:
            # Compatibility fallback:
            # Some decomposition functions still target the legacy graph-of-subgraphs API
            # and fail on the newer QuotientGraph type (e.g. missing `.nodes`).
            # In that case, use the legacy vectorization pipeline.
            return legacy_vectorize(
                graphs,
                decomposition_function=decomp,
                nbits=self.nbits,
                dense=False,
            )

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def fit(self, graphs):
        data_mtx = self.transform(graphs).astype(bool)
        n_instances, n_features = data_mtx.shape
        # find all features that are always present
        exist_feats = data_mtx.sum(axis=0).A.flatten().astype(bool)
        self.must_exist_features_vec = np.logical_or(np.zeros(n_features).astype(bool), (exist_feats == n_instances))
        return self

    def number_of_violations(self, graphs):
        assert self.must_exist_features_vec is not None, 'FeasibilityEstimatorFeatureMustExist is not fit'
        data_mtx = self.transform(graphs)
        preds = data_mtx.dot(self.must_exist_features_vec.astype(int))
        return preds

    def predict(self, graphs):
        assert self.must_exist_features_vec is not None, 'FeasibilityEstimatorFeatureMustExist is not fit'
        data_mtx = self.transform(graphs).astype(bool)
        preds = data_mtx.dot(self.must_exist_features_vec)
        return preds


class FeasibilityEstimatorFeatureCannotExist(object):
    def __init__(self, decomposition_function=None, nbits=14, parallel=True):
        self.decomposition_function = decomposition_function
        self.nbits = nbits
        self.cannot_exist_features_vec = None
        self.parallel = parallel

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def transform(self, graphs):
        """
        Convert a list of NetworkX graphs to a CSR feature matrix using the
        new QuotientGraph pipeline.  Remains sparse so downstream .dot() and
        .A calls behave exactly like before.
        """
        # Honour the old parallel flag
        n_jobs = -1 if self.parallel else 1

        # Default to identity decomposition if none provided
        decomp = self.decomposition_function or (lambda qg: qg)

        transformer = QuotientGraphTransformer(
            nbits=self.nbits,
            decomposition_function=decomp,
            return_dense=False,   # keep CSR (original code expects it)
            n_jobs=n_jobs,
        )
        return transformer.transform(graphs)

    def fit(self, graphs):
        data_mtx = self.transform(graphs).astype(bool)
        # find all missing features
        exist_feats = data_mtx.astype(bool).sum(axis=0).A.flatten().astype(bool)
        self.cannot_exist_features_vec = np.logical_not(exist_feats)
        return self

    def number_of_violations(self, graphs):
        assert self.cannot_exist_features_vec is not None, 'FeasibilityEstimatorFeatureCannotExist is not fit'
        data_mtx = self.transform(graphs)
        preds = data_mtx.dot(self.cannot_exist_features_vec.astype(int))
        return preds

    def predict(self, graphs):
        assert self.cannot_exist_features_vec is not None, 'FeasibilityEstimatorFeatureCannotExist is not fit'
        data_mtx = self.transform(graphs).astype(bool)
        cannot_exist = data_mtx.dot(self.cannot_exist_features_vec)
        preds = np.logical_not(cannot_exist)
        return preds


class FeasibilityEstimator(object):
    
    def __init__(self, feasibility_estimators, parallel=True):
        self.feasibility_estimators = feasibility_estimators
        self.set_parallel(parallel)        

    def set_parallel(self, parallel):
        feasibility_estimators = []
        for feasibility_estimator in self.feasibility_estimators:
            feasibility_estimator.parallel = parallel
            feasibility_estimators.append(feasibility_estimator)
        self.feasibility_estimators = feasibility_estimators

    def __repr__(self):
        infos = ['feasibility_estimator_%d:%s'%(i,feasibility_estimator) for i,feasibility_estimator in enumerate(self.feasibility_estimators)]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def fit(self, graphs):
        graphs = filter_graphs_without_node_and_edge_label_attribute(graphs)
        self.feasibility_estimators = [feasibility_estimator.fit(graphs) for feasibility_estimator in self.feasibility_estimators]
        return self

    def predict(self, graphs):
        preds = [feasibility_estimator.predict(graphs).reshape(-1,1) for feasibility_estimator in self.feasibility_estimators]
        preds = np.hstack(preds)
        preds = np.all(preds, axis=1)
        return preds

    def number_of_violations(self, graphs):
        preds = [feasibility_estimator.number_of_violations(graphs).reshape(-1,1) for feasibility_estimator in self.feasibility_estimators]
        preds = np.hstack(preds)
        preds = np.sum(preds, axis=1)
        return preds

    def filter(self, graphs, targets=None):
        graphs = filter_graphs_without_node_and_edge_label_attribute(graphs)
        is_feasible = self.predict(graphs)
        selected_graphs = [graphs[idx] for idx in range(len(graphs)) if is_feasible[idx]==True]
        if targets is not None: 
            selected_targets = [targets[idx] for idx in range(len(targets)) if is_feasible[idx]==True]
            return selected_graphs, selected_targets
        return selected_graphs


def ConcreteFeasibilityEstimator(min_size=1, max_size=None):
    feasibility_estimators = [FeasibilityEstimatorHasNodeAndEdgeLabelAttribute(),
                              FeasibilityEstimatorHasNoSelfLoops(),
                              FeasibilityEstimatorNumberOfNodesInRange(min_size=min_size, max_size=max_size),
                              FeasibilityEstimatorNumberOfEdgesInRange(min_size=1, max_size=None),
                              FeasibilityEstimatorIsConnected(number_connected_components=1)]
    return FeasibilityEstimator(feasibility_estimators=feasibility_estimators)



class FeasibilityRepairEstimator(object):
    def __init__(self, perturbation_generator, feasibility_estimator, n_iter=1, beam_size=1, min_number_of_violations=0, parallel=True):
        self.perturbation_generator = perturbation_generator
        self.feasibility_estimator = feasibility_estimator
        self.n_iter = n_iter
        self.beam_size = beam_size
        self.min_number_of_violations = min_number_of_violations
        self.parallel = parallel
        
    def fit(self, graphs):
        self.perturbation_generator.fit(graphs)
        self.feasibility_estimator.fit(graphs)
        return self
    
    def select(self, graphs, n_samples=1):
        if len(graphs) == 1 and n_samples == 1: return graphs
        number_of_violations = self.feasibility_estimator.number_of_violations(graphs)
        idxs = np.argsort(number_of_violations)
        idxs = idxs[:n_samples]
        selected_graphs = [graphs[idx] for idx in idxs]
        return selected_graphs

    def select_one(self, graphs):
        selected_graph = self.select(graphs, n_samples=1)[0]
        return selected_graph

    def repair_single(self, graph):
        selected_graphs = [graph]
        for it in range(self.n_iter):
            perturbed_graphs = self.perturbation_generator.neighbors(selected_graphs)
            perturbed_graphs = perturbed_graphs + selected_graphs
            selected_graphs = self.select(perturbed_graphs, n_samples=int(self.beam_size))
            number_of_violations = self.feasibility_estimator.number_of_violations(selected_graphs)
            if np.min(number_of_violations) <= self.min_number_of_violations: break
        selected_graph = self.select(selected_graphs, n_samples=1)[0]
        selected_graph.graph['repair_n_iter'] = it + 1
        return selected_graph

    def repair_parallel(self, graphs):
        n_cpus = mp.cpu_count()
        batch_size = len(graphs) // n_cpus
        if len(graphs) <= n_cpus:
            graphs_list = [[graph] for graph in graphs]
        else:
            graphs_list = list(partition_all(batch_size, graphs))
        pool = mp.Pool(n_cpus)
        results = pool.map(self.repair_sequential, graphs_list)
        pool.close()
        all_results = []
        for list_of_results in results:
            all_results.extend(list_of_results)
        return all_results

    def repair_sequential(self, graphs):
        selected_graphs = [self.repair_single(graph) for graph in graphs]
        return selected_graphs

    def repair(self, graphs):
        if self.parallel: selected_graphs = self.repair_parallel(graphs)
        else: selected_graphs = self.repair_sequential(graphs)
        return selected_graphs
        
