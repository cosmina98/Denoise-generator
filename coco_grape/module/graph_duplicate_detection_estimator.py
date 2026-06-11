
from toolz import partition_all
import multiprocessing_on_dill as mp
import time
import random
import numpy as np
import networkx as nx
import random
from collections import defaultdict
from itertools import permutations

from coco_grape.module.graph_hash import graph_hash

class GraphDuplicateDetectionEstimator(object):
    def __init__(self, graph_hash=graph_hash, parallel=True):
        self.parallel = parallel
        self.graph_hash = graph_hash

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def fit(self, graphs):
        self.graphs_hash_dict = self.compute_graphs_hash(graphs)
        return self

    def compute_graphs_hash(self, graphs):     
        if self.parallel:
            graphs_hash_dict = self.parallel_compute_graphs_hash(graphs, self.graph_hash)
        else:
            graphs_hash_dict = self.sequential_compute_graphs_hash(graphs, self.graph_hash)
        return graphs_hash_dict

    def sequential_compute_graphs_hash(self, graphs, graph_hash): 
        graphs_hash_dict = {graph_hash(graph):graph for graph in graphs}
        return graphs_hash_dict

    def parallel_compute_graphs_hash(self, graphs, graph_hash): 
        def func(graph):
            return (graph_hash(graph), graph)
        n_cpus = mp.cpu_count()
        pool = mp.Pool(n_cpus)
        results = pool.map(func, graphs)
        pool.close()
        graphs_hash_dict = dict()
        graphs_hash_dict.update(results)
        return graphs_hash_dict

    def filter(self, graphs):
        graphs_hash_dict = self.compute_graphs_hash(graphs)
        unique_graphs = [graphs_hash_dict[hash_key] for hash_key in graphs_hash_dict if hash_key not in self.graphs_hash_dict]
        return unique_graphs

    def fit_filter(self, graphs):
        self.graphs_hash_dict = self.compute_graphs_hash(graphs)
        unique_graphs = list(self.graphs_hash_dict.values())
        return unique_graphs

def many_lists2one_list_of_tuples(*lists): return list(zip(*lists))

def one_list_of_tuples2many_lists(list_of_tuples): return [list(tuple_of_items) for tuple_of_items in zip(*list_of_tuples)]

class GraphAndAuxiliaryInformationDuplicateDetectionEstimator(object):
    def __init__(self, graph_hash=graph_hash, parallel=True):
        self.parallel = parallel
        self.graph_hash = graph_hash

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def fit(self, graphs, auxiliary_infos):
        data = many_lists2one_list_of_tuples(graphs, auxiliary_infos)
        self.graphs_hash_dict = self.compute_graphs_hash(data)
        return self

    def compute_graphs_hash(self, data):     
        if self.parallel:
            graphs_hash_dict = self.parallel_compute_graphs_hash(data)
        else:
            graphs_hash_dict = self.sequential_compute_graphs_hash(data)
        return graphs_hash_dict

    def sequential_compute_graphs_hash(self, data): 
        graphs, auxiliary_infos = one_list_of_tuples2many_lists(data)
        graphs_hash_dict = {self.graph_hash(graph):(graph, auxiliary_info) for graph, auxiliary_info in zip(graphs, auxiliary_infos)}
        return graphs_hash_dict

    def parallel_compute_graphs_hash(self, data, graph_hash): 
        n_cpus = mp.cpu_count()
        pool = mp.Pool(n_cpus)
        results = pool.map(self.sequential_compute_graphs_hash, data)
        pool.close()
        graphs_hash_dict = dict()
        for result in results: 
            graphs_hash_dict.update(result)
        return graphs_hash_dict

    def filter(self, graphs, auxiliary_infos):
        data = many_lists2one_list_of_tuples(graphs, auxiliary_infos)
        graphs_hash_dict = self.compute_graphs_hash(data)
        unique_data = [graphs_hash_dict[hash_key] for hash_key in graphs_hash_dict if hash_key not in self.graphs_hash_dict]
        unique_graphs, unique_auxiliary_infos = one_list_of_tuples2many_lists(unique_data)
        return unique_graphs, unique_auxiliary_infos

    def fit_filter(self, graphs, auxiliary_infos):
        data = many_lists2one_list_of_tuples(graphs, auxiliary_infos)
        self.graphs_hash_dict = self.compute_graphs_hash(data)
        unique_data = list(self.graphs_hash_dict.values())
        unique_graphs, unique_auxiliary_infos = one_list_of_tuples2many_lists(unique_data)
        return unique_graphs, unique_auxiliary_infos
