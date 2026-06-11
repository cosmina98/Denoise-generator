from coco_grape.data_processor.generative.feasibility_estimator import FeasibilityEstimator, FeasibilityEstimatorFromBooleanFunction, FeasibilityEstimatorIsConnected
from coco_grape.data_processor.generative.feasibility_estimator import FeasibilityEstimatorFeatureMustExist, FeasibilityEstimatorFeatureCannotExist
from coco_grape.data_processor.generative.neighborhood_generator import GraphNeighborhoodGenerator
from coco_grape.data_processor.generative.neighborhood_generator import NeighborhoodEdgeSwap, NeighborhoodEdgeMove, NeighborhoodDecomposition, NeighborhoodBinaryDecomposition
from coco_grape.data_processor.processor import DataEstimator
from coco_grape.data_processor.processor import DataTransformer
from coco_grape.graph_vectorizer.paired_neighborhood_graph_vectorizer import PairedNeighborhoodGraphVectorizer
from coco_grape.module import *
from coco_grape.vector_embedder.vector_embedder import VectorEmbedder, SparseToDenseTransformer
from collections import defaultdict
from sklearn.svm import OneClassSVM
import copy
import multiprocessing_on_dill as mp
import networkx as nx
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

class GraphPerturbationGenerator(object):
    def __init__(self, graph_vectorizer=None, feasibility_estimator=None, generators=None, classifier=None, num_iterations=1, max_n_perturbed_graphs_per_input_graph=50, parallel=True):
        self.parallel = parallel
        self.num_iterations = num_iterations
        self.graph_neighborhood_generator = GraphNeighborhoodGenerator(generators, feasibility_estimator=feasibility_estimator, max_n_neighborhood_graphs=max_n_perturbed_graphs_per_input_graph, parallel=False)
        self.data_estimator = DataEstimator(data_transformer=graph_vectorizer, estimator=classifier)
        
    def fit(self, graphs, targets=None):
        self.graph_neighborhood_generator.feasibility_estimator.fit(graphs)
        self.graph_neighborhood_generator.fit(graphs)
        self.data_estimator.fit(graphs, targets)
        if self.parallel: self.graph_neighborhood_generator.feasibility_estimator.set_parallel(False)
        return self
    
    def transform_sequential(self, graphs):
        graph_list = [self.graph_neighborhood_generator.iterated_neighbors([graph], num_iterations=self.num_iterations) for graph in graphs]
        sample_graphs = sum(graph_list, [])
        return sample_graphs
            
    def transform_parallel(self, graphs):
        n_cpus = mp.cpu_count()
        if len(graphs) < n_cpus: n_cpus = len(graphs)
        batch_size = max(1, len(graphs) // n_cpus)
        batches = [graphs[i:i+batch_size] for i in range(0,len(graphs),batch_size)]
        pool = mp.Pool(n_cpus)
        results = pool.map(self.transform_sequential, batches)
        pool.close()
        sampled_graphs = sum(results, [])
        return sampled_graphs

    def perturb(self, graphs):
        if self.parallel: perturbed_graphs = self.transform_parallel(graphs)
        else: perturbed_graphs = self.transform_sequential(graphs)
        perturbed_targets = self.data_estimator.predict(perturbed_graphs)
        return perturbed_graphs, perturbed_targets
        
        
def ConcreteGraphPerturbationGenerator(generators=None, num_iterations=1, max_n_perturbed_graphs_per_input_graph=50, parallel=True):
    nbits = 12
    radius = 2
    distance = 4        
    
    connected_cycles_df = compose(unique(), filter_by_number_of_connected_components(size=1), combination(size=2), cycle(abstraction_level='unlabelled_graph_process'))
    feasibility_df = add(neighborhood(), cycle(), connected_cycles_df)
    feasibility_estimators = [FeasibilityEstimatorFeatureCannotExist(decomposition_function=feasibility_df, nbits=nbits)]
    feasibility_estimator = FeasibilityEstimator(feasibility_estimators, parallel=parallel)

    graph_vectorizer = PairedNeighborhoodGraphVectorizer(radius=radius,distance=distance, nbits=nbits, parallel=parallel)
    
    classifier = ExtraTreesClassifier(n_estimators=300, n_jobs=-1)
    
    generator = GraphPerturbationGenerator(
        graph_vectorizer=graph_vectorizer, 
        feasibility_estimator=feasibility_estimator, 
        generators=generators, 
        classifier=classifier, 
        num_iterations=num_iterations, 
        max_n_perturbed_graphs_per_input_graph=max_n_perturbed_graphs_per_input_graph, 
        parallel=parallel)
    return generator