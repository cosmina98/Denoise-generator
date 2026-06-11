from coco_grape.data_processor.generative.feasibility_estimator import FeasibilityEstimator, FeasibilityEstimatorFromBooleanFunction, FeasibilityEstimatorIsConnected
from coco_grape.data_processor.generative.feasibility_estimator import FeasibilityEstimatorFeatureMustExist, FeasibilityEstimatorFeatureCannotExist
from coco_grape.data_processor.generative.neighborhood_generator import GraphNeighborhoodGenerator
from coco_grape.data_processor.generative.neighborhood_generator import NeighborhoodEdgeSwap, NeighborhoodEdgeMove, NeighborhoodDecomposition, NeighborhoodBinaryDecomposition
from coco_grape.data_processor.processor import DataEstimator
from coco_grape.data_processor.processor import DataTransformer
from coco_grape.data_processor.unsupervised.spectral_equal_size_clustering import EqualSizeSpectralClustering
from coco_grape.graph_vectorizer.paired_neighborhood_graph_vectorizer import PairedNeighborhoodGraphVectorizer
from coco_grape.graph_vectorizer.paired_neighborhood_graph_vectorizer import PairedNeighborhoodGraphVectorizer
from coco_grape.module.composition import add, compose, binary_combine
from coco_grape.module.decompositions.binary_set_operations import binary_intersection
from coco_grape.module.decompositions.combination import combination
from coco_grape.module.decompositions.complement import complement
from coco_grape.module.decompositions.connected_component import connected_component
from coco_grape.module.decompositions.cycle import cycle, tree, cycle_tree
from coco_grape.module.decompositions.expand import expand
from coco_grape.module.decompositions.filter_by import filter_by_feature_importance, filter_by_node_size, filter_by_number_of_connected_components
from coco_grape.module.decompositions.merge import merge
from coco_grape.module.decompositions.neighborhood import neighborhood
from coco_grape.module.decompositions.path import path
from coco_grape.module.decompositions.unique import unique
from coco_grape.vector_embedder.vector_embedder import VectorEmbedder, SparseToDenseTransformer
from collections import defaultdict
from sklearn.svm import OneClassSVM
import copy
import multiprocessing_on_dill as mp
import networkx as nx
import numpy as np



def no_self_loops(graph): return nx.number_of_selfloops(graph)==0

def no_disconnected_components(graph): return nx.number_connected_components(graph)==1


class GraphNeighborhoodSampler(object):
    def __init__(self, graph_vectorizer=None, feasibility_estimator=None, generators=None, num_iterations=1, neighborhood_size=2, single_perturbation_max_size=20, max_n_neighborhood_graphs=50, fraction_outliers=0.5, parallel=True):
        self.parallel = parallel
        self.num_iterations = num_iterations
        if generators is None:
            generators = [NeighborhoodDecomposition(decomposition_function=neighborhood(size=neighborhood_size), nbits=16, size=single_perturbation_max_size, max_num_permutations=2)]
        self.graph_neighborhood_generator = GraphNeighborhoodGenerator(generators, feasibility_estimator=feasibility_estimator, max_n_neighborhood_graphs=max_n_neighborhood_graphs, parallel=False)
        estimator = OneClassSVM(gamma='auto', nu=fraction_outliers)
        self.data_estimator = DataEstimator(data_transformer=graph_vectorizer, estimator=estimator)
        
    def fit(self, graphs):
        self.graph_neighborhood_generator.feasibility_estimator.fit(graphs)
        self.graph_neighborhood_generator.fit(graphs)
        self.data_estimator.fit(graphs)
        self.graphs = graphs
        return self
    
    def sample_sequential(self, n_samples=1):
        idxs = np.random.choice(len(self.graphs), size=n_samples, replace=True)
        seed_graphs = [self.graphs[idx] for idx in idxs]
        sample_graphs = self.graph_neighborhood_generator.iterated_neighbors(seed_graphs, num_iterations=self.num_iterations)
        scores = self.data_estimator.predict(sample_graphs)
        idxs = np.argsort(-scores)
        idxs = idxs[:n_samples]
        selected_graphs = [sample_graphs[idx] for idx in idxs]
        return selected_graphs

    def sample_parallel(self, n_samples):
        n_cpus = mp.cpu_count()
        if n_samples < n_cpus: n_cpus = n_samples
        batch_size = max(1, n_samples // n_cpus)
        batch_sizes = [batch_size]*n_cpus
        pool = mp.Pool(n_cpus)
        results = pool.map(self.sample_sequential, batch_sizes)
        pool.close()
        sampled_graphs = sum(results, [])
        return sampled_graphs

    def sample(self, n_samples):
        if self.parallel: return self.sample_parallel(n_samples)
        else: return self.sample_sequential(n_samples)


def GraphClusteringEstimator(graph_vectorizer=None, clustering=None):
    return DataEstimator(data_transformer=graph_vectorizer, estimator=clustering)
    
class GraphPartitionEstimator(object):
    def __init__(self, graph_clustering=None):
        self.graph_clustering = graph_clustering

    def partition(self, graphs): 
        if self.graph_clustering is None:
            cluster_idxs = [0]*len(graphs)
        else:
            cluster_idxs = self.graph_clustering.predict(graphs)
        partitioned_graphs_dict = defaultdict(list)
        for graph, cluster_idx in zip(graphs, cluster_idxs):
            partitioned_graphs_dict[cluster_idx].append(graph)
        return partitioned_graphs_dict
        
    def fit(self, graphs, targets=None):
        if self.graph_clustering is not None:
            self.graph_clustering.fit(graphs)
        return self
    

class TargetGraphPartitionEstimator(object):
    def __init__(self, graph_partition_estimator=None):
        self.graph_partition_estimator = graph_partition_estimator
        self.graph_partition_estimators_dict = None

    def target_partition(self, graphs, targets): 
        target_partitioned_graphs_dict = defaultdict(list)
        for graph, target in zip(graphs, targets):
            target_partitioned_graphs_dict[target].append(graph)
        return target_partitioned_graphs_dict
        
    def fit(self, graphs, targets):
        target_partitioned_graphs_dict = self.target_partition(graphs, targets)
        self.graph_partition_estimators_dict = {target: copy.deepcopy(self.graph_partition_estimator) for target in target_partitioned_graphs_dict}
        for target in target_partitioned_graphs_dict:
            self.graph_partition_estimators_dict[target].fit(target_partitioned_graphs_dict[target])
        return self
    
    def partition(self, graphs, targets): 
        target_partitioned_graphs_dict = self.target_partition(graphs, targets)
        target_partitioned_graph_partitioned_dict =  {target: self.graph_partition_estimators_dict[target].partition(target_partitioned_graphs_dict[target]) for target in target_partitioned_graphs_dict}
        return target_partitioned_graph_partitioned_dict
    
class TargetGraphPartitionGraphSampler(object):
    def __init__(self, target_graph_partition_estimator=None, graph_sampler=None):
        self.target_graph_partition_estimator = target_graph_partition_estimator
        self.graph_sampler = graph_sampler
        
    def fit(self, graphs, targets):
        self.target_graph_partition_estimator.fit(graphs, targets)
        target_partitioned_graph_partitioned_dict = self.target_graph_partition_estimator.partition(graphs, targets)
        self.graph_samplers_dict = dict()
        for target in target_partitioned_graph_partitioned_dict:
            for part_id in target_partitioned_graph_partitioned_dict[target]:
                part_graphs = target_partitioned_graph_partitioned_dict[target][part_id]
                self.graph_samplers_dict[(target,part_id)] = copy.deepcopy(self.graph_sampler).fit(part_graphs)
        return self
    
    def sample(self, n_samples=1):
        all_sample_graphs = []
        all_targets = []
        for (target,part_id) in self.graph_samplers_dict:
            sample_graphs = self.graph_samplers_dict[(target,part_id)].sample(n_samples)
            all_sample_graphs.extend(sample_graphs)
            sample_targets = [target]*len(sample_graphs)
            all_targets.extend(sample_targets)
        return all_sample_graphs, all_targets
        

def ConcreteGraphPartitionEstimator(n_clusters):
    if n_clusters <= 1: return GraphPartitionEstimator(graph_clustering=None)
    graph_vectorizer = PairedNeighborhoodGraphVectorizer(radius=2,distance=4, nbits=12)
    vector_embedder = VectorEmbedder(transformers=[SparseToDenseTransformer()])
    data_transformer = DataTransformer(data_vectorizer=graph_vectorizer, vector_embedder=vector_embedder)

    clustering = EqualSizeSpectralClustering(n_clusters=n_clusters)
    graph_clustering = GraphClusteringEstimator(graph_vectorizer=data_transformer, clustering=clustering)
    return GraphPartitionEstimator(graph_clustering)


def ConcreteTargetGraphPartitionGraphNeighborhoodSampler(generators=None, n_clusters=2, num_iterations=1, neighborhood_size=2, single_perturbation_max_size=20, max_n_neighborhood_graphs=50, fraction_outliers=0.5, parallel=True):
    nbits = 12
    radius = 2
    distance = 4        
    
    connected_cycles_df = compose(unique(), filter_by_number_of_connected_components(size=1), combination(size=2), cycle(abstraction_level='unlabelled_graph_process'))
    feasibility_df = add(neighborhood(), cycle(), connected_cycles_df)
    feasibility_estimators = [
        #FeasibilityEstimatorFromBooleanFunction(boolean_function=no_self_loops),
        #FeasibilityEstimatorFromBooleanFunction(boolean_function=no_disconnected_components),
        FeasibilityEstimatorFeatureCannotExist(decomposition_function=feasibility_df, nbits=nbits)]
    feasibility_estimator = FeasibilityEstimator(feasibility_estimators, parallel=False)

    graph_vectorizer = PairedNeighborhoodGraphVectorizer(radius=radius,distance=distance, nbits=nbits, parallel=False)
        
    graph_sampler = GraphNeighborhoodSampler(
        graph_vectorizer=graph_vectorizer, 
        feasibility_estimator=feasibility_estimator,
        generators=generators, 
        num_iterations=num_iterations, 
        neighborhood_size=neighborhood_size, 
        single_perturbation_max_size=single_perturbation_max_size, 
        max_n_neighborhood_graphs=max_n_neighborhood_graphs, 
        fraction_outliers=fraction_outliers, 
        parallel=parallel)
    graph_partition_estimator = ConcreteGraphPartitionEstimator(n_clusters)
    target_graph_partition_estimator = TargetGraphPartitionEstimator(graph_partition_estimator=graph_partition_estimator)
    return TargetGraphPartitionGraphSampler(target_graph_partition_estimator=target_graph_partition_estimator, graph_sampler=graph_sampler)

