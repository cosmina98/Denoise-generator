import random
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from coco_grape.graph_vectorizer.graph_vectorizer import GraphVectorizer
from coco_grape.data_processor.generative.neighborhood_generator import GraphNeighborhoodGenerator, NeighborhoodBinaryDecomposition, NeighborhoodDecomposition
from coco_grape.module.vectorize import vectorize as decomposition_vectorize
from coco_grape.module.decompositions.filter_by import filter_by_feature_id
from coco_grape.data_processor.generative.feasibility_estimator import *        
from coco_grape.module import *
import multiprocessing_on_dill as mp
import networkx as nx
from coco_grape.data_processor.generative.feasibility_estimator import FeasibilityEstimatorNumberOfNodesInRange, FeasibilityEstimator
from coco_grape.graph_vectorizer.paired_neighborhood_graph_vectorizer import PairedNeighborhoodGraphVectorizer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier 
from coco_grape.data_processor.processor import DataEstimator

default_decomposition_function_list = [neighborhood(size=2), tree(), cycle(), neighborhood(min_size=1, max_size=3)]

def merge_two_lists(A,B):
    if len(B) < len(A):
        AB = sum([[a,b] for i,(a,b) in enumerate(zip(A[-len(B):],B))],[])
        AB = A[:len(B)]+AB
    elif len(B) == len(A):
        AB = sum([[a,b] for i,(a,b) in enumerate(zip(A,B))],[])
    else: #len(A)<len(B)
        AB = sum([[a,b] for i,(a,b) in enumerate(zip(A,B))],[])
        AB = AB+B[len(A):]
    return AB

def filter_by_size(graphs, graph_src, graph_dest, size_factor=0.025):
    sizes = [nx.number_of_nodes(graph_src),nx.number_of_nodes(graph_dest)]
    feasibility_estimators = [FeasibilityEstimatorNumberOfNodesInRange(min_size=int(min(sizes)*(1-size_factor)), max_size=int(max(sizes)*(1+size_factor)))]
    feasibility_estimator = FeasibilityEstimator(feasibility_estimators=feasibility_estimators)
    feasibility_estimator.set_parallel(False)
    sel_graphs = feasibility_estimator.filter(graphs)
    return sel_graphs

def sample_idx_of_most_similar_instance(idx, similarity_mtx, power_exponent_to_boost_similarity=1):
    similiarity_vector = np.absolute(similarity_mtx[idx])
    similiarity_vector[similiarity_vector==1] = 0 #eliminate the possibility of sampling identical elements
    aidxs = np.argsort(-similiarity_vector)
    similiarity_vector = np.power(similiarity_vector[aidxs], power_exponent_to_boost_similarity)
    p = similiarity_vector/np.sum(similiarity_vector)
    random_idx = np.random.choice(aidxs, size=1, p=p)[0]
    return random_idx

def sample_idx_of_most_similar_instance_with_different_target(idx, similarity_mtx, targets, power_exponent_to_boost_similarity=1):
    target_idx = targets[idx]
    similiarity_vector = np.absolute(similarity_mtx[idx])
    similiarity_vector[similiarity_vector==1] = 0 #eliminate the possibility of sampling identical elements
    aidxs = np.argsort(-similiarity_vector)
    most_similar_different_target_idxs = np.array([aidx for aidx in aidxs if targets[aidx] != target_idx])
    similiarity_vector = np.power(similiarity_vector[most_similar_different_target_idxs], power_exponent_to_boost_similarity)
    p = similiarity_vector/np.sum(similiarity_vector)
    random_mostly_similar_different_target_idx = np.random.choice(most_similar_different_target_idxs, size=1, p=p)[0]
    return random_mostly_similar_different_target_idx

def make_difference_decomposition_functions(graph_src, graph_dest, decomposition_function, nbits):
    X = decomposition_vectorize([graph_src, graph_dest], decomposition_function=decomposition_function, nbits=nbits).astype(bool).todense().A
    features_in_src_but_not_in_dest_mask = np.logical_and(X[0],np.logical_not(X[1]))
    features_in_dest_but_not_in_src_mask = np.logical_and(X[1],np.logical_not(X[0]))
    features_in_src_but_not_in_dest_decomposition_function = compose(filter_by_feature_id(feature_mask=features_in_src_but_not_in_dest_mask), decomposition_function)
    features_in_dest_but_not_in_src_decomposition_function = compose(filter_by_feature_id(feature_mask=features_in_dest_but_not_in_src_mask), decomposition_function)
    return features_in_src_but_not_in_dest_decomposition_function, features_in_dest_but_not_in_src_decomposition_function

def interpolation_generation(graph_src, graph_dest, decomposition_function, nbits, feasibility_estimator, n_elements_per_random_perturbation=30, n_iterations=1):
    features_in_src_but_not_in_dest_decomposition_function, features_in_dest_but_not_in_src_decomposition_function = make_difference_decomposition_functions(graph_src, graph_dest, decomposition_function, nbits)
    generators = [NeighborhoodBinaryDecomposition(
        size=n_elements_per_random_perturbation, 
        decomposition_function_source=features_in_src_but_not_in_dest_decomposition_function, 
        decomposition_function_destination=features_in_dest_but_not_in_src_decomposition_function, 
        nbits=nbits,
        max_num_permutations=2)]
    perturbation_generator = GraphNeighborhoodGenerator(
        generators, 
        feasibility_estimator=feasibility_estimator, 
        max_n_neighborhood_graphs=None, 
        parallel=False).fit([graph_src, graph_dest]) 
    perturbed_graphs = perturbation_generator.iterated_neighbors([graph_src], num_iterations=n_iterations)
    return perturbed_graphs


class InterpolationGraphSampler(object):
    def __init__(self, 
        decomposition_function_list, 
        nbits, 
        feasibility_estimator, 
        estimator, 
        n_elements_per_random_perturbation=30, 
        max_steps_in_interpolation=10, 
        n_output_samples_per_interpolation=6, 
        n_nearest_neighbors_per_interpolation=3, 
        metric='cosine', 
        parallel=True, 
        size_factor=0.025, 
        max_similarity=.8,
        use_improve=True):
        self.decomposition_function_list = decomposition_function_list
        self.nbits = nbits
        self.feasibility_estimator = feasibility_estimator
        self.estimator = estimator
        self.n_elements_per_random_perturbation = n_elements_per_random_perturbation
        self.max_steps_in_interpolation = max_steps_in_interpolation
        self.n_output_samples_per_interpolation = n_output_samples_per_interpolation
        self.n_nearest_neighbors_per_interpolation = n_nearest_neighbors_per_interpolation
        self.metric = metric
        self.parallel = parallel
        self.size_factor = size_factor
        self.max_similarity = max_similarity
        self.use_improve = use_improve
        self.vectorizer = PairedNeighborhoodGraphVectorizer(radius=2, distance=5, nbits=nbits)
        
    def fit(self, graphs, targets=None):
        self.training_graphs = graphs
        self.training_targets = targets
        if self.training_targets is not None: self.estimator.fit(self.training_graphs, self.training_targets)
        self.feasibility_estimator.fit(graphs)
        if self.parallel: self.feasibility_estimator.set_parallel(False) #Note: since Python does not allow daemonic processes to have children we have to disable parallelism here to allow parallelism in sample
        self.training_data_mtx = self.vectorizer.transform(self.training_graphs)
        self.training_similarity_mtx = pairwise_kernels(self.training_data_mtx, metric=self.metric)    
        return self
        
    def interpolate_single(self, graph, graph_src, graph_dest, decomposition_function):
        samples = interpolation_generation(
            graph, graph_dest, 
            decomposition_function=decomposition_function, 
            nbits=self.nbits, 
            feasibility_estimator=self.feasibility_estimator, 
            n_elements_per_random_perturbation=self.n_elements_per_random_perturbation, 
            n_iterations=1)
        samples = filter_by_size(samples, graph_src, graph_dest, size_factor=self.size_factor)
        if len(samples) < 1: return None
        next_graph = random.choice(samples)
        return next_graph
        
    def interpolate_src_to_dest(self, graph_src, graph_dest, n_output_samples_per_interpolation):
        interpolation_list = []
        graph = graph_src
        for it in range(self.max_steps_in_interpolation):
            local_interpolation_list = []
            for decomposition_function in self.decomposition_function_list:
                next_graph = self.interpolate_single(graph, graph_src, graph_dest, decomposition_function)
                if next_graph is None: next_graph = graph
                local_interpolation_list.append([next_graph])
                graph = next_graph
            samples = sum(local_interpolation_list, [])
            if len(samples) < 1: break
            interpolation_list.append(samples)
        samples = sum(interpolation_list, [])
        if n_output_samples_per_interpolation < len(samples):
            idxs = np.unique(np.linspace(0,len(samples)-1, n_output_samples_per_interpolation, dtype=int))
            samples = [samples[idx] for idx in idxs]
        return samples

    def interpolate(self, graph_src, graph_dest):
        interpolation_list_AB = self.interpolate_src_to_dest(graph_src, graph_dest, n_output_samples_per_interpolation=self.n_output_samples_per_interpolation)
        interpolation_list_BA = self.interpolate_src_to_dest(graph_dest, graph_src, n_output_samples_per_interpolation=self.n_output_samples_per_interpolation)
        interpolation_list_BA = interpolation_list_BA[::-1]
        #interpolation_list = interpolation_list_AB + interpolation_list_BA[::-1]
        
        interpolation_list = []
        for graph_src_loc, graph_dest_loc in zip(interpolation_list_AB, interpolation_list_BA):
            graph = graph_src_loc
            for decomposition_function in self.decomposition_function_list:
                next_graph = self.interpolate_single(graph, graph_src_loc, graph_dest_loc, decomposition_function)
                if next_graph is None: next_graph = graph
                graph = next_graph
            if graph is not None:
                interpolation_list.append(graph)
        return interpolation_list
    
    def sample_seed(self, idx_A=None):
        if idx_A is None: idx_A = np.random.choice(len(self.training_graphs))
        if self.training_targets is not None: idx_B = sample_idx_of_most_similar_instance_with_different_target(idx_A, self.training_similarity_mtx, self.training_targets)
        else: idx_B = sample_idx_of_most_similar_instance(idx_A, self.training_similarity_mtx)
        graph_src = self.training_graphs[idx_A]
        graph_dest = self.training_graphs[idx_B]
        return graph_src, graph_dest
            
    def sample_serial(self, n_samples=1):
        np.random.seed()
        samples = []
        for it in range(n_samples):
            graph_src, graph_dest = self.sample_seed()
            interpolation_list_AB = self.interpolate(graph_src, graph_dest)
            interpolation_list_BA = self.interpolate(graph_dest, graph_src)
            samples.extend(merge_two_lists(interpolation_list_AB, interpolation_list_BA[::-1]))
        return samples

    def sample_parallel(self, n_samples):
        n_cpus = mp.cpu_count()
        if n_samples < n_cpus: n_cpus = n_samples
        batch_size = n_samples // n_cpus
        pool = mp.Pool(n_cpus)
        results = pool.map(self.sample_serial, [batch_size]*n_cpus)
        pool.close()
        sampled_graphs = sum(results, [])
        return sampled_graphs

    def sample(self, n_samples):
        if self.parallel: sampled_graphs = self.sample_parallel(n_samples)
        else: sampled_graphs = self.sample_serial(n_samples)
        if self.training_targets is not None: 
            sampled_targets = self.estimator.predict(sampled_graphs)
            return sampled_graphs, sampled_targets
        else:
            return sampled_graphs

    def find_improved_neighbour(self, idx):
        for neigh_idx in self.training_similarity_mtx[idx].tolist():
            neigh_idx = int(neigh_idx)
            if self.training_targets[neigh_idx] > self.training_targets[idx]:
                return neigh_idx
        for neigh_idx in self.training_similarity_mtx[idx].tolist():
            neigh_idx = int(neigh_idx)
            if self.training_targets[neigh_idx] >= self.training_targets[idx]:
                return neigh_idx
        return idx

    def transform_serial(self, idxs):
        samples = []
        for idx_A in idxs:
            dest_idxs = [self.neighbors_idxs_mtx[idx_A, j] for j in range(1, self.n_nearest_neighbors_per_interpolation+1)]
            if self.use_improve: #strategy: for each of the neighbours find their respective neighbour with a higher target
                dest_idxs = [self.find_improved_neighbour(dest_idx) for dest_idx in dest_idxs]
            for idx_B in dest_idxs:
                graph_src, graph_dest = self.test_graphs[idx_A], self.training_graphs[idx_B]
                interpolation_list_AB = self.interpolate(graph_src, graph_dest)
                interpolation_list_BA = self.interpolate(graph_dest, graph_src)
                samples.extend(merge_two_lists(interpolation_list_AB, interpolation_list_BA[::-1]))
        return samples

    def transform_parallel(self, idxs):
        n_cpus = mp.cpu_count()
        if len(idxs) < n_cpus: n_cpus = len(idxs)
        batch_size = len(idxs) // n_cpus
        n = len(idxs)
        idxs_list = [[idxs[i+j] for j in range(batch_size) if i+j < n] for i in range(0, n, batch_size)]
        pool = mp.Pool(n_cpus)
        results = pool.map(self.transform_serial, idxs_list)
        pool.close()
        samples = sum(results, [])
        return samples

    def transform(self, graphs):
        self.test_graphs = graphs
        data_mtx = self.vectorizer.transform(self.test_graphs)
        similarity_mtx = pairwise_kernels(data_mtx, self.training_data_mtx, metric=self.metric)
        similarity_mtx[similarity_mtx >= self.max_similarity] = 0
        self.neighbors_idxs_mtx = np.argsort(-similarity_mtx, axis=1)
        if self.parallel:
            sampled_graphs = self.transform_parallel(idxs=range(len(graphs)))
        else:
            sampled_graphs = self.transform_serial(idxs=range(len(graphs)))
        if self.training_targets is not None: 
            sampled_targets = self.estimator.predict(sampled_graphs)
            return sampled_graphs, sampled_targets
        else:
            return sampled_graphs

def ConcreteInterpolationGraphSampler(
    decomposition_function_list=default_decomposition_function_list, 
    nbits=11, 
    n_elements_per_random_perturbation=30,
    max_steps_in_interpolation=10, 
    n_output_samples_per_interpolation=6,
    n_nearest_neighbors_per_interpolation=3,
    use_improve=False,
    task='classification',
    n_estimators=300):
    if task == 'classification': estimator = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
    elif task == 'regression': estimator = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1)
    else: assert False, 'ERROR: unknown task=%s'%task
    estimator = DataEstimator(data_transformer=PairedNeighborhoodGraphVectorizer(radius=3, distance=5, nbits=12), estimator=estimator)
    feasibility_df = add(
                neighborhood(), 
                cycle(abstraction_level='unlabelled_graph_process'),
                compose(unique(), filter_by_number_of_connected_components(size=1), combination(size=2), cycle(abstraction_level='unlabelled_graph_process'))
                )
    feasibility_estimators = [FeasibilityEstimatorFeatureCannotExist(decomposition_function=feasibility_df, nbits=19)]
    feasibility_estimator = FeasibilityEstimator(feasibility_estimators, parallel=True)
    generator = InterpolationGraphSampler(
        decomposition_function_list=decomposition_function_list, nbits=nbits, 
        feasibility_estimator=feasibility_estimator, 
        estimator=estimator,
        n_elements_per_random_perturbation=n_elements_per_random_perturbation, 
        max_steps_in_interpolation=max_steps_in_interpolation, 
        n_output_samples_per_interpolation=n_output_samples_per_interpolation, 
        n_nearest_neighbors_per_interpolation=n_nearest_neighbors_per_interpolation,
        metric='cosine',
        size_factor=0.025,
        use_improve=use_improve)
    return generator