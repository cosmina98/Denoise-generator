import random
import numpy as np
import random
import copy
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from coco_grape.data_processor.generative.neighborhood_generator import GraphNeighborhoodGenerator, NeighborhoodBinaryDecomposition
from coco_grape.data_processor.generative.neighborhood_generator import NeighborhoodProbabilisticBinaryDecomposition
from coco_grape.module.vectorize import vectorize, parallel_vectorize
from coco_grape.module.composition import compose
from coco_grape.module.decompositions.filter_by import filter_by_feature_id
from coco_grape.module.graph_duplicate_detection_estimator import GraphDuplicateDetectionEstimator
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.ensemble import ExtraTreesClassifier

import multiprocessing_on_dill as mp
from coco_grape.data_processor.generative.nearest_mutual_neighbours_sampler import NearestMutualNeighboursEstimator, NearestMutualNeighboursProbabilityEstimator
from coco_grape.data_processor.generative.nearest_mutual_neighbours_sampler import nuSVMSupportVectorProbabilityEstimator, ClassificationConfidenceEstimator, ProbabilityEstimator
from coco_grape.graph_vectorizer.graph_vectorizer import GraphVectorizer
from coco_grape.module import *
from coco_grape.data_processor.generative.feasibility_estimator import *
from coco_grape.data_processor.processor import DataEstimator
from coco_grape.data_processor.processor import DataTransformer
from coco_grape.data_processor.unsupervised.spectral_equal_size_clustering import EqualSizeSpectralClustering
from coco_grape.vector_embedder.vector_embedder import VectorEmbedder, SparseToDenseTransformer
from coco_grape.module.graph_duplicate_detection_estimator import GraphDuplicateDetectionEstimator


class NearestMutualNeighboursGraphSampler(object):
    def __init__(self, 
                 vectorizer=None, 
                 attribute_graph_graphicalizer=None, 
                 feasibility_estimator=None, 
                 duplicate_estimator=None,
                 nearest_mutual_neighbours_estimator=None, 
                 probability_estimator=None, 
                 decomposition_function=None,
                 nbits=12,
                 neighborhood_size=None,
                 max_n_neighborhood_graphs=100,
                 num_iterations=1,
                 n_iter_estimate_compatibility_score=100,
                 parallel=True,
                 use_probabilistic=True,
                 verbose=False):
        self.vectorizer = vectorizer
        self.attribute_graph_graphicalizer = attribute_graph_graphicalizer
        self.feasibility_estimator = feasibility_estimator
        self.duplicate_estimator = duplicate_estimator
        self.nearest_mutual_neighbours_estimator = nearest_mutual_neighbours_estimator
        self.probability_estimator = probability_estimator
        self.decomposition_function = decomposition_function
        self.nbits = nbits
        self.neighborhood_size = neighborhood_size 
        self.max_n_neighborhood_graphs = max_n_neighborhood_graphs
        self.num_iterations = num_iterations
        self.n_iter_estimate_compatibility_score = n_iter_estimate_compatibility_score
        self.parallel = parallel
        self.use_probabilistic = use_probabilistic
        self.verbose = verbose
        
    def fit(self, graphs, targets=None):
        if self.feasibility_estimator is not None: self.feasibility_estimator.fit(graphs) 
        if self.attribute_graph_graphicalizer is not None: self.attribute_graph_graphicalizer.fit(graphs)
        X = self.vectorizer.fit_transform(graphs).todense().A
        if targets is None:
            y = np.zeros(len(graphs), dtype=int)
        else:
            y = np.asarray(targets)
            if len(y) != len(graphs):
                raise ValueError("targets must have the same length as graphs")
        self.graphs = copy.deepcopy(graphs)
        self.data_mtx = copy.deepcopy(X)
        self.targets = copy.deepcopy(y)
        self.sampling_probability = self.probability_estimator.fit_predict_proba(X,y)
        self.k_nearest_mutual_neighbours = self.nearest_mutual_neighbours_estimator.fit_predict(X)
        return self
    
    def estimate_compatibility_score(self, idx_A, idx_B, idx_Bp):
        XA,XB,XBp = self.data_mtx[idx_A], self.data_mtx[idx_B], self.data_mtx[idx_Bp]
        features_in_B_but_not_in_Bp_mask = np.logical_and(XB,np.logical_not(XBp))
        features_in_Bp_but_not_in_B_mask = np.logical_and(XBp,np.logical_not(XB))
        features_in_B_but_not_in_Bp_and_in_A_mask = np.logical_and(np.logical_and(XB,np.logical_not(XBp)),XA)
        n_B = np.sum(features_in_B_but_not_in_Bp_mask)
        n_Bp = np.sum(features_in_Bp_but_not_in_B_mask)
        n_A = np.sum(features_in_B_but_not_in_Bp_and_in_A_mask)
        score = n_B * n_Bp * n_A
        return score

    def select_neighbours(self, sampling_probability):
        best_idxs = None
        best_score = 0
        for it in range(self.n_iter_estimate_compatibility_score):
            #select an instance at random
            idx_A = np.random.choice(len(self.k_nearest_mutual_neighbours),size=1, p=sampling_probability)[0]
            if len(self.k_nearest_mutual_neighbours[idx_A])<1: continue
            #select one of its neighbors at random
            idx_B = np.random.choice(self.k_nearest_mutual_neighbours[idx_A])
            if len(self.k_nearest_mutual_neighbours[idx_B])<1: continue
            #select one of the neighbors of idx_B at random
            idx_Bp = np.random.choice(self.k_nearest_mutual_neighbours[idx_B])
            if len(self.k_nearest_mutual_neighbours[idx_Bp])<1: continue            
            score = self.estimate_compatibility_score(idx_A, idx_B, idx_Bp)
            if score > best_score:
                best_score = score
                best_idxs = (idx_A, idx_B, idx_Bp)                
        idx_A, idx_B, idx_Bp = best_idxs
        assert best_score > 0
        return idx_A, idx_B, idx_Bp

    def offset_generation(self, idx_A, idx_B, idx_Bp):
        if self.use_probabilistic: return self.probabilistic_offset_generation(idx_A, idx_B, idx_Bp)
        else: return self.deterministic_offset_generation(idx_A, idx_B, idx_Bp)

    def probabilistic_offset_generation(self, idx_A, idx_B, idx_Bp):

        def make_probs(in_u,not_in_v):
            features_in_B_but_not_in_Bp_probs = in_u - not_in_v
            features_in_B_but_not_in_Bp_probs[features_in_B_but_not_in_Bp_probs<0]=0
            features_in_B_but_not_in_Bp_probs = features_in_B_but_not_in_Bp_probs/np.sum(features_in_B_but_not_in_Bp_probs)
            return features_in_B_but_not_in_Bp_probs
            
        graph_A, graph_B, graph_Bp = self.graphs[idx_A], self.graphs[idx_B], self.graphs[idx_Bp]
        XB,XBp = self.data_mtx[idx_B], self.data_mtx[idx_Bp]

        features_in_B_but_not_in_Bp_probs = make_probs(XB,XBp)
        features_in_Bp_but_not_in_B_probs = make_probs(XBp,XB)
        features_in_B_but_not_in_Bp_probs = 10*features_in_B_but_not_in_Bp_probs+features_in_Bp_but_not_in_B_probs
        features_in_B_but_not_in_Bp_probs = features_in_B_but_not_in_Bp_probs/np.sum(features_in_B_but_not_in_Bp_probs)
        features_in_Bp_but_not_in_B_probs = 10*features_in_Bp_but_not_in_B_probs + features_in_B_but_not_in_Bp_probs
        features_in_Bp_but_not_in_B_probs = features_in_Bp_but_not_in_B_probs/np.sum(features_in_Bp_but_not_in_B_probs)

        generators = [NeighborhoodProbabilisticBinaryDecomposition(
            size=self.neighborhood_size, 
            decomposition_function_source=self.decomposition_function, 
            decomposition_function_destination=self.decomposition_function,
            decomposition_probs_source=features_in_B_but_not_in_Bp_probs, 
            decomposition_probs_destination=features_in_Bp_but_not_in_B_probs,
            nbits=self.nbits,
            max_num_permutations=2)]
        perturbation_generator = GraphNeighborhoodGenerator(
            generators, 
            feasibility_estimator=self.feasibility_estimator, 
            max_n_neighborhood_graphs=self.max_n_neighborhood_graphs, 
            parallel=False).fit([graph_B, graph_Bp]) 
        if self.num_iterations > 1: # consider cumulative perturbations from one step through num_iterations
            perturbed_graphs = []
            for num_iterations in range(1, self.num_iterations + 1):
                perturbed_graphs.extend(perturbation_generator.iterated_neighbors([graph_A], num_iterations=num_iterations))
        else: perturbed_graphs = perturbation_generator.iterated_neighbors([graph_A], num_iterations=self.num_iterations)
        return perturbed_graphs

    def deterministic_offset_generation(self, idx_A, idx_B, idx_Bp):
        graph_A, graph_B, graph_Bp = self.graphs[idx_A], self.graphs[idx_B], self.graphs[idx_Bp]
        XB,XBp = self.data_mtx[idx_B], self.data_mtx[idx_Bp]
        features_in_B_but_not_in_Bp_mask = np.logical_and(XB,np.logical_not(XBp))
        features_in_Bp_but_not_in_B_mask = np.logical_and(XBp,np.logical_not(XB))
        features_in_B_but_not_in_Bp_decomposition_function = compose(filter_by_feature_id(feature_mask=features_in_B_but_not_in_Bp_mask), self.decomposition_function)
        features_in_Bp_but_not_in_B_decomposition_function = compose(filter_by_feature_id(feature_mask=features_in_Bp_but_not_in_B_mask), self.decomposition_function)
        generators = [NeighborhoodBinaryDecomposition(
            size=self.neighborhood_size, 
            decomposition_function_source=features_in_B_but_not_in_Bp_decomposition_function, 
            decomposition_function_destination=features_in_Bp_but_not_in_B_decomposition_function, 
            nbits=self.nbits,
            max_num_permutations=2)]
        perturbation_generator = GraphNeighborhoodGenerator(
            generators, 
            feasibility_estimator=self.feasibility_estimator, 
            max_n_neighborhood_graphs=self.max_n_neighborhood_graphs, 
            parallel=False).fit([graph_B, graph_Bp]) 
        if self.num_iterations > 1: # consider cumulative perturbations from one step through num_iterations
            perturbed_graphs = []
            for num_iterations in range(1, self.num_iterations + 1):
                perturbed_graphs.extend(perturbation_generator.iterated_neighbors([graph_A], num_iterations=num_iterations))
        else: perturbed_graphs = perturbation_generator.iterated_neighbors([graph_A], num_iterations=self.num_iterations)
        return perturbed_graphs

    def sample_single(self, data):
        n_samples, target = data
        if target is not None:
            #sample only instances from desired target class
            sampling_probability = copy.deepcopy(self.sampling_probability)
            sampling_probability[self.targets != target] = 0
            denom = np.sum(sampling_probability)
            if denom <= 0:
                if self.verbose:
                    print(f"No fitted graphs for target={target!r}")
                return []
            sampling_probability = sampling_probability / denom
        else: sampling_probability = self.sampling_probability
        sampled_graphs = []
        for it in range(n_samples):
            try:
                idx_A, idx_B, idx_Bp = self.select_neighbours(sampling_probability)
                samples = self.offset_generation(idx_A, idx_B, idx_Bp)
                sampled_graphs.extend(samples)
            except Exception as e:
                if self.verbose: print(e)
                pass
        if len(sampled_graphs) > n_samples: sampled_graphs = random.choices(sampled_graphs, k=n_samples)
        if self.attribute_graph_graphicalizer is not None: sampled_graphs = self.attribute_graph_graphicalizer.transform(sampled_graphs)
        return sampled_graphs
    
    def sample_parallel(self, n_samples, target=None):
        n_cpus = mp.cpu_count()
        if n_samples < n_cpus: n_cpus = n_samples
        if n_cpus <= 0:
            return []
        base = n_samples // n_cpus
        remainder = n_samples % n_cpus
        jobs = [(base + (1 if i < remainder else 0), target) for i in range(n_cpus)]
        pool = mp.Pool(n_cpus)
        results = pool.map(self.sample_single, jobs)
        pool.close()
        sampled_graphs = sum(results, [])
        return sampled_graphs

    def sample(self, n_samples, target=None):
        if self.parallel: sampled_graphs = self.sample_parallel(n_samples, target)
        else: sampled_graphs = self.sample_single((n_samples, target))
        if self.duplicate_estimator is not None:
            sampled_graphs = self.duplicate_estimator.fit_filter(sampled_graphs)
        return sampled_graphs


def ConcreteNearestMutualNeighboursGraphSampler(n_neighbours=10, metric='euclidean'):
    nearest_mutual_neighbours_estimator = NearestMutualNeighboursEstimator(n_neighbours, metric)
    probability_estimator = NearestMutualNeighboursProbabilityEstimator(n_neighbours, metric)
    decomposition_function = neighborhood(min_size=2, max_size=3)
    nbits = 12
    vectorizer = GraphVectorizer(decomposition_function=decomposition_function, nbits=nbits)
    feasibility_df = add(neighborhood(), cycle(abstraction_level='unlabelled_graph_process'), compose(unique(), filter_by_number_of_connected_components(size=1), combination(size=2), cycle(abstraction_level='unlabelled_graph_process')))
    feasibility_estimators = [FeasibilityEstimatorFeatureCannotExist(decomposition_function=feasibility_df, nbits=19)]
    feasibility_estimator = FeasibilityEstimator(feasibility_estimators, parallel=False)
    sampler = NearestMutualNeighboursGraphSampler(
        vectorizer=vectorizer, 
        attribute_graph_graphicalizer=None, 
        feasibility_estimator=feasibility_estimator, 
        duplicate_estimator=GraphDuplicateDetectionEstimator(),
        nearest_mutual_neighbours_estimator=nearest_mutual_neighbours_estimator, 
        probability_estimator=probability_estimator, 
        decomposition_function=decomposition_function,
        nbits=nbits,
        neighborhood_size=None,
        max_n_neighborhood_graphs=100,
        num_iterations=1)
    return sampler


class ClassConditionalNearestMutualNeighboursGraphSamplingTransformer(object):
    def __init__(self, sampler, resampling_factor=1, use_balanced=True):
        self.sampler = sampler
        self.resampling_factor = resampling_factor 
        self.use_balanced = use_balanced

    def fit(self, graphs,targets):
        self.sampler.fit(graphs,targets)
        self.n_classes = len(set(targets))
        return self
        
    def transform(self, graphs,targets):
        target_list = sorted(set(targets))
        target_counts = np.bincount(targets)
        max_target_counts = np.max(target_counts)
        all_graphs = []
        all_tragets = []
        for target in target_list:
            if self.use_balanced: samples = self.sampler.sample(n_samples=int(max_target_counts*self.resampling_factor), target=target)
            else: samples = self.sampler.sample(n_samples=int(target_counts[target]*self.resampling_factor), target=target)
            all_graphs.extend(samples)
            all_tragets.extend([target]*len(samples))
        all_tragets = np.hstack(all_tragets)
        return all_graphs, all_tragets

    def sample(self, n_samples):
        all_graphs = []
        all_tragets = []
        if isinstance(n_samples, list) is False: 
            n_samples = [n_samples]*self.n_classes

        for target, max_target_counts in enumerate(n_samples):
            samples = self.sampler.sample(n_samples=int(max_target_counts * self.resampling_factor), target=target)
            all_graphs.extend(samples)
            all_tragets.extend([target]*len(samples))
        all_tragets = np.hstack(all_tragets)
        return all_graphs, all_tragets

    def fit_transform(self, X,y):
        return self.fit(X,y).transform(X,y)


def ConcreteClassConditionalNearestMutualNeighboursGraphSamplingTransformer(decomposition_function=neighborhood(min_size=2, max_size=3), nbits=12, n_neighbours=10, metric='euclidean', resampling_factor=1, num_iterations=1, use_balanced=True, parallel=True):
    nearest_mutual_neighbours_estimator = NearestMutualNeighboursEstimator(n_neighbours, metric)
    probability_estimator = NearestMutualNeighboursProbabilityEstimator(n_neighbours, metric)
    decomposition_function = neighborhood(min_size=2, max_size=3)
    nbits = 12
    vectorizer = GraphVectorizer(decomposition_function=decomposition_function, nbits=nbits)
    feasibility_df = add(neighborhood(), cycle(abstraction_level='unlabelled_graph_process'), compose(unique(), filter_by_number_of_connected_components(size=1), combination(size=2), cycle(abstraction_level='unlabelled_graph_process')))
    feasibility_estimators = [FeasibilityEstimatorFeatureCannotExist(decomposition_function=feasibility_df, nbits=19)]
    feasibility_estimator = FeasibilityEstimator(feasibility_estimators, parallel=False)
    sampler = NearestMutualNeighboursGraphSampler(
        vectorizer=vectorizer, 
        attribute_graph_graphicalizer=None, 
        feasibility_estimator=feasibility_estimator, 
        duplicate_estimator=GraphDuplicateDetectionEstimator(),
        nearest_mutual_neighbours_estimator=nearest_mutual_neighbours_estimator, 
        probability_estimator=probability_estimator, 
        decomposition_function=decomposition_function,
        nbits=nbits,
        neighborhood_size=None,
        max_n_neighborhood_graphs=100,
        num_iterations=num_iterations,
        parallel=parallel)
    cc_sampler = ClassConditionalNearestMutualNeighboursGraphSamplingTransformer(sampler, resampling_factor=resampling_factor, use_balanced=use_balanced)
    return cc_sampler


def ConcreteSupportClassConditionalNearestMutualNeighboursGraphSamplingTransformer(decomposition_function=neighborhood(min_size=2, max_size=3), nbits=12, n_neighbours=10, support_instances_fraction=1, resampling_factor=1, use_balanced=True, metric='euclidean', parallel=True):
    nearest_mutual_neighbours_estimator = NearestMutualNeighboursEstimator(n_neighbours, metric)
    probability_estimators=[NearestMutualNeighboursProbabilityEstimator(n_neighbours, metric), 
                           nuSVMSupportVectorProbabilityEstimator(kernel='rbf', gamma='scale', nu_start=.01, nu_end=.99, n_steps=20, support_instances_fraction=support_instances_fraction)]
    probability_estimator = ProbabilityEstimator(probability_estimators)
    
    vectorizer = GraphVectorizer(decomposition_function=decomposition_function, nbits=nbits)
    feasibility_df = add(neighborhood(), cycle(abstraction_level='unlabelled_graph_process'), compose(unique(), filter_by_number_of_connected_components(size=1), combination(size=2), cycle(abstraction_level='unlabelled_graph_process')))
    feasibility_estimators = [FeasibilityEstimatorFeatureCannotExist(decomposition_function=feasibility_df, nbits=19)]
    feasibility_estimator = FeasibilityEstimator(feasibility_estimators, parallel=False)
    sampler = NearestMutualNeighboursGraphSampler(
        vectorizer=vectorizer, 
        attribute_graph_graphicalizer=None, 
        feasibility_estimator=feasibility_estimator, 
        duplicate_estimator=GraphDuplicateDetectionEstimator(),
        nearest_mutual_neighbours_estimator=nearest_mutual_neighbours_estimator, 
        probability_estimator=probability_estimator, 
        decomposition_function=decomposition_function,
        nbits=nbits,
        neighborhood_size=None,
        max_n_neighborhood_graphs=100,
        num_iterations=1,
        parallel=parallel)
    cc_sampler = ClassConditionalNearestMutualNeighboursGraphSamplingTransformer(sampler, resampling_factor=resampling_factor, use_balanced=use_balanced)
    return cc_sampler


class ClassConditionalClusteringNearestMutualNeighboursGraphSamplingTransformer(object):
    def __init__(self, sampler, clustering, resampling_factor=1, use_balanced=True):
        self.sampler = sampler
        self.clustering = clustering
        self.resampling_factor = resampling_factor 
        self.use_balanced = use_balanced

    def fit(self, graphs,targets):
        self.clustering.fit(graphs)
        clusters_ids = self.clustering.predict(graphs)
        #join clusters_ids and targets
        self.target_clusters_id_to_class_map = {paired_target_clusters_id:i for i,paired_target_clusters_id in enumerate(set([(target,clusters_id) for target,clusters_id in zip(targets,clusters_ids)]))}
        self.target_clusters_id_class_to_target_map = {i:paired_target_clusters_id[0] for i,paired_target_clusters_id in enumerate(set([(target,clusters_id) for target,clusters_id in zip(targets,clusters_ids)]))}
        self.targets_clusters_ids = [self.target_clusters_id_to_class_map[(target,clusters_id)] for target,clusters_id in zip(targets,clusters_ids)]
        self.sampler.fit(graphs,self.targets_clusters_ids)
        return self
        
    def transform(self, graphs, targets):
        clusters_ids = self.clustering.predict(graphs)
        targets_clusters_ids = [self.target_clusters_id_to_class_map[(target,clusters_id)] for target,clusters_id in zip(targets,clusters_ids)]
        generated_graphs, generated_tragets = self.generate(targets_clusters_ids)
        return generated_graphs, generated_tragets

    def sample(self, n_samples):
        targets_clusters_ids = np.random.choice(self.targets_clusters_ids, size=n_samples)
        generated_graphs, generated_tragets = self.generate(targets_clusters_ids)
        return generated_graphs, generated_tragets
        
    def generate(self, targets_clusters_ids):
        target_list = sorted(set(targets_clusters_ids))
        target_counts = np.bincount(targets_clusters_ids)
        max_target_counts = np.max(target_counts)
        generated_graphs = []
        generated_tragets = []
        for targets_clusters_id in target_list:
            if self.use_balanced: samples = self.sampler.sample(n_samples=int(max_target_counts*self.resampling_factor), target=targets_clusters_id)
            else: samples = self.sampler.sample(n_samples=int(target_counts[targets_clusters_id]*self.resampling_factor), target=targets_clusters_id)
            generated_graphs.extend(samples)
            generated_tragets.extend([self.target_clusters_id_class_to_target_map[targets_clusters_id]]*len(samples))
        generated_tragets = np.hstack(generated_tragets)
        return generated_graphs, generated_tragets

    def fit_transform(self, X,y):
        return self.fit(X,y).transform(X,y)
    
    
def ConcreteClassConditionalClusteringNearestMutualNeighboursGraphSamplingTransformer(decomposition_function=neighborhood(min_size=2, max_size=3), nbits=12, n_clusters=10, n_neighbours=10, metric='euclidean', resampling_factor=1, use_balanced=True, parallel=True):
    nearest_mutual_neighbours_estimator = NearestMutualNeighboursEstimator(n_neighbours, metric)
    probability_estimator = NearestMutualNeighboursProbabilityEstimator(n_neighbours, metric)
    decomposition_function = neighborhood(min_size=2, max_size=3)
    nbits = 12
    vectorizer = GraphVectorizer(decomposition_function=decomposition_function, nbits=nbits)
    feasibility_df = add(neighborhood(), cycle(abstraction_level='unlabelled_graph_process'), compose(unique(), filter_by_number_of_connected_components(size=1), combination(size=2), cycle(abstraction_level='unlabelled_graph_process')))
    feasibility_estimators = [FeasibilityEstimatorFeatureCannotExist(decomposition_function=feasibility_df, nbits=19)]
    feasibility_estimator = FeasibilityEstimator(feasibility_estimators, parallel=False)
    sampler = NearestMutualNeighboursGraphSampler(
        vectorizer=vectorizer, 
        attribute_graph_graphicalizer=None, 
        feasibility_estimator=feasibility_estimator, 
        duplicate_estimator=GraphDuplicateDetectionEstimator(),
        nearest_mutual_neighbours_estimator=nearest_mutual_neighbours_estimator, 
        probability_estimator=probability_estimator, 
        decomposition_function=decomposition_function,
        nbits=nbits,
        neighborhood_size=None,
        max_n_neighborhood_graphs=100,
        num_iterations=1,
        parallel=parallel)
    vector_embedder = VectorEmbedder(transformers=[SparseToDenseTransformer()])
    data_transformer = DataTransformer(data_vectorizer=vectorizer, vector_embedder=vector_embedder)
    clustering_estimator = EqualSizeSpectralClustering(n_clusters=n_clusters)
    clustering = DataEstimator(data_transformer=data_transformer, estimator=clustering_estimator)
    cc_sampler = ClassConditionalClusteringNearestMutualNeighboursGraphSamplingTransformer(sampler, clustering, resampling_factor=resampling_factor, use_balanced=use_balanced)
    return cc_sampler


def ConcreteSupportClassConditionalClusteringNearestMutualNeighboursGraphSamplingTransformer(decomposition_function=neighborhood(min_size=2, max_size=3), nbits=12, n_clusters=10, n_neighbours=10, support_instances_fraction=1, resampling_factor=1, use_balanced=True, metric='euclidean', parallel=True):
    nearest_mutual_neighbours_estimator = NearestMutualNeighboursEstimator(n_neighbours, metric)
    probability_estimators=[NearestMutualNeighboursProbabilityEstimator(n_neighbours, metric), 
                           nuSVMSupportVectorProbabilityEstimator(kernel='rbf', gamma='scale', nu_start=.01, nu_end=.99, n_steps=20, support_instances_fraction=support_instances_fraction)]
    probability_estimator = ProbabilityEstimator(probability_estimators)
    
    vectorizer = GraphVectorizer(decomposition_function=decomposition_function, nbits=nbits)
    feasibility_df = add(neighborhood(), cycle(abstraction_level='unlabelled_graph_process'), compose(unique(), filter_by_number_of_connected_components(size=1), combination(size=2), cycle(abstraction_level='unlabelled_graph_process')))
    feasibility_estimators = [FeasibilityEstimatorFeatureCannotExist(decomposition_function=feasibility_df, nbits=19)]
    feasibility_estimator = FeasibilityEstimator(feasibility_estimators, parallel=False)
    sampler = NearestMutualNeighboursGraphSampler(
        vectorizer=vectorizer, 
        attribute_graph_graphicalizer=None, 
        feasibility_estimator=feasibility_estimator, 
        duplicate_estimator=GraphDuplicateDetectionEstimator(),
        nearest_mutual_neighbours_estimator=nearest_mutual_neighbours_estimator, 
        probability_estimator=probability_estimator, 
        decomposition_function=decomposition_function,
        nbits=nbits,
        neighborhood_size=None,
        max_n_neighborhood_graphs=100,
        num_iterations=1,
        parallel=parallel)
    vector_embedder = VectorEmbedder(transformers=[SparseToDenseTransformer()])
    data_transformer = DataTransformer(data_vectorizer=vectorizer, vector_embedder=vector_embedder)
    clustering_estimator = EqualSizeSpectralClustering(n_clusters=n_clusters)
    clustering = DataEstimator(data_transformer=data_transformer, estimator=clustering_estimator)
    cc_sampler = ClassConditionalClusteringNearestMutualNeighboursGraphSamplingTransformer(sampler, clustering, resampling_factor=resampling_factor, use_balanced=use_balanced)
    return cc_sampler


class ConfidentGraphSamplingTransformer(object):
    def __init__(self, graph_sampling_transformer=None, graph_estimator=None, confidence_threshold=0.65, oversampling_factor=2):
        self.graph_sampling_transformer = graph_sampling_transformer
        self.graph_sampling_transformer.resampling_factor = self.graph_sampling_transformer.resampling_factor * oversampling_factor
        self.confidence_threshold = confidence_threshold
        self.graph_estimator = graph_estimator

    def fit(self, graphs, targets):
        self.graph_sampling_transformer.fit(graphs, targets)
        self.graph_estimator.fit(graphs, targets)
        return self
        
    def transform(self, graphs, targets):
        sample_graphs, sample_targets = self.graph_sampling_transformer.transform(graphs, targets)
        conf_sample_graphs, conf_sample_targets = self.select(sample_graphs, sample_targets)
        return conf_sample_graphs, conf_sample_targets

    def sample(self, n_samples):
        sample_graphs, sample_targets = self.graph_sampling_transformer.sample(n_samples)
        conf_sample_graphs, conf_sample_targets = self.select(sample_graphs, sample_targets)
        return conf_sample_graphs, conf_sample_targets

    def select(self, sample_graphs, sample_targets):
        sample_probs = self.graph_estimator.predict_proba(sample_graphs)
        sample_confidences = np.array([sample_prob[sample_target] for sample_target, sample_prob in zip(sample_targets, sample_probs)])
        conf_sample_graphs = [sample_graphs[idx] for idx in range(len(sample_graphs)) if sample_confidences[idx]>self.confidence_threshold]
        conf_sample_targets = [sample_targets[idx] for idx in range(len(sample_graphs)) if sample_confidences[idx]>self.confidence_threshold]
        return conf_sample_graphs, conf_sample_targets

    def fit_transform(self, X,y):
        return self.fit(X,y).transform(X,y)


def ConcreteConfidentGraphSamplingTransformer(graph_sampling_transformer=None, confidence_threshold=0.65, oversampling_factor=2):
    vectorizer = graph_sampling_transformer.sampler.vectorizer
    classifier = ExtraTreesClassifier(n_estimators=300, n_jobs=-1)
    graph_estimator = DataEstimator(data_transformer=vectorizer, estimator=classifier)
    sampler = ConfidentGraphSamplingTransformer(graph_sampling_transformer=graph_sampling_transformer, graph_estimator=graph_estimator, confidence_threshold=confidence_threshold, oversampling_factor=oversampling_factor)
    return sampler


class FilteredGraphSamplingTransformer(object):
    def __init__(self, graph_sampling_transformer=None, graph_vectorizer=None, probability_estimator=None, oversampling_factor=2):
        self.graph_sampling_transformer = graph_sampling_transformer
        self.graph_sampling_transformer.resampling_factor = self.graph_sampling_transformer.resampling_factor * oversampling_factor
        self.probability_estimator = probability_estimator
        self.graph_vectorizer = graph_vectorizer

    def fit(self, graphs, targets):
        self.graph_sampling_transformer.fit(graphs, targets)
        X = self.graph_vectorizer.fit_transform(graphs)
        self.probability_estimator.fit(X, targets)
        return self
        
    def transform(self, graphs, targets):
        sample_graphs, sample_targets = self.graph_sampling_transformer.transform(graphs, targets)
        conf_sample_graphs, conf_sample_targets = self.select(sample_graphs, sample_targets)
        return conf_sample_graphs, conf_sample_targets

    def sample(self, n_samples):
        sample_graphs, sample_targets = self.graph_sampling_transformer.sample(n_samples)
        conf_sample_graphs, conf_sample_targets = self.select(sample_graphs, sample_targets)
        return conf_sample_graphs, conf_sample_targets

    def select(self, sample_graphs, sample_targets):
        sample_X = self.graph_vectorizer.fit_transform(sample_graphs)
        sample_probs = self.probability_estimator.fit_predict_proba(sample_X, sample_targets)
        idxs = np.random.choice(len(sample_probs), size=len(sample_probs), p=sample_probs)
        conf_sample_graphs = [sample_graphs[idx] for idx in idxs]
        conf_sample_targets = [sample_targets[idx] for idx in idxs]
        return conf_sample_graphs, conf_sample_targets

    def fit_transform(self, X,y):
        return self.fit(X,y).transform(X,y)


def ConcreteFilteredGraphSamplingTransformer(graph_sampling_transformer=None, confidence_threshold=0.65, support_instances_fraction=0.3, oversampling_factor=2):
    vectorizer = graph_sampling_transformer.sampler.vectorizer
    classifier = ExtraTreesClassifier(n_estimators=300, n_jobs=-1)
    probability_estimators=[ClassificationConfidenceEstimator(estimator=classifier, confidence_threshold=confidence_threshold), 
                           nuSVMSupportVectorProbabilityEstimator(kernel='rbf', gamma='scale', nu_start=.01, nu_end=.99, n_steps=20, support_instances_fraction=support_instances_fraction)]
    probability_estimator = ProbabilityEstimator(probability_estimators)
    sampler = FilteredGraphSamplingTransformer(graph_sampling_transformer=graph_sampling_transformer, graph_vectorizer=vectorizer, probability_estimator=probability_estimator, oversampling_factor=oversampling_factor)
    return sampler
