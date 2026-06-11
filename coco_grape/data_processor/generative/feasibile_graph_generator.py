#!/usr/bin/env python
"""Provides interface."""

import copy
import networkx as nx 
import numpy as np
import random
from sklearn.metrics.pairwise import pairwise_distances
from coco_grape.data_processor.processor import DataEstimator
from coco_grape.graph_vectorizer.paired_neighborhood_graph_vectorizer import PairedNeighborhoodGraphVectorizer
from coco_grape.data_processor.unsupervised.spectral_equal_size_clustering import EqualSizeSpectralClustering


class FeasibleGraphGenerator(object):
    def __init__(self, feasibility_estimator, graph_neighborhood_generator, data_estimator=None):
        self.feasibility_estimator = feasibility_estimator
        self.graph_neighborhood_generator = graph_neighborhood_generator
        self.data_estimator = data_estimator

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def fit(self, graphs):
        self.seed_graphs = graphs
        self.feasibility_estimator.fit(graphs)
        self.graph_neighborhood_generator.feasibility_estimator = self.feasibility_estimator
        self.graph_neighborhood_generator.fit(graphs)
        if self.data_estimator is not None: self.data_estimator.fit(graphs)
        return self
        
    def sample(self, num_samples=1, num_iterations=1):
        seed_graphs = random.choices(self.seed_graphs, k=num_samples)
        output_graphs = self.graph_neighborhood_generator.iterated_neighbors(self.seed_graphs, num_iterations=num_iterations)
        if self.data_estimator is None: 
            output_graphs = random.choices(output_graphs, k=num_samples)
        else:
            scores = self.data_estimator.decision_function(output_graphs)
            idxs = np.argsort(-np.array(scores))[:num_samples]
            output_graphs = [output_graphs[idx] for idx in idxs]
        return output_graphs


class PartitionedFeasibleGraphGenerator(object):
    def __init__(self, feasible_graph_generator, graph_vectorizer=PairedNeighborhoodGraphVectorizer(radius=2,distance=3, nbits=15), num_clusters=1, equity_fraction=.9, data_estimator=None):
        self.feasible_graph_generator = feasible_graph_generator
        self.graph_vectorizer = graph_vectorizer
        self.num_clusters = num_clusters
        self.equity_fraction = equity_fraction
        self.data_estimator = data_estimator

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def partition(self, graphs, num_clusters, equity_fraction=.9): 
        if num_clusters == 1: return [graphs]
        embeddings = self.graph_vectorizer.transform(graphs)
        n_neighbors = max(2,int(len(graphs) * 0.1))
        cluster_idxs = EqualSizeSpectralClustering(n_clusters=num_clusters, n_neighbors=n_neighbors, equity_fraction=equity_fraction).fit_predict(embeddings)
        partitioned_graphs = [[graph for graph, cluster_idx in zip(graphs, cluster_idxs) if cluster_idx == curr_cluster_idx] for curr_cluster_idx in sorted(set(cluster_idxs))]
        return partitioned_graphs

    def fit(self, graphs, targets=None):
        if targets is not None and self.data_estimator is not None:
            self.data_estimator.fit(graphs, targets)
        if targets is None: 
            targets = [0]*len(graphs)
        target_classes = sorted(set(targets))
        self.graph_generator = dict()
        for target_class in target_classes:
            #consider data from each target class separately
            loc_graphs = [graph for graph, target in zip(graphs, targets) if target == target_class]
            #cluster graphs and for each cluster generate samples
            partitioned_graphs = self.partition(loc_graphs, self.num_clusters, self.equity_fraction)
            graph_generators = []
            for part_graphs in partitioned_graphs: 
                self.feasible_graph_generator.fit(part_graphs)
                graph_generators.append(copy.deepcopy(self.feasible_graph_generator))
            self.graph_generator[target_class] = copy.deepcopy(graph_generators)
        return self

    def sample(self, num_samples, num_iterations=1, excess_factor=2):
        num_parts = sum(len(self.graph_generator[target]) for target in self.graph_generator)
        if self.data_estimator is None: effective_excess_factor = 1
        else: effective_excess_factor = excess_factor
        desired_num_samples = max(1, int(num_samples / num_parts)) 
        effective_num_samples = desired_num_samples * effective_excess_factor
        generated_graphs = []
        generated_targets = []
        for target in self.graph_generator:
            for part_graph_generator in self.graph_generator[target]:
                generated_graphs_ = part_graph_generator.sample(num_samples=effective_num_samples, num_iterations=num_iterations)
                if self.data_estimator is not None:
                    #select instances that are predicted as most probable for the target under consideration
                    probs = self.data_estimator.predict_proba(generated_graphs_)
                    score = probs[:,target] #select the probability value for the target under consideration
                    idxs = np.argsort(-score)
                    idxs = idxs[:desired_num_samples]
                    generated_graphs_ = [generated_graphs_[idx] for idx in idxs]
                generated_graphs.extend(generated_graphs_)
                generated_targets.extend([target]*len(generated_graphs_))
        return generated_graphs, generated_targets
