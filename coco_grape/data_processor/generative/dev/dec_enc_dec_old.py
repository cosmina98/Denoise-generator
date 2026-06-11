#!/usr/bin/env python
"""Provides interface."""

import random
import copy
import time
import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
import dill as pickle
from coco_grape.module import *
from coco_grape.graph_vectorizer.graph_vectorizer import NodeGraphVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from coco_grape.data_processor.generative.nearest_mutual_neighbours_sampler import NearestMutualNeighboursEstimator
from coco_grape.data_processor.generative.nearest_mutual_neighbours_sampler import NearestMutualNeighboursProbabilityEstimator
from coco_grape.data_processor.generative.nearest_mutual_neighbours_sampler import NearestMutualNeighboursSampler
from coco_grape.data_processor.generative.nearest_mutual_neighbours_sampler import EncoderDecoderNearestMutualNeighboursSampler
from coco_grape.data_processor.generative.nearest_mutual_neighbours_sampler import ClassConditionalSamplingTransformer
from coco_grape.module import *
from coco_grape.data_processor.generative.feasibility_estimator import *
from coco_grape.data_processor.generative.feasibility_estimator import FeasibilityRepairEstimator, ConcreteFeasibilityEstimator
from coco_grape.data_processor.generative.neighborhood_generator import NeighborhoodEdgeRemove, NeighborhoodEdgeMove, NeighborhoodEdgeSwap, NeighborhoodEdgeAdd, GraphNeighborhoodGenerator
from coco_grape.vector_embedder.vector_embedder import VectorEmbedder, SVDTransformer, IdentityTransformer


# Suppress numpy warnings for invalid operations and divisions
np.seterr(invalid='ignore', divide='ignore')

class AdjacencyMatrixTransformer(object):
    """
    Transforms a list of NetworkX graphs into their corresponding adjacency matrices.
    
    Attributes:
        vectorize (bool): If True, reshapes each adjacency matrix into a 1D vector.
    """
    def __init__(self, vectorize=False):
        """
        Initializes the AdjacencyMatrixTransformer.
        
        Parameters:
            vectorize (bool): Whether to flatten the adjacency matrices into vectors.
        """
        self.vectorize = vectorize
        
    def fit(self, graphs, targets=None):
        """
        Fits the transformer. For this transformer, fitting does nothing.
        
        Parameters:
            graphs (list): List of NetworkX graphs.
            targets: Not used. Included for compatibility.
        
        Returns:
            self: Returns the instance itself.
        """
        return self
    
    def transform(self, graphs):
        """
        Transforms a list of graphs into their adjacency matrices.
        
        Parameters:
            graphs (list): List of NetworkX graphs.
        
        Returns:
            mtx_list (list): List of adjacency matrices as numpy arrays. 
                             If vectorize is True, each matrix is flattened.
        """
        # Convert each graph to its adjacency matrix as an integer numpy array
        mtx_list = [nx.to_numpy_array(graph).astype(int) for graph in graphs]
        # If vectorization is enabled, reshape each matrix to a 1D vector
        if self.vectorize:
            mtx_list = [A.reshape(1, -1) for A in mtx_list]
        return mtx_list
    
    def fit_transform(self, graphs, targets=None):
        """
        Fits the transformer and then transforms the graphs.
        
        Parameters:
            graphs (list): List of NetworkX graphs.
            targets: Not used. Included for compatibility.
        
        Returns:
            Transformed adjacency matrices.
        """
        return self.fit(graphs, targets).transform(graphs)


def adj_mtx_to_targets(adj_mtx_list):
    """
    Converts a list of adjacency matrices to a target vector for training.
    
    For each adjacency matrix, it extracts the presence (1) or absence (0) of edges 
    between distinct node pairs and appends them to a target list.
    
    Parameters:
        adj_mtx_list (list): List of adjacency matrices as numpy arrays.
    
    Returns:
        targets (numpy.ndarray): 1D array of edge presence indicators.
    """
    n_instances = len(adj_mtx_list)
    targets = []
    for t in range(n_instances):
        adj_mtx = adj_mtx_list[t]
        n_nodes = adj_mtx.shape[0]
        # Iterate over all possible node pairs
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    target = adj_mtx[i, j]
                    targets.append(target)
    # Convert the list to a numpy array for efficient processing
    targets = np.array(targets)
    return targets


def encodings_to_instances(encodings_list):
    """
    Converts a list of node encodings into instances for training the adjacency matrix predictor.
    
    Each instance consists of the graph encoding concatenated with the encodings of a pair of nodes.
    
    Parameters:
        encodings_list (list): List of numpy arrays, each containing node encodings for a graph.
    
    Returns:
        instances (numpy.ndarray): 2D array where each row is an instance.
    """
    n_instances = len(encodings_list)
    instances = []
    for t in range(n_instances):
        encodings = encodings_list[t]
        # Compute the graph encoding as the sum of all node encodings
        graph_encoding = np.sum(encodings, axis=0)
        n_nodes = encodings.shape[0]
        # Iterate over all possible node pairs
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    # Concatenate graph encoding with encodings of node i and node j
                    instance = np.hstack([graph_encoding, encodings[i], encodings[j]])
                    instances.append(instance)
    # Stack all instances into a single 2D array
    instances = np.vstack(instances)
    return instances


def encodings_and_adj_mtx_to_dataset(encodings_list, adj_mtx_list):
    """
    Combines node encodings and adjacency matrices to create a dataset for training.
    
    Parameters:
        encodings_list (list): List of node encodings for each graph.
        adj_mtx_list (list): List of adjacency matrices for each graph.
    
    Returns:
        instances (numpy.ndarray): Feature matrix.
        targets (numpy.ndarray): Target vector indicating edge presence.
    """
    instances = encodings_to_instances(encodings_list)
    targets = adj_mtx_to_targets(adj_mtx_list)
    return instances, targets


def encodings_and_graphs_to_node_label_dataset(encodings_list, graphs):
    """
    Creates a dataset for training node label predictors.
    
    Each instance consists of a node encoding, and the corresponding target is the node's label.
    
    Parameters:
        encodings_list (list): List of node encodings for each graph.
        graphs (list): List of NetworkX graphs.
    
    Returns:
        instances (numpy.ndarray): Node encodings stacked vertically.
        node_labels (list): List of node labels corresponding to each encoding.
    """
    n_instances = len(encodings_list)
    instances = []
    node_labels = []
    for t in range(n_instances):
        graph = graphs[t]
        encodings = encodings_list[t]
        n_nodes = encodings.shape[0]
        # Iterate over nodes with their indices and labels
        for i, u in zip(range(n_nodes), graph.nodes()):
            instances.append(encodings[i])
            node_labels.append(graph.nodes[u]['label'])
    # Stack all node encodings into a single 2D array
    instances = np.vstack(instances)
    return instances, node_labels


def encodings_and_graphs_to_edge_label_dataset(encodings_list, graphs):
    """
    Creates a dataset for training edge label predictors.
    
    Each instance consists of concatenated encodings of two connected nodes, and the target is the edge's label.
    
    Parameters:
        encodings_list (list): List of node encodings for each graph.
        graphs (list): List of NetworkX graphs.
    
    Returns:
        instances (numpy.ndarray): Concatenated node encodings for each edge.
        edge_labels (list): List of edge labels corresponding to each instance.
    """
    n_instances = len(encodings_list)
    instances = []
    edge_labels = []
    for t in range(n_instances):
        graph = graphs[t]
        encodings = encodings_list[t]
        n_nodes = encodings.shape[0]
        # Iterate over all node pairs to find existing edges
        for i, u in zip(range(n_nodes), graph.nodes()):
            for j, v in zip(range(n_nodes), graph.nodes()):
                if graph.has_edge(u, v):
                    # Concatenate encodings of node i and node j
                    instance = np.hstack([encodings[i], encodings[j]])
                    instances.append(instance)
                    label = graph.edges[u, v]['label']
                    edge_labels.append(label)
    # Stack all instances into a single 2D array
    instances = np.vstack(instances)
    return instances, edge_labels


def encodings_and_adj_mtx_to_edge_dataset(encodings_list, adj_mtx_list):
    """
    Creates a dataset for training edge presence predictors based on adjacency matrices.
    
    Each instance consists of concatenated encodings of two nodes where an edge exists.
    
    Parameters:
        encodings_list (list): List of node encodings for each graph.
        adj_mtx_list (list): List of adjacency matrices for each graph.
    
    Returns:
        instances (numpy.ndarray): Concatenated node encodings for existing edges.
    """
    n_instances = len(encodings_list)
    instances = []
    for t in range(n_instances):
        adj_mtx = adj_mtx_list[t]
        encodings = encodings_list[t]
        n_nodes = encodings.shape[0]
        # Iterate over all possible node pairs
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj_mtx[i, j] != 0:
                    # Concatenate encodings of node i and node j
                    instance = np.hstack([encodings[i], encodings[j]])
                    instances.append(instance)
    # Convert list to numpy array if there are any instances
    if len(instances) > 0:
        instances = np.vstack(instances)
    return instances


class DecompositionalNodeEncoderDecoder(object):
    """
    Encoder-Decoder model for nodes in a graph using decompositional approaches.
    
    It trains classifiers for node labels, edge labels, and adjacency matrices to decode graph structures.
    """
    def __init__(self, classifier=None, undirected=True, prob_threshold=0.01, verbose=True):
        """
        Initializes the DecompositionalNodeEncoderDecoder.
        
        Parameters:
            classifier: A machine learning classifier (e.g., ExtraTreesClassifier).
            undirected (bool): If True, ensures the adjacency matrix is symmetric.
            prob_threshold (float): Threshold to decide edge presence based on predicted probabilities.
            verbose (bool): If True, prints progress messages.
        """
        # Transformer to convert graphs to adjacency matrices
        self.adjacency_matrix_transformer = AdjacencyMatrixTransformer()
        # Deep copies of the classifier for different prediction tasks
        self.adjacency_matrix_classifier = copy.deepcopy(classifier)
        self.node_label_classifier = copy.deepcopy(classifier)
        self.edge_label_classifier = copy.deepcopy(classifier)
        self.undirected = undirected
        self.prob_threshold = prob_threshold
        self.verbose = verbose
        
    def fit(self, graphs, encodings_list):
        """
        Trains the node label classifier, edge label classifier, and adjacency matrix classifier.
        
        Parameters:
            graphs (list): List of NetworkX graphs.
            encodings_list (list): List of node encodings for each graph.
        
        Returns:
            self: Returns the instance itself.
        """
        # Train Node Label Classifier
        start = time.time()
        X, y = encodings_and_graphs_to_node_label_dataset(encodings_list, graphs)
        if self.verbose:
            print('Training node label predictor on %d instances with %d features' % (X.shape[0], X.shape[1]))
        self.node_label_classifier.fit(X, y)
        end = time.time()
        elapsed = end - start
        if self.verbose:
            print('  Time elapsed: %.1f s [%.1d m]\n' % (elapsed, elapsed / 60))
        
        # Train Edge Label Classifier
        start = time.time()
        X, y = encodings_and_graphs_to_edge_label_dataset(encodings_list, graphs)
        if self.verbose:
            print('Training edge label predictor on %d instances with %d features' % (X.shape[0], X.shape[1]))
        self.edge_label_classifier.fit(X, y)
        end = time.time()
        elapsed = end - start
        if self.verbose:
            print('  Time elapsed: %.1f s [%.1d m]\n' % (elapsed, elapsed / 60))
        
        # Train Adjacency Matrix Classifier
        start = time.time()
        adj_mtx_list = self.adjacency_matrix_transformer.fit_transform(graphs)
        X, y = encodings_and_adj_mtx_to_dataset(encodings_list, adj_mtx_list)
        if self.verbose:
            print('Training adjacency matrix predictor on %d instances with %d features' % (X.shape[0], X.shape[1]))
        self.adjacency_matrix_classifier.fit(X, y)
        end = time.time()
        elapsed = end - start
        if self.verbose:
            print('  Time elapsed: %.1f s [%.1d m]\n' % (elapsed, elapsed / 60))
        
        return self
        
    def decode_adjacency_matrix(self, encodings_list):
        """
        Predicts adjacency matrices for given node encodings using the trained classifier.
        
        Parameters:
            encodings_list (list): List of node encodings for each graph.
        
        Returns:
            adj_mtx_list (list): List of predicted adjacency matrices as numpy arrays.
        """
        # Calculate the number of edge predictions per graph
        sizes = [len(encoding)**2 - len(encoding) for encoding in encodings_list]
        # Create instances for adjacency matrix prediction
        X = encodings_to_instances(encodings_list)
        # Predict edge probabilities
        predicted_targets = self.adjacency_matrix_classifier.predict_proba(X)[:, -1]
        # Split predictions back into per-graph lists
        predicted_targets_list = np.split(predicted_targets, np.cumsum(sizes))
        
        adj_mtx_list = []
        for predicted_targets, encodings in zip(predicted_targets_list, encodings_list):
            n_nodes = len(encodings)
            p = len(predicted_targets)
            idx = 0
            # Initialize an empty adjacency matrix
            adj_mtx = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        adj_mtx[i, j] = predicted_targets[idx]
                        idx += 1
            # If undirected, symmetrize the adjacency matrix
            if self.undirected:
                adj_mtx = (adj_mtx + adj_mtx.T) / 2
            # Select the top probable edges based on node degree
            row_desc_argsorted_adj_mtx = np.argsort(-adj_mtx, axis=1)
            for i in range(n_nodes):
                n_edges = int(encodings[i, 1])  # Assuming feature 1 encodes the node's degree
                if n_edges > 0:
                    # Zero out edges that are not in the top n_edges for the node
                    adj_mtx[i, row_desc_argsorted_adj_mtx[i, n_edges:]] = 0
            # Apply probability threshold to decide on edge presence
            adj_mtx = np.where(adj_mtx >= self.prob_threshold, 1, 0)
            adj_mtx = adj_mtx.astype(int)
            # Optionally enforce undirected edges by requiring symmetry
            # if self.undirected:
            #     adj_mtx = (adj_mtx * adj_mtx.T)
            adj_mtx_list.append(adj_mtx)
        return adj_mtx_list
        
    def decode_node_labels(self, encodings_list):
        """
        Predicts node labels for given node encodings using the trained classifier.
        
        Parameters:
            encodings_list (list): List of node encodings for each graph.
        
        Returns:
            predicted_node_labels_list (list): List of predicted labels for each node in each graph.
        """
        # Stack all node encodings into a single matrix for prediction
        X = np.vstack(encodings_list)
        # Predict node labels
        predicted_node_labels = self.node_label_classifier.predict(X)
        # Determine the number of nodes per graph to split predictions accordingly
        sizes = [len(encoding) for encoding in encodings_list]
        predicted_node_labels_list = np.split(predicted_node_labels, np.cumsum(sizes))
        return predicted_node_labels_list
        
    def decode_edge_labels(self, encodings_list, adj_mtx_list):
        """
        Predicts edge labels for given adjacency matrices and node encodings using the trained classifier.
        
        Parameters:
            encodings_list (list): List of node encodings for each graph.
            adj_mtx_list (list): List of adjacency matrices for each graph.
        
        Returns:
            predicted_edge_labels_list (list): List of predicted labels for each edge in each graph.
        """
        # Create instances for edge label prediction based on adjacency matrices
        X = encodings_and_adj_mtx_to_edge_dataset(encodings_list, adj_mtx_list)
        if len(X) < 1:
            # If there are no edges, return a list of empty lists
            return [[] * len(encodings_list)]
        # Predict edge labels
        predicted_edge_labels = self.edge_label_classifier.predict(X)
        # Determine the number of edges per graph to split predictions accordingly
        sizes = [np.sum(adj_mtx) for adj_mtx in adj_mtx_list]
        predicted_edge_labels_list = np.split(predicted_edge_labels, np.cumsum(sizes))
        return predicted_edge_labels_list
        
    def decode(self, encodings_list):
        """
        Decodes node encodings into complete graphs with predicted node and edge labels.
        
        Parameters:
            encodings_list (list): List of node encodings for each graph.
        
        Returns:
            graphs (list): List of NetworkX graphs with predicted adjacency matrices and labels.
        """
        start = time.time()
        if self.verbose:
            print('Decoding %d instances' % (len(encodings_list)))

        # Predict adjacency matrices
        adj_mtx_list = self.decode_adjacency_matrix(encodings_list)
        # Predict node labels
        predicted_node_labels_list = self.decode_node_labels(encodings_list)
        # Predict edge labels
        predicted_edge_labels_list = self.decode_edge_labels(encodings_list, adj_mtx_list)

        graphs = []
        for predicted_node_labels, predicted_edge_labels, adj_mtx in zip(predicted_node_labels_list, predicted_edge_labels_list, adj_mtx_list):
            # Create a graph from the predicted adjacency matrix
            graph = nx.from_numpy_array(adj_mtx)
            # Map predicted labels to node attributes
            predicted_node_labels_map = {i: label for i, label in enumerate(predicted_node_labels)}
            nx.set_node_attributes(graph, predicted_node_labels_map, 'label')
            if np.sum(adj_mtx) > 0:  # Check if the adjacency matrix has any edges
                n_nodes = nx.number_of_nodes(graph)
                edge_idx = 0
                edge_attributes = dict()
                # Assign predicted labels to edges
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        if adj_mtx[i, j] != 0:
                            edge_attributes[(i, j)] = predicted_edge_labels[edge_idx]
                            edge_idx += 1
                nx.set_edge_attributes(graph, edge_attributes, 'label')
            graphs.append(graph)
        end = time.time()
        elapsed = end - start
        if self.verbose:
            print('  Time elapsed: %.1f s [%.1d m]\n' % (elapsed, elapsed / 60))
        return graphs
    
    def save(self, filename='generative_model.obj'):
        """
        Saves the trained encoder-decoder model to a file using dill.
        
        Parameters:
            filename (str): Path to the file where the model will be saved.
        """
        filehandler = open(filename, 'wb') 
        pickle.dump(self, filehandler)
    
    def load(self, filename='generative_model.obj'):
        """
        Loads a trained encoder-decoder model from a file using dill.
        
        Parameters:
            filename (str): Path to the file from which the model will be loaded.
        
        Returns:
            self: The loaded model instance.
        """
        filehandler = open(filename, 'rb') 
        self = pickle.load(filehandler)
        return self


class DecompositionalNodeAutoRegressor(object):
    """
    A regression model that predicts node encodings in a graph based on their approximate encodings
    and the graph's embedding. It uses a sliding window approach with an optional window size and
    can aggregate windowed encodings by summing them.

    Attributes:
        node_graph_vectorizer: Vectorizer for node features.
        node_regressor: Regressor model to predict node encodings.
        max_n_nodes (int): Maximum number of nodes to consider in any graph.
        min_n_nodes (int): Minimum number of nodes to consider in any graph.
        window_size (int or None): Size of the sliding window. If None, all preceding nodes are included.
        use_sum (bool): If True, sums the encodings within the window; otherwise, concatenates them.
        verbose (bool): If True, prints training and prediction progress.
        node_encoding_n_features (int): Number of features in each node encoding.
        position_encoding_n_features (int): Number of features in the position mask.
        n_features (int): Total number of features in each instance.
    """
    def __init__(self, node_graph_vectorizer=None, regressor=None, max_n_nodes=None, min_n_nodes=None, window_size=None, use_sum=True, verbose=True):
        """
        Initializes the DecompositionalNodeAutoRegressor.

        Parameters:
            node_graph_vectorizer: Vectorizer for node features.
            regressor: Regressor model to predict node encodings.
            max_n_nodes (int): Maximum number of nodes to consider.
            min_n_nodes (int): Minimum number of nodes to consider.
            window_size (int or None): Size of the sliding window (k). If None, use all preceding nodes.
            use_sum (bool): If True, sums the encodings within the window; otherwise, concatenates them.
            verbose (bool): If True, prints training progress.
        """
        self.node_graph_vectorizer = node_graph_vectorizer
        self.node_regressor = copy.deepcopy(regressor)
        self.max_n_nodes = max_n_nodes
        self.min_n_nodes = min_n_nodes
        self.window_size = window_size  # Sliding window size k; if None, use all preceding nodes
        self.use_sum = use_sum  # New parameter to determine encoding aggregation
        self.verbose = verbose
        self.node_encoding_n_features = None
        self.position_encoding_n_features = None  # Will be set to max_n_nodes
        self.n_features = None

    def make_instance(self, graph_encoding, encodings, current_index):
        """
        Creates an instance for training or prediction using a sliding window
        and binary mask position encoding.

        Parameters:
            graph_encoding (numpy.ndarray): Encoding of the entire graph (1D array).
            encodings (list or numpy.ndarray): List or array of node encodings up to current_index.
            current_index (int): The index of the current node to predict.

        Returns:
            instance (numpy.ndarray): A feature vector combining graph encoding, windowed node encodings (summed or concatenated), and position mask.
        """
        # Initialize instance with graph encoding converted to list
        instance = list(graph_encoding)

        # Determine the start index for the window
        if self.window_size is not None:
            # Use a fixed window size
            start_idx = max(current_index - self.window_size, 0)
        else:
            # Include all preceding nodes
            start_idx = 0
        end_idx = current_index  # Exclusive

        # Collect windowed node encodings
        window_encodings = []
        for i in range(start_idx, end_idx):
            window_encodings.extend(encodings[i])

        if self.use_sum:
            # If using sum, compute the sum of encodings within the window
            # Ensure the window is fully covered by padding if necessary
            if self.window_size is not None:
                num_missing = self.window_size - (end_idx - start_idx)
                if num_missing > 0:
                    # Pad with zeros to maintain consistent window size
                    window_encodings.extend([0] * (num_missing * self.node_encoding_n_features))
            else:
                # When window_size is None, pad with zeros up to max_n_nodes - 1
                expected_window_size = self.max_n_nodes - 1
                current_window_size = end_idx - start_idx
                num_missing = expected_window_size - current_window_size
                if num_missing > 0:
                    window_encodings.extend([0] * (num_missing * self.node_encoding_n_features))
            
            # Convert window_encodings to numpy array and compute sum
            window_array = np.array(window_encodings).reshape(-1, self.node_encoding_n_features)
            sum_window = np.sum(window_array, axis=0)
            # Append the summed window encoding to the instance
            instance.extend(sum_window.tolist())
        else:
            # If not using sum, append the individual windowed encodings as before
            if self.window_size is not None:
                num_missing = self.window_size - (end_idx - start_idx)
                if num_missing > 0:
                    # Pad with zeros to maintain consistent window size
                    window_encodings.extend([0] * (num_missing * self.node_encoding_n_features))
            else:
                # When window_size is None, pad with zeros up to max_n_nodes - 1
                expected_window_size = self.max_n_nodes - 1
                current_window_size = end_idx - start_idx
                num_missing = expected_window_size - current_window_size
                if num_missing > 0:
                    window_encodings.extend([0] * (num_missing * self.node_encoding_n_features))
            
            # Append windowed encodings to the instance
            instance.extend(window_encodings)

        # Create position mask vector
        position_mask = [0] * self.max_n_nodes
        for pos in range(start_idx, end_idx):
            if pos < self.max_n_nodes:
                position_mask[pos] = 1
        # If current_index is within bounds, include it
        if current_index < self.max_n_nodes:
            position_mask[current_index] = 1

        # Append position mask to the instance
        instance.extend(position_mask)

        # Convert the list back to a numpy array
        return np.array(instance)

    def fit(self, encodings_list):
        """
        Fits the regressor model using the provided encodings.

        Parameters:
            encodings_list (list of numpy.ndarray): List of node encodings for each graph.

        Returns:
            self: Returns the instance itself.
        """
        # Determine max and min number of nodes if not already set
        if self.max_n_nodes is None:
            self.max_n_nodes = max(encodings.shape[0] for encodings in encodings_list)
        if self.min_n_nodes is None:
            self.min_n_nodes = min(encodings.shape[0] for encodings in encodings_list)

        # Set the number of features per node encoding
        self.node_encoding_n_features = encodings_list[0].shape[1]
        # Set the length of the position mask to max_n_nodes
        self.position_encoding_n_features = self.max_n_nodes  # Binary mask length

        # If window_size is None, set it to include all preceding nodes (i.e., window_size = max_n_nodes - 1)
        if self.window_size is None:
            self.window_size = self.max_n_nodes - 1

        # Calculate the total number of features:
        # graph_encoding + window_encodings (summed or concatenated) + position_mask
        if self.use_sum:
            # graph_encoding + sum_window_encoding + position_mask
            self.n_features = self.node_encoding_n_features + \
                              self.node_encoding_n_features + \
                              self.position_encoding_n_features
        else:
            # graph_encoding + (window_size * node_encoding_n_features) + position_mask
            self.n_features = self.node_encoding_n_features + \
                              (self.window_size * self.node_encoding_n_features) + \
                              self.position_encoding_n_features

        instances = []
        targets = []
        for encodings in encodings_list:
            # Compute the graph encoding as the sum of all node encodings
            graph_encoding = np.sum(encodings, axis=0)
            n_nodes = encodings.shape[0]
            for i in range(n_nodes):
                # Create feature instance for node i
                instance = self.make_instance(graph_encoding, encodings, i)
                instances.append(instance)
                # The target is the original node encoding
                target = encodings[i]
                targets.append(target)

        # Convert lists to numpy arrays for efficient computation
        instances = np.vstack(instances)
        targets = np.vstack(targets)

        start = time.time()
        if self.verbose:
            print(f'Training regressor on {instances.shape[0]} instances with {instances.shape[1]} features to predict targets with {targets.shape[1]} features')
        # Fit the regression model
        self.node_regressor.fit(instances, targets)
        end = time.time()
        elapsed = end - start
        if self.verbose:
            print(f'Time elapsed: {elapsed:.1f} s [{elapsed/60:.1f} m]')

        return self

    def predict(self, graph_encodings):
        """
        Predicts node encodings for the provided graph encodings.

        Parameters:
            graph_encodings (list or numpy.ndarray): List of graph encodings.

        Returns:
            encodings_list (list of numpy.ndarray): List of predicted node encodings for each graph.
        """
        start = time.time()
        if self.verbose:
            print(f'Predicting node embeddings for {len(graph_encodings)} instances')

        encodings_list = []
        for graph_encoding in graph_encodings:
            # Ensure graph_encoding is a 1D array
            graph_encoding = graph_encoding.flatten()
            # Infer the number of nodes from the graph encoding if necessary
            # Here, we assume that the first element encodes the number of nodes
            n_nodes = int(graph_encoding[0])  # Modify as per actual encoding
            # Clamp the number of nodes between min_n_nodes and max_n_nodes
            n_nodes = max(self.min_n_nodes, n_nodes)
            n_nodes = min(self.max_n_nodes, n_nodes)
            encoding_list = []
            for i in range(n_nodes):
                # Create feature instance for node i
                instance = self.make_instance(graph_encoding, encoding_list, i)
                instance = instance.reshape(1, -1)
                # Predict the node encoding using the regression model
                encoding = self.node_regressor.predict(instance)
                encoding_list.append(encoding.flatten())
            # Ensure encodings are compatible with graph_encoding
            graph_encoding_from_node_encodings = np.sum(encoding_list, axis=0)
            # Compute normalization factor to align predicted encodings with graph encoding
            normalization_factor = np.divide(
                graph_encoding,
                graph_encoding_from_node_encodings,
                out=np.ones_like(graph_encoding, dtype=float),
                where=graph_encoding_from_node_encodings != 0
            )

            # Apply normalization to the predicted encodings
            encodings = np.array(encoding_list) * normalization_factor

            encodings_list.append(encodings)

        end = time.time()
        elapsed = end - start
        if self.verbose:
            print(f'Time elapsed: {elapsed:.1f} s [{elapsed/60:.1f} m]')
        return encodings_list

    def save(self, filename='generative_model.obj'):
        """
        Saves the trained model to a file.

        Parameters:
            filename (str): Path to the file where the model will be saved.
        """
        with open(filename, 'wb') as filehandler:
            pickle.dump(self, filehandler)

    def load(self, filename='generative_model.obj'):
        """
        Loads a trained model from a file.

        Parameters:
            filename (str): Path to the file from which the model will be loaded.

        Returns:
            self: The loaded model instance.
        """
        with open(filename, 'rb') as filehandler:
            loaded_obj = pickle.load(filehandler)
        # Update the current instance's attributes with the loaded object's attributes
        self.__dict__.update(loaded_obj.__dict__)
        return self



class DecompositionalEncoderDecoder(object):

    def __init__(self, node_graph_vectorizer=None, decompositional_node_autoregressor=None, decompositional_node_encoder_decoder=None, vector_embedder=None, verbose=True):
        self.node_graph_vectorizer = node_graph_vectorizer
        self.decompositional_node_autoregressor = decompositional_node_autoregressor
        self.decompositional_node_encoder_decoder = decompositional_node_encoder_decoder
        self.vector_embedder = vector_embedder
        self.verbose = verbose
        
    def fit(self, graphs):
        self.node_graph_vectorizer.fit(graphs)
        encodings_list = self.node_encode(graphs)
        graph_encodings = [np.sum(encodings, axis=0) for encodings in encodings_list]
        graph_encodings = np.vstack(graph_encodings)
        self.vector_embedder.fit(graph_encodings)
        self.decompositional_node_autoregressor.fit(encodings_list)
        self.decompositional_node_encoder_decoder.fit(graphs, encodings_list)
        return self
        
    def node_encode(self, graphs):
        start = time.time()
        if self.verbose: print('Encoding %d graphs'%(len(graphs)))
        encodings_list = self.node_graph_vectorizer.transform(graphs)
        end = time.time()
        elapsed = end - start
        if self.verbose: print('  Time elapsed: %.1f s [%.1d m]\n'%(elapsed, elapsed/60))
        return encodings_list
    
    def encode(self, graphs):
        encodings_list = self.node_encode(graphs)
        graph_encodings = [np.sum(encodings, axis=0) for encodings in encodings_list]
        graph_encodings = np.vstack(graph_encodings)
        graph_embeddings = self.vector_embedder.transform(graph_encodings)
        return graph_embeddings
    
    def decode(self, graph_embeddings):
        graph_encodings = self.vector_embedder.inverse_transform(graph_embeddings)
        encodings_list = self.decompositional_node_autoregressor.predict(graph_encodings)
        graphs = self.decompositional_node_encoder_decoder.decode(encodings_list)
        return graphs

    def save(self, filename='generative_model.obj'):
        filehandler = open(filename, 'wb') 
        pickle.dump(self, filehandler)

    def load(self, filename='generative_model.obj'):
        filehandler = open(filename, 'rb') 
        self = pickle.load(filehandler)
        return self


class DecompositionalRepairTransformer(object):
    """
    Transformer to repair approximate node encodings by learning a mapping 
    from approximate encodings and graph embeddings to original node encodings.
    
    It uses a regression model to predict the original node encodings based on 
    their approximate encodings and the graph-level embeddings.
    """
    def __init__(self, decompositional_encoder_decoder=None, regressor=None, verbose=True):
        """
        Initializes the DecompositionalRepairTransformer.
        
        Parameters:
            decompositional_encoder_decoder: An instance of DecompositionalEncoderDecoder that has been fitted.
            regressor: A regression model (e.g., ExtraTreesRegressor) used to learn the mapping.
            verbose (bool): If True, prints progress messages.
        """
        self.decompositional_encoder_decoder = decompositional_encoder_decoder
        self.regressor = regressor
        self.verbose = verbose

    def make_train_data(self, graph_embeddings, encodings_list, approx_encodings_list):
        """
        Creates training data by matching approximate node encodings to original node encodings
        using linear assignment matching.
        
        For each graph:
            - Computes a cost matrix based on Euclidean distance between encodings.
            - Finds the optimal one-to-one matching between original and approximate encodings.
            - Constructs input features by concatenating approximate encodings with graph embeddings.
            - Associates each input feature with the corresponding original node encoding as the target.
        
        Parameters:
            graph_embeddings (list or numpy.ndarray): List of graph embeddings for each graph.
            encodings_list (list of numpy.ndarray): List of original node encodings for each graph.
            approx_encodings_list (list of numpy.ndarray): List of approximate node encodings for each graph.
        
        Returns:
            X (numpy.ndarray): Feature matrix for training. Each row is [approx_node_encoding, graph_embedding].
            Y (numpy.ndarray): Target matrix for training. Each row is the original node_encoding.
        """
        X = []
        Y = []
        for graph_embedding, encodings, approx_encodings in zip(graph_embeddings, encodings_list, approx_encodings_list):
            n_nodes = encodings.shape[0]
            # Ensure that the number of nodes matches between encodings and approx_encodings
            if encodings.shape[0] != approx_encodings.shape[0]:
                raise ValueError("Number of nodes in encodings and approximate encodings must be the same.")
            
            # Compute the cost matrix (Euclidean distance) between original and approximate encodings
            cost_matrix = np.linalg.norm(encodings[:, np.newaxis] - approx_encodings[np.newaxis, :], axis=2)
            
            # Perform linear assignment to find the optimal matching
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Iterate over the matched pairs to construct X and Y
            for original_idx, approx_idx in zip(row_ind, col_ind):
                # Extract the matched encodings
                original_node_encoding = encodings[original_idx]
                approx_node_encoding = approx_encodings[approx_idx]
                
                # Concatenate approx_node_encoding with graph_embedding to form the input feature
                # Reshape to 1D array if necessary
                x = np.hstack([approx_node_encoding, graph_embedding])
                X.append(x)
                
                # The target is the original node encoding
                y = original_node_encoding
                Y.append(y)
        
        # Convert lists to numpy arrays for efficient computation
        X = np.array(X)
        Y = np.array(Y)
        return X, Y 

    def make_test_data(self, graph_embeddings, approx_encodings_list):
        """
        Creates test data by concatenating approximate node encodings with graph embeddings.
        
        Parameters:
            graph_embeddings (list or numpy.ndarray): List of graph embeddings for each graph.
            approx_encodings_list (list of numpy.ndarray): List of approximate node encodings for each graph.
        
        Returns:
            X (numpy.ndarray): Feature matrix for testing. Each row is [approx_node_encoding, graph_embedding].
        """
        X = []
        for graph_embedding, approx_encodings in zip(graph_embeddings, approx_encodings_list):
            for approx_node_encoding in approx_encodings:
                # Concatenate approx_node_encoding with graph_embedding to form the input feature
                x = np.hstack([approx_node_encoding, graph_embedding])
                X.append(x)
        X = np.array(X)
        return X

    def fit(self, graphs):
        """
        Fits the regression model to learn the mapping from approximate node encodings and graph embeddings
        to original node encodings.
        
        Parameters:
            graphs (list): List of NetworkX graphs to train on.
        
        Returns:
            self: Returns the instance itself.
        """
        # Encode the original graphs
        encodings_list = self.decompositional_encoder_decoder.node_encode(graphs)
        # Compute graph-level embeddings
        graph_embeddings = self.decompositional_encoder_decoder.encode(graphs)
        # Decode to get approximate graphs based on embeddings
        approx_graphs = self.decompositional_encoder_decoder.decode(graph_embeddings)
        # Encode the approximate graphs
        approx_encodings_list = self.decompositional_encoder_decoder.node_encode(approx_graphs)
        
        # Create training data by matching original and approximate encodings
        X, Y = self.make_train_data(graph_embeddings, encodings_list, approx_encodings_list)
        
        # Fit the regression model to learn the mapping
        self.regressor.fit(X, Y)
        
        if self.verbose:
            print(f"Regressor trained on {X.shape[0]} samples with {X.shape[1]} features.")
        
        return self

    def transform(self, graph_encodings, graphs):
        """
        Repairs the node encodings of approximate graphs by predicting the original encodings
        using the trained regression model.
        
        Parameters:
            graph_encodings (list or numpy.ndarray): List of graph embeddings for each graph to be repaired.
            graphs (list): List of approximate NetworkX graphs to be repaired.
        
        Returns:
            repaired_graphs (list): List of repaired NetworkX graphs with updated node encodings.
        """
        start = time.time()
        if self.verbose:
            print(f'Repairing {len(graphs)} graphs')
        
        # Encode the approximate graphs
        approx_encodings_list = self.decompositional_encoder_decoder.node_encode(graphs)
        encodings_list = []
        
        for graph_encoding, approx_node_encodings in zip(graph_encodings, approx_encodings_list):
            # Ensure graph_encoding is a 1D array
            graph_encoding = graph_encoding.flatten()
            
            # Create input features by concatenating approx_node_encoding with graph_embedding
            X = []
            for approx_node_encoding in approx_node_encodings:
                x = np.hstack([approx_node_encoding, graph_encoding])
                X.append(x)
            X = np.array(X)
            
            # Predict the original node encodings using the regression model
            node_encodings = self.regressor.predict(X)
            # Optionally, convert predictions to integers if necessary
            node_encodings = node_encodings.astype(int)
            encodings_list.append(node_encodings)
        
        # Decode the repaired node encodings back into graphs
        repaired_graphs = self.decompositional_encoder_decoder.decompositional_node_encoder_decoder.decode(encodings_list)
        
        end = time.time()
        elapsed = end - start
        if self.verbose:
            print(f'  Time elapsed: {elapsed:.1f} s [{elapsed/60:.1f} m]\n')
        
        return repaired_graphs

    def save(self, filename='generative_model.obj'):
        """
        Saves the trained transformer to a file using pickle.
        
        Parameters:
            filename (str): Path to the file where the transformer will be saved.
        """
        with open(filename, 'wb') as filehandler:
            pickle.dump(self, filehandler)
        if self.verbose:
            print(f'Transformer saved to {filename}.')

    def load(self, filename='generative_model.obj'):
        """
        Loads a trained transformer from a file using pickle.
        
        Parameters:
            filename (str): Path to the file from which the transformer will be loaded.
        
        Returns:
            self: The loaded transformer instance.
        """
        with open(filename, 'rb') as filehandler:
            loaded_obj = pickle.load(filehandler)
        # Update the current instance's attributes with the loaded object's attributes
        self.__dict__.update(loaded_obj.__dict__)
        if self.verbose:
            print(f'Transformer loaded from {filename}.')
        return self

    

class DecompositionalRepairedEncoderDecoder(object):

    def __init__(self, node_repair_regressor=None, node_graph_vectorizer=None, decompositional_node_autoregressor=None, decompositional_node_encoder_decoder=None, vector_embedder=None, verbose=True):
        self.decompositional_encoder_decoder = DecompositionalEncoderDecoder(
            node_graph_vectorizer=node_graph_vectorizer, 
            decompositional_node_autoregressor=decompositional_node_autoregressor, 
            decompositional_node_encoder_decoder=decompositional_node_encoder_decoder, 
            vector_embedder=vector_embedder, 
            verbose=verbose)
        self.decompositional_repair_transformer = DecompositionalRepairTransformer(
            decompositional_encoder_decoder=self.decompositional_encoder_decoder, 
            regressor=node_repair_regressor, 
            verbose=verbose)

    def fit(self, graphs):
        n_graphs = len(graphs)
        self.decompositional_encoder_decoder.fit(graphs[:n_graphs//2])
        self.decompositional_repair_transformer.decompositional_encoder_decoder = self.decompositional_encoder_decoder
        self.decompositional_repair_transformer.fit(graphs[n_graphs//2:])
        return self

    def encode(self, graphs):
        graph_embeddings = self.decompositional_encoder_decoder.encode(graphs)
        return graph_embeddings
    
    def decode(self, graph_embeddings):
        approx_graphs = self.decompositional_encoder_decoder.decode(graph_embeddings)
        graphs = self.decompositional_repair_transformer.transform(graph_embeddings, approx_graphs)
        return graphs


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def ConcreteDecompositionalEncoderDecoder(n_estimators=1000, decomposition_function=None, nbits=10, window_size=3, n_components=None, prob_threshold=.01, verbose=True):
    if decomposition_function is None: decomposition_function = add(neighborhood(), path(min_size=1, max_size=3), pairwise_neighborhood(size=0, min_distance=1, max_distance=5))
    node_graph_vectorizer = NodeGraphVectorizer(
        decomposition_function=decomposition_function,
        nbits=nbits,
        use_attributes=False,
        parallel=True,
        dense=True)
    classifier = ExtraTreesClassifier(n_estimators=n_estimators, n_jobs=-1, class_weight='balanced')
    regressor = ExtraTreesRegressor(n_estimators=n_estimators, n_jobs=-1)
    decompositional_node_autoregressor = DecompositionalNodeAutoRegressor(regressor=regressor, window_size=window_size, verbose=verbose)
    decompositional_node_encoder_decoder = DecompositionalNodeEncoderDecoder(classifier=classifier, prob_threshold=prob_threshold, verbose=verbose)
    if n_components is not None: vector_embedder = VectorEmbedder(transformers=[SVDTransformer(n_components=n_components)])
    else: vector_embedder = VectorEmbedder(transformers=[IdentityTransformer()])
    decompositional_encoder_decoder = DecompositionalEncoderDecoder(node_graph_vectorizer=node_graph_vectorizer, decompositional_node_autoregressor=decompositional_node_autoregressor, decompositional_node_encoder_decoder=decompositional_node_encoder_decoder, vector_embedder=vector_embedder, verbose=verbose)
    return decompositional_encoder_decoder


def ConcreteClassConditionalDecompositionalEncoderDecoderSampler(n_estimators=1000, decomposition_function=None, nbits=10, window_size=3, n_components=None, prob_threshold=.01, verbose=True, n_neighbours=5, resampling_factor=1, interpolation_factor=1, min_interpolation_factor=1, use_balanced=False, use_min_max_constraints=False, metric='euclidean'):
    nearest_mutual_neighbours_estimator = NearestMutualNeighboursEstimator(n_neighbours, metric)
    probability_estimator = NearestMutualNeighboursProbabilityEstimator(n_neighbours, metric)
    nearest_mutual_neighbours_sampler = NearestMutualNeighboursSampler(
        nearest_mutual_neighbours_estimator,
        probability_estimator,
        interpolation_factor=interpolation_factor,
        min_interpolation_factor=min_interpolation_factor,
        use_min_max_constraints=use_min_max_constraints
    )
    encoder_decoder = ConcreteDecompositionalEncoderDecoder(n_estimators=n_estimators, decomposition_function=decomposition_function, nbits=nbits, window_size=window_size, n_components=n_components, prob_threshold=prob_threshold, verbose=verbose)
    sampler = EncoderDecoderNearestMutualNeighboursSampler(encoder_decoder, nearest_mutual_neighbours_sampler)
    cc_sampler = ClassConditionalSamplingTransformer(sampler, resampling_factor=resampling_factor, use_balanced=use_balanced)
    return cc_sampler


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def ConcreteDecompositionalRepairedEncoderDecoder(n_estimators=1000, decomposition_function=None, nbits=10, window_size=3, n_components=None, prob_threshold=.01, verbose=True):
    if decomposition_function is None: decomposition_function = add(neighborhood(), path(min_size=1, max_size=3), pairwise_neighborhood(size=0, min_distance=1, max_distance=5))
    node_graph_vectorizer = NodeGraphVectorizer(
        decomposition_function=decomposition_function,
        nbits=nbits,
        use_attributes=False,
        parallel=True,
        dense=True)
    classifier = ExtraTreesClassifier(n_estimators=n_estimators, n_jobs=-1, class_weight='balanced')
    regressor = ExtraTreesRegressor(n_estimators=n_estimators, n_jobs=-1)
    decompositional_node_autoregressor = DecompositionalNodeAutoRegressor(regressor=regressor, window_size=window_size, verbose=verbose)
    decompositional_node_encoder_decoder = DecompositionalNodeEncoderDecoder(classifier=classifier, prob_threshold=prob_threshold, verbose=verbose)
    if n_components is not None: vector_embedder = VectorEmbedder(transformers=[SVDTransformer(n_components=n_components)])
    else: vector_embedder = VectorEmbedder(transformers=[IdentityTransformer()])
    node_repair_regressor = ExtraTreesRegressor(n_estimators=n_estimators, n_jobs=-1)
    decompositional_encoder_decoder = DecompositionalRepairedEncoderDecoder(node_repair_regressor=node_repair_regressor, node_graph_vectorizer=node_graph_vectorizer, decompositional_node_autoregressor=decompositional_node_autoregressor, decompositional_node_encoder_decoder=decompositional_node_encoder_decoder, vector_embedder=vector_embedder, verbose=verbose)
    return decompositional_encoder_decoder


def ConcreteClassConditionalDecompositionalRepairedEncoderDecoderSampler(n_estimators=1000, decomposition_function=None, nbits=10, window_size=3, n_components=None, prob_threshold=.01, verbose=True, n_neighbours=5, resampling_factor=1, interpolation_factor=1, min_interpolation_factor=1, use_balanced=False, use_min_max_constraints=False, metric='euclidean'):
    nearest_mutual_neighbours_estimator = NearestMutualNeighboursEstimator(n_neighbours, metric)
    probability_estimator = NearestMutualNeighboursProbabilityEstimator(n_neighbours, metric)
    nearest_mutual_neighbours_sampler = NearestMutualNeighboursSampler(
        nearest_mutual_neighbours_estimator,
        probability_estimator,
        interpolation_factor=interpolation_factor,
        min_interpolation_factor=min_interpolation_factor,
        use_min_max_constraints=use_min_max_constraints
    )
    encoder_decoder = ConcreteDecompositionalRepairedEncoderDecoder(n_estimators=n_estimators, decomposition_function=decomposition_function, nbits=nbits, window_size=window_size, n_components=n_components, prob_threshold=prob_threshold, verbose=verbose)
    sampler = EncoderDecoderNearestMutualNeighboursSampler(encoder_decoder, nearest_mutual_neighbours_sampler)
    cc_sampler = ClassConditionalSamplingTransformer(sampler, resampling_factor=resampling_factor, use_balanced=use_balanced)
    return cc_sampler


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class ConcreteFeasibleClassConditionalDecompositionalEncoderDecoderSampler(object):
    def __init__(self, n_estimators=1000, decomposition_function=None, nbits=10, window_size=3, n_components=None, prob_threshold=.01, verbose=True, n_neighbours=5, resampling_factor=1, interpolation_factor=1, min_interpolation_factor=1, use_balanced=False, use_min_max_constraints=False, metric='euclidean', feasibility_estimators=None):
        self.verbose = verbose
        if feasibility_estimators is None: 
            feasibility_df = add(neighborhood(), cycle(abstraction_level='unlabelled_graph_process'), compose(unique(), filter_by_number_of_connected_components(size=1), combination(size=2), cycle(abstraction_level='unlabelled_graph_process')))
            feasibility_estimators = [FeasibilityEstimatorIsConnected(), FeasibilityEstimatorFeatureCannotExist(decomposition_function=feasibility_df, nbits=19)]
        self.feasibility_estimator = FeasibilityEstimator(feasibility_estimators, parallel=False)

        self.sampler = ConcreteClassConditionalDecompositionalEncoderDecoderSampler(
            n_estimators=n_estimators, 
            decomposition_function=decomposition_function, 
            nbits=nbits, 
            window_size=window_size,
            n_components=n_components,
            prob_threshold=prob_threshold, 
            verbose=verbose, 
            n_neighbours=n_neighbours, 
            resampling_factor=resampling_factor, 
            interpolation_factor=interpolation_factor, 
            min_interpolation_factor=min_interpolation_factor, 
            use_balanced=use_balanced, 
            use_min_max_constraints=use_min_max_constraints, 
            metric=metric)

    def fit(self, graphs, targets=None):
        start = time.time()
        if self.verbose: print('Training feasibility estimator for %d instances'%(len(graphs)))
        self.feasibility_estimator.fit(graphs)
        end = time.time()
        elapsed = end - start
        if self.verbose: print('  Time elapsed: %.1f s [%.1d m]\n'%(elapsed, elapsed/60))
        
        self.sampler.fit(graphs, targets)
        return self

    def sample(self, n_samples):
        sampled_graphs, targets = self.sampler.sample(n_samples)
        selected_graphs, selected_targets = self.feasibility_estimator.filter(sampled_graphs, targets)
        if self.verbose: print('#generated graphs:%d  #feasible graphs:%d  [efficiency: %.2f perc]'%(len(sampled_graphs), len(selected_graphs), len(selected_graphs)/len(sampled_graphs)))
        return selected_graphs, selected_targets

    def save(self, filename='generative_model.obj'):
        filehandler = open(filename, 'wb') 
        pickle.dump(self, filehandler)

    def load(self, filename='generative_model.obj'):
        filehandler = open(filename, 'rb') 
        self = pickle.load(filehandler)
        return self



class ConcreteRepairedFeasibleClassConditionalDecompositionalEncoderDecoderSampler(object):
    def __init__(self, n_estimators=1000, decomposition_function=None, nbits=10, window_size=3, n_components=None, prob_threshold=.01, verbose=True, n_neighbours=5, resampling_factor=1, interpolation_factor=1, min_interpolation_factor=1, use_balanced=False, use_min_max_constraints=False, metric='euclidean', feasibility_estimators=None, max_num_repair_steps=50, beam_size=5, min_number_of_violations=0, max_n_neighborhood_graphs=10, size=30):
        self.verbose = verbose
        if feasibility_estimators is None: 
            feasibility_df = add(neighborhood(), cycle(abstraction_level='unlabelled_graph_process'), compose(unique(), filter_by_number_of_connected_components(size=1), combination(size=2), cycle(abstraction_level='unlabelled_graph_process')))
            feasibility_estimators = [FeasibilityEstimatorIsConnected(), FeasibilityEstimatorFeatureCannotExist(decomposition_function=feasibility_df, nbits=nbits)]
        self.feasibility_estimator = FeasibilityEstimator(feasibility_estimators, parallel=False)

        self.sampler = ConcreteClassConditionalDecompositionalEncoderDecoderSampler(
            n_estimators=n_estimators, 
            decomposition_function=decomposition_function, 
            nbits=nbits, 
            window_size=window_size,
            n_components=n_components,
            prob_threshold=prob_threshold, 
            verbose=verbose, 
            n_neighbours=n_neighbours, 
            resampling_factor=resampling_factor, 
            interpolation_factor=interpolation_factor, 
            min_interpolation_factor=min_interpolation_factor, 
            use_balanced=use_balanced, 
            use_min_max_constraints=use_min_max_constraints, 
            metric=metric)

        generators = [NeighborhoodEdgeMove(size=size), NeighborhoodEdgeSwap(size=size), NeighborhoodEdgeRemove(size=size), NeighborhoodEdgeAdd(size=size)]
        perturbation_generator = GraphNeighborhoodGenerator(generators, parallel=False, max_n_neighborhood_graphs=max_n_neighborhood_graphs)
        self.feasibility_repair_estimator = FeasibilityRepairEstimator(
            perturbation_generator=perturbation_generator, 
            feasibility_estimator=self.feasibility_estimator, 
            n_iter=max_num_repair_steps, 
            beam_size=beam_size,
            min_number_of_violations=min_number_of_violations)


    def fit(self, graphs, targets=None):
        start = time.time()
        if self.verbose: print('Training feasibility estimator for %d instances'%(len(graphs)))
        self.feasibility_estimator.fit(graphs)
        self.feasibility_repair_estimator.fit(graphs)
        end = time.time()
        elapsed = end - start
        if self.verbose: print('  Time elapsed: %.1f s [%.1d m]\n'%(elapsed, elapsed/60))
        
        self.sampler.fit(graphs, targets)
        return self

    def sample(self, n_samples, return_all=False):
        start = time.time()
        sampled_graphs, targets = self.sampler.sample(n_samples)
        end = time.time()
        elapsed = end - start
        if self.verbose: print('  Time elapsed: %.1f s [%.1d m]\n'%(elapsed, elapsed/60))
        start = time.time()
        if self.verbose: print('Repairing and filtering for feasibility %d instances'%(len(sampled_graphs)))
        repaired_sampled_graphs = self.feasibility_repair_estimator.repair(sampled_graphs)
        feasibile_graphs, feasibile_targets = self.feasibility_estimator.filter(repaired_sampled_graphs, targets)
        end = time.time()
        elapsed = end - start
        if self.verbose: print('  Time elapsed: %.1f s [%.1d m]\n'%(elapsed, elapsed/60))
        if self.verbose: print('#generated graphs:%d  #feasible graphs:%d  [efficiency: %.2f perc]'%(len(sampled_graphs), len(feasibile_graphs), 100*len(feasibile_graphs)/len(sampled_graphs)))
        if return_all: return feasibile_graphs, feasibile_targets, repaired_sampled_graphs, targets, sampled_graphs, targets
        return feasibile_graphs, feasibile_targets

    def save(self, filename='generative_model.obj'):
        filehandler = open(filename, 'wb') 
        pickle.dump(self, filehandler)

    def load(self, filename='generative_model.obj'):
        filehandler = open(filename, 'rb') 
        self = pickle.load(filehandler)
        return self


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class ConcreteAttributedRepairedFeasibleClassConditionalDecompositionalEncoderDecoderSampler(object):
    def __init__(self, n_estimators=1000, decomposition_function=None, nbits=10, window_size=3, n_components=None, prob_threshold=.01, verbose=True, n_neighbours=5, resampling_factor=1, interpolation_factor=1, min_interpolation_factor=1, use_balanced=False, use_min_max_constraints=False, metric='euclidean', feasibility_estimators=None, max_num_repair_steps=50, beam_size=5, min_number_of_violations=0, max_n_neighborhood_graphs=10, size=30):
        self.verbose = verbose
        if feasibility_estimators is None: 
            feasibility_df = add(neighborhood(), cycle(abstraction_level='unlabelled_graph_process'), compose(unique(), filter_by_number_of_connected_components(size=1), combination(size=2), cycle(abstraction_level='unlabelled_graph_process')))
            feasibility_estimators = [FeasibilityEstimatorIsConnected(), FeasibilityEstimatorFeatureCannotExist(decomposition_function=feasibility_df, nbits=nbits)]
        self.feasibility_estimator = FeasibilityEstimator(feasibility_estimators, parallel=False)

        self.sampler = ConcreteClassConditionalDecompositionalEncoderDecoderSampler(
            n_estimators=n_estimators, 
            decomposition_function=decomposition_function, 
            nbits=nbits, 
            window_size=window_size,
            n_components=n_components,
            prob_threshold=prob_threshold, 
            verbose=verbose, 
            n_neighbours=n_neighbours, 
            resampling_factor=resampling_factor, 
            interpolation_factor=interpolation_factor, 
            min_interpolation_factor=min_interpolation_factor, 
            use_balanced=use_balanced, 
            use_min_max_constraints=use_min_max_constraints, 
            metric=metric)

        generators = [NeighborhoodEdgeMove(size=size), NeighborhoodEdgeSwap(size=size), NeighborhoodEdgeRemove(size=size), NeighborhoodEdgeAdd(size=size)]
        perturbation_generator = GraphNeighborhoodGenerator(generators, parallel=False, max_n_neighborhood_graphs=max_n_neighborhood_graphs)
        self.feasibility_repair_estimator = FeasibilityRepairEstimator(
            perturbation_generator=perturbation_generator, 
            feasibility_estimator=self.feasibility_estimator, 
            n_iter=max_num_repair_steps, 
            beam_size=beam_size,
            min_number_of_violations=min_number_of_violations)

        regressor = ExtraTreesRegressor(n_estimators=n_estimators, n_jobs=-1)
        self.attribute_graph_graphicalizer = AttributeGraphGraphicalizer(regressor=regressor, decomposition_function=decomposition_function, nbits=nbits, attribute_key='vec', data_type=int, parallel=True)


    def fit(self, graphs, targets=None):
        start = time.time()
        if self.verbose: print('Training feasibility estimator, feasibility_repair_estimator and attribute_graph_graphicalizer for %d instances'%(len(graphs)))
        self.feasibility_estimator.fit(graphs)
        self.feasibility_repair_estimator.fit(graphs)
        self.attribute_graph_graphicalizer.fit(graphs)
        end = time.time()
        elapsed = end - start
        if self.verbose: print('  Time elapsed: %.1f s [%.1d m]\n'%(elapsed, elapsed/60))
        
        self.sampler.fit(graphs, targets)
        return self

    def sample(self, n_samples, return_all=False):
        start = time.time()
        sampled_graphs, targets = self.sampler.sample(n_samples)
        end = time.time()
        elapsed = end - start
        if self.verbose: print('  Time elapsed: %.1f s [%.1d m]\n'%(elapsed, elapsed/60))
        start = time.time()
        if self.verbose: print('Repairing, filtering for feasibility and predicting node and edge attributes %d instances'%(len(sampled_graphs)))
        repaired_sampled_graphs = self.feasibility_repair_estimator.repair(sampled_graphs)
        feasibile_graphs, feasibile_targets = self.feasibility_estimator.filter(repaired_sampled_graphs, targets)
        feasibile_graphs = self.attribute_graph_graphicalizer.transform(feasibile_graphs)
        end = time.time()
        elapsed = end - start
        if self.verbose: print('  Time elapsed: %.1f s [%.1d m]\n'%(elapsed, elapsed/60))
        if self.verbose: print('#generated graphs:%d  #feasible graphs:%d  [efficiency: %.2f perc]'%(len(sampled_graphs), len(feasibile_graphs), 100*len(feasibile_graphs)/len(sampled_graphs)))
        if return_all: return feasibile_graphs, feasibile_targets, repaired_sampled_graphs, targets, sampled_graphs, targets
        return feasibile_graphs, feasibile_targets

    def save(self, filename='generative_model.obj'):
        filehandler = open(filename, 'wb') 
        pickle.dump(self, filehandler)

    def load(self, filename='generative_model.obj'):
        filehandler = open(filename, 'rb') 
        self = pickle.load(filehandler)
        return self


