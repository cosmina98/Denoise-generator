# Import necessary modules for data splitting and model training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

# Import NumPy for numerical operations
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr
from pathos.multiprocessing import ProcessingPool as Pool  
from typing import Callable, Any, Dict, List, Tuple, Optional, Union

class ClassSpecificFeatureImportance(BaseEstimator, TransformerMixin):
    def __init__(self, model=None, binary_threshold=0):
        """
        Initialize the ClassSpecificFeatureImportance transformer.

        This transformer computes class-specific feature importance by leveraging 
        rank correlation (Spearman) between features and target classes. It also 
        incorporates feature frequency from a binary version of the feature matrix.

        Parameters:
        - model: A scikit-learn compatible classification model.
                 If None, defaults to RandomForestClassifier.
        - binary_threshold: Threshold to binarize the feature matrix X.
                            Features with values greater than this threshold are set to 1,
                            and others are set to 0. Useful for converting continuous
                            features to binary indicators.
        """
        # Initialize the classification model. If none is provided, use RandomForestClassifier.
        self.model = model if model is not None else RandomForestClassifier()
        # Set the threshold for binarizing the feature matrix.
        self.binary_threshold = binary_threshold

    def fit(self, X, y):
        """
        Fit the classification model to the data.

        This method trains the provided model using the feature matrix X and target vector y.
        It also stores the unique classes present in y for later use during transformation.

        Parameters:
        - X: Feature matrix of shape (n_samples, n_features). Type: numpy.ndarray or similar.
        - y: Target vector of shape (n_samples,). Type: numpy.ndarray or similar.

        Returns:
        - self: The fitted transformer instance.
        """
        # Fit the classification model to the data.
        self.model.fit(X, y)
        # Store the feature matrix for potential future use.
        self.X_ = X  # Shape: (n_samples, n_features)
        # Store the target vector for correlation computations.
        self.y_ = y  # Shape: (n_samples,)
        # Identify and store the unique classes present in the target vector.
        self.classes_ = np.unique(y)  # Shape: (n_classes,)
        return self

    def transform(self, X):
        """
        Transform the data by computing class-specific feature importance.

        This method performs the following steps:
        1. Computes global feature importance using the trained classification model.
        2. Calculates the Spearman rank correlation between each feature and each class.
        3. Identifies the most positively correlated class for each feature.
        4. Binarizes the feature matrix based on the specified threshold.
        5. Computes the frequency of each feature in the binary matrix.
        6. Updates feature importance by incorporating feature frequency.
        7. Combines correlation and importance to produce the final feature importance matrix.

        Parameters:
        - X: Feature matrix to transform, of shape (n_samples, n_features). Type: numpy.ndarray or similar.

        Returns:
        - correlation_feature_importance_matrix: A matrix representing the combined
                                                 correlation and feature importance
                                                 for each feature-class pair.
                                                 Shape: (n_features, n_classes)
                                                 Type: numpy.ndarray
        """
        # -------------------------------
        # Step 1: Compute Global Feature Importance
        # -------------------------------
        # Extract feature importance scores from the trained model.
        feature_importance = self.model.feature_importances_  # Shape: (n_features,)

        # -------------------------------
        # Step 2: Initialize Correlation Matrix
        # -------------------------------
        # Create a matrix to store Spearman correlation coefficients.
        # Shape: (n_features, n_classes)
        correlation_matrix = np.zeros((X.shape[1], len(self.classes_)))  # Type: numpy.ndarray

        # -------------------------------
        # Step 3: Compute Spearman Correlation
        # -------------------------------
        # Iterate over each feature.
        for feature_idx in range(X.shape[1]):
            # Iterate over each class.
            for class_idx, class_label in enumerate(self.classes_):
                # Create a binary target vector: 1 if the sample belongs to the current class, else 0.
                binary_target = (self.y_ == class_label).astype(int)  # Shape: (n_samples,)

                # Add a small amount of random noise to the binary target to handle potential ties
                # in the Spearman correlation calculation. This helps in obtaining a valid correlation.
                noise = np.random.normal(0, 1e-6, size=binary_target.shape)  # Shape: (n_samples,)

                # Compute Spearman rank correlation between the feature and the binary target.
                corr, _ = spearmanr(X[:, feature_idx] + noise, binary_target)  # corr: float

                # Store the correlation coefficient in the correlation matrix.
                correlation_matrix[feature_idx, class_idx] = corr  # Type: float

        # -------------------------------
        # Step 4: Compute Indicator Matrix
        # -------------------------------
        # Initialize an indicator matrix with the same shape as the correlation matrix.
        # This matrix will have a 1 in the position corresponding to the most positively
        # correlated class for each feature, and 0s elsewhere.
        indicator_matrix = np.zeros_like(correlation_matrix, dtype=int)  # Shape: (n_features, n_classes)

        # For each feature, identify the index of the class with the maximum correlation.
        max_corr_indices = np.argmax(correlation_matrix, axis=1)  # Shape: (n_features,)

        # Iterate over each feature to set the corresponding indicator.
        for feature_idx, class_idx in enumerate(max_corr_indices):
            # Mark the most positively correlated class for the current feature.
            indicator_matrix[feature_idx, class_idx] = 1  # Type: int

        # -------------------------------
        # Step 5: Reshape Feature Importance Vector
        # -------------------------------
        # Reshape the global feature importance vector to a column vector.
        # Shape: (n_features, 1)
        feature_importance_col_vec = feature_importance.reshape(-1, 1)  # Shape: (n_features, 1)

        # -------------------------------
        # Step 6: Binarize Feature Matrix
        # -------------------------------
        # Convert the feature matrix X to a binary matrix based on the binary_threshold.
        # Features with values greater than the threshold are set to 1; otherwise, 0.
        X_binary = (X > self.binary_threshold).astype(int)  # Shape: (n_samples, n_features)

        # -------------------------------
        # Step 7: Compute Frequency Vector
        # -------------------------------
        # Calculate the frequency of each feature in the binary matrix.
        # This is done by summing the binary values across all samples for each feature.
        # Shape: (n_features,)
        frequency_vector = X_binary.mean(axis=0)  # Shape: (n_features,)

        # -------------------------------
        # Step 8: Update Feature Importance with Frequency
        # -------------------------------
        # Incorporate feature frequency into the global feature importance scores.
        # This is achieved by multiplying each feature's importance by its frequency.
        feature_importance_col_vec = feature_importance_col_vec * frequency_vector.reshape(-1, 1)  # Shape: (n_features, 1)

        # -------------------------------
        # Step 9: Compute Feature Importance Matrix
        # -------------------------------
        # Multiply the indicator matrix with the updated feature importance vector.
        # This assigns the importance scores to the most correlated class for each feature.
        # Shape: (n_features, n_classes)
        feature_importance_matrix = indicator_matrix * feature_importance_col_vec  # Shape: (n_features, n_classes)

        # -------------------------------
        # Step 10: Compute Combined Correlation and Importance Matrix
        # -------------------------------
        # Element-wise multiply the feature importance matrix with the correlation matrix.
        # This combines both the correlation strength and the importance of each feature-class pair.
        # Shape: (n_features, n_classes)
        correlation_feature_importance_matrix = feature_importance_matrix * correlation_matrix  # Shape: (n_features, n_classes)

        return correlation_feature_importance_matrix  # Type: numpy.ndarray


class ImportanceQuotientGraphGraphicalizer(object):
    """
    A class to visualize node importance in graphs using feature importance derived from an ensemble classifier.

    This class utilizes graph decomposition and vectorization techniques to transform graphs into feature matrices.
    It then employs an ExtraTreesClassifier to compute feature importances, which are subsequently used to assign
    importance weights to nodes and edges within the graphs.

    Attributes:
        decomposition_function (callable): Function to decompose graphs into substructures.
        nbits (int): Number of bits for vectorization.
        importance_key (str): The key under which node and edge importance weights will be stored.
        feature_importance_n_iter (int): Number of iterations for training the classifier to ensure stable feature importance estimates.
        n_estimators (int): Number of trees in the ExtraTreesClassifier.
        quantile (float): Quantile threshold for filtering out low-importance features.
        parallel (bool): Whether to utilize parallel processing during vectorization and classifier training.
        feature_importance_vector (np.ndarray): Normalized feature importance scores after processing.
    """

    def __init__(self, node_vectorizer, importance_key='att', feature_importance_n_iter=5, n_estimators=100, 
                 quantile=0.75, parallel=True, normalize=True):
        """
        Initializes the ImportanceGraphGraphicalizer instance with specified parameters.

        Args:
            decomposition_function (callable, optional): Function to decompose graphs into substructures.
                Defaults to `neighborhood(min_size=1, max_size=2)`, which decomposes graphs into
                neighborhoods of size 1 to 2.
            nbits (int, optional): Number of bits for vectorization. Determines the dimensionality of the
                feature vectors. Defaults to 12.
            importance_key (str, optional): The key for storing node and edge importance in graph attributes.
                Defaults to 'att'.
            feature_importance_n_iter (int, optional): Number of iterations for classifier training to compute stable feature
                importance estimates. Defaults to 10.
            n_estimators (int, optional): Number of trees in the ExtraTreesClassifier. More trees can lead
                to more stable importance estimates but increase computational cost. Defaults to 100.
            quantile (float, optional): Quantile threshold for filtering out low-importance features. Features
                below this quantile are set to zero. Defaults to 0.5.
            parallel (bool, optional): Whether to utilize parallel processing during vectorization and classifier
                training. Defaults to True.
            normalize (bool, optional): Whether to normalize feature importance scores to [0,1] range.
                If True, scores are divided by their maximum value. Defaults to True.
        """
        self.node_vectorizer = node_vectorizer  # Expected type: object with a transform method
        self.importance_key = importance_key  # Type: str
        self.feature_importance_n_iter = feature_importance_n_iter  # Type: int
        self.n_estimators = n_estimators  # Type: int
        self.quantile = quantile  # Type: float
        self.parallel = parallel  # Type: bool
        self.normalize = normalize  # Type: bool
        self.feature_importance_vector = None  # Will be a 2D array [n_features, n_classes], Type: numpy.ndarray
        self.classes_ = None  # Store class labels, Type: numpy.ndarray

    def _vectorize_graphs(self, graphs):
        """Convert graphs to feature matrix.
        
        Args:
            graphs (list): List of graph objects to vectorize.

        Returns:
            numpy.ndarray: Feature matrix of shape (n_graphs, n_features).
        """
        node_embeddings_list = self.node_vectorizer.transform(graphs)  # Type: list of numpy.ndarray
        return np.vstack([node_embeddings.sum(axis=0) for node_embeddings in node_embeddings_list])  # Shape: (n_graphs, n_features)
    
    def _compute_feature_importance(self, X, targets):
        """
        Compute feature importance scores using parallel computation with the Pathos library.
        """
        self.classes_ = np.unique(targets)  # Identify unique classes
        n_iterations = self.feature_importance_n_iter  # Total number of iterations

        def compute_importance(it):
            """
            Helper function to compute feature importance for a single iteration.
            """
            # Split the data into training and (unused) test sets
            train_X, _, train_targets, _ = train_test_split(
                X, targets, train_size=0.7, random_state=it + 1
            )

            # Initialize the ExtraTreesClassifier with specified parameters
            base_estimator = ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                n_jobs=1,  # Use a single core per process to avoid oversubscription
                random_state=it + 1
            )

            # Initialize the ClassSpecificFeatureImportance transformer with the base estimator
            class_specific = ClassSpecificFeatureImportance(model=base_estimator)

            # Fit the transformer and compute class-specific feature importances
            importance = class_specific.fit(train_X, train_targets).transform(train_X)

            # Verify the shape
            expected_shape = (X.shape[1], len(self.classes_))
            if importance.shape != expected_shape:
                raise ValueError(f"Iteration {it}: Expected shape {expected_shape}, got {importance.shape}")

            return importance  # Shape: (n_features, n_classes)

        if self.parallel:
            # Initialize the processing pool with the desired number of processes
            with Pool() as pool:
                # Distribute the compute_importance function across iterations
                feature_importances_per_iter = pool.map(compute_importance, range(n_iterations))
        else:
            # Sequential computation if parallelization is not desired
            feature_importances_per_iter = [
                compute_importance(it) for it in range(n_iterations)
            ]

        # Ensure all importance matrices have the same shape before stacking
        for it, importance in enumerate(feature_importances_per_iter):
            if importance.shape != (X.shape[1], len(self.classes_)):
                raise ValueError(f"After computation: Iteration {it} has shape {importance.shape}, expected {(X.shape[1], len(self.classes_))}")

        # Stack the list of importance matrices into a single 3D NumPy array
        # Shape: (n_iterations, n_features, n_classes)
        return np.stack(feature_importances_per_iter)
    
    def fit(self, graphs, targets):
        """
        Fit model by computing feature importances from graphs and targets.

        Args:
            graphs (list): List of graph objects to analyze. Type: list
            targets (numpy.ndarray): Target labels for supervised training. Shape: (n_graphs,). Type: numpy.ndarray

        Returns:
            self: For method chaining

        Raises:
            ValueError: If inputs are invalid or empty
        """
        if not graphs or not targets:
            raise ValueError("Graphs and targets cannot be empty")
        if len(graphs) != len(targets):
            raise ValueError("Number of graphs must match number of targets")

        # Validate graph attributes
        self._validate_graphs(graphs)
        # Convert graphs to feature matrix
        X = self._vectorize_graphs(graphs)  # Shape: (n_graphs, n_features)
        # Compute feature importances across multiple iterations
        importance_matrix = self._compute_feature_importance(X, targets)  # Shape: (n_iterations, n_features, n_classes)
        # Calculate stable importance scores per class
        mean_imp = np.mean(importance_matrix, axis=0)  # Shape: (n_features, n_classes)
        std_imp = np.std(importance_matrix, axis=0)  # Shape: (n_features, n_classes)
        adjusted_imp = np.maximum(mean_imp - std_imp, 0)  # Shape: (n_features, n_classes)

        # Apply quantile threshold per class
        for i in range(adjusted_imp.shape[1]):
            threshold = np.quantile(adjusted_imp[:, i], self.quantile)  # Scalar
            adjusted_imp[adjusted_imp[:, i] < threshold, i] = 0  # Zero out low importance features
        # Normalize if needed (per class)
        if self.normalize:
            max_imp = np.max(adjusted_imp, axis=0)  # Shape: (n_classes,)
            max_imp[max_imp == 0] = 1  # Avoid division by zero
            adjusted_imp = adjusted_imp / max_imp  # Broadcasting division, Shape: (n_features, n_classes)

        self.feature_importance_vector = adjusted_imp  # Shape: (n_features, n_classes)
        return self

    def _validate_graphs(self, graphs):
        """
        Validate that all graphs have required node and edge attributes
        and sequential node IDs starting from 0.

        Args:
            graphs (list): List of networkx graphs to validate. Type: list

        Raises:
            ValueError: If any graph is missing required attributes or has invalid node IDs
        """
        for i, g in enumerate(graphs):
            # Check sequential node IDs
            node_ids = sorted(list(g.nodes()))  # List of node IDs, sorted
            expected_ids = list(range(len(node_ids)))  # Expected node IDs: [0, 1, 2, ..., n_nodes-1]
            if node_ids != expected_ids:
                raise ValueError(f"Graph {i} nodes must be sequential integers starting from 0")

            # Check node labels
            for node_id, node_data in g.nodes(data=True):
                if 'label' not in node_data:
                    raise ValueError(f"Graph {i}, node {node_id} is missing 'label' attribute")

            # Check edge labels 
            for u, v, edge_data in g.edges(data=True):
                if 'label' not in edge_data:
                    raise ValueError(f"Graph {i}, edge ({u}, {v}) is missing 'label' attribute")

    def _vectorize_nodes(self, graphs):
        """Vectorize nodes of input graphs.

        Args:
            graphs (list): List of graph objects to vectorize. Type: list

        Returns:
            list of numpy.ndarray: List where each element corresponds to the node feature matrix of a graph.
        """
        return self.node_vectorizer.transform(graphs)  # Type: list of numpy.ndarray

    def _compute_node_weights(self, node_feature_mtx):
        """Compute class-specific node importance weights.

        Args:
            node_feature_mtx (numpy.ndarray): Feature matrix for nodes. Shape: (n_nodes, n_features)

        Returns:
            numpy.ndarray: Node importance weights. Shape: (n_nodes, n_classes)
        """
        # Compute weights for each class
        weights = np.dot(node_feature_mtx, self.feature_importance_vector)  # Shape: (n_nodes, n_classes)

        # Normalize per class if needed
        if self.normalize:
            max_weights = np.max(weights, axis=0)  # Shape: (n_classes,)
            max_weights[max_weights == 0] = 1  # Avoid division by zero
            weights = weights / max_weights  # Broadcasting division, Shape: (n_nodes, n_classes)

        return weights  # Shape: (n_nodes, n_classes)

    def _compute_edge_weights(self, graph, node_weights):
        """Compute edge weights based on connected nodes.

        Args:
            graph (networkx.Graph): A single graph object.
            node_weights (numpy.ndarray): Node importance weights. Shape: (n_nodes, n_classes)

        Returns:
            dict: Mapping from edge tuples to importance weights.
                  Each key is a tuple (u, v), and each value is a numpy.ndarray of shape (n_classes,).
        """
        edge_weights = {}  # Initialize dictionary to store edge weights
        for u, v in graph.edges():
            weight_u = node_weights[u]  # Shape: (n_classes,)
            weight_v = node_weights[v]  # Shape: (n_classes,)
            edge_weights[(u, v)] = weight_u * weight_v  # Element-wise multiplication, Shape: (n_classes,)
        return edge_weights  # Type: dict

    def _create_transformed_graph(self, graph, node_weights, edge_weights):
        """Create new graph with importance weights.

        Args:
            graph (networkx.Graph): Original graph.
            node_weights (numpy.ndarray): Node importance weights. Shape: (n_nodes, n_classes)
            edge_weights (dict): Edge importance weights. Shape: dict with keys as edge tuples and values as numpy.ndarray of shape (n_classes,)

        Returns:
            networkx.Graph: Transformed graph with added importance attributes.
        """
        out_graph = graph.copy()  # Create a copy of the graph to modify

        # Add node weights
        for node_id, weight in zip(graph.nodes(), node_weights):
            out_graph.nodes[node_id][self.importance_key] = np.max(weight)  # Scalar, Type: float
            out_graph.nodes[node_id]['class_' + self.importance_key] = weight  # Array, Shape: (n_classes,)

        # Add edge weights
        for (u, v), weight in edge_weights.items():
            out_graph.edges[u, v][self.importance_key] = np.max(weight)  # Scalar, Type: float
            out_graph.edges[u, v]['class_' + self.importance_key] = weight  # Array, Shape: (n_classes,)

        return out_graph  # Type: networkx.Graph

    def compute_weights(self, graphs, node_feature_matrices) -> list:
        """
        Compute node and edge importance weights for each input graph.
        
        This method first vectorizes all nodes across all graphs once and then computes
        the weights either in parallel or sequentially based on the `self.parallel` flag.
        
        Args:
            node_feature_matrices (list): List of node feature matrices for each graph.
        
        Returns:
            list: List of tuples containing node_weights and edge_weights for each graph.
        
        Raises:
            ValueError: If inputs are invalid or empty.
        """
        weights_list = []
        for graph, node_features in zip(graphs, node_feature_matrices):
            node_weights = self._compute_node_weights(node_features)  # Shape: (n_nodes, n_classes)
            edge_weights = self._compute_edge_weights(graph, node_weights)  # Dict of edge weights
            weights_list.append((node_weights, edge_weights))
        return weights_list

    def transform(self, graphs):
        """
        Transforms the input graphs by assigning normalized importance weights to each node and edge based on feature importances.

        This method performs the following steps for each graph:
            1. Vectorizes the nodes of the graph using either parallel or sequential vectorization.
            2. Computes weights for each node by combining node features with the precomputed feature importance vector.
            3. Normalizes the node weights to scale them between 0 and 1.
            4. Assigns the normalized weights to the nodes and computes edge weights as the product of connected nodes' importance scores.

        Args:
            graphs (list): A list of graph objects to be transformed with node and edge importance weights. Type: list

        Returns:
            out_graphs (list): A list of transformed graph objects with updated node and edge importance attributes. Type: list of networkx.Graph
        """
        if not graphs:
            raise ValueError("Input graphs list cannot be empty")
        if self.feature_importance_vector is None:
            raise ValueError("Must call fit() before transform()")

        # Validate graph attributes
        self._validate_graphs(graphs)

        # Compute weights for all graphs
        node_feature_matrices = self._vectorize_nodes(graphs)  # Shape: list of (n_nodes, n_features)

        weights_list = self.compute_weights(graphs, node_feature_matrices) # Type: list of tuples

        transformed_graphs = []  # Initialize list to store transformed graphs
        for graph, (node_weights, edge_weights) in zip(graphs, weights_list):
            # Create transformed graph with added importance weights
            transformed_graph = self._create_transformed_graph(graph, node_weights, edge_weights)  # Type: networkx.Graph
            transformed_graphs.append(transformed_graph)  # Append to the list

        return transformed_graphs  # Type: list of networkx.Graph

    def extract(self, graphs):
        """
        Extract and sort subgraphs by their importance scores for each class.

        This method performs the following steps:
        1. Decomposes input graphs into subgraphs using the specified decomposition function.
        2. Creates a mapping between hash IDs and subgraphs.
        3. Sorts subgraphs by importance scores for each class.
        4. Returns only subgraphs with positive importance scores.

        Args:
            graphs (list): List of input graphs to extract subgraphs from. Type: list

        Returns:
            tuple: A pair containing:
                - dict: Mapping from class labels to lists of subgraphs sorted by importance. Type: dict
                - dict: Mapping from class labels to lists of importance scores corresponding to the subgraphs. Type: dict

        Example:
            >>> graphs = [graph1, graph2, graph3]
            >>> sorted_subgraphs, importances = extract(graphs)
            >>> print(sorted_subgraphs['class1'][0])  # Most important subgraph for class1
            >>> print(importances['class1'][0])       # Its importance score
        """
        # Step 1: Decompose input graphs into subgraphs
        # The 'decomposition' function processes each graph using the provided decomposition function,
        # number of bits, and parallelization settings.
        quotient_graphs_list = self.node_vectorizer.extract(graphs)  # Type: list of quotient_graph objects

        # Initialize an empty dictionary to map each hash ID to its corresponding subgraph
        feature_subgraphs_dict = {}  # Type: dict

        # Step 2: Create a mapping between hash IDs and subgraphs
        for quotient_graph in quotient_graphs_list:
            # Iterate over each node in the decomposed graph
            for u in quotient_graph.quotient_graph.nodes():
                # Extract the subgraph associated with the current node
                subgraph = quotient_graph.subgraph(u)  # Type: networkx.Graph
                # Extract the hash ID (label) associated with the current node
                hash_id = quotient_graph.get_quotient_node_label(u)  # Type: int or str
                # Map the hash ID to its corresponding subgraph
                feature_subgraphs_dict[hash_id] = subgraph  # Key: hash_id, Value: networkx.Graph

        # Initialize dictionaries to store sorted subgraphs and their importance scores for each class
        sorted_subgraphs = {}  # Type: dict, keys: class labels, values: list of networkx.Graph
        sorted_subgraph_importances = {}  # Type: dict, keys: class labels, values: list of float

        # Step 3: Sort subgraphs by importance scores for each class
        for class_idx, class_label in enumerate(self.classes_):
            # Retrieve the importance scores for the current class across all features
            # Assuming 'self.feature_importance_vector' is a 2D NumPy array where rows correspond to features
            # and columns correspond to classes
            class_importance = self.feature_importance_vector[:, class_idx]  # Shape: (n_features,)

            # Get the indices that would sort the importance scores in descending order
            idxs = np.argsort(-class_importance)  # Shape: (n_features,), Type: numpy.ndarray

            # Initialize lists to hold sorted subgraphs and their corresponding importance scores
            sorted_subgraphs[class_label] = []  # Type: list
            sorted_subgraph_importances[class_label] = []  # Type: list

            # Iterate over the sorted indices
            for idx in idxs:
                # Check if the current index exists in the feature_subgraphs_dict and has a positive importance score
                if idx in feature_subgraphs_dict and class_importance[idx] > 0:
                    # Append the corresponding subgraph to the sorted_subgraphs dictionary
                    sorted_subgraphs[class_label].append(feature_subgraphs_dict[idx])  # Append networkx.Graph
                    # Append the corresponding importance score to the sorted_subgraph_importances dictionary
                    sorted_subgraph_importances[class_label].append(class_importance[idx])  # Append float

        # Step 4: Return only subgraphs with positive importance scores
        # 'sorted_subgraphs' maps each class to a list of its subgraphs sorted by importance
        # 'sorted_subgraph_importances' maps each class to a list of corresponding importance scores
        return sorted_subgraphs, sorted_subgraph_importances  # Type: tuple of dicts
