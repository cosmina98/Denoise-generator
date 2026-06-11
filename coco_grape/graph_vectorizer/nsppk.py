import math
from collections import defaultdict, Counter
import copy
import hashlib
import networkx as nx
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import multiprocessing_on_dill as mp

# ------------------------
# Hash and Utility Functions
# ------------------------

def _hash(value):
    value_str = str(value)
    value_bytes = value_str.encode()
    sha256_hash = hashlib.sha256(value_bytes).hexdigest()
    hash_int = int(sha256_hash, 16) & (2**30 - 1)
    return hash_int

def masked_hash_value(value, bitmask=(2**20 - 1)):
    return _hash(value) & bitmask

def hash_value(value, nbits=10):
    max_index = 2 ** nbits
    h = masked_hash_value(value, max_index - 3)
    h += 2  # Offset by 2 to ensure hash is never 0 or 1
    return h

def hash_sequence(iterable):
    return _hash(tuple(iterable))

def hash_set(iterable):
    sorted_iterable = sorted(iterable)
    tuple_representation = tuple(sorted_iterable)
    return _hash(tuple_representation)

def node_hash(node_idx, graph):
    uh = _hash(graph.nodes[node_idx]['label'])
    edges_h = [
        _hash((_hash(graph.nodes[v]['label']), _hash(graph.edges[node_idx, v]['label'])))
        for v in graph.neighbors(node_idx)
    ]
    nh = hash_set(edges_h)
    ext_node_h = _hash((uh, nh))
    return ext_node_h

def invert_dict(mydict):
    reversed_dict = defaultdict(list)
    for key, value in mydict.items():
        reversed_dict[value].append(key)
    return reversed_dict

def rooted_graph_hashes(node_idx, graph, radius=1):
    node_idxs_to_dist_dict = nx.single_source_shortest_path_length(graph, node_idx, cutoff=radius)
    dist_to_node_idxs_dict = invert_dict(node_idxs_to_dist_dict)
    iso_distance_codes_list = [
        hash_set([graph.nodes[curr_node_idx]['node_label_hash'] if dist == 0 
                  else graph.nodes[curr_node_idx]['node_hash']
                  for curr_node_idx in node_idxs])
        for dist, node_idxs in sorted(dist_to_node_idxs_dict.items())
    ]
    h_list = [hash_sequence(iso_distance_codes_list[:i])
              for i in range(1, len(iso_distance_codes_list) + 1)]
    return h_list

def items_to_sparse_histogram(items, nbits):
    histogram_dict = Counter(items)
    # Create a LIL matrix first
    mat = sp.sparse.lil_matrix((1, 2**nbits), dtype=int)
    for col, value in histogram_dict.items():
        mat[0, col] = value
    return mat

def weighted_sparse_histogram(weighted_dict, nbits):
    # Create a LIL matrix first
    mat = sp.sparse.lil_matrix((1, 2**nbits), dtype=float)
    for col, value in weighted_dict.items():
        mat[0, col] = value
    return mat

def edge_triplet_hash(u, v, graph):
    return hash_set([_hash(graph.nodes[u]['label']), _hash(graph.edges[u, v]['label']), _hash(graph.nodes[v]['label'])])

def precompute_edge_triplet_hashes(graph):
    for u, v in graph.edges():
        graph.edges[u, v]['triplet_hash'] = edge_triplet_hash(u, v, graph)

def gaussian_weight(dist, sigma):
    if sigma is None:
        return 1.0
    return np.exp(- (dist ** 2) / (sigma ** 2))

# ------------------------
# Accumulator Classes
# ------------------------

class ListAccumulator:
    def __init__(self):
        self.data = []
    def add(self, code, weight):
        # For unweighted case weight is always 1.0
        self.data.append(code)
    def get(self):
        return self.data

class DictAccumulator:
    def __init__(self):
        self.data = defaultdict(float)
    def add(self, code, weight):
        self.data[code] += weight
    def get(self):
        return self.data

def _process_node_features(node_idx, graph, distance, connector, nbits, sigma, accumulator, use_edges_as_features=True):
    # Weighted: inline computation of weight using the Gaussian function.
    node_idxs_to_dist_dict = nx.single_source_shortest_path_length(graph, node_idx, cutoff=distance)
    dist_to_node_idxs_dict = invert_dict(node_idxs_to_dist_dict)

    if use_edges_as_features:
        for target_node, dist in node_idxs_to_dist_dict.items():
            w = gaussian_weight(dist, sigma)
            for neighbor in graph.neighbors(target_node):
                triplet_hash = graph.edges[target_node, neighbor].get('triplet_hash', None)
                if triplet_hash is not None:
                    distance_triplet_hash = hash_value(hash_sequence([dist, triplet_hash]), nbits=nbits)
                    accumulator.add(distance_triplet_hash, w)

    for code_i in graph.nodes[node_idx]['rooted_graph_hash']:
        for dist, node_idxs in sorted(dist_to_node_idxs_dict.items()):
            w = gaussian_weight(dist, sigma)
            for curr_node_idx in node_idxs:
                if connector > 0:
                    union_of_shortest_paths = set()
                    shortest_paths = list(nx.all_shortest_paths(graph, source=node_idx, target=curr_node_idx))
                    for path in shortest_paths:
                        union_of_shortest_paths.update(path)
                for connect in range(connector + 1):
                    for code_j in graph.nodes[curr_node_idx]['rooted_graph_hash']:
                        if connect == 0:
                            paired_code = hash_sequence([code_i, dist, code_j])
                        else:
                            union_of_shortest_paths_code = hash_set([
                                    hash_sequence([node_idxs_to_dist_dict[node], graph.nodes[node]['rooted_graph_hash'][connect - 1]])
                                    for node in union_of_shortest_paths
                                ])
                            paired_code = hash_sequence([code_i, dist, code_j, union_of_shortest_paths_code])
                        paired_code = hash_value(paired_code, nbits=nbits)
                        accumulator.add(paired_code, w)
    
def get_structural_node_vectors(original_graph, radius, distance, connector, nbits, degree_threshold=None, sigma=None, use_edges_as_features=True):
    """
    Generates a feature vector for each node in the graph.
    Uses an unweighted (faster) version if sigma is None, in which the weight is inlined as 1.0.
    If sigma is provided, the Gaussian weighted variant is used.
    """
    graph = original_graph.copy()
    
    # Precompute node and edge hashes.
    for node_idx in graph.nodes():
        graph.nodes[node_idx]['node_label_hash'] = _hash(graph.nodes[node_idx]['label'])
        graph.nodes[node_idx]['node_hash'] = node_hash(node_idx, graph)
    precompute_edge_triplet_hashes(graph)
    
    cutoff = max(radius, connector)
    for node_idx in graph.nodes():
        graph.nodes[node_idx]['rooted_graph_hash'] = np.zeros(cutoff + 1, dtype=int)
        degree = graph.degree[node_idx]
        effective_cutoff = 0 if degree_threshold is not None and degree > degree_threshold else cutoff
        for r, radius_r_rooted_graph_hash in enumerate(rooted_graph_hashes(node_idx, graph, radius=effective_cutoff)):
            graph.nodes[node_idx]['rooted_graph_hash'][r] = radius_r_rooted_graph_hash
            
    node_vectors = []
    if sigma is None:
        # Unweighted branch: use ListAccumulator and inline constant weight.
        accumulator_class = ListAccumulator
        convert_func = items_to_sparse_histogram
        for node_idx in graph.nodes():
            accumulator = accumulator_class()
            _process_node_features(node_idx, graph, distance, connector, nbits, sigma, accumulator, use_edges_as_features)
            node_vector = convert_func(accumulator.get(), nbits)
            # Add special features (degree info)
            node_vector[0, 0] = 1
            node_vector[0, 1] = graph.degree[node_idx]
            node_vector = node_vector.tocsr()
            node_vectors.append(node_vector)
    else:
        # Weighted branch: use DictAccumulator.
        accumulator_class = DictAccumulator
        convert_func = weighted_sparse_histogram
        for node_idx in graph.nodes():
            accumulator = accumulator_class()
            _process_node_features(node_idx, graph, distance, connector, nbits, sigma, accumulator, use_edges_as_features)
            node_vector = convert_func(accumulator.get(), nbits)
            # Add special features (degree info)
            node_vector[0,0] = 1
            node_vector[0,1] = graph.degree[node_idx]
            node_vector = node_vector.tocsr()
            node_vectors.append(node_vector)
    
    return sp.sparse.vstack(node_vectors)

def get_node_attribute_matrix(original_graph, node_attribute_key):
    """
    Extracts node attributes from a graph and stacks them into a matrix.

    Args:
        original_graph (networkx.Graph): The input graph.
        node_attribute_key (str): The key corresponding to the node attribute to extract.

    Returns:
        numpy.ndarray: The matrix of node attributes.
    """
    return np.vstack([
        original_graph.nodes[node_idx][node_attribute_key]
        for node_idx in original_graph.nodes()
    ])

def get_edge_attribute_mtx(original_graph, edge_attribute_key):
    """
    Extracts edge attributes from a graph and stacks them into a matrix.

    Args:
        original_graph (networkx.Graph): The input graph.
        edge_attribute_key (str): The key corresponding to the edge attribute to extract.

    Returns:
        numpy.ndarray: The matrix of edge attributes.
    """
    return np.vstack([
        original_graph.edges[edge][edge_attribute_key]
        for edge in original_graph.edges()
    ])

def reweight_node_vectors_mtx(node_vectors_mtx, original_graph, weight_key):
    """
    Reweights node feature vectors based on node weights.

    Args:
        node_vectors_mtx (scipy.sparse.csr_matrix): The matrix of node feature vectors.
        original_graph (networkx.Graph): The input graph.
        weight_key (str): The key corresponding to the node weight to apply.

    Returns:
        scipy.sparse.csr_matrix: The reweighted node feature matrix.
    """
    weight_vector = np.array([
        original_graph.nodes[node_idx][weight_key]
        for node_idx in original_graph.nodes()
    ])
    node_vectors_mtx = (node_vectors_mtx.todense().A.T * weight_vector).T
    return csr_matrix(node_vectors_mtx)


def reweight_edge_vectors_mtx(edge_vectors_mtx, original_graph, weight_key):
    """
    Reweights edge feature vectors based on edge weights.

    Args:
        edge_vectors_mtx (scipy.sparse.csr_matrix): The matrix of edge feature vectors.
        original_graph (networkx.Graph): The input graph.
        weight_key (str): The key corresponding to the edge weight to apply.

    Returns:
        scipy.sparse.csr_matrix: The reweighted edge feature matrix.
    """
    weight_vector = np.array([
        original_graph.edges[edge][weight_key]
        for edge in original_graph.edges()
    ])
    edge_vectors_mtx = (edge_vectors_mtx.todense().A.T * weight_vector).T
    return csr_matrix(edge_vectors_mtx)


def get_node_vectors(original_graph, radius, distance, connector, nbits, weight_key=None, node_attribute_key=None, degree_threshold=None, add_structural_node_information=True, sigma=None, use_edges_as_features=True):
    """
    Generates a feature vector for a single graph based on node and subgraph hashes.

    Args:
        original_graph (networkx.Graph): The input graph.
        radius (int): The radius for rooted graph hashing.
        distance (int): The distance parameter for paired hashing.
        connector (int): Connector thickness.
        nbits (int): Number of bits for hashing.
        weight_key (str, optional): Node weight key for reweighting features.
        node_attribute_key (str, optional): Node attribute key for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        scipy.sparse.csr_matrix: The sparse feature matrix representing the graph.
    """
    node_vectors_mtx = get_structural_node_vectors(
        original_graph, radius, distance, connector, nbits, degree_threshold, sigma, use_edges_as_features
    )
    if weight_key is not None:
        node_vectors_mtx = reweight_node_vectors_mtx(node_vectors_mtx, original_graph, weight_key)
    if node_attribute_key is not None:
        attribute_mtx = get_node_attribute_matrix(original_graph, node_attribute_key)
        feature_node_vectors_mtx = node_vectors_mtx.todense().A
        feature_node_vectors_mtx = attribute_mtx.T.dot(feature_node_vectors_mtx).dot(feature_node_vectors_mtx.T).T
        feature_node_vectors_mtx = np.power(np.abs(feature_node_vectors_mtx),1/3) #reduce magnitude of entries to avoid kernel diagonal dominance issues
        feature_node_vectors_mtx = csr_matrix(feature_node_vectors_mtx)
        if add_structural_node_information: node_vectors_mtx = sp.sparse.hstack([csr_matrix(attribute_mtx), feature_node_vectors_mtx, node_vectors_mtx])
        else: node_vectors_mtx = sp.sparse.hstack([csr_matrix(attribute_mtx), feature_node_vectors_mtx])
    return node_vectors_mtx


def get_node_graph_vectors(original_graph, radius, distance, connector, nbits, weight_key=None, node_attribute_key=None, degree_threshold=None, sigma=None, use_edges_as_features=True):
    """
    Generates a feature vector for a single graph based on node and subgraph hashes.

    Args:
        original_graph (networkx.Graph): The input graph.
        radius (int): The radius for rooted graph hashing.
        distance (int): The distance parameter for paired hashing.
        connector (int): Connector thickness.
        nbits (int): Number of bits for hashing.
        weight_key (str, optional): Node weight key for reweighting features.
        node_attribute_key (str, optional): Node attribute key for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        scipy.sparse.csr_matrix: The sparse feature vector representing the graph.
    """
    node_vectors_mtx = get_structural_node_vectors(original_graph, radius, distance, connector, nbits, degree_threshold, sigma, use_edges_as_features)
    node_vectors_mtx = node_vectors_mtx.todense().A
    attribute_node_vectors_mtx = get_node_vectors(original_graph, radius, distance, connector, nbits, weight_key, node_attribute_key, degree_threshold, add_structural_node_information=False, sigma=sigma, use_edges_as_features=use_edges_as_features)
    vector_ = attribute_node_vectors_mtx.T.dot(node_vectors_mtx)
    vector = vector_.reshape(1, -1)
    vector = np.power(np.abs(vector), 1/2)
    vector = csr_matrix(vector)
    return csr_matrix(vector)


def get_graph_vector(original_graph, radius, distance, connector, nbits, weight_key=None, node_attribute_key=None, edge_attribute_key=None, degree_threshold=None, sigma=None, use_edges_as_features=True):
    """
    Generates a feature vector for a single graph based on node and subgraph hashes.

    Args:
        original_graph (networkx.Graph): The input graph.
        radius (int): The radius for rooted graph hashing.
        distance (int): The distance parameter for paired hashing.
        connector (int): Connector thickness.
        nbits (int): Number of bits for hashing.
        weight_key (str, optional): Node weight key for reweighting features.
        node_attribute_key (str, optional): Node attribute key for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        scipy.sparse.csr_matrix: The sparse feature vector representing the graph.
    """
    node_vectors_mtx = get_structural_node_vectors(
        original_graph, radius, distance, connector, nbits, degree_threshold, sigma, use_edges_as_features
    )
    if weight_key is not None:
        node_vectors_mtx = reweight_node_vectors_mtx(node_vectors_mtx, original_graph, weight_key)
    if node_attribute_key is None:
        graph_vector = node_vectors_mtx.sum(axis=0)    
    else:
        attribute_mtx = get_node_attribute_matrix(original_graph, node_attribute_key)
        node_vectors_mtx = node_vectors_mtx.todense().A
        vector_ = attribute_mtx.T.dot(node_vectors_mtx)
        graph_vector = vector_.reshape(1, -1)
    graph_vector = csr_matrix(graph_vector)
    return graph_vector


def split_into_chunks(lst, n):
    """
    Splits a list into 'n' nearly equal chunks.

    Args:
        lst (list): The list to split.
        n (int): The number of chunks.

    Returns:
        list of lists: A list containing 'n' sublists.
    """
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def paired_graphs_vector_encoder(graphs, radius, distance, connector, nbits, parallel=True, weight_key=None, node_attribute_key=None, edge_attribute_key=None, degree_threshold=None, sigma=None, use_edges_as_features=True):
    """
    Encodes a list of graphs into a sparse matrix of feature vectors.

    Args:
        graphs (list of networkx.Graph): The list of graphs to encode.
        radius (int): The radius for rooted graph hashing.
        distance (int): The distance parameter for paired hashing.
        connector (int): Connector thickness.
        nbits (int): Number of bits for hashing.
        parallel (bool, optional): Whether to encode graphs in parallel. Defaults to True.
        weight_key (str, optional): Node weight key for reweighting features.
        node_attribute_key (str, optional): Node attribute key for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        scipy.sparse.csr_matrix: The sparse matrix where each row is a graph's feature vector.
    """
    n_cpus = mp.cpu_count()

    if parallel and len(graphs) > n_cpus:
        def process_subset(subset):
            return [
                get_graph_vector(graph, radius, distance, connector, nbits, weight_key, node_attribute_key, edge_attribute_key, degree_threshold, sigma, use_edges_as_features)
                for graph in subset
            ]

        subsets = split_into_chunks(graphs, n_cpus)
        with mp.Pool(n_cpus) as pool:
            graph_vectors_subsets = pool.map(process_subset, subsets)
        graph_vectors = [vec for subset in graph_vectors_subsets for vec in subset]
    else:
        graph_vectors = [
            get_graph_vector(graph, radius, distance, connector, nbits, weight_key, node_attribute_key, edge_attribute_key, degree_threshold, sigma, use_edges_as_features)
            for graph in graphs
        ]

    return sp.sparse.vstack(graph_vectors)


def paired_node_graphs_vector_encoder(graphs, radius, distance, connector, nbits, parallel=True, weight_key=None, node_attribute_key=None, edge_attribute_key=None, degree_threshold=None, sigma=None, use_edges_as_features=True):
    """
    Encodes a list of graphs into a sparse matrix of feature vectors.

    Args:
        graphs (list of networkx.Graph): The list of graphs to encode.
        radius (int): The radius for rooted graph hashing.
        distance (int): The distance parameter for paired hashing.
        connector (int): Connector thickness.
        nbits (int): Number of bits for hashing.
        parallel (bool, optional): Whether to encode graphs in parallel. Defaults to True.
        weight_key (str, optional): Node weight key for reweighting features.
        node_attribute_key (str, optional): Node attribute key for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        scipy.sparse.csr_matrix: The sparse matrix where each row is a graph's feature vector.
    """
    n_cpus = mp.cpu_count()

    if parallel and len(graphs) > n_cpus:
        def process_subset(subset):
            return [
                get_node_graph_vectors(graph, radius, distance, connector, nbits, weight_key, node_attribute_key, degree_threshold, sigma=sigma, use_edges_as_features=use_edges_as_features)
                for graph in subset
            ]

        subsets = split_into_chunks(graphs, n_cpus)
        with mp.Pool(n_cpus) as pool:
            graph_vectors_subsets = pool.map(process_subset, subsets)
        graph_vectors = [vec for subset in graph_vectors_subsets for vec in subset]
    else:
        graph_vectors = [
            get_node_graph_vectors(graph, radius, distance, connector, nbits, weight_key, node_attribute_key, degree_threshold, sigma=sigma, use_edges_as_features=use_edges_as_features)
            for graph in graphs
        ]

    return sp.sparse.vstack(graph_vectors)


def paired_node_vector_encoder(graphs, radius, distance, connector, nbits, parallel=True, weight_key=None, node_attribute_key=None, edge_attribute_key=None, degree_threshold=None, sigma=None, use_edges_as_features=True):
    """
    Encodes a list of graphs into node feature vectors.

    Args:
        graphs (list of networkx.Graph): The list of graphs to encode.
        radius (int): The radius for rooted graph hashing.
        distance (int): The distance parameter for paired hashing.
        connector (int): Connector thickness.
        nbits (int): Number of bits for hashing.
        parallel (bool, optional): Whether to encode graphs in parallel. Defaults to True.
        weight_key (str, optional): Node weight key for reweighting features.
        node_attribute_key (str, optional): Node attribute key for additional features.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.

    Returns:
        list of scipy.sparse.csr_matrix: A list where each element is a matrix of node vectors for a graph.
    """
    n_cpus = mp.cpu_count()

    if parallel and len(graphs) > n_cpus:
        def process_subset(subset):
            return [
                get_node_vectors(graph, radius, distance, connector, nbits, weight_key, node_attribute_key, degree_threshold, sigma=sigma, use_edges_as_features=use_edges_as_features)
                for graph in subset
            ]

        subsets = split_into_chunks(graphs, n_cpus)
        with mp.Pool(n_cpus) as pool:
            graph_node_vectors_subsets = pool.map(process_subset, subsets)
        graph_node_vectors = [vec for subset in graph_node_vectors_subsets for vec in subset]
    else:
        graph_node_vectors = [
            get_node_vectors(graph, radius, distance, connector, nbits, weight_key, node_attribute_key, degree_threshold, sigma=sigma, use_edges_as_features=use_edges_as_features)
            for graph in graphs
        ]

    return graph_node_vectors


class BaseNSPPK(BaseEstimator, TransformerMixin):
    """
    BaseNSPPK is an abstract base class for encoding graphs into feature vectors.

    This class is initialized with generic components for embedding, clustering, and classification,
    allowing for flexibility and extensibility. Subclasses can specialize these components to 
    implement specific encoding strategies.

    Parameters:
        embedder (transformer, optional): A transformer for attribute embedding (e.g., TruncatedSVD).
        clustering_predictor (estimator, optional): A clustering model for attribute clustering (e.g., KMeans).
        classifier (estimator, optional): A classifier for predicting cluster labels (e.g., ExtraTreesClassifier).
        radius (int, default=1): The radius for rooted graph hashing.
        distance (int, default=3): The distance parameter for paired hashing.
        connector (int, default=0): Connector thickness.
        nbits (int, default=10): Number of bits for hashing.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.
        dense (bool, default=True): Whether to convert the feature matrix to a dense format.
        parallel (bool, default=True): Whether to encode graphs in parallel.
        weight_key (str, optional): Node weight key for reweighting features.
        node_attribute_key (str, optional): Node attribute key to use for additional features.
        attribute_dim (int, optional): Dimension of the attribute vector. If not None, performs SVD for dimensionality reduction to this dimension.
        attribute_alphabet_size (int, optional): Number of clusters for discretizing node attributes.
        use_node_kernel (bool, default=False): Whether to use node-level kernel encoding.
    """

    def __init__(self, embedder=None, clustering_predictor=None, classifier=None, radius=1, distance=3, connector=0,
                 nbits=10, degree_threshold=None, dense=True, parallel=True, weight_key=None, 
                 node_attribute_key=None, edge_attribute_key=None, attribute_dim=None, attribute_alphabet_size=None, use_node_kernel=False, sigma=None, use_edges_as_features=True):
        self.embedder = embedder
        self.clustering_predictor = clustering_predictor
        self.classifier = classifier
        self.radius = radius
        self.distance = distance
        self.connector = connector
        self.nbits = nbits
        self.degree_threshold = degree_threshold
        self.dense = dense
        self.parallel = parallel
        self.weight_key = weight_key
        self.node_attribute_key = node_attribute_key
        self.edge_attribute_key = edge_attribute_key
        self.attribute_dim = attribute_dim
        self.attribute_alphabet_size = attribute_alphabet_size
        self.use_node_kernel = use_node_kernel
        self.sigma = sigma
        self.use_edges_as_features = use_edges_as_features

    def __repr__(self):
        """
        Returns a string representation of the BaseNSPPK instance.

        Returns:
            str: The string representation.
        """
        infos = ', '.join([f"{key}={value}" for key, value in self.__dict__.items()])
        return f"{self.__class__.__name__}({infos})"

    def fit(self, graphs, targets=None):
        """
        Fit the encoder on the given graphs.

        This method fits the embedder and clustering predictor if they are provided.
        If attribute clustering is specified, it also fits the classifier.

        Args:
            graphs (list of networkx.Graph): The input graphs to fit on.
            targets (array-like, optional): Target values (unused).

        Returns:
            self: The instance itself.
        """
        if self.node_attribute_key and (self.attribute_dim or self.attribute_alphabet_size):
            attribute_mtx = np.vstack([
                graph.nodes[node_idx][self.node_attribute_key]
                for graph in graphs
                for node_idx in graph.nodes()
            ])

            if self.attribute_dim and self.embedder:
                self.embedder.fit(attribute_mtx)
                attribute_mtx = self.embedder.transform(attribute_mtx)

            if self.attribute_alphabet_size and self.clustering_predictor:
                targets = self.clustering_predictor.fit_predict(attribute_mtx)
                self.classifier.fit(attribute_mtx, targets)

        return self

    def embed_attributes(self, graphs):
        """
        Apply the embedder to reduce the dimensionality of node attributes.

        Args:
            graphs (list of networkx.Graph): List of graphs with attributes to embed.

        Returns:
            list of networkx.Graph: Graphs with embedded attributes.
        """
        if not self.embedder:
            return graphs

        out_graphs = []
        for graph in graphs:
            attribute_mtx = np.vstack([
                graph.nodes[node_idx][self.node_attribute_key]
                for node_idx in graph.nodes()
            ])
            embeddings = self.embedder.transform(attribute_mtx)
            out_graph = graph.copy()
            for embedding, node_idx in zip(embeddings, out_graph.nodes()):
                out_graph.nodes[node_idx]['original_' + self.node_attribute_key] = graph.nodes[node_idx][self.node_attribute_key]
                out_graph.nodes[node_idx][self.node_attribute_key] = embedding
            out_graphs.append(out_graph)
        return out_graphs

    def set_discrete_labels(self, graphs):
        """
        Assign discrete labels to nodes in the graphs based on clustered attributes.

        Args:
            graphs (list of networkx.Graph): Input graphs.

        Returns:
            list of networkx.Graph: Graphs with updated node labels.
        """
        if not self.classifier:
            return graphs

        out_graphs = []
        for graph in graphs:
            attribute_mtx = np.vstack([
                graph.nodes[node_idx][self.node_attribute_key]
                for node_idx in graph.nodes()
            ])
            labels = self.classifier.predict(attribute_mtx)
            out_graph = graph.copy()
            for label, node_idx in zip(labels, out_graph.nodes()):
                out_graph.nodes[node_idx]['label'] = label
            out_graphs.append(out_graph)
        return out_graphs

    def transform(self, graphs):
        """
        Transforms the input graphs into feature vectors.

        This method handles attribute embedding, label assignment, and encoding of graphs into feature vectors.

        Args:
            graphs (list of networkx.Graph): The list of graphs to transform.

        Returns:
            numpy.ndarray or scipy.sparse.csr_matrix: The feature matrix.
        """
        if self.node_attribute_key and self.attribute_dim:
            graphs = self.embed_attributes(graphs)
        if self.node_attribute_key and self.attribute_alphabet_size:
            graphs = self.set_discrete_labels(graphs)

        if self.use_node_kernel:
            data_mtx = paired_node_graphs_vector_encoder(
                graphs, 
                self.radius, 
                self.distance, 
                self.connector, 
                self.nbits,
                self.parallel, 
                self.weight_key,
                self.node_attribute_key,
                self.degree_threshold,
                self.sigma,
                self.use_edges_as_features
            )
        else:
            data_mtx = paired_graphs_vector_encoder(
                graphs, 
                self.radius, 
                self.distance, 
                self.connector, 
                self.nbits,
                self.parallel, 
                self.weight_key,
                self.node_attribute_key,
                self.edge_attribute_key,
                self.degree_threshold,
                self.sigma,
                self.use_edges_as_features
            )

        if self.dense:
            data_mtx = data_mtx.todense().A

        return data_mtx

class NSPPK(BaseEstimator, TransformerMixin):
    """
    NSPPK (Neighborhood Subgraph Pairwise Propagation Kernel) class specialized from BaseNSPPK.

    This class encodes graphs into feature vectors suitable for machine learning models by capturing both
    local (node-level) and structural (subgraph-level) information. It specializes the BaseNSPPK by
    initializing it with specific components: TruncatedSVD for embedding, KMeans for clustering, and 
    ExtraTreesClassifier for classification.

    Parameters:
        radius (int, default=1): The radius for rooted graph hashing.
        distance (int, default=3): The distance parameter for paired hashing.
        connector (int, default=0): Connector thickness.
        nbits (int, default=10): Number of bits for hashing.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.
        dense (bool, default=True): Whether to convert the feature matrix to a dense format.
        parallel (bool, default=True): Whether to encode graphs in parallel.
        weight_key (str, optional): Node weight key for reweighting features.
        node_attribute_key (str, optional): Node attribute key to use for additional features.
        attribute_dim (int, optional): Dimension of the attribute vector. If not None, performs SVD for dimensionality reduction to this dimension.
        attribute_alphabet_size (int, optional): Number of clusters for discretizing node attributes.
        use_node_kernel (bool, default=False): Whether to use node-level kernel encoding.
        sigma (float, optional): The sigma parameter for Gaussian weighting. Defaults to None.
    """

    def __init__(self, radius=1, distance=3, connector=0, nbits=10, degree_threshold=None, dense=True, parallel=True, weight_key=None, 
                 node_attribute_key=None, edge_attribute_key=None, attribute_dim=None, attribute_alphabet_size=None, use_node_kernel=False, sigma=None, use_edges_as_features=True):
        self.radius = radius
        self.distance = distance
        self.connector = connector
        self.nbits = nbits
        self.degree_threshold = degree_threshold
        self.dense = dense
        self.parallel = parallel
        self.weight_key = weight_key
        self.node_attribute_key = node_attribute_key
        self.edge_attribute_key = edge_attribute_key
        self.attribute_dim = attribute_dim
        self.attribute_alphabet_size = attribute_alphabet_size
        self.use_node_kernel = use_node_kernel
        self.sigma = sigma
        self.use_edges_as_features = use_edges_as_features

        self._initialize_components()

    def set_nbits(self, nbits):
        """
        Set the number of bits for hashing.

        Args:
            nbits (int): Number of bits for hashing.
        """
        self.nbits = nbits
        self._initialize_components()
        
    def _initialize_components(self):
        """
        Initializes the internal estimators based on the current parameters.
        """
        self.embedder = TruncatedSVD(n_components=self.attribute_dim) if self.attribute_dim else None
        self.clustering_predictor = KMeans(n_clusters=self.attribute_alphabet_size) if self.attribute_alphabet_size else None
        self.classifier = ExtraTreesClassifier(
            n_estimators=300, 
            n_jobs=-1 if self.parallel else None
        ) if self.attribute_alphabet_size else None

        self.base_nsppk = BaseNSPPK(
            embedder=self.embedder,
            clustering_predictor=self.clustering_predictor,
            classifier=self.classifier,
            radius=self.radius,
            distance=self.distance,
            connector=self.connector,
            nbits=self.nbits,
            degree_threshold=self.degree_threshold,
            dense=self.dense,
            parallel=self.parallel,
            weight_key=self.weight_key,
            node_attribute_key=self.node_attribute_key,
            edge_attribute_key=self.edge_attribute_key,
            attribute_dim=self.attribute_dim,
            attribute_alphabet_size=self.attribute_alphabet_size,
            use_node_kernel=self.use_node_kernel,
            sigma=self.sigma,
            use_edges_as_features=self.use_edges_as_features
        )

    def __repr__(self):
        """
        Returns a string representation of the NSPPK instance.

        Returns:
            str: The string representation.
        """
        return f"{self.__class__.__name__}(radius={self.radius}, distance={self.distance}, connector={self.connector}, sigma={self.sigma}, use_edges_as_features={self.use_edges_as_features}" \
               f"nbits={self.nbits}, degree_threshold={self.degree_threshold}, dense={self.dense}, " \
               f"parallel={self.parallel}, weight_key={self.weight_key}, node_attribute_key={self.node_attribute_key}, " \
               f"attribute_dim={self.attribute_dim}, attribute_alphabet_size={self.attribute_alphabet_size})"

    def fit(self, graphs, targets=None):
        """
        Fit the NSPPK encoder on the given graphs by delegating to BaseNSPPK.

        Args:
            graphs (list of networkx.Graph): The input graphs to fit on.
            targets (array-like, optional): Target values (unused).

        Returns:
            self: The instance itself.
        """
        self.base_nsppk.fit(graphs, targets)
        return self

    def transform(self, graphs):
        """
        Transform the input graphs into feature vectors by delegating to BaseNSPPK.

        Args:
            graphs (list of networkx.Graph): The list of graphs to transform.

        Returns:
            numpy.ndarray or scipy.sparse.csr_matrix: The feature matrix.
        """
        check_is_fitted(self.base_nsppk, 'fit')
        return self.base_nsppk.transform(graphs)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Args:
            deep (bool, default=True): If True, will return the parameters for this estimator and
                                        contained sub-objects that are estimators.

        Returns:
            dict: Parameter names mapped to their values.
        """
        params = {
            'radius': self.radius,
            'distance': self.distance,
            'connector': self.connector,
            'nbits': self.nbits,
            'degree_threshold': self.degree_threshold,
            'dense': self.dense,
            'parallel': self.parallel,
            'weight_key': self.weight_key,
            'node_attribute_key': self.node_attribute_key,
            'edge_attribute_key': self.edge_attribute_key,
            'attribute_dim': self.attribute_dim,
            'attribute_alphabet_size': self.attribute_alphabet_size,
            'use_node_kernel': self.use_node_kernel,
            'sigma': self.sigma,
            'use_edges_as_features': self.use_edges_as_features
        }

        if deep:
            if self.embedder is not None:
                params.update({f'embedder__{k}': v for k, v in self.embedder.get_params().items()})
            if self.clustering_predictor is not None:
                params.update({f'clustering_predictor__{k}': v for k, v in self.clustering_predictor.get_params().items()})
            if self.classifier is not None:
                params.update({f'classifier__{k}': v for k, v in self.classifier.get_params().items()})

        return params

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Args:
            **params: Estimator parameters.

        Returns:
            self: The instance itself.
        """
        if not params:
            return self

        # Separate parameters for internal estimators
        embedder_params = {}
        clustering_predictor_params = {}
        classifier_params = {}
        main_params = {}

        for key, value in params.items():
            if key.startswith('embedder__'):
                embedder_params[key.split('__', 1)[1]] = value
            elif key.startswith('clustering_predictor__'):
                clustering_predictor_params[key.split('__', 1)[1]] = value
            elif key.startswith('classifier__'):
                classifier_params[key.split('__', 1)[1]] = value
            else:
                main_params[key] = value

        # Set main parameters and reinitialize components if needed
        if main_params:
            for key, value in main_params.items():
                setattr(self, key, value)
            self._initialize_components()

        # Set parameters for internal estimators
        if embedder_params and self.embedder is not None:
            self.embedder.set_params(**embedder_params)
        if clustering_predictor_params and self.clustering_predictor is not None:
            self.clustering_predictor.set_params(**clustering_predictor_params)
        if classifier_params and self.classifier is not None:
            self.classifier.set_params(**classifier_params)

        # Re-initialize BaseNSPPK with updated components
        if main_params:
            self.base_nsppk = BaseNSPPK(
                embedder=self.embedder,
                clustering_predictor=self.clustering_predictor,
                classifier=self.classifier,
                radius=self.radius,
                distance=self.distance,
                connector=self.connector,
                nbits=self.nbits,
                degree_threshold=self.degree_threshold,
                dense=self.dense,
                parallel=self.parallel,
                weight_key=self.weight_key,
                node_attribute_key=self.node_attribute_key,
                edge_attribute_key=self.edge_attribute_key,
                attribute_dim=self.attribute_dim,
                attribute_alphabet_size=self.attribute_alphabet_size,
                use_node_kernel=self.use_node_kernel,
                sigma=self.sigma,
                use_edges_as_features=self.use_edges_as_features
            )

        return self


class NodeNSPPK(BaseEstimator, TransformerMixin):
    """
    NodeNSPPK (Node Neighborhood Subgraph Pairwise Propagation Kernel) class compatible with scikit-learn's Transformer interface.

    This class encodes graphs into node-level feature vectors suitable for
    node-level machine learning models. It captures both local (node-level) and structural (subgraph-level) information
    by hashing node labels and the structure of their neighborhoods. Additionally, it supports dimensionality reduction
    of node attributes using Singular Value Decomposition (SVD).

    Parameters:
        radius (int, default=1): The radius for rooted graph hashing.
        distance (int, default=3): The distance parameter for paired hashing.
        connector (int, default=0): Connector thickness.
        nbits (int, default=10): Number of bits for hashing.
        degree_threshold (int, optional): Threshold for node degree to limit hashing. Defaults to None.
        dense (bool, default=True): Whether to convert the feature matrix to a dense format.
        parallel (bool, default=True): Whether to encode graphs in parallel.
        weight_key (str, optional): Node weight key for reweighting features.
        node_attribute_key (str, optional): Node attribute key to use for additional features.
        attribute_dim (int, optional): The target dimensionality for attribute vectors; if set, applies SVD for dimensionality reduction.
        attribute_alphabet_size (int, optional): Number of clusters for discretizing node attributes.
        sigma (float, optional): The sigma parameter for Gaussian weighting. Defaults to None.
    """

    def __init__(self, radius=1, distance=3, connector=0, nbits=10, degree_threshold=None, dense=True, parallel=True, weight_key=None, 
                 node_attribute_key=None, attribute_dim=None, attribute_alphabet_size=None, sigma=None, use_edges_as_features=True):
        self.nsppk = NSPPK(
            radius=radius,
            distance=distance,
            connector=connector,
            nbits=nbits,
            degree_threshold=degree_threshold,
            dense=dense,
            parallel=parallel,
            weight_key=weight_key,
            node_attribute_key=node_attribute_key,
            attribute_dim=attribute_dim,
            attribute_alphabet_size=attribute_alphabet_size,
            sigma=sigma,
            use_edges_as_features=use_edges_as_features
        )
        
    def set_nbits(self, nbits):
        """
        Set the number of bits for hashing.

        Args:
            nbits (int): Number of bits for hashing.
        """
        self.nsppk.set_nbits(nbits)

    def __repr__(self):
        """
        Generate a string representation of the NodeNSPPK instance.

        Returns:
            str: The string representation of the object.
        """
        return f"{self.__class__.__name__}({self.nsppk})"

    def fit(self, graphs, targets=None):
        """
        Fit the NodeNSPPK encoder on the given graphs by delegating to NSPPK.

        Args:
            graphs (list of networkx.Graph): The input graphs to fit on.
            targets (array-like, optional): Target values (unused).

        Returns:
            self: The instance itself.
        """
        self.nsppk.fit(graphs, targets)
        return self

    def transform(self, graphs):
        """
        Transform the input graphs into node-level feature vectors.

        This method encodes each graph into a list of node feature matrices, where each matrix corresponds to a graph
        and contains feature vectors for its nodes. If `attribute_dim` is specified, it applies SVD to reduce the
        dimensionality of node attributes before encoding. Additionally, if `attribute_alphabet_size` is specified,
        it assigns discrete labels to nodes based on clustered attributes.

        Args:
            graphs (list of networkx.Graph): The list of graphs to transform.

        Returns:
            list of numpy.ndarray or list of scipy.sparse.csr_matrix: A list where each element is a node feature matrix for a graph.
        """
        if self.nsppk.base_nsppk.node_attribute_key and self.nsppk.base_nsppk.attribute_dim:
            graphs = self.nsppk.base_nsppk.embed_attributes(graphs)
        if self.nsppk.base_nsppk.node_attribute_key and self.nsppk.base_nsppk.attribute_alphabet_size:
            graphs = self.nsppk.base_nsppk.set_discrete_labels(graphs)

        nodes_data_mtx_list = paired_node_vector_encoder(
            graphs,
            self.nsppk.base_nsppk.radius,
            self.nsppk.base_nsppk.distance,
            self.nsppk.base_nsppk.connector,
            self.nsppk.base_nsppk.nbits,
            self.nsppk.base_nsppk.parallel,
            self.nsppk.base_nsppk.weight_key,
            self.nsppk.base_nsppk.node_attribute_key,
            self.nsppk.base_nsppk.degree_threshold,
            self.nsppk.sigma,
            self.nsppk.use_edges_as_features
        )

        if self.nsppk.base_nsppk.dense:
            nodes_data_mtx_list = [mtx.todense().A for mtx in nodes_data_mtx_list]

        return nodes_data_mtx_list


class ImportanceNSPPK(object):
    """
    A class to visualize node importance in graphs using feature importance derived from an ensemble classifier.

    This class leverages the NSPPK (Neighborhood Subgraph Pairwise Path Kernel) method
    to decompose graphs into substructures and vectorize them. It then uses an ExtraTreesClassifier
    to compute feature importances, which are used to assign importance weights to nodes in the graphs.

    Attributes:
        node_nsppk (object): An instance of NSPPK or a similar class responsible for graph decomposition and vectorization.
        importance_key (str): The key under which node importance weights will be stored in the graph's node attributes.
        n_iter (int): Number of iterations for training the classifier to ensure stable feature importance estimates.
        n_estimators (int): Number of trees in the ExtraTreesClassifier.
        quantile (float): Quantile threshold for filtering out low-importance features.
        parallel (bool): Whether to utilize parallel processing during classifier training.
        normalize (bool): Whether to normalize node and edge weights to [0,1] range.
        feature_importance_vector (np.ndarray): Normalized feature importance scores after processing.
    """
    
    def __init__(self, node_nsppk, importance_key='att', n_iter=10, n_estimators=100, quantile=0.5, parallel=True, normalize=True):
        """
        Initializes the ImportanceNSPPK instance with specified parameters.

        Args:
            node_nsppk (object): An instance responsible for graph decomposition and vectorization (e.g., NSPPK).
            importance_key (str, optional): The key for storing node importance in graph attributes. Defaults to 'att'.
            n_iter (int, optional): Number of iterations for classifier training. Defaults to 10.
            n_estimators (int, optional): Number of trees in the ExtraTreesClassifier. Defaults to 100.
            quantile (float, optional): Quantile threshold for filtering feature importances. Defaults to 0.5.
            parallel (bool, optional): Whether to run classifier training in parallel. Defaults to True.
            normalize (bool, optional): Whether to normalize weights to [0,1]. Defaults to True.
        """
        self.node_nsppk = node_nsppk
        self.importance_key = importance_key
        self.n_iter = n_iter
        self.n_estimators = n_estimators
        self.quantile = quantile
        self.parallel = parallel
        self.normalize = normalize
        self.feature_importance_vector = None  # To store the computed normalized feature importance scores

    def fit(self, graphs, targets=None):
        """
        Fits the model by computing feature importances from the provided graphs and corresponding target labels.

        This method vectorizes the input graphs, aggregates node features, and trains an ExtraTreesClassifier
        multiple times to obtain stable estimates of feature importances. It then processes these importances
        by averaging, subtracting the standard deviation, thresholding, and normalizing.

        Args:
            graphs (list): A list of graph objects to be vectorized and used for training.
            targets (array-like, optional): Target labels corresponding to each graph. Required for supervised training.

        Returns:
            self: Returns the instance itself to allow method chaining.
        """
        # Vectorize the graphs using the provided NSPPK instance
        node_feature_mtx_list = self.node_nsppk.fit_transform(graphs)
        
        # Aggregate node features for each graph by summing across nodes
        X = np.vstack([node_feature_mtx.sum(axis=0) for node_feature_mtx in node_feature_mtx_list])

        feature_importances = []  # List to store feature importances from each iteration

        # Perform multiple iterations to compute stable feature importances
        for it in range(self.n_iter):
            # Split the data into training and testing sets with a fixed random state for reproducibility
            train_X, test_X, train_targets, test_targets = train_test_split(
                X, targets, train_size=0.7, random_state=it+1
            )
            
            # Initialize and train the ExtraTreesClassifier
            clf = ExtraTreesClassifier(
                n_estimators=self.n_estimators, 
                n_jobs=-1 if self.parallel else None,
                random_state=it+1  # Ensure reproducibility across iterations
            ).fit(train_X, train_targets)
            
            # Append the feature importances from the trained classifier
            feature_importances.append(clf.feature_importances_)
        
        # Convert the list of feature importances into a NumPy matrix for statistical processing
        feature_importances_mtx = np.vstack(feature_importances)
        
        # Compute the mean and standard deviation of feature importances across all iterations
        mean_importances = np.mean(feature_importances_mtx, axis=0)
        std_importances = np.std(feature_importances_mtx, axis=0)
        
        # Calculate the adjusted feature importance by subtracting the standard deviation from the mean
        adjusted_importances = mean_importances - std_importances
        
        # Set any negative importance values to zero to ensure non-negativity
        adjusted_importances[adjusted_importances < 0] = 0
        
        # Apply a quantile threshold to filter out features with low importance
        threshold = np.quantile(adjusted_importances, self.quantile)
        adjusted_importances[adjusted_importances < threshold] = 0
        
        # Normalize the feature importance vector by dividing by its maximum value to scale between 0 and 1
        max_importance = np.max(adjusted_importances)
        if (max_importance > 0):
            self.feature_importance_vector = adjusted_importances / max_importance
        else:
            self.feature_importance_vector = adjusted_importances  # Remain zero if max is zero
        
        return self  # Allow method chaining

    def transform(self, graphs):
        """
        Transforms the input graphs by assigning normalized importance weights to each node based on feature importances.

        This method vectorizes the nodes of each graph, computes weights by combining node features with the
        precomputed feature importance vector, and normalizes these weights to assign importance scores to nodes.
        Additionally, edge weights are assigned based on the product of the connected nodes' importance scores.

        Args:
            graphs (list): A list of graph objects to be transformed with node and edge importance weights.

        Returns:
            out_graphs (list): A list of transformed graph objects with updated node and edge importance attributes.
        """
        # Vectorize the nodes of the graphs using the provided NSPPK instance
        node_feature_mtx_list = self.node_nsppk.transform(graphs)
        
        out_graphs = []  # List to store the transformed graphs

        # Iterate over each graph and its corresponding node feature matrix
        for graph, node_feature_mtx in zip(graphs, node_feature_mtx_list):
            out_graph = graph.copy()  # Create a copy of the graph to avoid modifying the original
            
            # Compute the weights by multiplying node features with the feature importance vector
            weights = node_feature_mtx * self.feature_importance_vector
            
            # Sum the weights across all feature dimensions for each node to obtain a single weight per node
            weights = np.sum(weights, axis=1)
            
            if self.normalize:
                # Normalize the weights by the maximum weight to scale between 0 and 1
                max_weight = np.max(weights) if np.max(weights) > 0 else 1
                normalized_weights = weights / max_weight
            else:
                normalized_weights = weights
            
            # Assign the normalized weight to each node in the graph
            for i, weight in enumerate(normalized_weights):
                out_graph.nodes[i][self.importance_key] = weight
            
            # Assign edge weights based on the product of the connected nodes' importance scores
            for u, v in out_graph.edges():
                out_graph.edges[u, v][self.importance_key] = out_graph.nodes[u][self.importance_key] * out_graph.nodes[v][self.importance_key]
        
            for u, node_feature in zip(out_graph.nodes(), node_feature_mtx):
                out_graph.nodes[u]['node_feature'] = node_feature

            for u, v in out_graph.edges():
                out_graph.edges[u, v]['edge_feature'] = out_graph.nodes[u]['node_feature'] + out_graph.nodes[v]['node_feature']

            out_graphs.append(out_graph)  # Add the transformed graph to the output list
        
        return out_graphs  # Return the list of transformed graphs
