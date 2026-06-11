from coco_grape.module.quotient_graph_vectorize import quotient_graph_vectorize
from coco_grape.module.quotient_graph_vectorize import quotient_graph_node_vectorize
from coco_grape.module.quotient_graph import QuotientGraph
from coco_grape.module.quotient_graph_operators import *
from coco_grape.data_graphicalizer.graph.importance_quotient_graph_graphicalizer import ImportanceQuotientGraphGraphicalizer
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.sparse import vstack, csr_matrix

class QuotientGraphVectorizer(object):
    def __init__(self, 
                 decomposition_function=neighborhood(radius=(0,2)),
                 nbits=16,
                 dense=True,
                 parallel=True):
        self.decomposition_function = decomposition_function
        self.nbits = nbits
        self.dense = dense
        self.parallel = parallel
        
    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
    
    def fit(self, graphs, targets=None):
        return self
    
    def transform(self, graphs):
        encodings = quotient_graph_vectorize(graphs, decomposition_function=self.decomposition_function, nbits=self.nbits, parallel=self.parallel)
        if self.dense:
            encodings = encodings.todense().A
        return encodings

    def fit_transform(self, graphs, targets=None):
        return self.fit(graphs, targets).transform(graphs)

    def extract(self, graphs):
        quotient_graphs_list = [self.decomposition_function(QuotientGraph(graph, nbits=self.nbits)) for graph in graphs]
        return quotient_graphs_list


#--------------------------------------------------------------------------------------------------------------------------------------------
class QuotientGraphNodeVectorizer(object):
    def __init__(self, 
                 decomposition_function=None,
                 nbits=16,
                 dense=True,
                 parallel=True):
        self.decomposition_function = decomposition_function
        self.nbits = nbits
        self.dense = dense
        self.parallel = parallel

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def fit(self, graphs, targets=None):
        return self
    
    def transform(self, graphs):
        encodings_list = quotient_graph_node_vectorize(graphs, decomposition_function=self.decomposition_function, nbits=self.nbits, parallel=self.parallel)
        if self.dense:
            encodings_list = [encodings.todense().A for encodings in encodings_list]
        return encodings_list
    
    def extract(self, graphs):
        quotient_graphs_list = [self.decomposition_function(QuotientGraph(graph, nbits=self.nbits)) for graph in graphs]
        return quotient_graphs_list
    

#--------------------------------------------------------------------------------------------------------------------------------------------
def assign_bins_to_weights(node_weights, n_bins):
    node_weights = np.max(node_weights, axis=1).flatten()
    if np.all(node_weights == node_weights[0]):
        return np.zeros_like(node_weights, dtype=int)

    quantile_values = np.linspace(0, 1, n_bins+1)[1:-1]
    bin_thresholds = np.quantile(node_weights, quantile_values)
    bin_indices = np.digitize(node_weights, bin_thresholds, right=False)
    bin_indices = np.clip(bin_indices, 0, n_bins-1)
    return bin_indices

def process_single_graph(
    graph,
    node_weights,
    per_graph_encodings,
    n_bins
):
    """
    `per_graph_encodings`: a list of node-feature arrays [array_for_vec0, array_for_vec1, ...]
                           all corresponding exactly to this one graph.
    For node 'node_id':
      - `per_graph_encodings[bin_idx]` is the correct array
      - `per_graph_encodings[bin_idx][node_id]` is the node_id'th feature vector
    """
    bin_indices = assign_bins_to_weights(node_weights, n_bins)
    
    binned_encodings = []
    for node_id in graph.nodes():
        bin_idx = bin_indices[node_id]
        node_encoding = per_graph_encodings[bin_idx][node_id]
        binned_encodings.append(node_encoding)

    binned_encodings = np.vstack(binned_encodings)
    return np.sum(binned_encodings, axis=0)

class ImportanceQuotientGraphVectorizer(BaseEstimator, TransformerMixin):
    """
    Transforms graphs into vector representations by leveraging node importance scores and multiple vectorizers.

    Parameters:
    - decomposition_function (callable): Function to decompose graphs into substructures.
    - abstract_decomposition_functions (list): List of functions for abstract decomposition.
    - nbits (int): Number of bits for vectorization.
    - dense (bool): Whether to use dense vector representations.
    - feature_importance_n_iter (int, optional): Number of iterations for classifier training to compute stable feature
        importance estimates. Defaults to 10.
    - n_estimators (int, optional): Number of trees in the ExtraTreesClassifier. More trees can lead
        to more stable importance estimates but increase computational cost. Defaults to 100.
    - quantile (float, optional): Quantile threshold for filtering out low-importance features. Features
        below this quantile are set to zero. Defaults to 0.5.
    - parallel (bool, optional): Whether to utilize parallel processing during vectorization and classifier
        training. Defaults to True.
    - normalize (bool, optional): Whether to normalize feature importance scores to [0,1] range.
        If True, scores are divided by their maximum value. Defaults to True.

    Attributes:
    - node_vectorizer (QuotientGraphNodeVectorizer): Primary node vectorizer.
    - importance_quotient_graph_graphicalizer (ImportanceQuotientGraphGraphicalizer): Graphicalizer for importance scores.
    - abstract_node_vectorizers (list): List of abstract node vectorizers.
    - n_bins (int): Number of bins for weight assignment.
    """
    def __init__(self, 
                 decomposition_function=neighborhood(radius=(0,2)),
                 abstract_decomposition_functions=None,
                 nbits=16,
                 dense=True, 
                 feature_importance_n_iter=10, 
                 n_estimators=100, 
                 quantile=0.5, 
                 parallel=True, 
                 normalize=True):
        if abstract_decomposition_functions is None:
            warnings.warn("No abstract decomposition functions provided. Only primary vectorizer will be used.")
        if not isinstance(nbits, int) or nbits <= 0:
            raise ValueError("nbits must be a positive integer")
        if not isinstance(abstract_decomposition_functions, list):
            raise TypeError("abstract_decomposition_functions must be a list")

        self.decomposition_function = decomposition_function
        self.abstract_decomposition_functions = abstract_decomposition_functions
        self.nbits = nbits
        self.dense = dense
        self.parallel = parallel
        self.node_vectorizer = QuotientGraphNodeVectorizer(decomposition_function=self.decomposition_function, nbits=self.nbits)
        self.importance_quotient_graph_graphicalizer = ImportanceQuotientGraphGraphicalizer(
            node_vectorizer=self.node_vectorizer, 
            feature_importance_n_iter=feature_importance_n_iter, 
            n_estimators=n_estimators,
            quantile=quantile,
            parallel=parallel,
            normalize=normalize)
        self.abstract_node_vectorizers = [QuotientGraphNodeVectorizer(decomposition_function=abstract_decomposition_function, nbits=self.nbits) for abstract_decomposition_function in self.abstract_decomposition_functions]
        self.n_bins = len(self.abstract_decomposition_functions) + 1  # Number of bins

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
    
    def fit(self, graphs, targets):
        """
        Fit the graphicalizer with the provided graphs and targets.

        Parameters:
        - graphs (list): List of graph objects to fit.
        - targets (array-like): Target labels for supervised learning.

        Returns:
        - self: Fitted instance of ImportanceQuotientGraphVectorizer.
        """
        self.importance_quotient_graph_graphicalizer.fit(graphs, targets)
        return self
    
    def _transform(self, graphs):
        encodings_list_list = [self.node_vectorizer.transform(graphs)]
        encodings_list_list.extend([abstract_node_vectorizer.transform(graphs) for abstract_node_vectorizer in self.abstract_node_vectorizers])
        encodings_list_list = encodings_list_list[::-1] #revert the order so that the most abstract operator is first and the most discriminative is last
        return encodings_list_list
    
    def transform(self, graphs):
        # 1. Generate all encodings for each graph using our vectorizers
        encodings_list_list = self._transform(graphs)

        # 2. The last set of encodings is used to compute importance weights
        node_feature_matrices = encodings_list_list[-1]
        weights_list = self.importance_quotient_graph_graphicalizer.compute_weights(graphs, node_feature_matrices)
        
        # Extract only the node weights (ignore edge weights)
        node_weights_list = [weights_tuple[0] for weights_tuple in weights_list]

        # (Optionally) reduce node_weights if they're multi-class
        # e.g., if shape is (n_nodes, n_classes):
        #       node_weights = np.max(node_weights, axis=1)
        #       or some other aggregation

        transformed_graphs = []
        if self.parallel:
            with Pool(nodes=None) as pool:
                partial_args = []
                for graph_id, (graph, node_weights) in enumerate(zip(graphs, node_weights_list)):
                    # Build the list of node-encoding matrices for this graph
                    per_graph_encodings = [
                        encodings_list_list[j][graph_id]
                        for j in range(len(encodings_list_list))
                    ]
                    
                    partial_args.append((
                        graph,
                        node_weights,
                        per_graph_encodings,
                        self.n_bins
                    ))

                def star(fn):
                    def wrapper(args):
                        return fn(*args)
                    return wrapper

                transformed_graphs = pool.map(star(process_single_graph), partial_args)

        else:
            # Sequential
            for graph_id, (graph, node_weights) in enumerate(zip(graphs, node_weights_list)):
                per_graph_encodings = [
                    encodings_list_list[j][graph_id] 
                    for j in range(len(encodings_list_list))
                ]
                arr = process_single_graph(
                    graph,
                    node_weights,
                    per_graph_encodings,
                    self.n_bins
                )
                transformed_graphs.append(arr)

        if not transformed_graphs:
            encodings = np.zeros((0, self.nbits))
        else:
            encodings = np.vstack(transformed_graphs)

        return encodings


    
    def fit_transform(self, graphs, targets):
        return self.fit(graphs, targets).transform(graphs)