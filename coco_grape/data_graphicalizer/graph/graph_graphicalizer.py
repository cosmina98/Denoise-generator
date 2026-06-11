import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import pairwise_kernels
from scipy.sparse import vstack
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import TruncatedSVD
from coco_grape.graph_vectorizer.graph_vectorizer import GraphVectorizer
from coco_grape.module.construct import decomposition
from coco_grape.module import *

def make_difference_decomposition_function(d0):
    d1 = compose(merge(), d0)
    d2 = compose(complement(), d1)
    d3 = compose(node(), d2)
    d4 = add(d3, d0)
    decomposition_function = compose(edges_from_distance(min_size=0, max_size=1), d4)
    return decomposition_function

class GraphOfSubgraphsGraphicalizer(object):
    def __init__(self, decomposition_function=None, nbits=None, parallel=True, use_difference_decomposition_function=False):
        self.decomposition_function = decomposition_function
        self.nbits = nbits
        self.parallel = parallel
        self.use_difference_decomposition_function = use_difference_decomposition_function
        
    def fit(self, graphs, targets=None):
        return self
        
    def transform(self, graphs):
        if self.use_difference_decomposition_function: df = make_difference_decomposition_function(self.decomposition_function)
        else: df = self.decomposition_function
        graphofsubgraphss = decomposition(graphs, decomposition_function=df, nbits=self.nbits, parallel=self.parallel)
        return graphofsubgraphss

    def fit_transform(self, graphs, targets=None):
        return self.fit(graphs, targets).transform(graphs)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


def resize_right(data_mtx, desired_dim):
    if desired_dim < data_mtx.shape[1]: return data_mtx[:,:desired_dim]
    padding_dim = desired_dim - data_mtx.shape[1]
    return np.pad(data_mtx, pad_width=((0,0),(0,padding_dim)))

def normalized_laplacian_SVD(graph, n_components): 
    L = nx.normalized_laplacian_matrix(graph)
    effective_n_components = min(n_components, len(graph)-1)
    data_mtx = TruncatedSVD(n_components=effective_n_components).fit_transform(L)
    data_mtx = resize_right(data_mtx, desired_dim=n_components)
    return data_mtx

def annotate_normalized_laplacian_SVD(graph, n_components, attribute_key='vec'):
    out_graph = graph.copy()
    data_mtx = normalized_laplacian_SVD(graph, n_components)
    for node_idx in graph.nodes():
        if attribute_key not in graph.nodes[node_idx]:
            vec = data_mtx[node_idx,:].reshape(1,-1)
        else:
            vec = graph.nodes[node_idx][attribute_key]
            vec = np.array(vec).reshape(1,-1)
            vec = np.hstack(vec,data_mtx[node_idx,:])
        vec = vec.flatten()
        out_graph.nodes[node_idx][attribute_key] = vec
    return out_graph

class NormalizedLaplacianSVDGraphGraphicalizer(object):
    def __init__(self, n_components=10, attribute_key='vec'):
        self.n_components = n_components
        self.attribute_key = attribute_key
        
    def fit(self, graphs, targets=None):
        return self
    
    def transform(self, graphs):
        out_graphs = [annotate_normalized_laplacian_SVD(graph, n_components=self.n_components, attribute_key=self.attribute_key) for graph in graphs]
        return out_graphs

    def fit_transform(self, graphs, targets=None):
        return self.fit(graphs).transform(graphs)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


class NodeDecompositionalGraphGraphicalizer(object):
    def __init__(self, node_vectorizer, attribute_key='vec', todense=True):
        self.node_vectorizer = node_vectorizer
        self.attribute_key = attribute_key
        self.todense = todense
        
    def fit(self, graphs, targets=None):
        self.node_vectorizer.fit(graphs, targets)
        return self

    def transform(self, graphs, targets=None):
        #A target here is an auxiliary graph associated to the primary input graph:
        #the input and auxiliary graphs need to be paired, i.e. they have the same oredered sequence of nodes   
        #node embeddings are extracted from the node vecotrization in the auxiliary graph but transfered to the input nodes
        #this is useful when the input graphs are simpler (e.g. seqeunces) and the auxiliary graphs have been obtained by
        #complex processes (e.g. folding) 
        if targets is None:
            node_embeddings_list = self.node_vectorizer.transform(graphs)
        else:
            node_embeddings_list = self.node_vectorizer.transform(targets)
        out = [self.transform_single(graph, node_embeddings) for graph, node_embeddings in zip(graphs, node_embeddings_list)]
        return out

    def transform_single(self, graph, node_embeddings):
        assert len(graph) == len(node_embeddings), 'ERROR: graph dim (%d) should be the same as embedding dim (%d)'%(len(graph), len(node_embedding))
        for node_idx, node_embedding in zip(graph.nodes, node_embeddings):
            if self.todense: graph.nodes[node_idx][self.attribute_key] = node_embedding.todense().A.flatten()
            else: graph.nodes[node_idx][self.attribute_key] = node_embedding
        return graph
        
    def fit_transform(self, graphs, targets=None):
        return self.fit(graphs).transform(graphs)



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


class NodeEmbedderGraphGraphicalizer(object):
    def __init__(self, node_transformer, attribute_key='pred'):
        self.node_transformer = node_transformer
        self.attribute_key = attribute_key
        
    def fit(self, graphs, targets=None):
        self.node_transformer.fit(graphs)
        return self

    def transform(self, graphs):
        node_embeddings_list = self.node_transformer.transform(graphs)
        return [self.transform_single(graph, node_embeddings) for graph, node_embeddings in zip(graphs, node_embeddings_list)]

    def transform_single(self, graph, node_embeddings):
        for node_idx, node_embedding in zip(graph.nodes, node_embeddings):
            graph.nodes[node_idx][self.attribute_key] = node_embedding
        return graph
        
    def fit_transform(self, graphs, targets=None):
        return self.fit(graphs).transform(graphs)

        
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

class NodeClusteringGraphGraphicalizer(object):
    def __init__(self, node_vectorizer, clustering, classifier, edge_threshold=.9, add_minimum_spanning_tree=True):
        self.node_vectorizer = node_vectorizer
        self.clustering = clustering
        self.classifier = classifier
        self.edge_threshold = edge_threshold
        self.add_minimum_spanning_tree = add_minimum_spanning_tree
    
    def fit(self, graphs, targets=None):
        self.node_vectorizer.fit(graphs, targets)
        #extract nodes vectors
        node_embeddings_list = self.node_vectorizer.transform(graphs)
        node_embeddings = vstack(node_embeddings_list)
        node_embeddings = node_embeddings.toarray()
        #cluster them
        cluster_labels = self.clustering.fit_predict(node_embeddings)
        self.classifier.fit(node_embeddings, cluster_labels)
        return self

    def transform(self, graphs):
        return [self.transform_single(graph) for graph in graphs]

    def transform_single(self, graph):
        node_embeddings = self.node_vectorizer.transform([graph])[0]
        node_labels = self.classifier.predict(node_embeddings)
        A = pairwise_kernels(node_embeddings, metric='cosine')
        if self.add_minimum_spanning_tree:
            T = minimum_spanning_tree(A).toarray()
            if self.edge_threshold < 1:
                A[A<self.edge_threshold] = 0
            else:
                #add 2 to consider self
                th = np.sort(A, axis=1)[:,-(self.edge_threshold+2)]
                A[A<=th.reshape(-1,1)] = 0
            A = np.maximum(A,T)
        else: 
            if self.edge_threshold < 1: 
                A[A<self.edge_threshold] = 0
            else:
                th = np.sort(A, axis=1)[:,-(self.edge_threshold+2)]
                A[A<=th.reshape(-1,1)] = 0
        A = A + A.T
        out_graph = nx.from_numpy_array(A)
        for node_idx in range(len(node_labels)):
            out_graph.nodes[node_idx]['original_label'] = graph.nodes[node_idx]['label']
            out_graph.nodes[node_idx]['label'] = node_labels[node_idx]
        nx.set_edge_attributes(out_graph, '-', 'label')
        return out_graph
        
    def fit_transform(self, graphs, targets=None):
        return self.fit(graphs).transform(graphs)


class DecompositionNodeClusteringGraphGraphicalizer(object):
    def __init__(self, decomposition_function, nbits, node_alphabet_size, edge_threshold=3, add_minimum_spanning_tree=False):
        node_vectorizer = GraphVectorizer(decomposition_function=decomposition_function, nbits=nbits, feature_type='node_list')
        clustering = AgglomerativeClustering(n_clusters=node_alphabet_size)
        classifier = ExtraTreesClassifier(n_estimators=300, n_jobs=-1)
        self.graph_graphicalizer = NodeClusteringGraphGraphicalizer(node_vectorizer, clustering, classifier, edge_threshold=edge_threshold, add_minimum_spanning_tree=add_minimum_spanning_tree)
        
    def fit(self, graphs, targets=None):
        self.graph_graphicalizer.fit(graphs, targets)
        return self
    
    def transform(self, graphs):
        return self.graph_graphicalizer.transform(graphs)

    def fit_transform(self, graphs, targets=None):
        return self.fit(graphs).transform(graphs)


#--------------------------------------------------------------------------------------------------

def relabel(graph):
    graph = graph.copy()
    for u in graph.nodes():
        label_pair = graph.nodes[u]['label']
        label_src, label_dst = label_pair
        label_src, label_dst = '%s'%label_src, '%s'%label_dst
        if label_src > label_dst:
            label_src, label_dst = label_dst, label_src 
        label = '%s:%s' % (label_src, label_dst)
        graph.nodes[u]['label'] = label
    return graph

def graph_product(graph1, graph2):
    graph = nx.cartesian_product(graph1, graph2)
    graph = nx.convert_node_labels_to_integers(graph)
    graph = relabel(graph)
    return graph


class ProductGraphGraphicalizer(object):

    def __init__(self, factor_graphs):
        factor_graph = nx.Graph()
        for graph in factor_graphs:
            factor_graph = nx.disjoint_union(factor_graph, graph)
        self.factor_graph = factor_graph

    def fit(self, graphs, targets=None):
        return self

    def transform(self, graphs):
        graphs = [graph_product(graph, self.factor_graph) for graph in graphs]
        return graphs

    def fit_transform(self, graphs, targets=None):
        return self.fit(graphs, targets).transform(graphs)


#--------------------------------------------------------------------------------------------------


class CompositionalGraphGraphicalizer(object):

    def __init__(self, graphicalizers):
        self.graphicalizers = graphicalizers
        
    def __repr__(self):
        infos = []
        infos += ['graphicalizer_%d:%s'%(i,str(graphicalizer)) for i,graphicalizer in enumerate(self.graphicalizers)]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def fit(self, graphs, targets=None):
        """fit."""
        graphs = self.graphicalizers[0].fit_transform(graphs, targets)
        for graphicalizer in self.graphicalizers[1:]:
            graphs = graphicalizer.fit_transform(graphs, targets)
        return self

    def transform(self, graphs):
        """transform."""
        graphs = self.graphicalizers[0].transform(graphs)
        for graphicalizer in self.graphicalizers[1:]:
            graphs = graphicalizer.transform(graphs)
        return graphs

    def fit_transform(self, graphs, targets=None):
        return self.fit(graphs, targets).transform(graphs)


