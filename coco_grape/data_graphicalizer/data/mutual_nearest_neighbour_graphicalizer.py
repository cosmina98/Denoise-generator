# author Oskar Maier
# version r0.1.0
# since 2014-10-15
# status Release
# Adapted
# author Fabrizio Costa
# version 1.0
# since 21-Dic-2023

import warnings
import networkx as nx
import numpy as np
from scipy.sparse.csr import csr_matrix
from scipy.spatial.distance import pdist, squareform


def dist(objects, metric='euclidean', diagval = np.inf):
    distvec = pdist(objects, metric=metric)
    out = squareform(distvec)
    np.fill_diagonal(out, diagval)
    return out

class MutualNearestNeighbourGraphicalizer(object):
    def __init__(self, n_neighbours=5, n_dense_links=1, edge_label='-', metric='euclidean'):
        self.n_neighbours = n_neighbours
        self.n_dense_links = n_dense_links
        self.edge_label = edge_label
        self.metric = metric
        
    def fit(self, data, targets=None):
        return self
    
    def make_mutual_nearest_neighbour_graph(self, instance):
        # compute their pairwise-distances
        pdists = dist(instance, self.metric)
        density = np.sum(1/pdists, axis=1).flatten()

        # get the (k) nearest neighbours 
        nearest_neighbours = np.argsort(pdists)
        k_nearest_neighbours = nearest_neighbours[:,:self.n_neighbours]
        
        # create a mask denoting the k nearest neighbours 
        k_nearest_mutual_neighbours_mask = np.zeros(pdists.shape, bool)
        for _mask_row, _neighbours_row in zip(k_nearest_mutual_neighbours_mask, k_nearest_neighbours):
            _mask_row[_neighbours_row] = True
            
        # and with transposed to remove non-mutual nearest neighbours
        k_nearest_mutual_neighbours_mask &= k_nearest_mutual_neighbours_mask.T
        
        graph_mtx = csr_matrix(k_nearest_mutual_neighbours_mask)

        #add links to denser neighbors to ensure connectedness
        n = len(instance)
        for idx in range(n):
            dense_links_counter = 0
            for nb_idx in nearest_neighbours[idx]:
                if density[nb_idx] > density[idx]:
                    graph_mtx[idx, nb_idx] = graph_mtx[nb_idx, idx] = True
                    dense_links_counter += 1
                if dense_links_counter >= self.n_dense_links: 
                    break
        pdists[~graph_mtx.todense()] = 0
        graph_dist_mtx = csr_matrix(pdists)
        return graph_mtx, graph_dist_mtx
    
    def transform_single(self, instance):
        graph_mtx, graph_dist_mtx = self.make_mutual_nearest_neighbour_graph(instance)
        graph = nx.from_scipy_sparse_matrix(graph_dist_mtx, edge_attribute='distance')
        for node_idx in graph.nodes(): graph.nodes[node_idx]['label'] = node_idx
        nx.set_edge_attributes(graph, values=self.edge_label, name='label')
        return graph
        
    def transform(self, data):
        graphs = [self.transform_single(instance) for instance in data]
        return graphs
    
    def fit_transform(self, data, targets=None):
        return self.fit(data, targets).transform(data)
        