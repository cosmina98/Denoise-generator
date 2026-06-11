import networkx as nx
from sklearn.neighbors import kneighbors_graph


class NearestNeighborVectorGraphicalizer(object):
    def __init__(self, instance_n_neighbors, connectivity_n_neighbors, discretization_factor, attribute_key='vec'):
        self.instance_n_neighbors = instance_n_neighbors #the number of instances that will form each graph
        self.connectivity_n_neighbors = connectivity_n_neighbors #the numer of outgoing edges for each node 
        self.discretization_factor = discretization_factor
        self.attribute_key = attribute_key
        
    def fit(self, data, targets=None):
        return self
    
    def transform(self, data):
        #kneighbors_graph returns the sparse adjacency matrix 
        #instance.nonzero() returns teh idxs for rows and cols: since we consider one row at a time (an instance is a row of the adj mtx) we only look into the cols i.e. [1]
        neighborhoods_data = [data[instance.nonzero()[1]] for instance in kneighbors_graph(data, self.instance_n_neighbors, mode='connectivity', include_self=True)]
        graphs = []
        for neighborhood_data in neighborhoods_data:
            graph = nx.from_scipy_sparse_matrix(kneighbors_graph(neighborhood_data, self.connectivity_n_neighbors, mode='connectivity', include_self=True))
            for node_idx in graph.nodes(): 
                #note: since the kneighbors_graph returns the neighbors sorted by distance, the progressive id is an indicator of proximity, i.e. low id means closer to central instance
                #note: if discretization_factor = 1/n then we are going to have n nodes that have the same label
                graph.nodes[node_idx]['label'] = int(node_idx*self.discretization_factor)
                graph.nodes[node_idx][self.attribute_key] = neighborhood_data[node_idx]
            nx.set_edge_attributes(graph, '-','label')
            graphs.append(graph)
        return graphs

    def fit_transform(self, data, targets=None):
        return self.fit(data, targets).transform(data)