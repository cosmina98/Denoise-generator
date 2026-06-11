import numpy as np
import networkx as nx
from toolz import partition_all
import multiprocessing_on_dill as mp
from coco_grape.module.vectorize import get_node_attributes_matrix, get_edge_attributes_matrix
from coco_grape.module.vectorize import node_vectorize, parallel_node_vectorize
from coco_grape.module.vectorize import edge_vectorize, parallel_edge_vectorize
from coco_grape.module.vectorize import vectorize as decomposition_vectorize
from coco_grape.module.vectorize import parallel_vectorize as parallel_decomposition_vectorize
import copy 


def get_graph_nodes_mtx_list(graphs_mtx, node_mtx_list):
    graph_nodes_mtx_list = []
    for graph_vec, node_mtx in zip(graphs_mtx, node_mtx_list):
        n_nodes = node_mtx.shape[0]
        graph_mtx = np.vstack(np.tile(graph_vec.todense().A.reshape(1,-1), (n_nodes,1)))
        graph_nodes_mtx = np.hstack([graph_mtx, node_mtx.todense().A])
        graph_nodes_mtx_list.append(graph_nodes_mtx)
    return graph_nodes_mtx_list

def graph_nodes(graphs, decomposition_function, nbits, parallel=True):
    if parallel: graphs_mtx = parallel_decomposition_vectorize(graphs, decomposition_function, nbits)
    else: graphs_mtx = decomposition_vectorize(graphs, decomposition_function, nbits)
    if parallel: node_mtx_list = parallel_node_vectorize(graphs, decomposition_function, nbits)
    else: node_mtx_list = node_vectorize(graphs, decomposition_function, nbits)
    return graphs_mtx, node_mtx_list

def graph_nodes_to_node_attribute(graphs, decomposition_function, nbits, parallel=True):
    graphs_mtx, node_mtx_list = graph_nodes(graphs, decomposition_function, nbits, parallel)
    graph_nodes_mtx_list = get_graph_nodes_mtx_list(graphs_mtx, node_mtx_list)
    graph_nodes_mtx = np.vstack(graph_nodes_mtx_list)
    
    node_attribute_mtx_list = [get_node_attributes_matrix(graph).todense().A for graph in graphs]
    node_attribute_mtx = np.vstack(node_attribute_mtx_list)
    return graph_nodes_mtx, node_attribute_mtx


class NodeAttributeGraphGraphicalizer(object):
    def __init__(self, regressor, decomposition_function, nbits, attribute_key='vec', data_type=int, parallel=True):
        self.regressor = copy.deepcopy(regressor)
        self.decomposition_function = decomposition_function
        self.nbits = nbits
        self.attribute_key = attribute_key
        self.data_type = data_type
        self.parallel = parallel
        
    def fit(self, graphs):
        graph_nodes_mtx, node_attribute_mtx = graph_nodes_to_node_attribute(graphs, self.decomposition_function, self.nbits, self.parallel)
        self.regressor.fit(graph_nodes_mtx, node_attribute_mtx)
        return self
    
    def transform(self, graphs):
        if self.parallel: return self.transform_parallel(graphs)
        else: return self.transform_serial(graphs)

    def transform_parallel(self, graphs):
        n_cpus = mp.cpu_count()
        batch_size = len(graphs)//n_cpus
        graphs_list = list(partition_all(batch_size, graphs))
        pool = mp.Pool(n_cpus)
        results = pool.map(self.transform_serial, graphs_list)
        pool.close()
        out_graphs = sum(results,[])
        return out_graphs

    def transform_serial(self, orig_graphs):
        graphs = [nx.Graph(graph) for graph in orig_graphs]
        graphs_mtx, node_mtx_list = graph_nodes(graphs, self.decomposition_function, self.nbits, False)
        for graph, node_mtx in zip(graphs, node_mtx_list): assert nx.number_of_nodes(graph)==node_mtx.shape[0]
        graph_nodes_mtx_list = get_graph_nodes_mtx_list(graphs_mtx, node_mtx_list)
        for graph, graph_nodes_mtx in zip(graphs, graph_nodes_mtx_list):
            node_attribute_mtx = self.regressor.predict(graph_nodes_mtx)    
            for node_idx, node_attribute in zip(graph.nodes(), node_attribute_mtx):
                graph.nodes[node_idx][self.attribute_key] = node_attribute.reshape(-1).astype(self.data_type)
        return graphs


#-------------------------------------------------------------------------------------------------------------------------------------------

def get_graph_edges_mtx_list(graphs_mtx, edge_mtx_list):
    graph_edges_mtx_list = []
    for graph_vec, edge_mtx in zip(graphs_mtx, edge_mtx_list):
        n_edges = edge_mtx.shape[0]
        graph_mtx = np.vstack(np.tile(graph_vec.todense().A.reshape(1,-1), (n_edges,1)))
        graph_edges_mtx = np.hstack([graph_mtx, edge_mtx.todense().A])
        graph_edges_mtx_list.append(graph_edges_mtx)
    return graph_edges_mtx_list

def graph_edges(graphs, decomposition_function, nbits, parallel=True):
    if parallel: graphs_mtx = parallel_decomposition_vectorize(graphs, decomposition_function, nbits)
    else: graphs_mtx = decomposition_vectorize(graphs, decomposition_function, nbits)
    if parallel: edge_mtx_list = parallel_edge_vectorize(graphs, decomposition_function, nbits)
    else: edge_mtx_list = edge_vectorize(graphs, decomposition_function, nbits)
    return graphs_mtx, edge_mtx_list

def graph_edges_to_edge_attribute(graphs, decomposition_function, nbits, parallel=True):
    graphs_mtx, edge_mtx_list = graph_edges(graphs, decomposition_function, nbits, parallel)
    graph_edges_mtx_list = get_graph_edges_mtx_list(graphs_mtx, edge_mtx_list)
    graph_edges_mtx = np.vstack(graph_edges_mtx_list)
    
    edge_attribute_mtx_list = [get_edge_attributes_matrix(graph).todense().A for graph in graphs]
    edge_attribute_mtx = np.vstack(edge_attribute_mtx_list)
    return graph_edges_mtx, edge_attribute_mtx


class EdgeAttributeGraphGraphicalizer(object):
    def __init__(self, regressor, decomposition_function, nbits, attribute_key='vec', data_type=int, parallel=True):
        self.regressor = copy.deepcopy(regressor)
        self.decomposition_function = decomposition_function
        self.nbits = nbits
        self.attribute_key = attribute_key
        self.data_type = data_type
        self.parallel = parallel
        
    def fit(self, graphs):
        graph_edges_mtx, edge_attribute_mtx = graph_edges_to_edge_attribute(graphs, self.decomposition_function, self.nbits, self.parallel)
        self.regressor.fit(graph_edges_mtx, edge_attribute_mtx)
        return self
    
    def transform(self, graphs):
        if self.parallel: return self.transform_parallel(graphs)
        else: return self.transform_serial(graphs)

    def transform_parallel(self, graphs):
        n_cpus = mp.cpu_count()
        batch_size = len(graphs)//n_cpus
        graphs_list = list(partition_all(batch_size, graphs))
        pool = mp.Pool(n_cpus)
        results = pool.map(self.transform_serial, graphs_list)
        pool.close()
        out_graphs = sum(results,[])
        return out_graphs

    def transform_serial(self, orig_graphs):
        graphs = [nx.Graph(graph) for graph in orig_graphs]
        graphs_mtx, edge_mtx_list = graph_edges(graphs, self.decomposition_function, self.nbits, False)
        for graph, edge_mtx in zip(graphs, edge_mtx_list): assert nx.number_of_edges(graph)==edge_mtx.shape[0]
        graph_edges_mtx_list = get_graph_edges_mtx_list(graphs_mtx, edge_mtx_list)
        for graph, graph_edges_mtx in zip(graphs, graph_edges_mtx_list):
            edge_attribute_mtx = self.regressor.predict(graph_edges_mtx)    
            for edge_idx, edge_attribute in zip(graph.edges(), edge_attribute_mtx):
                graph.edges[edge_idx][self.attribute_key] = edge_attribute.reshape(-1).astype(self.data_type)
        return graphs

#-------------------------------------------------------------------------------------------------------------------------------------------

class AttributeGraphGraphicalizer(object):
    def __init__(self, regressor, decomposition_function, nbits, attribute_key='vec', data_type=int, parallel=True):
        self.node_attribute_graph_graphicalizer = NodeAttributeGraphGraphicalizer(regressor, decomposition_function, nbits, attribute_key, data_type, parallel)
        self.edge_attribute_graph_graphicalizer = EdgeAttributeGraphGraphicalizer(regressor, decomposition_function, nbits, attribute_key, data_type, parallel)

    def fit(self, graphs):
        self.node_attribute_graph_graphicalizer.fit(graphs)
        self.edge_attribute_graph_graphicalizer.fit(graphs)
        return self

    def transform(self, graphs):
        node_graphs = self.node_attribute_graph_graphicalizer.transform(graphs)
        node_edge_graphs = self.edge_attribute_graph_graphicalizer.transform(node_graphs)
        return node_edge_graphs
