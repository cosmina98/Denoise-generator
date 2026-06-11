import numpy as np
import networkx as nx
from coco_grape.module.vectorize import node_vectorize, get_node_attributes_matrix, parallel_node_vectorize, structures_vectorize, parallel_structures_vectorize


class LabelEncoder(object):
    def __init__(self, use_one_hot_encoding=True):
        self.use_one_hot_encoding = use_one_hot_encoding
        self.label_to_code_map = None
        self.classes_ = None
        self.num_classes = None
    
    def fit(self, seq):
        # Note: reserve last available code (i.e. self.num_classes-1) for unknown labels
        self.label_to_code_map = {str(s):i for i,s in enumerate(sorted(set(seq)))}
        self.num_classes = len(set(seq))+1
        self.classes_ = list(range(self.num_classes))
        self.unknown_label_code = self.num_classes - 1
        self.inverse_label_to_code_map = {v:k for k,v in self.label_to_code_map.items()}
        self.inverse_label_to_code_map[self.unknown_label_code] = self.unknown_label_code
        return self

    def one_hot_encoding(self, value):
        x = np.zeros(self.num_classes)
        x[value] = 1
        return x

    def inverse_one_hot_encoding(self, value):
        x = np.argmax(value)
        return x

    def encode(self, value):
        x = self.label_to_code_map.get(str(value), self.unknown_label_code)
        if self.use_one_hot_encoding:
            return self.one_hot_encoding(x)
        else:
            return x

    def inverse_encode(self, value):
        x = self.inverse_label_to_code_map.get(value, self.unknown_label_code)
        if self.use_one_hot_encoding:
            return self.inverse_one_hot_encoding(x)
        else:
            return x

    def transform(self, seq):
        return [self.encode(s) for s in seq]

    def inverse_transform(self, seq):
        return [self.inverse_encode(s) for s in seq]


class GraphOneHotLabelEncoder(object):
    def __init__(
        self,
        attribute_label='vec',
        append=True,
        encode_node_labels=True,
        encode_edge_labels=True,
    ):
        self.attribute_label = attribute_label
        self.node_one_hot_encoder = LabelEncoder(use_one_hot_encoding=True)
        self.edge_one_hot_encoder = LabelEncoder(use_one_hot_encoding=True)
        self.append = append
        self.encode_node_labels = bool(encode_node_labels)
        self.encode_edge_labels = bool(encode_edge_labels)
    
    def fit(self, graphs):
        node_labels = [graph.nodes[node_id]['label'] for graph in graphs for node_id in graph.nodes()]
        self.node_one_hot_encoder.fit(node_labels if node_labels else ['__UNLABELED__'])
        edge_labels = [graph.edges[edge_id]['label'] for graph in graphs for edge_id in graph.edges()]
        self.edge_one_hot_encoder.fit(edge_labels if edge_labels else ['__UNLABELED__'])
        return self
    
    def transform(self, graphs):
        graphs_ = []
        for graph in graphs:
            graph_ = graph.copy()
            #nodes
            if self.encode_node_labels and graph.number_of_nodes() > 0:
                labels = [graph.nodes[node_id]['label'] for node_id in graph.nodes()]
                one_hot_encoded_labels = self.node_one_hot_encoder.transform(labels)
                for node_id, one_hot_encoded_label in zip(graph.nodes(), one_hot_encoded_labels):
                    if self.append:
                        if self.attribute_label in graph_.nodes[node_id]:
                            graph_.nodes[node_id][self.attribute_label] = np.hstack([graph_.nodes[node_id][self.attribute_label].flatten(),one_hot_encoded_label]).flatten()
                        else: graph_.nodes[node_id][self.attribute_label] = one_hot_encoded_label
                    else: graph_.nodes[node_id][self.attribute_label] = one_hot_encoded_label
            #edges
            if self.encode_edge_labels and graph.number_of_edges() > 0:
                labels = [graph.edges[edge_id]['label'] for edge_id in graph.edges()]
                one_hot_encoded_labels = self.edge_one_hot_encoder.transform(labels)
                for edge_id, one_hot_encoded_label in zip(graph.edges(), one_hot_encoded_labels):
                    if self.append:
                        if self.attribute_label in graph_.edges[edge_id]:
                            graph_.edges[edge_id][self.attribute_label] = np.hstack([graph_.edges[edge_id][self.attribute_label].flatten(),one_hot_encoded_label]).flatten()
                        else: graph_.edges[edge_id][self.attribute_label] = one_hot_encoded_label
                    else: graph_.edges[edge_id][self.attribute_label] = one_hot_encoded_label
            graphs_.append(graph_)
        return graphs_

    def fit_transform(self, graphs):
        return self.fit(graphs).transform(graphs)


class DecompositionalNodeVectorizer(object):
    def __init__(self, 
                 decomposition_function, 
                 nbits=7, 
                 node_attribute_key='vec', 
                 parallel=True,
                 encode_node_labels=True,
                 encode_edge_labels=True):
        self.decomposition_function = decomposition_function
        self.nbits = nbits
        self.node_attribute_key = node_attribute_key
        self.parallel = parallel
        self.encode_node_labels = bool(encode_node_labels)
        self.encode_edge_labels = bool(encode_edge_labels)
        
    def fit(self, graphs, targets=None):
        self.max_n_nodes = max(nx.number_of_nodes(graph) for graph in graphs)
        self.graph_one_hot_label_encoder = GraphOneHotLabelEncoder(
            attribute_label=self.node_attribute_key,
            append=True,
            encode_node_labels=self.encode_node_labels,
            encode_edge_labels=self.encode_edge_labels,
        ).fit(graphs)
        self.data_shape = self.transform([graphs[0]])[0].shape
        return self
    
    def get_data_shape(self):
        return self.data_shape

    def get_node_attributes_matrix_list(self, graphs):
        return [get_node_attributes_matrix(graph, attribute_label=self.node_attribute_key) for graph in graphs]
    
    def pad_n_nodes(self, mtx):
        if self.max_n_nodes < mtx.shape[0]:
            mtx = mtx[:self.max_n_nodes,:]
        else:
            new_n_rows = max(0,self.max_n_nodes-mtx.shape[0])
            mtx = np.pad(mtx, ((0,new_n_rows),(0,0)))
        return mtx
    
    def todense_and_pad_n_nodes(self, mtx_list):
        dense_mtx = np.array([self.pad_n_nodes(mtx.todense()) for mtx in mtx_list])
        return dense_mtx

    def concatenate_node_attribute(self, graphs, dense_mtx):
        dense_attributes_mtx_list = self.get_node_attributes_matrix_list(graphs)
        dense_attributes_mtx = self.todense_and_pad_n_nodes(dense_attributes_mtx_list)
        dense_mtx = np.concatenate([dense_mtx, dense_attributes_mtx], axis=2)
        return dense_mtx

    def transform(self, graphs):
        one_hot_label_encoded_graphs = self.graph_one_hot_label_encoder.transform(graphs)
        if self.parallel: mtx_list = parallel_node_vectorize(one_hot_label_encoded_graphs, decomposition_function=self.decomposition_function, nbits=self.nbits)
        else: mtx_list = node_vectorize(one_hot_label_encoded_graphs, decomposition_function=self.decomposition_function, nbits=self.nbits)
        structure_mtx = self.todense_and_pad_n_nodes(mtx_list)
        dense_mtx = self.concatenate_node_attribute(graphs, structure_mtx)
        return dense_mtx

    def fit_transform(self, graphs, targets=None):
        return self.fit(graphs, targets).transform(graphs)


#------------------------------------------------------------------------------------------------------------------------------------------------------
class DecompositionalElementVectorizer(object):
    def __init__(self, 
                 decomposition_function, 
                 nbits=7, 
                 attribute_key='vec', 
                 parallel=True):
        self.decomposition_function = decomposition_function
        self.nbits = nbits
        self.attribute_key = attribute_key
        self.parallel = parallel
        
    def fit(self, graphs, targets=None):
        self.max_n_nodes = max(nx.number_of_nodes(graph) for graph in graphs)
        self.max_n_edges = max(nx.number_of_edges(graph) for graph in graphs)
        self.graph_one_hot_label_encoder = GraphOneHotLabelEncoder(attribute_label=self.attribute_key, append=True).fit(graphs)
        self.data_shape = [mtx.shape for mtx in self.transform([graphs[0]])[0]]
        return self
    
    def get_data_shape(self):
        return self.data_shape

    def pad_n_nodes(self, mtx, size):
        if size < mtx.shape[0]:
            mtx = mtx[:size,:]
        else:
            new_n_rows = max(0,size-mtx.shape[0])
            mtx = np.pad(mtx, ((0,new_n_rows),(0,0)))
        return mtx
    
    def todense_and_pad(self, mtxs_list):
        dense_mtxs_list = []
        for node_structure_mtx, node_attribute_mtx, edge_structure_mtx, edge_attribute_mtx in mtxs_list:
            dense_node_structure_mtx = self.pad_n_nodes(node_structure_mtx.todense(), self.max_n_nodes)
            dense_node_attribute_mtx = self.pad_n_nodes(node_attribute_mtx.todense(), self.max_n_nodes)
            dense_edge_structure_mtx = self.pad_n_nodes(edge_structure_mtx.todense(), self.max_n_edges)
            dense_edge_attribute_mtx = self.pad_n_nodes(edge_attribute_mtx.todense(), self.max_n_edges)
            dense_mtxs_list.append([dense_node_structure_mtx, dense_node_attribute_mtx, dense_edge_structure_mtx, dense_edge_attribute_mtx])
        return dense_mtxs_list

    def transform(self, graphs):
        one_hot_label_encoded_graphs = self.graph_one_hot_label_encoder.transform(graphs)
        if self.parallel: mtxs_list = parallel_structures_vectorize(one_hot_label_encoded_graphs, decomposition_function=self.decomposition_function, nbits=self.nbits)
        else: mtxs_list = structures_vectorize(one_hot_label_encoded_graphs, decomposition_function=self.decomposition_function, nbits=self.nbits)
        dense_mtxs_list = self.todense_and_pad(mtxs_list)
        return dense_mtxs_list

    def fit_transform(self, graphs, targets=None):
        return self.fit(graphs, targets).transform(graphs)
