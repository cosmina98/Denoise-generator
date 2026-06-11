import networkx as nx


class StringGraphicalizer(object):
    def __init__(self, separator='', edge_label='-', start_label=None, end_label=None):
        self.separator = separator
        self.edge_label = edge_label
        self.start_label = start_label
        self.end_label = end_label
    
    def fit(self, data, targets=None):
        return self
    
    def transform(self, data):
        graphs = []
        for seq in data:
            graph = nx.Graph()
            if self.separator == '':
                seq_ = list(seq)
            else:
                seq_ = seq.split(self.separator)
            if self.start_label is not None:
                graph.add_node(0, label=self.start_label)
            for label in seq_:
                node_idx = nx.number_of_nodes(graph)
                graph.add_node(node_idx, label=label)
            if self.end_label is not None:
                node_idx = nx.number_of_nodes(graph)
                graph.add_node(node_idx, label=self.end_label)
            for node_idx in graph.nodes():
                if node_idx > 0:
                    graph.add_edge(node_idx-1, node_idx, label=self.edge_label)
            graphs.append(graph)
        return graphs

    def fit_transform(self, data, targets=None):
        return self.fit(data, targets).transform(data)


class SequenceGraphicalizer(object):
    def __init__(self, edge_label='-', start_label=None, end_label=None):
        self.edge_label = edge_label
        self.start_label = start_label
        self.end_label = end_label
    
    def fit(self, data, targets=None):
        return self
    
    def transform(self, data):
        graphs = []
        for seq in data:
            graph = nx.Graph()
            if self.start_label is not None:
                graph.add_node(0, label=self.start_label)
            for label in seq:
                node_idx = nx.number_of_nodes(graph)
                graph.add_node(node_idx, label=label)
            if self.end_label is not None:
                node_idx = nx.number_of_nodes(graph)
                graph.add_node(node_idx, label=self.end_label)
            for node_idx in graph.nodes():
                if node_idx > 0:
                    graph.add_edge(node_idx-1, node_idx, label=self.edge_label)
            graphs.append(graph)
        return graphs

    def fit_transform(self, data, targets=None):
        return self.fit(data, targets).transform(data)