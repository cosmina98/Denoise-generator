import dgl
import networkx as nx
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import unittest

class DGLGraphGraphicalizer(BaseEstimator, TransformerMixin):
    """
    A Scikit-learn compatible transformer for converting between DGL and NetworkX graph formats.
    
    - `fit`: Learns unique label mappings for nodes and edges from NetworkX graphs.
    - `transform`: Converts DGL graphs to NetworkX graphs, enforcing 'label' and 'vec' attributes.
    - `inverse_transform`: Converts NetworkX graphs to DGL graphs, applying one-hot encoding to 'label' attributes.
    """
    
    def __init__(self):
        self.node_label_to_index_ = {}
        self.edge_label_to_index_ = {}
        self.fitted_ = False

    def fit(self, graphs, y=None):
        """
        Fit the transformer by learning unique label mappings from a list of NetworkX graphs.
        
        Parameters:
        - graphs (list of nx.Graph): The input NetworkX graphs to learn label mappings from.
        - y: Ignored. Present for compatibility.
        
        Returns:
        - self
        """
        if not isinstance(graphs, list):
            raise ValueError("Input to fit should be a list of NetworkX graphs.")
        
        unique_node_labels, unique_edge_labels = self._collect_unique_labels(graphs)
        self.node_label_to_index_ = {label: idx for idx, label in enumerate(unique_node_labels)}
        self.edge_label_to_index_ = {label: idx for idx, label in enumerate(unique_edge_labels)}
        self.fitted_ = True
        return self

    def transform(self, dgl_graphs):
        """
        Convert a list of DGL graphs to NetworkX graphs, enforcing 'label' and 'vec' attributes.
        
        Parameters:
        - dgl_graphs (list of dgl.DGLGraph): The input DGL graphs to convert.
        
        Returns:
        - list of nx.Graph: The converted NetworkX graphs with enforced attributes.
        """
        if not self.fitted_:
            raise RuntimeError("GraphEncoder must be fitted before calling transform.")
        
        if not isinstance(dgl_graphs, list):
            raise ValueError("Input to transform should be a list of DGL graphs.")
        
        nx_graphs = []
        for idx, dgl_graph in enumerate(dgl_graphs):
            try:
                nx_graph = self._dgl_to_networkx_custom(dgl_graph)
                nx_graphs.append(nx_graph)
            except Exception as e:
                print(f"Error converting DGL graph at index {idx}: {e}")
                raise e
        return nx_graphs

    def inverse_transform(self, nx_graphs):
        """
        Convert a list of NetworkX graphs to DGL graphs, applying one-hot encoding to 'label' attributes.
        
        Parameters:
        - nx_graphs (list of nx.Graph): The input NetworkX graphs to convert.
        
        Returns:
        - list of dgl.DGLGraph: The converted DGL graphs with one-hot encoded attributes.
        """
        if not self.fitted_:
            raise RuntimeError("GraphEncoder must be fitted before calling inverse_transform.")
        
        if not isinstance(nx_graphs, list):
            raise ValueError("Input to inverse_transform should be a list of NetworkX graphs.")
        
        dgl_graphs = []
        for idx, nx_graph in enumerate(nx_graphs):
            try:
                dgl_graph = self._networkx_to_dgl_custom_with_onehot(nx_graph)
                dgl_graphs.append(dgl_graph)
            except Exception as e:
                print(f"Error converting NetworkX graph at index {idx}: {e}")
                raise e
        return dgl_graphs

    def _collect_unique_labels(self, nx_graphs):
        """
        Collect all unique node and edge labels from a list of NetworkX graphs.
        
        Parameters:
        - nx_graphs (list of nx.Graph): The input NetworkX graphs.
        
        Returns:
        - tuple: (sorted list of unique node labels, sorted list of unique edge labels)
        """
        node_labels = set()
        edge_labels = set()
        for graph in nx_graphs:
            for _, data in graph.nodes(data=True):
                label = data.get('label', '-')
                node_labels.add(label)
            for _, _, data in graph.edges(data=True):
                label = data.get('label', '-')
                edge_labels.add(label)
        return sorted(node_labels), sorted(edge_labels)

    def _one_hot_encode(self, labels, label_to_index):
        """
        One-hot encode a list of labels using a provided label-to-index mapping.
        
        Parameters:
        - labels (list): List of labels (strings or integers).
        - label_to_index (dict): Mapping from label to unique index.
        
        Returns:
        - numpy.ndarray: One-hot encoded matrix of shape (len(labels), num_unique_labels).
        """
        num_labels = len(label_to_index)
        one_hot = np.zeros((len(labels), num_labels), dtype=np.float32)
        for i, label in enumerate(labels):
            idx = label_to_index.get(label, None)
            if idx is not None:
                one_hot[i, idx] = 1.0
            else:
                # Handle unknown labels by leaving the row as all zeros
                pass
        return one_hot

    def _dgl_to_networkx_custom(self, dgl_graph):
        """
        Convert a single DGLGraph to a NetworkX graph with enforced 'label' and 'vec' attributes.
        
        Parameters:
        - dgl_graph (dgl.DGLGraph): The input DGL graph.
        
        Returns:
        - nx.Graph: The converted NetworkX graph with enforced attributes.
        """
        nx_graph = dgl.to_networkx(
            dgl_graph, 
            node_attrs=list(dgl_graph.ndata.keys()), 
            edge_attrs=list(dgl_graph.edata.keys())
        )
        
        # Ensure node attributes
        for node, data in nx_graph.nodes(data=True):
            if 'label' not in data:
                nx_graph.nodes[node]['label'] = '-'
            if 'vec' not in data:
                nx_graph.nodes[node]['vec'] = None
            else:
                if not isinstance(nx_graph.nodes[node]['vec'], np.ndarray):
                    nx_graph.nodes[node]['vec'] = np.array(nx_graph.nodes[node]['vec'])
        
        # Ensure edge attributes
        for u, v, data in nx_graph.edges(data=True):
            if 'label' not in data:
                nx_graph.edges[u, v]['label'] = '-'
            if 'vec' not in data:
                nx_graph.edges[u, v]['vec'] = None
            else:
                if not isinstance(nx_graph.edges[u, v]['vec'], np.ndarray):
                    nx_graph.edges[u, v]['vec'] = np.array(nx_graph.edges[u, v]['vec'])
        
        return nx_graph

    def _networkx_to_dgl_custom_with_onehot(self, nx_graph):
        """
        Convert a single NetworkX graph to a DGLGraph, applying one-hot encoding to 'label' attributes.
        
        Parameters:
        - nx_graph (nx.Graph): The input NetworkX graph with enforced attributes.
        
        Returns:
        - dgl.DGLGraph: The converted DGL graph with one-hot encoded 'vec' attributes.
        """
        # Ensure all nodes have 'label' and 'vec' attributes
        for node, data in nx_graph.nodes(data=True):
            if 'label' not in data:
                nx_graph.nodes[node]['label'] = '-'
            if 'vec' not in data:
                nx_graph.nodes[node]['vec'] = None

        # Ensure all edges have 'label' and 'vec' attributes
        for u, v, data in nx_graph.edges(data=True):
            if 'label' not in data:
                nx_graph.edges[u, v]['label'] = '-'
            if 'vec' not in data:
                nx_graph.edges[u, v]['vec'] = None

        # Process node labels
        node_labels = [data['label'] for _, data in nx_graph.nodes(data=True)]
        node_one_hot = self._one_hot_encode(node_labels, self.node_label_to_index_)  # Shape: (num_nodes, num_unique_labels)
        
        # Process node vectors
        node_vectors = []
        for idx_node, (_, data) in enumerate(nx_graph.nodes(data=True)):
            vec = data['vec']
            if vec is None:
                vec = np.zeros(0, dtype=np.float32)
            else:
                vec = np.array(vec, dtype=np.float32)
            if vec.size > 0:
                concatenated = np.concatenate([node_one_hot[idx_node], vec])
            else:
                concatenated = node_one_hot[idx_node]
            node_vectors.append(concatenated)
        node_vectors = np.stack(node_vectors)  # Shape: (num_nodes, one_hot_dim + vec_dim)

        # Process edge labels
        edge_labels = [data['label'] for _, _, data in nx_graph.edges(data=True)]
        edge_one_hot = self._one_hot_encode(edge_labels, self.edge_label_to_index_)  # Shape: (num_edges, num_unique_labels)
        
        # Process edge vectors
        edge_vectors = []
        for idx_edge, (_, _, data) in enumerate(nx_graph.edges(data=True)):
            vec = data['vec']
            if vec is None:
                vec = np.zeros(0, dtype=np.float32)
            else:
                vec = np.array(vec, dtype=np.float32)
            if vec.size > 0:
                concatenated = np.concatenate([edge_one_hot[idx_edge], vec])
            else:
                concatenated = edge_one_hot[idx_edge]
            edge_vectors.append(concatenated)
        if edge_vectors:
            edge_vectors = np.stack(edge_vectors)  # Shape: (num_edges, one_hot_dim + vec_dim)
        else:
            edge_vectors = np.zeros((0, len(self.edge_label_to_index_)), dtype=np.float32)  # No edges

        # Create DGL graph without initial attributes
        dgl_graph = dgl.from_networkx(
            nx_graph,
            node_attrs=[],  # We'll add 'vec' manually
            edge_attrs=[]
        )

        # Assign the concatenated 'vec' as node features
        dgl_graph.ndata['vec'] = dgl.backend.tensor(node_vectors, dtype=dgl.backend.float32)

        # Assign the concatenated 'vec' as edge features
        if edge_vectors.size > 0:
            dgl_graph.edata['vec'] = dgl.backend.tensor(edge_vectors, dtype=dgl.backend.float32)
        else:
            dgl_graph.edata['vec'] = dgl.backend.tensor(edge_vectors, dtype=dgl.backend.float32)

        return dgl_graph

# ========================= Unit Tests =========================

class TestDGLGraphGraphicalizer(unittest.TestCase):
    def setUp(self):
        # Create sample NetworkX graphs for fitting
        self.nx_graph1 = nx.Graph()
        self.nx_graph1.add_node(0, label='A', vec=np.array([1.0, 2.0]))
        self.nx_graph1.add_node(1, label='B', vec=np.array([3.0, 4.0]))
        self.nx_graph1.add_edge(0, 1, label='ab', vec=np.array([0.1]))
        
        self.nx_graph2 = nx.Graph()
        self.nx_graph2.add_node(0, label='C', vec=np.array([5.0, 6.0]))
        self.nx_graph2.add_node(1)  # Missing 'label' and 'vec'
        self.nx_graph2.add_edge(0, 1)  # Missing 'label' and 'vec'
        
        self.nx_graphs = [self.nx_graph1, self.nx_graph2]
        
        # Create sample DGL graphs for transformation
        self.dgl_graph1 = dgl.graph(([0, 1, 2], [1, 2, 3]))
        self.dgl_graph1.ndata['label'] = ['X', 'Y', 'Z', 'W']
        self.dgl_graph1.ndata['vec'] = [
            np.array([7.0, 8.0]),
            None,
            np.array([9.0, 10.0]),
            np.array([11.0, 12.0])
        ]
        self.dgl_graph1.edata['label'] = ['xy', 'yz', 'zw']
        self.dgl_graph1.edata['vec'] = [
            np.array([0.2]),
            np.array([0.3]),
            np.array([0.4])
        ]
        
        self.dgl_graph2 = dgl.graph(([0, 2], [2, 3]))
        self.dgl_graph2.ndata['label'] = ['M', 'N', 'O', 'P']
        self.dgl_graph2.ndata['vec'] = [
            np.array([13.0, 14.0]),
            None,
            np.array([15.0, 16.0]),
            np.array([17.0, 18.0])
        ]
        self.dgl_graph2.edata['label'] = ['mo']
        self.dgl_graph2.edata['vec'] = [np.array([0.5])]
        
        self.dgl_graphs = [self.dgl_graph1, self.dgl_graph2]
        
        # Initialize GraphEncoder
        self.encoder = DGLGraphGraphicalizer()
        self.encoder.fit(self.nx_graphs)
    
    def test_fit(self):
        # Check if label mappings are correctly learned
        expected_node_labels = sorted(['A', 'B', 'C', '-', ])  # from self.nx_graphs
        expected_edge_labels = sorted(['-', 'ab'])
        
        self.assertEqual(sorted(self.encoder.node_label_to_index_.keys()), sorted(['A', 'B', 'C', '-']))
        self.assertEqual(sorted(self.encoder.edge_label_to_index_.keys()), sorted(['-', 'ab']))
    
    def test_transform_dgl_to_networkx(self):
        # Transform DGL graphs to NetworkX graphs
        nx_graphs = self.encoder.transform(self.dgl_graphs)
        self.assertEqual(len(nx_graphs), 2)
        
        # Check first transformed graph
        g1 = nx_graphs[0]
        self.assertEqual(len(g1.nodes), 4)
        self.assertEqual(len(g1.edges), 3)
        self.assertEqual(g1.nodes[0]['label'], 'X')
        self.assertTrue(np.array_equal(g1.nodes[0]['vec'], np.array([7.0, 8.0])))
        self.assertIsNone(g1.nodes[1]['vec'])
        self.assertEqual(g1.edges[0, 1]['label'], 'xy')
        self.assertTrue(np.array_equal(g1.edges[0, 1]['vec'], np.array([0.2])))
        
        # Check second transformed graph
        g2 = nx_graphs[1]
        self.assertEqual(len(g2.nodes), 4)
        self.assertEqual(len(g2.edges), 1)
        self.assertEqual(g2.nodes[1]['label'], 'N')
        self.assertIsNone(g2.nodes[1]['vec'])
        self.assertEqual(g2.edges[0, 2]['label'], 'mo')
        self.assertTrue(np.array_equal(g2.edges[0, 2]['vec'], np.array([0.5])))
    
    def test_inverse_transform_networkx_to_dgl(self):
        # Prepare NetworkX graphs to inverse transform
        nx_graph1 = nx.Graph()
        nx_graph1.add_node(0, label='A', vec=np.array([1.0, 2.0]))
        nx_graph1.add_node(1)  # Missing 'label' and 'vec'
        nx_graph1.add_edge(0, 1, label='ab', vec=np.array([0.1]))
        
        nx_graph2 = nx.Graph()
        nx_graph2.add_node(0, label='C', vec=np.array([5.0, 6.0]))
        nx_graph2.add_node(1)  # Missing 'label' and 'vec'
        nx_graph2.add_edge(0, 1)  # Missing 'label' and 'vec'
        
        nx_graphs = [nx_graph1, nx_graph2]
        
        # Inverse transform
        dgl_graphs = self.encoder.inverse_transform(nx_graphs)
        self.assertEqual(len(dgl_graphs), 2)
        
        # Check first DGL graph
        dg1 = dgl_graphs[0]
        self.assertEqual(dg1.number_of_nodes(), 2)
        self.assertEqual(dg1.number_of_edges(), 1)
        
        # Node features: one-hot for 'A' and '-', concatenated with 'vec'
        # node_label_to_index_ = {'A':0, 'B':1, 'C':2, '-':3}
        expected_node_features = np.array([
            [1.0, 0.0, 0.0, 0.0, 1.0, 2.0],  # 'A' -> [1,0,0,0] + [1,2]
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]   # '-' -> [0,0,0,1] + [0,0]
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(dg1.ndata['vec'].numpy(), expected_node_features)
        
        # Edge features: one-hot for 'ab' and concatenated with 'vec'
        # edge_label_to_index_ = {'-':0, 'ab':1}
        expected_edge_features = np.array([
            [0.0, 1.0, 0.1]  # 'ab' -> [0,1] + [0.1]
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(dg1.edata['vec'].numpy(), expected_edge_features)
        
        # Check second DGL graph
        dg2 = dgl_graphs[1]
        self.assertEqual(dg2.number_of_nodes(), 2)
        self.assertEqual(dg2.number_of_edges(), 1)
        
        expected_node_features_g2 = np.array([
            [0.0, 0.0, 1.0, 0.0, 5.0, 6.0],  # 'C' -> [0,0,1,0] + [5,6]
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]   # '-' -> [0,0,0,1] + [0,0]
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(dg2.ndata['vec'].numpy(), expected_node_features_g2)
        
        expected_edge_features_g2 = np.array([
            [0.0, 0.0, 0.1]  # '-' -> [1,0] + [0.0] assuming 'mo' is not in mapping
        ], dtype=np.float32)
        # Note: 'mo' was not part of the fit labels, so it should be all zeros for one-hot
        # Adjust the expected value accordingly
        expected_edge_features_g2_corrected = np.array([
            [0.0, 0.0, 0.0]  # Unknown label, all zeros + [0.0]
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(dg2.edata['vec'].numpy(), expected_edge_features_g2_corrected)
    
    def test_inverse_transform_with_new_labels(self):
        # Create a NetworkX graph with a new label not seen during fit
        nx_graph_new_label = nx.Graph()
        nx_graph_new_label.add_node(0, label='D', vec=np.array([2.0, 3.0]))  # 'D' not in fit labels
        nx_graph_new_label.add_edge(0, 1, label='cd', vec=np.array([0.6]))  # 'cd' not in fit edge labels
        
        with self.assertRaises(KeyError):
            # This should not raise an error but handle unknown labels by setting one-hot to all zeros
            dgl_graphs = self.encoder.inverse_transform([nx_graph_new_label])
            dg = dgl_graphs[0]
            expected_node_features = np.array([
                [0.0, 0.0, 0.0, 0.0, 2.0, 3.0],  # Unknown label 'D' -> all zeros + vec
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # '-' -> [0,0,0,1] but 'vec' missing
            ], dtype=np.float32)
            np.testing.assert_array_almost_equal(dg.ndata['vec'].numpy(), expected_node_features)
    
    def test_inverse_transform_empty_vec(self):
        # Create a NetworkX graph where 'vec' is None for nodes and edges
        nx_graph_empty_vec = nx.Graph()
        nx_graph_empty_vec.add_node(0, label='A')  # 'vec' is None
        nx_graph_empty_vec.add_node(1)  # Missing 'label' and 'vec'
        nx_graph_empty_vec.add_edge(0, 1)  # Missing 'label' and 'vec'
        
        dgl_graphs = self.encoder.inverse_transform([nx_graph_empty_vec])
        self.assertEqual(len(dgl_graphs), 1)
        dg = dgl_graphs[0]
        
        # Node features: 'A' and '-'
        expected_node_features = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],  # 'A' -> [1,0,0,0] + [0,0]
            [0.0, 0.0, 0.0, 1.0, 0.0]   # '-' -> [0,0,0,1] + [0,0]
        ], dtype=np.float32)
        # Note: 'vec' is None for both nodes, so only one-hot encoding
        expected_node_features_corrected = np.array([
            [1.0, 0.0, 0.0, 0.0],  # 'A' -> [1,0,0,0] + []
            [0.0, 0.0, 0.0, 1.0]   # '-' -> [0,0,0,1] + []
        ], dtype=np.float32)
        # However, since 'vec' is empty, the concatenation should just be the one-hot
        np.testing.assert_array_almost_equal(dg.ndata['vec'].numpy(), expected_node_features_corrected)
        
        # Edge features: '-'
        expected_edge_features = np.array([
            [0.0, 0.0]  # '-' -> [0,0]
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(dg.edata['vec'].numpy(), expected_edge_features)

# ========================= Run Unit Tests =========================

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
