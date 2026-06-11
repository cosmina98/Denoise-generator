import networkx as nx
import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
import unittest

class TensorGraphGraphicalizer(BaseEstimator, TransformerMixin):
    """
    A Scikit-learn compatible transformer for converting between Tensor-based (PyTorch) and NetworkX graph formats.
    
    - `fit`: Learns unique label mappings for nodes and edges from NetworkX graphs.
    - `transform`: Converts Tensor-based graphs to NetworkX graphs, enforcing 'label' and 'vec' attributes.
    - `inverse_transform`: Converts NetworkX graphs to Tensor-based graphs, applying one-hot encoding to 'label' attributes.
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

    def transform(self, tensor_graphs):
        """
        Convert a list of Tensor-based graphs to NetworkX graphs, enforcing 'label' and 'vec' attributes.
        
        Parameters:
        - tensor_graphs (list of dict): The input Tensor-based graphs to convert. Each graph should be a dict with keys:
            - 'node_features': torch.Tensor of shape (num_nodes, node_feature_dim)
            - 'edge_index': torch.LongTensor of shape (2, num_edges)
            - 'edge_features': torch.Tensor of shape (num_edges, edge_feature_dim)
            - 'node_labels': list of labels (strings or integers) of length num_nodes
            - 'edge_labels': list of labels (strings or integers) of length num_edges
        
        Returns:
        - list of nx.Graph: The converted NetworkX graphs with enforced attributes.
        """
        if not self.fitted_:
            raise RuntimeError("TensorGraphGraphicalizer must be fitted before calling transform.")
        
        if not isinstance(tensor_graphs, list):
            raise ValueError("Input to transform should be a list of Tensor-based graph dictionaries.")
        
        nx_graphs = []
        for idx, tensor_graph in enumerate(tensor_graphs):
            try:
                nx_graph = self._tensor_to_networkx(tensor_graph)
                nx_graphs.append(nx_graph)
            except Exception as e:
                print(f"Error converting Tensor graph at index {idx}: {e}")
                raise e
        return nx_graphs

    def inverse_transform(self, nx_graphs):
        """
        Convert a list of NetworkX graphs to Tensor-based graphs, applying one-hot encoding to 'label' attributes.
        
        Parameters:
        - nx_graphs (list of nx.Graph): The input NetworkX graphs to convert.
        
        Returns:
        - list of dict: The converted Tensor-based graphs with one-hot encoded attributes. Each graph is a dict with keys:
            - 'node_features': torch.Tensor of shape (num_nodes, one_hot_dim_node + vec_dim)
            - 'edge_index': torch.LongTensor of shape (2, num_edges)
            - 'edge_features': torch.Tensor of shape (num_edges, one_hot_dim_edge + vec_dim)
        """
        if not self.fitted_:
            raise RuntimeError("TensorGraphGraphicalizer must be fitted before calling inverse_transform.")
        
        if not isinstance(nx_graphs, list):
            raise ValueError("Input to inverse_transform should be a list of NetworkX graphs.")
        
        tensor_graphs = []
        for idx, nx_graph in enumerate(nx_graphs):
            try:
                tensor_graph = self._networkx_to_tensor(nx_graph)
                tensor_graphs.append(tensor_graph)
            except Exception as e:
                print(f"Error converting NetworkX graph at index {idx}: {e}")
                raise e
        return tensor_graphs

    def _collect_unique_labels(self, nx_graphs):
        """
        Collect all unique labels from nodes and edges across all NetworkX graphs.
        
        Parameters:
        - nx_graphs (list of nx.Graph): List of NetworkX graphs.
        
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

    def _tensor_to_networkx(self, tensor_graph):
        """
        Convert a single Tensor-based graph to a NetworkX graph with enforced 'label' and 'vec' attributes.
        
        Parameters:
        - tensor_graph (dict): The input Tensor-based graph with keys:
            - 'node_features': torch.Tensor of shape (num_nodes, node_feature_dim)
            - 'edge_index': torch.LongTensor of shape (2, num_edges)
            - 'edge_features': torch.Tensor of shape (num_edges, edge_feature_dim)
            - 'node_labels': list of labels (strings or integers) of length num_nodes
            - 'edge_labels': list of labels (strings or integers) of length num_edges
        
        Returns:
        - nx.Graph: The converted NetworkX graph with enforced attributes.
        """
        # Extract components
        node_features = tensor_graph.get('node_features')
        edge_index = tensor_graph.get('edge_index')
        edge_features = tensor_graph.get('edge_features')
        node_labels = tensor_graph.get('node_labels', [])
        edge_labels = tensor_graph.get('edge_labels', [])
        
        num_nodes = node_features.shape[0]
        num_edges = edge_index.shape[1]
        
        if len(node_labels) != num_nodes:
            raise ValueError("Length of 'node_labels' must match number of nodes in 'node_features'.")
        if len(edge_labels) != num_edges:
            raise ValueError("Length of 'edge_labels' must match number of edges in 'edge_index'.")
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes with attributes
        for i in range(num_nodes):
            label = node_labels[i] if node_labels[i] is not None else '-'
            vec = node_features[i].numpy() if node_features[i] is not None else None
            G.add_node(i, label=label, vec=vec)
        
        # Add edges with attributes
        for i in range(num_edges):
            u, v = edge_index[:, i].tolist()
            label = edge_labels[i] if edge_labels[i] is not None else '-'
            vec = edge_features[i].numpy() if edge_features[i] is not None else None
            G.add_edge(u, v, label=label, vec=vec)
        
        return G

    def _networkx_to_tensor(self, nx_graph):
        """
        Convert a single NetworkX graph to a Tensor-based graph with one-hot encoded 'label' attributes.
        
        Parameters:
        - nx_graph (nx.Graph): The input NetworkX graph with enforced 'label' and 'vec' attributes.
        
        Returns:
        - dict: The converted Tensor-based graph with keys:
            - 'node_features': torch.Tensor of shape (num_nodes, one_hot_dim_node + vec_dim)
            - 'edge_index': torch.LongTensor of shape (2, num_edges)
            - 'edge_features': torch.Tensor of shape (num_edges, one_hot_dim_edge + vec_dim)
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
        
        # Process node labels and one-hot encode
        node_labels = [data['label'] for _, data in nx_graph.nodes(data=True)]
        node_one_hot = self._one_hot_encode(node_labels, self.node_label_to_index_)  # Shape: (num_nodes, num_unique_labels_node)
        
        # Process node vectors
        node_vecs = []
        for _, data in nx_graph.nodes(data=True):
            vec = data['vec']
            if vec is None:
                vec = np.zeros(0, dtype=np.float32)
            else:
                vec = np.array(vec, dtype=np.float32)
            if vec.size > 0:
                concatenated = np.concatenate([node_one_hot[len(node_vecs)], vec])
            else:
                concatenated = node_one_hot[len(node_vecs)]
            node_vecs.append(concatenated)
        node_features = np.stack(node_vecs)  # Shape: (num_nodes, one_hot_dim_node + vec_dim)
        node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
        
        # Process edge labels and one-hot encode
        edge_labels = [data['label'] for _, _, data in nx_graph.edges(data=True)]
        edge_one_hot = self._one_hot_encode(edge_labels, self.edge_label_to_index_)  # Shape: (num_edges, num_unique_labels_edge)
        
        # Process edge vectors
        edge_vecs = []
        for i, (_, _, data) in enumerate(nx_graph.edges(data=True)):
            vec = data['vec']
            if vec is None:
                vec = np.zeros(0, dtype=np.float32)
            else:
                vec = np.array(vec, dtype=np.float32)
            if vec.size > 0:
                concatenated = np.concatenate([edge_one_hot[i], vec])
            else:
                concatenated = edge_one_hot[i]
            edge_vecs.append(concatenated)
        if edge_vecs:
            edge_features = np.stack(edge_vecs)  # Shape: (num_edges, one_hot_dim_edge + vec_dim)
        else:
            edge_features = np.zeros((0, len(self.edge_label_to_index_)), dtype=np.float32)  # No edges
        edge_features_tensor = torch.tensor(edge_features, dtype=torch.float32)
        
        # Process edge indices
        edge_indices = list(nx_graph.edges())
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()  # Shape: (2, num_edges)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Construct the tensor graph dictionary
        tensor_graph = {
            'node_features': node_features_tensor,
            'edge_index': edge_index,
            'edge_features': edge_features_tensor
        }
        
        return tensor_graph

# ========================= Unit Tests =========================

class TestTensorGraphGraphicalizer(unittest.TestCase):
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
        
        # Create sample Tensor-based graphs for transformation
        self.tensor_graph1 = {
            'node_features': torch.tensor([
                [7.0, 8.0],
                [9.0, 10.0],
                [11.0, 12.0]
            ], dtype=torch.float32),
            'edge_index': torch.tensor([
                [0, 1],
                [1, 2]
            ], dtype=torch.long),
            'edge_features': torch.tensor([
                [0.2],
                [0.3]
            ], dtype=torch.float32),
            'node_labels': ['X', 'Y', 'Z'],
            'edge_labels': ['xy', 'yz']
        }
        
        self.tensor_graph2 = {
            'node_features': torch.tensor([
                [13.0, 14.0],
                [15.0, 16.0],
                [17.0, 18.0],
                [19.0, 20.0]
            ], dtype=torch.float32),
            'edge_index': torch.tensor([
                [0, 2],
                [2, 3]
            ], dtype=torch.long),
            'edge_features': torch.tensor([
                [0.4],
                [0.5]
            ], dtype=torch.float32),
            'node_labels': ['M', 'N', 'O', 'P'],
            'edge_labels': ['mo', 'op']
        }
        
        self.tensor_graphs = [self.tensor_graph1, self.tensor_graph2]
        
        # Initialize TensorGraphGraphicalizer
        self.encoder = TensorGraphGraphicalizer()
        self.encoder.fit(self.nx_graphs)
    
    def test_fit(self):
        # Check if label mappings are correctly learned
        expected_node_labels = sorted(['A', 'B', 'C', '-'])  # from self.nx_graphs
        expected_edge_labels = sorted(['-', 'ab'])
        
        self.assertEqual(sorted(self.encoder.node_label_to_index_.keys()), expected_node_labels)
        self.assertEqual(sorted(self.encoder.edge_label_to_index_.keys()), expected_edge_labels)
    
    def test_transform_tensor_to_networkx(self):
        # Transform Tensor-based graphs to NetworkX graphs
        nx_transformed = self.encoder.transform(self.tensor_graphs)
        self.assertEqual(len(nx_transformed), 2)
        
        # Check first transformed graph
        g1 = nx_transformed[0]
        self.assertEqual(len(g1.nodes), 3)
        self.assertEqual(len(g1.edges), 2)
        self.assertEqual(g1.nodes[0]['label'], 'X')
        self.assertTrue(np.array_equal(g1.nodes[0]['vec'], np.array([7.0, 8.0])))
        self.assertEqual(g1.edges[0, 1]['label'], 'xy')
        self.assertTrue(np.array_equal(g1.edges[0, 1]['vec'], np.array([0.2])))
        
        # Check second transformed graph
        g2 = nx_transformed[1]
        self.assertEqual(len(g2.nodes), 4)
        self.assertEqual(len(g2.edges), 2)
        self.assertEqual(g2.nodes[1]['label'], 'N')  # Node 1
        self.assertTrue(np.array_equal(g2.nodes[1]['vec'], np.array([15.0, 16.0])))
        self.assertEqual(g2.edges[2, 3]['label'], 'op')
        self.assertTrue(np.array_equal(g2.edges[2, 3]['vec'], np.array([0.5])))
    
    def test_inverse_transform_networkx_to_tensor(self):
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
        tensor_graphs = self.encoder.inverse_transform(nx_graphs)
        self.assertEqual(len(tensor_graphs), 2)
        
        # Check first tensor graph
        tg1 = tensor_graphs[0]
        expected_node_features = np.array([
            [1.0, 0.0, 1.0, 2.0],  # 'A' -> [1,0,0] + [1,2] assuming 'A' is index 0
            [0.0, 0.0, 0.0, 0.0]   # '-' -> [0,0,1] + [0,0] assuming '-' is index 3
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(tg1['node_features'].numpy(), expected_node_features)
        
        expected_edge_features = np.array([
            [0.0, 1.0, 0.1]  # 'ab' -> [0,1] + [0.1] assuming 'ab' is index 1
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(tg1['edge_features'].numpy(), expected_edge_features)
        
        # Check second tensor graph
        tg2 = tensor_graphs[1]
        expected_node_features_g2 = np.array([
            [0.0, 0.0, 0.0, 0.0, 5.0, 6.0],  # 'C' -> [0,0,1,0] + [5,6]
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # '-' -> [0,0,1,0] + [0,0]
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(tg2['node_features'].numpy(), expected_node_features_g2)
        
        expected_edge_features_g2 = np.array([
            [0.0, 0.0, 0.0]  # '-' -> [0,0] + [0.0]
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(tg2['edge_features'].numpy(), expected_edge_features_g2)
    
    def test_one_hot_encoding_consistency(self):
        # Ensure that one-hot encoding is consistent across graphs based on the learned mappings
        dgl_graphs = self.encoder.transform(self.tensor_graphs)
        inverse_tensor_graphs = self.encoder.inverse_transform(dgl_graphs)
        
        # Verify node one-hot consistency
        for original, transformed in zip(self.tensor_graphs, inverse_tensor_graphs):
            original_labels = original['node_labels']
            transformed_features = transformed['node_features'].numpy()
            for i, label in enumerate(original_labels):
                if label in self.encoder.node_label_to_index_:
                    idx = self.encoder.node_label_to_index_[label]
                    expected_one_hot = np.zeros(len(self.encoder.node_label_to_index_), dtype=np.float32)
                    expected_one_hot[idx] = 1.0
                    # Extract one-hot from the transformed features
                    one_hot = transformed_features[i, :len(self.encoder.node_label_to_index_)]
                    np.testing.assert_array_almost_equal(one_hot, expected_one_hot)
                else:
                    # Unknown label, one-hot should be all zeros
                    one_hot = transformed_features[i, :len(self.encoder.node_label_to_index_)]
                    expected_one_hot = np.zeros(len(self.encoder.node_label_to_index_), dtype=np.float32)
                    np.testing.assert_array_almost_equal(one_hot, expected_one_hot)
        
        # Similarly, verify edge one-hot consistency
        for original, transformed in zip(self.tensor_graphs, inverse_tensor_graphs):
            original_labels = original['edge_labels']
            transformed_features = transformed['edge_features'].numpy()
            for i, label in enumerate(original_labels):
                if label in self.encoder.edge_label_to_index_:
                    idx = self.encoder.edge_label_to_index_[label]
                    expected_one_hot = np.zeros(len(self.encoder.edge_label_to_index_), dtype=np.float32)
                    expected_one_hot[idx] = 1.0
                    # Extract one-hot from the transformed features
                    one_hot = transformed_features[i, :len(self.encoder.edge_label_to_index_)]
                    np.testing.assert_array_almost_equal(one_hot, expected_one_hot)
                else:
                    # Unknown label, one-hot should be all zeros
                    one_hot = transformed_features[i, :len(self.encoder.edge_label_to_index_)]
                    expected_one_hot = np.zeros(len(self.encoder.edge_label_to_index_), dtype=np.float32)
                    np.testing.assert_array_almost_equal(one_hot, expected_one_hot)
    
    def test_inverse_transform_with_new_labels(self):
        # Create a NetworkX graph with new labels not seen during fit
        nx_graph_new_label = nx.Graph()
        nx_graph_new_label.add_node(0, label='D', vec=np.array([2.0, 3.0]))  # 'D' not in fit labels
        nx_graph_new_label.add_node(1)  # Missing 'label' and 'vec'
        nx_graph_new_label.add_edge(0, 1, label='cd', vec=np.array([0.6]))  # 'cd' not in fit edge labels
        
        # Inverse transform
        tensor_graphs = self.encoder.inverse_transform([nx_graph_new_label])
        self.assertEqual(len(tensor_graphs), 1)
        tg = tensor_graphs[0]
        
        # Node features: one-hot for 'D' should be all zeros, concatenated with 'vec'
        # node_label_to_index_ = {'A':0, 'B':1, 'C':2, '-':3}
        expected_node_features = np.array([
            [0.0, 0.0, 0.0, 0.0, 2.0, 3.0],  # 'D' -> [0,0,0,0] + [2,3]
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]   # '-' -> [0,0,0,1] + [0,0]
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(tg['node_features'].numpy(), expected_node_features)
        
        # Edge features: 'cd' not in fit edge labels, one-hot should be all zeros, concatenated with 'vec'
        expected_edge_features = np.array([
            [0.0, 0.0, 0.6]  # 'cd' -> [0,0] + [0.6]
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(tg['edge_features'].numpy(), expected_edge_features)
    
    def test_inverse_transform_empty_vec(self):
        # Create a NetworkX graph where 'vec' is None for nodes and edges
        nx_graph_empty_vec = nx.Graph()
        nx_graph_empty_vec.add_node(0, label='A')  # 'vec' is None
        nx_graph_empty_vec.add_node(1)  # Missing 'label' and 'vec'
        nx_graph_empty_vec.add_edge(0, 1)  # Missing 'label' and 'vec'
        
        # Inverse transform
        tensor_graphs = self.encoder.inverse_transform([nx_graph_empty_vec])
        self.assertEqual(len(tensor_graphs), 1)
        tg = tensor_graphs[0]
        
        # Node features: 'A' and '-', concatenated with 'vec' which is empty
        # 'A' has one-hot [1,0,0,0] and 'vec' [0,0]
        expected_node_features = np.array([
            [1.0, 0.0, 0.0, 0.0],  # 'A' -> [1,0,0,0] + [] since 'vec' is None
            [0.0, 0.0, 0.0, 1.0]   # '-' -> [0,0,0,1] + [] since 'vec' is None
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(tg['node_features'].numpy(), expected_node_features)
        
        # Edge features: '-', concatenated with 'vec' which is empty
        expected_edge_features = np.array([
            [0.0, 0.0]  # '-' -> [0,0] + []
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(tg['edge_features'].numpy(), expected_edge_features)

# ========================= Run Unit Tests =========================

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
