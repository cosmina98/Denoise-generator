import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

class QuotientGraphDataConverter:
    """
    A scikit-learn style transformer that processes lists of quotient graphs.

    Each quotient graph is a tuple (G_pre, G_img) where:
      - G_pre is a preimage graph (a networkx Graph) with nodes/edges having:
          'label': a discrete label,
          'vec': a vector attribute.
      - G_img is an image graph (a networkx Graph) that, in addition to the keys above,
          uses the key 'subgraph' in each node to store a copy of a subgraph (a networkx Graph)
          corresponding to the nodes (by id) in the preimage graph.

    The converter computes the following from the dataset:
      - Maximum number of nodes in preimage and image graphs.
      - Maximum number of edges in preimage and image graphs.
      - Maximum length of node attribute vectors and edge attribute vectors.
      - Maximum number of nodes in any subgraph from image graph nodes.
      - A single LabelEncoder for node labels (across preimage and image graphs).
      - A single LabelEncoder for edge labels.
    
    The transform method returns three types of tensors:
      1. Preimage graph data (dictionary with padded node labels/vec, edge_index, edge labels/vec).
      2. Image graph data (same structure as for preimage).
      3. Link data: for each image node (padded to max_image_nodes) a list of node indices (padded up to max_subgraph_nodes)
         that indicate which nodes in the preimage graph belong to the subgraph.
    """
    def __init__(self):
        # These will be determined in fit.
        self.max_preimage_nodes = None
        self.max_image_nodes = None
        self.max_preimage_edges = None
        self.max_image_edges = None
        self.max_node_attr_size = 0
        self.max_edge_attr_size = 0
        self.max_subgraph_nodes = None
        self.node_label_encoder = LabelEncoder()
        self.edge_label_encoder = LabelEncoder()
    
    def fit(self, X, y=None):
        """
        Fit the converter on a list of quotient graphs.

        Parameters
        ----------
        X : list
            Each element is a tuple (G_pre, G_img) where G_pre and G_img are networkx graphs.
        y : array-like, default=None
            Target vector (not used in this converter, but present for scikit-learn compatibility).

        Returns
        -------
        self
        """
        preimage_node_counts = []
        image_node_counts = []
        preimage_edge_counts = []
        image_edge_counts = []
        subgraph_sizes = []
        node_labels = []
        edge_labels = []
        
        # Loop over each quotient graph.
        for qg in X:
            G_pre, G_img = qg
            
            # Preimage graph stats.
            preimage_node_counts.append(G_pre.number_of_nodes())
            preimage_edge_counts.append(G_pre.number_of_edges())
            for _, data in G_pre.nodes(data=True):
                node_labels.append(data['label'])
                vec = data.get('vec', [])
                if isinstance(vec, (list, tuple, np.ndarray)):
                    self.max_node_attr_size = max(self.max_node_attr_size, len(vec))
                else:
                    self.max_node_attr_size = max(self.max_node_attr_size, 1)
            for _, _, edata in G_pre.edges(data=True):
                edge_labels.append(edata['label'])
                vec = edata.get('vec', [])
                if isinstance(vec, (list, tuple, np.ndarray)):
                    self.max_edge_attr_size = max(self.max_edge_attr_size, len(vec))
                else:
                    self.max_edge_attr_size = max(self.max_edge_attr_size, 1)
            
            # Image graph stats.
            image_node_counts.append(G_img.number_of_nodes())
            image_edge_counts.append(G_img.number_of_edges())
            for _, data in G_img.nodes(data=True):
                node_labels.append(data['label'])
                vec = data.get('vec', [])
                if isinstance(vec, (list, tuple, np.ndarray)):
                    self.max_node_attr_size = max(self.max_node_attr_size, len(vec))
                else:
                    self.max_node_attr_size = max(self.max_node_attr_size, 1)
                # For link data: extract subgraph sizes.
                subgraph = data.get('subgraph', None)
                if subgraph is not None and isinstance(subgraph, nx.Graph):
                    subgraph_sizes.append(subgraph.number_of_nodes())
            for _, _, edata in G_img.edges(data=True):
                edge_labels.append(edata['label'])
                vec = edata.get('vec', [])
                if isinstance(vec, (list, tuple, np.ndarray)):
                    self.max_edge_attr_size = max(self.max_edge_attr_size, len(vec))
                else:
                    self.max_edge_attr_size = max(self.max_edge_attr_size, 1)
        
        self.max_preimage_nodes = max(preimage_node_counts) if preimage_node_counts else 0
        self.max_image_nodes = max(image_node_counts) if image_node_counts else 0
        self.max_preimage_edges = max(preimage_edge_counts) if preimage_edge_counts else 0
        self.max_image_edges = max(image_edge_counts) if image_edge_counts else 0
        self.max_subgraph_nodes = max(subgraph_sizes) if subgraph_sizes else 0
        
        # Fit label encoders over all node labels and edge labels.
        self.node_label_encoder.fit(node_labels)
        self.edge_label_encoder.fit(edge_labels)
        
        return self
    
    def transform(self, X):
        """
        Transform the list of quotient graphs into tensors.

        Parameters
        ----------
        X : list
            A list of quotient graphs (tuples (G_pre, G_img)).

        Returns
        -------
        preimage_data : dict
            Contains tensors for the preimage graphs:
              - "node_labels": LongTensor of shape (num_graphs, max_preimage_nodes)
              - "node_vecs": FloatTensor of shape (num_graphs, max_preimage_nodes, max_node_attr_size)
              - "edge_index": LongTensor of shape (num_graphs, 2, max_preimage_edges)
              - "edge_labels": LongTensor of shape (num_graphs, max_preimage_edges)
              - "edge_vecs": FloatTensor of shape (num_graphs, max_preimage_edges, max_edge_attr_size)
        image_data : dict
            Analogous dictionary for the image graphs.
        link_data : LongTensor
            Tensor of shape (num_graphs, max_image_nodes, max_subgraph_nodes) where each row, for each
            node in the image graph, contains the (padded) indices of nodes in the corresponding preimage
            graph that are included in the subgraph.
        """
        num_graphs = len(X)
        # Initialize tensors for preimage graph data.
        pre_node_labels = torch.full((num_graphs, self.max_preimage_nodes), -1, dtype=torch.long)
        pre_node_vecs = torch.zeros((num_graphs, self.max_preimage_nodes, self.max_node_attr_size), dtype=torch.float)
        pre_edge_index = torch.full((num_graphs, 2, self.max_preimage_edges), -1, dtype=torch.long)
        pre_edge_labels = torch.full((num_graphs, self.max_preimage_edges), -1, dtype=torch.long)
        pre_edge_vecs = torch.zeros((num_graphs, self.max_preimage_edges, self.max_edge_attr_size), dtype=torch.float)
        
        # Initialize tensors for image graph data.
        im_node_labels = torch.full((num_graphs, self.max_image_nodes), -1, dtype=torch.long)
        im_node_vecs = torch.zeros((num_graphs, self.max_image_nodes, self.max_node_attr_size), dtype=torch.float)
        im_edge_index = torch.full((num_graphs, 2, self.max_image_edges), -1, dtype=torch.long)
        im_edge_labels = torch.full((num_graphs, self.max_image_edges), -1, dtype=torch.long)
        im_edge_vecs = torch.zeros((num_graphs, self.max_image_edges, self.max_edge_attr_size), dtype=torch.float)
        
        # Initialize tensor for the links between image nodes and preimage nodes.
        link_data = torch.full((num_graphs, self.max_image_nodes, self.max_subgraph_nodes), -1, dtype=torch.long)
        
        for i, qg in enumerate(X):
            G_pre, G_img = qg
            
            # -----------------
            # Preimage Graph
            # -----------------
            pre_nodes = sorted(G_pre.nodes())
            # Process nodes.
            for j, node in enumerate(pre_nodes):
                if j >= self.max_preimage_nodes:
                    break
                data = G_pre.nodes[node]
                # Encode label.
                encoded_label = self.node_label_encoder.transform([data['label']])[0]
                pre_node_labels[i, j] = encoded_label
                # Process vector attribute.
                vec = np.array(data.get('vec', []), dtype=float).flatten()
                padded_vec = np.zeros(self.max_node_attr_size, dtype=float)
                length = min(len(vec), self.max_node_attr_size)
                padded_vec[:length] = vec[:length]
                pre_node_vecs[i, j] = torch.from_numpy(padded_vec)
            
            # Process edges.
            pre_edges = list(G_pre.edges(data=True))
            for k, (u, v, edata) in enumerate(pre_edges):
                if k >= self.max_preimage_edges:
                    break
                # Use the ordering from pre_nodes.
                try:
                    u_idx = pre_nodes.index(u)
                    v_idx = pre_nodes.index(v)
                except ValueError:
                    continue
                pre_edge_index[i, 0, k] = u_idx
                pre_edge_index[i, 1, k] = v_idx
                encoded_label = self.edge_label_encoder.transform([edata['label']])[0]
                pre_edge_labels[i, k] = encoded_label
                vec = np.array(edata.get('vec', []), dtype=float).flatten()
                padded_vec = np.zeros(self.max_edge_attr_size, dtype=float)
                length = min(len(vec), self.max_edge_attr_size)
                padded_vec[:length] = vec[:length]
                pre_edge_vecs[i, k] = torch.from_numpy(padded_vec)
            
            # -----------------
            # Image Graph
            # -----------------
            im_nodes = sorted(G_img.nodes())
            # Process nodes.
            for j, node in enumerate(im_nodes):
                if j >= self.max_image_nodes:
                    break
                data = G_img.nodes[node]
                encoded_label = self.node_label_encoder.transform([data['label']])[0]
                im_node_labels[i, j] = encoded_label
                vec = np.array(data.get('vec', []), dtype=float).flatten()
                padded_vec = np.zeros(self.max_node_attr_size, dtype=float)
                length = min(len(vec), self.max_node_attr_size)
                padded_vec[:length] = vec[:length]
                im_node_vecs[i, j] = torch.from_numpy(padded_vec)
                
                # Process the subgraph for links.
                subgraph = data.get('subgraph', None)
                if subgraph is not None and isinstance(subgraph, nx.Graph):
                    sub_nodes = sorted(subgraph.nodes())
                    for k, sub_node in enumerate(sub_nodes):
                        if k >= self.max_subgraph_nodes:
                            break
                        # We assume that sub_node is a node from the preimage graph.
                        try:
                            pre_index = pre_nodes.index(sub_node)
                        except ValueError:
                            pre_index = -1
                        link_data[i, j, k] = pre_index
                        
            # Process edges.
            im_edges = list(G_img.edges(data=True))
            for k, (u, v, edata) in enumerate(im_edges):
                if k >= self.max_image_edges:
                    break
                try:
                    u_idx = im_nodes.index(u)
                    v_idx = im_nodes.index(v)
                except ValueError:
                    continue
                im_edge_index[i, 0, k] = u_idx
                im_edge_index[i, 1, k] = v_idx
                encoded_label = self.edge_label_encoder.transform([edata['label']])[0]
                im_edge_labels[i, k] = encoded_label
                vec = np.array(edata.get('vec', []), dtype=float).flatten()
                padded_vec = np.zeros(self.max_edge_attr_size, dtype=float)
                length = min(len(vec), self.max_edge_attr_size)
                padded_vec[:length] = vec[:length]
                im_edge_vecs[i, k] = torch.from_numpy(padded_vec)
        
        preimage_data = {
            'node_labels': pre_node_labels,
            'node_vecs': pre_node_vecs,
            'edge_index': pre_edge_index,
            'edge_labels': pre_edge_labels,
            'edge_vecs': pre_edge_vecs
        }
        image_data = {
            'node_labels': im_node_labels,
            'node_vecs': im_node_vecs,
            'edge_index': im_edge_index,
            'edge_labels': im_edge_labels,
            'edge_vecs': im_edge_vecs
        }
        return preimage_data, image_data, link_data

class EdgeConv(nn.Module):
    """
    Multi-layer Edge Convolution with skip connections and dropout.
    
    This module expects that node and edge features are already embedded into
    a common hidden space. At each layer, for each edge the module concatenates:
      - the source node features,
      - the target node features,
      - the edge attributes.
    The resulting tensor of shape (E, 3*hidden_dim) is passed through an MLP 
    that outputs hidden_dim–dimensional messages. These messages are aggregated 
    (summed) by target node. Then, a dropout is applied and the result is added as a 
    skip connection to the current node features before applying a ReLU.
    """
    def __init__(self, hidden_dim, num_layers=1, dropout=0.1):
        """
        Args:
            hidden_dim (int): Hidden dimension (for both node and edge embeddings).
            num_layers (int): Number of EdgeConv layers. If set to 0, this module is skipped.
            dropout (float): Dropout probability to apply in each layer.
        """
        super(EdgeConv, self).__init__()
        self.num_layers = num_layers
        self.dropout_layer = nn.Dropout(dropout)
        self.mlps = nn.ModuleList()
        # Each MLP takes an input of dimension 3*hidden_dim and outputs hidden_dim.
        for _ in range(num_layers):
            self.mlps.append(nn.Sequential(
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
    
    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x (Tensor): Node features of shape (N, hidden_dim).
            edge_index (LongTensor): Edge indices of shape (2, E) with rows [source, target].
            edge_attr (Tensor): Edge attributes of shape (E, hidden_dim).
        Returns:
            Tensor: Updated node features of shape (N, hidden_dim) after processing.
        """
        for i in range(self.num_layers):
            src, tgt = edge_index  # Unpack source and target indices.
            x_src = x[src]         # Shape: (E, hidden_dim)
            x_tgt = x[tgt]         # Shape: (E, hidden_dim)
            # Concatenate source, target node features, and edge attributes.
            edge_features = torch.cat([x_src, x_tgt, edge_attr], dim=1)  # (E, 3*hidden_dim)
            # Compute messages.
            messages = self.mlps[i](edge_features)  # (E, hidden_dim)
            # Initialize aggregation tensor.
            N = x.size(0)
            aggregated = torch.zeros(N, messages.size(1), device=x.device)
            aggregated = aggregated.index_add(0, tgt, messages)
            # Add dropout and a skip connection from the input.
            x = F.relu(x + self.dropout_layer(aggregated))
        return x

class SetTransformer(nn.Module):
    """
    Transformer-based module for sets with dropout.
    
    This module first projects node features (assumed to be of dimension hidden_dim)
    into the transformer space (again of size hidden_dim), pads (or truncates) to a fixed
    number of nodes (max_node_size), applies a TransformerEncoder (which includes built-in 
    dropout and skip connections), and finally performs mean pooling over the node dimension.
    """
    def __init__(self, node_feature_dim, max_node_size, num_layers, num_heads, transformer_hidden_dim, dropout=0.1):
        """
        Args:
            node_feature_dim (int): Dimensionality of input node features (should match hidden_dim).
            max_node_size (int): Maximum number of nodes for the transformer.
            num_layers (int): Number of transformer encoder layers.
            num_heads (int): Number of attention heads.
            transformer_hidden_dim (int): Hidden (model) dimension for the transformer (same as hidden_dim here).
            dropout (float): Dropout probability for the transformer layers.
        """
        super(SetTransformer, self).__init__()
        self.max_node_size = max_node_size
        # Project node features into transformer space.
        self.input_proj = nn.Linear(node_feature_dim, transformer_hidden_dim)
        # Create a Transformer encoder layer with dropout (includes skip connections).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_hidden_dim,
            nhead=num_heads,
            dim_feedforward=transformer_hidden_dim * 2,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): Node features of shape (batch_size, num_nodes, node_feature_dim).
            mask (Tensor, optional): Boolean mask of shape (batch_size, num_nodes) where True indicates padding.
        Returns:
            Tensor: Graph-level representation of shape (batch_size, transformer_hidden_dim).
        """
        batch_size, num_nodes, _ = x.shape
        # Pad or truncate node dimension to max_node_size.
        if num_nodes < self.max_node_size:
            pad_size = self.max_node_size - num_nodes
            padding = torch.zeros(batch_size, pad_size, x.size(2), device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)
            if mask is not None:
                padding_mask = torch.ones(batch_size, pad_size, device=x.device, dtype=torch.bool)
                mask = torch.cat([mask, padding_mask], dim=1)
        elif num_nodes > self.max_node_size:
            x = x[:, :self.max_node_size, :]
            if mask is not None:
                mask = mask[:, :self.max_node_size]
        
        # Project node features.
        x = self.input_proj(x)  # (batch_size, max_node_size, transformer_hidden_dim)
        # Transformer expects input shape (sequence_length, batch_size, d_model)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = x.transpose(0, 1)
        # Mean pooling over nodes to get the graph-level representation.
        pooled = x.mean(dim=1)
        return pooled

class GraphNetwork(nn.Module):
    """
    Graph neural network combining optional EdgeConv and optional SetTransformer.
    
    - It first embeds raw node and edge features into a shared hidden dimension.
    - Optionally applies multi-layer EdgeConv with skip connections and dropout.
    - Optionally applies a set transformer. If no transformer layers are used,
      it performs a simple mean pooling.
      
    The same dropout probability is used for both modules.
    """
    def __init__(self,
                 node_in_dim,
                 edge_in_dim,
                 hidden_dim,
                 max_node_size,
                 num_edgeconv_layers=1,
                 num_transformer_layers=1,
                 num_transformer_heads=4,
                 dropout=0.1):
        """
        Args:
            node_in_dim (int): Input dimensionality of node features.
            edge_in_dim (int): Input dimensionality of edge features.
            hidden_dim (int): Unified hidden dimension for node/edge embeddings and processing.
            max_node_size (int): Maximum number of nodes for the transformer.
            num_edgeconv_layers (int): Number of EdgeConv layers (set to 0 to skip).
            num_transformer_layers (int): Number of transformer layers (set to 0 to skip).
            num_transformer_heads (int): Number of attention heads for the transformer.
            dropout (float): Dropout probability to use in both modules.
        """
        super(GraphNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        # Initial embedding layers for nodes and edges.
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_in_dim, hidden_dim)
        # EdgeConv: Optional if num_edgeconv_layers is 0.
        if num_edgeconv_layers > 0:
            self.edge_conv = EdgeConv(hidden_dim, num_layers=num_edgeconv_layers, dropout=dropout)
        else:
            self.edge_conv = None
        # SetTransformer: Optional if num_transformer_layers is 0.
        if num_transformer_layers > 0:
            self.set_transformer = SetTransformer(
                node_feature_dim=hidden_dim,
                max_node_size=max_node_size,
                num_layers=num_transformer_layers,
                num_heads=num_transformer_heads,
                transformer_hidden_dim=hidden_dim,  # same hidden dimension
                dropout=dropout
            )
        else:
            self.set_transformer = None

    def forward(self, x, edge_index, edge_attr, mask=None):
        """
        Process a single graph.
        
        Args:
            x (Tensor): Raw node features of shape (num_nodes, node_in_dim).
            edge_index (Tensor): Edge indices of shape (2, num_edges).
            edge_attr (Tensor): Raw edge attributes of shape (num_edges, edge_in_dim).
            mask (Tensor, optional): Mask for padded nodes when used with the transformer.
        Returns:
            Tensor: Graph-level representation.
              - If the transformer is used: (batch_size, hidden_dim).
              - If the transformer is skipped: (1, hidden_dim) from mean pooling.
        """
        # Embed nodes and edges.
        x = self.node_embed(x)              # (num_nodes, hidden_dim)
        edge_attr = self.edge_embed(edge_attr)  # (num_edges, hidden_dim)
        
        # Optionally apply EdgeConv.
        if self.edge_conv is not None:
            x = self.edge_conv(x, edge_index, edge_attr)  # (num_nodes, hidden_dim)
        
        # If a transformer is used, add a batch dimension and pass through it.
        if self.set_transformer is not None:
            x_batch = x.unsqueeze(0)  # (1, num_nodes, hidden_dim)
            graph_repr = self.set_transformer(x_batch, mask)
        else:
            # Otherwise, simply apply mean pooling.
            graph_repr = x.mean(dim=0, keepdim=True)  # (1, hidden_dim)
        return graph_repr
