import networkx as nx
import copy
import numpy as np
from typing import Union, Any, Callable, Optional, Dict, Tuple, List
from scipy.sparse import csr_matrix
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.axes import Axes

# Import the actual hash_graph function from the module.
from coco_grape.module.hash_graph import hash_graph
import networkx as nx
import copy
import numpy as np
from typing import Union, Any, Callable, Optional
from scipy.sparse import csr_matrix

# Import the actual hash_graph function from the module.
from coco_grape.module.hash_graph import hash_graph

from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed
import numpy as np
from scipy.sparse import vstack as sparse_vstack

class QuotientGraphTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nbits: int, decomposition_function: callable, return_dense: bool = True, n_jobs: int = -1):
        """
        Parameters:
            nbits (int): The nbits parameter to initialize each QuotientGraph.
            decomposition_function (callable): A function that takes a QuotientGraph
                and returns a new QuotientGraph after decomposition.
            return_dense (bool): If True, .to_array() returns a dense numpy array.
                If False, .to_array() returns a CSR (sparse) matrix. Default is True.
            n_jobs (int): Number of jobs for parallel processing. Default is -1.
        """
        self.nbits = nbits
        self.decomposition_function = decomposition_function
        self.return_dense = return_dense
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        # No fitting necessary for this transformer.
        return self

    def _process_graph(self, graph):
        # Create the QuotientGraph from the input graph.
        qg = QuotientGraph(nbits=self.nbits).pre_image_from(graph)
        # Apply the provided decomposition function.
        qg_p = self.decomposition_function(qg)
        # Convert the quotient graph to an array using the return_dense flag.
        # If return_dense is False, it is expected to return a CSR matrix.
        arr = qg_p.to_array(return_dense=self.return_dense)
        arr = arr.sum(axis=0)  # Sum over all rows to get a single row.
        return arr

    def transform(self, X, y=None):
        """
        Transform a list of graphs into a stacked array.
        
        Parameters:
            X (list): A list of graphs.
            y: Ignored.
            
        Returns:
            np.ndarray or scipy.sparse.csr_matrix:
                If return_dense is True, returns a stacked dense numpy array.
                Otherwise, returns a stacked CSR matrix.
        """
        # Process each graph in parallel.
        arrays = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_graph)(graph) for graph in X
        )
        
        # Stack arrays based on the return_dense flag.
        if self.return_dense:
            return np.stack(arrays)
        else:
            return sparse_vstack(arrays)
        
class QuotientGraphNodeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nbits: int, decomposition_function: callable, return_dense: bool = True, n_jobs: int = -1):
        """
        Parameters:
            nbits (int): The nbits parameter to initialize each QuotientGraph.
            decomposition_function (callable): A function that takes a QuotientGraph
                and returns a new QuotientGraph after decomposition.
            return_dense (bool): If True, .to_array() returns a dense numpy array.
                If False, .to_array() returns a CSR (sparse) matrix. Default is True.
            n_jobs (int): Number of jobs for parallel processing. Default is -1.
        """
        self.nbits = nbits
        self.decomposition_function = decomposition_function
        self.return_dense = return_dense
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        # No fitting necessary for this transformer.
        return self

    def _process_graph(self, graph):
        # Create the QuotientGraph from the input graph.
        qg = QuotientGraph(nbits=self.nbits).pre_image_from(graph)
        # Apply the provided decomposition function.
        qg_p = self.decomposition_function(qg)
        # Convert the quotient graph to an array using the return_dense flag.
        # If return_dense is False, it is expected to return a CSR matrix.
        arr = qg_p.to_array(return_dense=self.return_dense)
        return arr

    def transform(self, X, y=None):
        """
        Transform a list of graphs into a stacked array.
        
        Parameters:
            X (list): A list of graphs.
            y: Ignored.
            
        Returns:
            np.ndarray or scipy.sparse.csr_matrix:
                If return_dense is True, returns a stacked dense numpy array.
                Otherwise, returns a stacked CSR matrix.
        """
        # Process each graph in parallel.
        arrays = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_graph)(graph) for graph in X
        )
        return arrays
    
# ===================== Default Functions =====================

def label_function(mapped_subgraph: Union[nx.Graph, 'QuotientGraph'], nbits: int) -> int:
    """
    If mapped_subgraph is an nx.Graph, returns the hash_graph of it.
    If it is a QuotientGraph, applies hash_graph on its image graph.
    """
    if isinstance(mapped_subgraph, QuotientGraph):
        h = hash_graph(mapped_subgraph.get_image_graph(), nbits=nbits)
    elif isinstance(mapped_subgraph, nx.Graph):
        h = hash_graph(mapped_subgraph, nbits=nbits)
    else:
        h = 0
    return h

def attribute_function(mapped_subgraph: Union[nx.Graph, 'QuotientGraph']) -> Union[np.ndarray, float]:
    """
    Iterates over the nodes of the mapped subgraph and sums the numeric value associated with 'attribute'.
    If mapped_subgraph is a QuotientGraph, uses its image graph.
    Attributes are expected to be scalars or arrays (or list/tuple convertible to an array).
    Returns the sum as a numpy array (or a scalar if the inputs are scalars).
    """
    if isinstance(mapped_subgraph, QuotientGraph):
        g = mapped_subgraph.get_image_graph()
    elif isinstance(mapped_subgraph, nx.Graph):
        g = mapped_subgraph
    else:
        return 0
    
    values = []
    for _, data in g.nodes(data=True):
        attr = data.get('attribute', 0)
        if isinstance(attr, (list, tuple)):
            attr = np.array(attr)
        values.append(attr)
    if not values:
        return 0
    return np.sum(values, axis=0)

def edge_label_function(mapped_subgraph1: Union[nx.Graph, 'QuotientGraph'],
                        mapped_subgraph2: Union[nx.Graph, 'QuotientGraph'], nbits: int) -> int:
    """
    Returns a label computed by summing the hash_graph values of both endpoint mappings.
    """
    if isinstance(mapped_subgraph1, QuotientGraph):
        h1 = hash_graph(mapped_subgraph1.get_image_graph(), nbits=nbits)
    elif isinstance(mapped_subgraph1, nx.Graph):
        h1 = hash_graph(mapped_subgraph1, nbits=nbits)
    else:
        h1 = 0

    if isinstance(mapped_subgraph2, QuotientGraph):
        h2 = hash_graph(mapped_subgraph2.get_image_graph(), nbits=nbits)
    elif isinstance(mapped_subgraph2, nx.Graph):
        h2 = hash_graph(mapped_subgraph2, nbits=nbits)
    else:
        h2 = 0

    return h1 + h2

def edge_attribute_function(mapped_subgraph1: Union[nx.Graph, 'QuotientGraph'],
                            mapped_subgraph2: Union[nx.Graph, 'QuotientGraph']) -> Union[np.ndarray, float]:
    """
    Returns a numpy array representing the sum of the 'attribute' values from both endpoint mappings.
    """
    attr1 = attribute_function(mapped_subgraph1)
    attr2 = attribute_function(mapped_subgraph2)
    return attr1 + attr2

# ===================== QuotientGraph Class =====================

class QuotientGraph:
    def __init__(self, 
                 label_function: Callable[[Union[nx.Graph, 'QuotientGraph'], int], int] = label_function,
                 attribute_function: Callable[[Union[nx.Graph, 'QuotientGraph']], Union[np.ndarray, float]] = attribute_function,
                 edge_label_function: Callable[[Union[nx.Graph, 'QuotientGraph'], Union[nx.Graph, 'QuotientGraph'], int], int] = edge_label_function,
                 edge_attribute_function: Callable[[Union[nx.Graph, 'QuotientGraph'], Union[nx.Graph, 'QuotientGraph']], Union[np.ndarray, float]] = edge_attribute_function,
                 nbits: int = 19) -> None:
        """
        Creates an empty QuotientGraph.
        
        - pre_image_graph is initially set as an empty NetworkX graph.
        - image_graph is initialized as an empty NetworkX graph.
        - label_function: Function that takes a mapped subgraph and nbits, returns an integer.
        - attribute_function: Function that takes a mapped subgraph and returns a scalar or numpy array.
        - edge_label_function: Function that takes two endpoint mapped subgraphs and nbits, returns an integer.
        - edge_attribute_function: Function that takes two endpoint mapped subgraphs and returns a scalar or numpy array.
        - nbits: An integer used by the hash functions.
        """
        self.pre_image_graph = nx.Graph()
        self.image_graph = nx.Graph()
        self.label_function = label_function
        self.attribute_function = attribute_function
        self.edge_label_function = edge_label_function
        self.edge_attribute_function = edge_attribute_function
        self.nbits = nbits

    def pre_image_from(self, G: Union[nx.Graph, 'QuotientGraph']) -> None:
        """
        Initializes the pre_image_graph from G (which can be a NetworkX graph or a QuotientGraph)
        and creates the image_graph with a single image node (id 0) whose mapping is the entire pre_image_graph.
        """
        if not isinstance(G, (nx.Graph, QuotientGraph)):
            raise TypeError("G must be either a networkx.Graph or a QuotientGraph instance.")
        
        if isinstance(G, QuotientGraph):
            self.pre_image_graph = copy.deepcopy(G)
        else:
            self.pre_image_graph = G.copy()
        
        # Initialize the image_graph with a single node mapping to the entire pre_image_graph.
        self.image_graph = nx.Graph()
        image_node_id = 0
        if isinstance(self.pre_image_graph, nx.Graph):
            mapping = self.pre_image_graph.copy()
        elif isinstance(self.pre_image_graph, QuotientGraph):
            mapping = copy.deepcopy(self.pre_image_graph)
        else:
            mapping = None  # Should not occur
        self.image_graph.add_node(image_node_id, mapping=mapping)
        return self

    def clear_pre_image_graph(self) -> None:
        """
        Clears the pre_image_graph.
        """
        self.pre_image_graph = nx.Graph()
        return self

    def clear_image_graph(self) -> None:
        """
        Clears the image_graph.
        """
        self.image_graph = nx.Graph()
        return self

    @classmethod
    def skeleton(cls, G: 'QuotientGraph') -> 'QuotientGraph':
        """
        Copies the meta-information (e.g., functions and parameters) but leaves out the actual graph data 
        and returns a new instance.
        """
        if not isinstance(G, QuotientGraph):
            raise TypeError("G must be a QuotientGraph instance.")
        new_instance = cls(label_function=G.label_function,
                           attribute_function=G.attribute_function,
                           edge_label_function=G.edge_label_function,
                           edge_attribute_function=G.edge_attribute_function,
                           nbits=G.nbits)
        new_instance.pre_image_graph = nx.Graph()
        new_instance.image_graph = nx.Graph()
        return new_instance

    @classmethod
    def copy(cls, G: 'QuotientGraph') -> 'QuotientGraph':
        """
        Deep copies the provided QuotientGraph G and returns a new instance.
        """
        if not isinstance(G, QuotientGraph):
            raise TypeError("G must be a QuotientGraph instance.")
        new_instance = cls(label_function=G.label_function,
                           attribute_function=G.attribute_function,
                           edge_label_function=G.edge_label_function,
                           edge_attribute_function=G.edge_attribute_function,
                           nbits=G.nbits)
        new_instance.pre_image_graph = copy.deepcopy(G.pre_image_graph)
        new_instance.image_graph = copy.deepcopy(G.image_graph)
        return new_instance

    def get_pre_image_graph(self) -> Union[nx.Graph, 'QuotientGraph']:
        """
        Returns the pre_image_graph.
        Raises an error if it is not initialized.
        """
        if self.pre_image_graph is None:
            raise ValueError("pre_image_graph is not initialized.")
        return self.pre_image_graph

    def get_base_graph(self) -> nx.Graph:
        """
        Recursively returns the underlying base graph.
        If pre_image_graph is itself a QuotientGraph, continues to unwrap until a plain networkx.Graph is reached.
        """
        current = self.pre_image_graph
        while isinstance(current, QuotientGraph):
            current = current.pre_image_graph
        if not isinstance(current, nx.Graph):
            raise ValueError("Base graph is not a networkx.Graph.")
        return current

    def get_image_graph(self) -> nx.Graph:
        """
        Returns the image_graph.
        Raises an error if it is not initialized.
        """
        if self.image_graph is None:
            raise ValueError("image_graph is not initialized.")
        return self.image_graph

    def get_image_node_mapping(self, image_node: Any) -> Union[nx.Graph, 'QuotientGraph']:
        """
        Given an image node identifier, returns the associated mapping (i.e., the full subgraph)
        from the pre_image_graph stored in that image node.
        """
        if self.image_graph is None or image_node not in self.image_graph.nodes:
            raise ValueError("Image node mapping not found.")
        return self.image_graph.nodes[image_node]['mapping']
    
    def get_subgraph(self, image_node: Any) -> Union[nx.Graph, 'QuotientGraph']:
        """
        Alias for get_image_node_mapping.
        """
        g = self.get_image_node_mapping(image_node)
        if isinstance(g, QuotientGraph):
            return g.get_image_graph()
        return g

    def get_image_nodes_mappings(self) -> list:
        """
        Returns a list of the associated subgraphs (mappings) for all image nodes in the image graph.
        """
        if self.image_graph is None:
            raise ValueError("image_graph is not initialized.")
        return [self.get_image_node_mapping(node) for node in self.image_graph.nodes()]
    
    def get_subgraphs(self) -> list:
        """
        Alias for get_image_nodes_mappings.
        """
        if self.image_graph is None:
            raise ValueError("image_graph is not initialized.")
        return [self.get_subgraph(node) for node in self.image_graph.nodes()]
    
    def add_image_node(self, nodes: list = None, edges: list = None, subgraph: nx.Graph = None) -> int:
        """
        Adds a new image node to the quotient graph.

        Exactly one of the parameters (nodes, edges, or subgraph) should be provided to define the 
        subgraph mapping. This method does not add any edges between image nodes.

        Parameters:
            nodes: A list of node identifiers from the pre_image_graph that define the subgraph.
                If pre_image_graph is a QuotientGraph, these are interpreted as keys in its image_graph.
            edges: A list of edge tuples defining the subgraph.
            subgraph: A complete NetworkX graph representing the mapping.

        Returns:
            The new image node identifier.

        Side Effects:
            - Adds a new node to the image_graph with its 'mapping' attribute set to the corresponding subgraph.
        """
        # Ensure that the pre_image_graph is initialized.
        if self.pre_image_graph is None:
            raise ValueError("pre_image_graph is not initialized.")

        # Ensure exactly one parameter is provided.
        provided = sum([nodes is not None, edges is not None, subgraph is not None])
        if provided != 1:
            raise ValueError("Provide exactly one of 'nodes', 'edges', or 'subgraph'.")

        # If the pre_image_graph is itself a QuotientGraph, then we want to work on its image_graph.
        if nodes is not None:
            if isinstance(self.pre_image_graph, QuotientGraph):
                # Use the image_graph of the nested QuotientGraph
                mapping = self.pre_image_graph.get_image_graph().subgraph(nodes).copy()
            else:
                mapping = self.pre_image_graph.subgraph(nodes).copy()
        elif edges is not None:
            if isinstance(self.pre_image_graph, QuotientGraph):
                mapping = nx.edge_subgraph(self.pre_image_graph.get_image_graph(), edges).copy()
            else:
                mapping = nx.edge_subgraph(self.pre_image_graph, edges).copy()
        else:  # subgraph is provided.
            mapping = subgraph.copy()

        # Determine a new unique image node identifier.
        new_id = max(self.image_graph.nodes()) + 1 if self.image_graph.nodes() else 0

        # Add the new image node with its mapping.
        self.image_graph.add_node(new_id, mapping=mapping)

        return new_id

    def update(self) -> None:
        """
        Applies the label_function and attribute_function to each image node's mapping,
        and applies the edge_label_function and edge_attribute_function to each edge's endpoints.
        The results are stored as node attributes 'label' and 'attribute' and edge attributes 'label' and 'attribute'.
        """
        if self.image_graph is None:
            raise ValueError("image_graph is not initialized.")
        
        # Update nodes.
        for node in self.image_graph.nodes():
            mapping = self.get_image_node_mapping(node)
            if self.label_function is not None:
                self.image_graph.nodes[node]['label'] = self.label_function(mapping, self.nbits)
            if self.attribute_function is not None:
                self.image_graph.nodes[node]['attribute'] = self.attribute_function(mapping)
        
        # Update edges.
        for u, v in self.image_graph.edges():
            mapping_u = self.get_image_node_mapping(u)
            mapping_v = self.get_image_node_mapping(v)
            if self.edge_label_function is not None:
                self.image_graph.edges[u, v]['label'] = self.edge_label_function(mapping_u, mapping_v, self.nbits)
            if self.edge_attribute_function is not None:
                self.image_graph.edges[u, v]['attribute'] = self.edge_attribute_function(mapping_u, mapping_v)
                    
    def __add__(self, other: 'QuotientGraph') -> 'QuotientGraph':
        # Ensure that 'other' is a QuotientGraph.
        if not isinstance(other, QuotientGraph):
            raise TypeError("Can only add another QuotientGraph")
        
        # Create a new QuotientGraph using the functions from the first operand.
        new_qg = QuotientGraph(
            label_function=self.label_function,
            attribute_function=self.attribute_function,
            edge_label_function=self.edge_label_function,
            edge_attribute_function=self.edge_attribute_function,
            nbits=self.nbits
        )
        
        # Combine the pre_image_graphs: use nx.compose to take the union, 
        # preserving only one copy of nodes with the same id.
        new_qg.pre_image_graph = nx.compose(self.get_pre_image_graph(), other.get_pre_image_graph())
        
        # Combine the image_graphs using disjoint_union so that nodes with the same id are duplicated 
        # and given different ids.
        new_qg.image_graph = nx.disjoint_union(self.get_image_graph(), other.get_image_graph())
            
        # Update the new QuotientGraph to reapply label and attribute functions.
        new_qg.update()
        
        return new_qg

    def to_graph(self):
        """
        Converts the QuotientGraph to a copy of the base NetworkX graph augmented with multisets.
        
        For each node in the base graph, computes the multiset (as a CSR sparse vector)
        of labels of top-level image graph nodes whose mapping contains that base node.
        Similarly, for each edge.
        The resulting base graph copy has new attributes on nodes and edges under key 'quotient_multiset'.
        """
        base = self.get_base_graph().copy()

        # Helper: convert a dictionary {label: count} to a CSR sparse row vector.
        def dict_to_sparse(d: dict) -> csr_matrix:
            if not d:
                return csr_matrix((1, 0))
            # Create a row vector where column indices are the label values.
            row = [0] * len(d)
            cols = list(d.keys())
            data = list(d.values())
            shape = (1, 2**self.nbits)
            return csr_matrix((data, (row, cols)), shape=shape)
        
        # For each base node, compute the multiset.
        # We'll iterate over each image node (top-level) and check if the base node is in its mapping.
        for n in base.nodes():
            counter = {}
            for img in self.image_graph.nodes():
                mapping = self.get_image_node_mapping(img)
                # Check if base node n is in the mapping.
                if isinstance(mapping, nx.Graph):
                    if n in mapping.nodes:
                        lbl = self.image_graph.nodes[img].get('label', None)
                        if lbl is not None:
                            counter[lbl] = counter.get(lbl, 0) + 1
                elif isinstance(mapping, QuotientGraph):
                    # If mapping is a QuotientGraph, get its base graph.
                    subg = mapping.get_base_graph()
                    if n in subg.nodes:
                        lbl = self.image_graph.nodes[img].get('label', None)
                        if lbl is not None:
                            counter[lbl] = counter.get(lbl, 0) + 1
            base.nodes[n]['quotient_multiset'] = dict_to_sparse(counter)
        
        # For each base edge, compute the multiset.
        for u, v, data in base.edges(data=True):
            counter = {}
            for img in self.image_graph.nodes():
                mapping = self.get_image_node_mapping(img)
                # For undirected graphs, check both (u,v) and (v,u)
                if isinstance(mapping, nx.Graph):
                    if mapping.has_edge(u, v) or mapping.has_edge(v, u):
                        lbl = self.image_graph.nodes[img].get('label', None)
                        if lbl is not None:
                            counter[lbl] = counter.get(lbl, 0) + 1
                elif isinstance(mapping, QuotientGraph):
                    subg = mapping.get_base_graph()
                    if subg.has_edge(u, v) or subg.has_edge(v, u):
                        lbl = self.image_graph.nodes[img].get('label', None)
                        if lbl is not None:
                            counter[lbl] = counter.get(lbl, 0) + 1
            data['quotient_multiset'] = dict_to_sparse(counter)
        
        return base

    def to_array(self, return_dense=True):
        """
        Returns an array of shape (n_nodes, n_features) representing the multisets 
        associated with each base node in the base graph. Here, n_nodes is the number of nodes 
        in the base graph (obtained via get_base_graph) and n_features is 2**nbits.
        Each row contains counts corresponding to the labels from the quotient multiset.

        Parameters:
            return_dense (bool): If True, returns a dense numpy array; if False, returns a csr_matrix.

        Returns:
            numpy.ndarray or csr_matrix: The matrix of counts.
        """
        # Obtain the base graph and list its nodes.
        base = self.get_base_graph()
        base_nodes = list(base.nodes())
        n_nodes = len(base_nodes)
        n_features = 2 ** self.nbits

        # Build a mapping from base node to its row index.
        base_index = {node: idx for idx, node in enumerate(base_nodes)}

        # Lists to accumulate row indices, column indices, and counts.
        rows = []
        cols = []
        data = []

        # Iterate over each image node in the quotient graph.
        for img in self.image_graph.nodes():
            mapping = self.get_image_node_mapping(img)

            # Determine the set of base nodes contained in this image node's mapping.
            if isinstance(mapping, nx.Graph):
                sub_nodes = set(mapping.nodes())
            elif isinstance(mapping, QuotientGraph):
                sub_nodes = set(mapping.get_base_graph().nodes())
            else:
                continue

            # Get the label for this quotient image node.
            lbl = self.image_graph.nodes[img].get('label', None)
            if lbl is None:
                continue

            # For every base node that appears in the mapping, increment the corresponding count.
            for n in sub_nodes:
                if n in base_index:
                    rows.append(base_index[n])
                    cols.append(lbl)
                    data.append(1)

        # Construct a sparse matrix.
        from scipy.sparse import csr_matrix
        mat = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_features), dtype=int)

        # Return dense or sparse according to return_dense.
        if return_dense:
            return mat.toarray()
        else:
            return mat

    def __str__(self) -> str:
        """
        Returns a string representation that includes:
          - The pre_image_graph (recursively if it is a QuotientGraph)
          - The image_graph: for each node, its attributes and the full details of its mapping,
            and all edge attributes.
        """
        def format_nx_graph(g: nx.Graph) -> str:
            lines = []
            lines.append("    Nodes:")
            for n, attrs in g.nodes(data=True):
                lines.append(f"      {n}: {attrs}")
            lines.append("    Edges:")
            for u, v, attrs in g.edges(data=True):
                lines.append(f"      ({u}, {v}): {attrs}")
            return "\n".join(lines)
        
        def format_mapping(mapping: Union[nx.Graph, 'QuotientGraph']) -> str:
            if isinstance(mapping, QuotientGraph):
                mapping_str = str(mapping)
                return "      " + mapping_str.replace("\n", "\n      ")
            elif isinstance(mapping, nx.Graph):
                return format_nx_graph(mapping)
            else:
                return str(mapping)
        
        lines = []
        # Pre-image graph.
        lines.append("Pre-Image Graph:")
        if self.pre_image_graph is None:
            lines.append("  Not initialized.")
        else:
            if isinstance(self.pre_image_graph, QuotientGraph):
                pre_str = str(self.pre_image_graph)
                lines.append("  " + pre_str.replace("\n", "\n  "))
            elif isinstance(self.pre_image_graph, nx.Graph):
                lines.append(format_nx_graph(self.pre_image_graph))
            else:
                lines.append(str(self.pre_image_graph))
        
        # Image graph.
        lines.append("Image Graph:")
        if self.image_graph is None or self.image_graph.number_of_nodes() == 0:
            lines.append("  Not initialized or empty.")
        else:
            for node, data in self.image_graph.nodes(data=True):
                lines.append(f"  Node {node}: {data}")
                if "mapping" in data:
                    lines.append("    Mapping:")
                    lines.append(format_mapping(data["mapping"]))
        lines.append("Image Graph Edges:")
        if self.image_graph is not None:
            for u, v, data in self.image_graph.edges(data=True):
                lines.append(f"  Edge ({u}, {v}): {data}")
        return "\n".join(lines)
    
    def display(self,
                base_style: Optional[Dict[str, Any]] = None,
                quotient_style: Optional[Dict[str, Any]] = None,
                connection_style: Optional[Dict[str, Any]] = None,
                size: Tuple[int, int] = (5, 4),
                ax: Optional[Any] = None,
                show_legend: bool = False) -> Optional[Any]:
        """
        Visualizes the full nested structure of a QuotientGraph.
        The leftmost level is always a plain NetworkX graph (base), and each nested quotient level 
        (i.e. the image_graph of a QuotientGraph) is drawn in an additional column to the right.
        Connection lines are drawn between a node in a higher level and every base node (from the 
        immediately lower level) that appears in its mapping.
        """
        # Set default styles.
        if base_style is None:
            base_style = {
                'node_size': 70,
                'edge_width': 1.0,
                'edge_style': 'solid',
                'node_border_width': 0.5,
                'node_alpha': 0.8,
                'edge_color': 'grey',
                'cmap': 'tab20'
            }
        else:
            base_style.setdefault('cmap', 'tab20')

        if quotient_style is None:
            quotient_style = {
                'node_size': 100,
                'edge_width': 2.0,
                'edge_style': 'solid',
                'node_border_width': 2.0,
                'node_alpha': 0.9,
                'edge_color': 'black',
                'cmap': 'tab20'
            }
        else:
            quotient_style.setdefault('cmap', 'tab20')

        if connection_style is None:
            connection_style = {
                'edge_width': 0.5,
                'edge_style': 'dashed',
                'edge_color': 'grey',
                'edge_alpha': 0.3
            }

        # Helper: recursively collect levels.
        # Each level is a tuple: (level_type, graph)
        # level_type is 'base' for a plain nx.Graph and 'quotient' for an image_graph.
        def get_levels(Q: "QuotientGraph"):
            levels = []
            if isinstance(Q.pre_image_graph, QuotientGraph):
                levels.extend(get_levels(Q.pre_image_graph))
            else:
                # Base level: plain networkx graph.
                levels.append(('base', Q.pre_image_graph))
            levels.append(('quotient', Q.image_graph))
            return levels

        # Helper: given a mapping (which may be a QuotientGraph or a plain nx.Graph),
        # return the list of node identifiers from the lower level.
        def get_mapping_nodes(mapping: Union[nx.Graph, "QuotientGraph"]):
            if isinstance(mapping, QuotientGraph):
                if mapping.image_graph is not None:
                    return list(mapping.image_graph.nodes())
                else:
                    return []
            elif isinstance(mapping, nx.Graph):
                return list(mapping.nodes())
            else:
                return []

        # Get the levels for the nested quotient.
        levels = get_levels(self)
        n_levels = len(levels)

        # Create a new figure/axis if one is not provided.
        need_show = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(size[0] * n_levels, size[1]))
            need_show = True

        layouts = []  # List of dicts mapping nodes to positions for each level.
        x_shift = 2.25  # Horizontal shift between levels.
        for i, (level_type, graph) in enumerate(levels):
            pos = nx.kamada_kawai_layout(graph)
            # Shift x-coordinate to separate levels.
            for node in pos:
                pos[node][0] += i * x_shift
            layouts.append(pos)

            # Select style: use base_style for the leftmost level, quotient_style for others.
            style = base_style if level_type == 'base' else quotient_style

            # Get a mapping from each node to its color.
            node_color_mapping = compute_node_color_mapping(graph, style['cmap'])
            # Create a list of colors for the nodes.
            node_colors = [node_color_mapping[node] for node in graph.nodes()]

            # Draw nodes and edges.
            nx.draw_networkx_nodes(graph, pos=pos, node_color=node_colors,
                                   node_size=style.get('node_size', 100),
                                   edgecolors='black',
                                   linewidths=style.get('node_border_width', 2.0),
                                   alpha=style.get('node_alpha', 0.9), ax=ax)
            nx.draw_networkx_edges(graph, pos=pos,
                                   edge_color=style.get('edge_color', 'black'),
                                   style=style.get('edge_style', 'solid'),
                                   width=style.get('edge_width', 2.0), ax=ax)

        # Draw connection lines between each pair of adjacent levels.
        connection_label_added = False
        for i in range(n_levels - 1):
            source_type, source_graph = levels[i]
            target_type, target_graph = levels[i + 1]
            source_pos = layouts[i]
            target_pos = layouts[i + 1]
            for tn, tdata in target_graph.nodes(data=True):
                mapping = tdata.get('mapping', None)
                if mapping is None:
                    continue
                sub_nodes = get_mapping_nodes(mapping)
                for sn in sub_nodes:
                    if sn in source_pos:
                        xs = [source_pos[sn][0], target_pos[tn][0]]
                        ys = [source_pos[sn][1], target_pos[tn][1]]
                        if not connection_label_added:
                            ax.plot(xs, ys,
                                    color=connection_style.get('edge_color', 'grey'),
                                    linestyle=connection_style.get('edge_style', 'dashed'),
                                    linewidth=connection_style.get('edge_width', 0.5),
                                    alpha=connection_style.get('edge_alpha', 0.3),
                                    label='Connections')
                            connection_label_added = True
                        else:
                            ax.plot(xs, ys,
                                    color=connection_style.get('edge_color', 'grey'),
                                    linestyle=connection_style.get('edge_style', 'dashed'),
                                    linewidth=connection_style.get('edge_width', 0.5),
                                    alpha=connection_style.get('edge_alpha', 0.3))

        ax.axis('off')

        if show_legend:
            patches = []
            # Legend for the base level.
            base_nodes = levels[0][1].nodes(data=True)
            cmap_base = plt.get_cmap(base_style['cmap'])
            base_label_to_color = {}
            for n, data in base_nodes:
                lbl = data.get('label')
                if lbl is not None and lbl not in base_label_to_color:
                    base_label_to_color[lbl] = cmap_base(len(base_label_to_color) / max(1, levels[0][1].number_of_nodes()))
            for lbl, col in base_label_to_color.items():
                patches.append(mpatches.Patch(color=col, label=f'Base: {lbl}'))
            # Legends for quotient levels.
            for lvl in range(1, n_levels):
                graph = levels[lvl][1]
                cmap_q = plt.get_cmap(quotient_style['cmap'])
                q_label_to_color = {}
                for n, data in graph.nodes(data=True):
                    lbl = data.get('label')
                    if lbl is not None and lbl not in q_label_to_color:
                        q_label_to_color[lbl] = cmap_q(len(q_label_to_color) / max(1, graph.number_of_nodes()))
                for lbl, col in q_label_to_color.items():
                    patches.append(mpatches.Patch(color=col, label=f'Quotient (Level {lvl}): {lbl}'))
            ax.legend(handles=patches, loc='upper left', bbox_to_anchor=(1, 1))

        if need_show:
            plt.tight_layout()
            plt.show()
        return ax
    
    def display_mappings(self,
                        subgraph_style: Optional[Dict[str, Any]] = None,
                        n_elements_per_row: int = 15,
                        size: float = 2.0,
                        level: int = 0) -> None:
        """
        Visualizes each distinct subgraph (mapping) from the quotient graph.
        
        - If self.pre_image_graph is a plain NetworkX graph, the mappings are displayed as currently done.
        - If self.pre_image_graph is a QuotientGraph, then:
            (a) the subgraphs of its image_graph (a plain nx.Graph) are displayed with titles
                prefixed with 'Level:<current level>' (e.g. "Level:0\nLabel:565\nCount:3"),
            (b) then the method recursively calls display_mappings on the nested QuotientGraph,
                incrementing the level (e.g. "Level:1", etc).
        
        Parameters:
            subgraph_style: Dictionary of plotting style options.
            n_elements_per_row: Number of subplots per row.
            size: Scale factor for subplot sizes.
            level: The current recursion level (default 0).
        """
        if self.image_graph is None or self.image_graph.number_of_nodes() == 0:
            print(f"No mappings to display at Level:{level}.")
            return

        # Set default style if none provided.
        if subgraph_style is None:
            subgraph_style = {
                'node_size': 70,
                'edge_width': 1.0,
                'node_alpha': 0.8,
                'edge_alpha': 0.5,
                'edge_color': 'black',
                'cmap': 'tab20'
            }
        else:
            subgraph_style.setdefault('cmap', 'tab20')

        # Group mappings by label.
        mapping_dict = {}
        frequency = defaultdict(int)
        for node in self.image_graph.nodes():
            lbl = self.image_graph.nodes[node].get('label')
            if lbl is not None:
                if lbl not in mapping_dict:
                    mapping_dict[lbl] = self.get_subgraph(node)
                frequency[lbl] += 1

        sorted_labels = sorted(frequency.keys(), key=lambda x: frequency[x], reverse=True)
        n_subgraphs = len(sorted_labels)
        n_rows = (n_subgraphs + n_elements_per_row - 1) // n_elements_per_row
        fig, axes = plt.subplots(n_rows, n_elements_per_row,
                                figsize=(n_elements_per_row * size, n_rows * size))
        if not hasattr(axes, '__iter__'):
            axes = [axes]
        else:
            axes = axes.flatten()

        # When pre_image_graph is a QuotientGraph, use its image_graph (a plain NetworkX graph)
        # as the base for coloring; otherwise, use pre_image_graph directly.
        if isinstance(self.pre_image_graph, QuotientGraph):
            base_for_colors = self.pre_image_graph.get_image_graph()
        else:
            base_for_colors = self.pre_image_graph
        pre_color_mapping = compute_node_color_mapping(base_for_colors, subgraph_style['cmap'])

        # Draw each mapping and set a title that includes the current level.
        for idx, lbl in enumerate(sorted_labels):
            ax = axes[idx]
            subg = mapping_dict[lbl]
            pos = nx.kamada_kawai_layout(subg)
            node_colors = [pre_color_mapping.get(n, 'grey') for n in subg.nodes()]
            nx.draw_networkx_nodes(subg, pos=pos, node_size=subgraph_style['node_size'],
                                edgecolors='black', node_color=node_colors,
                                alpha=subgraph_style['node_alpha'], ax=ax)
            nx.draw_networkx_edges(subg, pos=pos, width=subgraph_style['edge_width'],
                                edge_color=subgraph_style['edge_color'],
                                alpha=subgraph_style['edge_alpha'], ax=ax)
            ax.set_title(f'Level:{level}\nLabel:{lbl}\nCount:{frequency[lbl]}', fontsize=10)
            ax.axis('off')
        for j in range(idx + 1, len(axes)):
            axes[j].axis('off')
        plt.tight_layout()
        plt.show()

        # If the pre_image_graph is itself a QuotientGraph, recursively display its mappings.
        if isinstance(self.pre_image_graph, QuotientGraph):
            print(f"Displaying nested mappings at Level:{level+1}")
            self.pre_image_graph.display_mappings(subgraph_style=subgraph_style,
                                                n_elements_per_row=n_elements_per_row,
                                                size=size,
                                                level=level+1)

def compute_node_color_mapping(graph: nx.Graph, cmap_name: str = 'tab20') -> Dict[Any, Any]:
    """
    Computes a mapping from each node in the graph to a color based on its label.
    Nodes with the same label receive the same color.
    The color is determined using the specified colormap and normalizing by the total number of nodes.
    """
    cmap = plt.get_cmap(cmap_name)
    label_to_color = {}
    node_to_color = {}
    for node, data in graph.nodes(data=True):
        label = data.get('label')
        if label is not None:
            if label not in label_to_color:
                # Normalize by the total number of nodes in the graph.
                label_to_color[label] = cmap(len(label_to_color) / max(1, graph.number_of_nodes()))
            node_to_color[node] = label_to_color[label]
        else:
            node_to_color[node] = 'grey'
    return node_to_color
