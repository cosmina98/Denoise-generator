import networkx as nx
from typing import Any, Optional, List, Tuple, Dict, Union
import copy
from coco_grape.module.hash_graph import hash_graph, hash_set, hash_sequence, hash_bounded

#Consider: preimage_graph <-> image_graph

#------------------------------------------------------------------------------------------------------------
# QuotientGraph class
class QuotientGraph:
    """
    Represents a quotient graph derived from a base graph by hashing its structure.

    The QuotientGraph class initializes a new graph where each node represents a subgraph of the base graph.
    """

    def __init__(
        self,
        graph: Optional[Union[nx.Graph, 'QuotientGraph']] = None,
        nbits: int = 32
    ) -> None:
        """
        Initializes the QuotientGraph with a base graph or copies from another QuotientGraph.

        Args:
            graph (nx.Graph or QuotientGraph, optional): The base graph from which the quotient graph is derived,
                or an existing QuotientGraph instance to copy from. Defaults to None.
            nbits (int, optional): Number of bits to limit the final hash. Defaults to 19.
            

        Raises:
            ValueError: If `nbits` is not a positive integer.
            TypeError: If `graph` is neither a networkx.Graph nor a QuotientGraph instance.
        """
        if not isinstance(nbits, int) or nbits <= 0:
            raise ValueError("nbits must be a positive integer")

        # Store the number of bits for hash limitation
        self.nbits: int = nbits


        # Initialize the quotient graph as a new NetworkX graph
        self.quotient_graph: nx.Graph = nx.Graph()
        self.graph: Optional[nx.Graph] = nx.Graph()

        if isinstance(graph, QuotientGraph):
            # If initializing from another QuotientGraph, perform a deep copy
            self.copy(graph)
        elif isinstance(graph, nx.Graph):
            # If initializing from a base networkx.Graph, set and initialize normally
            self._initialize_from_graph(graph)
        elif graph is not None:
            raise TypeError("graph must be a networkx.Graph or QuotientGraph instance, instead is %s" % type(graph))
        else:
            pass
            
    def assign_default_labels(self, graph: nx.Graph, default_label: str = '-') -> None:
        """
        Assigns default labels to nodes and edges in a NetworkX graph where the 'label' attribute is missing.
        
        This function iterates through all nodes and edges of the provided graph. If a node does not have
        the 'label' attribute, it assigns the specified default label. Similarly, if an edge lacks the 'label'
        attribute, it assigns the specified default label.
        
        Args:
            graph (nx.Graph): The NetworkX graph to process. It can be either directed or undirected.
            default_label (str, optional): The default label to assign to nodes and edges missing the 'label' attribute.
                                        Defaults to '-'.
        
        Raises:
            TypeError: If the input `graph` is not an instance of `networkx.Graph` or its subclasses.
        """
        if not isinstance(graph, nx.Graph):
            raise TypeError("The input must be a NetworkX Graph or a subclass of it.")
        
        # Assign default label '-' to nodes missing the 'label' attribute
        for _, data in graph.nodes(data=True):
            data.setdefault('label', default_label)
        
        # Assign default label '-' to edges missing the 'label' attribute
        for _, _, data in graph.edges(data=True):
            data.setdefault('label', default_label)


    def _initialize_from_graph(self, graph: nx.Graph) -> None:
        """
        Initializes the QuotientGraph instance from a base networkx.Graph.

        Args:
            graph (nx.Graph): The base graph to initialize from.
        """
        # Deep copy the base graph to ensure independence
        self.set_graph(copy.deepcopy(graph))

        # Perform initial computations to set up the quotient graph
        self.add_quotient_node(nodes = list(self.graph.nodes()))
        self.set_quotient_node_and_edge_labels()
        return self

    def partial_copy(self, other: 'QuotientGraph') -> None:
        """
        Initializes the current instance by partially copying from another QuotientGraph.

        Args:
            other (QuotientGraph): The QuotientGraph instance to partially copy from.

        Raises:
            AttributeError: If the provided QuotientGraph instance does not have a 'graph' attribute.
        """
        # Deep copy the base graph
        if hasattr(other, 'graph'):
            self.set_graph(copy.deepcopy(other.graph))
        else:
            raise AttributeError("The provided QuotientGraph instance does not have a 'graph' attribute.")

        self.nbits = other.nbits
        return self

    def copy(self, other: 'QuotientGraph') -> None:
        """
        Initializes the current instance by deep copying from another QuotientGraph.

        Args:
            other (QuotientGraph): The QuotientGraph instance to copy from.

        Raises:
            AttributeError: If the provided QuotientGraph instance does not have a 'graph' attribute.
        """
        # Deep copy the base graph
        if hasattr(other, 'graph'):
            self.set_graph(copy.deepcopy(other.graph))
        else:
            raise AttributeError("The provided QuotientGraph instance does not have a 'graph' attribute.")

        # Deep copy the quotient_graph
        self.quotient_graph = copy.deepcopy(other.quotient_graph)
        self.nbits = other.nbits
        return self

    def set_graph(self, graph: nx.Graph) -> None:
        """
        Sets the base graph for the QuotientGraph.

        Args:
            graph (nx.Graph): The base graph to set.
        """
        self.graph = graph
        self.assign_default_labels(self.graph)
        return self

    def get_graph(self) -> nx.Graph:
        """
        Retrieves the base graph of the QuotientGraph.

        Returns:
            nx.Graph: The base graph.
        """
        return self.graph
        
    def number_of_quotient_nodes(self) -> int:
        """
        Returns the number of nodes in the quotient graph.

        Returns:
            int: The number of nodes in the quotient graph.
        """
        return self.quotient_graph.number_of_nodes()
    
    def number_of_quotient_edges(self) -> int:
        """
        Returns the number of edges in the quotient graph.

        Returns:
            int: The number of edges in the quotient graph.
        """
        return self.quotient_graph.number_of_edges()

    def add_quotient_node(
        self,
        nodes: Optional[List[Any]] = None,
        edges: Optional[List[Tuple[Any, Any]]] = None
    ) -> None:
        """
        Adds a new node to the quotient graph, representing a subgraph defined by either nodes or edges.

        This method allows the user to add a new node to the quotient graph by specifying a set of nodes or edges
        from the base graph. It computes a hash label for the defined subgraph and assigns it as the node's label.

        Args:
            nodes (Optional[List[Any]]): A list of node identifiers defining the subgraph.
            edges (Optional[List[Tuple[Any, Any]]]): A list of edge tuples defining the subgraph.

        Raises:
            ValueError: 
                - If neither `nodes` nor `edges` are provided.
                - If both `nodes` and `edges` are provided simultaneously.
        """
        # Validate input: Ensure that either nodes or edges are provided, but not both
        if nodes is not None and edges is not None:
            raise ValueError("Provide either nodes or edges, not both.")
        elif nodes is None and edges is None:
            raise ValueError("Either nodes or edges must be provided to add a subgraph.")

        # Generate new node ID as 1 + max existing ID (or 0 if empty)
        if self.quotient_graph.number_of_nodes() == 0:
            new_node_id: int = 0
        else:
            new_node_id: int = max(self.quotient_graph.nodes()) + 1

        # Create new node with provided subgraph definition
        self.quotient_graph.add_node(new_node_id)
        self.quotient_graph.nodes[new_node_id]['nodes'] = nodes
        self.quotient_graph.nodes[new_node_id]['edges'] = edges

        # Compute and assign hash label for new node
        self.set_quotient_node_label(new_node_id)

    def _get_next_node_id(self) -> int:
        """Get next available node ID safely."""
        return max(self.quotient_graph.nodes(), default=-1) + 1

    def _validate_node_id(self, node_id: int) -> None:
        """Validate node exists."""
        if node_id not in self.quotient_graph:
            raise ValueError(f"Node {node_id} does not exist")

    def set_quotient_node_label(self, node_id: int) -> None:
        """Assign hash label to quotient graph node.
        
        Args:
            node_id: Node identifier
            
        Raises:
            ValueError: If node_id invalid
        """
        self._validate_node_id(node_id)
        
        hash_label = self.subgraph_label(node_id)
        self.quotient_graph.nodes[node_id]['label'] = hash_label

    def get_quotient_node_label(self, node_id: int) -> int:
        """Get hash label of quotient graph node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            int: Hash label of node
        """
        self._validate_node_id(node_id)
        return self.quotient_graph.nodes[node_id]['label']
    
    def add_quotient_edge(
        self,
        source_node_id: int,
        destination_node_id: int
    ) -> None:
        """
        Adds an edge between two nodes in the quotient graph based on their hash labels.

        This method connects two nodes in the quotient graph based on their hash labels.
        It retrieves the hash labels of the source and destination nodes and creates an edge between them.

        Args:
            source_node_id (int): The node ID of the source node in the quotient graph.
            destination_node_id (int): The node ID of the destination node in the quotient graph.
        """
        # Add an edge between the source and destination nodes in the quotient graph
        self.quotient_graph.add_edge(source_node_id, destination_node_id)
        self.set_quotient_edge_label(source_node_id, destination_node_id)
        
    def set_quotient_edge_label(self, source_node_id: int, destination_node_id: int) -> None:
        """
        Computes the hash label for an edge between two nodes in the quotient graph.

        This method computes the hash label for an edge between two nodes in the quotient graph
        based on the hash labels of the source and destination nodes.

        Args:
            source_node_id (int): The node ID of the source node in the quotient graph.
            destination_node_id (int): The node ID of the destination node in the quotient graph.

        Returns:
            int: The hash label of the edge between the source and destination nodes.
        """
        # Get hash labels of endpoint nodes
        source_label: int = self.quotient_graph.nodes[source_node_id]['label']
        destination_label: int = self.quotient_graph.nodes[destination_node_id]['label']
        
        # Create edge label as bounded hash of unordered endpoint labels
        edge_label: int = hash_bounded(hash_set([source_label, destination_label]), self.nbits)
        self.quotient_graph.edges[source_node_id, destination_node_id]['label'] = edge_label

    def get_quotient_edge_label(self, source_node_id: int, destination_node_id: int) -> int:
        """
        Retrieves the hash label of an edge between two nodes in the quotient graph.

        This method retrieves the hash label of an edge between two nodes in the quotient graph
        based on the provided source and destination node IDs.

        Args:
            source_node_id (int): The node ID of the source node in the quotient graph.
            destination_node_id (int): The node ID of the destination node in the quotient graph.

        Returns:
            int: The hash label of the edge between the source and destination nodes.
        """
        self._validate_node_id(source_node_id)
        self._validate_node_id(destination_node_id)
        return self.quotient_graph.edges[source_node_id, destination_node_id]['label']
    
    def subgraph_nodes(self, node_id: int) -> List[int]:
        """
        Retrieves the nodes that define the subgraph corresponding to a given node in the quotient graph.

        Args:
            node_id (int): The ID of the node in the quotient graph.

        Returns:
            List[int]: A list of node identifiers defining the subgraph associated with the given node.
        """
        return self.quotient_graph.nodes[node_id].get('nodes', [])
    
    def quotient_nodes(self) -> List[int]:
        """
        Retrieves the node identifiers of all nodes in the quotient graph.

        Returns:
            List[int]: A list of node identifiers in the quotient graph.
        """
        return list(self.quotient_graph.nodes())
    
    def subgraph(
        self,
        node_id: int
    ) -> nx.Graph:
        """
        Creates a subgraph from the quotient graph based on provided nodes or edges.

        This method generates either a node-induced subgraph or an edge-induced subgraph from the base graph
        stored within the quotient graph's metadata. It requires either a list of nodes or a list of edges
        to define the subgraph.

        Args:
            nodes (Optional[List[Any]]): A list of node identifiers defining the subgraph.
            edges (Optional[List[Tuple[Any, Any]]]): A list of edge tuples defining the subgraph.

        Returns:
            nx.Graph: The resulting subgraph based on the provided nodes or edges.

        Raises:
            ValueError: If neither `nodes` nor `edges` are provided.
        """
        # Retrieve the nodes and edges defining the subgraph for node 'u'
        nodes: List[Any] = self.quotient_graph.nodes[node_id].get('nodes', [])
        edges: List[Tuple[Any, Any]] = self.quotient_graph.nodes[node_id].get('edges', [])

        if nodes is not None:
            # Create a node-induced subgraph based on the provided nodes
            subgraph = nx.subgraph(self.graph, nodes).copy()
            # It's important to create a copy to ensure the subgraph is independent
        elif edges is not None:
            # Create an edge-induced subgraph based on the provided edges
            subgraph = nx.edge_subgraph(self.graph, edges).copy()
            # Creating a copy ensures the subgraph is a standalone graph object
        else:
            # Raise an error if neither nodes nor edges are provided to define the subgraph
            raise ValueError("Either nodes or edges must be provided to compute subgraph label.")
        
        return subgraph

    def subgraph_label(
        self,
        node_id: int
    ) -> int:
        """
        Computes a hash label for a subgraph defined by either nodes or edges.

        This method generates a subgraph using the `subgraph` method and then computes its hash
        using the `hash_graph` function with the specified number of bits (`nbits`).

        Args:
            nodes (Optional[List[Any]]): A list of node identifiers defining the subgraph.
            edges (Optional[List[Tuple[Any, Any]]]): A list of edge tuples defining the subgraph.

        Returns:
            int: The hash label of the subgraph.

        Raises:
            ValueError: If neither nodes nor edges are provided.
        """
        # Generate the subgraph based on provided nodes or edges
        subgraph = self.subgraph(node_id)
        
        # Compute the hash of the subgraph using the hash_graph function
        h = hash_graph(subgraph, self.nbits)
        return h

    def set_quotient_node_and_edge_labels(self) -> None:
        """
        Assigns hash labels to all nodes and edges in the quotient graph based on their associated subgraphs.
        """
        for u in self.quotient_graph.nodes():
            self.set_quotient_node_label(u)
        for u,v in self.quotient_graph.edges():
            self.set_quotient_edge_label(u,v)

    def subgraphs(self) -> List[nx.Graph]:
        """
        Retrieves all subgraphs defined in the quotient graph.

        This method iterates through each node in the quotient graph, extracts the associated nodes and edges,
        and generates the corresponding subgraph using the `subgraph` method.

        Returns:
            List[nx.Graph]: A list of NetworkX graph objects representing the subgraphs.
        """
        # Initialize an empty list to store the subgraphs
        subgraphs_list: List[nx.Graph] = []

        # Iterate through each node in the quotient graph
        for node_id in self.quotient_graph.nodes():
            # Generate the subgraph using the `subgraph` method and append it to the list
            subgraph: nx.Graph = self.subgraph(node_id)
            subgraphs_list.append(subgraph)

        return subgraphs_list
            
    def __str__(self) -> str:
        """
        Provides a string representation of the quotient graph, detailing its nodes and edges.

        Returns:
            str: A formatted string listing all nodes with their attributes and all edges with their attributes.
        """
        parts: List[str] = []

        # Handle the base graph if it exists
        if self.graph is not None:
            parts.append("Base Graph:")
            if self.graph.number_of_nodes() > 0:
                parts.append("  Nodes:")
                for u, data in self.graph.nodes(data=True):
                    parts.append(f"    Node {u}: {data}")
            else:
                parts.append("  No nodes in the base graph.")

            if self.graph.number_of_edges() > 0:
                parts.append("  Edges:")
                for u, v, data in self.graph.edges(data=True):
                    parts.append(f"    Edge ({u}, {v}): {data}")
            else:
                parts.append("  No edges in the base graph.")
        else:
            parts.append("Base Graph: Not initialized.")

        parts.append("")  # Add a blank line for separation

        # Handle the quotient graph if it exists
        if self.quotient_graph is not None:
            parts.append("Quotient Graph:")
            if self.quotient_graph.number_of_nodes() > 0:
                parts.append("  Nodes:")
                for u, data in self.quotient_graph.nodes(data=True):
                    parts.append(f"    Node {u}: {data}")
            else:
                parts.append("  No nodes in the quotient graph.")

            if self.quotient_graph.number_of_edges() > 0:
                parts.append("  Edges:")
                for u, v, data in self.quotient_graph.edges(data=True):
                    parts.append(f"    Edge ({u}, {v}): {data}")
            else:
                parts.append("  No edges in the quotient graph.")
        else:
            parts.append("Quotient Graph: Not initialized.")

        return "\n".join(parts)
    
    def __add__(self, other: 'QuotientGraph') -> 'QuotientGraph':
        """
        Combines two QuotientGraph instances into a single QuotientGraph.

        This method creates a new QuotientGraph instance that merges the nodes and edges
        of the quotient graphs of the two input QuotientGraph instances.

        Args:
            other (QuotientGraph): Another QuotientGraph instance to combine with the current one.

        Returns:
            QuotientGraph: A new QuotientGraph instance containing the combined quotient graph.

        Raises:
            TypeError: If `other` is not an instance of QuotientGraph.
        """
        if not isinstance(other, QuotientGraph):
            raise TypeError(f"The operand must be an instance of QuotientGraph, got {type(other)} instead.")
        
        # Handle case where one of the QuotientGraph instances is null
        if self.quotient_graph.number_of_nodes() == 0 and self.quotient_graph.number_of_edges() == 0:
            return other
        if other.quotient_graph.number_of_nodes() == 0 and other.quotient_graph.number_of_edges() == 0:
            return self
            
        assert hash_graph(self.graph, self.nbits) == hash_graph(other.graph, self.nbits), "Base graphs must be the same"
        
        # Create a new QuotientGraph for the combined graph
        combined_quotient_graph = QuotientGraph(self)

        # Perform disjoint union of nodes and edges from both quotient graphs
        combined_quotient_graph.quotient_graph = nx.disjoint_union(self.quotient_graph, other.quotient_graph)

        return combined_quotient_graph
