import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import warnings
from typing import Optional, Callable, Any, List, Iterable, Tuple, Dict, Set 
from coco_grape.module.quotientgraph.definitions import graph_hash_label_function_factory, DEFAULT_NBITS
from coco_grape.module.quotientgraph.definitions import sum_attribute_function, null_edge_function

class QuotientGraph:
    """
    Represents a quotient graph derived from an original graph, where nodes represent equivalence classes of subgraphs.
    
    Attributes:
        preimage_graph (nx.Graph): The original graph from which this quotient graph is derived.
        image_graph (nx.Graph): The quotient graph itself, where each node represents a set of equivalent subgraphs.
        label_function (Callable[[nx.Graph], Any]): Function to compute labels for the image nodes based on their subgraphs.
        attribute_function (Callable[[nx.Graph], np.ndarray]): Function to compute attributes for the image nodes based on their subgraphs.
        edge_function (Callable[["QuotientGraph"], "QuotientGraph"]): Function to generate edges in the image graph based on the subgraph structure.
    """
    def __init__(
        self,
        graph: Optional[nx.Graph] = None,
        quotient_graph: Optional["QuotientGraph"] = None,
        label_function: Optional[Callable[[nx.Graph], Any]] = None,
        attribute_function: Optional[Callable[[nx.Graph], np.ndarray]] = None,
        edge_function: Optional[Callable[["QuotientGraph"], "QuotientGraph"]] = None,
    ) -> None:
        """
        Initializes a QuotientGraph.

        - If 'graph' is provided, it initializes from a standard graph.
        - If 'quotient_graph' is provided, it copies from an existing QuotientGraph.
        - Functional arguments inherit from 'quotient_graph' unless explicitly provided.

        """
        self.preimage_graph: nx.Graph = nx.Graph()
        self.image_graph: nx.Graph = nx.Graph()

        # Use explicitly provided functions, or inherit from quotient_graph, or fall back to default
        self.label_function = (
            label_function
            if label_function is not None
            else (quotient_graph.label_function if quotient_graph else graph_hash_label_function_factory(DEFAULT_NBITS))
        )
        self.attribute_function = (
            attribute_function
            if attribute_function is not None
            else (quotient_graph.attribute_function if quotient_graph else sum_attribute_function)
        )
        self.edge_function = (
            edge_function
            if edge_function is not None
            else (quotient_graph.edge_function if quotient_graph else null_edge_function)
        )

        if graph:
            self.from_graph(graph)
        elif quotient_graph:
            self.from_quotient_graph(quotient_graph)

    def copy(self) -> "QuotientGraph":
        """
        Return a deep copy of this QuotientGraph, including both
        preimage_graph and image_graph, and carrying over all
        label/attribute/edge functions.
        """
        # The easiest way is to call your own from_quotient_graph
        new = QuotientGraph(
            label_function=self.label_function,
            attribute_function=self.attribute_function,
            edge_function=self.edge_function,
        )
        # Copy the two graphs
        new.preimage_graph = self.preimage_graph.copy()
        new.image_graph    = self.image_graph.copy()
        return new

    def from_graph(self, graph: nx.Graph) -> "QuotientGraph":
        """
        Initializes the QuotientGraph from a given standard graph.
        
        The preimage_graph is a copy of the provided graph and the image_graph remains empty.
        
        Args:
            graph: The standard NetworkX graph to copy.
        
        Returns:
            The QuotientGraph instance (self).
        """
        self.preimage_graph = graph.copy()
        return self
    
    def from_quotient_graph(self, quotient_graph: "QuotientGraph") -> "QuotientGraph":
        """
        Copies an existing QuotientGraph.
        
        Args:
            quotient_graph: The QuotientGraph instance to copy.
        
        Returns:
            The QuotientGraph instance (self).
        """
        self.preimage_graph = quotient_graph.preimage_graph.copy()
        self.image_graph = quotient_graph.image_graph.copy()
        return self
    
    def _add_image_node(self, association: nx.Graph, meta: Optional[dict] = None) -> int:
        """
        Adds a new image node to the image_graph with the given subgraph and optional metadata.

        Returns the node ID of the newly added node.
        """
        node_id = len(self.image_graph)
        self.image_graph.add_node(
            node_id,
            association=association,
            label=None,
            attribute=None,
            meta=meta or {}
        )
        return node_id

    def create_default_image_node(self) -> None:
        subgraph = self.preimage_graph.copy()
        self._add_image_node(association=subgraph)
        return self

    def create_image_node_with_subgraph_from_nodes(self, nodes: Iterable[Any], meta: Optional[dict] = None) -> None:
        subgraph = self.preimage_graph.subgraph(set(nodes)).copy()
        self._add_image_node(association=subgraph, meta=meta)

    def create_image_node_with_subgraph_from_edges(self, edges: Iterable[Tuple[Any, Any]], meta: Optional[dict] = None) -> None:
        subgraph = self.preimage_graph.edge_subgraph(list(edges)).copy()
        self._add_image_node(association=subgraph, meta=meta)

    def create_image_node_with_subgraph_from_subgraph(self, subgraph: nx.Graph, meta: Optional[dict] = None) -> None:
        self._add_image_node(association=subgraph, meta=meta)

    def apply_label_function(self) -> None:
        """
        Applies the label_function (provided in __init__ or the default) to each image node 
        using its full node attribute dictionary (including meta info).

        The computed label is stored in the image node's attributes under the key 'label'.
        """
        if self.label_function is None:
            raise ValueError("No label_function provided during initialization")
        for node in self.image_graph.nodes:
            node_attributes = self.image_graph.nodes[node]
            self.image_graph.nodes[node]["label"] = self.label_function(node_attributes)
        return self

    def apply_attribute_function(self) -> None:
        """
        Aggregates attributes from nodes in the stored subgraph using the attribute_function.

        The computed attribute is stored in the image node's attributes under the key 'attribute'.
        """
        if self.attribute_function is None:
            raise ValueError("No attribute_function provided during initialization")
        for node in self.image_graph.nodes:
            subgraph = self.image_graph.nodes[node]["association"]
            self.image_graph.nodes[node]["attribute"] = self.attribute_function(subgraph)
        return self

    def apply_edge_function(self) -> None:
        """
        Applies the edge_function (provided in __init__ or the default) to generate edges in the image graph.
        """
        self.edge_function(self)
        return self

    def update(self) -> None:
        """
        Updates the image graph by applying the label, attribute, and edge functions.
        """
        self.apply_label_function()
        self.apply_attribute_function()
        self.apply_edge_function()
        return self

    def get_preimage_nodes_inverse_associations(self) -> List[nx.Graph]:
        """
        Computes the inverse association mapping from preimage nodes to image nodes.

        For each node 'p' in the preimage graph, it identifies all image graph nodes 'q'
        such that 'p' is part of the subgraph associated with 'q'. It then creates the
        induced subgraph of the image_graph containing these nodes 'q'. This induced
        subgraph is stored in the preimage node's data dictionary under the key
        'inverse_association'.

        The function returns a list of these induced subgraphs, where the i-th graph
        corresponds to the i-th node in `list(self.preimage_graph.nodes())`.

        Returns:
            List[nx.Graph]: A list of induced subgraphs from the image_graph,
                            ordered according to the nodes in the preimage graph.
                            Each graph represents the set of image nodes associated
                            with the corresponding preimage node.
        """
        # Step 1: Build the map from preimage node ID to set of image node IDs
        inverse_associations_map: Dict[Any, Set[Any]] = {
            node: set() for node in self.preimage_graph.nodes()
        }

        # Iterate through image nodes and their associated subgraphs
        for image_node_id, image_node_data in self.image_graph.nodes(data=True):
            association_subgraph = image_node_data.get("association")

            if association_subgraph is not None:
                # Iterate through nodes within the association subgraph
                for preimage_node_id in association_subgraph.nodes():
                    # Check if the preimage node exists in our map (initialized from preimage_graph)
                    # and add the image node ID to its inverse association set
                    if preimage_node_id in inverse_associations_map:
                        inverse_associations_map[preimage_node_id].add(image_node_id)
                    # else: # Optional: Warn if an association contains nodes not in the main preimage graph
                    #     warnings.warn(f"Node {preimage_node_id} from association of image node {image_node_id} "
                    #                   f"not found in preimage graph. Skipping inverse mapping for this instance.")

        # Step 2: Create induced subgraphs, store them in preimage nodes, and collect them in a list
        result_subgraphs: List[nx.Graph] = []
        preimage_node_list = list(self.preimage_graph.nodes()) # Ensure consistent order

        for preimage_node_id in preimage_node_list:
            # Get the set of associated image node IDs for the current preimage node
            associated_image_nodes = inverse_associations_map.get(preimage_node_id, set())

            # Create the induced subgraph from the image graph using the collected IDs
            # Use .copy() to get an independent graph object, not just a view
            induced_subgraph = self.image_graph.subgraph(associated_image_nodes).copy()

            # Store the induced subgraph in the corresponding preimage node's attributes
            # We iterate through preimage_node_list, so the node must exist.
            self.preimage_graph.nodes[preimage_node_id]['inverse_association'] = induced_subgraph

            # Add the created subgraph to the result list
            result_subgraphs.append(induced_subgraph)

        return result_subgraphs

    def get_image_nodes_associations(self) -> List[nx.Graph]:
        """
        Retrieves and returns a list of all subgraphs associated with the image nodes.
        
        Returns:
            A list of NetworkX subgraphs, one for each image node.
        """
        return [data["association"] for _, data in self.image_graph.nodes(data=True)]
        
    def __add__(self, other: object) -> "QuotientGraph":
        """
        Combines two QuotientGraphs:
        - Merges preimage graphs using nx.compose
        - Merges image graphs using nx.disjoint_union
        - Preserves self's label, attribute, and edge functions
        - Returns self unchanged if `other` is None or 0
        - Skips if `other` is not a QuotientGraph
        """
        if other is None or other == 0:
            return self

        if not isinstance(other, QuotientGraph):
            return NotImplemented

        new_qg = QuotientGraph(
            label_function=self.label_function,
            attribute_function=self.attribute_function,
            edge_function=self.edge_function,
        )

        new_qg.preimage_graph = nx.compose(self.preimage_graph, other.preimage_graph)
        new_qg.image_graph = nx.disjoint_union(self.image_graph, other.image_graph)

        return new_qg

    def __repr__(self) -> str:
        """
        Provides a string representation of the QuotientGraph.
        
        Uses graph_repr for the preimage_graph and image_graph. For the image graph, the 'subgraph'
        attribute is recursively represented with increased indentation.
        
        Returns:
            A formatted string representing the QuotientGraph.
        """

        def graph_repr(graph: nx.Graph, indent: int = 0) -> str:
            """
            Returns a nicely formatted string representation of a NetworkX graph.
            
            It lists the nodes with their associated attributes and the edges with their attributes.
            If a node has a 'subgraph' attribute that is a NetworkX graph, it recursively calls graph_repr on it
            with increased indentation.
            
            Args:
                graph: The NetworkX graph to represent.
                indent: The current indentation level.
            
            Returns:
                A formatted string representing the graph.
            """
            indent_str = "    " * indent
            lines = []
            lines.append(f"{indent_str}Nodes:")
            for node, data in graph.nodes(data=True):
                attr_parts = []
                for key, value in data.items():
                    if key == "association" and isinstance(value, nx.Graph):
                        # Recursively represent the subgraph with increased indentation.
                        subgraph_str = graph_repr(value, indent=indent+2)
                        attr_parts.append(f"{key}:\n{subgraph_str}")
                    else:
                        attr_parts.append(f"{key}: {value}")
                attr_str = "{" + ", ".join(attr_parts) + "}"
                lines.append(f"{indent_str}  {node}: {attr_str}")
            lines.append(f"{indent_str}Edges:")
            for u, v, edata in graph.edges(data=True):
                edata_str = "{" + ", ".join(f"{k}: {v}" for k, v in edata.items()) + "}"
                lines.append(f"{indent_str}  ({u}, {v}): {edata_str}")
            return "\n".join(lines)
        
        lines = []
        lines.append("QuotientGraph:")
        lines.append("Preimage Graph:")
        lines.append(graph_repr(self.preimage_graph, indent=1))
        lines.append("Image Graph:")
        lines.append(graph_repr(self.image_graph, indent=1))
        return "\n".join(lines)

    def to_graph(self, connection_label: str = "quotient") -> nx.Graph:
        """
        Converts the QuotientGraph into a single NetworkX graph with integer node IDs.

        - Preimage nodes are numbered from 0 to N-1 (same as in self.preimage_graph).
        - Image nodes are numbered starting from N.
        - Edges from image nodes to preimage nodes in their subgraphs are added with 'label' = connection_label.

        Returns:
            A combined NetworkX graph with unified integer node space.
        """
        G = nx.Graph()

        # Map original preimage node ids to consistent 0...N-1 ids
        preimage_nodes = list(self.preimage_graph.nodes())
        preimage_id_map = {orig_id: i for i, orig_id in enumerate(preimage_nodes)}
        next_node_id = len(preimage_nodes)

        # 1. Add preimage nodes and edges
        for orig_id, data in self.preimage_graph.nodes(data=True):
            G.add_node(preimage_id_map[orig_id], **data, kind="preimage", original_id=orig_id)
        for u, v, edata in self.preimage_graph.edges(data=True):
            G.add_edge(preimage_id_map[u], preimage_id_map[v], **edata)

        # 2. Add image nodes and edges
        image_id_map = {}  # map image graph node -> global node id
        for img_node, data in self.image_graph.nodes(data=True):
            image_id = next_node_id
            image_id_map[img_node] = image_id
            next_node_id += 1
            G.add_node(image_id, **{k: v for k, v in data.items() if k != "association"}, kind="image", original_id=img_node)

        for u, v, edata in self.image_graph.edges(data=True):
            G.add_edge(image_id_map[u], image_id_map[v], **edata)

        # 3. Connect image nodes to preimage nodes (association membership)
        for img_node, data in self.image_graph.nodes(data=True):
            subgraph = data.get("association", nx.Graph())
            image_node_id = image_id_map[img_node]
            for orig_pre_id in subgraph.nodes():
                if orig_pre_id in preimage_id_map:
                    G.add_edge(image_node_id, preimage_id_map[orig_pre_id], label=connection_label, kind="quotient")

        return G

    def to_array(self) -> csr_matrix:
        """
        Generates a sparse CSR array representing the counts of image node labels
        associated with each preimage node.

        The array has shape (n, m), where n is the number of preimage nodes and
        m depends on the `nbits` attribute of the `label_function` (m = 2**nbits).
        The value at (i, j) is the count of image nodes associated with the i-th
        preimage node (in `list(self.preimage_graph.nodes())` order) that have label j.

        This method first ensures image node labels are computed using `apply_label_function`.
        It then uses `get_preimage_nodes_inverse_associations` to find the links.

        Returns:
            scipy.sparse.csr_matrix: The sparse count matrix.

        Raises:
            ValueError: If it cannot automatically determine 'nbits' from `self.label_function`,
                        or if `self.label_function` is None.
            TypeError: If an image node label is not interpretable as an integer.
        """
        # 1. Ensure labels are computed for image nodes
        if self.label_function is None:
             raise ValueError("Cannot generate array without a label_function.")
        self.apply_label_function() # Computes 'label' attribute for image nodes

        # 2. Determine nbits and matrix dimensions
        nbits = getattr(self.label_function, 'nbits', None)
        if nbits is None:
            raise ValueError(
                "Could not automatically determine 'nbits' from the label_function. "
                "Ensure the label_function was created using a provided helper "
                "(e.g., graph_hash_label_function) or manually assign an 'nbits' attribute."
            )
        m = 2**nbits

        # 3. Get preimage nodes and create mapping to row indices
        preimage_nodes = list(self.preimage_graph.nodes())
        node_to_index = {node: i for i, node in enumerate(preimage_nodes)}
        n = len(preimage_nodes)

        # 4. Use lil_matrix for efficient construction
        count_matrix = lil_matrix((n, m), dtype=int)

        # 5. Get inverse associations (list of induced image subgraphs)
        #    We don't actually need the full graphs here, just the node IDs within them.
        #    Re-computing the map might be slightly more direct than calling the full get_preimage_nodes_inverse_associations
        #    Let's build the ID map again for clarity and efficiency here.

        inverse_associations_map: Dict[Any, Set[Any]] = {
            node: set() for node in self.preimage_graph.nodes()
        }
        for image_node_id, image_node_data in self.image_graph.nodes(data=True):
            association_subgraph = image_node_data.get("association")
            if association_subgraph is not None:
                for preimage_node_id in association_subgraph.nodes():
                    if preimage_node_id in inverse_associations_map:
                        inverse_associations_map[preimage_node_id].add(image_node_id)


        # 6. Iterate through preimage nodes and count labels of associated image nodes
        for preimage_node_id, associated_image_node_ids in inverse_associations_map.items():
            # Get the row index for the current preimage node
            preimage_node_index = node_to_index.get(preimage_node_id)
            if preimage_node_index is None:
                # This case should ideally not happen if map keys come from preimage_graph.nodes()
                warnings.warn(f"Preimage node {preimage_node_id} not found in node_to_index map. Skipping.")
                continue

            # Count labels of associated image nodes
            for image_node_id in associated_image_node_ids:
                if image_node_id not in self.image_graph:
                    # Should not happen if the map was built correctly from image_graph
                    warnings.warn(f"Associated image node {image_node_id} not found in image_graph. Skipping.")
                    continue

                # Get the pre-computed label
                image_node_label = self.image_graph.nodes[image_node_id].get('label')

                if image_node_label is None:
                    # This indicates apply_label_function might have failed or wasn't called properly
                    warnings.warn(f"Image node {image_node_id} has no label. Skipping count for this node.")
                    continue

                # Convert label to integer and increment count
                try:
                    label_int = int(image_node_label)
                    # Check if label is within the valid range [0, m-1]
                    if 0 <= label_int < m:
                        count_matrix[preimage_node_index, label_int] += 1
                    else:
                        warnings.warn(f"Image node {image_node_id} has label {label_int} outside the expected range [0, {m-1}). Skipping.")
                except (ValueError, TypeError) as e:
                     # Catch potential errors during int conversion
                     warnings.warn(f"Image node {image_node_id} has non-integer label '{image_node_label}' (Error: {e}). Skipping.")


        # 7. Convert to CSR format for efficient storage and computation
        return count_matrix.tocsr()