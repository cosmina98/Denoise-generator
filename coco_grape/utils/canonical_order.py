import networkx as nx
from collections import deque
import copy

def bfs_sorted_traversal_with_centrality_and_components(graph, verbose=False):
    """
    Perform a BFS traversal on the graph with the following steps:
    
    1. If the graph is disconnected, process each connected component separately.
       - Order components by:
         a. Descending number of nodes.
         b. If tied, descending number of edges.
         c. If still tied, descending diameter.
    
    2. For each component:
       a. Compute betweenness centrality.
       b. Select the starting node:
          - Node with the highest betweenness centrality.
          - If tied, select the node with the largest eccentricity.
       c. Perform BFS traversal from the starting node.
       d. Compute subtree sizes (NS) for each node in the BFS tree.
       e. Sort siblings based on NS (ascending).
       f. Generate traversal order for the component.
    
    3. Concatenate traversal orders of all components in the specified order.
    
    Parameters:
    - graph: A NetworkX graph (undirected or directed).
    - verbose: Boolean flag to enable/disable debug prints.
    
    Returns:
    - traversal_order: List of node IDs in the new traversal order.
    """
    
    if graph.number_of_nodes() == 0:
        if verbose:
            print("The input graph has no nodes. Returning an empty traversal order.")
        return []
    
    traversal_order = []
    
    # Step 1: Identify connected components
    if graph.is_directed():
        # For directed graphs, use weakly connected components
        components = list(nx.weakly_connected_components(graph))
    else:
        # For undirected graphs, use connected components
        components = list(nx.connected_components(graph))
    
    # Prepare component metadata for sorting
    component_info = []
    for comp in components:
        subgraph = graph.subgraph(comp)
        num_nodes = subgraph.number_of_nodes()
        num_edges = subgraph.number_of_edges()
        if num_nodes == 1:
            diameter = 0
        else:
            try:
                diameter = nx.diameter(subgraph)
            except nx.exception.NetworkXError:
                # If the graph is not connected, diameter is undefined
                diameter = float('inf')
        component_info.append({
            'nodes': comp,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'diameter': diameter,
            'subgraph': subgraph
        })
    
    # Step 2: Sort components
    # Sort by:
    # a. Descending number of nodes
    # b. Descending number of edges
    # c. Descending diameter
    component_info_sorted = sorted(
        component_info,
        key=lambda x: (-x['num_nodes'], -x['num_edges'], -x['diameter'])
    )
    
    # Debug: Print component ordering
    if verbose:
        print("Components ordered by size, edges, and diameter:")
        for idx, comp in enumerate(component_info_sorted, 1):
            print(f"Component {idx}: Nodes={comp['num_nodes']}, Edges={comp['num_edges']}, Diameter={comp['diameter']}")
    
    # Step 3: Process each component
    for comp_idx, comp in enumerate(component_info_sorted, 1):
        subgraph = comp['subgraph']
        if verbose:
            print(f"\nProcessing Component {comp_idx}:")
        
        # a. Compute betweenness centrality
        centrality = nx.betweenness_centrality(subgraph)
        max_centrality = max(centrality.values())
        candidates = [node for node, cent in centrality.items() if cent == max_centrality]
        if verbose:
            print(f" - Nodes with highest betweenness centrality ({max_centrality}): {candidates}")
        
        if len(candidates) == 1:
            start_node = candidates[0]
        else:
            # b. Break ties using eccentricity
            eccentricity = nx.eccentricity(subgraph)
            max_eccentricity = max(eccentricity[node] for node in candidates)
            candidates_ecc = [node for node in candidates if eccentricity[node] == max_eccentricity]
            if verbose:
                print(f" - Candidates after eccentricity tie-breaker ({max_eccentricity}): {candidates_ecc}")
            # If still tied, select the first one
            start_node = candidates_ecc[0]
        
        if verbose:
            print(f" - Selected start node: {start_node}")
        
        # c. Build BFS Tree
        bfs_tree = nx.bfs_tree(subgraph, start_node)
        
        # d. Compute Subtree Sizes (NS) using post-order traversal
        NS = {node: 1 for node in bfs_tree.nodes()}
        
        def compute_subtree_sizes(tree, root):
            stack = [(root, False)]
            while stack:
                node, visited_flag = stack.pop()
                if node is None:
                    continue
                if visited_flag:
                    # After visiting children
                    for child in tree.successors(node):
                        NS[node] += NS[child]
                else:
                    stack.append((node, True))
                    for child in tree.successors(node):
                        stack.append((child, False))
        
        compute_subtree_sizes(bfs_tree, start_node)
        
        # Debug: Print subtree sizes
        if verbose:
            print(" - Subtree sizes (NS) for each node:")
            for node, size in NS.items():
                print(f"   Node {node}: NS = {size}")
        
        # e. Sort the children of each node based on NS (ascending)
        sorted_bfs_tree = nx.DiGraph()
        sorted_bfs_tree.add_node(start_node)
        
        queue = deque([start_node])
        while queue:
            current = queue.popleft()
            children = list(bfs_tree.successors(current))
            # Sort children based on NS
            children_sorted = sorted(children, key=lambda x: NS[x])
            for child in children_sorted:
                sorted_bfs_tree.add_edge(current, child)
                queue.append(child)
        
        # f. Perform BFS on the sorted tree to get traversal order for this component
        component_traversal = []
        queue = deque([start_node])
        while queue:
            current = queue.popleft()
            component_traversal.append(current)
            for child in sorted_bfs_tree.successors(current):
                queue.append(child)
        
        if verbose:
            print(f" - Traversal Order for Component {comp_idx}: {component_traversal}")
        
        # Append to overall traversal order
        traversal_order.extend(component_traversal)
    
    return traversal_order

def canonicalise(graph, verbose=False):
    """
    Returns a canonicalised copy of the input NetworkX graph where node order has been sorted
    according to the BFS traversal with betweenness centrality and component ordering.
    
    Parameters:
    - graph: A NetworkX graph (undirected or directed).
    - verbose: Boolean flag to enable/disable debug prints.
    
    Returns:
    - canonical_graph: A new NetworkX graph with nodes ordered as per the traversal order.
    """
    # Get the traversal order
    traversal_order = bfs_sorted_traversal_with_centrality_and_components(graph, verbose=verbose)
    
    if verbose:
        print("\nFinal Traversal Order:", traversal_order)
    
    # Create a new graph of the same type
    if graph.is_directed():
        canonical_graph = nx.DiGraph()
    else:
        canonical_graph = nx.Graph()
    
    # Preserve graph attributes
    canonical_graph.graph = copy.deepcopy(graph.graph)
    
    # Add nodes in the traversal order, preserving node attributes
    for node in traversal_order:
        if graph.has_node(node):
            canonical_graph.add_node(node, **copy.deepcopy(graph.nodes[node]))
    
    # Add edges, preserving edge attributes
    for u, v, attrs in graph.edges(data=True):
        canonical_graph.add_edge(u, v, **copy.deepcopy(attrs))
    
    # Reorder nodes by traversal order
    # Since NetworkX uses insertion order from Python 3.7+, nodes are already ordered
    # in the order they were added above.
    
    return canonical_graph
