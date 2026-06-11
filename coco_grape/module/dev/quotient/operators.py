import networkx as nx
import numpy as np
import copy
from toolz import curry
from typing import Callable, Any, Dict, List, Tuple, Optional, Union
from coco_grape.module.quotient.graph import QuotientGraph
import inspect
import itertools
from itertools import combinations, product
from networkx.algorithms.community import kernighan_lin_bisection

#====================================================================================================
# HIGHER ORDER OPERATORS
#====================================================================================================
def add(*decomposition_functions):
    """
    Combines the results of multiple decomposition functions into a single quotient graph.    
    """
    def composed(quotient_graph: 'QuotientGraph'):
        return sum((func(quotient_graph) for func in decomposition_functions), QuotientGraph())
    composed.__name__ = "add"
    composed.decomposition_functions = decomposition_functions
    composed.operator_type = "add"
    return composed

#--------------------------------------------------------------------------------
def compose(*decomposition_functions):
    def composed(quotient_graph: 'QuotientGraph'):
        for func in reversed(decomposition_functions):
            quotient_graph = func(quotient_graph)
        return quotient_graph
    composed.__name__ = "compose"
    composed.chain = decomposition_functions
    composed.operator_type = "compose"  # Mark as a compose operator
    return composed

def forward_compose(*decomposition_functions):
    def composed(quotient_graph: 'QuotientGraph'):
        for func in decomposition_functions:
            quotient_graph = func(quotient_graph)
        return quotient_graph
    composed.__name__ = "forward_compose"
    composed.chain = decomposition_functions
    composed.operator_type = "forward_compose"  # Mark as a forward_compose operator
    return composed

#--------------------------------------------------------------------------------
def compose_product(combiner, *decomposition_functions):
    """
    Implements the product operation.

    Parameters:
    - combiner: A function to combine the results of the decomposition functions.
    - decomposition_functions: Functions to apply independently to the input.

    Returns:
    - A function that applies the decomposition functions in parallel and combines their results.
    """
    def composed(quotient_graph: 'QuotientGraph'):
        results = [func(quotient_graph) for func in decomposition_functions]
        return combiner(*results)
    composed.__name__ = "compose_product"
    composed.decomposition_functions = decomposition_functions
    composed.operator_type = "product"
    composed.combiner = combiner
    return composed

#====================================================================================================
# CONDITIONAL OPERATORS
#====================================================================================================

@curry
def if_then_else(
    quotient_graph: 'QuotientGraph',
    predicate: Callable[['QuotientGraph'], bool],
    then_function: Callable[['QuotientGraph'], 'QuotientGraph'],
    else_function: Callable[['QuotientGraph'], 'QuotientGraph']
) -> 'QuotientGraph':
    """
    Conditionally applies one of two decomposition functions to a QuotientGraph
    based on the evaluation of a predicate function.

    This operator enables conditional branching within a graph transformation
    pipeline. If the predicate returns True, the 'then_function' is applied to
    the quotient graph. Otherwise, the 'else_function' is applied.

    Args:
        quotient_graph (QuotientGraph): The input QuotientGraph to be transformed.
        predicate (Callable): A function that takes a QuotientGraph and returns a boolean.
            Determines which branch to execute.
        then_function (Callable): The function to apply if the predicate evaluates to True.
        else_function (Callable): The function to apply if the predicate evaluates to False.

    Returns:
        QuotientGraph: The resulting QuotientGraph after applying either the
        'then_function' or the 'else_function' based on the predicate.

    Example:
        workflow = forward_compose(
            connected_component(),
            if_then_else(
                predicate=lambda qg: qg.image_graph.number_of_nodes() > 10,
                then_function=merge(),
                else_function=cycle()
            ),
            filter_by_number_of_nodes(number_of_nodes=(4, 10))
        )
    """
    # Check the condition on the input quotient graph.
    if predicate(quotient_graph):
        # If the predicate is True, apply the 'then_function'.
        return then_function(quotient_graph)
    else:
        # If the predicate is False, apply the 'else_function'.
        return else_function(quotient_graph)


@curry
def if_then_elif_else(
    quotient_graph: 'QuotientGraph',
    conditions_functions: List[Tuple[Callable[['QuotientGraph'], bool], Callable[['QuotientGraph'], 'QuotientGraph']]],
    else_function: Callable[['QuotientGraph'], 'QuotientGraph']
) -> 'QuotientGraph':
    """
    Applies conditional branching to a QuotientGraph with multiple conditions (elif-style)
    and a default fallback function (else).

    This operator allows for defining multiple conditional decomposition paths,
    applied in order. The first predicate that evaluates to True will trigger
    its associated function. If none of the conditions are met, the else_function
    is applied.

    Args:
        quotient_graph (QuotientGraph):
            The input QuotientGraph to be transformed.
        conditions_functions (List[Tuple[Callable, Callable]]):
            A list of (predicate, function) pairs. Each predicate is a function
            that takes a QuotientGraph and returns a boolean. Each function is
            a decomposition to apply if its corresponding predicate is True.
        else_function (Callable):
            A function to apply if none of the predicates return True.

    Returns:
        QuotientGraph: The resulting QuotientGraph after applying the first
        matching decomposition function or the else_function if no conditions match.

    Example:
        workflow = forward_compose(
            connected_component(),
            if_then_elif_else(
                conditions_functions=[
                    (lambda qg: qg.image_graph.number_of_nodes() > 20, merge()),
                    (lambda qg: qg.image_graph.number_of_nodes() > 10, cycle()),
                    (lambda qg: qg.image_graph.number_of_nodes() > 5, clique())
                ],
                else_function=path()
            ),
            filter_by_number_of_nodes(number_of_nodes=(4, 10))
        )
    """
    # Iterate over each (predicate, function) pair in the provided conditions.
    for predicate, func in conditions_functions:
        # Apply the first function where the predicate evaluates to True.
        if predicate(quotient_graph):
            return func(quotient_graph)
    
    # If no predicates matched, apply the else_function.
    return else_function(quotient_graph)

#====================================================================================================
# ITERATION OPERATORS
#====================================================================================================

@curry
def for_loop(
    quotient_graph: 'QuotientGraph',
    function: Callable[['QuotientGraph'], 'QuotientGraph'],
    n_iterations: int = 1
) -> 'QuotientGraph':
    """
    Repeatedly applies a decomposition function to a QuotientGraph 
    for a fixed number of iterations.

    Args:
        quotient_graph (QuotientGraph): The input QuotientGraph.
        function (Callable): The decomposition function to repeatedly apply.
        n_iterations (int, optional): The number of times to apply the function. 
            Defaults to 1.

    Returns:
        QuotientGraph: The transformed QuotientGraph after applying the function
        n_iterations times.
    
    Example:
        workflow = forward_compose(
            connected_component(),
            for_loop(cycle(), n_iterations=3),
            filter_by_number_of_nodes(number_of_nodes=(4, 10))
        )
    """
    for _ in range(n_iterations):
        quotient_graph = function(quotient_graph)
    return quotient_graph

@curry
def while_loop(
    quotient_graph: 'QuotientGraph',
    function: Callable[['QuotientGraph'], 'QuotientGraph'],
    predicate: Callable[['QuotientGraph'], bool],
    max_iterations: int = 100
) -> 'QuotientGraph':
    """
    Repeatedly applies a decomposition function to a QuotientGraph
    as long as a predicate remains True, with an optional maximum number of iterations.

    Args:
        quotient_graph (QuotientGraph): The input QuotientGraph.
        function (Callable): The decomposition function to repeatedly apply.
        predicate (Callable): A function that takes a QuotientGraph and returns a boolean.
            The loop continues as long as this predicate returns True.
        max_iterations (int, optional): Maximum number of iterations to prevent infinite loops.
            Defaults to 100.

    Returns:
        QuotientGraph: The transformed QuotientGraph after applying the function 
        until the predicate is False or max_iterations is reached.

    Example:
        workflow = forward_compose(
            connected_component(),
            while_loop(
                cycle(),
                predicate=lambda qg: qg.image_graph.number_of_nodes() > 5,
                max_iterations=10
            ),
            merge()
        )
    """
    iteration = 0
    while predicate(quotient_graph) and iteration < max_iterations:
        quotient_graph = function(quotient_graph)
        iteration += 1
    return quotient_graph

#====================================================================================================
# UNARY OPERATORS
#====================================================================================================

@curry
def identity(
    quotient_graph: 'QuotientGraph',
    param=None
    ) -> 'QuotientGraph':
    out_quotient_graph = QuotientGraph().copy(quotient_graph)
    out_quotient_graph.update()
    return out_quotient_graph

#--------------------------------------------------------------------------------
@curry
def node(
    quotient_graph: 'QuotientGraph',
    param=None
    ) -> 'QuotientGraph':
    out_quotient_graph = QuotientGraph().copy(quotient_graph).clear_image_graph()

    for subgraph in quotient_graph.get_subgraphs():
        components = [[node] for node in subgraph.nodes()]
        for component in components:
            out_quotient_graph.add_image_node(nodes=component)

    out_quotient_graph.update()
    return out_quotient_graph


#--------------------------------------------------------------------------------
@curry
def edge(
    quotient_graph: 'QuotientGraph',
    param=None
    ) -> 'QuotientGraph':
    out_quotient_graph = QuotientGraph().copy(quotient_graph).clear_image_graph()
    for subgraph in quotient_graph.get_subgraphs():
        components = list(subgraph.edges())
        for component in components:
            out_quotient_graph.add_image_node(nodes=component)
    
    out_quotient_graph.update()
    return out_quotient_graph


#--------------------------------------------------------------------------------
def connected_component_decomposition_function(subgraph):
    components = list(nx.connected_components(subgraph))
    return components

@curry
def connected_component(
    quotient_graph: 'QuotientGraph',
    param=None
    ) -> 'QuotientGraph':
    out_quotient_graph = QuotientGraph().copy(quotient_graph).clear_image_graph()

    for subgraph in quotient_graph.get_subgraphs():
        components = connected_component_decomposition_function(subgraph)
        for component in components:
            out_quotient_graph.add_image_node(nodes=component)
    
    out_quotient_graph.update()
    return out_quotient_graph

#--------------------------------------------------------------------------------
def degree_decomposition_function(subgraph, min_degree=0, max_degree=2):
    deg = dict(nx.degree(subgraph))
    component = set([u for u in deg if max_degree >= deg[u] and  deg[u] >= min_degree])
    components = [component]
    return components

@curry
def degree(
    quotient_graph: 'QuotientGraph',
    d = (0,2)
    ) -> 'QuotientGraph':
    out_quotient_graph = QuotientGraph().copy(quotient_graph).clear_image_graph()

    for subgraph in quotient_graph.get_subgraphs():
        components = degree_decomposition_function(subgraph, min_degree=min(d), max_degree=max(d))
        for component in components:
            out_quotient_graph.add_image_node(nodes=component)
    
    out_quotient_graph.update()
    return out_quotient_graph

#--------------------------------------------------------------------------------
def split_decomposition_function(subgraph):
    """
    Given a connected NetworkX subgraph, partition it into two roughly equal parts
    using the Kernighan-Lin bisection algorithm.
    
    If the graph is not connected, this function returns the nodes of the subgraph
    as a single partition.
    
    Parameters:
      subgraph (nx.Graph): The input subgraph.
      
    Returns:
      List[set]: A list containing two sets of nodes if bisection succeeded,
                 otherwise a single set with all nodes.
    """
    # If subgraph is not connected, we simply return the whole node set.
    if not nx.is_connected(subgraph):
        return [set(subgraph.nodes())]
    
    try:
        part1, part2 = kernighan_lin_bisection(subgraph)
        return [set(part1), set(part2)]
    except Exception as e:
        # In case of any error, fall back to not splitting.
        return [set(subgraph.nodes())]

@curry
def split(quotient_graph: 'QuotientGraph',
    param=None) -> 'QuotientGraph':
    """
    For each subgraph mapping in the quotient graph, split it into two balanced parts
    using the Kernighan-Lin algorithm. Each of the two partitions is added as a new image node.
    
    Returns:
      A new QuotientGraph whose image_graph contains the partitions (as node mappings)
      of the original subgraphs.
    """
    out_quotient_graph = QuotientGraph().copy(quotient_graph).clear_image_graph()
    
    for subgraph in quotient_graph.get_subgraphs():
        # If the subgraph has less than 2 nodes, we cannot split it.
        if subgraph.number_of_nodes() < 2:
            out_quotient_graph.add_image_node(nodes=list(subgraph.nodes()))
        else:
            parts = split_decomposition_function(subgraph)
            for part in parts:
                out_quotient_graph.add_image_node(nodes=list(part))
    
    out_quotient_graph.update()
    return out_quotient_graph


#--------------------------------------------------------------------------------
def get_reachable_nodes_bfs(
    graph: nx.Graph,
    source: Any,
    cutoff: int
    ) -> List[Any]:
    """
    Retrieves all nodes reachable from a specified source node within a given cutoff depth using BFS.

    This function performs a breadth-first search (BFS) traversal starting from the `source` node
    and explores nodes up to the specified `cutoff` depth. Only nodes that are within the cutoff
    distance from the source are returned.

    Args:
        graph (nx.Graph): The NetworkX graph to traverse.
        source (Any): The starting node for BFS traversal.
        cutoff (int): The maximum depth to traverse from the source node. Must be non-negative.

    Returns:
        List[Any]: A list of nodes reachable from the source node within the specified cutoff.

    Raises:
        nx.NetworkXError: If the source node is not present in the graph.
        ValueError: If the cutoff value is negative.
    """
    # Validate that the cutoff is a non-negative integer
    if not isinstance(cutoff, int) or cutoff < 0:
        raise ValueError("Cutoff must be a non-negative integer.")

    # Validate that the source node exists in the graph
    if source not in graph:
        raise nx.NetworkXError(f"Source node {source} is not present in the graph.")

    if cutoff == 0:
        # If the cutoff is 0, only the source node is reachable
        return [source]
    
    # Utilize NetworkX's single_source_shortest_path_length to perform BFS up to the cutoff
    # This function returns a dictionary mapping nodes to their shortest path length from the source
    path_lengths = nx.single_source_shortest_path_length(graph, source, cutoff=cutoff)

    # Extract the keys from the dictionary, which are the reachable nodes
    reachable_nodes = list(path_lengths.keys())

    return reachable_nodes

@curry
def neighborhood(
    quotient_graph: 'QuotientGraph',
    radius=(0,1)
) -> 'QuotientGraph':
    """
    Generates a new quotient graph by expanding each subgraph within a specified radius.

    This function takes a list of `QuotientGraph` instances (expected to contain exactly one graph),
    iterates through each subgraph, and adds new nodes to an output quotient graph based on the
    reachable nodes within the specified radius from each source node in the subgraphs.

    Args:
        quotient_graphs (List['QuotientGraph']): A list containing a single `QuotientGraph` instance.
        min_radius (int, optional): The minimum radius to consider when generating neighborhoods. Defaults to 0.
        max_radius (Optional[int], optional): The maximum radius to consider. If `None`, it defaults to the number of nodes in each subgraph. Defaults to None.

    Returns:
        quotient_graphs (List['QuotientGraph']): A list containing a single `QuotientGraph` instance.

    Raises:
        AssertionError: If the `quotient_graphs` list does not contain exactly one `QuotientGraph` instance.
    """
    out_quotient_graph = QuotientGraph().copy(quotient_graph).clear_image_graph()

    for subgraph in quotient_graph.get_subgraphs():
        # Iterate through each radius from min_radius to max_radius (inclusive)
        for r in range(min(radius), max(radius) + 1):
            # Iterate through each node in the current subgraph as the source
            for source in subgraph.nodes():
                # Retrieve all nodes reachable from the source within the current radius using BFS
                reachable_nodes: List[Any] = get_reachable_nodes_bfs(subgraph, source, cutoff=r)
                # Add a new node to the output QuotientGraph representing the reachable nodes
                out_quotient_graph.add_image_node(nodes=reachable_nodes)
    
    out_quotient_graph.update()
    return out_quotient_graph


#--------------------------------------------------------------------------------
def get_edges_from_cycle(cycle):
    for i, c in enumerate(cycle):
        j = (i + 1) % len(cycle)
        u, v = cycle[i], cycle[j]
        if u < v:
            yield u, v
        else:
            yield v, u

def get_cycle_basis_edges(g):
    ebunch = []
    cs = nx.cycle_basis(g)
    for c in cs:
        ebunch += list(get_edges_from_cycle(c))
    return ebunch

def edge_complement(g, ebunch):
    edge_set = set(ebunch)
    other_ebunch = [e for e in g.edges() if e not in edge_set]
    return other_ebunch

def edge_subgraph(g, ebunch):
    if nx.is_directed(g):
        g2 = nx.DiGraph()
    else:
        g2 = nx.Graph()
    g2.add_nodes_from(g.nodes())
    for u, v in ebunch:
        g2.add_edge(u, v)
        g2.edges[u, v].update(g.edges[u, v])
    return g2

def edge_complement_subgraph(g, ebunch):
    """Induce graph from edges that are not in ebunch."""
    if nx.is_directed(g):
        g2 = nx.DiGraph()
    else:
        g2 = nx.Graph()
    g2.add_nodes_from(g.nodes())
    for e in g.edges():
        if e not in ebunch:
            u, v = e
            g2.add_edge(u, v)
            g2.edges[u, v].update(g.edges[u, v])
    return g2

def cycle_decomposition_function(subgraph):
    cs = nx.cycle_basis(subgraph)
    cycle_components = list(map(set, cs))
    return cycle_components

def non_cycle_decomposition_function(subgraph):
    cs = nx.cycle_basis(subgraph)
    cycle_components = list(map(set, cs))
    cycle_ebunch = get_cycle_basis_edges(subgraph)
    g2 = edge_complement_subgraph(subgraph, cycle_ebunch)
    non_cycle_components = nx.connected_components(g2)
    non_cycle_components = [c for c in non_cycle_components if len(c) >= 2]
    non_cycle_components = list(map(set, non_cycle_components))
    return non_cycle_components

@curry
def cycle(
    quotient_graph: 'QuotientGraph',
    param=None
    ) -> 'QuotientGraph':
    out_quotient_graph = QuotientGraph().copy(quotient_graph).clear_image_graph()

    for subgraph in quotient_graph.get_subgraphs():
        cycle_components = cycle_decomposition_function(subgraph)
        for cycle_component in cycle_components:
            out_quotient_graph.add_image_node(nodes=cycle_component)
    
    out_quotient_graph.update()
    return out_quotient_graph


@curry
def tree(
    quotient_graph: 'QuotientGraph',
    param=None
    ) -> 'QuotientGraph':
    out_quotient_graph = QuotientGraph().copy(quotient_graph).clear_image_graph()

    for subgraph in quotient_graph.get_subgraphs():
        non_cycle_components = non_cycle_decomposition_function(subgraph)
        for non_cycle_component in non_cycle_components:
            out_quotient_graph.add_image_node(nodes=non_cycle_component)
    
    out_quotient_graph.update()
    return out_quotient_graph


#--------------------------------------------------------------------------------
def path_decomposition_function(subgraph, min_number_of_edges=1, max_number_of_edges=None):
    if max_number_of_edges is None:
        max_number_of_edges = subgraph.number_of_nodes()
    edge_components = []
    for n in subgraph.nodes():
        ego_graph = nx.ego_graph(subgraph, n, radius=max_number_of_edges+1)
        for v in ego_graph.nodes():
            try:
                for path in nx.all_shortest_paths(ego_graph, source=n, target=v):
                    edge_component = set()
                    if len(path) >= min_number_of_edges + 1 and len(path) <= max_number_of_edges + 1:
                        for i, u in enumerate(path[:-1]):
                            w = path[i + 1]
                            edge_component.add(u)
                            edge_component.add(w)
                    if edge_component:
                        edge_component = tuple(sorted(edge_component))
                        edge_components.append(edge_component)
            except Exception:
                pass
    components = list(set(edge_components))
    return components

@curry
def path(
    quotient_graph: 'QuotientGraph',
    number_of_edges=(1,3)
    ) -> 'QuotientGraph':
    out_quotient_graph = QuotientGraph().copy(quotient_graph).clear_image_graph()

    for subgraph in quotient_graph.get_subgraphs():
        components = path_decomposition_function(subgraph, min_number_of_edges=min(number_of_edges), max_number_of_edges=max(number_of_edges))
        for component in components:
            out_quotient_graph.add_image_node(nodes=component)
    
    out_quotient_graph.update()
    return out_quotient_graph

#--------------------------------------------------------------------------------
def graphlet_decomposition_function(subgraph, radius=1, min_number_of_nodes=1, max_number_of_nodes=3):
    components = []
    for size in range(min_number_of_nodes, max_number_of_nodes + 1):
        for u in subgraph.nodes():
            ego_graph = nx.ego_graph(subgraph, u, radius=radius)
            for sub_nodes in itertools.combinations(ego_graph.nodes(), size):
                sub_subgraph = ego_graph.subgraph(sub_nodes)
                if nx.is_connected(sub_subgraph):
                    components.append(tuple(sorted(set(sub_nodes))))
    components = list(set(components))
    return components

@curry
def graphlet(
    quotient_graph: 'QuotientGraph',
    radius=1,
    number_of_nodes=(1,3)
    ) -> 'QuotientGraph':
    out_quotient_graph = QuotientGraph().copy(quotient_graph).clear_image_graph()

    for subgraph in quotient_graph.get_subgraphs():
        components = graphlet_decomposition_function(subgraph, radius=radius, min_number_of_nodes=min(number_of_nodes), max_number_of_nodes=max(number_of_nodes))
        for component in components:
            out_quotient_graph.add_image_node(nodes=component)
    
    out_quotient_graph.update()
    return out_quotient_graph

#--------------------------------------------------------------------------------    
def clique_decomposition_function(subgraph, min_number_of_nodes=1, max_number_of_nodes=None):
    if max_number_of_nodes is None:
        max_number_of_nodes = subgraph.number_of_nodes()
    components = []
    cliques = nx.enumerate_all_cliques(subgraph)
    components = list(filter(lambda x: min_number_of_nodes <= len(x) <= max_number_of_nodes, cliques))
    return components

@curry
def clique(
    quotient_graph: 'QuotientGraph',
    number_of_nodes=(1,3)
    ) -> 'QuotientGraph':
    out_quotient_graph = QuotientGraph().copy(quotient_graph).clear_image_graph()

    for subgraph in quotient_graph.get_subgraphs():
        components = clique_decomposition_function(subgraph, min_number_of_nodes=min(number_of_nodes), max_number_of_nodes=max(number_of_nodes))
        for component in components:
            out_quotient_graph.add_image_node(nodes=component)
    
    out_quotient_graph.update()
    return out_quotient_graph

#--------------------------------------------------------------------------------    
@curry
def complement(
    quotient_graph: 'QuotientGraph',
    param=None
    ) -> 'QuotientGraph':
    out_quotient_graph = QuotientGraph().copy(quotient_graph).clear_image_graph()

    for subgraph in quotient_graph.get_subgraphs():
        component = list(subgraph.nodes())
        negative_component = set(quotient_graph.pre_image_graph.nodes()).difference(set(component))
        out_quotient_graph.add_image_node(nodes=negative_component)
    
    out_quotient_graph.update()
    return out_quotient_graph

#--------------------------------------------------------------------------------    
def betweenness_centrality_decomposition_function(subgraph, number_of_nodes=1, use_perifery=False):
    n_dict = nx.betweenness_centrality(subgraph)
    if use_perifery: reverse = False
    else: reverse = True
    selected_ids = sorted(n_dict, key=lambda x: n_dict[x], reverse=reverse)[:number_of_nodes]
    components = [selected_ids] 
    return components

@curry
def betweenness_centrality(
    quotient_graph: 'QuotientGraph',
    number_of_nodes=1,
    use_perifery=False
    ) -> 'QuotientGraph':
    out_quotient_graph = QuotientGraph().copy(quotient_graph).clear_image_graph()

    for subgraph in quotient_graph.get_subgraphs():
        components = betweenness_centrality_decomposition_function(subgraph, number_of_nodes=number_of_nodes, use_perifery=use_perifery)
        for component in components:
            out_quotient_graph.add_image_node(nodes=component)
    
    out_quotient_graph.update()
    return out_quotient_graph


#--------------------------------------------------------------------------------    
@curry
def merge(
    quotient_graph: 'QuotientGraph',
    use_edges=False
    ) -> 'QuotientGraph':
    out_quotient_graph = QuotientGraph().copy(quotient_graph).clear_image_graph()

    if use_edges:
        component = []
        for subgraph in quotient_graph.get_subgraphs():
            component.extend(subgraph.edges())
        out_quotient_graph.add_image_node(edges=component)
        print(out_quotient_graph)##
    else:
        component = []
        for subgraph in quotient_graph.get_subgraphs():
            component.extend(subgraph.nodes())
        out_quotient_graph.add_image_node(nodes=component)
    
    out_quotient_graph.update()
    return out_quotient_graph

#--------------------------------------------------------------------------------   
def get_distance(graph1, graph2, basegraph):
    return min(nx.shortest_path_length(basegraph, source=u, target=v) for u in graph1.nodes() for v in graph2.nodes())

def get_distance_matrix(subgraphs1, subgraphs2, basegraph, max_distance, min_distance):
    distance_matrix = np.zeros((len(subgraphs1), len(subgraphs2)))
    for i, subgraph_i in enumerate(subgraphs1):
        for j, subgraph_j in enumerate(subgraphs2):
            try:
                dist = get_distance(subgraph_i, subgraph_j, basegraph)
                if min_distance <= dist <= max_distance:
                    distance_matrix[i,j] = dist
                    #distance_matrix[j,i] = dist
                else:
                    distance_matrix[i,j] = np.nan
                    #distance_matrix[j,i] = np.nan
            except Exception:
                distance_matrix[i,j] = np.nan
                #distance_matrix[j,i] = np.nan
                pass
    return distance_matrix

def all_distances_are_feasible(combination_idxs, distance_matrix):
    pairs = combinations(combination_idxs, 2)
    for i,j in pairs:
        distance = distance_matrix[i,j]
        if np.isnan(distance): 
            return False
    return True  

def combination_decomposition_function(
        subgraphs, 
        graph, 
        number_of_elements=(2,2),
        distance=(0,1)):
    distance_matrix = get_distance_matrix(subgraphs, subgraphs, graph, max(distance), min(distance))
    components = []
    
    component_combinations = [list(subgraph.nodes()) for subgraph in subgraphs]
    for order in range(min(number_of_elements), max(number_of_elements)+1):
        combination_idxs_list = combinations(range(len(component_combinations)), order)
        for combination_idxs in combination_idxs_list:
            if distance_matrix is not None and all_distances_are_feasible(combination_idxs, distance_matrix) is False: continue #i.e. skip to next combination_idxs
            component_combination = [component_combinations[combination_idx] for combination_idx in combination_idxs]
            component = set([node for combination_nodes in component_combination for node in combination_nodes])
            components.append(component)
            
    return components

@curry
def combination(
    quotient_graph: 'QuotientGraph',
    number_of_elements=(2,2),
    distance=(0,1)
    ) -> 'QuotientGraph':
    out_quotient_graph = QuotientGraph().copy(quotient_graph).clear_image_graph()

    components = combination_decomposition_function(quotient_graph.get_subgraphs(), quotient_graph.pre_image_graph, number_of_elements=number_of_elements, distance=distance)
    for component in components:
        out_quotient_graph.add_image_node(nodes=component)

    out_quotient_graph.update()
    return out_quotient_graph

#====================================================================================================
# EDGE OPERATORS
#====================================================================================================
@curry
def intersection_edges(quotient_graph: 'QuotientGraph', size_threshold=1, accept_connection_by_edge=False) -> 'QuotientGraph':
    out_quotient_graph = QuotientGraph().copy(quotient_graph)
    
    # Determine the graph to use for edge queries.
    if isinstance(quotient_graph.pre_image_graph, QuotientGraph):
        pre_img = quotient_graph.pre_image_graph.image_graph
    else:
        pre_img = quotient_graph.pre_image_graph

    for u in quotient_graph.image_graph.nodes():
        subgraph_u = quotient_graph.get_subgraph(u)
        nodes_u = list(subgraph_u.nodes())
        for v in quotient_graph.image_graph.nodes():
            if u != v:
                subgraph_v = quotient_graph.get_subgraph(v)
                nodes_v = list(subgraph_v.nodes())
                
                # Flag to decide whether to add an edge from u to v.
                add_edge = False
                
                # Normal behavior: add edge if the intersection size meets the threshold.
                if len(set(nodes_u).intersection(set(nodes_v))) >= size_threshold:
                    add_edge = True
                
                if accept_connection_by_edge:
                    # Check for any pre_image edge between nodes_u and nodes_v.
                    for node_u in nodes_u:
                        for node_v in nodes_v:
                            if pre_img.has_edge(node_u, node_v):
                                add_edge = True
                                break
                        if add_edge:
                            break
                
                if add_edge:
                    out_quotient_graph.image_graph.add_edge(u, v)
    
    out_quotient_graph.update()
    return out_quotient_graph

#--------------------------------------------------------------------------------    
# FILTERS
#--------------------------------------------------------------------------------    
@curry
def filter_by_number_of_connected_components(
    quotient_graph: 'QuotientGraph',
    number_of_components=(1,1)
    ) -> 'QuotientGraph':
    out_quotient_graph = QuotientGraph().copy(quotient_graph).clear_image_graph()

    for subgraph in quotient_graph.get_subgraphs():
        cc = list(nx.connected_components(subgraph))
        if min(number_of_components) <= len(cc) <= max(number_of_components):
            out_quotient_graph.add_image_node(nodes=subgraph.nodes())
    
    out_quotient_graph.update()
    return out_quotient_graph


#--------------------------------------------------------------------------------    
@curry
def filter_by_number_of_nodes(
    quotient_graph: 'QuotientGraph',
    number_of_nodes=(1,10)
    ) -> 'QuotientGraph':
    out_quotient_graph = QuotientGraph().copy(quotient_graph).clear_image_graph()


    for subgraph in quotient_graph.get_subgraphs():
        if subgraph.number_of_nodes() >= min(number_of_nodes):
            if subgraph.number_of_nodes() <= max(number_of_nodes): 
                out_quotient_graph.add_image_node(nodes=subgraph.nodes())
    
    out_quotient_graph.update()
    return out_quotient_graph


#--------------------------------------------------------------------------------    
@curry
def filter_by_number_of_edges(
    quotient_graph: 'QuotientGraph',
    number_of_edges=(1,10)
    ) -> 'QuotientGraph':
    out_quotient_graph = QuotientGraph().copy(quotient_graph).clear_image_graph()


    for subgraph in quotient_graph.get_subgraphs():
        if subgraph.number_of_edges() >= min(number_of_edges):
            if subgraph.number_of_edges() <= max(number_of_edges): 
                out_quotient_graph.add_image_node(nodes=subgraph.nodes())
    
    out_quotient_graph.update()
    return out_quotient_graph


#--------------------------------------------------------------------------------    
@curry
def filter_by_node_label(
    quotient_graph: 'QuotientGraph',
    key='label',
    must_have_one_of=[],
    cannot_have_any_in=[]
    ) -> 'QuotientGraph':
    out_quotient_graph = QuotientGraph().copy(quotient_graph).clear_image_graph()


    for subgraph in quotient_graph.get_subgraphs():
        if len(must_have_one_of) > 0:
            must_conditions_are_met = False
            for u in subgraph.nodes(): 
                if subgraph.nodes[u].get(key,0) in must_have_one_of:
                    must_conditions_are_met = True
                    break
        else: must_conditions_are_met = True

        if len(cannot_have_any_in) > 0:
            cannot_conditions_are_met = True
            for u in subgraph.nodes(): 
                if subgraph.nodes[u].get(key,0) in cannot_have_any_in:
                    cannot_conditions_are_met = False
                    break
        else:
            cannot_conditions_are_met = True

        if must_conditions_are_met and cannot_conditions_are_met:
            out_quotient_graph.add_image_node(nodes=subgraph.nodes())
    
    out_quotient_graph.update()
    return out_quotient_graph


#====================================================================================================
# BINARY OPERATORS
#====================================================================================================
def binary_combination_decomposition_function(subgraphs1, subgraphs2, graph, distance=(0,1)):
    distance_matrix = get_distance_matrix(subgraphs1, subgraphs2, graph, max(distance), min(distance))
    components = []
    component_combinations1 = [list(subgraph.nodes()) for subgraph in subgraphs1]
    component_combinations2 = [list(subgraph.nodes()) for subgraph in subgraphs2]
    combination_idxs_list = product(range(len(component_combinations1)), range(len(component_combinations2)))
    for combination_idxs in combination_idxs_list:
        if distance_matrix is not None and all_distances_are_feasible(combination_idxs, distance_matrix) is False: continue #i.e. skip to next combination_idxs
        nodes1_list = [node for node in component_combinations1[combination_idxs[0]]]
        nodes2_list = [node for node in component_combinations2[combination_idxs[1]]]
        component = set(nodes1_list+nodes2_list)
        components.append(component)
    return components

@curry
def binary_combination(
    first_quotient_graph: 'QuotientGraph',
    second_quotient_graph: 'QuotientGraph',
    distance=(0,1)
    ) -> 'QuotientGraph':
    out_quotient_graph = QuotientGraph().copy(first_quotient_graph).clear_image_graph()

    components = binary_combination_decomposition_function(first_quotient_graph.get_subgraphs(), second_quotient_graph.get_subgraphs(), first_quotient_graph.pre_image_graph, distance=distance)
    for component in components:
        out_quotient_graph.add_image_node(nodes=component)

    out_quotient_graph.update()
    return out_quotient_graph

#====================================================================================================
# PRE-IMAGE GRAPH OPERATORS
#====================================================================================================
@curry
def unlabel(
    quotient_graph: 'QuotientGraph', 
    label='-'
    ) -> 'QuotientGraph':
    out_quotient_graph = QuotientGraph().copy(quotient_graph)
    nx.set_node_attributes(out_quotient_graph.pre_image_graph, label, 'label')
    nx.set_edge_attributes(out_quotient_graph.pre_image_graph, label, 'label')
    
    out_quotient_graph.update()
    return out_quotient_graph


#====================================================================================================
# SCALAR OPERATORS
#====================================================================================================
@curry
def number_of_image_graph_nodes(
    quotient_graph: 'QuotientGraph',
    param=None
) -> int:
    return quotient_graph.image_graph.number_of_nodes()

def number_of_image_graph_edges(
    quotient_graph: 'QuotientGraph',
    param=None
) -> int:
    return quotient_graph.image_graph.number_of_edges()

def quantile_number_of_subgraph_nodes(
    quotient_graph: 'QuotientGraph',
    q=0.5
) -> int:
    return np.quantile([subgraph.number_of_nodes() for subgraph in quotient_graph.get_subgraphs()], q)

def quantile_number_of_subgraph_edges(
    quotient_graph: 'QuotientGraph',
    q=0.5
) -> int:
    return np.quantile([subgraph.number_of_edges() for subgraph in quotient_graph.get_subgraphs()], q)

def max_number_of_subgraph_nodes(
    quotient_graph: 'QuotientGraph',
    param=None
) -> int:
    return max([subgraph.number_of_nodes() for subgraph in quotient_graph.get_subgraphs()])

def min_number_of_subgraph_nodes(
    quotient_graph: 'QuotientGraph',
    param=None
) -> int:
    return min([subgraph.number_of_nodes() for subgraph in quotient_graph.get_subgraphs()])

def max_number_of_subgraph_edges(
    quotient_graph: 'QuotientGraph',
    param=None
) -> int:
    return max([subgraph.number_of_edges() for subgraph in quotient_graph.get_subgraphs()])

def min_number_of_subgraph_edges(
    quotient_graph: 'QuotientGraph',
    param=None
) -> int:
    return min([subgraph.number_of_edges() for subgraph in quotient_graph.get_subgraphs()])