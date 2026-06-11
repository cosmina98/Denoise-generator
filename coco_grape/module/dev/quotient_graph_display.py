from typing import Any, List, Tuple, Optional, Dict
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.axes import Axes
from collections import defaultdict
from coco_grape.module.quotient_graph import QuotientGraph

def display_quotient_graph(
    quotient_graph: 'QuotientGraph',
    base_style: Optional[Dict[str, Any]] = None,
    quotient_style: Optional[Dict[str, Any]] = None,
    connection_style: Optional[Dict[str, Any]] = None,
    size: Tuple[int, int] = (5, 4),
    ax: Optional[Axes] = None,
    show_legend: bool = False
) -> Optional[Axes]:
    """
    Visualizes a QuotientGraph object by drawing both the base graph and the quotient graph
    with customizable styles for each.
    
    The function performs the following:
    1. Draws the base graph with nodes and edges styled based on `base_style`.
    2. Draws the quotient graph with nodes and edges styled based on `quotient_style`.
    3. Connects base nodes to their corresponding quotient graph nodes with lines styled based on `connection_style`.
    
    Args:
        quotient_graph (QuotientGraph): The QuotientGraph instance to visualize.
        base_style (Optional[Dict[str, Any]]): Dictionary specifying the drawing style for the base graph.
            Expected keys:
                - 'node_size' (int): Size of the base graph nodes.
                - 'edge_width' (float): Thickness of the base graph edges.
                - 'edge_style' (str): Style of the base graph edges (e.g., 'solid', 'dashed').
                - 'node_border_width' (float): Thickness of the node borders.
                - 'node_alpha' (float): Transparency of the base graph nodes.
                - 'edge_color' (str or list): Color of the base graph edges.
                - 'cmap' (str, optional): Colormap name for base graph nodes. Defaults to 'tab20'.
        quotient_style (Optional[Dict[str, Any]]): Dictionary specifying the drawing style for the quotient graph.
            Expected keys:
                - 'node_size' (int): Size of the quotient graph nodes.
                - 'edge_width' (float): Thickness of the quotient graph edges.
                - 'edge_style' (str): Style of the quotient graph edges (e.g., 'solid', 'dashed').
                - 'node_border_width' (float): Thickness of the node borders.
                - 'node_alpha' (float): Transparency of the quotient graph nodes.
                - 'edge_color' (str or list): Color of the quotient graph edges.
                - 'cmap' (str, optional): Colormap name for quotient graph nodes. Defaults to 'tab20'.
        connection_style (Optional[Dict[str, Any]]): Dictionary specifying the drawing style for the connections.
            Expected keys:
                - 'edge_width' (float): Thickness of the connection edges.
                - 'edge_style' (str): Style of the connection edges (e.g., 'solid', 'dashed').
                - 'edge_color' (str): Color of the connection edges.
                - 'edge_alpha' (float): Transparency of the connection edges.
        size (Tuple[int, int], optional): Size of the figure as (width, height). Defaults to (14, 10).
        ax (Optional[Axes], optional): Matplotlib Axes object to plot on. If None, a new figure and axes are created.
        show_legend (bool, optional): Flag to display the legend. Defaults to False.
    
    Returns:
        Optional[Axes]: The Matplotlib Axes object if `ax` is provided, otherwise None.
    """
    # Set default styles if not provided
    if base_style is None:
        base_style = {
            'node_size': 70,
            'edge_width': 1.0,
            'edge_style': 'solid',
            'node_border_width': 0.5,
            'node_alpha': 0.8,
            'edge_color': 'grey',
            'cmap': 'tab20'  # Default colormap for base graph
        }
    else:
        # Ensure 'cmap' key exists in base_style
        base_style.setdefault('cmap', 'tab20')

    if quotient_style is None:
        quotient_style = {
            'node_size': 100,
            'edge_width': 2.0,
            'edge_style': 'solid',
            'node_border_width': 2.0,
            'node_alpha': 0.9,
            'edge_color': 'black',
            'cmap': 'tab20'  # Default colormap for quotient graph
        }
    else:
        # Ensure 'cmap' key exists in quotient_style
        quotient_style.setdefault('cmap', 'tab20')

    if connection_style is None:
        connection_style = {
            'edge_width': 0.5,
            'edge_style': 'dashed',
            'edge_color': 'grey',
            'edge_alpha': 0.3
        }

    # Extract the base graph and the quotient graph from the QuotientGraph instance
    base_graph: nx.Graph = quotient_graph.graph
    q_graph: nx.Graph = quotient_graph.quotient_graph

    # Determine if a new figure and axes need to be created
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
        need_show = True
    else:
        need_show = False

    # Generate layouts using kamada_kawai_layout
    base_pos = nx.kamada_kawai_layout(base_graph)
    q_pos = nx.kamada_kawai_layout(q_graph)

    # Shift the quotient graph's positions to the right to separate from the base graph
    x_shift = 2.25  # Shift amount along the x-axis
    for node in q_pos:
        q_pos[node][0] += x_shift  # Shift along the x-axis

    # Combine positions for plotting connections
    combined_pos: Dict[Any, Tuple[float, float]] = {**base_pos, **q_pos}

    # --- Mapping Labels to Colors for Base Graph ---
    # Create a mapping of base graph nodes to their colors based on their labels
    label_to_color = {}
    cmap_base = plt.get_cmap(base_style['cmap'])

    for node in base_graph.nodes(data=True):
        label = node[1].get('label')
        if label is not None:
            # Map the label to a unique color in the colormap
            if label not in label_to_color:
                label_to_color[label] = cmap_base(len(label_to_color) / len(base_graph.nodes))
    # Assign colors to base graph nodes based on their label
    base_node_colors: List[Any] = [
        label_to_color.get(base_graph.nodes[node].get('label', '-'), 'grey') for node in base_graph.nodes()
    ]

    # --- Mapping Labels to Colors for Quotient Graph ---
    # Create a mapping of quotient graph nodes to their colors based on their labels
    q_label_to_color = {}
    cmap_q = plt.get_cmap(quotient_style['cmap'])

    for node in q_graph.nodes(data=True):
        label = node[1].get('label')
        if label is not None:
            # Map the label to a unique color in the colormap
            if label not in q_label_to_color:
                q_label_to_color[label] = cmap_q(len(q_label_to_color) / len(q_graph.nodes()))
    # Assign colors to quotient graph nodes based on their label
    q_node_colors: List[Any] = [
        q_label_to_color.get(q_graph.nodes[node].get('label', '-'), 'grey') for node in q_graph.nodes()
    ]

    # Draw the base graph nodes with colors based on labels and specified styles
    nx.draw_networkx_nodes(
        base_graph,
        pos=base_pos,
        node_color=base_node_colors,
        node_size=base_style['node_size'],
        edgecolors='black',
        linewidths=base_style.get('node_border_width', 0.5),
        alpha=base_style.get('node_alpha', 0.8),
        ax=ax,
        label='Base Graph Nodes'
    )

    # Draw the base graph edges with specified styles
    nx.draw_networkx_edges(
        base_graph,
        pos=base_pos,
        edge_color=base_style.get('edge_color', 'grey'),
        style=base_style.get('edge_style', 'solid'),
        width=base_style.get('edge_width', 1.0),
        alpha=1.0,  # Edge transparency can be controlled via edge_color alpha if needed
        ax=ax,
        label='Base Graph Edges'
    )

    # Draw the quotient graph nodes with colors based on labels and specified styles
    nx.draw_networkx_nodes(
        q_graph,
        pos=q_pos,
        node_color=q_node_colors,
        node_size=quotient_style['node_size'],
        edgecolors='black',
        linewidths=quotient_style.get('node_border_width', 2.0),
        alpha=quotient_style.get('node_alpha', 0.9),
        ax=ax,
        label='Quotient Graph Nodes'
    )

    # Draw the quotient graph edges with specified styles
    nx.draw_networkx_edges(
        q_graph,
        pos=q_pos,
        edge_color=quotient_style.get('edge_color', 'black'),
        style=quotient_style.get('edge_style', 'solid'),
        width=quotient_style.get('edge_width', 2.0),
        alpha=1.0,
        ax=ax,
        label='Quotient Graph Edges'
    )

    # Establish a mapping from base nodes to their corresponding quotient nodes
    # Allow multiple quotient nodes per base node
    base_to_quotient: Dict[Any, List[Any]] = defaultdict(list)
    for q_node in q_graph.nodes():
        subgraph_nodes = q_graph.nodes[q_node].get('nodes', [])
        if subgraph_nodes is not None and len(subgraph_nodes) > 0:
            for b_node in subgraph_nodes:
                base_to_quotient[b_node].append(q_node)
        subgraph_edges = q_graph.nodes[q_node].get('edges', [])
        if subgraph_edges is not None and len(subgraph_edges) > 0:
            for b_node_i, b_node_j in subgraph_edges:
                base_to_quotient[b_node_i].append(q_node)
                base_to_quotient[b_node_j].append(q_node)

    # Draw connecting lines from base nodes to their corresponding quotient nodes
    # To ensure 'Connections' appears only once in the legend
    connection_label_added = False

    for b_node, q_nodes in base_to_quotient.items():
        for q_node in q_nodes:
            # Retrieve the positions of the base node and the corresponding quotient node
            x_values = [base_pos[b_node][0], q_pos[q_node][0]]
            y_values = [base_pos[b_node][1], q_pos[q_node][1]]
            # Plot a line with specified connection styles
            if not connection_label_added:
                ax.plot(
                    x_values,
                    y_values,
                    color=connection_style.get('edge_color', 'grey'),
                    linestyle=connection_style.get('edge_style', 'dashed'),
                    linewidth=connection_style.get('edge_width', 0.5),
                    alpha=connection_style.get('edge_alpha', 0.3),
                    label='Connections'
                )
                connection_label_added = True
            else:
                ax.plot(
                    x_values,
                    y_values,
                    color=connection_style.get('edge_color', 'grey'),
                    linestyle=connection_style.get('edge_style', 'dashed'),
                    linewidth=connection_style.get('edge_width', 0.5),
                    alpha=connection_style.get('edge_alpha', 0.3)
                )

    # Remove axis for better visualization
    ax.axis('off')

    # Create a legend if show_legend is True
    if show_legend:
        legend_patches = []

        # Create a legend for base graph labels
        for label, color in label_to_color.items():
            patch = mpatches.Patch(color=color, label=f'Base Label: {label}')
            legend_patches.append(patch)

        # Create a legend for quotient graph labels
        for label, color in q_label_to_color.items():
            patch = mpatches.Patch(color=color, label=f'Quotient Label: {label}')
            legend_patches.append(patch)

        # Add patches for base graph edges, quotient graph edges, and connections
        legend_patches.extend([
            mpatches.Patch(color=base_style.get('edge_color', 'grey'), label='Base Graph Edges'),
            mpatches.Patch(color=quotient_style.get('edge_color', 'black'), label='Quotient Graph Edges'),
            mpatches.Patch(color=connection_style.get('edge_color', 'grey'), label='Connections')
        ])

        # Add the legend outside the plot
        ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust layout to accommodate the legend
    if need_show:
        plt.tight_layout()
        plt.show()

    # If ax was provided, return it for further composition
    if not need_show:
        return ax
    else:
        return None


def display_quotient_graph_subgraphs(
    quotient_graph: 'QuotientGraph',
    subgraph_style: Optional[Dict[str, Any]] = None,
    n_elements_per_row: int = 15,
    size: float = 2.0
) -> None:
    """
    Visualizes each distinct subgraph of the quotient graph only once, displaying the label
    and the number of occurrences. Subgraphs are sorted by frequency and displayed in rows.
    Each node is colored based on its label in the base graph.
    
    Args:
        quotient_graph (QuotientGraph): The QuotientGraph instance containing the subgraphs.
        subgraph_style (Optional[Dict[str, Any]]): Dictionary specifying the drawing style for the subgraphs.
            Expected keys:
                - 'node_size' (int): Size of the subgraph nodes.
                - 'edge_width' (float): Thickness of the subgraph edges.
                - 'edge_color' (str or list): Color of the subgraph edges.
                - 'node_alpha' (float): Transparency of the subgraph nodes.
                - 'edge_alpha' (float): Transparency of the subgraph edges.
                - 'cmap' (str or matplotlib colormap): Colormap for node coloring based on label.
        n_elements_per_row (int, optional): Number of subgraphs to display per row. Defaults to 7.
        size (float, optional): Size of each individual subplot (both width and height). Defaults to 5.0.
    
    Returns:
        None: Displays the plot using matplotlib.
    """
    if quotient_graph.number_of_quotient_nodes() == 0:
        print("No subgraphs to display.")
        return
    
    # Set default subgraph styles if not provided
    if subgraph_style is None:
        subgraph_style = {
            'node_size': 70,
            'edge_width': 1.0,
            'edge_color': 'grey',
            'node_alpha': 0.8,
            'edge_alpha': 0.5,
            'cmap': 'tab20'  # Default colormap
        }
    else:
        # Ensure all expected keys are present with defaults if missing
        subgraph_style.setdefault('node_size', 70)
        subgraph_style.setdefault('edge_width', 1.0)
        subgraph_style.setdefault('edge_color', 'grey')
        subgraph_style.setdefault('node_alpha', 0.8)
        subgraph_style.setdefault('edge_alpha', 0.5)
        subgraph_style.setdefault('cmap', 'tab20')  # Default colormap

    # Extract the quotient graph and base graph
    q_graph: nx.Graph = quotient_graph.quotient_graph
    base_graph: nx.Graph = quotient_graph.graph  # Assuming the base graph is accessible

    # Create a mapping of base graph nodes to their colors based on their labels
    label_to_color = {}
    cmap = plt.get_cmap(subgraph_style['cmap'])
    
    for node in base_graph.nodes(data=True):
        label = node[1].get('label')
        if label is not None:
            # Map the label to a unique color in the colormap
            if label not in label_to_color:
                # Map the label to a unique color in the colormap
                label_to_color[label] = cmap(len(label_to_color) / len(base_graph.nodes))
    
    # Group quotient graph nodes by their subgraph label
    label_to_nodes = defaultdict(list)  # label -> list of quotient graph node IDs
    label_to_subgraph = {}  # label -> subgraph (base graph)

    for node_id in q_graph.nodes():
        label = quotient_graph.get_quotient_node_label(node_id)
        if label is not None:
            label_to_nodes[label].append(node_id)
            subgraph = quotient_graph.subgraph(node_id)
            # To ensure consistency, store one representative subgraph per label
            if label not in label_to_subgraph:
                label_to_subgraph[label] = subgraph

    # Calculate frequencies and sort labels by frequency descending
    label_frequency = {label: len(nodes) for label, nodes in label_to_nodes.items()}
    sorted_labels = sorted(label_frequency.keys(), key=lambda x: label_frequency[x], reverse=True)

    # Determine the number of unique subgraphs
    num_subgraphs = len(sorted_labels)

    # Calculate the number of rows needed
    n_rows = (num_subgraphs + n_elements_per_row - 1) // n_elements_per_row

    # Calculate overall figure size
    fig_width = n_elements_per_row * size
    fig_height = n_rows * size

    # Create the figure and axes
    fig, axes = plt.subplots(n_rows, n_elements_per_row, figsize=(fig_width, fig_height))
    if isinstance(axes, Axes):
        axes = [axes]  # Single subplot
    else:
        axes = axes.flatten()  # Flatten in case of multiple rows

    # Iterate over sorted labels and plot each subgraph
    for idx, label in enumerate(sorted_labels):
        ax = axes[idx]
        subgraph = label_to_subgraph[label]

        # Generate layout for the subgraph
        pos = nx.kamada_kawai_layout(subgraph)

        # List to hold colors for individual nodes in the subgraph
        node_colors = []
        for node in subgraph.nodes(data=True):
            subgraph_label = node[1].get('label')
            if subgraph_label is not None:
                # Get the color for this node based on its label from the base graph
                node_colors.append(label_to_color.get(subgraph_label, 'skyblue'))  # Default to 'skyblue' if no label

        # Draw nodes with a black border
        nx.draw_networkx_nodes(
            subgraph,
            pos=pos,
            node_size=subgraph_style['node_size'],
            node_color=node_colors,
            alpha=subgraph_style['node_alpha'],
            edgecolors='black',  # Black border around nodes
            ax=ax
        )

        # Draw edges
        nx.draw_networkx_edges(
            subgraph,
            pos=pos,
            width=subgraph_style['edge_width'],
            edge_color=subgraph_style['edge_color'],
            alpha=subgraph_style['edge_alpha'],
            ax=ax
        )

        # Optional: Draw labels (commented out for clarity)
        # nx.draw_networkx_labels(subgraph, pos=pos, font_size=8, ax=ax)

        # Set title with label and frequency
        frequency = label_frequency[label]
        ax.set_title(f'Label: {label}\nOccurrences: {frequency}', fontsize=10)

        # Remove axis
        ax.axis('off')

    # Hide any unused subplots
    for idx in range(num_subgraphs, len(axes)):
        axes[idx].axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the plot
    plt.show()


def display_quotient_graph_decomposition(graph, decomposition_function, nbits=10, verbose=False):
    """
    Decomposes a base graph into its quotient graph representation and visualizes the decomposition.

    This function initializes a QuotientGraph from the provided base graph, applies a decomposition function
    to it, and then displays the resulting quotient graph along with its node subgraphs. Optionally, it can
    print the string representation of the quotient graph for debugging or informational purposes.

    Args:
        graph (networkx.Graph or QuotientGraph):
            The base graph to be decomposed. It can be a NetworkX Graph instance or an existing QuotientGraph.
        
        decomposition_function (callable):
            A function that takes a list of QuotientGraph instances and returns a list of decomposed QuotientGraph instances.
            This function encapsulates the logic for decomposing the quotient graph, such as identifying substructures or applying
            specific graph algorithms.
        
        verbose (bool, optional):
            If set to True, the function will print the string representation of the quotient graph after decomposition.
            This is useful for debugging or gaining insights into the quotient graph's structure.
            Defaults to False.

    Returns:
        None

    Example:
        ```python
        import networkx as nx

        # Define a sample decomposition function
        def sample_decomposition(qg_list):
            # For demonstration, simply return the input list without modification
            return qg_list

        # Create a base graph
        G = nx.complete_graph(5)

        # Show the quotient graph decomposition
        show_quotient_graph_decomposition(G, sample_decomposition, verbose=True)
        ```
    """
    # Initialize a QuotientGraph instance with the provided base graph
    quotient_graph = QuotientGraph(graph, nbits=nbits)
    
    # Apply the decomposition function to the quotient graph.
    # The decomposition function is expected to return a list of QuotientGraph instances.
    quotient_graph = decomposition_function(quotient_graph)
    
    # If verbose mode is enabled, print the string representation of the quotient graph.
    # This is useful for debugging or inspecting the quotient graph's details.
    if verbose:
        print(quotient_graph)
    
    # Display the visual representation of the quotient graph.
    # The 'display_quotient_graph' function is assumed to handle visualization, such as plotting the graph.
    display_quotient_graph(quotient_graph)
    
    # Display the subgraphs associated with each node in the quotient graph.
    # The 'display_quotient_graph_subgraphs' function is assumed to visualize or list subgraphs for each node.
    display_quotient_graph_subgraphs(quotient_graph)
