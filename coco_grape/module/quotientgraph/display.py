import os
import inspect
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from toolz.functoolz import curry
from networkx.drawing.nx_agraph import to_agraph
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import hashlib
from typing import Optional, Dict, Any, Tuple, List, Callable
import math
from coco_grape.module.quotientgraph.type import QuotientGraph

def _format_param_value_for_label(value: Any) -> str:
    """
    Produce a compact string for parameter values in the decomposition graph.

    - If value is a list and longer than 3 elements, show the first two, then '...', then the last.
      Example: [a, b, ..., z]
    - Otherwise, fall back to str(value).

    Note: Tuples (including length-2 tuples) are left as-is via str(value).
    """
    try:
        if isinstance(value, list) and len(value) > 3:
            # Convert elements to strings safely
            head = ", ".join(str(x) for x in value[:2])
            tail = str(value[-1])
            return f"[{head}, ..., {tail}]"
    except Exception:
        # Fallback to default string conversion if anything goes wrong
        pass
    return str(value)

def stable_hash(x: str) -> int:
    """
    Computes a stable hash from a string using MD5.

    Args:
        x: The string to hash.
    
    Returns:
        An integer hash value.
    """
    return int(hashlib.md5(x.encode('utf-8')).hexdigest(), 16)

def get_color(label: Any, cmap_name: str = 'hsv') -> Any:
    """
    Maps a label deterministically to a color via a continuous colormap.

    Previously we reduced the hash modulo 20 and used the 'tab20' palette,
    which can cause collisions (different labels producing the same color).
    Using a continuous colormap like 'hsv' and a normalized hash in [0,1)
    greatly reduces collisions while remaining stable across runs.

    Args:
        label: Any value; converted to an integer via int(...) or stable hash.
        cmap_name: Matplotlib colormap name (default 'hsv').
    
    Returns:
        A color (RGBA tuple) as returned by the colormap.
    """
    cmap = cm.get_cmap(cmap_name)
    # Always hash the label (even if it's an int) to spread values across [0,1)
    # This avoids tiny normalized values (e.g., 9708/2^32) that cluster near hue 0 (red) in 'hsv'.
    num = stable_hash(str(label))
    # Normalize the integer into [0, 1) using 32-bit space to keep it stable.
    norm = (num % (2**32)) / float(2**32)
    return cmap(norm)

def display_graph(
    graph: nx.Graph,
    ax: Optional[plt.Axes] = None,
    style: Optional[Dict[str, Any]] = None,
    pos: Optional[Dict[Any, Tuple[float, float]]] = None,
    offset: Tuple[float, float] = (0, 0),
    size: Tuple[int, int] = (5, 4)
) -> Optional[plt.Axes]:
    """
    Draws a single NetworkX graph onto a Matplotlib axis with specified styling and offset.

    Node colors are assigned based on their 'label' attribute or a stable hash.

    Args:
        graph: The NetworkX graph to display.
        ax: The Matplotlib axis to draw on. If None, a new figure and axis are created.
        style: A dict of style parameters for drawing (node_size, edge_width, etc.).
        pos: Optional pre-calculated node positions. If None, Kamada-Kawai layout is used.
        offset: A tuple (x_offset, y_offset) to shift the graph's position.
        size: The figure size if `ax` is None.

    Returns:
        The Matplotlib axis containing the visualization.
    """
    # Create figure and axis if not provided.
    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=size)
        created_ax = True

    # Set default style if not provided.
    if style is None:
        style = {
            'node_size': 70, 'edge_width': 1.0, 'edge_style': 'solid',
            'node_border_width': 0.5, 'node_alpha': 0.8, 'edge_color': 'grey',
            'cmap': 'hsv'
        }
    # Ensure all expected keys have defaults if partially provided
    style.setdefault('node_size', 70)
    style.setdefault('edge_width', 1.0)
    style.setdefault('edge_style', 'solid')
    style.setdefault('node_border_width', 0.5)
    style.setdefault('node_alpha', 0.8)
    style.setdefault('edge_color', 'grey')
    style.setdefault('cmap', 'hsv')

    # Calculate positions if not provided. Handle empty graph.
    if pos is None:
        if graph.number_of_nodes() > 0:
            pos = nx.kamada_kawai_layout(graph)
        else:
            pos = {} # Empty position dict for empty graph

    # Apply offset to positions for drawing
    final_pos = {node: (x + offset[0], y + offset[1]) for node, (x, y) in pos.items()}

    # Determine node colors.
    node_colors: List[Any] = []
    cmap_name = style.get('cmap', 'hsv')
    for node, data in graph.nodes(data=True):
        label = data.get('label', None)
        # Treat None as missing label to avoid hashing the string "None"
        if label is not None:
            node_colors.append(get_color(label, cmap_name=cmap_name))
        else:
            node_colors.append(get_color(stable_hash(str(node)), cmap_name=cmap_name))

    # Draw the graph nodes.
    nx.draw_networkx_nodes(
        graph, final_pos,
        node_size=style['node_size'],
        alpha=style['node_alpha'],
        linewidths=style['node_border_width'],
        node_color=node_colors,
        edgecolors='black', # Keep consistent border
        ax=ax
    )

    # Draw the graph edges.
    nx.draw_networkx_edges(
        graph, final_pos,
        width=style['edge_width'],
        style=style['edge_style'],
        edge_color=style['edge_color'],
        ax=ax
    )

    if created_ax:
        ax.axis('off')
        plt.show()
        return None
    else:
        return ax

# --- Refactored display Function ---
def display(
    qgraph: "QuotientGraph",
    base_style: Optional[Dict[str, Any]] = None,
    quotient_style: Optional[Dict[str, Any]] = None,
    connection_style: Optional[Dict[str, Any]] = None,
    size: Tuple[int, int] = (5, 4),
    ax: Optional[plt.Axes] = None, # Use plt.Axes for type hint
    show_legend: bool = False
) -> Optional[plt.Axes]:
    """
    Visualizes the full nested structure of a QuotientGraph using display_graph.

    The leftmost level is the base (preimage) graph and the quotient level
    (image_graph) is drawn in an additional column to the right.
    Connection lines are drawn between a node in the quotient graph and every base node
    (from the preimage graph) that appears in its associated subgraph.

    Node colors are assigned consistently based on their numerical labels or stable hash.

    Args:
        qgraph: The QuotientGraph instance to visualize.
        base_style: A dict of style parameters for drawing the base graph.
        quotient_style: A dict of style parameters for drawing the quotient graph.
        connection_style: A dict of style parameters for drawing connection lines.
        size: The figure size as a tuple (width, height) if `ax` is None.
        ax: A Matplotlib axis to draw on. If None, a new figure and axis are created.
        show_legend: If True, displays a legend.

    Returns:
        The Matplotlib axis containing the visualization.
    """
    # Set default styles (can be simplified if defaults are handled in display_graph)
    if base_style is None:
        base_style = {
            'node_size': 70, 'edge_width': 1.0, 'edge_style': 'solid',
            'node_border_width': 0.5, 'node_alpha': 0.8, 'edge_color': 'grey', 'cmap': 'hsv'
        }
    if quotient_style is None:
        quotient_style = {
            'node_size': 100, 'edge_width': 2.0, 'edge_style': 'solid',
            'node_border_width': 2.0, 'node_alpha': 0.9, 'edge_color': 'black', 'cmap': 'hsv'
        }
    if connection_style is None:
        connection_style = {
            'edge_width': 0.5, 'edge_style': 'dashed',
            'edge_color': 'grey', 'edge_alpha': 0.3
        }
    # Ensure connection style defaults
    connection_style.setdefault('edge_width', 0.5)
    connection_style.setdefault('edge_style', 'dashed')
    connection_style.setdefault('edge_color', 'grey')
    connection_style.setdefault('edge_alpha', 0.3)

    # Create figure and axis if not provided.
    if ax is None:
        fig, ax = plt.subplots(figsize=size)

    # --- Calculate Layouts ---
    # Calculate base positions. Handle empty graph.
    if qgraph.preimage_graph.number_of_nodes() > 0:
        pos_base = nx.kamada_kawai_layout(qgraph.preimage_graph)
    else:
        pos_base = {}

    # Calculate quotient positions. Handle empty graph.
    if qgraph.image_graph.number_of_nodes() > 0:
        pos_quotient = nx.kamada_kawai_layout(qgraph.image_graph)
    else:
        pos_quotient = {}

    # --- Determine Offset ---
    # Shift the quotient graph positions to the right.
    if pos_base:
        # Find max x-coordinate, handle potential empty pos_base values if layout failed partially
        valid_x_coords = [x for x, _ in pos_base.values() if isinstance(x, (int, float))]
        max_x = max(valid_x_coords) if valid_x_coords else 0
    else:
        max_x = 0
    x_offset = max_x + 1.5  # leave some space between graphs

    # --- Draw Graphs using Helper ---
    # Draw the base graph (no offset)
    ax = display_graph(
        qgraph.preimage_graph,
        ax=ax,
        style=base_style,
        pos=pos_base,
        offset=(0, 0) # Explicitly no offset
    )

    # Draw the quotient graph (with x_offset)
    ax = display_graph(
        qgraph.image_graph,
        ax=ax,
        style=quotient_style,
        pos=pos_quotient, # Pass original positions
        offset=(x_offset, 0) # Apply offset
    )

    # --- Draw Connection Lines ---
    # Need the final, offsetted positions for the quotient graph for connections
    final_pos_quotient = {node: (x + x_offset, y) for node, (x, y) in pos_quotient.items()}

    for qnode, qdata in qgraph.image_graph.nodes(data=True):
        subg = qdata.get("association")
        if subg is None:
            continue
        # Ensure qnode exists in the final positions (handles empty quotient graph case)
        if qnode not in final_pos_quotient:
            continue
        qnode_pos = final_pos_quotient[qnode]

        for base_node in subg.nodes:
            # Ensure base_node exists in base positions
            if base_node in pos_base:
                base_pos = pos_base[base_node]
                ax.plot(
                    [qnode_pos[0], base_pos[0]], # x coordinates
                    [qnode_pos[1], base_pos[1]], # y coordinates
                    linewidth=connection_style['edge_width'],
                    linestyle=connection_style['edge_style'],
                    color=connection_style['edge_color'],
                    alpha=connection_style['edge_alpha'],
                    zorder=-1 # Draw connections behind nodes
                )

    # Optionally add a legend.
    if show_legend:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Base Node',
                   markerfacecolor='grey', markersize=8, linestyle='None'), # Added linestyle='None'
            Line2D([0], [0], marker='o', color='w', label='Quotient Node',
                   markerfacecolor='grey', markersize=10, linestyle='None'), # Added linestyle='None'
            Line2D([0], [0], color=connection_style['edge_color'], lw=connection_style['edge_width'],
                   linestyle=connection_style['edge_style'], label='Mapping')
        ]
        ax.legend(handles=legend_elements, loc='best') # Added loc='best'

    ax.axis('off')
    return ax


def display_mappings(
    qgraph: "QuotientGraph",
    subgraph_style: Optional[Dict[str, Any]] = None,
    n_elements_per_row: int = 15,
    size: float = 2.0
) -> None:
    """
    Visualizes each distinct subgraph (mapping) from the quotient graph.
    
    For each distinct mapping (grouped by the image node 'label'), a representative subgraph is displayed.
    The mappings are sorted by frequency (most frequent first). Each subplot shows the representative
    subgraph drawn with the same default style as in the display function, but the node colors are those 
    of the original preimage graph nodes (i.e. for each node in the subgraph, if it has a 'label' attribute, 
    its color is computed via get_color; otherwise, a stable hash of its identifier is used).
    
    The function arranges the displays in a grid with n_elements_per_row per row and sets the size for each subgraph.
    
    Args:
        qgraph: The QuotientGraph instance to visualize.
        subgraph_style: A dict of style parameters for drawing each subgraph.
        n_elements_per_row: Number of subgraph displays per row.
        size: The size (in inches) for each individual subgraph display.
    """
    if qgraph is None or qgraph.image_graph is None or len(qgraph.image_graph.nodes) == 0:
        print("[display_mappings] Empty quotient graph — nothing to display.")
        return

    # Set default subgraph style if not provided.
    if subgraph_style is None:
        subgraph_style = {
            'node_size': 70,
            'edge_width': 1.0,
            'edge_style': 'solid',
            'node_border_width': 0.5,
            'node_alpha': 0.8,
            'edge_color': 'grey',
            'cmap': 'hsv'
        }
    else:
        subgraph_style.setdefault('cmap', 'hsv')
    
    # Group image nodes by their label.
    mapping_dict: Dict[Any, List[nx.Graph]] = {}
    for node, data in qgraph.image_graph.nodes(data=True):
        label = data.get("label")
        if label is None:
            label = stable_hash(str(node))
        mapping_dict.setdefault(label, []).append(data["association"])
    
    # Sort the mappings by frequency (number of subgraphs) in descending order.
    sorted_mappings = sorted(mapping_dict.items(), key=lambda item: len(item[1]), reverse=True)
    
    n_mappings = len(sorted_mappings)
    n_rows = math.ceil(n_mappings / n_elements_per_row)
    fig, axes = plt.subplots(n_rows, n_elements_per_row, figsize=(n_elements_per_row * size, n_rows * size))
    
    # Flatten axes array to a 1D list.
    if n_rows == 1:
        axes = list(axes)
    else:
        axes = [ax for row in axes for ax in row]
    
    # For each distinct mapping, display a representative subgraph.
    for i, (label, subgraph_list) in enumerate(sorted_mappings):
        freq = len(subgraph_list)
        rep_subgraph = subgraph_list[0]
        # Use Kamada-Kawai layout if the subgraph is non-empty.
        pos = nx.kamada_kawai_layout(rep_subgraph) if rep_subgraph.number_of_nodes() > 0 else {}
        
        # Compute node colors from the original preimage graph node attributes.
        node_colors = []
        cmap_name = subgraph_style.get('cmap', 'tab20')
        for node, data in rep_subgraph.nodes(data=True):
            if 'label' in data:
                node_colors.append(get_color(data['label'], cmap_name=cmap_name))
            else:
                node_colors.append(get_color(stable_hash(str(node)), cmap_name=cmap_name))
        
        ax = axes[i]
        ax.set_title(f"Label: {label}\nFreq: {freq}", fontsize=8)
        ax.axis("off")
        
        # Draw the nodes and edges of the representative subgraph.
        nx.draw_networkx_nodes(
            rep_subgraph, pos,
            node_size=subgraph_style['node_size'],
            alpha=subgraph_style['node_alpha'],
            linewidths=subgraph_style['node_border_width'],
            node_color=node_colors,
            edgecolors='black',
            ax=ax
        )
        nx.draw_networkx_edges(
            rep_subgraph, pos,
            width=subgraph_style['edge_width'],
            style=subgraph_style['edge_style'],
            edge_color=subgraph_style['edge_color'],
            ax=ax
        )
    
    # Hide any unused subplots.
    for j in range(n_mappings, len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()
    plt.show()

# ===========================
# UTILITY FUNCTIONS
# ===========================
def get_underlying_function(f):
    """
    Unwraps a function to retrieve its underlying callable.
    """
    if hasattr(f, 'func'):
        return f.func
    if hasattr(f, '__wrapped__'):
        return f.__wrapped__
    return f

# Global constant for the initial input node.
GLOBAL_INPUT_NODE = "QuotientGraph"

# ===========================
# PARAMETER SUBGRAPH BUILDING
# ===========================
def build_parameter_subgraph(param_func, G, global_input):
    """
    For a callable parameter value:
      - If it accepts only one argument, assume it uses the global input.
      - Otherwise, recursively build its full decomposition subgraph.
    """
    if hasattr(param_func, '__code__') and param_func.__code__.co_argcount == 1:
        return global_input
    else:
        dummy_key = f"ParamInput_{id(param_func)}"
        # Ensure the global input node is present.
        if not G.has_node(GLOBAL_INPUT_NODE):
            G.add_node(GLOBAL_INPUT_NODE, data_type="global", label="QuotientGraph")
        else:
            dummy_key = GLOBAL_INPUT_NODE  # reuse global input
        output_key = build_decomposition_subgraph(param_func, dummy_key, G)
        return output_key

def add_parameters(G, parent_key, comp_func, global_input):
    """
    For each parameter of comp_func, creates two nodes:
      - A parameter value node (data_type "value").
      - A parameter name node (data_type "parameter").
    Then links the value node to the name node and connects the name node to parent_key.
    """
    # Process keyword parameters.
    if hasattr(comp_func, 'keywords') and comp_func.keywords:
        for key, value in comp_func.keywords.items():
            if callable(value):
                value_node_key = build_parameter_subgraph(value, G, global_input)
            else:
                # Use a composite key to ensure each occurrence is unique.
                value_node_key = f"param_value_{key}_{id(comp_func)}_{id(value)}"
                G.add_node(value_node_key, data_type="value", label=_format_param_value_for_label(value))
            param_name_key = f"param_name_{key}_{id(comp_func)}"
            G.add_node(param_name_key, data_type="parameter", label=key)
            G.add_edge(value_node_key, param_name_key)
            G.add_edge(param_name_key, parent_key)
    # Process positional parameters.
    if hasattr(comp_func, 'args') and comp_func.args:
        for i, arg in enumerate(comp_func.args):
            if callable(arg):
                value_node_key = build_parameter_subgraph(arg, G, global_input)
            else:
                value_node_key = f"param_value_arg{i}_{id(comp_func)}_{id(arg)}"
                G.add_node(value_node_key, data_type="value", label=_format_param_value_for_label(arg))
            param_name_key = f"param_name_arg{i}_{id(comp_func)}"
            G.add_node(param_name_key, data_type="parameter", label=f"arg{i}")
            G.add_edge(value_node_key, param_name_key)
            G.add_edge(param_name_key, parent_key)

# ===========================
# DECOMPOSITION GRAPH BUILDING
# ===========================
def build_decomposition_subgraph(comp_func, input_node, G):
    """
    Recursively builds a subgraph for a composite function (comp_func) starting at input_node.

    Operator handling:
      - For operators (with attribute operator_type "add" or "product"),
        create an operator node (data_type "operator") and recursively add its children.

    Composition flattening:
      - For compose/forward_compose calls (identified by their __name__ and chain attribute),
        flatten the chain (reversing for "compose" and preserving order for "forward_compose").

    Leaf nodes:
      - If comp_func is callable, create a node with data_type "function"; otherwise, "value".
      - Parameter nodes are added using add_parameters.
      
    Returns the key of the output node.
    """
    # --- Operator handling ---
    if hasattr(comp_func, 'operator_type'):
        op_type = comp_func.operator_type
        if op_type == "add":
            operator_key = f"operator_add_{id(comp_func)}"
            G.add_node(operator_key, data_type="operator", label="+")
            for child in comp_func.decomposition_functions:
                child_output = build_decomposition_subgraph(child, input_node, G)
                G.add_edge(child_output, operator_key)
            return operator_key
        elif op_type == "product":
            combiner = comp_func.combiner
            underlying_combiner = get_underlying_function(combiner)
            combiner_name = getattr(underlying_combiner, '__name__', repr(underlying_combiner))
            operator_key = f"operator_product_{id(comp_func)}"
            G.add_node(operator_key, data_type="operator", label=combiner_name)
            if hasattr(combiner, 'keywords') or hasattr(combiner, 'args'):
                add_parameters(G, operator_key, combiner, input_node)
            for child in comp_func.decomposition_functions:
                child_output = build_decomposition_subgraph(child, input_node, G)
                G.add_edge(child_output, operator_key, style="bold", penwidth="3")
            return operator_key

    # --- Composition flattening ---
    underlying = get_underlying_function(comp_func)
    if getattr(underlying, '__name__', None) in ("compose", "forward_compose") and hasattr(comp_func, "chain"):
        funcs = list(comp_func.chain)
        chain = list(reversed(funcs)) if underlying.__name__ == "compose" else funcs
    else:
        # Attempt to extract a function chain from the closure if available.
        funcs = []
        if inspect.isfunction(underlying):
            try:
                closure = inspect.getclosurevars(underlying)
                for _, val in closure.nonlocals.items():
                    if isinstance(val, tuple) and all(callable(x) for x in val):
                        funcs = list(val)
                        break
            except Exception:
                funcs = []
        chain = funcs

    if chain:
        current_input = input_node
        for f in chain:
            # If the function is an operator, recursively build its subgraph.
            if hasattr(f, 'operator_type'):
                child_output = build_decomposition_subgraph(f, current_input, G)
            else:
                underlying_f = get_underlying_function(f)
                func_name = getattr(underlying_f, '__name__', repr(underlying_f))
                node_key = f"{func_name}_{id(f)}"
                G.add_node(node_key, data_type="function", label=func_name)
                G.add_edge(current_input, node_key)
                if hasattr(f, 'keywords') or hasattr(f, 'args'):
                    add_parameters(G, node_key, f, input_node)
                child_output = node_key
            current_input = child_output
        return current_input
    else:
        # --- Leaf node handling ---
        real_name = getattr(underlying, '__name__', repr(underlying))
        if real_name == "QuotientGraph":
            return GLOBAL_INPUT_NODE
        leaf_key = f"{real_name}_{id(comp_func)}"
        data_type = "function" if callable(comp_func) else "value"
        G.add_node(leaf_key, data_type=data_type, label=real_name)
        G.add_edge(input_node, leaf_key)
        if hasattr(comp_func, 'keywords') or hasattr(comp_func, 'args'):
            add_parameters(G, leaf_key, comp_func, input_node)
        return leaf_key

def decomposition_to_graph(comp_func) -> nx.DiGraph:
    """
    Builds a full decomposition graph for comp_func.
    The graph starts with a single global input node and builds the subgraph recursively.
    """
    G = nx.DiGraph()
    G.add_node(GLOBAL_INPUT_NODE, data_type="global", label="QGraph")
    build_decomposition_subgraph(comp_func, GLOBAL_INPUT_NODE, G)
    return G

# ===========================
# GRAPH DISPLAY FUNCTION
# ===========================
def display_decomposition_graph(comp_func_or_graph, output_file: str = "decomposition_graph.png", figsize=(12, 8)) -> None:
    """
    Draws and displays the decomposition graph for a composite function.
    
    Node data_types and their corresponding visual styles:
      - "global": Global input node.
      - "value": Parameter value node.
      - "function": Function node.
      - "parameter": Parameter name node.
      - "operator": Operator node.
    
    The graph is laid out using Graphviz's dot (bottom-to-top orientation) and saved to output_file.
    """
    # Mapping from our data_type to Graphviz attributes.
    DATA_TYPE_STYLES = {
        "global": {"shape": "circle", "fillcolor": "ivory"},
        "value": {"shape": "circle", "fillcolor": "ivory"},
        "function": {"shape": "rectangle", "fillcolor": "lightskyblue2"},
        "parameter": {"shape": "oval", "fillcolor": "goldenrod1"},
        "operator": {"shape": "hexagon", "fillcolor": "lightsteelblue"},
    }
    
    if isinstance(comp_func_or_graph, nx.DiGraph):
        G = comp_func_or_graph
    else:
        G = decomposition_to_graph(comp_func_or_graph)
    
    A = to_agraph(G)
    
    for n in A.nodes():
        node_name = n.get_name()
        display_label = G.nodes[node_name].get("label", node_name)
        # Use "data_type" instead of "shape"
        data_type = G.nodes[node_name].get("data_type", "parameter")
        style = DATA_TYPE_STYLES.get(data_type, {"shape": "oval", "fillcolor": "grey"})
        
        n.attr['shape'] = style["shape"]
        n.attr['style'] = 'filled'
        n.attr['fillcolor'] = style["fillcolor"]
        n.attr['label'] = display_label
        
        if data_type == "operator":
            n.attr['fontsize'] = '14'
    
    # Set edge thickness based on the data_type of the tail node.
    for edge in A.edges():
        tail = edge[0]
        tail_data_type = G.nodes[tail].get("data_type", "parameter")
        if tail_data_type == "function":
            if not edge.attr.get("penwidth"):
                edge.attr["penwidth"] = "3"
        elif tail_data_type == "parameter":
            if not edge.attr.get("penwidth"):
                edge.attr["penwidth"] = "1"
        else:
            if not edge.attr.get("penwidth"):
                edge.attr["penwidth"] = "1"
    
    A.graph_attr.update(
        rankdir="BT",
        nodesep="0.8",
        ranksep="1.2",
        splines="true",
        overlap="false"
    )
    
    A.layout(prog="dot")
    A.draw(output_file)
    
    if os.path.exists(output_file):
        img = mpimg.imread(output_file)
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.axis("off")
        plt.show()
    else:
        print("Error: output file was not created.")
