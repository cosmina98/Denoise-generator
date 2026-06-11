import os
import inspect
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from toolz.functoolz import curry
from networkx.drawing.nx_agraph import to_agraph

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
                G.add_node(value_node_key, data_type="value", label=str(value))
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
                G.add_node(value_node_key, data_type="value", label=str(arg))
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
