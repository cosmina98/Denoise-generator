#!/usr/bin/env python
"""Provides interface."""

import numpy as np
import networkx as nx
import pylab as plt
from collections import Counter
from pprint import pprint
from coco_grape.module.construct import construct
import copy
import textwrap
#pip install pyvis
from pyvis.network import Network

def textwrap_node_attributes(graph, input_key_list, output_key, col_len=40):
    for u in graph.nodes():
        txt = [textwrap.fill(graph.nodes[u][key], 40) for key in input_key_list]
        txt = '\n\n'.join(txt)
        graph.nodes[u][output_key] = txt

def dynamic_graph(orig_graph, hide_edges=False, node_label_key='label', node_title_key=None, node_group_key=None, edge_label_key='label', input_key_list=None,output_key=None, col_len=40, night_mode=False, toggle_physics=True):
    #invoke as show_dynamic_graph(G).show('n.html')
    graph = nx.convert_node_labels_to_integers(orig_graph)
    if input_key_list is not None and output_key is not None: textwrap_node_attributes(graph, input_key_list=input_key_list, output_key=output_key, col_len=40)
    G = nx.Graph()
    G.add_nodes_from(graph.nodes())
    nx.set_node_attributes(G, nx.get_node_attributes(graph, name=node_label_key), 'label')
    if node_title_key is not None: nx.set_node_attributes(G, nx.get_node_attributes(graph, name=node_title_key), 'title')
    if node_group_key is not None: nx.set_node_attributes(G, nx.get_node_attributes(graph, name=node_group_key), 'group')
    G.add_edges_from(graph.edges())
    nx.set_edge_attributes(G, nx.get_edge_attributes(graph, name=edge_label_key), 'title')
    nx.set_edge_attributes(G, hide_edges, 'hidden')
    if night_mode: graph = Network(height="750px", width="100%", notebook=True, bgcolor="#222222", font_color="white")
    else: graph = Network(height="750px", width="100%", notebook=True)
    graph.from_nx(G, default_node_size=7, default_edge_weight=1)
    graph.toggle_physics(toggle_physics)
    graph.show_buttons(filter_=['physics'])
    graph.repulsion(node_distance=100, spring_length=100, spring_strength=0.05, damping=0.08)
    return graph


def graph_to_string(graph, use_vec=True, use_compact=True):
    txt = ''
    txt += 'Nodes\n'
    for u in graph.nodes():
        if use_vec: attributes = graph.nodes[u]
        else: 
            attributes = copy.deepcopy(graph.nodes[u])
            attributes.pop('vec', None)
        if not use_compact: 
            attributes = '\n'.join(['%s:%s'%(key,attributes[key]) for key in attributes])
        txt += 'node_id=%s   attributes=%s'%(u, attributes)
        txt += '\n'

    if nx.number_of_edges(graph)>0:
        txt += '\n'
        txt += 'Edges\n'
        for u,v in graph.edges():
            if use_vec: attributes = graph.edges[u,v]
            else: 
                attributes = copy.deepcopy(graph.edges[u,v])
                attributes.pop('vec', None)
            if not use_compact: 
                attributes = '\n'.join(['%s:%s'%(key,attributes[key]) for key in attributes])
            txt += 'node_ids=%s,%s   attributes=%s'%(u,v, attributes)
            txt += '\n'
    return txt

def graphofsubgraphs2graph_disjoint(graphofsubgraphs):
    gg = graphofsubgraphs
    n = gg.number_of_nodes()
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i, u in enumerate(gg.nodes()):
        graph.nodes[i]['label'] = gg.nodes[u]['label']
        graph.nodes[i]['level'] = 'high'
        n_start = len(graph)
        subgraph = gg.nodes[u]['subgraph']
        nx.set_edge_attributes(subgraph, 'base', 'level')
        nx.set_node_attributes(subgraph, 'base', 'level')
        graph = nx.disjoint_union(graph, subgraph)
        n_end = len(graph)
        for v in range(n_start, n_end):
            graph.add_edge(i, v, label='interlevel', level='interlevel')
    for i, u in enumerate(gg.nodes()):
        for j, v in enumerate(gg.nodes()):
            if gg.has_edge(u,v):
                label = gg.edges[u,v]['label']
                graph.add_edge(i, j, label=label, level='high')
    return graph


def graphofsubgraphs2graph_union(graphofsubgraphs):
    base_graph = graphofsubgraphs.graph['base']
    max_node_id = max(u for u in base_graph.nodes())
    nx.set_node_attributes(base_graph, 'base', 'level')
    nx.set_edge_attributes(base_graph, 'base', 'level')
    graph = nx.relabel_nodes(graphofsubgraphs, lambda u: u + max_node_id + 1)
    nx.set_node_attributes(graph, 'high', 'level')
    nx.set_edge_attributes(graph, 'high', 'level')
    graph = nx.union(graph, base_graph)
    for u in graph.nodes():
        if graph.nodes[u]['level'] == 'high':
            subgraph = graph.nodes[u]['subgraph']
            for v in subgraph.nodes():
                graph.add_edge(u, v, label='interlevel', level='interlevel')
    return graph    


def graphofsubgraphs2graph(graphofsubgraphs, use_disjoint=True):
    if use_disjoint is True:
        return graphofsubgraphs2graph_disjoint(graphofsubgraphs)
    else:
        return graphofsubgraphs2graph_union(graphofsubgraphs)

def filter_graph(graph, use_base_level=True, use_high_level=True):
    nbunch = []
    if use_base_level is True:
        nbunch += [u for u in graph.nodes() if graph.nodes[u]['level']=='base']
    if use_high_level is True:
        nbunch += [u for u in graph.nodes() if graph.nodes[u]['level']=='high']
    return nx.subgraph(graph, nbunch)


def draw_graphs(
    graphs,
    titles=None,
    use_pos=False,
    label='label',
    node_color='label',
    edge_color=None,
    secondary_labels=None,
    edge_label=None,
    secondary_edge_labels=None,
    size=3,
    n_graphs_per_line=13,
    cmap='Set3',
    edge_cmap='hot',
    node_size=100,
    draw_self_loops=False,
    edge_width=1,
    use_consistent_colors_from_graphs=None,
    use_base_level=True,
    use_high_level=True,
    use_abstraction=True,
    node_marker_edge_color='k',
    n_text_cols=80,
    ax=None  # Added ax parameter
):
    """
    Draws multiple NetworkX graphs with various customization options.

    Parameters:
    - graphs (list or NetworkX graph): Graphs to be drawn.
    - titles (list or None): Titles for each graph.
    - use_pos (bool): Whether to use predefined positions.
    - label (str): Node label attribute.
    - node_color (str): Node color attribute.
    - edge_color (str or None): Edge color attribute.
    - secondary_labels (list or None): Secondary node label attributes.
    - edge_label (str or None): Edge label attribute.
    - secondary_edge_labels (list or None): Secondary edge label attributes.
    - size (float): Size factor for the figure.
    - n_graphs_per_line (int): Number of graphs per line.
    - cmap (str): Colormap for node colors.
    - edge_cmap (str): Colormap for edge colors.
    - node_size (int): Size of the nodes.
    - draw_self_loops (bool): Whether to draw self-loops.
    - edge_width (float): Base width of the edges.
    - use_consistent_colors_from_graphs (list or None): Graphs to derive consistent colors.
    - use_base_level (bool): Whether to use base level in abstraction.
    - use_high_level (bool): Whether to use high level in abstraction.
    - use_abstraction (bool): Whether to use abstraction.
    - node_marker_edge_color (str): Edge color of node markers.
    - n_text_cols (int): Number of text columns for labels.
    - ax (matplotlib.axes.Axes or None): Matplotlib axis to draw the graph on.
    """

    assert len(graphs) > 0, 'ERROR: no graphs'

    def draw_graph(graph, title, ax):
        """
        Draws a single NetworkX graph on the given axis.

        Parameters:
        - graph (NetworkX graph): The graph to draw.
        - title (str or None): Title for the graph.
        - ax (matplotlib.axes.Axes): Axis to draw the graph on.
        """
        def make_label_string(graph, node_id, label, secondary_labels):
            txt = f"{graph.nodes[node_id].get(label, '')}"
            txt = textwrap.fill(txt, n_text_cols)
            if secondary_labels is not None:
                for secondary_label in secondary_labels:
                    txt_ = f"{graph.nodes[node_id].get(secondary_label, '')}"
                    txt_ = textwrap.fill(txt_, n_text_cols)
                    txt += '\n' + txt_
            return txt

        def make_edge_label_string(graph, src_node_id, dst_node_id, label, secondary_labels):
            txt = f"{graph.edges[src_node_id, dst_node_id].get(label, '')}"
            txt = textwrap.fill(txt, n_text_cols)
            if secondary_labels is not None:
                for secondary_label in secondary_labels:
                    txt_ = f"\n{graph.edges[src_node_id, dst_node_id].get(secondary_label, '')}"
                    txt_ = textwrap.fill(txt_, n_text_cols)
                    txt += '\n' + txt_
            return txt

        # Compute widths of nodes according to level
        node_levels = [graph.nodes[u].get('level', 'unknown') for u in graph.nodes()]
        node_widths = []
        for node_level in node_levels:
            if node_level == 'base':
                node_widths.append(2)
            elif node_level == 'high':
                node_widths.append(0.5)
            else:
                node_widths.append(1)

        # Determine positions
        if use_pos:
            fixed_pos = {u: graph.nodes[u]['coords'] for u in graph.nodes() if graph.nodes[u].get('coords', False) is not False}
            pos = nx.spring_layout(graph, pos=fixed_pos, fixed=list(fixed_pos.keys()))
        else:
            pos = nx.kamada_kawai_layout(graph)

        # Handle node colors
        if node_color is not None:
            node_colors = [node_labels_color_map.get(str(graph.nodes[u].get(node_color, -1)), 0) for u in graph.nodes()]
            nx.draw_networkx_nodes(
                graph,
                pos,
                node_size=node_size,
                node_color=node_colors,
                cmap=cmap,
                vmin=0,
                vmax=1,
                linewidths=node_widths,
                edgecolors=node_marker_edge_color,
                ax=ax
            )
        else:
            nx.draw_networkx_nodes(
                graph,
                pos,
                node_size=node_size,
                node_color='w',
                linewidths=node_widths,
                edgecolors=node_marker_edge_color,
                ax=ax
            )

        # Handle self-loops
        orig_graph = graph
        if not draw_self_loops:
            non_self_loop_graph = nx.Graph(graph)
            self_loops = [(u, v) for u, v in graph.edges() if u == v]
            non_self_loop_graph.remove_edges_from(self_loops)
            graph = non_self_loop_graph

        # Compute widths and styles of edges according to level
        edge_levels = [graph.edges[u, v].get('level', 'unknown') for u, v in graph.edges()]
        edge_widths = []
        edge_styles = []
        edge_color_values = []
        for edge_level in edge_levels:
            if edge_level == 'base':
                edge_widths.append(3 * edge_width)
                edge_styles.append('-')
                edge_color_values.append('gray')
            elif edge_level == 'high':
                edge_widths.append(1.5 * edge_width)
                edge_styles.append('-')
                edge_color_values.append('navy')
            elif edge_level == 'interlevel':
                edge_widths.append(0.15 * edge_width)
                edge_styles.append('-.')
                edge_color_values.append('gray')
            else:
                edge_widths.append(1 * edge_width)
                edge_styles.append('-')
                edge_color_values.append('gray')

        # Handle edge colors
        if edge_color is not None:
            edge_colors = [edge_labels_color_map.get(str(graph.edges[u, v].get(edge_color, 0)), 'gray') for u, v in graph.edges()]
            nx.draw_networkx_edges(
                graph,
                pos,
                width=edge_widths,
                style=edge_styles,
                edge_color=edge_colors,
                edge_cmap=plt.get_cmap(edge_cmap),
                ax=ax
            )
        else:
            nx.draw_networkx_edges(
                graph,
                pos,
                width=edge_widths,
                style=edge_styles,
                edge_color=edge_color_values,
                ax=ax
            )

        # Draw edge labels
        if edge_label is not None:
            edge_labels_dict = {
                (u, v): make_edge_label_string(graph, u, v, edge_label, secondary_edge_labels)
                for u, v in graph.edges() if u != v
            }
            nx.draw_networkx_edge_labels(
                graph,
                pos,
                edge_labels=edge_labels_dict,
                font_size=7,
                ax=ax
            )
            if draw_self_loops:
                self_loop_labels = {
                    (u, v): make_edge_label_string(graph, u, v, edge_label, secondary_edge_labels)
                    for u, v in graph.edges() if u == v
                }
                self_loop_pos = {key: (pos[key][0], pos[key][1] + 0.1) for key in pos}
                nx.draw_networkx_edge_labels(
                    graph,
                    self_loop_pos,
                    edge_labels=self_loop_labels,
                    font_size=7,
                    ax=ax
                )

        # Draw node labels
        if label is not None:
            node_labels = {
                u: make_label_string(graph, u, label, secondary_labels)
                for u in graph.nodes()
            }
            nx.draw_networkx_labels(
                graph,
                pos,
                labels=node_labels,
                font_size=7,
                ax=ax
            )

        # Finalize axis
        ax.axis('off')
        if title is not None:
            ax.set_title(title)

    # Handle single graph by ensuring it's in a list
    if not isinstance(graphs, list):
        graphs = [graphs]
        if titles is None:
            titles = [None]
        else:
            titles = [titles]
    else:
        if titles is None:
            titles = [None] * len(graphs)

    # Apply abstraction filters if necessary
    if use_abstraction and 'base' in graphs[0].graph:
        graphs = [filter_graph(graphofsubgraphs2graph(graph, use_disjoint=False), use_base_level, use_high_level) for graph in graphs]

    # Handle consistent colors
    if use_consistent_colors_from_graphs is not None:
        color_graphs = use_consistent_colors_from_graphs
    else:
        color_graphs = graphs

    # Create color maps for nodes and edges
    if node_color is not None:
        unique_node_labels = sorted(set(
            str(graph.nodes[u].get(node_color, 0)) for graph in color_graphs for u in graph.nodes()
        ))
        if use_abstraction and 'base' in color_graphs[0].graph:
            unique_node_labels += sorted(set(
                str(graph.graph['base'].nodes[u].get(node_color, 0)) for graph in color_graphs for u in graph.graph['base'].nodes()
            ))
        node_labels_color_map = {label: i / len(unique_node_labels) for i, label in enumerate(unique_node_labels)}
    if edge_color is not None:
        unique_edge_labels = sorted(set(
            str(graph.edges[u, v].get(edge_color, 0)) for graph in color_graphs for u, v in graph.edges()
        ))
        if use_abstraction and 'base' in color_graphs[0].graph:
            unique_edge_labels += sorted(set(
                str(graph.graph['base'].edges[u, v].get(edge_color, 0)) for graph in color_graphs for u, v in graph.graph['base'].edges()
            ))
        edge_labels_color_map = {label: i / len(unique_edge_labels) for i, label in enumerate(unique_edge_labels)}

    n = len(graphs)

    # If a single graph
    if n == 1:
        if ax is not None:
            # Use the provided axis
            draw_graph(graphs[0], titles[0], ax)
        else:
            # Create a new figure and axis
            fig, ax = plt.subplots(1, 1, figsize=(size, size))
            draw_graph(graphs[0], titles[0], ax)
            plt.tight_layout()
            plt.show()
    else:
        # Determine the number of lines (rows)
        n_lines = int(np.ceil(n / n_graphs_per_line))
        fig, axs = plt.subplots(n_lines, n_graphs_per_line, figsize=(size * n_graphs_per_line, size * n_lines))
        
        # Ensure axs is a 2D array for consistent indexing
        if n_lines == 1 and n_graphs_per_line == 1:
            axs = np.array([[axs]])
        elif n_lines == 1:
            axs = np.expand_dims(axs, axis=0)
        elif n_graphs_per_line == 1:
            axs = np.expand_dims(axs, axis=1)
        else:
            axs = np.array(axs)

        for idx, (graph, title) in enumerate(zip(graphs, titles)):
            row = idx // n_graphs_per_line
            col = idx % n_graphs_per_line
            draw_graph(graph, title, axs[row, col])

        # Turn off any remaining empty subplots
        total_subplots = n_lines * n_graphs_per_line
        for idx in range(n, total_subplots):
            row = idx // n_graphs_per_line
            col = idx % n_graphs_per_line
            axs[row, col].axis('off')

        plt.tight_layout()
        plt.show()


def display_subgraphs(graphofsubgraphs, n_graphs_per_line=11, size=3, max_num_graphs=None):
    labels = [graphofsubgraphs.nodes[u]['label'] for u in graphofsubgraphs.nodes()]
    graphs_counter = Counter(labels)
    graphs_dict = {graphofsubgraphs.nodes[u]['label']:graphofsubgraphs.nodes[u]['subgraph'] for u in graphofsubgraphs.nodes()}
    dim = len(graphs_dict)
    print('#%d (#%d)'%(len(labels), dim))
    gids = sorted(graphs_counter, key=lambda gid:graphs_counter[gid], reverse=True)
    if max_num_graphs is not None: gids = gids[:max_num_graphs]
    titles = ['%s [#%d]'%(gid,graphs_counter[gid]) for gid in gids]
    subgraphs = [graphs_dict[gid] for gid in gids]
    draw_graphs(subgraphs, titles, size=size, n_graphs_per_line=n_graphs_per_line, use_consistent_colors_from_graphs=subgraphs)


def display_graph(graph, decomposition_function, nbits=10, size=7, n_graphs_per_line=11, feature_size=3, max_num_graphs=None):
    gog = decomposition_function(construct(graph, nbits=nbits))
    draw_graphs([gog], size=size)
    display_subgraphs(gog, n_graphs_per_line, size=feature_size, max_num_graphs=max_num_graphs)

