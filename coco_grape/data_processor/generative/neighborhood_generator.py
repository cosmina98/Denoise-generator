from toolz import partition_all
import multiprocessing_on_dill as mp
import time
import random
import numpy as np
import networkx as nx
import random
from collections import defaultdict
from itertools import permutations

from coco_grape.module.graph_hash import graph_hash
from coco_grape.utils.pareto import pareto_select
from coco_grape.module.construct import construct
from coco_grape.module.composition import compose, add

from coco_grape.module.graph_duplicate_detection_estimator import GraphDuplicateDetectionEstimator

import logging
logger = logging.getLogger('GO')
#logger.setLevel(logging.INFO)
logger.setLevel(logging.WARNING)

def annotate_elapsed_time(msg, start, indent=0):
    end = time.time()
    elapsed = end - start
    if elapsed < 60:
        sfx = ' in %.1f s ' % (elapsed)
    elif elapsed < 3600:
        sfx = ' in %.1f m ' % (elapsed / 60)
    else:
        sfx = ' in %.1f h' % (elapsed / 3600)
    msg += sfx
    if indent > 0:
        pfx = ' ' * (2 * indent)
        msg = pfx + msg
    return msg


#-------------------------------------------------------------------------------------------------------------------------------------
class BaseNeighborhoodGenerator(object):

    def __init__(self, size=None):
        self.size = size
        
    def fit(self, graphs, targets=None):
        return self

    def graph_to_parts(self, graph, **kwargs):
        parts = [[node_id] for node_id in graph.nodes()]
        return parts
    
    def graph_and_part_to_neighbors(self, graph, part):
        return [graph]
    
    def graph_to_parts_and_neighbors(self, graph):
        parts = self.graph_to_parts(graph)
        if self.size is not None:
            parts = random.sample(parts, k=min(self.size, len(parts)))
        parts_and_neighbors = [(part, self.graph_and_part_to_neighbors(graph, part)) for part in parts]
        return parts_and_neighbors
        
    def neighbors(self, graph):
        parts_and_neighbors = self.graph_to_parts_and_neighbors(graph)
        neighbors_ = sum([neighs for part, neighs in parts_and_neighbors],[])
        if self.size is not None:
            neighbors_ = random.sample(neighbors_, k=min(self.size, len(neighbors_)))
        return neighbors_


class NeighborhoodEdgeSwap(BaseNeighborhoodGenerator):

    def __init__(self, size=None):
        self.size = size
        
    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def graph_and_part_to_neighbors(self, graph, part):
        def _swap_edge(g, e1, e2):
            s1, d1 = e1
            s2, d2 = e2
            ne1 = (s1, d2)
            ne2 = (s2, d1)
            args_e1 = g.edges[e1[0], e1[1]]
            args_e2 = g.edges[e2[0], e2[1]]
            g.remove_edge(*e1)
            g.remove_edge(*e2)
            g.add_edge(*ne1, **args_e1)
            g.add_edge(*ne2, **args_e2)

        graphs_ = []
        es = [e for e in graph.edges()]
        s1 = part[0]
        for d1 in graph.neighbors(s1):
            e1 = (s1,d1)
            for e2 in es:
                g = graph.copy()
                s1, d1 = e1
                s2, d2 = e2
                ne1 = (s1, d2)
                ne2 = (s2, d1)
                if s1 == s2 or s1 == d2 or s1 == d1 or d1 == s2 or d1 == d2 or s2 == d2 or g.has_edge(*ne1) or g.has_edge(*ne2):
                    continue
                _swap_edge(g, e1, e2)
                graphs_.append(g)
        return graphs_


class NeighborhoodEdgeRemove(BaseNeighborhoodGenerator):

    def __init__(self, size=None, return_largest_component=False):
        self.size = size
        self.return_largest_component = return_largest_component
            
    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def graph_and_part_to_neighbors(self, graph, part):
        graphs_ = []
        u = part[0]
        neighbors = set(graph.neighbors(u))
        for v in neighbors:
            g = graph.copy()
            g.remove_edge(u, v)
            if self.return_largest_component:
                max_cc = max(nx.connected_components(g), key=lambda x: len(x))
                graphs_.append(nx.subgraph(g, max_cc).copy())
            else:
                graphs_.append(g)
        return graphs_
    

class NeighborhoodNodeSmooth(BaseNeighborhoodGenerator):

    def __init__(self, size=None):
        self.size = size
        
    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def graph_and_part_to_neighbors(self, graph, part):
        graphs_ = []
        u = part[0]
        neighbors = set(graph.neighbors(u))
        for v in neighbors:
            v_neighbors = set(graph.neighbors(v))
            for w in v_neighbors:
                if w != u:
                    g = graph.copy()
                    g.add_edge(u, w, label=graph.edges[v,w]['label'])
                    g.remove_node(v)
                    graphs_.append(g)
        return graphs_
    

class NeighborhoodEdgeContract(BaseNeighborhoodGenerator):

    def __init__(self, size=None):
        self.size = size
        
    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def graph_and_part_to_neighbors(self, graph, part):
        graphs_ = []
        u = part[0]
        neighbors = set(graph.neighbors(u))
        for v in neighbors:
            v_neighbors = set(graph.neighbors(v))
            g = graph.copy()
            g.remove_node(v)
            for w in v_neighbors:
                if w != u:
                    g.add_edge(u, w, label=graph.edges[v,w]['label'])        
            graphs_.append(g)
        return graphs_


class NeighborhoodEdgeAdd(BaseNeighborhoodGenerator):

    def __init__(self, size=None):
        self.size = size
        
    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def fit(self, graphs, targets=None):
        self.edge_labels = set([g.edges[e]['label'] for g in graphs for e in g.edges()])
        return self

    def graph_and_part_to_neighbors(self, graph, part):
        graphs_ = []
        u = part[0]
        neighbors = set(graph.neighbors(u))
        nodes = set(graph.nodes())
        non_neighbors = nodes.difference(neighbors)
        for v in non_neighbors:
            for label in self.edge_labels:
                g = graph.copy()
                g.add_edge(u, v, label=label)
                graphs_.append(g)
        return graphs_


class NeighborhoodEdgeLabelMutation(BaseNeighborhoodGenerator):

    def __init__(self, size=None):
        self.size = size
        
    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def fit(self, graphs, targets=None):
        self.edge_labels = set([g.edges[e]['label'] for g in graphs for e in g.edges()])
        return self

    def graph_and_part_to_neighbors(self, graph, part):
        graphs_ = []
        u = part[0]
        neighbors = set(graph.neighbors(u))
        for v in neighbors:
            for label in self.edge_labels:
                if label != graph.edges[u,v]['label']:
                    g = graph.copy()
                    g.edges[u, v]['label'] = label
                    graphs_.append(g)
        return graphs_


class NeighborhoodNodeLabelMutation(BaseNeighborhoodGenerator):

    def __init__(self, size=None):
        self.size = size
        
    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def fit(self, graphs, targets=None):
        self.node_labels = set([g.nodes[u]['label'] for g in graphs for u in g.nodes()])
        return self

    def graph_and_part_to_neighbors(self, graph, part):
        graphs_ = []
        u = part[0]
        for label in self.node_labels:
            if label != graph.nodes[u]['label']:
                g = graph.copy()
                g.nodes[u]['label'] = label
                graphs_.append(g)
        return graphs_


class NeighborhoodNodeAdd(BaseNeighborhoodGenerator):

    def __init__(self, size=None):
        self.size = size
        
    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def fit(self, graphs, targets=None):
        self.node_labels = set([g.nodes[u]['label'] for g in graphs for u in g.nodes()])
        self.edge_labels = set([g.edges[e]['label'] for g in graphs for e in g.edges()])
        return self

    def graph_and_part_to_neighbors(self, graph, part):
        graphs_ = []
        u = part[0]
        v = max(graph.nodes())+1
        for node_label in self.node_labels:
            for edge_label in self.edge_labels:
                g = graph.copy()
                g.add_node(v, label=node_label)
                g.add_edge(u,v, label=edge_label)
                graphs_.append(g)
        return graphs_


class NeighborhoodNodeOnEdgeAdd(BaseNeighborhoodGenerator):

    def __init__(self, size=None):
        self.size = size
        
    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def fit(self, graphs, targets=None):
        self.node_labels = set([g.nodes[u]['label'] for g in graphs for u in g.nodes()])
        self.edge_labels = set([g.edges[e]['label'] for g in graphs for e in g.edges()])
        return self

    def graph_and_part_to_neighbors(self, graph, part):
        graphs_ = []
        u = part[0]
        w = max(graph.nodes())+1
        for v in graph.neighbors(u):
            for node_label in self.node_labels:
                for edge_label in self.edge_labels:
                    g = graph.copy()
                    g.add_node(w, label=node_label)
                    g.add_edge(u,w, label=edge_label)
                    g.add_edge(w,v, label=g.edges[u,v]['label'])
                    g.remove_edge(u,v)
                    graphs_.append(g)
        return graphs_


class NeighborhoodEdgeMove(BaseNeighborhoodGenerator):

    def __init__(self, size=None, forbid_self_loop=True):
        self.size = size
        self.forbid_self_loop = forbid_self_loop
        
    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def fit(self, graphs, targets=None):
        return self

    def graph_and_part_to_neighbors(self, graph, part):
        graphs_ = []
        u = part[0]
        neighbors = set(graph.neighbors(u))
        nodes = set(graph.nodes())
        if self.forbid_self_loop: 
            nodes.discard(u)
        non_neighbors = nodes.difference(neighbors)
        for v in neighbors:
            for w in non_neighbors:
                g = graph.copy()
                g.add_edge(u,w, label=graph.edges[u,v]['label'])
                g.remove_edge(u,v)
                graphs_.append(g)
        return graphs_



class NeighborhoodDecomposition(BaseNeighborhoodGenerator):

    def __init__(self, decomposition_function, nbits=16, size=None, max_num_permutations=None):
        self.max_num_permutations = max_num_permutations
        self.size = size
        self.decomposition_function = decomposition_function
        self.nbits = nbits 
        
    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def cut_edges_to_part_signature(self, cut_edges):
        # a part signature is the hash of the edge labels in the cut
        part_signature = hash(tuple(sorted([edge['label'] for e, edge in cut_edges])))
        return part_signature
    
    def part_decomposition(self, graph):
        graphofgraph = self.decomposition_function(construct(graph, nbits=self.nbits))
        base_graph = graphofgraph.graph['base']
        parts = defaultdict(list)
        for u in graphofgraph.nodes():
            subgraph = graphofgraph.nodes[u]['subgraph']
            inner_nodes = set(subgraph.nodes())
            # identify edges in the cut connecting the subgraph to the remainder of the graph
            # note: each edge is a 2-tuple where the first node belongs to the subgraph, the second to the remainder of the graph
            cut_edges = [((v,z), base_graph.edges[(v,z)]) for v in inner_nodes for z in base_graph.neighbors(v) if z not in inner_nodes]
            # a part is a 2-tuple: the list of the edges in the cut and the subgraph 
            # a part signature is the hash of the edge labels in the cut
            part_signature = self.cut_edges_to_part_signature(cut_edges)
            parts[part_signature].append((cut_edges, subgraph.copy()))
        return parts
    
    def graph_to_parts(self, graph):
        graphofgraph = self.decomposition_function(construct(graph, nbits=self.nbits))
        parts = [set(graphofgraph.nodes[u]['subgraph'].nodes()) for u in graphofgraph.nodes()]
        return parts
    
    def fit(self, graphs, targets=None):
        self.fit_parts(graphs, targets)
        return self

    def fit_parts(self, graphs, targets=None):
        self.parts = defaultdict(list)
        for graph in graphs:
            parts = self.part_decomposition(graph)
            for key in parts:
                self.parts[key].extend(parts[key])
        # remove redundant parts i.e. parts that have the same subgraph
        for key in self.parts:
            already_present_set = set()
            reduced_part = []
            for part in self.parts[key]:
                cut_edges, subgraph = part
                subgraph_code = graph_hash(subgraph)
                if subgraph_code not in already_present_set:
                    already_present_set.add(subgraph_code)
                    reduced_part.append(part)
            self.parts[key] = reduced_part
        return self

    def join(self, complement_graph, cut_edges, candidate_cut_edges, candidate_subgraph):
        graphs = []
        # map nodes of candidate_subgraph in node space for complement_graph
        start_node_id = max(complement_graph.nodes())+1
        candidate_subgraph2complement_graph_map = {node_id:start_node_id+node_id for node_id in candidate_subgraph.nodes()}
        # copy complement_graph
        graph = complement_graph.copy()
        # join candidate_subgraph to complement_graph copy
        for u in candidate_subgraph.nodes():
            node = candidate_subgraph.nodes[u]
            new_node_id = candidate_subgraph2complement_graph_map[u]
            graph.add_node(new_node_id)
            for key in node:
                graph.nodes[new_node_id][key] = node[key]
        # add links from candidate_subgraph
        for e in candidate_subgraph.edges():
            edge = candidate_subgraph.edges[e]
            src,dst = e
            new_src = candidate_subgraph2complement_graph_map[src]
            new_dst = candidate_subgraph2complement_graph_map[dst]
            graph.add_edge(new_src, new_dst)
            for key in edge:
                graph.edges[new_src, new_dst][key] = edge[key]
        # link according to cut_edges
        random.shuffle(cut_edges)
        for counter, permuted_cut_edges in enumerate(permutations(cut_edges)):
            if self.max_num_permutations is not None and counter > self.max_num_permutations:
                break
            graph_ = graph.copy()
            is_valid = True
            for e, e_candidate in zip(permuted_cut_edges, candidate_cut_edges):
                (e_src, e_dst), attributes_dict = e
                (e_candidate_src, e_candidate_dst), candidate_attributes_dict = e_candidate
                if attributes_dict['label'] != candidate_attributes_dict['label']:
                    is_valid = False
                    break
                e_candidate_src = candidate_subgraph2complement_graph_map[e_candidate_src]
                graph_.add_edge(e_candidate_src, e_dst, **attributes_dict)
            if is_valid:
                if nx.number_of_nodes(graph_) > 0 and nx.number_of_edges(graph_) > 0:
                    graphs.append(graph_)
        return graphs

    def graph_and_part_to_neighbors(self, graph, part):
        graphs_ = []
        all_nodes = set(graph.nodes())
        inner_nodes = part
        complement_nodes = all_nodes.difference(inner_nodes)
        if len(complement_nodes)==0: return graphs_    
        complement_graph = nx.subgraph(graph, complement_nodes)
        cut_edges = [((v,z), graph.edges[(v,z)]) for v in inner_nodes for z in graph.neighbors(v) if z not in inner_nodes]
        part_signature = self.cut_edges_to_part_signature(cut_edges)
        candidates = self.parts[part_signature]
        for candidate in candidates:
            candidate_cut_edges, candidate_subgraph = candidate
            neigh_graphs = self.join(complement_graph, cut_edges, candidate_cut_edges, candidate_subgraph)
            graphs_.extend(neigh_graphs)
        return graphs_
    


class NeighborhoodBinaryDecomposition(BaseNeighborhoodGenerator):

    def __init__(self, decomposition_function_source, decomposition_function_destination, nbits=16, size=None, max_num_permutations=None):
        self.max_num_permutations = max_num_permutations
        self.size = size
        self.decomposition_function_source = decomposition_function_source
        self.decomposition_function_destination = decomposition_function_destination
        self.nbits = nbits 
        
    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def cut_edges_to_part_signature(self, cut_edges):
        # a part signature is the hash of the edge labels in the cut
        part_signature = hash(tuple(sorted([edge['label'] for e, edge in cut_edges])))
        return part_signature
    
    def part_decomposition(self, graph, decomposition_function):
        graphofgraph = decomposition_function(construct(graph, nbits=self.nbits))
        base_graph = graphofgraph.graph['base']
        parts = defaultdict(list)
        for u in graphofgraph.nodes():
            subgraph = graphofgraph.nodes[u]['subgraph']
            inner_nodes = set(subgraph.nodes())
            # identify edges in the cut connecting the subgraph to the remainder of the graph
            # note: each edge is a 2-tuple where the first node belongs to the subgraph, the second to the remainder of the graph
            cut_edges = [((v,z), base_graph.edges[(v,z)]) for v in inner_nodes for z in base_graph.neighbors(v) if z not in inner_nodes]
            # a part is a 2-tuple: the list of the edges in the cut and the subgraph 
            # a part signature is the hash of the edge labels in the cut
            part_signature = self.cut_edges_to_part_signature(cut_edges)
            parts[part_signature].append((cut_edges, subgraph.copy()))
        return parts
    
    def graph_to_parts(self, graph):
        graphofgraph = self.decomposition_function_source(construct(graph, nbits=self.nbits))
        parts = [set(graphofgraph.nodes[u]['subgraph'].nodes()) for u in graphofgraph.nodes()]
        return parts

    def fit(self, graphs, targets=None):
        self.parts = self.fit_parts(graphs, targets, decomposition_function=self.decomposition_function_source)
        self.parts_destination = self.fit_parts(graphs, targets, decomposition_function=self.decomposition_function_destination)
        return self

    def fit_parts(self, graphs, targets=None, decomposition_function=None):
        parts_dict = defaultdict(list)
        for graph in graphs:
            parts = self.part_decomposition(graph, decomposition_function)
            for key in parts:
                parts_dict[key].extend(parts[key])
        # remove redundant parts i.e. parts that have the same subgraph
        for key in parts_dict:
            already_present_set = set()
            reduced_part = []
            for part in parts_dict[key]:
                cut_edges, subgraph = part
                subgraph_code = graph_hash(subgraph)
                if subgraph_code not in already_present_set:
                    already_present_set.add(subgraph_code)
                    reduced_part.append(part)
            parts_dict[key] = reduced_part
        return parts_dict

    def join(self, complement_graph, cut_edges, candidate_cut_edges, candidate_subgraph):
        graphs = []
        # map nodes of candidate_subgraph in node space for complement_graph
        start_node_id = max(complement_graph.nodes())+1
        candidate_subgraph2complement_graph_map = {node_id:start_node_id+node_id for node_id in candidate_subgraph.nodes()}
        # copy complement_graph
        graph = complement_graph.copy()
        # join candidate_subgraph to complement_graph copy
        for u in candidate_subgraph.nodes():
            node = candidate_subgraph.nodes[u]
            new_node_id = candidate_subgraph2complement_graph_map[u]
            graph.add_node(new_node_id)
            for key in node:
                graph.nodes[new_node_id][key] = node[key]
        # add links from candidate_subgraph
        for e in candidate_subgraph.edges():
            edge = candidate_subgraph.edges[e]
            src,dst = e
            new_src = candidate_subgraph2complement_graph_map[src]
            new_dst = candidate_subgraph2complement_graph_map[dst]
            graph.add_edge(new_src, new_dst)
            for key in edge:
                graph.edges[new_src, new_dst][key] = edge[key]
        # link according to cut_edges
        random.shuffle(cut_edges)
        for counter, permuted_cut_edges in enumerate(permutations(cut_edges)):
            if self.max_num_permutations is not None and counter > self.max_num_permutations:
                break
            graph_ = graph.copy()
            is_valid = True
            for e, e_candidate in zip(permuted_cut_edges, candidate_cut_edges):
                (e_src, e_dst), attributes_dict = e
                (e_candidate_src, e_candidate_dst), candidate_attributes_dict = e_candidate
                if attributes_dict['label'] != candidate_attributes_dict['label']:
                    is_valid = False
                    break
                e_candidate_src = candidate_subgraph2complement_graph_map[e_candidate_src]
                graph_.add_edge(e_candidate_src, e_dst, **attributes_dict)
            if is_valid:
                if nx.number_of_nodes(graph_) > 0 and nx.number_of_edges(graph_) > 0:
                    graphs.append(graph_)
        return graphs


    def graph_and_part_to_neighbors(self, graph, part):
        graphs_ = []
        all_nodes = set(graph.nodes())
        inner_nodes = part
        complement_nodes = all_nodes.difference(inner_nodes)
        if len(complement_nodes)==0: return graphs_    
        complement_graph = nx.subgraph(graph, complement_nodes)
        cut_edges = [((v,z), graph.edges[(v,z)]) for v in inner_nodes for z in graph.neighbors(v) if z not in inner_nodes]
        part_signature = self.cut_edges_to_part_signature(cut_edges)
        candidates = self.parts_destination[part_signature]
        for candidate in candidates:
            candidate_cut_edges, candidate_subgraph = candidate
            neigh_graphs = self.join(complement_graph, cut_edges, candidate_cut_edges, candidate_subgraph)
            graphs_.extend(neigh_graphs)
        return graphs_
    

class NeighborhoodProbabilisticBinaryDecomposition(BaseNeighborhoodGenerator):

    def __init__(self, decomposition_function_source, decomposition_function_destination, decomposition_probs_source=None, decomposition_probs_destination=None, sampling_size_factor=.5, nbits=16, size=None, max_num_permutations=None):
        self.max_num_permutations = max_num_permutations
        self.size = size
        self.decomposition_function_source = decomposition_function_source
        self.decomposition_function_destination = decomposition_function_destination
        self.decomposition_probs_source = decomposition_probs_source
        self.decomposition_probs_destination = decomposition_probs_destination
        self.sampling_size_factor = sampling_size_factor
        self.nbits = nbits 
        
    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def cut_edges_to_part_signature(self, cut_edges):
        # a part signature is the hash of the edge labels in the cut
        part_signature = hash(tuple(sorted([edge['label'] for e, edge in cut_edges])))
        return part_signature
    
    def part_decomposition(self, graph, decomposition_function, decomposition_probs=None):
        graphofgraph = decomposition_function(construct(graph, nbits=self.nbits))
        base_graph = graphofgraph.graph['base']
        parts = defaultdict(list)
        node_list = list(graphofgraph.nodes()) 
        if decomposition_probs is not None:
            node_probs_list = decomposition_probs[node_list]
            if np.sum(node_probs_list) == 0: #no features are present in graph so return an empty parts
                return parts
            node_probs_list = node_probs_list / np.sum(node_probs_list)
            node_list = np.random.choice(node_list, size=int(len(node_list)*self.sampling_size_factor), p=node_probs_list)
        for u in node_list:
            subgraph = graphofgraph.nodes[u]['subgraph']
            inner_nodes = set(subgraph.nodes())
            # identify edges in the cut connecting the subgraph to the remainder of the graph
            # note: each edge is a 2-tuple where the first node belongs to the subgraph, the second to the remainder of the graph
            cut_edges = [((v,z), base_graph.edges[(v,z)]) for v in inner_nodes for z in base_graph.neighbors(v) if z not in inner_nodes]
            # a part is a 2-tuple: the list of the edges in the cut and the subgraph 
            # a part signature is the hash of the edge labels in the cut
            part_signature = self.cut_edges_to_part_signature(cut_edges)
            parts[part_signature].append((cut_edges, subgraph.copy()))
        return parts
    
    def graph_to_parts(self, graph):
        graphofgraph = self.decomposition_function_source(construct(graph, nbits=self.nbits))
        parts = [set(graphofgraph.nodes[u]['subgraph'].nodes()) for u in graphofgraph.nodes()]
        return parts

    def fit(self, graphs, targets=None):
        self.parts = self.fit_parts(graphs, targets, decomposition_function=self.decomposition_function_source, decomposition_probs=self.decomposition_probs_source)
        self.parts_destination = self.fit_parts(graphs, targets, decomposition_function=self.decomposition_function_destination, decomposition_probs=self.decomposition_probs_destination)
        return self

    def fit_parts(self, graphs, targets=None, decomposition_function=None, decomposition_probs=None):
        parts_dict = defaultdict(list)
        for graph in graphs:
            parts = self.part_decomposition(graph, decomposition_function, decomposition_probs)
            for key in parts:
                parts_dict[key].extend(parts[key])
        # remove redundant parts i.e. parts that have the same subgraph
        for key in parts_dict:
            already_present_set = set()
            reduced_part = []
            for part in parts_dict[key]:
                cut_edges, subgraph = part
                subgraph_code = graph_hash(subgraph)
                if subgraph_code not in already_present_set:
                    already_present_set.add(subgraph_code)
                    reduced_part.append(part)
            parts_dict[key] = reduced_part
        return parts_dict

    def join(self, complement_graph, cut_edges, candidate_cut_edges, candidate_subgraph):
        graphs = []
        # map nodes of candidate_subgraph in node space for complement_graph
        start_node_id = max(complement_graph.nodes())+1
        candidate_subgraph2complement_graph_map = {node_id:start_node_id+node_id for node_id in candidate_subgraph.nodes()}
        # copy complement_graph
        graph = complement_graph.copy()
        # join candidate_subgraph to complement_graph copy
        for u in candidate_subgraph.nodes():
            node = candidate_subgraph.nodes[u]
            new_node_id = candidate_subgraph2complement_graph_map[u]
            graph.add_node(new_node_id)
            for key in node:
                graph.nodes[new_node_id][key] = node[key]
        # add links from candidate_subgraph
        for e in candidate_subgraph.edges():
            edge = candidate_subgraph.edges[e]
            src,dst = e
            new_src = candidate_subgraph2complement_graph_map[src]
            new_dst = candidate_subgraph2complement_graph_map[dst]
            graph.add_edge(new_src, new_dst)
            for key in edge:
                graph.edges[new_src, new_dst][key] = edge[key]
        # link according to cut_edges
        random.shuffle(cut_edges)
        for counter, permuted_cut_edges in enumerate(permutations(cut_edges)):
            if self.max_num_permutations is not None and counter > self.max_num_permutations:
                break
            graph_ = graph.copy()
            is_valid = True
            for e, e_candidate in zip(permuted_cut_edges, candidate_cut_edges):
                (e_src, e_dst), attributes_dict = e
                (e_candidate_src, e_candidate_dst), candidate_attributes_dict = e_candidate
                if attributes_dict['label'] != candidate_attributes_dict['label']:
                    is_valid = False
                    break
                e_candidate_src = candidate_subgraph2complement_graph_map[e_candidate_src]
                graph_.add_edge(e_candidate_src, e_dst, **attributes_dict)
            if is_valid:
                if nx.number_of_nodes(graph_) > 0 and nx.number_of_edges(graph_) > 0:
                    graphs.append(graph_)
        return graphs

    def graph_and_part_to_neighbors(self, graph, part):
        graphs_ = []
        all_nodes = set(graph.nodes())
        inner_nodes = part
        complement_nodes = all_nodes.difference(inner_nodes)
        if len(complement_nodes)==0: return graphs_    
        complement_graph = nx.subgraph(graph, complement_nodes)
        cut_edges = [((v,z), graph.edges[(v,z)]) for v in inner_nodes for z in graph.neighbors(v) if z not in inner_nodes]
        part_signature = self.cut_edges_to_part_signature(cut_edges)
        candidates = self.parts_destination[part_signature]
        for candidate in candidates:
            candidate_cut_edges, candidate_subgraph = candidate
            neigh_graphs = self.join(complement_graph, cut_edges, candidate_cut_edges, candidate_subgraph)
            graphs_.extend(neigh_graphs)
        return graphs_

    
def distance_relation_function(graph1, graph2, basegraph, min_size, max_size):
    try:
        dist = min(nx.shortest_path_length(basegraph, source=u, target=v) for u in graph1.nodes() for v in graph2.nodes())
        if min_size <= dist <= max_size: relation_value = dist
        else: relation_value = np.nan
    except Exception:
        relation_value = np.nan
        pass
    return relation_value 


def intersection_relation_function(graph1, graph2, basegraph, min_size, max_size, scale=10):
    try:
        nodes1 = set(u for u in graph1.nodes())
        nodes2 = set(u for u in graph2.nodes())
        score = nodes1.intersection(nodes2)
        score = score / sp.stats.gmean([len(nodes1), len(nodes2)])
        score = int(score * scale)
        if min_size <= score <= max_size: relation_value = score
        else: relation_value = np.nan
    except Exception:
        relation_value = np.nan
        pass
    return relation_value 


class NeighborhoodContextualDecomposition(BaseNeighborhoodGenerator):

    def __init__(self, decomposition_function, context_decomposition_function, relation_function, min_size=0, max_size=1, nbits=16, size=None, max_num_permutations=None):
        self.max_num_permutations = max_num_permutations
        self.size = size
        self.decomposition_function = decomposition_function
        self.context_decomposition_function = context_decomposition_function
        self.relation_function = relation_function
        self.min_size = min_size
        self.max_size = max_size
        self.nbits = nbits 
        
    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def cut_edges_to_part_signature(self, cut_edges, context_hash):
        # a part signature is the hash of the edge labels in the cut...
        part_signature = hash(tuple(sorted([edge['label'] for e, edge in cut_edges])))
        #...jointly with the context graph hash
        part_signature = hash((part_signature, context_hash))
        return part_signature
    
    def part_decomposition(self, graph):
        graphofgraph = self.decomposition_function(construct(graph, nbits=self.nbits))
        context_graphofgraph = self.context_decomposition_function(construct(graph, nbits=self.nbits))
        base_graph = graphofgraph.graph['base']
        parts = defaultdict(list)
        for v in context_graphofgraph.nodes():
            context_subgraph = context_graphofgraph.nodes[v]['subgraph']
            context_hash = context_graphofgraph.nodes[v]['label']
            for u in graphofgraph.nodes():
                subgraph = graphofgraph.nodes[u]['subgraph']
                if np.isnan(self.relation_function(subgraph, context_subgraph, base_graph, min_size=self.min_size, max_size=self.max_size)) == False: 
                    inner_nodes = set(subgraph.nodes())
                    # identify edges in the cut connecting the subgraph to the remainder of the graph
                    # note: each edge is a 2-tuple where the first node belongs to the subgraph, the second to the remainder of the graph
                    cut_edges = [((v,z), base_graph.edges[(v,z)]) for v in inner_nodes for z in base_graph.neighbors(v) if z not in inner_nodes]
                    # a part is a 2-tuple: the list of the edges in the cut and the subgraph 
                    # a part signature is the hash of the edge labels in the cut
                    part_signature = self.cut_edges_to_part_signature(cut_edges, context_hash)
                    parts[part_signature].append((cut_edges, subgraph.copy()))
        return parts
    
    def graph_to_parts(self, graph):
        graphofgraph = self.decomposition_function(construct(graph, nbits=self.nbits))
        parts = [set(graphofgraph.nodes[u]['subgraph'].nodes()) for u in graphofgraph.nodes()]
        return parts
    
    def fit(self, graphs, targets=None):
        self.fit_parts(graphs, targets)
        return self

    def fit_parts(self, graphs, targets=None):
        self.parts = defaultdict(list)
        for graph in graphs:
            parts = self.part_decomposition(graph)
            for key in parts:
                self.parts[key].extend(parts[key])
        # remove redundant parts i.e. parts that have the same subgraph
        for key in self.parts:
            already_present_set = set()
            reduced_part = []
            for part in self.parts[key]:
                cut_edges, subgraph = part
                subgraph_code = graph_hash(subgraph)
                if subgraph_code not in already_present_set:
                    already_present_set.add(subgraph_code)
                    reduced_part.append(part)
            self.parts[key] = reduced_part
        return self

    def join(self, complement_graph, cut_edges, candidate_cut_edges, candidate_subgraph):
        graphs = []
        # map nodes of candidate_subgraph in node space for complement_graph
        start_node_id = max(complement_graph.nodes())+1
        candidate_subgraph2complement_graph_map = {node_id:start_node_id+node_id for node_id in candidate_subgraph.nodes()}
        # copy complement_graph
        graph = complement_graph.copy()
        # join candidate_subgraph to complement_graph copy
        for u in candidate_subgraph.nodes():
            node = candidate_subgraph.nodes[u]
            new_node_id = candidate_subgraph2complement_graph_map[u]
            graph.add_node(new_node_id)
            for key in node:
                graph.nodes[new_node_id][key] = node[key]
        # add links from candidate_subgraph
        for e in candidate_subgraph.edges():
            edge = candidate_subgraph.edges[e]
            src,dst = e
            new_src = candidate_subgraph2complement_graph_map[src]
            new_dst = candidate_subgraph2complement_graph_map[dst]
            graph.add_edge(new_src, new_dst)
            for key in edge:
                graph.edges[new_src, new_dst][key] = edge[key]
        # link according to cut_edges
        random.shuffle(cut_edges)
        for counter, permuted_cut_edges in enumerate(permutations(cut_edges)):
            if self.max_num_permutations is not None and counter > self.max_num_permutations:
                break
            graph_ = graph.copy()
            is_valid = True
            for e, e_candidate in zip(permuted_cut_edges, candidate_cut_edges):
                (e_src, e_dst), attributes_dict = e
                (e_candidate_src, e_candidate_dst), candidate_attributes_dict = e_candidate
                if attributes_dict['label'] != candidate_attributes_dict['label']:
                    is_valid = False
                    break
                e_candidate_src = candidate_subgraph2complement_graph_map[e_candidate_src]
                graph_.add_edge(e_candidate_src, e_dst, **attributes_dict)
            if is_valid:
                if nx.number_of_nodes(graph_) > 0 and nx.number_of_edges(graph_) > 0:
                    graphs.append(graph_)
        return graphs

    def graph_and_part_to_neighbors(self, graph, part):
        context_graphofgraph = self.context_decomposition_function(construct(graph, nbits=self.nbits))
        graphs_ = []
        all_nodes = set(graph.nodes())
        inner_nodes = part
        #recreate the subgraph
        subgraph = nx.subgraph(graph, inner_nodes)
        complement_nodes = all_nodes.difference(inner_nodes)
        if len(complement_nodes)==0: return graphs_    
        complement_graph = nx.subgraph(graph, complement_nodes)
        cut_edges = [((v,z), graph.edges[(v,z)]) for v in inner_nodes for z in graph.neighbors(v) if z not in inner_nodes]
        for v in context_graphofgraph.nodes():
            context_subgraph = context_graphofgraph.nodes[v]['subgraph']
            context_hash = context_graphofgraph.nodes[v]['label']
            if np.isnan(self.relation_function(subgraph, context_subgraph, graph, min_size=self.min_size, max_size=self.max_size)) == False: 
                part_signature = self.cut_edges_to_part_signature(cut_edges, context_hash)
                candidates = self.parts[part_signature]
                for candidate in candidates:
                    candidate_cut_edges, candidate_subgraph = candidate
                    neigh_graphs = self.join(complement_graph, cut_edges, candidate_cut_edges, candidate_subgraph)
                    graphs_.extend(neigh_graphs)
        return graphs_
    



class NeighborhoodContextualBinaryDecomposition(BaseNeighborhoodGenerator):

    def __init__(self, decomposition_function_source, decomposition_function_destination, context_decomposition_function, relation_function, min_size=0, max_size=1, nbits=16, size=None, max_num_permutations=None):
        self.max_num_permutations = max_num_permutations
        self.size = size
        self.decomposition_function_source = decomposition_function_source
        self.decomposition_function_destination = decomposition_function_destination
        self.context_decomposition_function = context_decomposition_function
        self.relation_function = relation_function
        self.min_size = min_size
        self.max_size = max_size
        self.nbits = nbits 
        
    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def cut_edges_to_part_signature(self, cut_edges, context_hash):
        # a part signature is the hash of the edge labels in the cut...
        part_signature = hash(tuple(sorted([edge['label'] for e, edge in cut_edges])))
        #...jointly with the context graph hash
        part_signature = hash((part_signature, context_hash))
        return part_signature

    def part_decomposition(self, graph, decomposition_function):
        graphofgraph = decomposition_function(construct(graph, nbits=self.nbits))
        context_graphofgraph = self.context_decomposition_function(construct(graph, nbits=self.nbits))
        base_graph = graphofgraph.graph['base']
        parts = defaultdict(list)
        for v in context_graphofgraph.nodes():
            context_subgraph = context_graphofgraph.nodes[v]['subgraph']
            context_hash = context_graphofgraph.nodes[v]['label']
            for u in graphofgraph.nodes():
                subgraph = graphofgraph.nodes[u]['subgraph']
                if np.isnan(self.relation_function(subgraph, context_subgraph, base_graph, min_size=self.min_size, max_size=self.max_size)) == False: 
                    inner_nodes = set(subgraph.nodes())
                    # identify edges in the cut connecting the subgraph to the remainder of the graph
                    # note: each edge is a 2-tuple where the first node belongs to the subgraph, the second to the remainder of the graph
                    cut_edges = [((v,z), base_graph.edges[(v,z)]) for v in inner_nodes for z in base_graph.neighbors(v) if z not in inner_nodes]
                    # a part is a 2-tuple: the list of the edges in the cut and the subgraph 
                    # a part signature is the hash of the edge labels in the cut
                    part_signature = self.cut_edges_to_part_signature(cut_edges, context_hash)
                    parts[part_signature].append((cut_edges, subgraph.copy()))
        return parts    

    def graph_to_parts(self, graph):
        graphofgraph = self.decomposition_function_source(construct(graph, nbits=self.nbits))
        parts = [set(graphofgraph.nodes[u]['subgraph'].nodes()) for u in graphofgraph.nodes()]
        return parts

    def fit(self, graphs, targets=None):
        self.parts = self.fit_parts(graphs, targets, decomposition_function=self.decomposition_function_source)
        self.parts_destination = self.fit_parts(graphs, targets, decomposition_function=self.decomposition_function_destination)
        return self

    def fit_parts(self, graphs, targets=None, decomposition_function=None):
        parts_dict = defaultdict(list)
        for graph in graphs:
            parts = self.part_decomposition(graph, decomposition_function)
            for key in parts:
                parts_dict[key].extend(parts[key])
        # remove redundant parts i.e. parts that have the same subgraph
        for key in parts_dict:
            already_present_set = set()
            reduced_part = []
            for part in parts_dict[key]:
                cut_edges, subgraph = part
                subgraph_code = graph_hash(subgraph)
                if subgraph_code not in already_present_set:
                    already_present_set.add(subgraph_code)
                    reduced_part.append(part)
            parts_dict[key] = reduced_part
        return parts_dict

    def join(self, complement_graph, cut_edges, candidate_cut_edges, candidate_subgraph):
        graphs = []
        # map nodes of candidate_subgraph in node space for complement_graph
        start_node_id = max(complement_graph.nodes())+1
        candidate_subgraph2complement_graph_map = {node_id:start_node_id+node_id for node_id in candidate_subgraph.nodes()}
        # copy complement_graph
        graph = complement_graph.copy()
        # join candidate_subgraph to complement_graph copy
        for u in candidate_subgraph.nodes():
            node = candidate_subgraph.nodes[u]
            new_node_id = candidate_subgraph2complement_graph_map[u]
            graph.add_node(new_node_id)
            for key in node:
                graph.nodes[new_node_id][key] = node[key]
        # add links from candidate_subgraph
        for e in candidate_subgraph.edges():
            edge = candidate_subgraph.edges[e]
            src,dst = e
            new_src = candidate_subgraph2complement_graph_map[src]
            new_dst = candidate_subgraph2complement_graph_map[dst]
            graph.add_edge(new_src, new_dst)
            for key in edge:
                graph.edges[new_src, new_dst][key] = edge[key]
        # link according to cut_edges
        random.shuffle(cut_edges)
        for counter, permuted_cut_edges in enumerate(permutations(cut_edges)):
            if self.max_num_permutations is not None and counter > self.max_num_permutations:
                break
            graph_ = graph.copy()
            is_valid = True
            for e, e_candidate in zip(permuted_cut_edges, candidate_cut_edges):
                (e_src, e_dst), attributes_dict = e
                (e_candidate_src, e_candidate_dst), candidate_attributes_dict = e_candidate
                if attributes_dict['label'] != candidate_attributes_dict['label']:
                    is_valid = False
                    break
                e_candidate_src = candidate_subgraph2complement_graph_map[e_candidate_src]
                graph_.add_edge(e_candidate_src, e_dst, **attributes_dict)
            if is_valid:
                if nx.number_of_nodes(graph_) > 0 and nx.number_of_edges(graph_) > 0:
                    graphs.append(graph_)
        return graphs


    def graph_and_part_to_neighbors(self, graph, part):
        context_graphofgraph = self.context_decomposition_function(construct(graph, nbits=self.nbits))
        graphs_ = []
        all_nodes = set(graph.nodes())
        inner_nodes = part
        #recreate the subgraph
        subgraph = nx.subgraph(graph, inner_nodes)
        complement_nodes = all_nodes.difference(inner_nodes)
        if len(complement_nodes)==0: return graphs_    
        complement_graph = nx.subgraph(graph, complement_nodes)
        cut_edges = [((v,z), graph.edges[(v,z)]) for v in inner_nodes for z in graph.neighbors(v) if z not in inner_nodes]
        for v in context_graphofgraph.nodes():
            context_subgraph = context_graphofgraph.nodes[v]['subgraph']
            context_hash = context_graphofgraph.nodes[v]['label']
            if np.isnan(self.relation_function(subgraph, context_subgraph, graph, min_size=self.min_size, max_size=self.max_size)) == False: 
                part_signature = self.cut_edges_to_part_signature(cut_edges, context_hash)
                candidates = self.parts_destination[part_signature]
                for candidate in candidates:
                    candidate_cut_edges, candidate_subgraph = candidate
                    neigh_graphs = self.join(complement_graph, cut_edges, candidate_cut_edges, candidate_subgraph)
                    graphs_.extend(neigh_graphs)
        return graphs_

#-------------------------------------------------------------------------------------------------------------------------------------
class GraphNeighborhoodGenerator(object):

    def __init__(self, generators, feasibility_estimator=None, duplicate_estimator=GraphDuplicateDetectionEstimator(), parallel=True, max_n_neighborhood_graphs=None):
        self.generators = generators
        self.feasibility_estimator = feasibility_estimator
        self.parallel = parallel
        self.max_n_neighborhood_graphs = max_n_neighborhood_graphs
        self.duplicate_estimator = duplicate_estimator
        if self.duplicate_estimator is not None: self.duplicate_estimator.parallel = parallel

    def __repr__(self):
        infos = ['generator_%d:%s'%(generator_id,generator) for generator_id, generator in enumerate(self.generators)]
        infos += ['%s:%s'%(key,value) for key,value in self.__dict__.items() if key != 'generators']
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def fit(self, graphs, targets=None):
        if self.parallel:
            def _make_generator_fitting_func(graphs, targets):
                def generator_fitting_func(generator_id):
                    return self.generators[generator_id].fit(graphs, targets)
                return generator_fitting_func
            n_cpus = mp.cpu_count()
            generator_fitting_func = _make_generator_fitting_func(graphs, targets)
            pool = mp.Pool(n_cpus)
            n_generators = len(self.generators)
            results = pool.map(generator_fitting_func, range(n_generators))
            pool.close()
            self.generators = results
        else:
            self.generators = [generator.fit(graphs, targets) for generator in self.generators]
        return self

    def annotate_generator_id_history(self, graphs, generator_id):
        for graph in graphs:
            if 'history' in graph.graph:
                graph.graph['history'].append(generator_id)
            else:
                graph.graph['history'] = [generator_id]
        return graphs

    def graphs_to_parts_generator_ids(self, graphs):
        graphs_parts_generator_ids = []
        for graph in graphs:
            for generator_id, generator in enumerate(self.generators):
                parts = generator.graph_to_parts(graph)
                graphs_parts_generator_ids_list = [(graph, part, generator_id)  for part in parts]
                graphs_parts_generator_ids.extend(graphs_parts_generator_ids_list)
        return graphs_parts_generator_ids

    def graphs_to_parts_generator_ids_neighbors(self, graphs):
        graphs_parts_generator_ids_neighbors = []
        for graph in graphs:
            for generator_id, generator in enumerate(self.generators):
                parts_and_neighbors = generator.graph_to_parts_and_neighbors(graph)
                graphs_parts_generator_ids_neighbors_list = [(graph, part, generator_id, neighbors_)  for part, neighbors_ in parts_and_neighbors]
                graphs_parts_generator_ids_neighbors.extend(graphs_parts_generator_ids_neighbors_list)
        return graphs_parts_generator_ids_neighbors

    def neighbors(self, graphs, generators_sequence=None):
        if self.parallel: out_graphs = self.neighbors_parallel(graphs, generators_sequence=generators_sequence)
        else: out_graphs = self.neighbors_sequential(graphs, generators_sequence=generators_sequence)
        out_graphs = self.feasibility_and_duplication_postprocessing(out_graphs)
        out_graphs = [nx.convert_node_labels_to_integers(graph) for graph in out_graphs]
        return out_graphs   

    def iterated_neighbors(self, graphs, generators_sequence=None, num_iterations=1):
        out_graphs = graphs
        for it in range(num_iterations):
            out_graphs = self.neighbors(out_graphs, generators_sequence=generators_sequence)
        return out_graphs

    def neighbors_parallel(self, graphs, generators_sequence):
        def _make_func(generators):
            def func(id_generator_and_graphs):
                synthetic_graphs = []
                generator_id, graphs = id_generator_and_graphs
                for graph in graphs:
                    generated_graphs = generators[generator_id].neighbors(graph)
                    generated_graphs = self.annotate_generator_id_history(generated_graphs, generator_id)
                    synthetic_graphs.extend(generated_graphs)
                return synthetic_graphs
            return func
        start = time.time()
        n_cpus = mp.cpu_count()
        batch_size = len(graphs) // n_cpus
        if len(graphs) <= n_cpus:
            graphs_list = [[graph] for graph in graphs]
        else:
            graphs_list = list(partition_all(batch_size, graphs))
        
        if generators_sequence is None: generators_sequence = [generator_id for generator_id in enumerate(self.generators)]
        #duplicate graphs to paralleize generator processing
        id_generator_and_graphs_list = [(generator_id, gs[:]) for generator_id in generators_sequence for gs in graphs_list]
        func = _make_func(self.generators)
        pool = mp.Pool(n_cpus)
        results = pool.map(func, id_generator_and_graphs_list)
        pool.close()
        all_results = []
        for list_of_results in results:
            all_results.extend(list_of_results)
        msg = 'from %d graphs generated %d perturbation graphs' % (len(graphs), len(all_results))
        msg = annotate_elapsed_time(msg, start, 2)
        logger.info(msg)
        return all_results

    def neighbors_sequential(self, graphs, generators_sequence):
        synthetic_graphs = []
        for graph in graphs:
            if generators_sequence is None: generator_id_and_generators_sequence = [(generator_id, generator) for generator_id, generator in enumerate(self.generators)]
            else: generator_id_and_generators_sequence = [(generator_id, self.generators[generator_id]) for generator_id in generators_sequence]
            for generator_id, generator in generator_id_and_generators_sequence:
                generated_graphs = generator.neighbors(graph)
                generated_graphs = self.annotate_generator_id_history(generated_graphs, generator_id)
                synthetic_graphs.extend(generated_graphs)
        return synthetic_graphs

    def feasibility_and_duplication_postprocessing(self, synthetic_graphs):
        if self.max_n_neighborhood_graphs is not None and len(synthetic_graphs) > self.max_n_neighborhood_graphs:
            n_in = len(synthetic_graphs)
            synthetic_graphs = random.sample(synthetic_graphs, k=min(self.max_n_neighborhood_graphs, len(synthetic_graphs)))
            n_out = len(synthetic_graphs)
            msg = '\tfrom %d graphs sampled %d graphs uniformly at random' % (n_in, n_out)
            logger.info(msg)
        if self.feasibility_estimator is not None:
            start = time.time()
            n_in = len(synthetic_graphs)
            if n_in > 0 : synthetic_graphs = self.feasibility_estimator.filter(synthetic_graphs)
            n_out = len(synthetic_graphs)
            msg = 'from %d graphs selected %d feasible graphs' % (n_in, n_out)
            msg = annotate_elapsed_time(msg, start, 2)
            logger.info(msg)
        if self.duplicate_estimator is not None: 
            start = time.time()
            n_in = len(synthetic_graphs)
            if n_in > 0 : synthetic_graphs = self.remove_duplicates(synthetic_graphs)
            n_out = len(synthetic_graphs)
            msg = 'from %d graphs selected %d unique graphs' % (n_in, n_out)
            msg = annotate_elapsed_time(msg, start, 2)
            logger.info(msg)
        return synthetic_graphs

    def remove_duplicates(self, graphs):
        return self.duplicate_estimator.fit_filter(graphs)

    

