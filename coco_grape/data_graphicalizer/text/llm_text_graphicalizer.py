#pip install openai
#pip install anthropic
from coco_grape.data_processor.selection.explain import select_most_discriminative_explanation, select_most_discriminative_explanation_from_dict
from coco_grape.data_processor.unsupervised.spectral_equal_size_clustering import EqualSizeSpectralClustering
from coco_grape.graph_vectorizer.graph_vectorizer import GraphVectorizer
from coco_grape.module import *
from coco_grape.visualizer.display import draw_graphs, graph_to_string
from coco_grape.data_graphicalizer.text.local_llm import LocalLLM

from collections import defaultdict
from io import StringIO
from scipy.optimize import linear_sum_assignment
import anthropic
import copy
import dill as pickle
import json
import networkx as nx
import numpy as np
import openai 
import random
import re
import textwrap
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering



def compute_cost_same_label(i,j,g1,g2):
    if g1.nodes[i]['label'] == g2.nodes[j]['label']: return 0
    else: return 1
    
def compute_cost_distance_vec(i,j,g1,g2):
    if g1.nodes[i]['label'] != g2.nodes[j]['label']: return 1e6
    v1 = np.array(g1.nodes[i]['vec'])
    v2 = np.array(g2.nodes[j]['vec'])
    d = np.linalg.norm(v1-v2)
    return d
    
def compute_mapping(g1,g2,threshold, compute_cost_func):
    n1 = nx.number_of_nodes(g1)
    n2 = nx.number_of_nodes(g2)
    cost = np.zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            cost[i,j] = compute_cost_func(i,j,g1,g2) 

    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = dict()
    for row_idx, (col_idx, col_cost) in enumerate(zip(col_ind, cost[row_ind, col_ind])):
        if col_cost <= threshold:
            mapping[row_idx] = col_idx
    return mapping

def unify(g1,g2,mapping):
    offset = nx.number_of_nodes(g1)
    gp2 = nx.relabel_nodes(g2, lambda x: x+offset)
    new_mapping = {key:value+offset for key,value in mapping.items()}
    g = nx.compose(g1,gp2)
    gn = nx.relabel_nodes(g, new_mapping, copy=False)
    return gn

def unify_graphs(graphs, compute_cost_func, threshold=0.5, print_mapping=False):
    if len(graphs)<2: return graphs[0]
    g1 = graphs[0]
    g2 = graphs[1]
    g1 = nx.convert_node_labels_to_integers(g1)
    g2 = nx.convert_node_labels_to_integers(g2)

    mapping = compute_mapping(g1,g2, threshold=threshold, compute_cost_func=compute_cost_func)
    if print_mapping: print(mapping)
    unified_g = unify(g1,g2,mapping)
    
    for g in graphs[2:]:
        unified_g = nx.convert_node_labels_to_integers(unified_g)
        g = nx.convert_node_labels_to_integers(g)
        mapping = compute_mapping(unified_g,g, threshold=threshold, compute_cost_func=compute_cost_func)
        if print_mapping: print(mapping)
        unified_g = unify(unified_g,g, mapping)
    return unified_g

def serialize_graph(G, ring_sep='-', node_label_key='label', edge_label_key='label'):
    """
    Serializes a NetworkX graph into a SMILES-like string.
    
    Parameters:
      G: A NetworkX graph.
      ring_sep: Separator for ring closures.
      node_label_key: Key to use when retrieving a node's label.
      edge_label_key: Key to use when retrieving an edge's label.
    """
    visited = set()   # nodes that have been visited
    ringed  = set()   # nodes referenced as a ring closure

    def chain_length(n, parent, unvisited_set):
        nb = [m for m in G.neighbors(n) if m != parent and m in unvisited_set]
        if len(nb) == 1:
            return 1 + chain_length(nb[0], n, unvisited_set)
        else:
            return 1

    def build_dfs_tree(node, parent):
        if node not in visited:
            visited.add(node)
        tree = {'node': node, 'main': None, 'branches': []}
        neighbors = [n for n in G.neighbors(node) if n != parent]
        local_unvisited = set(G.nodes()) - visited
        unvisited_neighbors = [n for n in neighbors if n not in visited]
        unvisited_neighbors.sort(key=lambda n: (chain_length(n, node, local_unvisited),
                                                G.nodes[n].get(node_label_key, str(n))))
        visited_neighbors = [n for n in neighbors if n in visited]
        visited_neighbors.sort(key=lambda n: G.nodes[n].get(node_label_key, str(n)))
        ordered_neighbors = unvisited_neighbors + visited_neighbors

        first = True
        for n in ordered_neighbors:
            edge_label = G[node][n].get(edge_label_key, '')
            if n not in visited:
                subtree = build_dfs_tree(n, node)
                if first:
                    tree['main'] = (edge_label, subtree)
                else:
                    tree['branches'].append((edge_label, subtree))
            else:
                ringed.add(n)
                ring_ref = ('ring', n)
                if first:
                    tree['main'] = (edge_label, ring_ref)
                else:
                    tree['branches'].append((edge_label, ring_ref))
            first = False
        return tree

    ring_index    = {}  # node -> assigned ring index
    label_counter = {}  # label -> next available ring index

    def get_ring_index(node):
        label = G.nodes[node].get(node_label_key, str(node))
        if node not in ring_index:
            idx = label_counter.get(label, 0)
            ring_index[node] = idx
            label_counter[label] = idx + 1
        return ring_index[node]

    def serialize_tree(tree):
        node  = tree['node']
        label = G.nodes[node].get(node_label_key, str(node))
        if node in ringed:
            idx = get_ring_index(node)
            s = f"[{label}{ring_sep}{idx}]"
        else:
            s = f"[{label}]"
        if tree['main'] is not None:
            edge_label, child = tree['main']
            s += f"-{edge_label}-"
            if isinstance(child, dict):
                s += serialize_tree(child)
            else:
                _, n = child
                n_label = G.nodes[n].get(node_label_key, str(n))
                idx = get_ring_index(n)
                s += f"[{n_label}{ring_sep}{idx}]"
        for edge_label, child in tree['branches']:
            s += f"(-{edge_label}-"
            if isinstance(child, dict):
                s += serialize_tree(child)
            else:
                _, n = child
                n_label = G.nodes[n].get(node_label_key, str(n))
                idx = get_ring_index(n)
                s += f"[{n_label}{ring_sep}{idx}]"
            s += ")"
        return s

    components = list(nx.connected_components(G))
    parts = []
    for comp in components:
        subgraph = G.subgraph(comp)
        if not subgraph.nodes:
            continue
        central_node = min(
            comp,
            key=lambda n: (sum(nx.single_source_shortest_path_length(subgraph, n).values()),
                           G.nodes[n].get(node_label_key, str(n)))
        )
        tree = build_dfs_tree(central_node, None)
        parts.append(serialize_tree(tree))
    return " . ".join(parts)


entity_definition_instructions_dict = {
   'Physical Entity': "This category encompasses all things that have a physical existence, whether natural or man-made. It is a subset of 'entity' and is divided into further subcategories.",
   'Abstraction': "Opposite to physical entities, this category includes all abstract concepts or ideas that do not have a physical existence. It covers a wide range of concepts like mathematical theories, ideas, relationships, etc.",
   'Object': "A subset of 'physical entities', this category includes all inanimate, tangible, and visible entities.",
   'Living Thing': "This category encompasses all forms of life, including plants, animals, and microorganisms.",
   'Group': "This category refers to collections or assemblages of entities, both physical and abstract.",
   'Phenomenon': "This includes naturally occurring events or observable occurrences, often outside human control.",
   'Event': "A broader category that includes anything that happens or takes place, not limited to natural phenomena.",
   'State': "Refers to conditions or situations of entities.",
   'Process': "A series of actions or steps taken in order to achieve a particular end.",
   'Quantity': "This category includes concepts related to numbers, amounts, and measurements.",
   'Attribute': "Characteristics or qualities of entities.",
   'Time': "Concepts related to time, including specific points in time, durations, etc.",
   'Location': "Refers to places, positions, and locales."
}


relation_definition_instructions_dict = {
    'Spatial Relations': "This includes relationships based on location or physical space, like 'above', 'below', 'beside', 'inside', 'outside', etc.",
    'Temporal Relations': "These are relationships based on time, such as 'before', 'after', 'during', 'simultaneous', etc.",
    'Part/Whole Relations': "This category, also known as meronymy, involves relationships where one entity is a part of another, like 'wheel' is a part of 'car' (meronym), or 'tree' is part of 'forest' (holonym).",
    'Cause and Effect': "Relationships where one entity causes or results from another, like 'smoking' can cause 'lung cancer'.",
    'Functional Relations': "This includes relationships based on function or role, such as 'key' and 'lock', where the function of one is related to the other.",
    'Membership Relations': "These are relationships where an entity is a member of a group or category, like an 'individual' is a member of a 'population'.",
    'Similarity/Contrast Relations': "Relationships based on similarities or differences between entities, including synonyms and antonyms.",
    'Ownership/Possession': "This involves relationships where one entity owns or possesses another, like 'owner' and 'property'.",
    'Attributive Relations': "Relationships where one entity is an attribute or characteristic of another, such as 'color' is an attribute of 'object'.",
    'Semantic Relations': "Broader conceptual relationships that cover how concepts are related in meaning or context, such as 'teacher' and 'education'.",
    'Social Relations': "These involve relationships defined by social constructs or roles, like 'parent' and 'child', or 'employer' and 'employee'.",
    'Symbolic Relations': "Relationships where one entity symbolizes or represents another, like 'flag' representing a 'country'."
}


class GraphAttributeClusteringGraphicalizer(object):
    def __init__(self, clustering, attribute_key='vec', use_cluster_id_as_label_suffix=True, llm=None, make_cluster_definitions=True):
        self.llm = llm
        self.attribute_key = attribute_key
        self.clustering = clustering
        self.use_cluster_id_as_label_suffix = use_cluster_id_as_label_suffix
        self.make_cluster_definitions = make_cluster_definitions

    def build_cluster_definitions(self, graphs):
        entity_descriptions = defaultdict(list)
        for graph in graphs:
            for node_idx in graph.nodes():
                key = graph.nodes[node_idx]['label']
                entity_descriptions[key].append(graph.nodes[node_idx]['description'])
        cluster_description = dict()
        for key in entity_descriptions:
            cluster_text = ' '.join(entity_descriptions[key])
            cluster_description[key] = cluster_text
        pre_instructions = "Task: Your objective is to generate a concise paragraph summarizing the provided text, focusing on extracting the central concept at an abstract level that is recurrent and widely applicable. Your summary should capture the essence of the text's main idea while maintaining brevity and clarity. Provide a title at the beginning. \n\nText:\n"
        self.cluster_definitions = {key : self.llm.ask(cluster_description[key], pre_instructions) for key in cluster_description}
        self.cluster_definitions_title = dict()
        self.cluster_definitions_body = dict()
        for key in self.cluster_definitions:
            definition = self.cluster_definitions[key]
            definition = definition.replace('Title:','')
            definition = definition.replace('#','')
            definition = definition.replace('*','')
            title = definition.split('\n')[0].lstrip()
            self.cluster_definitions_title[key] = title
            body = ''.join(definition.split('\n')[1:])
            self.cluster_definitions_body[key] = body

    def fit(self, graphs, targets=None):
        data_mtx = np.array([graph.nodes[u][self.attribute_key] for graph in graphs for u in graph.nodes()])
        self.clustering.fit(data_mtx)
        return self
    
    def transform_single(self, graph):
        try:
            out_graph = graph.copy()
            data_mtx = np.array([out_graph.nodes[u][self.attribute_key] for u in out_graph.nodes()])
            cluster_labels = self.clustering.predict(data_mtx)
            for u, cluster_label in zip(out_graph.nodes(), cluster_labels):
                if self.use_cluster_id_as_label_suffix: new_label = '%s_%d'%(out_graph.nodes[u]['label'], cluster_label)
                else: new_label = cluster_label
                out_graph.nodes[u]['original_label'] = out_graph.nodes[u]['label']
                out_graph.nodes[u]['label'] = new_label
                out_graph.nodes[u]['cluster'] = cluster_label
        except Exception as e:
            print(e)
            out_graph = graph.copy()
            for u in out_graph.nodes():
                out_graph.nodes[u]['cluster'] = -1
        return out_graph
        
    def annotate_cluster_definition(self, graphs):
        for graph in graphs:
            for u in graph.nodes():
                key = graph.nodes[u]['label']
                graph.nodes[u]['cluster_title'] = self.cluster_definitions_title[key]
        return graphs

    def transform(self, graphs):
        out_graphs = [self.transform_single(graph) for graph in graphs]
        if self.make_cluster_definitions: 
            self.build_cluster_definitions(out_graphs)
            out_graphs = self.annotate_cluster_definition(out_graphs)
        return out_graphs

    def fit_transform(self, graphs, targets=None):
        return self.fit(graphs, targets).transform(graphs)

def xstr(s):
    if s is None: return ''
    return str(s)


class AnthropicAILLM(object):
    def __init__(self, api_key=None, model_name='claude-3-5-sonnet-20240620', max_tokens=4096, temperature=0.0, system_instructions='Respond briefly.', pre_instructions=None, post_instructions=None, cache=None, use_cache=True, verbose=False):
        self.api_key = api_key
        self.pre_instructions = pre_instructions
        self.post_instructions = post_instructions
        self.model = model_name
        self.temperature = temperature
        self.system_instructions = system_instructions
        
        self.max_tokens_to_sample = max_tokens
        if cache is None: self.cache = dict()
        else: self.cache = cache
        self.use_cache = use_cache
        self.verbose = verbose
        
    def build_query(self, document, pre_instructions=None, post_instructions=None):
        instruction = xstr(pre_instructions) + document + xstr(post_instructions)
        return instruction

    def build_query_with_instructions(self, document, pre_instructions=None, post_instructions=None):
        if pre_instructions is None: pre_instructions = self.pre_instructions
        if post_instructions is None: post_instructions = self.post_instructions
        document_ = ' '.join(document.split())
        query_with_instructions = self.build_query(document_, pre_instructions, post_instructions)
        key = hash(query_with_instructions)
        return key, query_with_instructions

    def answer(self, txt):
        client = anthropic.Anthropic(api_key=self.api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens_to_sample,
            temperature=self.temperature,
            system=self.system_instructions,
            messages=[
                {"role": "user", "content": txt}
            ]
        )
        ans = message.content[0].text
        return ans

    def ask(self, document, pre_instructions=None, post_instructions=None):
        key, query_with_instructions = self.build_query_with_instructions(document, pre_instructions, post_instructions)
        if self.verbose: print(query_with_instructions)
        if key not in self.cache or self.use_cache is False: 
            ans = self.answer(query_with_instructions)
            self.cache[key] = ans
        answer = self.cache[key]
        if self.verbose: print(answer)
        return answer

    def forget(self, document, pre_instructions=None, post_instructions=None):
        key, query_with_instructions = self.build_query_with_instructions(document, pre_instructions, post_instructions)
        self.cache.pop(key,None)

    def save(self, filename='anthropic_model.obj'):
        filehandler = open(filename, 'wb') 
        pickle.dump(self, filehandler)

    def load(self, filename='anthropic_model.obj'):
        filehandler = open(filename, 'rb') 
        self = pickle.load(filehandler)
        return self

class LocalLLMTextVectorizer(object):
    def __init__(self, model_name='deepseek-r1:latest', verbose=False):
        self.llm = LocalLLM(model=model_name)
        self.verbose = verbose

    def fit(self, documents, targets=None):
        return self 
    
    def embed(self, text):
        return self.llm.transform_single(text)
    
    def transform_single(self, document):
        return self.llm.transform_single(document)
    
    def transform(self, documents):
        return self.llm.transform(documents)
    
    def fit_transform(self, documents, targets=None):
        return self.fit(documents, targets).transform(documents)

class LocalTextLLM(object):
    def __init__(self, model_name='deepseek-r1:latest', pre_instructions=None, post_instructions=None, verbose=False):
        self.llm = LocalLLM(model=model_name)
        self.pre_instructions = pre_instructions
        self.post_instructions = post_instructions
        self.verbose = verbose
     
    def answer(self, query_text):
        answer_txt = self.llm.answer(query_text)
        return answer_txt

    def build_query(self, document, pre_instructions=None, post_instructions=None):
        instruction = xstr(pre_instructions) + document + xstr(post_instructions)
        return instruction
    
    def build_query_with_instructions(self, document, pre_instructions=None, post_instructions=None):
        if pre_instructions is None: pre_instructions = self.pre_instructions
        if post_instructions is None: post_instructions = self.post_instructions
        document_ = ' '.join(document.split())
        query_with_instructions = self.build_query(document_, pre_instructions, post_instructions)
        return query_with_instructions

    def ask(self, document, pre_instructions=None, post_instructions=None):
        query_with_instructions = self.build_query_with_instructions(document, pre_instructions, post_instructions)
        if self.verbose: print(query_with_instructions)
        answer = self.answer(query_with_instructions)
        if self.verbose: print(answer)
        return answer
    

class OpenAILLMTextVectorizer(object):
    def __init__(self, model='text-embedding-3-small', api_key=None, cache_embeddings=None):
        self.api_key = api_key
        self.model = model
        if cache_embeddings is None: self.cache_embeddings = dict()
        else: self.cache_embeddings = cache_embeddings

    def fit(self, documents, targets=None):
        return self 

    def embed(self, text):
        text = text.replace("\n", " ")
        client = openai.OpenAI(api_key=self.api_key)
        return client.embeddings.create(input = [text], model=self.model).data[0].embedding

    def transform_single(self, document):
        key = hash(document)
        if key not in self.cache_embeddings: self.cache_embeddings[key] = self.embed(document)
        return self.cache_embeddings[key]

    def transform(self, documents):
        return np.array([self.transform_single(document) for document in documents])

    def fit_transform(self, documents, targets=None):
        return self.fit(documents, targets).transform(documents)
  
class OpenAILLM(object):
    def __init__(self, model_name='gpt-4o', api_key=None, temperature=0, max_tokens=4096, pre_instructions=None, post_instructions=None, cache=None, use_cache=True, verbose=False):
        self.model = model_name
        self.api_key = api_key
        self.pre_instructions = pre_instructions
        self.post_instructions = post_instructions
        if cache is None: self.cache = dict()
        else: self.cache = cache
        self.use_cache = use_cache
        self.verbose = verbose

    def answer(self, query_text):
        client = openai.OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
          model=self.model,
          #response_format={ "type": "json_object" },
          messages=[
            {"role": "system", "content": "You are a helpful assistant designed to answer in a concise way."},
            {"role": "user", "content": query_text}
          ]
        )
        answer_txt = response.choices[0].message.content
        return answer_txt

    def build_query(self, document, pre_instructions=None, post_instructions=None):
        instruction = xstr(pre_instructions) + document + xstr(post_instructions)
        return instruction
    
    def build_query_with_instructions(self, document, pre_instructions=None, post_instructions=None):
        if pre_instructions is None: pre_instructions = self.pre_instructions
        if post_instructions is None: post_instructions = self.post_instructions
        document_ = ' '.join(document.split())
        query_with_instructions = self.build_query(document_, pre_instructions, post_instructions)
        key = hash(query_with_instructions)
        return key, query_with_instructions

    def ask(self, document, pre_instructions=None, post_instructions=None):
        key, query_with_instructions = self.build_query_with_instructions(document, pre_instructions, post_instructions)
        if self.verbose: print(query_with_instructions)
        if key not in self.cache or self.use_cache is False: 
            self.cache[key] = self.answer(query_with_instructions)
        answer = self.cache[key]
        if self.verbose: print(answer)
        return answer

    def forget(self, document, pre_instructions=None, post_instructions=None):
        key, query_with_instructions = self.build_query_with_instructions(document, pre_instructions, post_instructions)
        self.cache.pop(key,None)
        
    def save(self, filename='openai_model.obj'):
        filehandler = open(filename, 'wb') 
        pickle.dump(self, filehandler)

    def load(self, filename='openai_model.obj'):
        filehandler = open(filename, 'rb') 
        self = pickle.load(filehandler)
        return self




class NodeTextVectorizerGraphicalizer(object):
    def __init__(self,
        text_vectorizer=None,
        node_attribute_key='vec',
        entity_description_key='description'):
        self.text_vectorizer = text_vectorizer
        self.node_attribute_key = node_attribute_key
        self.entity_description_key = entity_description_key

    def transform_single(self, graph):
        new_graph = nx.Graph(graph)
        for node_idx in graph.nodes():
            entity_description = graph.nodes[node_idx][self.entity_description_key]
            embedding = self.text_vectorizer.transform_single(entity_description)
            new_graph.nodes[node_idx][self.node_attribute_key] = embedding
        return new_graph

    def transform(self, graphs):
        out_graphs = [self.transform_single(graph) for graph in graphs]
        return out_graphs


class LLMTextGraphicalizer(object):
    def __init__(self, 
        llm=None, 
        entity_definition_instructions_dict=entity_definition_instructions_dict, 
        relation_definition_instructions_dict=relation_definition_instructions_dict, 
        verbose=False):
        self.verbose = verbose
        self.entity_definition_instructions_dict = entity_definition_instructions_dict
        self.relation_definition_instructions_dict = relation_definition_instructions_dict
        self.pre_instructions, self.post_instructions = self.build_pre_and_post_instructions(entity_definition_instructions_dict, relation_definition_instructions_dict)
        self.description_pre_instructions = 'Give general definition, in a single sentence, of the following entity: "'
        self.description_post_instructions = '", be concise and start the sentence with the name of the entity and continue with: is ...  Do not start the sentence with "Here is a one sentence general definition of".'
        self.contextual_description_pre_instructions = 'Describe, in a single sentence, of the following entity: "'
        self.contextual_description_post_instructions = '" in the context of the following document: '
        self.llm = llm

    def build_pre_and_post_instructions(self, entity_definition_instructions_dict, relation_definition_instructions_dict):
        entity_definition_instructions = '\n'.join(['%s: %s'%(key, entity_definition_instructions_dict[key]) for key in entity_definition_instructions_dict])
        entity_labels = list(entity_definition_instructions_dict.keys())
        entity_labels_example = ', '.join(random.choices(entity_labels, k=3))
        relation_definition_instructions = '\n'.join(['%s: %s'%(key, relation_definition_instructions_dict[key]) for key in relation_definition_instructions_dict])
        relation_labels = list(relation_definition_instructions_dict.keys())
        relation_labels_example = ', '.join(random.choices(relation_labels, k=3))
        controlled_vocabulary_instructions = 'The entity type can be one of the following:\n' + entity_definition_instructions + '\n' + 'The relation type can be one of the following:\n' + relation_definition_instructions
        pre_instructions = 'You are an expert ontologist, and possess expertise in the principles of ontology design, the development of classification systems, and the use of semantic technologies to facilitate data interoperability and knowledge representation. Consider the following text.'
        post_instructions_list = ['List and enumerate the main entities in the text.',
                                  'Extract the main links and relationships between the entities previously identified, each entity should be included in at least 3 relationships, and express the links between each pair of entities as, for example,  1 > 3 to indicate a link between entity 1 and entity 3.',
                                  'Identify the type of each entity and the type of each relation according to the following ontology:',
                                  controlled_vocabulary_instructions,
                                  'Create a digraph using the DOT language that includes all the entities from the text, enumerated using integers and labeled with the entity\'s textual name.',
                                  'Add the ontology type from the previous analysis in a new distinct attribute called CONTEXT, do not change the attribute "label".',
                                  'Label the entities using the ontology types for entities previously identified, such as  '+entity_labels_example+', etc.',
                                  'Label the edges between these entities using the ontology types for the relations previously identified, such as '+relation_labels_example+', etc.',
                                  'All entities need to be connected.',
                                  'Do not create subgraphs.']
        post_instructions = '\n'.join(post_instructions_list)
        return pre_instructions, post_instructions

    def fit(self, documents, targets=None):
        return self
    
    def text_extract_dot_digraph(self, out):
        reverse = out[::-1]
        assert re.search(r'\{', reverse) is not None, 'ERROR: something went wrong: there are no \{ ... \} in the text, hence no graph can be built. Text:\n%s'%out
        start = len(out)-re.search(r'\{', reverse).start()
        end = re.search(r'\}', out).start()
        res = 'digraph {'+out[start:end]+'}'
        res = res.replace('\nCONTEXT', ', CONTEXT')
        res = res.replace('\\lCONTEXT', ', CONTEXT')
        return res

    def dot_digraph_to_graph(self, digraph):
        #convert dot digraph to networkx graph
        graph = nx.Graph(nx.nx_pydot.read_dot(StringIO(digraph)))

        #reformat strings and validate expected fields
        out_graph = nx.subgraph(graph, [node_id for node_id in graph.nodes() if 'label' in graph.nodes[node_id]])
        for node_id in out_graph.nodes():
            #remove double slashes, remove quotes
            l = out_graph.nodes[node_id]['label']
            l = str(l.replace('\\\\', '\\'))
            l = l.replace('"', '')
            out_graph.nodes[node_id]['label'] = l
            #consider last word, i.e. most specialised category
            if 'CONTEXT' in out_graph.nodes[node_id]: 
                l = out_graph.nodes[node_id]['CONTEXT']
            elif 'context' in out_graph.nodes[node_id]: 
                l = out_graph.nodes[node_id]['context']
                out_graph.nodes[node_id].pop('context', None)
            else: 
                l = 'entity'
            l = l.replace('"', '')
            l = l.split()[-1] 
            l = l.lower()
            out_graph.nodes[node_id]['CONTEXT'] = l 
        for edge_id in out_graph.edges():
            #remove quotes
            l = out_graph.edges[edge_id]['label']
            l = l.replace('"', '')
            l = l.lower()
            #if l not in relation_labels: raise Exception('ERROR: the relation is not in the ontology.')
            out_graph.edges[edge_id]['label'] = l 
        return out_graph

    def generate_entity_descriptions(self, graph, document):
        entities = [graph.nodes[u]['label'] for u in graph.nodes()]
        entity_descriptions = []
        contextual_entity_descriptions = []
        for entity in entities:
            entity_description = self.llm.ask(entity, pre_instructions=self.description_pre_instructions, post_instructions=self.description_post_instructions)
            entity_descriptions.append(entity_description)
            contextual_entity_description = self.llm.ask(entity, pre_instructions=self.contextual_description_pre_instructions, post_instructions=self.contextual_description_post_instructions+document)
            contextual_entity_descriptions.append(contextual_entity_description)
            if self.verbose: 
                print(entity, ':', entity_description)
                print(entity, ':', contextual_entity_description)
        return entity_descriptions, contextual_entity_descriptions

    def integrate_entity_descriptions(self, graph, entity_descriptions, contextual_entity_descriptions):
        new_graph = nx.Graph(graph)
        for node_idx, entity_description, contextual_entity_description in zip(graph.nodes(), entity_descriptions, contextual_entity_descriptions):
            new_graph.nodes[node_idx]['label'] = graph.nodes[node_idx]['CONTEXT']
            new_graph.nodes[node_idx]['entity'] = graph.nodes[node_idx]['label']
            new_graph.nodes[node_idx]['description'] = entity_description.strip()
            new_graph.nodes[node_idx]['contextual_description'] = contextual_entity_description.strip()
            new_graph.nodes[node_idx].pop('CONTEXT', None)
        return new_graph

    def _transform_single(self, document):
        output = self.llm.ask(document, pre_instructions=self.pre_instructions, post_instructions=self.post_instructions)
        if self.verbose: print(output)
        text_dot_digraph = self.text_extract_dot_digraph(output)
        graph = self.dot_digraph_to_graph(text_dot_digraph)
        if self.verbose: print('graph with %d nodes and %d edges'%(nx.number_of_nodes(graph), nx.number_of_edges(graph)))
        entity_descriptions, contextual_entity_descriptions = self.generate_entity_descriptions(graph, document)
        graph = self.integrate_entity_descriptions(graph, entity_descriptions, contextual_entity_descriptions)
        graph.graph['document'] = document
        graph.graph['output'] = output
        return graph

    def transform(self, documents):
        graphs = []
        for document in documents:
            #when a valid graph cannot be created output a syntactically valid but non-informative graph 
            graph = nx.Graph()
            graph.add_node(0,label='N/A', entity='N/A', description='N/A', contextual_description='N/A')
            try:
                graph = self._transform_single(document)
            except Exception as e:
                #something went wrong: forget try again
                print(e)
                #self.llm.forget(document, pre_instructions=self.pre_instructions, post_instructions=self.post_instructions)
            graphs.append(graph)
            if self.verbose: print('%d/%d documents processed'%(len(graphs), len(documents)))
        return graphs

    def fit_transform(self, documents, targets=None):
        return self.fit(documents, targets).transform(documents)

    def save(self, filename='model.obj'):
        filehandler = open(filename, 'wb') 
        pickle.dump(self, filehandler)

    def load(self, filename='model.obj'):
        filehandler = open(filename, 'rb') 
        self = pickle.load(filehandler)
        return self

    def display(self, graphs, show_graph=True, show_cluster_title=True, size=20):
        if show_graph: 
            if show_cluster_title: draw_graphs(graphs, node_color=None, label='label', secondary_labels=['cluster_title','entity'], edge_label='label', size=size, node_marker_edge_color='w')
            else: draw_graphs(graphs, node_color=None, label='label', secondary_labels=['entity'], edge_label='label', size=size, node_marker_edge_color='w')

        for graph in graphs:
            for u in graph.nodes():
                str_list  = [graph.nodes[u]['entity'], graph.nodes[u]['label'], graph.nodes[u]['description'], graph.nodes[u]['contextual_description']]
                print(' | '.join(str_list))
                print()
            for e in graph.edges():
                str_list = [graph.nodes[e[0]]['entity']+' ('+graph.nodes[e[0]]['label']+')', graph.edges[e]['label'], graph.nodes[e[1]]['entity']+' ('+graph.nodes[e[1]]['label']+')']
                print(' - '.join(str_list))
            print()
            print(serialize_graph(graph))
            print()
            print(graph.graph['document'])
            print('_'*100)


class LLMTextTransformer(object):
    def __init__(self, text_vectorizer=None, text_graphicalizer=None, node_vectorizer=None, node_clustering=None, graph_vectorizer=None):
        self.text_vectorizer = text_vectorizer
        self.text_graphicalizer = text_graphicalizer
        self.node_vectorizer = node_vectorizer
        self.node_clustering = node_clustering
        self.graph_vectorizer = graph_vectorizer
        
    def fit_transform(self, documents):
        graphs = self.text_graphicalizer.fit_transform(documents)
        node_embedding_graphs = self.node_vectorizer.transform(graphs)
        node_clustered_graphs = self.node_clustering.fit_transform(node_embedding_graphs)
        return node_clustered_graphs
    
    def transform(self, documents):
        graphs = self.text_graphicalizer.transform(documents)
        node_embedding_graphs = self.node_vectorizer.transform(graphs)
        node_clustered_graphs = self.node_clustering.transform(node_embedding_graphs)
        return node_clustered_graphs

    def transform_graph_groups(self, graphs_list, compute_cost_func, threshold, print_mapping=False):
        out_graphs = []
        for graphs in graphs_list:
            graph = unify_graphs(graphs, compute_cost_func=compute_cost_func, threshold=threshold, print_mapping=print_mapping)
            out_graphs.append(graph)
        return out_graphs

    def transform_document_groups(self, documents_list, compute_cost_func, threshold, print_mapping=False):
        out_graphs = []
        for documents in documents_list:
            graphs = self.transform(documents)
            graph = unify_graphs(graphs, compute_cost_func=compute_cost_func, threshold=threshold, print_mapping=print_mapping)
            out_graphs.append(graph)
        return out_graphs
    
    def embed(self, documents):
        node_clustered_graphs = self.transform(documents)
        return self.graph_embed(node_clustered_graphs)
        
    def graph_embed(self, graphs):
        return self.graph_vectorizer.fit_transform(graphs)
        
    def display(self, graphs):
        self.text_graphicalizer.display(graphs)
        
    def save(self, filename='model.obj'):
        filehandler = open(filename, 'wb') 
        pickle.dump(self, filehandler)

    def load(self, filename='model.obj'):
        filehandler = open(filename, 'rb') 
        self = pickle.load(filehandler)
        return self

    def save_graphs(self, graphs, filename='data.obj'):    
        filehandler = open(filename, 'wb') 
        pickle.dump(graphs, filehandler)
        
    def load_graphs(self, filename='data.obj'):    
        filehandler = open(filename, 'rb') 
        graphs = pickle.load(filehandler)
        return graphs

    def plot_entities_embeddings(self, node_clustered_graphs):
        node_clustering = self.node_clustering
        gs = node_clustered_graphs
        nodes_embeddings = np.vstack([g.nodes[u]['vec'] for g in gs for u in g.nodes()])
        nodes_clusters = [g.nodes[u]['label'] for g in gs for u in g.nodes()]
        lenc = LabelEncoder()
        nodes_clusters_idxs = lenc.fit_transform(nodes_clusters)
        cluster_names = lenc.classes_
        n_clusters = len(cluster_names)

        X2d = TSNE(n_components=2, perplexity=30.0).fit_transform(nodes_embeddings)
        cmap = mpl.colormaps['jet']
        colors = cmap(np.linspace(0, 1, n_clusters))
        size = 14
        plt.figure(figsize=(size,size))
        ax = plt.subplot(111)

        for cluster_idx in range(n_clusters):
            X_cluster_idx = X2d[nodes_clusters_idxs==cluster_idx]
            label = '%2d %13s:  %s'%(cluster_idx, cluster_names[cluster_idx], node_clustering.cluster_definitions_title[cluster_names[cluster_idx]])
            plt.scatter(*X_cluster_idx.T,label=label, s=120, color=colors[cluster_idx])
            for x,y in X_cluster_idx:
                plt.text(x,y,cluster_idx, horizontalalignment='center', verticalalignment='center', color='w')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'family':'monospace'})
        plt.show()

    def print_cluster_definitions(self):    
        for key in sorted(self.node_clustering.cluster_definitions):
            title = self.node_clustering.cluster_definitions_title[key]
            body = self.node_clustering.cluster_definitions_body[key]
            print('%s: %s\n%s\n'%(key, title, body))
    
    def draw_dendrogram(self, cumulative_graphs):
        X = self.graph_embed(cumulative_graphs).todense()
        # setting distance_threshold=0 ensures we compute the full tree.
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
        model = model.fit(X)
        plt.title("Hierarchical Clustering Dendrogram")
        # Create linkage matrix and then plot the dendrogram
        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, truncate_mode="level", p=10)

        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.show()


def ConcreteOpenAILLMTextTransformer(api_key, vectorizer_api_key, entity_definition_instructions_dict, relation_definition_instructions_dict, model_name='gpt-4o', temperature=0, max_tokens=4096, n_clusters=2, verbose=False):
    llm = OpenAILLM(
        model_name=model_name, 
        api_key=api_key, 
        temperature=temperature, 
        max_tokens=max_tokens, 
        pre_instructions=None, 
        post_instructions=None, 
        cache=None, 
        use_cache=True, 
        verbose=verbose)
    text_vectorizer = OpenAILLMTextVectorizer(api_key=vectorizer_api_key)
    text_graphicalizer = LLMTextGraphicalizer(
        llm=llm, 
        entity_definition_instructions_dict=entity_definition_instructions_dict, 
        relation_definition_instructions_dict=relation_definition_instructions_dict, 
        verbose=verbose)
    node_vectorizer = NodeTextVectorizerGraphicalizer(text_vectorizer=text_vectorizer)
    clustering = EqualSizeSpectralClustering(n_clusters=n_clusters)
    node_clustering = GraphAttributeClusteringGraphicalizer(clustering=clustering, llm=llm, make_cluster_definitions=True)
    graph_vectorizer = GraphVectorizer(
        decomposition_function=graphlet(size=3, radius=1),
        nbits=10,
        feature_type='feature', #feature, node, node_list
        use_attributes=True)
    estimator = LLMTextTransformer(text_vectorizer=text_vectorizer, text_graphicalizer=text_graphicalizer, node_vectorizer=node_vectorizer, node_clustering=node_clustering, graph_vectorizer=graph_vectorizer)
    return estimator


def ConcreteAnthropicLLMTextTransformer(api_key, vectorizer_api_key, entity_definition_instructions_dict, relation_definition_instructions_dict, model_name='claude-3-5-sonnet-20240620', temperature=0, max_tokens=4096, n_clusters=2, verbose=False):
    llm = AnthropicAILLM(
        model_name=model_name, 
        api_key=api_key, 
        temperature=temperature, 
        max_tokens=max_tokens, 
        pre_instructions=None, 
        post_instructions=None, 
        cache=None, 
        use_cache=True,
        verbose=verbose)
    text_vectorizer = OpenAILLMTextVectorizer(api_key=vectorizer_api_key)
    text_graphicalizer = LLMTextGraphicalizer(
        llm=llm, 
        entity_definition_instructions_dict=entity_definition_instructions_dict, 
        relation_definition_instructions_dict=relation_definition_instructions_dict, 
        verbose=verbose)
    node_vectorizer = NodeTextVectorizerGraphicalizer(text_vectorizer=text_vectorizer)
    clustering = EqualSizeSpectralClustering(n_clusters=n_clusters)
    node_clustering = GraphAttributeClusteringGraphicalizer(clustering=clustering, llm=llm, make_cluster_definitions=True)
    graph_vectorizer = GraphVectorizer(
        decomposition_function=graphlet(size=3, radius=1),
        nbits=10,
        feature_type='feature', #feature, node, node_list
        use_attributes=True)
    estimator = LLMTextTransformer(text_vectorizer=text_vectorizer, text_graphicalizer=text_graphicalizer, node_vectorizer=node_vectorizer, node_clustering=node_clustering, graph_vectorizer=graph_vectorizer)
    return estimator

def ConcreteLocalLLMTextTransformer(model_name=None, entity_definition_instructions_dict=None, relation_definition_instructions_dict=None, n_clusters=2, verbose=False):
    llm = LocalTextLLM(model_name=model_name, verbose=verbose)
    text_vectorizer = LocalLLMTextVectorizer(model_name=model_name, verbose=verbose)
    text_graphicalizer = LLMTextGraphicalizer(
        llm=llm, 
        entity_definition_instructions_dict=entity_definition_instructions_dict, 
        relation_definition_instructions_dict=relation_definition_instructions_dict, 
        verbose=False)
    node_vectorizer = NodeTextVectorizerGraphicalizer(text_vectorizer=text_vectorizer)
    clustering = EqualSizeSpectralClustering(n_clusters=n_clusters)
    node_clustering = GraphAttributeClusteringGraphicalizer(clustering=clustering, llm=llm, make_cluster_definitions=True)
    graph_vectorizer = GraphVectorizer(
        decomposition_function=graphlet(size=3, radius=1),
        nbits=10,
        feature_type='feature', #feature, node, node_list
        use_attributes=True)
    estimator = LLMTextTransformer(text_vectorizer=text_vectorizer, text_graphicalizer=text_graphicalizer, node_vectorizer=node_vectorizer, node_clustering=node_clustering, graph_vectorizer=graph_vectorizer)
    return estimator


#---------------------------------

def select_matching_subgraph(hash_id, graph, decomposition_function, nbits):
    graphofsubgraphs = decomposition([graph], decomposition_function, nbits)[0]
    for u in graphofsubgraphs.nodes():
        if graphofsubgraphs.nodes[u]['label'] == hash_id:
            return graphofsubgraphs.nodes[u]['subgraph']
    return None

def select_motif_occurrence(cumulative_graphs, sorted_graphs, discriminative_concepts, sel_idx, explanation_decomposition_function, nbits, verbose=True):
    hash_id = sorted_graphs[sel_idx].graph['hash_id']
    discriminative_concept = discriminative_concepts[sel_idx]
    if verbose:
        print('concept_idx: %d   hash_id: %s'%(sel_idx, hash_id))
        print('discriminative_concept: %s'%discriminative_concept)
    motif_list = []
    counter = 0
    for i, graph in enumerate(cumulative_graphs):
        motif = select_matching_subgraph(hash_id, graph=graph, decomposition_function=explanation_decomposition_function, nbits=nbits)
        if motif is not None:
            motif_list.append(motif)
            if verbose:
                print('instance_idx:%d   graph_id:%d'%(counter, i))
                draw_graphs(motif, size=10, n_text_cols=30, n_graphs_per_line=1, label='label', secondary_labels=['entity','cluster_title','description'], edge_label='label', node_color=None, node_marker_edge_color=None, node_size=300)
        counter += 1
    return motif_list


def convert_graph_to_dot_extended(g):
    gg= nx.Graph()
    for u in g.nodes():
        label = g.nodes[u]['entity'] + '. '+g.nodes[u]['cluster_title'] + '. '+g.nodes[u]['description'] + '. '+g.nodes[u]['contextual_description']+ '. '
        gg.add_node(u,label=label)
    for u,v in g.edges():
        gg.add_edge(u,v,label=g.edges[u,v]['label'])
    graph_str = str(nx.nx_pydot.to_pydot(gg))
    return graph_str

def motif_to_text_extended(motif_graph, llm):
    g_str = convert_graph_to_dot_extended(motif_graph)
    prompt = "Given a concept described as a graph in DOT format with entities in nodes and relations in edges, write a paragraph in the style of a dictionary entry to describe the concept encoded in the graph. Here's the graph: "+ g_str 
    answer = llm.ask(prompt)
    prompt ='You are a microbiologist university professor. Rewrite the following paragraph as an explanation to a graduate student in biology (do not mention papers, publications or nodes): '+ answer
    answer = llm.ask(prompt)
    return answer

def compute_refined_discriminative_concept(selected_motives, discriminative_concept, llm):
    text_motives = [motif_to_text_extended(selected_motive, llm) for selected_motive in selected_motives]
    text_motives = ' '.join(text_motives)
    text_motives = text_motives.strip()
    txt = discriminative_concept +'\n'+text_motives
    prompt ='You are a microbiologist university professor. DO not start your answer with "As a microbiologist, I\'d explain it this way:". Make a single sentence summary of the following text from the tag <START> until the tag <END>: <START>'+ txt +'<END>'
    refined_discriminative_concept = llm.ask(prompt)
    prompt ='Rewrite the following text in the style of a dictionary entry: '+ refined_discriminative_concept
    refined_discriminative_concept = llm.ask(prompt)
    return refined_discriminative_concept

def compute_refined_discriminative_concepts(sorted_graphs, cumulative_graphs, discriminative_concepts, explanation_decomposition_function, nbits, text_transformer, verbose=True):
    llm = text_transformer.text_graphicalizer.llm
    n_range = len(discriminative_concepts)
    selected_motives_list = []
    for sel_idx in range(n_range):
        selected_motives = select_motif_occurrence(
            cumulative_graphs, 
            sorted_graphs, 
            discriminative_concepts,
            sel_idx=sel_idx, 
            explanation_decomposition_function=explanation_decomposition_function,
            nbits=nbits, 
            verbose=verbose)
        discriminative_concept = discriminative_concepts[sel_idx]
        refined_discriminative_concept = compute_refined_discriminative_concept(selected_motives, discriminative_concept, llm)
        if verbose:
            print('\nRefined_discriminative_concept:\n%s\n\n'%refined_discriminative_concept)
        selected_motives_list.append((refined_discriminative_concept, selected_motives))
    return selected_motives_list

def get_cluster_definitions(text_transformer):    
    txt = ''
    for key in sorted(text_transformer.node_clustering.cluster_definitions):
        title = text_transformer.node_clustering.cluster_definitions_title[key]
        body = text_transformer.node_clustering.cluster_definitions_body[key]
        txt +='%s: %s\n%s\n'%(key, title, body)
    return txt

def convert_graph_to_dot(g):
    gg= nx.Graph()
    for u in g.nodes():
        label = g.nodes[u]['cluster_title']
        gg.add_node(u,label=label)
    for u,v in g.edges():
        gg.add_edge(u,v,label=g.edges[u,v]['label'])
    graph_str = str(nx.nx_pydot.to_pydot(gg))
    return graph_str

def motif_to_text(motif_graph, relation_definitions, cluster_definitions, llm):
    g_str = convert_graph_to_dot(motif_graph)
    prompt = "Given a concept described as a graph in DOT format with entities in nodes and relations in edges, write a sentence in the style of a dictionary entry to describe the concept encoded in the graph. Use the provided relation definitions: "+relation_definitions+"\n and the entities definition: "+cluster_definitions+".\n Here's the graph: "+ g_str 
    answer = llm.ask(prompt)
    prompt ='You are a microbiologist university professor. Rewrite the following paragraph as an explanation to a graduate student in biology (do not mention papers, publications or nodes): '+ answer
    answer = llm.ask(prompt)
    return answer

class SubGraphExplainer(object):
    def __init__(self, text_transformer, base_decomposition_function, explanation_decomposition_function, nbits=12, n_selected_features=.33, num_train_repetitions=5, parallel=False):
        self.text_transformer = text_transformer
        self.relation_definition_instructions_dict = text_transformer.text_graphicalizer.relation_definition_instructions_dict
        self.base_decomposition_function = base_decomposition_function
        self.explanation_decomposition_function = explanation_decomposition_function
        self.nbits = nbits
        self.n_selected_features = n_selected_features
        self.num_train_repetitions = num_train_repetitions
        self.parallel = parallel
        self.data_dict = None
        
    def fit(self, graphs, targets):
        self.data_dict = select_most_discriminative_explanation(
            graphs= graphs, 
            targets=targets, 
            base_decomposition_function=self.base_decomposition_function,
            explanation_decomposition_function=self.explanation_decomposition_function,
            nbits=self.nbits, 
            n_selected_features=self.n_selected_features,
            num_train_repetitions=self.num_train_repetitions,
            parallel=self.parallel)
        return self
    
    def explain(self, min_counts=1, min_pi=1e-1, min_freq=0.0, class_of_interest=1, min_size=2):
        sorted_graphs, sorted_titles = select_most_discriminative_explanation_from_dict(
            self.data_dict, 
            min_counts=min_counts,
            min_pi=min_pi, 
            min_freq=min_freq,
            class_of_interest=class_of_interest, 
            min_size=min_size)
        return sorted_graphs, sorted_titles
    
    def compute_explanation(self, selected_sorted_graphs, selected_sorted_titles):
        cluster_definitions = get_cluster_definitions(self.text_transformer)
        relation_definitions = ' '.join(['%s: %s\n'%(key, self.relation_definition_instructions_dict[key]) for key in self.relation_definition_instructions_dict])
        llm = self.text_transformer.text_graphicalizer.llm
        discriminative_concepts = [motif_to_text(selected_sorted_graph, relation_definitions, cluster_definitions, llm) for selected_sorted_graph in selected_sorted_graphs]
        wrapped_discriminative_concepts = [textwrap.fill(discriminative_concept, 80) for discriminative_concept in discriminative_concepts]
        wrapped_discriminative_concepts = ['%s\n%s\n%s'%(g.graph.get('hash_id','N/A'), c,t) for c,t,g in zip(wrapped_discriminative_concepts, selected_sorted_titles, selected_sorted_graphs)]
        return discriminative_concepts, wrapped_discriminative_concepts
    
    def explain_and_draw(self, max_num_concepts=9, label='cluster_title', min_counts=1, min_pi=1e-1, min_freq=0.0, class_of_interest=1, min_size=2, draw=True, n_graphs_per_line=3):
        sorted_graphs, sorted_titles = self.explain(min_counts, min_pi, min_freq, class_of_interest, min_size)
        selected_sorted_graphs = sorted_graphs[:max_num_concepts]
        selected_sorted_titles = sorted_titles[:max_num_concepts]
        discriminative_concepts, wrapped_discriminative_concepts = self.compute_explanation(selected_sorted_graphs, selected_sorted_titles)
        if draw: draw_graphs(selected_sorted_graphs, titles=wrapped_discriminative_concepts, size=10, n_graphs_per_line=n_graphs_per_line, label=label, secondary_labels=['label'], edge_label='label', node_color=None, node_marker_edge_color=None, node_size=300)
        return sorted_graphs, sorted_titles, discriminative_concepts
    
    def compute_refined_discriminative_concepts_and_draw(self, cumulative_graphs, max_num_concepts=9, label='cluster_title', min_counts=1, min_pi=1e-1, min_freq=0.0, class_of_interest=1, min_size=2, draw=True):
        concept_graphs, concept_titles, discriminative_concepts = self.explain_and_draw(max_num_concepts, label, min_counts, min_pi, min_freq, class_of_interest, min_size, False)
        refined_discriminative_concepts_list = compute_refined_discriminative_concepts(concept_graphs, cumulative_graphs, discriminative_concepts, self.explanation_decomposition_function, self.nbits, self.text_transformer, verbose=draw)
        return refined_discriminative_concepts_list
