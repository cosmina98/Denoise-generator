import numpy as np
import networkx as nx
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from functools import reduce
from collections import defaultdict
from scipy.stats import chisquare
from coco_grape.module.vectorize import parallel_vectorize
from coco_grape.module.construct import decomposition
from coco_grape.module.composition import compose, add, binary_combine
from coco_grape.module.graph_hash import graph_hash, hash_value, nodes_hash
from coco_grape.module.decompositions.connected_component import connected_component
from coco_grape.module.decompositions.merge import merge
from coco_grape.module.decompositions.filter_by import filter_by_feature_importance
from coco_grape.module.decompositions.binary_set_operations import binary_intersection


def select_most_discriminative_explanation(graphs, targets, base_decomposition_function, explanation_decomposition_function, nbits, n_selected_features, num_train_repetitions, parallel=True):
    feature_importances_list = []
    for i in range(num_train_repetitions):
        train_graphs, test_graphs, train_targets, test_targets = train_test_split(graphs, targets, train_size=.7)
        train_data = parallel_vectorize(train_graphs, decomposition_function=base_decomposition_function, nbits=nbits)
        clf = ExtraTreesClassifier(n_estimators=100, n_jobs=-1).fit(train_data, train_targets)
        feature_importances = clf.feature_importances_  
        feature_importances_list.append(feature_importances)

    feature_importance_func = lambda feature_importances : compose(connected_component(), merge(), filter_by_feature_importance(size=n_selected_features, feature_importances=feature_importances), base_decomposition_function)
    dfs = map(feature_importance_func, feature_importances_list)

    feature_intersection_func = lambda df1, df2 : binary_combine(binary_intersection(), df1, df2)
    df_ = reduce(feature_intersection_func, dfs)
    important_part_df = compose(connected_component(), merge(), df_)
    
    #for each class get the fragment id -> freq
    class_graph_list = [[graph for graph, target in zip(graphs, targets) if target == class_type] for class_type in sorted(set(targets))] 
    results = []
    for explain_graphs in class_graph_list:
        graphofsubgraphs_list = decomposition(explain_graphs, important_part_df, nbits, parallel=parallel)
        subgraphs = [[graphofsubgraphs.nodes[u]['subgraph'] for u in graphofsubgraphs.nodes()] for graphofsubgraphs in graphofsubgraphs_list]
        subgraphs = sum(subgraphs, [])
        graphofsubgraphs_list = decomposition(subgraphs, explanation_decomposition_function, nbits, parallel=parallel)
        
        analysis_subgraphs = []
        for graphofsubgraphs in graphofsubgraphs_list:
            for u in graphofsubgraphs.nodes():
                subgraph = graphofsubgraphs.nodes[u]['subgraph']
                hash_id = graphofsubgraphs.nodes[u]['label']
                subgraph.graph['hash_id'] = hash_id
                analysis_subgraphs.append((hash_id, subgraph))
        #analysis_subgraphs = sum([[(graphofsubgraphs.nodes[u]['label'], graphofsubgraphs.nodes[u]['subgraph']) for u in graphofsubgraphs.nodes()] for graphofsubgraphs in graphofsubgraphs_list],[])
        analysis_subgraph_dict = dict(analysis_subgraphs)
        freq_analysis_subgraph_dict = Counter([hash_id for hash_id,analysis_subgraph in analysis_subgraphs])
        sorted_analysis_subgraphs = [analysis_subgraph_dict[hash_id] for hash_id, count in freq_analysis_subgraph_dict.most_common()]
        sorted_counts = [count for hash_id, count in freq_analysis_subgraph_dict.most_common()]
        results.append((sorted_analysis_subgraphs, sorted_counts))

    counts_dict = dict()
    num_classes = len(results)
    for i, (subgraphs, counts) in enumerate(results):
        for subgraph, freq in zip(subgraphs, counts):
            key = graph_hash(subgraph, context=1, nbits=nbits)
            if key not in counts_dict:
                counts_dict[key] = [subgraph, np.zeros(num_classes, dtype=int)]
            counts_dict[key][1][i] = freq

    data_dict = dict()
    for key in counts_dict:
        counts = counts_dict[key][1]
        counts = np.array(counts)
        freqs = counts/np.sum(counts)
        pi_value = chisquare(counts)[1]
        max_counts = max(counts)
        max_freq = max(freqs)
        graph = counts_dict[key][0]
        class_with_max_counts = np.argmax(counts)
        size = nx.number_of_nodes(graph)
        data_dict[key] = [graph, pi_value, max_counts, counts, class_with_max_counts, freqs, max_freq, size]
        
    return data_dict

def select_most_discriminative_explanation_from_dict(data_dict, min_counts, min_pi, min_freq, class_of_interest, min_size):
    sel_graphs = [data_dict[key][0] for key in data_dict if (data_dict[key][2] > min_counts and data_dict[key][1]<min_pi and data_dict[key][4] == class_of_interest and data_dict[key][6] > min_freq and data_dict[key][7] >= min_size)]
    sel_titles = ['%.1e %s'%(data_dict[key][1],data_dict[key][3]) for key in data_dict if (data_dict[key][2] > min_counts and data_dict[key][1]<min_pi and data_dict[key][4] == class_of_interest and data_dict[key][6] > min_freq and data_dict[key][7] >= min_size)]
    sel_pis = [data_dict[key][1] for key in data_dict if (data_dict[key][2] > min_counts and data_dict[key][1]<min_pi and data_dict[key][4] == class_of_interest and data_dict[key][6] > min_freq and data_dict[key][7] >= min_size)]
    idxs = np.argsort(sel_pis)
    sorted_graphs = [sel_graphs[idx] for idx in idxs]
    sorted_titles = [sel_titles[idx] for idx in idxs]
    return sorted_graphs, sorted_titles

def do_explain(graphs, targets, base_decomposition_function, explanation_decomposition_function, nbits, n_selected_features, num_train_repetitions, min_counts, min_pi, min_freq, class_of_interest, min_size, parallel=True):
    data_dict = select_most_discriminative_explanation(graphs, targets, base_decomposition_function, explanation_decomposition_function, nbits, n_selected_features, num_train_repetitions, parallel)
    sorted_graphs, sorted_titles = select_most_discriminative_explanation_from_dict(data_dict, min_counts, min_pi, min_freq, class_of_interest, min_size)
    return sorted_graphs, sorted_titles