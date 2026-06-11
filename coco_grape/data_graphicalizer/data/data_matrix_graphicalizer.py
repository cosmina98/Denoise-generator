import numpy as np
import scipy as sp
import networkx as nx
from toolz import partition_all
import multiprocessing_on_dill as mp

def norm_importance_func(data_mtx):
    return np.linalg.norm(data_mtx, axis=0)

def var_importance_func(data_mtx):
    return np.std(data_mtx, axis=0)

def data_matrix_to_feature_graph(data_mtx, k, importance_func, labeled=True, use_rank_correlation=True):
    #correlation defines the metric
    if use_rank_correlation: C = sp.stats.spearmanr(data_mtx)[0]
    else: C = np.corrcoef(data_mtx.T)
    C = np.absolute(C)
    C = np.nan_to_num(C)

    #feature neighbours are sorted from the most correlated to the least
    neighbors_idxs_mtx = np.argsort(-C, axis=1)
    #feature importance is the norm of the column vector corresponding to the feature
    importance = importance_func(data_mtx)
    n_nodes = len(importance)
    parents_list = []
    for feature_idx, neighbors_idxs in enumerate(neighbors_idxs_mtx):
        counter = 0
        feature_parents = []
        #for each feature scan the neighbours in order
        for neighbor_idx in neighbors_idxs:
            #after finding a maximum of k parents stop
            if counter >= k: break
            #if a neighnour has higher importance than the current feature add it as a parent
            if importance[neighbor_idx] > importance[feature_idx]: 
                feature_parents.append(neighbor_idx)
                counter += 1
        parents_list.append(feature_parents)
    #build a graph connecting each feature to its (at most) k parents
    graph = nx.Graph()
    if labeled: graph.add_nodes_from([(idx, {'label': idx, 'importance':importance[idx]}) for idx in range(n_nodes)])
    else: graph.add_nodes_from([(idx, {'label': '-'}) for idx in range(n_nodes)])
    for node_idx, parents in enumerate(parents_list):
        for parent_idx in parents:
            w = C[node_idx, parent_idx]
            graph.add_edge(node_idx, parent_idx, label='-', weight=w)
    return graph


class DataMatrixGraphicalizer(object):
    def __init__(self, importance_func=norm_importance_func, n_edges=1, labeled=True, use_rank_correlation=True):
        self.importance_func = importance_func
        self.n_edges = n_edges
        self.labeled = labeled
        self.use_rank_correlation = use_rank_correlation
        
    def fit(self, data, targets=None):
        return self
    
    def transform(self, data):
        graphs = []
        for data_mtx in data:
            graph = data_matrix_to_feature_graph(data_mtx, self.n_edges, importance_func=self.importance_func, labeled=self.labeled, use_rank_correlation=self.use_rank_correlation)
            graphs.append(graph)
        return graphs

    def fit_transform(self, data, targets=None):
        return self.fit(data, targets).transform(data)



def data_to_graph(orig_data_mtx, targets=None, max_n_edges=5, min_corrcoef=0.9, max_corrcoef=0.99, min_corrcoef_to_target=0.5):
    eps=1e-6
    data_mtx = orig_data_mtx + np.random.rand(*orig_data_mtx.shape)*eps
    if targets is not None:
        data_mtx = np.hstack([data_mtx, targets.reshape(-1,1)])
    C = np.absolute(np.corrcoef(data_mtx.T))
    if targets is not None:
        node_to_target_corrcoeffs = C[-1,:-2]
        C = C[:-2,:-2]
    min_th = np.quantile(C,min_corrcoef)
    max_th = np.quantile(C,max_corrcoef)
    C[C < min_th] = 0
    C[C > max_th] = 0
    idxs = np.argsort(C, axis=1)
    idxs = idxs[:,:len(C)-max_n_edges]
    idxs_mtx = idxs[:,:C.shape[1]-max_n_edges]
    for i,idxs in enumerate(idxs_mtx):
        for j in idxs:
            C[i,j] = 0
    C = (C + C.T)/2
    C = C.astype(bool).astype(int)
    C = C - np.diag(np.diag(C))
    G = nx.from_numpy_array(C)
    nx.set_node_attributes(G,{i:str(i) for i in range(len(C))}, 'label')
    nx.set_edge_attributes(G,'-', 'label')
    if targets is not None:
        #remove all nodes that are not above threshold corr_coeff w.r.t target
        node_idxs = np.where(node_to_target_corrcoeffs >= min_corrcoef_to_target)[0]
        G = G.subgraph(node_idxs)
    return G

class FeatureCorrelationGraphicalizer(object):
    def __init__(self, max_n_edges=5, min_corrcoef=0.9, max_corrcoef=0.99, min_corrcoef_to_target=0.5, eps=1e-1, attribute_key='vec', parallel=True):
        self.max_n_edges = max_n_edges
        self.min_corrcoef = min_corrcoef
        self.max_corrcoef = max_corrcoef
        self.min_corrcoef_to_target = min_corrcoef_to_target
        self.eps = eps
        self.attribute_key = attribute_key
        self.parallel = parallel
        
    def fit(self, data, targets=None):
        self.graph_template = data_to_graph(data, targets, max_n_edges=self.max_n_edges, min_corrcoef=self.min_corrcoef, max_corrcoef=self.max_corrcoef, min_corrcoef_to_target=self.min_corrcoef_to_target)
        return self
    
    def transform_sequential(self, data):
        graphs = []
        for data_row in data:
            graph = nx.Graph(self.graph_template)
            for i, val in enumerate(data_row): 
                if i in graph.nodes(): 
                    if np.absolute(val) <= self.eps: graph.remove_node(i)
                    else: graph.nodes[i][self.attribute_key] = np.array([val])
            graphs.append(graph)
        return graphs

    def transform_parallel(self, data):
        n_cpus = mp.cpu_count()
        batch_size = len(data)//n_cpus
        if len(data) < n_cpus: data_list = [data]
        else: data_list = list(partition_all(batch_size, data))
        pool = mp.Pool(n_cpus)
        results = pool.map(self.transform_sequential, data_list)
        pool.close()
        all_list_of_graphs = []
        for list_of_graphs in results:
            all_list_of_graphs.extend(list_of_graphs)
        return all_list_of_graphs
    

    def transform(self, data):
        if self.parallel: graphs = self.transform_parallel(data)
        else: graphs = self.transform_sequential(data)
        return graphs
    

    def fit_transform(self, data, targets=None):
        return self.fit(data, targets).transform(data)
