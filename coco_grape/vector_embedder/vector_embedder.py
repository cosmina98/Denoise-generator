#!/usr/bin/env python
"""Provides scikit interface."""

import copy
import numpy as np
import scipy as sp
import networkx as nx
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.cluster import MiniBatchKMeans 
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.feature_selection import RFE
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.model_selection import RandomizedSearchCV
#from sklearn.utils.fixes import loguniform
from scipy.stats import loguniform
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from scipy.optimize import linear_sum_assignment
from scipy.stats import ortho_group
from sklearn.preprocessing import PowerTransformer


class IdentityTransformer(object):

    def __init__(self):
        self.n_components = None

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
        
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x

    def inverse_transform(self, embeddings):
        return embeddings

    def fit_transform(self, x, y=None):
        return x

class SparseToDenseTransformer(object):

    def __init__(self):
        self.n_components = None

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
        
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x.toarray()

    def fit_transform(self, x, y=None):
        return x.toarray()


class RandomFeatureSelectionTransformer(object):

    def __init__(self, n_components=None, seed=42):
        self.n_components = n_components
        self.seed = seed
        
    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
        
    def fit(self, x, y=None):
        n_components = x.shape[1]
        np.random.seed(seed=self.seed)
        self.selected_feature_idxs = np.random.permutation(n_components)[:self.n_components]
        return self

    def transform(self, x):
        embeddings = x[:,self.selected_feature_idxs]
        return embeddings

    def fit_transform(self, x, y=None):
        return self.fit(x,y).transform(x)


class RandomRotationTransformer(object):

    def __init__(self, seed=42):
        self.n_components = None
        self.seed = seed
        
    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
        
    def fit(self, x, y=None):
        self.n_components = x.shape[1]
        self.orthogonal_mtx = ortho_group.rvs(self.n_components, random_state=self.seed)
        return self

    def transform(self, x):
        embeddings = x.dot(self.orthogonal_mtx)
        return embeddings

    def fit_transform(self, x, y=None):
        return self.fit(x,y).transform(x)


class FunctionTransformer(object):

    def __init__(self, func):
        self.n_components = None
        self.func = func

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
        
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        embeddings = func(x)
        return embeddings

    def fit_transform(self, x, y=None):
        return self.fit(x,y).transform(x)


class GaussianTransformer(object):

    def __init__(self):
        self.n_components = None
        self.embedder = PowerTransformer()

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
        
    def fit(self, x, y=None):
        self.n_components = x.shape[1]
        self.embedder.fit(x)
        return self

    def transform(self, x):
        embeddings = self.embedder.transform(x)
        return embeddings

    def fit_transform(self, x, y=None):
        return self.fit(x,y).transform(x)


class FourierTransformer(object):

    def __init__(self):
        self.n_components = None

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
        
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        Xfft = np.fft.fft(x)
        Xfft = np.hstack([Xfft.real, Xfft.imag])
        return Xfft

    def fit_transform(self, x, y=None):
        z = self.fit(x, y).transform(x)
        if sp.sparse.issparse(z):
            z = z.todense()
        return z


class KPCATransformer(object):

    def __init__(self, n_components=2, kernel='rbf', gamma=1e-3, alpha=1e-3, fit_inverse_transform=False):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform
        self.embedder = None
        #self.is_fit = False

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
        
    def fit(self, X, y=None):
        #if self.is_fit: return self
        if self.gamma is None:
            parameters = dict(C=loguniform(1e0, 1e4),gamma=loguniform(1e-6, 1e0))
            svc = svm.SVC()
            clf = RandomizedSearchCV(svc, parameters, n_iter=20, n_jobs=-1)
            clf.fit(X, y)
            self.gamma = clf.best_params_['gamma']
        self.embedder = KernelPCA(n_components=self.n_components, kernel=self.kernel, gamma=self.gamma, fit_inverse_transform=self.fit_inverse_transform, alpha=self.alpha)
        self.embedder.fit(X)    
        return self

    def transform(self, X):
        Xtr = self.embedder.transform(X)
        return Xtr

    def inverse_transform(self, embeddings):
        X = self.embedder.inverse_transform(embeddings)
        return X

    def fit_transform(self, x, y=None):
        return self.fit(x, y).transform(x)


class LinearFeatureSelectionTransformer(object):

    def __init__(self, n_components=100, step=.3):
        self.n_components = n_components
        self.step = step
        self.embedder = RFE(estimator=SGDClassifier(penalty='elasticnet'), n_features_to_select=n_components, step=step)
        
    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
       
    def fit(self, x, y=None):
        self.embedder.fit(x,y)
        return self

    def transform(self, x):
        z = self.embedder.transform(x)
        if sp.sparse.issparse(z):
            z = z.todense()
        return z

    def fit_transform(self, x, y=None):
        z = self.fit(x, y).transform(x)
        if sp.sparse.issparse(z):
            z = z.todense()
        return z

class EnsembleFeatureSelectionTransformer(object):

    def __init__(self, n_components=100, step=.3, n_estimators=300):
        self.n_components = n_components
        self.step = step
        self.n_estimators = n_estimators
        self.embedder = RFE(estimator=ExtraTreesClassifier(n_estimators=n_estimators, n_jobs=-1), n_features_to_select=n_components, step=step)
        
    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
        
    def fit(self, x, y=None):
        self.embedder.fit(x,y)
        
    def transform(self, x):
        z = self.embedder.transform(x)
        if sp.sparse.issparse(z):
            z = z.todense()
        return z

    def fit_transform(self, x, y=None):
        z = self.fit(x, y).transform(x)
        if sp.sparse.issparse(z):
            z = z.todense()
        return z


class StandardMinMaxNormalizeTransformer(object):

    def __init__(self, use_standard_scaler=True, use_minmax_scaler=False, use_normalizer=False):
        self.n_components = -1
        self.use_standard_scaler = use_standard_scaler
        self.use_minmax_scaler = use_minmax_scaler
        self.use_normalizer = use_normalizer
        self.standardscaler = StandardScaler()
        self.minmaxscaler = MinMaxScaler()
        self.normalizer = Normalizer()

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
    
    def fit(self, x, y=None):
        #if self.is_fit: return self
        self.n_components = x.shape[1]
        if self.use_normalizer:
            x = self.normalizer.fit_transform(x)
        if self.use_minmax_scaler:
            x = self.minmaxscaler.fit_transform(x)
        if self.use_standard_scaler:
            x = self.standardscaler.fit_transform(x)
        return self

    def transform(self, x):
        if self.use_normalizer:
            x = self.normalizer.transform(x)
        if self.use_minmax_scaler:
            x = self.minmaxscaler.transform(x)
        if self.use_standard_scaler:
            x = self.standardscaler.transform(x)
        x = np.nan_to_num(x)
        return x

    def fit_transform(self, x, y=None):
        return self.fit(x, y).transform(x)


class RotoScaleTransformer(object):

    def __init__(self):
        self.n_components = -1
        self.rotation = self.xmin = self.xmax = self.xlen = None

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
    
    def fit(self, x, y=None):
        self.n_components = x.shape[1]
        # rotation
        u, s, vh = np.linalg.svd(x, full_matrices=True)
        self.rotation = vh.T
        x_low_rot = np.dot(x, self.rotation)

        # scale
        self.xmin = np.min(x_low_rot, axis=0).reshape(-1)
        self.xmax = np.max(x_low_rot, axis=0).reshape(-1)
        self.xlen = np.absolute(self.xmax - self.xmin).max()
        return self

    def transform(self, x):
        x_low_rot = np.dot(x, self.rotation)
        x_low_rot_rescaled = (x_low_rot - self.xmin) / self.xlen
        z = np.nan_to_num(x_low_rot_rescaled)
        return z 

    def fit_transform(self, x, y=None):
        z = self.fit(x, y).transform(x)
        z = np.nan_to_num(x_low_rot_rescaled)
        return z         


class SVDTransformer(object):

    def __init__(self, n_components=10):
        self.n_components = n_components
        if self.n_components is None:
            self.svd = TruncatedSVD(n_components=2)
        else:
            self.svd = TruncatedSVD(n_components=n_components)

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
    
    def fit(self, x, y=None):
        if self.n_components is None:
            self.n_components = min(x.shape) - 1
            self.svd = TruncatedSVD(n_components=self.n_components)
        self.svd.fit(x)
        return self

    def transform(self, x):
        return self.svd.transform(x)

    def inverse_transform(self, embeddings):
        return self.svd.inverse_transform(embeddings)

    def fit_transform(self, x, y=None):
        return self.fit(x, y).transform(x)



class LinearDiscriminantAnalysisTransformer(object):

    def __init__(self, n_components=10):
        self.n_components = n_components
        if self.n_components is None:
            self.estimator = LinearDiscriminantAnalysis(n_components=2)
        else:
            self.estimator = LinearDiscriminantAnalysis(n_components=n_components)

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
    
    def fit(self, x, y=None):
        #if self.is_fit: return self
        if self.n_components is None:
            self.n_components = min(x.shape) - 1
            self.estimator = LinearDiscriminantAnalysis(n_components=self.n_components)
        self.estimator.fit(x,y)
        return self

    def transform(self, x):
        return self.estimator.transform(x)

    def fit_transform(self, x, y=None):
        return self.fit(x, y).transform(x)


class NcaTransformer(object):

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.embedder = NeighborhoodComponentsAnalysis(n_components=self.n_components)
        
    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
    
    def fit(self, x, y=None):
        self.embedder.fit(x,y)
        return self

    def transform(self, x):
        return self.embedder.transform(x)

    def fit_transform(self, x, y=None):
        return self.fit(x, y).transform(x)



class ExtraTreesTransformer(object):

    def __init__(self, task='classification', n_estimators=100, max_depth=10, todense=False):
        self.task = task
        self.n_components = -1
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.todense = todense

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
    
    def fit(self, x, y=None):
        if self.task == 'classification':
            self.extratrees = ExtraTreesClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=42, n_jobs=-1)
        else:
            self.extratrees = ExtraTreesRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=42, n_jobs=-1)
        if y is None:
            y = np.random.randint(2, size=x.shape[0])
        self.extratrees.fit(x, y)
        return self

    def transform(self, x):
        xy, _ = self.extratrees.decision_path(x)
        if self.todense: xy = xy.todense()
        self.n_components = xy.shape[1]
        return xy

    def fit_transform(self, x, y=None):
        return self.fit(x, y).transform(x)



class TSNETransformer(object):

    def __init__(self, n_components=2, perplexity=30):
        self.n_components = n_components
        self.tsne = TSNE(n_components=n_components, perplexity=perplexity, init="pca", learning_rate="auto", n_iter=500, n_iter_without_progress=150)
        self.extratrees = ExtraTreesRegressor(n_estimators=1000, random_state=42, n_jobs=-1)
        
    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
    
    def fit(self, x, y=None):
        embeddings = self.tsne.fit_transform(x)
        self.extratrees.fit(x, embeddings)
        return self

    def transform(self, x):
        embeddings = self.extratrees.predict(x)
        return embeddings

    def fit_transform(self, x, y=None):
        return self.fit(x, y).transform(x)

class MDSTransformer(object):
    def __init__(self, n_components=None, mode='classical', max_iter=10000, add_original_features=False, n_init=5, regressor=ExtraTreesRegressor(n_estimators=300)):
        self.n_components = n_components
        self.mode = mode
        self.add_original_features = add_original_features
        self.mds = MDS(n_components=n_components, n_init=n_init, max_iter=max_iter)
        self.est = regressor
        
    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
    
    def classicalMDS(self, x):
        D = pairwise_distances(x)
        D2 = np.mat(D*D)
        n = np.shape(D2)[0]
        C = np.eye(n) - 1/n * np.ones(n)
        C = np.mat(C)
        B = -1/2*C*D2*C
        U,s,Vh = np.linalg.svd(B)
        Up = np.mat(U[:,:self.n_components])
        Sp = np.mat(np.diag(np.sqrt(s[:self.n_components])))
        Vp = np.mat(Vh.H[:,:self.n_components])
        x_low = Up*Sp
        x_low = x_low.A
        return x_low

    def _fit(self, x, y=None):
        if self.n_components is None:
            self.n_components = max(2, int(np.sqrt(x.shape[1])))
            self.mds.set_params(n_components=self.n_components)
        if self.mode == 'classical':
            x_low = self.classicalMDS(x)
        else:
            x_low = self.mds.fit_transform(x)
        if self.add_original_features:
            x_low = np.hstack([x_low, x])
        return x_low

    def fit(self, x, y=None):
        #if self.is_fit: return self
        x_low = self._fit(x, y)
        self.est.fit(x, x_low)
        return self

    def transform(self, x):
        return self.est.predict(x)

    def fit_transform(self, x, y=None):
        return self.fit(x,y).transform(x)


class EqualizerTransformer(object):

    def __init__(self, n_components=2, n_edges=3, metric='rbf', n_estimators=100):
        self.n_components = n_components
        self.n_edges = n_edges
        self.metric = metric
        self.estimator = ExtraTreesRegressor(n_estimators=n_estimators)
        self.inverse_estimator = ExtraTreesRegressor(n_estimators=n_estimators)

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
        
    def data_to_density_graph(self, data_mtx, n_edges=1, metric='rbf'):
        K = pairwise_kernels(data_mtx, metric=metric)
        neighbors = np.argsort(-K,axis=1)
        densities = np.sum(K,axis=1)

        graph = nx.Graph()
        graph.add_nodes_from(range(len(densities)))
        nx.set_node_attributes(graph, '-', 'label')
        for i, ns in enumerate(neighbors):
            counter = 0
            for n in ns:
                if n != i:
                    if densities[n] > densities[i]:
                        graph.add_edge(i,n, label='-')
                        counter += 1
                    if counter >= n_edges:
                        break
        return graph

    def graph_to_distance_matrix(self, graph):
        n = nx.number_of_nodes(graph)
        len_dict = dict(nx.all_pairs_shortest_path_length(graph))
        D = np.zeros((n,n))
        for i in len_dict:
            for j in len_dict[i]:
                D[i,j] = D[j,i] = len_dict[i][j]
        return D

    def fit(self, data_mtx, targets=None):
        graph = self.data_to_density_graph(data_mtx=data_mtx, n_edges=self.n_edges, metric=self.metric)
        distance_matrix = self.graph_to_distance_matrix(graph)
        embeddings = MDS(n_components=self.n_components, dissimilarity='precomputed').fit_transform(distance_matrix)
        self.estimator.fit(data_mtx, embeddings)
        self.inverse_estimator.fit(embeddings, data_mtx)
        return self

    def transform(self, data_mtx):
        return self.estimator.predict(data_mtx)

    def inverse_transform(self, embeddings):
        data_mtx = self.inverse_estimator.predict(embeddings)
        return data_mtx

    def fit_transform(self, data_mtx, targets=None):
        return self.fit(data_mtx, targets).transform(data_mtx)

# ----------------------------------------------------------------------------

def make_knn_graph(X, n_neighbors):
    A = kneighbors_graph(X, n_neighbors=n_neighbors)
    g = nx.from_numpy_matrix(A.todense())
    return g

def make_iterated_linear_sum_assignment_graph(X, y, n_edges_linear_assignment):
    G = nx.Graph()
    
    classes = sorted(set(y))
    # for each pairs of classes A,B
    ids = np.arange(X.shape[0])
    mask = np.zeros((X.shape[0],X.shape[0]))
    for i in range(n_edges_linear_assignment):
        for class_A, class_B in combinations(classes, 2):
            ids_A = ids[y==class_A]
            X_A = X[y==class_A]
            ids_B = ids[y==class_B]
            X_B = X[y==class_B]
            D = pairwise_distances(X_A, X_B)
            # remove previous matches
            for ii in range(D.shape[0]):
                for jj in range(D.shape[1]):
                    if mask[ids_A[ii], ids_B[jj]] == 1:
                        D[ii,jj] = np.inf
            # find best match between instance in class A and an instance in class B
            matched_ids_A, matched_ids_B = linear_sum_assignment(D)
            # add matches as edges
            for a_id, b_id in zip(matched_ids_A, matched_ids_B):
                mask[ids_A[a_id], ids_B[b_id]] = 1
                G.add_edge(ids_A[a_id], ids_B[b_id])
    return G

def make_iterated_mst_graph(X, n_iter_mst):
    D = pairwise_distances(X)
    dt = [('weight', float)]
    C = np.array(D, dtype=dt)
    gd = nx.from_numpy_matrix(C)

    for i in range(n_iter_mst):
        gmst = nx.minimum_spanning_tree(gd, weight='weight')
        if i == 0:
            g = gmst
        else:
            g = nx.compose(g, gmst)
        gdd = nx.difference(gd, g)
        gd = gd.edge_subgraph(gdd.edges()).copy()
    return g

def make_density_graph(X, n_edges=1):
    g = nx.Graph()
    g.add_nodes_from(range(np.shape(X)[0]))
    D = pairwise_distances(X)
    densities = np.sum(D, axis=0)
    ns_mtx = np.argsort(D, axis=1)
    for i, ns in enumerate(ns_mtx):
        d_i = densities[i]
        current_n_edges = 0
        for j in ns:
            if densities[j] > d_i:
                g.add_edge(i,j)
                current_n_edges += 1
                if current_n_edges >= n_edges:
                    break
    return g

def make_regression_target_graph(X, y, n_neighbors_regression_target=10, n_edges_regression_target=1):
    g = nx.Graph()
    g.add_nodes_from(range(np.shape(X)[0]))
    D = pairwise_distances(X)
    ns_mtx = np.argsort(D, axis=1)[:,:n_neighbors_regression_target]
    for i, ns in enumerate(ns_mtx):
        js = sorted(ns, key=lambda j:np.absolute(y[i]-y[j]))[:n_edges_regression_target+1]
        for j in js:
            if j != i:
                g.add_edge(i,j)
    return g

def make_graph_target_graph(X, y):
    g = nx.Graph()
    g.add_nodes_from(range(np.shape(X)[0]))
    for i, ns in enumerate(y):
        for j,n in enumerate(ns):
            if n > 0:
                g.add_edge(i,j)
    return g

def make_graph(X, y, n_neighbors, n_iter_mst, n_edges_density, n_edges_linear_assignment, n_edges_regression_target, n_neighbors_regression_target, auxiliary_information):
    if n_neighbors == 0:
        g_knn = nx.Graph()
    else:
        g_knn = make_knn_graph(X, n_neighbors)
    if n_iter_mst == 0:
        g_mst = nx.Graph()
    else:
        g_mst = make_iterated_mst_graph(X, n_iter_mst)
    if n_edges_density == 0:
        g_density = nx.Graph()
    else:
        g_density = make_density_graph(X, n_edges_density)
    if n_edges_linear_assignment == 0:
        g_linear_assignment = nx.Graph()
    else:
        g_linear_assignment = make_iterated_linear_sum_assignment_graph(X, y, n_edges_linear_assignment)
    if n_edges_regression_target == 0:
        g_regression_target = nx.Graph()
    else:
        g_regression_target = make_regression_target_graph(X, y, n_neighbors_regression_target=n_neighbors_regression_target, n_edges_regression_target=n_edges_regression_target)
    if auxiliary_information=='graph':
        g_graph_target = make_graph_target_graph(X, y)
    else:
        g_graph_target = nx.Graph()
    g = nx.compose(g_knn, g_mst)
    g = nx.compose(g, g_density)
    g = nx.compose(g, g_linear_assignment)
    g = nx.compose(g, g_regression_target)
    g = nx.compose(g, g_graph_target)
    return g


def euclidean_distance(x, z):
    return np.linalg.norm(x - z)
    
def annotate_graph(g, X, y, distance_type='unitary', within_class_shrinkage_factor=0.25, auxiliary_information='classification'):
    if distance_type == 'rank':
        D = pairwise_distances(X)
        ranks = np.log(rankdata(D,axis=1)+1)
    if y is not None:
        for u, t in zip(g.nodes(), y):
            g.nodes[u]['label'] = t
    else:
        for u in g.nodes():
            g.nodes[u]['label'] = '-'
    for u, v in g.edges():
        g.edges[u, v]['label'] = '-'
        if auxiliary_information == 'classification':
            if g.nodes[u]['label'] == g.nodes[v]['label']:
                factor = within_class_shrinkage_factor
            else:
                factor = 1
        elif auxiliary_information == 'regression':
            factor = within_class_shrinkage_factor * np.absolute(g.nodes[u]['label'] - g.nodes[v]['label'])
        elif auxiliary_information == 'graph':
            #consider the connectivity information encoded in the adjacency matrix y
            #if the two instances u and v are linked then the factor is 1/(1+link_strength)
            #if the link_strength is 0, then the factor is 1
            #if the link_strength is 2, then the factor is 1/3
            factor = 1/(1+y[u,v]) * within_class_shrinkage_factor
        if distance_type == 'distance':
            g.edges[u, v]['weight'] = factor * euclidean_distance(X[u], X[v])
        elif distance_type == 'unitary':
            g.edges[u, v]['weight'] = factor 
        elif distance_type == 'rank':
            g.edges[u, v]['weight'] = factor * np.abs(ranks[u,v])

def compute_distance_mtx(g):
    n = g.number_of_nodes()
    distance_mtx = np.zeros((n, n))
    dist_dict = dict(nx.shortest_path_length(g, weight='weight'))
    for k in sorted(dist_dict):
        for j in sorted(dist_dict[k]):
            distance_mtx[k, j] = dist_dict[k][j]
    return distance_mtx

class NetworkEmbeddingTransformer(object):

    def __init__(self, 
        n_components=None, 
        n_neighbors=3, 
        n_iter_mst=0, 
        density_n_edges=1, 
        n_edges_linear_assignment=0, 
        n_edges_regression_target=0, 
        n_neighbors_regression_target=0,
        distance_type='distance', 
        auxiliary_information='classification',
        within_class_shrinkage_factor=1, 
        add_original_features=False, 
        graph_embedding='MDS',
        max_iter=10000, 
        n_init=5, 
        regressor=ExtraTreesRegressor(n_estimators=300)):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.n_iter_mst = n_iter_mst
        self.density_n_edges = density_n_edges
        self.n_edges_linear_assignment = n_edges_linear_assignment
        self.n_edges_regression_target = n_edges_regression_target
        self.n_neighbors_regression_target = n_neighbors_regression_target
        self.distance_type = distance_type
        self.within_class_shrinkage_factor = within_class_shrinkage_factor
        self.add_original_features = add_original_features
        self.auxiliary_information = auxiliary_information
        self.graph_embedding = graph_embedding
        self.mds = MDS(n_components=n_components, n_init=n_init, max_iter=max_iter, dissimilarity='precomputed')
        self.est = regressor
        
    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
    
    def graph_layout(self, graph):
        if self.graph_embedding == 'MDS':
            distance_mtx = compute_distance_mtx(graph)
            x_low = self.mds.fit_transform(distance_mtx)
        elif self.graph_embedding == 'spectral':
            pos = nx.spectral_layout(graph, dim=self.n_components)
            x_low = np.vstack([pos[id] for id in sorted(pos)])
        return x_low

    def _fit(self, x, y=None):
        if self.n_components is None:
            self.n_components = max(2, int(np.sqrt(x.shape[1])))
            self.mds.set_params(n_components=self.n_components)
        graph = make_graph(x, y, 
            n_neighbors=self.n_neighbors, 
            n_iter_mst=self.n_iter_mst, 
            n_edges_density=self.density_n_edges, 
            n_edges_linear_assignment=self.n_edges_linear_assignment,
            n_edges_regression_target=self.n_edges_regression_target, 
            n_neighbors_regression_target=self.n_neighbors_regression_target, 
            auxiliary_information=self.auxiliary_information)
        annotate_graph(graph, x, y, distance_type=self.distance_type, within_class_shrinkage_factor=self.within_class_shrinkage_factor, auxiliary_information=self.auxiliary_information)
        x_low = self.graph_layout(graph) 
        if self.add_original_features:
            x_low = np.hstack([x_low, x])
        return x_low

    def fit(self, x, y=None):
        x_low = self._fit(x, y)
        self.est.fit(x, x_low)
        return self

    def transform(self, x):
        return self.est.predict(x)

    def fit_transform(self, x, y=None):
        return self.fit(x,y).transform(x)


class PredictiveBaggingRegressorTransformer():
    def __init__(self, base_estimator, n_components):
        self.base_estimator = base_estimator
        self.n_components = n_components
        self.estimators_ = []

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
        
    def fit(self, data_instances, targets):
        ids = np.arange(len(data_instances))
        for i in range(self.n_components):
            sel_ids = np.random.choice(ids, size=len(ids), replace=True)
            training_data_instances = [data_instances[id] for id in sel_ids]
            training_targets = [targets[id] for id in sel_ids]
            est = self.base_estimator.fit(training_data_instances, training_targets)
            self.estimators_.append(copy.deepcopy(est))
        return self
    
    def predict(self, data_instances):
        preds_mtx = self.transform(data_instances)
        preds = np.mean(preds_mtx, axis=1).flatten()
        return preds
    
    def transform(self, data_instances):
        preds_mtx = np.hstack([est.predict(data_instances).reshape(-1,1) for est in self.estimators_])
        return preds_mtx
    
    def fit_transform(self, data_instances, targets):
        return self.fit(data_instances, targets).transform(data_instances)

    def fit_predict(self, data_instances, targets=None):
        return self.fit(data_instances, targets).predict(data_instances)

class BaggingRegressorTransformer():
    def __init__(self, base_estimator, n_components):
        self.base_estimator = base_estimator
        self.n_components = n_components
        self.estimators_ = []

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
        
    def fit(self, data_instances, targets):
        self.estimators_ = []
        ids = np.arange(len(data_instances))
        for i in range(self.n_components):
            sel_ids = np.random.choice(ids, size=len(ids), replace=True)
            training_data_instances = np.array([data_instances[id] for id in sel_ids])
            training_targets = [targets[id] for id in sel_ids]
            est = self.base_estimator.fit(training_data_instances, training_targets)
            self.estimators_.append(copy.deepcopy(est))
        return self
    
    def transform(self, data_instances):
        preds_mtx = np.hstack([np.nan_to_num(est.predict_proba(data_instances)) for est in self.estimators_])
        return preds_mtx
    
    def fit_transform(self, data_instances, targets):
        return self.fit(data_instances, targets).transform(data_instances)

class ClusteringBaggingRegressorTransformer():
    def __init__(self, base_estimator, n_components, n_clusters):
        self.base_estimator = base_estimator
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.clusterer = MiniBatchKMeans(n_clusters=n_clusters)
        

    def __repr__(self):
        infos = ['%s:%s'%(key,value) for key,value in self.__dict__.items()]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)
        
    def fit(self, data_instances, targets):
        self.estimators_ = []
        clusters_ids = self.clusterer.fit_predict(data_instances)
        ids = np.arange(len(data_instances))
        for i in range(self.n_components):
            sel_ids = np.random.choice(ids, size=len(ids), replace=True)
            training_data_instances = np.array([data_instances[id] for id in sel_ids])
            training_targets = [clusters_ids[id] if targets[id]==1 else clusters_ids[id]+self.n_clusters for id in sel_ids]
            est = self.base_estimator.fit(training_data_instances, training_targets)
            self.estimators_.append(copy.deepcopy(est))
        return self
    
    def transform(self, data_instances):
        preds_mtx = np.hstack([np.nan_to_num(est.predict_proba(data_instances)) for est in self.estimators_])
        return preds_mtx
    
    def fit_transform(self, data_instances, targets):
        return self.fit(data_instances, targets).transform(data_instances)


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

class InverseEnabledTransformer(object):
    def __init__(self, transformer, regressor=ExtraTreesRegressor(n_estimators=300)):
        self.transformer = transformer
        self.regressor = regressor

    def fit(self, data_mtx, targets=None):
        self.transformer.fit(data_mtx, targets)
        embeddings = self.transformer.transform(data_mtx)
        self.regressor.fit(embeddings, data_mtx)
        return self

    def transform(self, data_mtx):
        embeddings = self.transformer.transform(data_mtx)
        return embeddings

    def inverse_transform(self, embeddings):
        if hasattr(self.transformer, 'inverse_transform'):
            return self.transformer.inverse_transform(embeddings)
        data_mtx = self.regressor.predict(embeddings)
        return data_mtx

    def fit_transform(self, data_mtx, targets=None):
        return self.fit(data_mtx, targets).transform(data_mtx)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

class VectorEmbedder(object):

    def __init__(self, transformers, enable_inverse_transform=False):
        if enable_inverse_transform: 
            self.transformers = [copy.deepcopy(InverseEnabledTransformer(copy.deepcopy(transformer))) for transformer in transformers]
        else:
            self.transformers = transformers

    def __repr__(self):
        infos = []
        infos += ['transformer_%d:%s'%(i,str(transformer)) for i,transformer in enumerate(self.transformers)]
        infos = ', '.join(infos) 
        return '%s(%s)'%(self.__class__.__name__, infos)

    def fit(self, instances, targets=None):
        """fit."""
        #if self.is_fit: return self     
        x = self.transformers[0].fit_transform(instances, targets)
        for transformer in self.transformers[1:]:
            x = transformer.fit_transform(x, targets)
        return self

    def transform(self, instances):
        """transform."""
        x = self.transformers[0].transform(instances)
        for transformer in self.transformers[1:]:
            x = transformer.transform(x)
        return x

    def inverse_transform(self, embeddings):
        x = self.transformers[-1].inverse_transform(embeddings)
        for transformer in self.transformers[-2::-1]:
            x = transformer.inverse_transform(x)
        return x        

    def fit_transform(self, instances, targets=None):
        return self.fit(instances, targets).transform(instances)
