import numpy as np
from sklearn.ensemble import RandomForestRegressor
import copy
import multiprocessing_on_dill as mp
from sklearn.ensemble import RandomForestRegressor


class ConsistencyGenerator(object):
    def __init__(self, base_estimator=RandomForestRegressor(n_estimators=30), n_features=10, n_iterations=2, iteration_weight=0.5, use_randomized=False, parallel=True, fraction_randomly_sampled_features=0.5):
        self.base_estimator = base_estimator
        self.n_features = n_features
        self.fraction_randomly_sampled_features = fraction_randomly_sampled_features
        self.n_iterations = n_iterations
        self.iteration_weight = iteration_weight
        self.use_randomized = use_randomized
        self.parallel = parallel
        
    def get_correlated_features_map(self, data_mtx):
        X = data_mtx+np.random.rand(*data_mtx.shape)*1e-6
        C = np.nan_to_num(np.abs(np.corrcoef(X.T)))
        C = C-np.diag(np.diag(C))
        C = np.nan_to_num(C/np.sum(C, axis=1).reshape(-1,1))
        if self.n_features > 1: size = self.n_features
        else: size = int(data_mtx.shape[1] * self.n_features)
        n_features_selected_deterministically = int((1-self.fraction_randomly_sampled_features) * size)
        n_features_selected_randomly = int(self.fraction_randomly_sampled_features * size)
        Id = np.argsort(-C, axis=1)[:,:n_features_selected_deterministically]
        Ir = [np.random.choice(len(row),size=n_features_selected_randomly, replace=False, p=row) for row in C]
        I = np.hstack([Id,Ir])
        return I

    def fit_(self, data_mtx):
        self.correlated_features = self.get_correlated_features_map(data_mtx)
        self.estimators = []
        for i,feature_idxs in enumerate(self.correlated_features):
            y = data_mtx[:,i]
            X = data_mtx[:,feature_idxs]
            estimator = copy.deepcopy(self.base_estimator).fit(X,y)
            self.estimators.append(estimator)
        return self
    
    def fit(self, data_mtx):
        self.correlated_features = self.get_correlated_features_map(data_mtx)
        self.n_dim = data_mtx.shape[1]
        self.estimators = [copy.deepcopy(self.base_estimator) for i in range(self.n_dim)]
        data = []
        for idx,feature_idxs in enumerate(self.correlated_features):
            y = data_mtx[:,idx]
            X = data_mtx[:,feature_idxs]
            data.append((idx, X, y))
        if self.parallel == False:
            for data_item in data:
                idx, X, y = data_item
                self.estimators[idx].fit(X,y)
        else:
            def func(data_item):
                idx, X, y = data_item
                return self.estimators[idx].fit(X, y)
            n_cpus = mp.cpu_count()
            pool = mp.Pool(n_cpus)
            self.estimators = pool.map(func, data)
            pool.close()      
        return self
    
    def predict(self, data_mtx):
        if self.use_randomized: return self.predict_randomized(data_mtx)
        else: return self.predict_serialized(data_mtx)
        
    def predict_serialized(self, data_mtx):
        X = copy.deepcopy(data_mtx)
        for it in range(self.n_iterations):
            Xp = [self.estimators[idx].predict(X[:,feature_idxs]) for idx, feature_idxs in enumerate(self.correlated_features)]
            Xp = np.array(Xp).T
            X = (1-self.iteration_weight) * X + self.iteration_weight * Xp 
        return X
    
    def predict_randomized(self, data_mtx):
        X = copy.deepcopy(data_mtx)
        n_features = self.correlated_features.shape[0]
        feature_idxs = np.random.randint(n_features, size=self.n_iterations)
        for feature_idx in feature_idxs:
            X_view = X[:,self.correlated_features[feature_idx]]
            X[:,feature_idx] = (1-self.iteration_weight) * X[:,feature_idx] +  self.iteration_weight * self.estimators[feature_idx].predict(X_view)
        return X
    

class ClassConditionalConsistencyGenerator(object):
    def __init__(self, consistency_generator=None):
        self.consistency_generator = consistency_generator
        self.consistency_generators = []

    def set_params(self, n_iterations=None, iteration_weight=None):
        for generator in self.consistency_generators:
            if n_iterations is not None: generator.n_iterations = n_iterations
            if iteration_weight is not None: generator.iteration_weight = iteration_weight
        return self
        
    def fit(self, data_mtx, targets):
        self.target_set = sorted(set(targets))
        self.consistency_generators = [copy.deepcopy(self.consistency_generator) for t in self.target_set]
        targets_masks_list = [targets==t for t in self.target_set]
        self.consistency_generators = [generator.fit(data_mtx[targets_mask]) for generator, targets_mask in zip(self.consistency_generators, targets_masks_list)]
        return self
    
    def predict(self, data_mtx, targets):
        X = copy.deepcopy(data_mtx)
        targets_masks_list = [targets==t for t in self.target_set]
        for generator, targets_mask in zip(self.consistency_generators, targets_masks_list):
            X[targets_mask] = generator.predict(data_mtx[targets_mask])
        return X
