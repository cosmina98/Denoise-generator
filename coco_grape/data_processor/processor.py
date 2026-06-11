#!/usr/bin/env python
"""Provides scikit interface."""

import numpy as np
import  scipy as sp
import networkx as nx
from copy import deepcopy
import pickle
import time
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics.pairwise import pairwise_kernels



class NodeVectorizerToGraphVectorizer(object):
    def __init__(self, node_vectorizer=None):
        self.node_vectorizer = node_vectorizer

    def __repr__(self):
        infos = ['node_vectorizer=%s' % str(self.node_vectorizer)]
        infos = ', '.join(infos)
        return '%s(%s)' % (self.__class__.__name__, infos)

    def fit(self, data_instances, targets=None):
        self.node_vectorizer.fit_transform(data_instances, targets)
        return self

    def transform(self, data_instances):
        node_encodings_list = self.node_vectorizer.transform(data_instances)
        encodings = [np.sum(node_encodings, axis=0) for node_encodings in node_encodings_list]
        encodings = np.vstack(encodings)
        return encodings

    def fit_transform(self, data_instances, targets=None):
        return self.fit(data_instances, targets).transform(data_instances)


class DataVectorizer(object):
    def __init__(self, data_graphicalizer=None, graph_vectorizer=None):
        self.data_graphicalizer = data_graphicalizer
        self.graph_vectorizer = graph_vectorizer

    def __repr__(self):
        infos = ['data_graphicalizer=%s' % str(self.data_graphicalizer), 'graph_vectorizer=%s'%str(self.graph_vectorizer)]
        infos = ', '.join(infos)
        return '%s(%s)' % (self.__class__.__name__, infos)

    def fit(self, data_instances, targets=None):
        graphs = self.data_graphicalizer.fit_transform(data_instances, targets)
        self.graph_vectorizer.fit(graphs, targets)
        return self

    def transform(self, data_instances):
        graphs = self.data_graphicalizer.transform(data_instances)
        encodings = self.graph_vectorizer.transform(graphs)
        return encodings

    def inverse_transform(self, encodings):
        graphs = self.graph_vectorizer.inverse_transform(encodings)
        data_instances = self.data_graphicalizer.inverse_transform(graphs)
        return data_instances

    def fit_transform(self, data_instances, targets=None):
        return self.fit(data_instances, targets).transform(data_instances)

    def save(self, filename='model.obj'):
        filehandler = open(filename, 'wb') 
        pickle.dump(self, filehandler)
        return self

    def load(self, filename='model.obj'):
        filehandler = open(filename, 'rb') 
        self = pickle.load(filehandler)
        return self

class DataTransformer(object):
    def __init__(self, data_vectorizer=None, vector_embedder=None):
        self.data_vectorizer = data_vectorizer
        self.vector_embedder = vector_embedder
        self.is_fit = False

    def __repr__(self):
        infos = ['data_vectorizer=%s' % str(self.data_vectorizer), 'vector_embedder=%s'%str(self.vector_embedder)]
        infos = ', '.join(infos)
        return '%s(%s)' % (self.__class__.__name__, infos)

    def fit(self, data_instances, targets=None):
        #if self.is_fit: return self
        encodings = self.data_vectorizer.fit_transform(data_instances, targets)
        self.vector_embedder.fit(encodings, targets)
        return self

    def transform(self, data_instances):
        encodings = self.data_vectorizer.transform(data_instances)
        encodings = self.vector_embedder.transform(encodings)
        return encodings

    def inverse_transform(self, encodings):
        encodings = self.vector_embedder.inverse_transform(encodings)
        data_instances = self.data_vectorizer.inverse_transform(encodings)
        return data_instances

    def fit_transform(self, data_instances, targets=None):
        return self.fit(data_instances, targets).transform(data_instances)

    def save(self, filename='model.obj'):
        filehandler = open(filename, 'wb') 
        pickle.dump(self, filehandler)
        return self

    def load(self, filename='model.obj'):
        filehandler = open(filename, 'rb') 
        self = pickle.load(filehandler)
        return self

class ConcatDataTransformer(object):
    def __init__(self, data_transformers, weights=None):
        self.data_transformers = data_transformers
        self.weights = weights

    def __repr__(self):
        infos = ['data_transformer %d = %s' % (i, data_transformer) for i, data_transformer in enumerate(self.data_transformers)]
        infos = ', '.join(infos)
        return '%s(%s)' % (self.__class__.__name__, infos)

    def fit(self, data_instances, targets=None):
        self.data_transformers = [data_transformer.fit(data_instances, targets) for data_transformer in self.data_transformers]
        return self

    def reweight(self, embeddings, weight):
        embeddings = weight * embeddings / np.sum(embeddings, axis=1)
        return embeddings

    def transform(self, data_instances):
        if self.weights is None: 
            encodings = np.hstack([data_transformer.transform(data_instances) for data_transformer in self.data_transformers])
        else:
            encodings = np.hstack([self.reweight(data_transformer.transform(data_instances), weight) for data_transformer, weight in zip(self.data_transformers, self.weights)])
        return encodings

    def fit_transform(self, data_instances, targets=None):
        return self.fit(data_instances, targets).transform(data_instances)

    def inverse_transform(self, embeddings):
        raise Exception('Not Implemented')

class SparseConcatDataTransformer(object):
    def __init__(self, data_transformers, weights=None):
        self.data_transformers = data_transformers
        self.weights = weights

    def __repr__(self):
        infos = ['data_transformer %d = %s' % (i, data_transformer) for i, data_transformer in enumerate(self.data_transformers)]
        infos = ', '.join(infos)
        return '%s(%s)' % (self.__class__.__name__, infos)

    def reweight(self, embeddings, weight):
        embeddings = sp.sparse.csr_matrix(weight*embeddings/embeddings.sum(axis=1))
        return embeddings

    def fit(self, data_instances, targets=None):
        self.data_transformers = [data_transformer.fit(data_instances, targets) for data_transformer in self.data_transformers]
        return self

    def transform(self, data_instances):
        if self.weights is None: 
            encodings = sp.sparse.hstack([data_transformer.transform(data_instances) for data_transformer in self.data_transformers])
        else:
            encodings = sp.sparse.hstack([self.reweight(data_transformer.transform(data_instances), weight) for data_transformer, weight in zip(self.data_transformers, self.weights)])
        return encodings

    def fit_transform(self, data_instances, targets=None):
        return self.fit(data_instances, targets).transform(data_instances)

    def inverse_transform(self, embeddings):
        raise Exception('Not Implemented')



class DataEstimator(object):
    def __init__(self, data_transformer=None, estimator=None):
        self.data_transformer = data_transformer
        self.estimator = estimator

    def get_params(self, deep=True):
        return {'data_transformer': self.data_transformer, 'estimator': self.estimator}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def __repr__(self):
        infos = ['data_transformer=%s'%(str(self.data_transformer).replace('\n',' ').replace('  ',' ')), 'estimator=%s'%str(self.estimator).replace('\n',' ').replace('  ',' ')]
        infos = ', '.join(infos)
        return '%s(%s)' % (self.__class__.__name__, infos)

    def fit(self, data_instances, targets=None):
        encodings = self.data_transformer.fit_transform(data_instances, targets)
        self.estimator.fit(encodings, targets)
        return self

    def transform(self, data_instances):
        return self.data_transformer.transform(data_instances)

    def predict(self, data_instances):
        encodings = self.data_transformer.transform(data_instances)
        preds = self.estimator.predict(encodings)
        return preds

    def decision_function(self, data_instances):
        encodings = self.data_transformer.transform(data_instances)
        preds = self.estimator.decision_function(encodings)
        return preds

    def predict_proba(self, data_instances):
        encodings = self.data_transformer.transform(data_instances)
        preds = self.estimator.predict_proba(encodings)
        return preds

    def predict_and_decision_function(self, data_instances):
        encodings = self.data_transformer.transform(data_instances)
        preds = self.estimator.predict(encodings)
        vals = self.estimator.decision_function(encodings)
        return preds, vals

    def predict_and_predict_proba(self, data_instances):
        encodings = self.data_transformer.transform(data_instances)
        preds = self.estimator.predict(encodings)
        probs = self.estimator.predict_proba(encodings)
        return preds, probs

    def save(self, filename='model.obj'):
        filehandler = open(filename, 'wb') 
        pickle.dump(self, filehandler)
        return self

    def load(self, filename='model.obj'):
        filehandler = open(filename, 'rb') 
        self = pickle.load(filehandler)
        return self


class IteratedDataEstimator(object):
    def __init__(self, vectorizers, estimator, max_uncertainty=0.25, cv=3, method='isotonic', ensemble=False, verbose=False):
        self.vectorizers = vectorizers
        self.margin = 0.5 - min(max_uncertainty, 0.5)
        self.cv = cv
        self.verbose = verbose
        self.calibrated_classifier = CalibratedClassifierCV(estimator, method=method, cv=cv, n_jobs=-1, ensemble=ensemble)
        self.calibrated_classifiers = []
        
    def fit(self, data, targets):
        working_data = deepcopy(data)
        working_targets = deepcopy(targets)
        for id_vectorizer, vectorizer in enumerate(self.vectorizers):
            start = time.time()
            working_train_data = vectorizer.transform(working_data)
            elapsed = time.time() - start
            self.calibrated_classifier.fit(working_train_data, working_targets)
            self.calibrated_classifiers.append(deepcopy(self.calibrated_classifier))
            if id_vectorizer < len(self.vectorizers) - 1: #execute the following code until the one before the last
                preds = cross_val_predict(deepcopy(self.calibrated_classifier), working_train_data, working_targets, cv=self.cv, method='predict_proba')[:,-1]
                confident_prediction_ids, uncertain_prediction_ids = self.partition_ids(preds)
                if self.verbose: print('train: #confident_predictions:%5d (%.2f)    #uncertain_predictions:%5d (%.2f)   [%4d instances vectorization time: %.1f s (%.1f min)]'%(len(confident_prediction_ids), len(confident_prediction_ids)/len(working_targets), len(uncertain_prediction_ids), len(uncertain_prediction_ids)/len(working_targets), len(working_targets), elapsed, elapsed/60))
                working_data = [working_data[id] for id in uncertain_prediction_ids]
                working_targets = [working_targets[id] for id in uncertain_prediction_ids]
                if self.break_condition(working_targets): break
        return self

    def partition_ids(self, preds):
        low_th  = 0.5 - self.margin
        high_th = 0.5 + self.margin
        uncertain_prediction_ids = [i for i,p in enumerate(preds) if p>low_th and p<high_th]
        confident_prediction_ids = [i for i,p in enumerate(preds) if p<=low_th or p>=high_th]
        return confident_prediction_ids, uncertain_prediction_ids
    
    def break_condition(self, working_targets):
        for class_id in sorted(set(working_targets)):
            n = len([t for t in working_targets if t == class_id])
            if n < self.cv: return True
        return False
        
    def predict(self, data):
        predictions = -1 * np.ones(len(data))
        probabilities = -1 * np.ones(len(data))
        working_data = data
        working_id_map = np.arange(len(data))
        for id_calibrated_classifier, (vectorizer, calibrated_classifier) in enumerate(zip(self.vectorizers, self.calibrated_classifiers)):
            start = time.time()
            working_train_data = vectorizer.transform(working_data)
            elapsed = time.time() - start
            preds = calibrated_classifier.predict_proba(working_train_data)[:,-1]
            confident_prediction_ids, uncertain_prediction_ids = self.partition_ids(preds)
            if self.verbose: print('test:  #confident_predictions:%5d (%.2f)    #uncertain_predictions:%5d (%.2f)   [%4d instances vectorization time: %.1f s (%.1f min)]'%(len(confident_prediction_ids), len(confident_prediction_ids)/len(working_id_map), len(uncertain_prediction_ids), len(uncertain_prediction_ids)/len(working_id_map), len(working_id_map), elapsed, elapsed/60))
            for id in confident_prediction_ids: probabilities[working_id_map[id]] = preds[id]
            for id in confident_prediction_ids: predictions[working_id_map[id]] = 1 if (preds[id]>0.5) else 0
            if len(working_data) <= 0 or id_calibrated_classifier == len(self.calibrated_classifiers) - 1: 
                for id in uncertain_prediction_ids: probabilities[working_id_map[id]] = preds[id]
                for id in uncertain_prediction_ids: predictions[working_id_map[id]] = 1 if (preds[id]>0.5) else 0                
                break
            working_data = [working_data[id] for id in uncertain_prediction_ids]
            working_id_map = [working_id_map[id] for id in uncertain_prediction_ids]
        self.predictions = predictions
        self.probabilities = probabilities
        return predictions



class StackedDataEstimator(object):
    def __init__(self, estimators=None):
        self.estimators = estimators

    def __repr__(self):
        infos = ['estimator %d = %s' % (i,clf) for i,clf in enumerate(self.estimators)]
        infos = ', '.join(infos)
        return '%s(%s)' % (self.__class__.__name__, infos)

    def fit(self, data_instances, targets=None):
        x = self.estimators[0].fit(data_instances, targets).predict(data_instances)
        if np.ndim(x) == 1:
            x = x.reshape(-1, 1)
        for estimator in self.estimators[1:]:
            x = estimator.fit(x, targets).predict(x)
            if np.ndim(x) == 1:
                x = x.reshape(-1, 1)
        return self

    def predict(self, data_instances):
        x = self.estimators[0].predict(data_instances)
        if np.ndim(x) == 1:
            x = x.reshape(-1, 1)
        for estimator in self.estimators[1:]:
            x = estimator.predict(x)
            if np.ndim(x) == 1:
                x = x.reshape(-1, 1)
        return x.flatten()

    def fit_predict(self, data_instances, targets=None):
        return self.fit(data_instances, targets).predict(data_instances)

    def save(self, filename='model.obj'):
        filehandler = open(filename, 'wb') 
        pickle.dump(self, filehandler)
        return self

    def load(self, filename='model.obj'):
        filehandler = open(filename, 'rb') 
        self = pickle.load(filehandler)
        return self

class MultiOutputEstimator():
    def __init__(self, data_transformer=None, estimator=None, use_targets_in_transformer=False):
        self.data_transformer = data_transformer
        self.data_transformers = []

        self.estimator = estimator
        self.estimators = []

        self.use_targets_in_transformer = use_targets_in_transformer

    def fit(self, data, multilabel_targets):
        self.n_estimators = multilabel_targets.shape[1]

        if not self.use_targets_in_transformer:
            encodings = self.data_transformer.fit_transform(data, targets=None)

        for i in range(self.n_estimators):
            targets = multilabel_targets[:, i]
            if self.use_targets_in_transformer:
                encodings = self.data_transformer.fit_transform(data, targets)
                self.data_transformers.append(deepcopy(self.data_transformer))
            estimator = self.estimator.fit(encodings, targets)
            self.estimators.append(deepcopy(estimator))
        return self

    def predict(self, data):
        if not self.use_targets_in_transformer:
            encodings = self.data_transformer.transform(data)
            return np.hstack([self.estimators[i].predict(encodings).reshape(-1, 1) for i in range(self.n_estimators)])
        else:
            return np.hstack([self.estimators[i].predict(self.data_transformers[i].transform(data)).reshape(-1,1) for i in range(self.n_estimators)])

    def predict_proba(self, data):
        if not self.use_targets_in_transformer:
            encodings = self.data_transformer.transform(data)
            return np.hstack([self.estimators[i].predict_proba(encodings)[:, -1].reshape(-1, 1) for i in range(self.n_estimators)])
        else:
            return np.hstack([self.estimators[i].predict_proba(self.data_transformers[i].transform(data))[:, -1].reshape(-1, 1) for i in range(self.n_estimators)])

    def predict_and_predict_proba(self, data):
        if not self.use_targets_in_transformer:
            encodings = self.data_transformer.transform(data)
            preds = np.hstack([self.estimators[i].predict(encodings).reshape(-1, 1) for i in range(self.n_estimators)])
            vals = np.hstack([self.estimators[i].predict_proba(encodings)[:, -1].reshape(-1, 1) for i in range(self.n_estimators)])
            return preds, vals
        else:
            preds = np.hstack([self.estimators[i].predict(self.data_transformers[i].transform(data)).reshape(-1, 1) for i in range(self.n_estimators)])
            vals = np.hstack([self.estimators[i].predict_proba(self.data_transformers[i].transform(data))[:, -1].reshape(-1, 1) for i in range(self.n_estimators)])
            return preds, vals

    def save(self, filename='model.obj'):
        filehandler = open(filename, 'wb') 
        pickle.dump(self, filehandler)
        return self

    def load(self, filename='model.obj'):
        filehandler = open(filename, 'rb') 
        self = pickle.load(filehandler)
        return self

class DataEstimatorWrapper(object):
    def __init__(self, objective_func):
        self.objective_func = objective_func

    def fit(self, graphs, targets):
        return self

    def predict(self, graphs):
        return self.objective_func(graphs)


class MultiObjectiveDataEstimator(object):
    def __init__(self, data_estimators, data_transformer):
        self.data_estimators = data_estimators
        self.data_transformer = data_transformer

    def fit(self, graphs, targets):
        for i in range(len(self.data_estimators)):
            self.data_estimators[i] = self.data_estimators[i].fit(graphs, targets[:,i])
        return self

    def predict(self, graphs):
        preds = [self.data_estimators[i].predict(graphs).reshape(-1,1) for i in range(len(self.data_estimators))]
        preds = np.hstack(preds)
        return preds

    def save(self, filename='model.obj'):
        filehandler = open(filename, 'wb') 
        pickle.dump(self, filehandler)
        return self

    def load(self, filename='model.obj'):
        filehandler = open(filename, 'rb') 
        self = pickle.load(filehandler)
        return self


class DataKernel(object):
    def __init__(self, data_transformer=None, metric='linear'):
        self.data_transformer = data_transformer
        if metric in ['additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine']:
            self.metric = metric
        else: raise Exception('Unknown metric:'%metric)

    def fit(self, graphs):
        self.data_mtx = self.data_transformer.fit_transform(graphs)
        return self

    def predict(self, graphs):
        data_mtx = self.data_transformer.transform(graphs)
        mtx = pairwise_kernels(data_mtx, self.data_mtx, metric=self.metric)
        return mtx
