import copy
import numpy as np
import scipy as sp
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics.pairwise import paired_distances
import multiprocessing_on_dill as mp
from toolz import partition_all

def partition_instances_by_target(instances, targets, n_classes):
    if sp.sparse.issparse(instances):
        partitioned_instances_list = [sp.sparse.vstack([instance for instance, target in zip(instances, targets) if target == target_class]) for target_class in range(n_classes)]
    elif isinstance(instances, np.ndarray):
        partitioned_instances_list = [np.array([instance for instance, target in zip(instances, targets) if target == target_class]) for target_class in range(n_classes)]
    else:
        partitioned_instances_list = [[instance for instance, target in zip(instances, targets) if target == target_class] for target_class in range(n_classes)]
    return partitioned_instances_list


class GenerativeEstimator(object):
    def __init__(self, generator, centre=True, metric='euclidean'):
        self.generator = generator
        self.scaler = StandardScaler(with_mean=centre, with_std=centre)
        self.metric = metric
        self.n_components = self.generator.n_components
        
    def fit(self, instances, targets=None):
        centred_instances = self.scaler.fit_transform(instances)
        self.generator.fit(centred_instances)
        return self
    
    def encode(self, instances):
        centred_instances = self.scaler.transform(instances)
        latents = self.generator.transform(centred_instances)
        return latents
    
    def decode(self, latents):
        reconstructed_centred_instances = self.generator.inverse_transform(latents)
        reconstructed_instances = self.scaler.inverse_transform(reconstructed_centred_instances)
        return reconstructed_instances
    
    def reconstruct(self, instances):
        latents = self.encode(instances)
        instances = self.decode(latents)
        return instances

    def reconstruction_error(self, instances):
        reconstructions = self.reconstruct(instances)
        return paired_distances(instances, reconstructions, metric=self.metric)


class GenerativeClassifier(object):
    def __init__(self, generative_estimator, n_classes, embedder=None):
        self.embedder = embedder
        self.n_classes = n_classes
        self.generative_estimator = generative_estimator
        self.n_components = self.generative_estimator.n_components
    
    def fit(self, instances, targets):
        if self.embedder is not None: embeddings = self.embedder.fit_transform(instances, targets)
        else: embeddings = instances
        partitioned_instances_list = partition_instances_by_target(embeddings, targets, self.n_classes)
        self.generative_estimators = [copy.deepcopy(self.generative_estimator.fit(partitioned_instances)) for partitioned_instances in partitioned_instances_list]
        return self
    
    def predict(self, instances):
        if self.embedder is not None: embeddings = self.embedder.transform(instances)
        else: embeddings = instances
        reconstruction_errors = np.hstack([generative_estimator.reconstruction_error(embeddings).reshape(-1,1) for generative_estimator in self.generative_estimators])
        preds = np.argmin(reconstruction_errors, axis=1)
        return preds
    
    def transform(self, instances):
        if self.embedder is not None: embeddings = self.embedder.transform(instances)
        else: embeddings = instances
        return np.hstack([generative_estimator.encode(embeddings) for generative_estimator in self.generative_estimators])
    
    def fit_transform(self, instances, targets):
        return self.fit(instances, targets).transform(instances)

    def fit_predict(self, instances, targets):
        return self.fit(instances, targets).predict(instances)

def bootstrap(instances, targets, n_classes, instance_probs=None): 
    if instance_probs is None: return bootstrap_(instances, targets)
    class_partitioned_data = [[(instances[i], targets[i], instance_probs[i]) for i in range(len(targets)) if targets[i]== curr_target]   for curr_target in range(n_classes)]  
    all_instances = []
    all_targets = []
    for class_partitioned_data_ in class_partitioned_data:
        instances_, targets_, instance_probs_ = list(zip(*class_partitioned_data_))
        if sp.sparse.issparse(instances): instances_ = sp.sparse.vstack(instances_)
        instances_, targets_ = bootstrap_(instances_, targets_, instance_probs_)
        all_instances.append(instances_)
        all_targets.append(targets_)
    if sp.sparse.issparse(instances): all_instances = sp.sparse.vstack(all_instances)
    elif isinstance(instances, np.ndarray): all_instances = np.vstack(all_instances)
    else: all_instances = sum(all_instances,[])
    all_targets = np.hstack(all_targets)
    return all_instances, all_targets
    
def bootstrap_(instances, targets, instance_probs=None):
    if instance_probs is not None: instance_probs = instance_probs / np.sum(instance_probs)
    size = len(targets)
    idxs = np.random.choice(size, size=size, replace=True, p=instance_probs)
    instances_ = [instances[idx] for idx in idxs]
    if sp.sparse.issparse(instances): instances_ = sp.sparse.vstack(instances_)
    elif isinstance(instances, np.ndarray): instances_ = np.vstack(instances_)
    else: pass
    targets_ = np.array([targets[idx] for idx in idxs])
    return instances_, targets_

def compute_row_frequencies(data_mtx, max_val):
    all_freqs = []
    for row in data_mtx:
        freq = np.zeros(max_val).astype(int)
        unique, counts = np.unique(row, return_counts=True)
        freq[unique] = counts
        freq = freq/np.sum(freq)
        all_freqs.append(freq)
    all_freqs = np.vstack(all_freqs)
    return all_freqs


class BaggedGenerativeClassifier(object):
    def __init__(self, generative_classifier, n_classes, n_bootstraps=10, parallel=False):
        self.n_classes = n_classes
        self.discretizer = KBinsDiscretizer(n_bins=n_classes, encode='ordinal', strategy='uniform')
        self.generative_classifiers = [copy.deepcopy(generative_classifier) for i in range(n_bootstraps)]
        self.n_components = generative_classifier.n_components
        self.parallel=parallel
    
    def fit_single(self, generative_classifier, instances, targets, instance_probs=None):
        return copy.deepcopy(generative_classifier.fit(*bootstrap(instances, targets, self.n_classes, instance_probs))) 

    def fit_sequential(self, instances, targets, instance_probs=None):
        generative_classifiers = [self.fit_single(generative_classifier, instances, targets, instance_probs) for generative_classifier in self.generative_classifiers]
        return generative_classifiers

    def fit_parallel(self, instances, targets, instance_probs=None):
        #TODO: make single input list of triplets
        n_cpus = mp.cpu_count()
        if len(targets) < n_cpus: n_cpus = len(targets)
        batch_size = len(targets)//n_cpus
        instances_list = list(partition_all(batch_size, instances))
        pool = mp.Pool(n_cpus)
        results = pool.map(self.fit_sequential, instances_list)
        pool.close()
        generative_classifier = sum(results,[])
        return generative_classifier

    def fit(self, instances, targets, instance_probs=None):
        discretized_targets = self.discretizer.fit_transform(targets.reshape(-1, 1)).flatten()
        if self.parallel:
            self.generative_classifiers = self.fit_parallel(instances, discretized_targets, instance_probs) 
        else:
            self.generative_classifiers = self.fit_sequential(instances, discretized_targets, instance_probs)
        return self
    
    def predict(self, instances):
        all_preds = np.hstack([generative_classifier.predict(instances).reshape(-1,1) for generative_classifier in self.generative_classifiers])
        discretized_preds = sp.stats.mode(all_preds, axis=1)[0].flatten()
        preds = self.discretizer.inverse_transform(discretized_preds.reshape(-1, 1)).flatten()
        preds = np.rint(preds)
        return preds
    
    def predict_proba(self, instances):
        all_preds = np.hstack([generative_classifier.predict(instances).reshape(-1,1) for generative_classifier in self.generative_classifiers])
        probs = compute_row_frequencies(all_preds, max_val=self.n_classes)
        return probs
    
    def transform(self, instances):
        embeddings = np.hstack([generative_classifier.transform(instances) for generative_classifier in self.generative_classifiers])
        return embeddings
    
    def fit_transform(self, instances, targets):
        return self.fit(instances, targets).transform(instances)

    def fit_predict(self, instances, targets):
        return self.fit(instances, targets).predict(instances)


class BoostedBaggedGenerativeClassifier(object):
    def __init__(self, bagged_generative_classifier, n_boosting_iterations=2):
        self.epsilon = 1e-3
        self.n_boosting_iterations = n_boosting_iterations
        self.bagged_generative_classifier = bagged_generative_classifier
        self.n_components = self.bagged_generative_classifier.n_components
        
    def fit(self, instances, targets):
        self.bagged_generative_classifier.fit(instances, targets)
        self.bagged_generative_classifiers = []
        self.bagged_generative_classifiers.append(copy.deepcopy(self.bagged_generative_classifier))
        for it in range(self.n_boosting_iterations):
            class_probs = self.bagged_generative_classifier.predict_proba(instances)
            #get the prob of the true target
            #use 1-p to sample more frequently the instances where mistakes are happening
            instance_probs = 1 - class_probs[range(len(targets)),targets]
            instance_probs = instance_probs + self.epsilon
            instance_probs = instance_probs / np.sum(instance_probs)
            #update predictors 
            self.bagged_generative_classifier.fit(instances, targets, instance_probs)
            self.bagged_generative_classifiers.append(copy.deepcopy(self.bagged_generative_classifier))
        return self
    
    def predict(self, instances):
        all_preds = np.hstack([bagged_generative_classifier.predict(instances).reshape(-1,1) for bagged_generative_classifier in self.bagged_generative_classifiers])
        preds = sp.stats.mode(all_preds, axis=1)[0].flatten()
        return preds
    
    def predict_proba(self, instances):
        curr_preds = None
        for bagged_generative_classifier in self.bagged_generative_classifiers:
            preds = bagged_generative_classifier.predict_proba(instances)
            if curr_preds is None: curr_preds = preds
            else: curr_preds = curr_preds * preds
        probs = (preds.T/np.sum(preds, axis=1).T).T
        return probs
    
    def transform(self, instances):
        embeddings = np.hstack([bagged_generative_classifier.transform(instances) for bagged_generative_classifier in self.bagged_generative_classifiers])
        return embeddings
    
    def fit_transform(self, instances, targets):
        return self.fit(instances, targets).transform(instances)

    def fit_predict(self, instances, targets):
        return self.fit(instances, targets).predict(instances)
