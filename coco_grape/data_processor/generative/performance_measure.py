
import sys
from coco_grape.graph_vectorizer.paired_neighborhood_graph_vectorizer import PairedNeighborhoodGraphVectorizer
from scipy.stats.mstats import gmean
from scipy.stats.mstats import trimmed_mean, trimmed_std
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import train_test_split
from toolz import curry
from toolz import partition_all
import multiprocessing_on_dill as mp
import numpy as np
import random
import scipy as sp


def estimate_mean_and_std_from_quantiles(data):

    """
    Estimate the mean and std of a vector of numbers using quantiles (IQR-based method).
    
    Wan, Xiang, Wenqian Wang, Jiming Liu, and Tiejun Tong. 2014. “Estimating the Sample Mean and Standard Deviation from the Sample Size, Median, Range And/or Interquartile Range.” BMC Medical Research Methodology 14 (135). doi:10.1186/1471-2288-14-135. 

    Parameters:
    data (list or numpy array): A list or array of numerical values.
    
    Returns:
    estimated_mean, estimated_std (float): The estimated mean and std of the data.
    """
    if len(data)==1: 
        return data[0], 0
        
    # Calculate the 25th and 75th percentiles (Q1 and Q3)
    q1 = np.percentile(data, 25)
    m = np.percentile(data, 50)
    q3 = np.percentile(data, 75)

    # Calculate the interquartile range (IQR)
    iqr = q3 - q1

    # Estimate variance using the IQR
    estimated_std = (iqr / 1.34898)  

    estimated_mean = np.mean([q1,m,q3])

    return estimated_mean ,estimated_std

def bootstrap(instances, targets, seed=None):
    if seed is not None: np.random.seed(seed)
    size = len(instances)
    idxs = np.random.choice(size, size=size, replace=True)
    instances_ = instances[idxs]
    targets_ = targets[idxs]
    return instances_, targets_

def resample(instances, targets, size, seed=42):
    if seed is not None: np.random.seed(seed)
    rng = np.random.default_rng(seed)
    if len(instances)-size < 3:
        return instances, targets
    if size > len(instances): #sample with replacement
        idxs = np.random.choice(len(instances), size=size, replace=True)
    else: #sample without replacement 
        idxs = np.random.choice(len(instances), size=size, replace=False)
    resampled_instances = instances[idxs]
    resampled_targets = np.array(targets)[idxs]
    return resampled_instances, resampled_targets

def class_equalize(instances, targets):
    class_instances = [np.vstack([instance for instance, target in zip(instances, targets) if target == curr_target]) for curr_target in sorted(set(targets))]
    max_size = max(len(class_instances) for class_instances in class_instances)
    resampled_class_instances_list = []
    resampled_targets_list = []
    for target, class_instances in enumerate(class_instances):
        if len(class_instances) < max_size:
            idxs = np.random.choice(len(class_instances), size=max_size, replace=True)
            resampled_class_instances = class_instances[idxs]
        else:
            resampled_class_instances = class_instances
        resampled_class_instances_list.append(resampled_class_instances)
        resampled_targets_list.append([target]*len(resampled_class_instances))
    resampled_instances = np.vstack(resampled_class_instances_list)
    resampled_targets = np.array(sum(resampled_targets_list, []))
    return resampled_instances, resampled_targets

def robust_statistics(vals, n_elements_to_trim=1):
    l = n_elements_to_trim/len(vals)
    mean = trimmed_mean(vals, limits=(l,l))
    std = trimmed_std(vals, limits=(l,l))
    return mean, std

def similarity_entropy(src_instances, dst_instances, metric = 'linear', n_neighbors=10):
    X = np.vstack([src_instances, dst_instances])
    targets = np.asarray([1]*src_instances.shape[0]+[0]*dst_instances.shape[0])
    scale = sp.stats.entropy(np.bincount(targets, minlength=2)/len(targets), base=2)
    K = pairwise_kernels(X, metric=metric)
    knbs = np.argsort(-K, axis=1)[:,:n_neighbors]
    target_knbs = targets[knbs]
    entropies = [sp.stats.entropy(np.bincount(x, minlength=2)/len(x), base=2)/scale for x in target_knbs]
    score = np.mean(entropies)
    return score

def estimate_instance_set_similarity(src_instances, src_targets, dst_instances, dst_targets, metric='cosine', n_neighbors=2):
    similarity_list = []
    target_classes = sorted(set(src_targets))
    for target_class in target_classes:
        loc_src_instances = np.vstack([src_instance for src_instance, src_target in zip(src_instances, src_targets) if src_target == target_class])
        loc_dst_instances = np.vstack([dst_instance for dst_instance, dst_target in zip(dst_instances, dst_targets) if dst_target == target_class])
        similarity = similarity_entropy(loc_src_instances, loc_dst_instances, metric=metric, n_neighbors=n_neighbors)
        similarity_list.append(similarity)
    similarity = gmean(similarity_list)
    return similarity

def adjusted_balanced_accuracy_score(test_targets, predicted_targets):
    return balanced_accuracy_score(test_targets, predicted_targets, adjusted=True)

def make_adjusted_score_func(score):
    def score_func(test_targets, preds):
        random_targets = np.random.choice(test_targets, size=len(test_targets), replace=False)
        random_score = score(test_targets, random_targets)
        perfect_score = score(test_targets, test_targets)
        adjusted_score = (score(test_targets, preds) - random_score)/(perfect_score - random_score)
        adjusted_score = max(0, adjusted_score)
        adjusted_score = min(1, adjusted_score)
        return adjusted_score
    return score_func

@curry
def compute_estimated_predictive_performance_score_func(train_instances, train_targets, test_instances=None, test_targets=None, data_estimator=None, discriminative_performance_func=None, n_rep=10):
    scores = []
    for it in range(n_rep):
        X_train, _, y_train, _ = train_test_split(train_instances, train_targets, stratify=train_targets, train_size=.7)
        predicted_targets = data_estimator.fit(X_train, y_train).predict(test_instances)
        score = discriminative_performance_func(test_targets, predicted_targets)
        scores.append(score)
    estimated_predictive_performance_score, estimated_predictive_performance_score_std = robust_statistics(scores, n_elements_to_trim=1)
    return estimated_predictive_performance_score, estimated_predictive_performance_score_std

def discriminative_generative_predictive_performances(data_estimator, generated_instances, generated_targets, real_train_instances, real_train_targets, real_reference_instances, real_reference_targets, test_instances, test_targets, discriminative_performance_func=None, n_rep=10):
    compute_estimated_predictive_performance_score = compute_estimated_predictive_performance_score_func(test_instances=test_instances, test_targets=test_targets, data_estimator=data_estimator, discriminative_performance_func=discriminative_performance_func, n_rep=n_rep)

    predictive_performance_with_real_train, predictive_performance_with_real_train_std = compute_estimated_predictive_performance_score(real_train_instances, real_train_targets)
    #ensure that num of generated_instances is the same as num of real_train_instances
    resampled_generated_instances, resampled_generated_targets = resample(generated_instances, generated_targets, len(real_train_instances))
    predictive_performance_with_generated, predictive_performance_with_generated_std = compute_estimated_predictive_performance_score(resampled_generated_instances, resampled_generated_targets)
    
    predictive_performance_with_real_train_and_reference, predictive_performance_with_real_train_and_reference_std = compute_estimated_predictive_performance_score(np.vstack([real_train_instances,real_reference_instances]), np.hstack([real_train_targets,real_reference_targets]))
    #ensure that num of generated_instances is the same as num of real_reference_instances
    resampled_generated_instances, resampled_generated_targets = resample(generated_instances, generated_targets, len(real_reference_instances))
    predictive_performance_with_real_train_and_generated, predictive_performance_with_real_train_and_generated_std = compute_estimated_predictive_performance_score(np.vstack([real_train_instances,resampled_generated_instances]), np.hstack([real_train_targets,resampled_generated_targets]))
    
    real_gen_instances = np.vstack([real_train_instances,resampled_generated_instances])
    real_gen_target = np.array([1]*len(real_train_instances)+[0]*len(resampled_generated_instances))
    
    predictive_performance_discriminate_real_from_generated_list = []
    for it in range(n_rep):
        train_real_gen_instances, test_real_gen_instances, train_real_gen_targets, test_real_gen_targets = train_test_split(real_gen_instances, real_gen_target, train_size=.7)
        train_real_gen_instances, train_real_gen_targets = class_equalize(train_real_gen_instances, train_real_gen_targets)
        compute_real_gen_estimated_predictive_performance_score = compute_estimated_predictive_performance_score_func(test_instances=test_real_gen_instances, test_targets=test_real_gen_targets, data_estimator=data_estimator, discriminative_performance_func=discriminative_performance_func, n_rep=n_rep)
        predictive_performance_discriminate_real_from_generated, predictive_performance_discriminate_real_from_generated_std = compute_real_gen_estimated_predictive_performance_score(train_real_gen_instances, train_real_gen_targets)
        predictive_performance_discriminate_real_from_generated_list.append(predictive_performance_discriminate_real_from_generated)
    predictive_performance_discriminate_real_from_generated = np.mean(predictive_performance_discriminate_real_from_generated_list)
    predictive_performances = [predictive_performance_with_real_train, predictive_performance_with_generated, predictive_performance_with_real_train_and_reference, predictive_performance_with_real_train_and_generated, predictive_performance_discriminate_real_from_generated]
    return predictive_performances

def std_of_ratio_of_normal_distributions(mu1, std1, mu2, std2):
    return np.sqrt(mu1**2 / mu2**2 * (std1**2 / mu1**2 + std2**2 / mu2**2))

def std_of_product_of_normal_distributions(mu1, std1, mu2, std2):
    return np.sqrt((std1**2 + mu1**2) * (std2**2 + mu2**2) - mu1**2 * mu2**2)

def std_of_sum_of_normal_distributions(mu1, std1, mu2, std2):
    return np.sqrt(std1**2 + std2**2)

def std_of_difference_of_normal_distributions(mu1, std1, mu2, std2):
    return std_of_sum_of_normal_distributions(mu1, std1, mu2, std2)

def std_of_average_of_normal_distributions(mu1, std1, mu2, std2):
    return std_of_sum_of_normal_distributions(mu1, std1, mu2, std2) / 2


class DiscriminativeGenerativeQualityScorer(object):

    def __init__(self, 
        data_estimator=None, 
        discriminative_performance_func=f1_score, 
        n_rep_estimator=3, 
        n_neighbors=3, 
        n_elements_to_trim=1, 
        metric='cosine', 
        verbose=True, 
        parallel=False, 
        n_cpus=None, 
        make_adjusted_score=True, 
        enforce_positive_definite=True,
        enforce_maximum=True):
        self.data_estimator = data_estimator
        if make_adjusted_score: self.effective_discriminative_performance_func = make_adjusted_score_func(discriminative_performance_func)
        else: self.effective_discriminative_performance_func = discriminative_performance_func
        self.n_rep_estimator = n_rep_estimator
        self.n_neighbors = n_neighbors
        self.n_elements_to_trim = n_elements_to_trim
        self.metric = metric
        self.verbose = verbose
        self.parallel = parallel
        self.n_cpus = n_cpus
        self.enforce_positive_definite = enforce_positive_definite
        self.enforce_maximum = enforce_maximum

        self.real_train_instances_list = []
        self.real_train_targets_list = []
        self.real_reference_instances_list = []
        self.real_reference_targets_list = []
        self.test_instances_list = []
        self.test_targets_list = []
        self.generated_instances_list =[]
        self.generated_targets_list = []
        self.similarity_generated_vs_real_train_avg = None
        self.similarity_generated_vs_real_train_std = None
        self.predictive_performance_with_real_train_avg = None
        self.predictive_performance_with_real_train_std = None
        self.predictive_performance_with_generated_avg = None
        self.predictive_performance_with_generated_std = None
        self.predictive_performance_with_real_train_and_reference_avg = None
        self.predictive_performance_with_real_train_and_reference_std = None
        self.predictive_performance_with_real_train_and_generated_avg = None
        self.predictive_performance_with_real_train_and_generated_std = None
        self.predictive_performance_discriminate_real_from_generated_avg = None
        self.predictive_performance_discriminate_real_from_generated_std = None

    def input_train(self, data_mtx, targets):
        self.real_train_instances_list.append(data_mtx)
        self.real_train_targets_list.append(targets)
        return self 
    
    def input_reference(self, data_mtx, targets):
        self.real_reference_instances_list.append(data_mtx)
        self.real_reference_targets_list.append(targets)
        return self 

    def input_test(self, data_mtx, targets):
        self.test_instances_list.append(data_mtx)
        self.test_targets_list.append(targets)
        return self 

    def input_generated(self, data_mtx, targets):
        self.generated_instances_list.append(data_mtx)
        self.generated_targets_list.append(targets)
        return self 

    def input_data(self, generated_instances, generated_targets, real_train_instances, real_train_targets, real_reference_instances, real_reference_targets, test_instances, test_targets):
        self.generated_instances_list.append(generated_instances)
        self.generated_targets_list.append(generated_targets)
        self.real_train_instances_list.append(real_train_instances)
        self.real_train_targets_list.append(real_train_targets)
        self.real_reference_instances_list.append(real_reference_instances)
        self.real_reference_targets_list.append(real_reference_targets)
        self.test_instances_list.append(test_instances)
        self.test_targets_list.append(test_targets)
        return self

    def resample_single(self, instances_list, targets_list, n_iterations=10, use_replacement=False, fraction=0.7, seed=42):  
        rng = np.random.default_rng(seed)        
        working_targets_list = []
        working_instances_list=[]
        for instances, targets in zip(instances_list, targets_list):
            for it in range(n_iterations):
                local_seed = int(rng.integers(1e9))
                if use_replacement:
                    working_instances, working_targets = bootstrap(instances, targets, seed=local_seed)
                else:
                    size = int(len(targets) * fraction)
                    working_instances, working_targets = resample(instances, targets, size, seed=local_seed)

                working_instances = np.array(working_instances)
                working_targets = np.array(working_targets)
                working_instances_list.append(working_instances)
                working_targets_list.append(working_targets)
        return working_instances_list, working_targets_list

    def resample(self, n_iterations=10, use_resampling=False, use_replacement=False, fraction=0.7,seed=42):
        rng = np.random.default_rng(seed)
        if use_resampling:
            self.generated_instances_list, self.generated_targets_list = self.resample_single(self.generated_instances_list, self.generated_targets_list, n_iterations=n_iterations, use_replacement=use_replacement, fraction=fraction,seed=int(rng.integers(1e9)))
            self.real_train_instances_list, self.real_train_targets_list = self.resample_single(self.real_train_instances_list, self.real_train_targets_list, n_iterations=n_iterations, use_replacement=use_replacement, fraction=fraction,seed=int(rng.integers(1e9)))
            self.real_reference_instances_list, self.real_reference_targets_list = self.resample_single(self.real_reference_instances_list, self.real_reference_targets_list, n_iterations=n_iterations, use_replacement=use_replacement, fraction=fraction,seed=int(rng.integers(1e9)))
            self.test_instances_list, self.test_targets_list = self.resample_single(self.test_instances_list, self.test_targets_list, n_iterations=n_iterations, use_replacement=use_replacement, fraction=fraction,seed=int(rng.integers(1e9)))
        else:
            self.generated_instances_list, self.generated_targets_list = self.resample_single(self.generated_instances_list, self.generated_targets_list, n_iterations=n_iterations, use_replacement=False, fraction=1,seed=int(rng.integers(1e9)))
            self.real_train_instances_list, self.real_train_targets_list = self.resample_single(self.real_train_instances_list, self.real_train_targets_list, n_iterations=n_iterations, use_replacement=False, fraction=1,seed=int(rng.integers(1e9)))
            self.real_reference_instances_list, self.real_reference_targets_list = self.resample_single(self.real_reference_instances_list, self.real_reference_targets_list, n_iterations=n_iterations, use_replacement=False, fraction=1,seed=int(rng.integers(1e9)))
            self.test_instances_list, self.test_targets_list = self.resample_single(self.test_instances_list, self.test_targets_list, n_iterations=n_iterations, use_replacement=False, fraction=1,seed=int(rng.integers(1e9)))

    def _compute_performance_indicators_single(self, generated_instances, generated_targets, real_train_instances, real_train_targets, real_reference_instances, real_reference_targets, test_instances, test_targets):
        similarity_generated_vs_real_train = estimate_instance_set_similarity(generated_instances, generated_targets, real_train_instances, real_train_targets, metric=self.metric, n_neighbors=self.n_neighbors)
        predictive_performances = discriminative_generative_predictive_performances(self.data_estimator, generated_instances, generated_targets, real_train_instances, real_train_targets, real_reference_instances, real_reference_targets, test_instances, test_targets, discriminative_performance_func=self.effective_discriminative_performance_func, n_rep=self.n_rep_estimator)
        predictive_performance_with_real_train, predictive_performance_with_generated, predictive_performance_with_real_train_and_reference, predictive_performance_with_real_train_and_generated, predictive_performance_discriminate_real_from_generated = predictive_performances
        if self.verbose: print('\t real:%.2f  gen:%.2f  real_and_ref:%.2f  real_and_gen:%.2f  real_vs_gen:%.2f  similarity:%.2f'%(predictive_performance_with_real_train, predictive_performance_with_generated, predictive_performance_with_real_train_and_reference, predictive_performance_with_real_train_and_generated, predictive_performance_discriminate_real_from_generated, similarity_generated_vs_real_train))
        return similarity_generated_vs_real_train, predictive_performance_with_real_train, predictive_performance_with_generated, predictive_performance_with_real_train_and_reference, predictive_performance_with_real_train_and_generated, predictive_performance_discriminate_real_from_generated
    
    def _compute_performance_indicators(self, data_list):
            predictive_performances_list = [self._compute_performance_indicators_single(*data) for data in data_list]
            return predictive_performances_list

    def _make_data_list(self): 
        data_list = []
        for generated_instances, generated_targets, real_train_instances, real_train_targets, real_reference_instances, real_reference_targets, test_instances, test_targets in zip(self.generated_instances_list, self.generated_targets_list, self.real_train_instances_list, self.real_train_targets_list, self.real_reference_instances_list, self.real_reference_targets_list, self.test_instances_list, self.test_targets_list):
            data = generated_instances, generated_targets, real_train_instances, real_train_targets, real_reference_instances, real_reference_targets, test_instances, test_targets
            data_list.append(data)
        return data_list
    def _compute_performance_indicators_with_seed(self, data_list, seed):
        np.random.seed(seed)  # NEW: control randomness inside subprocess
        random.seed(seed)
        return self._compute_performance_indicators(data_list)

    def compute_performance_indicators(self, seed=42):
        data_list = self._make_data_list()
        if self.parallel is False:
            # serial mode (already correct behavior)
            predictive_performances_list = self._compute_performance_indicators(data_list)
        else:
            # NEW: split data + seed per batch
            if self.n_cpus is None:
                self.n_cpus = mp.cpu_count()
            if len(data_list) < self.n_cpus:
                self.n_cpus = len(data_list)
            batch_size = len(data_list) // self.n_cpus
            batched_data_list = list(partition_all(batch_size, data_list))

            # NEW: assign fixed deterministic seeds
            rng = np.random.default_rng(seed)
            batch_seeds = [int(rng.integers(1e9)) for _ in range(len(batched_data_list))]

            # NEW: pass seed explicitly per worker
            with mp.Pool(self.n_cpus) as pool:
                results = pool.starmap(
                    self._compute_performance_indicators_with_seed,
                    zip(batched_data_list, batch_seeds)
                )
            predictive_performances_list = sum(results, [])
        self._store_predictive_performances_list(predictive_performances_list)
        return self


    def _store_predictive_performances_list(self, predictive_performances_list):
        self.similarity_generated_vs_real_train_list = []
        self.predictive_performance_with_real_train_list = []
        self.predictive_performance_with_generated_list = []
        self.predictive_performance_with_real_train_and_reference_list = []
        self.predictive_performance_with_real_train_and_generated_list = []
        self.predictive_performance_discriminate_real_from_generated_list = []
        for predictive_performances in predictive_performances_list:
            similarity_generated_vs_real_train, predictive_performance_with_real_train, predictive_performance_with_generated, predictive_performance_with_real_train_and_reference, predictive_performance_with_real_train_and_generated, predictive_performance_discriminate_real_from_generated = predictive_performances
            self.similarity_generated_vs_real_train_list.append(similarity_generated_vs_real_train)
            self.predictive_performance_with_real_train_list.append(predictive_performance_with_real_train)
            self.predictive_performance_with_generated_list.append(predictive_performance_with_generated)
            self.predictive_performance_with_real_train_and_reference_list.append(predictive_performance_with_real_train_and_reference)
            self.predictive_performance_with_real_train_and_generated_list.append(predictive_performance_with_real_train_and_generated)
            self.predictive_performance_discriminate_real_from_generated_list.append(predictive_performance_discriminate_real_from_generated)
        self.similarity_generated_vs_real_train_avg, self.similarity_generated_vs_real_train_std = robust_statistics(self.similarity_generated_vs_real_train_list, n_elements_to_trim=self.n_elements_to_trim)
        self.predictive_performance_with_real_train_avg, self.predictive_performance_with_real_train_std = robust_statistics(self.predictive_performance_with_real_train_list, n_elements_to_trim=self.n_elements_to_trim)
        self.predictive_performance_with_generated_avg, self.predictive_performance_with_generated_std = robust_statistics(self.predictive_performance_with_generated_list, n_elements_to_trim=self.n_elements_to_trim)
        self.predictive_performance_with_real_train_and_reference_avg, self.predictive_performance_with_real_train_and_reference_std = robust_statistics(self.predictive_performance_with_real_train_and_reference_list, n_elements_to_trim=self.n_elements_to_trim)
        self.predictive_performance_with_real_train_and_generated_avg, self.predictive_performance_with_real_train_and_generated_std = robust_statistics(self.predictive_performance_with_real_train_and_generated_list, n_elements_to_trim=self.n_elements_to_trim)
        self.predictive_performance_discriminate_real_from_generated_avg, self.predictive_performance_discriminate_real_from_generated_std = robust_statistics(self.predictive_performance_discriminate_real_from_generated_list, n_elements_to_trim=self.n_elements_to_trim)

    def feasibility_condition_enforcement(self, score):
        score = np.nan_to_num(score, posinf=0, neginf=0)
        if self.enforce_positive_definite: score = max(0, score)
        if self.enforce_maximum: score = min(1, score)
        return score

    def post_process_with_feasibility_enforcement(self, score_list):
        score, score_std = estimate_mean_and_std_from_quantiles(score_list)
        score = self.feasibility_condition_enforcement(score)
        score_std = self.feasibility_condition_enforcement(score_std)
        return score, score_std
        
    def get_similarity_list(self):
        return self.similarity_generated_vs_real_train_list

    def similarity(self):
        #similarity: how similar are the distribution of neighbors distances when we consider the neighbors in the real set to a generated instance and when we consider the neighbors in the generated set to a real instance
        #this is a failsafe quality measure that does not depend on the discriminator capacity
        similarity_list = self.get_similarity_list()
        similarity_score, similarity_score_std = self.post_process_with_feasibility_enforcement(similarity_list)
        return similarity_score, similarity_score_std
        
    def get_quality_list(self):
        quality_list = [predictive_performance_with_generated / predictive_performance_with_real_train for predictive_performance_with_generated, predictive_performance_with_real_train in zip(self.predictive_performance_with_generated_list, self.predictive_performance_with_real_train_list)]
        return quality_list

    def quality(self):
        #quality: training on generated data should yield comparable predictive performance on a test set as when training on original data
        quality_list = self.get_quality_list() 
        quality_score, quality_score_std = self.post_process_with_feasibility_enforcement(quality_list)
        return quality_score, quality_score_std
    
    def get_utility_list(self):
        eps = 1e-6
        utility_numerator_list = [max(eps, predictive_performance_with_real_train_and_generated - predictive_performance_with_real_train) for predictive_performance_with_real_train_and_generated, predictive_performance_with_real_train in zip(self.predictive_performance_with_real_train_and_generated_list, self.predictive_performance_with_real_train_list)]
        utility_denominator_list = [max(eps, predictive_performance_with_real_train_and_reference - predictive_performance_with_real_train) for predictive_performance_with_real_train_and_reference, predictive_performance_with_real_train in zip(self.predictive_performance_with_real_train_and_reference_list, self.predictive_performance_with_real_train_list)]
        utility_list = [utility_numerator/utility_denominator for utility_numerator, utility_denominator in zip(utility_numerator_list, utility_denominator_list)]
        return utility_list

    def utility(self):
        #utility: training on original data + generated data should yield comparable increase in predictive performance w.r.t. original data on test as original data + data from same distribution
        utility_list = self.get_utility_list()
        utility_score, utility_score_std = self.post_process_with_feasibility_enforcement(utility_list)
        return utility_score, utility_score_std
        
    def get_indistinguishability_list(self):
        indistinguishability_list = [1 - score for score in self.predictive_performance_discriminate_real_from_generated_list]
        return indistinguishability_list

    def indistinguishability(self):
        #indistinguishability: it should be difficult to accurately discriminate between original data and generated data
        indistinguishability_list = self.get_indistinguishability_list()
        indistinguishability_score, indistinguishability_score_std = self.post_process_with_feasibility_enforcement(indistinguishability_list)
        return indistinguishability_score, indistinguishability_score_std

    def get_exchangeability_list(self):
        quality_list = self.get_quality_list() 
        utility_list = self.get_utility_list()
        indistinguishability_list = self.get_indistinguishability_list()
        similarity_list = self.get_similarity_list()
        exchangeability_list = [np.mean([quality, utility])*np.mean([indistinguishability, similarity]) for quality, utility, indistinguishability, similarity in zip(quality_list, utility_list, indistinguishability_list, similarity_list)]
        return exchangeability_list

    def exchangeability(self):
        exchangeability_list = self.get_exchangeability_list()
        exchangeability_score, exchangeability_score_std = self.post_process_with_feasibility_enforcement(exchangeability_list)
        return exchangeability_score, exchangeability_score_std

    def get_creativity_list(self):
        quality_list = self.get_quality_list() 
        utility_list = self.get_utility_list()
        indistinguishability_list = self.get_indistinguishability_list()
        similarity_list = self.get_similarity_list()
        creativity_list = [np.mean([quality, utility])/(1+np.mean([indistinguishability, similarity])) for quality, utility, indistinguishability, similarity in zip(quality_list, utility_list, indistinguishability_list, similarity_list)]
        return creativity_list

    def creativity(self):
        creativity_list = self.get_creativity_list()
        creativity_score, creativity_score_std = self.post_process_with_feasibility_enforcement(creativity_list)
        return creativity_score, creativity_score_std
        
    def score(self):
        quality, quality_std, utility, utility_std,  indistinguishability, indistinguishability_std, similarity, similarity_std = *self.quality(), *self.utility(), *self.indistinguishability(), *self.similarity()
        exchangeability_score, exchangeability_score_std = self.exchangeability()
        creativity_score, creativity_score_std = self.creativity()
        return exchangeability_score, exchangeability_score_std, creativity_score, creativity_score_std, quality, quality_std, utility, utility_std, indistinguishability, indistinguishability_std, similarity, similarity_std

    def scores(self):
        quality, quality_std, utility, utility_std,  indistinguishability, indistinguishability_std, similarity, similarity_std = *self.quality(), *self.utility(), *self.indistinguishability(), *self.similarity()
        scores = np.array([quality, utility, indistinguishability, similarity]) 
        scores_std = np.array([quality_std, utility_std, indistinguishability_std, similarity_std]) 
        exchangeability_score, exchangeability_score_std = self.exchangeability()
        creativity_score, creativity_score_std = self.creativity()
        predictive_performances = [self.predictive_performance_with_real_train_avg, self.predictive_performance_with_generated_avg, self.predictive_performance_with_real_train_and_reference_avg, self.predictive_performance_with_real_train_and_generated_avg, self.predictive_performance_discriminate_real_from_generated_avg]
        predictive_performances_std = [self.predictive_performance_with_real_train_std, self.predictive_performance_with_generated_std, self.predictive_performance_with_real_train_and_reference_std, self.predictive_performance_with_real_train_and_generated_std, self.predictive_performance_discriminate_real_from_generated_std]
        # if self.verbose:
        print_score(exchangeability_score, creativity_score, scores, predictive_performances, exchangeability_score_std, creativity_score_std, scores_std, predictive_performances_std)
        return exchangeability_score, creativity_score, scores, predictive_performances, exchangeability_score_std, creativity_score_std, scores_std, predictive_performances_std


def print_score(exchangeability_score, creativity_score, scores, predictive_performances, exchangeability_score_std, creativity_score_std, scores_std, predictive_performances_std):
    quality, utility, indistinguishability, similarity = scores
    quality_std, utility_std, indistinguishability_std, similarity_std = scores_std
    predictive_performance_with_real_train_avg, predictive_performance_with_generated_avg, predictive_performance_with_real_train_and_reference_avg, predictive_performance_with_real_train_and_generated_avg, predictive_performance_discriminate_real_from_generated_avg = predictive_performances
    predictive_performance_with_real_train_std, predictive_performance_with_generated_std, predictive_performance_with_real_train_and_reference_std, predictive_performance_with_real_train_and_generated_std, predictive_performance_discriminate_real_from_generated_std = predictive_performances_std
    print('real: %.2f+-%.2f   generated: %.2f+-%.2f   real+reference: %.2f+-%.2f   real+generated: %.2f+-%.2f   real_vs_generated: %.2f+-%.2f'%(predictive_performance_with_real_train_avg, predictive_performance_with_real_train_std, predictive_performance_with_generated_avg, predictive_performance_with_generated_std, predictive_performance_with_real_train_and_reference_avg, predictive_performance_with_real_train_and_reference_std, predictive_performance_with_real_train_and_generated_avg, predictive_performance_with_real_train_and_generated_std, predictive_performance_discriminate_real_from_generated_avg, predictive_performance_discriminate_real_from_generated_std))
    print('quality: %.2f+-%.2f   utility: %.2f+-%.2f   indistinguishability: %.2f+-%.2f   similarity: %.2f+-%.2f'%(quality, quality_std, utility, utility_std, indistinguishability, indistinguishability_std, similarity, similarity_std))
    print('exploitable_exchangeability: %.2f+-%.2f   exploitable_creativity: %.2f+-%.2f'%(exchangeability_score, exchangeability_score_std, creativity_score, creativity_score_std))


def concrete_discriminative_generative_quality_score(generated_instances, generated_targets, real_train_instances, real_train_targets, real_reference_instances, real_reference_targets, test_instances, test_targets, n_iterations=10, use_resampling=False, use_replacement=False, fraction=0.7, data_estimator=ExtraTreesClassifier(n_estimators=100, n_jobs=-1), discriminative_performance_func=adjusted_balanced_accuracy_score, verbose=1, parallel=True):
    #verbose=0 no output; verbose=1 only print_score; verbose=2 print_score and print each iteration
    scorer = DiscriminativeGenerativeQualityScorer(
        data_estimator=data_estimator, 
        discriminative_performance_func=discriminative_performance_func, 
        n_rep_estimator=3, 
        n_neighbors=3, 
        n_elements_to_trim=1, 
        metric='cosine', 
        verbose=verbose>=2, 
        parallel=parallel, 
        n_cpus=None, 
        make_adjusted_score=False, 
        enforce_positive_definite=True)
    scorer.input_data(generated_instances, generated_targets, real_train_instances, real_train_targets, real_reference_instances, real_reference_targets, test_instances, test_targets)
    scorer.resample(n_iterations=n_iterations, use_resampling=use_resampling, use_replacement=use_replacement, fraction=fraction)
    scorer.compute_performance_indicators()
    res = scorer.scores()
    exchangeability_score, creativity_score, scores, predictive_performances, exchangeability_score_std, creativity_score_std, scores_std, predictive_performances_std = res
    return exchangeability_score, creativity_score, scores, predictive_performances, exchangeability_score_std, creativity_score_std, scores_std, predictive_performances_std


def concrete_graph_discriminative_generative_quality_score(generated_graphs, generated_targets, real_train_graphs, real_train_targets, real_reference_graphs, real_reference_targets, test_graphs, test_targets, n_iterations=10, verbose=1, parallel=True):
    vectorizer = PairedNeighborhoodGraphVectorizer(radius=2, distance=4, nbits=12)
    generated_instances = vectorizer.fit_transform(generated_graphs)
    real_train_instances = vectorizer.fit_transform(real_train_graphs)
    real_reference_instances = vectorizer.fit_transform(real_reference_graphs)
    test_instances = vectorizer.fit_transform(test_graphs)
    return concrete_discriminative_generative_quality_score(generated_instances, generated_targets, real_train_instances, real_train_targets, real_reference_instances, real_reference_targets, test_instances, test_targets, n_iterations=n_iterations, use_replacement=False, fraction=0.7, data_estimator=ExtraTreesClassifier(n_estimators=100, n_jobs=-1), discriminative_performance_func=f1_score, verbose=verbose, parallel=parallel)


def score_rank(scores_list, n_iter=1000, std_correction_factor=.1):

    def sample_scores(locs, stds): 
        return [np.random.normal(loc=loc, scale=std) for loc, std in zip(locs, stds)]

    def compute_is_dominated_mtx(score_mtx):
        #compute if one model is dominated by another model
        #input: each row is a vector of (sampled) scores for one model
        #output: is_dominated_mtx is nxn matrix with 1 in cell i,j if model i is dominated by model j 
        n_models = score_mtx.shape[0]
        is_dominated_mtx = np.zeros((n_models,n_models))
        for i in range(n_models):
            for j in range(n_models):
                is_dominated_mtx[i,j] = int(np.all(score_mtx[i]<=score_mtx[j])) #compare i with j: mark 1 when i is less on *all* objectives
        return is_dominated_mtx

    n_models = len(scores_list)
    is_dominated_mtx = np.zeros((n_models,n_models))
    for it in range(n_iter):
        score_mtx = np.array([sample_scores(locs, np.array(stds)*std_correction_factor) for locs, stds in scores_list])
        is_dominated_mtx += compute_is_dominated_mtx(score_mtx) #add 1 in cell i,j if model i is dominated by model j 
    is_dominated_counts_mtx = np.sum(is_dominated_mtx, axis=1) #cumulative number of times that model i is dominated by any other model
    ranks = sp.stats.rankdata(is_dominated_counts_mtx)
    return ranks

def discriminative_generative_quality_score_rank(scores_list, n_iter=1000, std_correction_factor=.1):
    #input: scores_list format: each row is a 2-tuple for a single model: (quality, utility, indistinguishability, similarity), (quality_std, utility_std, indistinguishability_std, similarity_std)
    #n_iter is the number of samples sampled from the mean+-std of each score
    #output: array with rank order by quality (1=best quality) of each model in same order as in scores_list
    return score_rank(scores_list, n_iter=n_iter, std_correction_factor=std_correction_factor)

def exploitable_exchangeability_and_exploitable_creativity_rank(scores_list, n_iter=1000, std_correction_factor=.1):
    #input: scores_list format: each row is a 2-tuple for a single model: (exploitable_exchangeability, exploitable_creativity), (exploitable_exchangeability_std, exploitable_creativity_std)
    #n_iter is the number of samples sampled from the mean+-std of each score
    #output: array with rank order by quality (1=best quality) of each model in same order as in scores_list
    return score_rank(scores_list, n_iter=n_iter, std_correction_factor=std_correction_factor)

