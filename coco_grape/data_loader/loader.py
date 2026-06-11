import numpy as np

class SupervisedDataSetLoader(object):
    def __init__(self, 
        load_func=None, 
        size=None, 
        use_targets_list=None, 
        use_equalized=False, 
        use_multiclass_to_binary=False, 
        use_regression_to_binary=False, 
        regression_to_binary_threshold=None):
        self.load_func = load_func
        self.size = size
        self.use_targets_list = use_targets_list
        self.use_equalized = use_equalized
        self.use_multiclass_to_binary = use_multiclass_to_binary
        self.use_regression_to_binary = use_regression_to_binary
        self.regression_to_binary_threshold = regression_to_binary_threshold
        
    def resize(self, data, targets, size):
        if isinstance(data, np.ndarray): data_is_numpy = True
        else: data_is_numpy = False
        if isinstance(targets, np.ndarray): target_is_numpy = True
        else: target_is_numpy = False
        idxs = np.random.choice(len(targets), size=size, replace=False)
        data = [data[idx] for idx in idxs]
        if data_is_numpy: data = np.asarray(data)
        targets = [targets[idx] for idx in idxs]
        if target_is_numpy: targets = np.asarray(targets)
        return data, targets
    
    def equalize(self, data, targets):
        if isinstance(data, np.ndarray): data_is_numpy = True
        else: data_is_numpy = False
        if isinstance(targets, np.ndarray): target_is_numpy = True
        else: target_is_numpy = False
        target_values = list(sorted(set(targets)))
        idxs_list = [[idx for idx in range(len(targets)) if targets[idx] == target_value]  for target_value in target_values]
        min_size = min(len(idxs) for idxs in idxs_list)
        idxs_list = [np.random.choice(idxs, size=min_size, replace=False) for idxs in idxs_list]
        data = sum([[data[idx] for idx in idxs] for idxs in idxs_list], [])
        if data_is_numpy: data = np.asarray(data)
        targets = sum([[targets[idx] for idx in idxs] for idxs in idxs_list], [])
        if target_is_numpy: targets = np.asarray(targets)            
        return data, targets
    
    def binarize_multiclass(self, targets):
        if isinstance(targets, np.ndarray): target_is_numpy = True
        else: target_is_numpy = False
        targets = [target%2 for target in targets]
        if target_is_numpy: targets = np.asarray(targets)
        return targets
    
    def binarize_regression(self, targets):
        if isinstance(targets, np.ndarray): target_is_numpy = True
        else: target_is_numpy = False
        targets = [target<self.regression_to_binary_threshold for target in targets]
        if target_is_numpy: targets = np.asarray(targets)
        return targets
    
    def filter_targets(self, data, targets, targets_list):
        if isinstance(data, np.ndarray): data_is_numpy = True
        else: data_is_numpy = False
        if isinstance(targets, np.ndarray): target_is_numpy = True
        else: target_is_numpy = False
        idxs = [idx for idx in range(len(targets)) if targets[idx] in targets_list]
        filtered_data = [data[idx] for idx in idxs]
        filtered_targets = [targets[idx] for idx in idxs]
        if data_is_numpy: filtered_data = np.asarray(filtered_data)
        if target_is_numpy: filtered_targets = np.asarray(filtered_targets)            
        return filtered_data, filtered_targets
    
    def load(self):
        data, targets = self.load_func()
        if self.use_targets_list is not None: data, targets = self.filter_targets(data, targets, targets_list=self.use_targets_list)
        if self.use_multiclass_to_binary: targets = self.binarize_multiclass(targets)
        if self.use_regression_to_binary: targets = self.binarize_regression(targets)
        if self.use_equalized: data, targets = self.equalize(data, targets)
        if len(targets) > self.size: data, targets = self.resize(data, targets, size=self.size)
        return data, targets