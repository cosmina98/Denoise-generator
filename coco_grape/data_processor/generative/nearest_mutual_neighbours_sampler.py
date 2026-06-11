import numpy as np
import copy
from scipy.sparse.csr import csr_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from coco_grape.data_processor.generative.consistency_generator import ConsistencyGenerator, ClassConditionalConsistencyGenerator


class KernelPCAEncoderDecoder(object):
    """
    A class to perform encoding and decoding of data using Kernel Principal Component Analysis (KernelPCA).

    Attributes:
    estimator (KernelPCA): The KernelPCA estimator used for transforming and inverse transforming the data.
    """

    def __init__(self, n_components=100):
        """
        Initializes the KernelPCAEncoderDecoder with the specified number of components.

        Parameters:
        n_components (int, optional): Number of components for KernelPCA. Default is 100.
        """
        self.estimator = KernelPCA(n_components=n_components, kernel='rbf', gamma=None, fit_inverse_transform=True)

    def fit(self, data_mtx):
        """
        Fits the KernelPCA estimator to the data matrix.

        Parameters:
        data_mtx (array-like): The data matrix to fit.

        Returns:
        self: Returns the instance itself.
        """
        self.estimator.fit(data_mtx)
        return self

    def encode(self, data_mtx):
        """
        Transforms the data matrix into the latent space using KernelPCA.

        Parameters:
        data_mtx (array-like): The data matrix to transform.

        Returns:
        array-like: The transformed latent matrix.
        """
        latent_mtx = self.estimator.transform(data_mtx)
        return latent_mtx

    def decode(self, latent_mtx):
        """
        Inverse transforms the latent matrix back to the original data space using KernelPCA.

        Parameters:
        latent_mtx (array-like): The latent matrix to inverse transform.

        Returns:
        array-like: The inverse transformed data matrix.
        """
        data_mtx = self.estimator.inverse_transform(latent_mtx)
        return data_mtx



class SVDEncoderDecoder(object):
    """
    A class to perform encoding and decoding of data using Truncated Singular Value Decomposition (SVD).

    Attributes:
    estimator (TruncatedSVD): The TruncatedSVD estimator used for transforming and inverse transforming the data.
    """

    def __init__(self, n_components=100):
        """
        Initializes the SVDEncoderDecoder with the specified number of components.

        Parameters:
        n_components (int, optional): Number of components for TruncatedSVD. Default is 100.
        """
        self.estimator = TruncatedSVD(n_components=n_components)

    def fit(self, data_mtx):
        """
        Fits the TruncatedSVD estimator to the data matrix.

        Parameters:
        data_mtx (array-like): The data matrix to fit.

        Returns:
        self: Returns the instance itself.
        """
        self.estimator.fit(data_mtx)
        return self

    def encode(self, data_mtx):
        """
        Transforms the data matrix into the latent space using TruncatedSVD.

        Parameters:
        data_mtx (array-like): The data matrix to transform.

        Returns:
        array-like: The transformed latent matrix.
        """
        latent_mtx = self.estimator.transform(data_mtx)
        return latent_mtx

    def decode(self, latent_mtx):
        """
        Inverse transforms the latent matrix back to the original data space using TruncatedSVD.

        Parameters:
        latent_mtx (array-like): The latent matrix to inverse transform.

        Returns:
        array-like: The inverse transformed data matrix.
        """
        data_mtx = self.estimator.inverse_transform(latent_mtx)
        return data_mtx



class ClassificationConfidenceEstimator(object):
    """
    A class to estimate classification confidence using a specified estimator and confidence threshold.

    Attributes:
    estimator (Classifier): The classifier used to predict probabilities.
    confidence_threshold (float): The threshold below which confidence scores are set to zero.
    """

    def __init__(self, estimator=None, confidence_threshold=0.65):
        """
        Initializes the ClassificationConfidenceEstimator with a specified estimator and confidence threshold.

        Parameters:
        estimator (Classifier, optional): The classifier used to predict probabilities.
        confidence_threshold (float, optional): The threshold below which confidence scores are set to zero. Default is 0.65.
        """
        self.estimator = estimator
        self.confidence_threshold = confidence_threshold

    def fit(self, X, y):
        """
        Fits the estimator to the data.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like): The target vector.

        Returns:
        self: Returns the instance itself.
        """
        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X, y):
        """
        Predicts probabilities and set confidences below the threshold to zero.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like): The target vector.

        Returns:
        array-like: The adjusted confidence probabilities.
        """
        # Predict probabilities using the estimator
        probs = self.estimator.predict_proba(X)

        # Calculate confidences for the correct class
        confidences = np.array([prob[target] for target, prob in zip(y, probs)])

        # Set confidences below the threshold to zero
        confidences[confidences < self.confidence_threshold] = 0

        # Normalize the confidences to sum to 1
        probs = confidences / np.sum(confidences)

        return probs

    def fit_predict_proba(self, X, y):
        """
        Fits the estimator to the data and then predicts adjusted probabilities.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like): The target vector.

        Returns:
        array-like: The adjusted confidence probabilities.
        """
        return self.fit(X, y).predict_proba(X, y)

        


class RegularizedSVMSupportVectorProbabilityEstimator(object):
    """
    A class to estimate the probability of data points being support vectors using a regularized SVM.

    Attributes:
    kernel (str): The kernel type to be used in the SVM algorithm.
    gamma (str): Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
    C_start (float): The starting exponent for the regularization parameter.
    C_end (float): The ending exponent for the regularization parameter.
    n_steps (int): The number of steps between C_start and C_end.
    """

    def __init__(self, kernel='rbf', gamma='scale', C_start=-1, C_end=1, n_steps=10):
        """
        Initializes the RegularizedSVMSupportVectorProbabilityEstimator with specified parameters.

        Parameters:
        kernel (str, optional): The kernel type to be used in the SVM algorithm. Default is 'rbf'.
        gamma (str, optional): Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. Default is 'scale'.
        C_start (float, optional): The starting exponent for the regularization parameter. Default is -1.
        C_end (float, optional): The ending exponent for the regularization parameter. Default is 1.
        n_steps (int, optional): The number of steps between C_start and C_end. Default is 10.
        """
        self.kernel = kernel
        self.gamma = gamma
        self.C_start = C_start
        self.C_end = C_end
        self.n_steps = n_steps

    def fit(self, X, y):
        """
        Fits the estimator to the data. This method is currently a no-op.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like): The target vector.

        Returns:
        self: Returns the instance itself.
        """
        return self

    def fit_predict_proba(self, X, y):
        """
        Fits the estimator to the data and predicts the probability of each data point being a support vector.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like): The target vector.

        Returns:
        array-like: The probability of each data point being a support vector.
        """
        # Generate an array of C values logarithmically spaced between C_start and C_end
        C_steps = np.linspace(self.C_start, self.C_end, self.n_steps)
        Cs = np.power(10, C_steps)

        # Initialize a list to store support vector indicators for each C
        is_support_list = []

        # Iterate over each C value
        for C in Cs:
            # Fit the SVM with the current C value
            est = SVC(C=C, kernel=self.kernel, gamma=self.gamma).fit(X, y)

            # Initialize an array to indicate support vectors
            is_support = np.zeros(len(y))

            # Identify the indices of support vectors where the dual coefficients match C
            support_idxs = est.support_[(np.abs(est.dual_coef_) == C).flatten()]

            # Mark the support vectors in the is_support array
            is_support[support_idxs] = 1

            # Append the support vector indicator array to the list
            is_support_list.append(is_support)

        # Calculate the mean support probability across all C values
        support_probability = np.mean(np.vstack(is_support_list), axis=0)

        # Normalize the support probabilities to sum to 1
        support_probability = support_probability / np.sum(support_probability)

        return support_probability



class nuSVMSupportVectorProbabilityEstimator(object):
    """
    A class to estimate the probability of data points being support vectors using a nu-SVM.

    A nu-SVM (ν-Support Vector Machine) is a variant of the standard Support Vector Machine that introduces 
    a parameter ν (nu) to control the number of support vectors and the margin errors. It allows users to 
    specify an upper bound on the fraction of margin errors and a lower bound on the fraction of 
    support vectors, providing more flexibility in model tuning and regularization compared to the traditional SVM. 

    Attributes:
    kernel (str): The kernel type to be used in the SVM algorithm.
    gamma (str): Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
    nu_start (float): The starting value for the nu parameter.
    nu_end (float): The ending value for the nu parameter.
    n_steps (int): The number of steps between nu_start and nu_end.
    support_instances_fraction (float): The fraction of support instances to consider.
    """

    def __init__(self, kernel='rbf', gamma='scale', nu_start=0.01, nu_end=0.99, n_steps=20, support_instances_fraction=1):
        """
        Initializes the nuSVMSupportVectorProbabilityEstimator with specified parameters.

        Parameters:
        kernel (str, optional): The kernel type to be used in the SVM algorithm. Default is 'rbf'.
        gamma (str, optional): Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. Default is 'scale'.
        nu_start (float, optional): The starting value for the nu parameter. Default is 0.01.
        nu_end (float, optional): The ending value for the nu parameter. Default is 0.99.
        n_steps (int, optional): The number of steps between nu_start and nu_end. Default is 20.
        support_instances_fraction (float, optional): The fraction of support instances to consider. Default is 1.
        """
        self.kernel = kernel
        self.gamma = gamma
        self.nu_start = nu_start
        self.nu_end = nu_end
        self.n_steps = n_steps
        self.support_instances_fraction = support_instances_fraction

    def fit(self, X, y):
        """
        Fits the estimator to the data. This method is currently a no-op.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like): The target vector.

        Returns:
        self: Returns the instance itself.
        """
        return self

    def fit_predict_proba(self, X, y):
        """
        Fits the estimator to the data and predicts the probability of each data point being a support vector.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like): The target vector.

        Returns:
        array-like: The probability of each data point being a support vector.
        """
        # Generate an array of nu values linearly spaced between nu_start and nu_end
        nus = np.linspace(self.nu_start, self.nu_end, self.n_steps)

        # Initialize a list to store support vector indicators for each nu
        is_support_list = []

        # Iterate over each nu value
        for nu in nus:
            try:
                # Fit the nu-SVM with the current nu value
                est = NuSVC(nu=nu, kernel=self.kernel, gamma=self.gamma).fit(X, y)
            except:
                pass
            else:
                # Initialize an array to indicate support vectors
                is_support = np.zeros(len(y))

                # Mark the support vectors in the is_support array
                is_support[est.support_] = 1

                # Append the support vector indicator array to the list
                is_support_list.append(is_support)

        # Calculate the mean support probability across all nu values
        support_probability = np.mean(np.vstack(is_support_list), axis=0)

        # Adjust the support probability if support_instances_fraction is less than 1
        if self.support_instances_fraction < 1:
            sorted_support_probability = -np.sort(-support_probability)
            support_probability_threshold = sorted_support_probability[int(len(sorted_support_probability) * self.support_instances_fraction)]
            support_probability[support_probability < support_probability_threshold] = 0

        # Normalize the support probabilities to sum to 1
        support_probability = support_probability / np.sum(support_probability)

        return support_probability




class NearestNeighboursEstimator(object):
    """
    A class to estimate the nearest neighbours of data points using a specified metric.

    Attributes:
    n_neighbours (int): The number of nearest neighbours to find.
    metric (str): The distance metric to use for finding nearest neighbours.
    """

    def __init__(self, n_neighbours=10, metric='euclidean'):
        """
        Initializes the NearestNeighboursEstimator with specified parameters.

        Parameters:
        n_neighbours (int, optional): The number of nearest neighbours to find. Default is 10.
        metric (str, optional): The distance metric to use for finding nearest neighbours. Default is 'euclidean'.
        """
        self.n_neighbours = n_neighbours
        self.metric = metric

    def dist(self, objects, metric='euclidean', diagval=np.inf):
        """
        Computes the pairwise distances between objects using the specified metric.

        Parameters:
        objects (array-like): The data points to compute distances between.
        metric (str, optional): The distance metric to use. Default is 'euclidean'.
        diagval (float, optional): The value to fill in the diagonal of the distance matrix. Default is infinity.

        Returns:
        array-like: The distance matrix with diagonal values set to diagval.
        """
        # Compute pairwise distances using the specified metric
        distvec = pdist(objects, metric=metric)

        # Convert the condensed distance matrix to a square matrix
        out = squareform(distvec)

        # Set the diagonal values to diagval
        np.fill_diagonal(out, diagval)

        return out

    def fit(self, X, y=None):
        """
        Fits the estimator to the data. This method is currently a no-op.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like, optional): The target vector. Default is None.

        Returns:
        self: Returns the instance itself.
        """
        return self

    def fit_predict(self, X):
        """
        Fits the estimator to the data and predicts the nearest neighbours for each data point.

        Parameters:
        X (array-like): The feature matrix.

        Returns:
        array-like: An array of indices of the nearest neighbours for each data point.
        """
        # Compute the pairwise distance matrix for the data points
        pdists = self.dist(X, self.metric)

        # Sort the distances to find the nearest neighbours for each data point
        nearest_neighbours = np.argsort(pdists)

        # Select the indices of the k-nearest neighbours
        k_nearest_neighbours = nearest_neighbours[:, :self.n_neighbours]

        return k_nearest_neighbours




class NearestMutualNeighboursEstimator(object):
    """
    A class to estimate the nearest mutual neighbours of data points using a specified metric.

    Attributes:
    n_neighbours (int): The number of nearest neighbours to find.
    metric (str): The distance metric to use for finding nearest neighbours.
    """

    def __init__(self, n_neighbours=10, metric='euclidean'):
        """
        Initializes the NearestMutualNeighboursEstimator with specified parameters.

        Parameters:
        n_neighbours (int, optional): The number of nearest neighbours to find. Default is 10.
        metric (str, optional): The distance metric to use for finding nearest neighbours. Default is 'euclidean'.
        """
        self.n_neighbours = n_neighbours
        self.metric = metric

    def dist(self, objects, metric='euclidean', diagval=np.inf):
        """
        Computes the pairwise distances between objects using the specified metric.

        Parameters:
        objects (array-like): The data points to compute distances between.
        metric (str, optional): The distance metric to use. Default is 'euclidean'.
        diagval (float, optional): The value to fill in the diagonal of the distance matrix. Default is infinity.

        Returns:
        array-like: The distance matrix with diagonal values set to diagval.
        """
        # Compute pairwise distances using the specified metric
        distvec = pdist(objects, metric=metric)

        # Convert the condensed distance matrix to a square matrix
        out = squareform(distvec)

        # Set the diagonal values to diagval
        np.fill_diagonal(out, diagval)

        return out

    def fit(self, X, y=None):
        """
        Fits the estimator to the data. This method is currently a no-op.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like, optional): The target vector. Default is None.

        Returns:
        self: Returns the instance itself.
        """
        return self

    def fit_predict_single(self, X):
        """
        Fits the estimator to the data and predicts the nearest mutual neighbours for each data point.

        Parameters:
        X (array-like): The feature matrix.

        Returns:
        list: A list of arrays containing the indices of the nearest mutual neighbours for each data point.
        """
        # Compute the pairwise distance matrix for the data points
        pdists = self.dist(X, self.metric)

        # Sort the distances to find the nearest neighbours for each data point
        nearest_neighbours = np.argsort(pdists)

        # Select the indices of the k-nearest neighbours
        k_nearest_neighbours = nearest_neighbours[:, :self.n_neighbours]

        # Initialize a mask to denote the k-nearest neighbours
        k_nearest_mutual_neighbours_mask = np.zeros(pdists.shape, bool)
        for _mask_row, _neighbours_row in zip(k_nearest_mutual_neighbours_mask, k_nearest_neighbours):
            _mask_row[_neighbours_row] = True

        # Perform element-wise AND with the transposed mask to remove non-mutual nearest neighbours
        k_nearest_mutual_neighbours_mask &= k_nearest_mutual_neighbours_mask.T

        # Extract the indices of the mutual nearest neighbours
        k_nearest_mutual_neighbours = [np.where(row == True)[0] for row in k_nearest_mutual_neighbours_mask]

        return k_nearest_mutual_neighbours

    def fit_predict(self, X, y=None):
        """
        Fits the estimator to the data and predicts the nearest mutual neighbours for each data point.
        If target labels are provided, mutual neighbours are found within the same class.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like, optional): The target vector. Default is None.

        Returns:
        list: A list of arrays containing the indices of the nearest mutual neighbours for each data point.
        """
        # If no target labels are provided, use the single fit_predict method
        if y is None:
            return self.fit_predict_single(X)

        # Convert target labels to a numpy array
        targets = np.asarray(y)

        # Create a mask for each unique target value
        targets_masks_list = [targets == t for t in sorted(set(y))]

        # Initialize a list to store the nearest mutual neighbours for each data point
        k_nearest_mutual_neighbours = [[] for _ in range(len(targets))]

        # Iterate over each target mask
        for targets_mask in targets_masks_list:
            idxs = [i for i, t in enumerate(targets_mask) if t]

            # Find mutual neighbours for data points within the same class
            k_nearest_mutual_neighbours_single = self.fit_predict_single(X[targets_mask])

            # Map the mutual neighbours back to the original indices
            for idx, ngbs in zip(idxs, k_nearest_mutual_neighbours_single):
                k_nearest_mutual_neighbours[idx] = [idxs[ngb] for ngb in ngbs]

        return k_nearest_mutual_neighbours



class NearestMutualNeighboursProbabilityEstimator(object):
    """
    A class to estimate the sampling probability of data points based on their nearest mutual neighbours.

    Attributes:
    nearest_mutual_neighbours_estimator (NearestMutualNeighboursEstimator): An instance of NearestMutualNeighboursEstimator.
    """

    def __init__(self, n_neighbours=10, metric='euclidean'):
        """
        Initializes the NearestMutualNeighboursProbabilityEstimator with specified parameters.

        Parameters:
        n_neighbours (int, optional): The number of nearest neighbours to find. Default is 10.
        metric (str, optional): The distance metric to use for finding nearest neighbours. Default is 'euclidean'.
        """
        self.nearest_mutual_neighbours_estimator = NearestMutualNeighboursEstimator(n_neighbours, metric)

    def fit(self, X, y=None):
        """
        Fits the estimator to the data. This method is currently a no-op.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like, optional): The target vector. Default is None.

        Returns:
        self: Returns the instance itself.
        """
        return self

    def fit_predict_proba_single(self, X):
        """
        Fits the estimator to the data and predicts the sampling probability for each data point.

        Parameters:
        X (array-like): The feature matrix.

        Returns:
        array-like: The sampling probability for each data point.
        """
        # Find the nearest mutual neighbours for each data point
        k_nearest_mutual_neighbours = self.nearest_mutual_neighbours_estimator.fit_predict(X)

        # Calculate the sampling probability based on the number of mutual neighbours
        p = np.array([len(neighbours) / self.nearest_mutual_neighbours_estimator.n_neighbours for neighbours in k_nearest_mutual_neighbours])

        # Normalize the sampling probabilities to sum to 1
        denom = np.sum(p)
        if denom <= 0:
            return np.ones(len(p), dtype=float) / max(1, len(p))
        sampling_probability = p / denom

        return sampling_probability

    def fit_predict_proba(self, X, y=None):
        """
        Fits the estimator to the data and predicts the sampling probability for each data point.
        If target labels are provided, sampling probabilities are calculated within each class.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like, optional): The target vector. Default is None.

        Returns:
        array-like: The sampling probability for each data point.
        """
        # If no target labels are provided, use the single fit_predict_proba method
        if y is None:
            return self.fit_predict_proba_single(X)

        # If X is sparse, convert it to a dense array for boolean indexing
        if hasattr(X, "toarray"):
            X = X.toarray()

        # Convert target labels to a numpy array
        targets = np.asarray(y)

        # Create a mask for each unique target value
        targets_masks_list = [targets == t for t in sorted(set(y))]

        # Initialize an array to store the sampling probabilities
        sampling_probability = np.zeros(len(y))

        # Iterate over each target mask
        for targets_mask in targets_masks_list:
            # Calculate the sampling probabilities for data points within the same class
            sampling_probability[targets_mask] = self.fit_predict_proba_single(X[targets_mask])

        # Normalize the sampling probabilities to sum to 1
        denom = np.sum(sampling_probability)
        if denom <= 0:
            return np.ones(len(sampling_probability), dtype=float) / max(1, len(sampling_probability))
        sampling_probability = sampling_probability / denom

        return sampling_probability



class ProbabilityEstimator(object):
    """
    A class to estimate sampling probabilities using a combination of multiple probability estimators.

    Attributes:
    probability_estimators (list): A list of probability estimator instances.
    """

    def __init__(self, probability_estimators=[]):
        """
        Initializes the ProbabilityEstimator with a list of probability estimators.

        Parameters:
        probability_estimators (list, optional): A list of probability estimator instances. Default is an empty list.
        """
        self.probability_estimators = probability_estimators

    def fit(self, X, y):
        """
        Fits all probability estimators to the data.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like): The target vector.

        Returns:
        self: Returns the instance itself.
        """
        # Fit each probability estimator to the data
        self.probability_estimators = [probability_estimator.fit(X, y) for probability_estimator in self.probability_estimators]
        return self

    def fit_predict_proba(self, X, y):
        """
        Fits all probability estimators to the data and predicts the combined sampling probability for each data point.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like): The target vector.

        Returns:
        array-like: The combined sampling probability for each data point.
        """
        # Compute the sampling probabilities for each estimator
        probs_mtx = np.array([probability_estimator.fit_predict_proba(X, y) for probability_estimator in self.probability_estimators]).T

        # Compute the product of probabilities across all estimators
        p = np.product(probs_mtx, axis=1)

        # Normalize the combined probabilities to sum to 1
        sampling_probability = p / np.sum(p)

        return sampling_probability



class NearestMutualNeighboursSampler(object):
    """
    A class to sample new data points based on nearest mutual neighbours and specified interpolation factors.

    Attributes:
    nearest_mutual_neighbours_estimator (NearestMutualNeighboursEstimator): An instance of NearestMutualNeighboursEstimator.
    probability_estimator (ProbabilityEstimator): An instance of ProbabilityEstimator.
    interpolation_factor (float): The maximum interpolation factor for generating new samples.
    min_interpolation_factor (float): The minimum interpolation factor for generating new samples.
    use_min_max_constraints (bool): Whether to apply min-max constraints on the generated samples.
    """

    def __init__(self, nearest_mutual_neighbours_estimator=None, probability_estimator=None, interpolation_factor=1, min_interpolation_factor=1, use_min_max_constraints=False):
        """
        Initializes the NearestMutualNeighboursSampler with specified parameters.

        Parameters:
        nearest_mutual_neighbours_estimator (NearestMutualNeighboursEstimator, optional): An instance of NearestMutualNeighboursEstimator.
        probability_estimator (ProbabilityEstimator, optional): An instance of ProbabilityEstimator.
        interpolation_factor (float, optional): The maximum interpolation factor for generating new samples. Default is 1.
        min_interpolation_factor (float, optional): The minimum interpolation factor for generating new samples. Default is 1.
        use_min_max_constraints (bool, optional): Whether to apply min-max constraints on the generated samples. Default is False.
        """
        self.nearest_mutual_neighbours_estimator = nearest_mutual_neighbours_estimator
        self.probability_estimator = probability_estimator
        self.interpolation_factor = interpolation_factor
        self.min_interpolation_factor = min_interpolation_factor
        self.use_min_max_constraints = use_min_max_constraints

    def fit(self, X, y=None):
        """
        Fits the sampler to the data and computes the necessary attributes for sampling.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like, optional): The target vector. Default is None.

        Returns:
        self: Returns the instance itself.
        """
        # Deep copy the data matrix and targets to avoid modifying the original data
        self.data_mtx = copy.deepcopy(X)
        self.targets = copy.deepcopy(np.asarray(y))

        # Compute the sampling probability and nearest mutual neighbours
        self.sampling_probability = self.probability_estimator.fit_predict_proba(X, y)
        self.k_nearest_mutual_neighbours = self.nearest_mutual_neighbours_estimator.fit_predict(X)

        return self

    def generate(self, data_mtx, k_nearest_mutual_neighbours, sampling_probability, interpolation_factor, min_interpolation_factor):
        """
        Generates a new data point based on nearest mutual neighbours and interpolation.

        Parameters:
        data_mtx (array-like): The data matrix.
        k_nearest_mutual_neighbours (list): A list of nearest mutual neighbours for each data point.
        sampling_probability (array-like): The sampling probability for each data point.
        interpolation_factor (float): The maximum interpolation factor.
        min_interpolation_factor (float): The minimum interpolation factor.

        Returns:
        array-like: A newly generated data point.
        """
        # Select an instance at random based on the sampling probability
        idx1 = np.random.choice(len(k_nearest_mutual_neighbours), size=1, p=sampling_probability)[0]

        # Select one of its neighbours at random
        idx2 = np.random.choice(k_nearest_mutual_neighbours[idx1])

        # Select one of the neighbours of the second instance at random
        idx3 = np.random.choice(k_nearest_mutual_neighbours[idx2])

        # Compute the scaled offset for interpolation
        alpha = np.random.rand() * (interpolation_factor - min_interpolation_factor) + min_interpolation_factor
        xn = data_mtx[idx1] + alpha * (data_mtx[idx3] - data_mtx[idx2])

        return xn

    def min_max_constraints(self, X, Xp):
        """
        Applies min-max constraints to the generated data points to ensure they are within the original data range.

        Parameters:
        X (array-like): The original data matrix.
        Xp (array-like): The generated data matrix.

        Returns:
        array-like: The constrained data matrix.
        """
        # Compute the min and max values for each feature in the original data
        mn = np.min(X, axis=0)
        mx = np.max(X, axis=0)

        # Apply min-max constraints to each feature in the generated data
        Xn = []
        for i in range(X.shape[1]):
            Xpi = Xp[:, i]
            Xpi[Xpi < mn[i]] = mn[i]
            Xpi[Xpi > mx[i]] = mx[i]
            Xn.append(Xpi.reshape(-1, 1))
        Xn = np.hstack(Xn)

        return Xn

    def sample(self, n_samples, target=None):
        """
        Generates new samples based on the fitted data and specified parameters.

        Parameters:
        n_samples (int): The number of samples to generate.
        target (int, optional): The target class for which to generate samples. Default is None.

        Returns:
        array-like: The generated samples.
        """
        # Adjust the sampling probability if a target class is specified
        if target is not None:
            sampling_probability = copy.deepcopy(self.sampling_probability)
            sampling_probability[self.targets != target] = 0
            sampling_probability = sampling_probability / np.sum(sampling_probability)
        else:
            sampling_probability = self.sampling_probability

        # Generate new samples
        sampled_data_mtx = []
        for _ in range(n_samples):
            try:
                x = self.generate(self.data_mtx, self.k_nearest_mutual_neighbours, sampling_probability, self.interpolation_factor, self.min_interpolation_factor)
                sampled_data_mtx.append(x)
            except:
                pass

        sampled_data_mtx = np.array(sampled_data_mtx)

        # Apply min-max constraints if specified
        if self.use_min_max_constraints:
            sampled_data_mtx = self.min_max_constraints(self.data_mtx, sampled_data_mtx)

        return sampled_data_mtx


def ConcreteNearestMutualNeighboursSampler(n_neighbours=10, interpolation_factor=1, min_interpolation_factor=1, metric='euclidean', use_min_max_constraints=False):
    """
    Creates an instance of NearestMutualNeighboursSampler with specified parameters and components.

    Parameters:
    n_neighbours (int, optional): The number of nearest neighbours to find. Default is 10.
    interpolation_factor (float, optional): The maximum interpolation factor for generating new samples. Default is 1.
    min_interpolation_factor (float, optional): The minimum interpolation factor for generating new samples. Default is 1.
    metric (str, optional): The distance metric to use for finding nearest neighbours. Default is 'euclidean'.
    use_min_max_constraints (bool, optional): Whether to apply min-max constraints on the generated samples. Default is False.

    Returns:
    NearestMutualNeighboursSampler: An instance of NearestMutualNeighboursSampler configured with the specified parameters.
    """
    # Create an instance of NearestMutualNeighboursEstimator with the specified number of neighbours and metric
    nearest_mutual_neighbours_estimator = NearestMutualNeighboursEstimator(n_neighbours, metric)

    # Create an instance of NearestMutualNeighboursProbabilityEstimator with the specified number of neighbours and metric
    probability_estimator = NearestMutualNeighboursProbabilityEstimator(n_neighbours, metric)

    # Create an instance of NearestMutualNeighboursSampler with the specified parameters and components
    sampler = NearestMutualNeighboursSampler(
        nearest_mutual_neighbours_estimator,
        probability_estimator,
        interpolation_factor=interpolation_factor,
        min_interpolation_factor=min_interpolation_factor,
        use_min_max_constraints=use_min_max_constraints
    )

    return sampler



class ConsistentSampler(object):
    """
    A class to generate samples consistently by combining a sampler and a consistency generator.

    Attributes:
    sampler (Sampler): An instance of a sampler used to generate initial samples.
    consistency_generator (ConsistencyGenerator): An instance of a consistency generator used to adjust the samples.
    """

    def __init__(self, sampler=None, consistency_generator=None):
        """
        Initializes the ConsistentSampler with specified sampler and consistency generator.

        Parameters:
        sampler (Sampler, optional): An instance of a sampler. Default is None.
        consistency_generator (ConsistencyGenerator, optional): An instance of a consistency generator. Default is None.
        """
        self.sampler = sampler
        self.consistency_generator = consistency_generator

    def fit(self, X):
        """
        Fits both the sampler and the consistency generator to the data.

        Parameters:
        X (array-like): The feature matrix to fit the models.

        Returns:
        self: Returns the instance itself.
        """
        # Fit the sampler to the data
        self.sampler.fit(X)
        
        # Fit the consistency generator to the data
        self.consistency_generator.fit(X)
        
        return self

    def sample(self, n_samples):
        """
        Generates samples using the sampler and then adjusts them using the consistency generator.

        Parameters:
        n_samples (int): The number of samples to generate.

        Returns:
        array-like: The adjusted samples.
        """
        # Generate initial samples using the sampler
        Xp = self.sampler.sample(n_samples)
        
        # Adjust the generated samples using the consistency generator
        Xpp = self.consistency_generator.predict(Xp)
        
        return Xpp




def ConcreteConsistentNearestMutualNeighboursSampler(n_features=0.5, n_iterations=100, iteration_weight=0.5, n_neighbours=10, interpolation_factor=1, min_interpolation_factor=1, metric='euclidean', use_min_max_constraints=False):
    """
    Creates an instance of ConsistentSampler with a NearestMutualNeighboursSampler and ConsistencyGenerator.

    Parameters:
    n_features (float, optional): Fraction of features to consider when fitting the consistency generator. Default is 0.5.
    n_iterations (int, optional): Number of iterations for the consistency generator. Default is 100.
    iteration_weight (float, optional): Weight of the iteration in the consistency generator. Default is 0.5.
    n_neighbours (int, optional): The number of nearest neighbours to find. Default is 10.
    interpolation_factor (float, optional): The maximum interpolation factor for generating new samples. Default is 1.
    min_interpolation_factor (float, optional): The minimum interpolation factor for generating new samples. Default is 1.
    metric (str, optional): The distance metric to use for finding nearest neighbours. Default is 'euclidean'.
    use_min_max_constraints (bool, optional): Whether to apply min-max constraints on the generated samples. Default is False.

    Returns:
    ConsistentSampler: An instance of ConsistentSampler configured with the specified parameters.
    """
    # Create an instance of NearestMutualNeighboursEstimator with the specified number of neighbours and metric
    nearest_mutual_neighbours_estimator = NearestMutualNeighboursEstimator(n_neighbours, metric)

    # Create an instance of NearestMutualNeighboursProbabilityEstimator with the specified number of neighbours and metric
    probability_estimator = NearestMutualNeighboursProbabilityEstimator(n_neighbours, metric)

    # Create an instance of NearestMutualNeighboursSampler with the specified parameters and components
    sampler = NearestMutualNeighboursSampler(
        nearest_mutual_neighbours_estimator,
        probability_estimator,
        interpolation_factor=interpolation_factor,
        min_interpolation_factor=min_interpolation_factor,
        use_min_max_constraints=use_min_max_constraints
    )

    # Create an instance of ConsistencyGenerator with a RandomForestRegressor as the base estimator
    consistency_generator = ConsistencyGenerator(
        base_estimator=RandomForestRegressor(n_estimators=30), 
        n_features=n_features, 
        n_iterations=n_iterations, 
        iteration_weight=iteration_weight, 
        use_randomized=False, 
        parallel=True
    )

    # Create an instance of ConsistentSampler with the NearestMutualNeighboursSampler and ConsistencyGenerator
    csampler = ConsistentSampler(sampler=sampler, consistency_generator=consistency_generator)

    return csampler



class ClassConditionalSamplingTransformer(object):
    """
    A class to perform class-conditional sampling transformation on a dataset using a specified sampler.

    Attributes:
    sampler (Sampler): An instance of a sampler used to generate samples.
    resampling_factor (float): Factor by which to resample the dataset.
    use_balanced (bool): Whether to balance the classes by resampling to the size of the largest class.
    """

    def __init__(self, sampler, resampling_factor=1, use_balanced=True):
        """
        Initializes the ClassConditionalSamplingTransformer with specified parameters.

        Parameters:
        sampler (Sampler): An instance of a sampler.
        resampling_factor (float, optional): Factor by which to resample the dataset. Default is 1.
        use_balanced (bool, optional): Whether to balance the classes by resampling to the size of the largest class. Default is True.
        """
        self.sampler = sampler
        self.resampling_factor = resampling_factor 
        self.use_balanced = use_balanced
        self.n_classes = None

    def fit(self, X, y):
        """
        Fits the sampler to the data.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like): The target vector.

        Returns:
        self: Returns the instance itself.
        """
        self.sampler.fit(X, y)
        self.n_classes = len(set(y))
        return self

    def transform(self, X, y):
        """
        Transforms the data by resampling each class according to the specified resampling factor.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like): The target vector.

        Returns:
        tuple: The resampled feature matrix and target vector.
        """
        # Get the list of unique target classes
        target_list = sorted(set(y))
        
        # Convert target vector to a numpy array
        targets = np.asarray(y)
        
        # Count the number of instances in each class
        target_counts = np.bincount(targets)
        
        # Find the maximum class count for balanced resampling
        max_target_counts = np.max(target_counts)
        
        # Initialize lists to store the resampled features and targets
        all_X = []
        all_targets = []
        
        # Iterate over each target class
        for target in target_list:
            if self.use_balanced:
                # Resample to the size of the largest class
                Xp = self.sampler.sample(n_samples=int(max_target_counts * self.resampling_factor), target=target)
            else:
                # Resample according to the original class size
                Xp = self.sampler.sample(n_samples=int(target_counts[target] * self.resampling_factor), target=target)
            
            # Append the resampled features and targets to the lists
            all_X.append(Xp)
            all_targets.append([target] * Xp.shape[0])
        
        # Stack the lists to form the resampled feature matrix and target vector
        Xs = np.vstack(all_X)
        ys = np.hstack(all_targets)
        
        return Xs, ys

    def sample(self, n_samples):
        Xs = []
        ys = []
        
        if isinstance(n_samples, list) is False: 
            n_samples = [n_samples]*self.n_classes

        for target, max_target_counts in enumerate(n_samples):
            Xp = self.sampler.sample(n_samples=int(max_target_counts * self.resampling_factor), target=target)
            Xs.append(Xp)
            ys.append([target] * len(Xp))
        
        # Stack the lists to form the resampled feature matrix and target vector
        if isinstance(Xs[0],np.ndarray): Xs = np.vstack(Xs)
        else: Xs = sum(Xs, [])
        ys = np.hstack(ys)
        
        return Xs, ys

    def fit_transform(self, X, y):
        """
        Fits the sampler to the data and then transforms it by resampling each class.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like): The target vector.

        Returns:
        tuple: The resampled feature matrix and target vector.
        """
        return self.fit(X, y).transform(X, y)



class ClassConditionalConsistentSamplingTransformer(object):
    """
    A class to perform class-conditional consistent sampling transformation on a dataset using a specified sampling transformer and consistency generator.

    Attributes:
    class_conditional_sampling_transformer (ClassConditionalSamplingTransformer): An instance of a class-conditional sampling transformer.
    consistency_generator (ConsistencyGenerator): An instance of a consistency generator.
    """

    def __init__(self, class_conditional_sampling_transformer=None, consistency_generator=None):
        """
        Initializes the ClassConditionalConsistentSamplingTransformer with specified parameters.

        Parameters:
        class_conditional_sampling_transformer (ClassConditionalSamplingTransformer, optional): An instance of a class-conditional sampling transformer. Default is None.
        consistency_generator (ConsistencyGenerator, optional): An instance of a consistency generator. Default is None.
        """
        self.class_conditional_sampling_transformer = class_conditional_sampling_transformer
        self.consistency_generator = consistency_generator

    def fit(self, X, y):
        """
        Fits both the class-conditional sampling transformer and the consistency generator to the data.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like): The target vector.

        Returns:
        self: Returns the instance itself.
        """
        # Fit the class-conditional sampling transformer to the data
        self.class_conditional_sampling_transformer.fit(X, y)
        
        # Fit the consistency generator to the data
        self.consistency_generator.fit(X, y)
        
        return self

    def sample(self, n_samples):
        Xp, yp = self.class_conditional_sampling_transformer.sample(n_samples)
        
        # Apply the consistency generator to the resampled data
        Xpp = self.consistency_generator.predict(Xp, yp)
        
        return Xpp, yp

    def transform(self, X, y):
        """
        Transforms the data by resampling each class and then applying the consistency generator.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like): The target vector.

        Returns:
        tuple: The transformed feature matrix and target vector.
        """
        # Resample the data using the class-conditional sampling transformer
        Xp, yp = self.class_conditional_sampling_transformer.transform(X, y)
        
        # Apply the consistency generator to the resampled data
        Xpp = self.consistency_generator.predict(Xp, yp)
        
        return Xpp, yp

    def fit_transform(self, X, y):
        """
        Fits the class-conditional sampling transformer and consistency generator to the data, and then transforms the data.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like): The target vector.

        Returns:
        tuple: The transformed feature matrix and target vector.
        """
        return self.fit(X, y).transform(X, y)



def ConcreteClassConditionalSamplingTransformer(n_neighbours, resampling_factor=1, interpolation_factor=1, min_interpolation_factor=1, use_balanced=False, use_min_max_constraints=False, metric='euclidean'):
    """
    Creates an instance of ClassConditionalSamplingTransformer with specified parameters and components.

    Parameters:
    n_neighbours (int): The number of nearest neighbours to find.
    resampling_factor (float, optional): Factor by which to resample the dataset. Default is 1.
    interpolation_factor (float, optional): The maximum interpolation factor for generating new samples. Default is 1.
    min_interpolation_factor (float, optional): The minimum interpolation factor for generating new samples. Default is 1.
    use_balanced (bool, optional): Whether to balance the classes by resampling to the size of the largest class. Default is False.
    use_min_max_constraints (bool, optional): Whether to apply min-max constraints on the generated samples. Default is False.
    metric (str, optional): The distance metric to use for finding nearest neighbours. Default is 'euclidean'.

    Returns:
    ClassConditionalSamplingTransformer: An instance of ClassConditionalSamplingTransformer configured with the specified parameters.
    """
    # Create an instance of NearestMutualNeighboursEstimator with the specified number of neighbours and metric
    nearest_mutual_neighbours_estimator = NearestMutualNeighboursEstimator(n_neighbours, metric)

    # Create an instance of NearestMutualNeighboursProbabilityEstimator with the specified number of neighbours and metric
    probability_estimator = NearestMutualNeighboursProbabilityEstimator(n_neighbours, metric)

    # Create an instance of NearestMutualNeighboursSampler with the specified parameters and components
    sampler = NearestMutualNeighboursSampler(
        nearest_mutual_neighbours_estimator,
        probability_estimator,
        interpolation_factor=interpolation_factor,
        min_interpolation_factor=min_interpolation_factor,
        use_min_max_constraints=use_min_max_constraints
    )

    # Create an instance of ClassConditionalSamplingTransformer with the specified resampling factor and balance option
    cc_sampler = ClassConditionalSamplingTransformer(sampler, resampling_factor=resampling_factor, use_balanced=use_balanced)

    return cc_sampler



def ConcreteSupportClassConditionalSamplingTransformer(n_neighbours, support_instances_fraction=1, resampling_factor=1, interpolation_factor=1, min_interpolation_factor=1, use_balanced=False, use_min_max_constraints=False, metric='euclidean'):
    """
    Creates an instance of ClassConditionalSamplingTransformer with support-based sampling and specified parameters.

    Parameters:
    n_neighbours (int): The number of nearest neighbours to find.
    support_instances_fraction (float, optional): Fraction of support instances to consider for sampling. Default is 1.
    resampling_factor (float, optional): Factor by which to resample the dataset. Default is 1.
    interpolation_factor (float, optional): The maximum interpolation factor for generating new samples. Default is 1.
    min_interpolation_factor (float, optional): The minimum interpolation factor for generating new samples. Default is 1.
    use_balanced (bool, optional): Whether to balance the classes by resampling to the size of the largest class. Default is False.
    use_min_max_constraints (bool, optional): Whether to apply min-max constraints on the generated samples. Default is False.
    metric (str, optional): The distance metric to use for finding nearest neighbours. Default is 'euclidean'.

    Returns:
    ClassConditionalSamplingTransformer: An instance of ClassConditionalSamplingTransformer configured with the specified parameters.
    """
    # Create an instance of NearestMutualNeighboursEstimator with the specified number of neighbours and metric
    nearest_mutual_neighbours_estimator = NearestMutualNeighboursEstimator(n_neighbours, metric)

    # Create a list of probability estimators: NearestMutualNeighboursProbabilityEstimator and nuSVMSupportVectorProbabilityEstimator
    probability_estimators = [
        NearestMutualNeighboursProbabilityEstimator(n_neighbours, metric),
        nuSVMSupportVectorProbabilityEstimator(kernel='rbf', gamma='scale', nu_start=0.01, nu_end=0.99, n_steps=20, support_instances_fraction=support_instances_fraction)
    ]

    # Create an instance of ProbabilityEstimator with the list of probability estimators
    probability_estimator = ProbabilityEstimator(probability_estimators)

    # Create an instance of NearestMutualNeighboursSampler with the specified parameters and components
    sampler = NearestMutualNeighboursSampler(
        nearest_mutual_neighbours_estimator,
        probability_estimator,
        interpolation_factor=interpolation_factor,
        min_interpolation_factor=min_interpolation_factor,
        use_min_max_constraints=use_min_max_constraints
    )

    # Create an instance of ClassConditionalSamplingTransformer with the specified resampling factor and balance option
    cc_sampler = ClassConditionalSamplingTransformer(sampler, resampling_factor=resampling_factor, use_balanced=use_balanced)

    return cc_sampler


def ConcreteConsistencyClassConditionalSamplingTransformer(n_neighbours, n_features=0.1, n_estimators=100, n_iterations=4, iteration_weight=0.5, resampling_factor=1, interpolation_factor=1, min_interpolation_factor=1, use_balanced=False, use_min_max_constraints=False, metric='euclidean'):
    """
    Creates an instance of ClassConditionalConsistentSamplingTransformer with consistency-based sampling and specified parameters.

    Parameters:
    n_neighbours (int): The number of nearest neighbours to find.
    n_features (float, optional): Fraction of features to consider when fitting the consistency generator. Default is 0.1.
    n_estimators (int, optional): Number of estimators for the RandomForestRegressor. Default is 100.
    n_iterations (int, optional): Number of iterations for the consistency generator. Default is 4.
    iteration_weight (float, optional): Weight of the iteration in the consistency generator. Default is 0.5.
    resampling_factor (float, optional): Factor by which to resample the dataset. Default is 1.
    interpolation_factor (float, optional): The maximum interpolation factor for generating new samples. Default is 1.
    min_interpolation_factor (float, optional): The minimum interpolation factor for generating new samples. Default is 1.
    use_balanced (bool, optional): Whether to balance the classes by resampling to the size of the largest class. Default is False.
    use_min_max_constraints (bool, optional): Whether to apply min-max constraints on the generated samples. Default is False.
    metric (str, optional): The distance metric to use for finding nearest neighbours. Default is 'euclidean'.

    Returns:
    ClassConditionalConsistentSamplingTransformer: An instance of ClassConditionalConsistentSamplingTransformer configured with the specified parameters.
    """
    # Create an instance of ConcreteClassConditionalSamplingTransformer with the specified parameters
    class_conditional_sampling_transformer = ConcreteClassConditionalSamplingTransformer(
        n_neighbours=n_neighbours,
        resampling_factor=resampling_factor,
        interpolation_factor=interpolation_factor,
        min_interpolation_factor=min_interpolation_factor,
        use_balanced=use_balanced,
        use_min_max_constraints=use_min_max_constraints,
        metric=metric
    )

    # Create an instance of ConsistencyGenerator with a RandomForestRegressor as the base estimator
    consistency_generator = ConsistencyGenerator(
        base_estimator=RandomForestRegressor(n_estimators=n_estimators),
        n_features=n_features,
        iteration_weight=iteration_weight
    )

    # Create an instance of ClassConditionalConsistencyGenerator with the specified parameters
    cc_consistency_generator = ClassConditionalConsistencyGenerator(consistency_generator=consistency_generator).set_params(
        n_iterations=n_iterations,
        iteration_weight=iteration_weight
    )

    # Create an instance of ClassConditionalConsistentSamplingTransformer with the class conditional sampling transformer and consistency generator
    cc_sampler = ClassConditionalConsistentSamplingTransformer(
        class_conditional_sampling_transformer=class_conditional_sampling_transformer,
        consistency_generator=cc_consistency_generator
    )

    return cc_sampler

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



class EncoderDecoderNearestMutualNeighboursSampler(object):
    """
    A class to perform sampling using an encoder-decoder model combined with a nearest mutual neighbours sampler.

    Attributes:
    encoder_decoder (EncoderDecoder): An instance of an encoder-decoder model.
    nearest_mutual_neighbours_sampler (NearestMutualNeighboursSampler): An instance of a nearest mutual neighbours sampler.
    """

    def __init__(self, encoder_decoder=None, nearest_mutual_neighbours_sampler=None):
        """
        Initializes the EncoderDecoderNearestMutualNeighboursSampler with specified encoder-decoder and nearest mutual neighbours sampler.

        Parameters:
        encoder_decoder (EncoderDecoder, optional): An instance of an encoder-decoder model. Default is None.
        nearest_mutual_neighbours_sampler (NearestMutualNeighboursSampler, optional): An instance of a nearest mutual neighbours sampler. Default is None.
        """
        self.encoder_decoder = encoder_decoder
        self.nearest_mutual_neighbours_sampler = nearest_mutual_neighbours_sampler

    def fit(self, X, y):
        """
        Fits the encoder-decoder to the data and then fits the nearest mutual neighbours sampler to the encoded data.

        Parameters:
        X (array-like): The feature matrix.
        y (array-like): The target vector.

        Returns:
        self: Returns the instance itself.
        """
        # Encode the data using the encoder-decoder model
        Z = self.encoder_decoder.fit(X).encode(X)

        # Fit the nearest mutual neighbours sampler to the encoded data
        self.nearest_mutual_neighbours_sampler.fit(Z, y)

        return self

    def sample(self, n_samples, target=None):
        """
        Generates samples using the nearest mutual neighbours sampler and then decodes them using the encoder-decoder model.

        Parameters:
        n_samples (int): The number of samples to generate.
        target (int, optional): The target class for which to generate samples. Default is None.

        Returns:
        array-like: The generated samples in the original feature space.
        """
        # Generate samples in the encoded space using the nearest mutual neighbours sampler
        Z = self.nearest_mutual_neighbours_sampler.sample(n_samples, target)

        # Decode the generated samples back to the original feature space
        X = self.encoder_decoder.decode(Z)

        return X


def ConcreteClassConditionalEncoderDecoderSamplingTransformer(n_neighbours, n_components=10, resampling_factor=1, interpolation_factor=1, min_interpolation_factor=1, use_balanced=False, use_min_max_constraints=False, metric='euclidean', use_linear_encoder=True):
    """
    Creates an instance of ClassConditionalSamplingTransformer with an encoder-decoder model and nearest mutual neighbours sampling.

    Parameters:
    n_neighbours (int): The number of nearest neighbours to find.
    n_components (int, optional): Number of components for the encoder-decoder model. Default is 10.
    resampling_factor (float, optional): Factor by which to resample the dataset. Default is 1.
    interpolation_factor (float, optional): The maximum interpolation factor for generating new samples. Default is 1.
    min_interpolation_factor (float, optional): The minimum interpolation factor for generating new samples. Default is 1.
    use_balanced (bool, optional): Whether to balance the classes by resampling to the size of the largest class. Default is False.
    use_min_max_constraints (bool, optional): Whether to apply min-max constraints on the generated samples. Default is False.
    metric (str, optional): The distance metric to use for finding nearest neighbours. Default is 'euclidean'.
    use_linear_encoder (bool, optional): Whether to use a linear encoder (SVD) or a non-linear encoder (KernelPCA). Default is True.

    Returns:
    ClassConditionalSamplingTransformer: An instance of ClassConditionalSamplingTransformer configured with the specified parameters.
    """
    # Choose the encoder-decoder model based on the use_linear_encoder flag
    if use_linear_encoder:
        encoder_decoder = SVDEncoderDecoder(n_components=n_components)
    else:
        encoder_decoder = KernelPCAEncoderDecoder(n_components=n_components)

    # Create an instance of NearestMutualNeighboursEstimator with the specified number of neighbours and metric
    nearest_mutual_neighbours_estimator = NearestMutualNeighboursEstimator(n_neighbours, metric)

    # Create an instance of NearestMutualNeighboursProbabilityEstimator with the specified number of neighbours and metric
    probability_estimator = NearestMutualNeighboursProbabilityEstimator(n_neighbours, metric)

    # Create an instance of NearestMutualNeighboursSampler with the specified parameters and components
    nearest_mutual_neighbours_sampler = NearestMutualNeighboursSampler(
        nearest_mutual_neighbours_estimator,
        probability_estimator,
        interpolation_factor=interpolation_factor,
        min_interpolation_factor=min_interpolation_factor,
        use_min_max_constraints=use_min_max_constraints
    )

    # Create an instance of EncoderDecoderNearestMutualNeighboursSampler with the encoder-decoder and nearest mutual neighbours sampler
    sampler = EncoderDecoderNearestMutualNeighboursSampler(encoder_decoder, nearest_mutual_neighbours_sampler)

    # Create an instance of ClassConditionalSamplingTransformer with the specified resampling factor and balance option
    cc_sampler = ClassConditionalSamplingTransformer(sampler, resampling_factor=resampling_factor, use_balanced=use_balanced)

    return cc_sampler


def ConcreteSupportClassConditionalEncoderDecoderSamplingTransformer(n_neighbours, n_components=10, support_instances_fraction=1, resampling_factor=1, interpolation_factor=1, min_interpolation_factor=1, use_balanced=False, use_min_max_constraints=False, metric='euclidean', use_linear_encoder=True):
    """
    Creates an instance of ClassConditionalSamplingTransformer with an encoder-decoder model, support-based sampling, and specified parameters.

    Parameters:
    n_neighbours (int): The number of nearest neighbours to find.
    n_components (int, optional): Number of components for the encoder-decoder model. Default is 10.
    support_instances_fraction (float, optional): Fraction of support instances to consider for sampling. Default is 1.
    resampling_factor (float, optional): Factor by which to resample the dataset. Default is 1.
    interpolation_factor (float, optional): The maximum interpolation factor for generating new samples. Default is 1.
    min_interpolation_factor (float, optional): The minimum interpolation factor for generating new samples. Default is 1.
    use_balanced (bool, optional): Whether to balance the classes by resampling to the size of the largest class. Default is False.
    use_min_max_constraints (bool, optional): Whether to apply min-max constraints on the generated samples. Default is False.
    metric (str, optional): The distance metric to use for finding nearest neighbours. Default is 'euclidean'.
    use_linear_encoder (bool, optional): Whether to use a linear encoder (SVD) or a non-linear encoder (KernelPCA). Default is True.

    Returns:
    ClassConditionalSamplingTransformer: An instance of ClassConditionalSamplingTransformer configured with the specified parameters.
    """
    # Choose the encoder-decoder model based on the use_linear_encoder flag
    if use_linear_encoder:
        encoder_decoder = SVDEncoderDecoder(n_components=n_components)
    else:
        encoder_decoder = KernelPCAEncoderDecoder(n_components=n_components)

    # Create an instance of NearestMutualNeighboursEstimator with the specified number of neighbours and metric
    nearest_mutual_neighbours_estimator = NearestMutualNeighboursEstimator(n_neighbours, metric)

    # Create a list of probability estimators: NearestMutualNeighboursProbabilityEstimator and nuSVMSupportVectorProbabilityEstimator
    probability_estimators = [
        NearestMutualNeighboursProbabilityEstimator(n_neighbours, metric),
        nuSVMSupportVectorProbabilityEstimator(kernel='rbf', gamma='scale', nu_start=0.01, nu_end=0.99, n_steps=20, support_instances_fraction=support_instances_fraction)
    ]

    # Create an instance of ProbabilityEstimator with the list of probability estimators
    probability_estimator = ProbabilityEstimator(probability_estimators)

    # Create an instance of NearestMutualNeighboursSampler with the specified parameters and components
    nearest_mutual_neighbours_sampler = NearestMutualNeighboursSampler(
        nearest_mutual_neighbours_estimator,
        probability_estimator,
        interpolation_factor=interpolation_factor,
        min_interpolation_factor=min_interpolation_factor,
        use_min_max_constraints=use_min_max_constraints
    )

    # Create an instance of EncoderDecoderNearestMutualNeighboursSampler with the encoder-decoder and nearest mutual neighbours sampler
    sampler = EncoderDecoderNearestMutualNeighboursSampler(encoder_decoder, nearest_mutual_neighbours_sampler)

    # Create an instance of ClassConditionalSamplingTransformer with the specified resampling factor and balance option
    cc_sampler = ClassConditionalSamplingTransformer(sampler, resampling_factor=resampling_factor, use_balanced=use_balanced)

    return cc_sampler



def ConcreteConsistencyClassConditionalEncoderDecoderSamplingTransformer(n_neighbours, n_features=0.1, n_estimators=100, n_iterations=4, iteration_weight=0.5, n_components=10, resampling_factor=1, interpolation_factor=1, min_interpolation_factor=1, use_balanced=False, use_min_max_constraints=False, metric='euclidean', use_linear_encoder=True):
    """
    Creates an instance of ClassConditionalConsistentSamplingTransformer with consistency-based sampling, encoder-decoder model, and specified parameters.

    Parameters:
    n_neighbours (int): The number of nearest neighbours to find.
    n_features (float, optional): Fraction of features to consider when fitting the consistency generator. Default is 0.1.
    n_estimators (int, optional): Number of estimators for the RandomForestRegressor. Default is 100.
    n_iterations (int, optional): Number of iterations for the consistency generator. Default is 4.
    iteration_weight (float, optional): Weight of the iteration in the consistency generator. Default is 0.5.
    n_components (int, optional): Number of components for the encoder-decoder model. Default is 10.
    resampling_factor (float, optional): Factor by which to resample the dataset. Default is 1.
    interpolation_factor (float, optional): The maximum interpolation factor for generating new samples. Default is 1.
    min_interpolation_factor (float, optional): The minimum interpolation factor for generating new samples. Default is 1.
    use_balanced (bool, optional): Whether to balance the classes by resampling to the size of the largest class. Default is False.
    use_min_max_constraints (bool, optional): Whether to apply min-max constraints on the generated samples. Default is False.
    metric (str, optional): The distance metric to use for finding nearest neighbours. Default is 'euclidean'.
    use_linear_encoder (bool, optional): Whether to use a linear encoder (SVD) or a non-linear encoder (KernelPCA). Default is True.

    Returns:
    ClassConditionalConsistentSamplingTransformer: An instance of ClassConditionalConsistentSamplingTransformer configured with the specified parameters.
    """
    # Create an instance of ConcreteClassConditionalEncoderDecoderSamplingTransformer with the specified parameters
    class_conditional_sampling_transformer = ConcreteClassConditionalEncoderDecoderSamplingTransformer(
        n_neighbours=n_neighbours,
        n_components=n_components,
        resampling_factor=resampling_factor,
        interpolation_factor=interpolation_factor,
        min_interpolation_factor=min_interpolation_factor,
        use_balanced=use_balanced,
        use_min_max_constraints=use_min_max_constraints,
        metric=metric,
        use_linear_encoder=use_linear_encoder
    )

    # Create an instance of ConsistencyGenerator with a RandomForestRegressor as the base estimator
    consistency_generator = ConsistencyGenerator(
        base_estimator=RandomForestRegressor(n_estimators=n_estimators),
        n_features=n_features,
        iteration_weight=iteration_weight
    )

    # Create an instance of ClassConditionalConsistencyGenerator with the specified parameters
    cc_consistency_generator = ClassConditionalConsistencyGenerator(consistency_generator=consistency_generator).set_params(
        n_iterations=n_iterations,
        iteration_weight=iteration_weight
    )

    # Create an instance of ClassConditionalConsistentSamplingTransformer with the class conditional sampling transformer and consistency generator
    cc_sampler = ClassConditionalConsistentSamplingTransformer(
        class_conditional_sampling_transformer=class_conditional_sampling_transformer,
        consistency_generator=cc_consistency_generator
    )

    return cc_sampler

