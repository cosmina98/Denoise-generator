import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
import scipy.linalg

class SlicedInverseRegressionEmbedder(BaseEstimator, TransformerMixin):
    """
    Sliced Inverse Regression (SIR) for dimension reduction.
    
    This estimator finds a set of linear directions that are most informative
    about the response variable y by slicing y into intervals, computing the 
    corresponding means of X, and solving a generalized eigenvalue problem.
    
    Parameters
    ----------
    n_directions : int, default=2
        Number of effective directions to extract.
    n_slices : int, default=10
        Number of slices to use for the response variable.
    reg : float, default=1e-6
        Regularization parameter added to the covariance matrix of X.
    
    Attributes
    ----------
    directions_ : array, shape (n_features, n_directions)
        The projection matrix (directions) learned from SIR.
    """
    
    def __init__(self, n_directions=2, n_slices=10, reg=1e-6):
        self.n_directions = n_directions
        self.n_slices = n_slices
        self.reg = reg
        
    def fit(self, X, y):
        X = check_array(X)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be a 1-dimensional array for SIR.")
        n_samples, n_features = X.shape

        # Center the predictors
        X_centered = X - np.mean(X, axis=0)

        # Compute covariance matrix of X and add regularization for stability.
        cov_X = np.cov(X_centered, rowvar=False)
        cov_X += self.reg * np.eye(n_features)
        
        # Slice the response variable using percentiles
        percentiles = np.linspace(0, 100, self.n_slices + 1)
        slice_edges = np.percentile(y, percentiles)
        
        # Compute slice means directly and keep track of slice counts.
        slice_means = []
        slice_counts = []
        for i in range(self.n_slices):
            # For the last slice, include equality on the right edge.
            if i == self.n_slices - 1:
                indices = np.where((y >= slice_edges[i]) & (y <= slice_edges[i+1]))[0]
            else:
                indices = np.where((y >= slice_edges[i]) & (y < slice_edges[i+1]))[0]
            count = len(indices)
            if count == 0:
                # Skip empty slices
                continue
            slice_counts.append(count)
            slice_means.append(np.mean(X_centered[indices, :], axis=0))
        
        if len(slice_means) == 0:
            raise ValueError("No valid slices were found. Check the distribution of y.")
        
        # Compute the weighted covariance of the slice means.
        cov_E = np.zeros((n_features, n_features))
        for count, m in zip(slice_counts, slice_means):
            cov_E += (count / n_samples) * np.outer(m, m)
        
        # Solve the generalized eigenvalue problem: cov_E v = λ cov_X v.
        eigvals, eigvecs = scipy.linalg.eigh(cov_E, cov_X)
        
        # Sort eigenvalues (and corresponding eigenvectors) in descending order.
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_indices]
        eigvecs = eigvecs[:, sorted_indices]
        
        # Check if n_directions is feasible given the effective rank.
        effective_rank = np.sum(eigvals > self.reg)
        if self.n_directions > effective_rank:
            raise ValueError("n_directions is greater than the effective rank of cov_E. "
                             f"Effective rank: {effective_rank}, requested: {self.n_directions}")
        
        # Select the top n_directions eigenvectors.
        self.directions_ = eigvecs[:, :self.n_directions]
        
        return self
    
    def transform(self, X):
        check_is_fitted(self, 'directions_')
        X = check_array(X)
        return np.dot(X, self.directions_)
