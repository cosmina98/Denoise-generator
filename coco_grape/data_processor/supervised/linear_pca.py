import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

class LinearPCA(BaseEstimator, TransformerMixin):
    """
    A transformer that:
      1. Fits a linear regressor on (X, y)
      2. Uses the normalized regression coefficients as the first axis
      3. Projects X onto the orthogonal complement of that direction and fits a PCA
      4. Transforms data so that the first component is the projection along the linear fit,
         and subsequent components are the PCA components.
         
    Parameters
    ----------
    n_components : int, default=2
        Total number of components to return. The first component is always the linear fit
        direction; if n_components > 1, then the remaining (n_components - 1) components are
        derived from PCA.
    """
    
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y):
        """
        Fit the transformer on the data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. If multi-dimensional, the norm of each row is used.
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Convert y to a NumPy array if not already, and if multi-dimensional, use the row-wise norm.
        y = np.array(y)
        if y.ndim > 1:
            y = np.linalg.norm(y, axis=1)
        
        # Fit linear regression and obtain coefficients
        self.linear_regressor_ = LinearRegression().fit(X, y)
        coef = self.linear_regressor_.coef_
        
        # Normalize the coefficient vector to get the first axis (normal direction)
        norm = np.linalg.norm(coef)
        if norm == 0:
            raise ValueError("The linear regression coefficients have zero norm; cannot determine a direction.")
        self.normal_ = coef / norm
        
        # Project X onto the orthogonal complement of self.normal_
        proj = np.outer(np.dot(X, self.normal_), self.normal_)
        X_orth = X - proj
        
        # If more than 1 component is requested, fit PCA on the residual space
        if self.n_components > 1:
            self.pca_ = PCA(n_components=self.n_components - 1)
            self.pca_.fit(X_orth)
        else:
            self.pca_ = None
        
        return self

    def transform(self, X):
        """
        Transform the data into the new component space.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to be transformed.
            
        Returns
        -------
        X_new : array of shape (n_samples, n_components)
            Transformed data where:
                - Column 0 is the projection along the linear regression normal.
                - Columns 1... are the PCA projections on the residual space (if applicable).
        """
        # Component 1: projection on the linear regression normal
        comp1 = np.dot(X, self.normal_).reshape(-1, 1)
        
        if self.n_components > 1:
            # Remove the component along the normal direction
            proj = np.outer(np.dot(X, self.normal_), self.normal_)
            X_orth = X - proj
            
            # Transform using the fitted PCA on the orthogonal subspace
            comp_rest = self.pca_.transform(X_orth)
            return np.hstack([comp1, comp_rest])
        else:
            return comp1
