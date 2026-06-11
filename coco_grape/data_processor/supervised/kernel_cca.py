import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_array, check_is_fitted
import scipy.linalg

class KernelCCA(BaseEstimator, TransformerMixin):
    """
    Kernel Canonical Correlation Analysis (Kernel CCA) transformer.

    This transformer implements a kernelized nonlinear CCA. It first computes kernel
    matrices for two views (X and Y), centers them, whitens them via eigen-decomposition
    (with regularization), and then performs SVD on the cross-covariance in the whitened
    space. The result is a set of canonical projection directions for X and Y that capture
    the shared nonlinear structure between the two views.

    Parameters
    ----------
    n_components : int, default=2
        Number of canonical components (dimensions) to extract.
    kernel : str or callable, default='rbf'
        Kernel function to use. If a string, it is passed to
        sklearn.metrics.pairwise.pairwise_kernels.
    kernel_params : dict, default=None
        Dictionary of parameters to pass to the kernel function. For example, for the
        RBF kernel you might specify {'gamma': 0.1}. If 'gamma' is not provided, it will
        default to 1/n_features (separately for X and Y).
    reg : float, default=1e-5
        Regularization parameter added to the eigenvalues during whitening.
    tol : float, default=1e-12
        Tolerance below which eigenvalues are considered negligible.

    Attributes
    ----------
    X_fit_ : array, shape (n_samples, n_features_X)
        Training data for view X.
    Y_fit_ : array, shape (n_samples, n_features_Y)
        Training data for view Y.
    Kx_fit_ : array, shape (n_samples, n_samples)
        The original (uncentered) training kernel matrix computed on X.
    Kx_fit_row_mean_ : array, shape (n_samples,)
        The row mean of the training kernel matrix for X.
    Kx_fit_total_mean_ : float
        The overall mean of the training kernel matrix for X.
    Wx_ : array, shape (n_samples, n_components)
        The projection matrix for view X.
    Wy_ : array, shape (n_samples, n_components)
        The projection matrix for view Y.
    canonical_correlations_ : array, shape (n_components,)
        The singular values (canonical correlations) corresponding to the components.
    kernel_params_X_ : dict
        The kernel parameters used for computing the kernel matrix on X.
    kernel_params_Y_ : dict
        The kernel parameters used for computing the kernel matrix on Y.

    References
    ----------
    Hardoon, D. R., Szedmak, S., & Shawe-Taylor, J. (2004). Canonical correlation analysis:
    An overview with application to learning methods. Neural computation, 16(12), 2639-2664.
    """
    def __init__(self, n_components=2, kernel='rbf', kernel_params=None, reg=1e-5, tol=1e-12):
        self.n_components = n_components
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.reg = reg
        self.tol = tol

    def _center_kernel(self, K):
        """Center a kernel matrix."""
        n = K.shape[0]
        one_n = np.ones((n, n)) / n
        return K - one_n @ K - K @ one_n + one_n @ K @ one_n

    def fit(self, X, Y):
        """
        Fit the Kernel CCA model using training data X and Y.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features_X)
            Data matrix for view X.
        Y : array-like, shape (n_samples, n_features_Y)
            Data matrix for view Y.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = check_array(X)
        Y = check_array(Y)
        self.X_fit_ = X
        self.Y_fit_ = Y
        n_samples, n_features_X = X.shape
        _, n_features_Y = Y.shape

        # Prepare kernel parameter dictionaries for X and Y.
        if self.kernel_params is None:
            base_params = {}
        else:
            base_params = self.kernel_params.copy()
        self.kernel_params_X_ = base_params.copy()
        self.kernel_params_Y_ = base_params.copy()

        if 'gamma' not in self.kernel_params_X_:
            self.kernel_params_X_['gamma'] = 1.0 / n_features_X
        if 'gamma' not in self.kernel_params_Y_:
            self.kernel_params_Y_['gamma'] = 1.0 / n_features_Y

        # Compute kernel matrices for X and Y using the provided kernel and parameters.
        Kx = pairwise_kernels(X, metric=self.kernel, **self.kernel_params_X_)
        Ky = pairwise_kernels(Y, metric=self.kernel, **self.kernel_params_Y_)

        # Save the original training kernel for X (needed for centering new data).
        self.Kx_fit_ = Kx.copy()
        self.Kx_fit_row_mean_ = np.mean(Kx, axis=0)
        self.Kx_fit_total_mean_ = np.mean(Kx)

        # Center the kernel matrices.
        Kx_centered = self._center_kernel(Kx)
        Ky_centered = self._center_kernel(Ky)

        # Eigen-decomposition of the centered kernels.
        Sx, Ux = np.linalg.eigh(Kx_centered)
        Sy, Uy = np.linalg.eigh(Ky_centered)

        # Regularize eigenvalues.
        Sx_reg = Sx + self.reg
        Sy_reg = Sy + self.reg

        # Filter eigenvalues above tolerance.
        idx_x = Sx_reg > self.tol
        idx_y = Sy_reg > self.tol
        if np.sum(idx_x) < self.n_components or np.sum(idx_y) < self.n_components:
            raise ValueError("Not enough non-negligible eigenvalues to extract the desired number of components.")

        Ux = Ux[:, idx_x]
        Sx_reg = Sx_reg[idx_x]
        Uy = Uy[:, idx_y]
        Sy_reg = Sy_reg[idx_y]

        # Whitening: compute the whitened representation.
        # Ax = Ux * (1/sqrt(Sx_reg)); using reshape for proper broadcasting.
        Ax = Ux / np.sqrt(Sx_reg.reshape(1, -1))
        Ay = Uy / np.sqrt(Sy_reg.reshape(1, -1))

        # Compute the cross-covariance in the whitened kernel space.
        M = Ax.T @ Ky_centered @ Ay

        # Perform SVD on the cross-covariance matrix.
        U_svd, s, Vt = np.linalg.svd(M)
        # Compute canonical directions in the whitened space.
        Wx = Ax @ U_svd[:, :self.n_components]
        Wy = Ay @ Vt.T[:, :self.n_components]

        self.Wx_ = Wx
        self.Wy_ = Wy
        self.canonical_correlations_ = s[:self.n_components]

        return self

    def transform(self, X, view='X'):
        """
        Transform new data using the fitted Kernel CCA projection.

        Parameters
        ----------
        X : array-like, shape (n_samples_new, n_features)
            New data for the specified view.
        view : str, default='X'
            Which view to transform; either 'X' or 'Y'.

        Returns
        -------
        X_proj : array, shape (n_samples_new, n_components)
            The projected data in the canonical space.
        """
        check_is_fitted(self, 'Wx_')
        X = check_array(X)

        if view == 'X':
            # Compute kernel between new X data and training X data.
            K_new = pairwise_kernels(X, self.X_fit_, metric=self.kernel, **self.kernel_params_X_)
            # Center the new kernel using the training kernel statistics.
            # Standard centering: K_new_centered = K_new - mean(K_fit, axis=0) - mean(K_new, axis=1, keepdims=True) + mean(K_fit)
            K_new_centered = K_new - self.Kx_fit_row_mean_ - np.mean(K_new, axis=1, keepdims=True) + self.Kx_fit_total_mean_
            return K_new_centered @ self.Wx_
        elif view == 'Y':
            # Compute kernel between new Y data and training Y data.
            Ky_new = pairwise_kernels(X, self.Y_fit_, metric=self.kernel, **self.kernel_params_Y_)
            # Compute training Y kernel statistics.
            Ky_fit = pairwise_kernels(self.Y_fit_, metric=self.kernel, **self.kernel_params_Y_)
            Ky_fit_row_mean = np.mean(Ky_fit, axis=0)
            Ky_fit_total_mean = np.mean(Ky_fit)
            Ky_new_centered = Ky_new - Ky_fit_row_mean - np.mean(Ky_new, axis=1, keepdims=True) + Ky_fit_total_mean
            return Ky_new_centered @ self.Wy_
        else:
            raise ValueError("view must be either 'X' or 'Y'.")
