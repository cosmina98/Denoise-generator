import time
from typing import Any, Dict, Iterable, Optional

import numpy as np
from scipy import linalg
from scipy import sparse
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import polynomial_kernel


def _to_numpy_2d(array: Any) -> np.ndarray:
    """Convert sparse/torch/numpy inputs into a dense float64 2D numpy array."""
    if array is None:
        raise ValueError("Input array cannot be None.")

    if sparse.issparse(array):
        array = array.toarray()

    try:
        import torch

        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
    except Exception:
        pass

    array = np.asarray(array)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D input, got shape={array.shape}")
    return array.astype(np.float64, copy=False)


class MMDEvaluation:
    """Notebook-compatible MMD evaluator (RBF or linear)."""

    def __init__(self, kernel: str = "rbf", sigma: str = "range", multiplier: str = "mean"):
        if multiplier == "mean":
            self.__get_sigma_mult_factor = self.__mean_pairwise_distance
        elif multiplier == "median":
            self.__get_sigma_mult_factor = self.__median_pairwise_distance
        else:
            self.__get_sigma_mult_factor = lambda *args, **kwargs: 1.0

        if "rbf" in kernel:
            if sigma == "range":
                self.base_sigmas = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0], dtype=float)
                self.name = f"mmd_rbf_{multiplier}"
            elif sigma == "one":
                self.base_sigmas = np.array([1.0], dtype=float)
                self.name = f"mmd_rbf_single_{multiplier}"
            else:
                raise ValueError(f"Invalid sigma: {sigma}")
            self.evaluate = self.calculate_mmd_rbf_quadratic
        elif "linear" in kernel:
            self.evaluate = self.calculate_mmd_linear_kernel
        else:
            raise ValueError(f"Invalid kernel: {kernel}")

    def __get_pairwise_distances(self, generated_dataset: np.ndarray, reference_dataset: np.ndarray) -> np.ndarray:
        return pairwise_distances(reference_dataset, generated_dataset, metric="euclidean", n_jobs=8) ** 2

    def __mean_pairwise_distance(self, dists_gr: np.ndarray) -> float:
        return float(np.sqrt(dists_gr.mean()))

    def __median_pairwise_distance(self, dists_gr: np.ndarray) -> float:
        return float(np.sqrt(np.median(dists_gr)))

    def get_sigmas(self, dists_gr: np.ndarray) -> np.ndarray:
        mult_factor = self.__get_sigma_mult_factor(dists_gr)
        return self.base_sigmas * mult_factor

    def calculate_mmd_rbf_quadratic(
        self, generated_dataset: Optional[np.ndarray] = None, reference_dataset: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        generated_dataset = _to_numpy_2d(generated_dataset)
        reference_dataset = _to_numpy_2d(reference_dataset)

        gg = self.__get_pairwise_distances(generated_dataset, generated_dataset)
        gr = self.__get_pairwise_distances(generated_dataset, reference_dataset)
        rr = self.__get_pairwise_distances(reference_dataset, reference_dataset)

        max_mmd = 0.0
        sigmas = self.get_sigmas(gr)
        for sigma in sigmas:
            gamma = 1.0 / (2.0 * (sigma ** 2 + 1e-12))
            k_gr = np.exp(-gamma * gr)
            k_gg = np.exp(-gamma * gg)
            k_rr = np.exp(-gamma * rr)
            mmd = float(k_gg.mean() + k_rr.mean() - 2.0 * k_gr.mean())
            max_mmd = max(max_mmd, mmd)
        return {self.name: max_mmd}

    def calculate_mmd_linear_kernel(
        self, generated_dataset: Optional[np.ndarray] = None, reference_dataset: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        generated_dataset = _to_numpy_2d(generated_dataset)
        reference_dataset = _to_numpy_2d(reference_dataset)
        g_bar = generated_dataset.mean(axis=0)
        r_bar = reference_dataset.mean(axis=0)
        z_bar = g_bar - r_bar
        mmd = float(z_bar.dot(z_bar))
        return {"mmd_linear": max(mmd, 0.0)}


class KIDEvaluation:
    """
    KID evaluator.
    Uses tensorflow_gan if available; otherwise falls back to a polynomial-kernel MMD estimate.
    """

    def evaluate(
        self, generated_dataset: Optional[np.ndarray] = None, reference_dataset: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        generated_dataset = _to_numpy_2d(generated_dataset)
        reference_dataset = _to_numpy_2d(reference_dataset)

        try:
            import tensorflow as tf
            import tensorflow_gan as tfgan

            gen_activations = tf.convert_to_tensor(generated_dataset, dtype=tf.float32)
            ref_activations = tf.convert_to_tensor(reference_dataset, dtype=tf.float32)
            kid = tfgan.eval.kernel_classifier_distance_and_std_from_activations(ref_activations, gen_activations)[
                0
            ].numpy()
            return {"kid": float(kid)}
        except Exception:
            return {"kid": float(_kid_polynomial_fallback(reference_dataset, generated_dataset))}


def _kid_polynomial_fallback(x: np.ndarray, y: np.ndarray, degree: int = 3, gamma: Optional[float] = None, coef0: float = 1.0) -> float:
    """Fallback KID estimator based on unbiased polynomial MMD^2."""
    if gamma is None:
        gamma = 1.0 / max(x.shape[1], 1)
    k_xx = polynomial_kernel(x, x, degree=degree, gamma=gamma, coef0=coef0)
    k_yy = polynomial_kernel(y, y, degree=degree, gamma=gamma, coef0=coef0)
    k_xy = polynomial_kernel(x, y, degree=degree, gamma=gamma, coef0=coef0)

    n = x.shape[0]
    m = y.shape[0]
    if n < 2 or m < 2:
        return float("nan")
    term_xx = (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
    term_yy = (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
    term_xy = 2.0 * k_xy.mean()
    return max(float(term_xx + term_yy - term_xy), 0.0)


class FIDEvaluation:
    def evaluate(
        self, generated_dataset: Optional[np.ndarray] = None, reference_dataset: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        generated_dataset = _to_numpy_2d(generated_dataset)
        reference_dataset = _to_numpy_2d(reference_dataset)
        mu_ref, cov_ref = self.__calculate_dataset_stats(reference_dataset)
        mu_gen, cov_gen = self.__calculate_dataset_stats(generated_dataset)
        fid = self.compute_fid(mu_ref, mu_gen, cov_ref, cov_gen)
        return {"fid": float(fid)}

    def __calculate_dataset_stats(self, activations: np.ndarray):
        mu = np.mean(activations, axis=0)
        cov = np.cov(activations, rowvar=False)
        return mu, cov

    def compute_fid(self, mu1: np.ndarray, mu2: np.ndarray, cov1: np.ndarray, cov2: np.ndarray, eps: float = 1e-6) -> float:
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(cov1 @ cov2, disp=False)
        if not np.isfinite(covmean).all():
            cov1 = cov1 + np.eye(cov1.shape[0]) * eps
            cov2 = cov2 + np.eye(cov2.shape[0]) * eps
            covmean, _ = linalg.sqrtm(cov1 @ cov2, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        tr_covmean = np.trace(covmean)
        return float(diff.dot(diff) + np.trace(cov1) + np.trace(cov2) - 2.0 * tr_covmean)


class PRDCEvaluation:
    def __init__(self, use_pr: bool = False):
        self.use_pr = use_pr

    def evaluate(
        self, generated_dataset: Optional[np.ndarray] = None, reference_dataset: Optional[np.ndarray] = None, nearest_k: int = 5
    ) -> Dict[str, float]:
        generated_dataset = _to_numpy_2d(generated_dataset)
        reference_dataset = _to_numpy_2d(reference_dataset)
        real_nnd = self.__compute_nearest_neighbour_distances(reference_dataset, nearest_k)
        dist_rf = self.__compute_pairwise_distance(reference_dataset, generated_dataset)

        if self.use_pr:
            fake_nnd = self.__compute_nearest_neighbour_distances(generated_dataset, nearest_k)
            precision = float((dist_rf <= np.expand_dims(real_nnd, axis=1)).any(axis=0).mean())
            recall = float((dist_rf <= np.expand_dims(fake_nnd, axis=0)).any(axis=1).mean())
            f1_pr = 2.0 / ((1.0 / (precision + 1e-5)) + (1.0 / (recall + 1e-5)))
            return {"precision": precision, "recall": recall, "f1_pr": float(f1_pr)}

        density = float((1.0 / float(nearest_k)) * (dist_rf <= np.expand_dims(real_nnd, axis=1)).sum(axis=0).mean())
        coverage = float((dist_rf.min(axis=1) <= real_nnd).mean())
        f1_dc = 2.0 / ((1.0 / (density + 1e-5)) + (1.0 / (coverage + 1e-5)))
        return {"density": density, "coverage": coverage, "f1_dc": float(f1_dc)}

    def __compute_pairwise_distance(self, data_x: np.ndarray, data_y: Optional[np.ndarray] = None) -> np.ndarray:
        return pairwise_distances(data_x, data_y, metric="euclidean", n_jobs=8)

    def __get_kth_value(self, unsorted: np.ndarray, k: int, axis: int = -1) -> np.ndarray:
        indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
        k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
        return k_smallests.max(axis=axis)

    def __compute_nearest_neighbour_distances(self, input_features: np.ndarray, nearest_k: int) -> np.ndarray:
        distances = self.__compute_pairwise_distance(input_features)
        return self.__get_kth_value(distances, k=nearest_k + 1, axis=-1)


# Notebook-compatible alias
prdcEvaluation = PRDCEvaluation


def evaluate_embedding_distributions(
    real_embeddings: Any,
    generated_embeddings: Any,
    nearest_k: int = 5,
    include_timing: bool = True,
) -> Dict[str, float]:
    """
    Compute FID/KID/PRDC/MMD on two embedding matrices.
    """
    real = _to_numpy_2d(real_embeddings)
    generated = _to_numpy_2d(generated_embeddings)

    evaluators: Iterable[Any] = [
        FIDEvaluation(),
        KIDEvaluation(),
        PRDCEvaluation(use_pr=True),
        PRDCEvaluation(use_pr=False),
        MMDEvaluation(kernel="rbf", sigma="range", multiplier="mean"),
        MMDEvaluation(kernel="linear"),
    ]

    out: Dict[str, float] = {}
    for evaluator in evaluators:
        name = evaluator.__class__.__name__.lower().replace("evaluation", "")
        t0 = time.time()
        try:
            # PRDC-style evaluators accept nearest_k; others do not.
            metrics = evaluator.evaluate(
                generated_dataset=generated,
                reference_dataset=real,
                nearest_k=nearest_k,
            )
        except TypeError:
            metrics = evaluator.evaluate(
                generated_dataset=generated,
                reference_dataset=real,
            )
        dt = time.time() - t0
        out.update({k: float(v) for k, v in metrics.items()})
        if include_timing:
            out[f"{name}_time"] = dt
    return out
