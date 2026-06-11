from typing import List, Any, Optional, Tuple
import numpy as np

class ConditionalNodeGeneratorBase:
    """
    Abstract base class for conditional node generators.

    All conditional node generators should inherit from this class and implement
    the fit and predict methods.

    Methods
    -------
    fit(
        node_encodings_list: List[np.ndarray],
        conditional_graph_encodings: Any,
        edge_pairs: Optional[List[Tuple[int, int, int]]] = None,
        edge_targets: Optional[np.ndarray] = None,
        node_mask: Optional[np.ndarray] = None
    )
        Fit the model to training data.

    predict(y)
        Generate samples conditioned on y.
    """

    def fit(
        self,
        node_encodings_list: List[np.ndarray],
        conditional_graph_encodings: Any,
        edge_pairs: Optional[List[Tuple[int, int, int]]] = None,
        edge_targets: Optional[np.ndarray] = None,
        node_mask: Optional[np.ndarray] = None
    ):
        raise NotImplementedError("fit() must be implemented by subclasses.")

    def predict(self, y):
        raise NotImplementedError("predict() must be implemented by subclasses.")