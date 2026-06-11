import numpy as np
from coco_grape.module.hash_graph import hash_graph
from joblib import Parallel, delayed
from scipy.sparse import vstack  # For stacking sparse matrices
from typing import Any, Callable, List, Optional, Union, Dict
from scipy.sparse import lil_matrix, csr_matrix
from coco_grape.module.quotientgraph.definitions import graph_hash_label_function_factory
from coco_grape.module.quotientgraph.type import QuotientGraph

def vectorize(quotientgraph: "QuotientGraph", nbits: int = 10, return_dense: bool = True) -> Union[np.ndarray, csr_matrix]:
    """
    Returns an array of shape (n_nodes, n_features) representing the multisets
    associated with each base node in the base graph, leveraging the to_array method.
    Here, n_nodes is the number of nodes in the base graph (obtained via
    quotientgraph.preimage_graph) and n_features is 2**nbits.

    The base count matrix is obtained using quotientgraph.to_array().
    The first two columns of the resulting matrix are then modified:
        - Feature 0 (column 0) is set to a constant 1 (bias term).
        - Feature 1 (column 1) is set to the degree of the corresponding base node
          in the preimage graph.

    Before calling to_array, the function updates the quotient graph's label_function
    to use hash_graph with the desired nbits.

    Parameters:
        quotientgraph (QuotientGraph): The quotient graph instance.
        nbits (int): Number of bits for the hash space (number of features = 2**nbits).
        return_dense (bool): If True, returns a dense numpy array; if False, returns a csr_matrix.

    Returns:
        Union[np.ndarray, csr_matrix]: The matrix of counts with the special features added.

    Raises:
        ValueError: If nbits < 2, as column 0 and 1 are reserved.
    """
    if nbits < 2:
        raise ValueError("nbits must be at least 2 to accommodate bias and degree features.")

    # 1. Set the label function with the correct nbits attribute.
    #    This is crucial for to_array to infer the correct matrix dimensions.
    quotientgraph.label_function = graph_hash_label_function_factory(nbits=nbits)

    # 2. Call to_array to get the base count matrix (always returns csr_matrix).
    #    to_array internally calls apply_label_function.
    try:
        M_sparse = quotientgraph.to_array()
    except ValueError as e:
        # Re-raise with a more specific context if needed, or just let it propagate.
        raise ValueError(f"Failed to generate base array using to_array. Original error: {e}")

    # 3. Get preimage graph info AND degrees in the correct order
    base_graph = quotientgraph.preimage_graph
    # Ensure the node order matches the row order implicitly used by to_array
    base_nodes: List[Any] = list(base_graph.nodes())
    n = len(base_nodes)

    # Handle empty graph case where M_sparse might be (0, m)
    if n == 0:
        # Return the matrix as is, converted to dense if requested
        return M_sparse.toarray() if return_dense else M_sparse

    # Calculate degrees in the order corresponding to matrix rows
    degrees = np.array([base_graph.degree[node] for node in base_nodes])

    # 4. Modify columns 0 and 1 using vectorized/optimized operations
    if return_dense:
        M = M_sparse.toarray()
        M[:, 0] = 1           # Assign 1 to the entire first column
        M[:, 1] = degrees     # Assign degrees to the second column
    else:
        # Use LIL format for efficient column assignment in sparse matrices
        M_lil = M_sparse.tolil()
        M_lil[:, 0] = 1           # Assign 1 to the entire first column
        # Assign degrees to the second column (reshape needed for LIL column assignment)
        M_lil[:, 1] = degrees.reshape(-1, 1)
        M = M_lil.tocsr()       # Convert back to CSR

    return M

class QuotientGraphTransformer:
    def __init__(self, nbits: int, decomposition_function: Callable[[QuotientGraph], QuotientGraph],
                 return_dense: bool = True, n_jobs: int = -1) -> None:
        """
        Parameters:
            nbits (int): The nbits parameter used in vectorize (and consequently in hash_graph).
            decomposition_function (callable): A function that takes a QuotientGraph and returns a new 
                QuotientGraph after decomposition (e.g. add(node(), edge())).
            return_dense (bool): If True, vectorize returns a dense numpy array;
                otherwise, it returns a CSR (sparse) matrix.
            n_jobs (int): Number of jobs for parallel processing. Default is -1.
        """
        self.nbits = nbits
        self.decomposition_function = decomposition_function
        self.return_dense = return_dense
        self.n_jobs = n_jobs

    def fit(self, X: List[Any], y: Optional[Any] = None) -> "QuotientGraphTransformer":
        # No fitting necessary.
        return self

    def fit_transform(self, X: List[Any], y: Optional[Any] = None) -> Any:
        """
        Fit to data, then transform it.

        Parameters:
            X (list): A list of graphs.
            y: Ignored.
        
        Returns:
            A stacked array (or sparse matrix) representing the transformed graphs.
        """
        self.fit(X, y)
        return self.transform(X, y)

    def _process_graph(self, graph: Any) -> Any:
        # Create the QuotientGraph from the input graph using the provided graph.
        # The following call creates a QuotientGraph and populates its image_graph with a default node.
        qg = QuotientGraph(graph=graph)
        qg.create_default_image_node()
        # Apply the provided decomposition function.
        qg = self.decomposition_function(qg)
        # Vectorize the quotient graph.
        arr = vectorize(qg, nbits=self.nbits, return_dense=self.return_dense)
        # Sum over rows to get a single feature vector per graph.
        arr = arr.sum(axis=0)
        if not self.return_dense:             # i.e. we promised a sparse output
            arr = csr_matrix(arr)             # 1 × n_features CSR row
        return arr

    def transform(self, X: List[Any], y: Optional[Any] = None) -> Any:
        """
        Transform a list of graphs into a stacked feature array.
        
        Parameters:
            X (list): A list of graphs.
            y: Ignored.
            
        Returns:
            If return_dense is True, returns a stacked dense numpy array;
            otherwise, returns a stacked CSR sparse matrix.
        """
        arrays = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_graph)(graph) for graph in X
        )
        if self.return_dense:
            return np.stack(arrays)
        else:
            return vstack(arrays)

class QuotientGraphNodeTransformer:
    def __init__(self, nbits: int, decomposition_function: Callable[[QuotientGraph], QuotientGraph],
                 return_dense: bool = True, n_jobs: int = -1) -> None:
        """
        Parameters:
            nbits (int): The nbits parameter used in vectorize (and hash_graph).
            decomposition_function (callable): A function that takes a QuotientGraph and returns a new 
                QuotientGraph after decomposition.
            return_dense (bool): If True, vectorize returns a dense numpy array;
                otherwise, it returns a CSR (sparse) matrix.
            n_jobs (int): Number of jobs for parallel processing. Default is -1.
        """
        self.nbits = nbits
        self.decomposition_function = decomposition_function
        self.return_dense = return_dense
        self.n_jobs = n_jobs

    def fit(self, X: List[Any], y: Optional[Any] = None) -> "QuotientGraphNodeTransformer":
        # No fitting necessary.
        return self

    def fit_transform(self, X: List[Any], y: Optional[Any] = None) -> List[Any]:
        """
        Fit to data, then transform it.

        Parameters:
            X (list): A list of graphs.
            y: Ignored.
        
        Returns:
            A list of feature matrices (each either a dense numpy array or a CSR matrix),
            one for each input graph.
        """
        self.fit(X, y)
        return self.transform(X, y)

    def _process_graph(self, graph: Any) -> Any:
        # Create the QuotientGraph from the input graph.
        qg = QuotientGraph(graph=graph)
        qg.create_default_image_node()
        # Apply the provided decomposition function.
        qg = self.decomposition_function(qg)
        # Vectorize the quotient graph.
        arr = vectorize(qg, nbits=self.nbits, return_dense=self.return_dense)
        return arr

    def transform(self, X: List[Any], y: Optional[Any] = None) -> List[Any]:
        """
        Transform a list of graphs into a list of feature matrices.
        
        Parameters:
            X (list): A list of graphs.
            y: Ignored.
            
        Returns:
            A list of feature matrices (each either a dense numpy array or a CSR matrix),
            one for each input graph.
        """
        arrays = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process_graph)(graph) for graph in X
        )
        return arrays
