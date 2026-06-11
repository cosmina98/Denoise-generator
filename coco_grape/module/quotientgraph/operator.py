#====================================================================================================
# SECTIONS:
#====================================================================================================
# UTILITIES
# HIGHER ORDER OPERATORS
# CONDITIONAL OPERATORS
# ITERATION OPERATORS
# UNARY OPERATORS
# EDGE OPERATORS
# FILTER OPERATORS
# BINARY OPERATORS
# PRE-IMAGE GRAPH OPERATORS
# SCALAR OPERATORS
#====================================================================================================

import networkx as nx
import numpy as np
import copy
from toolz import curry
from typing import Callable, Any, Dict, List, Tuple, Optional, Union
import inspect
import itertools
from itertools import combinations, product
from networkx.algorithms.community import kernighan_lin_bisection
from coco_grape.module.quotientgraph.type import QuotientGraph


#====================================================================================================
# UTILITIES
#====================================================================================================

def value_to_2tuple(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """
    Converts a value to a tuple of two identical integers.
    """
    if isinstance(value, tuple):
        return value
    elif isinstance(value, int):
        return (value, value)
    else:
        raise ValueError(f"Invalid value: {value}. Expected an int or a tuple.")
    

def build_meta_from_function_context(exclude_keys: Tuple[str, ...] = ("quotient_graph",)) -> Dict[str, Any]:
    """
    Attempts to construct a meta dictionary including:
    - source_function: inferred from the calling function (even if curried)
    - params: all parameters except excluded ones (like quotient_graph)

    Returns:
        dict: metadata with 'source_function' and 'params'
    """
    frame = inspect.currentframe()
    caller_frame = frame.f_back

    # Step 1: Get parameter values (excluding things like 'quotient_graph')
    args, _, _, values = inspect.getargvalues(caller_frame)
    params_dict = {k: values[k] for k in args if k not in exclude_keys}

    # Step 2: Try to infer function name from calling frame
    source_function = "unknown"
    try:
        func_obj = values.get(args[0], None)
        source_function = inspect.unwrap(func_obj).__name__
    except Exception:
        # fallback: use the code object
        try:
            source_function = caller_frame.f_code.co_name
        except Exception:
            pass

    return {
        "source_function": source_function,
        "params": params_dict
    }


#====================================================================================================
# HIGHER ORDER OPERATORS
#====================================================================================================
def add(*decomposition_functions):
    """
    Build an operator that adds (unions) the outputs of multiple decomposition functions.
    """
    def composed(quotient_graph: 'QuotientGraph'):
        """Additive composition of decomposition outputs over a shared preimage graph.
        Summary
            Given one QuotientGraph, run several decomposition functions on it and
            add their resulting quotient graphs together using the graph’s `+`
            semantics, producing a single aggregate QuotientGraph.

        Semantics
            - Input QG state:
                Uses the provided `quotient_graph` as the common source for all
                decompositions; reads its `label_function`, `attribute_function`,
                and `edge_function` to seed the result.
            - Output QG state:
                Returns a new QuotientGraph whose image graph is the additive
                combination of all per-function outputs; the preimage graph is
                aligned with the input (as defined by `__add__` on QuotientGraph).
            - Determinism:
                Deterministic given the input graph and the ordered list of
                decomposition functions.

        Parameters
            quotient_graph : QuotientGraph
                The graph to be decomposed by each function in `decomposition_functions`.

        Returns
            QuotientGraph
                A single quotient graph equivalent to
                `func_1(qg) + func_2(qg) + ... + func_m(qg)`, with operator settings
                (label/attribute/edge functions) preserved from the input.

        Algorithm
            1. Initialise an empty/base QuotientGraph carrying the input’s functional
               settings (label/attribute/edge functions).
            2. For each decomposition function `f` in order:
               a) Compute `f(quotient_graph)` → a QuotientGraph.
               b) Add it to the running `result` via `result = result + f(quotient_graph)`.
            3. Return the accumulated `result`.

        Complexity
            Let m be the number of functions, T_f the cost of each decomposition,
            and A the cost of a `+` merge:
              - Time:  Σ T_f  +  (m − 1)·A
              - Memory: proportional to the size of the union of image-node sets and
                any metadata/materialised edges produced by the functions.

        Interactions
            - Pairs naturally with decomposition functions such as:
              `connected_components_decomposition`, `cycle_decomposition`,
              `clique_decomposition`, `filter_by_*`.
            - Often followed by consolidation steps like `deduplicate`, `merge`,
              or `project` to normalise/aggregate overlapping image nodes.
            - Order matters if `+` is not strictly commutative/associative in the
              implementation (e.g., metadata precedence rules).

        Examples
            # Combine connected components and simple cycles into one operator
            cc_plus_cycles = add(connected_components_decomposition,
                                 cycle_decomposition)
            qg_out = cc_plus_cycles(qg_in)

            # Add several filters
            fused = add(filter_by_label('Person'),
                        filter_by_attribute('weight', '>', 70.0))(qg_in)

        Domain Analogies
            - Chemistry: union of detected motifs (e.g., rings + functional groups).
            - Social networks: merge of multiple community detections (by interests,
              by interaction frequency) into one layer.
            - Vision: combine edge maps from different detectors into a single feature layer.

        Failure Modes
            - Empty input list (`add()` with no functions) returns a base/empty
              quotient graph carrying only operator settings.
            - Incompatible outputs: if `__add__` requires matched preimage graphs
              or settings, and a decomposition violates those assumptions, a merge
              error may occur.
            - Non-idempotent `+`: repeated addition of overlapping image nodes may
              duplicate structures unless `__add__` handles deduplication.
            - Exceptions raised in any decomposition function propagate to the caller.
        """
        # Preserve the functional settings from the input graph
        base = QuotientGraph(
            label_function=quotient_graph.label_function,
            attribute_function=quotient_graph.attribute_function,
            edge_function=quotient_graph.edge_function,
        )

        result = base
        for func in decomposition_functions:
            result = result + func(quotient_graph)
        return result

    composed.__name__ = "add"
    composed.decomposition_functions = decomposition_functions
    composed.operator_type = "add"
    return composed

#--------------------------------------------------------------------------------
def compose(*decomposition_functions):
    def composed(quotient_graph: 'QuotientGraph'):
        """Reverse-order composition of decomposition functions on a QuotientGraph.
        Summary
            Applies a chain of decomposition functions to a quotient graph,
            evaluating them from right to left (last provided function runs first).

        Semantics
            - Input QG state:
                Receives one QuotientGraph as input.
            - Output QG state:
                Returns the transformed QuotientGraph after sequentially applying
                all decomposition functions in reversed order.
            - Determinism:
                Deterministic given the input and fixed function chain.

        Parameters
            quotient_graph : QuotientGraph
                The input graph to transform through the composition chain.

        Returns
            QuotientGraph
                The final graph after applying all functions in reverse order.

        Algorithm
            1. Initialise with the input `quotient_graph`.
            2. For each function f in `reversed(decomposition_functions)`:
                quotient_graph = f(quotient_graph)
            3. Return the final `quotient_graph`.

        Complexity
            Let m = number of functions, and T_f = cost of each:
              - Time: Σ T_f
              - Memory: governed by the largest intermediate QuotientGraph.

        Interactions
            - Pairs with decomposition primitives (`cycle`, `clique`, `filter_by_*`).
            - Often wrapped in higher-level pipelines for symbolic XML composition.
            - Useful for operators that require preprocessing by another function
              before they run (e.g. filtering before merging).

        Examples
            # Compose cycle detection after connected components
            cc_then_cycle = compose(cycle_decomposition,
                                    connected_components_decomposition)
            qg_out = cc_then_cycle(qg_in)
            # Equivalent to cycle_decomposition(connected_components_decomposition(qg_in))

        Domain Analogies
            - Mathematics: function composition f∘g, evaluated right-to-left.
            - Image processing: apply a blur after resizing.
            - Social networks: detect cliques inside already-partitioned communities.

        Failure Modes
            - Empty composition chain returns the input graph unchanged.
            - Any exception in a function aborts the chain.
            - Ordering mistakes: easy to confuse with `forward_compose`
              since semantics differ only by evaluation order.
        """
        for func in reversed(decomposition_functions):
            quotient_graph = func(quotient_graph)
        return quotient_graph

    composed.__name__ = "compose"
    composed.chain = decomposition_functions
    composed.operator_type = "compose"  # Mark as a compose operator
    return composed

def forward_compose(*decomposition_functions):
    def composed(quotient_graph: 'QuotientGraph'):
        """Forward-order composition of decomposition functions on a QuotientGraph.
        Summary
            Applies a chain of decomposition functions to a quotient graph,
            evaluating them from left to right (first provided function runs first).

        Semantics
            - Input QG state:
                Takes one QuotientGraph as the starting point.
            - Output QG state:
                Returns the transformed QuotientGraph after applying each function
                in forward order.
            - Determinism:
                Deterministic given the input and fixed function chain.

        Parameters
            quotient_graph : QuotientGraph
                The input graph to transform through the chain.

        Returns
            QuotientGraph
                The final graph after applying all functions in left-to-right order.

        Algorithm
            1. Initialise with the input `quotient_graph`.
            2. For each function f in `decomposition_functions`:
                quotient_graph = f(quotient_graph)
            3. Return the final `quotient_graph`.

        Complexity
            Let m = number of functions, and T_f = cost of each:
              - Time: Σ T_f
              - Memory: governed by the largest intermediate QuotientGraph.

        Interactions
            - Useful in pipelines where function order mirrors natural workflow.
            - Combines well with `filter_by_*` then `merge` style operations.
            - More intuitive than `compose` when read left-to-right.

        Examples
            # Apply connected components first, then cycle detection
            cc_then_cycle = forward_compose(connected_components_decomposition,
                                            cycle_decomposition)
            qg_out = cc_then_cycle(qg_in)
            # Equivalent to cycle_decomposition(connected_components_decomposition(qg_in))

        Domain Analogies
            - Functional programming: pipeline operator |> in F# or Elixir.
            - Image processing: resize → blur → sharpen (in order).
            - Social networks: partition into communities, then analyse cliques.

        Failure Modes
            - Empty composition chain returns the input graph unchanged.
            - Exceptions propagate immediately from any function in the chain.
            - Can be confused with `compose` if users forget directionality.
        """
        for func in decomposition_functions:
            quotient_graph = func(quotient_graph)
        return quotient_graph

    composed.__name__ = "forward_compose"
    composed.chain = decomposition_functions
    composed.operator_type = "forward_compose"  # Mark as a forward_compose operator
    return composed

#--------------------------------------------------------------------------------
def compose_product(combiner, *decomposition_functions):
    def composed(quotient_graph: 'QuotientGraph'):
        """Parallel product composition of decomposition functions with a combiner.
        Summary
            Applies multiple decomposition functions independently to the same
            input QuotientGraph, then fuses their outputs using a user-supplied
            `combiner` function.

        Semantics
            - Input QG state:
                One QuotientGraph, used as input to each decomposition function.
            - Output QG state:
                A new QuotientGraph returned by `combiner(*results)` where each
                result is the output of one decomposition function.
            - Determinism:
                Deterministic given deterministic decomposition functions and combiner.

        Parameters
            combiner : Callable[[QuotientGraph, ...], QuotientGraph]
                A function that takes all decomposition outputs and produces a
                single QuotientGraph (e.g. via addition, merge, intersection).
            decomposition_functions : tuple[Callable[[QuotientGraph], QuotientGraph], ...]
                The decomposition functions to apply in parallel.

        Returns
            QuotientGraph
                The combined result of applying all functions to the input
                QuotientGraph and reducing them with `combiner`.

        Algorithm
            1. For each f in `decomposition_functions`, compute f(quotient_graph).
            2. Collect all results into a list.
            3. Return combiner(*results).

        Complexity
            Let m = number of functions, T_f = cost of each, and C = cost of combiner:
              - Time: Σ T_f + C
              - Memory: sum of sizes of intermediate QuotientGraphs + size of combined output.

        Interactions
            - Natural generalisation of `add`: use `combiner = operator.add`.
            - Can encode intersections, unions, or custom fusions depending on
              combiner.
            - Useful for multi-view decomposition (e.g. apply different structural
              detectors and then fuse).

        Examples
            # Product with addition as combiner (union of outputs)
            union_op = compose_product(lambda a, b: a + b,
                                       cycle_decomposition,
                                       connected_components_decomposition)
            qg_out = union_op(qg_in)

            # Product with custom intersection combiner
            def intersect(a, b): return a & b
            intersect_op = compose_product(intersect,
                                           clique_decomposition,
                                           filter_by_label("Person"))
            qg_out = intersect_op(qg_in)

        Domain Analogies
            - Chemistry: detect rings and functional groups separately, then fuse.
            - Social networks: compute communities by geography and by activity,
              then combine.
            - Vision: apply edge detector and texture detector, then overlay results.

        Failure Modes
            - Empty function list: `compose_product` returns `combiner()` with no
              args → usually raises a TypeError.
            - Incompatible combiner: if outputs are not compatible with the combiner
              (e.g. types mismatch), runtime error.
            - Expensive combiner: cost may dominate total runtime if reduction is heavy.
        """
        results = [func(quotient_graph) for func in decomposition_functions]
        return combiner(*results)

    composed.__name__ = "compose_product"
    composed.decomposition_functions = decomposition_functions
    composed.operator_type = "product"
    composed.combiner = combiner
    return composed

#====================================================================================================
# CONDITIONAL OPERATORS
#====================================================================================================

@curry
def if_then_else(
    quotient_graph: 'QuotientGraph',
    predicate: Callable[['QuotientGraph'], bool],
    then_function: Callable[['QuotientGraph'], 'QuotientGraph'],
    else_function: Callable[['QuotientGraph'], 'QuotientGraph']
) -> 'QuotientGraph':
    """Conditional branching operator for QuotientGraph transformations.
    Summary
        Evaluates a predicate on the input QuotientGraph and applies either
        `then_function` (if True) or `else_function` (if False).

    Semantics
        - Input QG state:
            The given `quotient_graph` is passed unchanged to both predicate
            and whichever branch function is chosen.
        - Output QG state:
            Result of applying either `then_function` or `else_function` to
            the input graph.
        - Determinism:
            Deterministic if predicate and branch functions are deterministic.

    Parameters
        quotient_graph : QuotientGraph
            Input graph on which the predicate is evaluated.
        predicate : Callable[[QuotientGraph], bool]
            Function deciding which branch to execute.
        then_function : Callable[[QuotientGraph], QuotientGraph]
            Transformation applied if predicate returns True.
        else_function : Callable[[QuotientGraph], QuotientGraph]
            Transformation applied if predicate returns False.

    Returns
        QuotientGraph
            Output of the selected transformation function.

    Algorithm
        1. Evaluate `predicate(quotient_graph)`.
        2. If result is True, return `then_function(quotient_graph)`.
        3. Otherwise, return `else_function(quotient_graph)`.

    Complexity
        - Time: cost(predicate) + cost(branch) for whichever branch is chosen.
        - Memory: size of branch output.
        - No extra overhead beyond predicate and chosen function.

    Interactions
        - Integrates into pipelines built with `forward_compose` or `compose`
          to enable conditional flows.
        - Combines naturally with decomposition selectors like
          `filter_by_*` or `merge`.
        - Can emulate “switch”-style logic when nested.

    Examples
        # Branch by number of image nodes
        workflow = forward_compose(
            connected_component(),
            if_then_else(
                predicate=lambda qg: qg.image_graph.number_of_nodes() > 10,
                then_function=merge(),
                else_function=cycle()
            ),
            filter_by_number_of_nodes(number_of_nodes=(4, 10))
        )
        qg_out = workflow(qg_in)

    Domain Analogies
        - Programming: the classic `if ... then ... else ...` control structure.
        - Chemistry: apply a different reaction depending on whether a molecule
          exceeds a threshold property.
        - Social networks: choose community detection algorithm based on graph size.

    Failure Modes
        - Predicate exceptions: if predicate raises an error, execution halts.
        - Branch mismatch: if `then_function` or `else_function` produce outputs
          incompatible with downstream operators, pipeline may fail.
        - Non-deterministic predicates lead to unpredictable branching.
    """
    if predicate(quotient_graph):
        return then_function(quotient_graph)
    else:
        return else_function(quotient_graph)


@curry
def if_then_elif_else(
    quotient_graph: 'QuotientGraph',
    conditions_functions: List[Tuple[Callable[['QuotientGraph'], bool], Callable[['QuotientGraph'], 'QuotientGraph']]],
    else_function: Callable[['QuotientGraph'], 'QuotientGraph']
) -> 'QuotientGraph':
    """Multi-branch conditional operator for QuotientGraph transformations.
    Summary
        Evaluates a sequence of (predicate, function) pairs on the input
        QuotientGraph. The first predicate that evaluates True determines the
        branch function to apply. If none match, `else_function` is applied.

    Semantics
        - Input QG state:
            Input QuotientGraph is passed unchanged to all predicates and to
            the selected transformation function.
        - Output QG state:
            Result of applying the first matching branch function or else_function.
        - Determinism:
            Deterministic if predicates and branch functions are deterministic.

    Parameters
        quotient_graph : QuotientGraph
            Input graph to evaluate conditions against.
        conditions_functions : list[tuple[Callable[[QuotientGraph], bool], Callable[[QuotientGraph], QuotientGraph]]]
            Ordered list of (predicate, function) pairs. Each predicate decides
            whether its paired function should run.
        else_function : Callable[[QuotientGraph], QuotientGraph]
            Transformation to apply if none of the predicates evaluate True.

    Returns
        QuotientGraph
            Output of the first matching branch function, or else_function if
            no predicates match.

    Algorithm
        1. For each (predicate, func) in conditions_functions:
              if predicate(quotient_graph) is True:
                  return func(quotient_graph)
        2. If no predicate matched, return else_function(quotient_graph).

    Complexity
        - Time: sum of costs of predicates until first True + cost of one branch.
        - Memory: size of branch output (only one branch executed).
        - Worst case: evaluate all predicates if none match.

    Interactions
        - Extends `if_then_else` with multiple “elif” clauses.
        - Works inside pipelines with `forward_compose` or `compose` for
          conditional multi-path logic.
        - Often combined with threshold-based or structural tests on graphs.

    Examples
        # Branch by number of image nodes with multiple conditions
        workflow = forward_compose(
            connected_component(),
            if_then_elif_else(
                conditions_functions=[
                    (lambda qg: qg.image_graph.number_of_nodes() > 20, merge()),
                    (lambda qg: qg.image_graph.number_of_nodes() > 10, cycle()),
                    (lambda qg: qg.image_graph.number_of_nodes() > 5, clique())
                ],
                else_function=path()
            ),
            filter_by_number_of_nodes(number_of_nodes=(4, 10))
        )
        qg_out = workflow(qg_in)

    Domain Analogies
        - Programming: an `if … elif … elif … else …` chain.
        - Chemistry: apply different analysis depending on molecule size ranges.
        - Social networks: use different detection algorithms depending on group size.

    Failure Modes
        - Empty conditions list: always falls through to else_function.
        - Predicate exceptions: if any predicate raises, evaluation stops with error.
        - Branch incompatibility: if selected function produces outputs not
          compatible with downstream operators.
        - Ordering pitfalls: only the first True predicate is used, later matches ignored.
    """
    for predicate, func in conditions_functions:
        if predicate(quotient_graph):
            return func(quotient_graph)
    return else_function(quotient_graph)


#====================================================================================================
# ITERATION OPERATORS
#====================================================================================================

@curry
def for_loop(
    quotient_graph: 'QuotientGraph',
    function: Callable[['QuotientGraph'], 'QuotientGraph'],
    n_iterations: int = 1
) -> 'QuotientGraph':
    """Fixed-iteration loop operator for QuotientGraph transformations.
    Summary
        Repeatedly applies a decomposition function to the input graph a fixed
        number of times.

    Semantics
        - Input QG state:
            Takes the input QuotientGraph and repeatedly transforms it.
        - Output QG state:
            Result of applying `function` exactly `n_iterations` times in sequence.
        - Determinism:
            Deterministic if the function is deterministic.

    Parameters
        quotient_graph : QuotientGraph
            The starting graph.
        function : Callable[[QuotientGraph], QuotientGraph]
            Transformation to apply in each iteration.
        n_iterations : int, optional (default=1)
            Number of times to apply the function.

    Returns
        QuotientGraph
            The graph obtained after `n_iterations` applications of the function.

    Algorithm
        1. Initialise current = quotient_graph.
        2. Repeat `n_iterations` times:
              current = function(current).
        3. Return current.

    Complexity
        - Time: n_iterations × cost(function).
        - Memory: dominated by the largest intermediate QuotientGraph.

    Interactions
        - Useful when functions converge toward a fixed point within a bounded number of steps.
        - Can approximate iterative refinement (e.g. repeated filtering).
        - Often paired with `merge`, `cycle`, or `filter_by_*`.

    Examples
        workflow = forward_compose(
            connected_component(),
            for_loop(cycle(), n_iterations=3),
            filter_by_number_of_nodes(number_of_nodes=(4, 10))
        )
        qg_out = workflow(qg_in)

    Domain Analogies
        - Programming: a standard `for` loop with fixed iteration count.
        - Chemistry: apply the same reaction step repeatedly (e.g., washing cycles).
        - Social networks: re-apply clustering to stabilise group boundaries.

    Failure Modes
        - n_iterations <= 0 → function is never applied (returns input unchanged).
        - Non-idempotent or divergent function → unstable or meaningless result.
    """
    for _ in range(n_iterations):
        quotient_graph = function(quotient_graph)
    return quotient_graph

@curry
def while_loop(
    quotient_graph: 'QuotientGraph',
    function: Callable[['QuotientGraph'], 'QuotientGraph'],
    predicate: Callable[['QuotientGraph'], bool],
    max_iterations: int = 100
) -> 'QuotientGraph':
    """Predicate-controlled loop operator for QuotientGraph transformations.
    Summary
        Repeatedly applies a decomposition function as long as a predicate on
        the current graph is True, or until `max_iterations` is reached.

    Semantics
        - Input QG state:
            Starts with the provided QuotientGraph, repeatedly checks predicate.
        - Output QG state:
            Final state after zero or more iterations of `function`, stopped when
            predicate fails or iteration cap reached.
        - Determinism:
            Deterministic if function and predicate are deterministic.

    Parameters
        quotient_graph : QuotientGraph
            The starting graph.
        function : Callable[[QuotientGraph], QuotientGraph]
            Transformation to apply on each iteration.
        predicate : Callable[[QuotientGraph], bool]
            Loop continues while this condition evaluates True.
        max_iterations : int, optional (default=100)
            Upper bound on iterations to avoid infinite loops.

    Returns
        QuotientGraph
            The graph obtained after applying the function repeatedly until the
            predicate is False or the iteration limit is reached.

    Algorithm
        1. Initialise current = quotient_graph, iteration = 0.
        2. While predicate(current) is True and iteration < max_iterations:
              current = function(current)
              iteration += 1
        3. Return current.

    Complexity
        - Time: O(k × cost(function) + k × cost(predicate)), where k is the number of iterations executed.
        - Memory: governed by the largest intermediate QuotientGraph.

    Interactions
        - Enables fixed-point iteration until convergence criteria are met.
        - Works well with `merge` or `deduplicate` to shrink until stable.
        - Natural complement to `for_loop`.

    Examples
        workflow = forward_compose(
            connected_component(),
            while_loop(
                cycle(),
                predicate=lambda qg: qg.image_graph.number_of_nodes() > 5,
                max_iterations=10
            ),
            merge()
        )
        qg_out = workflow(qg_in)

    Domain Analogies
        - Programming: a `while` loop with condition and safety cap.
        - Chemistry: repeat titration steps until pH threshold reached.
        - Social networks: reapply clustering until no further group changes occur.

    Failure Modes
        - Predicate always False → input returned unchanged.
        - Predicate never False and function non-convergent → forced stop at max_iterations.
        - Predicate exceptions halt execution.
        - Function divergence may produce runaway growth in intermediate graphs.
    """
    iteration = 0
    while predicate(quotient_graph) and iteration < max_iterations:
        quotient_graph = function(quotient_graph)
        iteration += 1
    return quotient_graph

#====================================================================================================
# UNARY OPERATORS
#====================================================================================================

@curry
def identity(
    quotient_graph: 'QuotientGraph',
    param=None
) -> 'QuotientGraph':
    """Identity operator for QuotientGraph transformations.
    Summary
        Returns the input QuotientGraph unchanged, serving as a no-op in pipelines.

    Semantics
        - Input QG state:
            Accepts a QuotientGraph and creates a new wrapper that references it.
        - Output QG state:
            Equivalent to the input graph; no modification to preimage/image graphs.
        - Determinism:
            Deterministic (output always mirrors input).

    Parameters
        quotient_graph : QuotientGraph
            The input graph to be returned as output.
        param : Any, optional (default=None)
            Placeholder argument for consistency with operator signatures; unused.

    Returns
        QuotientGraph
            The same graph as the input, wrapped as a new QuotientGraph instance.

    Algorithm
        1. Construct a new QuotientGraph referencing the input.
        2. Return it directly.

    Complexity
        - Time: O(1).
        - Memory: O(1), except for wrapper instantiation overhead.

    Interactions
        - Useful as a placeholder in dynamically generated pipelines.
        - Can act as a neutral element in composition operators (`compose`, `add`).
        - Helpful for debugging: insert identity to check intermediate graph state.

    Examples
        # Use identity in a pipeline
        workflow = forward_compose(
            connected_component(),
            identity(),
            merge()
        )
        qg_out = workflow(qg_in)

    Domain Analogies
        - Mathematics: the identity function f(x) = x.
        - Chemistry: a reagent that leaves the molecule unchanged.
        - Social networks: “observe but do not intervene.”

    Failure Modes
        - None: safe in all contexts.
        - Potential confusion: some users may expect `identity` to clone deeply,
          but here it only returns an equivalent QuotientGraph wrapper.
    """
    out_quotient_graph = QuotientGraph(quotient_graph=quotient_graph)
    return out_quotient_graph

#--------------------------------------------------------------------------------
@curry
def node(
    quotient_graph: 'QuotientGraph',
    param=None
    ) -> 'QuotientGraph':
    """Emit one image node for each singleton vertex contained in the current image-node associations.
    Summary
        For every subgraph associated with an image node, decompose it into its constituent single vertices and
        create one new image node per vertex. Each new image node associates to an induced subgraph consisting
        of exactly that one vertex.

    Semantics
        - Input QG state: Reads quotient_graph.preimage_graph and all current image-node associations.
        - Output QG state: Returns a new QuotientGraph with the same preimage_graph and an image_graph in which
          each node corresponds to a singleton subgraph {v}. Provenance metadata is stored for traceability.
        - Determinism: Deterministic given the input graph; order of singleton emission is not semantically significant.

    Parameters
        param : Any, optional
            Placeholder argument for interface consistency. Ignored.

    Algorithm
        - Initialize a fresh QuotientGraph with the same preimage_graph.
        - For each subgraph in get_image_nodes_associations():
            * Iterate over all its nodes.
            * For each node, create a singleton subgraph [{node}].
            * Call create_image_node_with_subgraph_from_nodes(singleton, meta=build_meta_from_function_context()).

    Complexity
        Let S be the number of associations, and N_i their node counts.
        Time: Σ_i O(N_i) to iterate and create singleton image nodes.
        Memory: O(total number of nodes) for storing singletons.

    Side Effects & Metadata
        - Each created image node stores:
            * 'association': a subgraph containing a single vertex from the preimage.
            * 'meta': {'source_function': 'node', 'params': {...}} from build_meta_from_function_context().
        - Labels/attributes are not computed here; call update() to populate them.

    Interactions
        - Often used as the "finest granularity" seed for subsequent operators (e.g., neighborhoods, degree filters).
        - Useful in `add` to combine node-level views with larger motifs.
        - Composes naturally with `neighborhood` to generate ego-graphs around individual nodes.

    Constraints & Invariants
        - Works with any undirected or directed preimage graph.
        - Emits exactly one singleton per node in each input association.
        - If the association is empty, no image nodes are created.

    Examples
        # Break graph into singleton image nodes
        workflow = forward_compose(node())
        Q2 = workflow(Q).update()

        # Use node() + neighborhood to generate radius-1 ego graphs around every vertex
        workflow = forward_compose(node(), neighborhood(radius=(1,1)))
        Q2 = workflow(Q).update()

    Domain Analogies
        - Social networks: one feature node per individual user.
        - Computer networks: one feature node per device.
        - Chemistry: one feature node per atom in a molecule.

    Failure Modes & Diagnostics
        - Potential explosion in image node count for very large graphs; mitigate with sampling or filters.
        - Ensure downstream operators handle large numbers of singletons efficiently.
    """
    out_quotient_graph = QuotientGraph(
        graph=quotient_graph.preimage_graph,
        label_function=quotient_graph.label_function,
        attribute_function=quotient_graph.attribute_function,
        edge_function=quotient_graph.edge_function,
    )

    for subgraph in quotient_graph.get_image_nodes_associations():
        components = [[node] for node in subgraph.nodes()]
        for component in components:
            out_quotient_graph.create_image_node_with_subgraph_from_nodes(
                component,
                meta=build_meta_from_function_context()
            )

    return out_quotient_graph

#--------------------------------------------------------------------------------
@curry
def edge(
    quotient_graph: 'QuotientGraph',
    param=None
    ) -> 'QuotientGraph':
    """Emit one image node for each edge contained in the current image-node associations.
    Summary
        For every subgraph associated with an image node, decompose it into its constituent edges and
        create one new image node per edge. Each new image node associates to the induced subgraph
        consisting of exactly the two incident vertices and their connecting edge.

    Semantics
        - Input QG state: Reads quotient_graph.preimage_graph and all current image-node associations.
        - Output QG state: Returns a new QuotientGraph with the same preimage_graph and an image_graph in which
          each node corresponds to a single edge subgraph. Provenance metadata is attached for traceability.
        - Determinism: Deterministic given the input graph; order of edge emission is not semantically significant.

    Parameters
        param : Any, optional
            Placeholder argument for interface consistency. Ignored.

    Algorithm
        - Initialize a fresh QuotientGraph with the same preimage_graph.
        - For each subgraph in get_image_nodes_associations():
            * Iterate over its edge list.
            * For each edge (u, v), build a 2-node induced subgraph {u, v} with the connecting edge.
            * Call create_image_node_with_subgraph_from_nodes(edge, meta=build_meta_from_function_context()).

    Complexity
        Let S be the number of associations, and E_i their edge counts.
        Time: Σ_i O(E_i) to iterate and create edge-based image nodes.
        Memory: O(total number of edges) for storing induced 2-node subgraphs.

    Side Effects & Metadata
        - Each created image node stores:
            * 'association': a subgraph with exactly 2 vertices and 1 edge.
            * 'meta': {'source_function': 'edge', 'params': {...}} from build_meta_from_function_context().
        - Labels/attributes are not computed here; call update() to populate them.

    Interactions
        - Often paired with `node()` to produce both vertex-level and edge-level features.
        - Useful in `add` to mix edge-based subgraphs with higher-order motifs (cycles, cliques).
        - Can precede `neighborhood` to grow paths or ego-graphs from edges.

    Constraints & Invariants
        - Works with undirected and directed graphs (edges will reflect graph type).
        - Emits exactly one 2-node subgraph per edge in each input association.
        - If the association has no edges, no image nodes are created.

    Examples
        # Break graph into edge subgraphs
        workflow = forward_compose(edge())
        Q2 = workflow(Q).update()

        # Combine edges and cycles
        workflow = forward_compose(add(edge(), cycle()))
        Q2 = workflow(Q).update()

    Domain Analogies
        - Social networks: one feature node per friendship/connection.
        - Computer networks: one feature node per physical or logical link.
        - Chemistry: one feature node per bond between two atoms.

    Failure Modes & Diagnostics
        - Explosion in image node count for dense graphs (O(n^2) edges). Use filters or degree constraints upstream.
        - Directed graphs yield ordered edge pairs; ensure downstream operators handle orientation if relevant.
    """
    out_quotient_graph = QuotientGraph(
        graph=quotient_graph.preimage_graph,
        label_function=quotient_graph.label_function,
        attribute_function=quotient_graph.attribute_function,
        edge_function=quotient_graph.edge_function,
    )
    for subgraph in quotient_graph.get_image_nodes_associations():
        components = list(subgraph.edges())
        for component in components:
            out_quotient_graph.create_image_node_with_subgraph_from_nodes(
                component,
                meta=build_meta_from_function_context()
            )
    
    return out_quotient_graph

#--------------------------------------------------------------------------------
def connected_component_decomposition_function(subgraph):
    """Find connected components of a subgraph.

    Parameters
    ----------
    subgraph : networkx.Graph
        Input undirected graph to partition into connected components.

    Returns
    -------
    components : list[set]
        List of node sets, one per connected component.
    """
    components = list(nx.connected_components(subgraph))
    return components


@curry
def connected_component(
    quotient_graph: 'QuotientGraph',
    param=None
    ) -> 'QuotientGraph':
    """Emit one image node per connected component from each associated subgraph in the current quotient graph.
    Summary
        For every image-node association (subgraph), compute its connected components and create one new image node
        per component, preserving the original preimage graph and adding provenance metadata.

    Semantics
        - Input QG state: Reads quotient_graph.preimage_graph (as the base) and all current image-node associations.
        - Output QG state: Returns a new QuotientGraph with the same preimage_graph and an image_graph whose nodes
          each associate to a connected component (node-induced subgraph) of the original associations.
          Invariants: preimage_graph unchanged; newly created image nodes have 'association' set and 'meta' populated.
        - Determinism: Deterministic given inputs; the order of emitted components is not semantically significant.

    Parameters
        param : Any, optional
            Unused placeholder to keep a uniform operator signature. Ignored.

    Algorithm
        - Initialize `out_quotient_graph` with the same preimage_graph.
        - For each associated subgraph in `quotient_graph.get_image_nodes_associations()`:
            * Compute components = connected_component_decomposition_function(subgraph).
            * For each component (set of nodes), call
              `create_image_node_with_subgraph_from_nodes(component, meta=build_meta_from_function_context())`.

    Complexity
        Let S be the number of input associations and (V_i, E_i) their sizes.
        Time: Σ_i O(|V_i| + |E_i|) for components + overhead to create image nodes.
        Memory: O(total emitted nodes + edges) across all component subgraphs.

    Side Effects & Metadata
        - Each created image node stores:
            * 'association' : induced subgraph on the component’s node set.
            * 'meta' : {'source_function': 'connected_component', 'params': {...}} via build_meta_from_function_context().
        - Labels/attributes are not computed here; call `update()` later if needed.

    Interactions
        - Often followed by filters (e.g., `filter_by_number_of_nodes`) to bound instance counts.
        - Composes well with `add`, `product`, and distance-based combinators after reducing to components.

    Constraints & Invariants
        - Assumes associations are undirected or that connectedness is well-defined for them.
        - Empty associations emit no components.

    Examples
        # Minimal: break current associations into components
        workflow = forward_compose(connected_component())
        Q2 = workflow(Q).update()

        # With size filter to limit instances
        workflow = forward_compose(
            connected_component(),
            filter_by_number_of_nodes(number_of_nodes=(4, 50))
        )

    Domain Analogies
        - Social networks: split groups into disconnected communities before further analysis.
        - Computer networks: decompose a selected topology region into subnets.
        - Chemistry: separate disconnected fragments in a selected molecular region.

    Failure Modes & Diagnostics
        - If associations are directed graphs, nx.connected_components may fail; ensure associations are undirected
          or convert appropriately upstream.
        - Excessive instance counts if the input associations are highly fragmented; mitigate with size filters.
    """
    out_quotient_graph = QuotientGraph(
        graph=quotient_graph.preimage_graph,
        label_function=quotient_graph.label_function,
        attribute_function=quotient_graph.attribute_function,
        edge_function=quotient_graph.edge_function,
    )

    for subgraph in quotient_graph.get_image_nodes_associations():
        components = connected_component_decomposition_function(subgraph)
        for component in components:
            out_quotient_graph.create_image_node_with_subgraph_from_nodes(
                component,
                meta=build_meta_from_function_context()
            )
    
    return out_quotient_graph


#--------------------------------------------------------------------------------
def degree_decomposition_function(subgraph, min_degree=0, max_degree=2):
    """Select nodes in a subgraph whose degree lies within [min_degree, max_degree].

    Parameters
    ----------
    subgraph : networkx.Graph
        Input graph on which node degrees are computed.
    min_degree : int, default 0
        Inclusive lower bound for node degree.
    max_degree : int, default 2
        Inclusive upper bound for node degree.

    Returns
    -------
    components : list[set]
        A one-element list containing the set of nodes that satisfy the degree constraint.
    """
    deg = dict(nx.degree(subgraph))
    component = set([u for u in deg if max_degree >= deg[u] and deg[u] >= min_degree])
    components = [component]
    return components

@curry
def degree(
    quotient_graph: 'QuotientGraph',
    value = (0,2)
    ) -> 'QuotientGraph':
    """Emit one image node per subgraph containing all vertices whose degree lies within given bounds.
    Summary
        For every image-node association, select the subset of its vertices whose degree is between the
        specified bounds, and create a new image node whose association is induced on that node set.

    Semantics
        - Input QG state: Reads quotient_graph.preimage_graph and current image-node associations.
        - Output QG state: Returns a new QuotientGraph with the same preimage_graph and one image node
          per association, representing the degree-filtered set of vertices (possibly empty).
        - Determinism: Deterministic given input graph and degree bounds.

    Parameters
        value : int | tuple[int,int], default (0,2)
            Inclusive lower and upper bounds for degree selection.
            If a single int is given, treated as (value, value).

    Algorithm
        - Normalize value into (min_degree, max_degree).
        - For each subgraph in get_image_nodes_associations():
            * Call degree_decomposition_function(subgraph, min_degree, max_degree).
            * For the returned node set, create one image node via
              create_image_node_with_subgraph_from_nodes(component, meta=...).

    Complexity
        Let S be number of associations, and V_i their vertex counts.
        Time: Σ_i O(|V_i| + |E_i|) for degree calculation.
        Memory: O(total nodes across associations).

    Side Effects & Metadata
        - Each created image node stores:
            * 'association': induced subgraph on nodes satisfying the degree constraint.
            * 'meta': {'source_function': 'degree', 'params': {...}}.
        - Labels/attributes not computed here; call update() afterwards.

    Interactions
        - Commonly used to isolate hubs or periphery before applying further operators (neighborhood, cycle).
        - Can be combined with `filter_by_number_of_nodes` to suppress empty or tiny degree-selected sets.
        - Works naturally inside `add` to parallelize degree-based and structural decompositions.

    Constraints & Invariants
        - Works for both directed and undirected graphs, but degree is total degree for directed.
        - Produces exactly one image node per association (possibly empty subgraph).

    Examples
        # Extract all degree-1 vertices across associations
        workflow = forward_compose(degree(value=1))
        Q2 = workflow(Q).update()

        # Select low-degree (1–2) and then apply neighborhood expansion
        workflow = forward_compose(
            degree(value=(1,2)),
            neighborhood(radius=(1,1))
        )

    Domain Analogies
        - Social networks: isolate leaves (followers only) vs. hubs (many friends).
        - Computer networks: edge devices vs. backbone routers.
        - Chemistry: hydrogens (degree=1) vs. branching carbons.

    Failure Modes & Diagnostics
        - Produces empty associations when no nodes match; can accumulate many empty image nodes.
        - Explosion risk is low, but many singleton sets can be produced if graph is large.
    """
    value = value_to_2tuple(value)
    out_quotient_graph = QuotientGraph(
        graph=quotient_graph.preimage_graph,
        label_function=quotient_graph.label_function,
        attribute_function=quotient_graph.attribute_function,
        edge_function=quotient_graph.edge_function,
    )

    for subgraph in quotient_graph.get_image_nodes_associations():
        components = degree_decomposition_function(
            subgraph,
            min_degree=min(value),
            max_degree=max(value)
        )
        for component in components:
            out_quotient_graph.create_image_node_with_subgraph_from_nodes(
                component,
                meta=build_meta_from_function_context()
            )
    
    return out_quotient_graph


#--------------------------------------------------------------------------------
def split_decomposition_function(subgraph):
    """Bipartition a connected subgraph using Kernighan–Lin; else return the whole node set.

    Parameters
    ----------
    subgraph : networkx.Graph
        Input (undirected) subgraph. If disconnected or if bisection fails, no split is performed.

    Returns
    -------
    list[set]
        [set(part1), set(part2)] on success; otherwise [set(all_nodes)].
    """
    # If subgraph is not connected, we simply return the whole node set.
    if not nx.is_connected(subgraph):
        return [set(subgraph.nodes())]
    
    try:
        part1, part2 = kernighan_lin_bisection(subgraph)
        return [set(part1), set(part2)]
    except Exception:
        # In case of any error, fall back to not splitting.
        return [set(subgraph.nodes())]

@curry
def split(quotient_graph: 'QuotientGraph',
    param=None) -> 'QuotientGraph':
    """Emit up to two image nodes per association by bipartitioning each subgraph via Kernighan–Lin.
    Summary
        For every associated subgraph, attempt a balanced bisection using the Kernighan–Lin algorithm.
        If successful, create two image nodes (one per part). If splitting is impossible or fails,
        emit a single image node containing the original node set.

    Semantics
        - Input QG state: Reads `quotient_graph.preimage_graph` and current image-node associations.
        - Output QG state: Returns a new `QuotientGraph` with the same preimage; its image graph contains
          one or two nodes per input association, each associated to the induced subgraph on the part.
          Determinism: Deterministic given inputs and NetworkX KL implementation (no added randomness here).

    Parameters
        param : Any, optional
            Placeholder for interface uniformity; ignored.

    Algorithm
        - Initialize `out_quotient_graph` with the same preimage graph.
        - For each associated subgraph:
            * If it has < 2 nodes, emit that node set as-is (no split).
            * Else call `split_decomposition_function(subgraph)` to get one or two parts.
            * For each part, create a new image node with association = induced subgraph on that part.
              (Provenance metadata is attached via `build_meta_from_function_context()` in the split branch.)

    Complexity
        Let S be the number of associations, with sizes (V_i, E_i).
        - KL bisection per subgraph is roughly O(|E_i|) to O(|V_i|^2) depending on implementation and graph density.
        - Overall time: Σ_i KL_cost(subgraph_i) + image-node creation overhead.
        - Memory: proportional to emitted induced subgraphs.

    Side Effects & Metadata
        - For split parts, each created image node includes:
            * 'association' : induced subgraph on the part’s node set.
            * 'meta' : {'source_function': 'split', 'params': {...}} via build_meta_from_function_context().
        - Note: In the degenerate (<2 nodes) branch, the current implementation creates the image node without meta.

    Interactions
        - Common precursor to `filter_by_number_of_nodes/edges` to prune tiny or huge parts.
        - Works well before `clique`, `cycle`, or distance-based combinators to limit combinatorics.
        - Can be iterated via `for_loop(split(), n)` or `while_loop(...)` for coarse-to-fine partitioning.

    Constraints & Invariants
        - Assumes undirected graphs for KL; on disconnected subgraphs or KL failures, falls back to a single part.
        - Does not modify the preimage graph.

    Examples
        # One-shot bisection of current associations
        workflow = forward_compose(split())
        Q2 = workflow(Q).update()

        # Iterative partition then keep medium-sized parts
        workflow = forward_compose(
            split(), split(),
            filter_by_number_of_nodes(number_of_nodes=(10, 200))
        )

    Domain Analogies
        - Social networks: partition a community into two cohorts (e.g., interest-based halves).
        - Computer networks: split a subnet into two clusters.
        - Chemistry: divide a large fragment into two sub-fragments before motif extraction.

    Failure Modes & Diagnostics
        - Highly irregular or tiny subgraphs may not split meaningfully; expect single-part output.
        - For very dense graphs, KL can be expensive—consider bounding subgraph size upstream.
    """
    out_quotient_graph = QuotientGraph(
        graph=quotient_graph.preimage_graph,
        label_function=quotient_graph.label_function,
        attribute_function=quotient_graph.attribute_function,
        edge_function=quotient_graph.edge_function,
    )
    
    for subgraph in quotient_graph.get_image_nodes_associations():
        # If the subgraph has less than 2 nodes, we cannot split it.
        if subgraph.number_of_nodes() < 2:
            out_quotient_graph.create_image_node_with_subgraph_from_nodes(list(subgraph.nodes()))
        else:
            parts = split_decomposition_function(subgraph)
            for part in parts:
                component = list(part)
                out_quotient_graph.create_image_node_with_subgraph_from_nodes(
                    component,
                    meta=build_meta_from_function_context()
                )
    
    return out_quotient_graph


#--------------------------------------------------------------------------------
def get_reachable_nodes_bfs(
    graph: nx.Graph,
    source: Any,
    cutoff: int
    ) -> List[Any]:
    """Return nodes within a given BFS radius from a source node.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph in which BFS is performed.
    source : Any
        Node ID to start the BFS from.
    cutoff : int
        Maximum hop distance to include (must be ≥ 0).

    Returns
    -------
    list[Any]
        Nodes reachable from `source` within the cutoff radius.

    Raises
    ------
    nx.NetworkXError
        If `source` is not present in the graph.
    ValueError
        If `cutoff` is negative.
    """
    if not isinstance(cutoff, int) or cutoff < 0:
        raise ValueError("Cutoff must be a non-negative integer.")
    
    if source not in graph:
        raise nx.NetworkXError(f"Source node {source} not present in graph.")

    if cutoff == 0:
        return [source]
    
    path_lengths = nx.single_source_shortest_path_length(graph, source, cutoff=cutoff)
    return list(path_lengths.keys())


@curry
def neighborhood(
    quotient_graph: 'QuotientGraph',
    radius=(0,1)
) -> 'QuotientGraph':
    """Emit image nodes for BFS neighborhoods of each node in current associations, over a radius range.
    Summary
        For each node in each input subgraph, generate BFS balls of all radii r in [min_radius, max_radius].
        Each resulting ball is represented as a new image node in the output QuotientGraph.

    Semantics
        - Input QG state: Reads quotient_graph.preimage_graph and current image-node associations.
        - Output QG state: New QuotientGraph whose image graph contains one node per BFS neighborhood.
          Provenance metadata records operator and parameters.
        - Determinism: Fully deterministic given graph and radius.

    Parameters
        radius : int | tuple[int,int], default (0,1)
            Inclusive radius bounds. If a single int is given, interpreted as (r,r).

    Algorithm
        - Normalize radius bounds using value_to_2tuple().
        - For each subgraph in get_image_nodes_associations():
            * For r from min_radius to max_radius:
                - For each node in the subgraph:
                    - Call get_reachable_nodes_bfs(subgraph, source=node, cutoff=r).
                    - Create image node with induced subgraph on reachable nodes.

    Complexity
        Let N = total nodes, E = total edges, R = number of radius values.
        - BFS cost per node is O(E) worst-case; repeated for N × R nodes.
        - Output size: O(N × R) image nodes per association.

    Side Effects & Metadata
        - Each created image node stores:
            * 'association': induced subgraph on reachable set.
            * 'meta': {'source_function': 'neighborhood', 'params': {'radius': (rmin,rmax)}}.
        - Labels/attributes not computed; call update() downstream.

    Interactions
        - Naturally follows `node()` to expand singletons into ego-graphs.
        - Can be combined with `filter_by_number_of_nodes` to control explosion in large neighborhoods.
        - Works well with `add` to mix neighborhoods with other motifs.

    Constraints & Invariants
        - Works with any connected or disconnected graph.
        - Emits one image node per (node, r) pair.
        - Empty results only possible for r=0 (singleton neighborhoods).

    Examples
        # Expand nodes into radius-1 ego-graphs
        workflow = forward_compose(node(), neighborhood(radius=1))

        # Multi-scale neighborhoods up to radius 3
        workflow = forward_compose(neighborhood(radius=(1,3)))

    Domain Analogies
        - Social networks: “friends-of-friends” circles at increasing hop distance.
        - Computer networks: subnets at hop distance r from a given device.
        - Chemistry: atom-centered fragments at increasing bond distance.

    Failure Modes & Diagnostics
        - Potential combinatorial blow-up in dense graphs; mitigate with radius limits or filters.
        - For very large radius ranges, output size can exceed memory quickly.
    """
    radius = value_to_2tuple(radius)
    out_quotient_graph = QuotientGraph(
        graph=quotient_graph.preimage_graph,
        label_function=quotient_graph.label_function,
        attribute_function=quotient_graph.attribute_function,
        edge_function=quotient_graph.edge_function,
    )

    for subgraph in quotient_graph.get_image_nodes_associations():
        for r in range(min(radius), max(radius) + 1):
            for source in subgraph.nodes():
                component = get_reachable_nodes_bfs(subgraph, source, cutoff=r)
                out_quotient_graph.create_image_node_with_subgraph_from_nodes(
                    component,
                    meta=build_meta_from_function_context()
                )
    
    return out_quotient_graph

#--------------------------------------------------------------------------------
def get_edges_from_cycle(cycle):
    """Yield edges (u, v) along a cycle, ensuring u < v for consistency."""
    for i, c in enumerate(cycle):
        j = (i + 1) % len(cycle)
        u, v = cycle[i], cycle[j]
        if u < v:
            yield u, v
        else:
            yield v, u

def get_cycle_basis_edges(g):
    """Return the list of edges belonging to all cycles in the graph."""
    ebunch = []
    cs = nx.cycle_basis(g)
    for c in cs:
        ebunch += list(get_edges_from_cycle(c))
    return ebunch

def edge_complement(g, ebunch):
    """Return edges of g that are not in ebunch."""
    edge_set = set(ebunch)
    other_ebunch = [e for e in g.edges() if e not in edge_set]
    return other_ebunch

def edge_subgraph(g, ebunch):
    """Induce subgraph of g using only the given edge list ebunch."""
    if nx.is_directed(g):
        g2 = nx.DiGraph()
    else:
        g2 = nx.Graph()
    g2.add_nodes_from(g.nodes())
    for u, v in ebunch:
        g2.add_edge(u, v)
        g2.edges[u, v].update(g.edges[u, v])
    return g2

def edge_complement_subgraph(g, ebunch):
    """Induce subgraph from edges of g that are not in ebunch."""
    if nx.is_directed(g):
        g2 = nx.DiGraph()
    else:
        g2 = nx.Graph()
    g2.add_nodes_from(g.nodes())
    for e in g.edges():
        if e not in ebunch:
            u, v = e
            g2.add_edge(u, v)
            g2.edges[u, v].update(g.edges[u, v])
    return g2

def cycle_decomposition_function(subgraph):
    """Return node sets corresponding to all simple cycles in the subgraph."""
    cs = nx.cycle_basis(subgraph)
    cycle_components = list(map(set, cs))
    return cycle_components

def non_cycle_decomposition_function(subgraph):
    """Return node sets of acyclic connected components after removing cycle edges."""
    cs = nx.cycle_basis(subgraph)
    cycle_ebunch = get_cycle_basis_edges(subgraph)
    g2 = edge_complement_subgraph(subgraph, cycle_ebunch)
    non_cycle_components = nx.connected_components(g2)
    non_cycle_components = [c for c in non_cycle_components if len(c) >= 2]
    non_cycle_components = list(map(set, non_cycle_components))
    return non_cycle_components

@curry
def cycle(
    quotient_graph: 'QuotientGraph',
    param=None
    ) -> 'QuotientGraph':
    """Emit one image node per cycle in each associated subgraph.
    Summary
        For each input subgraph, compute its simple cycle basis and create one image node per cycle,
        with the association set to the induced subgraph on that cycle’s nodes.

    Semantics
        - Input QG state: Uses quotient_graph.preimage_graph and image-node associations.
        - Output QG state: New QuotientGraph where each image node corresponds to a simple cycle.
        - Determinism: Deterministic given the input graph; cycle_basis order is consistent per run.

    Parameters
        param : Any, optional
            Placeholder for operator interface; ignored.

    Algorithm
        - For each subgraph in get_image_nodes_associations():
            * Call cycle_decomposition_function(subgraph).
            * For each cycle (node set), create an image node via create_image_node_with_subgraph_from_nodes().

    Complexity
        - cycle_basis: O(|V| + |E|) per subgraph.
        - Total cost: sum over all associations.

    Metadata
        - Each emitted image node stores 'association' (the cycle-induced subgraph) and 'meta'
          with source_function='cycle'.

    Interactions
        - Complements `tree()` operator to split graph into cyclic vs. acyclic parts.
        - Useful before `combination` to link cycles with functional groups in chemistry,
          or with `neighborhood` to explore cycle context.

    Examples
        # Decompose into cycles
        workflow = forward_compose(connected_component(), cycle())

    Domain Analogies
        - Social networks: closed friendship circles.
        - Computer networks: routing loops.
        - Chemistry: aromatic rings or other cyclic motifs.
    """
    out_quotient_graph = QuotientGraph(
        graph=quotient_graph.preimage_graph,
        label_function=quotient_graph.label_function,
        attribute_function=quotient_graph.attribute_function,
        edge_function=quotient_graph.edge_function,
    )

    for subgraph in quotient_graph.get_image_nodes_associations():
        components = cycle_decomposition_function(subgraph)
        for component in components:
            out_quotient_graph.create_image_node_with_subgraph_from_nodes(
                component,
                meta=build_meta_from_function_context()
            )
    
    return out_quotient_graph

@curry
def tree(
    quotient_graph: 'QuotientGraph',
    param=None
    ) -> 'QuotientGraph':
    """Emit one image node per acyclic connected component in each associated subgraph.
    Summary
        For each subgraph, remove all cycle edges and compute the remaining connected components.
        For each nontrivial acyclic component, create an image node with association to that node set.

    Semantics
        - Input QG state: Uses quotient_graph.preimage_graph and image-node associations.
        - Output QG state: New QuotientGraph where each image node is an acyclic component (tree).
        - Determinism: Deterministic given input graph.

    Parameters
        param : Any, optional
            Placeholder argument; ignored.

    Algorithm
        - For each subgraph in get_image_nodes_associations():
            * Call non_cycle_decomposition_function(subgraph).
            * For each returned component (node set), create a new image node.

    Complexity
        - Cycle detection + complement graph construction: O(|V| + |E|).
        - Connected components: O(|V| + |E|).

    Metadata
        - Each image node stores 'association' (acyclic subgraph) and 'meta' with source_function='tree'.

    Interactions
        - Complements `cycle()` to partition graph into cyclic and acyclic parts.
        - Useful for chemistry (chains vs. rings), social nets (tree-like follower structures), or
          computer networks (tree-like spanning subnetworks).

    Examples
        # Extract tree-like parts
        workflow = forward_compose(connected_component(), tree())

    Domain Analogies
        - Social networks: hierarchical tree structures (e.g., org charts).
        - Computer networks: spanning trees, acyclic subnetworks.
        - Chemistry: chain structures vs. rings.

    Failure Modes
        - Tiny graphs (<2 nodes) may not yield meaningful components.
    """
    out_quotient_graph = QuotientGraph(
        graph=quotient_graph.preimage_graph,
        label_function=quotient_graph.label_function,
        attribute_function=quotient_graph.attribute_function,
        edge_function=quotient_graph.edge_function,
    )

    for subgraph in quotient_graph.get_image_nodes_associations():
        components = non_cycle_decomposition_function(subgraph)
        for component in components:
            out_quotient_graph.create_image_node_with_subgraph_from_nodes(
                component,
                meta=build_meta_from_function_context()
            )
    
    return out_quotient_graph

#--------------------------------------------------------------------------------
def path_decomposition_function(subgraph, min_number_of_edges=1, max_number_of_edges=None):
    """Return node sets corresponding to simple paths within a length range.

    Parameters
    ----------
    subgraph : networkx.Graph
        Input graph in which to search for paths.
    min_number_of_edges : int, default 1
        Minimum path length (in edges) to include.
    max_number_of_edges : int, optional
        Maximum path length (in edges). Defaults to number of nodes.

    Returns
    -------
    list[tuple]
        Unique tuples of node IDs representing paths within the length range.
    """
    if max_number_of_edges is None:
        max_number_of_edges = subgraph.number_of_nodes()
    edge_components = []
    for n in subgraph.nodes():
        ego_graph = nx.ego_graph(subgraph, n, radius=max_number_of_edges+1)
        for v in ego_graph.nodes():
            try:
                for path in nx.all_shortest_paths(ego_graph, source=n, target=v):
                    edge_component = set()
                    if len(path) >= min_number_of_edges + 1 and len(path) <= max_number_of_edges + 1:
                        for i, u in enumerate(path[:-1]):
                            w = path[i + 1]
                            edge_component.add(u)
                            edge_component.add(w)
                    if edge_component:
                        edge_component = tuple(sorted(edge_component))
                        edge_components.append(edge_component)
            except Exception:
                pass
    components = list(set(edge_components))
    return components

@curry
def path(
    quotient_graph: 'QuotientGraph',
    number_of_edges=(1,3)
    ) -> 'QuotientGraph':
    """Emit one image node per path within given edge-length bounds.
    Summary
        For each subgraph, enumerate simple paths whose length in edges lies between
        `min_number_of_edges` and `max_number_of_edges`. Each distinct path’s nodes
        form a new image node in the output QuotientGraph.

    Semantics
        - Input QG state: Reads preimage_graph and current image-node associations.
        - Output QG state: New QuotientGraph with additional image nodes, one per qualifying path.
        - Determinism: Deterministic given input graph and parameters.

    Parameters
        number_of_edges : int | tuple[int,int], default (1,3)
            Inclusive range of path lengths in edges. If a single int is given, it is treated as (n,n).

    Algorithm
        - Normalize number_of_edges with value_to_2tuple().
        - For each subgraph association:
            * Call path_decomposition_function(subgraph, min, max).
            * For each resulting node set, create a new image node with association=subgraph induced by that set.

    Complexity
        Path enumeration can grow exponentially with graph size.
        - all_shortest_paths dominates cost: O(#paths × path_length).
        - Filtering bounds keeps only paths within min/max.

    Metadata
        Each image node stores 'association' (induced path subgraph) and 'meta'
        with source_function='path' and parameters.

    Interactions
        - Useful in chemistry to capture chains of bonds of fixed lengths.
        - In social networks, can represent "friend-of-friend-of-friend" chains of specific depth.
        - Often paired with `combination` or `filter_by_number_of_nodes` to avoid blow-up.

    Examples
        # Extract paths of length 2–4
        workflow = forward_compose(
            connected_component(),
            path(number_of_edges=(2,4))
        )

    Domain Analogies
        - Chemistry: carbon chains of length n.
        - Social networks: paths of introductions (degree of separation).
        - Computer networks: routing paths up to certain hops.

    Failure Modes & Diagnostics
        - For large or dense graphs, path enumeration may explode in size.
        - Paths shorter than `min_number_of_edges` or longer than `max_number_of_edges` are ignored.
    """
    number_of_edges = value_to_2tuple(number_of_edges)
    out_quotient_graph = QuotientGraph(
        graph=quotient_graph.preimage_graph,
        label_function=quotient_graph.label_function,
        attribute_function=quotient_graph.attribute_function,
        edge_function=quotient_graph.edge_function,
    )

    for subgraph in quotient_graph.get_image_nodes_associations():
        components = path_decomposition_function(
            subgraph,
            min_number_of_edges=min(number_of_edges),
            max_number_of_edges=max(number_of_edges)
        )
        for component in components:
            out_quotient_graph.create_image_node_with_subgraph_from_nodes(
                component,
                meta=build_meta_from_function_context()
            )
    
    return out_quotient_graph

#--------------------------------------------------------------------------------
def graphlet_decomposition_function(subgraph, radius=1, min_number_of_nodes=1, max_number_of_nodes=3):
    """Enumerate connected ego-subgraphs of bounded size.

    Parameters
    ----------
    subgraph : networkx.Graph
        Input graph to search for subgraphs.
    radius : int, default 1
        Ego radius around each node to consider when forming graphlets.
    min_number_of_nodes : int, default 1
        Minimum number of nodes in each subgraph.
    max_number_of_nodes : int, default 3
        Maximum number of nodes in each subgraph.

    Returns
    -------
    list[tuple]
        Unique tuples of node IDs, each defining a connected graphlet.
    """
    components = []
    for size in range(min_number_of_nodes, max_number_of_nodes + 1):
        for u in subgraph.nodes():
            ego_graph = nx.ego_graph(subgraph, u, radius=radius)
            for sub_nodes in itertools.combinations(ego_graph.nodes(), size):
                sub_subgraph = ego_graph.subgraph(sub_nodes)
                if nx.is_connected(sub_subgraph):
                    components.append(tuple(sorted(set(sub_nodes))))
    components = list(set(components))
    return components

@curry
def graphlet(
    quotient_graph: 'QuotientGraph',
    radius=1,
    number_of_nodes=(1,3)
    ) -> 'QuotientGraph':
    """Emit image nodes for connected graphlets within ego neighborhoods.
    Summary
        For each input subgraph, enumerate all connected induced subgraphs
        (“graphlets”) of size between `min_number_of_nodes` and `max_number_of_nodes`
        inside ego neighborhoods of radius `r`. Each graphlet becomes a new image node.

    Semantics
        - Input QG state: Reads preimage_graph and image-node associations.
        - Output QG state: New QuotientGraph with one image node per connected graphlet.
        - Determinism: Deterministic enumeration, though order of graphlets is not guaranteed.

    Parameters
        radius : int, default 1
            Ego radius around each node to expand before sampling graphlets.
        number_of_nodes : int | tuple[int,int], default (1,3)
            Inclusive bounds on graphlet size (number of nodes).

    Algorithm
        - Normalize number_of_nodes with value_to_2tuple().
        - For each subgraph association:
            * Call graphlet_decomposition_function(subgraph, radius, min, max).
            * For each node tuple, create an image node representing the induced subgraph.

    Complexity
        - Exponential in subgraph size due to combinations.
        - Mitigated by limiting radius and max_number_of_nodes.

    Metadata
        - Each image node stores 'association' (graphlet subgraph) and 'meta'
          with source_function='graphlet' and params.

    Interactions
        - More fine-grained than `clique()` or `path()`.
        - Can be combined with `filter_by_number_of_nodes` to limit explosion.
        - Useful precursor to motif-based classification tasks.

    Examples
        # Extract all connected graphlets up to size 4
        workflow = forward_compose(
            connected_component(),
            graphlet(radius=2, number_of_nodes=(2,4))
        )

    Domain Analogies
        - Chemistry: small functional fragments around atoms.
        - Social networks: micro-groups (triads, quads).
        - Computer networks: small local subnetworks.

    Failure Modes & Diagnostics
        - Large radius or high max_number_of_nodes leads to combinatorial blow-up.
        - Disconnected candidate subgraphs are discarded.
    """
    number_of_nodes = value_to_2tuple(number_of_nodes) 
    out_quotient_graph = QuotientGraph(
        graph=quotient_graph.preimage_graph,
        label_function=quotient_graph.label_function,
        attribute_function=quotient_graph.attribute_function,
        edge_function=quotient_graph.edge_function,
    )

    for subgraph in quotient_graph.get_image_nodes_associations():
        components = graphlet_decomposition_function(
            subgraph,
            radius=radius,
            min_number_of_nodes=min(number_of_nodes),
            max_number_of_nodes=max(number_of_nodes)
        )
        for component in components:
            out_quotient_graph.create_image_node_with_subgraph_from_nodes(
                component,
                meta=build_meta_from_function_context()
            )
    
    return out_quotient_graph

#--------------------------------------------------------------------------------    
def clique_decomposition_function(subgraph, min_number_of_nodes=1, max_number_of_nodes=None):
    """Enumerate cliques within size bounds in a subgraph.

    Parameters
    ----------
    subgraph : networkx.Graph
        Input graph to search for cliques.
    min_number_of_nodes : int, default 1
        Minimum clique size to include.
    max_number_of_nodes : int, optional
        Maximum clique size to include. Defaults to size of subgraph.

    Returns
    -------
    list[list]
        List of cliques (as lists of node IDs) whose size lies in [min, max].
    """
    if max_number_of_nodes is None:
        max_number_of_nodes = subgraph.number_of_nodes()
    cliques = nx.enumerate_all_cliques(subgraph)
    components = list(filter(lambda x: min_number_of_nodes <= len(x) <= max_number_of_nodes, cliques))
    return components

@curry
def clique(
    quotient_graph: 'QuotientGraph',
    number_of_nodes=(1,3)
    ) -> 'QuotientGraph':
    """Emit one image node per clique of bounded size.
    Summary
        For each subgraph, enumerate all cliques (fully connected subgraphs) whose
        number of nodes lies between given bounds. Each clique becomes a new image node.

    Semantics
        - Input QG state: Uses quotient_graph.preimage_graph and image-node associations.
        - Output QG state: New QuotientGraph with image nodes corresponding to cliques.
        - Determinism: Deterministic given NetworkX’s clique enumeration.

    Parameters
        number_of_nodes : int | tuple[int,int], default (1,3)
            Inclusive range of clique sizes (number of nodes).

    Algorithm
        - Normalize number_of_nodes with value_to_2tuple().
        - For each subgraph association:
            * Call clique_decomposition_function(subgraph, min, max).
            * For each clique, create an image node with association=subgraph induced by that clique.

    Complexity
        Clique enumeration can be exponential in graph density and size.
        - Worst case: O(3^(n/3)) cliques.
        - Bounded by min/max size to limit blow-up.

    Metadata
        - Each image node stores 'association' (clique subgraph) and 'meta'
          with source_function='clique' and params.

    Interactions
        - Complements path(), cycle(), and graphlet() as structural motif extractors.
        - Useful in chemistry for aromatic rings or fully bonded functional groups.
        - In social networks, corresponds to tightly-knit groups.

    Examples
        # Extract cliques of size 3–5
        workflow = forward_compose(
            connected_component(),
            clique(number_of_nodes=(3,5))
        )

    Domain Analogies
        - Social networks: close friendship groups where everyone knows each other.
        - Computer networks: fully interconnected subnetworks.
        - Chemistry: ring systems where all atoms are bonded to each other.

    Failure Modes & Diagnostics
        - Large dense graphs may yield many cliques (combinatorial explosion).
        - Single-node cliques are included unless filtered by number_of_nodes.
    """
    number_of_nodes = value_to_2tuple(number_of_nodes)
    out_quotient_graph = QuotientGraph(
        graph=quotient_graph.preimage_graph,
        label_function=quotient_graph.label_function,
        attribute_function=quotient_graph.attribute_function,
        edge_function=quotient_graph.edge_function,
    )

    for subgraph in quotient_graph.get_image_nodes_associations():
        components = clique_decomposition_function(
            subgraph,
            min_number_of_nodes=min(number_of_nodes),
            max_number_of_nodes=max(number_of_nodes)
        )
        for component in components:
            out_quotient_graph.create_image_node_with_subgraph_from_nodes(
                component,
                meta=build_meta_from_function_context()
            )
    
    return out_quotient_graph

#--------------------------------------------------------------------------------    
@curry
def complement(
    quotient_graph: 'QuotientGraph',
    param=None
    ) -> 'QuotientGraph':
    """Emit image nodes representing the complement of each subgraph.
    Summary
        For every subgraph associated with an image node, create a new image node
        whose association consists of all preimage nodes *not* in the original subgraph.

    Semantics
        - Input QG state: Reads the preimage_graph node set and current image-node associations.
        - Output QG state: New QuotientGraph where each image node corresponds to the complement
          node set of its input association.
        - Determinism: Deterministic given the input graph and associations.

    Parameters
        param : ignored
            Present only for consistency with curried operator signatures.

    Algorithm
        - For each image-node subgraph:
            * Collect its node set.
            * Compute set difference with all nodes in preimage_graph.
            * Create a new image node with that complementary node set.

    Complexity
        - Time: O(N) per subgraph, where N is number of nodes in preimage_graph.
        - Memory: proportional to output size (number of complement sets).

    Metadata
        - Each output image node stores 'association' (induced subgraph of complement nodes)
          and 'meta' with source_function='complement'.

    Interactions
        - Often paired with cycle(), clique(), or path() to model inside–outside relationships.
        - Can be chained to build families like (subgraph, complement) pairs for contrastive features.
        - Useful for “negative space” reasoning: what is *not* included in a motif.

   

    Examples
        # Generate complements of connected components
        workflow = forward_compose(
            connected_component(),
            complement()
        )

    Domain Analogies
        - Social networks: people not in a given community.
        - Computer networks: devices outside a given subnet.
        - Chemistry: atoms outside a functional group.

    Failure Modes & Diagnostics
        - Empty complements arise if subgraph = entire preimage_graph.
        - Full complements (all nodes) arise if subgraph = ∅ (rare).
        - Complement size may dwarf the original subgraph; consider filtering.
    """
    out_quotient_graph = QuotientGraph(
        graph=quotient_graph.preimage_graph,
        label_function=quotient_graph.label_function,
        attribute_function=quotient_graph.attribute_function,
        edge_function=quotient_graph.edge_function,
    )

    for subgraph in quotient_graph.get_image_nodes_associations():
        component = list(subgraph.nodes())
        component = set(quotient_graph.preimage_graph.nodes()).difference(set(component))
        out_quotient_graph.create_image_node_with_subgraph_from_nodes(
            component,
            meta=build_meta_from_function_context()
        )
    
    return out_quotient_graph

#--------------------------------------------------------------------------------    
def betweenness_centrality_decomposition_function(subgraph, number_of_nodes=1, use_perifery=False):
    """Select nodes by betweenness centrality score.

    Parameters
    ----------
    subgraph : networkx.Graph
        Input graph to analyse.
    number_of_nodes : int, default 1
        Number of nodes to return.
    use_perifery : bool, default False
        If False, return the most central nodes; if True, return the least central nodes.

    Returns
    -------
    list[list]
        Single-element list containing a list of selected node IDs.
    """
    n_dict = nx.betweenness_centrality(subgraph)
    reverse = not use_perifery
    selected_ids = sorted(n_dict, key=lambda x: n_dict[x], reverse=reverse)[:number_of_nodes]
    components = [selected_ids] 
    return components

@curry
def betweenness_centrality(
    quotient_graph: 'QuotientGraph',
    number_of_nodes=1,
    use_perifery=False
    ) -> 'QuotientGraph':
    """Emit image nodes for nodes ranked by betweenness centrality.
    Summary
        For each subgraph, compute betweenness centrality scores and select either
        the top-k most central nodes or the bottom-k least central nodes. Each selected
        set is emitted as a new image node.

    Semantics
        - Input QG state: Reads preimage_graph and image-node associations.
        - Output QG state: New QuotientGraph with image nodes corresponding to
          sets of central or peripheral nodes.
        - Determinism: Deterministic given the input graph and parameters.

    Parameters
        number_of_nodes : int, default 1
            Number of nodes to select.
        use_perifery : bool, default False
            If False, select top central nodes. If True, select least central nodes.

    Algorithm
        - Compute betweenness centrality on each subgraph with NetworkX.
        - Sort nodes by centrality score (descending unless use_perifery=True).
        - Take the first `number_of_nodes`.
        - Create a new image node for that set.

    Complexity
        Betweenness centrality is O(|V||E|) on unweighted graphs.
        - Cost grows quickly with graph size; suitable mainly for small subgraphs.

    Metadata
        - Each image node stores 'association' (selected node set) and 'meta'
          with source_function='betweenness_centrality' and params.

    Interactions
        - Complements structural operators like cycle(), path(), or clique()
          by capturing centrality instead of topology alone.
        - Useful for filtering to “hub” or “periphery” roles in networks.

    Examples
        # Select 2 most central nodes in each connected component
        workflow = forward_compose(
            connected_component(),
            betweenness_centrality(number_of_nodes=2, use_perifery=False)
        )

    Domain Analogies
        - Social networks: most influential users (high centrality) vs. fringe users (low centrality).
        - Transportation: hub airports vs. peripheral stops.
        - Biology: bottleneck proteins in interaction networks.

    Failure Modes & Diagnostics
        - On large subgraphs, centrality computation may be expensive.
        - If number_of_nodes exceeds subgraph size, returns all nodes.
    """
    out_quotient_graph = QuotientGraph(
        graph=quotient_graph.preimage_graph,
        label_function=quotient_graph.label_function,
        attribute_function=quotient_graph.attribute_function,
        edge_function=quotient_graph.edge_function,
    )

    for subgraph in quotient_graph.get_image_nodes_associations():
        components = betweenness_centrality_decomposition_function(
            subgraph,
            number_of_nodes=number_of_nodes,
            use_perifery=use_perifery
        )
        for component in components:
            out_quotient_graph.create_image_node_with_subgraph_from_nodes(
                component,
                meta=build_meta_from_function_context()
            )
    
    return out_quotient_graph

#--------------------------------------------------------------------------------    
@curry
def merge(
    quotient_graph: 'QuotientGraph',
    use_edges=False
    ) -> 'QuotientGraph':
    """Merge all subgraphs into a single image node.
    Summary
        Combine the contents of all subgraph associations into one new image node.
        By default, collects all nodes; if `use_edges=True`, collects all edges instead.

    Semantics
        - Input QG state: Reads associations from all current image nodes.
        - Output QG state: New QuotientGraph with a single image node containing
          either the union of all nodes or the union of all edges.
        - Determinism: Deterministic union, order of accumulation does not matter.

    Parameters
        use_edges : bool, default False
            If False, merge all node sets into one.  
            If True, merge all edge sets into one.

    Algorithm
        - Initialize an empty component.
        - For each subgraph association:
            * Collect nodes (default) or edges (`use_edges=True`).
            * Extend the component list.
        - Create a single image node with this combined component.

    Complexity
        - Time: O(sum of sizes of all subgraphs).  
        - Memory: proportional to merged node or edge set.

    Metadata
        - Each output image node stores 'association' (merged subgraph) and 'meta'
          with source_function='merge' and params.

    Interactions
        - Useful as a “collapsing” step after decomposition, producing a single
          representative subgraph.
        - Can be paired with complement() or filters for aggregate reasoning.

    Examples
        # Merge all connected components into one node set
        workflow = forward_compose(
            connected_component(),
            merge()
        )

        # Merge all edges from cycle and tree decomposition
        workflow = add(cycle(), tree())
        workflow = forward_compose(workflow, merge(use_edges=True))

    Domain Analogies
        - Social networks: treat multiple communities as one collective group.
        - Computer networks: aggregate all subnetworks into one backbone.
        - Chemistry: union of functional groups into a single motif.

    Failure Modes & Diagnostics
        - If associations are empty, creates a single empty image node.
        - `use_edges=True` produces edges but may result in disconnected node sets.
    """
    out_quotient_graph = QuotientGraph(
        graph=quotient_graph.preimage_graph,
        label_function=quotient_graph.label_function,
        attribute_function=quotient_graph.attribute_function,
        edge_function=quotient_graph.edge_function,
    )

    if use_edges:
        component = []
        for subgraph in quotient_graph.get_image_nodes_associations():
            component.extend(subgraph.edges())
        out_quotient_graph.create_image_node_with_subgraph_from_nodes(
            component,
            meta=build_meta_from_function_context()
        )
    else:
        component = []
        for subgraph in quotient_graph.get_image_nodes_associations():
            component.extend(subgraph.nodes())
        out_quotient_graph.create_image_node_with_subgraph_from_nodes(
            component,
            meta=build_meta_from_function_context()
        )
    
    return out_quotient_graph

#--------------------------------------------------------------------------------    
@curry
def intersection(
    quotient_graph: 'QuotientGraph',
    node_size=None,
    must_be_connected: bool = True
) -> 'QuotientGraph':
    """Emit image nodes for intersections of every pair of associated subgraphs.
    Summary
        For each unordered pair of image nodes in the input QuotientGraph, compute the
        intersection of their associated subgraphs' node sets. If the intersection size
        is within the inclusive range `node_size`, create a new image node whose
        association is the induced subgraph on those intersecting nodes.

    Semantics
        - Input QG state: Reads associations from current image nodes and the preimage_graph.
        - Output QG state: New QuotientGraph with one image node per qualifying intersection.
        - Determinism: Deterministic given the input graph and `node_size`.

    Parameters
        node_size : None | int | tuple[int,int], default None
            When None, no size filtering is applied to the intersection.
            If an int k is given, it is treated as (k,k). If a tuple (min,max)
            is given, the intersection size must satisfy min ≤ |I| ≤ max.
        must_be_connected : bool, default True
            If True, accept the intersection only when its induced subgraph on the
            preimage graph forms exactly one connected component.

    Algorithm
        - Iterate over all unordered pairs of image nodes (u, v), u < v.
        - Compute intersection I = nodes(assoc[u]) ∩ nodes(assoc[v]).
        - If min(node_size) ≤ |I| ≤ max(node_size), create an image node with association = induced subgraph on I.

    Complexity
        - Time: O(M^2 · d) where M = number of image nodes, d = average subgraph size.
        - Memory: proportional to number and size of emitted intersections.

    Interactions
        - Complements `intersection_edges`, but creates new image nodes instead of edges.
        - Often followed by `filter_by_number_of_nodes` or connectivity filters.

    Examples
        # Intersections among neighborhoods of radius 1 with size between 2 and 5
        workflow = forward_compose(
            neighborhood(radius=1),
            intersection(node_size=(2,5))
        )
    """
    # Normalise size bounds if provided
    if node_size is not None:
        node_size = value_to_2tuple(node_size)

    out_quotient_graph = QuotientGraph(
        graph=quotient_graph.preimage_graph,
        label_function=quotient_graph.label_function,
        attribute_function=quotient_graph.attribute_function,
        edge_function=quotient_graph.edge_function,
    )

    # Work on unordered pairs to avoid duplicates (u,v) and (v,u)
    # Additionally, deduplicate identical intersections across different pairs.
    seen = set()  # set[frozenset]
    img_nodes = list(quotient_graph.image_graph.nodes())
    for u, v in combinations(img_nodes, 2):
        sub_u = quotient_graph.image_graph.nodes[u].get('association')
        sub_v = quotient_graph.image_graph.nodes[v].get('association')
        if sub_u is None or sub_v is None:
            continue
        inter_nodes = set(sub_u.nodes()).intersection(sub_v.nodes())
        inter_len = len(inter_nodes)

        # Apply optional size filtering
        size_ok = True
        if node_size is not None:
            size_ok = (min(node_size) <= inter_len <= max(node_size))
        if not size_ok:
            continue

        # Apply optional connectivity constraint
        if must_be_connected:
            if inter_len == 0:
                continue
            induced = quotient_graph.preimage_graph.subgraph(inter_nodes)
            try:
                cc_count = len(list(nx.connected_components(induced)))
            except Exception:
                cc_count = 0
            if cc_count != 1:
                continue

        key = frozenset(inter_nodes)
        if key in seen:
            continue
        seen.add(key)

        out_quotient_graph.create_image_node_with_subgraph_from_nodes(
            inter_nodes,
            meta=build_meta_from_function_context()
        )

    return out_quotient_graph

#--------------------------------------------------------------------------------   
def get_distance(graph1, graph2, basegraph):
    """Compute shortest-path distance between two subgraphs.

    Parameters
    ----------
    graph1, graph2 : networkx.Graph
        Subgraphs to compare (nodes only are used).
    basegraph : networkx.Graph
        Full graph in which distances are computed.

    Returns
    -------
    int
        Length of shortest path between any node in graph1 and any node in graph2.
    """
    return min(nx.shortest_path_length(basegraph, source=u, target=v)
               for u in graph1.nodes() for v in graph2.nodes())


def get_distance_matrix(subgraphs1, subgraphs2, basegraph, min_distance, max_distance):
    """Compute pairwise distance matrix between two sets of subgraphs."""
    distance_matrix = np.full((len(subgraphs1), len(subgraphs2)), np.nan)
    for i, subgraph_i in enumerate(subgraphs1):
        for j, subgraph_j in enumerate(subgraphs2):
            try:
                dist = get_distance(subgraph_i, subgraph_j, basegraph)
                if min_distance <= dist <= max_distance:
                    distance_matrix[i, j] = dist
            except Exception:
                pass  # Keep as NaN
    return distance_matrix


def all_distances_are_feasible(combination_idxs, distance_matrix):
    """Check if all pairwise distances within a combination are valid.

    Parameters
    ----------
    combination_idxs : iterable[int]
        Indices of subgraphs in the combination.
    distance_matrix : np.ndarray
        Pairwise distance matrix.

    Returns
    -------
    bool
        True if all pairwise distances are finite (non-NaN), else False.
    """
    pairs = combinations(combination_idxs, 2)
    for i, j in pairs:
        distance = distance_matrix[i, j]
        if np.isnan(distance):
            return False
    return True


def combination_decomposition_function(subgraphs, graph,
                                       number_of_elements=(2, 2),
                                       distance=(0, 1)):
    """Combine subgraphs into larger components based on distance constraints.

    Parameters
    ----------
    subgraphs : list[networkx.Graph]
        Input subgraphs to combine.
    graph : networkx.Graph
        Full graph used for distance computation.
    number_of_elements : tuple(int, int), default (2,2)
        Min and max number of subgraphs to combine.
    distance : tuple(int, int), default (0,1)
        Acceptable range for pairwise distances between combined subgraphs.

    Returns
    -------
    list[set]
        List of combined node sets formed from feasible subgraph combinations.
    """
    # NOTE: get_distance_matrix expects (min_distance, max_distance). Passing them
    # in the wrong order would filter out all valid pairs. Ensure correct ordering.
    distance_matrix = get_distance_matrix(
        subgraphs,
        subgraphs,
        graph,
        min(distance),
        max(distance)
    )
    components = []
    component_combinations = [list(subgraph.nodes()) for subgraph in subgraphs]
    for order in range(min(number_of_elements), max(number_of_elements) + 1):
        combination_idxs_list = combinations(range(len(component_combinations)), order)
        for combination_idxs in combination_idxs_list:
            if distance_matrix is not None and not all_distances_are_feasible(combination_idxs, distance_matrix):
                continue
            component_combination = [component_combinations[idx] for idx in combination_idxs]
            component = set(node for nodes in component_combination for node in nodes)
            components.append(component)
    return components

@curry
def combination(
    quotient_graph: 'QuotientGraph',
    number_of_elements=(2,2),
    distance=(0,1)
    ) -> 'QuotientGraph':
    """Emit image nodes formed by combining multiple subgraphs subject to distance constraints.
    Summary
        For each feasible combination of subgraphs, create a new image node whose
        association is the union of their node sets. A combination is feasible if:
        - The number of subgraphs lies within `number_of_elements`.
        - All pairwise distances between them (measured in preimage_graph) fall
          within the specified `distance` range.

    Semantics
        - Input QG state: Consumes the current set of image-node associations.
        - Output QG state: New QuotientGraph with image nodes representing unions
          of feasible subgraph combinations.
        - Determinism: Deterministic given input graph and parameters.

    Parameters
        number_of_elements : tuple(int,int), default (2,2)
            Minimum and maximum number of subgraphs to combine.
        distance : tuple(int,int), default (0,1)
            Inclusive range of allowed shortest-path distances between any two
            subgraphs in the combination.

    Algorithm
        - Build distance matrix for all pairs of subgraphs.
        - Enumerate all subgraph combinations of size within bounds.
        - Keep only those with feasible distances.
        - Form union of node sets for each valid combination.
        - Emit one image node per union.

    Complexity
        - Distance matrix: O(k² * |V| + |E|), where k = number of subgraphs.
        - Combinations: exponential in k, practical only for small subgraph sets.

    Metadata
        - Each output image node stores 'association' (unioned node set) and 'meta'
          with source_function='combination' and params.

    Interactions
        - Generalises cycle()+complement() patterns by considering multiple subgraphs together.
        - Can express multi-part motifs: e.g., “two cycles within distance 3”.

    Examples
        # Combine cycles (from QG1) with degree-1 nodes (from QG2) within 2 hops
        qg1 = forward_compose(connected_component(), cycle())(Q)
        qg2 = forward_compose(node(), degree(value=1))(Q)
        workflow = forward_compose(
            lambda _: qg1,  # assuming wrappers that return fixed QGs
            lambda _: qg2,
        )
        # Combine them:
        out = binary_combination(qg1, qg2, distance=(0,2))

    Domain Analogies
        - Social networks: groups formed by nearby communities.
        - Computer networks: subnetworks within bounded latency.
        - Chemistry: functional groups within certain bond distance.

    Failure Modes & Diagnostics
        - Infeasible parameters may yield zero combinations.
        - Large number_of_elements can explode runtime.
    """
    number_of_elements = value_to_2tuple(number_of_elements)
    distance = value_to_2tuple(distance)
    out_quotient_graph = QuotientGraph(
        graph=quotient_graph.preimage_graph,
        label_function=quotient_graph.label_function,
        attribute_function=quotient_graph.attribute_function,
        edge_function=quotient_graph.edge_function,
    )

    components = combination_decomposition_function(
        quotient_graph.get_image_nodes_associations(), 
        quotient_graph.preimage_graph, 
        number_of_elements=number_of_elements, 
        distance=distance
    )
    for component in components:
        out_quotient_graph.create_image_node_with_subgraph_from_nodes(
            component,
            meta=build_meta_from_function_context()
        )

    return out_quotient_graph


#====================================================================================================
# EDGE OPERATORS
#====================================================================================================
@curry
def intersection_edges(
    quotient_graph: 'QuotientGraph',
    size_threshold=1,
    accept_connection_by_edge=False
) -> 'QuotientGraph':
    """Add edges between image nodes whose subgraphs overlap or are adjacent.
    Summary
        For each pair of image nodes, add an edge in the image graph if:
        - Their associated subgraphs share at least `size_threshold` nodes, OR
        - (optional) any node in one subgraph is directly connected by an edge
          in the preimage graph to a node in the other subgraph
          (`accept_connection_by_edge=True`).

    Semantics
        - Input QG state: Reads image-node associations and preimage graph.
        - Output QG state: Returns a new QuotientGraph with extra edges
          added between image nodes satisfying the criteria.
        - Determinism: Deterministic given the input graph and parameters.

    Parameters
        size_threshold : int, default 1
            Minimum number of shared nodes between subgraphs required to add an edge.
        accept_connection_by_edge : bool, default False
            If True, also connect image nodes if any pair of their preimage nodes
            share an edge in the preimage graph.

    Algorithm
        - Iterate over all ordered pairs of image nodes (u,v).
        - Extract node sets from each subgraph association.
        - If |intersection| ≥ size_threshold, mark as connected.
        - If `accept_connection_by_edge=True`, also scan preimage edges
          between the node sets.
        - Add edge (u,v) if condition holds.

    Complexity
        - Time: O(M² * d) where M = number of image nodes and d is average subgraph size.
        - Memory: O(1) beyond input graphs and new image edges.

    Metadata
        - Output image edges are unlabeled unless edge_function later annotates them.

    Interactions
        - Complements decomposition operators: can create higher-level
          adjacency among substructures (e.g., overlapping cycles).
        - Useful for building “graph of motifs” where motifs are linked
          if they overlap or touch.

    Examples
        # Connect overlapping neighborhoods
        workflow = forward_compose(
            neighborhood(radius=(1,2)),
            intersection_edges(size_threshold=2)
        )

        # Connect cycles if they are adjacent in the molecule
        workflow = forward_compose(
            cycle(),
            intersection_edges(size_threshold=0, accept_connection_by_edge=True)
        )

    Domain Analogies
        - Social networks: overlap between friend groups.
        - Computer networks: subnetworks sharing routers.
        - Chemistry: functional groups sharing atoms or bonds.

    Failure Modes & Diagnostics
        - Large image graphs (many nodes) may make pairwise checks costly.
        - If no overlaps and `accept_connection_by_edge=False`, output graph
          may be entirely disconnected.
    """
    out_quotient_graph = QuotientGraph(quotient_graph=quotient_graph)
    
    # Determine the graph to use for edge queries.
    if isinstance(quotient_graph.preimage_graph, QuotientGraph):
        pre_img = quotient_graph.preimage_graph.image_graph
    else:
        pre_img = quotient_graph.preimage_graph

    for u in quotient_graph.image_graph.nodes():
        subgraph_u = quotient_graph.image_graph.nodes[u]['association']
        nodes_u = list(subgraph_u.nodes())
        for v in quotient_graph.image_graph.nodes():
            if u != v:
                subgraph_v = quotient_graph.image_graph.nodes[v]['association']
                nodes_v = list(subgraph_v.nodes())
                
                # Flag to decide whether to add an edge from u to v.
                add_edge = False
                
                # Normal behavior: add edge if the intersection size meets the threshold.
                if len(set(nodes_u).intersection(set(nodes_v))) >= size_threshold:
                    add_edge = True
                
                if accept_connection_by_edge:
                    # Check for any pre_image edge between nodes_u and nodes_v.
                    for node_u in nodes_u:
                        for node_v in nodes_v:
                            if pre_img.has_edge(node_u, node_v):
                                add_edge = True
                                break
                        if add_edge:
                            break
                
                if add_edge:
                    out_quotient_graph.image_graph.add_edge(u, v)
    
    return out_quotient_graph

#--------------------------------------------------------------------------------    
# FILTER OPERATORS
#--------------------------------------------------------------------------------    
@curry
def filter_by_number_of_connected_components(
    quotient_graph: 'QuotientGraph',
    number_of_components=(1,1)
    ) -> 'QuotientGraph':
    """Filter subgraphs by their number of connected components.
    Summary
        Retain only those subgraphs whose number of connected components
        falls within the specified interval.

    Semantics
        - Input QG state: Reads node associations of current image nodes.
        - Output QG state: Returns a new QuotientGraph containing only
          image nodes whose subgraphs satisfy the component-count filter.
        - Determinism: Deterministic given input and parameter range.

    Parameters
        number_of_components : tuple(int,int), default (1,1)
            Inclusive range [min,max] for the number of connected components
            allowed. E.g. (1,1) keeps only connected subgraphs.

    Algorithm
        - For each image-node subgraph:
            * Compute connected components with `nx.connected_components`.
            * Count how many components exist.
            * Keep the subgraph if count ∈ [min,max].
        - Discard otherwise.

    Complexity
        - Time: O(|V|+|E|) per subgraph (connected-components computation).
        - Memory: proportional to node count of the kept subgraphs.

    Metadata
        - Each surviving image node keeps original meta with added provenance
          (source_function='filter_by_number_of_connected_components').

    Interactions
        - Often used to enforce connectedness after decomposition
          (e.g., keeping only connected cliques).
        - Can prune trivial decompositions with too many disconnected pieces.

    Examples
        # Keep only connected neighborhoods
        workflow = forward_compose(
            neighborhood(radius=(1,2)),
            filter_by_number_of_connected_components((1,1))
        )

        # Allow up to 3 components
        workflow = forward_compose(
            complement(),
            filter_by_number_of_connected_components((1,3))
        )

    Domain Analogies
        - Social networks: require groups to be internally connected.
        - Computer networks: retain subnets with limited fragmentation.
        - Chemistry: filter fragments to ensure they form a connected molecule.

    Failure Modes & Diagnostics
        - Subgraphs with zero nodes yield 0 components and will be discarded
          unless 0 lies in the range.
        - Overly narrow ranges may filter out all subgraphs.
    """
    number_of_components = value_to_2tuple(number_of_components)
    out_quotient_graph = QuotientGraph(
        graph=quotient_graph.preimage_graph,
        label_function=quotient_graph.label_function,
        attribute_function=quotient_graph.attribute_function,
        edge_function=quotient_graph.edge_function,
    )

    for subgraph in quotient_graph.get_image_nodes_associations():
        cc = list(nx.connected_components(subgraph))
        if min(number_of_components) <= len(cc) <= max(number_of_components):
            out_quotient_graph.create_image_node_with_subgraph_from_nodes(subgraph.nodes())
    
    return out_quotient_graph

#--------------------------------------------------------------------------------    
@curry
def filter_by_number_of_nodes(
    quotient_graph: 'QuotientGraph',
    number_of_nodes=(1,10)
    ) -> 'QuotientGraph':
    """Filter subgraphs by their node count.
    Summary
        Retain only those subgraphs whose number of nodes lies within
        the specified inclusive range.

    Semantics
        - Input QG state: Reads node associations of current image nodes.
        - Output QG state: Returns a new QuotientGraph containing only
          image nodes whose subgraphs satisfy the node-count constraint.
        - Determinism: Deterministic given input and parameter range.

    Parameters
        number_of_nodes : tuple(int,int), default (1,10)
            Inclusive range [min,max] for the number of nodes allowed
            in each subgraph.

    Algorithm
        - For each image-node subgraph:
            * Compute its node count.
            * Keep the subgraph if count ∈ [min,max].
        - Discard otherwise.

    Complexity
        - Time: O(1) per subgraph (node count lookup).
        - Memory: proportional to number of retained subgraphs.

    Metadata
        - Each surviving image node keeps its original meta,
          with provenance marking the filter application.

    Interactions
        - Often paired with decomposition operators to constrain
          the granularity of extracted substructures (e.g., paths,
          cliques, neighborhoods).
        - Helps control combinatorial explosion by filtering overly
          large or trivial subgraphs.

    Examples
        # Keep only small cliques (≤ 5 nodes)
        workflow = forward_compose(
            clique(),
            filter_by_number_of_nodes((1,5))
        )

        # Keep only subgraphs of medium size (10–20 nodes)
        workflow = forward_compose(
            connected_component(),
            filter_by_number_of_nodes((10,20))
        )

    Domain Analogies
        - Social networks: keep groups within a desired size band.
        - Computer networks: select subnetworks of bounded size.
        - Chemistry: retain fragments with an atom count in a range.

    Failure Modes & Diagnostics
        - Empty subgraphs (0 nodes) are discarded unless 0 is in range.
        - Narrow ranges may filter out all subgraphs.
    """
    number_of_nodes = value_to_2tuple(number_of_nodes)
    out_quotient_graph = QuotientGraph(
        graph=quotient_graph.preimage_graph,
        label_function=quotient_graph.label_function,
        attribute_function=quotient_graph.attribute_function,
        edge_function=quotient_graph.edge_function,
    )

    for subgraph in quotient_graph.get_image_nodes_associations():
        if subgraph.number_of_nodes() >= min(number_of_nodes):
            if subgraph.number_of_nodes() <= max(number_of_nodes): 
                out_quotient_graph.create_image_node_with_subgraph_from_nodes(subgraph.nodes())
    
    return out_quotient_graph

#--------------------------------------------------------------------------------    
@curry
def filter_by_number_of_edges(
    quotient_graph: 'QuotientGraph',
    number_of_edges=(1,10)
    ) -> 'QuotientGraph':
    """Filter subgraphs by their edge count.
    Summary
        Retain only those subgraphs whose number of edges lies within
        the specified inclusive range.

    Semantics
        - Input QG state: Reads edge sets of current image-node subgraphs.
        - Output QG state: Returns a new QuotientGraph containing only
          image nodes whose subgraphs satisfy the edge-count constraint.
        - Determinism: Deterministic given input and parameter range.

    Parameters
        number_of_edges : tuple(int,int), default (1,10)
            Inclusive range [min,max] for the number of edges allowed
            in each subgraph.

    Algorithm
        - For each image-node subgraph:
            * Compute its edge count.
            * Keep the subgraph if count ∈ [min,max].
        - Discard otherwise.

    Complexity
        - Time: O(1) per subgraph (edge count lookup).
        - Memory: proportional to number of retained subgraphs.

    Metadata
        - Each surviving image node keeps its original meta,
          with provenance marking the filter application.

    Interactions
        - Complements filter_by_number_of_nodes by constraining edge
          density explicitly.
        - Helps discard trivial or overly dense subgraphs depending on task.

    Examples
        # Keep only sparse neighborhoods with ≤ 5 edges
        workflow = forward_compose(
            neighborhood(radius=(1,2)),
            filter_by_number_of_edges((1,5))
        )

        # Retain only large dense cliques (≥ 20 edges)
        workflow = forward_compose(
            clique(),
            filter_by_number_of_edges((20,100))
        )

    Domain Analogies
        - Social networks: filter groups by number of relationships.
        - Computer networks: retain subnetworks with bounded link counts.
        - Chemistry: restrict molecular fragments by bond count.

    Failure Modes & Diagnostics
        - Subgraphs with zero edges are discarded unless 0 is in range.
        - Narrow ranges may filter out all subgraphs.
    """
    number_of_edges = value_to_2tuple(number_of_edges)
    out_quotient_graph = QuotientGraph(
        graph=quotient_graph.preimage_graph,
        label_function=quotient_graph.label_function,
        attribute_function=quotient_graph.attribute_function,
        edge_function=quotient_graph.edge_function,
    )

    for subgraph in quotient_graph.get_image_nodes_associations():
        if subgraph.number_of_edges() >= min(number_of_edges):
            if subgraph.number_of_edges() <= max(number_of_edges): 
                out_quotient_graph.create_image_node_with_subgraph_from_nodes(subgraph.nodes())
    
    return out_quotient_graph

#--------------------------------------------------------------------------------    
@curry
def filter_by_node_label(
    quotient_graph: 'QuotientGraph',
    key='label',
    must_have_one_of=None,
    cannot_have_any_in=None
    ) -> 'QuotientGraph':
    """Filter subgraphs by the labels of their constituent nodes.
    Summary
        Retain only those subgraphs that (optionally) contain at least one node
        with a label in `must_have_one_of` and contain no nodes with labels
        in `cannot_have_any_in`.

    Semantics
        - Input QG state: Reads node attributes of current image-node subgraphs.
        - Output QG state: Returns a new QuotientGraph containing only
          image nodes that satisfy both inclusion and exclusion criteria.
        - Determinism: Deterministic given input graph and label sets.

    Parameters
        key : str, default "label"
            The attribute key to inspect on each node.
        must_have_one_of : list, default []
            If non-empty, subgraphs must contain at least one node whose
            `key` value is in this list.
        cannot_have_any_in : list, default []
            If non-empty, subgraphs are discarded if they contain any node
            whose `key` value is in this list.

    Algorithm
        - For each image-node subgraph:
            * Check if at least one node’s label ∈ must_have_one_of
              (if constraint provided).
            * Check that no node’s label ∈ cannot_have_any_in
              (if constraint provided).
            * Keep subgraph only if both conditions are met.
        - Discard otherwise.

    Complexity
        - Time: O(|V_sub|) per subgraph (node scan).
        - Memory: O(1) additional per subgraph.

    Metadata
        - Retained subgraphs inherit original metadata,
          with provenance of the filter operation.

    Interactions
        - Complements structural filters (by size or connectivity) with
          semantic filtering.
        - Enables domain-specific constraints (e.g., presence/absence of
          atom types in chemistry, role labels in social networks).

    Examples
        # Keep only subgraphs with at least one oxygen atom
        workflow = forward_compose(
            neighborhood(radius=(1,2)),
            filter_by_node_label(key='atom', must_have_one_of=['O'])
        )

        # Keep only user groups containing "admin" but no "banned"
        workflow = forward_compose(
            connected_component(),
            filter_by_node_label(key='role', must_have_one_of=['admin'], cannot_have_any_in=['banned'])
        )

    Domain Analogies
        - Social networks: require at least one influencer; exclude any group with bots.
        - Computer networks: keep subnets with a router, exclude those with deprecated nodes.
        - Chemistry: retain fragments with a functional atom, exclude toxic groups.

    Failure Modes & Diagnostics
        - If all constraints are empty, all subgraphs are passed through unchanged.
        - Overly strict filters may yield zero surviving subgraphs.
        - If the `key` attribute is missing on nodes, default `0` is used
          and constraints may fail unexpectedly.
    """
    must_have_one_of = [] if must_have_one_of is None else must_have_one_of
    cannot_have_any_in = [] if cannot_have_any_in is None else cannot_have_any_in
    
    out_quotient_graph = QuotientGraph(
        graph=quotient_graph.preimage_graph,
        label_function=quotient_graph.label_function,
        attribute_function=quotient_graph.attribute_function,
        edge_function=quotient_graph.edge_function,
    )

    for subgraph in quotient_graph.get_image_nodes_associations():
        if len(must_have_one_of) > 0:
            must_conditions_are_met = False
            for u in subgraph.nodes(): 
                if subgraph.nodes[u].get(key,0) in must_have_one_of:
                    must_conditions_are_met = True
                    break
        else:
            must_conditions_are_met = True

        if len(cannot_have_any_in) > 0:
            cannot_conditions_are_met = True
            for u in subgraph.nodes(): 
                if subgraph.nodes[u].get(key,0) in cannot_have_any_in:
                    cannot_conditions_are_met = False
                    break
        else:
            cannot_conditions_are_met = True

        if must_conditions_are_met and cannot_conditions_are_met:
            out_quotient_graph.create_image_node_with_subgraph_from_nodes(subgraph.nodes())
    
    return out_quotient_graph

#--------------------------------------------------------------------------------    
@curry
def select_top_by_feature_ranking(
    quotient_graph: 'QuotientGraph',
    ranked_features,
    max_num: int = 1,
) -> 'QuotientGraph':
    """Select top-K image nodes based on an external feature-importance ranking.
    Summary
        Given an ordered list of feature IDs (most important first), rank all
        current image nodes by their label and retain only the top `max_num`.

    Semantics
        - Input QG state: Reads image-node labels (computes them on the fly
          for nodes missing a 'label' attribute using the QG's label_function).
        - Output QG state: Returns a new QuotientGraph containing only the
          selected image nodes (their association subgraphs are copied). Operator
          settings (label/attribute/edge functions) are preserved.

    Parameters
        ranked_labels : Sequence[int] or Mapping[int, float]
            Label IDs in descending importance order, or a mapping from label
            to importance score. Labels not present receive lowest priority.
        max_num : int, default 1
            Number of top-ranked image nodes to keep (globally across the image graph).

    Notes
        - If labels are provided as an ordering, nodes are ranked by the index
          position (lower index = higher priority).
        - If a mapping is provided, nodes are ranked by score (higher is better).
        - Image-node labels are treated as integers consistent with hashing-based
          label functions. Labels not found in the ranking are assigned worst rank.
    """
    # Build scoring function from input ranking.
    score_map = None
    is_mapping = hasattr(ranked_features, 'items')
    if is_mapping:
        score_map = dict(ranked_features)
    else:
        # Higher priority → larger score; use reverse index so index 0 gets max score.
        n = len(ranked_features)
        score_map = {lbl: (n - i) for i, lbl in enumerate(ranked_features)}

    # Gather candidates with scores.
    candidates = []  # (score, node_id, association, meta)
    for node_id, data in quotient_graph.image_graph.nodes(data=True):
        label = data.get('label', None)
        if label is None and getattr(quotient_graph, 'label_function', None) is not None:
            try:
                label = quotient_graph.label_function(data)
            except Exception:
                label = None
        if label is None:
            continue  # cannot rank without a label
        score = score_map.get(label, float('-inf'))
        if score == float('-inf'):
            continue  # skip labels not present in the ranking
        assoc = data.get('association')
        meta = data.get('meta')
        candidates.append((score, node_id, assoc, meta))

    # Sort and select top-K.
    candidates.sort(key=lambda x: (-x[0], x[1]))
    selected = candidates[: max(0, int(max_num)+1)]

    # Construct the output QuotientGraph with selected associations.
    out_quotient_graph = QuotientGraph(
        graph=quotient_graph.preimage_graph,
        label_function=quotient_graph.label_function,
        attribute_function=quotient_graph.attribute_function,
        edge_function=quotient_graph.edge_function,
    )

    for _, _, assoc, meta in selected:
        if assoc is not None:
            # Preserve metadata if available
            out_quotient_graph.create_image_node_with_subgraph_from_subgraph(assoc.copy(), meta=meta)

    return out_quotient_graph

#====================================================================================================
# BINARY OPERATORS
#====================================================================================================
def binary_combination_decomposition_function(subgraphs1, subgraphs2, graph, distance=(0,1)):
    """Combine one subgraph from set1 with one from set2 if their pairwise distance is within bounds.

    Parameters
    ----------
    subgraphs1, subgraphs2 : list[networkx.Graph]
        Two collections of subgraphs to pairwise combine (one from each set).
    graph : networkx.Graph
        Base graph used to compute shortest-path distances between subgraphs.
    distance : tuple(int, int), default (0,1)
        Inclusive [min, max] bounds on shortest-path distance between any node
        of a subgraph from set1 and any node of a subgraph from set2.

    Returns
    -------
    list[set]
        List of combined node sets (union of nodes from each valid pair).
    """
    # NOTE: get_distance_matrix signature is (min_distance, max_distance). Ensure
    # we pass the bounds in the correct order to avoid discarding valid matches.
    distance_matrix = get_distance_matrix(
        subgraphs1,
        subgraphs2,
        graph,
        min(distance),
        max(distance)
    )
    components = []
    component_combinations1 = [list(subgraph.nodes()) for subgraph in subgraphs1]
    component_combinations2 = [list(subgraph.nodes()) for subgraph in subgraphs2]
    combination_idxs_list = product(range(len(component_combinations1)), range(len(component_combinations2)))
    for combination_idxs in combination_idxs_list:
        if distance_matrix is not None and all_distances_are_feasible(combination_idxs, distance_matrix) is False:
            continue
        nodes1_list = [node for node in component_combinations1[combination_idxs[0]]]
        nodes2_list = [node for node in component_combinations2[combination_idxs[1]]]
        component = set(nodes1_list + nodes2_list)
        components.append(component)
    return components

@curry
def binary_combination(
    first_quotient_graph: 'QuotientGraph',
    second_quotient_graph: 'QuotientGraph',
    distance=(0,1)
    ) -> 'QuotientGraph':
    """Emit image nodes by pairing subgraphs from two QGs when their inter-distance is within bounds.
    Summary
        Take one associated subgraph from the first QuotientGraph and one from the second; if the
        shortest-path distance between them (in the shared preimage graph) lies within `distance`,
        emit a new image node whose association is the union of both subgraphs’ node sets.

    Semantics
        - Input QG state: Reads image-node associations from two input QGs and uses the first QG’s
          preimage_graph to compute distances.
        - Output QG state: Returns a new QuotientGraph (with the first QG’s preimage_graph) whose
          image nodes each represent a valid pairwise combination (union) of subgraphs.
        - Determinism: Deterministic given inputs and parameters.

    Parameters
        distance : int | tuple[int,int], default (0,1)
            Inclusive [min, max] bounds for allowed shortest-path distances between any node in a
            subgraph from the first set and any node in a subgraph from the second set. A scalar `d`
            is treated as (d, d).

    Algorithm
        - Normalize `distance` via value_to_2tuple().
        - Compute pairwise distances via `get_distance_matrix(subgraphs1, subgraphs2, basegraph, ...)`.
        - Enumerate all pairs (i, j); keep only those whose distance is finite and within bounds.
        - For each valid pair, form the union of nodes and create a new image node with that induced subgraph.
        - Attach provenance via `build_meta_from_function_context()`.

    Complexity
        - Distance matrix computation: O(k1 * k2 * D) where k1, k2 are counts of subgraphs and D is
          the cost of shortest paths between node pairs (depends on basegraph size/structure).
        - Pair enumeration: O(k1 * k2).
        - Practical usage suggests applying upstream filters to keep k1, k2 small.

    Side Effects & Metadata
        - Each emitted image node stores:
            * 'association' : induced subgraph on the unioned node set.
            * 'meta'        : {'source_function': 'binary_combination', 'params': {...}}.
        - Labels/attributes are not computed here; call `update()` to populate them.

    Interactions
        - Expresses cross-family motifs, e.g., “a cycle near a high-betweenness node” by pairing
          outputs of `cycle()` (first QG) and `betweenness_centrality()` (second QG).
        - Can be followed by size/connectivity filters to control combinatorial growth.

    Constraints & Invariants
        - Assumes both QGs refer to the same underlying preimage node ID space.
        - If either QG has zero associations, the output will be empty.
        - If `distance` is too strict, no pairs may be produced.

    Examples
        # Pair cycles (from QG1) with degree-1 nodes (from QG2) within 2 hops
        qg1 = forward_compose(connected_component(), cycle())(Q)
        qg2 = forward_compose(node(), degree(value=1))(Q)
        workflow = forward_compose(
            lambda _: qg1,  # assuming wrappers that return fixed QGs
            lambda _: qg2,
        )
        # Combine them:
        out = binary_combination(qg1, qg2, distance=(0,2))

    Domain Analogies
        - Social networks: pair a community with a nearby influencer set.
        - Computer networks: pair a subnet with a nearby gateway/router group.
        - Chemistry: pair a ring system with a nearby heteroatom set.

    Failure Modes & Diagnostics
        - Large k1/k2 or loose distance bounds can cause quadratic blow-up in pairs.
        - Tight bounds (e.g., distance=(0,0)) may yield no combinations if subgraphs do not touch.
        - Ensure both QGs share the same preimage graph; otherwise distances are undefined.
    """
    distance = value_to_2tuple(distance)
    out_quotient_graph = QuotientGraph(
        graph=first_quotient_graph.preimage_graph,
        label_function=first_quotient_graph.label_function,
        attribute_function=first_quotient_graph.attribute_function,
        edge_function=first_quotient_graph.edge_function,
    )

    components = binary_combination_decomposition_function(
        first_quotient_graph.get_image_nodes_associations(),
        second_quotient_graph.get_image_nodes_associations(),
        first_quotient_graph.preimage_graph,
        distance=distance
    )
    for component in components:
        out_quotient_graph.create_image_node_with_subgraph_from_nodes(
            component,
            meta=build_meta_from_function_context()
        )

    return out_quotient_graph

#--------------------------------------------------------------------------------    
@curry
def binary_intersection(
    first_quotient_graph: 'QuotientGraph',
    second_quotient_graph: 'QuotientGraph',
    node_size=None,
    must_be_connected: bool = True
) -> 'QuotientGraph':
    """Emit image nodes for intersections between subgraphs from two QuotientGraphs.
    Summary
        For each pair consisting of one associated subgraph from the first QuotientGraph
        and one from the second, compute the intersection of their node sets. If the
        intersection satisfies the optional size bounds `node_size` and the connectivity
        constraint `must_be_connected`, emit a new image node whose association is the
        induced subgraph on the intersecting nodes.

    Semantics
        - Input QG state: Reads image-node associations from both input QGs and uses the
          first QG’s preimage_graph to induce intersection subgraphs.
        - Output QG state: Returns a new QuotientGraph (with the first QG’s preimage_graph)
          containing one image node per qualifying intersection.
        - Determinism: Deterministic given inputs and parameters.

    Parameters
        node_size : None | int | tuple[int,int], default None
            If None, no size filter is applied. If int k, treated as (k,k). If a tuple (min,max)
            is provided, require min ≤ |I| ≤ max for the intersection I.
        must_be_connected : bool, default True
            If True, accept only intersections whose induced subgraph forms exactly one
            connected component (empty intersections are rejected).

    Algorithm
        - Optionally normalise node_size via value_to_2tuple when not None.
        - For each subgraph a in first QG and each subgraph b in second QG:
            * Compute I = nodes(a) ∩ nodes(b).
            * Apply size filter (if any).
            * If must_be_connected, require induced subgraph on I to have exactly 1 component.
            * If accepted, create an image node for I.

    Complexity
        - Time: O(k1 · k2 · d) where k1, k2 are counts of subgraphs in the two QGs,
          and d is average subgraph size.
        - Memory: proportional to number and size of emitted intersections.

    Metadata
        - Each output image node stores 'association' and 'meta' with source_function='binary_intersection'.

    Interactions
        - Complements `binary_combination` by intersecting instead of unioning pairs.
        - Often followed by `filter_by_number_of_nodes` to bound sizes explicitly.

    Examples
        # Intersect cycles (from QG1) with neighborhoods (from QG2), require connected intersections
        qg1 = forward_compose(connected_component(), cycle())(Q)
        qg2 = forward_compose(node(), neighborhood(radius=2))(Q)
        out = binary_intersection(qg1, qg2, node_size=None, must_be_connected=True)
    """
    # Normalise size bounds if provided
    if node_size is not None:
        node_size = value_to_2tuple(node_size)

    out_quotient_graph = QuotientGraph(
        graph=first_quotient_graph.preimage_graph,
        label_function=first_quotient_graph.label_function,
        attribute_function=first_quotient_graph.attribute_function,
        edge_function=first_quotient_graph.edge_function,
    )

    subgraphs1 = first_quotient_graph.get_image_nodes_associations()
    subgraphs2 = second_quotient_graph.get_image_nodes_associations()

    # Deduplicate identical intersections across different pairs
    seen = set()  # set[frozenset]

    for sg1 in subgraphs1:
        nodes1 = set(sg1.nodes())
        for sg2 in subgraphs2:
            nodes2 = set(sg2.nodes())
            inter_nodes = nodes1.intersection(nodes2)
            inter_len = len(inter_nodes)

            # Size filter
            size_ok = True
            if node_size is not None:
                size_ok = (min(node_size) <= inter_len <= max(node_size))
            if not size_ok:
                continue

            # Connectivity filter
            if must_be_connected:
                if inter_len == 0:
                    continue
                induced = first_quotient_graph.preimage_graph.subgraph(inter_nodes)
                try:
                    cc_count = len(list(nx.connected_components(induced)))
                except Exception:
                    cc_count = 0
                if cc_count != 1:
                    continue

            key = frozenset(inter_nodes)
            if key in seen:
                continue
            seen.add(key)

            out_quotient_graph.create_image_node_with_subgraph_from_nodes(
                inter_nodes,
                meta=build_meta_from_function_context()
            )

    return out_quotient_graph

#====================================================================================================
# PRE-IMAGE GRAPH OPERATORS
#====================================================================================================
@curry
def unlabel(
    quotient_graph: 'QuotientGraph', 
    label='-'
    ) -> 'QuotientGraph':
    """Replace all node and edge labels in the preimage graph with a constant.
    Summary
        Reset the 'label' attribute of every node and edge in the preimage graph
        to the same constant value.

    Semantics
        - Input QG state: Reads preimage_graph of the given QuotientGraph.
        - Output QG state: Returns a new QuotientGraph with identical structure
          but all labels overwritten.
        - Determinism: Deterministic given `label`.

    Parameters
        label : str | int, default '-'
            The constant value to assign to every node and edge label.

    Algorithm
        - Copy the input QuotientGraph.
        - Apply `nx.set_node_attributes` and `nx.set_edge_attributes` with
          the given constant label.

    Complexity
        - Time: O(|V| + |E|).
        - Memory: O(1) extra beyond the graph copy.

    Interactions
        - Useful to erase semantic bias before structural decomposition.
        - Often paired with `prepend_label` to ensure uniform namespace.

    Examples
        # Strip all labels to a neutral "-"
        qg2 = unlabel(qg1, label='-')

    Domain Analogies
        - Chemistry: treat all atoms and bonds as identical.
        - Social networks: anonymize all node/edge roles.

    Failure Modes
        - If preimage_graph is empty, nothing is modified.
        - Only affects 'label' key; other attributes remain untouched.
    """
    out_quotient_graph = QuotientGraph(quotient_graph=quotient_graph)
    nx.set_node_attributes(out_quotient_graph.preimage_graph, label, 'label')
    nx.set_edge_attributes(out_quotient_graph.preimage_graph, label, 'label')
    return out_quotient_graph

@curry
def prepend_label(
    quotient_graph: 'QuotientGraph', 
    label: Union[str, int] = '-'
) -> 'QuotientGraph':
    """Prepend a string prefix to every node and edge label in the preimage graph.
    Summary
        Modify the 'label' attribute of each node and edge by concatenating
        the provided prefix in front of the existing label value.

    Semantics
        - Input QG state: Reads preimage_graph of the given QuotientGraph.
        - Output QG state: Returns a new QuotientGraph with labels modified by
          prepending the chosen prefix.
        - Determinism: Deterministic given `label`.

    Parameters
        label : str | int, default '-'
            Prefix to prepend. Converted to string if not already.

    Algorithm
        - Copy the input QuotientGraph.
        - For each node and edge:
            * Read existing 'label' (default to empty string).
            * Set new label = f"{prefix}{old_label}".

    Complexity
        - Time: O(|V| + |E|).
        - Memory: O(1) extra beyond the graph copy.

    Interactions
        - Can namespace multiple graphs before composition.
        - Useful to distinguish contributions from different workflows.

    Examples
        # Prefix all labels with "chem_"
        qg2 = prepend_label(qg1, label="chem_")

    Domain Analogies
        - Chemistry: prefix functional groups with context tags.
        - Social networks: add organizational prefixes to role labels.

    Failure Modes
        - If 'label' key is missing, treated as empty string.
        - Repeated application prepends multiple prefixes (idempotence not guaranteed).
    """
    label_str: str = str(label)
    out_quotient_graph = QuotientGraph(quotient_graph=quotient_graph)

    # Prepend the label to each node's existing label
    for node, data in out_quotient_graph.preimage_graph.nodes(data=True):
        current_label = data.get('label', '')
        new_label = f"{label_str}{str(current_label)}"
        data['label'] = new_label

    # Prepend the label to each edge's existing label
    for u, v, data in out_quotient_graph.preimage_graph.edges(data=True):
        current_label = data.get('label', '')
        new_label = f"{label_str}{str(current_label)}"
        data['label'] = new_label
        
    return out_quotient_graph

#====================================================================================================
# SCALAR OPERATORS
#====================================================================================================
@curry
def number_of_image_graph_nodes(
    quotient_graph: 'QuotientGraph',
    param=None
) -> int:
    """Count the number of image nodes.
    Summary
        Return the total number of nodes in the image_graph of the QuotientGraph.

    Parameters
        quotient_graph : QuotientGraph
            The input graph.

    Returns
        int : number of image nodes.
    """
    return quotient_graph.image_graph.number_of_nodes()


def number_of_image_graph_edges(
    quotient_graph: 'QuotientGraph',
    param=None
) -> int:
    """Count the number of image edges.
    Summary
        Return the total number of edges in the image_graph of the QuotientGraph.

    Parameters
        quotient_graph : QuotientGraph
            The input graph.

    Returns
        int : number of image edges.
    """
    return quotient_graph.image_graph.number_of_edges()


def quantile_number_of_subgraph_nodes(
    quotient_graph: 'QuotientGraph',
    q=0.5
) -> int:
    """Quantile of subgraph sizes by nodes.
    Summary
        Compute the q-quantile of the distribution of node counts
        across all image-node subgraphs.

    Parameters
        q : float, default 0.5
            Quantile in [0,1].

    Returns
        float : q-th quantile of subgraph node counts.
    """
    return np.quantile([subgraph.number_of_nodes() for subgraph in quotient_graph.get_image_nodes_associations()], q)


def quantile_number_of_subgraph_edges(
    quotient_graph: 'QuotientGraph',
    q=0.5
) -> int:
    """Quantile of subgraph sizes by edges.
    Summary
        Compute the q-quantile of the distribution of edge counts
        across all image-node subgraphs.

    Parameters
        q : float, default 0.5
            Quantile in [0,1].

    Returns
        float : q-th quantile of subgraph edge counts.
    """
    return np.quantile([subgraph.number_of_edges() for subgraph in quotient_graph.get_image_nodes_associations()], q)


def max_number_of_subgraph_nodes(
    quotient_graph: 'QuotientGraph',
    param=None
) -> int:
    """Maximum subgraph size by nodes.
    Summary
        Return the maximum number of nodes among all image-node subgraphs.

    Returns
        int : maximum node count across subgraphs.
    """
    return max([subgraph.number_of_nodes() for subgraph in quotient_graph.get_image_nodes_associations()])


def min_number_of_subgraph_nodes(
    quotient_graph: 'QuotientGraph',
    param=None
) -> int:
    """Minimum subgraph size by nodes.
    Summary
        Return the minimum number of nodes among all image-node subgraphs.

    Returns
        int : minimum node count across subgraphs.
    """
    return min([subgraph.number_of_nodes() for subgraph in quotient_graph.get_image_nodes_associations()])


def max_number_of_subgraph_edges(
    quotient_graph: 'QuotientGraph',
    param=None
) -> int:
    """Maximum subgraph size by edges.
    Summary
        Return the maximum number of edges among all image-node subgraphs.

    Returns
        int : maximum edge count across subgraphs.
    """
    return max([subgraph.number_of_edges() for subgraph in quotient_graph.get_image_nodes_associations()])


def min_number_of_subgraph_edges(
    quotient_graph: 'QuotientGraph',
    param=None
) -> int:
    """Minimum subgraph size by edges.
    Summary
        Return the minimum number of edges among all image-node subgraphs.

    Returns
        int : minimum edge count across subgraphs.
    """
    return min([subgraph.number_of_edges() for subgraph in quotient_graph.get_image_nodes_associations()])

#====================================================================================================
# XML REGISTRATION
#====================================================================================================
# Explicitly register all QuotientGraph operators with the XML serializer/deserializer.
# This avoids relying on implicit discovery and ensures stable round-trips by name.
try:
    from coco_grape.module.quotientgraph.quotientgraph_xml import register_operator

    # List only operators that operate on and/or return QuotientGraph instances or pipelines.
    # Scalar reducers (e.g., number_of_image_graph_nodes) are intentionally excluded from XML pipelines.
    _QG_OPERATORS = [
        # Higher-order composition
        add,
        compose,
        forward_compose,
        compose_product,

        # Conditionals and loops
        if_then_else,
        if_then_elif_else,
        for_loop,
        while_loop,

        # Unary / decomposition operators
        identity,
        node,
        edge,
        connected_component,
        degree,
        split,
        neighborhood,
        cycle,
        tree,
        path,
        graphlet,
        clique,

        # Unary graph transforms
        complement,
        betweenness_centrality,
        merge,
        combination,
        intersection,
        intersection_edges,

        # Filters
        filter_by_number_of_connected_components,
        filter_by_number_of_nodes,
        filter_by_number_of_edges,
        filter_by_node_label,

        # Binary composition & relabelling
        binary_combination,
        binary_intersection,
        unlabel,
        prepend_label,
    ]

    for _op in _QG_OPERATORS:
        try:
            register_operator(getattr(_op, "__name__", None))(_op)
        except Exception:
            # Best-effort registration; ignore individual failures to avoid import-time crashes.
            pass
    # Backward-compatibility alias: support legacy XML referring to 'pairwise_intersection'
    try:
        register_operator("pairwise_intersection")(intersection)
    except Exception:
        pass
except Exception:
    # If XML module is unavailable, skip registration without failing import.
    pass
