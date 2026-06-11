"""
XML round-trip for QuotientGraph operator pipelines.

Features
- Serialises operator pipelines: `compose`, `forward_compose`, `add`, `compose_product`.
- Handles conditional/iterative operators: `if_then_else`, `if_then_elif_else`, `for_loop`, `while_loop`.
- Works with toolz.curry-annotated operators: captures bound kwargs and callable args.
- Safe param values: only Python literals are embedded directly; callable kwargs are saved as references.
- Pluggable registries to resolve operators and combiners.

Usage
    from coco_grape.module.quotientgraph import operator as qg_ops
    from coco_grape.module.quotientgraph.quotientgraph_xml import register_from_module,
        operator_to_xml_string, operator_from_xml_string

    register_from_module(qg_ops)
    xml = operator_to_xml_string(pipeline)
    pipeline2 = operator_from_xml_string(xml)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional
import ast
import xml.etree.ElementTree as ET

# --------------------------------------------------------------------------------------
# Registries
# --------------------------------------------------------------------------------------

OPERATOR_REGISTRY: Dict[str, Callable[..., Any]] = {}
COMBINER_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_operator(name: Optional[str] = None):
    """Decorator to register an operator constructor by name."""

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        key = name or getattr(func, "__name__", None)
        if not key:
            raise ValueError("Cannot register operator without a name")
        OPERATOR_REGISTRY[key] = func
        return func

    return _decorator


def register_combiner(name: str, func: Callable[..., Any]) -> None:
    """Register a combiner function for compose_product round-trips."""
    if not name:
        raise ValueError("Combiner name must be non-empty")
    COMBINER_REGISTRY[name] = func


def register_from_module(module: Any, names: Optional[Iterable[str]] = None) -> None:
    """Bulk-register callables from a module by attribute name."""
    cand_names = names or [n for n in dir(module) if not n.startswith("_")]
    for n in cand_names:
        obj = getattr(module, n, None)
        if callable(obj):
            OPERATOR_REGISTRY[n] = obj


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _is_curry_obj(obj: Any) -> bool:
    return hasattr(obj, "func") and hasattr(obj, "args") and hasattr(obj, "keywords")


def _op_name(op: Any) -> str:
    """Resolve an operator's name for XML `type` attribute.

    Prefer `__name__` so that `compose_product` round-trips, since some composed
    ops in this codebase set `operator_type = "product"` but `__name__ = "compose_product"`.
    """
    if hasattr(op, "__name__") and isinstance(op.__name__, str) and op.__name__:
        return op.__name__
    if hasattr(op, "operator_type") and isinstance(op.operator_type, str) and op.operator_type:
        # Map legacy/product tag to constructor name for round-trip
        return "compose_product" if op.operator_type == "product" else op.operator_type
    if _is_curry_obj(op) and hasattr(op, "func") and hasattr(op.func, "__name__"):
        return op.func.__name__
    return type(op).__name__


def _op_bound_kwargs(op: Any) -> Dict[str, Any]:
    if hasattr(op, "params") and isinstance(getattr(op, "params"), dict):
        return dict(getattr(op, "params"))
    if _is_curry_obj(op) and hasattr(op, "keywords"):
        return dict(op.keywords or {})
    return {}


def _op_children(op: Any) -> List[Any]:
    for attr in ("children", "chain", "decomposition_functions"):
        if hasattr(op, attr):
            seq = getattr(op, attr)
            return list(seq) if seq else []
    if _is_curry_obj(op) and getattr(op, "args", None):
        return [a for a in op.args if callable(a)]
    return []


def _maybe_combiner_name(op: Any) -> Optional[str]:
    name = getattr(op, "combiner_name", None)
    if isinstance(name, str) and name:
        return name
    comb = getattr(op, "combiner", None)
    if comb is None:
        return None
    for k, v in COMBINER_REGISTRY.items():
        if v is comb:
            return k
    return getattr(comb, "__name__", None)


def _to_attr_value(value: Any) -> str:
    """Serialise a Python literal into an attribute string via repr."""
    return repr(value)


def _from_attr_value(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def _encode_param_value(value: Any) -> str:
    """Encode parameter values.
    - Callables are stored as "ref:<name>" and looked up in the operator registry.
    - Literals use repr so literal_eval can restore them.
    """
    if callable(value):
        return f"ref:{_op_name(value)}"
    return _to_attr_value(value)


def _decode_param_value(value: str) -> Any:
    if isinstance(value, str) and value.startswith("ref:"):
        ref_name = value.split(":", 1)[1]
        if ref_name not in OPERATOR_REGISTRY:
            raise KeyError(f"Unknown referenced callable '{ref_name}'. Register it first.")
        return OPERATOR_REGISTRY[ref_name]
    return _from_attr_value(value)


# --------------------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------------------

def operator_to_xml_element(op: Any) -> ET.Element:
    elem = ET.Element("Operator")
    elem.set("type", _op_name(op))

    # Parameters (kwargs). Encode callables as refs.
    for k, v in sorted(_op_bound_kwargs(op).items()):
        elem.set(k, _encode_param_value(v))

    # combiner for compose_product
    # Prefer legacy attribute when the combiner is a simple named function registered via COMBINER_REGISTRY.
    # Otherwise (e.g., curried/parameterised operators like binary_combination(distance=...)),
    # serialise as a nested <Combiner><Operator .../></Combiner> element for full round-trip of params.
    combiner_obj = getattr(op, "combiner", None)
    if combiner_obj is not None:
        bound_kwargs = _op_bound_kwargs(combiner_obj)
        is_curried = _is_curry_obj(combiner_obj)
        combiner_name = _maybe_combiner_name(op)

        use_legacy_attr = (
            isinstance(combiner_name, str)
            and combiner_name in COMBINER_REGISTRY
            and not bound_kwargs
            and not is_curried
        )

        if use_legacy_attr:
            elem.set("combiner", combiner_name)
        else:
            combiner_elem = ET.SubElement(elem, "Combiner")
            combiner_elem.append(operator_to_xml_element(combiner_obj))

    # Children (composition operands, or positional callables for loops, etc.)
    for child in _op_children(op):
        child_elem = ET.SubElement(elem, "Child")
        child_elem.append(operator_to_xml_element(child))

    return elem


def operator_to_xml_string(op: Any, pretty: bool = True) -> str:
    elem = operator_to_xml_element(op)
    xml = ET.tostring(elem, encoding="unicode")
    if not pretty:
        return xml
    try:
        import xml.dom.minidom as minidom
        return minidom.parseString(xml).toprettyxml(indent="  ")
    except Exception:
        return xml


# --------------------------------------------------------------------------------------
# Deserialisation
# --------------------------------------------------------------------------------------

def _resolve_operator_constructor(name: str) -> Callable[..., Any]:
    if name not in OPERATOR_REGISTRY:
        raise KeyError(f"Unknown operator type '{name}'. Register it first.")
    return OPERATOR_REGISTRY[name]


def _resolve_combiner(name: str) -> Callable[..., Any]:
    if name not in COMBINER_REGISTRY:
        raise KeyError(f"Unknown combiner '{name}'. Register it first.")
    return COMBINER_REGISTRY[name]


def _build_wrapped(
    name: str,
    builder: Callable[..., Any],
    children: List[Any],
    params: Dict[str, Any],
) -> Callable[[Any], Any]:
    """Wrap a constructor into a callable op expecting a QuotientGraph first argument."""

    def _op(quotient_graph):
        return builder(quotient_graph, *children, **params)

    _op.__name__ = name
    _op.operator_type = name  # type: ignore[attr-defined]
    _op.children = list(children)  # type: ignore[attr-defined]
    _op.params = dict(params)  # type: ignore[attr-defined]
    return _op


def operator_from_xml_element(elem: ET.Element) -> Any:
    name = elem.attrib.get("type")
    if not name:
        raise ValueError("Operator element missing 'type' attribute")

    # Parameters
    params: Dict[str, Any] = {}
    for k, v in elem.attrib.items():
        if k in ("type", "combiner"):
            continue
        params[k] = _decode_param_value(v)

    # Children
    child_ops: List[Any] = []
    for child in elem.findall("Child"):
        inner = child.find("Operator")
        if inner is None:
            raise ValueError("Child element missing nested Operator")
        child_ops.append(operator_from_xml_element(inner))

    # Compose variants and add
    if name in ("compose", "forward_compose", "add"):
        builder = _resolve_operator_constructor(name)
        return builder(*child_ops, **params)

    # Product composition with combiner
    if name == "compose_product" or name == "product":  # accept legacy tag
        builder = _resolve_operator_constructor("compose_product")

        # Prefer legacy attribute path when present
        comb_name = elem.attrib.get("combiner")
        comb = None
        if comb_name:
            comb = _resolve_combiner(comb_name)
        else:
            # Look for nested <Combiner><Operator/></Combiner>
            combiner_container = elem.find("Combiner")
            if combiner_container is None:
                raise ValueError("compose_product element missing combiner (no 'combiner' attribute or <Combiner> child)")
            inner = combiner_container.find("Operator")
            if inner is None:
                raise ValueError("<Combiner> element missing nested <Operator>")
            comb = operator_from_xml_element(inner)

        op = builder(comb, *child_ops, **params)
        if comb_name:
            try:
                op.combiner_name = comb_name  # type: ignore[attr-defined]
            except Exception:
                pass
        return op

    # Conditional and loops — return a wrapped callable that receives qg first
    if name in ("if_then_else", "if_then_elif_else", "for_loop", "while_loop"):
        builder = _resolve_operator_constructor(name)
        return _build_wrapped(name, builder, child_ops, params)

    # Leaf/general curried operators
    builder = _resolve_operator_constructor(name)
    try:
        op = builder(**params) if params else builder()
    except TypeError:
        # Fallback if builder expects the qg first (non-curried)
        op = _build_wrapped(name, builder, [], params)
    try:
        op.operator_type = name  # type: ignore[attr-defined]
        if params:
            op.params = dict(params)  # type: ignore[attr-defined]
    except Exception:
        pass
    return op


def operator_from_xml_string(xml: str) -> Any:
    root = ET.fromstring(xml)
    if root.tag != "Operator":
        raise ValueError("Root element must be <Operator>")
    return operator_from_xml_element(root)
