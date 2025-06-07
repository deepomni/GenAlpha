from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Tuple, Union

import networkx as nx
import numpy as np

from ._alpha import _Alpha
from ._function import _Function
from ._object_helper import make_function

__all__ = ["make_alpha", "_AlphaStatus"]


class _AlphaStatus(Enum):
    ALIVE = "ALIVE"
    DEAD = "DEAD"


def make_alpha(
    agid: str,
    function_set: Dict[str, Tuple[int, Callable[..., Any]]],
    n_variable: int,
    variable_names: Union[List[str], None] = None,
    constant_range: Union[Tuple[float, float], None] = (-1.0, 1.0),
    init_depth: Tuple[int, int] = (2, 6),
    init_method: Union[
        Literal["half_and_half", "full", "grow", "complete"], None
    ] = "half_and_half",
    alpha: Union[nx.DiGraph, None] = None,
    debug: bool = False,
) -> _Alpha:
    """
    Create a new alpha instance.
    """

    # Validate agid
    if not isinstance(agid, str):
        raise TypeError("agid must be a string")

    # Validate function_set
    if not isinstance(function_set, dict):
        raise TypeError("function_set must be a dictionary")
    if not function_set:
        raise ValueError("function_set must not be empty")
    if not all(isinstance(key, str) for key in function_set):
        raise TypeError("function_set keys must be strings")
    if not all(
        isinstance(value, tuple) and len(value) == 2 for value in function_set.values()
    ):
        raise ValueError("function_set values must be tuples with length 2")
    if not all(
        isinstance(value[0], int) and value[0] > 0 for value in function_set.values()
    ):
        raise ValueError("function_set arity must be positive integers")
    if not all(callable(value[1]) for value in function_set.values()):
        raise TypeError("function_set values' second element must be callable")

    # Validate n_variable and variable_names
    if not isinstance(n_variable, int):
        raise TypeError("n_variables must be an integer")
    if n_variable < 1:
        raise ValueError("n_variables must be positive")
    if not isinstance(variable_names, list):
        raise TypeError("variable_names must be a list")
    if variable_names is not None:
        if not variable_names:
            raise ValueError("variable_names must not be empty")
        if not all(isinstance(name, str) for name in variable_names):
            raise TypeError("variable_names elements must be strings")
        if len(variable_names) != n_variable:
            raise ValueError(
                "variable_names must have length equal to n_variables")

    # Validate constant_range
    if constant_range is not None:
        if not isinstance(constant_range, tuple):
            raise TypeError("constant_range must be a tuple")
        if not len(constant_range) == 2:
            raise ValueError("constant_range must have length 2")
        if not all(isinstance(value, float) for value in constant_range):
            raise TypeError("constant_range elements must be floats")
        if not all(np.isfinite(value) for value in constant_range):
            raise ValueError("constant_range elements must be finite")
        if not constant_range[0] < constant_range[1]:
            raise ValueError("constant_range must be increasing")

    # Validate init_depth
    if (
        not isinstance(init_depth, tuple)
        or len(init_depth) != 2
        or not all(isinstance(value, int) and value > 0 for value in init_depth)
    ):
        raise ValueError("init_depth must be a tuple of two positive integers")

    # Validate init_method
    if init_method is not None:
        if not isinstance(init_method, str) or init_method not in {
            "full",
            "grow",
            "half_and_half",
            "complete",
        }:
            raise ValueError(
                "init_method must be one of 'full', 'grow', 'half_and_half', 'complete'"
            )

    # Validate alpha (if provided)
    if alpha is not None and not isinstance(alpha, nx.DiGraph):
        raise TypeError("alpha must be a networkx.DiGraph")

    _agid: str = agid
    _function_set: List[_Function] = [
        make_function(name, function, arity)
        for name, (arity, function) in function_set.items()
    ]
    _n_variable: int = n_variable
    _variable_names: Union[List[str], None] = variable_names
    _constant_range: Union[Tuple[float, float], None] = constant_range
    _init_depth: Tuple[int, int] = init_depth
    _init_method: Union[Literal["half_and_half", "full", "grow", "complete"], None] = (
        init_method
    )
    _alpha: Union[nx.DiGraph, None] = alpha

    # Return the alpha instance
    return _Alpha(
        agid=_agid,
        function_set=_function_set,
        n_variable=_n_variable,
        variable_names=_variable_names,
        constant_range=_constant_range,
        init_depth=_init_depth,
        init_method=_init_method,
        alpha=_alpha,
        debug=debug,
    )
