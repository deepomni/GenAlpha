from typing import Any, Callable, Dict, List, Literal, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from joblib import wrap_non_picklable_objects

from ._algen import _Algen
from ._alpha import _Alpha
from ._constant import _Constant
from ._function import _Function, validate_function_arity
from ._variable import _Variable

__all__ = [
    "make_algen",
    "make_alpha",
    "make_function",
    "make_variable",
    "make_constant",
]


def make_algen(
    global_alpha_registry: pd.DataFrame,
    good_alpha_registry: pd.DataFrame,
    population_size: int,
    generation_count: int,
    crossover_rate: float,
    mutation_rate: float,
    debug: bool = False,
) -> _Algen:
    """
    Create a new algen instance.

    Args:
        global_alpha_registry (pd.DataFrame): The global alpha registry.
        good_alpha_registry (pd.DataFrame): The good alpha registry.
        population_size (int): The size of the population.
        generation_count (int): The number of generations.
        crossover_rate (float): The crossover rate.
        mutation_rate (float): The mutation rate.

    Raises:
        TypeError: If the global_alpha_registry is not a pandas DataFrame, the good_alpha_registry is not a pandas DataFrame, the population_size is not an integer, the generation_count is not an integer, the crossover_rate is not a float, the mutation_rate is not a float, or the debug is not a boolean.
        ValueError: If the crossover_rate is not between 0 and 1, or the mutation_rate is not between 0 and 1.
    """
    if not isinstance(global_alpha_registry, pd.DataFrame):
        raise TypeError("global_alpha_registry must be a pandas DataFrame")
    if not isinstance(good_alpha_registry, pd.DataFrame):
        raise TypeError("good_alpha_registry must be a pandas DataFrame")
    if not isinstance(population_size, int):
        raise TypeError("population_size must be an integer")
    if not isinstance(generation_count, int):
        raise TypeError("generation_count must be an integer")
    if not isinstance(crossover_rate, float):
        raise TypeError("crossover_rate must be a float")
    if not isinstance(mutation_rate, float):
        raise TypeError("mutation_rate must be a float")
    if not 0 <= crossover_rate <= 1:
        raise ValueError("crossover_rate must be between 0 and 1")
    if not 0 <= mutation_rate <= 1:
        raise ValueError("mutation_rate must be between 0 and 1")
    if not isinstance(debug, bool):
        raise TypeError("debug must be a boolean")

    _global_alpha_registry: pd.DataFrame = global_alpha_registry
    _good_alpha_registry: pd.DataFrame = good_alpha_registry
    _population_size: int = population_size
    _generation_count: int = generation_count
    _crossover_rate: float = crossover_rate
    _mutation_rate: float = mutation_rate
    _debug: bool = debug

    # TODO: Want to update it
    return _Algen(
        global_alpha_registry=_global_alpha_registry,
        good_alpha_registry=_good_alpha_registry,
        population_size=_population_size,
        generation_count=_generation_count,
        crossover_rate=_crossover_rate,
        mutation_rate=_mutation_rate,
        debug=_debug,
    )


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

    Args:
        agid (str): The ID of the alpha.
        function_set (Dict[str, Tuple[int, Callable[..., Any]]]): A dictionary of functions.
        n_variable (int): The number of variables.
        variable_names (Union[List[str], None]): The names of the variables.
        constant_range (Union[Tuple[float, float], None]): The range of constant values.
        init_depth (Tuple[int, int]): The initial depth of the alpha.
        init_method (Union[Literal["half_and_half", "full", "grow", "complete"], None]): The initialization method.
        alpha (Union[nx.DiGraph, None]): The alpha graph.

    Raises:
        TypeError: If the function_set is not a dictionary, the n_variable is not an integer, the variable_names is not a list, the constant_range is not a tuple, the init_depth is not a tuple, the init_method is not a string, or the alpha is not a networkx.DiGraph.
        ValueError: If the function_set is empty, the function_set keys are not strings, the function_set values are not tuples with length 2, the function_set arity is not a positive integer, the variable_names is empty, the variable_names elements are not strings, the variable_names length is not equal to n_variables, the constant_range is not a tuple with length 2, the constant_range elements are not floats, the constant_range elements are not finite, the constant_range is not increasing, the init_depth is not a tuple of two positive integers, the init_method is not one of 'half_and_half', 'full', 'grow', 'complete'.

    Returns:
        _Alpha: An instance of the _Alpha class representing the provided alpha.
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


def make_function(
    name: str,
    function: Callable[..., Any],
    arity: int,
    parallelize: bool = False,  # Optional flag for parallel processing
    debug: bool = False,
) -> _Function:
    """
    Create a new function object.

    Args:
        name (str): The name of the function.
        function (Callable[..., Any]): A callable object representing the function.
        arity (int): The number of arguments the function accepts.
        parallelize (bool): Whether to wrap the function for parallel execution.

    Raises:
        TypeError: If the name is not a string, the function is not callable, or the arity is not an integer.
        ValueError: If the arity is negative or does not match the function's signature.

    Returns:
        _Function: An instance of the _Function class representing the provided function.
    """
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    if not callable(function):
        raise TypeError("function must be callable")
    if not isinstance(arity, int):
        raise TypeError("arity must be an integer")
    if not arity > 0:
        raise ValueError("arity must be positive")

    validate_function_arity(function, arity)

    # TODO: Want to check the edge cases for the function

    if parallelize:
        # TODO: Want to add a method to execute using parallel processing
        function = wrap_non_picklable_objects(function)
        return _Function(name=name, function=function, arity=arity, parallelize=True)

    return _Function(name=name, function=function, arity=arity)


def make_variable(name: str, variable_number: int) -> _Variable:
    """
    Create a new variable instance.

    Args:
        name (str): The name of the variable.
        variable_number (int): The index of the variable associated with this variable.

    Raises:
        TypeError: If `name` is not a string or `variable_number` is not an integer.
        ValueError: If `variable_number` is negative or out of bounds for the dataset.

    Returns:
        _Variable: An instance of the `_Variable` class representing the variable.
    """
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    if not name.strip():
        raise ValueError("name must not be empty or whitespace-only")
    if not isinstance(variable_number, int):
        raise TypeError("variable_number must be an integer")
    if variable_number < 0:
        raise ValueError("variable_number must be positive")

    return _Variable(name=name, variable_number=variable_number)


def make_constant(value: float) -> _Constant:
    """
    Create a new constant.

    Args:
        value (float): The value of the constant.

    Raises:
        TypeError: If value is not an int or float.
        ValueError: If value is not finite.

    Returns:
        _Constant: An instance of the _Constant class.
    """
    if not isinstance(value, (int, float)):
        raise TypeError("value must be an integer or float")
    if not np.isfinite(value):
        raise ValueError("value must be a finite number")

    return _Constant(value=value)
