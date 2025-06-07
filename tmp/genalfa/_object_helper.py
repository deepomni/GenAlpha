from typing import Any, Callable

import numpy as np
from joblib import wrap_non_picklable_objects

from ._constant import _Constant
from ._function import _Function, validate_function_arity
from ._variable import _Variable

__all__ = ["make_function", "make_variable", "make_constant"]


def make_function(
    name: str,
    function: Callable[..., Any],
    arity: int,
    parallelize: bool = False,  # Optional flag for parallel processing
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
    if arity > 0:
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
