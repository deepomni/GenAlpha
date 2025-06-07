import re
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import networkx as nx
import pandas as pd

from ._alpha import _Alpha
from ._constant import _Constant
from ._crossover import _CrossOver
from ._fitness import _Fitness
from ._function import _Function
from ._mutate import _Mutate
from ._variable import _Variable
from .object import make_alpha, make_constant, make_function, make_variable

__all__ = ["convert_string_to_alpha",
           "convert_alpha_to_string", "coupling", "fitness"]


def _parse(
    tokens: List[str],
    functions: Dict[str, Tuple[int, Callable]],
    variable_names: List[str],
    constant_range: Union[Tuple[float, float], None] = (-1.0, 1.0),
    G: nx.DiGraph = nx.DiGraph(),
    node_id: int = 0,
) -> Optional[str]:
    """
    Parse the tokens and build a graph.

    Returns the root node of the subtree.
    """
    if len(tokens) == 0:
        return None

    token: str = tokens.pop(0)

    # If the token is a function
    if token in functions:
        arity: int = functions[token][0]
        function: Callable = functions[token][1]
        function_node: _Function = make_function(
            name=token,
            arity=arity,
            function=function,
        )
        current_node: str = f"{function_node.get_name()}_{node_id}"
        G.add_node(current_node, type="function", data=function_node)
        node_id += 1

        # Parse children of the function
        for _ in range(arity):
            child_node = _parse(
                tokens, functions, variable_names, constant_range, G, node_id
            )
            if child_node is not None:
                G.add_edge(current_node, child_node)

        return current_node

    # If the token is a variable
    elif token in variable_names:
        variable_node: _Variable = make_variable(
            name=token,
            variable_number=variable_names.index(token),
        )
        current_node: str = f"{variable_node.get_name()}_{node_id}"
        G.add_node(
            current_node,
            type="variable",
            data=variable_node,
        )
        return current_node

    # Process constants
    elif re.match(r"-?\d+(\.\d+)?", token):
        if constant_range is None:
            raise ValueError("Constant range is not defined.")
        value: float = float(token)
        if not constant_range[0] <= value <= constant_range[1]:
            raise ValueError(
                f"Constant value {value} is out of range ({constant_range[0]}, {constant_range[1]})."
            )
        constant_node: _Constant = make_constant(value=value)
        current_node: str = f"{constant_node.get_value()}_{node_id}"
        G.add_node(
            current_node,
            type="constant",
            data=constant_node,
        )
        return current_node

    # Process parentheses
    elif token in ("(", ")"):
        return _parse(tokens, functions, variable_names, constant_range, G, node_id)

    raise ValueError(f"Invalid token {token}.")


def convert_string_to_alpha(
    agid: str,
    expression: str,
    functions: Dict[str, Tuple[int, Callable]],
    n_variable: int,
    variable_names: Union[List[str], None] = None,
    constant_range: Union[Tuple[float, float], None] = (-1.0, 1.0),
    init_depth: Tuple[int, int] = (2, 6),
    init_method: Optional[
        Literal["full", "grow", "half_and_half", "complete"]
    ] = "half_and_half",
    node_id: int = 0,
    debug: bool = False,
) -> _Alpha:
    """
    Converts a mathematical string expression into a networkx graph (DiGraph).
    """
    if not isinstance(agid, str):
        raise TypeError("agid must be a string.")

    # Validate input expression
    if expression.count("(") != expression.count(")"):
        raise ValueError("Mismatched parentheses in the expression.")

    tokens: List[str] = re.findall(
        r"[A-Za-z_]\w*|\d+\.\d+|\d+|[-+*/()]", expression)

    if n_variable < 1:
        raise ValueError("Number of variables must be greater than 0.")

    if variable_names is None:
        variable_names = [f"X{i}" for i in range(n_variable)]
    elif len(variable_names) != n_variable:
        raise ValueError(
            "Number of variable names must match the number of variables.")

    if (
        not isinstance(init_depth, tuple)
        or len(init_depth) != 2
        or not all(isinstance(value, int) and value > 0 for value in init_depth)
    ):
        raise ValueError(
            "init_depth must be a tuple of two positive integers.")

    if init_method is not None:
        if init_method not in ("full", "grow", "half_and_half", "complete"):
            raise ValueError(
                "init_method must be one of 'full', 'grow', 'half_and_half', 'complete'."
            )

    G = nx.DiGraph()
    root = _parse(
        tokens=tokens,
        functions=functions,
        variable_names=variable_names,
        constant_range=constant_range,
        G=G,
        node_id=node_id,
    )

    if root is None:
        raise ValueError("Parse function did not construct a valid graph.")

    # Conver the graph to an alpha object
    alpha: _Alpha = make_alpha(
        agid=agid,
        function_set=functions,
        n_variable=n_variable,
        variable_names=variable_names,
        constant_range=constant_range,
        alpha=G,
        debug=debug,
        init_depth=init_depth,
        init_method=init_method,
    )

    # Fix the alpha node ID's
    alpha.fix()

    return alpha


def convert_alpha_to_string(
    alpha: _Alpha,
) -> str:
    """
    Converts an alpha object into a mathematical string expression.
    """
    return alpha.represent_alpha_as_string()


def coupling(
    parentX: _Alpha,
    parentY: _Alpha,
    random_seed: Optional[int] = None,
    debug: bool = False,
) -> _CrossOver:
    """
    Pairing two alpha objects for crossover.
    """
    # Validate the parent and donar
    if not parentX.validate_alpha():
        raise ValueError("Parent X alpha object is not valid.")

    if not parentY.validate_alpha():
        raise ValueError("Parent Y alpha object is not valid.")

    if parentX.get_n_variable() != parentY.get_n_variable():
        raise ValueError(
            "Number of variables in the Parent X and Parent Y must match.")

    if parentX.get_variable_names() != parentY.get_variable_names():
        raise ValueError(
            "Variable names in the Parent X and Parent Y must match.")

    if parentX.get_function_details() != parentY.get_function_details():
        raise ValueError(
            "Function set in the Parent X and Parent Y must match.")

    if parentX.get_constant_range() != parentY.get_constant_range():
        raise ValueError(
            "Constant range in the Parent X and Parent Y must match.")

    # Perform crossover
    crossover: _CrossOver = _CrossOver(
        parentX=parentX,
        parentY=parentY,
        random_seed=random_seed,
        debug=debug,
    )

    return crossover


def fitness(
    alpha: _Alpha,
    data: pd.DataFrame,
) -> _Fitness:
    """
    Calculate the fitness of the alpha.
    """
    if not isinstance(alpha, _Alpha):
        raise ValueError("alpha must be an instance of _Alpha")
    if not alpha.validate_alpha():
        raise ValueError("alpha is not valid")

    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame")
    if data.empty:
        raise ValueError("data is empty")

    return _Fitness(alpha=alpha, data=data)


def evolve(
    alpha: _Alpha,
    current_generation: int = 0,
    max_generation: int = 108,
    debug: bool = False,
) -> _Mutate:
    """
    Evolve the alpha object.
    """
    if not isinstance(alpha, _Alpha):
        raise ValueError("alpha must be an instance of _Alpha")
    if not alpha.validate_alpha():
        raise ValueError("alpha is not valid")

    if not isinstance(current_generation, int):
        raise ValueError("current_generation must be an integer")
    if current_generation < 0:
        raise ValueError(
            "current_generation must be greater than or equal to 0")

    if not isinstance(max_generation, int):
        raise ValueError("max_generation must be an integer")
    if max_generation < 1:
        raise ValueError("max_generation must be greater than 0")

    if current_generation > max_generation:
        raise ValueError("current_generation must be less than max_generation")

    mutate: _Mutate = _Mutate(
        alpha=alpha,
        current_generation=current_generation,
        max_generation=max_generation,
        debug=debug,
    )

    return mutate
