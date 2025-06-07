import inspect
from typing import Any, Callable

__all__ = ["validate_function_arity", "_Function"]


class _Function:
    """
    Representation of a function.
    """

    def __init__(
        self,
        name: str,
        function: Callable[..., Any],
        arity: int,
        parallelize: bool = False,
    ) -> None:
        self._name: str = name
        self._function: Callable[..., Any] = function
        self._arity: int = arity
        # TODO: Implement parallelization
        self._parallelize: bool = parallelize

    def __str__(self):
        return f"Function: {self._name}\nArity: {self._arity}"

    def __call__(self, *args: Any) -> Any:
        if len(args) != self._arity:
            raise ValueError(
                f"Expected {self._arity} arguments, but got {len(args)}")
        return self._function(*args)

    def is_parallelized(self) -> bool:
        return self._parallelize

    def get_data(self) -> str:
        return f"{self._name} ({self._arity})"

    def get_name(self) -> str:
        return self._name

    def get_function(self) -> Callable[..., Any]:
        return self._function

    def get_arity(self) -> int:
        return self._arity

    def get_type(self) -> str:
        return "function"


def validate_function_arity(function: Callable[..., Any], arity: int) -> None:
    signature = inspect.signature(function)
    if len(signature.parameters) != arity:
        raise ValueError(
            f"Function must accept {arity} arguments, but it accepts {len(signature.parameters)}"
        )
