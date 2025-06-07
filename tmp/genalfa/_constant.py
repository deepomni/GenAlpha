import numpy as np

__all__ = ["_Constant"]


class _Constant:
    """
    Representation of a constant.
    """

    def __init__(self, value: float) -> None:
        if not np.isfinite(value):
            raise ValueError("value must be a finite number")
        self._value: float = value

    def __str__(self) -> str:
        return f"Constant: {self._value}"

    def __repr__(self) -> str:
        return f"_Constant(value={self._value})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, _Constant):
            return False
        return self._value == other

    def get_data(self) -> str:
        return f"{self._value}"

    def get_value(self) -> float:
        return self._value

    def get_type(self) -> str:
        return "constant"
