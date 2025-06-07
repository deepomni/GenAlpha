__all__ = ["_Variable"]


class _Variable:
    """
    Representation of a variable.
    """

    def __init__(self, name: str, variable_number: int) -> None:
        self._name: str = name
        self._variable_number: int = variable_number

    def __str__(self) -> str:
        return f"Variable: {self._name} (Variable #{self._variable_number})"

    def __repr__(self) -> str:
        return f"_Variable(name={self._name}, variable_number={self._variable_number})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, _Variable):
            return False
        return (
            self._name == other._name
            and self._variable_number == other._variable_number
        )

    def get_data(self) -> str:
        return f"{self._name}"

    def get_name(self) -> str:
        return self._name

    def get_variable_number(self) -> int:
        return self._variable_number

    def get_type(self) -> str:
        return "variable"
