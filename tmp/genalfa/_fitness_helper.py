import pandas as pd

from ._alpha import _Alpha
from ._fitness import _Fitness

__all__ = ["fitness"]


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
