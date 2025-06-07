from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, ValidationError

from ._alpha import _Alpha
from ._alpha_object import _AlphaStatus
from ._constant import _Constant
from ._function import _Function
from ._mutate_object import _MutationType
from ._variable import _Variable

__all__ = ["_globalRegistry", "_currentGenerationRegistry"]


class _globalRegistry(BaseModel):
    """
    Representation of the global alpha registry.
    """

    AGID: str
    Alpha: str
    Raw_Fitness_Score: float
    Fitness_Score: float
    Penalty_Count: int
    Penalty_Score: float
    Crossover_Count: int
    Mutation_Count: int
    Mutation_Types: List[_MutationType]
    Zyrex_AGIDs: List[str]
    Zyra_Count: int
    Zyra_AGIDs: List[str]
    Generation_Count: int
    Status: _AlphaStatus
    Is_Good: bool

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class _currentGenerationRegistry(BaseModel):
    """
    Representation of the current generation registry.
    """

    AGID: str
    Alpha: _Alpha
    Raw_Fitness_Score: float
    Fitness_Score: float
    Penalty_Count: int
    Penalty_Score: float
    Crossover_Count: int
    Mutation_Count: int
    Mutation_Types: List[_MutationType]
    Zyrex_AGIDs: List[str]
    Zyra_Count: int
    Zyra_AGIDs: List[str]
    Generation_Count: int
    Status: _AlphaStatus
    Is_Good: bool
    Moved_to_Next_Generation: bool
    Crossovered_in_this_Generation: bool

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
