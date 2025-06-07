from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from ._algen_object import _currentGenerationRegistry, _globalRegistry
from ._alpha import _Alpha
from ._alpha_object import _AlphaStatus, make_alpha
from ._constant import _Constant
from ._function import _Function
from ._variable import _Variable

__all__ = ["_Algen"]


class _Algen:
    """
    Represents a algen ( a world for alphas )
    """

    def __init__(
        self,
        global_alpha_registry: pd.DataFrame,
        good_alpha_registry: pd.DataFrame,
        population_size: int,
        generation_count: int,
        function_set: Dict[str, Tuple[int, Callable[..., Any]]],
        n_variable: int,
        variable_names: Optional[List[str]] = None,
        constant_range: Optional[Tuple[float, float]] = (-1.0, 1.0),
        init_depth: Tuple[int, int] = (2, 6),
        init_method: Optional[
            Literal["half_and_half", "full", "grow", "complete"]
        ] = "half_and_half",
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        debug: bool = False,
    ) -> None:
        """
        Initialize the Algen ( A world full of Alpha's and Zyra's )
        """
        self._debug = debug
        if self._debug:
            print("Debug mode is on")
            print("Initializing Algen")

        if not isinstance(global_alpha_registry, pd.DataFrame):
            raise TypeError("Global alpha registry must be a pandas dataframe")
        if not isinstance(good_alpha_registry, pd.DataFrame):
            raise TypeError("Good alpha registry must be a pandas dataframe")
        if not isinstance(population_size, int):
            raise TypeError("Population size must be an integer")
        if not isinstance(generation_count, int):
            raise TypeError("Generations count must be an integer")
        if not isinstance(function_set, dict):
            raise TypeError("Function set must be a dictionary")
        if not all(isinstance(key, str) for key in function_set):
            raise TypeError("Function set keys must be strings")
        if not all(callable(value[1]) for value in function_set.values()):
            raise TypeError(
                "Function set values' second element must be callable")
        if not isinstance(n_variable, int):
            raise TypeError("Number of variables must be an integer")
        if not isinstance(variable_names, list):
            raise TypeError("Variable names must be a list")
        if variable_names is not None:
            if not all(isinstance(value, str) for value in variable_names):
                raise TypeError("Variable names must be strings")
        if constant_range is not None:
            if not isinstance(constant_range, tuple):
                raise TypeError("Constant range must be a tuple")
            if not all(isinstance(value, float) for value in constant_range):
                raise TypeError("Constant range values must be floats")
        if not isinstance(crossover_rate, float):
            raise TypeError("Crossover rate must be a float")
        if not isinstance(mutation_rate, float):
            raise TypeError("Mutation rate must be a float")
        if self._debug:
            print("Checking if the values are valid")

        if population_size < 1:
            raise ValueError("Population size must be greater than 0")
        if generation_count < 1:
            raise ValueError("Generations count must be greater than 0")
        if not function_set:
            raise ValueError("Function set must not be empty")
        if not all(
            isinstance(value, tuple) and len(value) == 2
            for value in function_set.values()
        ):
            raise ValueError(
                "Function set values must be tuples with length 2")
        if not all(
            isinstance(value[0], int) and value[0] > 0
            for value in function_set.values()
        ):
            raise ValueError("Function set arity must be positive integers")
        if n_variable < 1:
            raise ValueError("Number of variables must be greater than 0")
        if variable_names is not None:
            if not variable_names:
                raise ValueError("Variable names must not be empty")
            if len(variable_names) != n_variable:
                raise ValueError(
                    "Variable names must have length equal to number of variables"
                )
        if constant_range is not None:
            if not len(constant_range) == 2:
                raise ValueError("Constant range must have length 2")
            if not constant_range[0] < constant_range[1]:
                raise ValueError("Constant range must be increasing")
            if not all(np.isfinite(value) for value in constant_range):
                raise ValueError("Constant range elements must be finite")
        if (
            not isinstance(init_depth, tuple)
            or len(init_depth) != 2
            or not all(isinstance(value, int) and value > 0 for value in init_depth)
        ):
            raise ValueError(
                "Init depth must be a tuple of two positive integers")
        if init_method is not None:
            if init_method not in ["half_and_half", "full", "grow", "complete"]:
                raise ValueError(
                    "Init method must be one of 'half_and_half', 'full', 'grow', 'complete'"
                )
        if not 0 <= crossover_rate <= 1:
            raise ValueError("Crossover rate must be between 0 and 1")
        if not 0 <= mutation_rate <= 1:
            raise ValueError("Mutation rate must be between 0 and 1")

        if self._debug:
            print("Values are valid")

        self._global_alpha_registry: pd.DataFrame = global_alpha_registry
        self._good_alpha_registry: pd.DataFrame = good_alpha_registry

        self._population_size: int = population_size
        self._generation_count: int = generation_count
        self._crossover_rate: float = crossover_rate
        self._mutation_rate: float = mutation_rate

        if self._debug:
            print(f"Population size: {self._population_size}")
            print(f"Generations count: {self._generation_count}")
            print(f"Crossover rate: {self._crossover_rate}")
            print(f"Mutation rate: {self._mutation_rate}")
            print("Algen initialized")

        # Counter for the alpha's seeded in the current generation
        self._current_generation_alpha_count: int = 0

        self._current_generation_alpha_registry: pd.DataFrame = pd.DataFrame(
            columns=pd.Index(
                [
                    "AGID",
                    "Alpha",
                    "Raw_Fitness_Score",
                    "Fitness_Score",
                    "Penalty_Count",
                    "Penalty_Score",
                    "Crossover_Count",
                    "Mutation_Count",
                    "Mutation_Types",
                    "Zyrex_AGIDs",
                    "Zyra_Count",
                    "Zyra_AGIDs",
                    "Generation_Count",
                    "Status",
                    "Is_Good",
                    "Moved_to_Next_Generation",
                    "Crossovered_in_this_Generation",
                ]
            ),
        )

    def _seed_current_generation_population(
        self,
    ) -> None:
        """
        Seed the current generation population
        """
        if self._debug:
            print("Seeding the current generation population")

        # Initially get the "ALIVE" alpha's from the global alpha registry
        alive_alpha_registry = self._global_alpha_registry[
            self._global_alpha_registry["Status"] == _AlphaStatus.ALIVE
        ]
        total_alive_alpha_count: int = len(alive_alpha_registry)

        if self._debug:
            print(f"Total alive alpha count: {total_alive_alpha_count}")

        if total_alive_alpha_count > self._population_size:
            raise ValueError(
                "Total alive alpha count is greater than the population size"
            )

        if total_alive_alpha_count == self._population_size:
            if self._debug:
                print("Total alive alpha count is equal to the population size")
            return None

        # If the total alive alpha count is less than the population size
        if self._generation_count != 1:
            raise ValueError(
                "Total alive alpha count is less than the population size")

        # If the generation count is 1, then seed the population with random alpha's
        remaining_alpha_count: int = self._population_size - total_alive_alpha_count
        if self._debug:
            print(f"Remaining alpha count: {remaining_alpha_count}")

        self._current_generation_alpha_count += 1
        for alpha_count in range(remaining_alpha_count):
            _alpha_agid = (
                f"{self._generation_count}_{self._current_generation_alpha_count}"
            )
            _alpha_function_set: Dict[str, Tuple[int, Callable[..., Any]]] = {}

            _alpha: _Alpha = make_alpha(
                agid=_alpha_agid,
            )
