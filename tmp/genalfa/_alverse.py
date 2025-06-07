import copy
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from ._algen import _Algen
from ._alpha import _Alpha
from ._alpha_object import _AlphaStatus
from ._constant import _Constant
from ._function import _Function
from ._mutate_object import _MutationType
from ._variable import _Variable

__all__ = ["_Alverse"]


class _Alverse:
    """
    Represents the Alverse.
    """

    def __init__(
        self,
        program_count: int,
        function_set: Dict[str, Tuple[int, Callable[..., Any]]],
        n_variable: int,
        data: pd.DataFrame,
        global_alpha_registry_path: str,
        good_alpha_registry_path: str,
        threshold_for_good_alpha: float,
        generations: int = 11,
        population_size: int = 108,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        penalty_coefficient: float = 0.1,
        tournament_proportion: float = 0.1,
        n_elite_proportion: float = 0.2,
        stopping_fitness: Optional[float] = None,
        variable_names: Optional[List[str]] = None,
        constant_range: Optional[Tuple[float, float]] = (-1.0, 1.0),
        init_depth: Tuple[int, int] = (2, 6),
        init_method: Optional[
            Literal["half_and_half", "full", "grow", "complete"]
        ] = "half_and_half",
        random_seed: Optional[int] = None,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """
        Initializes the Alverse.
        """
        self._debug = debug
        if self._debug:
            print("Debug mode is on")
            print("Initializing Alverse")

        self._verbose = verbose
        if self._verbose:
            print("Verbose mode is activated !!!")

        if not isinstance(program_count, int):
            raise TypeError("program_count must be an integer")
        if program_count < 1:
            raise ValueError("program_count must be greater than 0")

        if not isinstance(function_set, dict):
            raise TypeError("function_set must be a dictionary")
        if not all(
            [
                isinstance(key, str)
                and isinstance(value, tuple)
                and len(value) == 2
                and isinstance(value[0], int)
                and value[0] > 0
                and callable(value[1])
                for key, value in function_set.items()
            ]
        ):
            raise ValueError(
                "function_set must be a dictionary with string keys and tuple values"
            )

        if not isinstance(n_variable, int):
            raise TypeError("n_variable must be an integer")
        if n_variable < 1:
            raise ValueError("n_variable must be greater than 0")

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if not isinstance(global_alpha_registry_path, str):
            raise TypeError("global_alpha_registry_path must be a string")
        if not os.path.exists(global_alpha_registry_path):
            raise FileNotFoundError(
                "global_alpha_registry_path does not exist")

        if not isinstance(good_alpha_registry_path, str):
            raise TypeError("good_alpha_registry_path must be a string")
        if not os.path.exists(good_alpha_registry_path):
            raise FileNotFoundError("good_alpha_registry_path does not exist")

        if not isinstance(threshold_for_good_alpha, float):
            raise TypeError("threshold_for_good_alpha must be a float")
        if threshold_for_good_alpha < 0.0:
            raise ValueError(
                "threshold_for_good_alpha must be greater than or equal to 0.0"
            )

        if not isinstance(generations, int):
            raise TypeError("generations must be an integer")
        if generations < 1:
            raise ValueError("generations must be greater than 0")

        if not isinstance(population_size, int):
            raise TypeError("population_size must be an integer")
        if population_size < 1:
            raise ValueError("population_size must be greater than 0")

        if not isinstance(crossover_rate, float):
            raise TypeError("crossover_rate must be a float")
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be between 0.0 and 1.0")

        if not isinstance(mutation_rate, float):
            raise TypeError("mutation_rate must be a float")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0.0 and 1.0")

        if not isinstance(penalty_coefficient, float):
            raise TypeError("penalty_coefficient must be a float")
        if penalty_coefficient < 0.0:
            raise ValueError(
                "penalty_coefficient must be greater than or equal to 0.0")

        if not isinstance(tournament_proportion, float):
            raise TypeError("tournament_proportion must be a float")
        if not 0.0 <= tournament_proportion <= 1.0:
            raise ValueError(
                "tournament_proportion must be between 0.0 and 1.0")

        if not isinstance(n_elite_proportion, float):
            raise TypeError("n_elite_proportion must be a float")
        if not 0.0 <= n_elite_proportion <= 1.0:
            raise ValueError("n_elite_proportion must be between 0.0 and 1.0")

        if stopping_fitness is not None:
            if not isinstance(stopping_fitness, float):
                raise TypeError("stopping_fitness must be a float")
            if stopping_fitness < 0.0:
                raise ValueError(
                    "stopping_fitness must be greater than or equal to 0.0"
                )

        if variable_names is not None:
            if not isinstance(variable_names, list):
                raise TypeError("variable_names must be a list")
            if not all(
                [isinstance(variable_name, str)
                 for variable_name in variable_names]
            ):
                raise ValueError("variable_names must be a list of strings")
            if len(variable_names) != n_variable:
                raise ValueError(
                    "variable_names must have the same length as n_variable"
                )

        if constant_range is not None:
            if not isinstance(constant_range, tuple):
                raise TypeError("constant_range must be a tuple")
            if not all([isinstance(constant, float) for constant in constant_range]):
                raise ValueError("constant_range must be a tuple of floats")
            if len(constant_range) != 2:
                raise ValueError("constant_range must have two elements")
            if constant_range[0] >= constant_range[1]:
                raise ValueError(
                    "constant_range must have the first element less than the second element"
                )

        if not isinstance(init_depth, tuple):
            raise TypeError("init_depth must be a tuple")
        if not all([isinstance(depth, int) for depth in init_depth]):
            raise ValueError("init_depth must be a tuple of integers")
        if len(init_depth) != 2:
            raise ValueError("init_depth must have two elements")
        if init_depth[0] < 1 or init_depth[1] < 1:
            raise ValueError(
                "init_depth must have both elements greater than 0")
        if init_depth[0] > init_depth[1]:
            raise ValueError(
                "init_depth must have the first element less than the second element"
            )

        if init_method is not None:
            if not isinstance(init_method, str):
                raise TypeError("init_method must be a string")
            if init_method not in ["half_and_half", "full", "grow", "complete"]:
                raise ValueError(
                    "init_method must be one of 'half_and_half', 'full', 'grow', 'complete'"
                )

        if random_seed is not None:
            if not isinstance(random_seed, int):
                raise TypeError("random_seed must be an integer")

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            if self._debug:
                print(f"Random seed: {random_seed}")
                print("Initialized random seed")

        self._program_count: int = program_count
        if self._debug:
            print(f"Program count: {self._program_count}")

        self._function_set: Dict[str,
                                 Tuple[int, Callable[..., Any]]] = function_set
        if self._debug:
            print("Function set:")
            for key, value in self._function_set.items():
                print(f"{key}: {value}")

        self._n_variable: int = n_variable
        if self._debug:
            print(f"Number of variables: {self._n_variable}")

        self._data: pd.DataFrame = data
        if self._debug:
            print(f"Data shape: {self._data.shape}")
            print(f"Data columns: {self._data.columns}")
            print(f"Data head:\n{self._data.head()}")
            print(f"Total number of data points: {len(self._data)}")

        self._global_alpha_registry_path: str = global_alpha_registry_path
        if self._debug:
            print(
                f"Global alpha registry path: {self._global_alpha_registry_path}")

        self._global_alpha_registry: pd.DataFrame = self._load_registry(
            registry_path=self._global_alpha_registry_path
        )
        if self._debug:
            print("Loaded global alpha registry")
            print(
                f"Global alpha registry shape: {self._global_alpha_registry.shape}")
            if not self._global_alpha_registry.empty:
                print(
                    f"Global alpha registry columns: {self._global_alpha_registry.columns}"
                )
                print(
                    f"Global alpha registry head:\n{self._global_alpha_registry.head()}"
                )
            else:
                print("Global alpha registry is empty")

        self._good_alpha_registry_path: str = good_alpha_registry_path
        if self._debug:
            print(
                f"Good alpha registry path: {self._good_alpha_registry_path}")

        self._good_alpha_registry: pd.DataFrame = self._load_registry(
            registry_path=self._good_alpha_registry_path
        )
        if self._debug:
            print("Loaded good alpha registry")
            print(
                f"Good alpha registry shape: {self._good_alpha_registry.shape}")
            if not self._good_alpha_registry.empty:
                print(
                    f"Good alpha registry columns: {self._good_alpha_registry.columns}"
                )
                print(
                    f"Good alpha registry head:\n{self._good_alpha_registry.head()}")
            else:
                print("Good alpha registry is empty")

        self._threshold_for_good_alpha: float = threshold_for_good_alpha
        if self._debug:
            print(
                f"Threshold for good alpha: {self._threshold_for_good_alpha}")

        self._generations: int = generations
        if self._debug:
            print(f"Generations: {self._generations}")

        self._population_size: int = population_size
        if self._debug:
            print(f"Population size: {self._population_size}")

        self._crossover_rate: float = crossover_rate
        if self._debug:
            print(f"Crossover rate: {self._crossover_rate}")

        self._mutation_rate: float = mutation_rate
        if self._debug:
            print(f"Mutation rate: {self._mutation_rate}")

        self._penalty_coefficient: float = penalty_coefficient
        if self._debug:
            print(f"Penalty coefficient: {self._penalty_coefficient}")

        self._tournament_proportion: float = tournament_proportion
        if self._debug:
            print(f"Tournament proportion: {self._tournament_proportion}")

        self._n_elite_proportion: float = n_elite_proportion
        if self._debug:
            print(f"Elite proportion: {self._n_elite_proportion}")

        self._stopping_fitness: Optional[float] = stopping_fitness
        if self._debug:
            print(f"Stopping fitness: {self._stopping_fitness}")

        self._variable_names: Optional[List[str]] = variable_names
        if self._debug:
            print(f"Variable names: {self._variable_names}")

        self._constant_range: Optional[Tuple[float, float]] = constant_range
        if self._debug:
            print(f"Constant range: {self._constant_range}")

        self._init_depth: Tuple[int, int] = init_depth
        if self._debug:
            print(f"Initialization depth: {self._init_depth}")

        self._init_method: Optional[
            Literal["half_and_half", "full", "grow", "complete"]
        ] = init_method
        if self._debug:
            print(f"Initialization method: {self._init_method}")

        # Storing the Training informations
        self._current_generation_count: int = 0
        self._current_generation: Optional[_Algen] = None
        self._maximum_fitness_score: float = -np.Infinity
        self._generation_registry: Dict[int, _Algen] = dict()

    def _load_registry(
        self,
        registry_path: str,
    ) -> pd.DataFrame:
        """
        Loads the registry.
        """
        registry: pd.DataFrame = pd.read_csv(
            filepath_or_buffer=registry_path,
            usecols=list(
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
                ]
            ),
            converters={
                "AGID": lambda x: str(x),
                "Alpha": lambda x: str(x),
                "Raw_Fitness_Score": lambda x: float(x),
                "Fitness_Score": lambda x: float(x),
                "Penalty_Count": lambda x: int(x),
                "Penalty_Score": lambda x: float(x),
                "Crossover_Count": lambda x: int(x),
                "Mutation_Count": lambda x: int(x),
                "Mutation_Types": lambda x: (
                    [
                        _MutationType(mutation_type.strip()[1:-1])
                        for mutation_type in x[1:-1].split(",")
                    ]
                ),
                "Zyrex_AGIDs": lambda x: (
                    [str(zyrex_agid.strip()[1:-1])
                     for zyrex_agid in x[1:-1].split(",")]
                ),
                "Zyra_Count": lambda x: int(x),
                "Zyra_AGIDs": lambda x: (
                    [str(zyra_agid.strip()[1:-1])
                     for zyra_agid in x[1:-1].split(",")]
                ),
                "Generation_Count": lambda x: int(x),
                "Status": lambda x: _AlphaStatus(x),
                "Is_Good": lambda x: bool(x),
            },
            dtype=str,
        )
        registry = registry.set_index(keys="AGID", drop=True)
        if self._debug:
            print("Loaded registry")

        return registry

    def _save_registry(
        self,
        registry: pd.DataFrame,
        registry_path: str,
    ) -> None:
        """
        Saves the registry.
        """
        registry["Mutation_Types"] = registry["Mutation_Types"].apply(
            lambda x: str([mutation_type.value for mutation_type in x])
        )
        registry["Zyrex_AGIDs"] = registry["Zyrex_AGIDs"].apply(
            lambda x: str([zyrex_agid for zyrex_agid in x])
        )
        registry["Zyra_AGIDs"] = registry["Zyra_AGIDs"].apply(
            lambda x: str([zyra_agid for zyra_agid in x])
        )
        registry["Status"] = registry["Status"].apply(lambda x: x.value)
        registry["Is_Good"] = registry["Is_Good"].apply(lambda x: int(x))
        registry.to_csv(
            path_or_buf=registry_path,
            index=True,
            index_label="AGID",
        )
        if self._debug:
            print("Saved registry")

    def seed_generation(
        self,
        debugging: Optional[bool] = None,
    ) -> None:
        """
        Seeds the generation.
        """
        debug: bool = self._debug if debugging is None else debugging
        if debug:
            print("Seeding Initial generation in the Alverse")

        if (self._current_generation_count != 0) and (
            self._current_generation is not None
        ):
            raise ValueError("Generation already seeded")

        self._current_generation = None
        if debug:
            print("Seeded Initial generation in the Alverse")

        self._current_generation_count += 1
        if debug:
            print(
                f"Current generation count: {self._current_generation_count}")

        return None

    def destroy(self) -> None:
        """
        Destroys the Alverse.
        """
        if self._debug or self._verbose:
            print("Destroying Alverse")

        self._save_registry(
            registry=self._global_alpha_registry,
            registry_path=self._global_alpha_registry_path,
        )
        if self._debug:
            print("Saved global alpha registry")

        self._save_registry(
            registry=self._good_alpha_registry,
            registry_path=self._good_alpha_registry_path,
        )
