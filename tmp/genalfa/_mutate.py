import copy
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from ._alpha import _Alpha
from ._constant import _Constant
from ._function import _Function
from ._mutate_object import (
    _RANDAM_GENERATED_ALPHA_AGID,
    _MutationProbability,
    _MutationStage,
    _MutationType,
)
from ._variable import _Variable
from .object import make_alpha

__all__ = ["_Mutate"]


class _Mutate:
    """
    Representation an alpha mutation operator.
    """

    def __init__(
        self,
        current_generation: int,
        max_generation: int,
        alpha: _Alpha,
        debug: bool = False,
    ) -> None:
        """
        Initialize the _Mutate class.
        """
        # Set the debugging mode
        self._debug: bool = debug
        if debug:
            print("Debugging mode is on.")
            print("Initializing the _Mutate class...")

        # Validating and Setting the alpha object
        if not isinstance(alpha, _Alpha):
            raise TypeError(
                "The alpha object must be an instance of the _Alpha class.")
        if not alpha.validate_alpha():
            raise ValueError("The alpha object is invalid.")

        self._alpha: _Alpha = copy.deepcopy(alpha)
        self._mutation_count: int = 0

        if debug:
            print("Initialized the alpha object.")
            print(f"Alpha: {self._alpha.get_all_nodes_list()}")
            print(f"Mutation Count: {self._mutation_count}")

        # Validating and Setting the current generation and maximum number of generations
        if not isinstance(current_generation, int):
            raise TypeError("The current generation must be an integer.")
        if not isinstance(max_generation, int):
            raise TypeError(
                "The maximum number of generations must be an integer.")
        if current_generation < 0:
            raise ValueError(
                "The current generation must be a non-negative integer.")
        if max_generation < 1:
            raise ValueError(
                "The maximum number of generations must be a positive integer."
            )
        if current_generation > max_generation:
            raise ValueError(
                "The current generation cannot exceed the maximum number of generations."
            )

        self._current_generation: int = current_generation
        self._max_generation: int = max_generation

        if debug:
            print(f"Current Generation: {self._current_generation}")
            print(f"Maximum Generations: {self._max_generation}")

        # Initialize the mutation probability, stage, and type classes
        self._mutation_probability: _MutationProbability = _MutationProbability()

        if debug:
            print("Initialized the mutation probability class.")

        # Determine the exploration and exploitation factors based on the current generation
        self._exploration_factor = max(
            0.1, 1 - (self._current_generation / self._max_generation)
        )
        self._exploitation_factor = 1 - self._exploration_factor

        if debug:
            print(f"Exploration Factor: {self._exploration_factor}")
            print(f"Exploitation Factor: {self._exploitation_factor}")

        # Determine the current stage based on the generation
        if self._current_generation <= self._max_generation * 0.2:
            self._mutation_stage: _MutationStage = _MutationStage.EXPLORATION
        elif self._current_generation <= self._max_generation * 0.4:
            self._mutation_stage: _MutationStage = _MutationStage.MID
        elif self._current_generation <= self._max_generation * 0.6:
            self._mutation_stage: _MutationStage = _MutationStage.EXPLOITATION
        elif self._current_generation <= self._max_generation * 0.8:
            self._mutation_stage: _MutationStage = _MutationStage.NEAR_END
        else:
            self._mutation_stage: _MutationStage = _MutationStage.FINAL

        # Setting the mutation type
        self._mutation_type: _MutationType = _MutationType.NONE

        if debug:
            print(f"Mutation Stage: {self._mutation_stage.value}")
            print(f"Mutation Type: {self._mutation_type.value}")

    def _get_random_terminal_node(self) -> Union[_Variable, _Constant]:
        """
        Get a random terminal node from the alpha object.
        """
        if not hasattr(self, "_alpha"):
            raise AttributeError("The alpha object is not available.")
        if not isinstance(self._alpha, _Alpha):
            raise TypeError(
                "The alpha object must be an instance of the _Alpha class.")
        if not self._alpha.validate_alpha():
            raise ValueError("The alpha object is invalid.")

        return self._alpha._get_random_terminal_node()

    def _get_random_subtree(self) -> nx.DiGraph:
        """
        Get a random subtree from the alpha object.
        """
        alpha: _Alpha = make_alpha(
            agid=_RANDAM_GENERATED_ALPHA_AGID,
            function_set=self._alpha.get_function_set_mapping(),
            n_variable=self._alpha.get_n_variable(),
            variable_names=self._alpha.get_variable_names(),
            constant_range=self._alpha.get_constant_range(),
            init_depth=self._alpha.get_initial_depth(),
            init_method=self._alpha.get_initial_method(),
            debug=self._debug,
        )

        return alpha.get_alpha()

    def _get_random_smaller_subtree(self) -> nx.DiGraph:
        """
        Get a random smaller subtree from the alpha object.
        """
        alpha: _Alpha = make_alpha(
            agid=_RANDAM_GENERATED_ALPHA_AGID,
            function_set=self._alpha.get_function_set_mapping(),
            n_variable=self._alpha.get_n_variable(),
            variable_names=self._alpha.get_variable_names(),
            constant_range=self._alpha.get_constant_range(),
            init_depth=(1, 2),
            init_method=self._alpha.get_initial_method(),
            debug=self._debug,
        )

        return alpha.get_alpha()

    def _get_random_alpha(self) -> _Alpha:
        """
        Get a random alpha object.
        """
        alpha: _Alpha = make_alpha(
            agid=_RANDAM_GENERATED_ALPHA_AGID,
            function_set=self._alpha.get_function_set_mapping(),
            n_variable=self._alpha.get_n_variable(),
            variable_names=self._alpha.get_variable_names(),
            constant_range=self._alpha.get_constant_range(),
            init_depth=self._alpha.get_initial_depth(),
            init_method=self._alpha.get_initial_method(),
            debug=self._debug,
        )

        return alpha

    def _get_random_smaller_alpha(self) -> _Alpha:
        """
        Get a random smaller alpha object.
        """
        alpha: _Alpha = make_alpha(
            agid=_RANDAM_GENERATED_ALPHA_AGID,
            function_set=self._alpha.get_function_set_mapping(),
            n_variable=self._alpha.get_n_variable(),
            variable_names=self._alpha.get_variable_names(),
            constant_range=self._alpha.get_constant_range(),
            init_depth=(1, 2),
            init_method=self._alpha.get_initial_method(),
            debug=self._debug,
        )

        return alpha

    def _apply_mutation(
        self,
        mutation_type: _MutationType,
        debugging: Optional[bool] = None,
    ) -> None:
        """
        Apply the mutation based on the selected mutation type.
        """
        debug: bool = self._debug if debugging is None else debugging

        # Set the mutation type
        self._mutation_type = mutation_type
        if debug:
            print(f"Applying Mutation Type: {self._mutation_type.value}")

        # Apply the mutation based on the selected mutation type
        if self._mutation_type == _MutationType.SUBTREE_MUTATION:
            self.subtree_mutation()
            if debug:
                print("Applied Subtree Mutation.")
                print(
                    f"Alpha after Subtree Mutation: {self._alpha.get_all_nodes_list()}"
                )
        elif self._mutation_type == _MutationType.REPLACEMENT_MUTATION:
            self.replacement_mutation()
            if debug:
                print("Applied Replacement Mutation.")
                print(
                    f"Alpha after Replacement Mutation: {self._alpha.get_all_nodes_list()}"
                )
        elif self._mutation_type == _MutationType.INSERTION_MUTATION:
            self.insertion_mutation()
            if debug:
                print("Applied Insertion Mutation.")
                print(
                    f"Alpha after Insertion Mutation: {self._alpha.get_all_nodes_list()}"
                )
        elif self._mutation_type == _MutationType.HOIST_MUTATION:
            self.hoist_mutation()
            if debug:
                print("Applied Hoist Mutation.")
                print(
                    f"Alpha after Hoist Mutation: {self._alpha.get_all_nodes_list()}")
        elif self._mutation_type == _MutationType.SHRINK_MUTATION:
            self.shrink_mutation()
            if debug:
                print("Applied Shrink Mutation.")
                print(
                    f"Alpha after Shrink Mutation: {self._alpha.get_all_nodes_list()}"
                )
        elif self._mutation_type == _MutationType.DELETION_MUTATION:
            self.deletion_mutation()
            if debug:
                print("Applied Deletion Mutation.")
                print(
                    f"Alpha after Deletion Mutation: {self._alpha.get_all_nodes_list()}"
                )
        elif self._mutation_type == _MutationType.DEPTH_LIMITED_MUTATION:
            self.depth_limited_mutation()
            if debug:
                print("Applied Depth-Limited Mutation.")
                print(
                    f"Alpha after Depth-Limited Mutation: {self._alpha.get_all_nodes_list()}"
                )
        elif self._mutation_type == _MutationType.POINT_MUTATION:
            self.point_mutation()
            if debug:
                print("Applied Point Mutation.")
                print(
                    f"Alpha after Point Mutation: {self._alpha.get_all_nodes_list()}")
        elif self._mutation_type == _MutationType.CREEP_MUTATION:
            self.creep_mutation()
            if debug:
                print("Applied Creep Mutation.")
                print(
                    f"Alpha after Creep Mutation: {self._alpha.get_all_nodes_list()}")
        elif self._mutation_type == _MutationType.SWAP_MUTATION:
            self.swap_mutation()
            if debug:
                print("Applied Swap Mutation.")
                print(
                    f"Alpha after Swap Mutation: {self._alpha.get_all_nodes_list()}")
        elif self._mutation_type == _MutationType.PERMUTATION_MUTATION:
            self.permutation_mutation()
            if debug:
                print("Applied Permutation Mutation.")
                print(
                    f"Alpha after Permutation Mutation: {self._alpha.get_all_nodes_list()}"
                )
        elif self._mutation_type == _MutationType.ROOT_MUTATION:
            self.root_mutation()
            if debug:
                print("Applied Root Mutation.")
                print(
                    f"Alpha after Root Mutation: {self._alpha.get_all_nodes_list()}")
        elif self._mutation_type == _MutationType.ROTATION_MUTATION:
            self.rotation_mutation()
            if debug:
                print("Applied Rotation Mutation.")
                print(
                    f"Alpha after Rotation Mutation: {self._alpha.get_all_nodes_list()}"
                )

        if debug:
            print(f"Mutation Type: {self._mutation_type.value} is applied.")

    def get_alpha(self) -> _Alpha:
        """
        Get the alpha object.
        """
        if not isinstance(self._alpha, _Alpha):
            raise TypeError(
                "The alpha object must be an instance of the _Alpha class.")
        if not self._alpha.validate_alpha():
            raise ValueError("The alpha object is invalid.")

        return self._alpha

    def get_mutated_alpha(self) -> _Alpha:
        """
        Get the mutated alpha object.
        """
        return copy.deepcopy(self._alpha)

    def get_current_generation(self) -> int:
        """
        Get the current generation.
        """
        if not isinstance(self._current_generation, int):
            raise TypeError("The current generation must be an integer.")
        if self._current_generation < 0:
            raise ValueError(
                "The current generation must be a non-negative integer.")
        if self._current_generation > self._max_generation:
            raise ValueError(
                "The current generation cannot exceed the maximum number of generations."
            )

        return self._current_generation

    def get_max_generation(self) -> int:
        """
        Get the maximum number of generations.
        """
        if not isinstance(self._max_generation, int):
            raise TypeError(
                "The maximum number of generations must be an integer.")
        if self._max_generation < 1:
            raise ValueError(
                "The maximum number of generations must be a positive integer."
            )
        if self._current_generation > self._max_generation:
            raise ValueError(
                "The current generation cannot exceed the maximum number of generations."
            )

        return self._max_generation

    def get_generations(self) -> Tuple[int, int]:
        """
        Get the current generation and maximum number of generations.
        """
        return self._current_generation, self._max_generation

    def get_exploration_factor(self) -> float:
        """
        Get the exploration factor.
        """
        if not isinstance(self._exploration_factor, float):
            raise TypeError("The exploration factor must be a float.")
        if self._exploration_factor < 0 or self._exploration_factor > 1:
            raise ValueError("The exploration factor must be between 0 and 1.")

        return self._exploration_factor

    def get_exploitation_factor(self) -> float:
        """
        Get the exploitation factor.
        """
        if not isinstance(self._exploitation_factor, float):
            raise TypeError("The exploitation factor must be a float.")
        if self._exploitation_factor < 0 or self._exploitation_factor > 1:
            raise ValueError(
                "The exploitation factor must be between 0 and 1.")

        return self._exploitation_factor

    def get_generation_factors(self) -> Tuple[float, float]:
        """
        Get the exploration and exploitation factors.
        """
        if not isinstance(self._exploration_factor, float):
            raise TypeError("The exploration factor must be a float.")
        if not isinstance(self._exploitation_factor, float):
            raise TypeError("The exploitation factor must be a float")

        if self._exploration_factor < 0 or self._exploration_factor > 1:
            raise ValueError("The exploration factor must be between 0 and 1.")
        if self._exploitation_factor < 0 or self._exploitation_factor > 1:
            raise ValueError(
                "The exploitation factor must be between 0 and 1.")

        if round(self._exploration_factor + self._exploitation_factor, 2) != 1:
            raise ValueError(
                "The sum of the exploration and exploitation factors must be equal to 1."
            )

        return self._exploration_factor, self._exploitation_factor

    def get_mutation_stage(self) -> _MutationStage:
        """
        Get the mutation stage.
        """
        if not isinstance(self._mutation_stage, _MutationStage):
            raise TypeError(
                "The mutation stage must be an instance of the _MutationStage class."
            )

        return self._mutation_stage

    def get_mutation_stage_value(self) -> str:
        """
        Get the mutation stage value.
        """
        if not isinstance(self._mutation_stage, _MutationStage):
            raise TypeError(
                "The mutation stage must be an instance of the _MutationStage class."
            )

        return self._mutation_stage.value

    def get_mutation_type(self) -> _MutationType:
        """
        Get the mutation type.
        """
        if not isinstance(self._mutation_type, _MutationType):
            raise TypeError(
                "The mutation type must be an instance of the _MutationType class."
            )

        return self._mutation_type

    def get_mutation_type_value(self) -> str:
        """
        Get the mutation type value.
        """
        if not isinstance(self._mutation_type, _MutationType):
            raise TypeError(
                "The mutation type must be an instance of the _MutationType class."
            )

        return self._mutation_type.value

    def get_mutation_probability(self) -> float:
        """
        Get the mutation probability for the current stage and type.
        """
        if not hasattr(self, "_mutation_probability"):
            raise AttributeError(
                "The mutation probability object is not available.")
        if not isinstance(self._mutation_probability, _MutationProbability):
            raise TypeError(
                "The mutation probability must be an instance of the _MutationProbability class."
            )

        if not hasattr(self, "_mutation_stage"):
            raise AttributeError("The mutation stage is not available.")
        if not isinstance(self._mutation_stage, _MutationStage):
            raise TypeError(
                "The mutation stage must be an instance of the _MutationStage class."
            )

        if not hasattr(self, "_mutation_type"):
            raise AttributeError("The mutation type is not available.")
        if not isinstance(self._mutation_type, _MutationType):
            raise TypeError(
                "The mutation type must be an instance of the _MutationType class."
            )

        mutation_probability: float = (
            self._mutation_probability.get_mutation_probability(
                stage=self._mutation_stage,
                mutation_type=self._mutation_type,
            )
        )

        return mutation_probability

    def get_mutation_probabilities(self) -> Dict[str, float]:
        """
        Get the mutation probabilities for the current stage.
        """
        if not hasattr(self, "_mutation_probability"):
            raise AttributeError(
                "The mutation probability object is not available.")
        if not isinstance(self._mutation_probability, _MutationProbability):
            raise TypeError(
                "The mutation probability must be an instance of the _MutationProbability class."
            )

        if not hasattr(self, "_mutation_stage"):
            raise AttributeError("The mutation stage is not available.")
        if not isinstance(self._mutation_stage, _MutationStage):
            raise TypeError(
                "The mutation stage must be an instance of the _MutationStage class."
            )

        mutation_probabilities: Dict[_MutationType, float] = (
            self._mutation_probability.get_mutation_probabilities(
                stage=self._mutation_stage
            )
        )

        return {k.value: v for k, v in mutation_probabilities.items()}

    def get_mutation_count(self) -> int:
        """
        Get the mutation count.
        """
        if not isinstance(self._mutation_count, int):
            raise TypeError("The mutation count must be an integer.")
        if self._mutation_count < 0:
            raise ValueError(
                "The mutation count must be a non-negative integer.")

        return self._mutation_count

    def subtree_mutation(
        self,
        debugging: Optional[bool] = None,
    ) -> None:
        """
        Perform a subtree mutation operation.
        """
        # Setting the debug mode
        debug: bool = self._debug if debugging is None else debugging

        # Setting the mutation type
        self._mutation_type = _MutationType.SUBTREE_MUTATION

        # Get the random subtree
        random_subtree: nx.DiGraph = self._get_random_subtree()
        if debug:
            print(f"Random Subtree: {random_subtree.nodes()}")

        random_subtree_nodes: List[int] = list(random_subtree.nodes())
        random_subtree_attributes: List[Dict[str, Any]] = []
        for node in random_subtree_nodes:
            random_subtree_attributes.append(random_subtree.nodes[node])
        if debug:
            print(f"Random Subtree Nodes: {random_subtree_nodes}")
            print(f"Random Subtree Attributes: {random_subtree_attributes}")

        # Get the random node from the alpha object
        random_node: int = random.choice(
            list(self._alpha.get_all_function_nodes_list_except_root())
        )
        if debug:
            print(f"Random Node: {random_node}")

        # Replace the random node with the random subtree
        self._alpha.replace_node_with_subtree(
            node=random_node,
            subtree=random_subtree,
            attributes=random_subtree_attributes,
        )

        if debug:
            print("Replaced the random node with the random subtree.")
            print(
                f"Alpha after Subtree Mutation: {self._alpha.get_all_nodes_list()}")

        # Increment the mutation count
        self._mutation_count += 1

        if debug:
            print(f"Mutation Count: {self._mutation_count}")

        return None

    def replacement_mutation(self) -> None:
        """
        Perform a replacement mutation operation.
        """
        pass

    def insertion_mutation(
        self,
        debugging: Optional[bool] = None,
    ) -> None:
        """
        Perform an insertion mutation operation.
        """
        # Setting the debug mode
        debug: bool = self._debug if debugging is None else debugging

        # Setting the mutation type
        self._mutation_type = _MutationType.INSERTION_MUTATION

        # Get the random subtree
        random_subtree: nx.DiGraph = self._get_random_subtree()
        if debug:
            print(f"Random Subtree: {random_subtree.nodes()}")

        random_subtree_nodes: List[int] = list(random_subtree.nodes())
        random_subtree_attributes: List[Dict[str, Any]] = []
        for node in random_subtree_nodes:
            random_subtree_attributes.append(random_subtree.nodes[node])
        if debug:
            print(f"Random Subtree Nodes: {random_subtree_nodes}")
            print(f"Random Subtree Attributes: {random_subtree_attributes}")

        # Get the random terminal node from the alpha object
        random_node: int = random.choice(
            list(self._alpha.get_all_terminal_nodes_list_except_root())
        )
        if debug:
            print(f"Random Terminal Node: {random_node}")

        # Replace the random terminal node with the random subtree
        self._alpha.replace_node_with_subtree(
            node=random_node,
            subtree=random_subtree,
            attributes=random_subtree_attributes,
        )

        if debug:
            print("Replaced the random terminal node with the random subtree.")
            print(
                f"Alpha after Insertion Mutation: {self._alpha.get_all_nodes_list()}")

        # Increment the mutation count
        self._mutation_count += 1

        if debug:
            print(f"Mutation Count: {self._mutation_count}")

        return None

    def hoist_mutation(
        self,
        debugging: Optional[bool] = None,
    ) -> None:
        """
        Perform a hoist mutation operation.
        """
        # Setting the debug mode
        debug: bool = self._debug if debugging is None else debugging

        # Setting the mutation type
        self._mutation_type = _MutationType.HOIST_MUTATION

        # Get the random subtree
        random_subtree: nx.DiGraph = self._get_random_subtree()
        if debug:
            print(f"Random Subtree: {random_subtree.nodes()}")

        random_subtree_nodes: List[int] = list(random_subtree.nodes())
        random_subtree_attributes: List[Dict[str, Any]] = []
        for node in random_subtree_nodes:
            random_subtree_attributes.append(random_subtree.nodes[node])
        if debug:
            print(f"Random Subtree Nodes: {random_subtree_nodes}")
            print(f"Random Subtree Attributes: {random_subtree_attributes}")

        # Get the random hoist node from the alpha object
        random_node: int = random.choice(
            list(self._alpha.get_hoist_nodes_list()))
        if debug:
            print(f"Random Hoist Node: {random_node}")

        # Replace the random hoist node with the random subtree
        self._alpha.replace_node_with_subtree(
            node=random_node,
            subtree=random_subtree,
            attributes=random_subtree_attributes,
        )

        if debug:
            print("Replaced the random hoist node with the random subtree.")
            print(
                f"Alpha after Hoist Mutation: {self._alpha.get_all_nodes_list()}")

        # Increment the mutation count
        self._mutation_count += 1

        if debug:
            print(f"Mutation Count: {self._mutation_count}")

        return None

    def shrink_mutation(
        self,
        debugging: Optional[bool] = None,
    ) -> None:
        """
        Perform a shrink mutation operation.
        """
        # Setting the debug mode
        debug: bool = self._debug if debugging is None else debugging

        # Setting the mutation type
        self._mutation_type = _MutationType.SHRINK_MUTATION

        # Get the choice for the shrink mutation
        # whether terminal or smaller subtree
        choice: int = random.choice([0, 1])
        if debug:
            print(
                f"Choice: {'Terminal' if choice == 0 else 'Smaller Subtree'}")

        if choice == 0:
            random_terminal_node: Union[_Variable, _Constant] = (
                self._get_random_terminal_node()
            )
            if debug:
                print(
                    f"Random Terminal Node Type: {random_terminal_node.get_type()}")
                print(f"Random Terminal Node: {random_terminal_node}")

            # Get the random node from the alpha object
            random_node: int = random.choice(
                list(self._alpha.get_all_function_nodes_list_except_root())
            )
            if debug:
                print(f"Random Node: {random_node}")

            # Replace the random node with the random terminal node
            self._alpha.replace_node_with_node(
                node=random_node,
                new_node=random_terminal_node,
                debugging=debug,
            )

            if debug:
                print("Replaced the random node with the random terminal node.")

        if choice == 1:
            # Get the random smaller subtree
            random_subtree: nx.DiGraph = self._get_random_smaller_subtree()
            if debug:
                print(f"Random Smaller Subtree: {random_subtree.nodes()}")

            random_subtree_nodes: List[int] = list(random_subtree.nodes())
            random_subtree_attributes: List[Dict[str, Any]] = []
            for node in random_subtree_nodes:
                random_subtree_attributes.append(random_subtree.nodes[node])
            if debug:
                print(f"Random Smaller Subtree Nodes: {random_subtree_nodes}")
                print(
                    f"Random Smaller Subtree Attributes: {random_subtree_attributes}")

            # Get the random node from the alpha object
            random_node: int = random.choice(
                list(self._alpha.get_all_function_nodes_list_except_root())
            )
            if debug:
                print(f"Random Node: {random_node}")

            # Replace the random node with the random subtree
            self._alpha.replace_node_with_subtree(
                node=random_node,
                subtree=random_subtree,
                attributes=random_subtree_attributes,
            )

            if debug:
                print("Replaced the random node with the random smaller subtree.")

        if debug:
            print(
                f"Alpha after Shrink Mutation: {self._alpha.get_all_nodes_list()}")

        # Increment the mutation count
        self._mutation_count += 1

        if debug:
            print(f"Mutation Count: {self._mutation_count}")

        return None

    def deletion_mutation(
        self,
        debugging: Optional[bool] = None,
    ) -> None:
        """
        Perform a deletion mutation operation.
        """
        # Setting the debug mode
        debug: bool = self._debug if debugging is None else debugging

        # Setting the mutation type
        self._mutation_type = _MutationType.DELETION_MUTATION

        # Get the random node object
        random_terminal_node: Union[_Variable, _Constant] = (
            self._get_random_terminal_node()
        )
        if debug:
            print(
                f"Random Terminal Node Type: {random_terminal_node.get_type()}")
            print(f"Random Terminal Node: {random_terminal_node}")

        # Get the random node from the alpha object
        random_node: int = random.choice(
            list(self._alpha.get_all_nodes_list_except_root())
        )
        if debug:
            print(f"Random Node: {random_node}")

        # Replace the random node with the random terminal node
        self._alpha.replace_node_with_node(
            node=random_node,
            new_node=random_terminal_node,
            debugging=debug,
        )

        if debug:
            print("Replaced the random node with the random terminal node.")
            print(
                f"Alpha after Deletion Mutation: {self._alpha.get_all_nodes_list()}")

        # Increment the mutation count
        self._mutation_count += 1

        if debug:
            print(f"Mutation Count: {self._mutation_count}")

        return None

    def depth_limited_mutation(self) -> None:
        """
        Perform a depth-limited mutation operation.
        """
        pass

    def point_mutation(self) -> None:
        """
        Perform a point mutation operation.
        """
        pass

    def creep_mutation(self) -> None:
        """
        Perform a creep mutation operation.
        """
        pass

    def swap_mutation(self) -> None:
        """
        Perform a swap mutation operation.
        """
        pass

    def permutation_mutation(self) -> None:
        """
        Perform a permutation mutation operation.
        """
        pass

    def root_mutation(self) -> None:
        """
        Perform a root mutation operation.
        """
        pass

    def rotation_mutation(self) -> None:
        """
        Perform a rotation mutation operation.
        """
        pass

    def mutate(
        self,
        debugging: Optional[bool] = None,
    ) -> None:
        """
        Select the mutation type based on the current generation and apply the mutation to the alpha.
        """
        debug: bool = self._debug if debugging is None else debugging

        # Get the mutation probabilities for the current stage
        mutation_probabilities: Dict[_MutationType, float] = (
            self._mutation_probability.get_mutation_probabilities(
                stage=self._mutation_stage
            )
        )

        if debug:
            print(f"Mutation Probabilities: {mutation_probabilities}")

        # Randomly select a mutation type based on the probability distribution
        mutation_type: _MutationType = random.choices(
            list(mutation_probabilities.keys()),
            weights=list(mutation_probabilities.values()),
            k=1,
        )[0]

        if debug:
            print(f"Selected Mutation Type: {mutation_type.value}")

        # Apply the mutation based on the selected mutation type
        self._apply_mutation(mutation_type=mutation_type)
