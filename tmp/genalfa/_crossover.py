import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from ._alpha import _Alpha
from ._crossover_object import _ADULT_ALPHA_AGID, _OFFSPRING_ALPHA_AGID
from ._fitness_helper import fitness
from .object import make_alpha

__all__ = ["_CrossOver"]


class _CrossOver:
    """
    Represents a crossover operator.
    """

    def __init__(
        self,
        parentX: _Alpha,
        parentY: _Alpha,
        random_seed: Optional[int] = None,
        debug: bool = False,
    ) -> None:
        """
        Initializes the _CrossOver class
        """
        # Set the debug mode
        self._debug: bool = debug
        if self._debug:
            print("Debug mode is on")
            print("Initializing _CrossOver class")

        # Validating the parent's
        if parentX is None or parentY is None:
            raise ValueError("ParentX and ParentY cannot be None")

        if not isinstance(parentX, _Alpha) or not isinstance(parentY, _Alpha):
            raise ValueError("ParentX and ParentY must be of type _Alpha")
        if not parentX.validate_alpha() or not parentY.validate_alpha():
            raise ValueError(
                "ParentX and ParentY are not valid _Alpha objects")

        if parentX.get_function_details() != parentY.get_function_details():
            raise ValueError(
                "ParentX and ParentY must have the same function details")

        if parentX.get_n_variable() != parentY.get_n_variable():
            raise ValueError(
                "ParentX and ParentY must have the same number of variables"
            )
        if parentX.get_variable_names() != parentY.get_variable_names():
            raise ValueError(
                "ParentX and ParentY must have the same variable names")

        if parentX.get_constant_range() != parentY.get_constant_range():
            raise ValueError(
                "ParentX and ParentY must have the same constant range")

        self._parentX: _Alpha = parentX
        self._parentY: _Alpha = parentY
        self._random_seed: Union[int, None] = random_seed

        self._function_set: Dict[str, Tuple[int, Callable[..., Any]]] = (
            parentX.get_function_set_mapping()
        )
        self._n_variable: int = parentX.get_n_variable()
        self._variable_names: List[str] = parentX.get_variable_names()
        self._constant_range: Union[Tuple[float, float], None] = (
            parentX.get_constant_range()
        )

        # Set random seed
        if self._random_seed is not None:
            random.seed(self._random_seed)
            np.random.seed(self._random_seed)

    def _crossover(self, debugging: Optional[bool] = None):
        """
        Perform crossover operation between parentX and parentY.
        """
        # Set the debugging mode
        debug: bool = self._debug if debugging is None else debugging

        # Get the parentX and parentY graphs
        parentX_graph: nx.DiGraph = self._parentX.get_alpha()
        parentY_graph: nx.DiGraph = self._parentY.get_alpha()

        if debug:
            print("ParentX graph nodes:", parentX_graph.nodes())
            print("ParentY graph nodes:", parentY_graph.nodes())

            print("ParentX graph nodes with data:",
                  parentX_graph.nodes(data=True))
            print("ParentY graph nodes with data:",
                  parentY_graph.nodes(data=True))

        # Copy the parentX and parentY graphs
        if debug:
            print("Copying parentX and parentY graphs")
        childX_graph: nx.DiGraph = self._parentX.reproduce()
        childY_graph: nx.DiGraph = self._parentY.reproduce()

        # Get the subtrees from parentX and parentY
        if debug:
            print("Getting random subtree from parentX and parentY")
        parentX_subtree: nx.DiGraph = self._parentX.get_random_subtree(
            debugging=debug)
        parentY_subtree: nx.DiGraph = self._parentY.get_random_subtree(
            debugging=debug)

        if len(parentX_subtree.nodes) == 0 or len(parentY_subtree.nodes) == 0:
            raise ValueError("Subtree is empty")

        # Get the subtree list of nodes
        parentX_subtree_nodes = list(parentX_subtree.nodes())
        parentY_subtree_nodes = list(parentY_subtree.nodes())

        if debug:
            print("ParentX subtree nodes:", parentX_subtree_nodes)
            print("ParentY subtree nodes:", parentY_subtree_nodes)

        # Get the node attributes of the subtrees from parentX and parentY
        if debug:
            print("Getting node attributes of the subtrees")
        parentX_subtree_nodes_attributes = [
            parentX_graph.nodes[node] for node in parentX_subtree_nodes
        ]
        parentY_subtree_nodes_attributes = [
            parentY_graph.nodes[node] for node in parentY_subtree_nodes
        ]

        if debug:
            print("ParentX subtree nodes attributes:",
                  parentX_subtree_nodes_attributes)
            print("ParentY subtree nodes attributes:",
                  parentY_subtree_nodes_attributes)

        # Get the root nodes of the subtrees
        parentX_subtree_root: int = list(parentX_subtree.nodes())[0]
        parentY_subtree_root: int = list(parentY_subtree.nodes())[0]

        if debug:
            print("ParentX subtree root:", parentX_subtree_root)
            print("ParentY subtree root:", parentY_subtree_root)

        # Get the parents of the subtree roots
        parentX_subtree_root_parent = list(
            parentX_graph.predecessors(parentX_subtree_root)
        )[0]
        parentY_subtree_root_parent = list(
            parentY_graph.predecessors(parentY_subtree_root)
        )[0]

        if debug:
            print("ParentX subtree root parent:", parentX_subtree_root_parent)
            print("ParentY subtree root parent:", parentY_subtree_root_parent)

        # Compute the starting index for new nodes to avoid conflicts
        max_index_X = max(childX_graph.nodes)
        max_index_Y = max(childY_graph.nodes)

        if debug:
            print(f"Max index X: {max_index_X}, Max index Y: {max_index_Y}")

        # Relabel nodes of subtrees to ensure unique IDs
        if debug:
            print("Relabeling nodes of the subtrees")
        parentX_subtree_mapping = {
            node: max_index_Y + i + 1 for i, node in enumerate(parentX_subtree.nodes())
        }
        parentY_subtree_mapping = {
            node: max_index_X + i + 1 for i, node in enumerate(parentY_subtree.nodes())
        }

        if debug:
            print("ParentX subtree mapping:", parentX_subtree_mapping)
            print("ParentY subtree mapping:", parentY_subtree_mapping)

        parentX_subtree = nx.relabel_nodes(
            parentX_subtree, parentX_subtree_mapping)
        parentY_subtree = nx.relabel_nodes(
            parentY_subtree, parentY_subtree_mapping)

        # Detach the subtrees from their parents
        childX_graph.remove_edge(
            parentX_subtree_root_parent, parentX_subtree_root)
        childY_graph.remove_edge(
            parentY_subtree_root_parent, parentY_subtree_root)

        # Add swapped subtrees to the children, including node attributes
        for new_node, attributes in zip(
            parentX_subtree_mapping.values(), parentX_subtree_nodes_attributes
        ):
            childY_graph.add_node(
                new_node,
                type=attributes["type"],
                data=attributes["data"],
            )

        for new_node, attributes in zip(
            parentY_subtree_mapping.values(), parentY_subtree_nodes_attributes
        ):
            childX_graph.add_node(
                new_node,
                type=attributes["type"],
                data=attributes["data"],
            )

        # Add edges from the relabeled subtrees
        childX_graph.add_edges_from(parentY_subtree.edges(data=True))
        childY_graph.add_edges_from(parentX_subtree.edges(data=True))

        # Connect the swapped subtree roots to their new parents
        childX_graph.add_edge(
            parentX_subtree_root_parent, list(
                parentY_subtree_mapping.values())[0]
        )
        childY_graph.add_edge(
            parentY_subtree_root_parent, list(
                parentX_subtree_mapping.values())[0]
        )

        # Remove the old subtrees from the children
        childX_graph.remove_nodes_from(parentX_subtree_nodes)
        childY_graph.remove_nodes_from(parentY_subtree_nodes)

        # Debug information
        if debug:
            print("Subtrees swapped successfully")
            print("ChildX graph nodes:", childX_graph.nodes(data=True))
            print("ChildY graph nodes:", childY_graph.nodes(data=True))

        return childX_graph, childY_graph

    def produce_offspring(
        self,
        debugging: Optional[bool] = None,
    ) -> _Alpha:
        """
        Produce offsprings from the parent.
        """
        debug: bool = self._debug if debugging is None else debugging

        if debug:
            print("Producing offspring from parentX and parentY")

        # Perform crossover operation
        childX_graph, childY_graph = self._crossover()

        if debug:
            print("Crossover operation performed successfully")
            print("Creating offspring from the childX graph")

        # Randomly select the child graph
        child_graph: nx.DiGraph = random.choice([childX_graph, childY_graph])

        if debug:
            print(f"Child graph Selected: {child_graph.nodes(data=True)}")
            print("Creating offspring from the child graph")

        # Create the offspring
        offspring: _Alpha = make_alpha(
            agid=_OFFSPRING_ALPHA_AGID,
            function_set=self._function_set,
            n_variable=self._n_variable,
            variable_names=self._variable_names,
            constant_range=self._constant_range,
            alpha=child_graph,
            debug=debug,
        )

        if debug:
            print("Offspring created successfully")

        if not offspring.validate_alpha():
            raise ValueError("Offspring is not a valid _Alpha object")

        if debug:
            print("Offspring is a valid _Alpha object")

        return offspring

    def produce_twins(
        self,
        debugging: Optional[bool] = None,
    ) -> Tuple[_Alpha, _Alpha]:
        """
        Produce twins from the parent.
        """
        debug: bool = self._debug if debugging is None else debugging

        if debug:
            print("Producing twins from parentX and parentY")

        # Perform crossover operation
        childX_graph, childY_graph = self._crossover()

        if debug:
            print("Crossover operation performed successfully")
            print(
                "Creating offspringX and offspringY from the childX and childY graphs"
            )

        # Create the offspring
        offspringX: _Alpha = make_alpha(
            agid=_OFFSPRING_ALPHA_AGID,
            function_set=self._function_set,
            n_variable=self._n_variable,
            variable_names=self._variable_names,
            constant_range=self._constant_range,
            alpha=childX_graph,
            debug=debug,
        )
        offspringY: _Alpha = make_alpha(
            agid=_OFFSPRING_ALPHA_AGID,
            function_set=self._function_set,
            n_variable=self._n_variable,
            variable_names=self._variable_names,
            constant_range=self._constant_range,
            alpha=childY_graph,
            debug=debug,
        )

        if debug:
            print("OffspringX created successfully")
            print("OffspringY created successfully")

        # Validate the offsprings
        if not offspringX.validate_alpha():
            raise ValueError("OffspringX is not a valid _Alpha object")
        if not offspringY.validate_alpha():
            raise ValueError("OffspringY is not a valid _Alpha object")

        if debug:
            print("OffspringX and OffspringY are valid _Alpha objects")

        return offspringX, offspringY

    def produce_best_offspring(
        self,
        data: pd.DataFrame,
        debugging: Optional[bool] = None,
    ) -> _Alpha:
        """
        Produce the best offspring from the parent.
        """
        debug: bool = self._debug if debugging is None else debugging

        if debug:
            print("Producing best offspring from parentX and parentY")

        offspringX, offspringY = self.produce_twins(debugging=debug)

        if not isinstance(offspringX, _Alpha):
            raise ValueError("OffspringX must be a valid _Alpha object")
        if not isinstance(offspringY, _Alpha):
            raise ValueError("OffspringY must be a valid _Alpha object")

        if debug:
            print("OffspringX and OffspringY are valid _Alpha objects")

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")

        offspringX_fitness = fitness(alpha=offspringX, data=data)
        offspringY_fitness = fitness(alpha=offspringY, data=data)

        if debug:
            print("OffspringX fitness:", offspringX_fitness)
            print("OffspringY fitness:", offspringY_fitness)

        offspringX_score: Dict[str,
                               np.float64] = offspringX_fitness.get_score()
        offspringY_score: Dict[str,
                               np.float64] = offspringY_fitness.get_score()

        if debug:
            print("OffspringX score:", offspringX_score)
            print("OffspringY score:", offspringY_score)

        offspringX_intra_diff_score: np.float64 = offspringX_score["Intra_diff"]
        offspringY_intra_diff_score: np.float64 = offspringY_score["Intra_diff"]

        if debug:
            print("OffspringX intra diff score:", offspringX_intra_diff_score)
            print("OffspringY intra diff score:", offspringY_intra_diff_score)

        return (
            offspringX
            if offspringX_intra_diff_score > offspringY_intra_diff_score
            else offspringY
        )
