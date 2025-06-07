import copy
import random
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx

from ._constant import _Constant
from ._function import _Function
from ._mutate_object import _MutationType
from ._object_helper import make_constant, make_variable
from ._variable import _Variable

__all__ = ["_Alpha"]


class _Alpha:
    """
    Representation of an alpha.
    """

    def __init__(
        self,
        agid: str,
        function_set: List[_Function],
        n_variable: int,
        variable_names: Union[List[str], None] = None,
        constant_range: Union[Tuple[float, float], None] = (-1.0, 1.0),
        init_depth: Tuple[int, int] = (2, 6),
        init_method: Union[
            Literal["half_and_half", "full", "grow", "complete"], None
        ] = "half_and_half",
        alpha: Union[nx.DiGraph, None] = None,
        debug: bool = False,
    ) -> None:
        """
        Initialize an alpha.
        """
        # Debugging mode
        self._debug: bool = debug

        # AGID
        # AGID Format : f"{generation_number}_{individual_number}"
        self._AGID: str = agid

        # Terminals
        self._n_variable: int = n_variable
        if variable_names is None:
            self._variables: List[_Variable] = [
                make_variable(name=f"X{index}", variable_number=index)
                for index in range(n_variable)
            ]
        else:
            self._variables: List[_Variable] = [
                make_variable(
                    name=name,
                    variable_number=index,
                )
                for index, name in enumerate(variable_names)
            ]
        self._constants: List[_Constant] = []
        self._constant_range: Union[Tuple[float, float], None] = constant_range

        # Functions
        self._functions: List[_Function] = function_set
        self._function_set: Dict[str, int] = {
            function.get_name(): function.get_arity() for function in function_set
        }

        # Initial depth and method
        self._init_depth: Tuple[int, int] = init_depth
        self._init_method: Union[
            Literal["half_and_half", "full", "grow", "complete"], None
        ] = init_method

        # Mutation fators
        self._mutation_count: int = 0
        self._mutation_types: List[_MutationType] = []

        # Penalty factors
        self._penalty_count: int = 0
        self._penalty_score: float = 0.0

        # Alpha description
        self._desc: str = f"""
        Alpha Description:
        ----------------

        AGID: {self._AGID}

        Function Details:
            -> Nuumber of Functions: {len(self._functions)}
            -> Function Set: [ {", ".join([function.get_name() for function in self._functions])} ]

        Variable Details:
            -> Number of Variables: {len(self._variables)}
            -> Variable Names: [ {", ".join([variable.get_name() for variable in self._variables])} ]

        Constant Details:
            -> Number of Constants: {len(self._constants)}
            -> Constant Range: {
            f"( {self._constant_range[0]}, {self._constant_range[1]} )" 
                if self._constant_range is not None else "None"
                }
            -> Constant Values: [ { ", ".join([str(constant.get_value()) for constant in self._constants])} ]

        Generation Details:
            -> Initial Depth: {self._init_depth}
            -> Initial Method: {self._init_method}

        Mutation Details:
            -> Mutation Count: {self._mutation_count}
            -> Mutation Types: [ {
                ", ".join(
                    [mutation_type.value  for mutation_type in self._mutation_types]
                )
            } ]

        Penalty Details:
            -> Penalty Count: {self._penalty_count}
            -> Penalty Score: {self._penalty_score}

        """

        # Alpha
        if alpha is None:
            if not self._init_method:
                self._alpha: nx.DiGraph = self.build_random_alpha()
            else:
                self._alpha: nx.DiGraph = self.build_random_alpha_based_on_init_method()
            if not self.validate_alpha():
                raise ValueError("The Randomly generated alpha is not valid.")
        else:
            # self._alpha: nx.DiGraph = alpha
            if self.validate_alpha(
                alpha=alpha,
            ):
                self._alpha: nx.DiGraph = alpha
            else:
                raise ValueError("Invalid alpha.")

        # Fix the alpha node IDs
        # self.fix()

    def __str__(self) -> str:
        """
        Get the description of the alpha.
        """
        return self._desc

    def __repr__(self) -> str:
        """
        Get the description of the alpha.
        """
        return self._desc

    def _length(self) -> int:
        """
        Get the length of the alpha.
        """
        return len(self._alpha)

    def _root_node(self) -> int:
        """
        Get the root node of the alpha.
        """
        root_nodes: List[Any] = [
            node for node in self._alpha.nodes if self._alpha.in_degree(node) == 0
        ]
        if len(root_nodes) != 1:
            raise ValueError("The alpha must have exactly one root node.")
        return root_nodes[0]

    def _get_root_node(self) -> Union[_Function, _Variable, _Constant]:
        """
        Get the root node of the alpha.
        """
        if self._alpha is None:
            raise ValueError("Alpha is empty.")

        return self._alpha.nodes[self._root_node()]["data"]

    def _get_all_nodes(self) -> List[int]:
        """
        Get all the nodes of the alpha.
        """
        return list(self._alpha.nodes)

    def _get_all_nodes_except_root(self) -> List[int]:
        """
        Get all the nodes of the alpha except the root node.
        """
        root_node: int = self._root_node()
        return [node for node in self._alpha.nodes if node != root_node]

    def _get_all_nodes_along_with_attributes(self) -> Dict[int, Dict[str, Any]]:
        """
        Get all the nodes along with their attributes.
        """
        if not self._alpha.nodes:
            raise ValueError("The graph has no nodes.")
        return {node: self._alpha.nodes[node] for node in self._alpha.nodes}

    def _get_all_function_nodes(self) -> List[int]:
        """
        Get all the function nodes of the alpha.
        """
        return [
            node
            for node in self._alpha.nodes
            if self._alpha.nodes[node]["type"] == "function"
        ]

    def _get_all_function_nodes_except_root(self) -> List[int]:
        """
        Get all the function nodes of the alpha except the root node.
        """
        root_node: int = self._root_node()
        return [
            node
            for node in self._alpha.nodes
            if self._alpha.nodes[node]["type"] == "function" and node != root_node
        ]

    def _get_all_variable_nodes(self) -> List[int]:
        """
        Get all the variable nodes of the alpha.
        """
        return [
            node
            for node in self._alpha.nodes
            if self._alpha.nodes[node]["type"] == "variable"
        ]

    def _get_all_constant_nodes(self) -> List[int]:
        """
        Get all the constant nodes of the alpha.
        """
        return [
            node
            for node in self._alpha.nodes
            if self._alpha.nodes[node]["type"] == "constant"
        ]

    def _get_all_terminal_nodes(self) -> List[int]:
        """
        Get all the terminal nodes of the alpha.
        """
        return [node for node in self._alpha.nodes if self._alpha.out_degree(node) == 0]

    def _get_all_terminal_nodes_except_root(self) -> List[int]:
        """
        Get all the terminal nodes of the alpha except the root node.
        """
        root_node: int = self._root_node()
        return [
            node
            for node in self._alpha.nodes
            if self._alpha.out_degree(node) == 0 and node != root_node
        ]

    def _get_all_hoist_nodes(self) -> List[int]:
        """
        Get all the hoist nodes of the alpha.

        Hoist nodes are the root or near-root nodes of the alpha.
        """
        if not hasattr(self, "_alpha"):
            raise ValueError("Alpha is not initialized.")

        if not self.validate_alpha(
            alpha=self._alpha,
        ):
            raise ValueError("Alpha is invalid.")

        root_node: int = self._root_node()

        # Perform a BFS to identify near-root nodes
        bfs_edges: List[Tuple[int, int]] = list(
            nx.bfs_edges(
                self._alpha,
                source=root_node,
            )
        )
        all_hoist_nodes: List[int] = [root_node]

        # Add children of the root node as potential hoist candidates
        for parent, child in bfs_edges:
            if parent == root_node:  # Direct children of the root node
                all_hoist_nodes.append(child)

        return all_hoist_nodes

    def _maximum_depth(self) -> int:
        """
        Get the maximum depth of the alpha.
        """
        if not nx.is_directed_acyclic_graph(self._alpha):
            raise ValueError(
                "The alpha must be a directed acyclic graph (DAG).")
        return nx.dag_longest_path_length(self._alpha)

    def _get_random_initial_depth(self) -> int:
        """
        Get a random initial depth.
        """
        if self._init_depth[0] > self._init_depth[1]:
            raise ValueError(
                "Invalid depth range: init_depth[0] must be <= init_depth[1]."
            )
        return random.randint(self._init_depth[0], self._init_depth[1])

    def _get_random_root(self) -> Union[_Function, _Variable, _Constant, None]:
        """
        Generate a random root node for building a graph.
        """
        # Randomly decide the type of the root node: function, variable, or constant
        node_type: Literal["function", "variable", "constant"] = random.choice(
            ["function", "variable", "constant"]
            if self._constant_range is not None
            else ["function", "variable"]
        )

        if node_type == "function":
            return self._get_random_function()
        elif node_type == "variable":
            return self._get_random_variable()
        elif node_type == "constant":
            return self._get_random_constant()
        return None

    def _get_random_root_function(self) -> _Function:
        """
        Get a random root function.
        """
        return self._get_random_function()

    def _get_random_function(self) -> _Function:
        """
        Get a random function from the function set.
        """
        if not self._functions:
            raise ValueError("Function set is empty.")
        return random.choice(self._functions)

    def _get_random_child(self) -> Union[_Function, _Variable, _Constant, None]:
        """
        Get a random child.
        """
        node_type: Literal["function", "variable", "constant"] = random.choice(
            ["function", "variable", "constant"]
            if self._constant_range is not None
            else ["function", "variable"]
        )
        if node_type == "function":
            return self._get_random_function()
        elif node_type == "variable":
            return self._get_random_variable()
        elif node_type == "constant":
            return self._get_random_constant()
        return None

    def _get_random_variable(self) -> _Variable:
        """
        Get a random variable from the variable set.
        """
        if not self._variables:
            raise ValueError("Variable set is empty.")
        return random.choice(self._variables)

    def _get_random_constant(self) -> _Constant:
        """
        Get a random constant from the constant set.
        """
        if not self._constant_range:
            raise ValueError("Constant range is not defined.")
        value: float = random.uniform(
            self._constant_range[0], self._constant_range[1])
        return make_constant(value=value)

    def _get_random_terminal(self) -> Union[_Variable, _Constant, None]:
        """
        Get a random terminal.
        """
        if self._constant_range is None:
            terminal_type: Literal["variable", "constant"] = "variable"
        else:
            terminal_type: Literal["variable", "constant"] = random.choice(
                ["variable", "constant"]
            )
        if terminal_type == "variable":
            return self._get_random_variable()
        elif terminal_type == "constant":
            return self._get_random_constant()
        return None

    def _get_random_node(self) -> Union[_Function, _Variable, _Constant]:
        """
        Get a random node.
        """
        if not self._alpha.nodes:
            raise ValueError("The graph has no nodes.")
        return self._alpha.nodes[random.choice(list(self._alpha.nodes))]["data"]

    def _get_random_function_node(self) -> _Function:
        """
        Get a random function node.
        """
        function_nodes: List[Any] = [
            node
            for node in self._alpha.nodes
            if isinstance(self._alpha.nodes[node]["data"], _Function)
            and (
                self._alpha.out_degree(node) > 0
                if isinstance(self._alpha.out_degree(node), int)
                else False
            )
        ]
        if function_nodes:
            return self._alpha.nodes[random.choice(function_nodes)]["data"]
        else:
            raise ValueError("No function nodes available.")

    def _get_random_variable_node(self) -> _Variable:
        """
        Get a random variable node.
        """
        variable_nodes: List[Any] = [
            node
            for node in self._alpha.nodes
            if self._alpha.out_degree(node) == 0
            and isinstance(self._alpha.nodes[node]["data"], _Variable)
        ]
        if variable_nodes:
            return self._alpha.nodes[random.choice(variable_nodes)]["data"]
        else:
            raise ValueError("No variable nodes available.")

    def _get_random_constant_node(self) -> _Constant:
        """
        Get a random constant node.
        """
        if self._constant_range is None:
            raise ValueError("Constant range is not defined.")
        constant_nodes: List[Any] = [
            node
            for node in self._alpha.nodes
            if self._alpha.out_degree(node) == 0
            and isinstance(self._alpha.nodes[node]["data"], _Constant)
        ]
        if constant_nodes:
            return self._alpha.nodes[random.choice(constant_nodes)]["data"]
        else:
            raise ValueError("No constant nodes available.")

    def _get_random_terminal_node(self) -> Union[_Variable, _Constant]:
        """
        Get a random terminal node.
        """
        if self._constant_range is None:
            terminal_nodes: List[Any] = [
                node
                for node in self._alpha.nodes
                if self._alpha.out_degree(node) == 0
                and isinstance(self._alpha.nodes[node]["data"], _Variable)
            ]
        else:
            terminal_nodes: List[Any] = [
                node for node in self._alpha.nodes if self._alpha.out_degree(node) == 0
            ]
        if terminal_nodes:
            return self._alpha.nodes[random.choice(terminal_nodes)]["data"]
        else:
            raise ValueError("No terminal nodes available.")

    def _get_subtree(self, node: int) -> nx.DiGraph:
        """
        Get the subtree rooted at the given node.
        """
        if not self._alpha.nodes:
            raise ValueError("The graph has no nodes.")
        if node not in self._alpha.nodes:
            raise ValueError("The node does not exist in the graph.")

        return nx.dfs_tree(self._alpha, source=node)

    def _get_random_subtree_along_with_attributes(
        self, debugging: Optional[bool] = None
    ) -> Tuple[nx.DiGraph, Dict[str, Any]]:
        """
        Get a random subtree along with its attributes.
        """
        subtree: nx.DiGraph = self._get_random_subtree(debugging=debugging)
        attributes: Dict[str, Any] = {}
        for node in subtree.nodes:
            attributes[node] = self._alpha.nodes[node]
        return subtree, attributes

    def _get_attributes_for_node(self, node: int) -> Dict[str, Any]:
        """
        Get the attributes for a node.
        """
        if not self._alpha.nodes:
            raise ValueError("The graph has no nodes.")
        if node not in self._alpha.nodes:
            raise ValueError("The node does not exist in the graph.")
        return self._alpha.nodes[node]

    def _get_attributes_for_subtree(self, node: int) -> Dict[str, Any]:
        """
        Get the attributes for a subtree.
        """
        subtree: nx.DiGraph = self._get_subtree(node)
        attributes: Dict[str, Any] = {}
        for node in subtree.nodes:
            attributes[str(node)] = self._alpha.nodes[node]
        return attributes

    def _get_random_subtree(self, debugging: Optional[bool] = None) -> nx.DiGraph:
        """
        Get a random subtree from the alpha graph, ensuring it has a minimum depth of 1
        and is not identical to the original graph.
        """
        debug: bool = self._debug
        if debugging is not None:
            debug = debugging

        # Validate the alpha graph
        if not hasattr(self, "_alpha") or not isinstance(self._alpha, nx.DiGraph):
            raise ValueError(
                "The graph (_alpha) is not initialized or is invalid.")
        if not self._alpha.nodes:
            raise ValueError("The graph (_alpha) has no nodes.")

        # Filter nodes to ensure subtree depth >= 1
        eligible_nodes: List[Any] = [
            node for node in self._alpha.nodes if list(self._alpha.successors(node))
        ]
        if not eligible_nodes:
            raise ValueError(
                "No eligible nodes with a minimum depth of 1 exist in the graph."
            )

        # Shuffle the eligible nodes to ensure randomness
        random.shuffle(eligible_nodes)

        # Iterate over eligible nodes to find a suitable subtree
        for root_node in eligible_nodes:
            # Generate the subtree using a depth-first search (DFS)
            subtree: nx.DiGraph = nx.dfs_tree(self._alpha, source=root_node)

            # Ensure the subtree is smaller than the original graph
            if len(subtree.nodes) < len(self._alpha.nodes):
                return subtree

        # Fallback: If no suitable subtree is found
        if debug:
            print("No suitable subtree found.")

        root_node: int = self._root_node()
        if debug:
            print("Root Node: ", root_node)

        # Exclude the root node and pick a random node from the graph
        fallback_eligible_nodes: List[Any] = [
            node for node in self._alpha.nodes if node != root_node
        ]
        if not fallback_eligible_nodes:
            raise ValueError("No eligible nodes exist in the graph.")

        # Shuffle the eligible nodes to ensure randomness
        random.shuffle(fallback_eligible_nodes)

        if debug:
            print("Fallback Eligible Nodes: ", fallback_eligible_nodes)

        # Pick a random node from the eligible nodes
        random_node: int = random.choice(fallback_eligible_nodes)

        if debug:
            print("Random Node: ", random_node)

        # Generate the subtree using a depth-first search (DFS)
        subtree: nx.DiGraph = nx.dfs_tree(self._alpha, source=random_node)

        return subtree

    def _node_to_string(
        self,
        node: Union[_Function, _Variable, _Constant],
        G: nx.DiGraph,
    ) -> str:
        """
        Convert a node to a string.
        """
        node_type: str = G.nodes[node]["type"]
        node_data: Union[_Function, _Variable,
                         _Constant] = G.nodes[node]["data"]

        if node_type == "function" and isinstance(node_data, _Function):
            function_name: str = node_data.get_name()
            return f"""{function_name}({', '.join(
            [self._node_to_string(child, G) for child in list(G.neighbors(node)) ]
            )})"""

        elif node_type == "variable" and isinstance(node_data, _Variable):
            return node_data.get_name()

        elif node_type == "constant" and isinstance(node_data, _Constant):
            return str(node_data.get_value())

        return ""

    def _add_node_to_graph(
        self,
        G: nx.DiGraph,
        node: Union[_Function, _Variable, _Constant, None],
        debugging: Optional[bool] = None,
    ) -> int:
        """
        Add a node to the graph and return a unique node key.
        """
        debug: bool = self._debug
        if debugging is not None:
            debug = debugging

        if node is None:
            raise ValueError("Node is None.")

        node_key: int = len(G.nodes)
        G.add_node(
            node_key,
            type=node.get_type(),
            data=node,
        )

        if debug:
            print("Node Added to Graph")
            print("Node Key: ", node_key)
            print("Node: ", node)

        return node_key

    def _build_random_alpha_recursive(
        self,
        G: nx.DiGraph,
        current_node_key: int,
        current_depth: int,
        max_depth: int,
        debugging: Optional[bool] = None,
    ) -> None:
        """
        Build a random alpha recursively.
        """
        debug: bool = self._debug
        if debugging is not None:
            debug = debugging

        if debug:
            print("Building Random Alpha Recursively")

        current_node: Union[_Function, _Variable, _Constant] = G.nodes[
            current_node_key
        ]["data"]
        if debug:
            print("Current Node Key: ", current_node_key)
            print("Current Node: ", current_node)
            print("Current Depth: ", current_depth)

        if current_depth >= max_depth:
            if debug:
                print("Current Depth >= Max Depth")
                print("Adding Terminal Nodes")

            if isinstance(current_node, _Function):
                if debug:
                    print("Current Node is a Function")

                arity: int = current_node.get_arity()
                if debug:
                    print("Arity: ", arity)

                for _ in range(arity):
                    if debug:
                        print("Adding Terminal Node to Function Node")

                    leaf_node: Union[_Variable, _Constant, None] = (
                        self._get_random_terminal()
                    )
                    leaf_node_key: int = self._add_node_to_graph(
                        G=G,
                        node=leaf_node,
                        debugging=debugging,
                    )
                    G.add_edge(current_node_key, leaf_node_key)

            if debug:
                print("Adding Leaf Node")

            leaf_node: Union[_Variable, _Constant,
                             None] = self._get_random_terminal()
            leaf_node_key: int = self._add_node_to_graph(
                G=G, node=leaf_node, debugging=debugging
            )
            G.add_edge(current_node_key, leaf_node_key)

            return

        if isinstance(current_node, _Function):
            if debug:
                print("Current Node is a Function")
                print("Adding Child Nodes")

            arity: int = current_node.get_arity()
            if debug:
                print("Arity: ", arity)

            for _ in range(arity):
                if debug:
                    print("Adding Child Node to Function Node")

                child_node: Union[_Function, _Variable, _Constant, None] = (
                    self._get_random_child()
                )
                child_node_key: int = self._add_node_to_graph(
                    G=G,
                    node=child_node,
                    debugging=debugging,
                )
                G.add_edge(current_node_key, child_node_key)

                if debug:
                    print("Calling to Build Random Alpha Recursively")
                if isinstance(child_node, _Function):
                    self._build_random_alpha_recursive(
                        G=G,
                        current_node_key=child_node_key,
                        current_depth=current_depth + 1,
                        max_depth=max_depth,
                        debugging=debugging,
                    )

    def _get_node_depth(
        self,
        node_key: int,
        G: Union[nx.DiGraph, None] = None,
    ) -> int:
        """
        Get the depth of a node.
        """
        if G is None:
            G = self._alpha

        depth: int = 0
        while list(G.predecessors(node_key)):
            node_key = list(G.predecessors(node_key))[0]
            depth += 1
        return depth

    def get_random_subtree(
        self,
        debugging: Optional[bool] = None,
    ) -> nx.DiGraph:
        """
        Get a random subtree from the alpha graph.
        """
        debug: bool = self._debug
        if debugging is not None:
            debug = debugging

        return self._get_random_subtree(debugging=debug)

    def get_function_details(self) -> Dict[str, int]:
        """
        Get the function details.
        """
        function_details: Dict[str, int] = {}
        for function in self._functions:
            function_name: str = function.get_name()
            function_arity: int = function.get_arity()
            function_details[function_name] = function_arity
        return function_details

    def get_function_mapping(self) -> Dict[str, Callable]:
        """
        Get the function mapping.
        """
        function_mapping: Dict[str, Callable] = {}
        for node in self._alpha.nodes:
            if self._alpha.nodes[node]["type"] == "function" and isinstance(
                self._alpha.nodes[node]["data"], _Function
            ):
                function_data: _Function = self._alpha.nodes[node]["data"]
                function_name: str = function_data.get_name()
                function_mapping[function_name] = self._alpha.nodes[node]["data"]
        return function_mapping

    def get_function_set_mapping(self) -> Dict[str, Tuple[int, Callable]]:
        """
        Get the function set mapping.
        """
        function_set_mapping: Dict[str, Tuple[int, Callable]] = {}

        if self._functions is None:
            print("Function set is empty.")
            return function_set_mapping

        for function in self._functions:
            function_name: str = function.get_name()
            function_arity: int = function.get_arity()
            function_call: Callable = function.get_function()
            function_set_mapping[function_name] = (
                function_arity, function_call)

        return function_set_mapping

    def get_depth(self) -> int:
        """
        Get the depth of the alpha.
        """
        if self._alpha is None:
            print("Alpha is empty.")
            return -1
        return nx.dag_longest_path_length(self._alpha)

    def get_alpha(self) -> nx.DiGraph:
        """
        Get the alpha.
        """
        if self._alpha is None:
            return nx.DiGraph()

        return copy.deepcopy(self._alpha)

    def get_n_variable(self) -> int:
        """
        Get the n_variable
        """
        if self._n_variable is None:
            print("Number of variables is not defined.")
            return -1
        return self._n_variable

    def get_variable_names(self) -> List[str]:
        """
        Get the variable names.
        """
        if self._variables is None:
            print("Variable set is empty.")
            return []
        return [variable.get_name() for variable in self._variables]

    def get_function_set(self) -> List[_Function]:
        """
        Get the function set.
        """
        if self._functions is None:
            print("Function set is empty.")
            return []
        return self._functions

    def get_variable_set(self) -> List[_Variable]:
        """
        Get the variable set.
        """
        if self._variables is None:
            print("Variable set is empty.")
            return []
        return self._variables

    def get_constant_set(self) -> List[_Constant]:
        """
        Get the constant set.
        """
        if self._constant_range is None or self._constants is None:
            print("Constant set is empty.")
            return []
        return self._constants

    def get_terminal_set(self) -> List[Union[_Variable, _Constant]]:
        """
        Get the terminal set.
        """
        if self._constant_range is None or self._constants is None:
            print("Terminal set is empty.")
            return []
        if self._variables is None:
            print("Terminal set is empty.")
            return []
        return self._variables + self._constants

    def get_constant_range(self) -> Union[Tuple[float, float], None]:
        """
        Get the constant range.
        """
        return self._constant_range

    def get_initial_depth(self) -> Tuple[int, int]:
        """
        Get the initial depth.
        """
        return self._init_depth

    def get_initial_method(
        self,
    ) -> Literal["half_and_half", "full", "grow", "complete"]:
        """
        Get the initial method.
        """
        if self._init_method is None:
            raise ValueError("Initial method is not defined.")
        return self._init_method

    def get_type(self) -> str:
        """
        Get the type of object.
        """
        return "alpha"

    def get_all_nodes_list(self) -> List[int]:
        """
        Get all the nodes of the alpha.
        """
        # Check if the alpha is empty
        if not list(self._alpha.nodes):
            print("Alpha is empty.")
            return []
        return self._get_all_nodes()

    def get_all_nodes_list_and_attributes(
        self,
    ) -> Tuple[List[int], Dict[int, Dict[str, Any]]]:
        """
        Get all the nodes of the alpha along with their attributes.
        """
        # Check if the alpha is empty
        if not list(self._alpha.nodes):
            print("Alpha is empty.")
            return [], {}
        return (
            self.get_all_nodes_list(),
            self._get_all_nodes_along_with_attributes(),
        )

    def get_all_nodes_list_except_root(self) -> List[int]:
        """
        Get all the nodes of the alpha except the root node.
        """
        # Check if the alpha is empty
        if not list(self._alpha.nodes):
            print("Alpha is empty.")
            return []
        nodes_list: List[int] = self._get_all_nodes_except_root()
        if not nodes_list:
            return self.get_all_nodes_list()
        return nodes_list

    def get_all_function_nodes_list_except_root(self) -> List[int]:
        """
        Get all the function nodes of the alpha except the root node.
        """
        # Check if the alpha is empty
        if not list(self._alpha.nodes):
            print("Alpha is empty.")
            return []
        nodes_list: List[int] = self._get_all_function_nodes_except_root()
        if not nodes_list:
            return self.get_all_nodes_list_except_root()
        return nodes_list

    def get_all_terminal_nodes_list_except_root(self) -> List[int]:
        """
        Get all the terminal nodes of the alpha except the root node.
        """
        # Check if the alpha is empty
        if not list(self._alpha.nodes):
            print("Alpha is empty.")
            return []
        nodes_list: List[int] = self._get_all_terminal_nodes_except_root()
        if not nodes_list:
            return self.get_all_nodes_list()
        return nodes_list

    def get_hoist_nodes_list(self) -> List[int]:
        """
        Get the hoist nodes of the alpha.
        """
        # Check if the alpha is empty
        if not list(self._alpha.nodes):
            print("Alpha is empty.")
            return []
        return self._get_all_hoist_nodes()

    def get_all_nodes(self) -> List[Union[_Function, _Variable, _Constant]]:
        """
        Get all the nodes of the alpha.
        """
        if self._alpha is None:
            print("Alpha is empty.")
            return []
        node_list: List[Union[_Function, _Variable, _Constant]] = []
        for node in self._alpha.nodes:
            node_list.append(self._alpha.nodes[node]["data"])
        return node_list

    def get_all_nodes_except_root(self) -> List[Union[_Function, _Variable, _Constant]]:
        """
        Get all the nodes of the alpha except the root node.
        """
        if self._alpha is None:
            print("Alpha is empty.")
            return []
        root_node: int = self._root_node()
        node_list: List[Union[_Function, _Variable, _Constant]] = []
        for node in self._alpha.nodes:
            if node != root_node:
                node_list.append(self._alpha.nodes[node]["data"])
        return node_list

    def get_all_function_nodes(self) -> List[_Function]:
        """
        Get the function set.
        """
        if self._functions is None:
            print("Function set is empty.")
            return []
        functions: List[_Function] = []
        for node in self._alpha.nodes:
            if self._alpha.nodes[node]["type"] == "function":
                functions.append(self._alpha.nodes[node]["data"])
        return functions

    def get_all_variable_nodes(self) -> List[_Variable]:
        """
        Get the variable set.
        """
        if self._variables is None:
            print("Variable set is empty.")
            return []
        variables: List[_Variable] = []
        for node in self._alpha.nodes:
            if self._alpha.nodes[node]["type"] == "variable":
                variables.append(self._alpha.nodes[node]["data"])
        return variables

    def get_all_constant_nodes(self) -> List[_Constant]:
        """
        Get the constant set.
        """
        if self._constant_range is None or self._constants is None:
            print("Constant set is empty.")
            return []
        constants: List[_Constant] = []
        for node in self._alpha.nodes:
            if self._alpha.nodes[node]["type"] == "constant":
                constants.append(self._alpha.nodes[node]["data"])
        return constants

    def get_all_terminal_nodes(self) -> List[Union[_Variable, _Constant]]:
        """
        Get the terminal set.
        """
        if self._constant_range is None or self._constants is None:
            print("Terminal set is empty.")
            return []
        if self._variables is None:
            print("Terminal set is empty.")
            return []
        terminals: List[Union[_Variable, _Constant]] = []
        for node in self._alpha.nodes:
            if self._alpha.nodes[node]["type"] == "variable":
                terminals.append(self._alpha.nodes[node]["data"])
            elif self._alpha.nodes[node]["type"] == "constant":
                terminals.append(self._alpha.nodes[node]["data"])
        return terminals

    def get_expression(self) -> str:
        """
        Get the expression of the alpha.
        """
        root_node: Any = self._root_node()
        return self._node_to_string(root_node, self._alpha)

    def represent_alpha_as_string(self) -> str:
        """
        Get the alpha as a string.
        """
        root_node: Any = self._root_node()
        return self._node_to_string(root_node, self._alpha)

    def represent_alpha_as_tree(
        self,
        current_node: Any = None,
        level: int = 0,
    ) -> str:
        """
        Get the alpha as a tree structure.
        """
        if current_node is None:
            current_node = self._root_node()

        result: str = " " * level * 2 + f"{current_node} -> "

        # Get the children (neighbors) of the current node
        children: List[Any] = list(self._alpha.neighbors(current_node))

        if not children:
            result += "None"
        else:
            # Access the `data` attribute of each child node
            result += ", ".join(
                [self._alpha.nodes[child]["data"].get_data()
                 for child in children]
            )

        result += "\n"

        # Recursively call the children
        for child in children:
            result += self.represent_alpha_as_tree(
                current_node=child,
                level=level + 1,
            )

        return result

    def represent_alpha_as_graph(self) -> None:
        """
        Represent the alpha as a top-to-bottom tree graph.
        """
        if self._alpha is None:
            print("Alpha is empty.")
            return None

        # Prepare a new graph with updated labels for visualization
        tree_graph: nx.DiGraph = nx.DiGraph()

        for node in self._alpha.nodes:
            node_data: Any = self._alpha.nodes[node]["data"]
            tree_graph.add_node(node, label=node_data.get_data())

        for edge in self._alpha.edges:
            tree_graph.add_edge(edge[0], edge[1])

        # Use a hierarchical layout (top-to-bottom)
        pos: Dict[Any, Tuple[float, float]] = nx.nx_agraph.graphviz_layout(
            tree_graph, prog="dot"
        )

        # Draw the graph
        plt.figure(figsize=(12, 8))
        nx.draw(
            tree_graph,
            pos=pos,
            with_labels=True,
            labels=nx.get_node_attributes(tree_graph, "label"),
            node_size=2000,
            node_color="skyblue",
            font_size=11,
            font_color="black",
            edge_color="gray",
        )

        # Show the graph
        plt.title("Alpha Equation Tree")
        plt.axis("off")
        plt.show()

    def represent_alpha_as_graph_with_id(self) -> None:
        """
        Represent the alpha as a top-to-bottom tree graph with node IDs.
        """
        if self._alpha is None:
            print("Alpha is empty.")
            return None

        # Prepare a new graph with updated labels for visualization
        tree_graph: nx.DiGraph = nx.DiGraph()

        for node in self._alpha.nodes:
            node_data: Any = self._alpha.nodes[node]["data"]
            tree_graph.add_node(node, label=f"{node} - {node_data.get_data()}")

        for edge in self._alpha.edges:
            tree_graph.add_edge(edge[0], edge[1])

        # Use a hierarchical layout (top-to-bottom)
        pos: Dict[Any, Tuple[float, float]] = nx.nx_agraph.graphviz_layout(
            tree_graph, prog="dot"
        )

        # Draw the graph
        plt.figure(figsize=(12, 8))
        nx.draw(
            tree_graph,
            pos=pos,
            with_labels=True,
            labels=nx.get_node_attributes(tree_graph, "label"),
            node_size=2000,
            node_color="skyblue",
            font_size=11,
            font_color="black",
            edge_color="gray",
        )

        # Show the graph
        plt.title("Alpha Equation Tree")
        plt.axis("off")
        plt.show()

    def build_random_alpha(
        self,
        debugging: Optional[bool] = None,
    ) -> nx.DiGraph:
        """
        Build a random alpha.
        """
        debug: bool = self._debug
        if debugging is not None:
            debug = debugging

        if debug:
            print("Building Random Alpha")
        # Check if the alpha is already built
        if hasattr(self, "_alpha"):
            raise ValueError("Alpha is already built.")

        # Check if the function set and variable set are empty
        if not self._functions:
            raise ValueError("Function set is empty.")
        if not self._variables:
            raise ValueError("Variable set is empty.")

        # Get the initial depth
        max_depth: int = self._get_random_initial_depth()
        if debug:
            print("Max Depth: ", max_depth)

        # Initialize the alpha
        alpha: nx.DiGraph = nx.DiGraph()
        if debug:
            print("Alpha Initialized.")

        # Randomly decide the root node
        root_node: Union[_Function, _Variable, _Constant, None] = (
            self._get_random_root()
        )
        if debug:
            print("Root Node: ", root_node)

        # Add the root node to the alpha
        root_node_key: int = self._add_node_to_graph(
            G=alpha,
            node=root_node,
            debugging=debugging,
        )
        if debug:
            print("Root Node Key: ", root_node_key)

        if debug:
            print("Building the alpha recursively.")
        # Build the alpha recursively
        self._build_random_alpha_recursive(
            G=alpha,
            current_node_key=root_node_key,
            current_depth=0,
            max_depth=max_depth,
            debugging=debugging,
        )

        if debug:
            print("Alpha Built.")

        # Return the alpha
        return alpha

    def build_random_alpha_based_on_init_method(
        self,
        debugging: Optional[bool] = None,
    ) -> nx.DiGraph:
        """
        Build a random alpha based on the initial method.
        """
        debug: bool = self._debug
        if debugging is not None:
            debug = debugging

        if debug:
            print("Building Random Alpha based on the initial method.")
        # Check if the alpha is already built
        if hasattr(self, "_alpha"):
            raise ValueError("Alpha is already built.")

        # Check if the function set and variable set are empty
        if not self._functions:
            raise ValueError("Function set is empty.")
        if not self._variables:
            raise ValueError("Variable set is empty.")

        # Get the maximum depth
        max_depth: int = self._get_random_initial_depth()
        if debug:
            print("Max Depth: ", max_depth)

        # Initialize the alpha
        alpha: nx.DiGraph = nx.DiGraph()
        if debug:
            print("Alpha Initialized.")

        # Determine the building method
        if self._init_method is None:
            raise ValueError("Initial method is not defined.")
        method: Literal["half_and_half", "full",
                        "grow", "complete"] = self._init_method
        if method == "half_and_half":
            method = random.choice(["full", "grow", "complete"])
        if debug:
            print("Method: ", method)

        # Get a random root function
        root_node: _Function = self._get_random_root_function()
        if debug:
            print("Root Node: ", root_node)

        root_node_key: int = self._add_node_to_graph(
            G=alpha,
            node=root_node,
            debugging=debugging,
        )
        if debug:
            print("Root Node Key: ", root_node_key)

        # Build a tree iteratively
        terminal_stack: List[Tuple[int, int]] = [
            (root_node_key, root_node.get_arity())]
        if debug:
            print("Building the alpha iteratively.")

        while terminal_stack:
            current_node_key, arity = terminal_stack.pop()
            if debug:
                print("Current Node Key: ", current_node_key)
                print("Arity: ", arity)

            for _ in range(arity):
                if debug:
                    print("Current Node Key: ", current_node_key)

                current_depth: int = self._get_node_depth(
                    node_key=current_node_key, G=alpha
                )
                if debug:
                    print("Current Depth: ", current_depth)

                is_terminal: bool = (
                    (method == "full" and current_depth == max_depth)
                    or (method == "grow" and current_depth >= max_depth)
                    or (method == "complete" and current_depth >= max_depth)
                )
                if debug:
                    print("Is Terminal: ", is_terminal)

                if is_terminal:
                    child_node: Any = self._get_random_terminal()
                    if debug:
                        print("Terminal Node: ", child_node)
                else:
                    child_node: Any = (
                        self._get_random_child()
                        if method in ["full", "grow"]
                        else self._get_random_function()
                    )
                    if debug:
                        print("Child Node: ", child_node)

                child_node_key: int = self._add_node_to_graph(
                    G=alpha,
                    node=child_node,
                    debugging=debugging,
                )
                if debug:
                    print("Child Node Key: ", child_node_key)

                alpha.add_edge(current_node_key, child_node_key)
                if debug:
                    print("Edge Added.")

                if isinstance(child_node, _Function):
                    terminal_stack.append(
                        (child_node_key, child_node.get_arity()))
                    if debug:
                        print("Child Node is a Function.")

        if debug:
            print("Alpha Built.")

        return alpha

    def validate_alpha(
        self,
        alpha: Union[nx.DiGraph, None] = None,
        debugging: Optional[bool] = None,
    ) -> bool:
        """
        Validate a given alpha.
        """
        debug: bool = self._debug
        if debugging is not None:
            debug = debugging

        if debug:
            print("Debugging mode is on.")
            print("Validating the alpha.")

        if alpha is None:
            alpha = self._alpha

        # Check if the alpha is a directed graph
        if not isinstance(alpha, nx.DiGraph):
            if debug:
                print("The alpha must be a directed graph.")
            return False

        # Check for the root node
        root_nodes: List[Any] = [
            node for node in alpha.nodes if alpha.in_degree(node) == 0
        ]
        if debug:
            print("Root Nodes: ", root_nodes)

        if len(root_nodes) != 1:
            if self._debug:
                print("The alpha must have exactly one root node.")
            return False

        # TODO: Need to decide must be a function or not
        # Check if the root node is a function
        # root_node = root_nodes[0]
        # if root_node["type"] != "function":
        #     return False
        # if root_node not in self._functions:
        #     return False

        if debug:
            print("Going to check the nodes.")
        # Check for the rest of the nodes
        for node_key in alpha.nodes:
            if debug:
                print("Node: ", node_key)
            node_attributes: Any = alpha.nodes[node_key]

            # Checking the node properties
            node_type: str = node_attributes["type"]
            node_data: Union[_Function, _Variable,
                             _Constant] = node_attributes["data"]
            if debug:
                print("Node Type: ", node_type)

            # Check if the node is a function
            if node_type == "function":
                if debug:
                    print("Function Node")
                # Checking the function instance
                if not isinstance(node_data, _Function):
                    if debug:
                        print("Not a function instance.")
                    return False
                arity: int = node_data.get_arity()
                if debug:
                    print("Arity: ", arity)
                out_degree: Any = alpha.out_degree(node_key)
                if debug:
                    print("Out Degree: ", out_degree)
                neighbors: int = len(list(alpha.successors(node_key)))
                if debug:
                    print("Neighbors: ", neighbors)
                if out_degree != arity:
                    if debug:
                        print("Invalid arity.")
                    return False

            # Check if the node is a variable
            elif node_type == "variable":
                if debug:
                    print("Variable Node")
                # Checking the variable instance
                if not isinstance(node_data, _Variable):
                    if debug:
                        print("Not a variable instance.")
                    return False
                out_degree: Any = alpha.out_degree(node_key)
                if debug:
                    print("Out Degree: ", out_degree)
                if out_degree != 0:
                    if debug:
                        print("Invalid out degree.")
                    return False
                if node_data not in self._variables:
                    if debug:
                        print("Variable not in the variable set.")
                    return False

            # Check if the node is a constant
            elif node_type == "constant" and self._constant_range is not None:
                if debug:
                    print("Constant Node")
                # Checking the constant instance
                if not isinstance(node_data, _Constant):
                    if debug:
                        print("Not a constant instance.")
                    return False
                out_degree: Any = alpha.out_degree(node_key)
                if debug:
                    print("Out Degree: ", out_degree)
                if out_degree != 0:
                    if debug:
                        print("Invalid out degree.")
                    return False
                values: float = node_data.get_value()
                if debug:
                    print("Constant Value: ", values)
                if values < self._constant_range[0] or values > self._constant_range[1]:
                    if debug:
                        print("Invalid constant value.")
                    return False

                # Add the constant to the list
                self._constants.append(node_data)
                if debug:
                    print("Added to the constant list.")

            # Invalid node type
            else:
                if debug:
                    print("Invalid node type.")
                return False

        # Checking the cycles
        if not nx.is_directed_acyclic_graph(alpha):
            if debug:
                print("The alpha must be a directed acyclic graph (DAG).")
            return False

        # Ensure all the nodes are reachable from the root node
        for node in alpha.nodes:
            if not nx.has_path(alpha, root_nodes[0], node):
                if debug:
                    print("Not all nodes are reachable from the root node.")
                return False

        if debug:
            print("Alpha is valid.")
        return True

    def reproduce(
        self,
        debugging: Optional[bool] = None,
    ) -> nx.DiGraph:
        """
        Reproduce the alpha.
        """
        debug: bool = self._debug
        if debugging is not None:
            debug = debugging

        if debug:
            print("Reproducing the alpha.")

        return copy.deepcopy(self.get_alpha())

    def is_valid(
        self,
        debugging: Optional[bool] = None,
    ) -> bool:
        """
        Check if the alpha is valid.
        """
        debug: bool = self._debug
        if debugging is not None:
            debug = debugging

        if debug:
            print("Checking if the alpha is valid.")

        return self.validate_alpha(debugging=debug)

    def fix(
        self,
        debugging: Optional[bool] = None,
    ) -> None:
        """
        Fix the alpha nodes ID's.
        """
        debug: bool = self._debug
        if debugging is not None:
            debug = debugging

        if debug:
            print("Debugging mode is on.")
            print("Fixing ID's of the alpha nodes.")

        if not hasattr(self, "_alpha"):
            raise ValueError("Alpha is not initialized.")

        if not self.validate_alpha(alpha=self._alpha, debugging=debug):
            raise ValueError("Alpha is invalid.")

        # Fix the node ID's
        node_id_mapping: Dict[Any, int] = {
            node: index for index, node in enumerate(self._alpha.nodes())
        }

        if debug:
            print("Node ID Mapping: ", node_id_mapping)

        # Update the node ID's
        self._alpha = nx.relabel_nodes(self._alpha, node_id_mapping, copy=True)

        if debug:
            print("Alpha nodes fixed.")

        return None

    def replace_node_with_subtree(
        self,
        node: int,
        subtree: nx.DiGraph,
        attributes: List[Dict[str, Any]],
        debugging: Optional[bool] = None,
    ) -> None:
        """
        Replace a subtree in the alpha.
        """
        debug: bool = self._debug
        if debugging is not None:
            debug = debugging

        if debug:
            print("Debugging mode is on.")
            print("Replacing the subtree.")
            print("Node: ", node)
            print("Subtree: ", subtree)
            print("Attributes: ", attributes)

        if not hasattr(self, "_alpha"):
            raise ValueError("Alpha is not initialized.")
        if not self.validate_alpha(alpha=self._alpha, debugging=debug):
            raise ValueError("Alpha is invalid.")
        if node not in self._alpha.nodes:
            raise ValueError("Root node is not in the alpha.")
        if not subtree:
            raise ValueError("Subtree is empty.")
        if not nx.is_directed_acyclic_graph(subtree):
            raise ValueError("Subtree must be a directed acyclic graph (DAG).")
        if not nx.has_path(subtree, list(subtree.nodes)[0], list(subtree.nodes)[-1]):
            raise ValueError("Subtree must be connected.")
        if debug:
            print("Subtree validated.")

        # Compute the starting index for the new nodes
        starting_index: int = len(self._alpha.nodes)

        # Get the subtree of the node in the alpha
        current_subtree: nx.DiGraph = self._get_subtree(node)
        current_subtree_nodes: List[int] = list(current_subtree.nodes())
        current_subtree_attributes: List[Any] = [
            self._alpha.nodes[node] for node in current_subtree_nodes
        ]
        if debug:
            print("Current Subtree: ", current_subtree)
            print("Current Subtree Nodes: ", current_subtree_nodes)
            print("Current Subtree Attributes: ", current_subtree_attributes)

        # Get the parent node of the node in the alpha
        parent_node: Any = list(self._alpha.predecessors(node))
        if debug:
            print("Parent Node: ", parent_node)

        if len(parent_node) >= 2:
            raise ValueError("The node has multiple parents.")

        # Remove the edges between the parent node and the node
        self._alpha.remove_edge(parent_node[0], node)
        if debug:
            print("Edge Removed.")

        # Remove the subtree of the node in the alpha
        self._alpha.remove_nodes_from(current_subtree.nodes)

        # Get the nodes and attributes of the subtree
        subtree_nodes: List[int] = list(subtree.nodes())
        subtree_attributes: List[Dict[str, Any]] = [
            attributes[node] for node in subtree.nodes
        ]

        if debug:
            print("Subtree Nodes: ", subtree_nodes)
            print("Subtree Attributes: ", subtree_attributes)

        # Update the node ID's of the subtree
        node_id_mapping: Dict[Any, int] = {
            node: starting_index + index + 1
            for index, node in enumerate(subtree.nodes())
        }

        if debug:
            print("Node ID Mapping: ", node_id_mapping)

        # Update the node ID's
        subtree = nx.relabel_nodes(subtree, node_id_mapping)

        if debug:
            print(f"Subtree Nodes Updated: {subtree.nodes()}")

        # Update the parent node of the subtree
        for new_node, attribute in zip(
            node_id_mapping.values(),
            subtree_attributes,
        ):
            self._alpha.add_node(
                new_node,
                type=attribute["type"],
                data=attribute["data"],
            )

        # Add the edges from the subtree
        self._alpha.add_edges_from(subtree.edges(data=True))

        if debug:
            print("Subtree Added.")

        # Add the edges between the parent node and the subtree
        self._alpha.add_edge(parent_node[0], list(subtree.nodes)[0])

        # Fix the node ID's
        self.fix(debugging=debug)

        if debug:
            print("Subtree Replaced.")

        return None

    def replace_node_with_node(
        self,
        node: int,
        new_node: Union[_Variable, _Constant],
        debugging: Optional[bool] = None,
    ) -> None:
        """
        Replace a node with another node.
        """
        debug: bool = self._debug
        if debugging is not None:
            debug = debugging

        if debug:
            print("Debugging mode is on.")
            print("Replacing the node.")
            print("Node: ", node)
            print("New Node: ", new_node)

        if not hasattr(self, "_alpha"):
            raise ValueError("Alpha is not initialized.")
        if not self.validate_alpha(alpha=self._alpha, debugging=debug):
            raise ValueError("Alpha is invalid.")
        if node not in self._alpha.nodes:
            raise ValueError("Node is not in the alpha.")
        if not new_node:
            raise ValueError("New node is empty.")
        if not isinstance(new_node, (_Variable, _Constant)):
            raise ValueError("New node must be a variable or a constant.")
        if debug:
            print("New node validated.")

        # Get the parent node of the node in the alpha
        parent_node: Any = list(self._alpha.predecessors(node))
        if debug:
            print("Parent Node: ", parent_node)

        if len(parent_node) >= 2:
            raise ValueError("The node has multiple parents.")

        # Get the starting index for the new node
        new_node_key: int = len(self._alpha.nodes)

        # Remove the edges between the parent node and the node
        self._alpha.remove_edge(parent_node[0], node)
        if debug:
            print("Edge Removed.")

        # Remove the node in the alpha
        subtree: nx.DiGraph = self._get_subtree(node)
        self._alpha.remove_nodes_from(subtree.nodes)
        if debug:
            print(f"Node Removed: {subtree.nodes()}")
            print("Node Removed.")

        # Add the new node to the alpha
        self._alpha.add_node(
            new_node_key,
            type=new_node.get_type(),
            data=new_node,
        )
        if debug:
            print("New Node Added.")

        # Add the edges between the parent node and the new node
        self._alpha.add_edge(parent_node[0], new_node_key)

        # Fix the node ID's
        self.fix(debugging=debug)

        if debug:
            print("Node Replaced.")

        return None

    def switch_node_with_node(
        self,
        node: int,
        new_node: Union[_Function, _Variable, _Constant],
        debugging: Optional[bool] = None,
    ) -> None:
        """
        Switch a node with another node.
        """
        debug: bool = self._debug
        if debugging is not None:
            debug = debugging

        if debug:
            print("Debugging mode is on.")
            print("Switching the node.")
            print("Node: ", node)
            print("New Node: ", new_node)

        if not hasattr(self, "_alpha"):
            raise ValueError("Alpha is not initialized.")
        if not self.validate_alpha(alpha=self._alpha, debugging=debug):
            raise ValueError("Alpha is invalid.")
        if node not in self._alpha.nodes:
            raise ValueError("Node is not in the alpha.")
        if not new_node:
            raise ValueError("New node is empty.")
        if debug:
            print("New node validated.")

        pass
