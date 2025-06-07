from enum import Enum
from typing import Dict

__all__ = [
    "_MutationStage",
    "_MutationType",
    "_MutationProbability",
    "_RANDAM_GENERATED_ALPHA_AGID",
]


# AGID's for the randomly created alphas
_RANDAM_GENERATED_ALPHA_AGID: str = "ZYPHI"


class _MutationStage(Enum):
    """Enum for the different stages of evolution."""

    EXPLORATION = "Exploration"
    MID = "Mid-Stage"
    EXPLOITATION = "Exploitation"
    NEAR_END = "Near End Stage"
    FINAL = "Final Stage"


class _MutationType(Enum):
    """
    Enum for the different mutation types.

    Types of _Mutation Operators:
    ---

     1. Subtree _Mutationx:
        - Description: Replaces a randomly selected subtree with a new, randomly generated subtree.
        - Use Case: Introduces significant structural diversity; ideal for escaping local optima.
        - Benefits: High exploration potential.

    ---

    2. Replacement _Mutation
        - Description: Replaces a randomly chosen subtree with an equivalent subtree from another individual in the population.
        - Use Case: Useful for sharing genetic material across solutions while maintaining diversity.
        - Benefits: Combines exploration and exploitation.

    ---

    3. Insertion _Mutation
        - Description: Adds a randomly generated subtree to a randomly chosen node in the tree.
        - Use Case: Introduces new complexity or functionality.
        - Benefits: Expands the search space.

    ---

    4. Hoist _Mutation
        - Description: Replaces the entire tree (or a significant part) with a randomly chosen subtree, effectively "hoisting" it to the root or near-root level.
        - Use Case: Reduces tree bloat and over-complexity; focuses on promising substructures.
        - Benefits: Simplifies the tree while preserving useful parts.

    ---

    5. Shrink _Mutation
        - Description: Removes a randomly selected subtree and replaces it with a simpler element (like a terminal node or a smaller subtree).
        - Use Case: Controls tree growth and reduces bloat.
        - Benefits: Keeps solutions compact and reduces computational overhead.

    ---

    6. Deletion _Mutation
        - Description: Removes a subtree or node, replacing it with a minimal structure (e.g., a terminal node).
        - Use Case: Counteracts overfitting and prevents tree bloat.
        - Benefits: Simplifies overly complex solutions.

    ---

    7. Depth-Limited _Mutation
        - Description: Applies mutation (e.g., subtree mutation) but limits the depth of the mutated subtree to prevent excessive tree growth.
        - Use Case: Balances exploration while keeping the solutions computationally feasible.
        - Benefits: Ensures controlled growth of tree depth.

    ---

    8. Point _Mutation
        - Description: Modifies a single node in the tree by replacing its function, operator, or value with another randomly chosen element.
        - Use Case: Fine-tunes solutions; suitable for late stages of evolution.
        - Benefits: Small, localized changes to improve solutions without drastic alterations.

    ---

    9. Creep _Mutation
        - Description: Slightly adjusts numerical values (e.g., constants or weights) in terminal nodes or parameters.
        - Use Case: Fine-tunes solutions with continuous-valued parameters.
        - Benefits: Allows for precise optimization of numerical solutions.

    ---

    10. Swap _Mutation
        - Description: Swaps two randomly selected subtrees or nodes in the same tree.
        - Use Case: Preserves the overall tree structure while introducing minor structural changes.
        - Benefits: Maintains solution integrity with moderate exploration.

    ---

    11. Permutation _Mutation
        - Description: Rearranges the order of child nodes or functions within the tree while keeping the structure and functionality intact.
        - Use Case: Tests the sensitivity of solutions to node arrangement.
        - Benefits: Preserves functional integrity but explores alternate configurations.

    ---

    12. Root _Mutation
        - Description: Modifies the root of the tree by replacing its function/operator or adding new children.
        - Use Case: Introduces top-level changes to influence the overall behavior of the tree.
        - Benefits: Affects the solution at a high level without disrupting the entire tree.

    ---

    13. Rotation _Mutation
        - Description: Rotates the tree structure around a randomly chosen node, changing the order of the branches.
        - Use Case: Tests the sensitivity of solutions to node arrangement.
        - Benefits: Preserves functional integrity but explores alternate configurations.

    ---

    """

    NONE = "No Mutation"
    SUBTREE_MUTATION = "Subtree Mutation"
    REPLACEMENT_MUTATION = "Replacement Mutation"
    INSERTION_MUTATION = "Insertion Mutation"
    HOIST_MUTATION = "Hoist Mutation"
    SHRINK_MUTATION = "Shrink Mutation"
    DELETION_MUTATION = "Deletion Mutation"
    DEPTH_LIMITED_MUTATION = "Depth-Limited Mutation"
    POINT_MUTATION = "Point Mutation"
    CREEP_MUTATION = "Creep Mutation"
    SWAP_MUTATION = "Swap Mutation"
    PERMUTATION_MUTATION = "Permutation Mutation"
    ROOT_MUTATION = "Root Mutation"
    ROTATION_MUTATION = "Rotation Mutation"


class _MutationProbability:
    """Class to store and manage mutation probabilities for each stage."""

    def __init__(self):
        # Initialize mutation probabilities for each stage
        self.probabilities: Dict[_MutationStage, Dict[_MutationType, float]] = {
            _MutationStage.EXPLORATION: {
                _MutationType.SUBTREE_MUTATION: 0.25,
                _MutationType.REPLACEMENT_MUTATION: 0.15,
                _MutationType.INSERTION_MUTATION: 0.15,
                _MutationType.HOIST_MUTATION: 0.10,
                _MutationType.SHRINK_MUTATION: 0.05,
                _MutationType.DELETION_MUTATION: 0.05,
                _MutationType.DEPTH_LIMITED_MUTATION: 0.10,
                _MutationType.POINT_MUTATION: 0.05,
                _MutationType.CREEP_MUTATION: 0.05,
                _MutationType.SWAP_MUTATION: 0.05,
                _MutationType.PERMUTATION_MUTATION: 0.05,
                _MutationType.ROOT_MUTATION: 0.05,
                _MutationType.ROTATION_MUTATION: 0.05,
            },
            _MutationStage.MID: {
                _MutationType.SUBTREE_MUTATION: 0.20,
                _MutationType.REPLACEMENT_MUTATION: 0.20,
                _MutationType.INSERTION_MUTATION: 0.15,
                _MutationType.HOIST_MUTATION: 0.15,
                _MutationType.SHRINK_MUTATION: 0.10,
                _MutationType.DELETION_MUTATION: 0.05,
                _MutationType.DEPTH_LIMITED_MUTATION: 0.10,
                _MutationType.POINT_MUTATION: 0.05,
                _MutationType.CREEP_MUTATION: 0.05,
                _MutationType.SWAP_MUTATION: 0.05,
                _MutationType.PERMUTATION_MUTATION: 0.05,
                _MutationType.ROOT_MUTATION: 0.05,
                _MutationType.ROTATION_MUTATION: 0.05,
            },
            _MutationStage.EXPLOITATION: {
                _MutationType.SUBTREE_MUTATION: 0.15,
                _MutationType.REPLACEMENT_MUTATION: 0.20,
                _MutationType.INSERTION_MUTATION: 0.10,
                _MutationType.HOIST_MUTATION: 0.10,
                _MutationType.SHRINK_MUTATION: 0.15,
                _MutationType.DELETION_MUTATION: 0.15,
                _MutationType.DEPTH_LIMITED_MUTATION: 0.05,
                _MutationType.POINT_MUTATION: 0.05,
                _MutationType.CREEP_MUTATION: 0.05,
                _MutationType.SWAP_MUTATION: 0.05,
                _MutationType.PERMUTATION_MUTATION: 0.05,
                _MutationType.ROOT_MUTATION: 0.05,
                _MutationType.ROTATION_MUTATION: 0.05,
            },
            _MutationStage.NEAR_END: {
                _MutationType.SUBTREE_MUTATION: 0.10,
                _MutationType.REPLACEMENT_MUTATION: 0.25,
                _MutationType.INSERTION_MUTATION: 0.05,
                _MutationType.HOIST_MUTATION: 0.05,
                _MutationType.SHRINK_MUTATION: 0.15,
                _MutationType.DELETION_MUTATION: 0.20,
                _MutationType.DEPTH_LIMITED_MUTATION: 0.10,
                _MutationType.POINT_MUTATION: 0.05,
                _MutationType.CREEP_MUTATION: 0.05,
                _MutationType.SWAP_MUTATION: 0.05,
                _MutationType.PERMUTATION_MUTATION: 0.05,
                _MutationType.ROOT_MUTATION: 0.05,
                _MutationType.ROTATION_MUTATION: 0.05,
            },
            _MutationStage.FINAL: {
                _MutationType.SUBTREE_MUTATION: 0.05,
                _MutationType.REPLACEMENT_MUTATION: 0.20,
                _MutationType.INSERTION_MUTATION: 0.05,
                _MutationType.HOIST_MUTATION: 0.05,
                _MutationType.SHRINK_MUTATION: 0.20,
                _MutationType.DELETION_MUTATION: 0.25,
                _MutationType.DEPTH_LIMITED_MUTATION: 0.05,
                _MutationType.POINT_MUTATION: 0.10,
                _MutationType.CREEP_MUTATION: 0.05,
                _MutationType.SWAP_MUTATION: 0.05,
                _MutationType.PERMUTATION_MUTATION: 0.05,
                _MutationType.ROOT_MUTATION: 0.05,
                _MutationType.ROTATION_MUTATION: 0.05,
            },
        }

    def __str__(self):
        """Return a formatted string of all mutation probabilities."""
        result = [
            "Mutation Probabilities:",
            "----------------------",
            "",
        ]
        for stage, mutations in self.probabilities.items():
            result.append(f"{stage.value}:")
            for mutation, prob in mutations.items():
                result.append(f"  - {mutation.value}: {prob}")
            result.append("")
        return "\n".join(result)

    def get_mutation_probability(
        self, stage: _MutationStage, mutation_type: _MutationType
    ) -> float:
        """Get the probability for a specific mutation at a given stage."""
        # Fetch probability from the dictionary using the stage and mutation type
        return self.probabilities.get(stage, {}).get(mutation_type, 0.0)

    def get_mutation_probabilities(
        self, stage: _MutationStage
    ) -> Dict[_MutationType, float]:
        """Get all mutation probabilities for a given stage."""
        # Fetch probabilities from the dictionary using the stage
        return self.probabilities.get(stage, {})

    def set_mutation_probability(
        self, stage: _MutationStage, mutation_type: _MutationType, probability: float
    ) -> None:
        """Set the probability for a specific mutation at a given stage."""
        if stage in self.probabilities:
            self.probabilities[stage][mutation_type] = probability
