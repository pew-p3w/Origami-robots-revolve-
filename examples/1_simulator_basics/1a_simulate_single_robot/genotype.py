"""Genotype for the ball-approach EA experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class Genotype:
    """A genotype represented as a flat array of CPG connection weights."""

    parameters: npt.NDArray[np.float_]

    def copy(self) -> Genotype:
        """
        Copy this genotype.

        :returns: A copy of this genotype.
        """
        return Genotype(self.parameters.copy())

    @classmethod
    def random(cls, num_parameters: int, rng: np.random.Generator) -> Genotype:
        """
        Create a random genotype.

        :param num_parameters: Number of CPG connection weights.
        :param rng: Random number generator.
        :returns: A random genotype.
        """
        return Genotype(rng.random(size=num_parameters) * 2 - 1)

    def mutate(self, rng: np.random.Generator, mutate_std: float) -> Genotype:
        """
        Mutate this genotype by adding gaussian noise.

        :param rng: Random number generator.
        :param mutate_std: Standard deviation of the gaussian noise.
        :returns: A mutated copy.
        """
        new_params = self.parameters + rng.normal(scale=mutate_std, size=len(self.parameters))
        return Genotype(np.clip(new_params, -1.0, 1.0))

    @classmethod
    def one_point_crossover(
        cls,
        parent1: Genotype,
        parent2: Genotype,
        rng: np.random.Generator,
    ) -> tuple[Genotype, Genotype]:
        """
        Perform one-point crossover and return two child genotypes.

        :param parent1: First parent.
        :param parent2: Second parent.
        :param rng: Random number generator.
        :returns: Two new child genotypes.
        :raises ValueError: If parent lengths differ.
        """
        if len(parent1.parameters) != len(parent2.parameters):
            raise ValueError("Cannot crossover genotypes with different lengths.")
        if len(parent1.parameters) < 2:
            return parent1.copy(), parent2.copy()

        crossover_point = int(rng.integers(1, len(parent1.parameters)))
        child1 = np.concatenate(
            [
                parent1.parameters[:crossover_point],
                parent2.parameters[crossover_point:],
            ]
        )
        child2 = np.concatenate(
            [
                parent2.parameters[:crossover_point],
                parent1.parameters[crossover_point:],
            ]
        )
        return Genotype(child1), Genotype(child2)
