""" Utility functions for data handling."""

from typing import Sequence
import jax.numpy as jnp
from jax.tree_util import tree_map

from src.data_handling.data_structures import GraphsTuple


def stack_graphs(graphs: Sequence[GraphsTuple]):
    """
    Collate function for the datasets, which uses the GraphsTuple class.

    Args:
        graphs: List of GraphsTuple objects.

    Returns:
        Stacked GraphsTuple object.
    """
    graph = tree_map(lambda *x: jnp.stack(x), *graphs)
    graph = graph._replace(
        coarse_particle_count=graphs[0].coarse_particle_count)
    return graph
