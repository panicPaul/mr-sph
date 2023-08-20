""" Multi-Resolution handling """

import jax.numpy as jnp
from jax import Array

from src.data_handling.data_structures import GraphsTuple, NodeFeatures, EdgeFeatures


def merge_and_split(nodes: NodeFeatures,
                    edges: EdgeFeatures,
                    merge_score: Array,
                    split_score: Array,
                    merge_idx: int,
                    split_idx: int,
                    merge_threshold: float,
                    ):
    """
    Single step of merge-split algorithm. Merges four nodes into one and splits one node
    into four to keep the number of nodes constant.

    Args:
    """
    # skip if we don't need to execute
    if merge_score[merge_idx] == jnp.inf:
        return nodes, edges, merge_score, split_score
    if merge_score[merge_idx] < merge_threshold:
        return nodes, edges, merge_score, split_score

    # look for 3 closest nodes

    # merge them

    # we don't need to update edges, since we recompute them after merge-split anyway

    # take the coarse node with the hightest score

    # split it into 4 nodes replacing the former fine nodes
    # replace coarse node with newly merged node
