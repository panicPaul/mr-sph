""" Utility functions for the models. """

from typing import Callable, Optional

import jax


def get_activation_fn(activation: Optional[str] = "relu") -> Callable:
    """
    Args:
        activation: Activation function to use.
    Returns:
        Activation function.
    """
    if activation == "relu":
        return jax.nn.relu
    elif activation == "sigmoid":
        return jax.nn.sigmoid
    elif activation == "tanh":
        return jax.nn.tanh
    elif activation == "swish":
        return jax.nn.swish
    elif activation == "elu":
        return jax.nn.elu
    else:
        raise RuntimeError(f"Invalid activation function: {activation}")


def get_aggregation_fn(aggregation_fn: Optional[str] = "sum") -> Callable:
    """
    Returns aggregation function.

    Args:
        aggregation_fn: Aggregation function to use.
    Returns:
        Aggregation function: Callable function with signature
            (data, segment_ids, num_segments=None)
    """
    if aggregation_fn == "sum":
        fn = jax.ops.segment_sum
    elif aggregation_fn == "mean":
        fn = jax.ops.segment_mean
    elif aggregation_fn == "max":
        fn = jax.ops.segment_max
    else:
        raise RuntimeError(f"Invalid aggregation function: {aggregation_fn}")

    return fn
