""" Multi resolution kit for sph simulations. """

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp


from src.utils import (
    MultiResolutionSettings,
    MultiResolutionState,
    GraphsTuple
)


@partial(jax.jit, static_argnames=("settings"))
def merge_only(
    graph: GraphsTuple,
    settings: MultiResolutionSettings,
    state: MultiResolutionState,
) -> Tuple[GraphsTuple, MultiResolutionState]:
    """merge only"""
    coarse_scores = graph.fine_nodes.coarse_score

    senders_shape_dtype = jax.ShapeDtypeStruct((settings.edge_capacity,), jnp.int32)

    merge_assignments_shape_dtype = jax.ShapeDtypeStruct((node_capacity,), jnp.int32)

    is_coarse_shape_dtype = jax.ShapeDtypeStruct((new_node_capacity,), jnp.bool_)
    is_padding_shape_dtype = jax.ShapeDtypeStruct((new_node_capacity,), jnp.bool_)
    edge_mask_shape_dtype = jax.ShapeDtypeStruct((settings.edge_capacity,), jnp.bool_)
    state_shape_dtype = state
    did_buffer_overflow_shape_dtype = jax.ShapeDtypeStruct((), jnp.bool_)

    shape_dtype = (
        senders_shape_dtype,
        senders_shape_dtype,
        merge_assignments_shape_dtype,
        is_coarse_shape_dtype,
        is_padding_shape_dtype,
        edge_mask_shape_dtype,
        state_shape_dtype,
        did_buffer_overflow_shape_dtype,
    )

    callback_kwargs = {
        "position": graph.nodes.position[-1],
        "coarse_scores": coarse_scores,
        "is_coarse": graph.nodes.is_coarse,
        "is_padding": graph.nodes.is_padding,
        "settings": settings,
        "state": state,
    }

    (
        senders,
        receivers,
        merge_assignments,
        is_coarse,
        is_padding,
        edge_mask,
        state,
        did_buffer_overflow,
    ) = jax.pure_callback(
        callback=external_merge, result_shape_dtypes=shape_dtype, **callback_kwargs
    )

    if graph.edges is not None and graph.edges.latent is not None:
        edge_dim = graph.edges.latent.shape[-1]
        latent_edges = jnp.zeros((senders.shape[0], edge_dim))
    else:
        latent_edges = None

    edges = EdgeFeatures(latent=latent_edges, is_padding=edge_mask)
    graph = graph._replace(senders=senders, receivers=receivers, edges=edges)

    operands = (graph, merge_assignments, is_coarse, is_padding)

    graph = jax.lax.cond(
        did_buffer_overflow,
        partial(get_junk_graph_merge_only, settings=settings),
        partial(jax_merge_only, settings=settings),
        *operands
    )

    return graph, state