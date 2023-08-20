""" Multi-Resolution handling """

import jax
import jax.numpy as jnp
from jax import Array

from src.data_handling.data_structures import GraphsTuple, NodeFeatures, EdgeFeatures


def split_position(center_position: Array,
                   box: Array,
                   split_offset: float,
                   key: jax.random.PRNGKey, 
) -> Array:
    """
    Splits the positions of the particles.

    Args:
        - center_position: Position of the particle to split.
        - box: Box size.
        - split_offset: Offset for splitting.
        - key: Random key.
    Returns:
        Array consisting of:
            - new_positions: Array containing the positions of the 4 new particles
    """
    # TODO: add random rotation
    rotation = jnp.eye(2) 
    ul_corner = center_position - rotation @ jnp.array([-1, 1]) * split_offset
    ur_corner = center_position - rotation @ jnp.array([1, 1]) * split_offset
    ll_corner = center_position - rotation @ jnp.array([-1, -1]) * split_offset
    lr_corner = center_position - rotation @ jnp.array([1, -1]) * split_offset
    new_positions = jnp.concatenate([ul_corner, ur_corner, ll_corner, lr_corner])

    # apply periodic boundary conditions
    new_positions = jnp.mod(new_positions, box)

    return new_positions

def merge_and_split(nodes: NodeFeatures,
                    senders: Array,
                    receivers: Array,
                    edges: EdgeFeatures,
                    max_edges: int,
                    fine_particle_count: int,
                    merge_score: Array,
                    split_score: Array,
                    merge_idx: int,
                    split_idx: int,
                    merge_threshold: float,
                    box: Array,
                    split_offset: float,
                    key: jax.random.PRNGKey,
                    ):
    """
    Single step of merge-split algorithm. Merges four nodes into one and splits one node
    into four to keep the number of nodes constant.

    Args:
        - nodes: Node features of the graph.
        - senders:  Senders of the edges.
        - receivers: Receivers of the edges.
        - edges: Edge features of the graph.
        - max_edges: Maximum number of edges per node.
        - merge_score: Score for merging nodes.
        - split_score: Score for splitting nodes. Sorted in ascending order.
            order.
        - merge_idx: Index of the node to merge.
        - split_idx: Index of the node to split.
        - merge_threshold: Threshold for merging nodes.
    """
    # skip if we don't need to execute
    if merge_score[merge_idx] == jnp.inf:
        return nodes, edges, merge_score, split_score, split_idx
    if merge_score[merge_idx] < merge_threshold:
        return nodes, edges, merge_score, split_score, split_idx

    # look for 3 closest nodes
    # TODO: filter out fill values
    neighbor_indices = receivers[jnp.where(senders == merge_idx, size=max_edges, fill_value=-1)[0]]
    neighbor_indices = jnp.where(neighbor_indices < fine_particle_count, neighbor_indices, -1)
    distances = edges.distance[neighbor_indices] * jnp.where(merge_score != jnp.inf, 1, jnp.inf)    
    distances, nn_indices = jax.lax.top_k(-distances, k=3) 
    
    if jnp.any(distances == jnp.inf):
        return nodes, edges, merge_score, split_score, split_idx
    
    if merge_score[merge_idx] + merge_score[nn_indices].sum() > 4 * merge_threshold:
        return nodes, edges, merge_score, split_score, split_idx
    
    if nodes.is_padding[merge_idx] or jnp.any(nodes.is_padding[nn_indices]):
        return nodes, edges, merge_score, split_score, split_idx
    
    # select the partcle to split
    if split_idx < 0:
        return nodes, edges, merge_score, split_score, split_idx
    
    # TODO: check for padding
    
    latent_c2f = nodes.latent[split_idx]
    key, subkey = jax.random.split(key)
    position_c2f = split_position(nodes.position[split_idx], box, split_offset, subkey)
    position_history_c2f =  jax.vmap(split_position, in_axes=0)(nodes.position_history[split_idx], box, split_offset, subkey)
    is_padding_c2f = nodes.is_padding[split_idx]
    original_id_c2f = nodes.original_id[split_idx]
    target_position_c2f = split_position(nodes.target_position[split_idx], box, split_offset, subkey)
    
    # merge them
    latent = nodes.latent.at[split_idx].set((jnp.sum(nodes.latent[nn_indices], axis=0) + nodes.latent[merge_idx])/4)
    position = nodes.position.at[split_idx].set((jnp.sum(nodes.position[nn_indices], axis=0) + nodes.position[merge_idx])/4)
    position_history = nodes.position_history.at[split_idx].set((jnp.sum(nodes.position_history[nn_indices], axis=0) + nodes.position_history[merge_idx])/4)
    is_padding = nodes.is_padding.at[split_idx].set(False)
    original_id = nodes.original_id.at[split_idx].set(jnp.concatenate([nodes.original_id[nn_indices], nodes.original_id[merge_idx]]))
    target_position = nodes.target_position.at[split_idx].set((jnp.sum(nodes.target_position[nn_indices], axis=0) + nodes.target_position[merge_idx])/4)
    nodes = nodes._replace(latent=latent, position=position, position_history=position_history, is_padding=is_padding, original_id=original_id, target_position=target_position)
    
    # NOTE: we don't need to update edges, since we recompute them after merge-split anyway

    # take the coarse node with the highest score
    latent_c2f = split_latent(latent_c2f, rotation)
    
    for i, idx in enumerate(jnp.concatenate([nn_indices, merge_idx])):
        latent = nodes.latent.at[idx].set(latent_c2f[i])
        position = nodes.position.at[idx].set(position_c2f[i])
        position_history = nodes.position_history.at[idx].set(position_history_c2f[i])
        is_padding = nodes.is_padding.at[idx].set(is_padding_c2f[i])
        original_id = nodes.original_id.at[idx].set(original_id_c2f[i])
        target_position = nodes.target_position.at[idx].set(target_position_c2f[i])
        nodes = nodes._replace(latent=latent, position=position, position_history=position_history, is_padding=is_padding, original_id=original_id, target_position=target_position)
        

    # split it into 4 nodes replacing the former fine nodes
    # replace coarse node with newly merged node
