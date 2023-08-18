import numpy as np
import jax.numpy as jnp

from src.utils import (
    MultiResolutionSettings,
    MultiResolutionState,
)

from src.data_structures import GraphsTuple



def merge_and_split(
    graph: GraphsTuple,
    is_padding_fine: jnp.ndarray,
    is_padding_coarse: jnp.ndarray,
    settings: MultiResolutionSettings,
) -> GraphsTuple:
    """
    Merges and splits particles, given the merge and split lists.

    Args:
        graph: GraphsTuple containing the current state of the simulation.
        merge_assignments: Array containing the merge assignments.
        split_assignments: Array containing the split assignments.
        is_coarse: Array containing the coarse assignments.
        is_padding: Array containing the padding assignments.
        is_coarse_after: Array containing the coarse assignments after merging and
            splitting.
        is_padding_after: Array containing the padding assignments after merging and
            splitting.
        settings: MultiResolutionSettings object.
    Returns:
        graph: GraphsTuple containing the updated merged and split particles.
    """
    # TODO placeholder name
    num_fine_nodes_after_merge = (
        graph.fine_nodes.position.shape[1] - 3 * settings.maximum_num_merges
    )
    
    position_fine = graph.fine_nodes.position
    position_coarse = graph.coarse_nodes.position
    
    merge_threshold = settings.merge_threshold
    split_threshold = settings.split_threshold

    # Sort scores and positions
    scores_coarse = graph.fine_nodes.coarse_score * ~is_padding_fine
    scores_fine = graph.coarse_nodes.coarse_score * ~is_padding_coarse
    
    merge_order = sorted(jnp.arange(position_fine.shape[0]), key=lambda x: scores_coarse[x], reverse=False)    
    split_order = sorted(jnp.arange(position_coarse.shape[0]), key=lambda x: scores_fine[x], reverse=False)
    
    # Apply merging first based on fine scores
    merge_check_iter = 0
    split_iter = 0
    current_merge_candidate_index = merge_order[merge_check_iter]
    
    while (scores_fine[current_merge_candidate_index] > merge_threshold):
        
        if scores_fine[current_merge_candidate_index] == jnp.inf:
            merge_check_iter+=1 
            current_merge_candidate_index = merge_order[merge_check_iter]
            continue
        
        # Merging
        neighbor_distances = jnp.where(graph.edges.senders == current_merge_candidate_index &
                                       graph.edges.receivers < graph.get_fine_particle_count(),
                                       graph.edges.distance,
                                       jnp.inf
        )
        top_3_edge_indices = jnp.argpartition(neighbor_distances, 3)[:3]
        top_3_neighbor_indices = graph.edges.receivers[top_3_edge_indices]
        
        # Merge check 
        # TODO change to jnp and concat
        average_score = np.mean(scores_coarse[top_3_neighbor_indices], 
                                scores_coarse[current_merge_candidate_index]
        )
        if average_score > merge_threshold:
            new_coarse_position = np.mean(position_fine[top_3_neighbor_indices], 
                                          position_fine[current_merge_candidate_index])
            
            # Find a coarse particle to be splitted unconditionally
            top_split_candidate_index = split_order[split_iter]
            pos_top_split_candidate = position_coarse[top_split_candidate_index]
            new_fine_positions = split_position(pos_top_split_candidate, settings)
            # TODO: random rotation of square shape 

            # Add the new coarse position
            position_coarse.at[top_split_candidate_index].set(new_coarse_position)
             
            # Add the 4 new fine positions
            position_fine.at[current_merge_candidate_index].set(new_fine_positions[0])
            position_fine.at[top_3_neighbor_indices[0]].set(new_fine_positions[1])
            position_fine.at[top_3_neighbor_indices[1]].set(new_fine_positions[2])
            position_fine.at[top_3_neighbor_indices[2]].set(new_fine_positions[3])
                        
            # Set the newly updated particle scores to 0                     
            scores_coarse.at[top_split_candidate_index].set(jnp.inf)
            scores_fine.at[current_merge_candidate_index].set(jnp.inf)
            scores_fine.at[top_3_neighbor_indices[0]].set(jnp.inf)
            scores_fine.at[top_3_neighbor_indices[1]].set(jnp.inf)
            scores_fine.at[top_3_neighbor_indices[2]].set(jnp.inf)
            
            split_iter+=1
               
        merge_check_iter+=1 
        current_merge_candidate_index = merge_order[merge_check_iter]
    
    
    # Re-sort fine particles based on the udpated scores, i.e., 0 when swapped 
    merge_order = sorted(jnp.arange(position_fine.shape[0]), key=lambda x: scores_coarse[x], reverse=False)    
    
    # Apply splitting based on coarse scores
    split_check_iter = split_iter
    merge_iter = 0
    top_split_candidate_index = split_order[split_check_iter]
    
    while (scores_coarse[top_split_candidate_index] > split_threshold):
        # Splitting
        new_fine_positions = split_position(position_coarse[top_split_candidate_index], 
                                                settings)
        
        # Find a fine top candidate particle to be merged unconditionally 
        # with its top 3 neighbors
        current_merge_candidate_index = merge_order[merge_iter]
        neighbor_distances = jnp.where(graph.edges.senders == current_merge_candidate_index &
                                       graph.edges.receivers < graph.get_fine_particle_count(),
                                       graph.edges.distance,
                                       jnp.inf
        )
        top_3_edge_indices = jnp.argpartition(neighbor_distances, 3)[:3]
        top_3_neighbor_indices = graph.edges.receivers[top_3_edge_indices]
        new_coarse_position = np.mean(position_fine[top_3_neighbor_indices], 
                                          position_fine[current_merge_candidate_index])
        
        # Add the new coarse position
        position_coarse.at[top_split_candidate_index].set(new_coarse_position)
            
        # Add the 4 new fine positions
        position_fine.at[current_merge_candidate_index].set(new_fine_positions[0])
        position_fine.at[top_3_neighbor_indices[0]].set(new_fine_positions[1])
        position_fine.at[top_3_neighbor_indices[1]].set(new_fine_positions[2])
        position_fine.at[top_3_neighbor_indices[2]].set(new_fine_positions[3])
                 
           
        merge_iter+=1
        split_check_iter+=1
        top_split_candidate_index = split_order[split_check_iter]
     
    return Harish



def split_position(
    center_position: np.array, settings: MultiResolutionSettings
) -> np.array:
    """
    Splits the positions of the particles.

    Args:
        position: Array containing the positions of the particles.
        split_assignments: Array containing the particles ids which are to be split.
        settings: MultiResolutionSettings containing the settings for the
            multi-resolution update.
    Returns:
        Tuple consisting of:
            - new_positions: Array containing the positions of the 4 new particles
    """
    offset = settings.split_offset
    ul_corner = center_position - np.array([-1, 1]) * offset
    ur_corner = center_position - np.array([1, 1]) * offset
    ll_corner = center_position - np.array([-1, -1]) * offset
    lr_corner = center_position - np.array([1, -1]) * offset
    new_positions = np.concatenate([ul_corner, ur_corner, ll_corner, lr_corner])

    # apply periodic boundary conditions
    box = np.array(settings.box)
    new_positions = np.mod(new_positions, box)

    return new_positions