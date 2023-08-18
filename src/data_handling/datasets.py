""" File containing the dataset classes. """

from typing import Union, Optional, Tuple
from jax.typing import ArrayLike
from jax import Array

from pathlib import Path
import h5py
from torch.utils.data import Dataset
import numpy as np
from matscipy.neighbours import neighbour_list as neighbor_list

from src.data_handling.data_structures import NodeFeatures, EdgeFeatures, GraphsTuple


def compute_connectivity(position: ArrayLike,
                         radius_cutoff: float,
                         box: ArrayLike,
                         pbc: ArrayLike,
                         edge_capacity: Optional[int] = None,
                         ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """
    Computes the edge connectivity for the given nodes.

    Args:
        position: Position of the nodes. Shape (n_nodes, spatial_dim)
        radius_cutoff: Radius cutoff for the edges.
        box: Box size. Shape (spatial_dim, )
        pbc: Periodic boundary conditions. Shape (spatial_dim, )
        edge_capacity: Maximum number of edges in the graph. If None, no limit is set.

    Returns:
        Tuple containing the senders, receivers and the distance and displacement 
        between the nodes and lastly the padding mask for the edges. The padding mask is
        True for padding edges and False for real edges.
    """
    if position.shape[-1] == 2:
        position = np.concatenate([position, np.zeros_like(position[..., :1])],
                                  axis=-1)
        box = np.concatenate([box, np.ones_like(box[..., :1])], axis=-1)
        pbc = np.concatenate([pbc, np.ones_like(pbc[..., :1])], axis=-1)

    senders, receivers, distance, displacement = neighbor_list('ijdD',
                                                               cutoff=radius_cutoff,
                                                               positions=position,
                                                               cell=np.eye(
                                                                   3) * box,
                                                               pbc=pbc)

    n_edges = senders.shape[0]
    edge_padding_mask = np.ones((n_edges,), dtype=bool)

    if edge_capacity is not None:
        assert n_edges <= edge_capacity, \
            f"Number of edges ({n_edges}) exceeds the edge capacity ({edge_capacity})."

        senders = np.pad(senders, (0, edge_capacity - n_edges))
        receivers = np.pad(receivers, (0, edge_capacity - n_edges))
        distance = np.pad(distance, (0, edge_capacity - n_edges))
        displacement = np.pad(displacement,
                              ((0, edge_capacity - n_edges), (0, 0)))
        edge_padding_mask = np.pad(
            edge_padding_mask, (0, edge_capacity - n_edges))

    return senders, receivers, distance, displacement, ~edge_padding_mask


class RPFDataset(Dataset):
    """ Reverse Poiseuille flow dataset """

    def __init__(self,
                 directory_path: Union[Path, str],
                 sequence_length: int,
                 edge_capacity: int,
                 dt: float,
                 fine_radius_cutoff: float,
                 mode: str = 'train',):
        """
        Args:
            directory_path: Path to the directory containing the dataset.
            sequence_length: Length of the sequences.
            edge_capacity: Maximum number of edges in the graph.
            dt: Time step.
            fine_radius_cutoff: Radius cutoff for fine nodes.
            mode: Mode of the dataset. Can be 'train', 'val' or 'test'.
        """

        super().__init__()
        self.path = Path(directory_path) / f"{mode}.h5"
        self.sequence_length = sequence_length
        self.particle_type = self._get_data("particle_type")

        # NOTE: the second -1 shouldn't be there but somehow the last
        #       frame seems to be missing
        self.length = 20000 - sequence_length - 1 - 1

        self.edge_capacity = edge_capacity
        self.dt = dt
        self.fine_radius_cutoff = fine_radius_cutoff
        self.box = np.array([1.0, 2.0])
        self.pbc = np.array([True, True])

    def __len__(self) -> int:
        return self.length

    def _get_data(self, key: str, index: Optional[int] = None):
        """
        Get data from the h5 file.

        Args:
            key: key of the data to get.
            index: index of the data to get.

        Returns:
            data: data from the h5 file.
        """
        with h5py.File(self.path, "r") as f:
            data = f["00000"][key]
            if index is not None:
                data = data[index]
            return np.array(data)

    def __getitem__(self, index: int) -> GraphsTuple:
        select_idx = np.arange(
            index, index + self.sequence_length + 1, dtype=np.int64)
        raw_positions = self._get_data("position", select_idx)
        position_history = raw_positions[:-2].swapaxes(0, 1)
        position = raw_positions[-2]
        target_position = raw_positions[-1]
        is_padding = np.zeros((position.shape[0],), dtype=bool)
        original_id = np.arange(position.shape[0], dtype=np.int64)

        nodes = NodeFeatures(
            position=position,
            position_history=position_history,
            is_padding=is_padding,
            original_id=original_id,
            target_position=target_position,
        )

        senders, receivers, distance, displacement, edge_padding_mask \
            = compute_connectivity(
                position=position,
                radius_cutoff=self.fine_radius_cutoff,
                box=self.box,
                pbc=self.pbc,
                edge_capacity=self.edge_capacity,
            )

        edges = EdgeFeatures(distance=distance,
                             displacement=displacement,
                             is_padding=edge_padding_mask)

        graph = GraphsTuple(senders=senders,
                            receivers=receivers,
                            nodes=nodes,
                            edges=edges,
                            coarse_particle_count=0)

        return graph
