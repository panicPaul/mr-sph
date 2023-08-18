""" This file contains all the data structures used in the project."""

from typing import NamedTuple, Optional
from jax.typing import ArrayLike
from jax import Array
from jax.tree_util import tree_map
from jax import vmap

# --------------------------- Graph data structures ---------------------------


class NodeFeatures(NamedTuple):
    """
    Node features.

    Attributes:
        - latent: Latent representation of the node. Shape (n_nodes, latent_dim)
        - position: Position of the node. Shape (n_nodes, spatial_dim)
        - position_history: Position history of the node. Shape
            (n_nodes, sequence_length - 1, spatial_dim)
        - is_padding: Whether the node is padding. Shape (n_nodes, )
        - original_id: Original index of the node. Shape (n_nodes, ) for fine nodes
            and (n_nodes, 4) for coarse nodes.
        - coarse_score: Coarse score of the node. Shape (n_nodes, )
        - acceleration_mean: Mean of the acceleration of the node. Shape (n_nodes, )
        - acceleration_covariance: Covariance of the acceleration of the node. Shape
            (n_nodes, spatial_dim, spatial_dim)
        - target_position: Target position of the node. Shape (n_nodes, spatial_dim)
    """

    latent: ArrayLike = None
    position: ArrayLike = None
    position_history: ArrayLike = None
    is_padding: ArrayLike = None

    original_id: ArrayLike = None
    coarse_score: Optional[Array] = None

    acceleration_mean: Optional[Array] = None
    acceleration_covariance: Optional[Array] = None
    target_position: Optional[ArrayLike] = None


class EdgeFeatures(NamedTuple):
    """
    Edge features.

    Attributes:
        - latent: Latent representation of the edge. Shape (n_edges, latent_dim)
        - distance: Distance between the nodes. Shape (n_edges, )
        - displacement: Displacement between the nodes. Shape (n_edges, spatial_dim)
        - is_padding: Whether the edge is padding. Shape (n_edges, )
    """

    latent: Array = None
    distance: Array = None
    displacement: Array = None
    is_padding: ArrayLike = None


class GraphsTuple(NamedTuple):
    """
    GraphsTuple with additional functionality.

    Attributes:
        - senders: Senders of the edges. Shape (n_edges, )
        - receivers: Receivers of the edges. Shape (n_edges, )
        - nodes: NodeFeatures
        - edges: EdgeFeatures
        - coarse_particle_count: Number of coarse particles in the graph.

    Methods:
        - get_latent_graph: Returns a graph with only latent features.
        - get_position_graph: Returns a graph with only position features.
        - replace_latent: Returns a graph with the latent nodes and edges replaced by
            the given nodes and edges.
        - replace_positional: Returns a graph with the positional information replaced.
    """

    senders: ArrayLike
    receivers: ArrayLike
    nodes: NodeFeatures
    edges: EdgeFeatures
    coarse_particle_count: int = 0

    def get_latent_graph(self) -> 'GraphsTuple':
        """Returns a graph with only latent features."""
        return self._replace(fine_nodes=self.fine_nodes.latent,
                             coarse_nodes=self.coarse_nodes.latent,
                             edges=self.edges.latent)

    def get_positional_graph(self) -> 'GraphsTuple':
        """Returns a graph with only the positional features."""
        return self._replace(fine_nodes=self.fine_nodes.position,
                             coarse_nodes=self.coarse_nodes.position)

    def replace_latent(
        self,
        fine_node_features: Optional[Array],
        coarse_node_features: Optional[Array],
        edge_features: Optional[Array],
    ):
        """
        Returns a graph with the latent nodes and edges replaced by the given nodes and
        edges.

        Args:
            fine_node_features: Latent features of the fine nodes. Shape (n_fine_nodes,
                latent_dim)
            coarse_node_features: Latent features of the coarse nodes. Shape
                (n_coarse_nodes, latent_dim)
            edge_features: Latent features of the edges. Shape (n_edges, latent_dim)
        """
        fine_nodes = self.fine_nodes._replace(latent=fine_node_features) \
            if fine_node_features is not None else self.fine_nodes
        coarse_nodes = self.coarse_nodes._replace(latent=coarse_node_features) \
            if coarse_node_features is not None else self.coarse_nodes
        edges = self.edges._replace(latent=edge_features) \
            if edge_features is not None else self.edges

        return self._replace(fine_nodes=fine_nodes,
                             coarse_nodes=coarse_nodes,
                             edges=edges)

    def fine_nodes(self) -> NodeFeatures:
        """ Returns the fine nodes. """

        nodes = self.nodes._asdict()

        if self.nodes.latent is None:
            nodes.pop('latent')
        if self.nodes.coarse_score is None:
            nodes.pop('coarse_score')
        if self.nodes.acceleration_mean is None:
            nodes.pop('acceleration_mean')
        if self.nodes.acceleration_covariance is None:
            nodes.pop('acceleration_covariance')
        if self.nodes.target_position is None:
            nodes.pop('target_position')

        if self.nodes.position.ndim == 2:
            nodes = tree_map(
                lambda x: x[:-self.coarse_particle_count], nodes)
        else:
            nodes = tree_map(
                lambda x: x[:, :-self.coarse_particle_count], nodes)

        return NodeFeatures(**nodes)

    def coarse_nodes(self) -> NodeFeatures:
        """ Returns the coarse nodes. """

        nodes = self.nodes._asdict()

        if self.nodes.latent is None:
            nodes.pop('latent')
        if self.nodes.coarse_score is None:
            nodes.pop('coarse_score')
        if self.nodes.acceleration_mean is None:
            nodes.pop('acceleration_mean')
        if self.nodes.acceleration_covariance is None:
            nodes.pop('acceleration_covariance')
        if self.nodes.target_position is None:
            nodes.pop('target_position')

        if self.nodes.position.ndim == 2:
            nodes = tree_map(
                lambda x: x[-self.coarse_particle_count:], nodes)
        else:
            nodes = tree_map(
                lambda x: x[:, -self.coarse_particle_count:], nodes)

        return NodeFeatures(**nodes)
