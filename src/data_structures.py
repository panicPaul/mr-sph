""" This file contains all the data structures used in the project."""

from typing import NamedTuple, Optional
from jax.typing import ArrayLike, Array


# --------------------------- Graph data structures ---------------------------
class NodeFeatures(NamedTuple):
    """
    Node features.

    Attributes:
        - latent: Latent representation of the node. Shape (n_nodes, latent_dim)
        - position: Position of the node. Shape (n_nodes, spatial_dim)
        - is_padding: Whether the node is padding. Shape (n_nodes, )
        - coarse_score: Coarse score of the node. Shape (n_nodes, )
        - original_id: Original index of the node. Shape (n_nodes, ) for fine nodes
            and (n_nodes, 4) for coarse nodes.
        - acceleration_mean: Mean of the acceleration of the node. Shape (n_nodes, )
        - acceleration_covariance: Covariance of the acceleration of the node. Shape
            (n_nodes, spatial_dim, spatial_dim)
        - target_position: Target position of the node. Shape (n_nodes, spatial_dim)
    """

    latent: ArrayLike = None
    position: ArrayLike = None
    is_padding: ArrayLike = None

    coarse_score: Optional[Array] = None
    original_id: Optional[Array] = None

    acceleration_mean: Optional[Array] = None
    acceleration_covariance: Optional[Array] = None
    target_position: Optional[ArrayLike] = None


class EdgeFeatures(NamedTuple):
    """
    Edge features.

    Attributes:
        - latent: Latent representation of the edge. Shape (n_edges, latent_dim)
        - is_padding: Whether the edge is padding. Shape (n_edges, )
    """

    latent: Array = None
    is_padding: ArrayLike = None


class GraphsTuple(NamedTuple):
    """
    GraphsTuple with additional functionality.

    Attributes:
        - senders: Senders of the edges. Shape (n_edges, )
        - receivers: Receivers of the edges. Shape (n_edges, )
        - fine_nodes: Fine nodes of the graph.
        - coarse_nodes: Coarse nodes of the graph.
        - edges: EdgeFeatures

    Methods:
        - get_latent_graph: Returns a graph with only latent features.
        - get_position_graph: Returns a graph with only position features.
        - replace_latent: Returns a graph with the latent nodes and edges replaced by
            the given nodes and edges.
        - replace_positional: Returns a graph with the positional information replaced.
    """

    senders: ArrayLike
    receivers: ArrayLike
    fine_nodes: NodeFeatures
    coarse_nodes: NodeFeatures
    edges: EdgeFeatures

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
