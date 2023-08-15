""" This file contains helpful utility functions for the JAX models."""
from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import jraph
import matplotlib.pyplot as plt
import networkx as nx


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


def _convert_jraph_to_networkx_graph(jraph_graph: jraph.GraphsTuple) -> nx.Graph:
    """
    Converts a jraph.GraphsTuple to a networkx graph. Adapted from the jraph tutorial.

    Args:
        jraph_graph: jraph.GraphsTuple
    Returns:
        nx_graph: nx.Graph
    """
    nodes, edges, receivers, senders, globals, n_node, n_edge = jraph_graph
    nx_graph = nx.DiGraph()
    for n in range(jnp.sum(n_node)):
        nx_graph.add_node(n, node_feature=nodes[n])
    for e in range(jnp.sum(n_edge)):
        nx_graph.add_edge(int(senders[e]), int(receivers[e]), edge_feature=edges[e])
    return nx_graph


def draw_jraph_graph_structure(
    jraph_graph: jraph.GraphsTuple, directional: bool = False
):
    """
    Draws a jraph.GraphsTuple using networkx. Adapted from the jraph tutorial.

    Args:
        jraph_graph: jraph.GraphsTuple
        directional: Whether to draw the graph as a directed graph.
    """
    if not directional:
        senders = jnp.concatenate([jraph_graph.senders, jraph_graph.receivers], axis=0)
        receivers = jnp.concatenate(
            [jraph_graph.receivers, jraph_graph.senders], axis=0
        )
        edges = jnp.concatenate([jraph_graph.edges] * 2, axis=0)
        n_edge = jraph_graph.n_edge * 2
        jraph_graph = jraph_graph._replace(
            senders=senders, receivers=receivers, edges=edges, n_edge=n_edge
        )
    nx_graph = _convert_jraph_to_networkx_graph(jraph_graph)
    pos = nx.spring_layout(nx_graph)
    plt.figure(figsize=(5, 4))
    nx.draw(nx_graph, pos=pos, with_labels=True, node_size=500, font_color="yellow")


def print_graph_properties(graph: jraph.GraphsTuple, name: Optional[str] = None):
    """
    Prints properties of a jraph.GraphsTuple.

    Args:
        graph: jraph.GraphsTuple
    """
    assert (
        graph.senders.shape == graph.receivers.shape == (jnp.sum(graph.n_edge),)
    ), "Invalid graph, number of senders and receivers does not match number of edges."
    print("-" * 80)
    if name is not None:
        print(f"{name} properties: \n")
    else:
        print("Graph properties:")
    print(f"Number of nodes: {graph.n_node}")
    print(f"Number of edges: {graph.n_edge}")
    print(f"Node features shape: {graph.nodes.shape}")
    print(f"Edge features shape: {graph.edges.shape}")
    print(f"Senders: {graph.senders}")
    print(f"Receivers: {graph.receivers}")
    print("-" * 80)
    print()


# -------------------------------  data structures  -------------------------------#


class NodeFeatures(NamedTuple):
    """
    Node features.

    Attributes:
        - latent: Latent representation of the node. Shape (n_nodes, latent_dim)
        - position: Position of the node. Shape (t, n_nodes, spatial_dim)
        - velocity: Velocity of the node. Shape (t, n_nodes, spatial_dim)
        - acceleration_mean: Mean acceleration of the node. Shape
            (t, n_nodes, spatial_dim)
        - acceleration_covariance: Covariance of the acceleration of the node. Shape
            (t, n_nodes, spatial_dim, spatial_dim)
        - mass: Mass of the node. Shape (n_nodes, )
        - is_coarse: Particle type of the node. Shape (n_nodes, )
        - target_position: Target position of the node. Shape (n_nodes, spatial_dim)
        - coarse_score: Probability of the node being in the coarse region. Shape
            (n_nodes, )
    """

    latent: jnp.ndarray = None

    position: jnp.ndarray = None
    velocity: jnp.ndarray = None
    acceleration_mean: jnp.ndarray = None
    acceleration_covariance: jnp.ndarray = None

    mass: jnp.ndarray = None

    is_coarse: jnp.ndarray = None
    is_padding: Optional[jnp.ndarray] = None

    target_position: Optional[jnp.ndarray] = None

    coarse_score: Optional[jnp.ndarray] = None


class EdgeFeatures(NamedTuple):
    """
    Edge features.

    Attributes:
        - latent: Latent representation of the edge. Shape (n_edges, latent_dim)
        - is_padding: Whether the edge is padding. Shape (n_edges, )
    """

    latent: jnp.ndarray = None
    is_padding: Optional[jnp.ndarray] = None


class GraphsTuple(NamedTuple):
    """
    GraphsTuple with additional functionality.

    Attributes:
        - senders: Senders of the edges. Shape (n_edges, )
        - receivers: Receivers of the edges. Shape (n_edges, )
        - nodes: NodeFeatures
        - edges: EdgeFeatures

    Methods:
        - get_latent_graph: Returns a graph with only latent features.
        - get_position_graph: Returns a graph with only position features.
        - replace_latent: Returns a graph with the latent nodes and edges replaced by
            the given nodes and edges.
        - replace_positional: Returns a graph with the positional information replaced.
    """

    senders: jnp.ndarray
    receivers: jnp.ndarray
    nodes: NodeFeatures
    edges: EdgeFeatures

    def get_latent_graph(self) -> jraph.GraphsTuple:
        """Returns a graph with only latent features."""
        return self._replace(nodes=self.nodes.latent, edges=self.edges.latent)

    def get_position_graph(self) -> jraph.GraphsTuple:
        """Returns a graph with only position features."""
        return self._replace(nodes=self.nodes.position)

    def replace_latent(
        self,
        latent_node_features: Optional[jnp.ndarray],
        latent_edge_features: Optional[jnp.ndarray],
    ):
        """
        Returns a graph with the latent nodes and edges replaced by the given nodes and
        edges.
        """
        assert (
            latent_node_features is not None or latent_edge_features is not None
        ), "At least one of the latent node or edge features must be given."
        if latent_node_features is None:
            latent_node_features = self.nodes.latent
        if latent_edge_features is None:
            latent_edge_features = self.edges.latent
        return self._replace(
            nodes=self.nodes._replace(latent=latent_node_features),
            edges=self.edges._replace(latent=latent_edge_features),
        )

    def replace_positional(
        self, acceleration: jnp.ndarray, velocity: jnp.ndarray, position: jnp.ndarray
    ):
        """
        Returns a graph with the positional nodes replaced by the given nodes.
        """
        nodes = self.nodes._replace(
            acceleration=acceleration, velocity=velocity, position=position
        )
        return self._replace(nodes=nodes)


def on_latent(function: Callable, graph: GraphsTuple, *args, **kwargs):
    """
    Executes a function on the latent graph of the given graphs tuple.

    Args:
        function: Function to execute on the latent graph.
        graph: Graphs tuple.
    Returns:
        The graphs tuple with the latent graph replaced by the result of the function.
    """
    latent_graph = graph.get_latent_graph()
    latent_graph = function(graph=latent_graph, *args, **kwargs)
    return graph.replace_latent(
        latent_edge_features=latent_graph.edges, latent_node_features=latent_graph.nodes
    )


class MultiResolutionSettings(NamedTuple):
    """
    Settings for multi resolution clustering and connectivity computation. Single values
    are used for each call to the multi resolution clustering function, lists are used
    to store the state of the multi resolution clustering function over multiple calls
    with different graphs, i.e. fine-grained and coarse-grained graphs.

    Attributes:
        - edge_capacity: Maximum number of edges after the multi resolution clustering.
        - maximum_num_splits: Maximum number of splits. The splits assignments will be
            padded to this length.
        - maximum_num_merges: Maximum number of merges. The number of merges will be
            padded to this amount.
        - coarse_radius_cutoff: Radius cutoff for the coarse-grained graph.
        - fine_radius_cutoff: Radius cutoff for the fine-grained graph.
        - box: Box size of the simulation. Used to compute the periodic boundary
            conditions.
        - periodic_boundary_conditions: Whether the periodic boundary conditions are
            active in the x, y and z direction.
        - interface_heights: Heights of the interfaces if used.
        - merge_threshold: Threshold for the merge assignment.
        - split_threshold: Threshold for the split assignment.
        - split_offset: Offset for the split assignment.
        - additional_node_padding: Additional padding for the nodes.
    """

    coarse_node_capacity: int
    edge_capacity: int
    maximum_num_splits: int
    maximum_num_merges: int

    coarse_radius_cutoff: float
    fine_radius_cutoff: float
    box: Tuple[float] = (1.0, 2.0)
    periodic_boundary_conditions: Tuple[bool] = (True, True, True)

    interface_heights: Optional[Tuple[float]] = (0.0, 0.25, 0.75, 1.25, 1.75, 2.0)
    merge_threshold: float = 0.5
    split_threshold: float = 0.5

    split_offset: float = 0.015

    additional_node_padding: int = 0

    def increase_buffer(
        self, state: "MultiResolutionState", capacity_factor: float = 1.25
    ):
        """
        Increases the buffer size of the multi resolution clustering.
        Note that in the case of a padding underflow, the changes are
        more drastic and do not happen here.

        Args:
            state: State of the multi resolution clustering.
            capacity_factor: Factor by which the capacity is increased.
        """
        maximum_num_merges = (
            int(state.merge_overflow * capacity_factor)
            if state.merge_overflow > 0
            else self.maximum_num_merges
        )
        maximum_num_splits = (
            int(state.split_overflow * capacity_factor)
            if state.split_overflow > 0
            else self.maximum_num_splits
        )
        edge_capacity = (
            int(state.edge_overflow * capacity_factor)
            if state.edge_overflow > 0
            else self.edge_capacity
        )
        return self._replace(
            maximum_num_merges=maximum_num_merges,
            maximum_num_splits=maximum_num_splits,
            edge_capacity=edge_capacity,
        )


class MultiResolutionState(NamedTuple):
    """
    State handling the overflow of the multi resolution clustering. Zero values are used
    to indicate that no overflow occured.

    Attributes:
        - merge_underflow: Number of merges that caused a merge underflow.
        - split_overflow: Number of splits that caused a split overflow.
        - padding_underflow: Number of padding nodes that caused a padding underflow.
    """

    merge_overflow: jnp.array = jnp.array(0)
    split_overflow: jnp.array = jnp.array(0)
    padding_underflow: jnp.array = jnp.array(0)
    edge_overflow: jnp.array = jnp.array(0)

    def did_buffer_overflow(self) -> bool:
        """
        Returns whether there was any buffer overflow.
        """
        return (
            self.merge_overflow > 0
            or self.split_overflow > 0
            or self.padding_underflow > 0
            or self.edge_overflow > 0
        )
