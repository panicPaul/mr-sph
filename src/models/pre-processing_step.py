""" Pre-processing step for the simulation. """

import jax.numpy as jnp
import haiku as hk

from src.data_handling.data_structures import GraphsTuple
from src.models.layers import EncoderProcessorDecoder


class PreProcessor(hk.Module):
    """ Pre-processing step for the simulation. """

    def __init__(self,
                 dt: float,
                 spatial_dim: int,
                 node_latent_size: int,
                 edge_latent_size: int,
                 message_passing_layers: int,
                 message_passing_steps: int,
                 preprocessing_target: str = "input",):
        """
        Args:
            dt: Time step.
            spatial_dim: Spatial dimension.
            node_latent_size: Size of the latent node features.
            edge_latent_size: Size of the latent edge features.
            preprocessing_target: Which features to preprocess. Can be "input" or
                "target".
        """
        super().__init__(name=f"{preprocessing_target}_preprocessor")
        self.dt = dt
        self.coarse_score_gnn = EncoderProcessorDecoder(
            name="coarse_score_gnn",
            node_input_size=spatial_dim,
            edge_input_size=spatial_dim + 1,
            node_output_size=node_latent_size,
            edge_output_size=edge_latent_size,
            message_passing_layers=message_passing_layers,
            message_passing_steps=message_passing_steps,
        )

    def featurizer(self, graph: GraphsTuple) -> GraphsTuple:
        """
        Computes the inital latent features of the graph.

        Args:
            graph: Input graph.

        Returns:
            Featurized graph.
        """
        # node features
        positions = jnp.concatenate(
            [graph.nodes.position_history,
             graph.nodes.position[:, None, :]],
            axis=1)

        velocities = jnp.diff(positions, axis=1) / self.dt
        latent_nodes = velocities.reshape(
            (velocities.shape[0], -1))
        nodes = graph.nodes._replace(latent=latent_nodes)

        # edge features
        latent_edges = jnp.concatenate(
            [graph.edges.displacement,
             graph.edges.distance[:, None]],
            axis=-1)
        edges = graph.edges._replace(latent=latent_edges)

        return graph._replace(nodes=nodes, edges=edges)
