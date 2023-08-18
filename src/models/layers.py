"""
This file contains the JAX implementation of the layers,
as well as basic building blocks, used in the model.
"""
from typing import Any, Optional, Union
from jaxtyping import Array, ArrayLike

from functools import partial
import haiku as hk
import jax
import jax.numpy as jnp


class _Residual(hk.Module):
    """Residual connection wrapper."""

    def __init__(self, module: hk.Module) -> None:
        """
        Args:
            module: Module to apply the residual connection to.
        """
        super().__init__()
        self._module = module

    def __call__(self, x: Array) -> jnp:
        """
        Args:
            x: Input to the module.
        Returns:
            Output of the module.
        """
        return x + self._module(x)


class _Act(hk.Module):
    """Activation function wrapper."""

    def __init__(self, activation: Optional[str] = "relu") -> None:
        """
        Args:
            activation: Activation function to use.
        """
        super().__init__()
        if activation == "relu":
            self.activation = jax.nn.relu
        elif activation == "sigmoid":
            self.activation = jax.nn.sigmoid
        elif activation == "tanh":
            self.activation = jax.nn.tanh
        elif activation == "swish" or activation == "silu":
            self.activation = jax.nn.swish
        elif activation == "elu":
            self.activation = jax.nn.elu
        elif activation == "identity" or activation == "id":
            self.activation = lambda x: x
        else:
            raise RuntimeError(f"Invalid activation function: {activation}")

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        Args:
            x: Input to the activation function.
        Returns:
            Output of the activation function.
        """
        return self.activation(x)


class _Dropout(hk.Module):
    """Dropout wrapper."""

    def __init__(self, rate: float = 0.5, name: Optional[str] = None) -> None:
        """
        Args:
            rate: Dropout rate.
        """
        super().__init__()
        self.rate = rate

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        if kwargs.get("is_training", True):
            return hk.dropout(hk.next_rng_key(), self.rate, x)
        else:
            return x


class MLP(hk.Module):
    """Multi-layer perceptron (MLP) module."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        n_layers: int = 2,
        latent_size: int | None = None,
        activation: Optional[str] = "relu",
        residual: Optional[bool] = False,
        batch_norm: Optional[bool] = False,
        batch_norm_kwargs: Optional[dict[str, Any]] = None,
        layer_norm: Optional[bool] = False,
        layer_norm_kwargs: Optional[dict[str, Any]] = None,
        dropout: Optional[float] = 0.0,
        output_activation: Optional[str] | None = None,
        name: Optional[str] = None,
    ) -> hk.Sequential:
        """
        Args:
            input_size: Input dimension.
            output_size: Output dimension.
            n_layers: Number of layers.
            latent_size: Latent dimension. If None, same as `output_size`.
            activation: Activation function to use.
            batch_norm: Whether to use batch normalization.
                Will be added after each linear layer.
            batch_norm_kwargs: Additional arguments for the batch normalization.
                If None, default arguments are used.
            layer_norm: Whether to use layer normalization.
                Will be added before each linear layer.
            layer_norm_kwargs: Additional arguments for the layer normalization.
                If None, default arguments are used.
            dropout: Dropout rate.
            output_activation: Activation function to use for the output layer.
                If None, same as `activation` and no dropout is applied.
            name: Name of the module.
        """
        super().__init__(name)
        self.output_dropout = True if output_activation is None else False
        latent_size = output_size if latent_size is None else latent_size
        self.residual = residual
        assert (
            not residual or input_size == output_size
        ), "Residual connection requires input and output size to be the same."
        self.dropout = dropout
        self.layers = []
        self.activation = _Act(activation)
        self.output_activation = (
            _Act(activation) if output_activation is None else _Act(
                output_activation)
        )
        self.layer_norm = None
        self.batch_norm = None

        if batch_norm:
            if batch_norm_kwargs is None:
                self.batch_norm = hk.BatchNorm(
                    create_scale=True, create_offset=True, decay_rate=0.999, eps=1e-5
                )
            else:
                self.batch_norm = hk.BatchNorm(**batch_norm_kwargs)
        if layer_norm:
            if layer_norm_kwargs is None:
                self.layer_norm = hk.LayerNorm(
                    axis=-1, create_scale=True, create_offset=True, eps=1e-5
                )
            else:
                self.layer_norm = hk.LayerNorm(axis=-1, **layer_norm_kwargs)

        for i in range(n_layers - 1):
            self.layers.append(hk.Linear(latent_size, name=f"linear_{i}"))
        self.output_layer = hk.Linear(
            output_size, name=f"linear_{n_layers - 1}")

    def __call__(
        self, x: jnp.ndarray, is_training: Optional[bool] = False
    ) -> jnp.ndarray:
        """
        Args:
            x: Input tensor.
            is_training: Whether the model is in training mode.
                For haiku initialization, this argument needs to be set to True.
        Returns:
            Output tensor.
        """
        y = x
        if self.layer_norm:
            y = self.layer_norm(y)

        for layer in self.layers:
            y = layer(y)
            y = self.activation(y)
            if is_training:
                y = hk.dropout(hk.next_rng_key(), self.dropout, y)
        y = self.output_layer(y)
        y = self.output_activation(y)
        if self.output_dropout and is_training:
            y = hk.dropout(hk.next_rng_key(), self.dropout, y)
        if self.batch_norm:
            y = self.batch_norm(y, is_training=is_training)
        if self.residual:
            y = y + x
        return y


class _EdgeBlock(hk.Module):
    """Edge block for message-passing networks. Performs the edge update."""

    def __init__(
        self,
        edge_input_size: int,
        node_size: int,
        edge_output_size: Optional[int] = None,
        fixed_edge_attributes: Optional[int] = None,
        name: Optional[str] = None,
        kwargs: Optional[dict] = {},
    ):
        super().__init__()
        """
        Args:
            edge_input_size: Dimension of the edge features.
            node_size: Dimension of the node features.
            edge_output_size: Dimension of the output edge features.
                If None, the output dimension is the same as the input dimension.
            fixed_edge_attributes: Dimension of the fixed edge attributes.
            name: Name of the module.
            kwargs: Additional arguments for the edge_update MLP.
        """
        super().__init__(name)
        if edge_output_size is None:
            edge_output_size = edge_input_size
        input_size = edge_input_size + 2 * node_size
        if fixed_edge_attributes is not None:
            edge_output_size -= fixed_edge_attributes
        self.edge_update_mlp = MLP(
            input_size, edge_output_size, name="edge_update_mlp", **kwargs
        )
        self.fixed_edge_attributes = fixed_edge_attributes

    def __call__(
        self, graph: jraph.GraphsTuple, is_training: Optional[bool] = True
    ) -> jraph.GraphsTuple:
        """
        Args:
            graph: GraphsTuple object. When using fixed edge attributes,
                the last fixed_edge_attributes dimensions of the edge features
                are assumed to be the fixed edge attributes.
            is_training: Whether the model is in training mode.
                For haiku initialization, this argument needs to be set to True.
        Returns:
            GraphsTuple object after the edge update.
        """
        sender_features = graph.nodes.latent[graph.senders]
        receiver_features = graph.nodes.latent[graph.receivers]

        edge_features = graph.edges.latent
        updated_edge_features = self.edge_update_mlp(
            jnp.concatenate(
                [edge_features, sender_features, receiver_features], axis=-1
            ),
            is_training=is_training,
        )
        if self.fixed_edge_attributes is not None:
            updated_edge_features = jnp.concatenate(
                [
                    updated_edge_features,
                    graph.edges.latent[:, -self.fixed_edge_attributes:],
                ],
                axis=-1,
            )

        return graph._replace(edges=graph.edges._replace(latent=updated_edge_features))


class _NodeBlock(hk.Module):
    """Node block for message-passing networks. Performs the node update."""

    def __init__(
        self,
        node_input_size: int,
        edge_input_size: int,
        node_output_size: int | None = None,
        aggregation: Optional[str] = "sum",
        fixed_node_attributes: Optional[int] = None,
        max_nodes: Optional[int] = None,
        name: Optional[str] = None,
        kwargs: Optional[dict] = {},
    ):
        super().__init__()
        """
        Args:
            node_input_size: Dimension of the  node features.
                If fixed_node_attributes is not None,
                the last fixed_node_attributes features are assumed to be fixed.
            edge_input_size: Dimension of the edge features.
            node_output_size: Dimension of the output node features.
                If None, the output dimension is the same as the input dimension.
            aggregation: Aggregation function to use for the edge features.
            fixed_node_attributes: Number of fixed node attributes.
                If None, the graph has no fixed node attributes.
            name: Name of the module.
            max_nodes: Maximum number of nodes in the graph.
                If None, the maximum number of nodes is not fixed.
                This is needed for jax jit compilation.
            kwargs: Additional arguments for the node_update MLP.
        """
        super().__init__(name)
        self.max_nodes = max_nodes
        self.fixed_node_attributes = fixed_node_attributes
        if node_output_size is None:
            node_output_size = node_input_size
        input_size = node_input_size + edge_input_size
        if fixed_node_attributes is not None:
            node_output_size = node_output_size - fixed_node_attributes
        self.node_update_mlp = MLP(
            input_size, node_output_size, name="node_update_mlp", **kwargs
        )

        if max_nodes is None:
            self.aggregation_fn = get_aggregation_fn(aggregation)
        else:
            self.aggregation_fn = partial(
                get_aggregation_fn(aggregation), num_segments=max_nodes
            )

    def __call__(
        self, graph: jraph.GraphsTuple, is_training: Optional[bool] = False
    ) -> jraph.GraphsTuple:
        """
        Args:
            graph: GraphsTuple object. If the graph has fixed node attributes,
                the last fixed_node_attributes columns of the node features
                are assumed to be fixed.
            is_training: Whether the model is in training mode.
                For haiku initialization, this argument needs to be set to True.
        Returns:
            GraphsTuple object after the node update.
        """
        node_features = graph.nodes.latent
        aggregated_edge_features = self.aggregation_fn(
            data=graph.edges.latent, segment_ids=graph.receivers
        )
        updated_node_features = self.node_update_mlp(
            jnp.concatenate(
                [node_features, aggregated_edge_features], axis=-1),
            is_training=is_training,
        )
        if self.fixed_node_attributes is not None:
            updated_node_features = jnp.concatenate(
                [
                    updated_node_features,
                    node_features[:, -self.fixed_node_attributes:],
                ],
                axis=-1,
            )
        return graph._replace(nodes=graph.nodes._replace(latent=updated_node_features))


class MessagePassingLayer(hk.Module):
    """
    Message-passing layer for graph neural networks.
    Implements a single message passing step.
    """

    def __init__(
        self,
        node_input_size: int,
        edge_input_size: int,
        node_output_size: int | None = None,
        edge_output_size: int | None = None,
        aggregation: Optional[str] = "sum",
        skip_connection: Optional[bool] = False,
        fixed_node_attributes: Optional[int] = None,
        fixed_edge_attributes: Optional[int] = None,
        max_nodes: Optional[int] = None,
        name: Optional[str] = None,
        edge_update_kwargs: Optional[dict] = {},
        node_update_kwargs: Optional[dict] = {},
    ):
        """
        Args:
            node_input_size: Dimension of the (learnable) node features.
            edge_input_size: Dimension of the edge features.
            node_output_size: Dimension of the (learnable) output node features.
                If None, the output dimension is the same as the input dimension.
            edge_output_size: Dimension of the output edge features.
                If None, the output dimension is the same as the input dimension.
            aggregation: Aggregation function to use for the edge features.
            skip_connection: Whether to add a skip connection to the output.
            fixed_node_attributes: Number of fixed node attributes.
                If None, the graph has no fixed node attributes.
            fixed_edge_attributes: Number of fixed edge attributes.
                If None, the graph has no fixed edge attributes.
            max_nodes: Maximum number of nodes in the graph.
                If None, the maximum number of nodes is not fixed.
                Needs to be set for jax jit compilation.
            name: Name of the module.
            edge_update_kwargs: Additional arguments for the edge_update MLP.
            node_update_kwargs: Additional arguments for the node_update MLP.

            MLP kwargs:
                - n_layers: Number of layers in the MLP.
                - latent_size: Dimension of the latent space.
                - activation: Activation function to use. Default: 'relu'.
                - batch_norm: Whether to use batch normalization. Default: False.
                - layer_norm: Whether to use layer normalization. Default: False.
                - dropout_rate: Dropout rate. Default: 0.0.
                - output_activation: Activation function for the output layer.
                    If None, same as activation. Default: None.
        """
        super().__init__(name)
        if node_output_size is None:
            node_output_size = node_input_size
        if edge_output_size is None:
            edge_output_size = edge_input_size
        if skip_connection:
            assert (
                node_input_size == node_output_size
            ), "Skip connection requires input \
                    and output node dimensions to be the same."
            assert (
                edge_input_size == edge_output_size
            ), "Skip connection requires input \
                    and output edge dimensions to be the same."
        if max_nodes is None:
            raise UserWarning(
                "max_nodes is None. \
                    This means, that the layer cannot be jit compiled."
            )
        self.skip_connection = skip_connection
        self.edge_block = _EdgeBlock(
            edge_input_size=edge_input_size,
            node_size=node_input_size,
            edge_output_size=edge_output_size,
            fixed_edge_attributes=fixed_edge_attributes,
            name="edge_block",
            kwargs=edge_update_kwargs,
        )

        self.node_block = _NodeBlock(
            node_input_size=node_input_size,
            edge_input_size=edge_output_size,
            node_output_size=node_output_size,
            aggregation=aggregation,
            fixed_node_attributes=fixed_node_attributes,
            max_nodes=max_nodes,
            name="node_block",
            kwargs=node_update_kwargs,
        )

    def __call__(
        self, graph: jraph.GraphsTuple, is_training: Optional[bool] = False
    ) -> jraph.GraphsTuple:
        """
        Args:
            graph: GraphsTuple object.
            is_training: Whether the model is in training mode.
                For haiku initialization, this argument needs to be set to True.
        Returns:
            GraphsTuple object after the message passing step.
        """
        y = self.edge_block(graph, is_training=is_training)
        y = self.node_block(y, is_training=is_training)
        if self.skip_connection:
            y = y.replace_latent(
                y.nodes.latent + graph.nodes.latent, y.edges.latent + graph.edges.latent
            )
        return y


class _Encoder(hk.Module):
    """
    Encodes the graph into a latent representation.
    Assumes, that the graph already has edge connectivity
    """

    def __init__(
        self,
        node_input_size: int,
        edge_input_size: int,
        node_output_size: int,
        edge_output_size: int,
        fixed_node_attributes: Optional[int] = None,
        fixed_edge_attributes: Optional[int] = None,
        name: Optional[str] = None,
        node_encoder_kwargs: Optional[dict] = {},
        edge_encoder_kwargs: Optional[dict] = {},
    ):
        """
        Args:
            node_input_size: Input dimension of the node features.
                If fixed_node_attributes is not None, then the last
                fixed_node_attributes columns of the node features
                are assumed to be fixed.
            edge_input_size: Input dimension of the edge features.
            node_output_size: Output dimension of the node features.
            edge_output_size: Output dimension of the edge features.
            fixed_node_attributes: Number of fixed node attributes.
                If None, no fixed node attributes are assumed.
            name: Name of the module.
            node_encoder_kwargs: Additional arguments for the node encoder MLP.
            edge_encoder_kwargs: Additional arguments for the edge encoder MLP.
        """
        super().__init__(name)
        self.fixed_node_attributes = fixed_node_attributes
        self.fixed_edge_attributes = fixed_edge_attributes
        if self.fixed_node_attributes is not None:
            node_output_size -= self.fixed_node_attributes
        if self.fixed_edge_attributes is not None:
            edge_output_size -= self.fixed_edge_attributes

        self.node_encoder = MLP(
            node_input_size, node_output_size, **node_encoder_kwargs
        )
        self.edge_encoder = MLP(
            edge_input_size, edge_output_size, **edge_encoder_kwargs
        )

    def __call__(
        self, graph: jraph.GraphsTuple, is_training: Optional[bool] = False
    ) -> jraph.GraphsTuple:
        """
        Args:
            graph: GraphsTuple object. If the graph has fixed node attributes,
                the last fixed_node_attributes columns of the node features are assumed
                to be fixed.
            is_training: Whether the network is in training mode.
                Needs to be true during the haiku init call.
        Returns:
            GraphsTuple object after the encoding.
        """
        node_features = self.node_encoder(
            graph.nodes.latent, is_training=is_training)
        edge_features = self.edge_encoder(
            graph.edges.latent, is_training=is_training)
        if self.fixed_node_attributes is not None:
            node_features = jnp.concatenate(
                [node_features, graph.nodes.latent[:, : self.fixed_node_attributes]],
                axis=-1,
            )
        if self.fixed_edge_attributes is not None:
            edge_features = jnp.concatenate(
                [edge_features, graph.edges.latent[:, : self.fixed_edge_attributes]],
                axis=-1,
            )

        return graph.replace_latent(node_features, edge_features)


class _Processor(hk.Module):
    """Processes the latent representation, with a message-passing GNN."""

    def __init__(
        self,
        node_size: int,
        edge_size: int,
        message_passing_layers: int,
        message_passing_steps: Optional[bool] = 1,
        residual_connections: Optional[bool] = False,
        fixed_node_attributes: Optional[int] = None,
        fixed_edge_attributes: Optional[int] = None,
        max_nodes: Optional[int] = None,
        name: Optional[str] = None,
        message_passing_layer_kwargs: Optional[Union[list[dict], dict]] = {},
    ):
        """
        Args:
            node_size: Dimension of the node features.
                If fixed_node_attributes is not None, then the last
                fixed_node_attributes columns of the node features
                are assumed to be fixed.
            edge_size: Dimension of the edge features.
            message_passing_layers: Number of message passing layers.
            message_passing_steps: Number of message passing steps.
            residual_connections: Whether to use residual connections
                for the message passing layers.
            fixed_node_attributes: Number of fixed node attributes.
                If None, no fixed node attributes are assumed.
            fixed_edge_attributes: Number of fixed edge attributes.
                If None, no fixed edge attributes are assumed.
            max_nodes: Maximum number of nodes in the graph. If None,
                no maximum number of nodes is assumed.
                Needs to be provided to be jit-able.
            name: Name of the module.
            message_passing_layer_kwargs: Additional arguments for the
                message passing layers.
                 If a list of dicts is provided, each dict is used for one layer.
                 If only one dict is provided, the same dict is used for all layers.
                 This is also the only case where we can have different node and edge
                 sizes for each layer, otherwise the node and edge sizes are fixed.
        """
        super().__init__(name)
        self.message_passing_steps = message_passing_steps
        self.residual_connections = residual_connections
        self.max_nodes = max_nodes
        if isinstance(message_passing_layer_kwargs, list):
            assert (
                len(message_passing_layer_kwargs) == message_passing_layers
            ), "Number of message passing layer configs must be equal to the number \
                    of message passing layers"
        self.message_passing_layers = []

        if isinstance(message_passing_layer_kwargs, list):
            # in this case, we have different configs for each layer,
            # this is also the only case where we can have different node and edge sizes
            # for each layer
            assert (
                message_passing_layer_kwargs[0]["node_output_size"] == node_size
            ), "First message passing layer must have the same node size as the input \
                    node size"
            assert (
                message_passing_layer_kwargs[0]["edge_output_size"] == edge_size
            ), "First message passing layer must have the same edge size as the input \
                    edge size"
            for i, kwargs in enumerate(message_passing_layer_kwargs):
                message_passing_layer_kwargs[i] = kwargs | {
                    "node_input_size": node_size,
                    "edge_input_size": edge_size,
                }

            for i in range(message_passing_layers):
                self.message_passing_layers.append(
                    MessagePassingLayer(
                        fixed_node_attributes=fixed_node_attributes,
                        fixed_edge_attributes=fixed_edge_attributes,
                        max_nodes=max_nodes,
                        name=f"message_passing_layer_{i}",
                        **message_passing_layer_kwargs[i],
                    )
                )
        else:
            for i in range(message_passing_layers):
                self.message_passing_layers.append(
                    MessagePassingLayer(
                        node_input_size=node_size,
                        edge_input_size=edge_size,
                        fixed_node_attributes=fixed_node_attributes,
                        fixed_edge_attributes=fixed_edge_attributes,
                        max_nodes=max_nodes,
                        name=f"message_passing_layer_{i}",
                        **message_passing_layer_kwargs,
                    )
                )

    def __call__(
        self, graph: jraph.GraphsTuple, is_training: Optional[bool] = False
    ) -> jraph.GraphsTuple:
        """
        Args:
            graph: GraphsTuple object. If the graph has fixed node attributes,
                the last fixed_node_attributes columns of the node features are
                assumed to be fixed.
            is_training: Whether the network is in training mode. Needs to be true
                during the haiku init call.
        Returns:
            GraphsTuple object after the processing.
        """
        for message_passing_layer in self.message_passing_layers:
            for _ in range(self.message_passing_steps):
                graph = message_passing_layer(graph, is_training=is_training)
        return graph


class _Decoder(hk.Module):
    """
    Decodes the latent representation into a graph. Only the node features are decoded.
    """

    def __init__(
        self,
        node_input_size: int,
        node_output_size: int,
        fixed_node_attributes: Optional[int] = None,
        name: Optional[str] = None,
        node_decoder_kwargs: Optional[dict] = {},
    ):
        """
        Args:
            node_input_size: Input dimension of the node features.
                If fixed_node_attributes is not None, the last fixed_node_attributes
                columns of the node features are assumed to be fixed.
            node_output_size: Output dimension of the node features. If the output
                should not have fixed node attributes, then fixed_node_attributes
                should be set to None.
            fixed_node_attributes: Number of fixed node attributes.
                If None, no fixed node attributes are assumed.
            name: Name of the module.
            node_decoder_kwargs: Additional arguments for the node decoder MLP.
        """
        super().__init__(name)
        self.fixed_node_attributes = fixed_node_attributes
        if fixed_node_attributes is not None:
            node_output_size -= fixed_node_attributes
        self.decoder_mlp = MLP(
            node_input_size, node_output_size, **node_decoder_kwargs)

    def __call__(
        self, graph: jraph.GraphsTuple, is_training: Optional[bool] = False
    ) -> jraph.GraphsTuple:
        """
        Args:
            graph: GraphsTuple object. If the graph has fixed node attributes,
                the last fixed_node_attributes columns of the node features are assumed
                to be fixed.
            is_training: Whether the model is in training model.
                Needs to be set to True for haiku initializers to work.
        Returns:
            GraphsTuple object after the decoding.
        """
        nodes = self.decoder_mlp(graph.nodes.latent, is_training=is_training)
        if self.fixed_node_attributes is not None:
            nodes = jnp.concatenate(
                [nodes, graph.nodes.latent[:, : self.fixed_node_attributes]], axis=-1
            )
        return graph._replace(nodes=graph.nodes._replace(latent=nodes))


class EncoderProcessorDecoder(hk.Module):
    """Complete EncoderProcessorDecoder graph neural network."""

    def __init__(
        self,
        node_input_size: int,
        edge_input_size: int,
        node_output_size: int,
        edge_output_size: int,
        message_passing_layers: int,
        message_passing_steps: int = 1,
        residual_connections: bool = False,
        node_latent_size: Optional[int] = None,
        edge_latent_size: Optional[int] = None,
        fixed_node_attributes: Optional[int] = None,
        fixed_edge_attributes: Optional[int] = None,
        max_nodes: Optional[int] = None,
        name: Optional[str] = None,
        node_encoder_kwargs: Optional[dict] = {},
        edge_encoder_kwargs: Optional[dict] = {},
        message_passing_layer_kwargs: Optional[Union[dict, list[dict]]] = {},
        node_decoder_kwargs: Optional[dict] = {},
        mlp_kwargs: Optional[dict] = {},
    ):
        """
        Args:
            node_input_size: Input dimension of the node features.
            edge_input_size: Input dimension of the edge features.
            node_output_size: Output dimension of the node features.
            edge_output_size: Output dimension of the edge features. Convenience
                argument assuring that the edge latent size is set.
            message_passing_layers: Number of message passing layers.
            message_passing_steps: Number of message passing steps.
            residual_connections: Whether to use residual connections in the message
                passing layers.
            node_latent_size: Latent dimension of the node features.
                If None, the latent dimension is set to the output dimension.
                The Processor will use the
            edge_latent_size: Latent dimension of the edge features.
                If None, the latent dimension is set to the output dimension.
            fixed_node_attributes: Number of fixed node attributes.
                If None, no fixed node attributes are assumed.
            fixed_edge_attributes: Number of fixed edge attributes.
                If None, no fixed edge attributes are assumed.
            max_nodes: Maximum number of nodes in the graph.
                If None, the module will not be jittable.
            name: Name of the module.
            node_encoder_kwargs: Additional arguments for the node encoder MLP.
            edge_encoder_kwargs: Additional arguments for the edge encoder MLP.
            message_passing_layer_kwargs: Additional arguments for the
                MessagePassingLayer.
                If a list of dicts is provided, each dict is used for one layer.
                If only one dict is provided, the same dict is used for all layers.
            node_decoder_kwargs: Additional arguments for the node decoder MLP.
            mlp_kwargs: Additional arguments for all MLPs. Specific MLP kwargs overwrite
                these kwargs.

        MessagePassingLayer kwargs:
            - **aggregation:** Aggregation function. *Default: 'sum'*.
            - **skip_connection:** Whether to use a skip connection to
                bypass the layer. *Default: False*.
            - **edge_update_kwargs:** Additional arguments for the edge update MLP.
            - **node_update_kwargs:** Additional arguments for the node update MLP.

        MLP kwargs:
            - **n_layers:** Number of layers.
            - **latent_size:** Latent dimension.
            - **activation:** Activation function. *Default: 'relu'*.
            - **batch_norm:** Whether to use batch normalization. *Default: False*.
            - **layer_norm:** Whether to use layer normalization. *Default: False*.
            - **dropout_rate:** Dropout rate. *Default: 0.0*.
            - **output_activation:** Output activation function. If None, same as
                activation. *Default: None*.
        """
        super().__init__(name)
        if node_latent_size is None:
            node_latent_size = node_output_size
        if edge_latent_size is None:
            edge_latent_size = edge_output_size

        # Update specific kwargs with general kwargs
        node_encoder_kwargs = mlp_kwargs | node_encoder_kwargs
        edge_encoder_kwargs = mlp_kwargs | edge_encoder_kwargs
        if isinstance(message_passing_layer_kwargs, list):
            message_passing_layer_kwargs = [
                mlp_kwargs | layer_kwargs
                for layer_kwargs in message_passing_layer_kwargs
            ]
        else:
            message_passing_layer_kwargs = mlp_kwargs | message_passing_layer_kwargs
        node_decoder_kwargs = mlp_kwargs | node_decoder_kwargs

        self.encoder = _Encoder(
            node_input_size=node_input_size,
            edge_input_size=edge_input_size,
            node_output_size=node_latent_size,
            edge_output_size=edge_latent_size,
            fixed_node_attributes=fixed_node_attributes,
            name="encoder",
            node_encoder_kwargs=node_encoder_kwargs,
            edge_encoder_kwargs=edge_encoder_kwargs,
        )
        self.processor = _Processor(
            node_size=node_latent_size,
            edge_size=edge_latent_size,
            message_passing_layers=message_passing_layers,
            message_passing_steps=message_passing_steps,
            residual_connections=residual_connections,
            fixed_node_attributes=fixed_node_attributes,
            fixed_edge_attributes=fixed_edge_attributes,
            max_nodes=max_nodes,
            name="processor",
            message_passing_layer_kwargs=message_passing_layer_kwargs,
        )
        self.decoder = _Decoder(
            node_input_size=node_latent_size,
            node_output_size=node_output_size,
            fixed_node_attributes=fixed_node_attributes,
            name="decoder",
            node_decoder_kwargs=node_decoder_kwargs,
        )

    def __call__(
        self, graph: jraph.GraphsTuple, is_training: Optional[bool] = False
    ) -> jraph.GraphsTuple:
        """
        Args:
            graph: GraphsTuple object. If the graph has fixed node attributes,
                the last fixed_node_attributes columns of the node features are assumed
                to be fixed.
            is_training: Whether the model is in training model.
                Needs to be set to True for haiku initializers to work.
        Returns:
            GraphsTuple object after the processing.
        """
        graph = self.encoder(graph, is_training=is_training)
        graph = self.processor(graph, is_training=is_training)
        graph = self.decoder(graph, is_training=is_training)
        return graph
