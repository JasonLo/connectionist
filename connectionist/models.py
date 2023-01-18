from typing import Dict, Union, List, Tuple
import tensorflow as tf
from connectionist.layers import PMSPLayer, HNSLayer
from connectionist.surgery import SurgeryPlan, Surgeon, make_recipient, copy_transplant


class PMSP(tf.keras.Model):
    """PMSP sim 3 model. A recurrent neural network with hidden layer, phonology layer and cleanup layer. It takes orthography as input and outputs phonology.

    See Plaut, McClelland, Seidenberg and Patterson (1996), simulation 3 for details.

    Args:
        tau: the time constant (smaller time constant = slower temporal dynamics, tau in [0, 1])
        h_units: the number of hidden units
        p_units: the number of phonology units
        c_units: the number of cleanup units
        connections: the connections between layers, default to: ['oh', 'ph', 'hp', 'pp', 'cp', 'pc']

    Call args:

        inputs: the input tensor, shape: (batch_size, time_steps, o_units)
        return_internals: whether to return the internal dynamics, default to: False.

            If `return_internals`=`True`, the model will return a dictionary with all the internal dynamics. Including:
                - activations in all layers (e.g., 'hidden')
                - raw inputs in each connection (e.g., 'oh' = $o @ w_oh$)

            If `return_internals`=`False` the model will return the phonology activation, i.e., $p$.

    Example:

    ```python
    import tensorflow as tf
    from connectionist.data import ToyOP
    from connectionist.models import PMSP

    data = ToyOP()
    model = PMSP(tau=0.2, h_units=10, p_units=9, c_units=5)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
    )
    model.fit(data.x_train, data.y_train, epochs=3, batch_size=20)
    ```

    """

    def __init__(
        self,
        tau: float,
        h_units: int,
        p_units: int,
        c_units: int,
        h_noise: float = 0.0,
        p_noise: float = 0.0,
        c_noise: float = 0.0,
        connections: List[str] = None,
        zero_out_rates: Dict[str, float] = None,
        l2: float = 0.0,
    ) -> None:
        super().__init__()

        self.tau = tau
        self.h_units = h_units
        self.p_units = p_units
        self.c_units = c_units
        self.h_noise = h_noise
        self.p_noise = p_noise
        self.c_noise = c_noise
        self.connections = connections
        self.zero_out_rates = zero_out_rates
        self.l2 = l2

        if connections is None:
            self.connections = ["oh", "ph", "hp", "pp", "cp", "pc"]

        if zero_out_rates is None:
            self.zero_out_rates = {conn: 0.0 for conn in self.connections}

        self.pmsp = PMSPLayer(
            tau=tau,
            h_units=self.h_units,
            p_units=self.p_units,
            c_units=self.c_units,
            h_noise=self.h_noise,
            p_noise=self.p_noise,
            c_noise=self.c_noise,
            connections=self.connections,
            zero_out_rates=self.zero_out_rates,
            l2=self.l2,
        )

    @property
    def all_layers_names(self) -> List[str]:
        return self.pmsp.all_layers_names

    @property
    def connection_locs(self) -> Dict[str, Dict[str, int]]:
        """A map that shows which axis of a weight is connecting to a layer."""

        return {layer: self._find_conn_locs(layer) for layer in self.all_layers_names}

    @property
    def weights_abbreviations(self) -> Dict[str, str]:
        """Weights name abbreviations. (e.g., w_oh, bias_hidden)."""

        weights = {f"w_{w}": f"{w}/kernel:0" for w in self.connections}

        biases = {f"bias_{layer}": f"{layer}/bias:0" for layer in self.all_layers_names}
        return {**weights, **biases}

    def call(
        self, inputs: tf.Tensor, training: bool = False, return_internals: bool = False
    ) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
        return self.pmsp(inputs, training=training, return_internals=return_internals)

    def get_config(self) -> Dict[str, Union[float, int, list]]:
        return dict(
            tau=self.tau,
            h_units=self.h_units,
            p_units=self.p_units,
            c_units=self.c_units,
            h_noise=self.h_noise,
            p_noise=self.p_noise,
            c_noise=self.c_noise,
            connections=self.connections,
            zero_out_rates=self.zero_out_rates,
            l2=self.l2,
        )

    # Miscellaneous internal methods

    def _find_conn_locs(self, layer: str) -> Dict[str, int]:
        """Find the connection locations of the target layer."""

        def find(l: List[str], prefix: str):
            """Locate connection by layer prefix."""
            return [i for i, x in enumerate(l) if x == prefix]

        prefix = layer[0]
        conn_weights = {}
        for connection in self.connections:
            if prefix in connection:
                conn_weights[f"w_{connection}"] = find(connection, prefix)

        conn_bias = {f"bias_{layer}": [0]}
        return {**conn_weights, **conn_bias}

    def _validate_layer(self, layer: str) -> None:
        """Validate the target layer."""

        if layer not in self.all_layers_names:
            raise ValueError(
                f"Unknown target layer: {layer}, please choose from {self.all_layers_names}"
            )

    def _validate_connections(self, connections: List[str]) -> None:
        if not all([c in self.connections for c in connections]):
            raise ValueError(
                f"Unknown connections: {connections}, please choose from {self.connections}"
            )

    def _to_units(self, layer: str) -> int:
        """Get the number of units in the target layer.

        Args:
            layer: the target layer, choose from ['hidden', 'phonology', 'cleanup']
        """

        mapping = {
            "hidden": self.h_units,
            "phonology": self.p_units,
            "cleanup": self.c_units,
        }
        return mapping[layer]

    # Extra methods for damaging the model

    def shrink_layer(self, layer: str, rate: float) -> tf.keras.Model:
        """Shrink the weights of the target layer.

        Args:
            layer: the target layer, choose from ['hidden', 'phonology', 'cleanup']
            rate: the shrink rate

        Returns:
            A new model with the same architecture, but with new weights shapes that match with the shrank layer.

        Example:

        ```python
        from connectionist.data import ToyOP
        from connectionist.models import PMSP

        model = PMSP(tau=0.2, h_units=10, p_units=9, c_units=5)

        # Build or call or fit a model to instantiate the weights
        y = model(data.x_train)

        new_model = model.shrink_layer('hidden', rate=0.5)
        ```

        """

        self._validate_layer(layer)

        plan = SurgeryPlan(
            layer=layer, original_units=self._to_units(layer), shrink_rate=rate
        )
        surgeon = Surgeon(surgery_plan=plan)

        # Make a new model with the same architecture, but with new weights shapes
        new_model = make_recipient(model=self, surgery_plan=plan, make_model_fn=PMSP)
        new_model.build(input_shape=self.pmsp._build_input_shape)

        surgeon.transplant(donor=self, recipient=new_model)
        return new_model

    def _build_and_transplant(self, config: dict) -> tf.keras.Model:
        """Build a new model with the same architecture with new config."""

        new_model = PMSP(**config)
        new_model.build(input_shape=self.pmsp._build_input_shape)

        # Copy weights
        for weight_name in new_model.weights_abbreviations.keys():
            copy_transplant(donor=self, recipient=new_model, weight_name=weight_name)

        return new_model

    def zero_out(self, zero_out_rates: Dict[str, float]) -> tf.keras.Model:
        """Zero out weights of the target connections.

        Args:
            zero_out_rates: the zero out rates for each connection. e.g., {'hc': 0.5, 'ph': 0.4}
                            higher zero out rates means more weights will be zeroed out.

        Returns:
            A new model with the same architecture, but with new weights.

        """

        model_config = self.get_config()
        model_config["zero_out_rates"].update(zero_out_rates)
        new_model = self._build_and_transplant(model_config)
        new_model.pmsp.cell.zero_out_weights()
        return new_model

    def cut_connections(self, connections: List[str]) -> tf.keras.Model:
        """Cut connections between two layers.

        Args:
            connections: the connections to be cut, the connections most be found in the original model, i.e., in `model.connections`.

        Returns:
            A new model with the same architecture, but with new connections.

        """

        self._validate_connections(connections)

        # Create recipient model with less connections
        model_config = self.get_config()
        remaining_connections = [c for c in self.connections if c not in connections]
        model_config.update(connections=remaining_connections)
        return self._build_and_transplant(model_config)

    def add_noise(self, layer: str, stddev: float) -> tf.keras.Model:
        """Add noise to the target layer.

        The noise is active in both training and inference.

        Args:
            layer: the target layer, choose from ['hidden', 'phonology', 'cleanup']
            stddev: the standard deviation of the noise

        Example:

        ```python
        from connectionist.data import ToyOP
        from connectionist.models import PMSP

        model = PMSP(tau=0.2, h_units=10, p_units=9, c_units=5)

        # Build or call or fit a model to instantiate the weights
        y = model(data.x_train)

        new_model = model.add_noise('hidden', stddev=0.1)
        ```

        """

        self._validate_layer(layer)

        # Create recipient model with noise at the target layer
        model_config = self.get_config()

        def _to_noise_name(layer_name: str) -> str:
            return f"{layer_name[0]}_noise"

        noise_name = _to_noise_name(layer)
        model_config[noise_name] = model_config[noise_name] + stddev
        return self._build_and_transplant(model_config)

    def add_l2(self, l2: float) -> tf.keras.Model:
        """Add L2 regularization to all the weights and biases in the model."""

        model_config = self.get_config()
        model_config.update(l2=l2)
        return self._build_and_transplant(model_config)


class HubAndSpokes(tf.keras.Model):
    def __init__(
        self,
        tau: float,
        hub_name: str,
        hub_units: int,
        spoke_names: List[str],
        spoke_units: List[int],
    ) -> None:

        super().__init__()

        self.tau = tau
        self.hub_name, self.hub_units = hub_name, hub_units
        self.spoke_names, self.spoke_units = spoke_names, spoke_units

        self.hns = HNSLayer(
            tau=tau,
            hub_name=hub_name,
            hub_units=hub_units,
            spoke_names=spoke_names,
            spoke_units=spoke_units,
        )

    def call(
        self, x: Dict[str, tf.Tensor], return_internals: bool = False
    ) -> Dict[str, tf.Tensor]:
        return self.hns(x, return_internals=return_internals)

    def train_step(self, data: Tuple[Dict[str, tf.Tensor]]) -> Dict[str, tf.Tensor]:
        """Train the model for one step, return losses and metrics."""

        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)

            all_losses = []

            # Only inject error according to y_train keys
            for y_name, y_target in y.items():
                sum_loss = tf.reduce_sum(
                    self.compiled_loss(y_true=y_target, y_pred=y_pred[y_name])
                )  # reduce over batch_size axis
                all_losses.append(sum_loss)

            loss_value = tf.reduce_sum(
                all_losses
            )  # Final loss value is the grand sum over every axis (output layer, batch size, time ticks, units).

        grads = tape.gradient(
            loss_value, self.trainable_weights
        )  # compute gradients dL/dw
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_weights)
        )  # update weights using stock optimizer

        self.compiled_metrics.update_state(y, y_pred)  # update metrics
        return {m.name: m.result() for m in self.metrics}
